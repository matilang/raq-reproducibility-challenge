"""
model.py
RAG model components for FEVER fact verification.
Covers DPR retrieval, BART generation, and the forward pass.
"""

import os
import torch
import faiss
import numpy as np

# BART vocabulary token IDs for label tokens "0", "1", "2"
# Verified by running:
# bart_tokenizer(["0","1","2"], add_special_tokens=False)["input_ids"]
LABEL_TOKEN_IDS = [288, 134, 176]


# ── Model loading ──────────────────────────────────────────────

def load_dpr_encoder(device="cuda"):
    """
    Load DPR question encoder and tokenizer.
    Sets encoder to train() mode — caller switches to eval() if needed.

    Args:
        device (str): torch device string. Defaults to "cuda".

    Returns:
        q_encoder:   DPRQuestionEncoder on device
        q_tokenizer: DPRQuestionEncoderTokenizerFast
    """
    from transformers import (
        DPRQuestionEncoder,
        DPRQuestionEncoderTokenizerFast,
    )

    q_tokenizer = DPRQuestionEncoderTokenizerFast.from_pretrained(
        "facebook/dpr-question_encoder-single-nq-base"
    )
    q_encoder = DPRQuestionEncoder.from_pretrained(
        "facebook/dpr-question_encoder-single-nq-base"
    ).to(device)

    q_encoder.train()
    return q_encoder, q_tokenizer


def load_bart(device="cuda"):
    """
    Load BART-large generator and its tokenizer.
    Sets model to train() mode — caller switches to eval() if needed.

    Args:
        device (str): torch device string. Defaults to "cuda".

    Returns:
        model:         BartForConditionalGeneration on device
        bart_tokenizer: BartTokenizer
    """
    from transformers import (
        BartForConditionalGeneration,
        BartTokenizer,
    )

    bart_tokenizer = BartTokenizer.from_pretrained("facebook/bart-large")
    model = BartForConditionalGeneration.from_pretrained(
        "facebook/bart-large"
    ).to(device)

    model.train()
    return model, bart_tokenizer


def load_models_from_checkpoint(checkpoint_dir, device="cuda"):
    """
    Load both models from saved checkpoints.
    Prefers GPU checkpoints (_best_gpu.pt) over CPU checkpoints (_best.pt).
    Sets both models to eval() mode.

    Args:
        checkpoint_dir (str): path to checkpoints directory
        device (str):         torch device string. Defaults to "cuda".

    Returns:
        q_encoder:     DPRQuestionEncoder loaded from checkpoint
        q_tokenizer:   DPRQuestionEncoderTokenizerFast
        model:         BartForConditionalGeneration loaded from checkpoint
        bart_tokenizer: BartTokenizer
    """
    q_encoder, q_tokenizer = load_dpr_encoder(device)
    model, bart_tokenizer  = load_bart(device)

    gpu_q    = os.path.join(checkpoint_dir, "q_encoder_best_gpu.pt")
    gpu_bart = os.path.join(checkpoint_dir, "bart_best_gpu.pt")
    cpu_q    = os.path.join(checkpoint_dir, "q_encoder_best.pt")
    cpu_bart = os.path.join(checkpoint_dir, "bart_best.pt")

    if os.path.exists(gpu_q):
        q_encoder.load_state_dict(
            torch.load(gpu_q,    map_location=device)
        )
        model.load_state_dict(
            torch.load(gpu_bart, map_location=device)
        )
        print("Loaded GPU best checkpoint")
    else:
        q_encoder.load_state_dict(
            torch.load(cpu_q,    map_location=device)
        )
        model.load_state_dict(
            torch.load(cpu_bart, map_location=device)
        )
        print("Loaded CPU best checkpoint")

    q_encoder.eval()
    model.eval()
    return q_encoder, q_tokenizer, model, bart_tokenizer


# ── Retrieval ──────────────────────────────────────────────────

def search_index(claim, index, passages, q_encoder,
                 q_tokenizer, n_docs=5, max_length=300,
                 training=False):
    """
    Encode a claim with DPR and retrieve top-n passages from FAISS index.

    Args:
        claim (str):       FEVER claim text
        index:             FAISS index (CPU or GPU)
        passages (list):   list of passage dicts with keys: title, text
        q_encoder:         DPRQuestionEncoder
        q_tokenizer:       DPRQuestionEncoderTokenizerFast
        n_docs (int):      number of passages to retrieve. Defaults to 5.
        max_length (int):  max query token length. Defaults to 300.
        training (bool):   if True, keeps gradient on query vector.
                           if False, wraps in torch.no_grad().

    Returns:
        results (list):    list of dicts with keys:
                           rank, score, title, text, idx
        query_vec:         raw query tensor if training=True, else None
    """
    encoded = q_tokenizer(
        claim,
        return_tensors="pt",
        truncation=True,
        max_length=max_length
    )

    if training:
        query_vec = q_encoder(
            input_ids=encoded["input_ids"].to("cuda"),
            attention_mask=encoded["attention_mask"].to("cuda")
        ).pooler_output
        query_vec_np = query_vec.detach().cpu().numpy()
    else:
        with torch.no_grad():
            query_vec = q_encoder(
                input_ids=encoded["input_ids"].to("cuda"),
                attention_mask=encoded["attention_mask"].to("cuda")
            ).pooler_output
        query_vec_np = query_vec.cpu().numpy()

    # normalise for cosine similarity via inner product
    query_vec_np_norm = query_vec_np.copy()
    faiss.normalize_L2(query_vec_np_norm)

    scores, indices = index.search(
        query_vec_np_norm.astype("float32"), n_docs
    )

    results = []
    for score, idx in zip(scores[0], indices[0]):
        results.append({
            "rank":  len(results) + 1,
            "score": float(score),
            "title": passages[idx]["title"],
            "text":  passages[idx]["text"],
            "idx":   int(idx)
        })

    return results, query_vec if training else None


# ── Forward pass ───────────────────────────────────────────────

def forward_pass(claim, retrieved_results, gold_label_str,
                 model, bart_tokenizer, label2id,
                 max_length=300,
                 label_token_ids=None):
    """
    Full RAG forward pass for one FEVER example.

    Steps:
        1. Build K BART inputs: "question: {claim} title: {t} context: {p}"
        2. Run BART on all K inputs simultaneously
        3. Extract logits at token position 0 for label tokens only → [K, 3]
        4. Softmax retrieval scores → p(z|x) for each passage [K]
        5. Marginalize: p(y|x) = Σ p(z|x) · p(y|x,z)  → [3]

    Args:
        claim (str):            FEVER claim text
        retrieved_results (list): list of dicts from search_index()
                                  each dict has: score, title, text
        gold_label_str (str):   "SUPPORTS", "REFUTES", or "NOT ENOUGH INFO"
        model:                  BartForConditionalGeneration
        bart_tokenizer:         BartTokenizer
        label2id (dict):        maps label string → token index string
                                e.g. {"SUPPORTS": "0", ...}
        max_length (int):       max source token length. Defaults to 300.
        label_token_ids (list): BART vocab IDs for "0", "1", "2".
                                Defaults to [288, 134, 176].

    Returns:
        marginalized (Tensor): shape [3] — weighted label logits
        loss (Tensor):         BART's internal loss (per-passage, pre-marginalization)
                               Use only for debugging, not for training loss.

    Notes:
        - Training loss should use CrossEntropyLoss on marginalized output
        - Document encoder and FAISS index are always frozen
        - Gradient flows through BART and query encoder only
    """
    if label_token_ids is None:
        label_token_ids = LABEL_TOKEN_IDS

    from data import prepare_bart_inputs

    # 1. tokenize K inputs
    encoded = prepare_bart_inputs(
        claim, retrieved_results, bart_tokenizer, max_length
    )

    # 2. retrieval scores and labels
    retrieval_scores = [r["score"] for r in retrieved_results]
    K             = len(retrieved_results)
    gold_token_id = label_token_ids[int(label2id[gold_label_str])]
    labels        = torch.full((K, 1), gold_token_id)

    # 3. BART forward pass
    output = model(
        input_ids      = encoded["input_ids"].to("cuda"),
        attention_mask = encoded["attention_mask"].to("cuda"),
        labels         = labels.to("cuda")
    )

    # 4. extract label logits at first generated token position
    # output.logits shape: [K, seq_len, vocab_size]
    logits       = output.logits[:, 0, :]       # [K, vocab_size]
    label_logits = logits[:, label_token_ids]   # [K, 3]

    # 5. retrieval probabilities p(z|x)
    retrieval_probs = torch.softmax(
        torch.tensor(retrieval_scores).to("cuda"), dim=0
    )  # [K]

    # 6. marginalize: p(y|x) = Σ p(z|x) · p(y|x,z)
    marginalized = (
        retrieval_probs.unsqueeze(1) * label_logits
    ).sum(dim=0)  # [3]

    return marginalized, output.loss


# ── Prediction ─────────────────────────────────────────────────

def predict(marginalized, label2id):
    """
    Convert marginalized logits to a predicted label string.

    Args:
        marginalized (Tensor): shape [3] from forward_pass()
        label2id (dict):       maps label string → token index string

    Returns:
        pred_label (str): "SUPPORTS", "REFUTES", or "NOT ENOUGH INFO"
    """
    id2label   = {v: k for k, v in label2id.items()}
    pred_idx   = torch.argmax(marginalized).item()
    return id2label[str(pred_idx)]