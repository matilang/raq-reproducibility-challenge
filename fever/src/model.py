"""
model.py
RAG model components for FEVER fact verification.
Covers DPR retrieval, BART generation, and the forward pass.
"""

# BART vocabulary token IDs for label tokens "0", "1", "2"
# Verified by running:
# bart_tokenizer(["0","1","2"], add_special_tokens=False)["input_ids"]
LABEL_TOKEN_IDS = [288, 134, 176]

# ------Load DPR encoder------------------------------
def load_dpr_encoder(device="cuda"):
    """
    Loads the DPR question encoder and tokenizer. And sets up encoder to train().

    Args:
        device: Defaults to "cuda"
        
    Returns: q_encoder, q_tokenizer : DPR encoder and its tokenizer
    """
    from transformers import (
        DPRQuestionEncoder,
        DPRQuestionEncoderTokenizerFast,
    )

    q_tokenizer = DPRQuestionEncoderTokenizerFast.from_pretrained(
        "facebook/dpr-question_encoder-single-nq-base")
    q_encoder = DPRQuestionEncoder.from_pretrained(
        "facebook/dpr-question_encoder-single-nq-base").to("cuda")
    
    q_encoder.train()
    
    return q_encoder, q_tokenizer
    


# ------Load BART------------------------------
def load_bart(device="cuda"):
    """
    Loads BART generator and its tokenizer

    Args:
        device: Defaults to "cuda"
    
    Returns: model, bart_tokenizer
    """
    
    from transformers import (
            BartForConditionalGeneration,
            BartTokenizer
    )
    
    bart_tokenizer = BartTokenizer.from_pretrained("facebook/bart-large")
    model = BartForConditionalGeneration.from_pretrained("facebook/bart-large").to("cuda")
    
    return model, bart_tokenizer
    
# ------Load model from checkpoints------------------------------
def load_model_from_checkpoint(checkpoint_dir, device="cuda"):
    """
    Loads models from checkpoint file. 

    Args:
        checkpoint_dir: str
        device: Defaults to "cuda"
    
    Returns: q_encoder, q_tokenizer, model, bart_tokenizer
    """
    
    q_encoder.load_state_dict(
        torch.load(
            os.path.join(checkpoint_dir, "q_encoder_best.pt"),
            map_location="cuda"
        )
    )
    model.load_state_dict(
        torch.load(
            os.path.join(checkpoint_dir, "bart_best.pt"),
            map_location="cuda"
        )
    )
    
    
# ------Search index function------------------------------
# ------forward pass------------------------------
# ------predict------------------------------