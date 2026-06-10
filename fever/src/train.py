"""
train.py
Fine-tuning loop for RAG on FEVER fact verification.
Handles training, validation, checkpoint saving,
and logging of loss and accuracy metrics.

Usage:
    # full reproduction run (matches paper setup)
    python fever/src/train.py

    # debug run on subset
    python fever/src/train.py --train_size 500 --epochs 1

    # custom hyperparameters
    python fever/src/train.py --lr 1e-5 --n_docs 10 --epochs 5

    # resume from checkpoint
    python fever/src/train.py --start_epoch 2 --checkpoint fever/results/checkpoints/q_encoder_epoch2_gpu.pt

    # evaluation only (no training)
    python fever/src/train.py --eval_only
"""

import os
import torch
from torch.amp import GradScaler
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup

from model import search_index, forward_pass, predict, LABEL_TOKEN_IDS
from data import LABEL2ID


def evaluate(data, model, q_encoder, bart_tokenizer,
             q_tokenizer, index, passages, label2id,
             config, max_length=300):
    """
    Run inference on a data split without gradient computation.
    Returns 3-way and 2-way label accuracy.

    3-way: accuracy over SUPPORTS / REFUTES / NOT ENOUGH INFO
    2-way: accuracy over SUPPORTS / REFUTES only (NEI examples excluded)

    Args:
        data (list):        list of FEVER examples (dicts with claim, label)
        model:              BartForConditionalGeneration
        q_encoder:          DPRQuestionEncoder
        bart_tokenizer:     BartTokenizer
        q_tokenizer:        DPRQuestionEncoderTokenizerFast
        index:              FAISS index (CPU or GPU)
        passages (list):    list of passage dicts
        label2id (dict):    maps label string to token index string
        config (dict):      loaded from fever_config.yaml
        max_length (int):   max source token length. Defaults to 300.

    Returns:
        acc_3way (float): 3-way accuracy
        acc_2way (float): 2-way accuracy (SUPPORTS/REFUTES only)
    """
    model.eval()
    q_encoder.eval()

    correct_3way = correct_2way = total_3way = total_2way = 0
    id2label = {v: k for k, v in label2id.items()}

    with torch.no_grad():
        for example in data:
            results, _ = search_index(
                example["claim"], index, passages,
                q_encoder, q_tokenizer,
                n_docs=config["model"]["n_docs"],
                max_length=max_length,
                training=False
            )
            marginalized, _ = forward_pass(
                example["claim"], results, example["label"],
                model, bart_tokenizer, label2id,
                max_length=max_length
            )

            pred_idx   = torch.argmax(marginalized).item()
            pred_label = id2label[str(pred_idx)]
            gold_label = example["label"]

            total_3way   += 1
            correct_3way += int(pred_label == gold_label)

            if gold_label != "NOT ENOUGH INFO":
                total_2way   += 1
                correct_2way += int(pred_label == gold_label)

    model.train()
    q_encoder.train()

    acc_3way = correct_3way / total_3way
    acc_2way = correct_2way / max(total_2way, 1)
    return acc_3way, acc_2way


def train(model, q_encoder, bart_tokenizer, q_tokenizer,
          train_data, val_data, index, passages, config,
          checkpoint_dir, start_epoch=0, n_epochs=None,
          max_length=300):
    """
    Full training loop for RAG on FEVER.

    Jointly fine-tunes BART generator and DPR query encoder.
    Document encoder and FAISS index are kept frozen throughout
    (as per the RAG paper, Section 2.4).

    Training details:
        - Loss: CrossEntropyLoss on marginalized logits
          implements Σⱼ −log p(yⱼ|xⱼ) from the RAG paper
        - Optimizer: AdamW (paper uses Adam — AdamW chosen as
          established best practice for transformer fine-tuning,
          see DECISIONS.md Decision 7)
        - Scheduler: linear warmup then linear decay
          (not specified in paper but standard practice)
        - Mixed precision: fp16 via GradScaler
        - Gradient accumulation over GRAD_ACCUM steps
        - Gradient clipping: max_norm from config

    Args:
        model:              BartForConditionalGeneration on cuda
        q_encoder:          DPRQuestionEncoder on cuda
        bart_tokenizer:     BartTokenizer
        q_tokenizer:        DPRQuestionEncoderTokenizerFast
        train_data (list):  list of training examples
        val_data (list):    list of validation examples
        index:              FAISS index for retrieval
        passages (list):    list of passage dicts
        config (dict):      loaded from fever_config.yaml
        checkpoint_dir (str): path to save checkpoints
        start_epoch (int):  epoch to resume from. Defaults to 0.
        n_epochs (int):     epochs to run. Defaults to config value.
        max_length (int):   max source token length. Defaults to 300.

    Returns:
        best_val_acc (float): best 3-way validation accuracy achieved
    """
    GRAD_ACCUM = config["training"]["gradient_accumulation_steps"]
    EPOCHS     = n_epochs if n_epochs is not None \
                 else config["training"]["epochs"]

    total_steps = (
        len(train_data) // config["training"]["batch_size"]
    ) * EPOCHS

    optimizer = AdamW([
        {"params": q_encoder.parameters(),
         "lr": config["training"]["learning_rate"]},
        {"params": model.parameters(),
         "lr": config["training"]["learning_rate"]}
    ], weight_decay=config["training"]["weight_decay"])

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config["training"]["warmup_steps"],
        num_training_steps=total_steps
    )

    scaler  = GradScaler()
    loss_fn = torch.nn.CrossEntropyLoss(
        label_smoothing=config["training"]["label_smoothing"]
    )

    os.makedirs(checkpoint_dir, exist_ok=True)
    best_val_acc = 0.0

    for epoch in range(start_epoch, start_epoch + EPOCHS):

        model.train()
        q_encoder.train()

        total_loss = correct = total = 0
        optimizer.zero_grad()

        for i, example in enumerate(train_data):

            retrieved_results, _ = search_index(
                example["claim"], index, passages,
                q_encoder, q_tokenizer,
                n_docs=config["model"]["n_docs"],
                max_length=max_length,
                training=True
            )

            marginalized, _ = forward_pass(
                example["claim"], retrieved_results,
                example["label"], model,
                bart_tokenizer, LABEL2ID,
                max_length=max_length
            )

            gold_idx = torch.tensor(
                [int(LABEL2ID[example["label"]])]
            ).to("cuda")
            loss = loss_fn(marginalized.unsqueeze(0), gold_idx)
            loss = loss / GRAD_ACCUM

            scaler.scale(loss).backward()

            if (i + 1) % GRAD_ACCUM == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    list(model.parameters()) +
                    list(q_encoder.parameters()),
                    config["training"]["max_grad_norm"]
                )
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()

            total_loss += loss.item() * GRAD_ACCUM
            pred        = torch.argmax(marginalized).item()
            correct    += int(pred == int(LABEL2ID[example["label"]]))
            total      += 1

            if i % 50 == 0:
                print(f"Epoch {epoch+1} | "
                      f"Step {i}/{len(train_data)} | "
                      f"Loss: {total_loss/(i+1):.4f} | "
                      f"Acc: {correct/max(total,1):.1%}")

        # handle remaining gradient accumulation steps
        if len(train_data) % GRAD_ACCUM != 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                list(model.parameters()) +
                list(q_encoder.parameters()),
                config["training"]["max_grad_norm"]
            )
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()

        # validation
        val_acc_3way, val_acc_2way = evaluate(
            val_data, model, q_encoder,
            bart_tokenizer, q_tokenizer,
            index, passages, LABEL2ID,
            config, max_length
        )

        print(f"\nEpoch {epoch+1} complete")
        print(f"  Train acc:     {correct/total:.1%}")
        print(f"  Val 3-way acc: {val_acc_3way:.1%}  (target 72.5%)")
        print(f"  Val 2-way acc: {val_acc_2way:.1%}  (target 89.5%)")

        # save per-epoch checkpoint
        torch.save(
            q_encoder.state_dict(),
            os.path.join(checkpoint_dir,
                         f"q_encoder_epoch{epoch+1}.pt")
        )
        torch.save(
            model.state_dict(),
            os.path.join(checkpoint_dir,
                         f"bart_epoch{epoch+1}.pt")
        )

        # save best checkpoint
        if val_acc_3way > best_val_acc:
            best_val_acc = val_acc_3way
            torch.save(
                q_encoder.state_dict(),
                os.path.join(checkpoint_dir, "q_encoder_best.pt")
            )
            torch.save(
                model.state_dict(),
                os.path.join(checkpoint_dir, "bart_best.pt")
            )
            print(f"  New best saved — acc: {best_val_acc:.1%}")

    print(f"\nTraining complete. Best val acc: {best_val_acc:.1%}")
    return best_val_acc


# ── CLI entry point ────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    import random
    import yaml
    import faiss

    from data import (
        apply_faiss_setup, load_fever, load_passages,
        move_index_to_gpu, get_train_val_split, path_setup
    )
    from model import load_dpr_encoder, load_bart

    # ── argument parser ────────────────────────────────────────
    parser = argparse.ArgumentParser(
        description="Fine-tune RAG on FEVER fact verification",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # data arguments
    data_group = parser.add_argument_group("data")
    data_group.add_argument(
        "--train_size", type=int, default=None,
        help="Number of training examples. None = full 228K dataset."
    )
    data_group.add_argument(
        "--val_size", type=int, default=500,
        help="Number of validation examples per epoch."
    )
    data_group.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for data shuffling."
    )

    # training arguments
    train_group = parser.add_argument_group("training")
    train_group.add_argument(
        "--epochs", type=int, default=3,
        help="Number of training epochs."
    )
    train_group.add_argument(
        "--lr", type=float, default=3e-5,
        help="Learning rate for AdamW optimizer."
    )
    train_group.add_argument(
        "--n_docs", type=int, default=5,
        help="Number of Wikipedia passages to retrieve per claim."
    )
    train_group.add_argument(
        "--batch_size", type=int, default=4,
        help="Training batch size (used for scheduler total_steps)."
    )
    train_group.add_argument(
        "--grad_accum", type=int, default=8,
        help="Gradient accumulation steps. "
             "Effective batch = batch_size × grad_accum."
    )
    train_group.add_argument(
        "--label_smoothing", type=float, default=0.1,
        help="Label smoothing epsilon for CrossEntropyLoss."
    )

    # resume arguments
    resume_group = parser.add_argument_group("resuming")
    resume_group.add_argument(
        "--start_epoch", type=int, default=0,
        help="Epoch number to start from when resuming training."
    )
    resume_group.add_argument(
        "--checkpoint_q", type=str, default=None,
        help="Path to query encoder checkpoint (.pt) to load before training."
    )
    resume_group.add_argument(
        "--checkpoint_bart", type=str, default=None,
        help="Path to BART checkpoint (.pt) to load before training."
    )

    # config and paths
    misc_group = parser.add_argument_group("config")
    misc_group.add_argument(
        "--config", type=str,
        default="fever/configs/fever_config.yaml",
        help="Path to YAML config file."
    )
    misc_group.add_argument(
        "--eval_only", action="store_true",
        help="Skip training and run validation only on the val set."
    )

    args = parser.parse_args()

    # ── load and override config ───────────────────────────────
    print("Loading config...")
    with open(args.config) as f:
        config = yaml.safe_load(f)

    # CLI args override config values
    config["training"]["learning_rate"]             = args.lr
    config["training"]["epochs"]                    = args.epochs
    config["training"]["batch_size"]                = args.batch_size
    config["training"]["gradient_accumulation_steps"] = args.grad_accum
    config["training"]["label_smoothing"]           = args.label_smoothing
    config["model"]["n_docs"]                       = args.n_docs
    config["data"]["seed"]                          = args.seed
    if args.train_size is not None:
        config["data"]["train_size"] = args.train_size

    print(f"  epochs:          {config['training']['epochs']}")
    print(f"  learning_rate:   {config['training']['learning_rate']}")
    print(f"  n_docs:          {config['model']['n_docs']}")
    print(f"  train_size:      {config['data']['train_size']} "
          f"(None = full dataset)")
    print(f"  val_size:        {args.val_size}")
    print(f"  grad_accum:      {config['training']['gradient_accumulation_steps']}")
    print(f"  effective_batch: "
          f"{config['training']['batch_size'] * config['training']['gradient_accumulation_steps']}")

    # ── paths and FAISS setup ──────────────────────────────────
    paths = path_setup()
    apply_faiss_setup()

    MAX_LENGTH = config["model"]["max_source_length"]

    # ── load data ──────────────────────────────────────────────
    print("\nLoading passages...")
    passages = load_passages(paths["passages_path"])
    print(f"Passages: {len(passages):,}")

    print("Loading FAISS index → GPU...")
    cpu_index = faiss.read_index(paths["faiss_index_path"])
    gpu_index = move_index_to_gpu(cpu_index)
    print(f"Index: {gpu_index.ntotal:,} vectors on GPU")

    print("Loading FEVER dataset...")
    dataset = load_fever(seed=config["data"]["seed"])
    train_data, val_data = get_train_val_split(
        dataset,
        train_size=config["data"]["train_size"],
        val_size=args.val_size,
        seed=config["data"]["seed"]
    )
    print(f"Train: {len(train_data):,} | Val: {len(val_data):,}")

    # ── load models ────────────────────────────────────────────
    print("\nLoading models...")
    q_encoder, q_tokenizer = load_dpr_encoder()
    model, bart_tokenizer  = load_bart()

    # load checkpoint if resuming
    if args.checkpoint_q and args.checkpoint_bart:
        print(f"Loading checkpoint from:")
        print(f"  {args.checkpoint_q}")
        print(f"  {args.checkpoint_bart}")
        q_encoder.load_state_dict(
            torch.load(args.checkpoint_q, map_location="cuda")
        )
        model.load_state_dict(
            torch.load(args.checkpoint_bart, map_location="cuda")
        )
        print("Checkpoint loaded successfully")
    elif args.checkpoint_q or args.checkpoint_bart:
        raise ValueError(
            "Provide both --checkpoint_q and --checkpoint_bart "
            "or neither."
        )

    # ── eval only mode ─────────────────────────────────────────
    if args.eval_only:
        print("\nEval-only mode — skipping training")
        acc_3way, acc_2way = evaluate(
            val_data, model, q_encoder,
            bart_tokenizer, q_tokenizer,
            gpu_index, passages, LABEL2ID,
            config, MAX_LENGTH
        )
        print(f"\nVal 3-way acc: {acc_3way:.1%}")
        print(f"Val 2-way acc: {acc_2way:.1%}")
        exit(0)

    # ── training ───────────────────────────────────────────────
    print(f"\nStarting training...")
    print(f"  start_epoch: {args.start_epoch}")
    print(f"  n_epochs:    {config['training']['epochs']}")

    best = train(
        model, q_encoder,
        bart_tokenizer, q_tokenizer,
        train_data, val_data,
        gpu_index, passages,
        config,
        checkpoint_dir=paths["checkpoint_dir"],
        start_epoch=args.start_epoch,
        n_epochs=config["training"]["epochs"],
        max_length=MAX_LENGTH
    )

    print(f"\nFinal best val 3-way accuracy: {best:.1%}")
    print(f"Paper target:                  72.5%")