import os
import argparse
import random
import time
import numpy as np
import torch

parser = argparse.ArgumentParser()

parser.add_argument("--model_type", choices=["sequence", "token"], default="sequence")
parser.add_argument("--dataset", required=True)
parser.add_argument("--dataset_config", default=None)
parser.add_argument("--index", default="compressed")
parser.add_argument("--n_docs", type=int, default=5)
parser.add_argument("--max_train_examples", type=int, default=None)
parser.add_argument("--epochs", type=int, default=1)
parser.add_argument("--batch_size", type=int, default=1)
parser.add_argument("--lr", type=float, default=3e-6)
parser.add_argument("--model_path", default=None)
parser.add_argument("--output_dir", required=True)
parser.add_argument("--freeze_question_encoder", action="store_true")

# NEW
parser.add_argument("--checkpoint_every", type=int, default=2000)
parser.add_argument("--log_every", type=int, default=100)

args = parser.parse_args()

PROJECT_DIR = os.path.expanduser("~/RAG_reproduction")
CACHE_ROOT = os.path.join(PROJECT_DIR, f"wiki_dpr_{args.index}_cache")
CACHE_DIR = os.path.join(CACHE_ROOT, "hf_cache")
DATASETS_DIR = os.path.join(CACHE_ROOT, "hf_datasets")

os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(DATASETS_DIR, exist_ok=True)
os.makedirs(args.output_dir, exist_ok=True)

os.environ["HF_HOME"] = CACHE_DIR
os.environ["HUGGINGFACE_HUB_CACHE"] = CACHE_DIR
os.environ["HF_DATASETS_CACHE"] = DATASETS_DIR
os.environ["TRANSFORMERS_CACHE"] = CACHE_DIR

from datasets import load_dataset
from torch.utils.data import DataLoader
from torch.optim import AdamW

from transformers import (
    RagTokenizer,
    RagRetriever,
    RagSequenceForGeneration,
    RagTokenForGeneration,
)

SEED = 42
MAX_INPUT_LENGTH = 64
MAX_TARGET_LENGTH = 32

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

if args.model_path is not None:
    MODEL_NAME = args.model_path
else:
    MODEL_NAME = (
        "facebook/rag-sequence-nq"
        if args.model_type == "sequence"
        else "facebook/rag-token-nq"
    )


def get_question_and_answer(example, dataset_name):
    if dataset_name == "nq_open":
        answers = example["answer"]
        if isinstance(answers, list):
            if len(answers) == 0:
                return example["question"], None
            return example["question"], answers[0]
        return example["question"], answers

    if dataset_name == "trivia_qa":
        question = example["question"]
        answer = example["answer"]

        if isinstance(answer, dict):
            if "value" in answer and answer["value"]:
                return question, answer["value"]
            if "aliases" in answer and answer["aliases"]:
                return question, answer["aliases"][0]

        return question, None

    if dataset_name in ["web_questions", "curated_trec"]:
        question = example["question"]
        answers = example.get("answers", example.get("answer", []))

        if isinstance(answers, list):
            if len(answers) == 0:
                return question, None
            return question, answers[0]

        return question, answers

    raise ValueError(f"Unsupported dataset: {dataset_name}")


print("=" * 80)
print("RAG QA FINE-TUNING WITH CHECKPOINTS")
print("=" * 80)
print(f"Model type: {args.model_type}")
print(f"Model path: {MODEL_NAME}")
print(f"Dataset: {args.dataset}")
print(f"Dataset config: {args.dataset_config}")
print(f"Index: {args.index}")
print(f"N_DOCS: {args.n_docs}")
print(f"MAX_TRAIN_EXAMPLES: {args.max_train_examples}")
print(f"Epochs: {args.epochs}")
print(f"Batch size: {args.batch_size}")
print(f"Learning rate: {args.lr}")
print(f"Freeze question encoder: {args.freeze_question_encoder}")
print(f"Checkpoint every: {args.checkpoint_every} steps")
print(f"Output dir: {args.output_dir}")
print(f"Cache root: {CACHE_ROOT}")
print("=" * 80)

print("Loading tokenizer...")
tokenizer = RagTokenizer.from_pretrained(MODEL_NAME)

print(f"Loading retriever with {args.index} Wiki DPR index...")
retriever = RagRetriever.from_pretrained(
    MODEL_NAME,
    index_name=args.index,
)

print(f"Loading RAG-{args.model_type} model...")
if args.model_type == "sequence":
    model = RagSequenceForGeneration.from_pretrained(
        MODEL_NAME,
        retriever=retriever,
    )
else:
    model = RagTokenForGeneration.from_pretrained(
        MODEL_NAME,
        retriever=retriever,
    )

if args.freeze_question_encoder:
    print("Freezing question encoder / retriever parameters...")
    for param in model.rag.question_encoder.parameters():
        param.requires_grad = False
else:
    print("Question encoder is trainable.")

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
model.train()

print(f"Using device: {device}")

split_name = (
    "train"
    if args.max_train_examples is None
    else f"train[:{args.max_train_examples}]"
)

print(f"Loading dataset: {args.dataset}")
print(f"Split: {split_name}")

if args.dataset_config is not None:
    dataset = load_dataset(args.dataset, args.dataset_config, split=split_name)
else:
    dataset = load_dataset(args.dataset, split=split_name)

print(f"Loaded {len(dataset)} training examples.")


def collate_fn(batch):
    questions = []
    answers = []

    for example in batch:
        question, answer = get_question_and_answer(example, args.dataset)

        if question is None or answer is None:
            continue

        questions.append(str(question))
        answers.append(str(answer))

    if len(questions) == 0:
        return None

    question_inputs = tokenizer.question_encoder(
        questions,
        padding=True,
        truncation=True,
        max_length=MAX_INPUT_LENGTH,
        return_tensors="pt",
    )

    answer_inputs = tokenizer.generator(
        answers,
        padding=True,
        truncation=True,
        max_length=MAX_TARGET_LENGTH,
        return_tensors="pt",
    )

    labels = answer_inputs["input_ids"].clone()
    labels[labels == tokenizer.generator.pad_token_id] = -100

    return {
        "input_ids": question_inputs["input_ids"],
        "attention_mask": question_inputs["attention_mask"],
        "labels": labels,
    }


dataloader = DataLoader(
    dataset,
    batch_size=args.batch_size,
    shuffle=True,
    collate_fn=collate_fn,
)

optimizer = AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=args.lr,
)

global_step = 0
start_time = time.time()


def save_checkpoint(step):
    ckpt_dir = os.path.join(args.output_dir, f"checkpoint_step_{step}")
    os.makedirs(ckpt_dir, exist_ok=True)

    print(f"\n[CHECKPOINT] Saving checkpoint at step {step}...")
    model.save_pretrained(ckpt_dir)
    tokenizer.save_pretrained(ckpt_dir)
    retriever.save_pretrained(ckpt_dir)
    print(f"[CHECKPOINT] Saved to: {ckpt_dir}\n")


try:
    for epoch in range(args.epochs):
        print("\n" + "=" * 80)
        print(f"Epoch {epoch + 1}/{args.epochs}")
        print("=" * 80)

        total_loss = 0.0
        valid_steps = 0

        for step, batch in enumerate(dataloader):
            if batch is None:
                continue

            optimizer.zero_grad()

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                n_docs=args.n_docs,
            )

            loss = outputs.loss
            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                filter(lambda p: p.requires_grad, model.parameters()),
                1.0,
            )

            optimizer.step()

            total_loss += loss.item()
            valid_steps += 1
            global_step += 1

            if global_step % args.log_every == 0:
                elapsed_h = (time.time() - start_time) / 3600

                if torch.cuda.is_available():
                    mem = torch.cuda.memory_allocated() / 1024**3
                    max_mem = torch.cuda.max_memory_allocated() / 1024**3
                    print(
                        f"Step {global_step} | "
                        f"Epoch step {step + 1}/{len(dataloader)} | "
                        f"Loss {total_loss / valid_steps:.4f} | "
                        f"Time {elapsed_h:.2f}h | "
                        f"GPU mem {mem:.2f}GB | "
                        f"Max GPU mem {max_mem:.2f}GB"
                    )
                else:
                    print(
                        f"Step {global_step} | "
                        f"Epoch step {step + 1}/{len(dataloader)} | "
                        f"Loss {total_loss / valid_steps:.4f} | "
                        f"Time {elapsed_h:.2f}h"
                    )

            if global_step % args.checkpoint_every == 0:
                save_checkpoint(global_step)

        print(f"Epoch {epoch + 1} average loss: {total_loss / max(valid_steps, 1):.4f}")

except KeyboardInterrupt:
    print("\n[INTERRUPTED] Training interrupted. Saving emergency checkpoint...")
    save_checkpoint(global_step)
    raise

except Exception as e:
    print(f"\n[ERROR] Training failed at step {global_step}: {repr(e)}")
    print("[ERROR] Saving emergency checkpoint...")
    save_checkpoint(global_step)
    raise

print("\n" + "#" * 80)
print("SAVING FINAL FINE-TUNED MODEL")
print("#" * 80)

model.save_pretrained(args.output_dir)
tokenizer.save_pretrained(args.output_dir)
retriever.save_pretrained(args.output_dir)

elapsed_h = (time.time() - start_time) / 3600
print(f"Saved final checkpoint to: {args.output_dir}")
print(f"Total training time: {elapsed_h:.2f} hours")
print(f"Total steps: {global_step}")