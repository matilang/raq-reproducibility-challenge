import os
import re
import string
import random
import argparse
import numpy as np
import torch

from datasets import load_dataset
from transformers import (
    RagTokenizer,
    RagRetriever,
    RagSequenceForGeneration,
    RagTokenForGeneration,
)

parser = argparse.ArgumentParser()

parser.add_argument("--model_type", choices=["sequence", "token"], default="sequence")
parser.add_argument("--dataset", required=True)
parser.add_argument("--dataset_config", default=None)
parser.add_argument("--index", default="compressed")
parser.add_argument("--n_docs", type=int, default=5)
parser.add_argument("--max_examples", type=int, default=None)
parser.add_argument("--model_path", default=None)

args = parser.parse_args()

PROJECT_DIR = os.path.expanduser("~/RAG_reproduction")

CACHE_ROOT = os.path.join(
    PROJECT_DIR,
    f"wiki_dpr_{args.index}_cache_{args.model_type}"
)

CACHE_DIR = os.path.join(CACHE_ROOT, "hf_cache")
DATASETS_DIR = os.path.join(CACHE_ROOT, "hf_datasets")

os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(DATASETS_DIR, exist_ok=True)

os.environ["HF_HOME"] = CACHE_DIR
os.environ["HUGGINGFACE_HUB_CACHE"] = CACHE_DIR
os.environ["HF_DATASETS_CACHE"] = DATASETS_DIR
os.environ["TRANSFORMERS_CACHE"] = CACHE_DIR

if args.model_path is not None:
    MODEL_NAME = args.model_path
else:
    if args.model_type == "sequence":
        MODEL_NAME = "facebook/rag-sequence-nq"
    else:
        MODEL_NAME = "facebook/rag-token-nq"

DATASET_NAME = args.dataset
DATASET_CONFIG = args.dataset_config
N_DOCS = args.n_docs
MAX_EXAMPLES = args.max_examples
INDEX_NAME = args.index

#new configuration

#MAX_LENGTH = 20
#NUM_BEAMS = 4

MAX_LENGTH = 20
NUM_BEAMS = 4
SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


def normalize_answer(text):
    text = str(text).lower()
    text = "".join(ch for ch in text if ch not in string.punctuation)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = " ".join(text.split())
    return text


def exact_match(prediction, gold_answers):
    pred = normalize_answer(prediction)
    return any(pred == normalize_answer(gold) for gold in gold_answers)


def get_default_split(dataset_name):
    if dataset_name in ["web_questions", "curated_trec"]:
        return "test"
    return "validation"


def get_question_and_answers(example, dataset_name):
    if dataset_name == "nq_open":
        return example["question"], example["answer"]

    if dataset_name == "trivia_qa":
        question = example["question"]
        gold_answers = []

        if isinstance(example.get("answer"), dict):
            ans = example["answer"]

            if "value" in ans and ans["value"]:
                gold_answers.append(ans["value"])

            if "aliases" in ans and ans["aliases"]:
                gold_answers.extend(ans["aliases"])

        return question, gold_answers

    if dataset_name == "web_questions":
        question = example["question"]

        if "answers" in example:
            gold_answers = example["answers"]
        elif "answer" in example:
            gold_answers = example["answer"]
        else:
            gold_answers = []

        if isinstance(gold_answers, str):
            gold_answers = [gold_answers]

        return question, gold_answers

    if dataset_name == "curated_trec":
        question = example["question"]

        if "answers" in example:
            gold_answers = example["answers"]
        elif "answer" in example:
            gold_answers = example["answer"]
        else:
            gold_answers = []

        if isinstance(gold_answers, str):
            gold_answers = [gold_answers]

        return question, gold_answers

    raise ValueError(f"Unsupported dataset: {dataset_name}")


print("=" * 80)
print("RAG QA EVALUATION")
print("=" * 80)
print(f"Model type: {args.model_type}")
print(f"Model: {MODEL_NAME}")
print(f"Dataset: {DATASET_NAME}")
print(f"Dataset config: {DATASET_CONFIG}")
print(f"Index: {INDEX_NAME}")
print(f"N_DOCS: {N_DOCS}")
print(f"MAX_EXAMPLES: {MAX_EXAMPLES}")
print(f"Cache root: {CACHE_ROOT}")
print("=" * 80)

print("Loading tokenizer...")
tokenizer = RagTokenizer.from_pretrained(MODEL_NAME)

print(f"Loading retriever with {INDEX_NAME} Wiki DPR index...")
retriever = RagRetriever.from_pretrained(
    MODEL_NAME,
    index_name=INDEX_NAME
)

print(f"Loading RAG-{args.model_type} model...")
if args.model_type == "sequence":
    model = RagSequenceForGeneration.from_pretrained(
        MODEL_NAME,
        retriever=retriever
    )
else:
    model = RagTokenForGeneration.from_pretrained(
        MODEL_NAME,
        retriever=retriever
    )

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
model.eval()

print(f"Using device: {device}")

base_split = get_default_split(DATASET_NAME)

if MAX_EXAMPLES is None:
    split_name = base_split
else:
    split_name = f"{base_split}[:{MAX_EXAMPLES}]"

print(f"Loading dataset: {DATASET_NAME}")
print(f"Split: {split_name}")

if DATASET_CONFIG is not None:
    dataset = load_dataset(DATASET_NAME, DATASET_CONFIG, split=split_name)
else:
    dataset = load_dataset(DATASET_NAME, split=split_name)

print(f"Loaded {len(dataset)} examples.")

correct = 0
total = 0

for idx, example in enumerate(dataset):
    question, gold_answers = get_question_and_answers(example, DATASET_NAME)

    if not gold_answers:
        continue

    inputs = tokenizer.question_encoder(
        question,
        return_tensors="pt"
    )

    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    with torch.no_grad():
        generated_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            n_docs=N_DOCS,
            num_beams=NUM_BEAMS,
            max_length=MAX_LENGTH,
            early_stopping=True,
        )

    prediction = tokenizer.batch_decode(
        generated_ids,
        skip_special_tokens=True
    )[0]

    is_correct = exact_match(prediction, gold_answers)

    correct += int(is_correct)
    total += 1

    if total % 100 == 0:
        print(
            f"Processed {total} examples | "
            f"Running EM: {(correct / total) * 100:.2f}%"
        )

print("\n" + "#" * 80)
print("FINAL RESULTS")
print("#" * 80)
print(f"Model type: {args.model_type}")
print(f"Model: {MODEL_NAME}")
print(f"Dataset: {DATASET_NAME}")
print(f"Dataset config: {DATASET_CONFIG}")
print(f"Split: {split_name}")
print(f"Index: {INDEX_NAME}")
print(f"N_DOCS: {N_DOCS}")
print(f"NUM_BEAMS: {NUM_BEAMS}")
print(f"MAX_LENGTH: {MAX_LENGTH}")
print(f"Evaluated examples: {total}")
print(f"Correct: {correct}")

if total > 0:
    em = (correct / total) * 100
    print(f"Exact Match: {em:.2f}%")
else:
    print("No examples were evaluated.")