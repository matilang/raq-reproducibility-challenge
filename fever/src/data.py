"""
data.py
FEVER dataset loading, preprocessing, and path utilities.
"""

import os
import sys
import json
import random
from collections import Counter
from datasets import load_dataset

# ------constants------------------------------

LABEL2ID = {
    "SUPPORTS"        : "0",
    "REFUTES"         : "1",
    "NOT ENOUGH INFO" : "2"
}

LABEL2ID_2WAY = {
    "SUPPORTS" : "0",
    "REFUTES"  : "1",
}

ID2LABEL = {v: k for k, v in LABEL2ID.items()}

# ------path setup------------------------------

def path_setup(notebook_dir=None):
    """
    Returns a dict of all project paths.
    Call with notebook_dir=os.getcwd() from a notebook,
    or leave None to auto-detect from this file's location.

    Args:
        notebook_dir: defaults to None
        
    Returns: paths(dict): directory with project paths
    """
    if notebook_dir is not None:
        fever_dir = os.path.abspath(
            os.path.join(notebook_dir, "..")
        )
    else:
        fever_dir = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..")
        )
    
    repo_root = os.path.abspath(os.path.join(fever_dir, ".."))
    
    paths = {
        "repo_root"       : repo_root,
        "fever_dir"       : fever_dir,
        "data_dir"        : os.path.join(fever_dir, "data"),
        "results_dir"     : os.path.join(fever_dir, "results"),
        "config_dir"      : os.path.join(fever_dir, "config"),
        "checkpoint_dir"  : os.path.join(fever_dir, "results", "checkpoints"),
        "passages_path"   : os.path.join(fever_dir, "data", "fever_passages.jsonl"),
        "embeddings_path" : os.path.join(fever_dir, "data", "fever_embeddings.npy"),
        "faiss_index_path": os.path.join(fever_dir, "data", "fever_faiss.index"),
    }
    
    for key in ["data_dir", "results_dir", "checkpoint_dir"]:
        os.makedirs(paths[key], exist_ok=True)

    return paths

# ------faiss setup------------------------------
def apply_faiss_setup():
    """
    Patch transformers' FAISS detection.
    Required because faiss-gpu-cu12 is not recognised by
    transformers' internal is_faiss_available() check.
    Must be called before any transformers component
    that requires FAISS is imported.
    """
    import transformers.utils.import_utils as import_utils
    import transformers.utils as tu

    if hasattr(import_utils.is_faiss_available, "cache_clear"):
        import_utils.is_faiss_available.cache_clear()

    import_utils._faiss_available = True
    import_utils.is_faiss_available = lambda: True
    tu.is_faiss_available = lambda: True

# ------loading Wikipedia passages from created file------------------------------
def load_passages(passages_path):
    """
    Load Wikipedia passages from jsonl file.
    Built from wikimedia/wikipedia 20231101.en dump,
    filtered to 23,733 articles referenced in FEVER dataset.
    See DECISIONS.md Decision 2 for why 2018 dump was not used.

    Args:
        passages_path: Path to data directory.
        
    Returns:
        passages: list of dicts with keys: id, title, text
    """
    
    passages = []
    with open(passages_path) as f:
        for line in f:
            passages.append(json.loads(line))
    return passages

# ------loading FEVER------------------------------
def load_fever(seed=42):
    """
    Load FEVER dataset from HuggingFace.
    Uses copenlu/fever_gold_evidence which is compatible
    with newer datasets versions (Parquet format).
    
    Args:
        seed (int, optional): Defaults to 42.
        
    Return:
        dataset: DatasetDict with train/validation/test splits
    """
    
    dataset = load_dataset("copenlu/fever_gold_evidence")
    
    return dataset
    
# ------loading FAISS index------------------------------
def load_faiss(index_path):
    """
    Load faiss index created during EDA with wikipedia passages.

    Args:
        index_path: Path to data directory.
    
    Returns:
        cpu_index:  
    """
    import faiss
    cpu_index = faiss.read_index(index_path)
    
    return cpu_index

# ------Prepare input to BART------------------------------
def prepare_bart_inputs(claim, retrieved_passages, bart_tokenizer, max_length):
    """
    Prepare tokenized BART inputs for one claim.
    Concatenates claim with each retrieved passage using
    the format from the RAG paper:
        "question: {claim} title: {title} context: {text}"

    Args:
        claim: str - claim from Wikipedia dataset
        retrieved_passages: list of dictionaries from search_index (FAISS) fcn
        bart_tokenizer: BartTokenizer
        max_length: maximal length of the passage -> const from config file
    
    Returns:
        dict with input_ids and attention_mask,
        both shape [K, max_length]
    """
    texts = []
    for p in retrieved_passages:
        text = (
            f"question: {claim} "
            f"title: {p['title']} "
            f"context: {p['text']}"
        )
        texts.append(text)
        
    encoded = bart_tokenizer(
        texts, 
        max_length=max_length,
        truncation=True, padding="max_length",
        return_tensors="pt"
    )
    
    return encoded
    
# ------Train/Validation split------------------------------
def get_train_val_split(dataset, train_size=None,
                         val_size=500, seed=42):
    """
    Prepare train and validation data lists.

    Args:
        dataset:    DatasetDict from load_fever()
        train_size: int or None — if None use full train set
        val_size:   int — number of validation examples
        seed:       random seed for reproducibility

    Returns:
        train_data: list of examples
        val_data:   list of examples
    """
    random.seed(seed)

    full_train = list(dataset["train"])
    random.shuffle(full_train)

    if train_size is not None:
        train_data = full_train[:train_size]
    else:
        train_data = full_train

    val_data = random.sample(
        list(dataset["validation"]), val_size
    )

    return train_data, val_data


# ------FEVER CLEAN FCN------------------------------
def clean_fever_title(title):
    """
    Convert FEVER-encoded article titles to normal Wikipedia titles.
    FEVER uses special bracket notation:
        -LRB- → (    -RRB- → )
        -LSB- → [    -RSB- → ]
        -LCB- → {    -RCB- → }
        _     → space
    """
    title = title.replace("-LRB-", "(")
    title = title.replace("-RRB-", ")")
    title = title.replace("-LSB-", "[")
    title = title.replace("-RSB-", "]")
    title = title.replace("-LCB-", "{")
    title = title.replace("-RCB-", "}")
    title = title.replace("_", " ")
    return title.strip()

# ------Weight labels------------------------------
def get_label_weights(train_data, device="cuda"):
    """
    Compute inverse frequency weights for each label.
    Used to handle class imbalance in CrossEntropyLoss.

    Args:
        train_data: list of examples from get_train_val_split()
        device:     torch device string

    Returns:
        torch.Tensor of shape [3] — one weight per label
    """
    import torch
    counts = Counter(ex["label"] for ex in train_data)
    total  = sum(counts.values())
    weights = torch.tensor([
        total / counts["SUPPORTS"],
        total / counts["REFUTES"],
        total / counts["NOT ENOUGH INFO"]
    ]).to(device)
    return weights

# ------Move index to gpu------------------------------
def move_index_to_gpu(cpu_index):
    """
    Move a FAISS CPU index to GPU for faster search.
    8.6x speedup measured on A40: 107ms → 12ms per query.

    Args:
        cpu_index: faiss CPU index

    Returns:
        gpu_index: faiss GPU index
    """
    import faiss
    res = faiss.StandardGpuResources()
    return faiss.index_cpu_to_gpu(res, 0, cpu_index)