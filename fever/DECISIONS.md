# FEVER Section — Technical Context & Decisions

## Decision 1 — FEVER dataset source

**What we use:** `copenlu/fever_gold_evidence`

**Why not the original `fever` dataset:**
The original dataset uses an old-style loading script (`fever.py`).
Newer `datasets` library no longer supports loading scripts:
`RuntimeError: Dataset scripts are no longer supported`
`trust_remote_code=True` also no longer supported.

**Why `copenlu/fever_gold_evidence` is better:**
- Compatible with new datasets versions (Parquet format)
- Includes gold evidence annotations directly in each example
- Evidence structure: `[article_title, sentence_id, sentence_text]`
- 228K train examples vs paper's 145K — includes all evidence
  annotations per claim, not just unique claims

---

## Decision 2 — Wikipedia index strategy

**What we do:** Build a FEVER-specific FAISS index from ~30K Wikipedia
articles referenced in the FEVER dataset.

**What the paper does:** Full Wikipedia index of 21M passages from
December 2018 dump.

**Why we cannot replicate the paper's exact approach:**

*Attempt 1 — RagRetriever with compressed index:*
```python
retriever = RagRetriever.from_pretrained(
    "facebook/rag-token-nq",
    index_name="compressed",
    use_dummy_dataset=False
)
```
Fails: `RuntimeError: Dataset scripts are no longer supported,
but found wiki_dpr.py`

*Attempt 2 — Wikipedia API (29K individual fetches):*
Estimated ~3 hours due to network round-trip per request.
Rejected as impractical.

*Attempt 3 — Full wikimedia/wikipedia download:*
Unnecessary — we only need ~30K of 6.7M articles.

*Attempt 4 — 2018 Wikipedia dump:*
Not accessible via HuggingFace datasets in compatible format.

**Chosen approach — Stream and filter wikimedia/wikipedia:**
```python
wiki_dataset = load_dataset(
    "wikimedia/wikipedia",
    "20231101.en",
    split="train",
    streaming=True
)
```
Stream through all 6,407,814 articles, keep only FEVER articles.

**Trade-offs vs paper:**
| Aspect | Paper | Ours |
|--------|-------|-------|
| Wikipedia dump | December 2018 | November 2023 |
| Index size | 21M passages | ~574K passages |
| Coverage | All of Wikipedia | FEVER articles only |

**Why the trade-offs are acceptable:**
1. FEVER claims were written from Wikipedia — relevant articles
   exist in both 2018 and 2023 dumps for most topics.
2. A domain-specific index is a legitimate research design choice.
   Searching 574K passages instead of 21M is faster with no loss
   for FEVER-specific queries since all relevant articles are included.
3. Noted in presentation as a methodological difference from paper.

---

## Decision 3 — FAISS detection patch

**Problem:** `transformers==4.56.2` uses `lru_cache` on
`is_faiss_available()`. Even though `faiss-gpu-cu12` is installed
and importable, the check returns `False` because transformers
checks for package name `faiss`, `faiss-cpu`, or `faiss-gpu`
in installed packages — not whether the module is actually importable.

**Fix applied at start of every relevant notebook:**
```python
import faiss
import transformers.utils.import_utils as import_utils
import transformers.utils as tu

if hasattr(import_utils.is_faiss_available, "cache_clear"):
    import_utils.is_faiss_available.cache_clear()
import_utils._faiss_available = True
import_utils.is_faiss_available = lambda: True
tu.is_faiss_available = lambda: True
```

**Why not just install `faiss-cpu`:**
`faiss-gpu-cu12` is already installed and uses the GPU — installing
`faiss-cpu` alongside risks version conflicts and loses GPU acceleration.

---

## Decision 4 — Wikipedia title cleaning

**Problem:** FEVER encodes article titles with special notation:
- `-LRB-` = `(`
- `-RRB-` = `)`
- `-LSB-` = `[`
- `-RSB-` = `]`
- `-LCB-` = `{`
- `-RCB-` = `}`
- `_` = space

**Cleaning function used throughout project:**
```python
def clean_fever_title(title):
    title = title.replace("-LRB-", "(")
    title = title.replace("-RRB-", ")")
    title = title.replace("-LSB-", "[")
    title = title.replace("-RSB-", "]")
    title = title.replace("-LCB-", "{")
    title = title.replace("-RCB-", "}")
    title = title.replace("_", " ")
    return title.strip()
```

**Matching strategy:** Case-insensitive lowercase comparison.
Wikipedia titles are case-sensitive but FEVER title formatting
is inconsistent. Lowercase matching recovers more articles at
the cost of potential false positives (rare in practice since
article titles are usually unique even case-insensitively).

---

## Decision 5 — Wikipedia title coverage (79.8%)

**Result:** 23,733 / 29,756 articles found → 574,197 passages

**Why 6,023 articles are missing from the 2023 dump:**
Articles renamed or restructured between 2017 (FEVER creation)
and 2023 Wikipedia dump. Confirmed examples:

| FEVER title | Status in 2023 dump |
|-------------|---------------------|
| `Kareena Kapoor` | Renamed to `Kareena Kapoor Khan` |
| `Charlotte Brontë` | Only biography page found |
| `Nicolas Cage` | Only filmography/awards found |
| `North Korea` | Only sub-topic articles found |
| `Halle Berry` | Only award lists found |
| `1989 (Taylor Swift album)` | Not found under this title |

**Why we did not attempt recovery:**
Building a mapping table for 6,023 renamed articles is out of scope.
Related articles for missing topics ARE present in our index and
provide partial coverage. Limitation is noted in the presentation.

**Impact on results:**
- Claims about missing articles may have lower retrieval recall
- BART parametric knowledge may compensate for major well-known topics
- 574K passages provides strong coverage for majority of FEVER claims

---

## Data files (fever/data/)
| File | Description |
|------|-------------|
| `fever_passages.jsonl` | 574,197 passages from 23,733 articles |
| `not_found_articles.json` | 6,023 articles not matched in 2023 dump |

---

## Environment
| Component | Version / Spec |
|-----------|----------------|
| GPU | NVIDIA A40 (46GB VRAM) |
| CPU RAM | 472GB |
| Disk | 213TB available (shared cluster) |
| PyTorch | 2.8.0+cu128 |
| CUDA | 12.8 |
| Transformers | 4.56.2 |
| FAISS | faiss-gpu-cu12 1.14.1 |
| datasets | latest |

---

## Label mapping
```python
LABEL2ID = {
    "SUPPORTS":        "0",
    "REFUTES":         "1",
    "NOT ENOUGH INFO": "2"
}
LABEL2ID_2WAY = {
    "SUPPORTS": "0",
    "REFUTES":  "1"
}
```

---

## Hyperparameters (paper-aligned, from fever_config.yaml)
```yaml
learning_rate: 3.0e-5
epochs: 3
warmup_steps: 500
gradient_accumulation_steps: 8
weight_decay: 0.001
adam_epsilon: 1.0e-8
max_grad_norm: 0.1
label_smoothing: 0.1
batch_size: 4
n_docs: 5
max_source_length: 300
max_target_length: 25
fp16: true
```