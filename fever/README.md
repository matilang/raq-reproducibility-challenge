# FEVER Fact Verification — RAG Reproduction

Reproduction of the FEVER fact-verification task from [Lewis et al. (2021)](https://arxiv.org/abs/2005.11401) using Retrieval-Augmented Generation.

**Author:** Mateusz Łangowski  
**Part of:** RAG Reproducibility Challenge — group project

---

## Task

Given a claim (e.g. *"Barack Obama was born in Hawaii"*), classify it as one of:

| Label | Meaning |
|---|---|
| SUPPORTS | Evidence retrieved from Wikipedia supports the claim |
| REFUTES | Evidence retrieved from Wikipedia refutes the claim |
| NOT ENOUGH INFO | No relevant evidence found |

---

## Results

| Setting | Ours | Paper (RAG-Token) |
|---|---|---|
| 3-way accuracy (SUPPORTS / REFUTES / NEI) | **66.2%** | 72.5% |
| 2-way accuracy (SUPPORTS / REFUTES only) | **75.1%** | 89.5% |

The 6.3 point gap is explained by three quantifiable factors:
- Wikipedia index covers 79.8% of FEVER articles (23,733 / 29,756) — 2023 dump used instead of paper's unavailable 2018 dump
- Article restructuring over 5 years affects high-profile topics
- Validation sample imbalance during training (48% NEI vs 40% true distribution)

See [`DECISIONS.md`](DECISIONS.md) for full technical context.

### Per-class breakdown

| Label | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| SUPPORTS | 0.714 | 0.531 | 0.609 | 6,456 |
| REFUTES | 0.749 | 0.664 | 0.704 | 4,889 |
| NOT ENOUGH INFO | 0.572 | 0.841 | 0.681 | 4,694 |

Notable finding: the model over-predicts NEI (84.1% recall, 57.2% precision),
defaulting to uncertainty when retrieval confidence is low. This is consistent
with pre-training retrieval recall of 65.4% top-1 on FEVER claims.

---

## Approach

- **Pretrained checkpoint:** `facebook/rag-token-nq` (Natural Questions)
- **Fine-tuning:** query encoder (BERT) + BART generator jointly; document encoder and FAISS index frozen throughout (RAG paper Section 2.4)
- **Index:** FEVER-specific FAISS index built from `wikimedia/wikipedia 20231101.en`, filtered to 23,733 articles referenced in FEVER dataset → 574,197 passages
- **Training:** full 228,277 training examples, 3 epochs, AdamW with linear warmup
- **Label mapping:** SUPPORTS → `"0"`, REFUTES → `"1"`, NEI → `"2"` — single output token makes RAG-Token and RAG-Sequence equivalent for FEVER

---

## Environment

| Component | Version |
|---|---|
| GPU | NVIDIA A40 (46GB VRAM) |
| PyTorch | 2.8.0+cu128 |
| CUDA | 12.8 |
| Transformers | 4.56.2 |
| FAISS | faiss-gpu-cu12 1.14.1 |

---

## Structure

```
fever/
├── configs/
│   └── fever_config.yaml          ← all hyperparameters
├── data/
│   ├── index_metadata.json        ← FAISS index build info
│   └── not_found_articles.json    ← 6,023 articles missing from 2023 dump
├── notebooks/
│   ├── 01_data_exploration.ipynb        ← FEVER dataset EDA
│   ├── 02_creation_of_passages.ipynb    ← Wikipedia index build
│   ├── 03_retrieval_verification.ipynb  ← pre-training retrieval recall
│   ├── 04_fine_tuning.ipynb             ← training epochs 1-2 (CPU FAISS)
│   ├── 04_fine_tuning_epoch3_gpu.ipynb  ← epoch 3 (GPU FAISS, 8.6x faster)
│   └── 05_evaluation.ipynb             ← full test set evaluation
├── results/
│   ├── dataset_summary.json
│   ├── label_distribution.png
│   ├── retrieval_recall.json
│   ├── retrieval_verification.png
│   └── final_evaluation.png
├── src/
│   ├── data.py       ← dataset loading, label mapping, path utilities
│   ├── model.py      ← DPR retrieval, BART forward pass, marginalization
│   ├── train.py      ← training loop with CLI
│   └── evaluate.py   ← standalone evaluation script
├── DECISIONS.md      ← technical decisions and known limitations
└── setup_env.py      ← cluster environment setup
```

---

## Quick start

### 1. Clone the repo and navigate to the fever section

```bash
git clone https://github.com/matilang/raq-reproducibility-challenge
cd raq-reproducibility-challenge
```

### 2. Install dependencies

```bash
pip install torch transformers datasets faiss-gpu-cu12 \
            evaluate accelerate sentencepiece scikit-learn \
            matplotlib seaborn pyyaml
```

> **Note:** On the JupyterLab cluster, run `%run fever/setup_env.py`
> at the start of each session — the cluster resets pip installs on restart.

### 3. Build the Wikipedia passage index

This step downloads and filters Wikipedia articles referenced in FEVER,
encodes them with DPR, and builds a FAISS index. Run once — takes ~30 min on A40.

```bash
# run notebook 02 end-to-end
jupyter nbconvert --to notebook --execute \
    fever/notebooks/02_creation_of_passages.ipynb
```

Or open `02_creation_of_passages.ipynb` in JupyterLab and run all cells.

This produces:
- `fever/data/fever_passages.jsonl` — 574,197 Wikipedia passages
- `fever/data/fever_embeddings.npy` — DPR embeddings (1.64 GB)
- `fever/data/fever_faiss.index`    — FAISS index (1.64 GB)

### 4. Fine-tune RAG on FEVER

**Via CLI (recommended):**

```bash
# full reproduction run — 3 epochs on full 228K training set (~10 hrs/epoch)
python fever/src/train.py

# debug run — confirm pipeline works in minutes
python fever/src/train.py --train_size 500 --epochs 1

# resume from checkpoint
python fever/src/train.py \
  --start_epoch 2 \
  --checkpoint_q fever/results/checkpoints/q_encoder_epoch2_gpu.pt \
  --checkpoint_bart fever/results/checkpoints/bart_epoch2_gpu.pt
```

**Via notebook:**  
Open `fever/notebooks/04_fine_tuning_epoch3_gpu.ipynb` in JupyterLab.

### 5. Evaluate on the test set

**Via CLI:**

```bash
python fever/src/evaluate.py
```

**Via notebook:**  
Open `fever/notebooks/05_evaluation.ipynb` in JupyterLab and run all cells.
Produces confusion matrix and accuracy comparison plots in `fever/results/`.

**Check a checkpoint without retraining:**

```bash
python fever/src/train.py \
  --checkpoint_q fever/results/checkpoints/q_encoder_best_gpu.pt \
  --checkpoint_bart fever/results/checkpoints/bart_best_gpu.pt \
  --eval_only
```

---

## Reproducing with a subset (faster)

As noted by the course instructor, a representative subset is acceptable.
To reproduce on 10K examples instead of the full 228K:

```bash
python fever/src/train.py --train_size 10000 --epochs 3
```

Expected accuracy: 55–62% 3-way (lower than full run due to less training data).

---

## Key hyperparameters

See `fever/configs/fever_config.yaml` for all values. Key ones:

| Parameter | Value | Source |
|---|---|---|
| Learning rate | 3e-5 | Official HuggingFace RAG scripts |
| Epochs | 3 | Official HuggingFace RAG scripts |
| Warmup steps | 500 | Official HuggingFace RAG scripts |
| Weight decay | 0.001 | Official HuggingFace RAG scripts |
| Label smoothing | 0.1 | Official HuggingFace RAG scripts |
| n_docs | 5 | RAG paper |
| Max source length | 300 | RAG paper |
| Gradient accumulation | 8 | Official HuggingFace RAG scripts |

> **Note on hyperparameter sources:** The RAG paper (Lewis et al., 2021)
> specifies only the optimizer (Adam) and loss function. Remaining
> hyperparameters follow the official HuggingFace RAG fine-tuning scripts.
> Gradient accumulation was added due to single-example processing
> constraint — the paper trained on 8×32GB V100s with true batch processing.

---

## Known limitations and differences from paper

| Aspect | Paper | This reproduction |
|---|---|---|
| Wikipedia dump | December 2018 | November 2023 |
| Index size | 21M passages | 574K passages (FEVER articles only) |
| Article coverage | 100% | 79.8% (6,023 articles renamed/restructured) |
| Optimizer | Adam | AdamW (established best practice) |
| Scheduler | Not specified | Linear warmup + decay |
| Training time | Not reported | ~10 hrs/epoch (batch_size=1, no batching) |

See [`DECISIONS.md`](DECISIONS.md) for full explanation of each difference.