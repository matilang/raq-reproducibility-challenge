# FEVER Fact Verification — RAG Reproduction

Reproduction of the FEVER fact-verification task from [Lewis et al. (2021)](https://arxiv.org/abs/2005.11401) using Retrieval-Augmented Generation.

**Author:** Mateusz Łangowski

## Task

Given a claim (e.g. *"Barack Obama was born in Hawaii"*), classify it as one of:

| Label | Meaning |
|---|---|
| SUPPORTS | Evidence retrieved from Wikipedia supports the claim |
| REFUTES | Evidence retrieved from Wikipedia refutes the claim |
| NOT ENOUGH INFO | No relevant evidence found |

## Target metrics (RAG-Token, from paper)

| Setting | Accuracy |
|---|---|
| 3-way (SUPPORTS / REFUTES / NEI) | 72.5% |
| 2-way (SUPPORTS / REFUTES only) | 89.5% |

## Approach

- **Checkpoint:** `facebook/rag-token-nq` (pretrained on Natural Questions)
- **Fine-tuning:** query encoder (BERT) + BART generator only; document encoder and FAISS index kept frozen
- **Index:** full Wikipedia dump (21M passages, 768-dim DPR vectors)
- **Training subset:** 10–20K examples from FEVER's ~145K training set
- **Label mapping:** each label mapped to a single output token so RAG-Token and RAG-Sequence behave identically

## Structure

```
fever/
  configs/
    fever_config.yaml
  data/
    fever_train.jsonl
    fever_dev.jsonl
  notebooks/
    01_data_exploration.ipynb
    02_creation_of_passages.ipynb
    03_retrieval_verification.ipynb
    04_finetuning.ipynb
    05_evaluation.ipynb
  src/
    data.py        # dataset loading and label mapping
    train.py       # fine-tuning loop
    evaluate.py    # metric computation
    model.py       # model loading, retrieval and forward pass
  configs/
    fever_config.yaml
  results/
    dataset_summary.json
    label_distribution.png
    retrieval_recall.json
    retrieval_verification.png
```

## Quick start

> Work in progress — full instructions will be added after fine-tuning is complete.
# cluster setup confirmed
