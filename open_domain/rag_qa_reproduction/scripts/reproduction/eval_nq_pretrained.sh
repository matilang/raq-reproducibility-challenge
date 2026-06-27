#!/bin/bash

# ============================================================
# Evaluate released RAG-Sequence NQ checkpoint on full NQ-Open eval set
# No additional fine-tuning is performed.
# ============================================================

python src/evaluation/evaluate_rag_qa.py \
  --model_type sequence \
  --dataset nq_open \
  --index compressed \
  --n_docs 5 \
  --model_path facebook/rag-sequence-nq


# ============================================================
# Evaluate released RAG-Token NQ checkpoint on full NQ-Open eval set
# No additional fine-tuning is performed.
# ============================================================

python src/evaluation/evaluate_rag_qa.py \
  --model_type token \
  --dataset nq_open \
  --index compressed \
  --n_docs 5 \
  --model_path facebook/rag-token-nq
