#!/bin/bash

# ============================================================
# Evaluate fine-tuned RAG-Sequence on WebQuestions
# ============================================================

python src/evaluation/evaluate_rag_qa.py \
  --model_type sequence \
  --dataset web_questions \
  --index compressed \
  --n_docs 5 \
  --model_path checkpoints/rag_sequence_webquestions_2epoch


# ============================================================
# Evaluate fine-tuned RAG-Token on WebQuestions
# ============================================================

python src/evaluation/evaluate_rag_qa.py \
  --model_type token \
  --dataset web_questions \
  --index compressed \
  --n_docs 5 \
  --model_path checkpoints/rag_token_webquestions_2epoch
