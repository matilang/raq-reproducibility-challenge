#!/bin/bash

# ============================================================
# Evaluate 30k fine-tuned RAG-Sequence on full TriviaQA eval set
# ============================================================

python src/evaluation/evaluate_rag_qa.py \
  --model_type sequence \
  --dataset trivia_qa \
  --dataset_config rc \
  --index compressed \
  --n_docs 5 \
  --model_path checkpoints/rag_sequence_triviaqa_30k


# ============================================================
# Evaluate 30k fine-tuned RAG-Token on full TriviaQA eval set
# ============================================================

python src/evaluation/evaluate_rag_qa.py \
  --model_type token \
  --dataset trivia_qa \
  --dataset_config rc \
  --index compressed \
  --n_docs 5 \
  --model_path checkpoints/rag_token_triviaqa_30k
