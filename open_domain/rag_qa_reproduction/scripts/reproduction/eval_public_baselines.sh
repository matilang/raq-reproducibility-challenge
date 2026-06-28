#!/bin/bash

# ============================================================
# Evaluate public RAG-Sequence checkpoint on WebQuestions
# ============================================================

python src/evaluation/evaluate_rag_qa.py \
  --model_type sequence \
  --dataset web_questions \
  --index compressed \
  --n_docs 5 \
  --model_path facebook/rag-sequence-nq


# ============================================================
# Evaluate public RAG-Token checkpoint on WebQuestions
# ============================================================

python src/evaluation/evaluate_rag_qa.py \
  --model_type token \
  --dataset web_questions \
  --index compressed \
  --n_docs 5 \
  --model_path facebook/rag-token-nq


# ============================================================
# Evaluate public RAG-Sequence checkpoint on TriviaQA
# ============================================================

python src/evaluation/evaluate_rag_qa.py \
  --model_type sequence \
  --dataset trivia_qa \
  --dataset_config rc \
  --index compressed \
  --n_docs 5 \
  --model_path facebook/rag-sequence-nq


# ============================================================
# Evaluate public RAG-Token checkpoint on TriviaQA
# ============================================================

python src/evaluation/evaluate_rag_qa.py \
  --model_type token \
  --dataset trivia_qa \
  --dataset_config rc \
  --index compressed \
  --n_docs 5 \
  --model_path facebook/rag-token-nq
