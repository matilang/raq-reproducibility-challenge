#!/bin/bash

# ============================================================
# Fine-tune RAG-Sequence on WebQuestions
# ============================================================

python src/training/finetune_rag_qa.py \
  --model_type sequence \
  --dataset web_questions \
  --index compressed \
  --n_docs 5 \
  --epochs 2 \
  --batch_size 1 \
  --lr 1e-5 \
  --checkpoint_every 2000 \
  --output_dir checkpoints/rag_sequence_webquestions_2epoch


# ============================================================
# Fine-tune RAG-Token on WebQuestions
# ============================================================

python src/training/finetune_rag_qa.py \
  --model_type token \
  --dataset web_questions \
  --index compressed \
  --n_docs 5 \
  --epochs 2 \
  --batch_size 1 \
  --lr 1e-5 \
  --checkpoint_every 2000 \
  --output_dir checkpoints/rag_token_webquestions_2epoch
