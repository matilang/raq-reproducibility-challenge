#!/bin/bash

# ============================================================
# Fine-tune RAG-Sequence on TriviaQA using 30k examples
# ============================================================

python src/training/finetune_rag_qa.py \
  --model_type sequence \
  --dataset trivia_qa \
  --dataset_config rc \
  --index compressed \
  --n_docs 5 \
  --max_train_examples 30000 \
  --epochs 1 \
  --batch_size 1 \
  --lr 1e-6 \
  --freeze_question_encoder \
  --checkpoint_every 2000 \
  --output_dir checkpoints/rag_sequence_triviaqa_30k


# ============================================================
# Fine-tune RAG-Token on TriviaQA using 30k examples
# ============================================================

python src/training/finetune_rag_qa.py \
  --model_type token \
  --dataset trivia_qa \
  --dataset_config rc \
  --index compressed \
  --n_docs 5 \
  --max_train_examples 30000 \
  --epochs 1 \
  --batch_size 1 \
  --lr 1e-6 \
  --freeze_question_encoder \
  --checkpoint_every 2000 \
  --output_dir checkpoints/rag_token_triviaqa_30k
