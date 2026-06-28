# Reproduction Protocol

This document describes the setup used for the RAG question answering reproduction experiments.

## Objective

The goal of this reproduction is to evaluate Retrieval-Augmented Generation (RAG) for open-domain question answering using RAG-Sequence and RAG-Token models.

The reproduction focuses on three QA datasets:

- WebQuestions
- TriviaQA
- Natural Questions Open (NQ-Open)

## Models

The experiments use Hugging Face RAG checkpoints:

- facebook/rag-sequence-nq
- facebook/rag-token-nq

Both models use a DPR-based retriever with the compressed Wikipedia DPR index.

## Retrieval setup

All experiments use:

- Index: compressed
- Number of retrieved documents: n_docs=5

## WebQuestions setup

For WebQuestions, both RAG-Sequence and RAG-Token were evaluated as public-checkpoint baselines and then fine-tuned for 2 epochs.

Main settings:

- Batch size: 1
- Learning rate: 1e-5
- Number of retrieved documents: 5
- DPR compressed index

## TriviaQA setup

For TriviaQA, both RAG-Sequence and RAG-Token were evaluated as public-checkpoint baselines and then fine-tuned on 30,000 examples for 1 epoch.

Main settings:

- Dataset configuration: rc
- Maximum training examples: 30,000
- Batch size: 1
- Learning rate: 1e-6
- Question encoder frozen
- Number of retrieved documents: 5
- DPR compressed index

The question encoder was frozen to reduce training cost and improve stability on the university cluster.

## NQ-Open setup

For NQ-Open, no additional fine-tuning was performed. The released RAG-NQ checkpoints were evaluated directly.

This makes the NQ experiment an evaluation-only baseline.

## Cluster execution

The experiments were executed on a university computing cluster. The fine-tuning script includes periodic checkpointing and logging to support long-running jobs.

Model checkpoints and Hugging Face cache files are not included in this repository because of their large size. The scripts required to reproduce the checkpoints are provided instead.

## Reproduction scripts

The main scripts are located in:

scripts/reproduction/

Available scripts:

- train_webquestions.sh
- eval_webquestions.sh
- train_triviaqa.sh
- eval_triviaqa.sh
- eval_nq_pretrained.sh
- eval_public_baselines.sh

## Notes

The reproduction does not aim to exactly match the full compute scale of the original paper. Instead, it provides a controlled reproduction using available university cluster resources and documents the experimental setup clearly.
