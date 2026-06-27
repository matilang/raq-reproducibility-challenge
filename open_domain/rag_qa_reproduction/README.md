# RAG QA Reproduction

This repository contains a reproduction of Retrieval-Augmented Generation (RAG) for open-domain question answering.

The project evaluates RAG-Sequence and RAG-Token models on:

- Natural Questions Open (NQ-Open)
- WebQuestions
- TriviaQA

The experiments use Hugging Face RAG checkpoints with a DPR-based retriever and the compressed Wikipedia DPR index.

## Project structure

- src/training/finetune_rag_qa.py
- src/evaluation/evaluate_rag_qa.py
- scripts/reproduction/
- results/reproduction_results.md
- docs/reproduction_protocol.md

## Setup

Install the main dependencies:

pip install -r requirements.txt

The file environment.txt contains the full cluster environment snapshot. The file requirements.txt contains the cleaner minimal dependency list.

## Models

The experiments use:

- facebook/rag-sequence-nq
- facebook/rag-token-nq

Both models use DPR retrieval with the compressed Wikipedia DPR index.

## Reproduction scripts

### Public checkpoint baselines

bash scripts/reproduction/eval_public_baselines.sh

### NQ-Open evaluation only

bash scripts/reproduction/eval_nq_pretrained.sh

### WebQuestions fine-tuning and evaluation

bash scripts/reproduction/train_webquestions.sh
bash scripts/reproduction/eval_webquestions.sh

### TriviaQA fine-tuning and evaluation

bash scripts/reproduction/train_triviaqa.sh
bash scripts/reproduction/eval_triviaqa.sh

## Results

| Dataset | Experiment | Model | EM |
|---|---|---|---:|
| NQ-Open | Public checkpoint | RAG-Sequence | 34.97% |
| NQ-Open | Public checkpoint | RAG-Token | 33.66% |
| WebQuestions | Public checkpoint | RAG-Sequence | 15.06% |
| WebQuestions | Fine-tuned, 2 epochs | RAG-Sequence | 32.33% |
| WebQuestions | Public checkpoint | RAG-Token | 12.00% |
| WebQuestions | Fine-tuned, 2 epochs | RAG-Token | 29.20% |
| TriviaQA | Public checkpoint | RAG-Sequence | 33.57% |
| TriviaQA | Fine-tuned on 30k examples | RAG-Sequence | 38.20% |
| TriviaQA | Public checkpoint | RAG-Token | 30.60% |
| TriviaQA | Fine-tuned on 30k examples | RAG-Token | 35.20% |

## Main observations

Fine-tuning improved performance on both WebQuestions and TriviaQA.

On WebQuestions, RAG-Sequence improved from 15.06% EM to 32.33% EM, while RAG-Token improved from 12.00% EM to 29.20% EM.

On TriviaQA, RAG-Sequence improved from 33.57% EM to 38.20% EM, while RAG-Token improved from 30.60% EM to 35.20% EM.

For NQ-Open, the released RAG-NQ checkpoints were evaluated without additional fine-tuning.

## Checkpoints and cache files

Model checkpoints, Hugging Face cache files, and DPR index cache files are not included in this repository because of their large size.

The repository provides the code and scripts needed to reproduce the checkpoints and evaluations.

## Notes

This reproduction was executed on a university computing cluster. The training script includes checkpointing and logging to support long-running jobs.

The reproduction is not intended to exactly match the full compute scale of the original paper. Instead, it documents a controlled reproduction using available academic computing resources.
