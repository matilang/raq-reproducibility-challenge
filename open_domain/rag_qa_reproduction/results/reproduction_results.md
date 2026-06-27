# RAG QA Reproduction Results

This file summarizes the QA reproduction experiments for RAG-Sequence and RAG-Token on open-domain question answering datasets.

## Experiment summary

| Dataset | Model | Training setting | Evaluation setting | Notes |
|---|---|---|---|---|
| WebQuestions | RAG-Sequence | Public NQ checkpoint | Full evaluation split | Baseline evaluation |
| WebQuestions | RAG-Sequence | Fine-tuned for 2 epochs | Full evaluation split | DPR compressed index, n_docs=5 |
| WebQuestions | RAG-Token | Public NQ checkpoint | Full evaluation split | Baseline evaluation |
| WebQuestions | RAG-Token | Fine-tuned for 2 epochs | Full evaluation split | DPR compressed index, n_docs=5 |
| TriviaQA | RAG-Sequence | Public NQ checkpoint | Full evaluation split | Baseline evaluation |
| TriviaQA | RAG-Sequence | Fine-tuned on 30k examples for 1 epoch | Full evaluation split | Question encoder frozen |
| TriviaQA | RAG-Token | Public NQ checkpoint | Full evaluation split | Baseline evaluation |
| TriviaQA | RAG-Token | Fine-tuned on 30k examples for 1 epoch | Full evaluation split | Question encoder frozen |
| NQ-Open | RAG-Sequence | Released RAG-NQ checkpoint only | Full evaluation split | No additional fine-tuning |
| NQ-Open | RAG-Token | Released RAG-NQ checkpoint only | Full evaluation split | No additional fine-tuning |

## Final metrics

| Dataset | Experiment | Model | EM |
|---|---|---|---:|
| NQ-Open | Baseline / public checkpoint | RAG-Sequence | 34.97% |
| NQ-Open | Baseline / public checkpoint | RAG-Token | 33.66% |
| WebQuestions | Baseline / public checkpoint | RAG-Sequence | 15.06% |
| WebQuestions | Fine-tuned, 2 epochs | RAG-Sequence | 32.33% |
| WebQuestions | Baseline / public checkpoint | RAG-Token | 12.00% |
| WebQuestions | Fine-tuned, 2 epochs | RAG-Token | 29.20% |
| TriviaQA | Baseline / public checkpoint | RAG-Sequence | 33.57% |
| TriviaQA | Fine-tuned on 30k examples | RAG-Sequence | 38.20% |
| TriviaQA | Baseline / public checkpoint | RAG-Token | 30.60% |
| TriviaQA | Fine-tuned on 30k examples | RAG-Token | 35.20% |

## Main observations

Fine-tuning improved performance on both WebQuestions and TriviaQA.

On WebQuestions, RAG-Sequence improved from 15.06% EM to 32.33% EM, while RAG-Token improved from 12.00% EM to 29.20% EM.

On TriviaQA, RAG-Sequence improved from 33.57% EM to 38.20% EM, while RAG-Token improved from 30.60% EM to 35.20% EM.

For NQ-Open, the released public RAG-NQ checkpoints were evaluated without additional fine-tuning. RAG-Sequence achieved 34.97% EM and RAG-Token achieved 33.66% EM.
