# RAG QA Reproduction

This repository contains a reproduction of Retrieval-Augmented Generation (RAG) for open-domain question answering.

The project evaluates RAG-Sequence and RAG-Token models on:

* Natural Questions Open (NQ-Open)
* WebQuestions
* TriviaQA

The experiments use Hugging Face RAG checkpoints with a DPR-based retriever and the compressed Wikipedia DPR index.

## Project structure

* `src/training/finetune_rag_qa.py`
* `src/evaluation/evaluate_rag_qa.py`
* `scripts/reproduction/`
* `results/reproduction_results.md`
* `docs/reproduction_protocol.md`

  rag_qa_reproduction/
├── src/
│   ├── training/
│   └── evaluation/
├── scripts/
│   └── reproduction/
├── results/
│   └── reproduction_results.md
├── docs/
│   └── reproduction_protocol.md
├── README.md
└── environment.txt

## Setup

Install the main dependencies:

```bash
pip install -r requirements.txt
```

The file `environment.txt` contains the full cluster environment snapshot. The file `requirements.txt` contains the cleaner minimal dependency list.

## Models

The experiments use:

* `facebook/rag-sequence-nq`
* `facebook/rag-token-nq`

Both models use DPR retrieval with the compressed Wikipedia DPR index.

## Experimental Configuration

### Retrieval and generation

All experiments used Hugging Face RAG models with a DPR-based retriever and the compressed Wiki-DPR index.

| Parameter                       | Setting                     |
| ------------------------------- | --------------------------- |
| Retriever                       | DPR-based RAG retriever     |
| Index                           | `compressed` Wiki-DPR index |
| Retrieved documents             | `n_docs=5`                  |
| Decoding                        | Beam search                 |
| Number of beams                 | `num_beams=4`               |
| Maximum generated answer length | `max_length=20`             |
| Evaluation metric               | Exact Match (EM)            |

### Dataset-specific settings

| Dataset      | Models                  | Split      | Training / evaluation setting                                           |
| ------------ | ----------------------- | ---------- | ----------------------------------------------------------------------- |
| NQ-Open      | RAG-Sequence, RAG-Token | validation | Public RAG-NQ checkpoints evaluated directly; no additional fine-tuning |
| WebQuestions | RAG-Sequence, RAG-Token | test       | Public checkpoint baseline and task-specific fine-tuning for 2 epochs   |
| TriviaQA     | RAG-Sequence, RAG-Token | validation | Public checkpoint baseline and fine-tuning on 30k examples for 1 epoch  |

### Fine-tuning settings

| Parameter           |           WebQuestions |                     TriviaQA |
| ------------------- | ---------------------: | ---------------------------: |
| Batch size          |                      1 |                            1 |
| Learning rate       |                 `1e-5` |                       `1e-6` |
| Epochs              |                      2 |                            1 |
| Retrieved documents |                      5 |                            5 |
| Optimizer           |                  AdamW |                        AdamW |
| Checkpointing       |       Every 2000 steps |             Every 2000 steps |
| Question encoder    | Trainable in main runs | Frozen in the stable 30k run |

The document encoder and compressed Wiki-DPR index were kept fixed. The main fine-tuning experiments updated the RAG model around the existing retrieval setup, while the stable TriviaQA run used a frozen question encoder to reduce memory pressure and improve training stability on the available university cluster resources.

### Evaluation details

Evaluation was performed using Exact Match after answer normalization. The normalization lowercases predictions and gold answers, removes punctuation and articles, and normalizes whitespace. For datasets with multiple gold answers or aliases, a prediction was counted as correct if it matched any available gold answer after normalization.

## Reproduction scripts

### Public checkpoint baselines

```bash
bash scripts/reproduction/eval_public_baselines.sh
```

### NQ-Open evaluation only

```bash
bash scripts/reproduction/eval_nq_pretrained.sh
```

### WebQuestions fine-tuning and evaluation

```bash
bash scripts/reproduction/train_webquestions.sh
bash scripts/reproduction/eval_webquestions.sh
```

### TriviaQA fine-tuning and evaluation

```bash
bash scripts/reproduction/train_triviaqa.sh
bash scripts/reproduction/eval_triviaqa.sh
```

## Results

| Dataset      | Experiment                 | Model        |     EM |
| ------------ | -------------------------- | ------------ | -----: |
| NQ-Open      | Public checkpoint          | RAG-Sequence | 34.97% |
| NQ-Open      | Public checkpoint          | RAG-Token    | 33.66% |
| WebQuestions | Public checkpoint          | RAG-Sequence | 15.06% |
| WebQuestions | Fine-tuned, 2 epochs       | RAG-Sequence | 32.33% |
| WebQuestions | Public checkpoint          | RAG-Token    | 12.00% |
| WebQuestions | Fine-tuned, 2 epochs       | RAG-Token    | 29.20% |
| TriviaQA     | Public checkpoint          | RAG-Sequence | 33.57% |
| TriviaQA     | Fine-tuned on 30k examples | RAG-Sequence | 38.20% |
| TriviaQA     | Public checkpoint          | RAG-Token    | 30.60% |
| TriviaQA     | Fine-tuned on 30k examples | RAG-Token    | 35.20% |

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
