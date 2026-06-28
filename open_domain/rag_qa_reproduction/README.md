# RAG QA Reproduction

This folder contains the open-domain question answering reproduction part of the **RAG Reproducibility Challenge** group project.

The goal of this part is to reproduce and evaluate Retrieval-Augmented Generation (RAG) for open-domain question answering using Hugging Face RAG checkpoints, DPR-based retrieval, and the compressed Wikipedia DPR index.

**Author:** Noor Yasser
**Part of:** RAG Reproducibility Challenge — group project
**Related extension:** The Graph-RAG / Hybrid Graph-RAG extension is implemented separately in [`../graphrag_extension/`](../graphrag_extension/).

---

## Task

Open-domain question answering requires a model to answer natural language questions without being given a gold evidence passage. Instead, the system must retrieve relevant evidence from a large text collection and generate an answer from the retrieved evidence.

This reproduction evaluates RAG models on three open-domain QA datasets:

* Natural Questions Open (NQ-Open)
* WebQuestions
* TriviaQA

The experiments use RAG-Sequence and RAG-Token models with a DPR-based retriever and the compressed Wikipedia DPR index.

---

## Related Paper

This reproduction is based on:

Lewis et al., **Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks**.

The paper introduces RAG models that combine:

1. a neural retriever based on Dense Passage Retrieval (DPR), and
2. a sequence-to-sequence generator.

The reproduced model families are:

* `facebook/rag-sequence-nq`
* `facebook/rag-token-nq`

---

## Scope of This Reproduction

This folder focuses on the standard open-domain QA reproduction.

It includes:

* evaluation of public RAG-NQ checkpoints,
* task-specific fine-tuning on WebQuestions,
* task-specific fine-tuning on TriviaQA,
* Exact Match evaluation after answer normalization,
* comparison between public checkpoint baselines and fine-tuned models.

This folder does **not** include the Graph-RAG extension. The graph-based retrieval and hybrid DPR + graph experiments are documented separately in:

```text
../graphrag_extension/
```

---

## Project Structure

```text
rag_qa_reproduction/
├── src/
│   ├── training/
│   │   └── finetune_rag_qa.py
│   │
│   └── evaluation/
│       └── evaluate_rag_qa.py
│
├── scripts/
│   └── reproduction/
│       ├── eval_public_baselines.sh
│       ├── eval_nq_pretrained.sh
│       ├── train_webquestions.sh
│       ├── eval_webquestions.sh
│       ├── train_triviaqa.sh
│       └── eval_triviaqa.sh
│
├── results/
│   └── reproduction_results.md
│
├── docs/
│   └── reproduction_protocol.md
│
├── README.md
├── requirements.txt
└── environment.txt
```

---

## Setup

Install the main dependencies:

```bash
pip install -r requirements.txt
```

The file `requirements.txt` contains the cleaner minimal dependency list.

The file `environment.txt` contains a full snapshot of the university cluster environment used for the experiments. This is included for reproducibility because package versions, CUDA versions, and cluster setup can affect long-running RAG experiments.

---

## Models

The experiments use the following Hugging Face RAG checkpoints:

| Model                      | Description                                                  |
| -------------------------- | ------------------------------------------------------------ |
| `facebook/rag-sequence-nq` | RAG-Sequence model trained for the Natural Questions setting |
| `facebook/rag-token-nq`    | RAG-Token model trained for the Natural Questions setting    |

Both models use DPR retrieval with the compressed Wikipedia DPR index.

---

## Dataset Splits

The reproduction uses the standard available splits for each open-domain QA dataset. Not all datasets provide train, validation, and test splits in the same way.

| Dataset                         |  Train |   Validation |         Test | Split used in this reproduction                                                  |
| ------------------------------- | -----: | -----------: | -----------: | -------------------------------------------------------------------------------- |
| NQ-Open                         | 87,925 |        3,610 | Not provided | Validation split used for evaluation                                             |
| WebQuestions                    |  3,778 | Not provided |        2,032 | Train split used for fine-tuning; test split used for evaluation                 |
| TriviaQA `unfiltered.nocontext` | 87,622 |       11,313 |       10,832 | 30k training examples used for fine-tuning; validation split used for evaluation |

For NQ-Open, the public RAG-NQ checkpoints were evaluated directly on the validation split without additional fine-tuning.

For WebQuestions, the full training split was used for task-specific fine-tuning, and the test split was used for evaluation.

For TriviaQA, a 30k-example subset of the training split was used for one epoch of fine-tuning due to the available university cluster resources. Evaluation was performed on the validation split.

---

## Chronological Workflow

The reproduction was performed in the following order:

1. Evaluate the released RAG-NQ checkpoints on NQ-Open.
2. Evaluate the same public checkpoints as baselines on WebQuestions and TriviaQA.
3. Fine-tune RAG-Sequence and RAG-Token on WebQuestions.
4. Fine-tune RAG-Sequence and RAG-Token on TriviaQA.
5. Evaluate all models using normalized Exact Match.
6. Compare public checkpoint baselines against the fine-tuned models.

This order separates the original public checkpoint behavior from the effect of task-specific fine-tuning.

---

## Experimental Configuration

### Retrieval and Generation

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

Beam search with `num_beams=4` was used during evaluation because it produced stronger and more stable answer generation in this reproduction setup than greedy decoding.

---

## Dataset-Specific Settings

| Dataset      | Models                  |                                  Training split |           Evaluation split | Training / evaluation setting                                           |
| ------------ | ----------------------- | ----------------------------------------------: | -------------------------: | ----------------------------------------------------------------------- |
| NQ-Open      | RAG-Sequence, RAG-Token |                                        Not used |  3,610 validation examples | Public RAG-NQ checkpoints evaluated directly; no additional fine-tuning |
| WebQuestions | RAG-Sequence, RAG-Token |                            3,778 train examples |        2,032 test examples | Public checkpoint baseline and task-specific fine-tuning for 2 epochs   |
| TriviaQA     | RAG-Sequence, RAG-Token | 30,000 examples sampled from the training split | 11,313 validation examples | Public checkpoint baseline and task-specific fine-tuning for 1 epoch    |

---

## Fine-Tuning Settings

| Parameter           |     WebQuestions |         TriviaQA |
| ------------------- | ---------------: | ---------------: |
| Batch size          |                1 |                1 |
| Learning rate       |           `1e-5` |           `1e-6` |
| Epochs              |                2 |                1 |
| Retrieved documents |                5 |                5 |
| Optimizer           |            AdamW |            AdamW |
| Checkpointing       | Every 2000 steps | Every 2000 steps |
| Question encoder    |        Trainable |        Trainable |
| Generator           |        Trainable |        Trainable |

All trainable RAG components were fine-tuned in the main runs, including the question encoder and the generator.

The compressed Wiki-DPR index was kept fixed. This means that the stored Wikipedia passage embeddings and FAISS index were not rebuilt during fine-tuning. The model was fine-tuned around the existing retrieval setup.

For TriviaQA, the fine-tuning run used 30k training examples for one epoch due to the available university cluster resources and memory constraints.

---

## Evaluation Details

Evaluation was performed using Exact Match after answer normalization.

The normalization step:

* lowercases predictions and gold answers,
* removes punctuation,
* removes articles,
* normalizes whitespace.

For datasets with multiple gold answers or aliases, a prediction was counted as correct if it matched any available gold answer after normalization.

---

## Reproduction Scripts

### Public Checkpoint Baselines

```bash
bash scripts/reproduction/eval_public_baselines.sh
```

### NQ-Open Evaluation Only

```bash
bash scripts/reproduction/eval_nq_pretrained.sh
```

### WebQuestions Fine-Tuning and Evaluation

```bash
bash scripts/reproduction/train_webquestions.sh
bash scripts/reproduction/eval_webquestions.sh
```

### TriviaQA Fine-Tuning and Evaluation

```bash
bash scripts/reproduction/train_triviaqa.sh
bash scripts/reproduction/eval_triviaqa.sh
```

---

## Results

Scores are reported as Exact Match percentages.

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

---

## Main Observations

Fine-tuning improved performance on both WebQuestions and TriviaQA.

On WebQuestions:

* RAG-Sequence improved from **15.06% EM** to **32.33% EM**.
* RAG-Token improved from **12.00% EM** to **29.20% EM**.

On TriviaQA:

* RAG-Sequence improved from **33.57% EM** to **38.20% EM**.
* RAG-Token improved from **30.60% EM** to **35.20% EM**.

For NQ-Open, the released RAG-NQ checkpoints were evaluated without additional fine-tuning, because these checkpoints were already trained for the Natural Questions setting.

Overall, the results show that task-specific fine-tuning is important when transferring RAG checkpoints from NQ-Open to other open-domain QA datasets such as WebQuestions and TriviaQA.

---

## Known Limitations and Differences from the Original Paper

This reproduction was executed under academic computing constraints and is not intended to exactly match the full compute scale of the original RAG paper.

| Aspect            | Original RAG paper              | This reproduction                                      |
| ----------------- | ------------------------------- | ------------------------------------------------------ |
| Compute scale     | Large-scale training setup      | University cluster resources                           |
| Retrieval index   | Full DPR Wikipedia setup        | Hugging Face compressed Wiki-DPR index                 |
| Training          | Paper-scale training and tuning | Controlled fine-tuning runs                            |
| Batch size        | Larger effective training setup | Batch size 1 due to memory constraints                 |
| TriviaQA training | Full task-scale setup           | Stable run on 30k examples                             |
| Checkpoints       | Paper-reported trained models   | Public Hugging Face checkpoints plus local fine-tuning |
| Evaluation        | Paper benchmark results         | Reproduced EM with normalized answers                  |

Because of these differences, the goal is not to exactly reproduce every paper number, but to document a controlled and reproducible RAG QA pipeline using available resources.

---

## Checkpoints and Cache Files

Model checkpoints, Hugging Face cache files, and DPR index cache files are not included in this repository because of their large size.

This repository provides the code, scripts, configuration, and documentation needed to reproduce the checkpoints and evaluations.

Large generated files should be stored outside the Git repository or regenerated using the provided scripts.

---

## Relation to the Graph-RAG Extension

This reproduction serves as the baseline open-domain QA component of the project.

The Graph-RAG / Hybrid Graph-RAG extension builds on this work by testing whether graph-structured evidence can complement DPR-based dense passage retrieval.

The extension is documented separately in:

```text
../graphrag_extension/
```

The separation is intentional:

* `rag_qa_reproduction/` contains the standard RAG QA reproduction.
* `graphrag_extension/` contains the entity-relation extraction, graph construction, graph retrieval, and hybrid DPR + graph evidence experiments.

---

## Notes

This reproduction was executed on a university computing cluster. The training script includes checkpointing and logging to support long-running jobs.

The reproduction documents a practical RAG QA setup under available academic computing resources. It should be interpreted as a controlled reproduction and analysis of RAG behavior on open-domain QA tasks, rather than a full-scale reimplementation of the original training infrastructure.

