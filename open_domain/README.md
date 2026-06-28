# Open-Domain QA: RAG Reproduction and Graph-RAG Extension

This folder contains the open-domain question answering part of the RAG Reproducibility Challenge group project.

The work is divided into two connected parts:

1. **RAG QA Reproduction**
2. **Graph-RAG / Hybrid Graph-RAG Extension**

The first part reproduces RAG-style open-domain question answering using DPR retrieval and RAG-Sequence / RAG-Token models. The second part extends the reproduction with graph-structured evidence and hybrid DPR + graph retrieval.

---

## Folder Structure

```text
open_domain/
├── rag_qa_reproduction/
│   ├── src/
│   ├── scripts/
│   ├── results/
│   ├── docs/
│   └── README.md
│
└── graphrag_extension/
    ├── notebooks/
    ├── graph_data/
    ├── results/
    └── README.md
```

---

## 1. RAG QA Reproduction

Folder: [`rag_qa_reproduction/`](rag_qa_reproduction/)

This part reproduces Retrieval-Augmented Generation for open-domain question answering.

It evaluates RAG-Sequence and RAG-Token on:

* Natural Questions Open
* WebQuestions
* TriviaQA

The experiments use Hugging Face RAG checkpoints, DPR-based retrieval, and the compressed Wikipedia DPR index.

The reproduction includes:

* public checkpoint evaluation,
* task-specific fine-tuning,
* Exact Match evaluation,
* comparison between baseline and fine-tuned results.

For full details, see:

[`rag_qa_reproduction/README.md`](rag_qa_reproduction/README.md)

---

## 2. Graph-RAG / Hybrid Graph-RAG Extension

Folder: [`graphrag_extension/`](graphrag_extension/)

This part extends the open-domain QA reproduction with graph-structured retrieval.

The extension builds a controlled one-shard proof of concept using one Wikipedia DPR shard. It extracts entities and relations from Wikipedia passages, builds structured evidence, and evaluates whether this evidence can complement DPR retrieval.

The extension includes:

* LLM-based entity-relation extraction,
* structured JSON relation retrieval,
* explicit graph construction,
* graph evidence retrieval,
* hybrid DPR + graph retrieval,
* T5 answer generation using the same controlled generator.

For full details, see:

[`graphrag_extension/README.md`](graphrag_extension/README.md)

---

## Chronological Workflow

The open-domain work was developed in the following order:

1. Reproduce the standard RAG QA setup.
2. Evaluate public RAG checkpoints on open-domain QA datasets.
3. Fine-tune RAG models on WebQuestions and TriviaQA.
4. Build a controlled DPR + T5 baseline over one Wikipedia DPR shard.
5. Extract entities and relations from the DPR shard using an LLM.
6. Test entity-relation JSON retrieval.
7. Build an explicit graph from extracted entities and relations.
8. Combine DPR retrieval with graph evidence in a hybrid setup.

---

## Summary

The `rag_qa_reproduction/` folder represents the standard RAG reproduction.

The `graphrag_extension/` folder represents the extension beyond the original reproduction.

Together, they show both the baseline RAG QA behavior and a controlled experiment testing whether graph-structured evidence can improve or complement dense DPR retrieval.
