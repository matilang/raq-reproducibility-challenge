# Graph-RAG / Hybrid Graph-RAG Extension

This folder contains the Graph-RAG extension for the open-domain QA reproduction.

The extension builds on the reproduced RAG question answering pipeline and explores whether graph-structured evidence can complement DPR-based dense passage retrieval. The experiment is implemented as a one-shard proof of concept using one Wikipedia DPR shard from the full DPR corpus.

## Related Papers

This extension is based on three main papers:

| Paper                                                                                                                                   | Role in this project                                                                                     |
| --------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------- |
| Lewis et al., *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*                                                        | Main reproduction paper. Used for RAG-Sequence, RAG-Token, DPR retrieval, and open-domain QA evaluation. |
| Edge et al., *From Local to Global: A GraphRAG Approach to Query-Focused Summarization*                                                 | Motivation for using LLM-extracted entities and relationships as a graph-based retrieval layer.          |
| Sarmah et al., *HybridRAG: Integrating Knowledge Graphs and Vector Retrieval Augmented Generation for Efficient Information Extraction* | Motivation for combining vector retrieval and graph retrieval into a hybrid RAG setup.                   |

## Scope of This Extension

This is not a full-scale GraphRAG implementation over the complete Wikipedia corpus. Instead, it is a controlled extension over one DPR Wikipedia shard.

The goal is to test whether LLM-extracted entities and relations can provide useful structured evidence alongside DPR-retrieved passages.

## Notebooks

| Notebook                               | Purpose                                                                                                                          |
| -------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------- |
| `LLM_graph_extraction_layer.ipynb`     | Builds the extraction layer by loading one DPR shard, chunking article text, and using an LLM to extract entities and relations. |
| `final_explicit_graphrag_hybrid.ipynb` | Builds the final explicit graph retrieval and hybrid DPR + graph evidence experiment.                                            |

## Experiment 1: LLM Graph Extraction Layer

The first part of the extension constructs the graph evidence source.

Main steps:

1. Load one shard from the DPR Wikipedia passage collection.
2. Group passages by Wikipedia article title.
3. Chunk article text into manageable context windows.
4. Use an LLM to extract entities and relations from each chunk.
5. Save the extracted entity-relation JSON outputs.
6. Use the extracted entities and relations as the basis for graph retrieval.

This experiment creates the structured knowledge layer used by the later graph-based retrieval experiments.

## Experiment 2: Entity-Relation Retrieval from Extracted JSON

The second experiment tests whether the extracted JSON evidence can be used directly for retrieval before building a fully explicit graph.

Main idea:

1. Extract or tokenize important question terms.
2. Match question terms against extracted document entities.
3. Retrieve relations connected to the matched entities.
4. Use the retrieved entity-relation evidence as additional context for question answering.

This stage is an intermediate experiment because it uses the extracted entity-relation structure without requiring a full NetworkX graph implementation.

## Experiment 2 Results: Implicit Entity-Relation Retrieval

Experiment 2 evaluated retrieval directly from the LLM-extracted entity-relation JSON outputs before constructing the final explicit graph.

This experiment was the most promising Graph-RAG result because the hybrid setup improved over the DPR baseline on both Exact Match and F1.

| Setting                 | Examples | Top-k | Max graph relations | Baseline EM | Graph-only EM | Hybrid EM | Baseline F1 | Graph-only F1 | Hybrid F1 |
| ----------------------- | -------: | ----: | ------------------: | ----------: | ------------: | --------: | ----------: | ------------: | --------: |
| JSON relation retrieval |    3,610 |     5 |                   3 |        5.82 |          5.10 |      6.23 |       10.58 |          9.35 |     11.07 |
| JSON relation retrieval |    3,610 |     5 |                   5 |        5.82 |          4.40 |      6.51 |       10.58 |          8.60 |     11.44 |
| JSON relation retrieval |    3,610 |     5 |                  10 |        5.82 |          4.24 |      6.26 |       10.58 |          7.99 |     10.84 |

Graph evidence was available for 100% of the evaluated examples.

The best result was obtained with `max_graph_relations=5`, where the hybrid setup improved Exact Match from 5.82 to 6.51 and F1 from 10.58 to 11.44.

This suggests that the extracted entity-relation JSON contains useful structured evidence, but that it is most effective when combined with DPR retrieval rather than used alone.


## Experiment 3: Explicit Hybrid Graph-RAG Retrieval

The final experiment builds an explicit graph-based retrieval setup and combines it with DPR retrieval.

Main steps:

1. Build graph nodes from extracted entities.
2. Build graph edges from extracted relations.
3. Match question terms or entities to graph nodes.
4. Retrieve neighboring graph evidence from connected relations.
5. Combine graph evidence with DPR-retrieved passages.
6. Evaluate graph-only and hybrid DPR + graph evidence variants.

The hybrid setup is the main extension because it does not replace DPR. Instead, it adds graph evidence as a complementary retrieval signal.

## Experimental Results

The final explicit graph experiment was evaluated on NQ-Open questions using DPR retrieval, graph-only evidence, and the hybrid DPR + graph evidence setup.

### Graph Construction Output

The explicit graph built from the one-shard extraction contained:

| Component               |   Count |
| ----------------------- | ------: |
| Graph nodes             | 117,447 |
| Graph edges             | 151,256 |
| Chunks with graph edges |  17,221 |

### Selected Evaluation Results

| Setting                  | Examples | Top-k | Max graph relations | Baseline EM | Graph-only EM | Hybrid EM | Baseline F1 | Graph-only F1 | Hybrid F1 |
| ------------------------ | -------: | ----: | ------------------: | ----------: | ------------: | --------: | ----------: | ------------: | --------: |
| Small ablation           |      100 |     5 |                   2 |        4.00 |          4.00 |      6.00 |        8.47 |          9.32 |     10.14 |
| Small ablation           |      100 |     5 |                   3 |        4.00 |          6.00 |      6.00 |        8.47 |         10.62 |     11.70 |
| Small ablation           |      100 |     5 |                   5 |        4.00 |          5.00 |      6.00 |        8.47 |          9.52 |     12.62 |
| Larger sample            |    1,000 |     5 |                   3 |        5.50 |          5.30 |      6.50 |       10.50 |          9.67 |     11.49 |
| Full NQ evaluation split |    3,610 |     5 |                   3 |        5.82 |          4.68 |      6.26 |       10.58 |          8.35 |     10.94 |

Graph evidence was available for almost all evaluated examples, reaching 100% in the smaller runs and approximately 99.97% on the full NQ evaluation split.

### Result Interpretation

The graph-only setup did not consistently outperform the DPR baseline. This is expected because the graph was built from only one DPR shard, so graph coverage is limited and question-to-entity matching is still relatively simple.

However, the hybrid setup achieved the best Exact Match and F1 in the full evaluation:

* Baseline DPR EM: 5.82
* Graph-only EM: 4.68
* Hybrid DPR + graph EM: 6.26
* Baseline DPR F1: 10.58
* Graph-only F1: 8.35
* Hybrid DPR + graph F1: 10.94

This suggests that the graph evidence is most useful as a complementary retrieval signal rather than as a standalone replacement for DPR retrieval. The result supports the main idea of the extension: combining dense passage retrieval with structured entity-relation evidence can improve the RAG context in a controlled one-shard setting.


## Method Summary

The overall Graph-RAG extension follows this pipeline:

```text
DPR Wikipedia shard
        ↓
Article grouping and chunking
        ↓
LLM entity-relation extraction
        ↓
Graph / relation evidence construction
        ↓
Question-to-entity matching
        ↓
Graph evidence retrieval
        ↓
Hybrid DPR + graph evidence context
        ↓
Answer generation / evaluation
```

## Interpretation

Dense DPR retrieval is useful for retrieving semantically relevant passages. Graph evidence can add explicit entity-relation connections that may not be easy to recover from passage similarity alone.

Therefore, the final experiment should be interpreted as a Hybrid Graph-RAG proof of concept: DPR provides the vector retrieval component, while the extracted entity-relation graph provides structured retrieval evidence.

## Limitations

* The graph is built from only one DPR shard, not the full 157-shard Wikipedia DPR corpus.
* Entity and relation quality depends on the LLM extraction step.
* Question matching is based on extracted entities and keyword overlap, so it is simpler than a fully trained graph retriever.
* The experiment is a proof of concept rather than a production-scale GraphRAG system.
