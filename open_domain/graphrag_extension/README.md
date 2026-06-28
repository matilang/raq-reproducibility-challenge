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

The extension should be interpreted as a proof of concept. It does not rebuild the full 157-shard DPR Wikipedia index.

## Notebooks

| Notebook                               | Purpose                                                                                                                                                               |
| -------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `LLM_graph_extraction_layer.ipynb`     | Builds the extraction layer by loading one DPR shard, chunking article text, and using an LLM to extract entities and relations.                                      |
| `final_explicit_graphrag_hybrid.ipynb` | Builds the one-shard DPR + T5 baseline, evaluates entity-relation JSON retrieval, constructs the explicit graph, and evaluates hybrid DPR + graph evidence retrieval. |

## Method Overview

The extension has three main parts:

1. **LLM graph extraction layer**
   This step creates the structured entity-relation evidence source from one DPR Wikipedia shard.

2. **Entity-relation retrieval from extracted JSON**
   This method retrieves structured relation evidence directly from the extracted JSON outputs.

3. **Explicit Hybrid Graph-RAG retrieval**
   This method builds an explicit graph from the extracted entities and relations, then retrieves graph evidence from connected nodes and edges.

The LLM extraction layer is not treated as a standalone QA experiment. Instead, it is the evidence construction step used by both retrieval evaluations.

Baseline Definition

For the graph-based evaluations, the baseline is a controlled one-shard DPR + T5 pipeline. Passage embeddings are created from one Wikipedia DPR shard, indexed locally, and retrieved using DPR-based dense retrieval. The retrieved passages are then passed to a T5 generator to produce answers.

This baseline uses only DPR-retrieved text evidence. It does not use the extracted JSON relations or the explicit graph evidence.

The same T5 generator is used across the baseline, relation-based, graph-based, and hybrid settings. This keeps the comparison focused on the retrieved evidence rather than changing the answer generation model.

## Retrieval and Generation Setup

The second notebook implements the retrieval and generation pipeline used for the graph-based evaluations.

Instead of relying only on a ready-made RAG checkpoint, the notebook builds a controlled one-shard QA setup:

1. A DPR-based dense retrieval baseline is built over one Wikipedia DPR shard.
2. Passage embeddings are created and indexed locally for retrieval.
3. Retrieved passages are passed to a T5 generator to produce answers.
4. This DPR + T5 setup is used as the baseline.
5. The same generator is then evaluated with additional structured evidence from the extracted entity-relation JSON.
6. Finally, an explicit graph is built from the extracted entities and relations, and graph evidence is combined with DPR evidence in a hybrid setup.

This makes the comparison controlled: the generator remains the same, while the retrieved evidence changes across the DPR-only, relation-based, graph-based, and hybrid settings.

The evaluated evidence settings are:

| Setting                        | Description                                                                                                  |
| ------------------------------ | ------------------------------------------------------------------------------------------------------------ |
| DPR + T5 baseline              | Retrieves passages from the one-shard DPR index and generates answers using T5.                              |
| Entity-relation JSON retrieval | Uses LLM-extracted entities and relations directly as structured evidence.                                   |
| Explicit graph retrieval       | Builds graph nodes and edges from extracted entities and relations, then retrieves connected graph evidence. |
| Hybrid retrieval               | Combines DPR-retrieved passages with structured relation or graph evidence before answer generation.         |

## Evaluation Results Summary

The extension evaluates two graph-based retrieval methods built from the same LLM-extracted entity-relation evidence:

1. **Entity-relation retrieval from extracted JSON**
2. **Explicit Hybrid Graph-RAG retrieval**

Both methods were evaluated on NQ-Open using the one-shard DPR + T5 setup as the baseline. Scores are reported as percentages.

### Best Result from Each Method

| Method                         | Examples | Top-k | Max graph relations | Baseline EM | Structured-only EM | Hybrid EM | Baseline F1 | Structured-only F1 | Hybrid F1 |
| ------------------------------ | -------: | ----: | ------------------: | ----------: | -----------------: | --------: | ----------: | -----------------: | --------: |
| Entity-relation JSON retrieval |    3,610 |     5 |                   5 |        5.82 |               4.40 |      6.51 |       10.58 |               8.60 |     11.44 |
| Explicit graph retrieval       |    3,610 |     5 |                   3 |        5.82 |               4.68 |      6.26 |       10.58 |               8.35 |     10.94 |

The strongest result came from the entity-relation JSON retrieval method. Its hybrid setup improved Exact Match from 5.82 to 6.51 and F1 from 10.58 to 11.44.

The explicit graph retrieval method also improved over the DPR + T5 baseline in the hybrid setting, reaching 6.26 EM and 10.94 F1 on the full NQ evaluation split.

Overall, both methods support the same conclusion: graph-structured evidence is most useful when combined with DPR retrieval rather than used as a standalone replacement.

## LLM Graph Extraction Layer

The first part of the extension constructs the graph evidence source.

Main steps:

1. Load one shard from the DPR Wikipedia passage collection.
2. Group passages by Wikipedia article title.
3. Chunk article text into manageable context windows.
4. Use an OpenAI mini model through OpenRouter with `temperature=0` to extract entities and relations from each chunk.
5. Save the extracted entity-relation JSON outputs.
6. Use the extracted entities and relations as the basis for graph retrieval.

This layer creates the structured knowledge source used by the later graph-based retrieval evaluations.

### LLM Extraction Reproducibility

The LLM extraction step was performed through the OpenRouter API using an OpenAI mini model. The extraction was run with `temperature=0` to make the entity and relation extraction as deterministic and reproducible as possible.

To reproduce the extraction step, an OpenRouter API key must be provided locally. The API key is not included in this repository.

On Linux / macOS:

```bash
export OPENROUTER_API_KEY="your_openrouter_api_key"
```

On Windows Command Prompt:

```bash
set OPENROUTER_API_KEY=your_openrouter_api_key
```

The notebook expects the key to be available as an environment variable:

```python
os.environ["OPENROUTER_API_KEY"]
```

Using the same model, prompt, chunking setup, and `temperature=0` is recommended when reproducing the extraction results.

### Extraction Quality Evaluation

To validate the LLM extraction layer, a small manual evaluation was performed on 20 sampled chunks. The evaluation compared extracted entities and relations against manually checked expected outputs.

| Metric             | Score |
| ------------------ | ----: |
| Entity precision   | 0.836 |
| Entity recall      | 0.780 |
| Entity F1          | 0.806 |
| Relation precision | 0.690 |
| Relation recall    | 0.660 |
| Relation F1        | 0.675 |

The extraction results show that the LLM extraction layer was reasonably reliable for entity extraction, with entity F1 around 0.81. Relation extraction was weaker than entity extraction, but still usable as a structured evidence source for the later retrieval evaluations.

This supports the design of the extension: the graph retrieval layer depends on automatically extracted entities and relations, so evaluating extraction quality helps justify using the extracted JSON outputs as graph evidence.

## Entity-Relation Retrieval from Extracted JSON

This retrieval evaluation tests whether the extracted JSON evidence can be used directly for retrieval before building a fully explicit graph.

Main idea:

1. Extract or tokenize important question terms.
2. Match question terms against extracted document entities.
3. Retrieve relations connected to the matched entities.
4. Use the retrieved entity-relation evidence as additional context for question answering.
5. Generate answers using the same T5 generator used in the DPR + T5 baseline.

This method uses the extracted entity-relation structure directly, without requiring a fully explicit graph implementation.

### Results

This was the most promising Graph-RAG result because the hybrid setup improved over the DPR + T5 baseline on both Exact Match and F1.

Scores are reported as percentages.

| Setting                 | Examples | Top-k | Max graph relations | Baseline EM | Relation-only EM | Hybrid EM | Baseline F1 | Relation-only F1 | Hybrid F1 |
| ----------------------- | -------: | ----: | ------------------: | ----------: | ---------------: | --------: | ----------: | ---------------: | --------: |
| JSON relation retrieval |    3,610 |     5 |                   3 |        5.82 |             5.10 |      6.23 |       10.58 |             9.35 |     11.07 |
| JSON relation retrieval |    3,610 |     5 |                   5 |        5.82 |             4.40 |      6.51 |       10.58 |             8.60 |     11.44 |
| JSON relation retrieval |    3,610 |     5 |                  10 |        5.82 |             4.24 |      6.26 |       10.58 |             7.99 |     10.84 |

Graph evidence was available for 100% of the evaluated examples.

The best result was obtained with `max_graph_relations=5`, where the hybrid setup improved Exact Match from 5.82 to 6.51 and F1 from 10.58 to 11.44. This suggests that the extracted entity-relation JSON contains useful structured evidence, but that it is most effective when combined with DPR retrieval rather than used alone.

## Explicit Hybrid Graph-RAG Retrieval

The final retrieval evaluation builds an explicit graph-based retrieval setup and combines it with DPR retrieval.

Main steps:

1. Build graph nodes from extracted entities.
2. Build graph edges from extracted relations.
3. Match question terms or entities to graph nodes.
4. Retrieve neighboring graph evidence from connected relations.
5. Combine graph evidence with DPR-retrieved passages.
6. Generate answers using the same T5 generator used in the DPR + T5 baseline.
7. Evaluate graph-only and hybrid DPR + graph evidence variants.

The hybrid setup does not replace DPR. Instead, it adds graph evidence as a complementary retrieval signal.

### Graph Construction Output

The explicit graph built from the one-shard extraction contained:

| Component               |   Count |
| ----------------------- | ------: |
| Graph nodes             | 117,447 |
| Graph edges             | 151,256 |
| Chunks with graph edges |  17,221 |

### Results

The explicit graph retrieval setup was evaluated on NQ-Open questions using DPR retrieval, graph-only evidence, and hybrid DPR + graph evidence.

Scores are reported as percentages.

| Setting                  | Examples | Top-k | Max graph relations | Baseline EM | Graph-only EM | Hybrid EM | Baseline F1 | Graph-only F1 | Hybrid F1 |
| ------------------------ | -------: | ----: | ------------------: | ----------: | ------------: | --------: | ----------: | ------------: | --------: |
| Small ablation           |      100 |     5 |                   2 |        4.00 |          4.00 |      6.00 |        8.47 |          9.32 |     10.14 |
| Small ablation           |      100 |     5 |                   3 |        4.00 |          6.00 |      6.00 |        8.47 |         10.62 |     11.70 |
| Small ablation           |      100 |     5 |                   5 |        4.00 |          5.00 |      6.00 |        8.47 |          9.52 |     12.62 |
| Larger sample            |    1,000 |     5 |                   3 |        5.50 |          5.30 |      6.50 |       10.50 |          9.67 |     11.49 |
| Full NQ evaluation split |    3,610 |     5 |                   3 |        5.82 |          4.68 |      6.26 |       10.58 |          8.35 |     10.94 |

Graph evidence was available for almost all evaluated examples, reaching 100% in the smaller runs and approximately 99.97% on the full NQ evaluation split.

## Pipeline Summary

The overall Graph-RAG extension follows this pipeline:

```text
DPR Wikipedia shard
        ↓
Article grouping and chunking
        ↓
LLM entity-relation extraction
        ↓
Structured JSON relation evidence
        ↓
Local DPR embeddings and index
        ↓
DPR + T5 baseline generation
        ↓
Entity-relation JSON retrieval
        ↓
Explicit graph construction
        ↓
Graph evidence retrieval
        ↓
Hybrid DPR + graph evidence context
        ↓
T5 answer generation and evaluation
```

## Result Interpretation

The relation-only and graph-only setups did not consistently outperform the DPR + T5 baseline. This is expected because the graph evidence was built from only one DPR shard, so graph coverage is limited and question-to-entity matching is still relatively simple.

The strongest result came from the entity-relation JSON retrieval evaluation, where the hybrid setup combined DPR evidence with LLM-extracted relation evidence. In the best run, the hybrid setup improved:

* Exact Match: 5.82 → 6.51
* F1: 10.58 → 11.44

The explicit graph retrieval evaluation also showed a smaller hybrid improvement on the full NQ evaluation split:

* Exact Match: 5.82 → 6.26
* F1: 10.58 → 10.94

Overall, the results suggest that graph evidence is most useful as a complementary retrieval signal rather than as a standalone replacement for DPR retrieval. Dense DPR retrieval provides semantically relevant passages, while graph evidence adds explicit entity-relation connections that may not be easy to recover from passage similarity alone.

Therefore, the extension should be interpreted as a Hybrid Graph-RAG proof of concept: DPR provides the vector retrieval component, while the extracted entity-relation graph provides structured retrieval evidence, and T5 is used as the controlled answer generator.

## Limitations

* The graph is built from only one DPR shard, not the full 157-shard Wikipedia DPR corpus.
* Entity and relation quality depends on the LLM extraction step.
* Question matching is based on extracted entities and keyword overlap, so it is simpler than a fully trained graph retriever.
* The DPR + T5 pipeline is a controlled one-shard baseline, not a full reproduction of the original large-scale RAG retrieval setup.
* The experiment is a proof of concept rather than a production-scale GraphRAG system.
