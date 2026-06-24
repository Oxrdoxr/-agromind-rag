# Agro-Mind RAG Pipeline Documentation

## Project: Autonomous Agricultural Customer Support Agent

---

## Table of Contents

1. [What the Pipeline Does](#1-what-the-pipeline-does)
2. [Data Sources and Formats](#2-data-sources-and-formats)
3. [Embedding and Retrieval Flow](#3-embedding-and-retrieval-flow)
4. [Evaluation Artifacts in `quality/`](#4-evaluation-artifacts-in-quality)

---

## 1. What the Pipeline Does

The pipeline transforms a raw Chinese agricultural product catalog and real customer support conversations into a queryable bilingual vector knowledge base. The retriever is the data layer for an autonomous support agent that answers farmer queries about product usage, disease diagnosis, dosage, logistics, and safety.

**End-to-end flow:**

```
Raw Chinese XLSX (114 products)
      ↓  Audit → Clean → Translate
Bilingual XLSX (产品目录_CLEANED.xlsx)
      ↓  Entity Extraction
Structured JSON entities (product_entities_normalized_v2.json)
      ↓  Retrieval Document Generation
Two document sets: structured (compact) + full (bilingual prose)
      ↓  AI-as-Judge quality gate
228 documents verified (structured avg 10.0/10, full avg 8.97/10)
      ↓  Embedding (BGE-M3 via Ollama, 1024 dims)
ChromaDB — two collections persisted at ./chromadb/
      ↓  Retriever (src/retriever.py)
AgroMindRetriever — product search + support case search
```

**Parallel fine-tuning track (evaluation only):**

```
230 real customer service chat examples (4 JSONL files)
      ↓  Safety cleaning (removed unsafe examples, added escalation examples)
234 cleaned training examples
      ↓  Gemini 2.5 Flash SFT via Vertex AI — SUCCEEDED
Tuned model: projects/548742106129/locations/us-central1/models/6556995316602634240@1
```

The fine-tuned Gemini model is not the production agent. The production agent uses Qwen2.5-7B-Instruct locally via Ollama. The fine-tuning output is used for evaluation and to derive safety behavior patterns for the system prompt.

**Core design principle:** Honest nulls over hallucinated data. Where the source catalog had no ingredient or dosage data, those gaps are preserved in the knowledge base. This is a safety-critical system; wrong dosage or ingredient information can cause real harm.

---

## 2. Data Sources and Formats

### Product Catalog

| File | Format | Description |
|---|---|---|
| `data/产品目录_CLEANED.xlsx` | Excel | Audited Chinese source, 114 products, 11 columns. Used as the canonical source of record. |
| `data/product_entities_normalized_v2.json` | JSON array | 114 structured product entities with normalized fields. Used as ChromaDB metadata. |
| `data/product_entities_normalized_v2.csv` | CSV | Same data as the JSON above, tabular form. |

**Catalog columns (from audit report):**
- 产品ID (Product ID)
- 产品名称 (Chinese name)
- 英文名称 (English name)
- 产品类型 (Product type)
- 中文使用说明 (Chinese usage instructions)
- 使用方法(兑水) (Dilution method)
- 对应农作物/植物 (Target crops/plants)
- 规格 (Specifications)
- 数据来源 (Data source)
- 主要成分 (Active ingredients) — **blank for 6 products** (AF0014, AF0017, AF0026, AF0029, AF0030, AF0035)
- 使用方法和剂量 (Usage method and dosage) — **blank for 17 products**

**Product types covered:** pesticides, microbial agents, fertilizers, adjuvants, herbicides, and related categories (51 raw type variants normalized to 24 canonical types).

### Customer Support Conversations

Four JSONL files in `data/`, each containing real customer service examples:

| File | Category | Count |
|---|---|---|
| `data/cat1_usage_product_real.jsonl` | Usage and product queries | 60 examples |
| `data/cat2_diagnosis_real.jsonl` | Crop diagnosis | 60 examples |
| `data/cat3_aftersales_logistics_real.jsonl` | After-sales and logistics | 60 examples |
| `data/cat4_safety_sensitive_real.jsonl` | Safety-sensitive queries | 50 examples |

These conversations are indexed in ChromaDB as the `agromind_full_v4` (support) collection. They are also used as the fine-tuning dataset after safety cleaning.

### Plant Disease Images

`data/plantvillage_unlabeled/class_1/` contains unlabeled JPG images from the PlantVillage dataset. These are referenced by config for a future image retrieval collection (`agromind_images_v4`). Image retrieval is not yet implemented in the active retriever.

---

## 3. Embedding and Retrieval Flow

### Embedding Model

**Model:** `bge-m3` via Ollama  
**Dimensions:** 1024  
**Endpoint:** `http://localhost:11434/api/embeddings`  

The embedding layer is implemented in `src/embeddings.py` as two classes:

- `OllamaEmbeddingFunction` — ChromaDB-compatible (`EmbeddingFunction` interface). Called by ChromaDB when indexing.
- `OllamaEmbeddingsWrapper` — LangChain-compatible (`Embeddings` interface). Used by `AgroMindRetriever` at query time via the `ollama_wrapper` singleton.

Both classes apply distinct prefixes to queries vs. documents (asymmetric embedding, required by BGE-M3):

| Use | Prefix |
|---|---|
| Documents at index time | `"Represent this document for retrieval: "` |
| Queries at retrieval time | `"Represent this query for retrieval: "` |

Both classes normalize the output vector to unit length when `config.embedding.normalize` is `true` (the current default). Normalization is applied after the prefix is prepended and the HTTP call returns.

**Singleton instances exported from `src/embeddings.py`:**

```python
ollama_embeddings   # OllamaEmbeddingFunction — ChromaDB interface
ollama_wrapper      # OllamaEmbeddingsWrapper — LangChain interface
```

### ChromaDB Collections

Two persistent collections are created at `./chromadb/` (path from `config.yaml`):

| Collection name | Config key | Contents |
|---|---|---|
| `agromind_structured_v4` | `retrieval.structured_collection` | 114 compact English product summary documents. Optimized for precise field-level queries (product ID, disease name, crop, product type). |
| `agromind_full_v4` | `retrieval.full_collection` | Bilingual documents: structured summary + English prose + original Chinese prose. Also used to index the support conversation JSONL files. Optimized for semantic/symptom queries. |

A third collection (`agromind_images_v4`) is declared in `config.yaml` but is not yet populated or queried.

### Retriever: `AgroMindRetriever` (`src/retriever.py`)

`AgroMindRetriever` connects to both ChromaDB collections on instantiation and validates that neither is empty. It raises `RuntimeError` if a collection is missing or empty.

**Methods:**

| Method | Collection | Description |
|---|---|---|
| `get_product(product_id)` | `agromind_structured_v4` | Exact lookup by ID. Returns metadata dict or `None`. |
| `search_products(query, k=5)` | `agromind_structured_v4` | Vector search. Returns list of product dicts with `distance` field. |
| `search_disease_products(disease, k=5)` | `agromind_structured_v4` | Alias for `search_products`. Passes `disease` string as the query. |
| `search_support_cases(query, k=3, category=None)` | `agromind_full_v4` | Vector search on support cases. Optional `category` filter maps to a ChromaDB `where` clause. |
| `retrieve_context(query, product_k=5, support_k=3, support_category=None)` | both | Agent entry point. Returns `{"products": [...], "support_cases": [...]}`. |
| `health()` | both | Returns document counts and the active embedding model name. |

**Retrieval parameters (from `config.yaml`):**

| Parameter | Value | Description |
|---|---|---|
| `top_k` | 5 | Default number of results |
| `confidence_threshold` | 0.85 | Distance threshold for confidence gating |
| `bm25_weight` | 0.4 | BM25 weight for hybrid search (configured, not yet active in retriever) |
| `vector_weight` | 0.6 | Vector weight for hybrid search |
| `hybrid_search.enabled` | true | Hybrid search enabled in config |
| `hybrid_search.bm25_alpha` | 0.4 | BM25 vs. vector weight ratio |
| `hybrid_search.rerank_top_k` | 3 | Rerank window after fusion |

Current retriever implementation uses vector search only (ChromaDB native `query`). The `bm25_weight`, `vector_weight`, and `hybrid_search` config fields are defined for future implementation.

### Configuration Loading

`src/config.py` loads `config.yaml` from the project root using `pyyaml`, validates all fields with Pydantic v2 models, and exposes a cached singleton:

```python
from src.config import config
```

Paths (`chromadb`, `data`, `logs`) are resolved relative to the project root at access time. `logs/` is created automatically on first access if it does not exist.

Environment variables override two config fields at load time: `LANGSMITH_API_KEY` and `LANGSMITH_TRACING`.

---

## 4. Evaluation Artifacts in `quality/`

All files in `quality/` are outputs from the pipeline evaluation phases. They are read-only records; no code reads from them at runtime.

### `quality/audit_report.json`

Output of the pre-translation data audit. Records:

- Total products: 114
- Blank counts per column
- Lists of product IDs with blank `主要成分` (active ingredients) and `使用方法和剂量` (usage/dosage)
- Ingredient recovery result: all 6 blank-ingredient products confirmed unrecoverable (source contains marketing copy only, no chemical data)
- Duplicate IDs found: 0

### `quality/judge_results_structured_v2.json`

Per-document AI-as-Judge scores for the 114 structured documents. Each entry records the product ID, score (0–10), and judge reasoning.

### `quality/judge_results_full_v2.json`

Per-document AI-as-Judge scores for the 114 full bilingual documents. Scores range from 7 to 10.

### `quality/judge_summary_v2.json`

Aggregate summary of judge scores and a list of products flagged for source data issues:

| Collection | Total | Avg score | Min | Max | Pass (≥7) | Fail |
|---|---|---|---|---|---|---|
| Structured | 114 | 10.0 | 10 | 10 | 114 | 0 |
| Full | 114 | 8.97 | 7 | 10 | 114 | 0 |

9 products scored exactly 7 on the full document: PN0004, PN0005, PN0007, PN0008, PN0011, PN0014, PN0031, PN0037, PN0041. The judge identified cross-section ingredient inconsistencies in each (active ingredient names conflict between the structured summary and the prose sections). These products are indexed as-is; the inconsistencies are flagged for client review.

All 228 documents (114 structured + 114 full) passed the ≥7 threshold and were cleared for embedding.

### `quality/retrieval_test_results_v2.json`

Per-query retrieval test results across 26 test queries. Each record includes:

- `query` and `query_type` (e.g., `disease`, `product_name`, `bilingual`)
- `collection` tested
- `expected_ids` — the ground-truth product IDs for this query
- `top5_ids` and `top5_names` — what the retriever actually returned
- `top5_distances` — ChromaDB cosine distances
- `hit_top1`, `hit_top3`, `hit_top5` — boolean accuracy flags

**Summary accuracy across 26 queries:**

| Metric | Value |
|---|---|
| Top-1 accuracy | 73.1% (19/26) |
| Top-3 accuracy | 96.2% (25/26) |
| Top-5 accuracy | 100% (26/26) |

### `quality/fine_tuning_summary.json`

Summary of the fine-tuning data preparation: example counts, categories, safety corrections applied, and training split details.

### `quality/gemini_tuning_job_v2.json`

Vertex AI fine-tuning job record. Contains the job ID, status (`SUCCEEDED`), and the tuned model resource name:

```
projects/548742106129/locations/us-central1/models/6556995316602634240@1
```

This model is for evaluation only and is not used by the production retriever or agent.

---

*SDA Agentic AI Capstone — Agro-Mind Autonomous Customer Support Agent*
