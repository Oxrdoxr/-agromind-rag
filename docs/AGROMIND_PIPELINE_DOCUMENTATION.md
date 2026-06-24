# Agro-Mind RAG Pipeline Documentation

## Project: Autonomous Agricultural Customer Support Agent

---

## Table of Contents

1. [What the Pipeline Does](#1-what-the-pipeline-does)
2. [Data Sources and Formats](#2-data-sources-and-formats)
3. [Embedding and Retrieval Flow](#3-embedding-and-retrieval-flow)
4. [ChromaDB Collections](#4-chromadb-collections)
5. [Evaluation Results](#5-evaluation-results)
6. [What Changed in This Sprint](#6-what-changed-in-this-sprint)

---

## 1. What the Pipeline Does

The pipeline transforms a raw Chinese agricultural product catalog and real customer support conversations into a queryable bilingual vector knowledge base. A separate image pipeline indexes annotated crop disease photos. Together they serve an autonomous support agent that answers farmer queries about product usage, disease diagnosis, dosage, logistics, and safety.

**Text pipeline:**

```
Raw Chinese XLSX (114 products)
      ↓  Audit → Clean → Translate
Bilingual XLSX (产品目录_CLEANED.xlsx)
      ↓  Entity Extraction
Structured JSON entities (clean_entities.json)
      ↓  scripts/rebuild_db.py  [BGE-M3 via Ollama]
agromind_structured_v4 — product search collection

4 support conversation JSONL files (230 examples)
      ↓  scripts/build_full_collection.py  [BGE-M3 via Ollama]
agromind_full_v4 — support case search collection

AgroMindRetriever (src/retriever.py)
      ↓  retrieve_context()
{"products": [...], "support_cases": [...]}
```

**Image pipeline:**

```
Annotated crop disease images (data/annotated_images/)
      ↓  scripts/add_image_collection.py  [CLIP ViT-B/32]
agromind_images_v4 — image search collection

Query image
      ↓  src/image_retriever.py  — CLIP embedding + ChromaDB query
Top-K similar disease images with crop/disease metadata
      ↓  src/diagnosis_tool.py
Combined output: diagnosis + recommended products + historical cases
```

**Core design principle:** Honest nulls over hallucinated data. Where the source catalog had no ingredient or dosage data, those gaps are preserved in the knowledge base. Wrong dosage or ingredient information can cause real harm.

**Fine-tuning (parallel track, evaluation only):**

```
230 real customer service chat examples
      ↓  Safety cleaning (removed unsafe examples, added escalation examples)
234 cleaned training examples
      ↓  Gemini 2.5 Flash SFT via Vertex AI — SUCCEEDED
Tuned model: projects/548742106129/locations/us-central1/models/6556995316602634240@1
```

The fine-tuned model is for evaluation only. The production agent uses Qwen2.5-7B-Instruct locally via Ollama.

---

## 2. Data Sources and Formats

### Product Catalog

| File | Format | Description |
|---|---|---|
| `data/产品目录_CLEANED.xlsx` | Excel | Audited Chinese source, 114 products, 11 columns. Canonical source of record. |
| `data/clean_entities.json` | JSON array | 114 structured product entities extracted and normalized from the catalog. Used by `scripts/rebuild_db.py` to build `agromind_structured_v4`. |
| `data/clean_products.json` | JSON | Alternative product representation. Produced alongside `clean_entities.json` during the entity extraction phase. |

**Catalog columns** (from `quality/audit_report.json`):
- 产品ID, 产品名称, 英文名称, 产品类型
- 中文使用说明, 使用方法(兑水), 对应农作物/植物, 规格, 数据来源
- 主要成分 (Active ingredients) — **blank for 6 products:** AF0014, AF0017, AF0026, AF0029, AF0030, AF0035
- 使用方法和剂量 (Usage/dosage) — **blank for 17 products**

**Entity fields** (from `data/clean_entities.json`):
`product_id`, `name_cn`, `name_en`, `target_crops` (list), `target_diseases` (list), `target_pests` (list), `active_ingredients` (list), `symptoms_addressed` (list), `has_diseases`, `has_pests`, `has_ingredients`, `is_pesticide`, `is_microbial`, `is_fertilizer`, `data_quality`

### Customer Support Conversations

Four JSONL files indexed into `agromind_full_v4` via `scripts/build_full_collection.py`:

| File | Category | Count |
|---|---|---|
| `data/cat1_usage_product_real.jsonl` | Product usage and dosage queries | 60 examples |
| `data/cat2_diagnosis_real.jsonl` | Crop diagnosis conversations | 60 examples |
| `data/cat3_aftersales_logistics_real.jsonl` | After-sales and logistics | 60 examples |
| `data/cat4_safety_sensitive_real.jsonl` | Safety-sensitive queries | 50 examples |

These same files were used as the fine-tuning dataset after safety cleaning (2 self-harm examples removed, 2 unsafe factual claims removed, 8 safety escalation examples added → 234 total).

### Plant Disease Images

**Unlabeled (present, not indexed):** `data/plantvillage_unlabeled/class_0/` and `class_1/` contain unlabeled JPG images from the PlantVillage dataset. These are not the annotated dataset used for `agromind_images_v4`.

**Annotated (required, not present):** `scripts/add_image_collection.py` expects `data/annotated_images/annotations.xlsx` (columns: Seq, Filename, Crop, Disease, Disease Type) and `data/annotated_images/images/`. This dataset is maintained separately and must be obtained from the team before the image collection can be built. The image dataset used for the `agromind_images_v4` collection is maintained in the `agromind-rag` repository; integration into the active retriever is pending.

### Chunking Experiments

`chuncks/` contains three generations of serialized chunk files from iterative chunking experiments:
- `all_chunks.pkl` (401 KB)
- `all_chunks_v2.pkl` (436 KB)
- `all_chunks_v3.pkl` (488 KB)

These are intermediate artifacts and are not used at runtime.

---

## 3. Embedding and Retrieval Flow

### Text Embeddings — BGE-M3

**Model:** `bge-m3` via Ollama  
**Dimensions:** 1024  
**Endpoint:** `http://localhost:11434/api/embeddings`

Implemented in `src/embeddings.py` as two classes:

- **`OllamaEmbeddingFunction`** — implements ChromaDB's `EmbeddingFunction`. Used when building collections via `Chroma.from_texts`.
- **`OllamaEmbeddingsWrapper`** — implements LangChain's `Embeddings`. Used by `AgroMindRetriever` at query time via the `ollama_wrapper` singleton.

Both apply asymmetric prefixes required by BGE-M3:

| Use | Prefix applied |
|---|---|
| Documents at index time | `"Represent this document for retrieval: "` |
| Queries at retrieval time | `"Represent this query for retrieval: "` |

Both normalize the output vector to unit length when `config.embedding.normalize` is `true` (default). A persistent `requests.Session()` reuses the HTTP connection across calls.

**Singleton exports from `src/embeddings.py`:**
- `ollama_embeddings` — `OllamaEmbeddingFunction` (ChromaDB interface)
- `ollama_wrapper` — `OllamaEmbeddingsWrapper` (LangChain interface)

### Image Embeddings — CLIP ViT-B/32

**Model:** `ViT-B/32` via `openai/clip`  
**Dimensions:** 512 (ViT-B/32 output)  
**Source:** `src/image_embeddings.py`

Implemented as `CLIPEmbeddings`, a LangChain `Embeddings` subclass with a singleton pattern (one CLIP model load per process). Supports:
- `embed_image(image_path)` — CLIP visual encoder, returns normalized float32 numpy array
- `embed_text(text)` — CLIP text encoder, used when searching images by text
- `embed_documents(texts)` and `embed_query(text)` — LangChain interface
- `cosine_similarity(emb1, emb2)` — utility

CLIP runs on CUDA if available, CPU otherwise.

**Singleton export:** `clip_embeddings` — `CLIPEmbeddings` instance

### AgroMindRetriever (`src/retriever.py`)

Connects to both text collections on instantiation. Raises `RuntimeError` if either collection is empty or missing.

**Methods:**

| Method | Collection queried | Description |
|---|---|---|
| `get_product(product_id)` | `agromind_structured_v4` | Exact ChromaDB get by ID. Returns metadata dict or `None`. |
| `search_products(query, k=5)` | `agromind_structured_v4` | Vector search. Returns list of product dicts with `distance`. |
| `search_disease_products(disease, k=5)` | `agromind_structured_v4` | Passes `disease` string to `search_products`. |
| `search_support_cases(query, k=3, category=None)` | `agromind_full_v4` | Vector search with optional `where` filter on `category` field. |
| `retrieve_context(query, product_k=5, support_k=3, support_category=None)` | both | Agent entry point. Returns `{"products": [...], "support_cases": [...]}`. |
| `health()` | both | Returns document counts and active embedding model name. |

**Module-level singleton:** `retrieval_tool = AgroMindRetriever()` — import directly.

### ImageRetriever (`src/image_retriever.py`)

Connects to `agromind_images_v4` on instantiation. Raises `RuntimeError: Image collection not found` if the collection does not exist.

**Methods:**

| Method | Description |
|---|---|
| `search_by_image(image_path, k=5)` | CLIP-encodes the image, queries `agromind_images_v4` by embedding. Returns raw ChromaDB result dict. |
| `diagnose(image_path, k=5)` | Wraps `search_by_image`. Returns the best match as structured diagnosis: `crop`, `disease`, `disease_type`, `confidence`, `top_matches`. |

### diagnosis_tool (`src/diagnosis_tool.py`)

Exports `diagnose_crop_image(image_path, user_text=None)` — the top-level agent entry point for image-based queries. Internally:

1. Calls `ImageRetriever.diagnose()` to identify crop and disease from the image
2. Calls `retrieval_tool.search_products(f"{crop} {disease}", k=5)` to find matching products
3. Calls `retrieval_tool.search_support_cases(disease, k=3, category="diagnosis")` for historical cases
4. Returns a combined dict: `{"success": True, "diagnosis": {...}, "recommended_products": [...], "historical_cases": [...], "similar_images": [...]}`

### Configuration Loading

`src/config.py` loads `config.yaml` from the project root using PyYAML, validates all fields with Pydantic v2 models, and exposes a cached singleton via `from src.config import config`. Paths are resolved relative to the project root at access time. `logs/` is created automatically if missing. Environment variables `LANGSMITH_API_KEY` and `LANGSMITH_TRACING` override config values at load time.

---

## 4. ChromaDB Collections

All collections persist at `./chromadb/` (from `config.yaml: paths.chromadb`).

| Collection | Config key | Built by | Embedding | Contents |
|---|---|---|---|---|
| `agromind_structured_v4` | `retrieval.structured_collection` | `scripts/rebuild_db.py` | BGE-M3 (1024d) | 114 bilingual product documents. Text includes product ID, Chinese name, English name, diseases, crops, active ingredients (both CN and EN). Metadata includes `product_id`, `name_cn`, `name_en`, `product_type`, `is_pesticide`, `is_microbial`, `is_fertilizer`, `diseases`, `crops`. |
| `agromind_full_v4` | `retrieval.full_collection` | `scripts/build_full_collection.py` | BGE-M3 (1024d) | Historical support conversations from the 4 JSONL files. Metadata includes `category` field (filterable) and `source`. |
| `agromind_images_v4` | `retrieval.image_collection` | `scripts/add_image_collection.py` | CLIP ViT-B/32 (512d) | Annotated crop disease images. Metadata: `crop`, `disease`, `disease_type`, `file_name`, `image_path`, `embedding_model`, `source="annotated"`, `type="image"`. **Not yet populated** — requires annotated dataset. |

**Rebuild commands:**

```bash
# Structured collection only (safe: keeps full + image collections)
python scripts/rebuild_db.py

# Full/support collection
python scripts/build_full_collection.py

# Image collection (requires data/annotated_images/)
python scripts/add_image_collection.py

# Check collection health
python scripts/check_collections_health.py

# Inspect collection contents
python scripts/check_chromadb.py
```

---

## 5. Evaluation Results

All artifacts in `quality/` are read-only outputs from pipeline evaluation phases.

### `quality/audit_report.json`

Pre-translation data audit results:
- 114 total products, 0 duplicate IDs
- 6 products with unrecoverable blank `主要成分` (active ingredients)
- 17 products with blank `使用方法和剂量` (usage/dosage)

### `quality/judge_summary_v2.json`

AI-as-Judge scores from GPT-4 evaluation of all 228 generated documents:

| Collection | Total | Avg score | Min | Max | Pass (≥7) | Fail |
|---|---|---|---|---|---|---|
| Structured | 114 | 10.0 | 10 | 10 | 114 | 0 |
| Full | 114 | 8.97 | 7 | 10 | 114 | 0 |

9 products scored exactly 7 on the full document due to cross-section ingredient inconsistencies in the source data: PN0004, PN0005, PN0007, PN0008, PN0011, PN0014, PN0031, PN0037, PN0041. All 228 documents passed the ≥7 threshold and were cleared for embedding.

### `quality/retrieval_test_results_v2.json`

Per-query results across 26 test queries against the vector collections:

| Metric | Result |
|---|---|
| Top-1 accuracy | 73.1% (19/26 queries) |
| Top-3 accuracy | 96.2% (25/26 queries) |
| Top-5 accuracy | 100% (26/26 queries) |

### `quality/fine_tuning_summary.json` and `quality/gemini_tuning_job_v2.json`

Fine-tuning track records. The Vertex AI SFT job succeeded. Tuned model:
`projects/548742106129/locations/us-central1/models/6556995316602634240@1`

This model is for evaluation only, not the production retriever.

### Retrieval evaluation scripts (`evaluation/`)

Five scripts for ongoing retrieval quality checks:

| Script | Purpose |
|---|---|
| `evaluation/eval_retrieval.py` | 9-query test set against the structured collection using `SmartRetriever` |
| `evaluation/evaluate_image_retrieval.py` | Evaluates CLIP image search quality against `agromind_images_v4` |
| `evaluation/evaluation_set.py` | Defines the evaluation query set |
| `evaluation/retrieval_eval.py` | Lightweight retrieval check |
| `evaluation/extract_dosage_cases.py` | Extracts dosage-related support cases for targeted evaluation |

Note: `evaluation/eval_retrieval.py` references an older collection path (`data/v4/chroma_db`, collection name `agromind_v4`) that does not match the current `agromind_structured_v4` collection. This script needs updating before use.

---

## 6. What Changed in This Sprint

This section describes what was built in `agromind-v3` that is now reflected in the `agromind-rag` repository.

### Image pipeline — fully implemented

Three new source files in `src/`:

**`src/image_embeddings.py`** — `CLIPEmbeddings` class using ViT-B/32. Singleton pattern, CUDA-aware, normalizes output vectors. Embeds both images and text in the same CLIP embedding space for cross-modal search.

**`src/image_retriever.py`** — `ImageRetriever` class querying the `agromind_images_v4` ChromaDB collection. Provides `search_by_image()` and `diagnose()` methods. Returns structured diagnosis metadata (crop, disease, disease_type, confidence score).

**`src/diagnosis_tool.py`** — `diagnose_crop_image()` function, the top-level entry point for image-based agent queries. Combines CLIP image diagnosis with product and support case retrieval into a single response dict.

### Image collection indexing script

**`scripts/add_image_collection.py`** — `ImageIndexer` class that reads an annotations Excel file and a folder of images, embeds each image with CLIP, and upserts into `agromind_images_v4`. Handles batching (default batch_size=50), skips already-indexed images, and logs missing files.

### Support conversation collection builder

**`scripts/build_full_collection.py`** — `FullCollectionBuilder` class that reads the 4 support JSONL files, converts each conversation into a ChromaDB document, and builds `agromind_full_v4`. Previously this collection was documented as pre-built; the build script now exists in the repo.

### Evaluation infrastructure

`evaluation/` folder with 5 scripts added for ongoing retrieval quality checks. `tests/` folder added with `test_agent_pipeline.py` (full pipeline integration test) and `test_retriever.py`.

### Additional test scripts at project root

`test_image_search.py`, `test_search.py`, `test_langsmith.py`, `final_test.py` — added alongside the existing `test_embedding.py` for component-level testing.

### Data files

`data/clean_products.json` — added alongside `data/clean_entities.json` as an alternative product representation.

### What is NOT yet done

- `agromind_images_v4` collection is not yet populated — annotated image dataset is required
- Image pipeline dependencies (`clip`, `torch`, `Pillow`, `pandas`) are not in `requirements.txt`
- `evaluation/eval_retrieval.py` references a stale collection name (`agromind_v4`) and path (`data/v4/chroma_db`)
- `agent.multimodal.enabled` in `config.yaml` is `false` — no agent-level multimodal routing yet

---

*SDA Agentic AI Capstone — Agro-Mind Autonomous Customer Support Agent*
