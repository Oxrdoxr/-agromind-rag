# Agro-Mind RAG

## Project Summary

Bilingual (Chinese + English) RAG knowledge base for agricultural products, powering an autonomous customer support agent. The system answers farmer queries about product usage, crop diagnosis, dosage, logistics, and safety without hallucinating.

- **114 agricultural products** (pesticides, microbial agents, fertilizers, adjuvants, herbicides) sourced from a Chinese product catalog
- **LLM:** Qwen2.5-7B-Instruct via Ollama — fully local inference, no cloud cost per query
- **Text embeddings:** BGE-M3 via Ollama — 1024-dimensional, bilingual, local
- **Image embeddings:** CLIP ViT-B/32 via `openai/clip` — used for crop disease image search
- **Vector store:** ChromaDB with three named collections: `agromind_structured_v4` (products), `agromind_full_v4` (support conversations), `agromind_images_v4` (crop disease images)
- **Tracing:** LangSmith (optional, configurable via `.env`)

The design principle throughout: honest nulls over hallucinated data. 6 products have no active ingredient data in the source; those gaps are preserved rather than invented.

**Source of truth for all code:** `agromind-v3` work folder. The files in `src/` are identical between agromind-v3 and agromind-rag.

---

## Requirements

Python packages (see `requirements.txt`):

| Package | Version | Purpose |
|---|---|---|
| `chromadb` | >=0.4.0 | Vector store |
| `ollama` | >=0.1.0 | Ollama Python client |
| `langchain` | >=0.1.0 | Agent framework |
| `langchain-chroma` | >=0.1.0 | LangChain ChromaDB integration |
| `langchain-ollama` | >=0.1.0 | LangChain Ollama integration |
| `langchain-community` | >=0.1.0 | LangChain community tools |
| `pydantic` | >=2.0.0 | Config validation |
| `pyyaml` | >=6.0 | Config file parsing |
| `requests` | >=2.31.0 | Ollama HTTP calls |
| `python-dotenv` | >=1.0.0 | `.env` loading |
| `langsmith` | >=0.1.0 | Tracing (optional) |
| `pytest` | >=7.0.0 | Testing |
| `numpy` | >=1.24.0 | Numeric utilities |

**Additional dependencies for the image pipeline (not yet in requirements.txt):**
`src/image_embeddings.py` imports `clip`, `torch`, `PIL` (Pillow), and `numpy`. `scripts/add_image_collection.py` imports `pandas`. These packages are required if using the image retrieval or diagnosis features but are not listed in `requirements.txt`:

```bash
pip install openai-clip torch Pillow pandas
```

**External requirement — Ollama:** must be installed separately from ollama.com and running locally on port `11434`.

---

## Installation

**1. Install Ollama and pull the required models**

```bash
# Install from ollama.com, then:
ollama pull bge-m3               # Text embedding model (1024 dims)
ollama pull qwen2.5:7b-instruct  # Agent LLM
ollama serve                      # Keep running in a separate terminal
```

**2. Clone the repo and create a virtual environment**

```bash
git clone <repo-url>
cd agromind-rag
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
```

**3. Install Python dependencies**

```bash
pip install -r requirements.txt
# If using image features:
pip install openai-clip torch Pillow pandas
```

**4. Obtain the ChromaDB vector store**

The pre-built ChromaDB is not in git (it is large). Get the `chromadb/` folder from the team shared drive and place it at the project root. The config expects it at `./chromadb/`.

**5. Create a `.env` file** (optional — only needed for LangSmith tracing)

```
LANGSMITH_API_KEY=your_key_here
LANGSMITH_TRACING=true
```

---

## Run the Project

### Text retrieval (main entry point)

`AgroMindRetriever` in `src/retriever.py` exports a module-level singleton `retrieval_tool`.

```python
from src.retriever import retrieval_tool

# Health check — confirms collections are loaded and non-empty
print(retrieval_tool.health())
# {'structured_collection': 114, 'support_collection': <N>, 'embedding_model': 'bge-m3'}

# Search the product catalog by natural language
products = retrieval_tool.search_products("root rot citrus", k=5)
for p in products:
    print(p["product_id"], p["name_en"], p["distance"])

# Fetch a product by exact ID
product = retrieval_tool.get_product("AF0001")

# Search historical support conversations
cases = retrieval_tool.search_support_cases("yellow leaves citrus", k=3)

# Filter support cases by category
cases = retrieval_tool.search_support_cases(
    "dosage spray rate", k=3, category="usage_product"
)

# Combined retrieval — main entry point for the agent
context = retrieval_tool.retrieve_context(
    query="what treats downy mildew on vegetables",
    product_k=5,
    support_k=3
)
# Returns: {"products": [...], "support_cases": [...]}
```

### Image-based crop diagnosis

`src/diagnosis_tool.py` combines CLIP image search with text retrieval to diagnose a crop disease from a photo and return matching products.

```python
from src.diagnosis_tool import diagnose_crop_image

result = diagnose_crop_image(
    image_path="path/to/crop_photo.jpg",
    user_text=None   # optional text context
)

if result["success"]:
    print(result["diagnosis"])          # crop, disease, disease_type, confidence
    print(result["recommended_products"])  # from agromind_structured_v4
    print(result["historical_cases"])   # from agromind_full_v4
    print(result["similar_images"])     # from agromind_images_v4
```

This requires the `agromind_images_v4` collection to be populated first (see Known Issues).

### Rebuild collections from source data

```bash
# Rebuild the structured product collection from data/clean_entities.json
python scripts/rebuild_db.py

# Rebuild the support conversation collection from data/cat*.jsonl files
python scripts/build_full_collection.py

# Index annotated crop disease images into agromind_images_v4
# Requires: data/annotated_images/annotations.xlsx + data/annotated_images/images/
python scripts/add_image_collection.py
```

### Verify components independently

```bash
python -m src.embeddings   # Test BGE-M3 connection
python -m src.config       # Test config loading
python test_embedding.py   # Broader Ollama connectivity test
```

---

## API Keys & Environment Variables

| Variable | Required | Description |
|---|---|---|
| `LANGSMITH_API_KEY` | No | Enables LangSmith tracing. Also referenced as `LANGCHAIN_API_KEY` in `config.yaml`. |
| `LANGSMITH_TRACING` | No | `true` or `false`. Defaults to `true` if not set. Overrides `config.yaml`. |

All other settings (model names, collection names, thresholds, paths) come from `config.yaml` and require no environment variables.

The Ollama server does not require an API key. It must be running at `http://localhost:11434`.

---

## Known Issues

**ChromaDB not found on first run**
The `chromadb/` directory is not included in the repository. `AgroMindRetriever.__init__` raises `RuntimeError` if either `agromind_structured_v4` or `agromind_full_v4` is empty or missing. Obtain the pre-built store from the team shared drive, or rebuild with `scripts/rebuild_db.py` and `scripts/build_full_collection.py`.

**Ollama must be running**
All BGE-M3 embedding calls go to `http://localhost:11434/api/embeddings`. If Ollama is not serving, every query fails with a connection error. Run `ollama serve` in a separate terminal before starting the project.

**Dimension mismatch error**
If `bge-m3` returns a different number of dimensions than `config.yaml` specifies (`embedding.dimensions: 1024`), `OllamaEmbeddingFunction._embed` raises a `ValueError`. Confirm the correct model is loaded with `ollama list`.

**6 products have no active ingredient data**
Products AF0014, AF0017, AF0026, AF0029, AF0030, and AF0035 have blank ingredient fields in the source catalog. The audit confirmed the data is unrecoverable from source. These products are indexed with the gap preserved.

**17 products have blank usage/dosage fields**
The source catalog did not include usage method or dosage for these products. Affected IDs are in `quality/audit_report.json`.

**Image pipeline requires annotated dataset not in this repo**
`src/image_embeddings.py`, `src/image_retriever.py`, and `src/diagnosis_tool.py` are fully implemented in code. The `agromind_images_v4` collection is built from an annotated image dataset (`data/annotated_images/annotations.xlsx` + `data/annotated_images/images/`) that is not included in this repository. Until that dataset is present and `scripts/add_image_collection.py` is run, any call to `ImageRetriever` or `diagnose_crop_image` will raise `RuntimeError: Image collection not found`. The unlabeled PlantVillage images in `data/plantvillage_unlabeled/` are not the annotated dataset.

**Image pipeline dependencies missing from requirements.txt**
`src/image_embeddings.py` imports `clip`, `torch`, `PIL`, and `numpy`. `scripts/add_image_collection.py` imports `pandas`. None of these are in `requirements.txt`. Install them separately if using the image features.

**`agent.multimodal.enabled` is false**
`config.yaml` sets `agent.multimodal.enabled: false`. The image collection name and ViT-B/32 model are configured, but the multimodal toggle is off. This does not block `diagnosis_tool.py` directly (it calls `ImageRetriever` directly), but any agent-level orchestration that checks this flag will skip multimodal routing.
