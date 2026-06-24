# Agro-Mind RAG

## Project Summary

Bilingual (Chinese + English) RAG knowledge base for agricultural products, powering an autonomous customer support agent. The system answers farmer queries about product usage, crop diagnosis, dosage, logistics, and safety without hallucinating.

- **114 agricultural products** (pesticides, microbial agents, fertilizers, adjuvants, herbicides) sourced from a Chinese product catalog
- **LLM:** Qwen2.5-7B-Instruct via Ollama — fully local inference, no cloud cost per query
- **Embeddings:** BGE-M3 via Ollama — 1024-dimensional, bilingual, local
- **Vector store:** ChromaDB with two named collections (`agromind_structured_v4`, `agromind_full_v4`)
- **Tracing:** LangSmith (optional, configurable via `.env`)

The design principle throughout: honest nulls over hallucinated data. 6 products have no active ingredient data in the source; those gaps are preserved rather than invented.

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

**External requirement — Ollama:** must be installed separately from [ollama.com](https://ollama.com) and running locally on port `11434`.

---

## Installation

**1. Install Ollama and pull the required models**

```bash
# Install from ollama.com, then:
ollama pull bge-m3               # Embedding model (1024 dims)
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

The main entry point is `AgroMindRetriever` in `src/retriever.py`. A module-level singleton `retrieval_tool` is exported for direct import.

```python
from src.retriever import retrieval_tool

# Health check — confirms collections are loaded
print(retrieval_tool.health())
# {'structured_collection': 114, 'support_collection': 230, 'embedding_model': 'bge-m3'}

# Search the product catalog by natural language
products = retrieval_tool.search_products("root rot citrus", k=5)
for p in products:
    print(p["product_id"], p["name_en"], p["distance"])

# Fetch a product by exact ID
product = retrieval_tool.get_product("AF0001")

# Search historical support cases
cases = retrieval_tool.search_support_cases("yellow leaves citrus", k=3)

# Search support cases filtered by category
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

**Verify embeddings work independently:**

```bash
python -m src.embeddings
```

**Verify configuration loads correctly:**

```bash
python -m src.config
```

---

## API Keys & Environment Variables

| Variable | Required | Description |
|---|---|---|
| `LANGSMITH_API_KEY` | No | Enables LangSmith tracing. Also read as `LANGCHAIN_API_KEY` (see `config.yaml`). |
| `LANGSMITH_TRACING` | No | `true` or `false`. Defaults to `true` if not set. Overrides `config.yaml`. |

All other settings (model names, collection names, thresholds, paths) come from `config.yaml` and require no environment variables.

The Ollama server does not require an API key. It must be running at `http://localhost:11434`.

---

## Known Issues

**ChromaDB not found on first run**
The `chromadb/` directory is not included in the repository. `AgroMindRetriever.__init__` will raise a `RuntimeError` if either collection (`agromind_structured_v4` or `agromind_full_v4`) is empty or missing. Obtain the pre-built store from the team shared drive.

**Ollama must be running**
All embedding calls go to `http://localhost:11434/api/embeddings`. If Ollama is not serving, every query will fail with a connection error. Run `ollama serve` in a separate terminal before starting the project.

**Dimension mismatch error**
If `bge-m3` returns a different number of dimensions than `config.yaml` specifies (`embedding.dimensions: 1024`), `OllamaEmbeddingFunction._embed` raises a `ValueError`. This can happen if a different Ollama model responds at the same endpoint. Confirm the correct model is loaded with `ollama list`.

**6 products have no active ingredient data**
Products AF0014, AF0017, AF0026, AF0029, AF0030, and AF0035 have blank ingredient fields in the source catalog. The audit confirmed the data is unrecoverable from source. These products are indexed with the gap preserved.

**17 products have blank usage/dosage fields**
The source catalog did not include usage method or dosage for these products. Affected IDs are documented in `quality/audit_report.json`.

**Multimodal support is disabled**
`config.yaml` includes image collection and ViT-B/32 settings, but `agent.multimodal.enabled` is set to `false`. The `agromind_images_v4` collection is defined in `config.yaml`, but the image data lives in the separate `agromind-rag` repository and has not yet been integrated into this retriever.
