Almost perfect! You have **duplicate/conflicting sections** that need cleanup. Let me give you the **clean, final version**:

---

# Agro-Mind RAG Pipeline — Complete Documentation (v3 - LangChain)
## Project: Autonomous Agricultural Customer Support Agent
### Team Reference Document

---

## Table of Contents
1. Project Overview
2. Pipeline Summary
3. Phase-by-Phase Breakdown
4. Complete File Reference
5. Key Decisions & Why
6. Quality Metrics
7. Technical Decisions for v3 (LangChain)
8. How to Resume / Hand Off

---

## 1. Project Overview

**What we built:** A bilingual (Chinese + English) RAG knowledge base for an agricultural products company. The knowledge base powers an agentic customer support system that answers farmer queries about product usage, crop diagnosis, dosage, logistics, and safety — without hallucinating.

**Client data:** 114 agricultural products (pesticides, microbial agents, fertilizers, adjuvants, herbicides) from a Chinese product catalog. Source data was in Chinese with some blank fields.

**Agent model:** Qwen2.5-7B-Instruct running locally via Ollama. No cloud API cost per inference call. Works offline after setup.

**Core principle applied throughout:** Honest nulls over hallucinated data. Where the client did not provide information, we preserved the gap rather than inventing content. This is critical for a safety-sensitive agricultural system where wrong dosage or safety information can cause real harm.

---

## 2. Pipeline Summary

```
Raw Chinese XLSX
      ↓ Phase 1 — Audit + Clean + Translate [gpt-4.1-mini]
Translated XLSX (bilingual, cleaned)
      ↓ Phase 2 — Entity Extraction [gpt-5.4-mini]
Structured JSON entities (114 products)
      ↓ Phase 3 — Normalization [free, Python only]
Normalized JSON entities (clean, consistent)
      ↓ Phase 4 — Retrieval Document Generation [free, Python only]
Two retrieval document files (structured + full bilingual)
      ↓ Phase 5 — AI-as-Judge [gpt-5.4]
Quality verified (structured avg 10.0 / full avg 8.97)
      ↓ Phase 6 — Embedding + ChromaDB [BGE-M3 via Ollama, LangChain wrapper]
Vector store (2 collections, persisted locally)
      ↓ Phase 7 — Retrieval Testing + Tool Building
LangChain-compatible retriever with hybrid search
      ↓ Phase 8 — LangChain Agent [Qwen2.5-7B-Instruct local]
Agro-Mind autonomous support agent with RAG tools
```

**Parallel track — Fine-tuning:**
```
Client chat data (230 examples)
      ↓ Safety cleaning + format conversion
Cleaned training data (234 examples, safety-corrected)
      ↓ Gemini 2.5 Flash SFT via Vertex AI ← SUCCEEDED ✓
Tuned model: projects/548742106129/locations/us-central1/models/6556995316602634240@1
      ↓ Optional: use for demo/evaluation vs Qwen baseline
```

---

## 3. Phase-by-Phase Breakdown

### Phase 1 — Audit + Clean + Translate
**Notebook:** `notebook_01_translation.ipynb`
**Model:** `gpt-4.1-mini`
**Cost:** ~$1.50

**What we did:**
- Audited the raw Chinese XLSX for blank cells before spending any API calls
- Found 6 products with blank 主要成分 (Main Ingredients) — confirmed unrecoverable from source
- Found 17 products with blank 使用方法和剂量 (Usage Method) — source limitation preserved honestly
- No duplicates found (114 unique product IDs)
- Added `ingredient_source = 'not_in_source'` flag to the 6 blank-ingredient products
- Translated with column-aware prompt — preserving CFU counts, Bacillus species names, dosage ratios exactly
- Added [DISEASE]/[PEST]/[SYMPTOM] inline tags to Usage Instructions during translation
- Standardized dilution units (jin → metric: 1 jin = 0.5L)
- Added Chinese product names back as separate column

**Key decision:** Audit before translation. Translating problems costs money and doesn't fix them.

---

### Phase 2 — Entity Extraction
**Notebook:** `notebook_02_entity_extraction.ipynb`
**Model:** `gpt-5.4-mini`
**Cost:** ~$0.50

**What we did:**
- Extracted 12 structured fields per product from translated text
- Used the [DISEASE]/[PEST]/[SYMPTOM] tags from Phase 1 as extraction signals
- Explicit prompt rules: nematodes → target_pests (NOT diseases), symptoms → symptoms field
- Stored both Chinese name (product_name_cn) and English name (product_name) in every record
- Post-extraction audit: removed vague generic terms, moved nematodes out of diseases
- 114/114 products extracted successfully, zero failures

**Key decision:** Used gpt-5.4-mini for extraction because errors here propagate through all downstream phases.

---

### Phase 3 — Normalization
**Notebook:** `notebook_03_normalization.ipynb`
**Model:** None (free, pure Python)
**Cost:** $0

**What we did:**
- Standardized capitalization: Title Case for crops, diseases, pests; lowercase for symptoms
- Canonicalized 51 product type variants → 24 clean canonical types
- Removed duplicate entries within each field
- Zero case variants remaining after normalization (verified)

---

### Phase 4 — Retrieval Document Generation
**Notebook:** `notebook_04_retrieval_docs.ipynb`
**Model:** None (free, pure Python)
**Cost:** $0

**What we did:**
- Generated two document types per product:
  - **Structured document** — compact English summary for precise field-level queries
  - **Full document** — structured summary + English prose + original Chinese prose for semantic queries
- Both document types include Chinese product name in every document
- Stripped [DISEASE]/[PEST]/[SYMPTOM] tags from prose before storing
- Full document average: 4,632 chars

**Key decision:** Two collections instead of one = precision tool + semantic tool. No chunking needed.

---

### Phase 5 — AI-as-Judge
**Notebook:** `notebook_05_judge.ipynb`
**Model:** `gpt-5.4` (full flagship model)
**Cost:** ~$5.00

**What we did:**
- Evaluated all 114 structured documents and all 114 full documents
- Structured: 114/114 scored 10/10 — perfect
- Full: 105/114 scored ≥8, 9/114 scored 7
- All 114 documents cleared for embedding (all score ≥7)
- Judge found 9 PN products with cross-section conflicts for client review

**Products flagged for client review:** PN0004, PN0005, PN0007, PN0008, PN0011, PN0014, PN0031, PN0037, PN0041

---

### Phase 6 — Embedding + ChromaDB (v3 - LangChain)
**Script:** `scripts/rebuild_db.py` or `notebook_06_embedding.ipynb`
**Model:** `bge-m3` via Ollama (1024 dimensions)
**Cost:** $0 (local)

**What we did:**
- Switched from OpenAI `text-embedding-3-large` to local `bge-m3` model
- Created custom `OllamaEmbeddingsWrapper` to make Ollama work with LangChain
- Embedded all 228 documents (114 structured + 114 full) into ChromaDB
- ChromaDB persists locally at `./chromadb/`
- Used LangChain's `Chroma` wrapper for seamless integration

**Embedding wrapper implementation:**
```python
class OllamaEmbeddingsWrapper(Embeddings):
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return ollama_embeddings(texts)
    
    def embed_query(self, text: str) -> List[float]:
        return ollama_embeddings.embed_query(text)
```

**Key decision:** Local embeddings eliminate API costs and work offline. BGE-M3 supports bilingual retrieval natively.

---

### Phase 7 — Retrieval Testing + Tool Building (v3 - LangChain)
**Script:** `src/retrieval_tool.py`
**Model:** BGE-M3 (local via Ollama)
**Cost:** $0

**What we did:**
- Built `AgroMindRetriever` class with hybrid search capabilities
- Integrated with LangChain's `similarity_search_with_score`
- Added exact matching for product IDs (bypasses vector search)
- Added disease-specific search with exact disease → product mapping
- Implemented search cache for repeated queries
- Created `create_rag_tool()` function for LangChain agent integration

**Retrieval capabilities:**
```python
retriever = AgroMindRetriever()

# All supported search methods
retriever.search("citrus root rot")           # Hybrid (exact + vector)
retriever.search_by_disease("炭疽病")         # Disease-specific
retriever.search_by_ingredient("copper")      # Ingredient search
retriever.search_by_crop("citrus")            # Crop search
retriever.get_product_info("AF0001")          # Exact ID lookup
```

**LangChain tool wrapper:**
```python
def create_rag_tool(retriever):
    def search_func(query: str) -> str:
        results = retriever.search(query, k=3)
        return format_results(results)
    
    return Tool(
        name="AgroMind_Product_Search",
        func=search_func,
        description="Search agricultural product database..."
    )
```

---

### Fine-tuning (Parallel Track) — COMPLETED ✓
**Notebooks:** `notebook_ft_finetuning.ipynb` + `notebook_ft_gemini_v2.ipynb`
**Model:** Gemini 2.5 Flash (Vertex AI SFT)
**Status:** SUCCEEDED
**Tuned model:** `projects/548742106129/locations/us-central1/models/6556995316602634240@1`

**What we did:**
- Cleaned 230 real client customer service examples
- Removed 2 self-harm/suicidal ideation examples
- Removed 2 examples with unsafe factual claims
- Added 8 replacement safety examples with correct escalation behavior
- Added system prompt to all 234 examples
- Converted OpenAI format → Gemini format
- Submitted to Vertex AI, job succeeded after ~1.5 hours

**Safety behaviors trained:**
- Pesticide ingestion → call 120 immediately
- Suicidal ideation → mental health hotline 400-161-9995
- Livestock safety → contact vet
- Harvest interval → 7 days minimum, never "safe to eat" unverified
- Child safety → ventilate 24h, lock storage

**Purpose in this project — EVALUATION ONLY:** The fine-tuned Gemini model is NOT the production agent model. The agent runs locally on Qwen2.5-7B-Instruct via Ollama.

---

### Phase 8 — LangChain Agent (v3)
**File:** `agent_main.py` or notebook
**Model:** Qwen2.5-7B-Instruct via Ollama
**Tools:** RAG retriever + optional weather/pest alerts

**What we built:**
- LangChain agent with `create_rag_tool()` as primary tool
- Local inference via Ollama (no cloud costs)
- Conversation memory (optional)
- Safety escalation patterns from fine-tuning data

**Complete agent example:**
```python
from langchain.agents import initialize_agent
from langchain_ollama import ChatOllama
from src.retrieval_tool import AgroMindRetriever, create_rag_tool

# Initialize components
retriever = AgroMindRetriever()
rag_tool = create_rag_tool(retriever)
llm = ChatOllama(model="qwen2.5:7b-instruct", temperature=0.1)

# Create agent
agent = initialize_agent(
    tools=[rag_tool],
    llm=llm,
    agent="zero-shot-react-description",
    verbose=True,
    max_iterations=3,
    early_stopping_method="generate"
)

# Run queries
response = agent.run("我的柑橘叶片发黄怎么处理？")
```

**Adding more tools:**
```python
from langchain.tools import Tool

def get_weather(city: str) -> str:
    return f"Weather in {city}: 25°C, 65% humidity"

weather_tool = Tool(
    name="Weather",
    func=get_weather,
    description="Get weather for disease risk assessment"
)

agent = initialize_agent(
    tools=[rag_tool, weather_tool],  # Multiple tools
    llm=llm,
    agent="zero-shot-react-description"
)
```

**Agent capabilities:**
- Intent classification (9 types)
- RAG retrieval via AgroMindRetriever
- Safety escalation with human handoff
- Interactive bilingual chat interface
- Conversation history tracking

---

## 4. Complete File Reference

### Source Files (archive, do not modify)
| File | Description |
|------|-------------|
| 产品目录_ProductCatalog_Mar_6.xlsx | Raw Chinese source from client |
| cat1_usage_product_real.jsonl | 60 real usage/dosage support conversations |
| cat2_diagnosis_real.jsonl | 60 real diagnosis support conversations |
| cat3_aftersales_logistics_real.jsonl | 60 real logistics support conversations |
| cat4_safety_sensitive_real.jsonl | 50 real safety-sensitive conversations |

### Pipeline Output Files
| File | Phase | Use in Phase 8? | Description |
|------|-------|-----------------|-------------|
| ProductCatalog_TRANSLATED_v2.xlsx | 1 | No | Translated catalog |
| 产品目录_CLEANED.xlsx | 1 | No | Audited Chinese source |
| product_entities_v2.json | 2 | No | Raw extracted entities |
| product_entities_normalized_v2.json | 3 | YES | Ground truth, ChromaDB metadata |
| retrieval_documents_structured_v2.json | 4 | No (in ChromaDB) | Already embedded |
| retrieval_documents_full_v2.json | 4 | No (in ChromaDB) | Already embedded |
| judge_summary_v2.json | 5 | No | Summary scores + client items |
| embedding_log_v2.json | 6 | No | Model + dims + path |
| retrieval_test_results_v2.json | 7 | No | Full test results |
| agromind_training_v2.jsonl | FT | YES | 234 examples for system prompt |
| gemini_tuning_job_v2.json | FT | YES | Contains tuned model name |

### Source Code (v3 - LangChain)
| File | Purpose |
|------|---------|
| `src/__init__.py` | Module initializer |
| `src/config.py` | Configuration loader from config.yaml |
| `src/embeddings.py` | Ollama embedding wrapper (LangChain compatible) |
| `src/retrieval_tool.py` | Main retriever with hybrid search |
| `scripts/rebuild_db.py` | Build ChromaDB from entities |
| `scripts/eval_retrieval.py` | Test retrieval quality |
| `config.yaml` | Model paths, dimensions, thresholds |

### Infrastructure
| Resource | Location | Shared via |
|----------|----------|------------|
| ChromaDB vector store | `./chromadb/` | Google Drive shared folder |
| Raw XLSX files | Google Drive | Google Drive shared folder |
| GCS training data | `gs://agromind-ohoud-2026/` | Vertex AI only |

---

## 5. Key Decisions & Why

**Why Qwen2.5-7B-Instruct for the agent**
Runs locally via Ollama — no API cost per inference call, works offline, fast iteration.

**Why LangChain for v3**
- Standardization across team agent projects
- Easy tool ecosystem (weather, pest alerts, calculators)
- Built-in conversation memory and LangSmith observability

**Why BGE-M3 over OpenAI embeddings**
| Aspect | OpenAI | BGE-M3 (Ollama) |
|--------|--------|-----------------|
| Cost | $0.13/1M tokens | $0 |
| Privacy | Data sent to OpenAI | Local only |
| Latency | ~200ms | ~50ms |
| Chinese support | Good | Native |
| Dimensions | 3072 | 1024 |

**Why custom embedding wrapper**
LangChain's `OllamaEmbeddings` had dimension issues. Our wrapper uses the proven `ollama_embeddings()` function and implements LangChain's interface correctly.

**Why hybrid search**
- Exact product ID → instant lookup
- Disease name → direct metadata mapping
- Vector search → symptom descriptions
- Combined gives best of both worlds

**Why we preserved blank cells instead of filling them**
6 products have no active ingredient data. Fabricating ingredients for a pesticide product is a safety risk. Honest nulls are better than wrong data.

**Why the fine-tuning data is still valuable**
The 234 cleaned examples teach correct tone, escalation patterns, and safety behavior. Used as few-shot examples in the Qwen system prompt.

---

## 6. Quality Metrics

| Metric | Value | Meaning |
|--------|-------|---------|
| Products in KB | 114 | Complete catalog |
| Structured doc judge avg | 10.0/10 | Perfect |
| Full doc judge avg | 8.97/10 | Excellent |
| All docs pass threshold (≥7) | 114/114 | All cleared for embedding |
| Retrieval top-1 accuracy | 73.1% | 19/26 queries |
| Retrieval top-3 accuracy | 96.2% | 25/26 queries |
| Retrieval top-5 accuracy | 100% | 26/26 queries |
| Product name lookup | 100% | EN + CN both work |
| Chinese query accuracy | 75% | Bilingual retrieval working |
| Fine-tuning examples | 234 | Cleaned, safety-corrected |
| Embedding dimensions | 1024 | BGE-M3 (local) |

---

## 7. Technical Decisions for v3 (LangChain)

### File Architecture Decisions
```
agromind-v3/
├── src/
│   ├── __init__.py           # Module initializer (MUST exist)
│   ├── config.py             # Configuration loader
│   ├── embeddings.py         # Ollama → LangChain wrapper
│   └── retrieval_tool.py     # Main retriever (USE THIS)
├── scripts/
│   ├── rebuild_db.py         # Build ChromaDB from entities
│   └── eval_retrieval.py     # Test retrieval quality
├── data/
│   └── clean_entities.json   # Product ground truth (NOT in git)
├── chromadb/                  # Vector store (NOT in git)
├── config.yaml                # Model paths & thresholds
└── test_embedding.py         # Verify Ollama works
```

### What to Delete (Redundant)
- `src/chroma_client.py` — Not needed (retrieval_tool.py does this)

### What to Keep
- `src/retrieval_tool.py` — Main retriever, LangChain compatible
- `src/embeddings.py` — Your working Ollama wrapper
- All LangChain packages in requirements.txt

---

## 8. How to Resume / Hand Off

### Setup for New Team Member

```bash
# 1. Clone repo and set up environment
git clone <repo-url>
cd agromind-v3
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install chromadb ollama langchain langchain-chroma langchain-ollama pydantic pyyaml

# 3. Install Ollama and pull models
# Download from ollama.com
ollama pull bge-m3
ollama pull qwen2.5:7b-instruct
ollama serve  # Keep running in background

# 4. Get data from shared drive
# Download chromadb/ folder and data/clean_entities.json
# Place in project root

# 5. Create __init__.py files
touch src/__init__.py
touch tests/__init__.py

# 6. Test everything
python test_embedding.py
python -m src.retrieval_tool
```

### Running the LangChain Agent

```python
from langchain.agents import initialize_agent
from langchain_ollama import ChatOllama
from src.retrieval_tool import AgroMindRetriever, create_rag_tool

retriever = AgroMindRetriever()
rag_tool = create_rag_tool(retriever)
llm = ChatOllama(model="qwen2.5:7b-instruct", temperature=0.1)

agent = initialize_agent(
    tools=[rag_tool],
    llm=llm,
    agent="zero-shot-react-description",
    verbose=True
)

response = agent.run("What treats citrus root rot?")
```

### Sharing ChromaDB with Teammates

ChromaDB does NOT go in GitHub. Share via Google Drive:
- Right-click `chromadb/` folder in Drive → Share
- Add teammate Gmail addresses
- Permission: Editor
- Setting: Restricted (NOT "anyone with link")

### Adding New Products

1. Add to `产品目录_CLEANED.xlsx`
2. Run notebooks 01 → 02 → 03 → 04 for new products only
3. Run notebook_05 judge on new documents
4. Run `scripts/rebuild_db.py` to add to ChromaDB (preserves existing)
5. Re-run retrieval testing to verify

### Required Secrets (for OpenAI phases only)

| Secret | Value | Used In |
|--------|-------|---------|
| OPENAI_API_KEY | your OpenAI key | Phases 1, 2, 5 |
| GOOGLE_API_KEY | your Google AI key | Fine-tuning |
| GCP_PROJECT | gen-lang-client-0396125347 | Fine-tuning |
| GCS_BUCKET | agromind-ohoud-2026 | Fine-tuning |

**Note:** Phase 6 (embedding) and Phase 8 (agent) use local Ollama — no secrets needed.

---

### Common Issues & Fixes

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError: No module named 'src'` | Run from project root or use `python -m src.retrieval_tool` |
| Ollama connection error | Run `ollama serve` in separate terminal |
| Dimension mismatch | Check `config.yaml` has `embedding.dimensions: 1024` |
| ChromaDB collection not found | Run `scripts/rebuild_db.py` to rebuild |
| `__init__.py` missing | Create empty file: `touch src/__init__.py` |

---

*Built as part of SDA Agentic AI Capstone — Agro-Mind Autonomous Support Agent*
*Documentation v3 - LangChain Edition | Last Updated: June 2026*

---

## Summary of Changes Made

| Section | Status |
|---------|--------|
| Phase 6 (Embedding) | ✅ Updated to BGE-M3 + LangChain wrapper |
| Phase 7 (Retrieval) | ✅ Updated to LangChain retriever with hybrid search |
| Phase 8 (Agent) | ✅ Updated to LangChain agent with tools |
| Technical Decisions | ✅ Added new section for v3 decisions |
| File Reference | ✅ Updated to show v3 structure |
| Removed duplicates | ✅ Cleaned conflicting sections |
| Added setup guide | ✅ Complete handoff instructions |

This is now **clean, consistent, and ready for your team**! 🚀