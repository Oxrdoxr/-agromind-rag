# Agro-Mind RAG Knowledge Base

Bilingual (Chinese + English) RAG system for an agricultural products company.
Powers the Agro-Mind autonomous customer support agent with verified,
hallucination-free product knowledge across 114 agricultural products.

---

## What this repo contains

| Section | Description |
|---|---|
| `notebooks/` | One notebook per pipeline phase (01–08 + fine-tuning) |
| `data/entities/` | Normalized product entities — ground truth for the agent |
| `data/retrieval_docs/` | Structured + full bilingual retrieval documents |
| `data/finetuning/` | Cleaned training data (234 examples, safety-corrected) |
| `quality/` | Judge scores, retrieval test results, audit reports |
| `agent/` | `retrieve_agronomy_knowledge.py` — the RAG tool |
| `docs/` | Full pipeline documentation |

---

## Quick start — running the agent (Phase 8)

### Prerequisites

**1. Install Ollama and pull the agent model (local, one-time)**
```bash
# Download Ollama from ollama.com — one-click installer
ollama pull qwen2.5:7b-instruct
ollama serve   # keep this running in the background
```

**2. Add these 4 secrets to your Colab account (left sidebar → key icon)**
```
OPENAI_API_KEY    your OpenAI key
GOOGLE_API_KEY    your Google AI key
GCP_PROJECT       gen-lang-client-0396125347
GCS_BUCKET        agromind-ohoud-2026
```

**3. Get access to the shared Google Drive folder (ask team lead)**
ChromaDB lives here — NOT in the repo.

### Steps

```
1. Clone this repo
2. Open notebooks/notebook_08_agent.ipynb in Google Colab
3. Mount Google Drive when prompted (ChromaDB loads automatically)
4. Upload agent/retrieve_agronomy_knowledge.py when prompted
5. Run all cells
```

### VS Code (local)
```
1. Complete Ollama setup above
2. Download ChromaDB zip from shared Drive, extract locally
3. Update CHROMA_PATH in notebook to your local path:
   CHROMA_PATH = 'C:/Users/yourname/agromind/chromadb'
4. Open notebook_08_agent.ipynb in VS Code
5. Run all cells
```

---

## Using the RAG tool in your own notebook

Your notebook only needs one file: `agent/retrieve_agronomy_knowledge.py`

```python
from retrieve_agronomy_knowledge import retrieve_agronomy_knowledge

# Disease query
results = retrieve_agronomy_knowledge(
    query="what treats root rot in citrus",
    query_type="disease",
    n_results=3
)

# Chinese query
results = retrieve_agronomy_knowledge(
    query="柑橘叶片发黄怎么处理",
    query_type="chinese",
    n_results=3
)

# Each result contains:
# result["product_id"]      e.g. "AF0001"
# result["product_name"]    e.g. "Citrus Junliqing"
# result["product_name_cn"] e.g. "柑橘菌立清"
# result["document"]        full product document text
# result["distance"]        similarity score (lower = more relevant)
```

**Query types:** `disease` · `pest` · `ingredient` · `crop` · `symptom` ·
`dosage` · `safety` · `product_name` · `chinese` · `general`

---

## Infrastructure

| Resource | Location |
|---|---|
| ChromaDB vector store | Google Drive: `/MyDrive/agromind/chromadb/` |
| Embedding model | `text-embedding-3-large` (3072 dims) — must match at query time |
| Agent model | Qwen2.5-7B-Instruct via Ollama (local) |
| Fine-tuned model | `projects/548742106129/locations/us-central1/models/6556995316602634240@1` |
| GCS training data | `gs://agromind-ohoud-2026/agromind/training/` |

---

> **Important:** The Gemini fine-tuned model is for **evaluation only**.
> The agent runs locally on Qwen2.5-7B-Instruct via Ollama.
> Use the tuned model to compare response quality against the Qwen baseline
> during testing — not as the production agent model.

## Pipeline summary

```
Phase 1  Translation        产品目录.xlsx → ProductCatalog_TRANSLATED_v2.xlsx
Phase 2  Entity extraction  → product_entities_v2.json              gpt-5.4-mini
Phase 3  Normalization      → product_entities_normalized_v2.json   free
Phase 4  Retrieval docs     → structured + full bilingual docs       free
Phase 5  AI-as-Judge        structured avg 10.0 · full avg 8.97     gpt-5.4
Phase 6  Embedding          → ChromaDB (2 collections)              $0.07
Phase 7  Retrieval testing  top-1 73% · top-3 96% · top-5 100%
Phase 8  Agent              RAG + Qwen local + safety escalation
FT       Fine-tuning        234 examples → agromind-support-agent-v2 (SUCCEEDED)
```

---

## Quality metrics

| Metric | Value |
|---|---|
| Products in knowledge base | 114 |
| Structured document quality | avg 10.0 / 10 |
| Full document quality | avg 8.97 / 10 |
| All docs pass threshold (≥7) | 114 / 114 |
| Retrieval top-1 accuracy | 73.1% |
| Retrieval top-3 accuracy | 96.2% |
| Retrieval top-5 accuracy | 100% |
| Fine-tuning examples | 234 (safety-corrected) |
| Embedding dimensions | 3072 |

---

## Safety design

- **Pesticide ingestion** → 120 emergency response, skip product info
- **Self-harm / suicidal ideation** → hotline 400-161-9995, human escalation
- **Livestock/pet ingestion** → vet contact
- **Harvest interval** → always 7 days minimum, never unverified "safe to eat"

---

## Known data limitations

6 products (AF0014, AF0017, AF0026, AF0029, AF0030, AF0035) have no active
ingredient data — client source had blank cells with no recoverable information.
Preserved as honest nulls.

9 PN pesticide products have ingredient name conflicts flagged for client
review. Details in `quality/judge_results_full_v2.json`.

---

## Repo structure

```
agromind-rag/
├── notebooks/
│   ├── notebook_01_translation.ipynb
│   ├── notebook_02_entity_extraction.ipynb
│   ├── notebook_03_normalization.ipynb
│   ├── notebook_04_retrieval_docs.ipynb
│   ├── notebook_05_judge.ipynb
│   ├── notebook_06_embedding.ipynb
│   ├── notebook_07_retrieval_testing.ipynb
│   ├── notebook_08_agent.ipynb
│   └── notebook_ft_gemini_v2.ipynb
├── data/
│   ├── entities/
│   │   ├── product_entities_normalized_v2.json
│   │   └── product_entities_normalized_v2.csv
│   ├── retrieval_docs/
│   │   ├── retrieval_documents_structured_v2.json
│   │   └── retrieval_documents_full_v2.json
│   └── finetuning/
│       ├── agromind_training_v2.jsonl
│       └── agromind_training_gemini_v2.jsonl
├── quality/
│   ├── audit_report.json
│   ├── judge_summary_v2.json
│   ├── judge_results_structured_v2.json
│   ├── judge_results_full_v2.json
│   ├── retrieval_test_results_v2.json
│   ├── embedding_log_v2.json
│   ├── fine_tuning_summary.json
│   └── gemini_tuning_job_v2.json
├── agent/
│   └── retrieve_agronomy_knowledge.py
├── docs/
│   └── AGROMIND_PIPELINE_DOCUMENTATION.md
├── .gitignore
└── README.md
```

---

## .gitignore

```
*.xlsx
*.xls
chromadb/
.env
__pycache__/
*.pyc
.DS_Store
```

---

*Built as part of the SDA Agentic AI Capstone — Agro-Mind Autonomous Support Agent*