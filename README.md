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

Add these 4 secrets to your Colab account (left sidebar → key icon):

```
OPENAI_API_KEY    your OpenAI key
GOOGLE_API_KEY    your Google AI key
GCP_PROJECT       gen-lang-client-0396125347
GCS_BUCKET        agromind-ohoud-2026
```

### Steps

```
1. Clone this repo
2. Get access to the shared Google Drive folder (ask team lead)
3. Open notebooks/notebook_08_agent.ipynb in Google Colab
4. Mount Google Drive when prompted (ChromaDB loads automatically)
5. Upload agent/retrieve_agronomy_knowledge.py when prompted
6. Run all cells
```

The agent starts an interactive chat — type in English or Chinese.

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
results = retrieve_agromind_knowledge(
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
| Embedding model | `text-embedding-3-large` (3072 dims) |
| Agent model | `gpt-5.4` |
| Fine-tuned model | see `quality/gemini_tuning_job_v2.json` |
| GCS training data | `gs://agromind-ohoud-2026/agromind/training/` |

---

## Pipeline summary

```
Phase 1  Translation        产品目录.xlsx → ProductCatalog_TRANSLATED_v2.xlsx
Phase 2  Entity extraction  → product_entities_v2.json              gpt-5.4-mini
Phase 3  Normalization      → product_entities_normalized_v2.json   free
Phase 4  Retrieval docs     → structured + full bilingual docs       free
Phase 5  AI-as-Judge        structured avg 10.0 · full avg 8.97     gpt-5.4
Phase 6  Embedding          → ChromaDB (2 collections)              $0.07
Phase 7  Retrieval testing  top-1 73% · top-3 96% · top-5 100%
Phase 8  Agent              RAG + tuned model + safety escalation
FT       Fine-tuning        234 examples → agromind-support-agent-v2
```

---

## Quality metrics

| Metric | Value |
|---|---|
| Products in knowledge base | 114 |
| Structured document quality | avg 10.0 / 10 |
| Full document quality | avg 8.97 / 10 |
| Retrieval top-1 accuracy | 73.1% |
| Retrieval top-3 accuracy | 96.2% |
| Retrieval top-5 accuracy | 100% |
| Fine-tuning examples | 234 (safety-corrected) |
| Embedding dimensions | 3072 |

---

## Safety design

The agent handles safety-sensitive cases with zero tolerance for errors:

- **Pesticide ingestion** → immediate 120 emergency response, skip product info
- **Self-harm / suicidal ideation** → mental health hotline 400-161-9995, human escalation
- **Livestock/pet ingestion** → vet contact, not product safety info
- **Harvest interval queries** → always 7 days minimum, never "safe to eat" without verification

These behaviors are enforced through both the system prompt (few-shot examples
from real client conversations) and the fine-tuned Gemini model.

---

## Known data limitations

6 products (AF0014, AF0017, AF0026, AF0029, AF0030, AF0035) have no
active ingredient data — the client's source file had blank cells and the
information was not present in any other column. These are preserved as
honest nulls rather than fabricated values.

9 PN pesticide products have ingredient name conflicts between the product
name and the label text. Details in `quality/judge_results_full_v2.json`.
These are flagged for client review.

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
