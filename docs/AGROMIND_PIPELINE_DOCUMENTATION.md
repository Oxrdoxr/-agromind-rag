# Agro-Mind RAG Pipeline — Complete Documentation
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
7. What's Left — Phase 8
8. How to Resume / Hand Off

---

## 1. Project Overview

**What we built:** A bilingual (Chinese + English) RAG knowledge base for an
agricultural products company. The knowledge base powers an agentic customer
support system that answers farmer queries about product usage, crop diagnosis,
dosage, logistics, and safety — without hallucinating.

**Client data:** 114 agricultural products (pesticides, microbial agents,
fertilizers, adjuvants, herbicides) from a Chinese product catalog. Source
data was in Chinese with some blank fields.

**Agent model:** Qwen2.5-7B-Instruct running locally via Ollama. No cloud API
cost per inference call. Works offline after setup.

**Core principle applied throughout:** Honest nulls over hallucinated data.
Where the client did not provide information, we preserved the gap rather than
inventing content. This is critical for a safety-sensitive agricultural system
where wrong dosage or safety information can cause real harm.

---

## 2. Pipeline Summary

```
Raw Chinese XLSX
      ↓ Phase 1 — Audit + Clean + Translate
Translated XLSX (bilingual, cleaned)
      ↓ Phase 2 — Entity Extraction  [gpt-5.4-mini]
Structured JSON entities (114 products)
      ↓ Phase 3 — Normalization  [free, Python only]
Normalized JSON entities (clean, consistent)
      ↓ Phase 4 — Retrieval Document Generation  [free, Python only]
Two retrieval document files (structured + full bilingual)
      ↓ Phase 5 — AI-as-Judge  [gpt-5.4]
Quality verified (structured avg 10.0 / full avg 8.97)
      ↓ Phase 6 — Embedding + ChromaDB  [text-embedding-3-large]
Vector store (2 collections, persisted to Google Drive)
      ↓ Phase 7 — Retrieval Testing
RAG tool built and verified (73.1% top-1 / 96.2% top-3 / 100% top-5)
      ↓ Phase 8 — Agent  [NEXT]
Agro-Mind autonomous support agent (Qwen2.5-7B-Instruct local)
```

**Parallel track — Fine-tuning:**
```
Client chat data (230 examples)
      ↓ Safety cleaning + format conversion
Cleaned training data (234 examples, safety-corrected)
      ↓ Gemini 2.5 Flash SFT via Vertex AI  ← SUCCEEDED ✓
Tuned model: projects/548742106129/locations/us-central1/models/6556995316602634240@1
      ↓ Optional: use for demo/evaluation vs Qwen baseline
```

**Note on OpenAI fine-tuning:** OpenAI deprecated self-serve fine-tuning for
new organizations in May 2026. Training data was prepared in OpenAI format
(agromind_training_v2.jsonl), converted to Gemini format, and successfully
trained on Vertex AI instead. The training data is model-agnostic and can be
used to fine-tune Qwen or any future model.

---

## 3. Phase-by-Phase Breakdown

### Phase 1 — Audit + Clean + Translate
**Notebook:** `notebook_01_translation.ipynb`
**Model:** `gpt-4.1-mini`
**Cost:** ~$1.50

**What we did:**
- Audited the raw Chinese XLSX for blank cells before spending any API calls
- Found 6 products with blank 主要成分 (Main Ingredients) — confirmed
  unrecoverable from source (marketing prose only, no chemical data)
- Found 17 products with blank 使用方法和剂量 (Usage Method) — source
  limitation preserved honestly
- No duplicates found (114 unique product IDs)
- Added `ingredient_source = 'not_in_source'` flag to the 6 blank-ingredient
  products for downstream tracking
- Translated with column-aware prompt — preserving CFU counts, Bacillus
  species names, dosage ratios exactly
- Added [DISEASE]/[PEST]/[SYMPTOM] inline tags to Usage Instructions during
  translation — this eliminated category mixing in extraction
- Standardized dilution units (jin → metric: 1 jin = 0.5L)
- Added Chinese product names back as separate column (critical for Chinese
  user retrieval)

**Key decision:** Audit before translation. Translating problems costs money
and doesn't fix them.

---

### Phase 2 — Entity Extraction
**Notebook:** `notebook_02_entity_extraction.ipynb`
**Model:** `gpt-5.4-mini`
**Cost:** ~$0.50

**What we did:**
- Extracted 12 structured fields per product from translated text
- Used the [DISEASE]/[PEST]/[SYMPTOM] tags from Phase 1 as extraction signals
- Explicit prompt rules: nematodes → target_pests (NOT diseases), symptoms →
  symptoms field
- Stored both Chinese name (product_name_cn) and English name (product_name)
  in every record — critical for bilingual retrieval
- Post-extraction audit: removed vague generic terms (bacteria, fungi,
  viruses), moved nematodes out of diseases, removed CFU specs from
  ingredients
- 114/114 products extracted successfully, zero failures

**Key decision:** Used gpt-5.4-mini (a strong mid-tier model) for extraction
because errors here propagate through all downstream phases. Quality at
extraction time saves correction cost later.

---

### Phase 3 — Normalization
**Notebook:** `notebook_03_normalization.ipynb`
**Model:** None (free, pure Python)
**Cost:** $0

**What we did:**
- Standardized capitalization: Title Case for crops, diseases, pests;
  lowercase for symptoms
- Canonicalized 51 product type variants → 24 clean canonical types
  e.g. "Adjuvant Growth", "Auxiliary Growth", "Growth Aid" → "Adjuvant Growth
  Promotion"
  e.g. "Aqueous solution", "Soluble liquid" → "Soluble Concentrate"
- Removed duplicate entries within each field
- Zero case variants remaining after normalization (verified)

---

### Phase 4 — Retrieval Document Generation
**Notebook:** `notebook_04_retrieval_docs.ipynb`
**Model:** None (free, pure Python)
**Cost:** $0

**What we did:**
- Generated two document types per product:

  Structured document — compact English summary of all extracted fields.
  Use for: precise field-level queries ("what products contain Bacillus
  subtilis", "dosage for AF0001")

  Full document — structured summary + English translated prose + original
  Chinese prose (3 sections).
  Use for: semantic queries ("my citrus leaves are yellowing") and Chinese
  farmer queries ("柑橘叶片发黄怎么处理")

- Both document types include Chinese product name in every document
- Stripped [DISEASE]/[PEST]/[SYMPTOM] tags from prose before storing (useful
  for extraction, noise for search)
- Full document average: 4,632 chars — rich enough for semantic search

**Key decision:** Two collections instead of one = precision tool + semantic
tool. No chunking needed — all documents fit within embedding model's 8,191
token limit.

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
- Judge found 9 PN products with cross-section conflicts — active ingredient
  names differ between product label and structured fields. Source data issues
  to report to client.

**Products flagged for client review:**
PN0004, PN0005, PN0007, PN0008, PN0011, PN0014, PN0031, PN0037, PN0041

---

### Phase 6 — Embedding + ChromaDB
**Notebook:** `notebook_06_embedding.ipynb`
**Model:** `text-embedding-3-large` (3072 dimensions)
**Cost:** ~$0.07

**What we did:**
- Embedded all 228 documents (114 structured + 114 full)
- Stored in ChromaDB with rich metadata per product:
  product_types, target_crops, target_diseases, target_pests,
  active_ingredients, is_pesticide, is_microbial, is_fertilizer,
  has_diseases, has_pests
- Persisted to Google Drive: /MyDrive/agromind/chromadb/
- Two collections: collection_structured and collection_full

**Key decision:** text-embedding-3-large over text-embedding-3-small —
multilingual retrieval matters for Chinese queries, cost difference on 228
documents is ~5 cents.

---

### Phase 7 — Retrieval Testing
**Notebook:** `notebook_07_retrieval_testing.ipynb`
**Model:** `text-embedding-3-large` (query embedding only)
**Cost:** ~$0.001

**What we did:**
- Built a 26-query test set covering 9 query types
- Results: 73.1% top-1 / 96.2% top-3 / 100% top-5
- Product name queries: 100%
- Ingredient queries: 100% after metadata filtering
- Chinese queries: 75%
- Built retrieve_agronomy_knowledge() with learned routing rules:
  Disease/pest/ingredient → metadata filter first, then semantic search
  Symptom/Chinese/general → full bilingual collection
  Safety → pesticide-only filter

---

### Fine-tuning (Parallel Track) — COMPLETED ✓
**Notebooks:** `notebook_ft_finetuning.ipynb` + `notebook_ft_gemini_v2.ipynb`
**Model:** Gemini 2.5 Flash (Vertex AI SFT)
**Status:** SUCCEEDED
**Tuned model:** projects/548742106129/locations/us-central1/models/6556995316602634240@1

**What we did:**
- Cleaned 230 real client customer service examples:
  Removed 2 self-harm/suicidal ideation examples
  Removed 2 examples with unsafe factual claims
  Added 8 replacement safety examples with correct escalation behavior
- Added system prompt to all 234 examples
- Converted OpenAI format → Gemini format, fixed last-turn validation error
- Submitted to Vertex AI, job succeeded after ~1.5 hours

**Safety behaviors trained:**
- Pesticide ingestion → call 120 immediately
- Suicidal ideation → mental health hotline 400-161-9995
- Livestock safety → contact vet
- Harvest interval → 7 days minimum, never "safe to eat" unverified
- Child safety → ventilate 24h, lock storage

**Usage in Phase 8:**
The fine-tuned Gemini model is a cloud model (Vertex AI only). The primary
agent model is Qwen2.5-7B-Instruct running locally. The tuned model can be
used for:
1. Evaluation: compare Qwen baseline vs fine-tuned Gemini on same queries
2. Demo: use Gemini tuned model for presentation if Qwen output is insufficient
3. Future: fine-tune Qwen locally using the same training data

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
| product_entities_normalized_v2.json | 3 | YES — metadata | Ground truth, ChromaDB metadata |
| product_entities_normalized_v2.csv | 3 | No | Human-readable review |
| retrieval_documents_structured_v2.json | 4 | No (in ChromaDB) | Already embedded |
| retrieval_documents_full_v2.json | 4 | No (in ChromaDB) | Already embedded |
| judge_summary_v2.json | 5 | No | Summary scores + client items |
| judge_results_structured_v2.json | 5 | No | Detailed scores per product |
| judge_results_full_v2.json | 5 | No | Detailed scores per product |
| embedding_log_v2.json | 6 | No | Model + dims + path |
| retrieval_test_results_v2.json | 7 | No | Full test results |
| retrieve_agronomy_knowledge.py | 7 | YES — RAG TOOL | The function the agent calls |
| agromind_training_v2.jsonl | FT | YES — few-shot | 234 examples for system prompt |
| agromind_training_gemini_v2.jsonl | FT | No | Gemini format — already submitted |
| gemini_tuning_job_v2.json | FT | YES — model ID | Contains tuned model name |
| fine_tuning_summary.json | FT | No | Documents fine-tuning decisions |
| audit_report.json | 1 | No | Raw data audit findings |

### Infrastructure
| Resource | Location | Shared via |
|----------|----------|------------|
| ChromaDB vector store | /MyDrive/agromind/chromadb/ | Google Drive shared folder |
| Raw XLSX files | Google Drive shared folder | Google Drive shared folder |
| GCS training data | gs://agromind-ohoud-2026/agromind/training/ | Vertex AI only |

---

## 5. Key Decisions & Why

**Why Qwen2.5-7B-Instruct for the agent**
Runs locally via Ollama — no API cost per inference call, works offline,
fast iteration during development. The RAG tool still calls OpenAI embeddings
for query embedding (small cost per query).

**Why we preserved blank cells instead of filling them**
6 products have no active ingredient data. The client's source had genuinely
blank cells with no chemical information in any column. Fabricating ingredients
for a pesticide product is a safety risk. Honest nulls are better than wrong
data.

**Why two retrieval collections**
collection_structured for precise factual queries, collection_full for
semantic and Chinese queries. The agent routes between them based on query
type learned from Phase 7 testing.

**Why no chunking**
All 228 documents fit within the embedding model's 8,191 token limit.
Chunking would break product context.

**Why we removed nematodes from target_diseases**
Nematodes are pests (animals), not diseases (pathogens). Fixed in Phase 2
extraction prompt and verified in Phase 3.

**Why the fine-tuning data is still valuable even though we use Qwen**
The 234 cleaned examples teach correct tone, escalation patterns, and safety
behavior. They are used as few-shot examples in the Qwen system prompt.
The same data can fine-tune Qwen locally when time permits.

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
| Embedding dimensions | 3072 | text-embedding-3-large |

---

## 7. What's Left — Phase 8

**notebook_08_agent.ipynb** — builds the agent:

```
retrieve_agronomy_knowledge.py    (Phase 7 RAG tool)
        +
ChromaDB on Google Drive          (Phase 6 vector store)
        +
System prompt with few-shot       (from agromind_training_v2.jsonl)
        +
Qwen2.5-7B-Instruct via Ollama    (local agent model)
        ↓
Agro-Mind autonomous support agent
```

**Setup for Qwen local model:**
```bash
# 1. Download Ollama from ollama.com
# 2. Pull model
ollama pull qwen2.5:7b-instruct
# 3. Start server
ollama serve
# Server runs at http://localhost:11434
```

**Agent capabilities:**
- Intent classification (9 types)
- RAG retrieval via retrieve_agronomy_knowledge()
- Safety escalation with create_human_alert()
- Human handoff for poisoning, self-harm, complaints
- Interactive bilingual chat interface
- Conversation history tracking

---

## 8. How to Resume / Hand Off

### Running the agent

**In Google Colab:**
```
1. Clone the GitHub repo
2. Get Google Drive shared folder access (ask team lead)
3. Add 4 Colab secrets: OPENAI_API_KEY, GOOGLE_API_KEY,
   GCP_PROJECT, GCS_BUCKET
4. Open notebook_08_agent.ipynb
5. Mount Google Drive when prompted
6. Upload agent/retrieve_agronomy_knowledge.py when prompted
7. Run all cells
```

**In VS Code (local):**
```
1. Install Ollama + pull qwen2.5:7b-instruct
2. ollama serve  (keep running in background)
3. Download ChromaDB zip from shared Drive, extract locally
4. Update CHROMA_PATH in notebook to local path
5. Open notebook_08_agent.ipynb in VS Code
6. Run all cells
```

### Sharing ChromaDB with teammates
ChromaDB does NOT go in the GitHub repo (binary + client data privacy).
Share via Google Drive:
- Right-click agromind/ folder in Drive → Share
- Add teammate Gmail addresses
- Permission: Editor
- Setting: Restricted (NOT "anyone with link")
- Communicate the share via team channel (WhatsApp/Slack), not public link

### Adding new products
```
1. Add to 产品目录_CLEANED.xlsx
2. Run notebook_01 → 02 → 03 → 04 for new products only
3. Run notebook_05 judge on new documents
4. In notebook_06: add to existing ChromaDB (do NOT delete existing)
5. Re-run notebook_07 to verify retrieval still works
```

### Required secrets
| Secret | Value |
|--------|-------|
| OPENAI_API_KEY | your OpenAI key |
| GOOGLE_API_KEY | your Google AI key |
| GCP_PROJECT | gen-lang-client-0396125347 |
| GCS_BUCKET | agromind-ohoud-2026 |

### Critical: embedding model must match
When querying ChromaDB, always use `text-embedding-3-large`.
Using a different model produces wrong similarity scores.
