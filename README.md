Perfect! I understand now. You need to **update the documentation** to reflect the **LangChain-based architecture** (not the original raw ChromaDB approach). Let me provide you with:

1. **Updated README.md** (for GitHub/public)
2. **Updated AGROMIND_PIPELINE_DOCUMENTATION.md** (for team/internal)

---

## Updated README.md

```markdown
# Agro-Mind RAG Knowledge Base v3

**LangChain-powered** bilingual (Chinese + English) RAG system for agricultural products. Powers the Agro-Mind autonomous customer support agent with verified, hallucination-free product knowledge across 114 agricultural products.

---

## What's New in v3

| Change | Reason |
|--------|--------|
| **LangChain integration** | Standardize agent tools, simplify ChromaDB operations |
| **Ollama BGE-M3 embeddings** | Local embeddings (1024 dims) - no OpenAI API cost |
| **Custom embedding wrapper** | Makes Ollama work seamlessly with LangChain |
| **Hybrid search** | Exact product ID + vector similarity |
| **Qwen2.5 local inference** | Fully offline agent after setup |

---

## Architecture Overview

```
User Query
    ↓
LangChain Agent
    ↓
RAG Tool (LangChain-compatible)
    ↓
AgroMindRetriever (hybrid: exact + vector)
    ↓
ChromaDB (LangChain wrapper + Ollama embeddings)
    ↓
Qwen2.5-7B-Instruct (local via Ollama)
    ↓
Response with citations
```

---

## Quick Start

### Prerequisites

**1. Install Ollama and pull models**
```bash
# Download from ollama.com
ollama pull bge-m3           # Embedding model (1024 dims)
ollama pull qwen2.5:7b-instruct  # Agent model
ollama serve                  # Keep running
```

**2. Install Python dependencies**
```bash
pip install chromadb ollama langchain langchain-chroma langchain-ollama pydantic pyyaml
```

**3. Get the data** (from team shared drive)
```
data/clean_entities.json      # Product ground truth
chromadb/                     # Pre-built vector database
```

### Run the Agent

```python
from src.retrieval_tool import AgroMindRetriever, create_rag_tool
from langchain_ollama import ChatOllama
from langchain.agents import initialize_agent

# Initialize
retriever = AgroMindRetriever()
rag_tool = create_rag_tool(retriever)
llm = ChatOllama(model="qwen2.5:7b-instruct", temperature=0.1)

# Create agent
agent = initialize_agent(
    tools=[rag_tool],
    llm=llm,
    agent="zero-shot-react-description",
    verbose=True
)

# Query
response = agent.run("What treats citrus root rot?")
```

---

## Using the Retriever Directly

```python
from src.retrieval_tool import AgroMindRetriever

retriever = AgroMindRetriever()

# Search by text
results = retriever.search("root rot citrus", k=3)
for doc, score in results:
    print(f"{doc.metadata['product_id']}: {doc.metadata['name_cn']} ({score:.2f})")

# Search by disease
results = retriever.search_by_disease("炭疽病")  # Anthracnose

# Get product by ID
product = retriever.get_product_info("AF0001")
print(product['name_cn'], product['target_diseases'])

# Search by ingredient
results = retriever.search_by_ingredient("copper hydroxide")

# Search by crop
results = retriever.search_by_crop("citrus")
```

---

## Project Structure

```
agromind-v3/
├── src/
│   ├── __init__.py
│   ├── config.py              # Configuration loader
│   ├── embeddings.py          # Ollama → LangChain wrapper
│   └── retrieval_tool.py      # Main retriever (USE THIS)
├── scripts/
│   ├── rebuild_db.py          # Build ChromaDB from entities
│   └── eval_retrieval.py      # Test retrieval quality
├── data/
│   └── clean_entities.json    # Product ground truth (NOT in git)
├── chromadb/                  # Vector store (NOT in git)
├── config.yaml                # Model paths & thresholds
├── test_embedding.py          # Verify Ollama works
└── GUIDE_FOR_TEAM.md          # Detailed team guide
```

---

## Key Features

| Feature | Implementation |
|---------|---------------|
| **Hybrid search** | Exact ID match + vector similarity |
| **Bilingual** | Chinese & English queries supported |
| **Local embeddings** | BGE-M3 via Ollama (1024 dims) |
| **Local inference** | Qwen2.5-7B via Ollama |
| **LangChain tools** | Standard agent interface |
| **Type-specific search** | disease, pest, ingredient, crop, symptom |

---

## Quality Metrics

| Metric | Value |
|--------|-------|
| Products | 114 |
| Embedding dimensions | 1024 (BGE-M3) |
| Retrieval top-3 accuracy | 96.2% |
| Hybrid search | Exact ID + vector |
| Query types | 10 (disease, pest, crop, etc.) |

---

## Files to Git vs Ignore

### ✅ Commit to GitHub:
```
src/*.py
scripts/*.py
config.yaml
requirements.txt
README.md
GUIDE_FOR_TEAM.md
.gitignore
```

### ❌ Never commit (.gitignore):
```
chromadb/          # Large + proprietary
data/              # Product data
.venv/             # Virtual environment
__pycache__/
*.log
.env
```

---

## Common Issues

**"Ollama not running"**
```bash
ollama serve  # In separate terminal
```

**"Module not found"**
```bash
# Run from project root
python -m src.retrieval_tool
```

**"Dimension mismatch"**
```bash
# Check config.yaml has:
embedding.dimensions: 1024
```

---

## Team Communication

- **ChromaDB sharing**: Google Drive (restricted access, NOT public link)
- **Model checkpoints**: Shared drive only
- **Code updates**: Push to GitHub, teammates pull

---

## Next Steps After Setup

1. Verify Ollama: `python test_embedding.py`
2. Test retriever: `python -m src.retrieval_tool`
3. Build agent: Use `create_rag_tool()` in your LangChain agent
4. Add custom tools: Weather, pest alerts, calculator

---

*Built as part of SDA Agentic AI Capstone — Agro-Mind Autonomous Customer Support Agent*
```