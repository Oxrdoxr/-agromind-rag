Here's the complete `GUIDE_FOR_TEAM.md` file for your team members:

```markdown
# Agro-Mind RAG System - Team Guide

**Version:** 3.0 (LangChain + Local Ollama)  
**Last Updated:** June 2026  
**Maintainer:** Agro-Mind Development Team

---

## 📋 Table of Contents

1. [What This Guide Covers](#what-this-guide-covers)
2. [System Overview](#system-overview)
3. [Initial Setup (One-Time)](#initial-setup-one-time)
4. [Project Structure](#project-structure)
5. [How to Use the Retriever](#how-to-use-the-retriever)
6. [Building the Agent](#building-the-agent)
7. [Adding New Tools](#adding-new-tools)
8. [Common Tasks](#common-tasks)
9. [Troubleshooting](#troubleshooting)
10. [What to Share vs Not Share](#what-to-share-vs-not-share)
11. [Quick Reference](#quick-reference)

---

## What This Guide Covers

This guide helps team members:
- Set up the Agro-Mind RAG system on their machine
- Use the retriever in their own code
- Build LangChain agents with the RAG tool
- Add new tools (weather, pest alerts, calculators)
- Troubleshoot common issues

**Prerequisites:** Basic Python, virtual environments, and LangChain familiarity.

---

## System Overview

### Architecture
```
User Query
    ↓
LangChain Agent
    ↓
RAG Tool (create_rag_tool)
    ↓
AgroMindRetriever
    ├── Exact ID lookup (product_id, disease name)
    └── Vector search (BGE-M3 embeddings)
    ↓
ChromaDB (local, 114 products)
    ↓
Qwen2.5-7B-Instruct (local via Ollama)
    ↓
Response with citations
```

### Key Components

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Retriever** | Custom `AgroMindRetriever` | Hybrid search (exact + vector) |
| **Embeddings** | BGE-M3 via Ollama (1024 dims) | Convert text to vectors |
| **Vector DB** | ChromaDB (LangChain wrapper) | Store product embeddings |
| **LLM** | Qwen2.5-7B-Instruct | Generate answers |
| **Agent** | LangChain `initialize_agent` | Orchestrate tools |
| **Data** | `clean_entities.json` | Product ground truth |

---

## Initial Setup (One-Time)

### Step 1: Clone Repository
```bash
git clone https://github.com/your-org/agromind-v3.git
cd agromind-v3
```

### Step 2: Create Virtual Environment
```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# Mac/Linux
python -m venv .venv
source .venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install chromadb ollama langchain langchain-chroma langchain-ollama pydantic pyyaml
```

### Step 4: Install Ollama
```bash
# Download from https://ollama.com
# Then pull required models:
ollama pull bge-m3              # Embedding model (1.1GB)
ollama pull qwen2.5:7b-instruct # Agent model (4.1GB)

# Start Ollama server (keep this terminal open)
ollama serve
```

### Step 5: Get Data Files (from Team Shared Drive)
Download these files and place in correct locations:

```
📁 agromind-v3/
├── 📁 data/
│   └── clean_entities.json          ← FROM SHARED DRIVE
├── 📁 chromadb/                      ← FROM SHARED DRIVE (if pre-built)
│   ├── chroma.sqlite3
│   └── ...
└── 📁 src/
    └── (code files already in repo)
```

**Shared Drive Access:** Ask team lead for Google Drive link (restricted access)

### Step 6: Create `__init__.py` Files
```bash
# Windows
echo. > src\__init__.py
echo. > tests\__init__.py

# Mac/Linux
touch src/__init__.py
touch tests/__init__.py
```

### Step 7: Test Installation
```bash
# Test embeddings wrapper
python -m src.embeddings

# Test retriever
python -m src.retrieval_tool

# Expected output:
# ✅ Document embedding: 1024 dimensions
# ✅ Query embedding: 1024 dimensions
# ✅ Batch embeddings: 3 texts → 1024 dims each
# ✅ Wrapper works! Ready to use with ChromaDB
```

---

## Project Structure

```
agromind-v3/
├── src/                          # Main source code
│   ├── __init__.py               # Makes src a Python package
│   ├── config.py                 # Configuration loader (reads config.yaml)
│   ├── embeddings.py             # Ollama → LangChain wrapper (WORKING ✓)
│   └── retrieval_tool.py         # MAIN RETRIEVER (USE THIS)
│
├── scripts/                      # Utility scripts
│   ├── rebuild_db.py             # Build ChromaDB from clean_entities.json
│   └── eval_retrieval.py         # Test retrieval quality
│
├── tests/                        # Unit tests
│   └── test_knowledge_base.py    # Test retriever functionality
│
├── data/                         # Data files (NOT in git)
│   └── clean_entities.json       # Product ground truth (FROM SHARED DRIVE)
│
├── chromadb/                     # Vector database (NOT in git)
│   ├── chroma.sqlite3
│   └── ...
│
├── config.yaml                   # Configuration (model names, paths)
├── test_embedding.py             # Quick Ollama connectivity test
├── requirements.txt              # Python dependencies
├── README.md                     # Project overview
└── GUIDE_FOR_TEAM.md             # This file
```

### Files You Should Never Modify
- `src/embeddings.py` - Working wrapper, changes break everything
- `data/clean_entities.json` - Source of truth (read-only)

### Files You Can Modify
- `src/retrieval_tool.py` - Add new search methods
- `config.yaml` - Change models, paths, thresholds
- `scripts/rebuild_db.py` - Adjust chunking/processing

---

## How to Use the Retriever

### Basic Usage

```python
from src.retrieval_tool import AgroMindRetriever

# Initialize (do this once at app start)
retriever = AgroMindRetriever(
    db_path="data/v4/chroma_db",           # Path to ChromaDB
    collection_name="agromind_v4",          # Collection name
    entities_path="data/clean_entities.json" # Product data
)

# Search by text query
results = retriever.search("root rot treatment for citrus", k=3)

# Each result is a tuple: (Document, score)
for doc, score in results:
    print(f"Product: {doc.metadata['name_cn']}")
    print(f"ID: {doc.metadata['product_id']}")
    print(f"Score: {score:.3f}")
    print(f"Details: {doc.page_content[:200]}...")
    print("-" * 40)
```

### Search Methods

| Method | Use Case | Example |
|--------|----------|---------|
| `search(query, k)` | General text search | `retriever.search("citrus yellow leaves")` |
| `search_by_disease(disease, k)` | Find products for specific disease | `retriever.search_by_disease("炭疽病")` |
| `search_by_ingredient(ingredient, k)` | Products with active ingredient | `retriever.search_by_ingredient("copper")` |
| `search_by_crop(crop, k)` | Products for specific crop | `retriever.search_by_crop("citrus")` |
| `get_product_info(product_id)` | Exact product lookup | `retriever.get_product_info("AF0001")` |

### Complete Example

```python
from src.retrieval_tool import AgroMindRetriever

retriever = AgroMindRetriever()

# Example 1: Disease query (Chinese)
print("=== 柑橘黄龙病 treatment ===")
results = retriever.search_by_disease("柑橘黄龙病", k=3)
for doc, score in results:
    print(f"✓ {doc.metadata['name_cn']} (score: {score:.2f})")

# Example 2: Symptom description
print("\n=== 'leaves turning yellow with spots' ===")
results = retriever.search("leaves turning yellow with spots", k=3)
for doc, score in results:
    print(f"✓ {doc.metadata['name_cn']}")

# Example 3: Exact product lookup
print("\n=== Product AF0012 ===")
product = retriever.get_product_info("AF0012")
if product:
    print(f"Name: {product['name_cn']}")
    print(f"Diseases: {', '.join(product.get('target_diseases', [])[:3])}")

# Example 4: Format results for LLM
def format_for_llm(results):
    context = "Relevant products:\n\n"
    for i, (doc, score) in enumerate(results, 1):
        context += f"{i}. {doc.metadata['name_cn']} (ID: {doc.metadata['product_id']})\n"
        context += f"   {doc.page_content[:150]}...\n\n"
    return context

print("\n=== Formatted for LLM ===")
print(format_for_llm(results[:2]))
```

### Performance Tips

```python
# Cache results for repeated queries
retriever = AgroMindRetriever()
results1 = retriever.search("citrus canker")  # Slow (first time)
results2 = retriever.search("citrus canker")  # Fast (cached)

# Clear cache when needed
retriever.clear_cache()

# Adjust k based on need
top_1 = retriever.search("query", k=1)      # Fastest
top_10 = retriever.search("query", k=10)    # Slower but more results
```

---

## Building the Agent

### Minimal Agent (Single Tool)

```python
from langchain.agents import initialize_agent
from langchain_ollama import ChatOllama
from src.retrieval_tool import AgroMindRetriever, create_rag_tool

# 1. Initialize components
retriever = AgroMindRetriever()
rag_tool = create_rag_tool(retriever)
llm = ChatOllama(
    model="qwen2.5:7b-instruct",
    temperature=0.1,      # Low for factual answers
    num_predict=1024      # Max response length
)

# 2. Create agent
agent = initialize_agent(
    tools=[rag_tool],
    llm=llm,
    agent="zero-shot-react-description",
    verbose=True,         # See agent's reasoning
    max_iterations=3,
    early_stopping_method="generate"
)

# 3. Run queries
response = agent.run("What product treats citrus root rot?")
print(response)

# Chinese query
response = agent.run("柑橘叶片发黄怎么处理？")
print(response)
```

### Agent with Memory

```python
from langchain.memory import ConversationBufferMemory
from langchain.agents import initialize_agent

# Add memory to remember conversation
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

agent = initialize_agent(
    tools=[rag_tool],
    llm=llm,
    agent="conversational-react-description",
    memory=memory,
    verbose=True
)

# Now agent remembers context
agent.run("What treats citrus diseases?")
agent.run("How much should I apply?")  # Knows you're still talking about citrus
```

### Agent with Multiple Tools

```python
from langchain.tools import Tool

# Tool 1: RAG (from above)
rag_tool = create_rag_tool(retriever)

# Tool 2: Weather (example)
def get_weather(location: str) -> str:
    """Get current weather for disease risk assessment"""
    # In production, call weather API
    return f"Weather in {location}: 25°C, 65% humidity, low disease risk"

weather_tool = Tool(
    name="Weather",
    func=get_weather,
    description="Get weather conditions for a location. Input: city name"
)

# Tool 3: Calculator (built-in)
from langchain.tools import tool

@tool
def calculate_dosage(area_hectares: str, rate_per_hectare: str) -> str:
    """Calculate total product needed. Input: 'area, rate' e.g., '5, 2.5'"""
    try:
        area, rate = map(float, area_hectares.split(','))
        total = area * rate
        return f"Total needed: {total} L"
    except:
        return "Invalid input. Use: 'area_ha, rate_per_ha'"

# Create agent with all tools
agent = initialize_agent(
    tools=[rag_tool, weather_tool, calculate_dosage],
    llm=llm,
    agent="zero-shot-react-description",
    verbose=True
)

# Agent will automatically choose the right tool
response = agent.run("What product for citrus canker? Also, what's the weather in Guangzhou?")
```

---

## Adding New Tools

### Template for New Tools

```python
from langchain.tools import Tool

def my_custom_function(input_string: str) -> str:
    """
    Brief description of what this tool does.
    
    Args:
        input_string: Description of expected input format
    
    Returns:
        Formatted output string
    """
    # Your logic here
    result = do_something(input_string)
    
    # Return a clean string
    return f"Result: {result}"

my_tool = Tool(
    name="MyToolName",                    # Used by agent to select tool
    func=my_custom_function,              # Function to call
    description="Detailed description of when to use this tool. Include input format."
)

# Add to agent
agent = initialize_agent(
    tools=[rag_tool, my_tool],            # Include new tool
    llm=llm,
    agent="zero-shot-react-description"
)
```

### Example: Pest Alert Tool

```python
import requests
from datetime import datetime

def get_pest_alerts(region: str) -> str:
    """
    Get current pest outbreak alerts for a region.
    
    Args:
        region: City or province name (e.g., "Guangdong")
    
    Returns:
        Pest alerts or "No active alerts"
    """
    # In production, call real API
    # response = requests.get(f"https://api.pestwatch.org/alerts?region={region}")
    
    # Mock data for demonstration
    alerts_db = {
        "guangdong": "⚠️ Rice blast outbreak reported in Zhaoqing",
        "hunan": "⚠️ Citrus red mite population increasing",
        "default": "No active pest alerts in this region"
    }
    
    alert = alerts_db.get(region.lower(), alerts_db["default"])
    return f"Pest alerts for {region} ({datetime.now().strftime('%Y-%m-%d')}):\n{alert}"

pest_alert_tool = Tool(
    name="PestAlerts",
    func=get_pest_alerts,
    description="Get current pest outbreak alerts. Input: region name (e.g., 'Guangdong')"
)

# Use in agent
agent = initialize_agent(
    tools=[rag_tool, pest_alert_tool],
    llm=llm,
    agent="zero-shot-react-description"
)

response = agent.run("Are there any pest outbreaks in Guangdong right now?")
```

### Example: Dosage Calculator Tool

```python
from langchain.tools import tool

@tool
def calculate_mix_ratio(water_liters: str, product_rate: str) -> str:
    """
    Calculate how much product to add to water.
    
    Input format: 'water_liters, product_rate_per_100L'
    Example: '200, 50' (200L water, 50ml product per 100L)
    
    Returns:
        Total product needed in ml
    """
    try:
        water, rate = map(float, water_liters.split(','))
        total_ml = (water / 100) * rate
        return f"For {water}L water, add {total_ml:.1f}ml of product"
    except:
        return "Invalid input. Use: 'water_liters, rate_per_100L'"

# Add to agent
agent = initialize_agent(
    tools=[rag_tool, calculate_mix_ratio],
    llm=llm,
    agent="zero-shot-react-description"
)

response = agent.run("I have 300L of water and the product says 40ml per 100L. How much do I add?")
```

---

## Common Tasks

### Task 1: Rebuild ChromaDB from Scratch

Run when `clean_entities.json` is updated:

```bash
python scripts/rebuild_db.py
```

If `scripts/rebuild_db.py` doesn't exist, create it:

```python
# scripts/rebuild_db.py
import json
from pathlib import Path
from src.retrieval_tool import AgroMindRetriever
from langchain_chroma import Chroma
from src.embeddings import OllamaEmbeddingsWrapper

def rebuild_database():
    """Rebuild ChromaDB from clean_entities.json"""
    
    # Load entities
    with open("data/clean_entities.json", "r", encoding="utf-8") as f:
        entities = json.load(f)
    
    # Create documents
    documents = []
    for entity in entities:
        text = f"""Product ID: {entity['product_id']}
Name: {entity.get('name_cn', '')} / {entity.get('name_en', '')}
Diseases: {', '.join(entity.get('target_diseases', []))}
Crops: {', '.join(entity.get('target_crops', []))}
Ingredients: {', '.join(entity.get('active_ingredients', []))}
"""
        documents.append(text)
    
    # Create vector store
    embeddings = OllamaEmbeddingsWrapper()
    vectorstore = Chroma.from_texts(
        texts=documents,
        embedding=embeddings,
        persist_directory="data/v4/chroma_db",
        collection_name="agromind_v4"
    )
    
    print(f"✅ Rebuilt database with {len(documents)} products")

if __name__ == "__main__":
    rebuild_database()
```

### Task 2: Test Retrieval Quality

```bash
python scripts/eval_retrieval.py
```

### Task 3: Add New Product to Database

```python
# 1. Add to clean_entities.json manually or via script
# 2. Rebuild database
python scripts/rebuild_db.py

# 3. Test retrieval for new product
from src.retrieval_tool import AgroMindRetriever
retriever = AgroMindRetriever()
result = retriever.get_product_info("NEW001")
print(result)
```

### Task 4: Export Agent Responses for Analysis

```python
import json
from datetime import datetime

def log_conversation(user_query, agent_response, retrieved_docs):
    """Save conversations for quality analysis"""
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "query": user_query,
        "response": agent_response,
        "retrieved_products": [doc.metadata['product_id'] for doc, _ in retrieved_docs],
        "scores": [score for _, score in retrieved_docs]
    }
    
    with open("logs/conversations.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
```

---

## Troubleshooting

### Issue 1: "ModuleNotFoundError: No module named 'src'"

**Fix:** Run from project root or set PYTHONPATH
```bash
# Option A: Run from project root
cd C:\path\to\agromind-v3
python -m src.retrieval_tool

# Option B: Set PYTHONPATH (Windows)
set PYTHONPATH=%CD%
python src/retrieval_tool.py

# Option C: Set PYTHONPATH (Mac/Linux)
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
python src/retrieval_tool.py
```

### Issue 2: "Ollama not running" or connection refused

**Fix:** Start Ollama server
```bash
# In a separate terminal, keep this running
ollama serve

# Verify it's working
curl http://localhost:11434/api/tags
```

### Issue 3: "Collection not found" or empty results

**Fix:** Database not built or wrong path
```bash
# Check if chromadb exists
ls chromadb/

# If empty, rebuild
python scripts/rebuild_db.py

# Or check config.yaml path
cat config.yaml | grep chromadb
```

### Issue 4: Dimension mismatch error

**Fix:** Update config.yaml
```yaml
# config.yaml
embedding:
  dimensions: 1024  # Must match BGE-M3 output
```

Then rebuild database:
```bash
python scripts/rebuild_db.py
```

### Issue 5: LangChain agent not using the tool

**Fix:** Make tool description clearer
```python
# Bad description (agent won't know when to use)
Tool(description="Search products")

# Good description (agent understands)
Tool(description="Search agricultural product database for disease treatments, pest control, or product information. Use this when farmers ask about specific diseases (e.g., 'root rot'), products (e.g., 'AF0001'), or symptoms (e.g., 'yellow leaves').")
```

### Issue 6: Slow response times

**Solutions:**
```python
# 1. Reduce number of retrieved documents
results = retriever.search(query, k=3)  # instead of k=10

# 2. Enable cache (default on)
retriever = AgroMindRetriever()  # Cache is enabled by default

# 3. Use exact lookup when possible
if query.upper().startswith("AF"):
    product = retriever.get_product_info(query)  # Fast!
else:
    results = retriever.search(query)  # Slower
```

### Issue 7: Chinese queries not working well

**Fix:** BGE-M3 natively supports Chinese, but ensure:
```python
# Use search_by_disease for known disease names
results = retriever.search_by_disease("炭疽病")  # Better than generic search

# Or use full-text search in Chinese
results = retriever.search("柑橘叶片发黄")
```

---

## What to Share vs Not Share

### ✅ Commit to GitHub (Public/Team)

```
src/*.py              # All source code
scripts/*.py          # Utility scripts
tests/*.py            # Tests
config.yaml           # Configuration (no secrets)
requirements.txt      # Dependencies
README.md             # Project overview
GUIDE_FOR_TEAM.md     # This file
.gitignore            # Git ignore rules
```

### ❌ Never Commit (Add to .gitignore)

```
chromadb/             # Vector database (large + proprietary)
data/                 # Product data (get from shared drive)
.venv/                # Virtual environment
__pycache__/          # Python cache
*.log                 # Log files
.env                  # Environment variables (secrets)
*.key                 # API keys
*.pem                 # Certificates
```

### Shared Drive (For Team Only)

Share these via **restricted Google Drive** (not public links):

```
📁 agromind-data/
├── clean_entities.json       # Product ground truth
├── chromadb.zip              # Pre-built vector database
├── product_catalog.xlsx      # Raw source data
└── training_data.jsonl       # Fine-tuning examples
```

**Sharing settings:**
- Right-click folder → Share
- Add teammate email addresses
- Permission: Editor
- General access: Restricted (NOT "Anyone with link")

---

## Quick Reference

### One-Line Commands

```bash
# Setup
ollama serve                                    # Start Ollama
ollama pull bge-m3                              # Download embedding model
python -m src.embeddings                        # Test embeddings

# Use retriever
python -m src.retrieval_tool                    # Test retriever

# Rebuild database
python scripts/rebuild_db.py                    # Rebuild from entities

# Run agent
python agent_main.py                            # Start interactive agent
```

### Common Code Snippets

**Basic retriever:**
```python
from src.retrieval_tool import AgroMindRetriever
retriever = AgroMindRetriever()
results = retriever.search("query", k=3)
```

**Basic agent:**
```python
from langchain.agents import initialize_agent
from langchain_ollama import ChatOllama
from src.retrieval_tool import create_rag_tool, AgroMindRetriever

retriever = AgroMindRetriever()
agent = initialize_agent(
    tools=[create_rag_tool(retriever)],
    llm=ChatOllama(model="qwen2.5:7b-instruct", temperature=0.1),
    agent="zero-shot-react-description"
)
response = agent.run("your query")
```

**Search by type:**
```python
retriever.search("text")                # General
retriever.search_by_disease("name")     # Disease-specific
retriever.search_by_ingredient("name")  # Ingredient search
retriever.search_by_crop("name")        # Crop search
retriever.get_product_info("AF0001")    # Exact ID
```

### File Paths (Default)

| Resource | Path |
|----------|------|
| ChromaDB | `./data/v4/chroma_db` |
| Entities | `./data/clean_entities.json` |
| Config | `./config.yaml` |
| Logs | `./logs/` |

### Configuration (config.yaml)

```yaml
llm:
  model: "qwen2.5:7b-instruct"
  temperature: 0.1

embedding:
  model: "bge-m3"
  dimensions: 1024

retrieval:
  top_k: 5
  confidence_threshold: 0.85

paths:
  chromadb: "./data/v4/chroma_db"
  data: "./data"
  logs: "./logs"
```

---

## Getting Help

### Check These First
1. Is Ollama running? `curl http://localhost:11434/api/tags`
2. Are you in the project root? `pwd` should show `.../agromind-v3`
3. Does `clean_entities.json` exist? `ls data/`
4. Did you create `__init__.py`? `ls src/__init__.py`

### Ask Team Lead For
- Shared Drive access (if you don't have it)
- ChromaDB zip file (if rebuilding takes too long)
- API keys (if adding external services)

### Error Messages & Solutions

| Error | Likely Fix |
|-------|-----------|
| `No module named 'src'` | Run from project root or set PYTHONPATH |
| `Connection refused` | Start `ollama serve` |
| `Collection not found` | Run `python scripts/rebuild_db.py` |
| `Dimension mismatch` | Update `config.yaml` dimensions to 1024 |
| `File not found: clean_entities.json` | Download from shared drive |

---

## Final Notes

- **Always run from project root** (`cd agromind-v3` before any python command)
- **Keep Ollama running** in a separate terminal
- **Never commit data files** to GitHub (use shared drive)
- **Test after any changes** with `python -m src.retrieval_tool`
- **Cache is your friend** - reuse retriever instance when possible

---

**Questions?** Ask the team lead or open an issue in the GitHub repo.

**Last Updated:** June 2026  
**Version:** 3.0 (LangChain + Local Ollama)

Happy coding! 🚀
```

---

This guide is comprehensive and should help any team member get up to speed quickly. Save it as `GUIDE_FOR_TEAM.md` in your project root.