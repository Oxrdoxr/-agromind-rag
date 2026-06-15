"""
config.py — AgroMind Configuration Loader
Reads config.yaml and exposes typed settings to all modules.
Import: from src.config import config
"""

from pathlib import Path
from functools import lru_cache
from typing import Optional
import yaml
from pydantic import BaseModel
# In load_config(), add environment variable loading
import os
from dotenv import load_dotenv

load_dotenv()  # Load .env file

# ── Sub-models ────────────────────────────────────────────────────────────────

class LLMConfig(BaseModel):
    provider: str
    model: str
    temperature: float
    max_tokens: int


class EmbeddingConfig(BaseModel):
    provider: str
    model: str
    dimensions: int
    query_prefix: str
    doc_prefix: str


class RetrievalConfig(BaseModel):
    structured_collection: str
    full_collection: str
    image_collection: str
    top_k: int
    confidence_threshold: float
    bm25_weight: float
    vector_weight: float


class PathsConfig(BaseModel):
    chromadb: str
    data: str
    logs: str
    escalations_log: str


class AgentConfig(BaseModel):
    max_retries: int
    faithfulness_threshold: float
    conversation_window: int


class LangSmithConfig(BaseModel):
    project: str
    endpoint: str = "https://api.smith.langchain.com"
    tracing_enabled: bool = True
    api_key: Optional[str] = None  # Load from env



# After creating config, override with env
if os.getenv("LANGSMITH_API_KEY"):
    config.langsmith.api_key = os.getenv("LANGSMITH_API_KEY")

class AgroMindConfig(BaseModel):
    llm: LLMConfig
    embedding: EmbeddingConfig
    retrieval: RetrievalConfig
    paths: PathsConfig
    agent: AgentConfig
    langsmith: LangSmithConfig

    @property
    def chromadb_path(self) -> str:
        """Resolve chromadb path relative to project root."""
        root = Path(__file__).parent.parent
        return str(root / self.paths.chromadb)

    @property
    def data_path(self) -> str:
        root = Path(__file__).parent.parent
        return str(root / self.paths.data)

    @property
    def logs_path(self) -> str:
        root = Path(__file__).parent.parent
        p = root / self.paths.logs
        p.mkdir(parents=True, exist_ok=True)
        return str(p)

    @property
    def escalations_log_path(self) -> str:
        root = Path(__file__).parent.parent
        return str(root / self.paths.escalations_log)


# ── Loader ────────────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def load_config() -> AgroMindConfig:
    """
    Load config.yaml from project root.
    Cached — only reads from disk once per process.
    """
    config_path = Path(__file__).parent.parent / "config.yaml"
    assert config_path.exists(), f"config.yaml not found at {config_path}"

    with open(config_path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    return AgroMindConfig(**raw)


# ── Singleton ─────────────────────────────────────────────────────────────────
# Import this everywhere: from src.config import config
config = load_config()


# ── Quick verification ────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=== AgroMind Config ===")
    print(f"LLM:       {config.llm.model} (temp={config.llm.temperature})")
    print(f"Embedding: {config.embedding.model} ({config.embedding.dimensions}d)")
    print(f"ChromaDB:  {config.chromadb_path}")
    print(f"Data:      {config.data_path}")
    print(f"Top-K:     {config.retrieval.top_k}")
    print(f"Threshold: {config.retrieval.confidence_threshold}")
    print(f"Project:   {config.langsmith.project}")
    print("Config loaded successfully.")