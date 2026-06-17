"""
Embedding wrapper that makes Ollama look like OpenAI to ChromaDB.
Supports:
- ChromaDB EmbeddingFunction
- LangChain Embeddings
- Optional vector normalization
- Persistent HTTP session (faster)
"""

import sys
from pathlib import Path
from typing import List

import requests
from chromadb.api.types import EmbeddingFunction
from langchain_core.embeddings import Embeddings

# Add parent directory to path if running directly
if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import config


class OllamaEmbeddingFunction(EmbeddingFunction):
    """
    ChromaDB-compatible embedding function using local Ollama.
    """

    def __init__(self):

        self.model = config.embedding.model
        self.dimensions = config.embedding.dimensions
        self.query_prefix = config.embedding.query_prefix
        self.doc_prefix = config.embedding.doc_prefix

        self.ollama_url = (
            "http://localhost:11434/api/embeddings"
        )

        # Reuse connection
        self.session = requests.Session()

    # --------------------------------------------------
    # Helpers
    # --------------------------------------------------

    def _normalize(
        self,
        embedding: List[float]
    ) -> List[float]:

        if not config.embedding.normalize:
            return embedding

        norm = (
            sum(x * x for x in embedding)
            ** 0.5
        )

        if norm == 0:
            return embedding

        return [
            x / norm
            for x in embedding
        ]

    def _embed(
        self,
        text: str
    ) -> List[float]:

        response = self.session.post(
            self.ollama_url,
            json={
                "model": self.model,
                "prompt": text
            },
            timeout=60
        )

        response.raise_for_status()

        embedding = (
            response.json()["embedding"]
        )

        if len(embedding) != self.dimensions:

            raise ValueError(
                f"Expected {self.dimensions} dimensions, "
                f"got {len(embedding)} from {self.model}"
            )

        return self._normalize(
            embedding
        )

    # --------------------------------------------------
    # Chroma Interface
    # --------------------------------------------------

    def __call__(
        self,
        input: List[str]
    ) -> List[List[float]]:

        embeddings = []

        for text in input:

            prefixed_text = (
                self.doc_prefix +
                text
            )

            embeddings.append(
                self._embed(
                    prefixed_text
                )
            )

        return embeddings

    def embed_query(
        self,
        query: str
    ) -> List[float]:

        prefixed_query = (
            self.query_prefix +
            query
        )

        return self._embed(
            prefixed_query
        )


class OllamaEmbeddingsWrapper(Embeddings):
    """
    LangChain-compatible wrapper.
    """

    def __init__(self):

        self.model = config.embedding.model
        self.dimensions = config.embedding.dimensions

        self.query_prefix = (
            config.embedding.query_prefix
        )

        self.doc_prefix = (
            config.embedding.doc_prefix
        )

        self.ollama_url = (
            "http://localhost:11434/api/embeddings"
        )

        # Reuse connection
        self.session = requests.Session()

    # --------------------------------------------------
    # Helpers
    # --------------------------------------------------

    def _normalize(
        self,
        embedding: List[float]
    ) -> List[float]:

        if not config.embedding.normalize:
            return embedding

        norm = (
            sum(x * x for x in embedding)
            ** 0.5
        )

        if norm == 0:
            return embedding

        return [
            x / norm
            for x in embedding
        ]

    def _embed(
        self,
        text: str
    ) -> List[float]:

        response = self.session.post(
            self.ollama_url,
            json={
                "model": self.model,
                "prompt": text
            },
            timeout=60
        )

        response.raise_for_status()

        embedding = (
            response.json()["embedding"]
        )

        if len(embedding) != self.dimensions:

            raise ValueError(
                f"Expected {self.dimensions} dimensions, "
                f"got {len(embedding)} from {self.model}"
            )

        return self._normalize(
            embedding
        )

    # --------------------------------------------------
    # LangChain Interface
    # --------------------------------------------------

    def embed_documents(
        self,
        texts: List[str]
    ) -> List[List[float]]:

        embeddings = []

        for text in texts:

            prefixed_text = (
                self.doc_prefix +
                text
            )

            embeddings.append(
                self._embed(
                    prefixed_text
                )
            )

        return embeddings

    def embed_query(
        self,
        text: str
    ) -> List[float]:

        prefixed_text = (
            self.query_prefix +
            text
        )

        return self._embed(
            prefixed_text
        )


# --------------------------------------------------
# Singleton Instances
# --------------------------------------------------

ollama_embeddings = (
    OllamaEmbeddingFunction()
)

ollama_wrapper = (
    OllamaEmbeddingsWrapper()
)


# --------------------------------------------------
# Direct Helper
# --------------------------------------------------

def get_embedding(
    text: str,
    for_query: bool = False
) -> List[float]:

    if for_query:

        return (
            ollama_embeddings
            .embed_query(text)
        )

    return (
        ollama_embeddings([text])[0]
    )


# --------------------------------------------------
# Quick Test
# --------------------------------------------------

if __name__ == "__main__":

    print(
        "=" * 60
    )

    print(
        "TESTING OLLAMA EMBEDDINGS"
    )

    print(
        "=" * 60
    )

    doc_embedding = get_embedding(
        "Citrus greening disease causes yellow shoots"
    )

    print(
        f"Document embedding: "
        f"{len(doc_embedding)} dims"
    )

    query_embedding = get_embedding(
        "yellow citrus leaves treatment",
        for_query=True
    )

    print(
        f"Query embedding: "
        f"{len(query_embedding)} dims"
    )

    batch = [

        "Product AF0001 treats root rot",

        "Product AF0002 treats citrus canker",

        "Apply 2ml per liter of water"
    ]

    embeddings = (
        ollama_wrapper
        .embed_documents(batch)
    )

    print(
        f"Batch embeddings: "
        f"{len(embeddings)}"
    )

    print(
        "\n✅ Embedding wrapper ready."
    )