"""
Embedding wrapper that makes Ollama look like OpenAI to ChromaDB.
ChromaDB calls this, we forward to Ollama, translate response.
"""

import sys
from pathlib import Path
from typing import List
from unittest.mock import Mock
import requests

# Add parent directory to path if running directly
if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent.parent))

from chromadb.api.types import EmbeddingFunction
from langchain_core.embeddings import Embeddings
from src.config import config


class OllamaEmbeddingFunction(EmbeddingFunction):
    """
    ChromaDB-compatible embedding function using local Ollama.
    ChromaDB thinks it's calling OpenAI, but we intercept and call Ollama.
    """
    
    def __init__(self):
        self.model = config.embedding.model
        self.dimensions = config.embedding.dimensions
        self.query_prefix = config.embedding.query_prefix
        self.doc_prefix = config.embedding.doc_prefix
        self.ollama_url = "http://localhost:11434/api/embeddings"
        
    def __call__(self, input: List[str]) -> List[List[float]]:
        """
        ChromaDB calls this with a list of texts.
        Returns list of embeddings (each a list of floats).
        
        Args:
            input: List of text strings to embed
            
        Returns:
            List of embedding vectors
        """
        embeddings = []
        for text in input:
            # Add the document prefix
            prefixed_text = self.doc_prefix + text
            
            # Call Ollama
            response = requests.post(
                self.ollama_url,
                json={
                    "model": self.model,
                    "prompt": prefixed_text
                },
                timeout=30
            )
            response.raise_for_status()
            
            # Extract embedding from Ollama's response
            embedding = response.json()["embedding"]
            
            # Verify dimension matches config
            if len(embedding) != self.dimensions:
                raise ValueError(
                    f"Expected {self.dimensions} dimensions, "
                    f"got {len(embedding)} from {self.model}"
                )
            
            embeddings.append(embedding)
        
        return embeddings
    
    def embed_query(self, query: str) -> List[float]:
        """
        Special method for query embedding (uses query_prefix).
        Call this separately for search queries.
        """
        prefixed_query = self.query_prefix + query
        
        response = requests.post(
            self.ollama_url,
            json={
                "model": self.model,
                "prompt": prefixed_query
            },
            timeout=30
        )
        response.raise_for_status()
        
        embedding = response.json()["embedding"]
        
        if len(embedding) != self.dimensions:
            raise ValueError(
                f"Expected {self.dimensions} dimensions, "
                f"got {len(embedding)} from {self.model}"
            )
        
        return embedding


class OllamaEmbeddingsWrapper(Embeddings):
    """
    LangChain-compatible wrapper for Ollama embeddings.
    This makes your working Ollama function work with LangChain.
    """
    
    def __init__(self):
        self.model = config.embedding.model
        self.dimensions = config.embedding.dimensions
        self.query_prefix = config.embedding.query_prefix
        self.doc_prefix = config.embedding.doc_prefix
        self.ollama_url = "http://localhost:11434/api/embeddings"
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        LangChain calls this for documents.
        Returns list of embeddings.
        """
        embeddings = []
        for text in texts:
            prefixed_text = self.doc_prefix + text
            response = requests.post(
                self.ollama_url,
                json={"model": self.model, "prompt": prefixed_text},
                timeout=30
            )
            response.raise_for_status()
            embedding = response.json()["embedding"]
            
            if len(embedding) != self.dimensions:
                raise ValueError(
                    f"Expected {self.dimensions} dimensions, "
                    f"got {len(embedding)} from {self.model}"
                )
            embeddings.append(embedding)
        
        return embeddings
    
    def embed_query(self, text: str) -> List[float]:
        """
        LangChain calls this for queries.
        Returns a single embedding.
        """
        prefixed_text = self.query_prefix + text
        response = requests.post(
            self.ollama_url,
            json={"model": self.model, "prompt": prefixed_text},
            timeout=30
        )
        response.raise_for_status()
        embedding = response.json()["embedding"]
        
        if len(embedding) != self.dimensions:
            raise ValueError(
                f"Expected {self.dimensions} dimensions, "
                f"got {len(embedding)} from {self.model}"
            )
        
        return embedding


# Singleton instances
ollama_embeddings = OllamaEmbeddingFunction()
ollama_wrapper = OllamaEmbeddingsWrapper()


# Helper function for direct use (outside ChromaDB)
def get_embedding(text: str, for_query: bool = False) -> List[float]:
    """
    Get embedding for a single text.
    
    Args:
        text: The text to embed
        for_query: If True, uses query_prefix; if False, uses doc_prefix
    
    Returns:
        Embedding vector as list of floats
    """
    if for_query:
        return ollama_embeddings.embed_query(text)
    else:
        return ollama_embeddings([text])[0]


# Quick test when run directly
if __name__ == "__main__":
    print("Testing Ollama embedding wrapper...")
    
    # Test 1: Document embedding
    doc_embedding = get_embedding("Citrus greening disease causes yellow shoots", for_query=False)
    print(f"✅ Document embedding: {len(doc_embedding)} dimensions")
    
    # Test 2: Query embedding
    query_embedding = get_embedding("yellow citrus leaves treatment", for_query=True)
    print(f"✅ Query embedding: {len(query_embedding)} dimensions")
    
    # Test 3: ChromaDB batch compatibility
    batch_input = [
        "Product AF0001 treats root rot",
        "Product AF0002 is for citrus canker",
        "Apply 2ml per liter of water"
    ]
    batch_embeddings = ollama_embeddings(batch_input)
    print(f"✅ Batch embeddings: {len(batch_embeddings)} texts → {len(batch_embeddings[0])} dims each")
    
    # Test 4: LangChain wrapper
    wrapper_embeddings = ollama_wrapper.embed_documents(batch_input)
    print(f"✅ LangChain wrapper: {len(wrapper_embeddings)} texts → {len(wrapper_embeddings[0])} dims each")
    
    print("\n✅ Wrapper works! Ready to use with ChromaDB")