"""
Embedding wrapper that makes Ollama look like OpenAI to ChromaDB.
ChromaDB calls this, we forward to Ollama, translate response.
"""

from typing import List
from unittest.mock import Mock
import requests
from chromadb.api.types import EmbeddingFunction

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
        # Determine if this is a query or document based on context
        # ChromaDB doesn't tell us, so we assume documents during indexing
        # For queries, you'll call .embed_query() separately
        prefix = self.doc_prefix
        
        embeddings = []
        for text in input:
            # Add the appropriate prefix
            prefixed_text = prefix + text
            
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


class OpenAICompatibleEmbeddingFunction(EmbeddingFunction):
    """
    Alternative: Makes Ollama look EXACTLY like OpenAI API.
    Use this if ChromaDB is configured for OpenAI and you can't change it.
    """
    
    def __init__(self):
        self.model = config.embedding.model
        self.dimensions = config.embedding.dimensions
        self.query_prefix = config.embedding.query_prefix
        self.doc_prefix = config.embedding.doc_prefix
        
        # Create a mock OpenAI client that actually calls Ollama
        self.client = Mock()
        self.client.embeddings.create = self._mock_embed_create
    
    def _mock_embed_create(self, model: str, input: List[str]):
        """Pretend to be OpenAI's embeddings.create method."""
        # This returns a mock object that ChromaDB will interrogate
        mock_response = Mock()
        mock_response.data = []
        
        for i, text in enumerate(input):
            # Call Ollama for each text
            prefixed_text = self.doc_prefix + text
            response = requests.post(
                "http://localhost:11434/api/embeddings",
                json={"model": self.model, "prompt": prefixed_text},
                timeout=30
            )
            response.raise_for_status()
            embedding = response.json()["embedding"]
            
            # Create mock embedding object
            mock_embedding = Mock()
            mock_embedding.embedding = embedding
            mock_embedding.index = i
            mock_response.data.append(mock_embedding)
        
        return mock_response
    
    def __call__(self, input: List[str]) -> List[List[float]]:
        """ChromaDB calls this - use the mock OpenAI client."""
        response = self.client.embeddings.create(
            model=self.model,
            input=input
        )
        # Extract embeddings in order
        embeddings = [item.embedding for item in sorted(response.data, key=lambda x: x.index)]
        return embeddings


# Singleton instance
ollama_embeddings = OllamaEmbeddingFunction()


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
        # Call __call__ with single item list
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
    
    print("\n✅ Wrapper works! Ready to use with ChromaDB")