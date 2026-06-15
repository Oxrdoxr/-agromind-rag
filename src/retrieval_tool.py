# src/retrieval_tool.py - LangChain compatible version (KEEP THIS)
"""Retrieval tool for AgroMind - LangChain compatible"""
import json
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

# Import YOUR working wrapper
from src.embeddings import ollama_embeddings


class OllamaEmbeddingsWrapper(Embeddings):
    """
    Wrapper to make your Ollama function compatible with LangChain.
    This lets LangChain use your working embeddings.
    """
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """LangChain calls this for documents"""
        return ollama_embeddings(texts)
    
    def embed_query(self, text: str) -> List[float]:
        """LangChain calls this for queries"""
        return ollama_embeddings.embed_query(text)


class AgroMindRetriever:
    """Hybrid retriever - LangChain compatible, using your working embeddings"""
    
    def __init__(self, 
                 db_path: str = "data/v4/chroma_db",
                 collection_name: str = "agromind_v4",
                 entities_path: str = "data/clean_entities.json"):
        
        # Use your working embeddings wrapped for LangChain
        self.embeddings = OllamaEmbeddingsWrapper()
        
        # LangChain's Chroma wrapper (now using your working embeddings)
        self.vectorstore = Chroma(
            persist_directory=db_path,
            embedding_function=self.embeddings,  # ← Your working wrapper!
            collection_name=collection_name
        )
        
        # Load entities for exact matching
        self.entities_path = Path(entities_path)
        self._load_entities()
        
        # Cache
        self._cache = {}
    
    def _load_entities(self):
        """Load product entities for exact matching"""
        if self.entities_path.exists():
            with open(self.entities_path, "r", encoding="utf-8") as f:
                entities = json.load(f)
            self.product_by_id = {e["product_id"]: e for e in entities}
            self.product_names = {e["product_id"]: e.get("name_cn", e.get("name", "")) for e in entities}
            self.product_diseases = {}
            for e in entities:
                diseases = e.get("target_diseases", [])
                if diseases:
                    for disease in diseases:
                        if disease not in self.product_diseases:
                            self.product_diseases[disease] = []
                        self.product_diseases[disease].append(e["product_id"])
        else:
            print(f"⚠️ Warning: Entities file not found at {self.entities_path}")
            self.product_by_id = {}
            self.product_names = {}
            self.product_diseases = {}
    
    def search(self, query: str, k: int = 10, use_hybrid: bool = True) -> List[Tuple[Document, float]]:
        """Search using LangChain's similarity_search"""
        cache_key = f"{query}_{k}"
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        results = []
        seen_ids = set()
        
        # Exact match for product ID
        if use_hybrid and query.upper() in self.product_names:
            product_id = query.upper()
            product = self.product_by_id[product_id]
            exact_doc = self._create_document_from_entity(product)
            results.append((exact_doc, 1.0))
            seen_ids.add(product_id)
            
            # Vector search for similar products
            vector_results = self.vectorstore.similarity_search_with_score(
                product.get("name_cn", ""),
                k=k
            )
            
            for doc, score in vector_results:
                if doc.metadata.get("product_id") not in seen_ids:
                    results.append((doc, score))
                    if len(results) >= k:
                        break
        
        else:
            # Regular vector search via LangChain
            results = self.vectorstore.similarity_search_with_score(query, k=k)
        
        self._cache[cache_key] = results
        return results
    
    def search_by_disease(self, disease: str, k: int = 10) -> List[Tuple[Document, float]]:
        """Search for products that control a specific disease"""
        if disease in self.product_diseases:
            exact_products = self.product_diseases[disease]
            results = []
            seen_ids = set()
            
            for pid in exact_products[:k]:
                if pid in self.product_by_id:
                    doc = self._create_document_from_entity(self.product_by_id[pid])
                    results.append((doc, 1.0))
                    seen_ids.add(pid)
            
            remaining = k - len(results)
            if remaining > 0:
                vector_results = self.vectorstore.similarity_search_with_score(
                    f"Controls {disease}",
                    k=remaining * 2
                )
                for doc, score in vector_results:
                    if doc.metadata.get("product_id") not in seen_ids:
                        results.append((doc, score))
                        if len(results) >= k:
                            break
            
            return results[:k]
        
        return self.vectorstore.similarity_search_with_score(f"Controls {disease}", k=k)
    
    def search_by_id(self, product_id: str) -> Optional[Dict[str, Any]]:
        """Get product by exact ID"""
        product_id = product_id.upper()
        if product_id in self.product_by_id:
            return self.product_by_id[product_id]
        return None
    
    def get_product_info(self, product_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed product information by ID"""
        return self.search_by_id(product_id)
    
    def search_by_ingredient(self, ingredient: str, k: int = 10) -> List[Tuple[Document, float]]:
        """Search for products containing a specific ingredient"""
        return self.vectorstore.similarity_search_with_score(f"Ingredients: {ingredient}", k=k)
    
    def search_by_crop(self, crop: str, k: int = 10) -> List[Tuple[Document, float]]:
        """Search for products that work on a specific crop"""
        return self.vectorstore.similarity_search_with_score(f"Crops: {crop}", k=k)
    
    def _create_document_from_entity(self, entity: Dict[str, Any]) -> Document:
        """Create a Document from an entity dictionary"""
        product_id = entity.get("product_id", "")
        name_cn = entity.get("name_cn", "")
        name_en = entity.get("name_en", "")
        diseases = entity.get("target_diseases", [])
        crops = entity.get("target_crops", [])
        ingredients = entity.get("active_ingredients", [])
        
        text_parts = [
            f"Product ID: {product_id}",
            f"Chinese Name: {name_cn}",
            f"English Name: {name_en}",
        ]
        
        if diseases:
            text_parts.append(f"Diseases: {', '.join(diseases)}")
        if crops:
            text_parts.append(f"Crops: {', '.join(crops)}")
        if ingredients:
            text_parts.append(f"Ingredients: {', '.join(ingredients)}")
        
        return Document(
            page_content="\n".join(text_parts),
            metadata={
                "product_id": product_id,
                "name_cn": name_cn,
                "name_en": name_en,
                "diseases": diseases,
                "crops": crops
            }
        )
    
    def clear_cache(self):
        """Clear the search cache"""
        self._cache.clear()


# LangChain Tool wrapper for agent use
def create_rag_tool(retriever: AgroMindRetriever):
    """Create a LangChain tool from the retriever"""
    from langchain.tools import Tool
    
    def search_func(query: str) -> str:
        results = retriever.search(query, k=3)
        if not results:
            return "No relevant products found in the knowledge base."
        
        context = "Relevant products from AgroMind knowledge base:\n\n"
        for i, (doc, score) in enumerate(results, 1):
            context += f"{i}. **{doc.metadata['name_cn']}** (ID: {doc.metadata['product_id']})\n"
            context += f"   {doc.page_content}\n"
            context += f"   Confidence: {score:.2f}\n\n"
        
        return context
    
    return Tool(
        name="AgroMind_Product_Search",
        func=search_func,
        description="""Search the AgroMind agricultural product database. 
Use this tool when farmers ask about:
- Disease treatments (e.g., "root rot", "anthracnose", "powdery mildew")
- Pest control products
- Product information by ID (e.g., "AF0001")
- Crop-specific treatments (e.g., "citrus diseases", "tomato blight")
- Active ingredients

Input: Natural language query about agricultural products or diseases.
Output: List of relevant products with their details and confidence scores."""
    )


# Quick test
if __name__ == "__main__":
    print("="*60)
    print("TESTING AGROMIND RETRIEVER (LangChain Compatible)")
    print("="*60)
    
    retriever = AgroMindRetriever()
    
    # Test search
    print("\n1. Search 'citrus root rot':")
    results = retriever.search("citrus root rot treatment", k=3)
    for doc, score in results:
        print(f"   {doc.metadata['product_id']}: {doc.metadata['name_cn']} (score: {score:.4f})")
    
    # Test LangChain tool
    print("\n2. Testing LangChain Tool:")
    tool = create_rag_tool(retriever)
    result = tool.func("What treats citrus canker?")
    print(result[:300] + "...")
    
    print("\n✅ Retriever ready for LangChain agent!")