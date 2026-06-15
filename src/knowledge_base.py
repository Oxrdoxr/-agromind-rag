# src/knowledge_base.py - COMPLETE FIXED VERSION
import json
from pathlib import Path
from typing import List, Dict, Optional
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

__version__ = "1.0.0"

class AgroMindKB:
    """Knowledge base for agricultural products"""
    
    def __init__(self, 
                 db_path: str = None,
                 entities_path: str = None,
                 model: str = "bge-m3"):
        
        # Auto-detect paths
        if db_path is None:
            possible_db_paths = [
                Path("data/v4/chroma_db"),
                Path("data/vector_db/chroma_db"),
            ]
            for p in possible_db_paths:
                if p.exists():
                    db_path = str(p)
                    break
            else:
                db_path = "data/v4/chroma_db"
        
        if entities_path is None:
            possible_entities_paths = [
                Path("data/clean_entities.json"),
                Path("clean_entities.json"),
            ]
            for p in possible_entities_paths:
                if p.exists():
                    entities_path = str(p)
                    break
            else:
                raise FileNotFoundError("clean_entities.json not found")
        
        # Load entities
        with open(entities_path, "r", encoding="utf-8") as f:
            entities = json.load(f)
        
        self.products_by_id = {e["product_id"]: e for e in entities}
        self.products_by_name = {e["name_cn"]: e["product_id"] for e in entities}
        
        # Build disease index
        self.disease_index = {}
        for e in entities:
            for disease in e.get("target_diseases", []):
                if disease not in self.disease_index:
                    self.disease_index[disease] = []
                self.disease_index[disease].append(e["product_id"])
        
        # Initialize vector search
        self.embeddings = OllamaEmbeddings(model=model)
        self.db = Chroma(
            persist_directory=db_path,
            embedding_function=self.embeddings,
            collection_name="agromind_v4"
        )
        
        print(f"✅ KB Ready: {len(self.products_by_id)} products, {len(self.disease_index)} diseases")
    
    def search(self, query: str, k: int = 5) -> List[Dict]:
        """Search the knowledge base"""
        # Check for exact ID match
        if query.upper() in self.products_by_id:
            product = self.products_by_id[query.upper()]
            return [{
                "product_id": product["product_id"],
                "name": product["name_cn"],
                "score": 1.0,
                "match_type": "exact_id"
            }]
        
        # Check for exact name match
        if query in self.products_by_name:
            pid = self.products_by_name[query]
            product = self.products_by_id[pid]
            return [{
                "product_id": product["product_id"],
                "name": product["name_cn"],
                "score": 1.0,
                "match_type": "exact_name"
            }]
        
        # Vector search - FIXED: properly extract results
        vector_results = self.db.similarity_search_with_score(query, k=k)
        results = []
        seen_ids = set()
        
        for doc, score in vector_results:
            pid = doc.metadata.get("product_id")
            if pid and pid not in seen_ids:
                seen_ids.add(pid)
                results.append({
                    "product_id": pid,
                    "name": doc.metadata.get("name_cn", "Unknown"),
                    "score": float(score),
                    "match_type": "vector"
                })
        
        return results
    
    def search_by_disease(self, disease: str, k: int = 5) -> List[Dict]:
        """Find products for a disease"""
        if disease in self.disease_index:
            products = []
            for pid in self.disease_index[disease][:k]:
                p = self.products_by_id.get(pid)
                if p:
                    products.append({
                        "product_id": p["product_id"],
                        "name": p["name_cn"],
                        "match_type": "exact_disease"
                    })
            return products
        return self.search(f"Controls {disease}", k=k)
    
    def get_product(self, product_id: str) -> Optional[Dict]:
        """Get product by ID"""
        return self.products_by_id.get(product_id.upper())
    
    @property
    def stats(self) -> Dict:
        """Get statistics"""
        return {
            "products": len(self.products_by_id),
            "diseases": len(self.disease_index),
            "model": "bge-m3"
        }