#!/usr/bin/env python3

"""
Unified AgroMind Retriever

Collections:
- Structured Product Collection
- Historical Support Collection

Uses:
- Same embedding model used during indexing
- ChromaDB native retrieval
"""

from typing import Dict
from typing import List
from typing import Optional

import chromadb

from src.config import config
from src.embeddings import ollama_wrapper


class AgroMindRetriever:

    def __init__(self):

        self.client = chromadb.PersistentClient(
            path=config.chromadb_path
        )

        self.products = self.client.get_collection(
            config.retrieval.structured_collection
        )

        self.support = self.client.get_collection(
            config.retrieval.full_collection
        )

        self.embeddings = ollama_wrapper

        self._validate_collections()

    # =====================================================
    # INTERNAL
    # =====================================================

    def _validate_collections(self):

        if self.products.count() == 0:
            raise RuntimeError(
                f"{config.retrieval.structured_collection} is empty"
            )

        if self.support.count() == 0:
            raise RuntimeError(
                f"{config.retrieval.full_collection} is empty"
            )

    def _embed_query(
        self,
        query: str
    ):

        return self.embeddings.embed_query(
            query
        )

    def _normalize_distance(
        self,
        distance: float
    ) -> float:

        return round(
            float(distance),
            4
        )

    # =====================================================
    # PRODUCT LOOKUP
    # =====================================================

    def get_product(
        self,
        product_id: str
    ) -> Optional[Dict]:

        result = self.products.get(
            ids=[product_id]
        )

        if not result["ids"]:
            return None

        metadata = result["metadatas"][0]

        return {
            "product_id":
                metadata.get("product_id"),

            "name_cn":
                metadata.get("name_cn"),

            "name_en":
                metadata.get("name_en"),

            "product_type":
                metadata.get("product_type"),

            "diseases":
                metadata.get("diseases", []),

            "crops":
                metadata.get("crops", []),

            "is_pesticide":
                metadata.get("is_pesticide", False),

            "is_microbial":
                metadata.get("is_microbial", False),

            "is_fertilizer":
                metadata.get("is_fertilizer", False)
        }

    # =====================================================
    # PRODUCT SEARCH
    # =====================================================

    def search_products(
        self,
        query: str,
        k: int = 5
    ) -> List[Dict]:

        query_embedding = self._embed_query(
            query
        )

        results = self.products.query(
            query_embeddings=[
                query_embedding
            ],
            n_results=k,
            include=[
                "metadatas",
                "distances"
            ]
        )

        products = []

        for metadata, distance in zip(
            results["metadatas"][0],
            results["distances"][0]
        ):

            products.append({

                "product_id":
                    metadata.get(
                        "product_id"
                    ),

                "name_cn":
                    metadata.get(
                        "name_cn"
                    ),

                "name_en":
                    metadata.get(
                        "name_en"
                    ),

                "product_type":
                    metadata.get(
                        "product_type"
                    ),

                "diseases":
                    metadata.get(
                        "diseases",
                        []
                    ),

                "crops":
                    metadata.get(
                        "crops",
                        []
                    ),

                "distance":
                    self._normalize_distance(
                        distance
                    )
            })

        return products

    # =====================================================
    # SUPPORT SEARCH
    # =====================================================

    def search_support_cases(
        self,
        query: str,
        k: int = 3,
        category: Optional[str] = None
    ) -> List[Dict]:

        query_embedding = self._embed_query(
            query
        )

        query_kwargs = {
            "query_embeddings": [
                query_embedding
            ],
            "n_results": k,
            "include": [
                "documents",
                "metadatas",
                "distances"
            ]
        }

        if category:

            query_kwargs["where"] = {
                "category": category
            }

        results = self.support.query(
            **query_kwargs
        )

        cases = []

        for doc, metadata, distance in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0]
        ):

            cases.append({

                "category":
                    metadata.get(
                        "category"
                    ),

                "source":
                    metadata.get(
                        "source"
                    ),

                "conversation":
                    doc,

                "distance":
                    self._normalize_distance(
                        distance
                    )
            })

        return cases

    # =====================================================
    # DISEASE SEARCH
    # =====================================================

    def search_disease_products(
        self,
        disease: str,
        k: int = 5
    ) -> List[Dict]:

        return self.search_products(
            query=disease,
            k=k
        )

    # =====================================================
    # AGENT ENTRYPOINT
    # =====================================================

    def retrieve_context(
        self,
        query: str,
        product_k: int = 5,
        support_k: int = 3,
        support_category: Optional[str] = None
    ) -> Dict:

        return {

            "products":
                self.search_products(
                    query=query,
                    k=product_k
                ),

            "support_cases":
                self.search_support_cases(
                    query=query,
                    k=support_k,
                    category=support_category
                )
        }

    # =====================================================
    # HEALTH
    # =====================================================

    def health(self) -> Dict:

        return {

            "structured_collection":
                self.products.count(),

            "support_collection":
                self.support.count(),

            "embedding_model":
                config.embedding.model
        }


retrieval_tool = AgroMindRetriever()