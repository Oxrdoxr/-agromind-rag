# src/retrieval_tool.py - LangChain compatible version
"""Retrieval tool for AgroMind - LangChain compatible"""

import json
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from src.config import config
from src.embeddings import ollama_embeddings


class OllamaEmbeddingsWrapper(Embeddings):
    """
    Wrapper to make the local Ollama embedding function compatible with LangChain.
    """

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """LangChain calls this for document embeddings."""
        return ollama_embeddings(texts)

    def embed_query(self, text: str) -> List[float]:
        """LangChain calls this for query embeddings."""
        return ollama_embeddings.embed_query(text)


class AgroMindRetriever:
    """
    Hybrid retriever for Agro-Mind.

    Supports:
    - exact product ID lookup
    - exact disease matching from clean_entities.json
    - vector search through ChromaDB
    """

    def __init__(
        self,
        db_path: Optional[str] = None,
        collection_name: Optional[str] = None,
        entities_path: Optional[str] = None,
    ):
        db_path = db_path or config.chromadb_path
        collection_name = collection_name or config.retrieval.structured_collection
        entities_path = entities_path or str(Path(config.data_path) / "clean_entities.json")

        self.db_path = db_path
        self.collection_name = collection_name
        self.entities_path = Path(entities_path)

        self.embeddings = OllamaEmbeddingsWrapper()

        self.vectorstore = Chroma(
            persist_directory=self.db_path,
            embedding_function=self.embeddings,
            collection_name=self.collection_name,
        )

        self._load_entities()
        self._cache: Dict[str, List[Tuple[Document, float]]] = {}

    def _load_entities(self) -> None:
        """Load product entities for exact matching."""
        if self.entities_path.exists():
            with open(self.entities_path, "r", encoding="utf-8") as file:
                entities = json.load(file)

            self.product_by_id = {
                entity["product_id"].upper(): entity
                for entity in entities
                if entity.get("product_id")
            }

            self.product_names = {
                entity["product_id"].upper(): entity.get(
                    "name_cn",
                    entity.get("name_en", entity.get("name", "")),
                )
                for entity in entities
                if entity.get("product_id")
            }

            self.product_diseases: Dict[str, List[str]] = {}

            for entity in entities:
                product_id = entity.get("product_id", "").upper()
                diseases = entity.get("target_diseases", [])

                for disease in diseases:
                    if disease not in self.product_diseases:
                        self.product_diseases[disease] = []
                    self.product_diseases[disease].append(product_id)

        else:
            print(f"Warning: Entities file not found at {self.entities_path}")
            self.product_by_id = {}
            self.product_names = {}
            self.product_diseases = {}

    def search(
        self,
        query: str,
        k: int = 10,
        use_hybrid: bool = True,
    ) -> List[Tuple[Document, float]]:
        """Search using exact product ID match first, then vector search."""
        query = query.strip()
        cache_key = f"{query}_{k}_{use_hybrid}"

        if cache_key in self._cache:
            return self._cache[cache_key]

        results: List[Tuple[Document, float]] = []
        seen_ids = set()

        query_upper = query.upper()

        if use_hybrid and query_upper in self.product_by_id:
            product = self.product_by_id[query_upper]
            exact_doc = self._create_document_from_entity(product)

            results.append((exact_doc, 1.0))
            seen_ids.add(query_upper)

            vector_results = self.vectorstore.similarity_search_with_score(
                product.get("name_cn", "") or product.get("name_en", ""),
                k=k,
            )

            for doc, score in vector_results:
                product_id = str(doc.metadata.get("product_id", "")).upper()

                if product_id not in seen_ids:
                    results.append((doc, score))
                    seen_ids.add(product_id)

                if len(results) >= k:
                    break

        else:
            results = self.vectorstore.similarity_search_with_score(query, k=k)

        self._cache[cache_key] = results
        return results

    def search_by_disease(
        self,
        disease: str,
        k: int = 10,
    ) -> List[Tuple[Document, float]]:
        """Search for products that control a specific disease."""
        disease = disease.strip()

        if disease in self.product_diseases:
            exact_products = self.product_diseases[disease]
            results: List[Tuple[Document, float]] = []
            seen_ids = set()

            for product_id in exact_products[:k]:
                product_id = product_id.upper()

                if product_id in self.product_by_id:
                    doc = self._create_document_from_entity(self.product_by_id[product_id])
                    results.append((doc, 1.0))
                    seen_ids.add(product_id)

            remaining = k - len(results)

            if remaining > 0:
                vector_results = self.vectorstore.similarity_search_with_score(
                    f"Controls {disease}",
                    k=remaining * 2,
                )

                for doc, score in vector_results:
                    product_id = str(doc.metadata.get("product_id", "")).upper()

                    if product_id not in seen_ids:
                        results.append((doc, score))
                        seen_ids.add(product_id)

                    if len(results) >= k:
                        break

            return results[:k]

        return self.vectorstore.similarity_search_with_score(
            f"Controls {disease}",
            k=k,
        )

    def search_by_id(self, product_id: str) -> Optional[Dict[str, Any]]:
        """Get product by exact ID."""
        product_id = product_id.strip().upper()
        return self.product_by_id.get(product_id)

    def get_product_info(self, product_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed product information by ID."""
        return self.search_by_id(product_id)

    def search_by_ingredient(
        self,
        ingredient: str,
        k: int = 10,
    ) -> List[Tuple[Document, float]]:
        """Search for products containing a specific ingredient."""
        return self.vectorstore.similarity_search_with_score(
            f"Ingredients: {ingredient}",
            k=k,
        )

    def search_by_crop(
        self,
        crop: str,
        k: int = 10,
    ) -> List[Tuple[Document, float]]:
        """Search for products that work on a specific crop."""
        return self.vectorstore.similarity_search_with_score(
            f"Crops: {crop}",
            k=k,
        )

    def _create_document_from_entity(self, entity: Dict[str, Any]) -> Document:
        """Create a LangChain Document from an entity dictionary."""
        product_id = entity.get("product_id", "")
        name_cn = entity.get("name_cn", "")
        name_en = entity.get("name_en", "")
        diseases = entity.get("target_diseases", [])
        crops = entity.get("target_crops", [])
        pests = entity.get("target_pests", [])
        ingredients = entity.get("active_ingredients", [])
        symptoms = entity.get("symptoms_addressed", [])

        text_parts = [
            f"Product ID: {product_id}",
            f"Chinese Name: {name_cn}",
            f"English Name: {name_en}",
        ]

        if crops:
            text_parts.append(f"Crops: {', '.join(crops)}")

        if diseases:
            text_parts.append(f"Diseases: {', '.join(diseases)}")

        if pests:
            text_parts.append(f"Pests: {', '.join(pests)}")

        if ingredients:
            text_parts.append(f"Ingredients: {', '.join(ingredients)}")

        if symptoms:
            text_parts.append(f"Symptoms addressed: {', '.join(symptoms)}")

        return Document(
            page_content="\n".join(text_parts),
            metadata={
                "product_id": product_id,
                "name_cn": name_cn,
                "name_en": name_en,
                "diseases": diseases,
                "crops": crops,
                "pests": pests,
                "ingredients": ingredients,
                "symptoms": symptoms,
            },
        )

    def clear_cache(self) -> None:
        """Clear the search cache."""
        self._cache.clear()


def create_rag_tool(retriever: AgroMindRetriever):
    """Create a LangChain tool from the retriever."""

    try:
        from langchain_core.tools import Tool
    except ImportError:
        from langchain.tools import Tool

    def search_func(query: str) -> str:
        results = retriever.search(query, k=3)

        if not results:
            return "No relevant products found in the AgroMind knowledge base."

        context = "Relevant products from AgroMind knowledge base:\n\n"

        for index, (doc, score) in enumerate(results, 1):
            metadata = doc.metadata or {}

            name_cn = metadata.get("name_cn", "Unknown Chinese name")
            name_en = metadata.get("name_en", "Unknown English name")
            product_id = metadata.get("product_id", "Unknown ID")

            context += f"{index}. {name_cn} / {name_en} (ID: {product_id})\n"
            context += f"{doc.page_content}\n"
            context += f"Retrieval score: {score:.4f}\n\n"

        return context

    return Tool(
        name="AgroMind_Product_Search",
        func=search_func,
        description=(
            "Search the AgroMind agricultural product database. "
            "Use this tool when farmers ask about disease treatments, pest control, "
            "product information by ID, crop-specific treatments, symptoms, or active ingredients. "
            "Input should be a natural language query about agricultural products or diseases. "
            "Output is a list of relevant products with details and retrieval scores."
        ),
    )


if __name__ == "__main__":
    print("=" * 60)
    print("TESTING AGROMIND RETRIEVER (LangChain Compatible)")
    print("=" * 60)

    print(f"ChromaDB path: {config.chromadb_path}")
    print(f"Collection: {config.retrieval.full_collection}")
    print(f"Data path: {config.data_path}")

    retriever = AgroMindRetriever()

    print(f"Entities: {retriever.entities_path}")

    print("\n1. Search 'citrus root rot':")
    results = retriever.search("citrus root rot treatment", k=3)

    if not results:
        print("   No vector results found.")
    else:
        for doc, score in results:
            print(
                f"   {doc.metadata.get('product_id')}: "
                f"{doc.metadata.get('name_cn')} / {doc.metadata.get('name_en')} "
                f"(score: {score:.4f})"
            )

    print("\n2. Exact product lookup 'AF0001':")
    product = retriever.get_product_info("AF0001")

    if product:
        print(
            f"   {product.get('product_id')}: "
            f"{product.get('name_cn')} / {product.get('name_en')}"
        )
    else:
        print("   Product not found.")

    print("\n3. Testing LangChain Tool:")
    tool = create_rag_tool(retriever)
    result = tool.func("What treats citrus canker?")
    print(result[:500] + "...")

    print("\n✅ Retriever ready for LangChain agent!")