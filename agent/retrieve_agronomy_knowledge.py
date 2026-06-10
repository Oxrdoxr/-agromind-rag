"""
retrieve_agronomy_knowledge — RAG tool for Agro-Mind agent
Generated from Phase 7 retrieval testing results.
 
Usage:
    results = retrieve_agronomy_knowledge(
        query="what treats root rot in citrus",
        query_type="disease",        # disease/pest/ingredient/crop/symptom/
                                     # dosage/safety/product_name/general
        n_results=3,
        use_full_docs=False          # True for semantic/Chinese queries
    )
"""
import os
import chromadb
from openai import OpenAI
 
EMBED_MODEL  = "text-embedding-3-large"
CHROMA_PATH  = "/content/drive/MyDrive/agromind/chromadb"
 
_client       = None
_chroma       = None
_col_struct   = None
_col_full     = None
 
def _init():
    global _client, _chroma, _col_struct, _col_full
    if _client is None:
        _client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    if _chroma is None:
        _chroma     = chromadb.PersistentClient(path=CHROMA_PATH)
        _col_struct = _chroma.get_collection("collection_structured")
        _col_full   = _chroma.get_collection("collection_full")
 
# Metadata filters by query type — learned from Phase 7 testing
QUERY_TYPE_FILTERS = {
    "disease":      {"has_diseases":  True},
    "pest":         {"has_pests":     True},
    "ingredient":   {"has_ingredients": True},
    "safety":       {"is_pesticide":  True},
    "microbial":    {"is_microbial":  True},
    "fertilizer":   {"is_fertilizer": True},
    "crop":         None,
    "symptom":      None,
    "dosage":       None,
    "product_name": None,
    "chinese":      None,
    "general":      None,
}
 
# Query types that benefit from full bilingual documents
FULL_DOC_TYPES = {"symptom", "chinese", "general", "safety"}
 
def retrieve_agronomy_knowledge(
    query: str,
    query_type: str = "general",
    n_results: int = 3,
    use_full_docs: bool = None,  # None = auto-decide based on query_type
) -> list[dict]:
    """
    Retrieve relevant agricultural product documents from ChromaDB.
 
    Args:
        query:         User query string (English or Chinese)
        query_type:    Type of query for metadata pre-filtering
                       Options: disease, pest, ingredient, crop, symptom,
                                dosage, safety, product_name, chinese, general
        n_results:     Number of results to return (default 3)
        use_full_docs: True = search full bilingual collection
                       False = search structured collection
                       None = auto-decide (full for semantic/Chinese queries)
 
    Returns:
        List of dicts with product_id, product_name, product_name_cn,
        document, distance
    """
    _init()
 
    # Auto-decide collection
    if use_full_docs is None:
        use_full_docs = query_type in FULL_DOC_TYPES
 
    collection = _col_full if use_full_docs else _col_struct
 
    # Get metadata filter for this query type
    where = QUERY_TYPE_FILTERS.get(query_type, None)
 
    # Embed query
    resp     = _client.embeddings.create(model=EMBED_MODEL, input=[query])
    q_vector = resp.data[0].embedding
 
    # Search
    kwargs = {
        "query_embeddings": [q_vector],
        "n_results":        n_results,
        "include":          ["documents", "metadatas", "distances"],
    }
    if where:
        kwargs["where"] = where
 
    raw = collection.query(**kwargs)
 
    return [
        {
            "product_id":      raw["ids"][0][i],
            "product_name":    raw["metadatas"][0][i]["product_name"],
            "product_name_cn": raw["metadatas"][0][i]["product_name_cn"],
            "document":        raw["documents"][0][i],
            "distance":        round(raw["distances"][0][i], 4),
            "collection":      "full" if use_full_docs else "structured",
        }
        for i in range(len(raw["ids"][0]))
    ]
 
 
if __name__ == "__main__":
    # Quick test
    results = retrieve_agronomy_knowledge(
        "citrus leaf yellowing treatment",
        query_type="symptom"
    )
    for r in results:
        print(f"{r['product_id']} | {r['product_name']} | dist={r['distance']}")
