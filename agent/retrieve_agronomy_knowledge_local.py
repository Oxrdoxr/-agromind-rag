"""
retrieve_agronomy_knowledge_local.py
Plan B — fully local RAG tool
Uses nomic-embed-text via Ollama for query embedding
Uses chromadb_local/ built by 06_embedding_local.py
"""
import ollama
import chromadb

EMBED_MODEL  = 'nomic-embed-text'
CHROMA_PATH  = './chromadb_local'

_client      = None
_col_struct  = None
_col_full    = None

def _init():
    global _client, _col_struct, _col_full
    if _client is None:
        _client     = chromadb.PersistentClient(path=CHROMA_PATH)
        _col_struct = _client.get_collection('collection_structured_local')
        _col_full   = _client.get_collection('collection_full_local')

QUERY_TYPE_FILTERS = {
    'disease':      {'has_diseases':    True},
    'pest':         {'has_pests':       True},
    'ingredient':   {'has_ingredients': True},
    'safety':       {'is_pesticide':    True},
    'microbial':    {'is_microbial':    True},
    'fertilizer':   {'is_fertilizer':  True},
    'crop':         None,
    'symptom':      None,
    'dosage':       None,
    'product_name': None,
    'chinese':      None,
    'general':      None,
}

FULL_DOC_TYPES = {'symptom', 'chinese', 'general', 'safety'}

def retrieve_agronomy_knowledge(
    query: str,
    query_type: str = 'general',
    n_results: int = 3,
    use_full_docs: bool = None,
) -> list[dict]:
    """
    Retrieve relevant agricultural product documents.
    Fully local — no API calls required.

    Args:
        query:         User query (English or Chinese)
        query_type:    disease/pest/ingredient/crop/symptom/dosage/
                       safety/product_name/chinese/general
        n_results:     Number of results (default 3)
        use_full_docs: None = auto-decide based on query_type

    Returns:
        List of dicts: product_id, product_name, product_name_cn,
                       document, distance, collection
    """
    _init()

    if use_full_docs is None:
        use_full_docs = query_type in FULL_DOC_TYPES

    collection = _col_full if use_full_docs else _col_struct
    where      = QUERY_TYPE_FILTERS.get(query_type, None)

    # Embed query locally
    q_vector = ollama.embeddings(model=EMBED_MODEL, prompt=query[:2000]).embedding

    kwargs = {
        'query_embeddings': [q_vector],
        'n_results':        n_results,
        'include':          ['documents', 'metadatas', 'distances'],
    }
    if where:
        kwargs['where'] = where

    raw = collection.query(**kwargs)

    return [
        {
            'product_id':      raw['ids'][0][i],
            'product_name':    raw['metadatas'][0][i]['product_name'],
            'product_name_cn': raw['metadatas'][0][i]['product_name_cn'],
            'document':        raw['documents'][0][i],
            'distance':        round(raw['distances'][0][i], 2),
            'collection':      'full' if use_full_docs else 'structured',
        }
        for i in range(len(raw['ids'][0]))
    ]


if __name__ == '__main__':
    print('=== RETRIEVAL TOOL TEST ===\n')
    tests = [
        ('citrus yellowing leaves', 'symptom'),
        ('what treats root rot',    'disease'),
        ('柑橘叶片发黄怎么处理',       'chinese'),
        ('Bacillus subtilis',       'ingredient'),
        ('glyphosate herbicide',    'pest'),
    ]
    for query, qtype in tests:
        results = retrieve_agronomy_knowledge(query, query_type=qtype, n_results=2)
        print(f'[{qtype}] "{query}"')
        for r in results:
            print(f'  → {r["product_id"]} | {r["product_name"]} | dist={r["distance"]}')
        print()