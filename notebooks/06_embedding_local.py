import os, json, ollama, chromadb
from tqdm import tqdm

EMBED_MODEL = 'nomic-embed-text'
CHROMA_PATH = './chromadb_local'
DATA_PATH   = './data'

def embed(text):
    return ollama.embeddings(model=EMBED_MODEL, prompt=text[:2000]).embedding

def build_metadata(entity):
    def sj(v): return ', '.join(str(x) for x in v) if v else ''
    return {
        'product_id':         entity['product_id'],
        'product_name':       entity['product_name'],
        'product_name_cn':    entity['product_name_cn'],
        'product_types':      sj(entity.get('product_types', [])[:5]),
        'target_crops':       sj(entity.get('target_crops', [])[:10]),
        'target_diseases':    sj(entity.get('target_diseases', [])[:10]),
        'target_pests':       sj(entity.get('target_pests', [])[:10]),
        'active_ingredients': sj(entity.get('active_ingredients', [])[:5]),
        'has_diseases':       len(entity.get('target_diseases', [])) > 0,
        'has_pests':          len(entity.get('target_pests', [])) > 0,
        'has_ingredients':    len(entity.get('active_ingredients', [])) > 0,
        'is_pesticide':       any(t in ['Pesticide','Insecticide','Fungicide','Herbicide','Acaricide']
                                  for t in entity.get('product_types', [])),
        'is_microbial':       'Microbial Agent' in entity.get('product_types', []),
        'is_fertilizer':      any('Fertilizer' in t for t in entity.get('product_types', [])),
    }

with open(f'{DATA_PATH}/product_entities_normalized_v2.json', encoding='utf-8') as f:
    entities = {e['product_id']: e for e in json.load(f)}
with open(f'{DATA_PATH}/retrieval_documents_structured_v2.json', encoding='utf-8') as f:
    structured_docs = json.load(f)
with open(f'{DATA_PATH}/retrieval_documents_full_v2.json', encoding='utf-8') as f:
    full_docs = json.load(f)

print(f'Loaded {len(entities)} entities')
print(f'Loaded {len(structured_docs)} structured docs')
print(f'Loaded {len(full_docs)} full docs')

os.makedirs(CHROMA_PATH, exist_ok=True)
client = chromadb.PersistentClient(path=CHROMA_PATH)

for name in ['collection_structured_local', 'collection_full_local']:
    try:
        client.delete_collection(name)
        print(f'Deleted existing {name}')
    except Exception:
        pass

col_s = client.create_collection('collection_structured_local')
col_f = client.create_collection('collection_full_local')
print('ChromaDB initialized fresh ✓\n')

print('Embedding structured docs...')
for doc in tqdm(structured_docs):
    pid = doc['product_id']
    col_s.add(
        ids=[pid],
        documents=[doc['document'][:2000]],
        embeddings=[embed(doc['document'])],
        metadatas=[build_metadata(entities[pid])]
    )
print(f'Stored {col_s.count()} structured docs ✓\n')

print('Embedding full docs...')
for doc in tqdm(full_docs):
    pid = doc['product_id']
    col_f.add(
        ids=[pid],
        documents=[doc['document'][:2000]],
        embeddings=[embed(doc['document'])],
        metadatas=[build_metadata(entities[pid])]
    )
print(f'Stored {col_f.count()} full docs ✓\n')

print('=== SMOKE TEST ===')
for query in ['citrus yellowing leaves', '柑橘叶片发黄', 'root rot treatment']:
    results = col_f.query(
        query_embeddings=[embed(query)],
        n_results=2,
        include=['metadatas', 'distances']
    )
    top  = results['metadatas'][0][0]
    dist = results['distances'][0][0]
    print(f'"{query}" → {top["product_id"]} ({top["product_name"]}) dist={dist:.1f}')

with open('embedding_log_local.json', 'w') as f:
    json.dump({
        'embed_model':  EMBED_MODEL,
        'dims':         768,
        'structured':   col_s.count(),
        'full':         col_f.count(),
        'truncated_at': 2000,
        'plan':         'B — fully local'
    }, f, indent=2)

print('\n=== DONE ✓ ===')
print(f'Structured: {col_s.count()} | Full: {col_f.count()}')
print(f'ChromaDB → {CHROMA_PATH}')