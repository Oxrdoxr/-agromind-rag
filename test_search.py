from src.retrieval_tool import AgroMindRetriever

# Initialize retriever
r = AgroMindRetriever()

# Check if products loaded
print(f"✅ Products loaded: {len(r.product_by_id)}")
print(f"Sample products: {list(r.product_by_id.keys())[:5]}")

# Search for root rot treatment
results = r.search("root rot", k=3)

print("\n=== Root Rot Search ===")
for doc, score in results:
    print(f"✅ {doc.metadata['product_id']}: {doc.metadata['name_cn']}")
    print(f"   Score: {score:.4f}")
    print(f"   Diseases: {doc.metadata.get('diseases', [])[:3]}")
    print()

# Search for anthracnose in Chinese
results = r.search("炭疽病", k=3)

print("\n=== 炭疽病 Search ===")
for doc, score in results:
    print(f"✅ {doc.metadata['product_id']}: {doc.metadata['name_cn']}")
    print(f"   Score: {score:.4f}")
    print()
    print(f"   Diseases: {doc.metadata.get('diseases', [])[:3]}")

# Search for citrus products
results = r.search("citrus", k=5)

print("\n=== Citrus Products ===")
for doc, score in results:
    print(f"✅ {doc.metadata['product_id']}: {doc.metadata['name_cn']}")
    print(f"   Score: {score:.4f}")
    print()

# Search for citrus in Chinese
results = r.search("柑橘", k=5)

print("\n=== 柑橘 Products ===")
for doc, score in results:
    print(f"✅ {doc.metadata['product_id']}: {doc.metadata['name_cn']}")
    print(f"   Score: {score:.4f}")
    print()

# Get specific product
product = r.get_product_info("AF0001")

print("\n=== Product AF0001 ===")
if product:
    print(f"Name: {product.get('name_cn')}")
    print(f"English: {product.get('name_en')}")
    print(f"Diseases: {product.get('target_diseases', [])[:5]}")
    print(f"Crops: {product.get('target_crops', [])[:5]}")
else:
    print("❌ Product not found")

# Search for yellow leaves symptom
results = r.search("yellow leaves", k=3)

print("\n=== Yellow Leaves Search ===")
for doc, score in results:
    print(f"✅ {doc.metadata['product_id']}: {doc.metadata['name_cn']}")
    print(f"   Score: {score:.4f}")
    print()

# Search for yellow leaves symptom
results = r.search("yellow leaves", k=3)

print("\n=== Yellow Leaves Search ===")
for doc, score in results:
    print(f"✅ {doc.metadata['product_id']}: {doc.metadata['name_cn']}")
    print(f"   Score: {score:.4f}")
    print()

# Search for Bacillus subtilis
results = r.search("枯草芽孢杆菌", k=3)

print("\n=== 枯草芽孢杆菌 Search ===")
for doc, score in results:
    print(f"✅ {doc.metadata['product_id']}: {doc.metadata['name_cn']}")
    print(f"   Score: {score:.4f}")
    print()

# Run all tests at once
def test_search():
    from src.retrieval_tool import AgroMindRetriever
    
    r = AgroMindRetriever()
    
    test_queries = [
        ("root rot", "Disease (EN)"),
        ("炭疽病", "Disease (CN)"),
        ("citrus", "Crop (EN)"),
        ("柑橘", "Crop (CN)"),
        ("yellow leaves", "Symptom"),
        ("枯草芽孢杆菌", "Ingredient"),
    ]
    
    print("="*60)
    print("SEARCH TEST RESULTS")
    print("="*60)
    
    for query, desc in test_queries:
        results = r.search(query, k=3)
        print(f"\n{desc}: '{query}'")
        if results:
            for doc, score in results[:3]:
                print(f"  ✅ {doc.metadata['product_id']}: {doc.metadata['name_cn']} ({score:.3f})")
        else:
            print("  ❌ No results")
    
    print("\n" + "="*60)
    print(f"✅ Total products loaded: {len(r.product_by_id)}")
    print("="*60)

# Run the test
test_search()

# Direct ChromaDB check
from src.config import config
import chromadb

client = chromadb.PersistentClient(path=config.chromadb_path)
collections = client.list_collections()

print("\n=== ChromaDB Status ===")
print(f"Collections: {[c.name for c in collections]}")

for c in collections:
    count = c.count()
    print(f"  {c.name}: {count} documents")