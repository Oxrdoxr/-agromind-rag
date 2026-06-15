"""Unit tests for retrieval"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.retrieve import ProductRetriever

def test_exact_id():
    retriever = ProductRetriever()
    results = retriever.search_by_id("AF0001", k=3)
    ids = [meta["product_id"] for meta, _ in results]
    assert "AF0001" in ids, f"AF0001 not found in {ids}"
    print("✓ Exact ID test passed")

def test_disease_search():
    retriever = ProductRetriever()
    results = retriever.search_by_disease("炭疽病", k=10)
    ids = [meta["product_id"] for meta, _ in results]
    # At least one product should be found
    assert len(ids) > 0, "No products found for 炭疽病"
    print(f"✓ Disease search found {len(ids)} products")

if __name__ == "__main__":
    test_exact_id()
    test_disease_search()
    print("\n✅ All tests passed!")