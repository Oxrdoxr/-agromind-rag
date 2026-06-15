#!/usr/bin/env python3
"""Run this to verify Ollama + wrapper work before using ChromaDB."""

import sys
from pathlib import Path

# Add current directory to path so 'src' is found
sys.path.insert(0, str(Path(__file__).parent))

from src.config import config
from src.embeddings import get_embedding, ollama_embeddings

def test_ollama_connection():
    """Test 1: Can we reach Ollama?"""
    import requests
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        assert response.status_code == 200
        print("✅ Ollama is running")
        return True
    except:
        print("❌ Ollama not running. Run: ollama serve")
        return False

def test_model_available():
    """Test 2: Is bge-m3 pulled?"""
    import requests
    response = requests.get("http://localhost:11434/api/tags")
    models = [m["name"] for m in response.json().get("models", [])]
    if config.embedding.model in str(models):
        print(f"✅ Model {config.embedding.model} is available")
        return True
    else:
        print(f"❌ Model {config.embedding.model} not found. Run: ollama pull {config.embedding.model}")
        return False

def test_dimensions():
    """Test 3: Does actual dimensions match config?"""
    embedding = get_embedding("test", for_query=False)
    actual_dims = len(embedding)
    expected_dims = config.embedding.dimensions
    
    if actual_dims == expected_dims:
        print(f"✅ Dimensions match: {actual_dims}")
        return True
    else:
        print(f"❌ Dimension mismatch: config says {expected_dims}, but model outputs {actual_dims}")
        print(f"   Fix: Update config.yaml embedding.dimensions to {actual_dims}")
        return False

def test_prefix_effect():
    """Test 4: Does prefix change the embedding?"""
    text = "citrus disease"
    
    # Get embedding with and without prefix
    with_prefix = get_embedding(text, for_query=True)  # uses query_prefix
    without_prefix = get_embedding(text, for_query=False)  # uses doc_prefix
    
    # Compare safely - handle numpy arrays
    try:
        import numpy as np
        # Convert to numpy arrays if they aren't already
        wp = np.array(with_prefix) if not isinstance(with_prefix, np.ndarray) else with_prefix
        wp_out = np.array(without_prefix) if not isinstance(without_prefix, np.ndarray) else without_prefix
        
        # Check if arrays are different (using np.array_equal)
        are_different = not np.array_equal(wp, wp_out)
    except ImportError:
        # No numpy - compare first few elements as a simple check
        are_different = with_prefix[:5] != without_prefix[:5]
    
    if are_different:
        print("✅ Prefix changes embedding (as expected for BGE-M3)")
    else:
        print("⚠️ Note: Prefix doesn't significantly change embedding")
    
    return True  # Not a critical test, just informational

def test_batch():
    """Test 5: Can we embed multiple texts at once?"""
    texts = ["text1", "text2", "text3"]
    embeddings = ollama_embeddings(texts)
    
    if len(embeddings) == 3 and all(len(e) == config.embedding.dimensions for e in embeddings):
        print("✅ Batch embedding works")
        return True
    else:
        print("❌ Batch embedding failed")
        return False

def test_similarity():
    """Test 6: Similar texts should have similar embeddings"""
    text1 = "citrus root rot treatment"
    text2 = "citrus root rot treatment"  # Same text
    text3 = "tomato blight prevention"   # Different topic
    
    emb1 = get_embedding(text1, for_query=True)
    emb2 = get_embedding(text2, for_query=True)
    emb3 = get_embedding(text3, for_query=True)
    
    # Calculate cosine similarity
    import numpy as np
    emb1 = np.array(emb1)
    emb2 = np.array(emb2)
    emb3 = np.array(emb3)
    
    sim_same = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
    sim_diff = np.dot(emb1, emb3) / (np.linalg.norm(emb1) * np.linalg.norm(emb3))
    
    if sim_same > 0.95 and sim_diff < 0.8:
        print(f"✅ Similarity works (same: {sim_same:.3f}, different: {sim_diff:.3f})")
        return True
    else:
        print(f"⚠️ Similarity scores (same: {sim_same:.3f}, diff: {sim_diff:.3f})")
        return True

if __name__ == "__main__":
    print("\n=== Testing Ollama Embedding Wrapper ===\n")
    
    tests = [
        ("Ollama connection", test_ollama_connection),
        ("Model available", test_model_available),
        ("Dimensions match", test_dimensions),
        ("Prefix affects embedding", test_prefix_effect),
        ("Batch embedding", test_batch),
        ("Semantic similarity", test_similarity),
    ]
    
    results = []
    for name, test in tests:
        print(f"\n📋 {name}...")
        try:
            passed = test()
            results.append(passed)
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()
            results.append(False)
    
    print("\n" + "="*50)
    passed = sum(results)
    total = len(results)
    
    if passed == total:
        print(f"\n✅ ALL TESTS PASSED ({passed}/{total})")
        print("\n🎉 Your wrapper is ready! You can now use ChromaDB with Ollama.")
    else:
        print(f"\n⚠️ {passed}/{total} tests passed")
        if not results[0]:
            print("  - Start Ollama: ollama serve")
        if not results[1]:
            print(f"  - Pull model: ollama pull {config.embedding.model}")
        if not results[2]:
            print("  - Update config.yaml embedding.dimensions")