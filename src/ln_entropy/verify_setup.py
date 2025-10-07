#!/usr/bin/env python3
"""
Simple verification script to check if all dependencies are available
检查所有依赖项是否可用的简单验证脚本
"""

def check_imports():
    """Check if all required modules can be imported"""
    print("🔍 Checking LN Entropy dependencies...")
    
    try:
        import torch
        print("✓ PyTorch available")
    except ImportError as e:
        print(f"✗ PyTorch not available: {e}")
        return False
    
    try:
        import transformers
        print("✓ Transformers available")
    except ImportError as e:
        print(f"✗ Transformers not available: {e}")
        return False
    
    try:
        import sentence_transformers
        print("✓ sentence-transformers available")
    except ImportError as e:
        print(f"✗ sentence-transformers not available: {e}")
        return False
    
    try:
        import sklearn
        print("✓ scikit-learn available")
    except ImportError as e:
        print(f"✗ scikit-learn not available: {e}")
        return False
    
    try:
        import numpy
        print("✓ NumPy available")
    except ImportError as e:
        print(f"✗ NumPy not available: {e}")
        return False
    
    try:
        import pandas
        print("✓ Pandas available")
    except ImportError as e:
        print(f"✗ Pandas not available: {e}")
        return False
    
    print("\n🎉 All dependencies are available!")
    print("✓ LN Entropy implementation is ready to use")
    return True

def test_sentence_transformers():
    """Test sentence-transformers functionality"""
    try:
        from sentence_transformers import SentenceTransformer
        print("\n🔍 Testing sentence-transformers...")
        
        # Try to load a small model for testing
        model = SentenceTransformer('all-MiniLM-L6-v2')
        print("✓ Successfully loaded sentence-transformers model")
        
        # Test encoding
        sentences = ["This is a test sentence.", "Another test sentence."]
        embeddings = model.encode(sentences)
        print(f"✓ Successfully encoded {len(sentences)} sentences")
        print(f"  Embedding dimension: {embeddings.shape[1]}")
        
        return True
    except Exception as e:
        print(f"✗ sentence-transformers test failed: {e}")
        return False

if __name__ == "__main__":
    deps_ok = check_imports()
    if deps_ok:
        test_sentence_transformers()
    else:
        print("\n❌ Please install missing dependencies before using LN Entropy")
        print("Run: pip install -r src/ln_entropy/requirements.txt")
