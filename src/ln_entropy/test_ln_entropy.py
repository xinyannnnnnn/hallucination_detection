#!/usr/bin/env python3
"""
Test script for LN Entropy implementation
测试LN熵实现的脚本
"""

import sys
import numpy as np
from pathlib import Path

# Add src directory to path
CURRENT_DIR = Path(__file__).resolve().parent
SRC_DIR = CURRENT_DIR.parent
sys.path.insert(0, str(SRC_DIR))

from ln_entropy.main import LNEntropyCalculator

def test_ln_entropy_basic():
    """Test basic LN Entropy functionality"""
    print("Testing basic LN Entropy functionality...")
    
    # Initialize calculator with optimized multilingual model
    calc = LNEntropyCalculator(
        embedding_model='paraphrase-multilingual-MiniLM-L12-v2',
        device='cpu'  # Use CPU for testing
    )
    
    # Test case 1: Identical responses (should have low entropy)
    identical_responses = [
        "Paris is the capital of France.",
        "Paris is the capital of France.",
        "Paris is the capital of France."
    ]
    
    entropy1, info1 = calc.compute_uncertainty(identical_responses)
    print(f"Identical responses - LN Entropy: {entropy1:.4f}, Clusters: {info1['num_clusters']}")
    
    # Test case 2: Different responses (should have higher entropy)
    different_responses = [
        "Paris is the capital of France.",
        "London is the capital of England.", 
        "Berlin is the capital of Germany."
    ]
    
    entropy2, info2 = calc.compute_uncertainty(different_responses)
    print(f"Different responses - LN Entropy: {entropy2:.4f}, Clusters: {info2['num_clusters']}")
    
    # Test case 3: Semantically similar but different wording
    similar_responses = [
        "Paris is the capital of France.",
        "The capital of France is Paris.",
        "France's capital city is Paris."
    ]
    
    entropy3, info3 = calc.compute_uncertainty(similar_responses)
    print(f"Similar responses - LN Entropy: {entropy3:.4f}, Clusters: {info3['num_clusters']}")
    
    # Verify expected behavior
    assert entropy1 <= entropy3 <= entropy2, "Entropy ordering should be: identical <= similar <= different"
    assert info1['num_clusters'] <= info3['num_clusters'] <= info2['num_clusters'], "Cluster count should follow same pattern"
    
    print("✓ All tests passed!")

def test_edge_cases():
    """Test edge cases"""
    print("\nTesting edge cases...")
    
    calc = LNEntropyCalculator(
        embedding_model='paraphrase-multilingual-MiniLM-L12-v2',
        device='cpu'
    )
    
    # Single response
    single_response = ["Paris is the capital of France."]
    entropy_single, info_single = calc.compute_uncertainty(single_response)
    print(f"Single response - LN Entropy: {entropy_single:.4f}, Clusters: {info_single['num_clusters']}")
    assert entropy_single == 0.0, "Single response should have zero entropy"
    
    # Empty responses
    empty_responses = []
    entropy_empty, info_empty = calc.compute_uncertainty(empty_responses)
    print(f"Empty responses - LN Entropy: {entropy_empty:.4f}, Clusters: {info_empty['num_clusters']}")
    assert entropy_empty == 0.0, "Empty responses should have zero entropy"
    
    print("✓ Edge case tests passed!")

if __name__ == "__main__":
    test_ln_entropy_basic()
    test_edge_cases()
    print("\n🎉 All LN Entropy tests completed successfully!")
