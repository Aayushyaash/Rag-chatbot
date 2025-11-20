"""
Unit tests for embedding functionality.
"""
import pytest
import numpy as np
from backend.embedder import Embedder


def test_embedder_initialization():
    """Test that embedder initializes without errors."""
    embedder = Embedder()
    assert embedder is not None


def test_embedder_encode_single():
    """Test encoding single text."""
    embedder = Embedder()
    texts = ["This is a test sentence."]
    
    embeddings = embedder.encode(texts)
    
    assert embeddings.shape[0] == 1
    assert embeddings.shape[1] > 0  # Has embedding dimension
    
    # Check normalization (L2 norm should be ~1)
    norm = np.linalg.norm(embeddings[0])
    assert 0.99 < norm < 1.01


def test_embedder_encode_multiple():
    """Test encoding multiple texts."""
    embedder = Embedder()
    texts = ["First sentence.", "Second sentence.", "Third sentence."]
    
    embeddings = embedder.encode(texts)
    
    assert embeddings.shape[0] == 3
    assert embeddings.shape[1] > 0
    
    # Check all are normalized
    for i in range(3):
        norm = np.linalg.norm(embeddings[i])
        assert 0.99 < norm < 1.01


def test_embedder_consistency():
    """Test that same text produces same embedding."""
    embedder = Embedder()
    text = "Consistency test sentence."
    
    emb1 = embedder.encode([text])
    emb2 = embedder.encode([text])
    
    # Should be identical
    assert np.allclose(emb1, emb2)


def test_embedder_empty_list():
    """Test encoding empty list."""
    embedder = Embedder()
    embeddings = embedder.encode([])
    assert len(embeddings) == 0
