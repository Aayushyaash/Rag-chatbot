"""
Embedding layer using nomic-embed-text-v1 via sentence-transformers.
Provides lazy loading, CPU/GPU detection, and normalized vector output.
"""
import os
import logging
from typing import List, Optional
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class Embedder:
    """Wrapper for nomic-embed-text-v1 embedding model."""
    
    MODEL_NAME = "nomic-ai/nomic-embed-text-v1"
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize embedder with lazy loading.
        
        Args:
            model_path: Optional custom path for model files. If None, uses local models/ directory.
        """
        if model_path is None:
            # Default to ../models/nomic-embed-text-v1 relative to this file
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            self.model_path = os.path.join(base_dir, "models", "nomic-embed-text-v1")
        else:
            self.model_path = model_path

        self._model: Optional[SentenceTransformer] = None
        self._embedding_dim: Optional[int] = None
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Embedder initialized. Device: {self._device}")
    
    def _load_model(self):
        """Load the model on first use."""
        if self._model is not None:
            return
        
        try:
            logger.info(f"Loading embedding model: {self.MODEL_NAME}")
            
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(
                    f"Embedding model not found at: {self.model_path}\n"
                    "Please run 'python download_model.py' to download the model first."
                )
            
            # Load from local path
            logger.info(f"Loading model from local path: {self.model_path}")
            self._model = SentenceTransformer(self.model_path, device=self._device, trust_remote_code=True)
            
            # Validate model by getting embedding dimension
            test_embedding = self._model.encode(["test"], normalize_embeddings=True)
            self._embedding_dim = test_embedding.shape[1]
            logger.info(f"Model loaded successfully. Embedding dimension: {self._embedding_dim}")
            
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise RuntimeError(
                f"Model load failed: {e}\n\n"
                "Please ensure you have run 'python download_model.py' to download the model."
            ) from e
    
    def encode(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Encode texts into normalized embeddings.
        
        Args:
            texts: List of text strings to embed
            batch_size: Batch size for encoding (default: 32)
        
        Returns:
            Numpy array of shape (len(texts), embedding_dim) with L2-normalized vectors
        """
        if not texts:
            return np.array([])
        
        # Lazy load model on first encode call
        self._load_model()
        
        try:
            embeddings = self._model.encode(
                texts,
                batch_size=batch_size,
                normalize_embeddings=True,  # L2 normalization
                show_progress_bar=len(texts) > 100,
                convert_to_numpy=True
            )
            
            logger.debug(f"Encoded {len(texts)} texts into embeddings of shape {embeddings.shape}")
            return embeddings
            
        except Exception as e:
            logger.error(f"Encoding failed: {e}")
            raise
    
    @property
    def embedding_dim(self) -> int:
        """Get embedding dimension (loads model if not already loaded)."""
        if self._embedding_dim is None:
            self._load_model()
        return self._embedding_dim
    
    @property
    def device(self) -> str:
        """Get device being used (cpu or cuda)."""
        return self._device


# Singleton instance
_embedder_instance: Optional[Embedder] = None


def get_embedder(model_path: Optional[str] = None) -> Embedder:
    """
    Get or create the singleton embedder instance.
    
    Args:
        model_path: Optional custom path for model files
    
    Returns:
        Embedder instance
    """
    global _embedder_instance
    
    if _embedder_instance is None:
        _embedder_instance = Embedder(model_path=model_path)
    
    return _embedder_instance
