"""
Ensemble embedding functionality for conversation analysis.
"""

import os
# Set tokenizers parallelism to avoid warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import torch
from typing import Dict, List, Optional, Tuple
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import logging
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class EnsembleEmbedder:
    """
    Manages ensemble of embedding models to capture "shadows" of higher-dimensional structure.
    """
    
    DEFAULT_ENSEMBLE = [
        {'name': 'MiniLM-L6', 'model_id': 'all-MiniLM-L6-v2', 'dim': 384},
        {'name': 'MPNet', 'model_id': 'all-mpnet-base-v2', 'dim': 768},
        {'name': 'MiniLM-L12', 'model_id': 'all-MiniLM-L12-v2', 'dim': 384},
        {'name': 'DistilRoBERTa', 'model_id': 'all-distilroberta-v1', 'dim': 768},
        {'name': 'E5-small', 'model_id': 'intfloat/e5-small-v2', 'dim': 384},
    ]
    
    def __init__(self, 
                 ensemble_config: Optional[List[Dict]] = None,
                 device: Optional[str] = None,
                 cache_dir: Optional[str] = None):
        """
        Initialize ensemble embedder.
        
        Args:
            ensemble_config: List of model configurations
            device: Device to use ('cuda' or 'cpu')
            cache_dir: Directory to cache models
        """
        # GPU is required
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA-capable GPU is required but not found. Please ensure you have a GPU and CUDA installed.")
            
        self.device = 'cuda'
        self.cache_dir = cache_dir
        self.ensemble_config = ensemble_config or self.DEFAULT_ENSEMBLE
        self.models = {}
        
        # GPU configuration
        logger.info(f"✓ GPU detected! Using {torch.cuda.get_device_name(0)} for acceleration")
        self.gpu_batch_size = 256  # RTX 4070 Super can handle large batches
        torch.cuda.empty_cache()  # Clear GPU cache
            
        self._load_models()
        
    def _load_models(self):
        """Load all models in the ensemble."""
        logger.info(f"Loading {len(self.ensemble_config)} models for ensemble...")
        
        for config in self.ensemble_config:
            model_name = config['name']
            model_id = config['model_id']
            
            try:
                logger.info(f"Loading {model_name} ({model_id})...")
                model = SentenceTransformer(
                    model_id, 
                    device=self.device,
                    cache_folder=self.cache_dir
                )
                self.models[model_name] = {
                    'model': model,
                    'dim': config['dim'],
                    'id': model_id
                }
                logger.info(f"✓ Loaded {model_name} with {config['dim']} dimensions")
                
            except Exception as e:
                logger.error(f"Failed to load {model_name}: {e}")
                raise
                
    def embed_texts(self, 
                    texts: List[str], 
                    batch_size: Optional[int] = None,
                    show_progress: bool = True) -> Dict[str, np.ndarray]:
        """
        Embed texts with all models in the ensemble.
        
        Args:
            texts: List of text strings to embed
            batch_size: Batch size for encoding (uses GPU optimized default if None)
            show_progress: Whether to show progress bar
            
        Returns:
            Dictionary mapping model names to embedding arrays
        """
        if batch_size is None:
            batch_size = self.gpu_batch_size
            
        embeddings = {}
        
        # Use tqdm for progress tracking
        model_iterator = tqdm(self.models.items(), desc="Embedding with models", disable=not show_progress)
        
        for model_name, model_info in model_iterator:
            model = model_info['model']
            
            if show_progress:
                model_iterator.set_description(f"Embedding with {model_name}")
            
            # Encode texts with GPU optimization
            model_embeddings = model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=False,  # We use our own progress bar
                convert_to_numpy=True,
                device=self.device
            )
            
            embeddings[model_name] = model_embeddings
            
        # Clear GPU cache after batch processing
        torch.cuda.empty_cache()
            
        return embeddings
    
    def embed_conversation(self, 
                          messages: List[Dict],
                          text_key: str = 'content') -> Dict[str, np.ndarray]:
        """
        Embed a conversation's messages.
        
        Args:
            messages: List of message dictionaries
            text_key: Key in message dict containing text
            
        Returns:
            Dictionary mapping model names to embedding arrays
        """
        texts = [msg.get(text_key, '') for msg in messages]
        return self.embed_texts(texts)
    
    def calculate_ensemble_agreement(self, 
                                   embeddings: Dict[str, np.ndarray]) -> Dict[str, float]:
        """
        Calculate agreement metrics between models in ensemble.
        
        Args:
            embeddings: Dictionary of embeddings from each model
            
        Returns:
            Agreement metrics including pairwise correlations
        """
        model_names = list(embeddings.keys())
        
        if len(model_names) < 2:
            return {'mean_correlation': 1.0, 'min_correlation': 1.0}
            
        correlations = []
        
        # Calculate pairwise correlations
        for i, model1 in enumerate(model_names):
            for model2 in model_names[i+1:]:
                emb1 = embeddings[model1]
                emb2 = embeddings[model2]
                
                # Calculate trajectory similarity
                if len(emb1) > 1:
                    # Use trajectory differences as similarity measure
                    diff1 = np.diff(emb1, axis=0)
                    diff2 = np.diff(emb2, axis=0)
                    
                    # Normalize to unit vectors
                    norm1 = np.linalg.norm(diff1, axis=1, keepdims=True)
                    norm2 = np.linalg.norm(diff2, axis=1, keepdims=True)
                    
                    diff1_norm = diff1 / (norm1 + 1e-8)
                    diff2_norm = diff2 / (norm2 + 1e-8)
                    
                    # Calculate cosine similarities
                    similarities = np.sum(diff1_norm * diff2_norm, axis=1)
                    corr = np.mean(similarities)
                    correlations.append(corr)
                    
        return {
            'mean_correlation': np.mean(correlations) if correlations else 0,
            'min_correlation': np.min(correlations) if correlations else 0,
            'std_correlation': np.std(correlations) if correlations else 0,
            'pairwise_correlations': correlations
        }
    
    def get_model_info(self) -> Dict[str, Dict]:
        """Get information about loaded models."""
        return {
            name: {
                'model_id': info['id'],
                'dimension': info['dim'],
                'device': str(self.device)
            }
            for name, info in self.models.items()
        }