"""
GPU utility functions for accelerated computation.
"""

import torch
import numpy as np
from typing import Union, Optional, List, Tuple, Dict
import logging

logger = logging.getLogger(__name__)


class GPUAccelerator:
    """
    GPU acceleration utilities for embedding analysis.
    """
    
    def __init__(self):
        """Initialize GPU accelerator."""
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA-capable GPU is required")
            
        self.device = torch.device('cuda')
        torch.cuda.empty_cache()
        
    def to_gpu(self, data: Union[np.ndarray, List]) -> torch.Tensor:
        """Convert numpy array or list to GPU tensor."""
        if isinstance(data, torch.Tensor):
            return data.to(self.device)
        return torch.tensor(data, device=self.device, dtype=torch.float32)
        
    def to_cpu(self, tensor: torch.Tensor) -> np.ndarray:
        """Convert GPU tensor to numpy array."""
        return tensor.cpu().numpy()
        
    def gpu_mean(self, data: Union[np.ndarray, torch.Tensor], axis: Optional[int] = None) -> Union[float, np.ndarray]:
        """GPU-accelerated mean calculation."""
        tensor = self.to_gpu(data)
        if axis is None:
            return tensor.mean().item()
        else:
            return self.to_cpu(tensor.mean(dim=axis))
            
    def gpu_std(self, data: Union[np.ndarray, torch.Tensor], axis: Optional[int] = None) -> Union[float, np.ndarray]:
        """GPU-accelerated standard deviation calculation."""
        tensor = self.to_gpu(data)
        if axis is None:
            return tensor.std().item()
        else:
            return self.to_cpu(tensor.std(dim=axis))
            
    def gpu_percentile(self, data: Union[np.ndarray, torch.Tensor], percentiles: Union[float, List[float]]) -> Union[float, np.ndarray]:
        """GPU-accelerated percentile calculation."""
        tensor = self.to_gpu(data).flatten()
        
        if isinstance(percentiles, (list, tuple, np.ndarray)):
            results = []
            for p in percentiles:
                k = int(round(p / 100.0 * len(tensor)))
                k = max(1, min(len(tensor), k))
                results.append(torch.kthvalue(tensor, k).values.item())
            return np.array(results)
        else:
            k = int(round(percentiles / 100.0 * len(tensor)))
            k = max(1, min(len(tensor), k))
            return torch.kthvalue(tensor, k).values.item()
            
    def gpu_norm(self, data: Union[np.ndarray, torch.Tensor]) -> float:
        """GPU-accelerated norm calculation."""
        tensor = self.to_gpu(data)
        return torch.norm(tensor).item()
        
    def gpu_pairwise_distances(self, embeddings: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """GPU-accelerated pairwise distance calculation."""
        embeddings_tensor = self.to_gpu(embeddings)
        
        # Efficient pairwise distance calculation
        distances = torch.cdist(embeddings_tensor, embeddings_tensor, p=2)
        
        return self.to_cpu(distances)
        
    def gpu_cosine_similarity(self, embeddings1: Union[np.ndarray, torch.Tensor], 
                            embeddings2: Optional[Union[np.ndarray, torch.Tensor]] = None) -> np.ndarray:
        """GPU-accelerated pairwise cosine similarity."""
        emb1_tensor = self.to_gpu(embeddings1)
        
        if embeddings2 is None:
            emb2_tensor = emb1_tensor
        else:
            emb2_tensor = self.to_gpu(embeddings2)
            
        # Normalize embeddings
        emb1_norm = torch.nn.functional.normalize(emb1_tensor, p=2, dim=1)
        emb2_norm = torch.nn.functional.normalize(emb2_tensor, p=2, dim=1)
        
        # Compute cosine similarity
        cosine_sim = torch.mm(emb1_norm, emb2_norm.t())
        
        return self.to_cpu(cosine_sim)
        
    def gpu_batch_velocity_calculation(self, embeddings: Union[np.ndarray, torch.Tensor]) -> Dict[str, np.ndarray]:
        """GPU-accelerated batch velocity calculation."""
        X = self.to_gpu(embeddings)
        
        # Calculate differences between consecutive embeddings
        diffs = X[1:] - X[:-1]
        
        # Calculate velocities (norms of differences)
        velocities = torch.norm(diffs, dim=1)
        
        # Calculate accelerations
        if len(velocities) > 1:
            accel_diffs = velocities[1:] - velocities[:-1]
            accelerations = accel_diffs
        else:
            accelerations = torch.tensor([], device=self.device)
            
        return {
            'velocities': self.to_cpu(velocities),
            'accelerations': self.to_cpu(accelerations),
            'mean_velocity': velocities.mean().item(),
            'std_velocity': velocities.std().item() if len(velocities) > 1 else 0
        }
        
    def gpu_pca(self, data: Union[np.ndarray, torch.Tensor], n_components: int = 3) -> Dict[str, np.ndarray]:
        """GPU-accelerated PCA using PyTorch."""
        X = self.to_gpu(data)
        
        # Center the data
        X_mean = X.mean(dim=0)
        X_centered = X - X_mean
        
        # Compute covariance matrix
        n_samples = X.shape[0]
        cov_matrix = torch.mm(X_centered.t(), X_centered) / (n_samples - 1)
        
        # Compute eigenvalues and eigenvectors
        eigenvalues, eigenvectors = torch.linalg.eigh(cov_matrix)
        
        # Sort by eigenvalues (descending)
        idx = eigenvalues.argsort(descending=True)
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # Select top components
        components = eigenvectors[:, :n_components]
        eigenvalues = eigenvalues[:n_components]
        
        # Transform data
        X_transformed = torch.mm(X_centered, components)
        
        # Calculate explained variance ratio
        total_variance = eigenvalues.sum()
        explained_variance_ratio = eigenvalues / total_variance
        
        return {
            'transformed': self.to_cpu(X_transformed),
            'components': self.to_cpu(components),
            'explained_variance_ratio': self.to_cpu(explained_variance_ratio),
            'mean': self.to_cpu(X_mean)
        }
        
    def clear_cache(self):
        """Clear GPU memory cache."""
        torch.cuda.empty_cache()