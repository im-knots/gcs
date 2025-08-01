"""
Trajectory analysis functionality for conversation embeddings.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from scipy.spatial.distance import euclidean, cosine
import logging
from ..utils import GPUAccelerator

logger = logging.getLogger(__name__)


class TrajectoryAnalyzer:
    """
    Analyzes trajectories through embedding space.
    """
    
    def __init__(self):
        """Initialize trajectory analyzer."""
        self.gpu = GPUAccelerator()
        
    def calculate_trajectory_metrics(self, 
                                   embeddings: np.ndarray,
                                   timestamps: Optional[np.ndarray] = None) -> Dict:
        """
        Calculate comprehensive trajectory metrics.
        
        Args:
            embeddings: Array of embeddings (n_messages, embedding_dim)
            timestamps: Optional array of timestamps
            
        Returns:
            Dictionary of trajectory metrics
        """
        n_messages = len(embeddings)
        
        if n_messages < 2:
            return self._empty_metrics()
            
        metrics = {}
        
        # Calculate velocities using GPU
        velocity_results = self.gpu.gpu_batch_velocity_calculation(embeddings)
        velocities = velocity_results['velocities']
        metrics['velocities'] = velocities
        metrics['velocity_mean'] = velocity_results['mean_velocity']
        metrics['velocity_std'] = velocity_results['std_velocity']
        metrics['velocity_max'] = np.max(velocities) if len(velocities) > 0 else 0
        
        # Calculate accelerations
        accelerations = velocity_results['accelerations']
        if len(accelerations) > 0:
            metrics['acceleration_mean'] = self.gpu.gpu_mean(np.abs(accelerations))
            metrics['acceleration_max'] = np.max(np.abs(accelerations))
        else:
            metrics['acceleration_mean'] = 0
            metrics['acceleration_max'] = 0
            
        # Calculate curvature
        curvatures = self._calculate_curvatures(embeddings)
        metrics['curvature_mean'] = np.mean(curvatures) if curvatures else 0
        metrics['curvature_max'] = np.max(curvatures) if curvatures else 0
        metrics['curvature_std'] = np.std(curvatures) if curvatures else 0
        
        # Calculate trajectory length
        metrics['total_distance'] = np.sum(velocities)
        
        # Calculate displacement
        metrics['displacement'] = euclidean(embeddings[0], embeddings[-1])
        metrics['efficiency'] = metrics['displacement'] / metrics['total_distance'] if metrics['total_distance'] > 0 else 0
        
        # Angular metrics
        angular_changes = self._calculate_angular_changes(embeddings)
        metrics['angular_change_mean'] = np.mean(angular_changes) if angular_changes else 0
        metrics['angular_change_total'] = np.sum(angular_changes) if angular_changes else 0
        
        return metrics
        
    def _calculate_velocities(self, 
                            embeddings: np.ndarray,
                            timestamps: Optional[np.ndarray] = None) -> np.ndarray:
        """Calculate velocities between consecutive embeddings."""
        velocities = []
        
        for i in range(1, len(embeddings)):
            distance = euclidean(embeddings[i], embeddings[i-1])
            
            if timestamps is not None:
                dt = timestamps[i] - timestamps[i-1]
                velocity = distance / dt if dt > 0 else distance
            else:
                velocity = distance
                
            velocities.append(velocity)
            
        return np.array(velocities)
        
    def _calculate_curvatures(self, embeddings: np.ndarray) -> List[float]:
        """Calculate curvature at each point."""
        curvatures = []
        
        for i in range(1, len(embeddings) - 1):
            curvature = self._point_curvature(
                embeddings[i-1],
                embeddings[i],
                embeddings[i+1]
            )
            curvatures.append(curvature)
            
        return curvatures
        
    def _point_curvature(self, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
        """Calculate curvature at p2 given three consecutive points."""
        v1 = p2 - p1
        v2 = p3 - p2
        
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)
        
        if norm_v1 < 1e-8 or norm_v2 < 1e-8:
            return 0
            
        # Angle between vectors
        cos_angle = np.dot(v1, v2) / (norm_v1 * norm_v2)
        angle = np.arccos(np.clip(cos_angle, -1, 1))
        
        # Curvature approximation
        avg_length = (norm_v1 + norm_v2) / 2
        curvature = angle / avg_length if avg_length > 0 else 0
        
        return curvature
        
    def _calculate_angular_changes(self, embeddings: np.ndarray) -> List[float]:
        """Calculate angular changes along trajectory."""
        angular_changes = []
        
        for i in range(1, len(embeddings) - 1):
            v1 = embeddings[i] - embeddings[i-1]
            v2 = embeddings[i+1] - embeddings[i]
            
            norm_v1 = np.linalg.norm(v1)
            norm_v2 = np.linalg.norm(v2)
            
            if norm_v1 > 1e-8 and norm_v2 > 1e-8:
                cos_sim = np.dot(v1, v2) / (norm_v1 * norm_v2)
                angle = np.arccos(np.clip(cos_sim, -1, 1))
                angular_changes.append(angle)
                
        return angular_changes
        
    def _empty_metrics(self) -> Dict:
        """Return empty metrics dictionary."""
        return {
            'velocities': np.array([]),
            'velocity_mean': 0,
            'velocity_std': 0,
            'velocity_max': 0,
            'acceleration_mean': 0,
            'acceleration_max': 0,
            'curvature_mean': 0,
            'curvature_max': 0,
            'curvature_std': 0,
            'total_distance': 0,
            'displacement': 0,
            'efficiency': 0,
            'angular_change_mean': 0,
            'angular_change_total': 0
        }
        
    def calculate_ensemble_trajectories(self, 
                                      ensemble_embeddings: Dict[str, np.ndarray]) -> Dict:
        """
        Calculate trajectory metrics for each model in ensemble.
        
        Args:
            ensemble_embeddings: Dict mapping model names to embeddings
            
        Returns:
            Dict mapping model names to trajectory metrics
        """
        ensemble_metrics = {}
        
        for model_name, embeddings in ensemble_embeddings.items():
            metrics = self.calculate_trajectory_metrics(embeddings)
            ensemble_metrics[model_name] = metrics
            
        # Add cross-model consistency metrics
        ensemble_metrics['consistency'] = self._calculate_trajectory_consistency(ensemble_metrics)
        
        return ensemble_metrics
        
    def _calculate_trajectory_consistency(self, ensemble_metrics: Dict) -> Dict:
        """Calculate consistency metrics across ensemble."""
        model_names = [name for name in ensemble_metrics.keys() if name != 'consistency']
        
        if len(model_names) < 2:
            return {'velocity_correlation': 1.0, 'curvature_correlation': 1.0}
            
        # Extract velocity sequences
        velocity_sequences = []
        for model in model_names:
            if 'velocities' in ensemble_metrics[model]:
                velocity_sequences.append(ensemble_metrics[model]['velocities'])
                
        # Calculate correlations
        consistency = {}
        
        if velocity_sequences and all(len(v) > 0 for v in velocity_sequences):
            # Align sequences to same length
            min_len = min(len(v) for v in velocity_sequences)
            aligned_velocities = [v[:min_len] for v in velocity_sequences]
            
            # Pairwise correlations
            correlations = []
            for i in range(len(aligned_velocities)):
                for j in range(i+1, len(aligned_velocities)):
                    corr = np.corrcoef(aligned_velocities[i], aligned_velocities[j])[0, 1]
                    if not np.isnan(corr):
                        correlations.append(corr)
                        
            consistency['velocity_correlation'] = np.mean(correlations) if correlations else 0
            consistency['velocity_correlation_std'] = np.std(correlations) if correlations else 0
            
        return consistency
        
    def detect_trajectory_anomalies(self, 
                                  embeddings: np.ndarray,
                                  threshold_percentile: float = 95) -> List[int]:
        """
        Detect anomalous points in trajectory.
        
        Args:
            embeddings: Array of embeddings
            threshold_percentile: Percentile for anomaly threshold
            
        Returns:
            List of anomalous turn indices
        """
        if len(embeddings) < 3:
            return []
            
        # Calculate local deviation scores
        deviation_scores = []
        
        for i in range(1, len(embeddings) - 1):
            # Expected position (linear interpolation)
            expected = (embeddings[i-1] + embeddings[i+1]) / 2
            actual = embeddings[i]
            
            deviation = euclidean(expected, actual)
            deviation_scores.append((i, deviation))
            
        # Find anomalies
        if deviation_scores:
            scores = [s[1] for s in deviation_scores]
            threshold = np.percentile(scores, threshold_percentile)
            
            anomalies = [idx for idx, score in deviation_scores if score > threshold]
            return anomalies
            
        return []