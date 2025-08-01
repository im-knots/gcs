"""
Trajectory analysis functionality for conversation embeddings.

This module contains comprehensive trajectory analysis methods including:
- Basic trajectory metrics (velocity, acceleration, curvature)
- Advanced normalization techniques (adaptive, entropy-based, information-theoretic)
- Ensemble curvature calculations using multiple methods
- Cross-model consistency analysis
- Anomaly detection in trajectories

The functionality from advanced_trajectory.py has been merged into this module
to provide a unified interface for all trajectory analysis needs.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from scipy.spatial.distance import euclidean, cosine
from scipy import interpolate, stats, signal
from scipy.optimize import minimize_scalar
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
                                   timestamps: Optional[np.ndarray] = None,
                                   use_advanced: bool = False) -> Dict:
        """
        Calculate comprehensive trajectory metrics.
        
        Args:
            embeddings: Array of embeddings (n_messages, embedding_dim)
            timestamps: Optional array of timestamps
            use_advanced: Whether to use advanced normalization methods
            
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
        if use_advanced:
            # Use ensemble curvature methods
            curvature_results = self.calculate_curvature_ensemble(embeddings)
            metrics['curvature_mean'] = curvature_results['ensemble']['mean']
            metrics['curvature_max'] = max(
                curvature_results['discrete']['max'],
                curvature_results['spline']['max'],
                curvature_results['frenet']['max']
            )
            metrics['curvature_std'] = curvature_results['ensemble']['std']
            metrics['curvature_methods'] = curvature_results
        else:
            # Use simple curvature calculation
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
        
        # Add advanced normalization if requested
        if use_advanced:
            # Get normalized metrics
            normalized_metrics = self.analyze_trajectory_with_normalization(
                embeddings, method='adaptive'
            )
            metrics['normalized'] = normalized_metrics
            
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
    
    # Advanced trajectory analysis methods (merged from advanced_trajectory.py)
    
    def analyze_trajectory_with_normalization(self, 
                                            embeddings: np.ndarray,
                                            method: str = 'adaptive') -> Dict:
        """
        Analyze trajectory with sophisticated length normalization.
        
        Args:
            embeddings: Array of embeddings (n_messages, embedding_dim)
            method: Normalization method ('adaptive', 'entropy', 'information', 'none')
            
        Returns:
            Dictionary of normalized trajectory metrics
        """
        n_messages = len(embeddings)
        
        # Calculate raw metrics first
        raw_metrics = self._calculate_raw_metrics(embeddings)
        
        # Apply normalization based on method
        if method == 'adaptive':
            norm_factor = self._adaptive_normalization(embeddings, raw_metrics)
        elif method == 'entropy':
            norm_factor = self._entropy_based_normalization(embeddings)
        elif method == 'information':
            norm_factor = self._information_theoretic_normalization(embeddings)
        else:
            norm_factor = 1.0
            
        # Normalize extensive properties
        normalized_metrics = self._apply_normalization(raw_metrics, norm_factor, n_messages)
        
        # Add normalization metadata
        normalized_metrics['normalization'] = {
            'method': method,
            'factor': norm_factor,
            'n_messages': n_messages,
            'effective_length': n_messages / norm_factor
        }
        
        return normalized_metrics
        
    def _calculate_raw_metrics(self, embeddings: np.ndarray) -> Dict:
        """Calculate raw trajectory metrics without normalization."""
        metrics = {}
        
        # Calculate velocities
        velocities = []
        for i in range(1, len(embeddings)):
            v = np.linalg.norm(embeddings[i] - embeddings[i-1])
            velocities.append(v)
        velocities = np.array(velocities)
        
        # Calculate path length
        metrics['total_distance'] = np.sum(velocities)
        metrics['mean_velocity'] = np.mean(velocities) if len(velocities) > 0 else 0
        metrics['std_velocity'] = np.std(velocities) if len(velocities) > 0 else 0
        
        # Calculate accelerations
        if len(velocities) > 1:
            accelerations = np.diff(velocities)
            metrics['mean_acceleration'] = np.mean(np.abs(accelerations))
            metrics['std_acceleration'] = np.std(accelerations)
        else:
            metrics['mean_acceleration'] = 0
            metrics['std_acceleration'] = 0
            
        return metrics
        
    def _adaptive_normalization(self, embeddings: np.ndarray, raw_metrics: Dict) -> float:
        """
        Adaptive normalization based on trajectory characteristics.
        
        Instead of assuming random walk (sqrt), we estimate the actual scaling behavior.
        """
        n = len(embeddings)
        
        # Analyze trajectory compactness
        centroid = np.mean(embeddings, axis=0)
        radii = [np.linalg.norm(emb - centroid) for emb in embeddings]
        compactness = np.std(radii) / (np.mean(radii) + 1e-8)
        
        # Analyze velocity autocorrelation to detect persistence
        velocities = []
        for i in range(1, len(embeddings)):
            v = embeddings[i] - embeddings[i-1]
            velocities.append(v)
        
        if len(velocities) > 10:
            # Calculate velocity autocorrelation
            autocorr = self._velocity_autocorrelation(velocities)
            persistence_length = self._estimate_persistence_length(autocorr)
            
            # Effective independent segments
            n_effective = n / max(1, persistence_length)
            
            # Scaling exponent: 0.5 for random walk, 1.0 for ballistic
            scaling_exponent = 0.5 + 0.5 * (1 - np.exp(-persistence_length / 5))
        else:
            n_effective = n
            scaling_exponent = 0.5
            
        # Combine compactness and persistence for normalization
        norm_factor = (n / 100.0) ** scaling_exponent
        
        # Adjust for trajectory compactness
        norm_factor *= (1 + compactness)
        
        return norm_factor
        
    def _entropy_based_normalization(self, embeddings: np.ndarray) -> float:
        """
        Normalization based on trajectory entropy.
        
        High entropy (random) trajectories need different normalization than
        low entropy (structured) trajectories.
        """
        n = len(embeddings)
        
        # Calculate directional entropy
        directions = []
        for i in range(1, len(embeddings)):
            v = embeddings[i] - embeddings[i-1]
            if np.linalg.norm(v) > 1e-8:
                directions.append(v / np.linalg.norm(v))
                
        if len(directions) < 2:
            return np.sqrt(n / 100.0)
            
        # Estimate entropy using angle distribution
        angles = []
        for i in range(1, len(directions)):
            cos_angle = np.clip(np.dot(directions[i], directions[i-1]), -1, 1)
            angle = np.arccos(cos_angle)
            angles.append(angle)
            
        # Discretize angles and calculate entropy
        hist, _ = np.histogram(angles, bins=20, density=True)
        hist = hist[hist > 0]
        entropy = -np.sum(hist * np.log(hist + 1e-10))
        
        # High entropy -> more random -> sqrt scaling
        # Low entropy -> more directed -> linear scaling
        max_entropy = np.log(20)  # Maximum possible entropy with 20 bins
        randomness = entropy / max_entropy
        scaling_exponent = 0.5 + 0.5 * (1 - randomness)
        
        return (n / 100.0) ** scaling_exponent
        
    def _information_theoretic_normalization(self, embeddings: np.ndarray) -> float:
        """
        Normalization based on information content of trajectory.
        
        Uses compression ratio as a proxy for information content.
        """
        n = len(embeddings)
        
        # Convert trajectory to symbolic sequence
        # Discretize based on direction changes
        symbols = []
        for i in range(2, len(embeddings)):
            v1 = embeddings[i-1] - embeddings[i-2]
            v2 = embeddings[i] - embeddings[i-1]
            
            if np.linalg.norm(v1) > 1e-8 and np.linalg.norm(v2) > 1e-8:
                # Compute angle between consecutive steps
                cos_angle = np.clip(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)), -1, 1)
                angle = np.arccos(cos_angle)
                
                # Discretize angle into symbols
                symbol = int(angle / (np.pi / 8))  # 8 directional bins
                symbols.append(symbol)
                
        if len(symbols) < 10:
            return np.sqrt(n / 100.0)
            
        # Estimate compression ratio using run-length encoding
        compression_ratio = self._estimate_compression_ratio(symbols)
        
        # High compression -> repetitive -> different scaling
        # Low compression -> complex -> sqrt scaling
        scaling_exponent = 0.5 + 0.5 * compression_ratio
        
        return (n / 100.0) ** scaling_exponent
        
    def _velocity_autocorrelation(self, velocities: List[np.ndarray]) -> np.ndarray:
        """Calculate velocity autocorrelation function."""
        velocities = np.array(velocities)
        n = len(velocities)
        
        # Normalize velocities
        v_norms = [np.linalg.norm(v) for v in velocities]
        normalized_v = [v / (norm + 1e-8) for v, norm in zip(velocities, v_norms)]
        
        # Calculate autocorrelation
        max_lag = min(n // 2, 50)
        autocorr = []
        
        for lag in range(max_lag):
            if lag == 0:
                corr = 1.0
            else:
                valid_pairs = [(normalized_v[i], normalized_v[i + lag]) 
                              for i in range(n - lag)]
                if valid_pairs:
                    corr = np.mean([np.dot(v1, v2) for v1, v2 in valid_pairs])
                else:
                    corr = 0
            autocorr.append(corr)
            
        return np.array(autocorr)
        
    def _estimate_persistence_length(self, autocorr: np.ndarray) -> float:
        """Estimate persistence length from autocorrelation."""
        # Find where autocorrelation drops to 1/e
        threshold = 1 / np.e
        
        for i, corr in enumerate(autocorr):
            if corr < threshold:
                return float(i)
                
        return float(len(autocorr))
        
    def _estimate_compression_ratio(self, symbols: List[int]) -> float:
        """Estimate compression ratio using run-length encoding."""
        if not symbols:
            return 0.5
            
        # Simple run-length encoding
        runs = []
        current_symbol = symbols[0]
        current_length = 1
        
        for s in symbols[1:]:
            if s == current_symbol:
                current_length += 1
            else:
                runs.append(current_length)
                current_symbol = s
                current_length = 1
        runs.append(current_length)
        
        # Compression ratio estimate
        original_length = len(symbols)
        compressed_length = len(runs) * 2  # Each run needs symbol + length
        
        return compressed_length / original_length
        
    def _apply_normalization(self, raw_metrics: Dict, norm_factor: float, n_messages: int) -> Dict:
        """Apply normalization to metrics."""
        normalized = raw_metrics.copy()
        
        # Extensive properties (scale with trajectory length)
        extensive_properties = ['total_distance', 'mean_acceleration', 'std_acceleration']
        for prop in extensive_properties:
            if prop in normalized:
                normalized[f'{prop}_normalized'] = normalized[prop] / norm_factor
                
        # Intensive properties (don't scale)
        # mean_velocity, std_velocity remain unchanged
        
        return normalized
        
    def calculate_curvature_ensemble(self, embeddings: np.ndarray) -> Dict:
        """
        Calculate curvature using multiple methods for robustness.
        
        Returns:
            Dictionary with curvature estimates from different methods
        """
        results = {}
        
        # Method 1: Discrete approximation (original)
        results['discrete'] = self._calculate_discrete_curvature(embeddings)
        
        # Method 2: Spline-based smooth curvature
        results['spline'] = self._calculate_spline_curvature(embeddings)
        
        # Method 3: Frenet-Serret curvature
        results['frenet'] = self._calculate_frenet_curvature(embeddings)
        
        # Ensemble statistics
        all_curvatures = []
        for method_curvatures in results.values():
            if method_curvatures['values'] is not None:
                all_curvatures.extend(method_curvatures['values'])
                
        if all_curvatures:
            results['ensemble'] = {
                'mean': np.mean(all_curvatures),
                'std': np.std(all_curvatures),
                'median': np.median(all_curvatures),
                'agreement': self._calculate_method_agreement(results)
            }
        else:
            results['ensemble'] = {
                'mean': 0,
                'std': 0,
                'median': 0,
                'agreement': 0
            }
            
        return results
        
    def _calculate_discrete_curvature(self, embeddings: np.ndarray) -> Dict:
        """Original discrete curvature calculation."""
        curvatures = []
        
        for i in range(1, len(embeddings) - 1):
            v1 = embeddings[i] - embeddings[i-1]
            v2 = embeddings[i+1] - embeddings[i]
            
            norm_v1 = np.linalg.norm(v1)
            norm_v2 = np.linalg.norm(v2)
            
            if norm_v1 > 1e-8 and norm_v2 > 1e-8:
                # Angle between vectors
                cos_angle = np.clip(np.dot(v1, v2) / (norm_v1 * norm_v2), -1, 1)
                angle = np.arccos(cos_angle)
                
                # Curvature approximation
                avg_length = (norm_v1 + norm_v2) / 2
                curvature = angle / avg_length if avg_length > 0 else 0
                curvatures.append(curvature)
                
        return {
            'values': curvatures if curvatures else None,
            'mean': np.mean(curvatures) if curvatures else 0,
            'max': np.max(curvatures) if curvatures else 0
        }
        
    def _calculate_spline_curvature(self, embeddings: np.ndarray) -> Dict:
        """Calculate curvature using spline interpolation for smoothness."""
        n_points = len(embeddings)
        
        if n_points < 4:  # Need at least 4 points for cubic spline
            return {'values': None, 'mean': 0, 'max': 0}
            
        # Parameterize by arc length
        distances = [0]
        for i in range(1, n_points):
            d = np.linalg.norm(embeddings[i] - embeddings[i-1])
            distances.append(distances[-1] + d)
        distances = np.array(distances)
        
        # Skip if trajectory is too short
        if distances[-1] < 1e-6:
            return {'values': None, 'mean': 0, 'max': 0}
            
        # Remove duplicate distances to ensure strictly increasing sequence
        unique_indices = [0]  # Always include first point
        for i in range(1, len(distances)):
            if distances[i] > distances[unique_indices[-1]] + 1e-10:  # Small tolerance
                unique_indices.append(i)
                
        # Need at least 4 points for cubic spline
        if len(unique_indices) < 4:
            return {'values': None, 'mean': 0, 'max': 0}
            
        # Use only unique points
        unique_distances = distances[unique_indices]
        unique_embeddings = embeddings[unique_indices]
        
        # Fit splines to each dimension
        splines = []
        for dim in range(unique_embeddings.shape[1]):
            # Use cubic spline with not-a-knot boundary conditions
            spline = interpolate.CubicSpline(unique_distances, unique_embeddings[:, dim], bc_type='not-a-knot')
            splines.append(spline)
            
        # Sample curvature at regular intervals
        n_samples = max(50, n_points * 2)
        sample_points = np.linspace(unique_distances[0], unique_distances[-1], n_samples)
        
        curvatures = []
        for s in sample_points[1:-1]:  # Skip endpoints
            # First and second derivatives
            first_derivs = np.array([spl(s, 1) for spl in splines])
            second_derivs = np.array([spl(s, 2) for spl in splines])
            
            # Curvature formula: |r' × r''| / |r'|^3
            speed = np.linalg.norm(first_derivs)
            if speed > 1e-8:
                # For high dimensions, use generalized curvature
                curvature = np.linalg.norm(second_derivs - 
                                         np.dot(second_derivs, first_derivs) / speed**2 * first_derivs) / speed**2
                curvatures.append(curvature)
                
        return {
            'values': curvatures if curvatures else None,
            'mean': np.mean(curvatures) if curvatures else 0,
            'max': np.max(curvatures) if curvatures else 0
        }
        
    def _calculate_frenet_curvature(self, embeddings: np.ndarray) -> Dict:
        """Calculate curvature using Frenet-Serret formulas."""
        n_points = len(embeddings)
        
        if n_points < 3:
            return {'values': None, 'mean': 0, 'max': 0}
            
        curvatures = []
        
        # Use finite differences for derivatives
        for i in range(2, n_points - 2):
            # Use 5-point stencil for better accuracy
            # First derivative (velocity)
            v = (-embeddings[i+2] + 8*embeddings[i+1] - 8*embeddings[i-1] + embeddings[i-2]) / 12.0
            
            # Second derivative (acceleration)
            a = (-embeddings[i+2] + 16*embeddings[i+1] - 30*embeddings[i] + 
                 16*embeddings[i-1] - embeddings[i-2]) / 12.0
                 
            speed = np.linalg.norm(v)
            if speed > 1e-8:
                # Frenet curvature: |v × a| / |v|^3
                # For high dimensions: |a_perp| / |v|^2
                a_parallel = np.dot(a, v) / speed**2 * v
                a_perp = a - a_parallel
                curvature = np.linalg.norm(a_perp) / speed**2
                curvatures.append(curvature)
                
        return {
            'values': curvatures if curvatures else None,
            'mean': np.mean(curvatures) if curvatures else 0,
            'max': np.max(curvatures) if curvatures else 0
        }
        
    def _calculate_method_agreement(self, results: Dict) -> float:
        """Calculate agreement between different curvature methods."""
        methods_with_values = []
        
        for method, data in results.items():
            if method != 'ensemble' and data['values'] is not None:
                methods_with_values.append(data['mean'])
                
        if len(methods_with_values) < 2:
            return 0.0
            
        # Calculate coefficient of variation
        mean_curvature = np.mean(methods_with_values)
        if mean_curvature > 1e-8:
            cv = np.std(methods_with_values) / mean_curvature
            # Convert to agreement score (1 - cv, bounded)
            agreement = max(0, 1 - cv)
        else:
            agreement = 1.0 if np.std(methods_with_values) < 1e-8 else 0.0
            
        return agreement