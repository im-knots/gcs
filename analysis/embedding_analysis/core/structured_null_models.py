"""
Structured null models that preserve certain properties while destroying others.

These null models help distinguish between true geometric invariance and
artifacts of shared linguistic structure.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from scipy.stats import multivariate_normal
import logging

logger = logging.getLogger(__name__)


class StructuredNullModels:
    """
    Generate null models that preserve specific aspects of conversation structure.
    
    This addresses the criticism that simple random nulls are too weak.
    """
    
    def __init__(self):
        """Initialize null model generator."""
        pass
    
    def generate_topic_preserving_null(self,
                                     embeddings: np.ndarray,
                                     n_topics: int = 5,
                                     n_samples: int = 1) -> List[np.ndarray]:
        """
        Generate null that preserves topic structure but destroys trajectory.
        
        Args:
            embeddings: Original conversation embeddings
            n_topics: Number of topics to identify
            n_samples: Number of null samples to generate
            
        Returns:
            List of null embedding sequences
        """
        null_samples = []
        
        # Identify topics using clustering
        kmeans = KMeans(n_clusters=n_topics, random_state=42)
        topic_labels = kmeans.fit_predict(embeddings)
        topic_centers = kmeans.cluster_centers_
        
        # Get topic sequence
        topic_sequence = topic_labels
        
        for _ in range(n_samples):
            null_embeddings = []
            
            for topic_idx in topic_sequence:
                # Sample from topic distribution
                center = topic_centers[topic_idx]
                
                # Estimate covariance from points in this topic
                topic_points = embeddings[topic_labels == topic_idx]
                if len(topic_points) > 1:
                    cov = np.cov(topic_points.T)
                    # Regularize for stability
                    cov += 1e-4 * np.eye(len(center))
                else:
                    # Use identity covariance if only one point
                    cov = 0.1 * np.eye(len(center))
                
                # Sample new point from topic distribution
                try:
                    new_point = multivariate_normal.rvs(mean=center, cov=cov)
                except:
                    # Fallback to simple noise
                    new_point = center + np.random.randn(len(center)) * 0.1
                
                null_embeddings.append(new_point)
            
            null_samples.append(np.array(null_embeddings))
        
        return null_samples
    
    def generate_syntax_preserving_null(self,
                                      embeddings: np.ndarray,
                                      window_size: int = 10,
                                      n_samples: int = 1) -> List[np.ndarray]:
        """
        Preserve local syntactic patterns while destroying global semantics.
        
        Args:
            embeddings: Original embeddings
            window_size: Size of local windows to preserve
            n_samples: Number of samples
            
        Returns:
            Null embedding sequences
        """
        null_samples = []
        n_messages = len(embeddings)
        
        for _ in range(n_samples):
            # Segment conversation into windows
            n_windows = (n_messages + window_size - 1) // window_size
            windows = []
            
            for i in range(n_windows):
                start = i * window_size
                end = min((i + 1) * window_size, n_messages)
                windows.append(embeddings[start:end].copy())
            
            # Shuffle windows to destroy global structure
            np.random.shuffle(windows)
            
            # Apply random rotation to each window
            rotated_windows = []
            for window in windows:
                # Random rotation matrix
                rotation = self._random_rotation_matrix(embeddings.shape[1])
                rotated = window @ rotation.T
                rotated_windows.append(rotated)
            
            # Concatenate
            null_embedding = np.vstack(rotated_windows)[:n_messages]
            null_samples.append(null_embedding)
        
        return null_samples
    
    def generate_markov_null(self,
                           embeddings: np.ndarray,
                           order: int = 2,
                           n_samples: int = 1) -> List[np.ndarray]:
        """
        Generate null using Markov model of specified order.
        
        Preserves local transition statistics up to specified order.
        """
        null_samples = []
        n_messages = len(embeddings)
        
        if order >= n_messages - 1:
            logger.warning(f"Markov order {order} too high for {n_messages} messages")
            order = max(1, n_messages // 2)
        
        # Build transition model
        transitions = {}
        
        for i in range(order, n_messages):
            # Get context (previous 'order' embeddings)
            context = tuple(range(i-order, i))
            next_idx = i
            
            if context not in transitions:
                transitions[context] = []
            transitions[context].append(next_idx)
        
        for _ in range(n_samples):
            null_sequence = []
            
            # Start with first 'order' messages
            null_sequence.extend(embeddings[:order].copy())
            
            # Generate rest using Markov model
            while len(null_sequence) < n_messages:
                # Get current context
                if len(null_sequence) >= order:
                    # Use similarity to find best matching context
                    current_embeddings = null_sequence[-order:]
                    
                    best_context = None
                    best_similarity = -np.inf
                    
                    for context_indices in transitions:
                        context_embeddings = [embeddings[idx] for idx in context_indices]
                        
                        # Compute similarity
                        similarity = 0
                        for i in range(order):
                            cos_sim = np.dot(current_embeddings[i], context_embeddings[i]) / (
                                np.linalg.norm(current_embeddings[i]) * 
                                np.linalg.norm(context_embeddings[i]) + 1e-8
                            )
                            similarity += cos_sim
                        
                        if similarity > best_similarity:
                            best_similarity = similarity
                            best_context = context_indices
                    
                    if best_context and best_context in transitions:
                        # Sample next index
                        possible_next = transitions[best_context]
                        next_idx = np.random.choice(possible_next)
                        
                        # Add noise to break exact repetition
                        next_embedding = embeddings[next_idx] + np.random.randn(embeddings.shape[1]) * 0.05
                        null_sequence.append(next_embedding)
                    else:
                        # Random fallback
                        null_sequence.append(embeddings[np.random.randint(n_messages)])
                else:
                    # Not enough context yet
                    null_sequence.append(embeddings[len(null_sequence)])
            
            null_samples.append(np.array(null_sequence[:n_messages]))
        
        return null_samples
    
    def generate_semantic_drift_null(self,
                                   embeddings: np.ndarray,
                                   drift_rate: float = 0.1,
                                   n_samples: int = 1) -> List[np.ndarray]:
        """
        Preserve local coherence but add systematic semantic drift.
        
        This tests whether invariance is due to bounded semantic space.
        """
        null_samples = []
        
        for _ in range(n_samples):
            null_sequence = []
            
            # Random drift direction
            drift_direction = np.random.randn(embeddings.shape[1])
            drift_direction /= np.linalg.norm(drift_direction)
            
            cumulative_drift = 0
            
            for i, embedding in enumerate(embeddings):
                # Add systematic drift
                drifted = embedding + cumulative_drift * drift_direction
                
                # Add small random noise
                noise = np.random.randn(embeddings.shape[1]) * 0.02
                drifted += noise
                
                null_sequence.append(drifted)
                
                # Increase drift
                cumulative_drift += drift_rate
            
            null_samples.append(np.array(null_sequence))
        
        return null_samples
    
    def generate_permutation_block_null(self,
                                      embeddings: np.ndarray,
                                      block_size: int = 20,
                                      n_samples: int = 1) -> List[np.ndarray]:
        """
        Permute blocks of messages to preserve local structure.
        
        Tests importance of global ordering vs local coherence.
        """
        null_samples = []
        n_messages = len(embeddings)
        
        for _ in range(n_samples):
            # Divide into blocks
            blocks = []
            for i in range(0, n_messages, block_size):
                end = min(i + block_size, n_messages)
                blocks.append(embeddings[i:end].copy())
            
            # Permute blocks
            permuted_blocks = blocks.copy()
            np.random.shuffle(permuted_blocks)
            
            # Concatenate
            null_embedding = np.vstack(permuted_blocks)[:n_messages]
            null_samples.append(null_embedding)
        
        return null_samples
    
    def generate_phase_preserving_null(self,
                                     embeddings: np.ndarray,
                                     phases: List[Dict],
                                     n_samples: int = 1) -> List[np.ndarray]:
        """
        Preserve phase structure but randomize within phases.
        
        Args:
            embeddings: Original embeddings
            phases: List of phase dictionaries with 'start' and 'end' keys
            n_samples: Number of samples
            
        Returns:
            Null samples preserving phase boundaries
        """
        null_samples = []
        
        if not phases:
            # No phases, fall back to random permutation
            return [embeddings[np.random.permutation(len(embeddings))] 
                   for _ in range(n_samples)]
        
        for _ in range(n_samples):
            null_sequence = embeddings.copy()
            
            # Process each phase
            for phase in phases:
                start = phase.get('start', phase.get('start_turn', 0))
                end = phase.get('end', phase.get('end_turn', len(embeddings)))
                
                if start < end and end <= len(embeddings):
                    # Permute within phase
                    phase_indices = np.arange(start, end)
                    permuted_indices = np.random.permutation(phase_indices)
                    null_sequence[phase_indices] = null_sequence[permuted_indices]
            
            null_samples.append(null_sequence)
        
        return null_samples
    
    def _random_rotation_matrix(self, dim: int) -> np.ndarray:
        """Generate random rotation matrix using QR decomposition."""
        # Generate random matrix
        A = np.random.randn(dim, dim)
        
        # QR decomposition
        Q, R = np.linalg.qr(A)
        
        # Ensure determinant is 1 (proper rotation)
        Q = Q @ np.diag(np.sign(np.diag(R)))
        
        return Q
    
    def compare_null_models(self,
                          original: np.ndarray,
                          null_models: Dict[str, List[np.ndarray]],
                          metric_func: callable) -> Dict[str, Dict]:
        """
        Compare different null models using specified metric.
        
        Args:
            original: Original embedding sequence
            null_models: Dict mapping null type to list of null samples
            metric_func: Function to compute metric of interest
            
        Returns:
            Comparison results
        """
        results = {}
        
        # Compute metric for original
        original_metric = metric_func(original)
        
        for null_type, null_samples in null_models.items():
            null_metrics = [metric_func(null) for null in null_samples]
            
            # Statistical comparison
            null_mean = np.mean(null_metrics)
            null_std = np.std(null_metrics)
            
            # Z-score
            if null_std > 0:
                z_score = (original_metric - null_mean) / null_std
            else:
                z_score = np.inf if original_metric != null_mean else 0
            
            # Percentile
            percentile = np.mean([original_metric > nm for nm in null_metrics]) * 100
            
            results[null_type] = {
                'original_metric': original_metric,
                'null_mean': null_mean,
                'null_std': null_std,
                'z_score': z_score,
                'percentile': percentile,
                'significant': abs(z_score) > 2
            }
        
        return results


class NullModelValidator:
    """
    Validate that null models preserve intended properties.
    """
    
    def __init__(self):
        """Initialize validator."""
        pass
    
    def validate_topic_preservation(self,
                                  original: np.ndarray,
                                  null: np.ndarray,
                                  n_topics: int = 5) -> Dict[str, float]:
        """Check if topic structure is preserved."""
        # Cluster both sequences
        kmeans_orig = KMeans(n_clusters=n_topics, random_state=42)
        kmeans_null = KMeans(n_clusters=n_topics, random_state=42)
        
        labels_orig = kmeans_orig.fit_predict(original)
        labels_null = kmeans_null.fit_predict(null)
        
        # Compare topic distributions
        topic_dist_orig = np.bincount(labels_orig, minlength=n_topics) / len(labels_orig)
        topic_dist_null = np.bincount(labels_null, minlength=n_topics) / len(labels_null)
        
        # KL divergence between distributions
        from scipy.stats import entropy
        kl_div = entropy(topic_dist_orig + 1e-8, topic_dist_null + 1e-8)
        
        # Topic sequence similarity (normalized mutual information)
        from sklearn.metrics import normalized_mutual_info_score
        nmi = normalized_mutual_info_score(labels_orig, labels_null)
        
        return {
            'kl_divergence': kl_div,
            'normalized_mutual_info': nmi,
            'topic_preserved': kl_div < 0.5  # Threshold
        }
    
    def validate_local_structure(self,
                               original: np.ndarray,
                               null: np.ndarray,
                               window_size: int = 5) -> Dict[str, float]:
        """Check if local structure is preserved."""
        # Compute local statistics
        local_corrs_orig = []
        local_corrs_null = []
        
        for i in range(len(original) - window_size):
            # Original window
            window_orig = original[i:i+window_size]
            mean_orig = np.mean(window_orig, axis=0)
            
            # Null window
            window_null = null[i:i+window_size]
            mean_null = np.mean(window_null, axis=0)
            
            # Correlation between mean vectors
            corr = np.corrcoef(mean_orig, mean_null)[0, 1]
            if not np.isnan(corr):
                local_corrs_orig.append(corr)
        
        return {
            'mean_local_correlation': np.mean(local_corrs_orig) if local_corrs_orig else 0,
            'std_local_correlation': np.std(local_corrs_orig) if local_corrs_orig else 0,
            'local_preserved': np.mean(local_corrs_orig) > 0.7 if local_corrs_orig else False
        }