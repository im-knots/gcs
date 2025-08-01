"""
Geometric invariance analysis for conversation embeddings.

This module tests the hypothesis that conversations exhibit model-invariant
geometric signatures in embedding space.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from scipy import stats
from scipy.spatial.distance import pdist, squareform, cdist
from sklearn.metrics import pairwise_distances
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)


class GeometricSignatureComputer:
    """
    Extracts geometric properties from conversation embeddings.
    
    These properties should be invariant across different embedding models
    if our hypothesis is correct.
    """
    
    def __init__(self):
        """Initialize the geometric signature computer."""
        self.metrics_computed = {}
        
    def compute_all_signatures(self, embeddings: np.ndarray, 
                             conversation_id: str) -> Dict[str, np.ndarray]:
        """
        Compute all geometric signatures for a conversation.
        
        Args:
            embeddings: Embedding vectors (n_messages, embedding_dim)
            conversation_id: Unique identifier for the conversation
            
        Returns:
            Dictionary of geometric signatures
        """
        signatures = {}
        
        # 1. Distance matrix (full pairwise distances)
        signatures['distance_matrix'] = self._compute_distance_matrix(embeddings)
        
        # 2. Sequential distances (trajectory)
        signatures['trajectory_distances'] = self._compute_trajectory_distances(embeddings)
        
        # 3. Velocity profile
        signatures['velocity_profile'] = self._compute_velocity_profile(embeddings)
        
        # 4. Curvature sequence
        signatures['curvature_sequence'] = self._compute_curvature_sequence(embeddings)
        
        # 5. Angular velocities
        signatures['angular_velocities'] = self._compute_angular_velocities(embeddings)
        
        # 6. Global measures
        signatures['global_measures'] = self._compute_global_measures(embeddings)
        
        # 7. Persistence features (topological)
        signatures['persistence_features'] = self._compute_persistence_features(embeddings)
        
        # Store for later analysis
        self.metrics_computed[conversation_id] = signatures
        
        return signatures
        
    def _compute_distance_matrix(self, embeddings: np.ndarray) -> np.ndarray:
        """Compute pairwise Euclidean distance matrix."""
        return squareform(pdist(embeddings, metric='euclidean'))
        
    def _compute_trajectory_distances(self, embeddings: np.ndarray) -> np.ndarray:
        """Compute sequential distances between consecutive messages."""
        if len(embeddings) < 2:
            return np.array([])
            
        distances = []
        for i in range(len(embeddings) - 1):
            dist = np.linalg.norm(embeddings[i+1] - embeddings[i])
            distances.append(dist)
            
        return np.array(distances)
        
    def _compute_velocity_profile(self, embeddings: np.ndarray) -> np.ndarray:
        """Compute velocity profile (normalized by embedding dimension)."""
        trajectory_distances = self._compute_trajectory_distances(embeddings)
        if len(trajectory_distances) == 0:
            return np.array([])
            
        # Normalize by sqrt of embedding dimension for scale invariance
        dim = embeddings.shape[1]
        return trajectory_distances / np.sqrt(dim)
        
    def _compute_curvature_sequence(self, embeddings: np.ndarray) -> np.ndarray:
        """Compute discrete curvature at each point."""
        if len(embeddings) < 3:
            return np.array([])
            
        curvatures = []
        for i in range(1, len(embeddings) - 1):
            # Three consecutive points
            p1, p2, p3 = embeddings[i-1], embeddings[i], embeddings[i+1]
            
            # Vectors
            v1 = p2 - p1
            v2 = p3 - p2
            
            # Norms
            norm1 = np.linalg.norm(v1)
            norm2 = np.linalg.norm(v2)
            
            if norm1 > 1e-8 and norm2 > 1e-8:
                # Angle between vectors
                cos_angle = np.dot(v1, v2) / (norm1 * norm2)
                cos_angle = np.clip(cos_angle, -1, 1)
                angle = np.arccos(cos_angle)
                
                # Discrete curvature approximation
                avg_norm = (norm1 + norm2) / 2
                if avg_norm > 1e-8:
                    curvature = angle / avg_norm
                else:
                    curvature = 0
            else:
                curvature = 0
                
            curvatures.append(curvature)
            
        return np.array(curvatures)
        
    def _compute_angular_velocities(self, embeddings: np.ndarray) -> np.ndarray:
        """Compute angular velocity (direction change rate)."""
        if len(embeddings) < 3:
            return np.array([])
            
        angular_velocities = []
        for i in range(1, len(embeddings) - 1):
            # Direction vectors
            v1 = embeddings[i] - embeddings[i-1]
            v2 = embeddings[i+1] - embeddings[i]
            
            # Normalize
            norm1 = np.linalg.norm(v1)
            norm2 = np.linalg.norm(v2)
            
            if norm1 > 1e-8 and norm2 > 1e-8:
                v1_norm = v1 / norm1
                v2_norm = v2 / norm2
                
                # Angular change
                cos_angle = np.dot(v1_norm, v2_norm)
                cos_angle = np.clip(cos_angle, -1, 1)
                angle = np.arccos(cos_angle)
                
                angular_velocities.append(angle)
            else:
                angular_velocities.append(0)
                
        return np.array(angular_velocities)
        
    def _compute_global_measures(self, embeddings: np.ndarray) -> Dict[str, float]:
        """Compute global trajectory measures."""
        measures = {}
        
        # Total path length
        trajectory_distances = self._compute_trajectory_distances(embeddings)
        measures['path_length'] = np.sum(trajectory_distances)
        
        # Displacement (start to end)
        if len(embeddings) >= 2:
            measures['displacement'] = np.linalg.norm(embeddings[-1] - embeddings[0])
            measures['efficiency'] = measures['displacement'] / measures['path_length'] if measures['path_length'] > 0 else 0
        else:
            measures['displacement'] = 0
            measures['efficiency'] = 0
            
        # Spread (average pairwise distance)
        if len(embeddings) >= 2:
            dist_matrix = self._compute_distance_matrix(embeddings)
            measures['spread'] = np.mean(dist_matrix[np.triu_indices_from(dist_matrix, k=1)])
        else:
            measures['spread'] = 0
            
        # Convex hull volume proxy (using PCA)
        if len(embeddings) >= 3:
            from sklearn.decomposition import PCA
            pca = PCA(n_components=min(3, embeddings.shape[1]))
            transformed = pca.fit_transform(embeddings)
            
            # Use variance as proxy for volume
            measures['volume_proxy'] = np.prod(np.var(transformed, axis=0) ** 0.5)
        else:
            measures['volume_proxy'] = 0
            
        return measures
        
    def _compute_persistence_features(self, embeddings: np.ndarray) -> Dict[str, float]:
        """
        Compute simplified topological features.
        
        Full persistence diagrams would require additional libraries,
        so we compute proxy features here.
        """
        features = {}
        
        # Distance-based features
        dist_matrix = self._compute_distance_matrix(embeddings)
        
        # Distribution of distances (captures clustering)
        if dist_matrix.size > 0:
            distances = dist_matrix[np.triu_indices_from(dist_matrix, k=1)]
            features['distance_entropy'] = stats.entropy(np.histogram(distances, bins=20)[0] + 1e-10)
            features['distance_variance'] = np.var(distances)
            features['distance_skewness'] = stats.skew(distances)
        else:
            features['distance_entropy'] = 0
            features['distance_variance'] = 0
            features['distance_skewness'] = 0
            
        # Local density variations
        if len(embeddings) >= 5:
            k = min(5, len(embeddings) - 1)
            local_densities = []
            
            for i in range(len(embeddings)):
                # k-nearest neighbors distance
                distances_i = dist_matrix[i]
                k_nearest = np.sort(distances_i)[1:k+1]  # Exclude self
                local_density = 1.0 / (np.mean(k_nearest) + 1e-8)
                local_densities.append(local_density)
                
            features['density_variation'] = np.std(local_densities)
        else:
            features['density_variation'] = 0
            
        return features


class InvarianceAnalyzer:
    """
    Analyzes cross-model correlations of geometric signatures.
    """
    
    def __init__(self):
        """Initialize the invariance analyzer."""
        self.correlation_results = {}
        
    def compute_invariance_metrics(self, 
                                 signatures_by_model: Dict[str, Dict[str, np.ndarray]],
                                 conversation_id: str) -> Dict[str, Dict[str, float]]:
        """
        Compute correlations between models for each geometric signature.
        
        Args:
            signatures_by_model: Dict mapping model name to geometric signatures
            conversation_id: Unique identifier for the conversation
            
        Returns:
            Dictionary of correlation matrices for each signature type
        """
        model_names = list(signatures_by_model.keys())
        n_models = len(model_names)
        
        if n_models < 2:
            logger.warning("Need at least 2 models for invariance analysis")
            return {}
            
        # Get signature types from first model
        signature_types = list(next(iter(signatures_by_model.values())).keys())
        
        correlations = {}
        
        for sig_type in signature_types:
            # Skip global measures (dict type)
            if sig_type == 'global_measures' or sig_type == 'persistence_features':
                # Handle dict-type signatures separately
                correlations[sig_type] = self._compute_dict_correlations(
                    signatures_by_model, sig_type, model_names
                )
                continue
                
            # Create correlation matrix for this signature type
            corr_matrix = np.ones((n_models, n_models))
            
            for i, model1 in enumerate(model_names):
                for j, model2 in enumerate(model_names):
                    if i < j:  # Upper triangle
                        sig1 = signatures_by_model[model1][sig_type]
                        sig2 = signatures_by_model[model2][sig_type]
                        
                        if len(sig1) > 0 and len(sig2) > 0 and len(sig1) == len(sig2):
                            # Handle different signature types
                            if sig_type == 'distance_matrix':
                                # For matrices, flatten and correlate
                                corr = self._compute_matrix_correlation(sig1, sig2)
                            else:
                                # For vectors, use Spearman correlation
                                corr, _ = stats.spearmanr(sig1, sig2)
                                
                            corr_matrix[i, j] = corr
                            corr_matrix[j, i] = corr
                        else:
                            # Different lengths or empty - no correlation
                            corr_matrix[i, j] = np.nan
                            corr_matrix[j, i] = np.nan
                            
            correlations[sig_type] = {
                'correlation_matrix': corr_matrix,
                'model_names': model_names,
                'mean_correlation': np.nanmean(corr_matrix[np.triu_indices_from(corr_matrix, k=1)])
            }
            
        self.correlation_results[conversation_id] = correlations
        return correlations
        
    def _compute_matrix_correlation(self, matrix1: np.ndarray, matrix2: np.ndarray) -> float:
        """Compute correlation between two matrices."""
        # Use upper triangle only (excluding diagonal)
        upper_indices = np.triu_indices_from(matrix1, k=1)
        vec1 = matrix1[upper_indices]
        vec2 = matrix2[upper_indices]
        
        if len(vec1) > 2:
            corr, _ = stats.spearmanr(vec1, vec2)
            return corr
        else:
            return np.nan
            
    def _compute_dict_correlations(self, signatures_by_model: Dict, 
                                 sig_type: str, model_names: List[str]) -> Dict:
        """Compute correlations for dictionary-type signatures."""
        # Extract all keys
        all_keys = set()
        for model in model_names:
            all_keys.update(signatures_by_model[model][sig_type].keys())
            
        # Compute correlation for each key
        key_correlations = {}
        
        for key in all_keys:
            corr_matrix = np.ones((len(model_names), len(model_names)))
            
            # Extract values for this key from all models
            values_by_model = {}
            for model in model_names:
                if key in signatures_by_model[model][sig_type]:
                    values_by_model[model] = signatures_by_model[model][sig_type][key]
                    
            # Compute pairwise correlations
            for i, model1 in enumerate(model_names):
                for j, model2 in enumerate(model_names):
                    if i < j and model1 in values_by_model and model2 in values_by_model:
                        # For single values, use absolute difference
                        val1 = values_by_model[model1]
                        val2 = values_by_model[model2]
                        
                        # Convert to correlation-like metric (1 = identical, 0 = very different)
                        if val1 + val2 > 0:
                            similarity = 1 - abs(val1 - val2) / (val1 + val2)
                        else:
                            similarity = 1.0
                            
                        corr_matrix[i, j] = similarity
                        corr_matrix[j, i] = similarity
                        
            key_correlations[key] = np.nanmean(corr_matrix[np.triu_indices_from(corr_matrix, k=1)])
            
        return {
            'key_correlations': key_correlations,
            'mean_correlation': np.mean(list(key_correlations.values()))
        }
        
    def aggregate_invariance_scores(self, all_conversations: Dict[str, Dict]) -> Dict:
        """
        Aggregate invariance scores across all conversations.
        
        Args:
            all_conversations: Dict mapping conversation_id to correlation results
            
        Returns:
            Aggregated statistics
        """
        # Collect all correlations by signature type
        correlations_by_type = {}
        
        for conv_id, conv_correlations in all_conversations.items():
            for sig_type, corr_data in conv_correlations.items():
                if sig_type not in correlations_by_type:
                    correlations_by_type[sig_type] = []
                    
                if 'mean_correlation' in corr_data:
                    correlations_by_type[sig_type].append(corr_data['mean_correlation'])
                    
        # Compute statistics for each signature type
        stats_by_type = {}
        for sig_type, correlations in correlations_by_type.items():
            if correlations:
                stats_by_type[sig_type] = {
                    'mean': np.mean(correlations),
                    'std': np.std(correlations),
                    'median': np.median(correlations),
                    'min': np.min(correlations),
                    'max': np.max(correlations),
                    'n_conversations': len(correlations)
                }
                
        # Overall invariance score
        all_correlations = []
        for corrs in correlations_by_type.values():
            all_correlations.extend(corrs)
            
        overall_stats = {
            'mean_invariance': np.mean(all_correlations) if all_correlations else 0,
            'std_invariance': np.std(all_correlations) if all_correlations else 0,
            'median_invariance': np.median(all_correlations) if all_correlations else 0,
            'signature_type_stats': stats_by_type
        }
        
        return overall_stats


class HypothesisTester:
    """
    Statistical hypothesis testing for geometric invariance.
    """
    
    def __init__(self, n_bootstrap: int = 10000):
        """
        Initialize hypothesis tester.
        
        Args:
            n_bootstrap: Number of bootstrap samples
        """
        self.n_bootstrap = n_bootstrap
        
    def test_invariance_hypothesis(self, 
                                 invariance_scores: Dict,
                                 null_scores: Optional[Dict] = None) -> Dict:
        """
        Test the main hypothesis that conversations have invariant geometric signatures.
        
        Args:
            invariance_scores: Real conversation invariance scores
            null_scores: Null model invariance scores (if available)
            
        Returns:
            Hypothesis test results
        """
        results = {}
        
        # Extract overall invariance score
        mean_invariance = invariance_scores['mean_invariance']
        std_invariance = invariance_scores['std_invariance']
        
        # Bootstrap confidence interval
        bootstrap_means = []
        signature_types = list(invariance_scores['signature_type_stats'].keys())
        
        # Create synthetic bootstrap samples
        for _ in range(self.n_bootstrap):
            # Sample with replacement from each signature type
            bootstrap_sample = []
            for sig_type in signature_types:
                stats = invariance_scores['signature_type_stats'][sig_type]
                # Simulate from normal distribution (parametric bootstrap)
                sample = np.random.normal(stats['mean'], stats['std'], 
                                        stats['n_conversations'])
                bootstrap_sample.extend(sample)
                
            bootstrap_means.append(np.mean(bootstrap_sample))
            
        # Confidence intervals
        ci_lower = np.percentile(bootstrap_means, 2.5)
        ci_upper = np.percentile(bootstrap_means, 97.5)
        
        results['invariance_score'] = mean_invariance
        results['confidence_interval'] = (ci_lower, ci_upper)
        results['bootstrap_se'] = np.std(bootstrap_means)
        
        # Test against null hypothesis of no invariance (correlation = 0)
        # One-sample t-test equivalent
        t_statistic = mean_invariance / (std_invariance / np.sqrt(len(signature_types)))
        p_value = 2 * (1 - stats.t.cdf(abs(t_statistic), df=len(signature_types)-1))
        
        results['t_statistic'] = t_statistic
        results['p_value'] = p_value
        results['reject_null'] = p_value < 0.05
        
        # Effect size (Cohen's d)
        results['cohens_d'] = mean_invariance / std_invariance
        
        # If null scores available, compare distributions
        if null_scores:
            results['null_comparison'] = self._compare_to_null(invariance_scores, null_scores)
            
        # Determine if hypothesis is supported
        results['hypothesis_supported'] = (
            mean_invariance > 0.7 and  # Strong correlation
            ci_lower > 0.5 and  # Lower bound still shows moderate correlation
            p_value < 0.001  # Highly significant
        )
        
        return results
        
    def _compare_to_null(self, real_scores: Dict, null_scores: Dict) -> Dict:
        """Compare real scores to null model scores."""
        comparison = {}
        
        # Extract distributions
        real_values = []
        null_values = []
        
        for sig_type in real_scores['signature_type_stats']:
            if sig_type in null_scores.get('signature_type_stats', {}):
                real_stats = real_scores['signature_type_stats'][sig_type]
                null_stats = null_scores['signature_type_stats'][sig_type]
                
                # Simulate values
                real_values.extend(np.random.normal(real_stats['mean'], real_stats['std'], 100))
                null_values.extend(np.random.normal(null_stats['mean'], null_stats['std'], 100))
                
        if real_values and null_values:
            # Mann-Whitney U test
            u_stat, p_value = stats.mannwhitneyu(real_values, null_values, alternative='greater')
            
            comparison['u_statistic'] = u_stat
            comparison['p_value'] = p_value
            comparison['effect_size'] = (np.mean(real_values) - np.mean(null_values)) / np.std(real_values)
            comparison['real_mean'] = np.mean(real_values)
            comparison['null_mean'] = np.mean(null_values)
            
        return comparison
        
    def multiple_comparison_correction(self, p_values: List[float], 
                                     method: str = 'bonferroni') -> List[float]:
        """
        Apply multiple comparison correction.
        
        Args:
            p_values: List of p-values
            method: Correction method ('bonferroni' or 'fdr')
            
        Returns:
            Corrected p-values
        """
        n_tests = len(p_values)
        
        if method == 'bonferroni':
            return [min(p * n_tests, 1.0) for p in p_values]
        elif method == 'fdr':
            # Benjamini-Hochberg procedure
            sorted_indices = np.argsort(p_values)
            sorted_p_values = np.array(p_values)[sorted_indices]
            
            corrected = np.zeros_like(sorted_p_values)
            for i in range(n_tests):
                corrected[i] = sorted_p_values[i] * n_tests / (i + 1)
                
            # Ensure monotonicity
            for i in range(n_tests - 2, -1, -1):
                corrected[i] = min(corrected[i], corrected[i + 1])
                
            # Unsort
            unsorted_corrected = np.zeros_like(corrected)
            unsorted_corrected[sorted_indices] = corrected
            
            return unsorted_corrected.tolist()
        else:
            raise ValueError(f"Unknown method: {method}")


class NullModelComparator:
    """
    Generate and analyze null models for comparison.
    """
    
    def __init__(self):
        """Initialize null model comparator."""
        self.null_generators = {
            'word_shuffle': self._generate_word_shuffle,
            'message_shuffle': self._generate_message_shuffle,
            'random_walk': self._generate_random_walk,
            'brownian': self._generate_brownian_motion,
            'markov': self._generate_markov_text
        }
        
    def generate_null_conversations(self, 
                                  real_conversation: Dict,
                                  null_type: str,
                                  n_samples: int = 100) -> List[Dict]:
        """
        Generate null conversations of specified type.
        
        Args:
            real_conversation: Real conversation to base null model on
            null_type: Type of null model
            n_samples: Number of null samples to generate
            
        Returns:
            List of null conversations
        """
        if null_type not in self.null_generators:
            raise ValueError(f"Unknown null type: {null_type}")
            
        null_conversations = []
        for i in range(n_samples):
            null_conv = self.null_generators[null_type](real_conversation)
            null_conv['metadata']['null_type'] = null_type
            null_conv['metadata']['null_sample'] = i
            null_conversations.append(null_conv)
            
        return null_conversations
        
    def _generate_word_shuffle(self, conversation: Dict) -> Dict:
        """Shuffle words within each message."""
        import copy
        import random
        
        null_conv = copy.deepcopy(conversation)
        
        for message in null_conv['messages']:
            words = message['content'].split()
            random.shuffle(words)
            message['content'] = ' '.join(words)
            
        return null_conv
        
    def _generate_message_shuffle(self, conversation: Dict) -> Dict:
        """Shuffle order of messages."""
        import copy
        import random
        
        null_conv = copy.deepcopy(conversation)
        random.shuffle(null_conv['messages'])
        
        return null_conv
        
    def _generate_random_walk(self, conversation: Dict) -> Dict:
        """Generate random walk with matched statistics."""
        import copy
        
        null_conv = copy.deepcopy(conversation)
        
        # Extract text statistics
        message_lengths = [len(msg['content'].split()) for msg in conversation['messages']]
        vocab = set()
        for msg in conversation['messages']:
            vocab.update(msg['content'].lower().split())
        vocab_list = list(vocab)
        
        # Generate random messages with similar lengths
        for i, msg in enumerate(null_conv['messages']):
            target_length = message_lengths[i]
            words = np.random.choice(vocab_list, size=target_length, replace=True)
            msg['content'] = ' '.join(words)
            
        return null_conv
        
    def _generate_brownian_motion(self, conversation: Dict) -> Dict:
        """Generate pure random text (Brownian motion in text space)."""
        import copy
        import string
        
        null_conv = copy.deepcopy(conversation)
        
        # Character set
        chars = string.ascii_lowercase + ' '
        
        for msg in null_conv['messages']:
            length = len(msg['content'])
            random_text = ''.join(np.random.choice(list(chars), size=length))
            msg['content'] = random_text
            
        return null_conv
        
    def _generate_markov_text(self, conversation: Dict) -> Dict:
        """Generate text using Markov chain to preserve local structure."""
        import copy
        from collections import defaultdict
        
        null_conv = copy.deepcopy(conversation)
        
        # Build transition matrix
        transitions = defaultdict(list)
        
        for msg in conversation['messages']:
            words = msg['content'].lower().split()
            for i in range(len(words) - 1):
                transitions[words[i]].append(words[i + 1])
                
        # Generate new messages
        for msg in null_conv['messages']:
            target_length = len(msg['content'].split())
            
            # Start with random word
            current_word = np.random.choice(list(transitions.keys()))
            generated = [current_word]
            
            for _ in range(target_length - 1):
                if current_word in transitions and transitions[current_word]:
                    next_word = np.random.choice(transitions[current_word])
                else:
                    next_word = np.random.choice(list(transitions.keys()))
                generated.append(next_word)
                current_word = next_word
                
            msg['content'] = ' '.join(generated)
            
        return null_conv