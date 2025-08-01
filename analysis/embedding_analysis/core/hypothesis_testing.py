"""
Statistical hypothesis testing framework for geometric invariance in conversation embeddings.

Main Hypothesis:
Conversations exhibit invariant geometric signatures in embedding space, with distance matrices 
showing significant correlations (ρ > 0.8) across transformer-based embedding models despite 
dimensionality differences (384-768 dims), suggesting an underlying conversational structure 
that transcends specific model architectures.
"""

import numpy as np
from scipy import stats
from scipy.spatial.distance import pdist, squareform
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances
from typing import Dict, List, Tuple, Optional
import logging
from statsmodels.stats.multitest import multipletests
import warnings

logger = logging.getLogger(__name__)


class GeometricInvarianceHypothesisTester:
    """
    Test for geometric invariance across embedding models.
    """
    
    def __init__(self, significance_level: float = 0.05):
        """
        Initialize hypothesis tester.
        
        Args:
            significance_level: Alpha level for statistical tests
        """
        self.alpha = significance_level
        self.results = {}
        
    def test_main_hypothesis(self, 
                           transformer_embeddings: Dict[str, np.ndarray],
                           control_embeddings: Optional[Dict[str, np.ndarray]] = None,
                           scrambled_embeddings: Optional[Dict[str, np.ndarray]] = None) -> Dict:
        """
        Test the main hypothesis about geometric invariance.
        
        Args:
            transformer_embeddings: Dict of transformer model embeddings
            control_embeddings: Optional dict of non-transformer embeddings (Word2Vec, GloVe)
            scrambled_embeddings: Optional dict of scrambled conversation embeddings
            
        Returns:
            Dictionary with test results and statistics
        """
        results = {
            'main_hypothesis': {},
            'geometric_properties': {},
            'controls': {},
            'summary': {}
        }
        
        # Test 1: Distance Matrix Correlations
        logger.info("Testing distance matrix correlations...")
        distance_results = self.test_distance_matrix_invariance(transformer_embeddings)
        results['geometric_properties']['distance_matrices'] = distance_results
        
        # Test 2: Trajectory Curvature Patterns
        logger.info("Testing trajectory curvature patterns...")
        curvature_results = self.test_curvature_invariance(transformer_embeddings)
        results['geometric_properties']['curvature_patterns'] = curvature_results
        
        # Test 3: Phase Transition Boundaries
        logger.info("Testing phase transition boundaries...")
        phase_results = self.test_phase_boundary_invariance(transformer_embeddings)
        results['geometric_properties']['phase_transitions'] = phase_results
        
        # Test 4: Velocity Profiles
        logger.info("Testing velocity profiles...")
        velocity_results = self.test_velocity_profile_invariance(transformer_embeddings)
        results['geometric_properties']['velocity_profiles'] = velocity_results
        
        # Control Tests
        if control_embeddings:
            logger.info("Running control tests with non-transformer embeddings...")
            results['controls']['non_transformer'] = self.test_control_embeddings(
                transformer_embeddings, control_embeddings
            )
            
        if scrambled_embeddings:
            logger.info("Running scrambled conversation tests...")
            results['controls']['scrambled'] = self.test_scrambled_baseline(
                transformer_embeddings, scrambled_embeddings
            )
            
        # Main hypothesis test
        results['main_hypothesis'] = self.evaluate_main_hypothesis(results)
        
        # Multiple testing correction
        results['corrected'] = self.apply_multiple_testing_correction(results)
        
        # Effect sizes
        results['effect_sizes'] = self.calculate_effect_sizes(results)
        
        # Summary statistics
        results['summary'] = self.create_summary(results)
        
        return results
        
    def test_distance_matrix_invariance(self, embeddings: Dict[str, np.ndarray]) -> Dict:
        """
        Test if distance matrices show significant correlations across models.
        
        H0: Distance matrix correlations between models are ≤ 0.8
        H1: Distance matrix correlations between models are > 0.8
        """
        model_names = list(embeddings.keys())
        n_models = len(model_names)
        
        # Calculate distance matrices for each model
        distance_matrices = {}
        for model, emb in embeddings.items():
            # Normalize embeddings to account for dimensionality differences
            scaler = StandardScaler()
            emb_normalized = scaler.fit_transform(emb)
            
            # Calculate pairwise distances
            dist_matrix = pairwise_distances(emb_normalized, metric='euclidean')
            distance_matrices[model] = dist_matrix
            
        # Calculate pairwise correlations
        correlations = []
        correlation_pairs = []
        
        for i in range(n_models):
            for j in range(i+1, n_models):
                model1, model2 = model_names[i], model_names[j]
                
                # Flatten upper triangular parts
                dist1 = distance_matrices[model1][np.triu_indices_from(distance_matrices[model1], k=1)]
                dist2 = distance_matrices[model2][np.triu_indices_from(distance_matrices[model2], k=1)]
                
                # Spearman correlation (robust to monotonic transformations)
                corr, p_value = stats.spearmanr(dist1, dist2)
                correlations.append(corr)
                correlation_pairs.append((model1, model2, corr, p_value))
                
        # Test against threshold of 0.8
        threshold = 0.8
        correlations = np.array(correlations)
        
        # One-sample t-test against threshold
        t_stat, p_value = stats.ttest_1samp(correlations, threshold, alternative='greater')
        
        # Bootstrap confidence interval
        bootstrap_ci = self._bootstrap_confidence_interval(correlations, n_bootstrap=10000)
        
        return {
            'correlations': correlations,
            'mean_correlation': np.mean(correlations),
            'std_correlation': np.std(correlations),
            'min_correlation': np.min(correlations),
            'correlation_pairs': correlation_pairs,
            'hypothesis_test': {
                'null_hypothesis': 'ρ ≤ 0.8',
                'alternative': 'ρ > 0.8',
                't_statistic': t_stat,
                'p_value': p_value,
                'reject_null': p_value < self.alpha,
                'confidence_interval': bootstrap_ci
            },
            'passes_threshold': np.mean(correlations) > threshold
        }
        
    def test_curvature_invariance(self, embeddings: Dict[str, np.ndarray]) -> Dict:
        """
        Test if trajectory curvature patterns are consistent across models.
        """
        from .trajectory import TrajectoryAnalyzer
        analyzer = TrajectoryAnalyzer()
        
        # Calculate curvature for each model
        curvatures = {}
        for model, emb in embeddings.items():
            curv_results = analyzer.calculate_curvature_ensemble(emb)
            curvatures[model] = curv_results['ensemble']['mean']
            
        # Test consistency using coefficient of variation
        curv_values = list(curvatures.values())
        cv = np.std(curv_values) / (np.mean(curv_values) + 1e-8)
        
        # Permutation test for consistency
        observed_cv = cv
        permuted_cvs = []
        
        for _ in range(1000):
            # Randomly shuffle curvature assignments
            shuffled = np.random.permutation(curv_values)
            perm_cv = np.std(shuffled) / (np.mean(shuffled) + 1e-8)
            permuted_cvs.append(perm_cv)
            
        p_value = np.mean(permuted_cvs <= observed_cv)
        
        return {
            'curvatures': curvatures,
            'coefficient_of_variation': cv,
            'consistency_test': {
                'observed_cv': observed_cv,
                'p_value': p_value,
                'consistent': cv < 0.2 and p_value < self.alpha
            }
        }
        
    def test_phase_boundary_invariance(self, embeddings: Dict[str, np.ndarray]) -> Dict:
        """
        Test if phase transitions occur at similar points across models.
        """
        from ..models.ensemble_phase_detector import EnsemblePhaseDetector
        detector = EnsemblePhaseDetector()
        
        # Detect phases for ensemble
        phase_results = detector.detect_phases_ensemble(embeddings)
        
        # Analyze agreement
        model_phases = phase_results.get('model_phases', {})
        if not model_phases:
            return {'error': 'No phase detections available'}
            
        # Extract phase transition points
        transition_points = {}
        for model, phases in model_phases.items():
            transition_points[model] = [p['turn'] for p in phases]
            
        # Calculate pairwise agreement using Jaccard index with tolerance
        agreements = []
        tolerance = 5  # turns
        
        models = list(transition_points.keys())
        for i in range(len(models)):
            for j in range(i+1, len(models)):
                points1 = set(transition_points[models[i]])
                points2 = set(transition_points[models[j]])
                
                # Count matches within tolerance
                matches = 0
                for p1 in points1:
                    for p2 in points2:
                        if abs(p1 - p2) <= tolerance:
                            matches += 1
                            break
                            
                jaccard = matches / (len(points1) + len(points2) - matches + 1e-8)
                agreements.append(jaccard)
                
        # Test if agreement is significant
        mean_agreement = np.mean(agreements) if agreements else 0
        
        # Permutation test
        observed_agreement = mean_agreement
        permuted_agreements = []
        
        n_messages = len(next(iter(embeddings.values())))
        for _ in range(1000):
            # Generate random phase points
            perm_agreements = []
            for _ in range(len(agreements)):
                n_phases1 = np.random.randint(1, 6)
                n_phases2 = np.random.randint(1, 6)
                random_points1 = set(np.random.choice(n_messages, n_phases1, replace=False))
                random_points2 = set(np.random.choice(n_messages, n_phases2, replace=False))
                
                matches = 0
                for p1 in random_points1:
                    for p2 in random_points2:
                        if abs(p1 - p2) <= tolerance:
                            matches += 1
                            break
                            
                jaccard = matches / (len(random_points1) + len(random_points2) - matches + 1e-8)
                perm_agreements.append(jaccard)
                
            permuted_agreements.append(np.mean(perm_agreements))
            
        p_value = np.mean(permuted_agreements >= observed_agreement)
        
        return {
            'transition_points': transition_points,
            'mean_agreement': mean_agreement,
            'agreement_scores': agreements,
            'significance_test': {
                'observed': observed_agreement,
                'p_value': p_value,
                'significant': p_value < self.alpha
            }
        }
        
    def test_velocity_profile_invariance(self, embeddings: Dict[str, np.ndarray]) -> Dict:
        """
        Test if velocity profiles are correlated across models.
        """
        # Calculate velocity profiles
        velocity_profiles = {}
        
        for model, emb in embeddings.items():
            velocities = []
            for i in range(1, len(emb)):
                v = np.linalg.norm(emb[i] - emb[i-1])
                velocities.append(v)
            velocity_profiles[model] = np.array(velocities)
            
        # Normalize profiles
        for model in velocity_profiles:
            profile = velocity_profiles[model]
            if np.std(profile) > 0:
                velocity_profiles[model] = (profile - np.mean(profile)) / np.std(profile)
                
        # Calculate pairwise correlations
        models = list(velocity_profiles.keys())
        correlations = []
        
        for i in range(len(models)):
            for j in range(i+1, len(models)):
                corr, _ = stats.pearsonr(velocity_profiles[models[i]], 
                                        velocity_profiles[models[j]])
                correlations.append(corr)
                
        # Test significance
        mean_corr = np.mean(correlations)
        
        # Fisher z-transformation for testing
        z_scores = [np.arctanh(r) for r in correlations]
        z_mean = np.mean(z_scores)
        z_se = np.std(z_scores) / np.sqrt(len(z_scores))
        
        # Test against null hypothesis of 0 correlation
        z_stat = z_mean / z_se
        p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
        
        return {
            'correlations': correlations,
            'mean_correlation': mean_corr,
            'significance_test': {
                'z_statistic': z_stat,
                'p_value': p_value,
                'significant': p_value < self.alpha
            }
        }
        
    def test_control_embeddings(self, 
                               transformer_embeddings: Dict[str, np.ndarray],
                               control_embeddings: Dict[str, np.ndarray]) -> Dict:
        """
        Compare transformer embeddings to non-transformer controls.
        """
        # Calculate average correlation within transformers
        trans_corrs = []
        trans_models = list(transformer_embeddings.keys())
        
        for i in range(len(trans_models)):
            for j in range(i+1, len(trans_models)):
                dist1 = pairwise_distances(transformer_embeddings[trans_models[i]])
                dist2 = pairwise_distances(transformer_embeddings[trans_models[j]])
                
                corr, _ = stats.spearmanr(dist1.flatten(), dist2.flatten())
                trans_corrs.append(corr)
                
        within_transformer_corr = np.mean(trans_corrs)
        
        # Calculate correlations between transformers and controls
        cross_corrs = []
        
        for trans_model in transformer_embeddings:
            for control_model in control_embeddings:
                dist1 = pairwise_distances(transformer_embeddings[trans_model])
                dist2 = pairwise_distances(control_embeddings[control_model])
                
                # Ensure same shape
                min_size = min(dist1.shape[0], dist2.shape[0])
                dist1 = dist1[:min_size, :min_size]
                dist2 = dist2[:min_size, :min_size]
                
                corr, _ = stats.spearmanr(dist1.flatten(), dist2.flatten())
                cross_corrs.append(corr)
                
        cross_correlation = np.mean(cross_corrs) if cross_corrs else 0
        
        # Two-sample t-test
        if len(trans_corrs) > 1 and len(cross_corrs) > 1:
            t_stat, p_value = stats.ttest_ind(trans_corrs, cross_corrs, alternative='greater')
        else:
            t_stat, p_value = 0, 1
            
        return {
            'within_transformer': within_transformer_corr,
            'transformer_to_control': cross_correlation,
            'difference': within_transformer_corr - cross_correlation,
            'significance_test': {
                't_statistic': t_stat,
                'p_value': p_value,
                'transformers_more_similar': p_value < self.alpha
            }
        }
        
    def test_scrambled_baseline(self,
                               real_embeddings: Dict[str, np.ndarray],
                               scrambled_embeddings: Dict[str, np.ndarray]) -> Dict:
        """
        Compare real conversations to scrambled versions.
        """
        # Calculate correlations for real conversations
        real_corrs = []
        models = list(real_embeddings.keys())
        
        for i in range(len(models)):
            for j in range(i+1, len(models)):
                if models[i] in real_embeddings and models[j] in real_embeddings:
                    dist1 = pairwise_distances(real_embeddings[models[i]])
                    dist2 = pairwise_distances(real_embeddings[models[j]])
                    
                    corr, _ = stats.spearmanr(dist1.flatten(), dist2.flatten())
                    real_corrs.append(corr)
                    
        real_mean = np.mean(real_corrs) if real_corrs else 0
        
        # Calculate correlations for scrambled conversations
        scrambled_corrs = []
        
        for i in range(len(models)):
            for j in range(i+1, len(models)):
                if models[i] in scrambled_embeddings and models[j] in scrambled_embeddings:
                    dist1 = pairwise_distances(scrambled_embeddings[models[i]])
                    dist2 = pairwise_distances(scrambled_embeddings[models[j]])
                    
                    corr, _ = stats.spearmanr(dist1.flatten(), dist2.flatten())
                    scrambled_corrs.append(corr)
                    
        scrambled_mean = np.mean(scrambled_corrs) if scrambled_corrs else 0
        
        # Paired t-test if we have matching pairs
        if len(real_corrs) == len(scrambled_corrs) and len(real_corrs) > 1:
            t_stat, p_value = stats.ttest_rel(real_corrs, scrambled_corrs, alternative='greater')
        elif len(real_corrs) > 1 and len(scrambled_corrs) > 1:
            t_stat, p_value = stats.ttest_ind(real_corrs, scrambled_corrs, alternative='greater')
        else:
            t_stat, p_value = 0, 1
            
        # Effect size (Cohen's d)
        if len(real_corrs) > 0 and len(scrambled_corrs) > 0:
            pooled_std = np.sqrt((np.var(real_corrs) + np.var(scrambled_corrs)) / 2)
            cohens_d = (real_mean - scrambled_mean) / (pooled_std + 1e-8)
        else:
            cohens_d = 0
            
        return {
            'real_correlation': real_mean,
            'scrambled_correlation': scrambled_mean,
            'difference': real_mean - scrambled_mean,
            'effect_size': cohens_d,
            'significance_test': {
                't_statistic': t_stat,
                'p_value': p_value,
                'real_more_structured': p_value < self.alpha
            }
        }
        
    def evaluate_main_hypothesis(self, results: Dict) -> Dict:
        """
        Evaluate the main hypothesis based on all tests.
        """
        # Extract key results
        distance_pass = results['geometric_properties']['distance_matrices']['passes_threshold']
        distance_significant = results['geometric_properties']['distance_matrices']['hypothesis_test']['reject_null']
        
        curvature_consistent = results['geometric_properties']['curvature_patterns']['consistency_test']['consistent']
        
        phase_significant = results['geometric_properties']['phase_transitions'].get(
            'significance_test', {}).get('significant', False)
            
        velocity_significant = results['geometric_properties']['velocity_profiles']['significance_test']['significant']
        
        # Control tests
        transformer_specific = False
        if 'non_transformer' in results['controls']:
            transformer_specific = results['controls']['non_transformer']['significance_test']['transformers_more_similar']
            
        not_random = False
        if 'scrambled' in results['controls']:
            not_random = results['controls']['scrambled']['significance_test']['real_more_structured']
            
        # Overall evaluation
        geometric_invariance = (distance_pass and distance_significant and 
                              (curvature_consistent or phase_significant or velocity_significant))
                              
        hypothesis_supported = geometric_invariance
        if transformer_specific or not_random:
            hypothesis_supported = hypothesis_supported and (transformer_specific or not_random)
            
        return {
            'geometric_invariance_found': geometric_invariance,
            'hypothesis_supported': hypothesis_supported,
            'evidence': {
                'distance_matrices_correlated': distance_pass and distance_significant,
                'curvature_patterns_consistent': curvature_consistent,
                'phase_boundaries_aligned': phase_significant,
                'velocity_profiles_correlated': velocity_significant,
                'transformer_specific': transformer_specific,
                'not_due_to_randomness': not_random
            },
            'conclusion': self._generate_conclusion(hypothesis_supported, results)
        }
        
    def apply_multiple_testing_correction(self, results: Dict) -> Dict:
        """
        Apply Benjamini-Hochberg correction for multiple testing.
        """
        # Collect all p-values
        p_values = []
        test_names = []
        
        # Distance matrix test
        if 'distance_matrices' in results['geometric_properties']:
            p = results['geometric_properties']['distance_matrices']['hypothesis_test']['p_value']
            p_values.append(p)
            test_names.append('distance_matrix_correlation')
            
        # Curvature test
        if 'curvature_patterns' in results['geometric_properties']:
            p = results['geometric_properties']['curvature_patterns']['consistency_test']['p_value']
            p_values.append(p)
            test_names.append('curvature_consistency')
            
        # Phase test
        if 'phase_transitions' in results['geometric_properties']:
            p = results['geometric_properties']['phase_transitions'].get(
                'significance_test', {}).get('p_value', 1)
            p_values.append(p)
            test_names.append('phase_alignment')
            
        # Velocity test
        if 'velocity_profiles' in results['geometric_properties']:
            p = results['geometric_properties']['velocity_profiles']['significance_test']['p_value']
            p_values.append(p)
            test_names.append('velocity_correlation')
            
        # Apply correction
        if p_values:
            rejected, corrected_p, alpha_Sidak, alpha_Bonf = multipletests(
                p_values, alpha=self.alpha, method='fdr_bh'
            )
            
            corrected_results = {
                'n_tests': len(p_values),
                'tests': list(zip(test_names, p_values, corrected_p, rejected)),
                'any_significant': any(rejected),
                'n_significant': sum(rejected)
            }
        else:
            corrected_results = {'error': 'No p-values to correct'}
            
        return corrected_results
        
    def calculate_effect_sizes(self, results: Dict) -> Dict:
        """
        Calculate effect sizes for key comparisons.
        """
        effect_sizes = {}
        
        # Distance matrix correlations
        if 'distance_matrices' in results['geometric_properties']:
            corrs = results['geometric_properties']['distance_matrices']['correlations']
            # Convert correlation to Cohen's d equivalent
            mean_r = np.mean(corrs)
            # Fisher's z transformation
            z = np.arctanh(mean_r)
            # Convert to d
            d = 2 * z / np.sqrt(1 - mean_r**2)
            effect_sizes['distance_correlation'] = {
                'mean_correlation': mean_r,
                'cohens_d_equivalent': d,
                'interpretation': self._interpret_effect_size(d)
            }
            
        # Scrambled comparison
        if 'scrambled' in results['controls']:
            d = results['controls']['scrambled']['effect_size']
            effect_sizes['real_vs_scrambled'] = {
                'cohens_d': d,
                'interpretation': self._interpret_effect_size(d)
            }
            
        return effect_sizes
        
    def create_summary(self, results: Dict) -> Dict:
        """
        Create a summary of all test results.
        """
        summary = {
            'main_hypothesis_supported': results['main_hypothesis']['hypothesis_supported'],
            'key_findings': [],
            'statistical_power': self._estimate_power(results),
            'limitations': []
        }
        
        # Key findings
        if results['main_hypothesis']['evidence']['distance_matrices_correlated']:
            mean_corr = results['geometric_properties']['distance_matrices']['mean_correlation']
            summary['key_findings'].append(
                f"Distance matrices show high correlation (ρ={mean_corr:.3f}) across models"
            )
            
        if results['main_hypothesis']['evidence']['not_due_to_randomness']:
            diff = results['controls']['scrambled']['difference']
            summary['key_findings'].append(
                f"Real conversations show {diff:.3f} higher correlation than scrambled"
            )
            
        # Limitations
        if results['corrected']['n_tests'] > 10:
            summary['limitations'].append(
                "Multiple testing correction may be conservative with many tests"
            )
            
        return summary
        
    def _bootstrap_confidence_interval(self, data: np.ndarray, n_bootstrap: int = 10000) -> Tuple[float, float]:
        """Calculate bootstrap confidence interval."""
        bootstrap_means = []
        
        for _ in range(n_bootstrap):
            sample = np.random.choice(data, size=len(data), replace=True)
            bootstrap_means.append(np.mean(sample))
            
        return np.percentile(bootstrap_means, [2.5, 97.5])
        
    def _interpret_effect_size(self, d: float) -> str:
        """Interpret Cohen's d effect size."""
        if abs(d) < 0.2:
            return "negligible"
        elif abs(d) < 0.5:
            return "small"
        elif abs(d) < 0.8:
            return "medium"
        else:
            return "large"
            
    def _estimate_power(self, results: Dict) -> float:
        """Estimate statistical power based on effect sizes and sample sizes."""
        # Simplified power estimation
        if 'distance_matrices' in results['geometric_properties']:
            n_comparisons = len(results['geometric_properties']['distance_matrices']['correlations'])
            
            if n_comparisons >= 10:
                return 0.8  # Adequate power
            elif n_comparisons >= 6:
                return 0.6  # Moderate power
            else:
                return 0.4  # Low power
        return 0.5
        
    def _generate_conclusion(self, supported: bool, results: Dict) -> str:
        """Generate a conclusion statement."""
        if supported:
            return ("The hypothesis is supported: Conversations exhibit invariant geometric signatures "
                   "across transformer embedding models, with correlations exceeding 0.8 threshold.")
        else:
            reasons = []
            if not results['main_hypothesis']['evidence']['distance_matrices_correlated']:
                reasons.append("distance matrix correlations below threshold")
            if not results['main_hypothesis']['evidence'].get('not_due_to_randomness', True):
                reasons.append("patterns not significantly different from scrambled conversations")
                
            return f"The hypothesis is not supported due to: {', '.join(reasons)}"