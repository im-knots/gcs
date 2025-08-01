"""
Statistical analysis utilities for phase detection evaluation.
"""

import numpy as np
from scipy import stats
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


def calculate_phase_correlation(annotated_phases: List[Dict], 
                               detected_phases: List[Dict],
                               n_messages: int,
                               tolerance: int = 5) -> Dict:
    """
    Calculate correlation between annotated and detected phase transitions.
    
    Args:
        annotated_phases: List of annotated phase transitions
        detected_phases: List of detected phase transitions  
        n_messages: Total number of messages
        tolerance: Turn tolerance for matching phases
        
    Returns:
        Dictionary with correlation metrics
    """
    # Convert to binary arrays indicating phase transitions
    annotated_array = np.zeros(n_messages)
    detected_array = np.zeros(n_messages)
    
    # Mark phase transitions
    for phase in annotated_phases:
        turn = phase.get('turn', phase.get('start_turn', 0))
        if 0 <= turn < n_messages:
            annotated_array[turn] = 1
            
    for phase in detected_phases:
        turn = phase.get('turn', phase.get('start_turn', 0))
        if 0 <= turn < n_messages:
            detected_array[turn] = 1
            
    # Apply tolerance window
    if tolerance > 0:
        # Expand detected phases to account for tolerance
        detected_expanded = np.zeros_like(detected_array)
        for i in range(n_messages):
            if detected_array[i] == 1:
                start = max(0, i - tolerance)
                end = min(n_messages, i + tolerance + 1)
                detected_expanded[start:end] = 1
        detected_array = detected_expanded
        
    # Calculate correlation metrics
    correlation, p_value = stats.pearsonr(annotated_array, detected_array)
    
    # Calculate precision, recall, F1
    true_positives = np.sum((annotated_array == 1) & (detected_array == 1))
    false_positives = np.sum((annotated_array == 0) & (detected_array == 1))
    false_negatives = np.sum((annotated_array == 1) & (detected_array == 0))
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    # Cohen's kappa for agreement
    observed_agreement = np.sum(annotated_array == detected_array) / n_messages
    expected_agreement = (np.sum(annotated_array == 1) * np.sum(detected_array == 1) + 
                         np.sum(annotated_array == 0) * np.sum(detected_array == 0)) / (n_messages ** 2)
    
    cohen_kappa = (observed_agreement - expected_agreement) / (1 - expected_agreement) if expected_agreement < 1 else 0
    
    return {
        'correlation': correlation,
        'p_value': p_value,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'cohen_kappa': cohen_kappa,
        'n_annotated': len(annotated_phases),
        'n_detected': len(detected_phases),
        'true_positives': int(true_positives),
        'false_positives': int(false_positives),
        'false_negatives': int(false_negatives)
    }


def calculate_model_agreement(model_phases: Dict[str, List[Dict]], 
                            n_messages: int,
                            tolerance: int = 3) -> Dict:
    """
    Calculate agreement between different models' phase detections.
    
    Args:
        model_phases: Dictionary mapping model names to detected phases
        n_messages: Total number of messages
        tolerance: Turn tolerance for matching phases
        
    Returns:
        Dictionary with agreement metrics
    """
    model_names = list(model_phases.keys())
    n_models = len(model_names)
    
    if n_models < 2:
        return {'mean_agreement': 1.0, 'pairwise_agreements': {}}
        
    # Calculate pairwise agreements
    pairwise_agreements = {}
    correlations = []
    
    for i in range(n_models):
        for j in range(i + 1, n_models):
            model1, model2 = model_names[i], model_names[j]
            
            # Create binary arrays
            array1 = np.zeros(n_messages)
            array2 = np.zeros(n_messages)
            
            for phase in model_phases[model1]:
                turn = phase['turn']
                if 0 <= turn < n_messages:
                    array1[turn] = 1
                    
            for phase in model_phases[model2]:
                turn = phase['turn']
                if 0 <= turn < n_messages:
                    array2[turn] = 1
                    
            # Apply tolerance
            if tolerance > 0:
                array1_expanded = np.zeros_like(array1)
                array2_expanded = np.zeros_like(array2)
                
                for idx in range(n_messages):
                    if array1[idx] == 1:
                        start = max(0, idx - tolerance)
                        end = min(n_messages, idx + tolerance + 1)
                        array1_expanded[start:end] = 1
                        
                    if array2[idx] == 1:
                        start = max(0, idx - tolerance)
                        end = min(n_messages, idx + tolerance + 1)
                        array2_expanded[start:end] = 1
                        
                agreement = np.sum((array1_expanded == 1) & (array2_expanded == 1)) / max(np.sum(array1), np.sum(array2), 1)
            else:
                agreement = np.sum((array1 == 1) & (array2 == 1)) / max(np.sum(array1), np.sum(array2), 1)
                
            pair_key = f"{model1}-{model2}"
            pairwise_agreements[pair_key] = agreement
            correlations.append(agreement)
            
    return {
        'mean_agreement': np.mean(correlations) if correlations else 0,
        'std_agreement': np.std(correlations) if correlations else 0,
        'pairwise_agreements': pairwise_agreements,
        'n_models': n_models
    }


def calculate_phase_timing_correlation(annotated_phases: List[Dict],
                                     model_phases: Dict[str, List[Dict]],
                                     n_messages: int) -> Dict:
    """
    Calculate correlation of phase timing across models and annotations.
    
    Args:
        annotated_phases: List of annotated phase transitions
        model_phases: Dictionary mapping model names to detected phases
        n_messages: Total number of messages
        
    Returns:
        Dictionary with timing correlation metrics
    """
    results = {}
    
    # Get annotated phase turns
    annotated_turns = [p.get('turn', p.get('start_turn', 0)) for p in annotated_phases]
    
    for model_name, phases in model_phases.items():
        detected_turns = [p['turn'] for p in phases]
        
        # Calculate timing differences for matched phases
        timing_diffs = []
        for ann_turn in annotated_turns:
            # Find closest detected phase
            if detected_turns:
                closest_idx = np.argmin(np.abs(np.array(detected_turns) - ann_turn))
                closest_turn = detected_turns[closest_idx]
                diff = abs(closest_turn - ann_turn)
                
                # Only count if within reasonable range
                if diff <= 10:
                    timing_diffs.append(diff)
                    
        if timing_diffs:
            results[model_name] = {
                'mean_timing_error': np.mean(timing_diffs),
                'std_timing_error': np.std(timing_diffs),
                'max_timing_error': np.max(timing_diffs),
                'matched_phases': len(timing_diffs),
                'match_rate': len(timing_diffs) / len(annotated_turns) if annotated_turns else 0
            }
        else:
            results[model_name] = {
                'mean_timing_error': float('inf'),
                'std_timing_error': 0,
                'max_timing_error': float('inf'),
                'matched_phases': 0,
                'match_rate': 0
            }
            
    return results


def test_model_phase_agreement(model_phases: Dict[str, List[Dict]], 
                              n_messages: int,
                              tolerance: int = 3) -> Dict:
    """
    Perform statistical tests on model agreement for phase detection.
    
    Args:
        model_phases: Dictionary mapping model names to detected phases
        n_messages: Total number of messages
        tolerance: Turn tolerance for matching phases
        
    Returns:
        Dictionary with statistical test results
    """
    model_names = list(model_phases.keys())
    n_models = len(model_names)
    
    # Create binary matrix for each model
    phase_matrix = np.zeros((n_models, n_messages))
    
    for i, model_name in enumerate(model_names):
        for phase in model_phases[model_name]:
            turn = phase['turn']
            if 0 <= turn < n_messages:
                # Apply tolerance window
                start = max(0, turn - tolerance)
                end = min(n_messages, turn + tolerance + 1)
                phase_matrix[i, start:end] = 1
    
    # Fleiss' Kappa for multi-rater agreement
    # Count agreements at each turn
    n_positive = np.sum(phase_matrix, axis=0)
    n_negative = n_models - n_positive
    
    # Calculate P_o (observed agreement)
    p_o = np.mean((n_positive * (n_positive - 1) + n_negative * (n_negative - 1)) / (n_models * (n_models - 1)))
    
    # Calculate P_e (expected agreement)
    p_positive = np.sum(phase_matrix) / (n_models * n_messages)
    p_negative = 1 - p_positive
    p_e = p_positive**2 + p_negative**2
    
    # Fleiss' Kappa
    fleiss_kappa = (p_o - p_e) / (1 - p_e) if p_e < 1 else 0
    
    # Kendall's W (coefficient of concordance)
    # Rank each position by number of models detecting a phase
    ranks = stats.rankdata(n_positive)
    mean_rank = np.mean(ranks)
    ss_total = np.sum((ranks - mean_rank)**2)
    kendall_w = (12 * ss_total) / (n_models**2 * (n_messages**3 - n_messages))
    
    # Chi-square test for Kendall's W
    chi2_stat = n_models * (n_messages - 1) * kendall_w
    chi2_df = n_messages - 1
    chi2_p_value = 1 - stats.chi2.cdf(chi2_stat, chi2_df)
    
    # Intraclass correlation coefficient (ICC)
    # Using one-way random effects model
    phase_counts = np.sum(phase_matrix, axis=1)
    between_models_var = np.var(phase_counts) * n_messages
    within_models_var = np.sum([np.var(phase_matrix[i, :]) for i in range(n_models)])
    
    icc = between_models_var / (between_models_var + within_models_var) if (between_models_var + within_models_var) > 0 else 0
    
    return {
        'fleiss_kappa': fleiss_kappa,
        'kendall_w': kendall_w,
        'kendall_w_p_value': chi2_p_value,
        'icc': icc,
        'mean_phases_per_model': np.mean([len(phases) for phases in model_phases.values()]),
        'std_phases_per_model': np.std([len(phases) for phases in model_phases.values()]),
        'interpretation': {
            'fleiss_kappa': _interpret_kappa(fleiss_kappa),
            'kendall_w': _interpret_kendall_w(kendall_w),
            'agreement_level': _interpret_agreement_level(fleiss_kappa, kendall_w, icc)
        }
    }


def test_phase_detection_accuracy(annotated_phases: List[Dict],
                                 model_phases: Dict[str, List[Dict]],
                                 n_messages: int,
                                 tolerance: int = 5) -> Dict:
    """
    Test accuracy of phase detection against annotated ground truth.
    
    Args:
        annotated_phases: List of annotated phase transitions
        model_phases: Dictionary mapping model names to detected phases
        n_messages: Total number of messages
        tolerance: Turn tolerance for matching phases
        
    Returns:
        Dictionary with accuracy test results
    """
    results = {
        'per_model': {},
        'ensemble': {},
        'statistical_tests': {}
    }
    
    # Get annotated turns
    annotated_turns = np.array([p.get('turn', p.get('start_turn', 0)) for p in annotated_phases])
    
    # Test each model individually
    for model_name, phases in model_phases.items():
        detected_turns = np.array([p['turn'] for p in phases])
        
        # Calculate matches within tolerance
        matches = []
        for ann_turn in annotated_turns:
            if len(detected_turns) > 0:
                distances = np.abs(detected_turns - ann_turn)
                min_dist = np.min(distances)
                if min_dist <= tolerance:
                    matches.append(min_dist)
        
        # Statistical tests
        if matches:
            # Wilcoxon signed-rank test for timing bias
            wilcoxon_stat, wilcoxon_p = stats.wilcoxon(matches) if len(matches) > 1 else (0, 1)
            
            # One-sample t-test against expected value of 0 (perfect timing)
            t_stat, t_p = stats.ttest_1samp(matches, 0)
        else:
            wilcoxon_stat, wilcoxon_p = 0, 1
            t_stat, t_p = 0, 1
        
        results['per_model'][model_name] = {
            'precision': len(matches) / len(detected_turns) if len(detected_turns) > 0 else 0,
            'recall': len(matches) / len(annotated_turns) if len(annotated_turns) > 0 else 0,
            'mean_timing_error': np.mean(matches) if matches else float('inf'),
            'median_timing_error': np.median(matches) if matches else float('inf'),
            'timing_bias_test': {
                'wilcoxon_statistic': wilcoxon_stat,
                'wilcoxon_p_value': wilcoxon_p,
                't_statistic': t_stat,
                't_p_value': t_p,
                'significant_bias': t_p < 0.05
            }
        }
    
    # Test ensemble consensus
    # Create consensus phases where multiple models agree
    all_detected_turns = []
    for phases in model_phases.values():
        all_detected_turns.extend([p['turn'] for p in phases])
    
    if all_detected_turns:
        # Use kernel density estimation to find consensus peaks
        from scipy.stats import gaussian_kde
        kde = gaussian_kde(all_detected_turns, bw_method=0.1)
        x = np.arange(n_messages)
        density = kde(x)
        
        # Find peaks in consensus
        consensus_threshold = len(model_phases) * 0.5 / n_messages
        consensus_turns = x[density > consensus_threshold]
        
        # Test consensus accuracy
        consensus_matches = []
        for ann_turn in annotated_turns:
            if len(consensus_turns) > 0:
                min_dist = np.min(np.abs(consensus_turns - ann_turn))
                if min_dist <= tolerance:
                    consensus_matches.append(min_dist)
        
        results['ensemble'] = {
            'consensus_phases': len(consensus_turns),
            'precision': len(consensus_matches) / len(consensus_turns) if len(consensus_turns) > 0 else 0,
            'recall': len(consensus_matches) / len(annotated_turns) if len(annotated_turns) > 0 else 0,
            'mean_timing_error': np.mean(consensus_matches) if consensus_matches else float('inf')
        }
    
    # Overall statistical tests
    # McNemar's test for comparing model performances
    model_names = list(model_phases.keys())
    if len(model_names) >= 2:
        model1_phases = model_phases[model_names[0]]
        model2_phases = model_phases[model_names[1]]
        
        # Create contingency table
        both_correct = 0
        only_model1 = 0
        only_model2 = 0
        both_wrong = 0
        
        for ann_turn in annotated_turns:
            model1_match = any(abs(p['turn'] - ann_turn) <= tolerance for p in model1_phases)
            model2_match = any(abs(p['turn'] - ann_turn) <= tolerance for p in model2_phases)
            
            if model1_match and model2_match:
                both_correct += 1
            elif model1_match and not model2_match:
                only_model1 += 1
            elif not model1_match and model2_match:
                only_model2 += 1
            else:
                both_wrong += 1
        
        # McNemar's test
        n = only_model1 + only_model2
        if n > 0:
            mcnemar_stat = (abs(only_model1 - only_model2) - 1)**2 / n
            mcnemar_p = 1 - stats.chi2.cdf(mcnemar_stat, 1)
        else:
            mcnemar_stat, mcnemar_p = 0, 1
        
        results['statistical_tests']['mcnemar'] = {
            'statistic': mcnemar_stat,
            'p_value': mcnemar_p,
            'model1_better': only_model1 > only_model2,
            'significant_difference': mcnemar_p < 0.05
        }
    
    return results


def _interpret_kappa(kappa: float) -> str:
    """Interpret Fleiss' Kappa value."""
    if kappa < 0:
        return "Poor agreement (worse than chance)"
    elif kappa < 0.20:
        return "Slight agreement"
    elif kappa < 0.40:
        return "Fair agreement"
    elif kappa < 0.60:
        return "Moderate agreement"
    elif kappa < 0.80:
        return "Substantial agreement"
    else:
        return "Almost perfect agreement"


def _interpret_kendall_w(w: float) -> str:
    """Interpret Kendall's W value."""
    if w < 0.1:
        return "Very weak agreement"
    elif w < 0.3:
        return "Weak agreement"
    elif w < 0.5:
        return "Moderate agreement"
    elif w < 0.7:
        return "Strong agreement"
    else:
        return "Very strong agreement"


def _interpret_agreement_level(kappa: float, w: float, icc: float) -> str:
    """Overall interpretation of agreement level."""
    avg_score = (kappa + w + icc) / 3
    
    if avg_score < 0.2:
        return "Poor overall agreement between models"
    elif avg_score < 0.4:
        return "Fair overall agreement between models"
    elif avg_score < 0.6:
        return "Moderate overall agreement between models"
    elif avg_score < 0.8:
        return "Good overall agreement between models"
    else:
        return "Excellent overall agreement between models"


def calculate_distance_matrix_correlations(distance_matrices: Dict[str, np.ndarray]) -> np.ndarray:
    """
    Calculate pairwise correlations between distance matrices from different models.
    
    Args:
        distance_matrices: Dictionary mapping model names to distance matrices
        
    Returns:
        Correlation matrix between models
    """
    model_names = list(distance_matrices.keys())
    n_models = len(model_names)
    corr_matrix = np.ones((n_models, n_models))
    
    for i, model1 in enumerate(model_names):
        for j, model2 in enumerate(model_names):
            if i != j:
                mat1 = distance_matrices[model1].flatten()
                mat2 = distance_matrices[model2].flatten()
                # Handle NaN values
                mask = ~(np.isnan(mat1) | np.isnan(mat2))
                if np.sum(mask) > 0:
                    corr = np.corrcoef(mat1[mask], mat2[mask])[0, 1]
                    corr_matrix[i, j] = corr
                else:
                    corr_matrix[i, j] = 0
                    
    return corr_matrix


def calculate_velocity_profile_correlations(embeddings: Dict[str, np.ndarray]) -> np.ndarray:
    """
    Calculate trajectory correlations based on velocity patterns between models.
    
    Args:
        embeddings: Dictionary mapping model names to embedding arrays
        
    Returns:
        Correlation matrix of velocity profiles between models
    """
    model_names = list(embeddings.keys())
    n_models = len(model_names)
    corr_matrix = np.ones((n_models, n_models))
    
    # Calculate velocities for each model
    velocities = {}
    for model, emb in embeddings.items():
        vel = []
        for i in range(1, len(emb)):
            vel.append(np.linalg.norm(emb[i] - emb[i-1]))
        velocities[model] = np.array(vel)
        
    # Calculate correlations
    for i, model1 in enumerate(model_names):
        for j, model2 in enumerate(model_names):
            if i != j:
                v1 = velocities[model1]
                v2 = velocities[model2]
                if len(v1) > 0 and len(v2) > 0 and np.std(v1) > 0 and np.std(v2) > 0:
                    corr = np.corrcoef(v1, v2)[0, 1]
                    corr_matrix[i, j] = corr
                else:
                    corr_matrix[i, j] = 0
                    
    return corr_matrix


def calculate_topology_preservation(embeddings: Dict[str, np.ndarray], k: Optional[int] = None) -> np.ndarray:
    """
    Calculate topology preservation between models by checking if nearest neighbors are preserved.
    
    Args:
        embeddings: Dictionary mapping model names to embedding arrays
        k: Number of nearest neighbors to consider (default: min(10, n_points//5))
        
    Returns:
        Matrix showing topology preservation scores between models
    """
    model_names = list(embeddings.keys())
    n_models = len(model_names)
    topo_matrix = np.ones((n_models, n_models))
    
    # Determine k if not provided
    n_points = len(next(iter(embeddings.values())))
    if k is None:
        k = min(10, n_points // 5)
    
    # Ensure k is valid
    if k >= n_points:
        k = n_points - 1
    if k < 1:
        return topo_matrix
    
    for i, model1 in enumerate(model_names):
        for j, model2 in enumerate(model_names):
            if i != j:
                # Calculate k-nearest neighbors for each point in both models
                emb1 = embeddings[model1]
                emb2 = embeddings[model2]
                
                preservation_scores = []
                for idx in range(len(emb1)):
                    # Find k nearest neighbors in model1
                    dists1 = [np.linalg.norm(emb1[idx] - emb1[jdx]) 
                             for jdx in range(len(emb1)) if idx != jdx]
                    nn1 = np.argsort(dists1)[:k]
                    
                    # Find k nearest neighbors in model2
                    dists2 = [np.linalg.norm(emb2[idx] - emb2[jdx]) 
                             for jdx in range(len(emb2)) if idx != jdx]
                    nn2 = np.argsort(dists2)[:k]
                    
                    # Calculate overlap
                    overlap = len(set(nn1) & set(nn2)) / k
                    preservation_scores.append(overlap)
                    
                topo_matrix[i, j] = np.mean(preservation_scores)
                
    return topo_matrix


def calculate_ensemble_trajectory_statistics(embeddings: Dict[str, np.ndarray]) -> Dict:
    """
    Calculate comprehensive trajectory statistics for ensemble embeddings.
    
    Args:
        embeddings: Dictionary mapping model names to embedding arrays
        
    Returns:
        Dictionary containing various trajectory statistics and correlations
    """
    results = {
        'distance_correlations': calculate_distance_matrix_correlations(
            {model: _calculate_distance_matrix(emb) for model, emb in embeddings.items()}
        ),
        'velocity_correlations': calculate_velocity_profile_correlations(embeddings),
        'topology_preservation': calculate_topology_preservation(embeddings),
        'summary': {}
    }
    
    # Calculate summary statistics
    dist_corrs = results['distance_correlations']
    vel_corrs = results['velocity_correlations']
    topo_pres = results['topology_preservation']
    
    # Extract off-diagonal elements (excluding self-correlations)
    n = len(dist_corrs)
    mask = ~np.eye(n, dtype=bool)
    
    results['summary'] = {
        'mean_distance_correlation': np.mean(dist_corrs[mask]),
        'std_distance_correlation': np.std(dist_corrs[mask]),
        'mean_velocity_correlation': np.mean(vel_corrs[mask]),
        'std_velocity_correlation': np.std(vel_corrs[mask]),
        'mean_topology_preservation': np.mean(topo_pres[mask]),
        'std_topology_preservation': np.std(topo_pres[mask])
    }
    
    return results


def _calculate_distance_matrix(embeddings: np.ndarray) -> np.ndarray:
    """Helper function to calculate distance matrix for embeddings."""
    n = len(embeddings)
    dist_matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(i+1, n):
            dist = np.linalg.norm(embeddings[i] - embeddings[j])
            dist_matrix[i, j] = dist
            dist_matrix[j, i] = dist
            
    return dist_matrix