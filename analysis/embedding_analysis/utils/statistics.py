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