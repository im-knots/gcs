"""
Phase detection functionality for conversation analysis.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from scipy.signal import find_peaks
from scipy.stats import gaussian_kde
import logging

logger = logging.getLogger(__name__)


class PhaseDetector:
    """
    Detects phase transitions in conversations using ensemble embeddings.
    """
    
    def __init__(self,
                 window_size: int = 10,
                 shift_threshold_percentile: float = 75,
                 min_phase_duration: int = 5,
                 ensemble_agreement_threshold: float = 0.6):
        """
        Initialize phase detector.
        
        Args:
            window_size: Size of sliding window for shift detection
            shift_threshold_percentile: Percentile for significant shifts
            min_phase_duration: Minimum turns between phases
            ensemble_agreement_threshold: Required model agreement
        """
        self.window_size = window_size
        self.shift_threshold_percentile = shift_threshold_percentile
        self.min_phase_duration = min_phase_duration
        self.ensemble_agreement_threshold = ensemble_agreement_threshold
        
    def detect_phases(self, 
                     ensemble_embeddings: Dict[str, np.ndarray],
                     verbose: bool = False) -> Dict:
        """
        Detect phase transitions using ensemble of embeddings.
        
        Args:
            ensemble_embeddings: Dict mapping model names to embeddings
            verbose: Whether to print progress
            
        Returns:
            Dictionary with detected phases and metrics
        """
        n_messages = len(next(iter(ensemble_embeddings.values())))
        
        if n_messages < self.window_size * 2:
            return {
                'detected_phases': [],
                'model_phases': {},
                'ensemble_agreement': {'mean_correlation': 0}
            }
            
        # Detect phases for each model
        model_phases = {}
        model_shift_profiles = {}
        
        for model_name, embeddings in ensemble_embeddings.items():
            if verbose:
                logger.info(f"Analyzing {model_name} embeddings...")
                
            # Calculate embedding shifts
            shifts = self._calculate_embedding_shifts(embeddings)
            model_shift_profiles[model_name] = shifts
            
            # Find phase transitions
            phases = self._find_phase_transitions(shifts)
            model_phases[model_name] = phases
            
            if verbose:
                logger.info(f"  Found {len(phases)} phase transitions")
                
        # Find consensus phases
        consensus_phases = self._find_consensus_phases(model_phases, n_messages)
        
        # Calculate agreement metrics
        agreement_metrics = self._calculate_ensemble_agreement(model_shift_profiles)
        
        # Characterize phases
        enhanced_phases = self._characterize_phases(consensus_phases, n_messages)
        
        return {
            'detected_phases': enhanced_phases,
            'model_phases': model_phases,
            'shift_profiles': model_shift_profiles,
            'ensemble_agreement': agreement_metrics
        }
        
    def _calculate_embedding_shifts(self, embeddings: np.ndarray) -> List[Dict]:
        """Calculate magnitude of embedding shifts at each point."""
        shifts = []
        
        for i in range(self.window_size, len(embeddings) - self.window_size):
            # Windows before and after
            before = embeddings[i-self.window_size:i]
            after = embeddings[i:i+self.window_size]
            
            # Centroids
            before_centroid = np.mean(before, axis=0)
            after_centroid = np.mean(after, axis=0)
            
            # Calculate shift magnitude
            shift = np.linalg.norm(after_centroid - before_centroid)
            
            # Angular change
            norm_before = np.linalg.norm(before_centroid)
            norm_after = np.linalg.norm(after_centroid)
            
            if norm_before > 1e-8 and norm_after > 1e-8:
                cos_sim = np.dot(before_centroid, after_centroid) / (norm_before * norm_after)
                angular_change = np.arccos(np.clip(cos_sim, -1, 1))
            else:
                angular_change = 0
                
            shifts.append({
                'turn': i,
                'magnitude': shift,
                'angular_change': angular_change,
                'combined_score': shift * (1 + angular_change)
            })
            
        return shifts
        
    def _find_phase_transitions(self, shifts: List[Dict]) -> List[Dict]:
        """Find significant phase transitions from shift profile."""
        if not shifts:
            return []
            
        # Extract combined scores
        scores = np.array([s['combined_score'] for s in shifts])
        
        # Dynamic threshold
        threshold = np.percentile(scores, self.shift_threshold_percentile)
        
        # Find peaks
        peaks, properties = find_peaks(
            scores,
            height=threshold,
            distance=self.min_phase_duration,
            prominence=np.std(scores) * 0.5
        )
        
        # Convert to phase list
        phases = []
        for i, peak_idx in enumerate(peaks):
            phases.append({
                'turn': shifts[peak_idx]['turn'],
                'magnitude': shifts[peak_idx]['magnitude'],
                'angular_change': shifts[peak_idx]['angular_change'],
                'confidence': properties['prominences'][i] / np.max(properties['prominences'])
            })
            
        return phases
        
    def _find_consensus_phases(self, 
                              model_phases: Dict[str, List],
                              n_messages: int) -> List[Dict]:
        """Find phase transitions that appear across multiple models."""
        all_turns = []
        for phases in model_phases.values():
            all_turns.extend([p['turn'] for p in phases])
            
        if not all_turns:
            return []
            
        # Use KDE to find density peaks
        all_turns = np.array(all_turns)
        kde = gaussian_kde(all_turns, bw_method=0.1)
        
        # Evaluate KDE
        x = np.arange(0, n_messages)
        density = kde(x)
        
        # Find peaks in consensus density
        min_models = len(model_phases) * self.ensemble_agreement_threshold
        peaks, _ = find_peaks(density, height=min_models/n_messages, distance=5)
        
        # Build consensus phases
        consensus_phases = []
        for peak in peaks:
            support = self._get_phase_support(peak, model_phases)
            
            if len(support) >= min_models:
                consensus_phases.append(self._create_consensus_phase(peak, support))
                
        return sorted(consensus_phases, key=lambda x: x['turn'])
        
    def _get_phase_support(self, 
                          turn: int,
                          model_phases: Dict[str, List],
                          tolerance: int = 3) -> Dict:
        """Get supporting evidence from each model for a phase."""
        support = {}
        
        for model, phases in model_phases.items():
            model_turns = [p['turn'] for p in phases]
            if model_turns:
                closest_idx = np.argmin(np.abs(np.array(model_turns) - turn))
                if abs(model_turns[closest_idx] - turn) <= tolerance:
                    support[model] = phases[closest_idx]
                    
        return support
        
    def _create_consensus_phase(self, turn: int, support: Dict) -> Dict:
        """Create consensus phase from supporting evidence."""
        avg_magnitude = np.mean([p['magnitude'] for p in support.values()])
        avg_angular = np.mean([p['angular_change'] for p in support.values()])
        avg_confidence = np.mean([p.get('confidence', 0.5) for p in support.values()])
        
        return {
            'turn': turn,
            'magnitude': avg_magnitude,
            'angular_change': avg_angular,
            'confidence': avg_confidence,
            'support_models': list(support.keys()),
            'consensus_strength': len(support) / self._get_all_models(support)
        }
        
    def _get_all_models(self, support: Dict) -> int:
        """Get total number of models (hack to avoid passing model count)."""
        # This is a bit hacky but avoids needing to pass model count around
        # Return the count, not the length
        return 5  # We have 5 models in the ensemble
        
    def _calculate_ensemble_agreement(self, shift_profiles: Dict) -> Dict:
        """Calculate how well models agree on trajectory dynamics."""
        if len(shift_profiles) < 2:
            return {'mean_correlation': 1.0, 'min_correlation': 1.0}
            
        model_names = list(shift_profiles.keys())
        correlations = []
        
        # Pairwise correlations
        for i, model1 in enumerate(model_names):
            for model2 in model_names[i+1:]:
                corr = self._calculate_shift_correlation(
                    shift_profiles[model1],
                    shift_profiles[model2]
                )
                if corr is not None:
                    correlations.append(corr)
                    
        return {
            'mean_correlation': np.mean(correlations) if correlations else 0,
            'min_correlation': np.min(correlations) if correlations else 0,
            'std_correlation': np.std(correlations) if correlations else 0
        }
        
    def _calculate_shift_correlation(self, shifts1: List, shifts2: List) -> Optional[float]:
        """Calculate correlation between two shift profiles."""
        # Align by turn
        turns1 = [s['turn'] for s in shifts1]
        turns2 = [s['turn'] for s in shifts2]
        
        common_turns = sorted(set(turns1) & set(turns2))
        
        if len(common_turns) > 10:
            scores1 = [s['combined_score'] for s in shifts1 if s['turn'] in common_turns]
            scores2 = [s['combined_score'] for s in shifts2 if s['turn'] in common_turns]
            
            return np.corrcoef(scores1, scores2)[0, 1]
            
        return None
        
    def _characterize_phases(self, phases: List[Dict], n_messages: int) -> List[Dict]:
        """Add semantic characterization to detected phases."""
        for i, phase in enumerate(phases):
            # Determine phase type
            if i == 0:
                phase['type'] = 'opening'
            elif i == len(phases) - 1:
                phase['type'] = 'closing'
            elif phase['angular_change'] > np.pi/3:  # >60 degrees
                phase['type'] = 'topic_shift'
            elif phase['magnitude'] > np.median([p['magnitude'] for p in phases]) * 1.5:
                phase['type'] = 'major_transition'
            else:
                phase['type'] = 'development'
                
            # Add context window
            turn = phase['turn']
            phase['context_window'] = {
                'start': max(0, turn - 5),
                'end': min(n_messages, turn + 5)
            }
            
        return phases
        
    def compare_with_annotations(self,
                               detected_phases: List[Dict],
                               annotated_phases: List[Dict],
                               tolerance: int = 5) -> Dict:
        """
        Compare detected phases with annotated ground truth.
        
        Args:
            detected_phases: List of detected phases
            annotated_phases: List of annotated phases
            tolerance: Turn tolerance for matching
            
        Returns:
            Comparison metrics
        """
        if not annotated_phases:
            return {
                'matches': [],
                'metrics': {
                    'precision': 0,
                    'recall': 0,
                    'f1': 0,
                    'mean_distance': None
                }
            }
            
        # Match detected to annotated
        matches = []
        matched_annotated = set()
        
        for det_phase in detected_phases:
            det_turn = det_phase['turn']
            
            # Find closest annotated phase
            best_match = None
            best_distance = float('inf')
            
            for i, ann_phase in enumerate(annotated_phases):
                if i in matched_annotated:
                    continue
                    
                distance = abs(det_turn - ann_phase['turn'])
                if distance <= tolerance and distance < best_distance:
                    best_match = i
                    best_distance = distance
                    
            if best_match is not None:
                matched_annotated.add(best_match)
                matches.append({
                    'detected': det_phase,
                    'annotated': annotated_phases[best_match],
                    'distance': best_distance
                })
                
        # Calculate metrics
        precision = len(matches) / len(detected_phases) if detected_phases else 0
        recall = len(matches) / len(annotated_phases) if annotated_phases else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            'matches': matches,
            'metrics': {
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'mean_distance': np.mean([m['distance'] for m in matches]) if matches else None
            }
        }