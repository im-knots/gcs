"""
Multi-scale analysis framework for conversation trajectories.

Separates analysis into global (whole conversation), meso (semantic segments),
and local (individual transitions) scales to better understand where invariance
holds and where it breaks down.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
import logging

logger = logging.getLogger(__name__)


class MultiScaleAnalyzer:
    """
    Analyze conversation trajectories at multiple scales.
    
    This addresses the global-local dichotomy by explicitly separating
    different scales of analysis.
    """
    
    def __init__(self):
        """Initialize multi-scale analyzer."""
        self.scales = {
            'global': GlobalScaleAnalyzer(),
            'meso': MesoScaleAnalyzer(),
            'local': LocalScaleAnalyzer()
        }
        
    def analyze_all_scales(self, 
                          embeddings: Dict[str, np.ndarray],
                          metadata: Optional[Dict] = None) -> Dict[str, Dict]:
        """
        Perform analysis at all scales.
        
        Args:
            embeddings: Dict mapping model names to embedding arrays
            metadata: Optional conversation metadata
            
        Returns:
            Analysis results at each scale
        """
        results = {}
        
        # Global scale - entire conversation
        results['global'] = self.scales['global'].analyze(embeddings)
        
        # Meso scale - semantic segments
        results['meso'] = self.scales['meso'].analyze(embeddings)
        
        # Local scale - individual transitions
        results['local'] = self.scales['local'].analyze(embeddings)
        
        # Cross-scale relationships
        results['cross_scale'] = self._analyze_cross_scale_relationships(results)
        
        return results
    
    def _analyze_cross_scale_relationships(self, scale_results: Dict) -> Dict:
        """Analyze relationships between different scales."""
        relationships = {}
        
        # Check if meso-scale segments align with local transitions
        if 'segment_boundaries' in scale_results['meso']:
            boundaries = scale_results['meso']['segment_boundaries']
            local_peaks = scale_results['local'].get('transition_peaks', {})
            
            alignment_scores = {}
            for model_name, peaks in local_peaks.items():
                if model_name in boundaries:
                    # Check how many local peaks fall near segment boundaries
                    model_boundaries = boundaries[model_name]
                    aligned = 0
                    for peak in peaks:
                        for boundary in model_boundaries:
                            if abs(peak - boundary) <= 5:  # Within 5 messages
                                aligned += 1
                                break
                    
                    alignment_scores[model_name] = aligned / len(peaks) if peaks else 0
            
            relationships['meso_local_alignment'] = alignment_scores
        
        # Compare global trajectory properties with meso-scale variability
        global_efficiency = scale_results['global'].get('trajectory_efficiency', {})
        meso_variability = scale_results['meso'].get('segment_variability', {})
        
        efficiency_variability_correlation = {}
        for model in global_efficiency:
            if model in meso_variability:
                # Higher efficiency should correlate with lower variability
                efficiency_variability_correlation[model] = {
                    'efficiency': global_efficiency[model],
                    'variability': meso_variability[model]
                }
        
        relationships['efficiency_variability'] = efficiency_variability_correlation
        
        return relationships


class GlobalScaleAnalyzer:
    """Analyze whole-conversation properties."""
    
    def analyze(self, embeddings: Dict[str, np.ndarray]) -> Dict:
        """
        Analyze global properties that should be most invariant.
        
        These are the properties that show high correlation in your results.
        """
        results = {}
        
        # Trajectory shape descriptors
        trajectory_shapes = {}
        for model_name, emb in embeddings.items():
            shape_desc = self._compute_shape_descriptors(emb)
            trajectory_shapes[model_name] = shape_desc
        
        results['trajectory_shapes'] = trajectory_shapes
        
        # Global efficiency (displacement / path length)
        trajectory_efficiency = {}
        for model_name, emb in embeddings.items():
            if len(emb) >= 2:
                displacement = np.linalg.norm(emb[-1] - emb[0])
                path_length = np.sum([np.linalg.norm(emb[i+1] - emb[i]) 
                                     for i in range(len(emb)-1)])
                efficiency = displacement / path_length if path_length > 0 else 0
                trajectory_efficiency[model_name] = efficiency
        
        results['trajectory_efficiency'] = trajectory_efficiency
        
        # Conversation "spread" in embedding space
        conversation_spread = {}
        for model_name, emb in embeddings.items():
            # Use eigenvalues of covariance as spread measure
            if len(emb) > 1:
                cov = np.cov(emb.T)
                eigenvalues = np.linalg.eigvalsh(cov)
                # Use top k eigenvalues
                k = min(10, len(eigenvalues))
                spread = np.sum(np.sort(eigenvalues)[-k:])
                conversation_spread[model_name] = spread
        
        results['conversation_spread'] = conversation_spread
        
        # Persistence of direction
        direction_persistence = {}
        for model_name, emb in embeddings.items():
            persistence = self._compute_direction_persistence(emb)
            direction_persistence[model_name] = persistence
        
        results['direction_persistence'] = direction_persistence
        
        return results
    
    def _compute_shape_descriptors(self, embeddings: np.ndarray) -> Dict[str, float]:
        """Compute invariant shape descriptors."""
        descriptors = {}
        
        if len(embeddings) < 3:
            return descriptors
        
        # Centroid
        centroid = np.mean(embeddings, axis=0)
        
        # Distances from centroid
        distances = np.array([np.linalg.norm(emb - centroid) for emb in embeddings])
        
        # Shape statistics
        descriptors['mean_radius'] = np.mean(distances)
        descriptors['std_radius'] = np.std(distances)
        descriptors['radius_ratio'] = np.max(distances) / (np.mean(distances) + 1e-8)
        
        # Trajectory "compactness"
        pairwise_distances = []
        for i in range(len(embeddings)):
            for j in range(i+1, len(embeddings)):
                pairwise_distances.append(np.linalg.norm(embeddings[i] - embeddings[j]))
        
        if pairwise_distances:
            descriptors['compactness'] = np.mean(distances) / (np.mean(pairwise_distances) + 1e-8)
        
        return descriptors
    
    def _compute_direction_persistence(self, embeddings: np.ndarray) -> float:
        """
        Compute how persistent the direction of movement is.
        
        High persistence indicates consistent direction (low curvature).
        """
        if len(embeddings) < 3:
            return 0.0
        
        # Compute direction vectors
        directions = []
        for i in range(len(embeddings) - 1):
            direction = embeddings[i+1] - embeddings[i]
            norm = np.linalg.norm(direction)
            if norm > 1e-8:
                directions.append(direction / norm)
        
        if len(directions) < 2:
            return 0.0
        
        # Compute average dot product between consecutive directions
        dot_products = []
        for i in range(len(directions) - 1):
            dot_prod = np.dot(directions[i], directions[i+1])
            dot_products.append(dot_prod)
        
        # Average persistence (1 = straight line, -1 = reversal)
        return np.mean(dot_products)


class MesoScaleAnalyzer:
    """Analyze intermediate-scale structures (semantic segments)."""
    
    def analyze(self, embeddings: Dict[str, np.ndarray]) -> Dict:
        """
        Analyze meso-scale properties like semantic segments.
        
        This scale shows moderate invariance in your results.
        """
        results = {}
        
        # Identify semantic segments using clustering
        segment_boundaries = {}
        segment_characteristics = {}
        
        for model_name, emb in embeddings.items():
            boundaries, chars = self._identify_semantic_segments(emb)
            segment_boundaries[model_name] = boundaries
            segment_characteristics[model_name] = chars
        
        results['segment_boundaries'] = segment_boundaries
        results['segment_characteristics'] = segment_characteristics
        
        # Analyze segment variability
        segment_variability = {}
        for model_name, chars in segment_characteristics.items():
            if chars:
                # Variability in segment properties
                lengths = [c['length'] for c in chars]
                variability = np.std(lengths) / (np.mean(lengths) + 1e-8) if lengths else 0
                segment_variability[model_name] = variability
        
        results['segment_variability'] = segment_variability
        
        # Semantic flow patterns
        flow_patterns = {}
        for model_name, emb in embeddings.items():
            pattern = self._analyze_semantic_flow(emb)
            flow_patterns[model_name] = pattern
        
        results['flow_patterns'] = flow_patterns
        
        return results
    
    def _identify_semantic_segments(self, 
                                  embeddings: np.ndarray,
                                  min_segment_size: int = 10) -> Tuple[List[int], List[Dict]]:
        """
        Identify semantic segments using hierarchical clustering.
        
        Returns:
            segment_boundaries: Indices where segments begin
            segment_characteristics: Properties of each segment
        """
        if len(embeddings) < min_segment_size * 2:
            return [], []
        
        # Use sliding window statistics to find segment boundaries
        window_size = min_segment_size
        change_scores = []
        
        for i in range(window_size, len(embeddings) - window_size):
            # Compare distributions before and after point i
            before = embeddings[i-window_size:i]
            after = embeddings[i:i+window_size]
            
            # Use mean shift as change score
            mean_shift = np.linalg.norm(np.mean(after, axis=0) - np.mean(before, axis=0))
            change_scores.append(mean_shift)
        
        if not change_scores:
            return [], []
        
        # Smooth and find peaks
        smoothed_scores = gaussian_filter1d(change_scores, sigma=3)
        
        # Find peaks (segment boundaries)
        peaks, properties = find_peaks(smoothed_scores, 
                                     height=np.percentile(smoothed_scores, 75),
                                     distance=min_segment_size)
        
        # Adjust indices to account for window offset
        boundaries = [0] + [p + window_size for p in peaks] + [len(embeddings)]
        
        # Characterize segments
        characteristics = []
        for i in range(len(boundaries) - 1):
            start, end = boundaries[i], boundaries[i+1]
            segment = embeddings[start:end]
            
            char = {
                'start': start,
                'end': end,
                'length': end - start,
                'centroid': np.mean(segment, axis=0),
                'spread': np.std(segment, axis=0).mean()
            }
            characteristics.append(char)
        
        return boundaries[1:-1], characteristics  # Exclude artificial boundaries
    
    def _analyze_semantic_flow(self, embeddings: np.ndarray) -> Dict[str, float]:
        """Analyze the flow of semantic content."""
        if len(embeddings) < 10:
            return {}
        
        # Compute semantic velocity (rate of meaning change)
        semantic_velocities = []
        for i in range(1, len(embeddings)):
            # Cosine distance as semantic change
            cos_sim = np.dot(embeddings[i], embeddings[i-1]) / (
                np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[i-1]) + 1e-8
            )
            semantic_velocity = 1 - cos_sim  # Convert similarity to distance
            semantic_velocities.append(semantic_velocity)
        
        # Analyze flow patterns
        patterns = {
            'mean_semantic_velocity': np.mean(semantic_velocities),
            'semantic_acceleration': np.std(semantic_velocities),
            'semantic_smoothness': 1 / (1 + np.std(semantic_velocities))
        }
        
        # Detect semantic "bursts" (rapid changes)
        burst_threshold = np.percentile(semantic_velocities, 90)
        bursts = [i for i, v in enumerate(semantic_velocities) if v > burst_threshold]
        patterns['semantic_burst_rate'] = len(bursts) / len(semantic_velocities)
        
        return patterns


class LocalScaleAnalyzer:
    """Analyze fine-grained local properties."""
    
    def analyze(self, embeddings: Dict[str, np.ndarray]) -> Dict:
        """
        Analyze local properties that show high variability.
        
        This is where your phase detection struggles.
        """
        results = {}
        
        # Local transition detection
        transition_peaks = {}
        transition_characteristics = {}
        
        for model_name, emb in embeddings.items():
            peaks, chars = self._detect_local_transitions(emb)
            transition_peaks[model_name] = peaks
            transition_characteristics[model_name] = chars
        
        results['transition_peaks'] = transition_peaks
        results['transition_characteristics'] = transition_characteristics
        
        # Local stability analysis
        stability_profiles = {}
        for model_name, emb in embeddings.items():
            stability = self._compute_local_stability(emb)
            stability_profiles[model_name] = stability
        
        results['stability_profiles'] = stability_profiles
        
        # Micro-pattern detection
        micro_patterns = {}
        for model_name, emb in embeddings.items():
            patterns = self._detect_micro_patterns(emb)
            micro_patterns[model_name] = patterns
        
        results['micro_patterns'] = micro_patterns
        
        return results
    
    def _detect_local_transitions(self, 
                                embeddings: np.ndarray,
                                window_size: int = 5) -> Tuple[List[int], List[Dict]]:
        """
        Detect fine-grained local transitions.
        
        These are the phase boundaries that vary across models.
        """
        if len(embeddings) < window_size * 2:
            return [], []
        
        # Multiple transition detection methods
        transitions_ensemble = []
        
        # Method 1: Angle changes
        angle_changes = []
        for i in range(1, len(embeddings) - 1):
            v1 = embeddings[i] - embeddings[i-1]
            v2 = embeddings[i+1] - embeddings[i]
            
            norm1, norm2 = np.linalg.norm(v1), np.linalg.norm(v2)
            if norm1 > 1e-8 and norm2 > 1e-8:
                cos_angle = np.dot(v1, v2) / (norm1 * norm2)
                angle = np.arccos(np.clip(cos_angle, -1, 1))
                angle_changes.append(angle)
            else:
                angle_changes.append(0)
        
        if angle_changes:
            angle_peaks, _ = find_peaks(angle_changes, 
                                      height=np.percentile(angle_changes, 80))
            transitions_ensemble.append(set(angle_peaks + 1))  # Adjust index
        
        # Method 2: Embedding space density changes
        density_changes = []
        for i in range(window_size, len(embeddings) - window_size):
            # Local density estimate
            local_points = embeddings[i-window_size:i+window_size+1]
            center = embeddings[i]
            
            distances = [np.linalg.norm(p - center) for p in local_points]
            density = 1 / (np.mean(distances) + 1e-8)
            density_changes.append(density)
        
        if len(density_changes) > 1:
            # Find sudden density changes
            density_diff = np.abs(np.diff(density_changes))
            density_peaks, _ = find_peaks(density_diff,
                                        height=np.percentile(density_diff, 80))
            transitions_ensemble.append(set(density_peaks + window_size))
        
        # Consensus transitions
        all_transitions = set()
        for transitions in transitions_ensemble:
            all_transitions.update(transitions)
        
        # Filter transitions that appear in multiple methods
        consensus_transitions = []
        for t in sorted(all_transitions):
            count = sum(1 for trans_set in transitions_ensemble if t in trans_set)
            if count >= len(transitions_ensemble) // 2:  # Majority vote
                consensus_transitions.append(t)
        
        # Characterize transitions
        characteristics = []
        for t in consensus_transitions:
            char = {
                'position': t,
                'angle_change': angle_changes[t-1] if t-1 < len(angle_changes) else 0,
                'methods_agreed': sum(1 for trans_set in transitions_ensemble if t in trans_set)
            }
            characteristics.append(char)
        
        return consensus_transitions, characteristics
    
    def _compute_local_stability(self, 
                               embeddings: np.ndarray,
                               window_size: int = 5) -> List[float]:
        """
        Compute local stability score at each point.
        
        High stability = low local variance.
        """
        stability_scores = []
        
        for i in range(len(embeddings)):
            # Get local neighborhood
            start = max(0, i - window_size // 2)
            end = min(len(embeddings), i + window_size // 2 + 1)
            
            if end - start > 1:
                local_embeddings = embeddings[start:end]
                # Local variance as instability measure
                local_variance = np.mean(np.var(local_embeddings, axis=0))
                stability = 1 / (1 + local_variance)
            else:
                stability = 1.0
            
            stability_scores.append(stability)
        
        return stability_scores
    
    def _detect_micro_patterns(self, 
                             embeddings: np.ndarray,
                             pattern_length: int = 3) -> Dict[str, List[int]]:
        """
        Detect recurring micro-patterns in the trajectory.
        
        These are small-scale motifs that might repeat.
        """
        if len(embeddings) < pattern_length * 2:
            return {}
        
        patterns = {
            'loops': [],  # Returns to similar embedding
            'spirals': [],  # Consistent rotation
            'jumps': []  # Sudden large movements
        }
        
        # Detect loops
        for i in range(pattern_length, len(embeddings)):
            # Check if current position is close to earlier position
            for j in range(max(0, i - 20), i - pattern_length):
                distance = np.linalg.norm(embeddings[i] - embeddings[j])
                if distance < np.percentile([np.linalg.norm(embeddings[k+1] - embeddings[k]) 
                                           for k in range(len(embeddings)-1)], 10):
                    patterns['loops'].append(i)
                    break
        
        # Detect spirals (consistent angular momentum)
        for i in range(2, len(embeddings) - 2):
            # Check if trajectory is spiraling
            v1 = embeddings[i] - embeddings[i-1]
            v2 = embeddings[i+1] - embeddings[i]
            v3 = embeddings[i+2] - embeddings[i+1]
            
            # Compute cross products to check rotation direction
            if v1.shape[0] >= 3:  # Need at least 3D for cross product
                cross1 = np.cross(v1[:3], v2[:3])
                cross2 = np.cross(v2[:3], v3[:3])
                
                # Consistent rotation direction
                if np.dot(cross1, cross2) > 0 and np.linalg.norm(cross1) > 1e-8:
                    patterns['spirals'].append(i)
        
        # Detect jumps
        distances = [np.linalg.norm(embeddings[i+1] - embeddings[i]) 
                    for i in range(len(embeddings)-1)]
        if distances:
            jump_threshold = np.percentile(distances, 95)
            patterns['jumps'] = [i for i, d in enumerate(distances) if d > jump_threshold]
        
        return patterns