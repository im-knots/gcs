"""
Ensemble phase detection using multiple algorithms for improved accuracy.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from scipy import signal, stats
from scipy.ndimage import gaussian_filter1d
try:
    import ruptures as rpt
    HAS_RUPTURES = True
except ImportError:
    HAS_RUPTURES = False
from sklearn.mixture import GaussianMixture
from sklearn.cluster import DBSCAN
try:
    from hmmlearn import hmm
    HAS_HMMLEARN = True
except ImportError:
    HAS_HMMLEARN = False
import logging
import warnings

logger = logging.getLogger(__name__)


class EnsemblePhaseDetector:
    """
    Detect conversation phases using an ensemble of methods for robustness.
    """
    
    def __init__(self):
        """Initialize ensemble phase detector."""
        self.methods = {
            'embedding_shift': self._detect_by_embedding_shift,
            'change_point': self._detect_by_change_point,
            'hmm': self._detect_by_hmm,
            'spectral': self._detect_by_spectral,
            'clustering': self._detect_by_clustering
        }
        
    def detect_phases_ensemble(self, 
                             embeddings: Dict[str, np.ndarray],
                             annotated_phases: Optional[List[Dict]] = None) -> Dict:
        """
        Detect phases using multiple methods and combine results.
        
        Args:
            embeddings: Dictionary of embeddings from different models
            annotated_phases: Optional ground truth phases for evaluation
            
        Returns:
            Dictionary with ensemble phase detection results
        """
        results = {
            'method_results': {},
            'ensemble_phases': [],
            'method_agreement': {},
            'evaluation': {}
        }
        
        # Run each detection method
        for method_name, method_func in self.methods.items():
            try:
                logger.info(f"Running {method_name} phase detection...")
                method_phases = method_func(embeddings)
                results['method_results'][method_name] = method_phases
            except Exception as e:
                logger.warning(f"Method {method_name} failed: {e}")
                results['method_results'][method_name] = []
                
        # Combine results using voting
        results['ensemble_phases'] = self._combine_phase_detections(
            results['method_results'], 
            n_messages=len(next(iter(embeddings.values())))
        )
        
        # Calculate method agreement
        results['method_agreement'] = self._calculate_method_agreement(
            results['method_results']
        )
        
        # Evaluate against ground truth if available
        if annotated_phases:
            results['evaluation'] = self._evaluate_detection(
                results['ensemble_phases'],
                annotated_phases
            )
            
        return results
        
    def _detect_by_embedding_shift(self, embeddings: Dict[str, np.ndarray]) -> List[Dict]:
        """
        Original method: Detect phases by large embedding shifts.
        Enhanced with adaptive thresholding.
        """
        all_phases = []
        
        for model_name, emb in embeddings.items():
            # Calculate embedding shifts
            shifts = []
            window_size = 10
            
            for i in range(window_size, len(emb) - window_size):
                before = emb[i-window_size:i]
                after = emb[i:i+window_size]
                
                before_centroid = np.mean(before, axis=0)
                after_centroid = np.mean(after, axis=0)
                
                shift = np.linalg.norm(after_centroid - before_centroid)
                shifts.append({'turn': i, 'magnitude': shift})
                
            if not shifts:
                continue
                
            # Adaptive thresholding using MAD (Median Absolute Deviation)
            magnitudes = [s['magnitude'] for s in shifts]
            median = np.median(magnitudes)
            mad = np.median(np.abs(magnitudes - median))
            threshold = median + 3 * mad  # 3 MADs above median
            
            # Find peaks above threshold
            peaks, _ = signal.find_peaks(
                magnitudes,
                height=threshold,
                distance=20  # Minimum 20 turns between phases
            )
            
            for peak in peaks:
                all_phases.append({
                    'turn': shifts[peak]['turn'],
                    'confidence': min(1.0, shifts[peak]['magnitude'] / threshold - 1),
                    'method': 'embedding_shift',
                    'model': model_name
                })
                
        return all_phases
        
    def _detect_by_change_point(self, embeddings: Dict[str, np.ndarray]) -> List[Dict]:
        """
        Use change point detection algorithms (PELT, Binary Segmentation).
        """
        all_phases = []
        
        for model_name, emb in embeddings.items():
            # Use PELT (Pruned Exact Linear Time) algorithm
            # Project to 1D using first principal component for efficiency
            from sklearn.decomposition import PCA
            
            if emb.shape[0] < 10:
                continue
                
            pca = PCA(n_components=1)
            emb_1d = pca.fit_transform(emb).flatten()
            
            # PELT with adaptive penalty
            if not HAS_RUPTURES:
                logger.warning("ruptures not installed, skipping PELT change point detection")
                return []
            try:
                algo = rpt.Pelt(model='rbf', min_size=10).fit(emb_1d)
                # Use elbow method for penalty selection
                penalty = self._select_penalty_elbow(emb_1d, algo)
                change_points = algo.predict(pen=penalty)
                
                # Convert to phase format
                for cp in change_points[:-1]:  # Last point is always the end
                    all_phases.append({
                        'turn': cp,
                        'confidence': 0.8,  # PELT doesn't provide confidence
                        'method': 'change_point_pelt',
                        'model': model_name
                    })
            except Exception as e:
                logger.warning(f"PELT failed for {model_name}: {e}")
                
            # Binary Segmentation as backup
            try:
                algo_binseg = rpt.Binseg(model='l2', min_size=10).fit(emb_1d)
                n_bkps = min(5, len(emb_1d) // 50)  # Heuristic for number of breakpoints
                change_points_binseg = algo_binseg.predict(n_bkps=n_bkps)
                
                for cp in change_points_binseg[:-1]:
                    all_phases.append({
                        'turn': cp,
                        'confidence': 0.7,
                        'method': 'change_point_binseg',
                        'model': model_name
                    })
            except Exception as e:
                logger.warning(f"Binary segmentation failed for {model_name}: {e}")
                
        return all_phases
        
    def _detect_by_hmm(self, embeddings: Dict[str, np.ndarray]) -> List[Dict]:
        """
        Use Hidden Markov Models to detect phase transitions.
        """
        all_phases = []
        
        for model_name, emb in embeddings.items():
            if emb.shape[0] < 20:
                continue
                
            # Reduce dimensionality for HMM
            from sklearn.decomposition import PCA
            pca = PCA(n_components=min(10, emb.shape[1]))
            emb_reduced = pca.fit_transform(emb)
            
            # Try different numbers of states
            if not HAS_HMMLEARN:
                logger.warning("hmmlearn not installed, skipping HMM phase detection")
                return []
                
            best_model = None
            best_score = -np.inf
            best_n_states = 3
            
            for n_states in range(3, min(8, emb.shape[0] // 20)):
                try:
                    model = hmm.GaussianHMM(
                        n_components=n_states,
                        covariance_type="diag",
                        n_iter=100,
                        random_state=42
                    )
                    model.fit(emb_reduced)
                    score = model.score(emb_reduced)
                    
                    # Use BIC for model selection
                    n_params = n_states * (n_states - 1) + n_states * emb_reduced.shape[1] * 2
                    bic = -2 * score + n_params * np.log(emb_reduced.shape[0])
                    
                    if best_model is None or bic < best_score:
                        best_model = model
                        best_score = bic
                        best_n_states = n_states
                except:
                    continue
                    
            if best_model is not None:
                # Get most likely state sequence
                states = best_model.predict(emb_reduced)
                
                # Find state transitions
                for i in range(1, len(states)):
                    if states[i] != states[i-1]:
                        # Calculate transition probability as confidence
                        trans_prob = best_model.transmat_[states[i-1], states[i]]
                        
                        all_phases.append({
                            'turn': i,
                            'confidence': trans_prob,
                            'method': 'hmm',
                            'model': model_name,
                            'from_state': int(states[i-1]),
                            'to_state': int(states[i])
                        })
                        
        return all_phases
        
    def _detect_by_spectral(self, embeddings: Dict[str, np.ndarray]) -> List[Dict]:
        """
        Use spectral methods to detect regime changes.
        """
        all_phases = []
        
        for model_name, emb in embeddings.items():
            if emb.shape[0] < 50:
                continue
                
            # Calculate trajectory in embedding space
            trajectory = []
            for i in range(1, len(emb)):
                trajectory.append(np.linalg.norm(emb[i] - emb[i-1]))
            trajectory = np.array(trajectory)
            
            # Apply wavelet transform
            try:
                # Use continuous wavelet transform
                widths = np.arange(5, min(30, len(trajectory) // 4))
                cwt_matrix = signal.cwt(trajectory, signal.ricker, widths)
                
                # Find ridges (local maxima across scales)
                ridges = signal.find_peaks_cwt(
                    trajectory, 
                    widths, 
                    min_snr=2,
                    noise_perc=10
                )
                
                for ridge in ridges:
                    if 0 < ridge < len(trajectory) - 1:
                        # Estimate confidence from ridge strength
                        ridge_strength = np.max(np.abs(cwt_matrix[:, ridge]))
                        confidence = min(1.0, ridge_strength / np.mean(np.abs(cwt_matrix)))
                        
                        all_phases.append({
                            'turn': ridge,
                            'confidence': confidence,
                            'method': 'spectral_wavelet',
                            'model': model_name
                        })
            except Exception as e:
                logger.warning(f"Spectral method failed for {model_name}: {e}")
                
            # Also try Fourier-based change detection
            try:
                # Short-time Fourier transform
                window_size = min(20, len(trajectory) // 5)
                if window_size > 10:
                    f, t, Sxx = signal.spectrogram(
                        trajectory, 
                        nperseg=window_size,
                        noverlap=window_size//2
                    )
                    
                    # Detect changes in spectral content
                    spectral_change = np.diff(np.mean(Sxx, axis=0))
                    change_peaks, _ = signal.find_peaks(
                        np.abs(spectral_change),
                        height=np.std(spectral_change) * 2
                    )
                    
                    for peak in change_peaks:
                        turn = int(t[peak] * len(trajectory) / t[-1])
                        all_phases.append({
                            'turn': turn,
                            'confidence': 0.6,
                            'method': 'spectral_fourier',
                            'model': model_name
                        })
            except:
                pass
                
        return all_phases
        
    def _detect_by_clustering(self, embeddings: Dict[str, np.ndarray]) -> List[Dict]:
        """
        Use clustering in sliding windows to detect phase changes.
        """
        all_phases = []
        
        for model_name, emb in embeddings.items():
            if emb.shape[0] < 30:
                continue
                
            window_size = 20
            step_size = 5
            
            # Calculate clustering metrics for sliding windows
            cluster_labels = []
            
            for i in range(0, len(emb) - window_size, step_size):
                window_emb = emb[i:i+window_size]
                
                # Use DBSCAN for clustering
                try:
                    clustering = DBSCAN(eps=0.5, min_samples=3).fit(window_emb)
                    n_clusters = len(set(clustering.labels_)) - (1 if -1 in clustering.labels_ else 0)
                    cluster_labels.append(n_clusters)
                except:
                    cluster_labels.append(1)
                    
            if len(cluster_labels) < 3:
                continue
                
            # Smooth the cluster count signal
            cluster_labels = gaussian_filter1d(cluster_labels, sigma=1)
            
            # Find significant changes in cluster structure
            cluster_changes = np.abs(np.diff(cluster_labels))
            threshold = np.mean(cluster_changes) + 2 * np.std(cluster_changes)
            
            change_points = np.where(cluster_changes > threshold)[0]
            
            for cp in change_points:
                turn = cp * step_size + window_size // 2
                confidence = min(1.0, cluster_changes[cp] / threshold - 1)
                
                all_phases.append({
                    'turn': turn,
                    'confidence': confidence,
                    'method': 'clustering',
                    'model': model_name
                })
                
        return all_phases
        
    def _select_penalty_elbow(self, signal_1d: np.ndarray, algo) -> float:
        """
        Select penalty using elbow method.
        """
        penalties = np.logspace(0, 4, 20)
        n_changepoints = []
        
        for pen in penalties:
            try:
                cps = algo.predict(pen=pen)
                n_changepoints.append(len(cps) - 1)
            except:
                n_changepoints.append(0)
                
        # Find elbow
        if len(set(n_changepoints)) > 1:
            # Simple elbow: where rate of change decreases most
            diffs = np.diff(n_changepoints)
            if len(diffs) > 0 and np.any(diffs < 0):
                elbow_idx = np.argmax(diffs) + 1
                return penalties[elbow_idx]
                
        return penalties[len(penalties)//2]  # Default to middle
        
    def _combine_phase_detections(self, 
                                 method_results: Dict[str, List[Dict]], 
                                 n_messages: int) -> List[Dict]:
        """
        Combine phase detections from multiple methods using consensus voting.
        """
        # Collect all detected phase points
        all_turns = []
        for method, phases in method_results.items():
            for phase in phases:
                all_turns.append(phase['turn'])
                
        if not all_turns:
            return []
            
        # Use kernel density estimation to find consensus peaks
        from scipy.stats import gaussian_kde
        
        all_turns = np.array(all_turns)
        if len(all_turns) < 2:
            return [{'turn': int(all_turns[0]), 'confidence': 0.5, 'n_votes': 1}]
            
        kde = gaussian_kde(all_turns, bw_method=0.1)
        x = np.arange(0, n_messages)
        density = kde(x)
        
        # Find peaks in consensus density
        min_votes = len(method_results) * 0.3  # At least 30% of methods agree
        peaks, properties = signal.find_peaks(
            density,
            height=min_votes / n_messages,
            distance=15
        )
        
        # Create consensus phases
        consensus_phases = []
        for i, peak in enumerate(peaks):
            # Count supporting methods
            support_count = 0
            confidence_sum = 0
            
            for method, phases in method_results.items():
                for phase in phases:
                    if abs(phase['turn'] - peak) <= 10:  # Within 10 turns
                        support_count += 1
                        confidence_sum += phase.get('confidence', 0.5)
                        break
                        
            if support_count >= min_votes:
                consensus_phases.append({
                    'turn': int(peak),
                    'confidence': confidence_sum / support_count,
                    'n_votes': support_count,
                    'density': density[peak]
                })
                
        return sorted(consensus_phases, key=lambda x: x['turn'])
        
    def _calculate_method_agreement(self, method_results: Dict[str, List[Dict]]) -> Dict:
        """
        Calculate agreement between different detection methods.
        """
        methods = list(method_results.keys())
        n_methods = len(methods)
        
        if n_methods < 2:
            return {'error': 'Need at least 2 methods for agreement'}
            
        # Pairwise agreement using Jaccard index with tolerance
        agreement_matrix = np.ones((n_methods, n_methods))
        tolerance = 10  # turns
        
        for i in range(n_methods):
            for j in range(i+1, n_methods):
                phases1 = method_results[methods[i]]
                phases2 = method_results[methods[j]]
                
                if not phases1 or not phases2:
                    agreement_matrix[i, j] = 0
                    agreement_matrix[j, i] = 0
                    continue
                    
                # Count matches
                matches = 0
                for p1 in phases1:
                    for p2 in phases2:
                        if abs(p1['turn'] - p2['turn']) <= tolerance:
                            matches += 1
                            break
                            
                jaccard = matches / (len(phases1) + len(phases2) - matches)
                agreement_matrix[i, j] = jaccard
                agreement_matrix[j, i] = jaccard
                
        return {
            'agreement_matrix': agreement_matrix,
            'mean_agreement': np.mean(agreement_matrix[np.triu_indices_from(agreement_matrix, k=1)]),
            'method_pairs': [(methods[i], methods[j], agreement_matrix[i, j]) 
                           for i in range(n_methods) for j in range(i+1, n_methods)]
        }
        
    def _evaluate_detection(self, 
                           detected_phases: List[Dict],
                           annotated_phases: List[Dict]) -> Dict:
        """
        Evaluate detection accuracy against ground truth.
        """
        if not annotated_phases:
            return {'error': 'No annotated phases provided'}
            
        # Extract turn numbers
        detected_turns = [p['turn'] for p in detected_phases]
        annotated_turns = [p.get('turn', p.get('start_turn', 0)) for p in annotated_phases]
        
        # Calculate metrics with tolerance
        tolerance = 10
        true_positives = 0
        
        for ann_turn in annotated_turns:
            for det_turn in detected_turns:
                if abs(ann_turn - det_turn) <= tolerance:
                    true_positives += 1
                    break
                    
        precision = true_positives / len(detected_turns) if detected_turns else 0
        recall = true_positives / len(annotated_turns) if annotated_turns else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # Calculate timing errors for matched phases
        timing_errors = []
        for ann_turn in annotated_turns:
            min_error = float('inf')
            for det_turn in detected_turns:
                error = abs(ann_turn - det_turn)
                if error <= tolerance:
                    min_error = min(min_error, error)
            if min_error <= tolerance:
                timing_errors.append(min_error)
                
        return {
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'true_positives': true_positives,
            'false_positives': len(detected_turns) - true_positives,
            'false_negatives': len(annotated_turns) - true_positives,
            'timing_errors': {
                'mean': np.mean(timing_errors) if timing_errors else float('inf'),
                'std': np.std(timing_errors) if timing_errors else 0,
                'median': np.median(timing_errors) if timing_errors else float('inf')
            }
        }