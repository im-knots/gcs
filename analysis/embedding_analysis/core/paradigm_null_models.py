"""
Paradigm-Specific Null Models for Hypothesis Testing

This module implements null models that account for the fundamental differences
between embedding paradigms (e.g., averaging vs. CLS token, dimensionality).
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from scipy import signal
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class ParadigmSpecificNullModels:
    """
    Generate null models that respect paradigm-specific constraints.
    """
    
    def __init__(self, seed: Optional[int] = None):
        self.rng = np.random.RandomState(seed)
        
    def phase_scramble_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Phase scramble embeddings to preserve frequency content but destroy temporal structure.
        
        Args:
            embeddings: (n_messages, embedding_dim) array
            
        Returns:
            Phase-scrambled embeddings
        """
        n_messages, embedding_dim = embeddings.shape
        
        # Apply FFT to each dimension
        scrambled = np.zeros_like(embeddings)
        
        for dim in range(embedding_dim):
            # Get the signal for this dimension
            signal_1d = embeddings[:, dim]
            
            # Apply FFT
            fft = np.fft.fft(signal_1d)
            
            # Randomize phases while preserving magnitudes
            magnitudes = np.abs(fft)
            phases = np.angle(fft)
            
            # Generate random phases (-pi to pi)
            # Keep DC component phase unchanged
            random_phases = self.rng.uniform(-np.pi, np.pi, size=len(phases))
            random_phases[0] = phases[0]  # Preserve DC
            
            # For real signals, ensure conjugate symmetry
            if n_messages % 2 == 0:
                # Even length - special handling for Nyquist frequency
                random_phases[n_messages//2] = 0  # Nyquist frequency must have phase 0 or pi
                random_phases[n_messages//2+1:] = -random_phases[1:n_messages//2][::-1]
            else:
                # Odd length
                random_phases[n_messages//2+1:] = -random_phases[1:n_messages//2+1][::-1]
            
            # Reconstruct with randomized phases
            fft_scrambled = magnitudes * np.exp(1j * random_phases)
            
            # Inverse FFT
            scrambled[:, dim] = np.real(np.fft.ifft(fft_scrambled))
        
        return scrambled
    
    def generate_averaging_null(self, embeddings: np.ndarray, 
                               window_sizes: List[int] = [1, 3, 5, 10]) -> Dict[str, np.ndarray]:
        """
        Generate null model for averaging-based embeddings (Word2Vec/GloVe).
        
        Accounts for the fact that these models average word embeddings,
        losing positional information.
        
        Args:
            embeddings: Original embeddings
            window_sizes: Different averaging windows to test
            
        Returns:
            Dict of null embeddings with different averaging windows
        """
        nulls = {}
        
        # Shuffle then apply different averaging windows
        shuffled = embeddings.copy()
        self.rng.shuffle(shuffled)
        
        for window in window_sizes:
            if window == 1:
                nulls[f'avg_window_{window}'] = shuffled
            else:
                # Apply moving average
                averaged = np.zeros_like(shuffled)
                for i in range(len(shuffled)):
                    start = max(0, i - window // 2)
                    end = min(len(shuffled), i + window // 2 + 1)
                    averaged[i] = np.mean(shuffled[start:end], axis=0)
                nulls[f'avg_window_{window}'] = averaged
        
        return nulls
    
    def generate_positional_null(self, embeddings: np.ndarray,
                                max_position: int = 512) -> np.ndarray:
        """
        Generate null model for transformer embeddings with positional encoding.
        
        Accounts for the fact that transformers use positional encodings.
        
        Args:
            embeddings: Original embeddings
            max_position: Maximum position for encoding
            
        Returns:
            Null embeddings with randomized positions
        """
        n_messages, embedding_dim = embeddings.shape
        
        # Generate random positional encodings
        positions = self.rng.randint(0, min(max_position, n_messages), size=n_messages)
        
        # Create sinusoidal position encodings (simplified)
        position_encodings = np.zeros((n_messages, embedding_dim))
        
        for pos in range(n_messages):
            for i in range(0, embedding_dim, 2):
                position_encodings[pos, i] = np.sin(positions[pos] / 
                                                   (10000 ** (i / embedding_dim)))
                if i + 1 < embedding_dim:
                    position_encodings[pos, i + 1] = np.cos(positions[pos] / 
                                                           (10000 ** (i / embedding_dim)))
        
        # Add position encodings to shuffled embeddings
        shuffled = embeddings.copy()
        self.rng.shuffle(shuffled)
        
        # Scale position encodings to match embedding magnitude
        scale = np.std(embeddings) * 0.1  # 10% of embedding magnitude
        
        return shuffled + scale * position_encodings
    
    def generate_dimension_matched_null(self, source_embeddings: np.ndarray,
                                      target_dim: int) -> np.ndarray:
        """
        Generate null embeddings matched to a specific dimensionality.
        
        Useful for comparing across different dimensional spaces (300 vs 384 vs 768).
        
        Args:
            source_embeddings: Original embeddings
            target_dim: Target dimensionality
            
        Returns:
            Dimension-matched null embeddings
        """
        n_messages, source_dim = source_embeddings.shape
        
        if source_dim == target_dim:
            # Just shuffle if dimensions match
            null = source_embeddings.copy()
            self.rng.shuffle(null)
            return null
        
        elif source_dim > target_dim:
            # Project down using random projection
            projection_matrix = self.rng.randn(source_dim, target_dim)
            # Normalize columns
            projection_matrix /= np.linalg.norm(projection_matrix, axis=0)
            
            # Shuffle then project
            shuffled = source_embeddings.copy()
            self.rng.shuffle(shuffled)
            
            return shuffled @ projection_matrix
        
        else:
            # Pad with noise to match higher dimension
            null = np.zeros((n_messages, target_dim))
            
            # Copy shuffled original
            shuffled = source_embeddings.copy()
            self.rng.shuffle(shuffled)
            null[:, :source_dim] = shuffled
            
            # Fill remaining dimensions with matched noise
            noise_std = np.std(source_embeddings)
            null[:, source_dim:] = self.rng.randn(n_messages, 
                                                  target_dim - source_dim) * noise_std
            
            return null
    
    def generate_conversation_structure_null(self, embeddings: np.ndarray,
                                           preserve_local: bool = True,
                                           local_window: int = 5) -> np.ndarray:
        """
        Generate null that preserves local structure but destroys global patterns.
        
        Args:
            embeddings: Original embeddings
            preserve_local: Whether to preserve local coherence
            local_window: Size of local window to preserve
            
        Returns:
            Structure-preserving null embeddings
        """
        n_messages = len(embeddings)
        
        if not preserve_local:
            # Complete shuffle
            null = embeddings.copy()
            self.rng.shuffle(null)
            return null
        
        # Preserve local structure
        null = np.zeros_like(embeddings)
        
        # Divide into chunks
        n_chunks = n_messages // local_window
        remainder = n_messages % local_window
        
        # Get chunk indices
        chunk_indices = list(range(n_chunks))
        self.rng.shuffle(chunk_indices)
        
        # Rearrange chunks
        output_idx = 0
        for chunk_idx in chunk_indices:
            start = chunk_idx * local_window
            end = start + local_window
            
            # Copy chunk
            null[output_idx:output_idx + local_window] = embeddings[start:end]
            output_idx += local_window
        
        # Handle remainder
        if remainder > 0:
            null[output_idx:] = embeddings[-remainder:]
        
        return null
    
    def generate_paradigm_specific_ensemble(self, embeddings_dict: Dict[str, np.ndarray],
                                          n_samples: int = 100) -> Dict[str, List[np.ndarray]]:
        """
        Generate ensemble of paradigm-specific nulls for multiple models.
        
        Args:
            embeddings_dict: Dict mapping model names to embeddings
            n_samples: Number of null samples to generate
            
        Returns:
            Dict mapping model names to lists of null embeddings
        """
        null_ensemble = {}
        
        for model_name, embeddings in embeddings_dict.items():
            nulls = []
            
            for i in range(n_samples):
                # Set seed for reproducibility within sample
                self.rng.seed(i)
                
                if 'word2vec' in model_name.lower() or 'glove' in model_name.lower():
                    # Classical models - use averaging null
                    avg_nulls = self.generate_averaging_null(embeddings)
                    # Pick a random window size
                    window = self.rng.choice([1, 3, 5, 10])
                    null = avg_nulls[f'avg_window_{window}']
                
                elif any(trans in model_name.lower() for trans in ['bert', 'roberta', 'mpnet']):
                    # Transformer models - use positional null
                    null = self.generate_positional_null(embeddings)
                
                else:
                    # Unknown model - use phase scrambling
                    null = self.phase_scramble_embeddings(embeddings)
                
                nulls.append(null)
            
            null_ensemble[model_name] = nulls
        
        return null_ensemble
    
    def test_null_validity(self, original: np.ndarray, null: np.ndarray) -> Dict[str, float]:
        """
        Test that null model preserves desired properties while destroying structure.
        
        Args:
            original: Original embeddings
            null: Null model embeddings
            
        Returns:
            Dictionary of validity metrics
        """
        metrics = {}
        
        # Check mean preservation (should be similar)
        metrics['mean_difference'] = np.abs(np.mean(original) - np.mean(null))
        
        # Check variance preservation (should be similar)
        metrics['variance_ratio'] = np.var(null) / np.var(original)
        
        # Check temporal autocorrelation (should be destroyed)
        def autocorr(x, lag=1):
            n = len(x)
            c0 = np.dot(x[:-lag], x[:-lag])
            c1 = np.dot(x[:-lag], x[lag:])
            return c1 / c0 if c0 > 0 else 0
        
        # Average autocorrelation across dimensions
        orig_autocorr = np.mean([autocorr(original[:, i]) for i in range(original.shape[1])])
        null_autocorr = np.mean([autocorr(null[:, i]) for i in range(null.shape[1])])
        
        metrics['autocorr_original'] = orig_autocorr
        metrics['autocorr_null'] = null_autocorr
        metrics['autocorr_reduction'] = (orig_autocorr - null_autocorr) / orig_autocorr
        
        # Check spectral properties (for phase scrambling)
        # Power spectrum should be preserved
        orig_power = np.mean([np.abs(np.fft.fft(original[:, i]))**2 
                             for i in range(original.shape[1])])
        null_power = np.mean([np.abs(np.fft.fft(null[:, i]))**2 
                             for i in range(null.shape[1])])
        
        metrics['power_ratio'] = null_power / orig_power
        
        return metrics


class MessageLevelNullModels:
    """
    Null models that operate at the message level rather than embedding level.
    """
    
    def __init__(self, seed: Optional[int] = None):
        self.rng = np.random.RandomState(seed)
    
    def shuffle_within_speakers(self, messages: List[Dict], embeddings: np.ndarray) -> np.ndarray:
        """
        Shuffle messages within each speaker, preserving speaker patterns.
        
        Args:
            messages: List of message dictionaries with 'role' field
            embeddings: Corresponding embeddings
            
        Returns:
            Reordered embeddings
        """
        # Group by speaker
        speaker_indices = {}
        for i, msg in enumerate(messages):
            role = msg.get('role', 'unknown')
            if role not in speaker_indices:
                speaker_indices[role] = []
            speaker_indices[role].append(i)
        
        # Create mapping of positions to roles
        position_to_role = {msg.get('position', i): msg.get('role', 'unknown') 
                           for i, msg in enumerate(messages)}
        
        # Shuffle indices within each speaker group
        shuffled_by_role = {}
        for role, indices in speaker_indices.items():
            shuffled = indices.copy()
            self.rng.shuffle(shuffled)
            shuffled_by_role[role] = shuffled
        
        # Reconstruct maintaining alternation pattern
        role_counters = {role: 0 for role in speaker_indices}
        new_order = []
        
        for pos in sorted(position_to_role.keys()):
            role = position_to_role[pos]
            if role_counters[role] < len(shuffled_by_role[role]):
                new_order.append(shuffled_by_role[role][role_counters[role]])
                role_counters[role] += 1
        
        return embeddings[new_order]
    
    def reverse_conversation_flow(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Reverse conversation flow to test directional dependencies.
        
        Args:
            embeddings: Original embeddings
            
        Returns:
            Reversed embeddings
        """
        return embeddings[::-1].copy()
    
    def bootstrap_conversation_segments(self, embeddings: np.ndarray,
                                      segment_length: int = 10,
                                      n_samples: int = 100) -> List[np.ndarray]:
        """
        Bootstrap conversation segments to create null distribution.
        
        Args:
            embeddings: Original embeddings
            segment_length: Length of segments to bootstrap
            n_samples: Number of bootstrap samples
            
        Returns:
            List of bootstrapped embedding sequences
        """
        n_messages = len(embeddings)
        n_segments = n_messages // segment_length
        
        bootstrapped = []
        
        # Handle case where segment_length >= n_messages
        if n_segments == 0:
            # Just return copies of the original
            for _ in range(n_samples):
                bootstrapped.append(embeddings.copy())
            return bootstrapped
        
        for _ in range(n_samples):
            # Sample segments with replacement
            segment_indices = self.rng.choice(n_segments, size=n_segments, replace=True)
            
            # Reconstruct conversation
            new_embeddings = []
            for seg_idx in segment_indices:
                start = seg_idx * segment_length
                end = min(start + segment_length, n_messages)
                new_embeddings.append(embeddings[start:end])
            
            bootstrapped.append(np.vstack(new_embeddings))
        
        return bootstrapped