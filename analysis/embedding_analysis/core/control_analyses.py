"""
Control Analyses for Hypothesis Testing

This module implements control analyses to validate that observed patterns
are not artifacts of message length, conversation type, or other confounds.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from scipy import stats
from sklearn.preprocessing import StandardScaler
import logging

logger = logging.getLogger(__name__)


class ControlAnalyses:
    """
    Implement control analyses for validating geometric invariance.
    """
    
    def __init__(self):
        self.scaler = StandardScaler()
    
    def control_for_message_length(self, embeddings: np.ndarray, 
                                 message_lengths: np.ndarray,
                                 correlation_func) -> Dict:
        """
        Control for message length effects on correlations.
        
        Args:
            embeddings: Embedding vectors
            message_lengths: Length of each message in tokens/characters
            correlation_func: Function to compute correlations
            
        Returns:
            Dict with partial correlations and statistics
        """
        results = {}
        
        # Compute raw correlation
        raw_corr = correlation_func(embeddings)
        results['raw_correlation'] = raw_corr
        
        # Compute correlation with message length
        # First, compute trajectory metrics that might correlate with length
        trajectory_distances = np.linalg.norm(np.diff(embeddings, axis=0), axis=1)
        
        # Correlation between trajectory distance and message length
        # Handle case where message lengths have no variation
        length_diffs = np.diff(message_lengths)
        if np.std(length_diffs) == 0:
            # No variation in message length differences
            length_corr = 0.0
            length_p = 1.0
        else:
            length_corr, length_p = stats.pearsonr(
                trajectory_distances[:len(message_lengths)-1],
                length_diffs
            )
            # Handle NaN values
            if np.isnan(length_corr):
                length_corr = 0.0
                length_p = 1.0
        
        results['length_correlation'] = length_corr
        results['length_p_value'] = length_p
        
        # Compute partial correlation controlling for length
        # Using regression approach
        from sklearn.linear_model import LinearRegression
        
        # Regress out length effects
        reg = LinearRegression()
        
        # For each embedding dimension, remove length effects
        embeddings_controlled = embeddings.copy()
        for dim in range(embeddings.shape[1]):
            reg.fit(message_lengths.reshape(-1, 1), embeddings[:, dim])
            residuals = embeddings[:, dim] - reg.predict(message_lengths.reshape(-1, 1))
            embeddings_controlled[:, dim] = residuals
        
        # Compute correlation on length-controlled embeddings
        controlled_corr = correlation_func(embeddings_controlled)
        results['partial_correlation'] = controlled_corr
        results['n_conversations'] = len(embeddings)
        
        # Effect of controlling
        results['correlation_change'] = abs(raw_corr - controlled_corr)
        results['relative_change'] = results['correlation_change'] / abs(raw_corr) if raw_corr != 0 else 0
        
        return results
    
    def analyze_by_conversation_type(self, conversations_by_type: Dict[str, List],
                                   analysis_func) -> Dict:
        """
        Analyze patterns separately by conversation type.
        
        Args:
            conversations_by_type: Dict mapping type to list of conversations
            analysis_func: Function to analyze each conversation set
            
        Returns:
            Dict with per-type results and cross-type comparisons
        """
        results = {
            'per_type': {},
            'cross_type_comparison': {}
        }
        
        # Analyze each type
        all_correlations = {}
        for conv_type, conversations in conversations_by_type.items():
            if len(conversations) < 5:  # Need minimum conversations
                logger.warning(f"Skipping type {conv_type} - only {len(conversations)} conversations")
                continue
                
            type_results = analysis_func(conversations)
            results['per_type'][conv_type] = type_results
            
            # Collect correlations for comparison
            if 'mean_correlation' in type_results:
                all_correlations[conv_type] = type_results['correlations']
        
        # Compare across types
        if len(all_correlations) >= 2:
            # Kruskal-Wallis test for differences
            h_stat, p_value = stats.kruskal(*all_correlations.values())
            results['cross_type_comparison']['kruskal_h'] = h_stat
            results['cross_type_comparison']['p_value'] = p_value
            
            # Pairwise comparisons
            type_names = list(all_correlations.keys())
            pairwise = {}
            
            for i in range(len(type_names)):
                for j in range(i+1, len(type_names)):
                    type1, type2 = type_names[i], type_names[j]
                    u_stat, p_val = stats.mannwhitneyu(
                        all_correlations[type1],
                        all_correlations[type2],
                        alternative='two-sided'
                    )
                    pairwise[f"{type1}_vs_{type2}"] = {
                        'u_statistic': u_stat,
                        'p_value': p_val,
                        'effect_size': self._compute_rank_biserial(
                            all_correlations[type1],
                            all_correlations[type2]
                        )
                    }
            
            results['cross_type_comparison']['pairwise'] = pairwise
            
            # Check if patterns are consistent
            mean_corrs = [np.mean(corrs) for corrs in all_correlations.values()]
            results['cross_type_comparison']['consistency'] = {
                'mean_of_means': np.mean(mean_corrs),
                'std_of_means': np.std(mean_corrs),
                'cv': np.std(mean_corrs) / np.mean(mean_corrs) if np.mean(mean_corrs) > 0 else np.inf
            }
        
        return results
    
    def _compute_rank_biserial(self, group1: np.ndarray, group2: np.ndarray) -> float:
        """Compute rank-biserial correlation as effect size."""
        from scipy.stats import rankdata
        
        # Combine and rank
        combined = np.concatenate([group1, group2])
        ranks = rankdata(combined)
        
        # Sum of ranks for group 1
        n1 = len(group1)
        n2 = len(group2)
        r1 = np.sum(ranks[:n1])
        
        # Rank-biserial correlation
        rb = 1 - (2 * r1) / (n1 * (n1 + n2 + 1))
        return rb
    
    def temporal_stability_analysis(self, conversations: List[Dict],
                                  window_size: int = 50) -> Dict:
        """
        Analyze temporal stability of patterns over conversation sequence.
        
        Args:
            conversations: List of conversations with timestamps/order
            window_size: Size of rolling window
            
        Returns:
            Dict with temporal stability metrics
        """
        results = {
            'rolling_correlations': [],
            'timestamps': [],
            'stability_metrics': {}
        }
        
        # Sort by timestamp if available
        if 'timestamp' in conversations[0]:
            conversations = sorted(conversations, key=lambda x: x['timestamp'])
        
        # Compute rolling window correlations
        n_windows = len(conversations) - window_size + 1
        
        for i in range(n_windows):
            window_convs = conversations[i:i+window_size]
            
            # Compute correlation for this window
            window_corr = self._compute_window_correlation(window_convs)
            results['rolling_correlations'].append(window_corr)
            
            # Track timestamp
            if 'timestamp' in conversations[0]:
                results['timestamps'].append(conversations[i+window_size//2]['timestamp'])
        
        # Analyze stability
        rolling_corrs = np.array(results['rolling_correlations'])
        
        # Augmented Dickey-Fuller test for stationarity
        from statsmodels.tsa.stattools import adfuller
        adf_result = adfuller(rolling_corrs)
        
        results['stability_metrics']['adf_statistic'] = adf_result[0]
        results['stability_metrics']['adf_p_value'] = adf_result[1]
        results['stability_metrics']['is_stationary'] = adf_result[1] < 0.05
        
        # Trend analysis
        x = np.arange(len(rolling_corrs))
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, rolling_corrs)
        
        results['stability_metrics']['trend'] = {
            'slope': slope,
            'p_value': p_value,
            'r_squared': r_value**2
        }
        
        # Variance over time
        results['stability_metrics']['rolling_std'] = pd.Series(rolling_corrs).rolling(10).std().values
        
        return results
    
    def _compute_window_correlation(self, conversations: List[Dict]) -> float:
        """Compute average correlation for a window of conversations."""
        correlations = []
        
        for conv in conversations:
            if 'invariance_score' in conv:
                correlations.append(conv['invariance_score'])
            elif 'correlation' in conv:
                correlations.append(conv['correlation'])
        
        return np.mean(correlations) if correlations else np.nan
    
    def outlier_robustness_check(self, all_correlations: np.ndarray,
                                outlier_threshold: float = 3.0) -> Dict:
        """
        Check robustness of results to outlier conversations.
        
        Args:
            all_correlations: Array of correlation values
            outlier_threshold: Z-score threshold for outliers
            
        Returns:
            Dict with robustness metrics
        """
        results = {}
        
        # Identify outliers using z-score
        z_scores = np.abs(stats.zscore(all_correlations))
        outlier_mask = z_scores > outlier_threshold
        
        n_outliers = np.sum(outlier_mask)
        results['n_outliers'] = n_outliers
        results['outlier_percentage'] = 100 * n_outliers / len(all_correlations)
        
        # Statistics with and without outliers
        results['with_outliers'] = {
            'mean': np.mean(all_correlations),
            'median': np.median(all_correlations),
            'std': np.std(all_correlations)
        }
        
        clean_correlations = all_correlations[~outlier_mask]
        results['without_outliers'] = {
            'mean': np.mean(clean_correlations),
            'median': np.median(clean_correlations),
            'std': np.std(clean_correlations)
        }
        
        # Change in statistics
        results['outlier_impact'] = {
            'mean_change': abs(results['with_outliers']['mean'] - 
                             results['without_outliers']['mean']),
            'relative_mean_change': abs(results['with_outliers']['mean'] - 
                                      results['without_outliers']['mean']) / 
                                   results['with_outliers']['mean']
        }
        
        # Test if conclusions change
        threshold = 0.5  # Hypothesis threshold
        results['conclusion_robust'] = (
            (results['with_outliers']['mean'] > threshold) ==
            (results['without_outliers']['mean'] > threshold)
        )
        
        # Identify specific outlier conversations
        if n_outliers > 0:
            outlier_indices = np.where(outlier_mask)[0]
            results['outlier_details'] = {
                'indices': outlier_indices.tolist(),
                'values': all_correlations[outlier_mask].tolist(),
                'z_scores': z_scores[outlier_mask].tolist()
            }
        
        return results
    
    def cross_validation_split(self, conversations: List[Dict],
                             n_folds: int = 5,
                             stratify_by: Optional[str] = None) -> List[Tuple]:
        """
        Create cross-validation splits for robustness testing.
        
        Args:
            conversations: List of all conversations
            n_folds: Number of CV folds
            stratify_by: Optional field to stratify by
            
        Returns:
            List of (train, test) index tuples
        """
        n_conversations = len(conversations)
        indices = np.arange(n_conversations)
        
        if stratify_by and stratify_by in conversations[0]:
            # Stratified split
            from sklearn.model_selection import StratifiedKFold
            
            labels = [conv[stratify_by] for conv in conversations]
            skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
            splits = list(skf.split(indices, labels))
        else:
            # Regular k-fold
            from sklearn.model_selection import KFold
            
            kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
            splits = list(kf.split(indices))
        
        return splits
    
    def bootstrap_confidence_intervals(self, data: np.ndarray,
                                     statistic_func,
                                     n_bootstrap: int = 1000,
                                     confidence: float = 0.95) -> Dict:
        """
        Compute bootstrap confidence intervals for any statistic.
        
        Args:
            data: Input data array
            statistic_func: Function to compute statistic
            n_bootstrap: Number of bootstrap samples
            confidence: Confidence level
            
        Returns:
            Dict with bootstrap results
        """
        # Original statistic
        original_stat = statistic_func(data)
        
        # Bootstrap
        bootstrap_stats = []
        n_samples = len(data)
        
        for _ in range(n_bootstrap):
            # Resample with replacement
            bootstrap_sample = np.random.choice(data, size=n_samples, replace=True)
            bootstrap_stats.append(statistic_func(bootstrap_sample))
        
        bootstrap_stats = np.array(bootstrap_stats)
        
        # Compute confidence intervals
        alpha = 1 - confidence
        lower_percentile = 100 * (alpha / 2)
        upper_percentile = 100 * (1 - alpha / 2)
        
        results = {
            'original_statistic': original_stat,
            'bootstrap_mean': np.mean(bootstrap_stats),
            'bootstrap_std': np.std(bootstrap_stats),
            'confidence_interval': {
                'lower': np.percentile(bootstrap_stats, lower_percentile),
                'upper': np.percentile(bootstrap_stats, upper_percentile),
                'confidence': confidence
            },
            'bias': np.mean(bootstrap_stats) - original_stat,
            'bootstrap_distribution': bootstrap_stats
        }
        
        return results