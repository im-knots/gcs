#!/usr/bin/env python3
"""
Unit tests for ControlAnalyses class.

Tests control analyses for message length, conversation type, and robustness checks.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from embedding_analysis.core.control_analyses import ControlAnalyses
from tests.test_utils import TestDataGenerator, assert_correlation_in_range


class TestControlAnalyses:
    """Test the ControlAnalyses implementation."""
    
    @pytest.fixture
    def analyzer(self):
        """Create a ControlAnalyses instance."""
        return ControlAnalyses()
    
    @pytest.fixture
    def data_generator(self):
        """Create test data generator."""
        return TestDataGenerator(seed=42)
    
    @pytest.fixture
    def sample_embeddings(self, data_generator):
        """Generate sample embeddings."""
        return data_generator.generate_conversation_embeddings(n_messages=100)
    
    def test_control_for_message_length(self, analyzer, sample_embeddings):
        """Test controlling for message length effects."""
        # Generate message lengths that correlate with trajectory
        n_messages = len(sample_embeddings)
        message_lengths = np.arange(n_messages) * 2 + np.random.randn(n_messages) * 5 + 20
        message_lengths = np.maximum(message_lengths, 5)  # Ensure positive
        
        # Mock correlation function
        def mock_correlation_func(embeddings):
            # Return correlation that would be affected by length
            return 0.8 + np.mean(embeddings) * 0.01
        
        result = analyzer.control_for_message_length(
            sample_embeddings,
            message_lengths,
            mock_correlation_func
        )
        
        # Check result structure
        assert 'raw_correlation' in result
        assert 'partial_correlation' in result
        assert 'length_correlation' in result
        assert 'correlation_change' in result
        assert 'relative_change' in result
        
        # Check that controlling changed the correlation
        assert result['correlation_change'] > 0
        assert 0 <= result['relative_change'] <= 1
        
        # Check length correlation is computed
        assert -1 <= result['length_correlation'] <= 1
        assert result['length_p_value'] >= 0
    
    def test_analyze_by_conversation_type(self, analyzer, data_generator):
        """Test analysis by conversation type."""
        # Create conversations of different types
        conversations_by_type = {
            'full_reasoning': [
                {'correlation': 0.85 + np.random.normal(0, 0.05)} 
                for _ in range(20)
            ],
            'light_reasoning': [
                {'correlation': 0.75 + np.random.normal(0, 0.05)} 
                for _ in range(20)
            ],
            'no_reasoning': [
                {'correlation': 0.65 + np.random.normal(0, 0.05)} 
                for _ in range(20)
            ]
        }
        
        def analysis_func(conversations):
            corrs = [c['correlation'] for c in conversations]
            return {
                'correlations': corrs,
                'mean_correlation': np.mean(corrs)
            }
        
        result = analyzer.analyze_by_conversation_type(
            conversations_by_type,
            analysis_func
        )
        
        # Check structure
        assert 'per_type' in result
        assert 'cross_type_comparison' in result
        
        # Check per-type results
        for conv_type in conversations_by_type:
            assert conv_type in result['per_type']
            assert 'mean_correlation' in result['per_type'][conv_type]
        
        # Check cross-type comparison
        comparison = result['cross_type_comparison']
        assert 'kruskal_h' in comparison
        assert 'p_value' in comparison
        assert comparison['p_value'] < 0.05  # Should detect differences
        
        # Check pairwise comparisons
        assert 'pairwise' in comparison
        assert 'full_reasoning_vs_no_reasoning' in comparison['pairwise']
        
        # Check consistency metrics
        assert 'consistency' in comparison
        assert 'cv' in comparison['consistency']
        assert comparison['consistency']['cv'] < 0.5  # Reasonable variation
    
    def test_temporal_stability_analysis(self, analyzer):
        """Test temporal stability analysis."""
        # Create conversations with temporal trend
        n_conversations = 100
        conversations = []
        
        for i in range(n_conversations):
            # Add slight upward trend in correlations
            base_corr = 0.7 + (i / n_conversations) * 0.1
            conversations.append({
                'timestamp': i * 3600,  # Hourly intervals
                'invariance_score': base_corr + np.random.normal(0, 0.05)
            })
        
        result = analyzer.temporal_stability_analysis(
            conversations,
            window_size=20
        )
        
        # Check structure
        assert 'rolling_correlations' in result
        assert 'timestamps' in result
        assert 'stability_metrics' in result
        
        # Check rolling correlations computed
        assert len(result['rolling_correlations']) == n_conversations - 20 + 1
        
        # Check stability metrics
        metrics = result['stability_metrics']
        assert 'adf_statistic' in metrics
        assert 'adf_p_value' in metrics
        assert 'is_stationary' in metrics
        
        # Check trend detection
        assert 'trend' in metrics
        assert metrics['trend']['slope'] > 0  # Should detect upward trend
        assert metrics['trend']['p_value'] < 0.05  # Should be significant
    
    def test_outlier_robustness_check(self, analyzer):
        """Test outlier robustness checking."""
        # Create correlations with some outliers
        n_samples = 100
        correlations = np.random.normal(0.8, 0.05, n_samples)
        
        # Add some outliers
        outlier_indices = [10, 25, 50]
        for idx in outlier_indices:
            correlations[idx] = 0.2  # Much lower than others
        
        result = analyzer.outlier_robustness_check(
            correlations,
            outlier_threshold=2.5
        )
        
        # Check structure
        assert 'n_outliers' in result
        assert 'outlier_percentage' in result
        assert 'with_outliers' in result
        assert 'without_outliers' in result
        assert 'outlier_impact' in result
        assert 'conclusion_robust' in result
        
        # Should detect outliers
        assert result['n_outliers'] >= len(outlier_indices)
        
        # Check statistics change
        assert result['with_outliers']['mean'] < result['without_outliers']['mean']
        assert result['outlier_impact']['mean_change'] > 0
        
        # Check outlier details if found
        if result['n_outliers'] > 0:
            assert 'outlier_details' in result
            assert 'indices' in result['outlier_details']
            assert 'values' in result['outlier_details']
            assert 'z_scores' in result['outlier_details']
    
    def test_cross_validation_split(self, analyzer, data_generator):
        """Test cross-validation splitting."""
        # Create conversations
        n_conversations = 50
        conversations = []
        
        for i in range(n_conversations):
            conv = data_generator.generate_conversation_data(n_messages=20)
            conversations.append(conv)
        
        # Test regular k-fold
        splits = analyzer.cross_validation_split(
            conversations,
            n_folds=5
        )
        
        assert len(splits) == 5
        
        # Check each split
        for train_idx, test_idx in splits:
            assert len(train_idx) + len(test_idx) == n_conversations
            assert len(set(train_idx) & set(test_idx)) == 0  # No overlap
            assert len(test_idx) == n_conversations // 5  # Roughly equal splits
        
        # Test stratified split
        splits_stratified = analyzer.cross_validation_split(
            conversations,
            n_folds=5,
            stratify_by='type'
        )
        
        assert len(splits_stratified) == 5
    
    def test_bootstrap_confidence_intervals(self, analyzer):
        """Test bootstrap confidence interval calculation."""
        # Create sample data
        data = np.random.normal(0.8, 0.1, 100)
        
        # Define statistic function (mean)
        def mean_func(x):
            return np.mean(x)
        
        result = analyzer.bootstrap_confidence_intervals(
            data,
            mean_func,
            n_bootstrap=500,
            confidence=0.95
        )
        
        # Check structure
        assert 'original_statistic' in result
        assert 'bootstrap_mean' in result
        assert 'bootstrap_std' in result
        assert 'confidence_interval' in result
        assert 'bias' in result
        assert 'bootstrap_distribution' in result
        
        # Check values
        assert np.isclose(result['original_statistic'], np.mean(data), rtol=0.01)
        assert np.isclose(result['bootstrap_mean'], np.mean(data), rtol=0.05)
        
        # Check confidence interval
        ci = result['confidence_interval']
        assert ci['lower'] < result['original_statistic'] < ci['upper']
        assert ci['confidence'] == 0.95
        
        # Check bootstrap distribution
        assert len(result['bootstrap_distribution']) == 500
        assert result['bootstrap_std'] > 0
    
    def test_message_length_edge_cases(self, analyzer):
        """Test edge cases for message length control."""
        # Test with no variation in lengths
        embeddings = np.random.randn(50, 100)
        constant_lengths = np.ones(50) * 20
        
        result = analyzer.control_for_message_length(
            embeddings,
            constant_lengths,
            lambda x: 0.8
        )
        
        # Should handle gracefully
        assert result['length_correlation'] == 0  # No correlation possible
        
        # Test with very few messages
        small_embeddings = np.random.randn(3, 100)
        small_lengths = np.array([10, 20, 30])
        
        result = analyzer.control_for_message_length(
            small_embeddings,
            small_lengths,
            lambda x: 0.8
        )
        
        assert result['n_conversations'] == 3
    
    def test_conversation_type_with_few_samples(self, analyzer):
        """Test conversation type analysis with few samples per type."""
        conversations_by_type = {
            'type1': [{'correlation': 0.8}],  # Only 1 sample
            'type2': [{'correlation': 0.7}, {'correlation': 0.75}],  # 2 samples
            'type3': [{'correlation': 0.6 + i*0.01} for i in range(10)]  # 10 samples
        }
        
        def analysis_func(conversations):
            corrs = [c['correlation'] for c in conversations]
            return {
                'correlations': corrs,
                'mean_correlation': np.mean(corrs)
            }
        
        result = analyzer.analyze_by_conversation_type(
            conversations_by_type,
            analysis_func
        )
        
        # Should skip type1 (too few samples)
        assert 'type1' not in result['per_type']
        assert 'type2' not in result['per_type']
        assert 'type3' in result['per_type']
    
    def test_temporal_stability_with_missing_timestamps(self, analyzer):
        """Test temporal stability when timestamps are missing."""
        conversations = []
        
        for i in range(50):
            conversations.append({
                'invariance_score': 0.7 + np.random.normal(0, 0.05)
                # No timestamp field
            })
        
        result = analyzer.temporal_stability_analysis(
            conversations,
            window_size=10
        )
        
        # Should still work, using order as proxy for time
        assert len(result['rolling_correlations']) == 41  # 50 - 10 + 1
        assert len(result['timestamps']) == 0  # No timestamps available


class TestControlAnalysesIntegration:
    """Integration tests for control analyses with other components."""
    
    def test_with_real_embeddings(self):
        """Test control analyses with realistic embedding data."""
        generator = TestDataGenerator()
        analyzer = ControlAnalyses()
        
        # Generate realistic conversation
        embeddings_dict = generator.generate_model_ensemble_embeddings(n_messages=100)
        first_model_embeddings = list(embeddings_dict.values())[0]
        
        # Generate correlated message lengths
        n_messages = len(first_model_embeddings)
        # Longer messages later in conversation
        message_lengths = np.linspace(10, 50, n_messages) + np.random.randn(n_messages) * 5
        message_lengths = np.maximum(message_lengths, 5)
        
        # Define correlation function that uses actual trajectory
        def trajectory_correlation(embeddings):
            velocities = np.linalg.norm(np.diff(embeddings, axis=0), axis=1)
            # Correlate with position (simulating length effect)
            positions = np.arange(len(velocities))
            return np.corrcoef(velocities, positions)[0, 1]
        
        result = analyzer.control_for_message_length(
            first_model_embeddings,
            message_lengths,
            trajectory_correlation
        )
        
        # Should show some length effect
        assert result['length_correlation'] != 0
        assert result['correlation_change'] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])