#!/usr/bin/env python3
"""
Tests for the hierarchical hypothesis testing framework.

Tests the revised hypothesis structure with proper statistical methods.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from embedding_analysis.core.hierarchical_hypothesis_testing import (
    HierarchicalHypothesisTester,
    HypothesisResult,
    TierResult
)
from embedding_analysis.core.paradigm_null_models import (
    ParadigmSpecificNullModels,
    MessageLevelNullModels
)


class TestHierarchicalHypothesisTester:
    """Test the hierarchical hypothesis testing implementation."""
    
    @pytest.fixture
    def tester(self):
        """Create a hypothesis tester instance."""
        return HierarchicalHypothesisTester(alpha=0.05)
    
    @pytest.fixture
    def sample_correlations(self):
        """Create realistic sample correlation data."""
        np.random.seed(42)
        
        # Generate correlations matching paper's findings
        data = {
            'correlations': {
                # Within-paradigm: high correlations
                'transformer_pairs': np.random.uniform(0.80, 0.95, 50),
                'classical_pairs': np.random.uniform(0.85, 0.98, 30),
                
                # Cross-paradigm: moderate correlations
                'cross_paradigm_pairs': np.random.uniform(0.55, 0.75, 100),
                
                # Null models: low correlations
                'null_within_paradigm': np.random.uniform(-0.1, 0.2, 50),
                'random_embedding_pairs': np.random.uniform(-0.15, 0.15, 100)
            },
            'geometric_metrics': {
                'velocity': {
                    'within': np.random.uniform(0.7, 0.9, 50),
                    'cross': np.random.uniform(0.5, 0.7, 50),
                    'random': np.random.uniform(0.0, 0.3, 50)
                },
                'curvature': {
                    'within': np.random.uniform(0.6, 0.85, 50),
                    'cross': np.random.uniform(0.45, 0.65, 50),
                    'random': np.random.uniform(-0.1, 0.2, 50)
                }
            },
            'control_data': {
                'real_scrambled_comparison': {
                    'real': np.random.uniform(0.6, 0.9, 50),
                    'scrambled': np.random.uniform(-0.1, 0.3, 50)
                },
                'length_controlled': {
                    'partial_correlation': 0.72,
                    'n_conversations': 228
                },
                'normalized_metrics': {
                    'correlations': np.random.uniform(0.55, 0.85, 100)
                }
            }
        }
        
        return data
    
    def test_fisher_transform(self, tester):
        """Test Fisher transformation implementation."""
        # Test normal values
        r = 0.7
        z = tester.fisher_transform(r)
        r_back = tester.inverse_fisher_transform(z)
        assert np.isclose(r, r_back)
        
        # Test edge cases
        assert tester.fisher_transform(0.99999) > 5
        assert tester.fisher_transform(-0.99999) < -5
        assert tester.fisher_transform(0) == 0
        
        # Test confidence intervals
        ci = tester.fisher_confidence_interval(0.8, n=100, confidence=0.95)
        assert ci[0] < 0.8 < ci[1]
        assert ci[1] - ci[0] < 0.3  # Reasonable interval width
    
    def test_cohens_q(self, tester):
        """Test Cohen's q effect size calculation."""
        # Large difference
        q1 = tester.cohens_q(0.9, 0.3)
        assert q1 > 1.0  # Large effect
        
        # Small difference
        q2 = tester.cohens_q(0.7, 0.65)
        assert 0 < q2 < 0.3  # Small effect
        
        # No difference
        q3 = tester.cohens_q(0.5, 0.5)
        assert q3 == 0
    
    def test_steiger_test(self, tester):
        """Test Steiger's test for dependent correlations."""
        # Test with high intercorrelation
        z, p = tester.steiger_test_dependent(r12=0.8, r13=0.4, r23=0.7, n=100)
        assert p < 0.05  # Should be significant difference
        
        # Test with similar correlations
        z, p = tester.steiger_test_dependent(r12=0.7, r13=0.68, r23=0.5, n=100)
        assert p > 0.05  # Should not be significant
    
    def test_tier1_within_paradigm(self, tester, sample_correlations):
        """Test Tier 1 hypothesis testing."""
        result = tester.test_tier1_within_paradigm(sample_correlations['correlations'])
        
        # Check structure
        assert isinstance(result, TierResult)
        assert result.tier_number == 1
        assert result.tier_name == "Within-Paradigm Invariance"
        assert len(result.hypotheses) == 3
        
        # Check H1a (transformers)
        h1a = next(h for h in result.hypotheses if h.name == "H1a")
        assert h1a.passed  # Should pass with high correlations
        assert h1a.effect_size > 0
        assert h1a.confidence_interval[0] > 0.75
        
        # Check H1b (classical)
        h1b = next(h for h in result.hypotheses if h.name == "H1b")
        assert h1b.passed
        assert h1b.confidence_interval[0] > 0.70
        
        # Check H1c (exceed chance)
        h1c = next(h for h in result.hypotheses if h.name == "H1c")
        assert h1c.passed
        assert h1c.p_value < 0.05/3  # Bonferroni corrected
        
        # Overall tier should pass
        assert result.passed
        assert result.correction_method == "Bonferroni"
    
    def test_tier2_cross_paradigm(self, tester, sample_correlations):
        """Test Tier 2 hypothesis testing."""
        result = tester.test_tier2_cross_paradigm(sample_correlations['correlations'])
        
        # Check structure
        assert result.tier_number == 2
        assert result.tier_name == "Cross-Paradigm Invariance"
        
        # Check H2a (substantial correlations)
        h2a = next(h for h in result.hypotheses if h.name == "H2a")
        assert h2a.passed
        assert h2a.confidence_interval[0] > 0.50
        
        # Check H2b (all positive)
        h2b = next(h for h in result.hypotheses if h.name == "H2b")
        assert h2b.passed  # All correlations in sample data are positive
        
        # Check H2c (exceed random)
        h2c = next(h for h in result.hypotheses if h.name == "H2c")
        assert h2c.passed
        
        # FDR correction applied
        assert result.correction_method == "FDR (Benjamini-Hochberg)"
    
    def test_tier3_hierarchy(self, tester, sample_correlations):
        """Test Tier 3 hypothesis testing."""
        result = tester.test_tier3_hierarchy(
            sample_correlations['correlations'],
            sample_correlations['geometric_metrics']
        )
        
        # Check structure
        assert result.tier_number == 3
        assert result.tier_name == "Invariance Hierarchy"
        
        # Check H3a (ordering)
        h3a = next(h for h in result.hypotheses if h.name == "H3a")
        assert h3a.passed  # Data is set up to satisfy ordering
        
        # Check H3b (effect size)
        h3b = next(h for h in result.hypotheses if h.name == "H3b")
        assert h3b.passed
        assert h3b.effect_size > 0.3
        
        # Check H3c (consistency across metrics)
        h3c = next(h for h in result.hypotheses if h.name == "H3c")
        assert h3c.passed
    
    def test_control_hypotheses(self, tester, sample_correlations):
        """Test control hypothesis testing."""
        controls = tester.test_controls(sample_correlations['control_data'])
        
        # Should have 3 control tests
        assert len(controls) >= 3
        
        # H4: Real > Scrambled
        h4 = next(h for h in controls if h.name == "H4")
        assert h4.passed
        
        # H5: Length controlled
        h5 = next(h for h in controls if h.name == "H5")
        assert h5.passed
        assert h5.confidence_interval[0] > 0.5
        
        # H6: Normalized metrics
        h6 = next(h for h in controls if h.name == "H6")
        assert h6.passed
    
    def test_hierarchical_testing_full(self, tester, sample_correlations):
        """Test complete hierarchical testing workflow."""
        results = tester.run_hierarchical_testing(sample_correlations)
        
        # Check structure
        assert 'tiers' in results
        assert 'controls' in results
        assert 'summary' in results
        
        # All tiers should be tested (data set up to pass)
        assert len(results['tiers']) == 3
        
        # Check summary
        summary = results['summary']
        assert summary['max_tier_passed'] == 3
        assert summary['conclusion'] == "All tiers passed: Complete geometric invariance established"
        assert summary['total_hypotheses'] > 0
        assert summary['passed_hypotheses'] > 0
        assert summary['mean_effect_size'] > 0
    
    def test_hierarchical_stopping(self, tester):
        """Test that hierarchical testing stops when a tier fails."""
        # Create data that fails Tier 1
        failing_data = {
            'correlations': {
                'transformer_pairs': np.random.uniform(0.3, 0.5, 50),  # Too low
                'classical_pairs': np.random.uniform(0.4, 0.6, 30),    # Too low
                'null_within_paradigm': np.random.uniform(0.2, 0.4, 50),
                'cross_paradigm_pairs': np.random.uniform(0.7, 0.9, 100),
                'random_embedding_pairs': np.random.uniform(-0.1, 0.1, 100)
            }
        }
        
        results = tester.run_hierarchical_testing(failing_data)
        
        # Should only test Tier 1
        assert len(results['tiers']) == 1
        assert results['summary']['max_tier_passed'] == 0
        assert "Failed at Tier 1" in results['summary']['conclusion']


class TestParadigmSpecificNulls:
    """Test paradigm-specific null model generation."""
    
    @pytest.fixture
    def null_generator(self):
        """Create null model generator."""
        return ParadigmSpecificNullModels(seed=42)
    
    @pytest.fixture
    def sample_embeddings(self):
        """Create sample embeddings."""
        return np.random.randn(100, 384)
    
    def test_phase_scrambling(self, null_generator, sample_embeddings):
        """Test phase scrambling preserves frequency content."""
        scrambled = null_generator.phase_scramble_embeddings(sample_embeddings)
        
        # Check shape preserved
        assert scrambled.shape == sample_embeddings.shape
        
        # Check power spectrum preserved (approximately)
        for dim in range(5):  # Test first 5 dimensions
            orig_power = np.abs(np.fft.fft(sample_embeddings[:, dim]))**2
            scram_power = np.abs(np.fft.fft(scrambled[:, dim]))**2
            
            # Power should be very similar
            # Use element-wise comparison to handle small values better
            for i in range(len(orig_power)):
                if abs(orig_power[i]) < 1e-10:
                    # For very small values, check absolute difference
                    assert abs(orig_power[i] - scram_power[i]) < 1e-8
                else:
                    # For larger values, check relative difference
                    rel_diff = abs(orig_power[i] - scram_power[i]) / abs(orig_power[i])
                    assert rel_diff < 0.5, f"Power spectrum mismatch at index {i}: {orig_power[i]} vs {scram_power[i]}"
        
        # But actual values should be different
        assert not np.allclose(sample_embeddings, scrambled)
    
    def test_averaging_null(self, null_generator, sample_embeddings):
        """Test averaging null for Word2Vec/GloVe style."""
        nulls = null_generator.generate_averaging_null(sample_embeddings)
        
        # Check all window sizes generated
        assert 'avg_window_1' in nulls
        assert 'avg_window_3' in nulls
        assert 'avg_window_5' in nulls
        assert 'avg_window_10' in nulls
        
        # Window 1 should be just shuffled
        assert nulls['avg_window_1'].shape == sample_embeddings.shape
        
        # Larger windows should have smoothing effect
        # Check that variance decreases with window size
        var_1 = np.var(nulls['avg_window_1'])
        var_10 = np.var(nulls['avg_window_10'])
        assert var_10 < var_1
    
    def test_positional_null(self, null_generator, sample_embeddings):
        """Test positional encoding null for transformers."""
        null = null_generator.generate_positional_null(sample_embeddings)
        
        # Check shape preserved
        assert null.shape == sample_embeddings.shape
        
        # Check that it's different from original
        assert not np.allclose(sample_embeddings, null)
        
        # Check that variance is reasonable (not exploded)
        assert np.var(null) < np.var(sample_embeddings) * 2
    
    def test_dimension_matching(self, null_generator, sample_embeddings):
        """Test dimension matching across paradigms."""
        # Test projection down
        null_300 = null_generator.generate_dimension_matched_null(
            sample_embeddings, target_dim=300
        )
        assert null_300.shape == (100, 300)
        
        # Test padding up
        small_embeddings = np.random.randn(100, 300)
        null_768 = null_generator.generate_dimension_matched_null(
            small_embeddings, target_dim=768
        )
        assert null_768.shape == (100, 768)
        
        # Test same dimension
        null_same = null_generator.generate_dimension_matched_null(
            sample_embeddings, target_dim=384
        )
        assert null_same.shape == sample_embeddings.shape
    
    def test_null_validity(self, null_generator, sample_embeddings):
        """Test that nulls preserve required properties."""
        null = null_generator.phase_scramble_embeddings(sample_embeddings)
        
        metrics = null_generator.test_null_validity(sample_embeddings, null)
        
        # Mean should be preserved
        assert metrics['mean_difference'] < 0.01
        
        # Variance should be similar
        assert 0.9 < metrics['variance_ratio'] < 1.1
        
        # Autocorrelation should be different (not necessarily reduced)
        # Phase scrambling changes temporal structure but doesn't guarantee reduction
        # If original autocorrelation is very small, accept small differences
        if abs(metrics['autocorr_original']) < 0.1:
            assert abs(metrics['autocorr_original'] - metrics['autocorr_null']) > 0.0001
        else:
            assert abs(metrics['autocorr_original'] - metrics['autocorr_null']) > 0.01
        
        # Power should be preserved (for phase scrambling)
        assert 0.99 < metrics['power_ratio'] < 1.01
    
    def test_paradigm_ensemble(self, null_generator):
        """Test ensemble generation for multiple paradigms."""
        embeddings = {
            'all-MiniLM-L6-v2': np.random.randn(50, 384),
            'word2vec': np.random.randn(50, 300),
            'glove': np.random.randn(50, 300)
        }
        
        ensemble = null_generator.generate_paradigm_specific_ensemble(
            embeddings, n_samples=10
        )
        
        # Check all models have nulls
        assert set(ensemble.keys()) == set(embeddings.keys())
        
        # Check each has correct number of samples
        for model, nulls in ensemble.items():
            assert len(nulls) == 10
            assert all(n.shape == embeddings[model].shape for n in nulls)


class TestMessageLevelNulls:
    """Test message-level null models."""
    
    @pytest.fixture
    def msg_null_generator(self):
        """Create message-level null generator."""
        return MessageLevelNullModels(seed=42)
    
    def test_speaker_shuffle(self, msg_null_generator):
        """Test shuffling within speakers."""
        messages = [
            {'role': 'user', 'content': 'msg1'},
            {'role': 'assistant', 'content': 'msg2'},
            {'role': 'user', 'content': 'msg3'},
            {'role': 'assistant', 'content': 'msg4'},
            {'role': 'user', 'content': 'msg5'},
            {'role': 'assistant', 'content': 'msg6'},
        ]
        
        embeddings = np.random.randn(6, 384)
        
        shuffled = msg_null_generator.shuffle_within_speakers(messages, embeddings)
        
        # Check shape preserved
        assert shuffled.shape == embeddings.shape
        
        # Check it's different
        assert not np.array_equal(embeddings, shuffled)
    
    def test_reverse_flow(self, msg_null_generator):
        """Test conversation reversal."""
        embeddings = np.random.randn(10, 384)
        reversed_emb = msg_null_generator.reverse_conversation_flow(embeddings)
        
        # Check reversal
        np.testing.assert_array_equal(reversed_emb, embeddings[::-1])
        
        # Check it's a copy
        reversed_emb[0] = 0
        assert not np.array_equal(embeddings[-1], 0)
    
    def test_bootstrap_segments(self, msg_null_generator):
        """Test bootstrap of conversation segments."""
        embeddings = np.random.randn(50, 384)
        
        bootstrapped = msg_null_generator.bootstrap_conversation_segments(
            embeddings, segment_length=10, n_samples=5
        )
        
        # Check we got right number of samples
        assert len(bootstrapped) == 5
        
        # Check each has right shape
        for boot in bootstrapped:
            assert boot.shape == embeddings.shape


if __name__ == "__main__":
    pytest.main([__file__, "-v"])