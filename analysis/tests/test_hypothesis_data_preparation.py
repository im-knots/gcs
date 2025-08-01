#!/usr/bin/env python3
"""
Functional tests for hypothesis data preparation.

Tests the _prepare_hypothesis_testing_data method and related functionality.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from run_analysis import ConversationAnalysisPipeline
from tests.test_utils import TestDataGenerator, MockComponents


class TestHypothesisDataPreparation:
    """Test hypothesis data preparation functionality."""
    
    @pytest.fixture
    def pipeline(self, tmp_path):
        """Create pipeline instance for testing."""
        # Patch the embedder to avoid loading models
        with patch('run_analysis.EnsembleEmbedder', MockComponents.mock_embedder):
            return ConversationAnalysisPipeline(
                output_dir=str(tmp_path),
                checkpoint_enabled=False,
                log_level="ERROR"
            )
    
    @pytest.fixture
    def data_generator(self):
        """Create test data generator."""
        return TestDataGenerator(seed=42)
    
    @pytest.fixture
    def mock_invariance_results(self, data_generator):
        """Create mock invariance analysis results."""
        return data_generator.generate_invariance_results(n_conversations=30)
    
    @pytest.fixture
    def mock_conversations(self, data_generator):
        """Create mock conversations with embeddings."""
        conversations = []
        
        for i in range(30):
            conv = data_generator.generate_conversation_data(n_messages=50)
            # Add ensemble embeddings
            conv['ensemble_embeddings'] = data_generator.generate_model_ensemble_embeddings(50)
            conversations.append(conv)
        
        return conversations
    
    def test_prepare_hypothesis_data_structure(self, pipeline, mock_invariance_results, 
                                             mock_conversations):
        """Test that prepared data has correct structure."""
        # Mock the heavy components to avoid loading models
        pipeline.embedder = MockComponents.mock_embedder()
        pipeline.trajectory_analyzer = MockComponents.mock_trajectory_analyzer()
        
        data = pipeline._prepare_hypothesis_testing_data(
            mock_invariance_results,
            mock_conversations
        )
        
        # Check top-level structure
        assert 'correlations' in data
        assert 'geometric_metrics' in data
        assert 'control_data' in data
        
        # Check correlations structure
        corr_data = data['correlations']
        required_keys = [
            'transformer_pairs', 'classical_pairs', 'cross_paradigm_pairs',
            'null_within_paradigm', 'random_embedding_pairs'
        ]
        for key in required_keys:
            assert key in corr_data
            assert isinstance(corr_data[key], list)
            assert len(corr_data[key]) > 0
        
        # Check geometric metrics structure
        geom_data = data['geometric_metrics']
        for metric in ['velocity', 'curvature', 'distance']:
            assert metric in geom_data
            for category in ['within', 'cross', 'random']:
                assert category in geom_data[metric]
                assert isinstance(geom_data[metric][category], list)
        
        # Check control data structure
        control = data['control_data']
        assert 'real_scrambled_comparison' in control
        assert 'length_controlled' in control
        assert 'normalized_metrics' in control
    
    def test_correlation_categorization(self, pipeline, mock_invariance_results, 
                                      mock_conversations):
        """Test that correlations are correctly categorized by model type."""
        pipeline.trajectory_analyzer = MockComponents.mock_trajectory_analyzer()
        
        data = pipeline._prepare_hypothesis_testing_data(
            mock_invariance_results,
            mock_conversations
        )
        
        # Check that correlations are in expected ranges
        # Transformer pairs should be high (0.80-0.95)
        transformer_corrs = data['correlations']['transformer_pairs']
        assert all(0.75 <= c <= 1.0 for c in transformer_corrs)
        
        # Classical pairs should be high (0.85-0.98)
        classical_corrs = data['correlations']['classical_pairs']
        assert all(0.80 <= c <= 1.0 for c in classical_corrs)
        
        # Cross-paradigm should be moderate (0.55-0.75)
        cross_corrs = data['correlations']['cross_paradigm_pairs']
        assert all(0.50 <= c <= 0.80 for c in cross_corrs)
        
        # Null should be low (but with some tolerance for random variations)
        null_corrs = data['correlations']['null_within_paradigm']
        assert all(-0.5 <= c <= 0.5 for c in null_corrs)
    
    def test_geometric_metrics_extraction(self, pipeline, data_generator):
        """Test extraction of geometric metrics from trajectory data."""
        # Create more detailed invariance results with trajectory metrics
        n_conversations = 10
        conversation_results = []
        
        for i in range(n_conversations):
            conv_result = {
                'session_id': f'session_{i}',
                'invariance_metrics': {
                    'pairwise_correlations': {
                        'all-MiniLM-L6-v2-all-mpnet-base-v2': 0.85,
                        'word2vec-glove': 0.90,
                        'all-MiniLM-L6-v2-word2vec': 0.65
                    }
                },
                'trajectory_metrics': {
                    'advanced': {
                        'all-MiniLM-L6-v2': {
                            'velocities': list(np.random.rand(50) * 0.5 + 0.3)
                        },
                        'all-mpnet-base-v2': {
                            'velocities': list(np.random.rand(50) * 0.5 + 0.32)
                        },
                        'word2vec': {
                            'velocities': list(np.random.rand(50) * 0.5 + 0.28)
                        }
                    },
                    'curvature_ensemble': {
                        'all-MiniLM-L6-v2': list(np.random.rand(48) * 0.3),
                        'all-mpnet-base-v2': list(np.random.rand(48) * 0.32),
                        'word2vec': list(np.random.rand(48) * 0.35)
                    },
                    'consistency': {
                        'distance_correlation': 0.75
                    }
                }
            }
            conversation_results.append(conv_result)
        
        invariance_results = {
            'conversation_results': conversation_results,
            'aggregate_statistics': {'mean_invariance': 0.8}
        }
        
        # Create minimal conversations
        conversations = [{'ensemble_embeddings': data_generator.generate_model_ensemble_embeddings(50)} 
                        for _ in range(10)]
        
        pipeline.trajectory_analyzer = MockComponents.mock_trajectory_analyzer()
        
        data = pipeline._prepare_hypothesis_testing_data(
            invariance_results,
            conversations
        )
        
        # Check that velocity metrics were extracted
        velocity_within = data['geometric_metrics']['velocity']['within']
        assert len(velocity_within) > 0
        
        # Check that curvature metrics were extracted
        curvature_within = data['geometric_metrics']['curvature']['within']
        assert len(curvature_within) > 0
    
    def test_null_model_generation(self, pipeline, mock_invariance_results, 
                                  mock_conversations):
        """Test that null models are properly generated."""
        pipeline.trajectory_analyzer = MockComponents.mock_trajectory_analyzer()
        
        # Mock the null model generator to track calls
        mock_generate = Mock(return_value={
            'all-MiniLM-L6-v2': [np.random.randn(50, 384) for _ in range(3)],
            'word2vec': [np.random.randn(50, 300) for _ in range(3)]
        })
        pipeline.null_model_generator.generate_paradigm_specific_ensemble = mock_generate
        
        data = pipeline._prepare_hypothesis_testing_data(
            mock_invariance_results,
            mock_conversations[:10]  # Use subset
        )
        
        # Check that null model generator was called
        assert mock_generate.call_count > 0
        
        # Check that null correlations were computed
        assert len(data['correlations']['null_within_paradigm']) >= 20
        assert len(data['correlations']['random_embedding_pairs']) >= 20
    
    def test_control_data_preparation(self, pipeline, mock_invariance_results, 
                                    mock_conversations):
        """Test preparation of control data."""
        pipeline.trajectory_analyzer = MockComponents.mock_trajectory_analyzer()
        
        # Add message content to conversations for length calculation
        for conv in mock_conversations:
            for msg in conv['messages']:
                msg['content'] = ' '.join(['word'] * np.random.randint(10, 50))
        
        data = pipeline._prepare_hypothesis_testing_data(
            mock_invariance_results,
            mock_conversations
        )
        
        control = data['control_data']
        
        # Check real vs scrambled comparison
        assert len(control['real_scrambled_comparison']['real']) > 0
        assert len(control['real_scrambled_comparison']['scrambled']) > 0
        
        # Check length control
        assert 'partial_correlation' in control['length_controlled']
        assert 0 <= control['length_controlled']['partial_correlation'] <= 1
        assert control['length_controlled']['n_conversations'] == len(mock_conversations)
        
        # Check normalized metrics
        assert len(control['normalized_metrics']['correlations']) > 0
    
    def test_conversation_type_analysis_inclusion(self, pipeline, mock_invariance_results):
        """Test that conversation type analysis is included when types are available."""
        # Create conversations with type metadata
        conversations = []
        types = ['full_reasoning', 'light_reasoning', 'no_reasoning']
        
        for i in range(30):
            conv = TestDataGenerator().generate_conversation_data(n_messages=30)
            conv['metadata']['type'] = types[i % 3]
            conv['ensemble_embeddings'] = TestDataGenerator().generate_model_ensemble_embeddings(30)
            conversations.append(conv)
        
        pipeline.trajectory_analyzer = MockComponents.mock_trajectory_analyzer()
        
        # Mock the control analyzer's type analysis
        mock_type_analysis = {
            'per_type': {
                'full_reasoning': {'mean_correlation': 0.85},
                'light_reasoning': {'mean_correlation': 0.75},
                'no_reasoning': {'mean_correlation': 0.65}
            },
            'cross_type_comparison': {
                'p_value': 0.001,
                'kruskal_h': 25.3
            }
        }
        
        pipeline.control_analyzer.analyze_by_conversation_type = Mock(
            return_value=mock_type_analysis
        )
        
        data = pipeline._prepare_hypothesis_testing_data(
            mock_invariance_results,
            conversations
        )
        
        # Check that conversation type analysis was included
        assert 'conversation_type_analysis' in data['control_data']
        assert data['control_data']['conversation_type_analysis'] == mock_type_analysis
    
    def test_handling_missing_data(self, pipeline):
        """Test graceful handling of missing or incomplete data."""
        # Create minimal invariance results
        minimal_results = {
            'conversation_results': [{
                'session_id': 'test',
                'invariance_metrics': {
                    'pairwise_correlations': {
                        'all-MiniLM-L6-v2-all-mpnet-base-v2': 0.85
                    }
                }
                # Missing trajectory_metrics
            }],
            'aggregate_statistics': {'mean_invariance': 0.8}
        }
        
        # Create minimal conversations
        minimal_conversations = [{
            'ensemble_embeddings': {
                'all-MiniLM-L6-v2': np.random.randn(30, 384)
            },
            'messages': [{'content': 'test'}] * 30
        }]
        
        pipeline.trajectory_analyzer = MockComponents.mock_trajectory_analyzer()
        
        # Should not crash
        data = pipeline._prepare_hypothesis_testing_data(
            minimal_results,
            minimal_conversations
        )
        
        # Should still have basic structure
        assert 'correlations' in data
        assert 'geometric_metrics' in data
        assert 'control_data' in data
    
    def test_logging_output(self, tmp_path, mock_invariance_results, 
                          mock_conversations, caplog):
        """Test that preparation logs summary statistics."""
        # Create pipeline with INFO logging
        with patch('run_analysis.EnsembleEmbedder', MockComponents.mock_embedder):
            pipeline = ConversationAnalysisPipeline(
                output_dir=str(tmp_path),
                checkpoint_enabled=False,
                log_level="INFO"
            )
        pipeline.trajectory_analyzer = MockComponents.mock_trajectory_analyzer()
        
        with caplog.at_level('INFO'):
            data = pipeline._prepare_hypothesis_testing_data(
                mock_invariance_results,
                mock_conversations
            )
        
        # Check that summary was logged
        log_text = caplog.text
        assert "Prepared hypothesis testing data:" in log_text
        assert "Transformer pairs:" in log_text
        assert "Classical pairs:" in log_text
        assert "Cross-paradigm pairs:" in log_text
        assert "Null samples:" in log_text
        assert "Random samples:" in log_text


class TestDataPreparationEdgeCases:
    """Test edge cases in data preparation."""
    
    def test_empty_conversations(self, tmp_path):
        """Test handling of empty conversation list."""
        pipeline = ConversationAnalysisPipeline(
            output_dir=str(tmp_path),
            checkpoint_enabled=False,
            log_level="ERROR"
        )
        
        empty_results = {
            'conversation_results': [],
            'aggregate_statistics': {'mean_invariance': 0}
        }
        
        data = pipeline._prepare_hypothesis_testing_data(empty_results, [])
        
        # Should return valid structure with empty lists
        assert len(data['correlations']['transformer_pairs']) == 0
        assert len(data['correlations']['classical_pairs']) == 0
        
        # But should have synthetic null samples
        assert len(data['correlations']['null_within_paradigm']) >= 20
    
    def test_single_model_only(self, tmp_path):
        """Test with only one model type in results."""
        pipeline = ConversationAnalysisPipeline(
            output_dir=str(tmp_path),
            checkpoint_enabled=False,
            log_level="ERROR"
        )
        
        # Results with only transformer correlations
        single_model_results = {
            'conversation_results': [{
                'session_id': 'test',
                'invariance_metrics': {
                    'pairwise_correlations': {
                        'all-MiniLM-L6-v2-all-mpnet-base-v2': 0.85,
                        'all-MiniLM-L6-v2-all-distilroberta-v1': 0.83
                    }
                }
            }],
            'aggregate_statistics': {'mean_invariance': 0.84}
        }
        
        conversations = [{
            'ensemble_embeddings': {
                'all-MiniLM-L6-v2': np.random.randn(30, 384),
                'all-mpnet-base-v2': np.random.randn(30, 768)
            },
            'messages': [{'content': 'test'}] * 30
        }]
        
        pipeline.trajectory_analyzer = MockComponents.mock_trajectory_analyzer()
        
        data = pipeline._prepare_hypothesis_testing_data(
            single_model_results,
            conversations
        )
        
        # Should have transformer pairs but no classical
        assert len(data['correlations']['transformer_pairs']) > 0
        assert len(data['correlations']['classical_pairs']) == 0
        assert len(data['correlations']['cross_paradigm_pairs']) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])