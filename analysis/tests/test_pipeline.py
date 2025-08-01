#!/usr/bin/env python3
"""
Tests for the conversation analysis pipeline.

This module tests the overall conversation analysis workflow that runs after
generating all PNG and PDF files, including:
- Conversation loading and filtering
- Batch embedding processing
- Phase detection and combination across models
- Invariance analysis and geometric signatures
- Visualization generation
- Full pipeline integration
"""

import pytest
import numpy as np
import tempfile
import shutil
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, List

# Import the main pipeline and components
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from run_analysis import ConversationAnalysisPipeline
from embedding_analysis import (
    EnsembleEmbedder,
    TrajectoryAnalyzer,
    TrajectoryVisualizer
)
from embedding_analysis.core import ConversationLoader
from embedding_analysis.models.ensemble_phase_detector import EnsemblePhaseDetector
from embedding_analysis.core.geometric_invariance import (
    GeometricSignatureComputer,
    InvarianceAnalyzer
)


class TestConversationAnalysisPipeline:
    """Test the main conversation analysis pipeline."""
    
    @pytest.fixture
    def temp_output_dir(self):
        """Create a temporary output directory."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def sample_conversations(self):
        """Create sample conversation data for testing."""
        conversations = [
            {
                'metadata': {'session_id': 'test_conv_1'},
                'messages': [
                    {'content': 'Hello, how can I help you?', 'role': 'assistant'},
                    {'content': 'I need help with Python', 'role': 'user'},
                    {'content': 'Sure, what specifically?', 'role': 'assistant'},
                    {'content': 'How do I read a file?', 'role': 'user'},
                    {'content': 'You can use the open() function', 'role': 'assistant'}
                ],
                'phases': [
                    {'turn': 0, 'phase': 'greeting'},
                    {'turn': 2, 'phase': 'problem_identification'},
                    {'turn': 4, 'phase': 'solution'}
                ]
            },
            {
                'metadata': {'session_id': 'test_conv_2'},
                'messages': [
                    {'content': 'Hi there!', 'role': 'user'},
                    {'content': 'Hello! How are you?', 'role': 'assistant'},
                    {'content': 'Good, can you explain loops?', 'role': 'user'},
                    {'content': 'Loops allow you to repeat code', 'role': 'assistant'}
                ]
            }
        ]
        return conversations
    
    @pytest.fixture
    def mock_embeddings(self):
        """Create mock embeddings for testing."""
        # Create embeddings for 5 messages across 3 models
        n_messages = 5
        embeddings = {
            'all-MiniLM-L6-v2': np.random.randn(n_messages, 384),
            'all-mpnet-base-v2': np.random.randn(n_messages, 768),
            'all-distilroberta-v1': np.random.randn(n_messages, 768)
        }
        return embeddings
    
    def test_pipeline_initialization(self, temp_output_dir):
        """Test that the pipeline initializes correctly."""
        pipeline = ConversationAnalysisPipeline(
            output_dir=str(temp_output_dir),
            checkpoint_enabled=False,
            log_level="ERROR",
            batch_size=10,
            figure_format="png"
        )
        
        assert pipeline.output_dir == temp_output_dir
        assert pipeline.batch_size == 10
        assert pipeline.figure_format == "png"
        assert pipeline.checkpoint_manager is None  # Disabled
        
        # Check that components are initialized
        assert isinstance(pipeline.loader, ConversationLoader)
        assert isinstance(pipeline.embedder, EnsembleEmbedder)
        assert isinstance(pipeline.trajectory_analyzer, TrajectoryAnalyzer)
        assert isinstance(pipeline.ensemble_phase_detector, EnsemblePhaseDetector)
        
        # Check output directories exist
        assert (temp_output_dir / "logs").exists()
    
    def test_load_conversations(self, temp_output_dir, sample_conversations):
        """Test conversation loading and filtering."""
        pipeline = ConversationAnalysisPipeline(
            output_dir=str(temp_output_dir),
            checkpoint_enabled=False,
            log_level="ERROR"
        )
        
        # Create test data directory with sample conversations
        test_data_dir = temp_output_dir / "test_data"
        test_data_dir.mkdir()
        
        for i, conv in enumerate(sample_conversations):
            with open(test_data_dir / f"conv_{i}.json", 'w') as f:
                json.dump(conv, f)
        
        # Mock the loader to return our sample conversations
        with patch.object(pipeline.loader, 'load_conversations_batch', 
                         return_value=sample_conversations):
            with patch.object(pipeline.loader, 'filter_conversations',
                             return_value=sample_conversations):
                
                conversations = pipeline.load_all_conversations(
                    [test_data_dir], 
                    max_conversations=10
                )
                
                assert len(conversations) == 2
                assert conversations[0]['metadata']['session_id'] == 'test_conv_1'
    
    def test_batch_embedding_processing(self, temp_output_dir, sample_conversations, mock_embeddings):
        """Test that conversations are processed in batches for embedding."""
        pipeline = ConversationAnalysisPipeline(
            output_dir=str(temp_output_dir),
            checkpoint_enabled=False,
            log_level="ERROR",
            batch_size=1  # Force multiple batches
        )
        
        # Mock the embedder to return pre-defined embeddings
        with patch.object(pipeline.embedder, 'embed_texts') as mock_embed:
            # Set up the mock to return embeddings based on the number of texts
            def mock_embed_texts(texts, show_progress=False):
                n_texts = len(texts)
                return {
                    'all-MiniLM-L6-v2': np.random.randn(n_texts, 384),
                    'all-mpnet-base-v2': np.random.randn(n_texts, 768),
                    'all-distilroberta-v1': np.random.randn(n_texts, 768)
                }
            
            mock_embed.side_effect = mock_embed_texts
            
            # Mock other required methods
            with patch.object(pipeline.trajectory_analyzer, 'calculate_ensemble_trajectories', 
                             return_value={}):
                with patch.object(pipeline.trajectory_analyzer, 'analyze_trajectory_with_normalization',
                                 return_value={}):
                    with patch.object(pipeline.trajectory_analyzer, 'calculate_curvature_ensemble',
                                     return_value={}):
                        with patch.object(pipeline.ensemble_phase_detector, 'detect_phases_ensemble',
                                         return_value={'ensemble_phases': [], 'method_results': {}}):
                            with patch.object(pipeline.signature_computer, 'compute_all_signatures',
                                             return_value={}):
                                with patch.object(pipeline.invariance_analyzer, 'compute_invariance_metrics',
                                                 return_value={}):
                                    with patch.object(pipeline.ensemble_visualizer, 'create_comprehensive_ensemble_plot'):
                                        
                                        results = pipeline.analyze_conversations_for_invariance(sample_conversations)
                                        
                                        # Should have been called twice (once per batch with batch_size=1)
                                        assert mock_embed.call_count == 2
                                        
                                        # Check results structure
                                        assert 'n_conversations' in results
                                        assert results['n_conversations'] == 2
                                        assert len(results['conversation_results']) == 2
    
    def test_phase_detection_and_combination(self, temp_output_dir):
        """Test phase detection and combination across models."""
        pipeline = ConversationAnalysisPipeline(
            output_dir=str(temp_output_dir),
            checkpoint_enabled=False,
            log_level="ERROR"
        )
        
        # Test _combine_cross_model_phases
        all_phases = [
            {'turn': 10, 'model': 'model1'},
            {'turn': 12, 'model': 'model2'},
            {'turn': 11, 'model': 'model3'},
            {'turn': 50, 'model': 'model1'},
            {'turn': 52, 'model': 'model2'}
        ]
        
        consensus_phases = pipeline._combine_cross_model_phases(all_phases, n_messages=100)
        
        # Should have 2 consensus phases (around turns 10-12 and 50-52)
        assert len(consensus_phases) == 2
        assert consensus_phases[0]['turn'] == 11  # Median of 10, 11, 12
        assert consensus_phases[0]['n_models'] == 3
        assert consensus_phases[1]['turn'] == 51  # Median of 50, 52
        assert consensus_phases[1]['n_models'] == 2
        
        # Test _calculate_phase_variance
        model_phases = {
            'model1': [{'turn': 10}, {'turn': 50}],
            'model2': [{'turn': 12}, {'turn': 52}],
            'model3': [{'turn': 11}]
        }
        
        variance_stats = pipeline._calculate_phase_variance(model_phases, n_messages=100)
        
        assert 'mean_std' in variance_stats
        assert 'max_std' in variance_stats
        assert variance_stats['mean_std'] > 0  # Should have some variance
    
    def test_phase_comparison_with_annotations(self, temp_output_dir):
        """Test comparison of detected phases with annotations."""
        pipeline = ConversationAnalysisPipeline(
            output_dir=str(temp_output_dir),
            checkpoint_enabled=False,
            log_level="ERROR"
        )
        
        detected_phases = [
            {'turn': 10},
            {'turn': 25},
            {'turn': 40}
        ]
        
        annotated_phases = [
            {'turn': 12},  # Close to 10
            {'turn': 26},  # Close to 25
            {'turn': 50}   # Not close to any detected
        ]
        
        comparison = pipeline._compare_phases_with_annotations(detected_phases, annotated_phases)
        
        assert comparison['n_detected'] == 3
        assert comparison['n_annotated'] == 3
        assert comparison['n_matches'] == 2  # First two are within threshold
        assert comparison['metrics']['precision'] == 2/3
        assert comparison['metrics']['recall'] == 2/3
        assert comparison['metrics']['f1'] > 0
    
    def test_invariance_analysis(self, temp_output_dir, sample_conversations):
        """Test geometric invariance analysis."""
        pipeline = ConversationAnalysisPipeline(
            output_dir=str(temp_output_dir),
            checkpoint_enabled=False,
            log_level="ERROR"
        )
        
        # Create mock signatures for different models
        mock_signatures = {
            'model1': {'curvature': 0.5, 'velocity': 1.2},
            'model2': {'curvature': 0.52, 'velocity': 1.18},
            'model3': {'curvature': 0.49, 'velocity': 1.22}
        }
        
        mock_invariance_metrics = {
            'mean_correlation': 0.95,
            'std_correlation': 0.02,
            'invariance_score': 0.93
        }
        
        # Mock the necessary methods
        with patch.object(pipeline.signature_computer, 'compute_all_signatures',
                         side_effect=lambda emb, id: mock_signatures[id.split('_')[-1]]):
            with patch.object(pipeline.invariance_analyzer, 'compute_invariance_metrics',
                             return_value=mock_invariance_metrics):
                with patch.object(pipeline.invariance_analyzer, 'aggregate_invariance_scores',
                                 return_value={'mean_invariance': 0.93, 'std_invariance': 0.05}):
                    # Mock other required components
                    with patch.object(pipeline.embedder, 'embed_texts',
                                     return_value={'model1': np.random.randn(5, 384)}):
                        with patch.object(pipeline.trajectory_analyzer, 'calculate_ensemble_trajectories',
                                         return_value={}):
                            with patch.object(pipeline.trajectory_analyzer, 'analyze_trajectory_with_normalization',
                                             return_value={}):
                                with patch.object(pipeline.trajectory_analyzer, 'calculate_curvature_ensemble',
                                                 return_value={}):
                                    with patch.object(pipeline.ensemble_phase_detector, 'detect_phases_ensemble',
                                                     return_value={'ensemble_phases': [], 'method_results': {}}):
                                        with patch.object(pipeline.ensemble_visualizer, 'create_comprehensive_ensemble_plot'):
                                            
                                            # Use only one conversation for simplicity
                                            results = pipeline.analyze_conversations_for_invariance([sample_conversations[0]])
                                            
                                            assert 'aggregate_statistics' in results
                                            assert results['aggregate_statistics']['mean_invariance'] == 0.93
                                            assert results['aggregate_statistics']['std_invariance'] == 0.05
    
    def test_visualization_generation(self, temp_output_dir, sample_conversations):
        """Test that visualizations are generated with correct formats."""
        # Test PNG only
        pipeline = ConversationAnalysisPipeline(
            output_dir=str(temp_output_dir),
            checkpoint_enabled=False,
            log_level="ERROR",
            figure_format="png"
        )
        
        # Mock visualization call
        mock_viz = Mock()
        pipeline.ensemble_visualizer.create_comprehensive_ensemble_plot = mock_viz
        
        # Mock other required methods for minimal execution
        with patch.object(pipeline.embedder, 'embed_texts',
                         return_value={'model1': np.random.randn(5, 384)}):
            with patch.object(pipeline.trajectory_analyzer, 'calculate_ensemble_trajectories',
                             return_value={}):
                with patch.object(pipeline.trajectory_analyzer, 'analyze_trajectory_with_normalization',
                                 return_value={}):
                    with patch.object(pipeline.trajectory_analyzer, 'calculate_curvature_ensemble',
                                     return_value={}):
                        with patch.object(pipeline.ensemble_phase_detector, 'detect_phases_ensemble',
                                         return_value={'ensemble_phases': [], 'method_results': {}}):
                            with patch.object(pipeline.signature_computer, 'compute_all_signatures',
                                             return_value={}):
                                with patch.object(pipeline.invariance_analyzer, 'compute_invariance_metrics',
                                                 return_value={}):
                                    with patch.object(pipeline.invariance_analyzer, 'aggregate_invariance_scores',
                                                     return_value={'mean_invariance': 0.9, 'std_invariance': 0.1}):
                                        
                                        pipeline.analyze_conversations_for_invariance([sample_conversations[0]])
                                        
                                        # Check visualization was called with PNG path only
                                        mock_viz.assert_called_once()
                                        call_args = mock_viz.call_args[1]
                                        assert call_args['save_path'] is not None
                                        assert str(call_args['save_path']).endswith('.png')
                                        assert call_args['save_pdf'] is None
        
        # Test PDF only
        pipeline.figure_format = "pdf"
        mock_viz.reset_mock()
        
        with patch.object(pipeline.embedder, 'embed_texts',
                         return_value={'model1': np.random.randn(5, 384)}):
            with patch.object(pipeline.trajectory_analyzer, 'calculate_ensemble_trajectories',
                             return_value={}):
                with patch.object(pipeline.trajectory_analyzer, 'analyze_trajectory_with_normalization',
                                 return_value={}):
                    with patch.object(pipeline.trajectory_analyzer, 'calculate_curvature_ensemble',
                                     return_value={}):
                        with patch.object(pipeline.ensemble_phase_detector, 'detect_phases_ensemble',
                                         return_value={'ensemble_phases': [], 'method_results': {}}):
                            with patch.object(pipeline.signature_computer, 'compute_all_signatures',
                                             return_value={}):
                                with patch.object(pipeline.invariance_analyzer, 'compute_invariance_metrics',
                                                 return_value={}):
                                    with patch.object(pipeline.invariance_analyzer, 'aggregate_invariance_scores',
                                                     return_value={'mean_invariance': 0.9, 'std_invariance': 0.1}):
                                        
                                        pipeline.analyze_conversations_for_invariance([sample_conversations[0]])
                                        
                                        # Check visualization was called with PDF path only
                                        mock_viz.assert_called_once()
                                        call_args = mock_viz.call_args[1]
                                        assert call_args['save_path'] is None
                                        assert call_args['save_pdf'] is not None
                                        assert str(call_args['save_pdf']).endswith('.pdf')
        
        # Test both formats
        pipeline.figure_format = "both"
        mock_viz.reset_mock()
        
        with patch.object(pipeline.embedder, 'embed_texts',
                         return_value={'model1': np.random.randn(5, 384)}):
            with patch.object(pipeline.trajectory_analyzer, 'calculate_ensemble_trajectories',
                             return_value={}):
                with patch.object(pipeline.trajectory_analyzer, 'analyze_trajectory_with_normalization',
                                 return_value={}):
                    with patch.object(pipeline.trajectory_analyzer, 'calculate_curvature_ensemble',
                                     return_value={}):
                        with patch.object(pipeline.ensemble_phase_detector, 'detect_phases_ensemble',
                                         return_value={'ensemble_phases': [], 'method_results': {}}):
                            with patch.object(pipeline.signature_computer, 'compute_all_signatures',
                                             return_value={}):
                                with patch.object(pipeline.invariance_analyzer, 'compute_invariance_metrics',
                                                 return_value={}):
                                    with patch.object(pipeline.invariance_analyzer, 'aggregate_invariance_scores',
                                                     return_value={'mean_invariance': 0.9, 'std_invariance': 0.1}):
                                        
                                        pipeline.analyze_conversations_for_invariance([sample_conversations[0]])
                                        
                                        # Check visualization was called with both paths
                                        mock_viz.assert_called_once()
                                        call_args = mock_viz.call_args[1]
                                        assert call_args['save_path'] is not None
                                        assert str(call_args['save_path']).endswith('.png')
                                        assert call_args['save_pdf'] is not None
                                        assert str(call_args['save_pdf']).endswith('.pdf')
    
    def test_full_pipeline_integration(self, temp_output_dir, sample_conversations):
        """Test the full pipeline execution end-to-end."""
        pipeline = ConversationAnalysisPipeline(
            output_dir=str(temp_output_dir),
            checkpoint_enabled=False,
            log_level="ERROR",
            batch_size=10,
            figure_format="png"
        )
        
        # Create test data directory
        test_data_dir = temp_output_dir / "test_data"
        test_data_dir.mkdir()
        
        # Mock all external calls
        with patch.object(pipeline, 'load_all_conversations', return_value=sample_conversations):
            with patch.object(pipeline, 'analyze_conversations_for_invariance') as mock_analyze:
                mock_analyze.return_value = {
                    'aggregate_statistics': {'mean_invariance': 0.9, 'std_invariance': 0.1},
                    'conversation_results': [],
                    'geometric_signatures': {},
                    'invariance_scores': {}
                }
                
                with patch.object(pipeline.hypothesis_tester_new, 'test_invariance_hypothesis',
                                 return_value={'hypothesis_accepted': True}):
                    with patch.object(pipeline, 'analyze_null_models', return_value={}):
                        with patch.object(pipeline, 'generate_invariance_visualizations'):
                            with patch.object(pipeline, 'generate_invariance_reports'):
                                
                                # Run the full pipeline
                                pipeline.run_analysis([test_data_dir], max_conversations=2)
                                
                                # Verify key methods were called
                                mock_analyze.assert_called_once()
                                assert len(mock_analyze.call_args[0][0]) == 2  # Two conversations
    
    def test_error_handling(self, temp_output_dir):
        """Test error handling in the pipeline."""
        pipeline = ConversationAnalysisPipeline(
            output_dir=str(temp_output_dir),
            checkpoint_enabled=False,
            log_level="ERROR"
        )
        
        # Test with non-existent directory
        non_existent_dir = temp_output_dir / "does_not_exist"
        conversations = pipeline.load_all_conversations([non_existent_dir])
        assert len(conversations) == 0  # Should handle gracefully
        
        # Test with empty conversations
        result = pipeline.analyze_conversations_for_invariance([])
        assert result['n_conversations'] == 0
        assert len(result['conversation_results']) == 0


class TestBatchProcessing:
    """Test batch processing functionality specifically."""
    
    def test_gpu_memory_management(self, temp_output_dir):
        """Test that GPU memory is properly managed during batch processing."""
        pipeline = ConversationAnalysisPipeline(
            output_dir=str(temp_output_dir),
            checkpoint_enabled=False,
            log_level="ERROR",
            batch_size=2
        )
        
        # Create larger set of conversations
        conversations = []
        for i in range(5):
            conversations.append({
                'metadata': {'session_id': f'test_conv_{i}'},
                'messages': [
                    {'content': f'Message {j}', 'role': 'user' if j % 2 == 0 else 'assistant'}
                    for j in range(10)
                ]
            })
        
        # Mock torch.cuda.empty_cache
        with patch('torch.cuda.empty_cache') as mock_empty_cache:
            # Mock other required methods
            with patch.object(pipeline.embedder, 'embed_texts',
                             return_value={'model1': np.random.randn(50, 384)}):
                with patch.object(pipeline.trajectory_analyzer, 'calculate_ensemble_trajectories',
                                 return_value={}):
                    with patch.object(pipeline.trajectory_analyzer, 'analyze_trajectory_with_normalization',
                                     return_value={}):
                        with patch.object(pipeline.trajectory_analyzer, 'calculate_curvature_ensemble',
                                         return_value={}):
                            with patch.object(pipeline.ensemble_phase_detector, 'detect_phases_ensemble',
                                             return_value={'ensemble_phases': [], 'method_results': {}}):
                                with patch.object(pipeline.signature_computer, 'compute_all_signatures',
                                                 return_value={}):
                                    with patch.object(pipeline.invariance_analyzer, 'compute_invariance_metrics',
                                                     return_value={}):
                                        with patch.object(pipeline.invariance_analyzer, 'aggregate_invariance_scores',
                                                         return_value={'mean_invariance': 0.9, 'std_invariance': 0.1}):
                                            with patch.object(pipeline.ensemble_visualizer, 'create_comprehensive_ensemble_plot'):
                                                
                                                pipeline.analyze_conversations_for_invariance(conversations)
                                                
                                                # Should have cleared GPU cache after each batch
                                                # 5 conversations with batch_size=2 = 3 batches
                                                assert mock_empty_cache.call_count == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])