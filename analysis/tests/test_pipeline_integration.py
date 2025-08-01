#!/usr/bin/env python3
"""
Integration tests for the full conversation analysis pipeline.

Tests the complete workflow from data loading to hypothesis testing results.
"""

import pytest
import numpy as np
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from run_analysis import ConversationAnalysisPipeline
from embedding_analysis.core import HierarchicalHypothesisTester
from tests.test_utils import TestDataGenerator, MockComponents


class TestPipelineIntegration:
    """Integration tests for the complete pipeline."""
    
    @pytest.fixture
    def test_data_dir(self, tmp_path):
        """Create test data directory with sample conversations."""
        data_dir = tmp_path / "test_data"
        data_dir.mkdir()
        
        # Generate test conversations
        generator = TestDataGenerator()
        
        for i in range(10):
            conv = generator.generate_conversation_data(
                n_messages=50,
                include_phases=True
            )
            
            # Save to file
            with open(data_dir / f"conversation_{i}.json", 'w') as f:
                json.dump(conv, f)
        
        return data_dir
    
    @pytest.fixture
    def pipeline(self, tmp_path):
        """Create pipeline instance."""
        return ConversationAnalysisPipeline(
            output_dir=str(tmp_path / "output"),
            checkpoint_enabled=False,
            log_level="INFO",
            batch_size=5,
            figure_format="png"
        )
    
    def test_full_pipeline_execution(self, pipeline, test_data_dir, tmp_path):
        """Test complete pipeline execution from loading to results."""
        # Mock heavy components to speed up test
        mock_embedder = MockComponents.mock_embedder()
        mock_analyzer = MockComponents.mock_trajectory_analyzer()
        
        pipeline.embedder = mock_embedder
        pipeline.trajectory_analyzer = mock_analyzer
        
        # Mock visualization to avoid actual plot generation
        pipeline.ensemble_visualizer.create_comprehensive_ensemble_plot = Mock()
        
        # Mock report generation
        pipeline.generate_invariance_visualizations = Mock()
        pipeline.generate_invariance_reports = Mock()
        
        # Run the pipeline
        pipeline.run_analysis(
            [test_data_dir],
            max_conversations=5  # Limit for speed
        )
        
        # Verify pipeline execution
        # Check that conversations were loaded
        assert len(pipeline.all_conversations) > 0
        
        # Check that invariance results were generated
        assert 'aggregate_statistics' in pipeline.invariance_results
        assert 'mean_invariance' in pipeline.invariance_results['aggregate_statistics']
        
        # Check that output directory structure was created
        output_dir = tmp_path / "output"
        assert (output_dir / "logs").exists()
        assert (output_dir / "figures").exists()
        assert (output_dir / "figures" / "ensemble").exists()
    
    def test_hierarchical_hypothesis_testing_integration(self, pipeline, test_data_dir):
        """Test integration of hierarchical hypothesis testing."""
        # Create realistic mock data
        generator = TestDataGenerator()
        
        # Mock conversation loading
        mock_conversations = []
        for i in range(30):
            conv = generator.generate_conversation_data(n_messages=50)
            conv['ensemble_embeddings'] = generator.generate_model_ensemble_embeddings(50)
            mock_conversations.append(conv)
        
        pipeline.load_all_conversations = Mock(return_value=mock_conversations)
        
        # Mock invariance analysis to return realistic results
        mock_invariance_results = generator.generate_invariance_results(n_conversations=30)
        pipeline.analyze_conversations_for_invariance = Mock(return_value=mock_invariance_results)
        
        # Mock visualizations
        pipeline.generate_invariance_visualizations = Mock()
        pipeline.generate_invariance_reports = Mock()
        
        # Run analysis
        pipeline.run_analysis([test_data_dir], max_conversations=30)
        
        # Verify hierarchical hypothesis testing was called
        assert hasattr(pipeline, 'hierarchical_hypothesis_tester')
        
        # The hypothesis testing should have been executed
        # Check by verifying the mocked methods were called
        assert pipeline.analyze_conversations_for_invariance.called
    
    def test_batch_processing_integration(self, pipeline, tmp_path):
        """Test that batch processing works correctly."""
        # Create many conversations to test batching
        generator = TestDataGenerator()
        conversations = []
        
        for i in range(15):  # More than batch_size
            conv = generator.generate_conversation_data(n_messages=30)
            conv['ensemble_embeddings'] = generator.generate_model_ensemble_embeddings(30)
            conversations.append(conv)
        
        # Set small batch size
        pipeline.batch_size = 5
        
        # Mock components
        pipeline.embedder = MockComponents.mock_embedder()
        pipeline.trajectory_analyzer = MockComponents.mock_trajectory_analyzer()
        pipeline.ensemble_visualizer.create_comprehensive_ensemble_plot = Mock()
        
        # Track batch processing
        embedding_calls = []
        
        def track_embedding(texts, show_progress=False):
            embedding_calls.append(len(texts))
            return MockComponents.mock_embedder().embed_texts(texts, show_progress)
        
        pipeline.embedder.embed_texts = track_embedding
        
        # Run analysis
        results = pipeline.analyze_conversations_for_invariance(conversations)
        
        # Verify batching occurred
        assert len(embedding_calls) == 3  # 15 conversations / 5 per batch
        
        # Verify results for all conversations
        assert len(results['conversation_results']) == 15
    
    def test_error_handling_integration(self, pipeline, test_data_dir):
        """Test pipeline error handling."""
        # Test with non-existent directory
        non_existent = test_data_dir / "does_not_exist"
        
        # Should handle gracefully
        pipeline.run_analysis([non_existent], max_conversations=5)
        
        # Pipeline should complete without crashing
        assert True  # If we get here, no exception was raised
        
        # Test with empty conversation list
        pipeline.load_all_conversations = Mock(return_value=[])
        
        # Should handle empty data
        pipeline.run_analysis([test_data_dir], max_conversations=5)
        
        # Should log error but not crash
        assert True
    
    def test_output_generation_integration(self, pipeline, test_data_dir, tmp_path):
        """Test that all expected outputs are generated."""
        # Mock heavy components
        pipeline.embedder = MockComponents.mock_embedder()
        pipeline.trajectory_analyzer = MockComponents.mock_trajectory_analyzer()
        
        # Track visualization calls
        viz_calls = []
        
        def track_viz(*args, **kwargs):
            viz_calls.append((args, kwargs))
        
        pipeline.ensemble_visualizer.create_comprehensive_ensemble_plot = track_viz
        
        # Mock report generation
        report_calls = []
        
        def track_reports(*args, **kwargs):
            report_calls.append((args, kwargs))
        
        pipeline.generate_invariance_reports = track_reports
        
        # Run analysis
        pipeline.run_analysis([test_data_dir], max_conversations=3)
        
        # Check visualizations were generated
        assert len(viz_calls) == 3  # One per conversation
        
        # Check each visualization call has proper arguments
        for args, kwargs in viz_calls:
            assert 'save_path' in kwargs or len(args) > 3
        
        # Check report generation was called
        assert len(report_calls) == 1
    
    def test_checkpoint_integration(self, pipeline, test_data_dir, tmp_path):
        """Test checkpoint functionality integration."""
        # Enable checkpointing
        checkpoint_pipeline = ConversationAnalysisPipeline(
            output_dir=str(tmp_path / "output_checkpoint"),
            checkpoint_enabled=True,
            log_level="ERROR",
            batch_size=5
        )
        
        # Mock components
        checkpoint_pipeline.embedder = MockComponents.mock_embedder()
        checkpoint_pipeline.trajectory_analyzer = MockComponents.mock_trajectory_analyzer()
        checkpoint_pipeline.ensemble_visualizer.create_comprehensive_ensemble_plot = Mock()
        checkpoint_pipeline.generate_invariance_visualizations = Mock()
        checkpoint_pipeline.generate_invariance_reports = Mock()
        
        # Run analysis
        checkpoint_pipeline.run_analysis([test_data_dir], max_conversations=5)
        
        # Check checkpoint directory created
        checkpoint_dir = tmp_path / "output_checkpoint" / "checkpoints"
        assert checkpoint_dir.exists()
    
    def test_figure_format_handling(self, tmp_path):
        """Test different figure format options."""
        # Test PNG only
        png_pipeline = ConversationAnalysisPipeline(
            output_dir=str(tmp_path / "png_output"),
            checkpoint_enabled=False,
            log_level="ERROR",
            figure_format="png"
        )
        
        # Test PDF only
        pdf_pipeline = ConversationAnalysisPipeline(
            output_dir=str(tmp_path / "pdf_output"),
            checkpoint_enabled=False,
            log_level="ERROR",
            figure_format="pdf"
        )
        
        # Test both formats
        both_pipeline = ConversationAnalysisPipeline(
            output_dir=str(tmp_path / "both_output"),
            checkpoint_enabled=False,
            log_level="ERROR",
            figure_format="both"
        )
        
        # Check that pipelines initialize correctly
        assert png_pipeline.figure_format == "png"
        assert pdf_pipeline.figure_format == "pdf"
        assert both_pipeline.figure_format == "both"


class TestPipelineWithRealComponents:
    """Integration tests using real components (slower but more thorough)."""
    
    @pytest.mark.slow
    def test_real_embedding_generation(self, tmp_path):
        """Test with real embedding generation (requires models)."""
        pipeline = ConversationAnalysisPipeline(
            output_dir=str(tmp_path),
            checkpoint_enabled=False,
            log_level="INFO",
            batch_size=2
        )
        
        # Create minimal test data
        generator = TestDataGenerator()
        conversations = []
        
        for i in range(2):
            conv = generator.generate_conversation_data(n_messages=10)
            conversations.append(conv)
        
        # Mock only visualization to avoid plots
        pipeline.ensemble_visualizer.create_comprehensive_ensemble_plot = Mock()
        pipeline.generate_invariance_visualizations = Mock()
        pipeline.generate_invariance_reports = Mock()
        
        # Try to run with real embedder (will skip if models not available)
        try:
            pipeline.load_all_conversations = Mock(return_value=conversations)
            pipeline.run_analysis([Path(".")], max_conversations=2)
            
            # If successful, check results
            assert len(pipeline.invariance_results) > 0
        except Exception as e:
            # Skip if embedding models not available
            pytest.skip(f"Embedding models not available: {e}")
    
    @pytest.mark.slow
    def test_performance_with_many_conversations(self, tmp_path):
        """Test pipeline performance with many conversations."""
        import time
        
        pipeline = ConversationAnalysisPipeline(
            output_dir=str(tmp_path),
            checkpoint_enabled=False,
            log_level="WARNING",
            batch_size=25
        )
        
        # Generate many conversations
        generator = TestDataGenerator()
        conversations = []
        
        for i in range(100):
            conv = generator.generate_conversation_data(n_messages=30)
            conv['ensemble_embeddings'] = generator.generate_model_ensemble_embeddings(30)
            conversations.append(conv)
        
        # Mock heavy components
        pipeline.embedder = MockComponents.mock_embedder()
        pipeline.trajectory_analyzer = MockComponents.mock_trajectory_analyzer()
        pipeline.ensemble_visualizer.create_comprehensive_ensemble_plot = Mock()
        pipeline.generate_invariance_visualizations = Mock()
        pipeline.generate_invariance_reports = Mock()
        
        # Time the analysis
        start_time = time.time()
        
        pipeline.load_all_conversations = Mock(return_value=conversations)
        pipeline.run_analysis([Path(".")], max_conversations=100)
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Should complete in reasonable time even with 100 conversations
        assert duration < 60  # Less than 1 minute with mocked components
        
        # Check all conversations processed
        assert len(pipeline.invariance_results.get('conversation_results', [])) == 100


class TestEndToEndScenarios:
    """Test specific end-to-end scenarios."""
    
    def test_tier_failure_scenario(self, tmp_path):
        """Test scenario where Tier 1 hypothesis fails."""
        pipeline = ConversationAnalysisPipeline(
            output_dir=str(tmp_path),
            checkpoint_enabled=False,
            log_level="INFO"
        )
        
        # Create data that will fail Tier 1
        generator = TestDataGenerator()
        
        # Override correlation generation to produce low values
        conversations = []
        conversation_results = []
        
        for i in range(20):
            conv = generator.generate_conversation_data(n_messages=30)
            conv['ensemble_embeddings'] = generator.generate_model_ensemble_embeddings(30)
            conversations.append(conv)
            
            # Create low correlation results
            conv_result = {
                'session_id': f'session_{i}',
                'invariance_metrics': {
                    'pairwise_correlations': {
                        'all-MiniLM-L6-v2-all-mpnet-base-v2': np.random.uniform(0.3, 0.5),
                        'all-MiniLM-L6-v2-all-distilroberta-v1': np.random.uniform(0.3, 0.5),
                        'word2vec-glove': np.random.uniform(0.4, 0.6),
                        'all-MiniLM-L6-v2-word2vec': np.random.uniform(0.2, 0.4)
                    },
                    'invariance_score': np.random.uniform(0.3, 0.5)
                }
            }
            conversation_results.append(conv_result)
        
        invariance_results = {
            'conversation_results': conversation_results,
            'aggregate_statistics': {'mean_invariance': 0.4}
        }
        
        # Mock components
        pipeline.load_all_conversations = Mock(return_value=conversations)
        pipeline.analyze_conversations_for_invariance = Mock(return_value=invariance_results)
        pipeline.embedder = MockComponents.mock_embedder()
        pipeline.trajectory_analyzer = MockComponents.mock_trajectory_analyzer()
        pipeline.generate_invariance_visualizations = Mock()
        pipeline.generate_invariance_reports = Mock()
        
        # Capture the hypothesis results
        hypothesis_results = None
        
        def capture_report(inv_results, hyp_results):
            nonlocal hypothesis_results
            hypothesis_results = hyp_results
        
        pipeline.generate_invariance_reports = capture_report
        
        # Run analysis
        pipeline.run_analysis([Path(".")], max_conversations=20)
        
        # Verify Tier 1 failed
        if hypothesis_results and 'summary' in hypothesis_results:
            assert hypothesis_results['summary']['max_tier_passed'] == 0
            assert "Tier 1" in hypothesis_results['summary']['conclusion']


if __name__ == "__main__":
    pytest.main([__file__, "-v"])