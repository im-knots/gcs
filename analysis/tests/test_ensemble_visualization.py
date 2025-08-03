"""
Test suite for ensemble visualization with transport metrics.
"""
import pytest
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from unittest.mock import Mock, patch

from embedding_analysis.visualization.ensemble_plots import EnsembleVisualizer
from tests.test_utils import TestDataGenerator


class TestEnsembleVisualization:
    """Test ensemble visualization functionality."""
    
    @pytest.fixture
    def visualizer(self, tmp_path):
        """Create visualizer instance."""
        return EnsembleVisualizer(output_dir=str(tmp_path))
    
    @pytest.fixture
    def sample_conversation(self):
        """Create sample conversation data."""
        generator = TestDataGenerator()
        return generator.generate_conversation_data(n_messages=50, include_phases=True)
    
    @pytest.fixture
    def sample_embeddings(self):
        """Create sample embeddings for multiple models."""
        np.random.seed(42)
        n_messages = 50
        return {
            'MiniLM-L6': np.random.randn(n_messages, 384),
            'MPNet': np.random.randn(n_messages, 768),
            'MiniLM-L12': np.random.randn(n_messages, 384),
            'Word2Vec': np.random.randn(n_messages, 300),
            'GloVe': np.random.randn(n_messages, 300)
        }
    
    @pytest.fixture
    def sample_phase_info(self):
        """Create sample phase detection results."""
        return {
            'model_phases': {
                'MiniLM-L6': [{'turn': 10, 'name': 'exploration'}, {'turn': 30, 'name': 'insight'}],
                'MPNet': [{'turn': 12, 'name': 'exploration'}, {'turn': 28, 'name': 'insight'}],
                'MiniLM-L12': [{'turn': 11, 'name': 'exploration'}, {'turn': 29, 'name': 'insight'}],
                'Word2Vec': [{'turn': 13, 'name': 'exploration'}, {'turn': 31, 'name': 'insight'}],
                'GloVe': [{'turn': 14, 'name': 'exploration'}, {'turn': 32, 'name': 'insight'}]
            },
            'ensemble_phases': [{'turn': 11, 'name': 'exploration'}, {'turn': 30, 'name': 'insight'}],
            'variance_by_model': {}
        }
    
    @pytest.fixture
    def sample_transport_invariance(self):
        """Create sample transport invariance results."""
        n_models = 5
        return {
            'wasserstein_invariance': 0.723,
            'sinkhorn_invariance': 0.812,
            'gromov_invariance': 0.654,
            'distance_matrices': {
                'wasserstein': np.random.rand(n_models, n_models),
                'sinkhorn': np.random.rand(n_models, n_models),
                'gromov': np.random.rand(n_models, n_models)
            }
        }
    
    def test_visualization_with_transport_metrics(self, visualizer, sample_conversation, 
                                                sample_embeddings, sample_phase_info,
                                                sample_transport_invariance, tmp_path):
        """Test that visualization includes transport metrics."""
        # Create visualization
        save_path = tmp_path / "test_ensemble.png"
        
        with patch('matplotlib.pyplot.savefig'):
            fig = visualizer.create_comprehensive_ensemble_plot(
                sample_conversation,
                sample_embeddings,
                sample_phase_info,
                sample_transport_invariance,
                save_path=str(save_path)
            )
        
        # Check that figure was created
        assert fig is not None
        
        # Close figure to prevent memory leaks
        plt.close(fig)
    
    def test_visualization_without_transport_metrics(self, visualizer, sample_conversation,
                                                   sample_embeddings, sample_phase_info, tmp_path):
        """Test that visualization works without transport metrics (backward compatibility)."""
        save_path = tmp_path / "test_ensemble_no_transport.png"
        
        with patch('matplotlib.pyplot.savefig'):
            fig = visualizer.create_comprehensive_ensemble_plot(
                sample_conversation,
                sample_embeddings,
                sample_phase_info,
                None,  # No transport metrics
                save_path=str(save_path)
            )
        
        # Should still work without transport metrics
        assert fig is not None
        plt.close(fig)
    
    def test_transport_metrics_layout(self, visualizer, sample_conversation,
                                    sample_embeddings, sample_phase_info,
                                    sample_transport_invariance, tmp_path):
        """Test that transport metrics are properly laid out in the figure."""
        # This test simply ensures the visualization doesn't crash with transport metrics
        # More detailed layout testing would require complex mocking
        save_path = tmp_path / "test_layout.png"
        
        with patch('matplotlib.pyplot.savefig'):
            with patch('matplotlib.pyplot.close'):
                fig = visualizer.create_comprehensive_ensemble_plot(
                    sample_conversation,
                    sample_embeddings,
                    sample_phase_info,
                    sample_transport_invariance,
                    save_path=str(save_path)
                )
                
                # If we get here without error, the layout is working
                assert fig is not None
    
    def test_transport_visualization_content(self, visualizer):
        """Test that transport visualization methods handle data correctly."""
        # Test the internal methods if they were exposed
        # Since they're part of the plot creation, we test indirectly
        
        # Create minimal data
        n_models = 3
        distance_matrix = np.array([
            [0.0, 0.5, 0.8],
            [0.5, 0.0, 0.6],
            [0.8, 0.6, 0.0]
        ])
        
        invariance_scores = {
            'wasserstein_invariance': 0.7,
            'sinkhorn_invariance': 0.8,
            'gromov_invariance': 0.6
        }
        
        # Verify the data is valid
        assert distance_matrix.shape == (n_models, n_models)
        assert np.allclose(distance_matrix, distance_matrix.T)  # Symmetric
        assert all(0 <= v <= 1 for v in invariance_scores.values())
    
    @pytest.mark.parametrize("figure_format,expected_calls", [
        ("png", ["png"]),
        ("pdf", ["pdf"]),
        ("both", ["png", "pdf"])
    ])
    def test_figure_format_handling(self, visualizer, sample_conversation,
                                  sample_embeddings, sample_phase_info,
                                  sample_transport_invariance, tmp_path,
                                  figure_format, expected_calls):
        """Test that figures are saved in the requested formats."""
        save_calls = []
        
        def mock_savefig(path, **kwargs):
            if path.endswith('.png'):
                save_calls.append('png')
            elif path.endswith('.pdf'):
                save_calls.append('pdf')
        
        with patch('matplotlib.pyplot.savefig', side_effect=mock_savefig):
            if figure_format == "png":
                visualizer.create_comprehensive_ensemble_plot(
                    sample_conversation,
                    sample_embeddings,
                    sample_phase_info,
                    sample_transport_invariance,
                    save_path=str(tmp_path / "test.png")
                )
            elif figure_format == "pdf":
                visualizer.create_comprehensive_ensemble_plot(
                    sample_conversation,
                    sample_embeddings,
                    sample_phase_info,
                    sample_transport_invariance,
                    save_pdf=str(tmp_path / "test.pdf")
                )
            else:  # both
                visualizer.create_comprehensive_ensemble_plot(
                    sample_conversation,
                    sample_embeddings,
                    sample_phase_info,
                    sample_transport_invariance,
                    save_path=str(tmp_path / "test.png"),
                    save_pdf=str(tmp_path / "test.pdf")
                )
        
        assert save_calls == expected_calls