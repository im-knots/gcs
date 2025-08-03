"""
Test suite for multiscale analysis and visualization.
"""

import pytest
import numpy as np
import tempfile
from pathlib import Path
import matplotlib.pyplot as plt

from embedding_analysis.core.multiscale_analysis import MultiScaleAnalyzer
from embedding_analysis.visualization.multiscale_plots import (
    MultiScaleVisualizer,
    create_scale_comparison_figure
)


class TestMultiScaleAnalysis:
    """Test multiscale analysis functionality."""
    
    @pytest.fixture
    def sample_embeddings(self):
        """Create sample embeddings for testing."""
        n_messages = 100
        n_dims = 384
        
        # Create embeddings with different patterns at different scales
        embeddings = {}
        
        # Create embeddings for multiple models
        for model_name in ['Model1', 'Model2', 'Model3']:
            # Global trend: linear drift
            global_trend = np.linspace(0, 1, n_messages).reshape(-1, 1)
            
            # Meso patterns: sinusoidal segments
            meso_pattern = np.sin(np.linspace(0, 4*np.pi, n_messages)).reshape(-1, 1)
            
            # Local noise
            local_noise = np.random.randn(n_messages, 1) * 0.1
            
            # Combine patterns
            base_pattern = global_trend + 0.5 * meso_pattern + local_noise
            
            # Extend to full dimensionality
            emb = np.tile(base_pattern, (1, n_dims))
            
            # Add model-specific variations
            emb += np.random.randn(n_messages, n_dims) * 0.05
            
            embeddings[model_name] = emb
            
        return embeddings
    
    def test_multiscale_analyzer_structure(self, sample_embeddings):
        """Test that MultiScaleAnalyzer returns expected structure."""
        analyzer = MultiScaleAnalyzer()
        results = analyzer.analyze_all_scales(sample_embeddings)
        
        # Check top-level keys
        assert 'global' in results
        assert 'meso' in results
        assert 'local' in results
        assert 'cross_scale' in results
        
        # Check global scale results
        global_results = results['global']
        assert isinstance(global_results, dict)
        expected_global_keys = ['trajectory_efficiency', 'trajectory_shapes', 
                               'conversation_spread', 'direction_persistence']
        for key in expected_global_keys:
            assert key in global_results, f"Missing key '{key}' in global results"
        
        # Check meso scale results
        meso_results = results['meso']
        assert isinstance(meso_results, dict)
        expected_meso_keys = ['segment_boundaries', 'segment_characteristics',
                             'segment_variability', 'flow_patterns']
        for key in expected_meso_keys:
            assert key in meso_results, f"Missing key '{key}' in meso results"
        
        # Check local scale results
        local_results = results['local']
        assert isinstance(local_results, dict)
        expected_local_keys = ['transition_peaks', 'transition_characteristics',
                              'stability_profiles', 'micro_patterns']
        for key in expected_local_keys:
            assert key in local_results, f"Missing key '{key}' in local results"
    
    def test_multiscale_visualizer_creation(self, sample_embeddings):
        """Test that MultiScaleVisualizer can create figures without errors."""
        analyzer = MultiScaleAnalyzer()
        results = analyzer.analyze_all_scales(sample_embeddings)
        
        # Create conversation data structure
        conv_data = {
            'ensemble_embeddings': sample_embeddings,
            'metadata': {'session_id': 'test_session'}
        }
        
        visualizer = MultiScaleVisualizer()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / 'test_multiscale.png'
            
            # This should not raise an exception
            fig = visualizer.create_multiscale_figure(
                conv_data,
                results,
                save_path=save_path
            )
            
            assert fig is not None
            assert save_path.exists()
            
            # Clean up
            plt.close(fig)
    
    def test_scale_comparison_figure(self, sample_embeddings):
        """Test scale comparison figure generation."""
        # Create multiple conversation results
        analyzer = MultiScaleAnalyzer()
        
        scale_results = {}
        for i in range(3):
            # Slightly modify embeddings for each conversation
            conv_embeddings = {
                k: v + np.random.randn(*v.shape) * 0.01 
                for k, v in sample_embeddings.items()
            }
            
            results = analyzer.analyze_all_scales(conv_embeddings)
            
            # Add mock correlations for visualization
            # (since the actual analyzer doesn't compute these)
            results['global']['correlations'] = np.random.uniform(0.8, 0.95, 10).tolist()
            results['meso']['correlations'] = np.random.uniform(0.6, 0.85, 10).tolist()
            results['local']['correlations'] = np.random.uniform(0.4, 0.7, 10).tolist()
            
            scale_results[f'conv_{i}'] = results
        
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / 'test_scale_comparison.png'
            
            # This should not raise an exception
            fig = create_scale_comparison_figure(scale_results, save_path=save_path)
            
            assert fig is not None
            assert save_path.exists()
            
            # Clean up
            plt.close(fig)
    
    def test_multiscale_with_missing_data(self):
        """Test robustness with missing or incomplete data."""
        # Test with empty embeddings
        analyzer = MultiScaleAnalyzer()
        results = analyzer.analyze_all_scales({})
        
        # Should still return structure but with empty results
        assert 'global' in results
        assert 'meso' in results
        assert 'local' in results
        
        # Test with single model
        single_embedding = {'Model1': np.random.randn(50, 100)}
        results = analyzer.analyze_all_scales(single_embedding)
        
        # Should process without errors
        assert 'global' in results
        assert 'Model1' in results['global'].get('trajectory_efficiency', {})
    
    @pytest.mark.parametrize("n_messages,n_models", [
        (20, 2),    # Small conversation
        (100, 5),   # Medium conversation
        (500, 3),   # Large conversation
    ])
    def test_multiscale_scaling(self, n_messages, n_models):
        """Test multiscale analysis with different sizes."""
        embeddings = {}
        for i in range(n_models):
            embeddings[f'Model{i}'] = np.random.randn(n_messages, 384)
        
        analyzer = MultiScaleAnalyzer()
        results = analyzer.analyze_all_scales(embeddings)
        
        # Verify results are computed for all models
        assert len(results['global']['trajectory_efficiency']) == n_models
        assert len(results['meso']['segment_boundaries']) == n_models
        assert len(results['local']['transition_peaks']) == n_models
    
    def test_visualization_error_handling(self):
        """Test that visualization handles errors gracefully."""
        visualizer = MultiScaleVisualizer()
        
        # Test with invalid data
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / 'test_error.png'
            
            # Should handle None data
            try:
                fig = visualizer.create_multiscale_figure(
                    None, None, save_path=save_path
                )
            except Exception:
                # Expected to fail, but should not crash
                pass
            
            # Should handle empty data
            try:
                fig = visualizer.create_multiscale_figure(
                    {}, {}, save_path=save_path
                )
            except Exception:
                # Expected to fail, but should not crash
                pass


class TestMultiScaleIntegration:
    """Test integration with the main pipeline."""
    
    @pytest.fixture
    def sample_embeddings(self):
        """Create sample embeddings for testing."""
        n_messages = 100
        n_dims = 384
        
        embeddings = {}
        for model_name in ['Model1', 'Model2', 'Model3']:
            embeddings[model_name] = np.random.randn(n_messages, n_dims)
            
        return embeddings
    
    def test_checkpoint_includes_multiscale(self, sample_embeddings):
        """Test that checkpoints include multiscale results."""
        from embedding_analysis.utils import CheckpointManager
        
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_mgr = CheckpointManager(tmpdir)
            
            # Create mock conversation results
            analyzer = MultiScaleAnalyzer()
            multiscale_results = analyzer.analyze_all_scales(sample_embeddings)
            
            checkpoint_data = {
                'session_id': 'test_session',
                'embeddings': sample_embeddings,
                'multiscale_results': multiscale_results,
                'conv_results': {
                    'session_id': 'test_session',
                    'multiscale_results': multiscale_results
                }
            }
            
            # Save checkpoint
            checkpoint_mgr.save(checkpoint_data, 'test_checkpoint')
            
            # Load checkpoint
            loaded_data = checkpoint_mgr.load('test_checkpoint')
            
            # Verify multiscale results are preserved
            assert 'multiscale_results' in loaded_data
            assert 'global' in loaded_data['multiscale_results']
            assert 'meso' in loaded_data['multiscale_results']
            assert 'local' in loaded_data['multiscale_results']