"""
Unit tests for transport metrics module.
"""
import pytest
import numpy as np
import torch
from embedding_analysis.core.transport_metrics import (
    TransportMetrics, 
    GeometricInvarianceAnalyzer,
    HAS_OT,
    USE_GPU,
    DEVICE
)


class TestTransportMetrics:
    """Test TransportMetrics class functionality."""
    
    @pytest.fixture
    def transport_metrics(self):
        """Create a TransportMetrics instance."""
        return TransportMetrics()
    
    @pytest.fixture
    def sample_trajectories(self):
        """Create sample trajectory data."""
        np.random.seed(42)
        # Create trajectories with different dimensions
        traj1 = np.random.randn(50, 384)  # MiniLM dimension
        traj2 = np.random.randn(50, 768)  # MPNet dimension
        traj3 = np.random.randn(50, 300)  # Word2Vec dimension
        return traj1, traj2, traj3
    
    def test_dimension_mismatch_handling(self, transport_metrics, sample_trajectories):
        """Test that dimension mismatches are handled properly."""
        traj1, traj2, traj3 = sample_trajectories
        
        # Should not raise error despite different dimensions
        dist12 = transport_metrics.compute_wasserstein_distance(traj1, traj2)
        dist13 = transport_metrics.compute_wasserstein_distance(traj1, traj3)
        dist23 = transport_metrics.compute_wasserstein_distance(traj2, traj3)
        
        assert isinstance(dist12, float)
        assert isinstance(dist13, float)
        assert isinstance(dist23, float)
        assert dist12 > 0
        assert dist13 > 0
        assert dist23 > 0
    
    def test_gpu_tensor_handling(self, transport_metrics, sample_trajectories):
        """Test handling of pre-loaded GPU tensors."""
        if not USE_GPU or not HAS_OT:
            pytest.skip("GPU not available or POT not installed")
        
        traj1, traj2, _ = sample_trajectories
        
        # Pre-load to GPU
        traj1_gpu = torch.from_numpy(traj1).float().to(DEVICE)
        traj2_gpu = torch.from_numpy(traj2).float().to(DEVICE)
        
        # Should handle GPU tensors without error
        dist = transport_metrics.compute_wasserstein_distance(traj1_gpu, traj2_gpu)
        assert isinstance(dist, float)
        assert dist > 0
    
    def test_sinkhorn_divergence(self, transport_metrics, sample_trajectories):
        """Test Sinkhorn divergence computation."""
        traj1, traj2, _ = sample_trajectories
        
        # Test with same dimensions
        traj2_same_dim = traj2[:, :traj1.shape[1]]
        div = transport_metrics.compute_sinkhorn_divergence(traj1, traj2_same_dim)
        
        assert isinstance(div, float)
        
        # Test with different dimensions
        div_diff = transport_metrics.compute_sinkhorn_divergence(traj1, traj2)
        assert isinstance(div_diff, float)
    
    def test_gromov_wasserstein(self, transport_metrics, sample_trajectories):
        """Test Gromov-Wasserstein computation."""
        traj1, traj2, _ = sample_trajectories
        
        # Should work with different dimensions since it uses internal distances
        dist = transport_metrics.compute_gromov_wasserstein(traj1, traj2)
        assert isinstance(dist, float)
        assert dist >= 0
    
    def test_trajectory_coupling(self, transport_metrics, sample_trajectories):
        """Test optimal transport coupling computation."""
        traj1, _, traj3 = sample_trajectories
        
        # Test with different dimensions
        coupling = transport_metrics.compute_trajectory_coupling(traj1, traj3)
        
        assert isinstance(coupling, np.ndarray)
        assert coupling.shape == (len(traj1), len(traj3))
        # Check that rows sum to approximately 1/n
        assert np.allclose(coupling.sum(axis=1), 1/len(traj1), atol=1e-6)


class TestGeometricInvarianceAnalyzer:
    """Test GeometricInvarianceAnalyzer functionality."""
    
    @pytest.fixture
    def analyzer(self):
        """Create analyzer instance."""
        return GeometricInvarianceAnalyzer()
    
    @pytest.fixture
    def model_trajectories(self):
        """Create sample trajectories from different models."""
        np.random.seed(42)
        return {
            'MiniLM-L6': np.random.randn(50, 384),
            'MPNet': np.random.randn(50, 768),
            'MiniLM-L12': np.random.randn(50, 384),
            'Word2Vec': np.random.randn(50, 300),
            'GloVe': np.random.randn(50, 300)
        }
    
    def test_compute_transport_invariance_score(self, analyzer, model_trajectories):
        """Test transport invariance score computation."""
        scores = analyzer.compute_transport_invariance_score(model_trajectories)
        
        assert isinstance(scores, dict)
        assert 'wasserstein_invariance' in scores
        assert 'sinkhorn_invariance' in scores
        assert 'gromov_invariance' in scores
        assert 'distance_matrices' in scores
        
        # Check invariance scores are in [0, 1]
        assert 0 <= scores['wasserstein_invariance'] <= 1
        assert 0 <= scores['sinkhorn_invariance'] <= 1
        assert 0 <= scores['gromov_invariance'] <= 1
        
        # Check distance matrices
        n_models = len(model_trajectories)
        assert scores['distance_matrices']['wasserstein'].shape == (n_models, n_models)
        assert scores['distance_matrices']['sinkhorn'].shape == (n_models, n_models)
        assert scores['distance_matrices']['gromov'].shape == (n_models, n_models)
    
    def test_batch_computation(self, analyzer, model_trajectories):
        """Test batch computation of transport invariance."""
        # Create batch of 3 conversations
        batch = [model_trajectories, model_trajectories, model_trajectories]
        
        results = analyzer.compute_transport_invariance_batch(batch)
        
        assert isinstance(results, list)
        assert len(results) == 3
        
        # Each result should have the same structure
        for result in results:
            assert 'wasserstein_invariance' in result
            assert 'sinkhorn_invariance' in result
            assert 'gromov_invariance' in result
    
    def test_progress_bar_functionality(self, analyzer, model_trajectories):
        """Test that progress bars work correctly."""
        # This mainly tests that show_progress doesn't break anything
        scores = analyzer.compute_transport_invariance_score(
            model_trajectories, 
            show_progress=True
        )
        
        assert isinstance(scores, dict)
        assert 'wasserstein_invariance' in scores
    
    @pytest.mark.skipif(not USE_GPU or not HAS_OT, 
                        reason="GPU not available or POT not installed")
    def test_gpu_acceleration(self, analyzer, model_trajectories):
        """Test that GPU acceleration works when available."""
        import time
        
        # Time CPU computation (force CPU by modifying global)
        import embedding_analysis.core.transport_metrics as tm
        original_use_gpu = tm.USE_GPU
        tm.USE_GPU = False
        
        start = time.time()
        scores_cpu = analyzer.compute_transport_invariance_score(model_trajectories)
        cpu_time = time.time() - start
        
        # Restore GPU
        tm.USE_GPU = original_use_gpu
        
        # Time GPU computation
        start = time.time()
        scores_gpu = analyzer.compute_transport_invariance_score(model_trajectories)
        gpu_time = time.time() - start
        
        # Results should be similar (not identical due to numerical differences)
        assert np.abs(scores_cpu['wasserstein_invariance'] - 
                     scores_gpu['wasserstein_invariance']) < 0.1
        
        # GPU should be faster (though this might not always be true for small data)
        print(f"CPU time: {cpu_time:.3f}s, GPU time: {gpu_time:.3f}s")