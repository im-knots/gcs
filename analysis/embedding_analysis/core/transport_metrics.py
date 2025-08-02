"""
Optimal transport and information-geometric metrics for conversation analysis.

This module implements advanced geometric measures based on optimal transport
and information geometry to strengthen theoretical foundations.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
try:
    import ot  # Python Optimal Transport
    HAS_OT = True
except ImportError:
    HAS_OT = False
    import warnings
    warnings.warn("POT (Python Optimal Transport) not installed. Transport metrics will use fallback implementations.")
from scipy.special import logsumexp
from scipy.stats import entropy
from scipy.spatial.distance import cdist
import logging

logger = logging.getLogger(__name__)


class TransportMetrics:
    """
    Compute optimal transport-based metrics between conversation trajectories.
    
    These metrics provide a principled way to compare trajectories that is
    invariant to certain transformations and grounded in measure theory.
    """
    
    def __init__(self, regularization: float = 0.1):
        """
        Initialize transport metrics computer.
        
        Args:
            regularization: Entropic regularization parameter for Sinkhorn
        """
        self.regularization = regularization
        
    def compute_wasserstein_distance(self, 
                                   traj1: np.ndarray, 
                                   traj2: np.ndarray,
                                   metric: str = 'euclidean') -> float:
        """
        Compute 2-Wasserstein distance between trajectories.
        
        Args:
            traj1: First trajectory (n_points, dim)
            traj2: Second trajectory (m_points, dim)
            metric: Distance metric to use
            
        Returns:
            Wasserstein distance
        """
        if not HAS_OT:
            # Fallback: Use average pairwise distance as approximation
            distances = cdist(traj1, traj2, metric=metric)
            return np.mean(distances)
        
        # Uniform weights for simplicity (can be extended)
        a = np.ones(len(traj1)) / len(traj1)
        b = np.ones(len(traj2)) / len(traj2)
        
        # Cost matrix
        M = ot.dist(traj1, traj2, metric=metric)
        
        # Compute transport distance
        return ot.emd2(a, b, M)
    
    def compute_sinkhorn_divergence(self,
                                   traj1: np.ndarray,
                                   traj2: np.ndarray) -> float:
        """
        Compute Sinkhorn divergence (debiased Sinkhorn distance).
        
        More robust than Wasserstein for high dimensions.
        """
        if not HAS_OT:
            # Fallback: Use regularized distance
            distances = cdist(traj1, traj2)
            # Simple entropy-regularized approximation
            return np.mean(np.exp(-distances / self.regularization))
        
        a = np.ones(len(traj1)) / len(traj1)
        b = np.ones(len(traj2)) / len(traj2)
        
        M = ot.dist(traj1, traj2)
        
        # Sinkhorn distance
        sinkhorn_ab = ot.sinkhorn2(a, b, M, self.regularization)
        
        # Self-distances for debiasing
        M_aa = ot.dist(traj1, traj1)
        sinkhorn_aa = ot.sinkhorn2(a, a, M_aa, self.regularization)
        
        M_bb = ot.dist(traj2, traj2)
        sinkhorn_bb = ot.sinkhorn2(b, b, M_bb, self.regularization)
        
        # Sinkhorn divergence
        return sinkhorn_ab - 0.5 * (sinkhorn_aa + sinkhorn_bb)
    
    def compute_trajectory_coupling(self,
                                  traj1: np.ndarray,
                                  traj2: np.ndarray) -> np.ndarray:
        """
        Compute optimal transport coupling between trajectories.
        
        Returns:
            Coupling matrix showing correspondence between points
        """
        if not HAS_OT:
            # Fallback: Simple nearest neighbor coupling
            n1, n2 = len(traj1), len(traj2)
            coupling = np.zeros((n1, n2))
            distances = cdist(traj1, traj2)
            for i in range(n1):
                j = np.argmin(distances[i])
                coupling[i, j] = 1 / n1
            return coupling
        
        a = np.ones(len(traj1)) / len(traj1)
        b = np.ones(len(traj2)) / len(traj2)
        
        M = ot.dist(traj1, traj2)
        
        # Compute optimal transport plan
        return ot.emd(a, b, M)
    
    def compute_gromov_wasserstein(self,
                                  traj1: np.ndarray,
                                  traj2: np.ndarray) -> float:
        """
        Compute Gromov-Wasserstein distance.
        
        This is invariant to isometric transformations and doesn't require
        trajectories to be in the same space.
        """
        if not HAS_OT:
            # Fallback: Compare distance matrix statistics
            C1 = cdist(traj1, traj1)
            C2 = cdist(traj2, traj2)
            # Simple approximation using Frobenius norm
            return np.linalg.norm(C1.flatten() - C2.flatten()) / (len(traj1) * len(traj2))
        
        # Internal distance matrices
        C1 = ot.dist(traj1, traj1)
        C2 = ot.dist(traj2, traj2)
        
        # Uniform distributions
        p = np.ones(len(traj1)) / len(traj1)
        q = np.ones(len(traj2)) / len(traj2)
        
        # Gromov-Wasserstein distance
        return ot.gromov_wasserstein2(C1, C2, p, q)


class InformationGeometry:
    """
    Information-geometric analysis of conversation trajectories.
    
    Treats embedding distributions as points on a statistical manifold.
    """
    
    def __init__(self):
        """Initialize information geometry analyzer."""
        pass
    
    def compute_fisher_information_matrix(self,
                                        embeddings: np.ndarray,
                                        window_size: int = 10) -> np.ndarray:
        """
        Estimate local Fisher information matrix along trajectory.
        
        Args:
            embeddings: Trajectory embeddings (n_points, dim)
            window_size: Window for local estimation
            
        Returns:
            Fisher information matrices at each point
        """
        n_points = len(embeddings)
        dim = embeddings.shape[1]
        fisher_matrices = []
        
        for i in range(n_points):
            # Get local window
            start = max(0, i - window_size // 2)
            end = min(n_points, i + window_size // 2)
            local_embeddings = embeddings[start:end]
            
            if len(local_embeddings) > 1:
                # Estimate covariance as proxy for Fisher information
                centered = local_embeddings - np.mean(local_embeddings, axis=0)
                cov = np.cov(centered.T)
                
                # Regularize for stability
                cov += 1e-6 * np.eye(dim)
                
                # Fisher information is inverse covariance
                try:
                    fisher = np.linalg.inv(cov)
                except np.linalg.LinAlgError:
                    fisher = np.eye(dim)
            else:
                fisher = np.eye(dim)
                
            fisher_matrices.append(fisher)
            
        return np.array(fisher_matrices)
    
    def compute_geodesic_distance(self,
                                 embeddings: np.ndarray,
                                 start_idx: int,
                                 end_idx: int) -> float:
        """
        Approximate geodesic distance on the statistical manifold.
        
        Uses Fisher information as the metric tensor.
        """
        fisher_matrices = self.compute_fisher_information_matrix(embeddings)
        
        # Approximate geodesic distance by summing infinitesimal distances
        total_distance = 0
        
        for i in range(start_idx, end_idx):
            # Tangent vector
            if i < len(embeddings) - 1:
                v = embeddings[i+1] - embeddings[i]
                
                # Riemannian distance: sqrt(v^T G v) where G is metric tensor
                G = fisher_matrices[i]
                distance = np.sqrt(np.dot(v, np.dot(G, v)))
                total_distance += distance
                
        return total_distance
    
    def compute_trajectory_curvature_tensor(self,
                                          embeddings: np.ndarray) -> List[float]:
        """
        Compute curvature along trajectory using information geometry.
        
        Returns:
            Scalar curvatures at each point
        """
        fisher_matrices = self.compute_fisher_information_matrix(embeddings)
        curvatures = []
        
        for i in range(1, len(embeddings) - 1):
            # Finite difference approximation of curvature
            G_prev = fisher_matrices[i-1]
            G_curr = fisher_matrices[i]
            G_next = fisher_matrices[i+1]
            
            # Christoffel symbols (simplified)
            dG = G_next - G_prev
            
            # Scalar curvature (trace of Ricci tensor approximation)
            try:
                G_inv = np.linalg.inv(G_curr)
                ricci_trace = np.trace(np.dot(G_inv, dG))
                curvatures.append(abs(ricci_trace))
            except np.linalg.LinAlgError:
                curvatures.append(0.0)
                
        return curvatures
    
    def compute_kl_divergence_trajectory(self,
                                       embeddings: np.ndarray,
                                       reference_dist: Optional[np.ndarray] = None) -> List[float]:
        """
        Compute KL divergence from reference distribution along trajectory.
        
        Treats each embedding as parameters of a distribution.
        """
        if reference_dist is None:
            # Use mean embedding as reference
            reference_dist = np.mean(embeddings, axis=0)
            
        # Normalize embeddings to probability distributions
        # Using softmax for stability
        def to_prob_dist(embedding):
            # Shift for numerical stability
            shifted = embedding - np.max(embedding)
            exp_vals = np.exp(shifted)
            return exp_vals / np.sum(exp_vals)
        
        ref_prob = to_prob_dist(reference_dist)
        
        kl_divergences = []
        for embedding in embeddings:
            prob = to_prob_dist(embedding)
            kl = entropy(prob, ref_prob)
            kl_divergences.append(kl)
            
        return kl_divergences


class GeometricInvarianceAnalyzer:
    """
    Analyze geometric invariance using transport and information geometry.
    """
    
    def __init__(self):
        """Initialize analyzer."""
        self.transport = TransportMetrics()
        self.info_geom = InformationGeometry()
        
    def compute_transport_invariance_score(self,
                                         trajectories: Dict[str, np.ndarray]) -> Dict[str, float]:
        """
        Compute invariance score based on optimal transport distances.
        
        Args:
            trajectories: Dict mapping model names to trajectory embeddings
            
        Returns:
            Invariance scores based on different metrics
        """
        model_names = list(trajectories.keys())
        n_models = len(model_names)
        
        # Compute pairwise transport distances
        wasserstein_distances = np.zeros((n_models, n_models))
        sinkhorn_distances = np.zeros((n_models, n_models))
        gromov_distances = np.zeros((n_models, n_models))
        
        for i in range(n_models):
            for j in range(i+1, n_models):
                traj_i = trajectories[model_names[i]]
                traj_j = trajectories[model_names[j]]
                
                # Wasserstein distance
                w_dist = self.transport.compute_wasserstein_distance(traj_i, traj_j)
                wasserstein_distances[i, j] = w_dist
                wasserstein_distances[j, i] = w_dist
                
                # Sinkhorn divergence
                s_dist = self.transport.compute_sinkhorn_divergence(traj_i, traj_j)
                sinkhorn_distances[i, j] = s_dist
                sinkhorn_distances[j, i] = s_dist
                
                # Gromov-Wasserstein (invariant to isometries)
                g_dist = self.transport.compute_gromov_wasserstein(traj_i, traj_j)
                gromov_distances[i, j] = g_dist
                gromov_distances[j, i] = g_dist
        
        # Compute invariance scores (lower distances = higher invariance)
        # Normalize by average distance
        avg_wasserstein = np.mean(wasserstein_distances[np.triu_indices(n_models, k=1)])
        avg_sinkhorn = np.mean(sinkhorn_distances[np.triu_indices(n_models, k=1)])
        avg_gromov = np.mean(gromov_distances[np.triu_indices(n_models, k=1)])
        
        return {
            'wasserstein_invariance': 1 / (1 + avg_wasserstein),
            'sinkhorn_invariance': 1 / (1 + avg_sinkhorn),
            'gromov_invariance': 1 / (1 + avg_gromov),
            'distance_matrices': {
                'wasserstein': wasserstein_distances,
                'sinkhorn': sinkhorn_distances,
                'gromov': gromov_distances
            }
        }
    
    def compute_information_geometric_invariance(self,
                                               trajectories: Dict[str, np.ndarray]) -> Dict[str, float]:
        """
        Compute invariance using information-geometric measures.
        """
        scores = {}
        
        # Compute curvature profiles for each trajectory
        curvature_profiles = {}
        for model_name, traj in trajectories.items():
            curvatures = self.info_geom.compute_trajectory_curvature_tensor(traj)
            curvature_profiles[model_name] = curvatures
        
        # Compare curvature profiles
        model_names = list(trajectories.keys())
        curvature_correlations = []
        
        for i in range(len(model_names)):
            for j in range(i+1, len(model_names)):
                curv_i = curvature_profiles[model_names[i]]
                curv_j = curvature_profiles[model_names[j]]
                
                # Align lengths
                min_len = min(len(curv_i), len(curv_j))
                if min_len > 0:
                    corr = np.corrcoef(curv_i[:min_len], curv_j[:min_len])[0, 1]
                    if not np.isnan(corr):
                        curvature_correlations.append(corr)
        
        if curvature_correlations:
            scores['curvature_correlation'] = np.mean(curvature_correlations)
        else:
            scores['curvature_correlation'] = 0.0
            
        # Compare KL divergence trajectories
        kl_profiles = {}
        for model_name, traj in trajectories.items():
            kl_trajectory = self.info_geom.compute_kl_divergence_trajectory(traj)
            kl_profiles[model_name] = kl_trajectory
            
        kl_correlations = []
        for i in range(len(model_names)):
            for j in range(i+1, len(model_names)):
                kl_i = kl_profiles[model_names[i]]
                kl_j = kl_profiles[model_names[j]]
                
                min_len = min(len(kl_i), len(kl_j))
                if min_len > 0:
                    corr = np.corrcoef(kl_i[:min_len], kl_j[:min_len])[0, 1]
                    if not np.isnan(corr):
                        kl_correlations.append(corr)
                        
        if kl_correlations:
            scores['kl_trajectory_correlation'] = np.mean(kl_correlations)
        else:
            scores['kl_trajectory_correlation'] = 0.0
            
        return scores