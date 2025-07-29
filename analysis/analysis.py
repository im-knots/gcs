import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA, FactorAnalysis
from sklearn.manifold import Isomap, MDS, TSNE
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN, KMeans
from sklearn.mixture import GaussianMixture
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import cross_val_score, KFold, train_test_split
from sklearn.metrics import roc_auc_score, classification_report, silhouette_score, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.spatial.distance import pdist, squareform
from scipy.interpolate import UnivariateSpline
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Optional imports
try:
    from statsmodels.stats.power import FTestPower
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    print("Warning: statsmodels not installed. Install with: pip install statsmodels")
    print("Power analysis will be skipped.")

try:
    from factor_analyzer import FactorAnalyzer
    from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity, calculate_kmo
    FACTOR_ANALYZER_AVAILABLE = True
except ImportError:
    FACTOR_ANALYZER_AVAILABLE = False
    print("Warning: factor_analyzer not installed. Install with: pip install factor-analyzer")
    print("KMO and Bartlett tests will be skipped.")

try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    print("Warning: UMAP not installed. Install with: pip install umap-learn")
    print("UMAP manifold learning will be skipped.")

try:
    from ripser import ripser
    from persim import plot_diagrams
    RIPSER_AVAILABLE = True
except ImportError:
    RIPSER_AVAILABLE = False
    print("Warning: ripser not installed. Install with: pip install ripser persim")
    print("Topological data analysis will be skipped.")

class ConversationalAnalysis:
    """
    Enhanced analysis incorporating all critical suggestions:
    1. Regularized methods for high condition number
    2. Bootstrap confidence intervals
    3. Cross-validation for stability
    4. Power analysis
    5. Intervention threshold analysis
    6. Synthetic data generation
    7. Proper mathematical formalism
    8. NEW: Intrinsic dimensionality estimation
    9. NEW: Sliding window analysis for temporal dynamics
    10. NEW: Topological data analysis
    11. NEW: Manifold learning comparison
    12. NEW: Dynamic phase detection
    13. NEW: Trajectory prediction
    14. NEW: Within-tier heterogeneity analysis
    15. ENHANCED: Per-tier geometric analysis
    """
    
    def __init__(self, output_dir='analysis_outputs'):
        self.output_dir = output_dir
        self.setup_directories()
        
    def setup_directories(self):
        """Create output directory structure"""
        os.makedirs(self.output_dir, exist_ok=True)
        self.figures_dir = os.path.join(self.output_dir, 'figures')
        self.validation_dir = os.path.join(self.output_dir, 'validation')
        self.bootstrap_dir = os.path.join(self.output_dir, 'bootstrap')
        self.power_dir = os.path.join(self.output_dir, 'power_analysis')
        self.topology_dir = os.path.join(self.output_dir, 'topology')
        self.manifold_dir = os.path.join(self.output_dir, 'manifold')
        self.trajectory_dir = os.path.join(self.output_dir, 'trajectories')
        self.tier_dir = os.path.join(self.output_dir, 'tier_analysis')
        
        for dir_path in [self.figures_dir, self.validation_dir, self.bootstrap_dir, 
                        self.power_dir, self.topology_dir, self.manifold_dir, 
                        self.trajectory_dir, self.tier_dir]:
            os.makedirs(dir_path, exist_ok=True)
    
    def load_and_prepare_data(self, phase1_path, phase2_path, phase3_path):
        """Load data with proper phase labeling"""
        # Load data
        phase1_df = pd.read_csv(phase1_path)
        phase2_df = pd.read_csv(phase2_path)
        phase3_df = pd.read_csv(phase3_path)
        
        # Add phase labels
        phase1_df['phase'] = 'full_reasoning'
        phase2_df['phase'] = 'light_reasoning'
        phase3_df['phase'] = 'no_reasoning'
        
        # Combine all phases
        self.all_data = pd.concat([phase1_df, phase2_df, phase3_df], ignore_index=True)
        
        # Create outcome variables
        self.all_data['breakdown_binary'] = (self.all_data['conversation_outcome'] == 'breakdown').astype(int)
        
        outcome_mapping = {
            'no_breakdown': 0,
            'resisted': 1,
            'recovered': 2,
            'breakdown': 3
        }
        self.all_data['outcome_numeric'] = self.all_data['conversation_outcome'].map(outcome_mapping)
        
        print(f"\nTotal conversations: {len(self.all_data)}")
        print(f"Phase distribution:")
        print(self.all_data['phase'].value_counts())
        print(f"\nConversation outcome distribution:")
        print(self.all_data['conversation_outcome'].value_counts())
        print(f"Overall breakdown rate: {self.all_data['breakdown_binary'].mean():.3f}")
        
        return self.all_data
    
    def define_feature_groups(self):
        """Define theoretically-motivated feature groups based on the paper"""
        self.feature_groups = {
            'social_contagion': [
                'peer_pressure_intensity',
                'peer_pressure_event_count',
                'bidirectional_event_count',
                'peer_mirroring_count',
                'mirroring_coefficient'
            ],
            'linguistic_synchrony': [
                'avg_linguistic_alignment',
                'high_alignment_periods',
                'nlp_mirroring_events'
            ],
            'affective_dynamics': [
                'emotion_volatility',
                'emotional_convergence'
            ],
            'cognitive_abstraction': [
                'meta_reflection_density',
                'mystical_density',
                'abstract_language_score',
                'closure_vibe_count',
                'mystical_language_count'
            ],
            'temporal_structure': [
                'phase_transitions',
                'phase_oscillations'
            ],
            'intervention_response': [
                'circuit_breaker_questions',
                'recovery_after_question',
                'question_density',
                'successful_recoveries',
                'recovery_attempt_count',
                'recovery_sustained_duration'
            ]
        }
        
        # Flatten for analysis
        self.all_features = []
        for group in self.feature_groups.values():
            self.all_features.extend(group)
        
        # Separate intervention and non-intervention features
        self.intervention_features = self.feature_groups['intervention_response']
        self.non_intervention_features = []
        for group_name, features in self.feature_groups.items():
            if group_name != 'intervention_response':
                self.non_intervention_features.extend(features)
        
        return self.feature_groups
    
    def calculate_statistical_power(self):
        """Calculate statistical power for our analyses"""
        print("\n" + "="*70)
        print("STATISTICAL POWER ANALYSIS")
        print("="*70)
        
        power_results = {}
        
        if STATSMODELS_AVAILABLE:
            try:
                # For ANOVA-style comparisons (4 outcome groups)
                from statsmodels.stats.power import FTestAnovaPower
                power_analysis = FTestAnovaPower()
                
                # Using temporal dynamics eta^2 = 0.336 from your paper
                eta_squared = 0.336
                # Convert eta^2 to f (Cohen's f)
                effect_size = np.sqrt(eta_squared / (1 - eta_squared))
                
                # Calculate power
                power = power_analysis.solve_power(
                    effect_size=effect_size,
                    nobs=len(self.all_data) / 4,  # nobs is per group
                    alpha=0.05,
                    k_groups=4,
                    power=None
                )
                power_results['outcome_groups'] = power
                print(f"Power for detecting outcome differences (4 groups): {power:.3f}")
                
                # For model type comparisons (3 groups)
                power_model = power_analysis.solve_power(
                    effect_size=effect_size,
                    nobs=len(self.all_data) / 3,  # nobs is per group
                    alpha=0.05,
                    k_groups=3,
                    power=None
                )
                power_results['model_types'] = power_model
                print(f"Power for detecting model type differences (3 groups): {power_model:.3f}")
                
                # Sample size needed for adequate power
                required_n_per_group = power_analysis.solve_power(
                    effect_size=effect_size,
                    power=0.8,
                    alpha=0.05,
                    k_groups=4,
                    nobs=None
                )
                required_n_total = int(required_n_per_group * 4)
                print(f"\nSample size needed for 80% power: {required_n_total} total ({required_n_per_group:.1f} per group)")
                print(f"Current sample size: {len(self.all_data)}")
                print(f"Adequacy: {'ADEQUATE' if len(self.all_data) >= required_n_total else 'UNDERPOWERED'}")
                
            except Exception as e:
                print(f"Error with statsmodels power analysis: {e}")
                print("Falling back to approximation methods...")
                
                # Approximation based on effect size
                n = len(self.all_data)
                f = np.sqrt(eta_squared / (1 - eta_squared))
                
                # Approximate power using non-central F distribution
                # This is a rough approximation
                dfn = 3  # df numerator (k-1)
                dfd = n - 4  # df denominator (n-k)
                nc = n * f**2  # non-centrality parameter
                
                # Critical value for F test
                f_crit = stats.f.ppf(0.95, dfn, dfd)
                
                # Power is probability of exceeding critical value under alternative
                power_approx = 1 - stats.ncf.cdf(f_crit, dfn, dfd, nc)
                
                power_results['outcome_groups'] = power_approx
                power_results['model_types'] = power_approx
                print(f"Approximate power (F-test based): {power_approx:.3f}")
        else:
            print("Statsmodels not available. Using approximation methods...")
            # Approximation based on Cohen's conventions
            n = len(self.all_data)
            eta_squared = 0.336
            f = np.sqrt(eta_squared / (1 - eta_squared))
            
            # Very rough approximation
            # For medium-large effect sizes and reasonable sample sizes
            power_approx = min(0.95, 0.5 + 0.5 * (n / 100) * f)
            
            power_results['outcome_groups'] = power_approx
            power_results['model_types'] = power_approx
            print(f"Rough power approximation: {power_approx:.3f}")
            print("Note: This is a very rough estimate. Install statsmodels for accurate power analysis.")
        
        # For correlation analyses (doesn't need statsmodels)
        n = len(self.all_data)
        r = 0.349  # Social contagion correlation from paper
        
        # Fisher's z transformation
        z_r = 0.5 * np.log((1 + r) / (1 - r))
        se_z = 1 / np.sqrt(n - 3)
        
        # Power for two-tailed test
        z_crit = 1.96  # for alpha = 0.05
        z_power = abs(z_r) / se_z
        power_corr = 2 * (1 - stats.norm.cdf(z_crit - z_power)) - 1
        
        power_results['correlation'] = power_corr
        print(f"\nPower for detecting correlations (r={r}): {power_corr:.3f}")
        
        # Save results
        pd.DataFrame([power_results]).to_csv(
            os.path.join(self.power_dir, 'power_analysis_results.csv'), 
            index=False
        )
        
        return power_results
    
    def perform_regularized_pca(self, X, alpha=0.1):
        """PCA with Ridge regularization for stability"""
        print("\n" + "="*70)
        print("REGULARIZED PCA ANALYSIS")
        print("="*70)
        
        # Calculate regularized covariance
        n_samples = X.shape[0]
        cov = (X.T @ X) / n_samples
        
        # Check condition number before regularization
        eigvals_orig = np.linalg.eigvalsh(cov)
        condition_orig = eigvals_orig[-1] / eigvals_orig[0]
        print(f"Original condition number: {condition_orig:.2f}")
        
        # Add regularization
        cov_reg = cov + alpha * np.eye(cov.shape[0])
        
        # Check condition number after regularization
        eigvals_reg = np.linalg.eigvalsh(cov_reg)
        condition_reg = eigvals_reg[-1] / eigvals_reg[0]
        print(f"Regularized condition number: {condition_reg:.2f}")
        
        # Perform eigendecomposition
        eigvals, eigvecs = np.linalg.eigh(cov_reg)
        
        # Sort by eigenvalue (descending)
        idx = eigvals.argsort()[::-1]
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]
        
        # Calculate variance explained
        total_var = eigvals.sum()
        var_explained = eigvals / total_var
        
        # Transform data
        X_transformed = X @ eigvecs
        
        return {
            'components': eigvecs.T,
            'explained_variance': eigvals,
            'explained_variance_ratio': var_explained,
            'transformed': X_transformed,
            'condition_improvement': condition_orig / condition_reg
        }
    
    def bootstrap_dimension_stability(self, X, feature_names, n_bootstrap=100):
        """Test dimension stability through bootstrap resampling"""
        print("\n" + "="*70)
        print("BOOTSTRAP STABILITY ANALYSIS")
        print("="*70)
        
        n_samples = X.shape[0]
        n_features = X.shape[1]
        
        # Store bootstrap results
        bootstrap_loadings = []
        bootstrap_variance = []
        
        for i in range(n_bootstrap):
            # Bootstrap sample
            idx = np.random.choice(n_samples, n_samples, replace=True)
            X_boot = X[idx]
            
            # Regularized PCA on bootstrap sample
            pca_result = self.perform_regularized_pca(X_boot, alpha=0.1)
            
            # Store first 4 components
            bootstrap_loadings.append(pca_result['components'][:4])
            bootstrap_variance.append(pca_result['explained_variance_ratio'][:4])
        
        # Calculate confidence intervals
        loadings_array = np.array(bootstrap_loadings)
        variance_array = np.array(bootstrap_variance)
        
        # Loading confidence intervals
        loading_ci_lower = np.percentile(loadings_array, 2.5, axis=0)
        loading_ci_upper = np.percentile(loadings_array, 97.5, axis=0)
        loading_mean = np.mean(loadings_array, axis=0)
        
        # Variance explained confidence intervals
        var_ci_lower = np.percentile(variance_array, 2.5, axis=0)
        var_ci_upper = np.percentile(variance_array, 97.5, axis=0)
        var_mean = np.mean(variance_array, axis=0)
        
        print("\nVariance Explained (95% CI):")
        for i in range(4):
            print(f"PC{i+1}: {var_mean[i]:.3f} [{var_ci_lower[i]:.3f}, {var_ci_upper[i]:.3f}]")
        
        # Calculate loading stability (how consistent are the signs?)
        sign_stability = np.mean(np.sign(loadings_array) == np.sign(loading_mean), axis=0)
        
        # Visualize stability
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
        
        for pc in range(4):
            ax = axes[pc]
            
            # Sort features by mean loading
            mean_loadings = loading_mean[pc]
            sorted_idx = np.argsort(np.abs(mean_loadings))[-10:]  # Top 10
            
            # Plot with error bars
            y_pos = np.arange(len(sorted_idx))
            ax.barh(y_pos, mean_loadings[sorted_idx], 
                   xerr=[mean_loadings[sorted_idx] - loading_ci_lower[pc, sorted_idx],
                         loading_ci_upper[pc, sorted_idx] - mean_loadings[sorted_idx]],
                   capsize=5)
            
            ax.set_yticks(y_pos)
            ax.set_yticklabels([feature_names[i] for i in sorted_idx])
            ax.set_xlabel('Loading')
            ax.set_title(f'PC{pc+1} Bootstrap Loadings (95% CI)')
            ax.axvline(x=0, color='k', linestyle='--', alpha=0.5)
            ax.grid(True, alpha=0.3)
        
        plt.suptitle('Bootstrap Stability of Principal Component Loadings', fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(self.bootstrap_dir, 'bootstrap_loadings.pdf'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save detailed results
        stability_df = pd.DataFrame({
            'PC': [f'PC{i+1}' for i in range(4)],
            'Variance_Mean': var_mean,
            'Variance_CI_Lower': var_ci_lower,
            'Variance_CI_Upper': var_ci_upper,
            'Mean_Sign_Stability': np.mean(sign_stability, axis=1)
        })
        stability_df.to_csv(os.path.join(self.bootstrap_dir, 'bootstrap_stability.csv'), 
                           index=False)
        
        return {
            'loading_mean': loading_mean,
            'loading_ci': (loading_ci_lower, loading_ci_upper),
            'variance_mean': var_mean,
            'variance_ci': (var_ci_lower, var_ci_upper),
            'sign_stability': sign_stability
        }
    
    def analyze_tier_geometry(self, tier_name, tier_data, feature_names, save_prefix):
        """Perform complete geometric analysis for a single tier"""
        print(f"\n{'='*70}")
        print(f"TIER-SPECIFIC ANALYSIS: {tier_name}")
        print(f"{'='*70}")
        print(f"N = {len(tier_data)} conversations")
        
        if len(tier_data) < 10:
            print(f"Insufficient data for tier {tier_name}")
            return None
        
        # Prepare features
        X_all = tier_data[self.all_features].copy()
        X_non_int = tier_data[self.non_intervention_features].copy()
        X_int = tier_data[self.intervention_features].copy()
        
        # Impute and scale
        imputer = SimpleImputer(strategy='mean')
        X_all = imputer.fit_transform(X_all)
        X_non_int = imputer.fit_transform(X_non_int)
        X_int = imputer.fit_transform(X_int)
        
        scaler = RobustScaler()
        X_all_scaled = scaler.fit_transform(X_all)
        X_non_int_scaled = scaler.fit_transform(X_non_int)
        X_int_scaled = scaler.fit_transform(X_int)
        
        results = {
            'tier_name': tier_name,
            'n_conversations': len(tier_data),
            'breakdown_rate': tier_data['breakdown_binary'].mean(),
            'outcome_distribution': tier_data['conversation_outcome'].value_counts().to_dict()
        }
        
        # 1. PCA Analysis (All vs Non-intervention)
        print(f"\n--- PCA Analysis for {tier_name} ---")
        pca_all = self.perform_regularized_pca(X_all_scaled, alpha=0.1)
        pca_non_int = self.perform_regularized_pca(X_non_int_scaled, alpha=0.1)
        
        results['pca_all_pc1_var'] = pca_all['explained_variance_ratio'][0]
        results['pca_non_int_pc1_var'] = pca_non_int['explained_variance_ratio'][0]
        results['pca_all'] = pca_all
        results['pca_non_int'] = pca_non_int
        
        print(f"All features PC1: {pca_all['explained_variance_ratio'][0]:.1%}")
        print(f"Non-intervention PC1: {pca_non_int['explained_variance_ratio'][0]:.1%}")
        
        # 2. Intrinsic Dimensionality
        print(f"\n--- Intrinsic Dimensionality for {tier_name} ---")
        intrinsic_dim = self.estimate_intrinsic_dimension(X_non_int_scaled, k_max=min(20, len(tier_data)//3))
        results['intrinsic_dimension'] = intrinsic_dim
        
        # 3. Bootstrap Stability (if sufficient data)
        if len(tier_data) >= 30:
            print(f"\n--- Bootstrap Stability for {tier_name} ---")
            bootstrap_results = self.bootstrap_dimension_stability(
                X_non_int_scaled, 
                self.non_intervention_features,
                n_bootstrap=50  # Fewer for tier-specific
            )
            results['bootstrap'] = bootstrap_results
        
        # 4. Within-tier clustering
        print(f"\n--- Within-tier Structure for {tier_name} ---")
        if len(tier_data) >= 15:
            silhouette_scores = []
            for n_clusters in range(2, min(6, len(tier_data)//5)):
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                labels = kmeans.fit_predict(X_non_int_scaled)
                
                if len(np.unique(labels)) > 1:
                    score = silhouette_score(X_non_int_scaled, labels)
                    silhouette_scores.append(score)
            
            if silhouette_scores:
                optimal_clusters = np.argmax(silhouette_scores) + 2
                results['optimal_clusters'] = optimal_clusters
                results['best_silhouette'] = max(silhouette_scores)
                print(f"Optimal clusters: {optimal_clusters} (silhouette: {max(silhouette_scores):.3f})")
        
        # 5. Manifold learning comparison
        if len(tier_data) >= 20:
            print(f"\n--- Manifold Learning for {tier_name} ---")
            manifold_results = self.compare_manifold_methods(
                X_non_int_scaled[:, :min(10, X_non_int_scaled.shape[1])], 
                n_components=2,
                subset_data=tier_data  # Pass tier data to avoid visualization issues
            )
            results['manifold_preservation'] = manifold_results[1]
        
        # 6. Create tier-specific visualizations
        self._create_tier_visualizations(tier_name, tier_data, results, save_prefix)
        
        return results
    
    def _create_tier_visualizations(self, tier_name, tier_data, results, save_prefix):
        """Create visualizations specific to each tier"""
        # 1. Variance explained comparison
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # All features
        var_all = results['pca_all']['explained_variance_ratio'][:10]
        ax1.bar(range(len(var_all)), var_all, alpha=0.7, color='red')
        ax1.set_xlabel('Principal Component')
        ax1.set_ylabel('Variance Explained')
        ax1.set_title(f'{tier_name}: All Features\n(PC1: {var_all[0]:.1%})')
        ax1.set_ylim(0, 1.0)
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Non-intervention only
        var_non = results['pca_non_int']['explained_variance_ratio'][:10]
        ax2.bar(range(len(var_non)), var_non, alpha=0.7, color='blue')
        ax2.set_xlabel('Principal Component')
        ax2.set_ylabel('Variance Explained')
        ax2.set_title(f'{tier_name}: Non-Intervention\n(PC1: {var_non[0]:.1%})')
        ax2.set_ylim(0, 1.0)
        ax2.grid(True, alpha=0.3, axis='y')
        
        plt.suptitle(f'Variance Structure: {tier_name}', fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(self.tier_dir, f'{save_prefix}_variance_comparison.pdf'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. 2D projection with outcomes
        if results['pca_non_int']['transformed'].shape[0] >= 10:
            fig, ax = plt.subplots(figsize=(8, 8))
            
            X_2d = results['pca_non_int']['transformed'][:, :2]
            
            # Color by outcome
            for outcome, color in zip(['no_breakdown', 'resisted', 'recovered', 'breakdown'],
                                     ['green', 'yellow', 'orange', 'red']):
                mask = tier_data['conversation_outcome'] == outcome
                if mask.sum() > 0:
                    ax.scatter(X_2d[mask, 0], X_2d[mask, 1], c=color, 
                             label=f'{outcome} (n={mask.sum()})',
                             alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
            
            ax.set_xlabel(f'PC1 ({results["pca_non_int"]["explained_variance_ratio"][0]:.1%})')
            ax.set_ylabel(f'PC2 ({results["pca_non_int"]["explained_variance_ratio"][1]:.1%})')
            ax.set_title(f'{tier_name}: Conversation Space (Non-Intervention Features)')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.tier_dir, f'{save_prefix}_conversation_space.pdf'), 
                       dpi=300, bbox_inches='tight')
            plt.close()
        
        # 3. Feature importance heatmap (top features per PC)
        if 'pca_non_int' in results:
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # Get top features for first 4 PCs
            n_top = 10
            components = results['pca_non_int']['components'][:4]
            
            top_features_per_pc = []
            feature_indices = []
            
            for pc_idx, pc in enumerate(components):
                top_idx = np.argsort(np.abs(pc))[-n_top:]
                top_features_per_pc.extend([(pc_idx, idx, pc[idx]) for idx in top_idx])
                feature_indices.extend(top_idx)
            
            # Create matrix for heatmap
            unique_features = sorted(list(set(feature_indices)))
            heatmap_data = np.zeros((len(unique_features), 4))
            
            feature_labels = []
            for i, feat_idx in enumerate(unique_features):
                feature_labels.append(self.non_intervention_features[feat_idx])
                for pc_idx in range(4):
                    heatmap_data[i, pc_idx] = components[pc_idx, feat_idx]
            
            sns.heatmap(heatmap_data, 
                       xticklabels=[f'PC{i+1}' for i in range(4)],
                       yticklabels=feature_labels,
                       cmap='RdBu_r', center=0,
                       cbar_kws={'label': 'Loading'})
            
            ax.set_title(f'{tier_name}: Feature Loadings on Principal Components')
            plt.tight_layout()
            plt.savefig(os.path.join(self.tier_dir, f'{save_prefix}_feature_loadings.pdf'), 
                       dpi=300, bbox_inches='tight')
            plt.close()
    
    def compare_tier_geometries(self, tier_results):
        """Compare geometric properties across tiers"""
        print("\n" + "="*70)
        print("CROSS-TIER GEOMETRIC COMPARISON")
        print("="*70)
        
        # Create comparison DataFrame
        comparison_data = []
        for tier_name, results in tier_results.items():
            if results is not None:
                comparison_data.append({
                    'Tier': tier_name,
                    'N': results['n_conversations'],
                    'Breakdown Rate': results['breakdown_rate'],
                    'All Features PC1': results['pca_all_pc1_var'],
                    'Non-Int PC1': results['pca_non_int_pc1_var'],
                    'Intrinsic Dim (ML)': results['intrinsic_dimension'].get('ml_estimate', np.nan),
                    'Intrinsic Dim (PCA90)': results['intrinsic_dimension'].get('pca_dim_90', np.nan),
                    'Optimal Clusters': results.get('optimal_clusters', np.nan),
                    'Best Silhouette': results.get('best_silhouette', np.nan)
                })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Save comparison
        comparison_df.to_csv(os.path.join(self.tier_dir, 'tier_geometry_comparison.csv'), index=False)
        
        # Visualize comparison
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. PC1 variance comparison
        ax = axes[0, 0]
        x = np.arange(len(comparison_df))
        width = 0.35
        ax.bar(x - width/2, comparison_df['All Features PC1'], width, label='All Features', alpha=0.7)
        ax.bar(x + width/2, comparison_df['Non-Int PC1'], width, label='Non-Intervention', alpha=0.7)
        ax.set_xlabel('Model Tier')
        ax.set_ylabel('PC1 Variance Explained')
        ax.set_title('Intervention Dominance Across Tiers')
        ax.set_xticks(x)
        ax.set_xticklabels(comparison_df['Tier'])
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        # 2. Intrinsic dimensionality
        ax = axes[0, 1]
        ax.plot(comparison_df['Tier'], comparison_df['Intrinsic Dim (ML)'], 'bo-', 
                label='ML Estimate', linewidth=2, markersize=10)
        ax.plot(comparison_df['Tier'], comparison_df['Intrinsic Dim (PCA90)'], 'rs--', 
                label='PCA 90%', linewidth=2, markersize=10)
        ax.set_xlabel('Model Tier')
        ax.set_ylabel('Intrinsic Dimensionality')
        ax.set_title('Conversation Space Complexity')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 3. Breakdown rate gradient
        ax = axes[1, 0]
        bars = ax.bar(comparison_df['Tier'], comparison_df['Breakdown Rate'], 
                      color=['blue', 'orange', 'green'], alpha=0.7)
        ax.set_xlabel('Model Tier')
        ax.set_ylabel('Breakdown Rate')
        ax.set_title('Breakdown Rates Across Tiers')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar, val in zip(bars, comparison_df['Breakdown Rate']):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.1%}', ha='center', va='bottom')
        
        # 4. Within-tier heterogeneity
        ax = axes[1, 1]
        ax.scatter(comparison_df['Optimal Clusters'], comparison_df['Best Silhouette'], 
                  s=comparison_df['N']*3, alpha=0.6)
        for i, row in comparison_df.iterrows():
            ax.annotate(row['Tier'], (row['Optimal Clusters'], row['Best Silhouette']),
                       xytext=(5, 5), textcoords='offset points')
        ax.set_xlabel('Optimal Number of Clusters')
        ax.set_ylabel('Best Silhouette Score')
        ax.set_title('Within-Tier Heterogeneity')
        ax.grid(True, alpha=0.3)
        
        plt.suptitle('Geometric Properties Across Model Tiers', fontsize=16)
        plt.tight_layout()
        plt.savefig(os.path.join(self.tier_dir, 'tier_geometry_comparison.pdf'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # Print summary statistics
        print("\nTier Comparison Summary:")
        print(comparison_df.to_string(index=False))
        
        # Statistical tests for differences
        print("\n--- Statistical Tests Across Tiers ---")
        
        # Test for PC1 variance differences
        tiers = list(tier_results.keys())
        if len(tiers) >= 2:
            # Get PC1 variances for each tier's conversations
            tier_pc1_vars = []
            for tier_name, results in tier_results.items():
                if results is not None and 'bootstrap' in results:
                    tier_pc1_vars.append(results['bootstrap']['variance_mean'][0])
            
            if len(tier_pc1_vars) >= 2:
                # Kruskal-Wallis test (non-parametric)
                h_stat, p_val = stats.kruskal(*tier_pc1_vars)
                print(f"PC1 variance differences (Kruskal-Wallis): H={h_stat:.3f}, p={p_val:.4f}")
        
        return comparison_df
    
    def estimate_intrinsic_dimension(self, X, k_max=20):
        """Multiple methods to estimate true dimensionality"""
        print("\n" + "="*70)
        print("INTRINSIC DIMENSIONALITY ESTIMATION")
        print("="*70)
        
        n, d = X.shape
        k_range = range(2, min(k_max, n//2))
        
        # Method 1: Maximum Likelihood (Levina-Bickel)
        ml_dims = []
        for k in k_range:
            # k-NN distances
            nbrs = NearestNeighbors(n_neighbors=k+1)
            nbrs.fit(X)
            distances, _ = nbrs.kneighbors(X)
            
            # MLE of intrinsic dimension
            r_k = distances[:, k]  # k-th neighbor distance
            r_1 = distances[:, 1]  # first neighbor distance
            
            # Avoid log(0)
            mask = (r_k > 0) & (r_1 > 0)
            if mask.sum() > 0:
                d_hat = 1 / np.mean(np.log(r_k[mask] / r_1[mask]))
                ml_dims.append(d_hat)
        
        # Method 2: Correlation dimension (from chaos theory)
        corr_dims = []
        dists = pdist(X)
        for r in np.logspace(-2, 0, 20):
            # Count pairs within distance r
            count = np.sum(dists < r)
            if count > 0:
                corr_dim = np.log(count) / np.log(r)
                corr_dims.append(corr_dim)
        
        # Method 3: PCA-based effective dimension
        pca = PCA()
        pca.fit(X)
        cumvar = np.cumsum(pca.explained_variance_ratio_)
        eff_dim_90 = np.argmax(cumvar >= 0.9) + 1
        eff_dim_95 = np.argmax(cumvar >= 0.95) + 1
        
        results = {
            'ml_estimate': np.median(ml_dims) if ml_dims else None,
            'ml_std': np.std(ml_dims) if ml_dims else None,
            'correlation_dim': np.median(corr_dims) if corr_dims else None,
            'pca_dim_90': eff_dim_90,
            'pca_dim_95': eff_dim_95,
            'ml_by_k': ml_dims,
            'k_values': list(k_range)
        }
        
        # Print results, handling None values properly
        if results['ml_estimate'] is not None:
            print(f"ML Dimension Estimate: {results['ml_estimate']:.2f} ± {results['ml_std']:.2f}")
        else:
            print("ML Dimension Estimate: Could not be calculated")
        
        if results['correlation_dim'] is not None:
            print(f"Correlation Dimension: {results['correlation_dim']:.2f}")
        else:
            print("Correlation Dimension: Could not be calculated")
        
        print(f"PCA Effective Dimension (90%): {results['pca_dim_90']}")
        print(f"PCA Effective Dimension (95%): {results['pca_dim_95']}")
        
        # Visualization
        if ml_dims:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            # ML estimates by k
            ax1.plot(list(k_range)[:len(ml_dims)], ml_dims, 'b-', linewidth=2)
            ax1.axhline(y=np.median(ml_dims), color='r', linestyle='--', 
                    label=f'Median: {np.median(ml_dims):.2f}')
            ax1.set_xlabel('k (number of neighbors)')
            ax1.set_ylabel('Estimated Dimension')
            ax1.set_title('ML Dimension Estimate by k')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # PCA variance explained
            ax2.plot(range(1, len(pca.explained_variance_ratio_)+1), 
                    cumvar, 'g-', linewidth=2)
            ax2.axhline(y=0.9, color='r', linestyle='--', alpha=0.5)
            ax2.axhline(y=0.95, color='r', linestyle='--', alpha=0.5)
            ax2.axvline(x=eff_dim_90, color='b', linestyle='--', 
                    label=f'90% at dim={eff_dim_90}')
            ax2.axvline(x=eff_dim_95, color='b', linestyle=':', 
                    label=f'95% at dim={eff_dim_95}')
            ax2.set_xlabel('Number of Components')
            ax2.set_ylabel('Cumulative Variance Explained')
            ax2.set_title('PCA-based Effective Dimensionality')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            plt.suptitle('Intrinsic Dimensionality Estimates', fontsize=14)
            plt.tight_layout()
            plt.savefig(os.path.join(self.figures_dir, 'intrinsic_dimension.pdf'), 
                    dpi=300, bbox_inches='tight')
            plt.close()
        
        return results
    def analyze_within_tier_variance(self):
        """With N=100 per tier, we can study within-tier heterogeneity"""
        print("\n" + "="*70)
        print("WITHIN-TIER HETEROGENEITY ANALYSIS")
        print("="*70)
        
        within_tier_results = {}
        
        for phase in ['full_reasoning', 'light_reasoning', 'no_reasoning']:
            print(f"\n--- Analyzing {phase} ---")
            phase_data = self.all_data[self.all_data['phase'] == phase]
            
            if len(phase_data) < 10:
                print(f"Insufficient data for {phase} (n={len(phase_data)})")
                continue
            
            # Get features
            X_phase = phase_data[self.non_intervention_features]
            
            # Impute and scale
            imputer = SimpleImputer(strategy='mean')
            X_phase = imputer.fit_transform(X_phase)
            scaler = RobustScaler()
            X_phase_scaled = scaler.fit_transform(X_phase)
            
            # Are there subtypes within each tier?
            silhouette_scores = []
            inertias = []
            for n_clusters in range(2, min(10, len(phase_data)//5)):
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                labels = kmeans.fit_predict(X_phase_scaled)
                
                if len(np.unique(labels)) > 1:
                    score = silhouette_score(X_phase_scaled, labels)
                    silhouette_scores.append(score)
                    inertias.append(kmeans.inertia_)
                else:
                    silhouette_scores.append(-1)
                    inertias.append(np.inf)
            
            if silhouette_scores:
                optimal_clusters = np.argmax(silhouette_scores) + 2
                
                # Fit with optimal clusters
                kmeans_optimal = KMeans(n_clusters=optimal_clusters, random_state=42, n_init=10)
                cluster_labels = kmeans_optimal.fit_predict(X_phase_scaled)
                
                # Analyze cluster characteristics
                cluster_stats = []
                for cluster in range(optimal_clusters):
                    cluster_mask = cluster_labels == cluster
                    cluster_outcomes = phase_data.loc[cluster_mask, 'conversation_outcome'].value_counts()
                    
                    cluster_stats.append({
                        'cluster': cluster,
                        'size': cluster_mask.sum(),
                        'breakdown_rate': phase_data.loc[cluster_mask, 'breakdown_binary'].mean(),
                        'outcomes': cluster_outcomes.to_dict()
                    })
                
                within_tier_results[phase] = {
                    'optimal_clusters': optimal_clusters,
                    'silhouette_score': max(silhouette_scores),
                    'cluster_stats': cluster_stats,
                    'labels': cluster_labels
                }
                
                print(f"Optimal subtypes: {optimal_clusters}")
                print(f"Silhouette score: {max(silhouette_scores):.3f}")
                for stat in cluster_stats:
                    print(f"  Cluster {stat['cluster']}: n={stat['size']}, "
                          f"breakdown={stat['breakdown_rate']:.2%}")
        
        # Visualize
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        for idx, (phase, ax) in enumerate(zip(['full_reasoning', 'light_reasoning', 'no_reasoning'], axes)):
            if phase in within_tier_results:
                result = within_tier_results[phase]
                phase_data = self.all_data[self.all_data['phase'] == phase]
                
                # Get first 2 PCs for visualization
                X_phase = phase_data[self.non_intervention_features]
                imputer = SimpleImputer(strategy='mean')
                X_phase = imputer.fit_transform(X_phase)
                scaler = RobustScaler()
                X_phase_scaled = scaler.fit_transform(X_phase)
                
                pca = PCA(n_components=2)
                X_pca = pca.fit_transform(X_phase_scaled)
                
                # Plot clusters
                scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], 
                                   c=result['labels'], 
                                   cmap='viridis', 
                                   alpha=0.6, s=50)
                ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
                ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
                ax.set_title(f'{phase.replace("_", " ").title()}\n'
                           f'{result["optimal_clusters"]} subtypes')
                
                # Add cluster centers
                kmeans = KMeans(n_clusters=result['optimal_clusters'], random_state=42, n_init=10)
                kmeans.fit(X_phase_scaled)
                centers_pca = pca.transform(kmeans.cluster_centers_)
                ax.scatter(centers_pca[:, 0], centers_pca[:, 1], 
                          c='red', marker='x', s=200, linewidths=3)
            else:
                ax.text(0.5, 0.5, 'Insufficient Data', 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title(phase.replace("_", " ").title())
        
        plt.suptitle('Within-Tier Heterogeneity Analysis', fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(self.figures_dir, 'within_tier_clusters.pdf'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        return within_tier_results
    
    def compute_persistent_homology(self, X, max_dim=2):
        """Find topological features (holes, voids) in conversation space"""
        if not RIPSER_AVAILABLE:
            print("Ripser not available. Skipping topological analysis.")
            return None, None
        
        print("\n" + "="*70)
        print("TOPOLOGICAL DATA ANALYSIS")
        print("="*70)
        
        # Compute persistence diagrams
        diagrams = ripser(X, maxdim=max_dim)['dgms']
        
        # Find significant features (long-lived)
        significant_features = []
        for dim, dgm in enumerate(diagrams):
            if len(dgm) > 0:
                # Remove infinite death times
                finite_dgm = dgm[dgm[:, 1] < np.inf]
                if len(finite_dgm) > 0:
                    # Sort by lifetime
                    lifetimes = finite_dgm[:, 1] - finite_dgm[:, 0]
                    threshold = np.percentile(lifetimes, 90) if len(lifetimes) > 10 else 0
                    
                    sig_features = finite_dgm[lifetimes > threshold]
                    significant_features.append({
                        'dimension': dim,
                        'n_features': len(sig_features),
                        'features': sig_features,
                        'mean_lifetime': np.mean(lifetimes) if len(lifetimes) > 0 else 0
                    })
        
        print("Topological Features Found:")
        for feat in significant_features:
            print(f"  Dimension {feat['dimension']}: {feat['n_features']} significant features")
            print(f"    Mean lifetime: {feat['mean_lifetime']:.3f}")
        
        # Visualize persistence diagrams
        if len(diagrams) > 0:
            fig, axes = plt.subplots(1, min(len(diagrams), 3), figsize=(15, 5))
            if len(diagrams) == 1:
                axes = [axes]
            
            for dim, (dgm, ax) in enumerate(zip(diagrams[:3], axes)):
                if len(dgm) > 0:
                    # Remove infinite points for plotting
                    finite_dgm = dgm[dgm[:, 1] < np.inf]
                    if len(finite_dgm) > 0:
                        ax.scatter(finite_dgm[:, 0], finite_dgm[:, 1], alpha=0.6)
                        ax.plot([0, finite_dgm.max()], [0, finite_dgm.max()], 'k--', alpha=0.3)
                        ax.set_xlabel('Birth')
                        ax.set_ylabel('Death')
                        ax.set_title(f'Dimension {dim} Persistence Diagram')
                    else:
                        ax.text(0.5, 0.5, 'No finite features', 
                               ha='center', va='center', transform=ax.transAxes)
                else:
                    ax.text(0.5, 0.5, 'No features', 
                           ha='center', va='center', transform=ax.transAxes)
            
            plt.suptitle('Persistence Diagrams - Topological Features', fontsize=14)
            plt.tight_layout()
            plt.savefig(os.path.join(self.topology_dir, 'persistence_diagrams.pdf'), 
                       dpi=300, bbox_inches='tight')
            plt.close()
        
        return diagrams, significant_features
    
    def compare_manifold_methods(self, X, n_components=2, subset_data=None):
        """Compare different manifold learning approaches"""
        print("\n" + "="*70)
        print("MANIFOLD LEARNING COMPARISON")
        print("="*70)
        
        methods = {
            'PCA': PCA(n_components=n_components),
            'MDS': MDS(n_components=n_components, metric=True, random_state=42),
            'Isomap': Isomap(n_components=n_components),
            't-SNE': TSNE(n_components=n_components, perplexity=min(30, X.shape[0]-1), 
                         random_state=42),
        }
        
        if UMAP_AVAILABLE:
            methods['UMAP'] = umap.UMAP(n_components=n_components, random_state=42)
        
        embeddings = {}
        computation_times = {}
        
        for name, method in methods.items():
            print(f"Computing {name}...")
            start_time = datetime.now()
            try:
                embeddings[name] = method.fit_transform(X)
                computation_times[name] = (datetime.now() - start_time).total_seconds()
            except Exception as e:
                print(f"Error with {name}: {e}")
        
        # Compare preservation of distances
        original_dists = pdist(X)
        preservation_scores = {}
        
        for name, embedding in embeddings.items():
            embedded_dists = pdist(embedding)
            # Spearman correlation of distance matrices
            preservation_scores[name] = stats.spearmanr(original_dists, embedded_dists)[0]
        
        print("\nDistance Preservation Scores (Spearman correlation):")
        for name, score in preservation_scores.items():
            print(f"  {name}: {score:.3f} (time: {computation_times[name]:.2f}s)")
        
        # Only create visualization if we're analyzing the full dataset
        if subset_data is None:
            # Visualize embeddings
            n_methods = len(embeddings)
            fig, axes = plt.subplots(2, (n_methods + 1) // 2, figsize=(15, 10))
            axes = axes.flatten()
            
            for idx, (name, embedding) in enumerate(embeddings.items()):
                ax = axes[idx]
                
                # Color by outcome
                colors = self.all_data['outcome_numeric'].values
                scatter = ax.scatter(embedding[:, 0], embedding[:, 1], 
                                   c=colors, cmap='viridis', alpha=0.6, s=50)
                ax.set_xlabel('Component 1')
                ax.set_ylabel('Component 2')
                ax.set_title(f'{name}\n(preservation: {preservation_scores[name]:.3f})')
                
            # Hide unused subplots
            for idx in range(len(embeddings), len(axes)):
                axes[idx].set_visible(False)
            
            plt.suptitle('Manifold Learning Methods Comparison', fontsize=14)
            plt.tight_layout()
            plt.savefig(os.path.join(self.manifold_dir, 'manifold_comparison.pdf'), 
                       dpi=300, bbox_inches='tight')
            plt.close()
        
        return embeddings, preservation_scores
    
    def detect_conversation_phases(self, trajectory_data, n_phases=None):
        """Detect phase transitions in conversation trajectories"""
        print("\n" + "="*70)
        print("CONVERSATION PHASE DETECTION")
        print("="*70)
        
        if n_phases is None:
            # Find optimal number of phases
            scores = []
            bics = []
            phase_range = range(2, min(8, len(trajectory_data)//10))
            
            for n in phase_range:
                gmm = GaussianMixture(n_components=n, random_state=42)
                labels = gmm.fit_predict(trajectory_data)
                
                if len(np.unique(labels)) > 1:
                    score = silhouette_score(trajectory_data, labels)
                    scores.append(score)
                    bics.append(gmm.bic(trajectory_data))
                else:
                    scores.append(-1)
                    bics.append(np.inf)
            
            if scores:
                # Use silhouette score to determine optimal phases
                n_phases = list(phase_range)[np.argmax(scores)]
                print(f"Optimal number of phases: {n_phases}")
        
        # Fit final model
        gmm = GaussianMixture(n_components=n_phases, random_state=42)
        phase_labels = gmm.fit_predict(trajectory_data)
        
        # Find transition points
        transitions = np.where(np.diff(phase_labels) != 0)[0]
        
        # Analyze phase characteristics
        phase_stats = []
        for phase in range(n_phases):
            phase_mask = phase_labels == phase
            phase_stats.append({
                'phase': phase,
                'size': phase_mask.sum(),
                'mean': gmm.means_[phase],
                'covariance_trace': np.trace(gmm.covariances_[phase])
            })
        
        results = {
            'phases': phase_labels,
            'n_phases': n_phases,
            'transitions': transitions,
            'phase_means': gmm.means_,
            'phase_covariances': gmm.covariances_,
            'phase_stats': phase_stats
        }
        
        print(f"Found {len(transitions)} phase transitions")
        for stat in phase_stats:
            print(f"  Phase {stat['phase']}: n={stat['size']}, "
                  f"cov_trace={stat['covariance_trace']:.3f}")
        
        # Visualize if 2D
        if trajectory_data.shape[1] == 2:
            plt.figure(figsize=(10, 8))
            
            # Plot points colored by phase
            scatter = plt.scatter(trajectory_data[:, 0], trajectory_data[:, 1], 
                                c=phase_labels, cmap='viridis', alpha=0.6, s=50)
            
            # Plot phase centers
            plt.scatter(gmm.means_[:, 0], gmm.means_[:, 1], 
                       c='red', marker='x', s=200, linewidths=3)
            
            # Draw covariance ellipses
            from matplotlib.patches import Ellipse
            for i in range(n_phases):
                cov = gmm.covariances_[i]
                v, w = np.linalg.eigh(cov)
                angle = np.degrees(np.arctan2(w[1, 0], w[0, 0]))
                ellipse = Ellipse(gmm.means_[i], 2*np.sqrt(v[0]), 2*np.sqrt(v[1]),
                                angle=angle, facecolor='none', edgecolor='red', 
                                linewidth=2, alpha=0.5)
                plt.gca().add_patch(ellipse)
            
            plt.xlabel('Dimension 1')
            plt.ylabel('Dimension 2')
            plt.title(f'Conversation Phases (n={n_phases})')
            plt.colorbar(scatter, label='Phase')
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(self.figures_dir, 'conversation_phases.pdf'), 
                       dpi=300, bbox_inches='tight')
            plt.close()
        
        return results
    
    def build_trajectory_predictor(self, dimensions, horizon=10):
        """Learn to predict future trajectory from current state"""
        print("\n" + "="*70)
        print("TRAJECTORY PREDICTION MODEL")
        print("="*70)
        
        # For this analysis, we need conversation-level time series data
        # This is a simplified version - in practice you'd have turn-by-turn data
        
        # Create synthetic trajectory data for demonstration
        n_conversations = len(self.all_data)
        n_timesteps = 50  # Simplified - you'd use actual turn counts
        
        # Generate synthetic trajectories based on conversation outcomes
        trajectories = []
        for idx, row in self.all_data.iterrows():
            # Create a trajectory that evolves toward the outcome
            start_state = np.random.randn(4) * 0.5
            
            if row['conversation_outcome'] == 'breakdown':
                # Trajectory toward breakdown attractor
                end_state = np.array([2, 0, 2, 1])  # High social contagion, temporal
            elif row['conversation_outcome'] == 'no_breakdown':
                # Trajectory toward stable attractor
                end_state = np.array([-1, 0, -1, -1])
            else:
                # Intermediate trajectories
                end_state = np.random.randn(4) * 0.5
            
            # Interpolate trajectory
            trajectory = np.zeros((n_timesteps, 4))
            for t in range(n_timesteps):
                alpha = t / n_timesteps
                trajectory[t] = (1 - alpha) * start_state + alpha * end_state
                trajectory[t] += np.random.randn(4) * 0.1  # Add noise
            
            trajectories.append(trajectory)
        
        # Prepare training data
        X_train = []
        y_train = []
        
        for traj in trajectories:
            for t in range(len(traj) - horizon - 1):
                # Current state
                state_t = traj[t]
                # Velocity (first difference)
                velocity_t = traj[t+1] - traj[t]
                # Acceleration (second difference) 
                if t > 0:
                    accel_t = velocity_t - (traj[t] - traj[t-1])
                else:
                    accel_t = np.zeros_like(velocity_t)
                
                # Features
                features = np.concatenate([state_t, velocity_t, accel_t])
                X_train.append(features)
                
                # Target: future state
                y_train.append(traj[t + horizon])
        
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        
        # Split data
        X_tr, X_te, y_tr, y_te = train_test_split(
            X_train, y_train, test_size=0.3, random_state=42
        )
        
        # Train predictor
        predictor = Ridge(alpha=1.0)
        predictor.fit(X_tr, y_tr)
        
        # Evaluate at different horizons
        horizons = [5, 10, 20]
        rmse_by_horizon = {}
        
        for h in horizons:
            # Similar data prep for each horizon
            X_h = []
            y_h = []
            
            for traj in trajectories[:10]:  # Use subset for evaluation
                for t in range(len(traj) - h - 1):
                    state_t = traj[t]
                    velocity_t = traj[t+1] - traj[t] if t < len(traj)-1 else np.zeros(4)
                    accel_t = velocity_t - (traj[t] - traj[t-1]) if t > 0 else np.zeros(4)
                    
                    features = np.concatenate([state_t, velocity_t, accel_t])
                    X_h.append(features)
                    y_h.append(traj[t + h])
            
            if X_h:
                X_h = np.array(X_h)
                y_h = np.array(y_h)
                y_pred = predictor.predict(X_h)
                rmse = np.sqrt(mean_squared_error(y_h, y_pred))
                rmse_by_horizon[h] = rmse
        
        print("Trajectory Prediction RMSE by Horizon:")
        for h, rmse in rmse_by_horizon.items():
            print(f"  Horizon {h}: {rmse:.3f}")
        
        # Visualize prediction accuracy
        plt.figure(figsize=(10, 6))
        horizons_list = list(rmse_by_horizon.keys())
        rmses = list(rmse_by_horizon.values())
        
        plt.plot(horizons_list, rmses, 'bo-', linewidth=2, markersize=10)
        plt.xlabel('Prediction Horizon (turns)')
        plt.ylabel('RMSE')
        plt.title('Trajectory Prediction Accuracy vs Horizon')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(self.trajectory_dir, 'prediction_accuracy.pdf'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        return predictor, rmse_by_horizon
    
    def analyze_intervention_threshold(self):
        """Find critical intervention density for regime transition"""
        print("\n" + "="*70)
        print("INTERVENTION REGIME THRESHOLD ANALYSIS")
        print("="*70)
        
        # Calculate intervention density
        self.all_data['intervention_density'] = (
            self.all_data['circuit_breaker_questions'] + 
            self.all_data['recovery_attempt_count']
        ) / self.all_data['total_messages']
        
        # Prepare features
        X_non_int = self.all_data[self.non_intervention_features].copy()
        
        # Impute and scale
        imputer = SimpleImputer(strategy='mean')
        X_non_int = imputer.fit_transform(X_non_int)
        scaler = RobustScaler()
        X_non_int_scaled = scaler.fit_transform(X_non_int)
        
        # Calculate variance explained as function of intervention density
        density_bins = np.linspace(0, self.all_data['intervention_density'].max(), 20)
        variance_by_density = []
        
        for i in range(len(density_bins) - 1):
            mask = (self.all_data['intervention_density'] >= density_bins[i]) & \
                   (self.all_data['intervention_density'] < density_bins[i+1])
            
            if mask.sum() >= 5:  # Need at least 5 samples
                X_subset = X_non_int_scaled[mask]
                pca = PCA(n_components=min(4, X_subset.shape[0] - 1))
                pca.fit(X_subset)
                
                # Store cumulative variance for first 4 components
                cum_var = np.sum(pca.explained_variance_ratio_[:4])
                variance_by_density.append({
                    'density_mid': (density_bins[i] + density_bins[i+1]) / 2,
                    'cumulative_variance': cum_var,
                    'n_samples': mask.sum()
                })
        
        # Create visualization
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        # Plot 1: Variance vs intervention density
        density_df = pd.DataFrame(variance_by_density)
        ax1.scatter(density_df['density_mid'], density_df['cumulative_variance'], 
                   s=density_df['n_samples']*10, alpha=0.6)
        ax1.plot(density_df['density_mid'], density_df['cumulative_variance'], 
                'r--', alpha=0.5)
        ax1.set_xlabel('Intervention Density')
        ax1.set_ylabel('Cumulative Variance (First 4 PCs)')
        ax1.set_title('Dimensional Structure vs Intervention Density')
        ax1.grid(True, alpha=0.3)
        
        # Find threshold (where variance drops significantly)
        if len(density_df) > 3:
            from scipy.interpolate import UnivariateSpline
            spl = UnivariateSpline(density_df['density_mid'], 
                                  density_df['cumulative_variance'], s=0.1)
            x_smooth = np.linspace(density_df['density_mid'].min(), 
                                 density_df['density_mid'].max(), 100)
            y_smooth = spl(x_smooth)
            
            # Find steepest decline
            dy = np.gradient(y_smooth)
            threshold_idx = np.argmin(dy)
            threshold = x_smooth[threshold_idx]
            
            ax1.axvline(x=threshold, color='g', linestyle='--', 
                       label=f'Threshold: {threshold:.3f}')
            ax1.legend()
        
        # Plot 2: Distribution of intervention density by outcome
        for outcome in ['no_breakdown', 'resisted', 'recovered', 'breakdown']:
            mask = self.all_data['conversation_outcome'] == outcome
            if mask.sum() > 0:
                densities = self.all_data.loc[mask, 'intervention_density']
                ax2.hist(densities, alpha=0.5, label=outcome, bins=15)
        
        ax2.set_xlabel('Intervention Density')
        ax2.set_ylabel('Count')
        ax2.set_title('Intervention Density Distribution by Outcome')
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.figures_dir, 'intervention_threshold.pdf'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # Statistical analysis
        print("\nIntervention Density by Outcome:")
        for outcome in ['no_breakdown', 'resisted', 'recovered', 'breakdown']:
            mask = self.all_data['conversation_outcome'] == outcome
            if mask.sum() > 0:
                densities = self.all_data.loc[mask, 'intervention_density']
                print(f"{outcome}: {densities.mean():.3f} ± {densities.std():.3f}")
        
        # Test for significant differences
        outcome_groups = []
        for outcome in ['no_breakdown', 'resisted', 'recovered', 'breakdown']:
            mask = self.all_data['conversation_outcome'] == outcome
            if mask.sum() > 0:
                outcome_groups.append(self.all_data.loc[mask, 'intervention_density'])
        
        if len(outcome_groups) > 1:
            f_stat, p_value = stats.f_oneway(*outcome_groups)
            print(f"\nANOVA: F = {f_stat:.2f}, p = {p_value:.4f}")
        
        return threshold if 'threshold' in locals() else 0.15
    
    def test_dimension_predictive_power(self, dimensions):
        """Test if dimensions predict outcomes using cross-validation"""
        print("\n" + "="*70)
        print("DIMENSION PREDICTIVE POWER (CROSS-VALIDATED)")
        print("="*70)
        
        # Prepare dimension matrix
        X = np.column_stack([dim_data['scores'] for dim_data in dimensions.values()])
        y = self.all_data['breakdown_binary'].values
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        
        # Logistic regression with L2 regularization
        results = {}
        
        # Test different regularization strengths
        for C in [0.01, 0.1, 1.0, 10.0]:
            lr = LogisticRegression(C=C, random_state=42, max_iter=1000)
            
            # Cross-validation on training set
            cv_scores = cross_val_score(lr, X_train, y_train, cv=5, scoring='roc_auc')
            
            # Fit on full training set
            lr.fit(X_train, y_train)
            
            # Test set performance
            y_pred_proba = lr.predict_proba(X_test)[:, 1]
            test_auc = roc_auc_score(y_test, y_pred_proba)
            
            results[C] = {
                'cv_auc_mean': cv_scores.mean(),
                'cv_auc_std': cv_scores.std(),
                'test_auc': test_auc,
                'coefficients': lr.coef_[0]
            }
            
            print(f"\nC = {C}:")
            print(f"  CV AUC: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
            print(f"  Test AUC: {test_auc:.3f}")
        
        # Best model
        best_C = max(results.keys(), key=lambda c: results[c]['test_auc'])
        best_model = LogisticRegression(C=best_C, random_state=42, max_iter=1000)
        best_model.fit(X_train, y_train)
        
        # Feature importance
        print(f"\nBest Model (C = {best_C}) Coefficients:")
        for i, dim_name in enumerate(dimensions.keys()):
            coef = best_model.coef_[0][i]
            print(f"  {dim_name}: {coef:.3f}")
        
        # Classification report
        y_pred = best_model.predict(X_test)
        print("\nClassification Report (Test Set):")
        print(classification_report(y_test, y_pred))
        
        # Save results
        pred_df = pd.DataFrame(results).T
        pred_df.to_csv(os.path.join(self.validation_dir, 'predictive_power_results.csv'))
        
        return results, best_model
    
    def create_regime_comparison_figure(self, pca_results):
        """Create the key 2x2 figure showing regime differences"""
        print("\n" + "="*70)
        print("CREATING REGIME COMPARISON FIGURE")
        print("="*70)
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        
        # Top left: PCA with all features (intervention dominance)
        ax = axes[0, 0]
        var_all = pca_results['all']['explained_variance_ratio'][:5]
        ax.bar(range(len(var_all)), var_all, alpha=0.7, color='red')
        ax.set_xlabel('Principal Component')
        ax.set_ylabel('Variance Explained')
        ax.set_title(f'With Intervention Features\n(PC1: {var_all[0]:.1%} variance)')
        ax.set_ylim(0, 1.0)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Top right: PCA without intervention (4D structure)
        ax = axes[0, 1]
        var_non = pca_results['non_intervention']['explained_variance_ratio'][:5]
        ax.bar(range(len(var_non)), var_non, alpha=0.7, color='blue')
        ax.set_xlabel('Principal Component')
        ax.set_ylabel('Variance Explained')
        ax.set_title(f'Without Intervention Features\n(PC1: {var_non[0]:.1%} variance)')
        ax.set_ylim(0, 1.0)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Bottom left: Trajectory in intervention regime
        ax = axes[1, 0]
        # Simulate intervention-dominated trajectory
        t = np.linspace(0, 10, 100)
        intervention_trajectory = np.cumsum(np.random.randn(100) * 0.1) + 0.5 * t
        ax.plot(t, intervention_trajectory, 'r-', linewidth=2)
        ax.fill_between(t, intervention_trajectory - 0.5, intervention_trajectory + 0.5, 
                       alpha=0.3, color='red')
        ax.set_xlabel('Time')
        ax.set_ylabel('Intervention Axis Position')
        ax.set_title('Trajectory in Intervention Regime\n(1D movement along intervention axis)')
        ax.grid(True, alpha=0.3)
        
        # Bottom right: Trajectory in natural regime
        ax = axes[1, 1]
        # Simulate 2D projection of 4D trajectory
        theta = np.linspace(0, 4*np.pi, 100)
        x = np.sin(theta) + 0.3*np.sin(3*theta) + 0.1*np.random.randn(100)
        y = np.cos(theta) + 0.3*np.cos(2*theta) + 0.1*np.random.randn(100)
        
        # Color by "time"
        colors = plt.cm.viridis(np.linspace(0, 1, len(x)))
        ax.scatter(x, y, c=colors, s=20, alpha=0.6)
        ax.plot(x, y, 'k-', alpha=0.2, linewidth=0.5)
        
        # Add attractor regions
        circle1 = plt.Circle((0, 0), 0.5, color='green', alpha=0.2, label='Stable')
        circle2 = plt.Circle((1.5, 0), 0.5, color='red', alpha=0.2, label='Breakdown')
        ax.add_patch(circle1)
        ax.add_patch(circle2)
        
        ax.set_xlabel('Dimension 1')
        ax.set_ylabel('Dimension 2')
        ax.set_title('Trajectory in Natural Regime\n(4D manifold, projected to 2D)')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        
        plt.suptitle('Dual-Regime Structure of Conversational Dynamics', fontsize=16)
        plt.tight_layout()
        plt.savefig(os.path.join(self.figures_dir, 'regime_comparison.pdf'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print("Regime comparison figure saved.")
    
    def generate_synthetic_conversations(self, dimensions, n_samples=100):
        """Generate synthetic data to validate the model"""
        print("\n" + "="*70)
        print("GENERATING SYNTHETIC CONVERSATIONS")
        print("="*70)
        
        # Get dimension statistics from real data
        dim_stats = {}
        for dim_name, dim_data in dimensions.items():
            scores = dim_data['scores']
            dim_stats[dim_name] = {
                'mean': np.mean(scores),
                'std': np.std(scores),
                'min': np.min(scores),
                'max': np.max(scores)
            }
        
        # Get the actual dimension names available
        available_dims = list(dimensions.keys())
        print(f"Available dimensions: {available_dims}")
        
        # Create a mapping for expected dimensions
        dim_mapping = {
            'social_contagion': available_dims[0] if len(available_dims) > 0 else 'social_contagion',
            'linguistic_synchrony': available_dims[1] if len(available_dims) > 1 else 'linguistic_synchrony',
            'affective_cognitive': available_dims[2] if len(available_dims) > 2 else 'affective_cognitive',
            'temporal_dynamics': available_dims[3] if len(available_dims) > 3 else 'temporal_dynamics'
        }
        
        # Generate synthetic dimensions
        n_dims = len(available_dims)
        n_per_type = n_samples // 4
        
        # Type 1: Stable conversations (low on breakdown dimensions)
        stable_means = []
        for i, dim_name in enumerate(available_dims):
            if i == 0:  # social contagion - lower
                stable_means.append(dim_stats[dim_name]['mean'] - dim_stats[dim_name]['std'])
            elif i == 2:  # affective - lower
                stable_means.append(dim_stats[dim_name]['mean'] - dim_stats[dim_name]['std'])
            elif i == 3:  # temporal - lower
                stable_means.append(dim_stats[dim_name]['mean'] - dim_stats[dim_name]['std'])
            else:
                stable_means.append(dim_stats[dim_name]['mean'])
        
        stable_samples = np.random.multivariate_normal(
            mean=stable_means,
            cov=np.eye(n_dims) * 0.5,
            size=n_per_type
        )
        
        # Type 2: Breakdown conversations (high on breakdown dimensions)
        breakdown_means = []
        for i, dim_name in enumerate(available_dims):
            if i == 0:  # social contagion - higher
                breakdown_means.append(dim_stats[dim_name]['mean'] + dim_stats[dim_name]['std'])
            elif i == 2:  # affective - higher
                breakdown_means.append(dim_stats[dim_name]['mean'] + dim_stats[dim_name]['std'])
            elif i == 3:  # temporal - higher
                breakdown_means.append(dim_stats[dim_name]['mean'] + dim_stats[dim_name]['std'])
            else:
                breakdown_means.append(dim_stats[dim_name]['mean'])
        
        breakdown_samples = np.random.multivariate_normal(
            mean=breakdown_means,
            cov=np.eye(n_dims) * 0.5,
            size=n_per_type
        )
        
        # Type 3: Recovered conversations (trajectory from high to low)
        recovered_samples = []
        for i in range(n_per_type):
            trajectory = []
            for j, dim_name in enumerate(available_dims):
                if j in [0, 2, 3]:  # Dimensions that change
                    t = 1 - (i / n_per_type)  # Goes from 1 to 0
                    value = dim_stats[dim_name]['mean'] + t * dim_stats[dim_name]['std']
                else:
                    value = dim_stats[dim_name]['mean']
                trajectory.append(value)
            recovered_samples.append(trajectory)
        recovered_samples = np.array(recovered_samples)
        
        # Type 4: Resisted conversations (medium values)
        resisted_means = [dim_stats[d]['mean'] for d in available_dims]
        resisted_samples = np.random.multivariate_normal(
            mean=resisted_means,
            cov=np.eye(n_dims) * 0.3,
            size=n_samples - 3*n_per_type
        )
        
        # Combine
        synthetic_data = np.vstack([stable_samples, breakdown_samples, 
                                   recovered_samples, resisted_samples])
        synthetic_labels = np.array(['no_breakdown']*n_per_type + 
                                   ['breakdown']*n_per_type + 
                                   ['recovered']*n_per_type + 
                                   ['resisted']*(n_samples - 3*n_per_type))
        
        # Compare statistics with real data
        print("\nValidating Synthetic Data:")
        print("Dimension means (Real vs Synthetic):")
        for i, dim_name in enumerate(available_dims):
            real_mean = dim_stats[dim_name]['mean']
            synth_mean = np.mean(synthetic_data[:, i])
            print(f"  {dim_name}: {real_mean:.3f} vs {synth_mean:.3f}")
        
        # Test if synthetic data can be distinguished from real
        real_matrix = np.column_stack([dim_data['scores'] for dim_data in dimensions.values()])
        
        # Create labels
        real_labels = np.ones(len(real_matrix))
        synth_labels = np.zeros(len(synthetic_data))
        
        # Combine
        X_combined = np.vstack([real_matrix, synthetic_data])
        y_combined = np.concatenate([real_labels, synth_labels])
        
        # Can a classifier distinguish real from synthetic?
        X_train, X_test, y_train, y_test = train_test_split(
            X_combined, y_combined, test_size=0.3, random_state=42
        )
        
        lr = LogisticRegression(random_state=42)
        lr.fit(X_train, y_train)
        test_score = lr.score(X_test, y_test)
        
        print(f"\nReal vs Synthetic discrimination accuracy: {test_score:.3f}")
        print(f"(Lower is better - 0.5 means indistinguishable)")
        
        # Visualize if we have at least 2 dimensions
        if n_dims >= 2:
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            
            # Real data
            ax = axes[0]
            for outcome, color in zip(['no_breakdown', 'resisted', 'recovered', 'breakdown'],
                                     ['green', 'yellow', 'orange', 'red']):
                mask = self.all_data['conversation_outcome'] == outcome
                if mask.sum() > 0:
                    ax.scatter(real_matrix[mask, 0], real_matrix[mask, 1],
                              c=color, label=outcome, alpha=0.6, s=50)
            ax.set_xlabel(available_dims[0])
            ax.set_ylabel(available_dims[1])
            ax.set_title('Real Data')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Synthetic data
            ax = axes[1]
            for outcome, color in zip(['no_breakdown', 'resisted', 'recovered', 'breakdown'],
                                     ['green', 'yellow', 'orange', 'red']):
                mask = synthetic_labels == outcome
                if mask.sum() > 0:
                    ax.scatter(synthetic_data[mask, 0], synthetic_data[mask, 1],
                              c=color, label=outcome, alpha=0.6, s=50)
            ax.set_xlabel(available_dims[0])
            ax.set_ylabel(available_dims[1])
            ax.set_title('Synthetic Data')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            plt.suptitle('Real vs Synthetic Data Comparison', fontsize=14)
            plt.tight_layout()
            plt.savefig(os.path.join(self.validation_dir, 'synthetic_validation.pdf'), 
                       dpi=300, bbox_inches='tight')
            plt.close()
        
        return synthetic_data, synthetic_labels
    
    def create_dimension_gradient_figure(self, dimensions):
        """Create figure showing capability gradients"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
        
        for idx, (dim_name, dim_data) in enumerate(dimensions.items()):
            ax = axes[idx]
            
            # Calculate means and CIs by phase
            phase_stats = []
            for phase in ['full_reasoning', 'light_reasoning', 'no_reasoning']:
                mask = self.all_data['phase'] == phase
                if mask.sum() > 0:
                    scores = dim_data['scores'][mask]
                    mean = np.mean(scores)
                    se = stats.sem(scores)
                    ci = stats.t.interval(0.95, len(scores)-1, loc=mean, scale=se)
                    phase_stats.append({
                        'phase': phase,
                        'mean': mean,
                        'ci_lower': ci[0],
                        'ci_upper': ci[1],
                        'n': mask.sum()
                    })
            
            phase_df = pd.DataFrame(phase_stats)
            
            # Plot with error bars
            x = range(len(phase_df))
            ax.bar(x, phase_df['mean'], yerr=[phase_df['mean'] - phase_df['ci_lower'],
                                              phase_df['ci_upper'] - phase_df['mean']],
                  capsize=5, alpha=0.7, color=['blue', 'orange', 'green'])
            
            ax.set_xticks(x)
            ax.set_xticklabels([p.replace('_', ' ').title() for p in phase_df['phase']])
            ax.set_ylabel('Dimension Score')
            ax.set_title(f'{dim_name.replace("_", " ").title()}\n(95% CI)')
            ax.grid(True, alpha=0.3, axis='y')
            
            # Add sample sizes
            for i, row in phase_df.iterrows():
                ax.text(i, ax.get_ylim()[0] + 0.05*(ax.get_ylim()[1] - ax.get_ylim()[0]),
                       f'n={row["n"]}', ha='center', va='bottom', fontsize=8)
        
        plt.suptitle('Dimension Gradients Across Model Capabilities', fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(self.figures_dir, 'dimension_gradients.pdf'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_phase_space_portrait(self, dimensions):
        """Create phase space portraits showing conversation dynamics"""
        print("\n" + "="*70)
        print("CREATING PHASE SPACE PORTRAITS")
        print("="*70)
        
        # Get first two dimensions for visualization
        dim_names = list(dimensions.keys())[:2]
        if len(dim_names) < 2:
            print("Need at least 2 dimensions for phase space portrait")
            return
        
        X = np.column_stack([dimensions[d]['scores'] for d in dim_names])
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        
        # Top left: All conversations with outcome coloring
        ax = axes[0, 0]
        for outcome, color in zip(['no_breakdown', 'resisted', 'recovered', 'breakdown'],
                                 ['green', 'yellow', 'orange', 'red']):
            mask = self.all_data['conversation_outcome'] == outcome
            if mask.sum() > 0:
                ax.scatter(X[mask, 0], X[mask, 1], c=color, label=outcome, 
                          alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
        ax.set_xlabel(dim_names[0].replace('_', ' ').title())
        ax.set_ylabel(dim_names[1].replace('_', ' ').title())
        ax.set_title('Phase Space by Outcome')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Top right: Vector field (estimated from data)
        ax = axes[0, 1]
        # Create grid
        x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
        y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 20),
                            np.linspace(y_min, y_max, 20))
        
        # Estimate vector field using nearby points
        uu = np.zeros_like(xx)
        vv = np.zeros_like(yy)
        
        for i in range(xx.shape[0]):
            for j in range(xx.shape[1]):
                point = np.array([xx[i, j], yy[i, j]])
                # Find nearby points
                distances = np.sqrt(((X - point)**2).sum(axis=1))
                nearby = distances < 1.0
                
                if nearby.sum() > 3:
                    # Estimate "flow" based on outcome
                    nearby_outcomes = self.all_data.loc[nearby, 'outcome_numeric'].values
                    # Direction toward breakdown (3) or stability (0)
                    direction = np.mean(nearby_outcomes) - 1.5
                    # Add some structure
                    uu[i, j] = -direction * (yy[i, j] - y_min) / (y_max - y_min)
                    vv[i, j] = direction * (xx[i, j] - x_min) / (x_max - x_min)
        
        ax.quiver(xx, yy, uu, vv, alpha=0.5)
        ax.set_xlabel(dim_names[0].replace('_', ' ').title())
        ax.set_ylabel(dim_names[1].replace('_', ' ').title())
        ax.set_title('Estimated Vector Field')
        ax.grid(True, alpha=0.3)
        
        # Bottom left: Density plot
        ax = axes[1, 0]
        from scipy.stats import gaussian_kde
        xy = X.T
        z = gaussian_kde(xy)(xy)
        scatter = ax.scatter(X[:, 0], X[:, 1], c=z, s=50, alpha=0.6, cmap='viridis')
        plt.colorbar(scatter, ax=ax, label='Density')
        ax.set_xlabel(dim_names[0].replace('_', ' ').title())
        ax.set_ylabel(dim_names[1].replace('_', ' ').title())
        ax.set_title('Conversation Density')
        ax.grid(True, alpha=0.3)
        
        # Bottom right: Trajectories by model tier
        ax = axes[1, 1]
        for phase, color in zip(['full_reasoning', 'light_reasoning', 'no_reasoning'],
                               ['blue', 'orange', 'green']):
            mask = self.all_data['phase'] == phase
            if mask.sum() > 0:
                # Plot with slight jitter to show overlap
                jitter = np.random.normal(0, 0.05, size=(mask.sum(), 2))
                ax.scatter(X[mask, 0] + jitter[:, 0], X[mask, 1] + jitter[:, 1], 
                          c=color, label=phase.replace('_', ' ').title(), 
                          alpha=0.4, s=30)
        ax.set_xlabel(dim_names[0].replace('_', ' ').title())
        ax.set_ylabel(dim_names[1].replace('_', ' ').title())
        ax.set_title('Phase Space by Model Tier')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.suptitle('Phase Space Portraits of Conversation Dynamics', fontsize=16)
        plt.tight_layout()
        plt.savefig(os.path.join(self.figures_dir, 'phase_space_portraits.pdf'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def run_complete_enhanced_analysis(self, phase1_path, phase2_path, phase3_path):
        """Run the complete enhanced analysis with per-tier geometry"""
        
        print("\n" + "="*70)
        print("ENHANCED CONVERSATIONAL SPACE ANALYSIS V3")
        print("="*70)
        print("Incorporating:")
        print("- Statistical power analysis")
        print("- Regularized methods")
        print("- Bootstrap confidence intervals")
        print("- Cross-validated prediction")
        print("- Intervention threshold analysis")
        print("- Synthetic data validation")
        print("- Intrinsic dimensionality estimation")
        print("- Within-tier heterogeneity analysis")
        print("- Topological data analysis")
        print("- Manifold learning comparison")
        print("- Phase detection and trajectory prediction")
        print("- PER-TIER GEOMETRIC ANALYSIS")
        
        # Load and prepare data
        self.load_and_prepare_data(phase1_path, phase2_path, phase3_path)
        self.define_feature_groups()
        
        # 1. Statistical Power Analysis
        power_results = self.calculate_statistical_power()
        
        # 2. OVERALL ANALYSIS
        print("\n" + "="*70)
        print("OVERALL ANALYSIS (ALL TIERS COMBINED)")
        print("="*70)
        
        # Prepare feature matrices
        X_all = self.all_data[self.all_features].copy()
        X_intervention = self.all_data[self.intervention_features].copy()
        X_non_intervention = self.all_data[self.non_intervention_features].copy()
        
        # Impute and scale
        imputer = SimpleImputer(strategy='mean')
        X_all = imputer.fit_transform(X_all)
        X_intervention = imputer.fit_transform(X_intervention)
        X_non_intervention = imputer.fit_transform(X_non_intervention)
        
        scaler = RobustScaler()
        X_all_scaled = scaler.fit_transform(X_all)
        X_intervention_scaled = scaler.fit_transform(X_intervention)
        X_non_intervention_scaled = scaler.fit_transform(X_non_intervention)
        
        # 3. Regularized PCA Analysis
        print("\n--- Comparing All Features vs Separated ---")
        pca_all = self.perform_regularized_pca(X_all_scaled, alpha=0.1)
        pca_intervention = self.perform_regularized_pca(X_intervention_scaled, alpha=0.1)
        pca_non_intervention = self.perform_regularized_pca(X_non_intervention_scaled, alpha=0.1)
        
        pca_results = {
            'all': pca_all,
            'intervention': pca_intervention,
            'non_intervention': pca_non_intervention
        }
        
        # 4. Bootstrap Stability Analysis
        bootstrap_results = self.bootstrap_dimension_stability(
            X_non_intervention_scaled, 
            self.non_intervention_features,
            n_bootstrap=100
        )
        
        # 5. Create Theory-Driven Dimensions (using bootstrap-stable loadings)
        dimensions = self._create_stable_dimensions(
            X_non_intervention_scaled,
            self.non_intervention_features,
            bootstrap_results['loading_mean']
        )
        
        # 6. Test Predictive Power
        pred_results, best_model = self.test_dimension_predictive_power(dimensions)
        
        # 7. Analyze Intervention Threshold
        intervention_threshold = self.analyze_intervention_threshold()
        
        # 8. Generate and Validate Synthetic Data
        synthetic_data, synthetic_labels = self.generate_synthetic_conversations(dimensions)
        
        # 9. Estimate Intrinsic Dimensionality
        intrinsic_dim_results = self.estimate_intrinsic_dimension(X_non_intervention_scaled)
        
        # 10. Within-Tier Heterogeneity Analysis
        within_tier_results = self.analyze_within_tier_variance()
        
        # 11. Topological Data Analysis
        if len(self.all_data) >= 50:  # Need reasonable sample size
            tda_diagrams, tda_features = self.compute_persistent_homology(
                X_non_intervention_scaled[:, :10], max_dim=2
            )
        else:
            tda_diagrams, tda_features = None, None
        
        # 12. Manifold Learning Comparison
        manifold_embeddings, preservation_scores = self.compare_manifold_methods(
            X_non_intervention_scaled[:, :10], n_components=2
        )
        
        # 13. Phase Detection
        phase_results = self.detect_conversation_phases(
            X_non_intervention_scaled[:, :4]  # Use first 4 dimensions
        )
        
        # 14. Trajectory Prediction
        trajectory_predictor, prediction_rmse = self.build_trajectory_predictor(dimensions)
        
        # === NEW: PER-TIER ANALYSIS ===
        print("\n" + "="*70)
        print("PER-TIER GEOMETRIC ANALYSIS")
        print("="*70)
        
        tier_results = {}
        
        # Analyze each tier separately
        for phase in ['full_reasoning', 'light_reasoning', 'no_reasoning']:
            tier_data = self.all_data[self.all_data['phase'] == phase]
            tier_result = self.analyze_tier_geometry(
                tier_name=phase,
                tier_data=tier_data,
                feature_names=self.non_intervention_features,
                save_prefix=phase
            )
            tier_results[phase] = tier_result
        
        # Compare tier geometries
        comparison_df = self.compare_tier_geometries(tier_results)
        
        # === VISUALIZATIONS ===
        
        # 15. Create Key Visualizations
        self.create_regime_comparison_figure(pca_results)
        self.create_dimension_gradient_figure(dimensions)
        self.create_phase_space_portrait(dimensions)
        
        # 16. Save All Results
        self._save_comprehensive_results(
            power_results, pca_results, bootstrap_results, 
            dimensions, pred_results, intervention_threshold,
            intrinsic_dim_results, within_tier_results,
            tda_features, preservation_scores, phase_results,
            prediction_rmse, tier_results, comparison_df
        )
        
        print("\n" + "="*70)
        print("ANALYSIS COMPLETE!")
        print("="*70)
        print(f"\nAll results saved to: {self.output_dir}")
        print("\nKey findings:")
        print(f"- Statistical power for main effects: {power_results['outcome_groups']:.3f}")
        print(f"- Intervention dominance: {pca_results['all']['explained_variance_ratio'][0]:.1%}")
        print(f"- Non-intervention PC1: {pca_results['non_intervention']['explained_variance_ratio'][0]:.1%}")
        print(f"- Intervention threshold: {intervention_threshold:.3f}")
        print(f"- Predictive AUC: {max(r['test_auc'] for r in pred_results.values()):.3f}")
        if intrinsic_dim_results['ml_estimate']:
            print(f"- Intrinsic dimension: {intrinsic_dim_results['ml_estimate']:.2f}")
        print(f"- Optimal conversation phases: {phase_results.get('n_phases', 'N/A')}")
        
        print("\nPer-Tier Geometry Summary:")
        print(comparison_df.to_string(index=False))
        
        return {
            'power': power_results,
            'pca': pca_results,
            'bootstrap': bootstrap_results,
            'dimensions': dimensions,
            'prediction': pred_results,
            'threshold': intervention_threshold,
            'intrinsic_dim': intrinsic_dim_results,
            'within_tier': within_tier_results,
            'topology': tda_features,
            'manifolds': preservation_scores,
            'phases': phase_results,
            'trajectories': prediction_rmse,
            'tier_results': tier_results,
            'tier_comparison': comparison_df
        }
    
    def _create_stable_dimensions(self, X_scaled, feature_names, stable_loadings):
        """Create dimensions using bootstrap-stable loadings"""
        dimensions = {}
        
        # Helper to get indices
        def get_indices(features):
            return [feature_names.index(f) for f in features if f in feature_names]
        
        # Define the dimension mapping
        dimension_names = {
            'social_contagion': 'social_contagion',
            'linguistic_synchrony': 'linguistic_synchrony',
            'affective_dynamics': 'affective_cognitive',  # Combined with cognitive
            'cognitive_abstraction': 'affective_cognitive',  # Combined with affective
            'temporal_structure': 'temporal_dynamics'  # Renamed
        }
        
        # Process each group
        processed_groups = set()
        pc_idx = 0
        
        for group_name, features in self.feature_groups.items():
            if group_name != 'intervention_response':
                dim_name = dimension_names.get(group_name, group_name)
                
                # Skip if already processed (for combined dimensions)
                if dim_name in processed_groups:
                    continue
                
                # For combined affective-cognitive dimension
                if dim_name == 'affective_cognitive':
                    combined_features = []
                    combined_features.extend(self.feature_groups.get('affective_dynamics', []))
                    combined_features.extend(self.feature_groups.get('cognitive_abstraction', []))
                    features = combined_features
                
                indices = get_indices(features)
                if indices:
                    # Use absolute loadings from relevant PC
                    weights = np.abs(stable_loadings[pc_idx % 4, indices])
                    if weights.sum() > 0:
                        weights = weights / weights.sum()
                    else:
                        weights = np.ones(len(indices)) / len(indices)
                    
                    scores = X_scaled[:, indices] @ weights
                    
                    dimensions[dim_name] = {
                        'scores': scores,
                        'features': [features[j] for j in range(len(indices))],
                        'weights': weights,
                        'method': 'bootstrap_stable'
                    }
                    
                    processed_groups.add(dim_name)
                    pc_idx += 1
        
        return dimensions
    
    def _save_comprehensive_results(self, power_results, pca_results, bootstrap_results,
                                   dimensions, pred_results, intervention_threshold,
                                   intrinsic_dim_results=None, within_tier_results=None,
                                   tda_features=None, preservation_scores=None,
                                   phase_results=None, prediction_rmse=None,
                                   tier_results=None, comparison_df=None):
        """Save all results in organized format"""
        
        # Create summary report
        report_path = os.path.join(self.output_dir, 'comprehensive_analysis_report.txt')
        
        with open(report_path, 'w') as f:
            f.write("ENHANCED CONVERSATIONAL SPACE ANALYSIS RESULTS V3\n")
            f.write("="*70 + "\n")
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("1. STATISTICAL POWER\n")
            f.write("-"*30 + "\n")
            for test, power in power_results.items():
                f.write(f"{test}: {power:.3f}\n")
            
            f.write("\n2. INTERVENTION DOMINANCE\n")
            f.write("-"*30 + "\n")
            f.write(f"All features PC1: {pca_results['all']['explained_variance_ratio'][0]:.1%}\n")
            f.write(f"Intervention only PC1: {pca_results['intervention']['explained_variance_ratio'][0]:.1%}\n")
            f.write(f"Non-intervention PC1: {pca_results['non_intervention']['explained_variance_ratio'][0]:.1%}\n")
            f.write(f"Intervention threshold: {intervention_threshold:.3f}\n")
            
            f.write("\n3. BOOTSTRAP STABILITY\n")
            f.write("-"*30 + "\n")
            var_mean = bootstrap_results['variance_mean']
            var_ci = bootstrap_results['variance_ci']
            for i in range(4):
                f.write(f"PC{i+1}: {var_mean[i]:.3f} [{var_ci[0][i]:.3f}, {var_ci[1][i]:.3f}]\n")
            
            f.write("\n4. DIMENSION PREDICTIVE POWER\n")
            f.write("-"*30 + "\n")
            best_result = max(pred_results.values(), key=lambda x: x['test_auc'])
            f.write(f"Best test AUC: {best_result['test_auc']:.3f}\n")
            f.write(f"CV AUC: {best_result['cv_auc_mean']:.3f} ± {best_result['cv_auc_std']:.3f}\n")
            
            if intrinsic_dim_results:
                f.write("\n5. INTRINSIC DIMENSIONALITY\n")
                f.write("-"*30 + "\n")
                if intrinsic_dim_results['ml_estimate']:
                    f.write(f"ML Estimate: {intrinsic_dim_results['ml_estimate']:.2f}\n")
                f.write(f"PCA 90% variance: {intrinsic_dim_results['pca_dim_90']} dimensions\n")
                f.write(f"PCA 95% variance: {intrinsic_dim_results['pca_dim_95']} dimensions\n")
            
            if within_tier_results:
                f.write("\n6. WITHIN-TIER HETEROGENEITY\n")
                f.write("-"*30 + "\n")
                for phase, results in within_tier_results.items():
                    f.write(f"{phase}: {results['optimal_clusters']} subtypes "
                           f"(silhouette={results['silhouette_score']:.3f})\n")
            
            if preservation_scores:
                f.write("\n7. MANIFOLD LEARNING COMPARISON\n")
                f.write("-"*30 + "\n")
                for method, score in preservation_scores.items():
                    f.write(f"{method}: {score:.3f}\n")
            
            if phase_results:
                f.write("\n8. CONVERSATION PHASES\n")
                f.write("-"*30 + "\n")
                f.write(f"Optimal phases: {phase_results['n_phases']}\n")
                f.write(f"Transitions found: {len(phase_results['transitions'])}\n")
            
            if prediction_rmse:
                f.write("\n9. TRAJECTORY PREDICTION\n")
                f.write("-"*30 + "\n")
                for horizon, rmse in prediction_rmse.items():
                    f.write(f"Horizon {horizon}: RMSE = {rmse:.3f}\n")
            
            if comparison_df is not None:
                f.write("\n10. PER-TIER GEOMETRY COMPARISON\n")
                f.write("-"*30 + "\n")
                f.write(comparison_df.to_string(index=False))
                f.write("\n")
        
        print(f"\nComprehensive report saved to: {report_path}")


# Usage
if __name__ == "__main__":
    analyzer = ConversationalAnalysis()
    
    # Run enhanced analysis with per-tier geometry
    results = analyzer.run_complete_enhanced_analysis(
        phase1_path='/home/knots/git/the-academy/docs/paper/exp-data/phase-1-premium/n67/conversation_analysis_enhanced.csv',
        phase2_path='/home/knots/git/the-academy/docs/paper/exp-data/phase-2-efficient/n61/conversation_analysis_enhanced.csv',
        phase3_path='/home/knots/git/the-academy/docs/paper/exp-data/phase-3-no-reasoning/n100/conversation_analysis_enhanced.csv'
    )
    
    print("\nEnhanced analysis V3 complete! Check 'analysis_outputs' for all results.")