import pandas as pd
import numpy as np
from sklearn.decomposition import PCA, FactorAnalysis
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score, KFold
from sklearn.covariance import EmpiricalCovariance
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.linalg import inv, det
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class ConversationalSpacePCA:
    """
    Rigorous PCA analysis for identifying latent dimensions in conversational dynamics.
    Based on theoretical framework from AI peer pressure dynamics research.
    """
    
    def __init__(self, output_dir='pca_analysis_outputs'):
        self.output_dir = output_dir
        self.setup_directories()
        
    def setup_directories(self):
        """Create output directory structure"""
        os.makedirs(self.output_dir, exist_ok=True)
        self.figures_dir = os.path.join(self.output_dir, 'figures')
        self.data_dir = os.path.join(self.output_dir, 'data')
        self.validation_dir = os.path.join(self.output_dir, 'validation')
        
        for dir_path in [self.figures_dir, self.data_dir, self.validation_dir]:
            os.makedirs(dir_path, exist_ok=True)
    
    def load_and_prepare_data(self, phase1_path, phase2_path, phase3_path):
        """Load data with proper phase labeling"""
        # Load your data
        phase1_df = pd.read_csv(phase1_path)
        phase2_df = pd.read_csv(phase2_path)
        phase3_df = pd.read_csv(phase3_path)
        
        # Add phase labels
        phase1_df['phase'] = 'full_reasoning'
        phase2_df['phase'] = 'light_reasoning'
        phase3_df['phase'] = 'no_reasoning'
        
        # Combine all phases
        self.all_data = pd.concat([phase1_df, phase2_df, phase3_df], ignore_index=True)
        print(f"Total conversations: {len(self.all_data)}")
        
        return self.all_data
    
    def define_theoretically_motivated_features(self):
        """
        Define features based on theoretical framework rather than kitchen-sink approach
        """
        # Based on your peer pressure dynamics paper, we should have theoretically motivated groupings
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
        self.features_for_pca = []
        for group in self.feature_groups.values():
            self.features_for_pca.extend(group)
        
        return self.features_for_pca
    
    def calculate_kmo(self, X):
        """Calculate Kaiser-Meyer-Olkin (KMO) measure of sampling adequacy"""
        corr = np.corrcoef(X.T)
        
        # Partial correlations
        try:
            inv_corr = inv(corr)
            partial_corr = -inv_corr / np.sqrt(np.outer(np.diag(inv_corr), np.diag(inv_corr)))
            np.fill_diagonal(partial_corr, 0)
        except:
            return 0.0  # Return 0 if matrix is singular
        
        # KMO calculation
        squared_corr = corr ** 2
        squared_partial = partial_corr ** 2
        
        # Sum of squared correlations
        sum_squared_corr = np.sum(squared_corr) - np.trace(squared_corr)
        # Sum of squared partial correlations
        sum_squared_partial = np.sum(squared_partial) - np.trace(squared_partial)
        
        # KMO
        kmo = sum_squared_corr / (sum_squared_corr + sum_squared_partial)
        
        return kmo
    
    def calculate_bartlett_sphericity(self, X):
        """Calculate Bartlett's test of sphericity"""
        n, p = X.shape
        corr = np.corrcoef(X.T)
        
        # Calculate determinant
        corr_det = det(corr)
        
        # Avoid log(0)
        if corr_det <= 0:
            return 0, 1.0  # Return non-significant result
        
        # Test statistic
        chi_square = -(n - 1 - (2 * p + 5) / 6) * np.log(corr_det)
        
        # Degrees of freedom
        dof = p * (p - 1) / 2
        
        # P-value
        p_value = 1 - stats.chi2.cdf(chi_square, dof)
        
        return chi_square, p_value
    
    def test_assumptions(self, X):
        """Test PCA assumptions rigorously"""
        print("\n=== Testing PCA Assumptions ===")
        
        # 1. Sample size adequacy
        n_samples, n_features = X.shape
        sample_ratio = n_samples / n_features
        print(f"Sample to feature ratio: {sample_ratio:.2f} (should be > 5)")
        
        # 2. KMO Test for sampling adequacy
        try:
            kmo_model = self.calculate_kmo(X)
            print(f"KMO Measure of Sampling Adequacy: {kmo_model:.3f}")
            if kmo_model >= 0.9:
                print("  -> Marvelous adequacy")
            elif kmo_model >= 0.8:
                print("  -> Meritorious adequacy")
            elif kmo_model >= 0.7:
                print("  -> Middling adequacy")
            elif kmo_model >= 0.6:
                print("  -> Mediocre adequacy")
            else:
                print("  -> WARNING: Miserable adequacy, reconsider PCA")
        except:
            print("KMO test failed - check for singular matrix")
        
        # 3. Bartlett's test of sphericity
        try:
            chi_square_value, p_value = self.calculate_bartlett_sphericity(X)
            print(f"\nBartlett's test of sphericity:")
            print(f"  Chi-square value: {chi_square_value:.2f}")
            print(f"  p-value: {p_value:.2e}")
            if p_value < 0.05:
                print("  -> Reject null hypothesis: Variables are correlated")
            else:
                print("  -> WARNING: Fail to reject null, variables may be uncorrelated")
        except:
            print("Bartlett's test failed")
        
        # 4. Multicollinearity check
        corr_matrix = np.corrcoef(X.T)
        eigenvalues = np.linalg.eigvals(corr_matrix)
        condition_number = np.max(eigenvalues) / np.min(eigenvalues)
        print(f"\nCondition number: {condition_number:.2f}")
        if condition_number > 1000:
            print("  -> WARNING: Severe multicollinearity detected")
        elif condition_number > 100:
            print("  -> Moderate multicollinearity detected")
        else:
            print("  -> Acceptable multicollinearity")
        
        # Save assumption test results
        with open(os.path.join(self.validation_dir, 'assumption_tests.txt'), 'w') as f:
            f.write("PCA ASSUMPTION TESTS\n")
            f.write(f"Sample size: {n_samples}\n")
            f.write(f"Number of features: {n_features}\n")
            f.write(f"Sample to feature ratio: {sample_ratio:.2f}\n")
            f.write(f"KMO: {kmo_model:.3f}\n")
            f.write(f"Bartlett's p-value: {p_value:.2e}\n")
            f.write(f"Condition number: {condition_number:.2f}\n")
    
    def perform_robust_pca(self, X):
        """Perform PCA with multiple validation approaches"""
        
        # 1. Standard PCA
        pca_standard = PCA()
        X_pca_standard = pca_standard.fit_transform(X)
        
        # 2. Cross-validated PCA stability
        n_components_test = min(10, X.shape[1])
        stability_scores = self.test_pca_stability(X, n_components_test)
        
        # 3. Bootstrap confidence intervals for loadings
        loading_cis = self.bootstrap_loadings(X, n_components=5, n_bootstrap=100)
        
        # 4. Parallel analysis for component selection
        n_components_parallel = self.parallel_analysis(X)
        
        print(f"\nParallel analysis suggests {n_components_parallel} components")
        
        # Final PCA with optimal components
        self.pca = PCA(n_components=n_components_parallel)
        self.X_pca = self.pca.fit_transform(X)
        
        return self.pca, self.X_pca
    
    def test_pca_stability(self, X, n_components):
        """Test PCA stability using cross-validation"""
        print("\n=== Testing PCA Stability ===")
        
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        loading_similarities = []
        
        for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
            X_train = X[train_idx]
            X_test = X[test_idx]
            
            # Fit PCA on train
            pca_train = PCA(n_components=n_components)
            pca_train.fit(X_train)
            
            # Fit PCA on test
            pca_test = PCA(n_components=n_components)
            pca_test.fit(X_test)
            
            # Compare loadings using absolute correlation
            for i in range(n_components):
                # Account for sign flipping
                corr = np.abs(np.corrcoef(pca_train.components_[i], 
                                         pca_test.components_[i])[0, 1])
                loading_similarities.append(corr)
        
        mean_similarity = np.mean(loading_similarities)
        std_similarity = np.std(loading_similarities)
        
        print(f"Average loading similarity across folds: {mean_similarity:.3f} ± {std_similarity:.3f}")
        
        if mean_similarity < 0.7:
            print("WARNING: Low stability detected. Components may not be reliable.")
        
        return loading_similarities
    
    def bootstrap_loadings(self, X, n_components=5, n_bootstrap=100):
        """Bootstrap confidence intervals for PCA loadings"""
        print("\n=== Bootstrap Loading Confidence Intervals ===")
        
        n_features = X.shape[1]
        bootstrap_loadings = np.zeros((n_bootstrap, n_components, n_features))
        
        for i in range(n_bootstrap):
            # Resample with replacement
            idx = np.random.choice(X.shape[0], X.shape[0], replace=True)
            X_boot = X[idx]
            
            # Fit PCA
            pca_boot = PCA(n_components=n_components)
            pca_boot.fit(X_boot)
            
            # Store loadings (account for sign flipping)
            for j in range(n_components):
                if i > 0:
                    # Align signs with first bootstrap
                    sign = np.sign(np.dot(bootstrap_loadings[0, j], pca_boot.components_[j]))
                    bootstrap_loadings[i, j] = sign * pca_boot.components_[j]
                else:
                    bootstrap_loadings[i, j] = pca_boot.components_[j]
        
        # Calculate confidence intervals
        ci_lower = np.percentile(bootstrap_loadings, 2.5, axis=0)
        ci_upper = np.percentile(bootstrap_loadings, 97.5, axis=0)
        
        # Identify stable loadings (CI doesn't include 0)
        stable_loadings = np.logical_or(
            np.logical_and(ci_lower > 0, ci_upper > 0),
            np.logical_and(ci_lower < 0, ci_upper < 0)
        )
        
        return {'lower': ci_lower, 'upper': ci_upper, 'stable': stable_loadings}
    
    def parallel_analysis(self, X, n_simulations=100):
        """Parallel analysis for component selection"""
        print("\n=== Parallel Analysis ===")
        
        n_samples, n_features = X.shape
        
        # Real data eigenvalues
        pca_real = PCA()
        pca_real.fit(X)
        real_eigenvalues = pca_real.explained_variance_
        
        # Simulated eigenvalues
        simulated_eigenvalues = np.zeros((n_simulations, n_features))
        
        for i in range(n_simulations):
            # Generate random normal data with same shape
            X_random = np.random.normal(0, 1, (n_samples, n_features))
            
            # Standardize
            X_random = StandardScaler().fit_transform(X_random)
            
            # Get eigenvalues
            pca_sim = PCA()
            pca_sim.fit(X_random)
            simulated_eigenvalues[i] = pca_sim.explained_variance_
        
        # 95th percentile of simulated eigenvalues
        simulated_95 = np.percentile(simulated_eigenvalues, 95, axis=0)
        
        # Components to retain
        n_components = np.sum(real_eigenvalues > simulated_95)
        
        # Plot
        plt.figure(figsize=(8, 6))
        plt.plot(range(1, len(real_eigenvalues) + 1), real_eigenvalues, 
                'bo-', label='Real data', markersize=8)
        plt.plot(range(1, len(simulated_95) + 1), simulated_95, 
                'ro--', label='95th percentile random', markersize=6)
        plt.axvline(x=n_components + 0.5, color='gray', linestyle=':', 
                   label=f'Suggested components: {n_components}')
        plt.xlabel('Component Number')
        plt.ylabel('Eigenvalue')
        plt.title('Parallel Analysis for Component Selection')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(self.figures_dir, 'parallel_analysis.pdf'), 
                    dpi=300, bbox_inches='tight')
        plt.close()
        
        return n_components
    
    def interpret_components_theoretically(self):
        """Interpret components based on theoretical framework"""
        interpretations = []
        
        loadings_df = pd.DataFrame(
            self.pca.components_.T,
            index=self.features_for_pca,
            columns=[f'PC{i+1}' for i in range(self.pca.n_components_)]
        )
        
        for i in range(self.pca.n_components_):
            pc_loadings = loadings_df[f'PC{i+1}']
            
            # Group loadings by theoretical construct
            group_contributions = {}
            for group_name, features in self.feature_groups.items():
                group_mask = pc_loadings.index.isin(features)
                group_contribution = np.mean(np.abs(pc_loadings[group_mask]))
                group_contributions[group_name] = group_contribution
            
            # Sort by contribution
            sorted_groups = sorted(group_contributions.items(), 
                                 key=lambda x: x[1], reverse=True)
            
            # Create interpretation
            dominant_construct = sorted_groups[0][0]
            secondary_construct = sorted_groups[1][0] if len(sorted_groups) > 1 else None
            
            interpretation = f"{dominant_construct.replace('_', ' ').title()}"
            if secondary_construct and sorted_groups[1][1] > 0.2:
                interpretation += f" + {secondary_construct.replace('_', ' ').title()}"
            
            interpretations.append({
                'Component': f'PC{i+1}',
                'Variance_Explained': self.pca.explained_variance_ratio_[i],
                'Interpretation': interpretation,
                'Dominant_Construct': dominant_construct,
                'Construct_Contributions': group_contributions
            })
        
        return pd.DataFrame(interpretations)
    
    def test_rotation_solutions(self, X):
        """Test different rotation solutions using sklearn's FactorAnalysis"""
        print("\n=== Testing Rotation Solutions ===")
        
        from sklearn.decomposition import FactorAnalysis
        
        n_factors = self.pca.n_components_
        rotation_results = {}
        
        # Test orthogonal solution (similar to varimax)
        try:
            fa = FactorAnalysis(n_components=n_factors, rotation=None, random_state=42)
            fa.fit(X)
            
            rotation_results['orthogonal'] = {
                'loadings': fa.components_.T,
                'variance': np.var(fa.components_, axis=1),
                'mean_factor_correlation': 0.0  # Orthogonal by design
            }
            
            print(f"Orthogonal: Mean factor correlation = 0.000 (by design)")
            
        except Exception as e:
            print(f"Failed to compute orthogonal solution: {str(e)}")
        
        # For oblique solutions, we'll note that sklearn doesn't support them
        print("\nNote: Oblique rotations (promax, oblimin) require specialized packages.")
        print("Consider using R or the factor_analyzer package for oblique solutions.")
        
        return rotation_results
    
    def create_publication_figures(self):
        """Create publication-quality figures with theoretical grounding"""
        
        # Set publication style
        plt.style.use('seaborn-v0_8-paper')
        plt.rcParams.update({
            'font.size': 10,
            'axes.labelsize': 11,
            'axes.titlesize': 12,
            'xtick.labelsize': 9,
            'ytick.labelsize': 9,
            'legend.fontsize': 9,
            'figure.dpi': 300,
            'font.family': 'sans-serif',
            'font.sans-serif': ['Arial', 'DejaVu Sans'],
        })
        
        # 1. Theory-driven loading plot
        self._create_theoretical_loading_plot()
        
        # 2. Dimensional space visualization
        self._create_dimensional_space_plot()
        
        # 3. Model type separation in latent space
        self._create_model_separation_plot()
    
    def _create_theoretical_loading_plot(self):
        """Create loading plot organized by theoretical constructs"""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Prepare data - use actual number of components
        n_components_to_plot = min(5, self.pca.n_components_)
        loadings_df = pd.DataFrame(
            self.pca.components_[:n_components_to_plot].T,
            index=self.features_for_pca,
            columns=[f'PC{i+1}' for i in range(n_components_to_plot)]
        )
        
        # Order features by theoretical group
        ordered_features = []
        group_positions = []
        current_pos = 0
        
        for group_name, features in self.feature_groups.items():
            group_features = [f for f in features if f in loadings_df.index]
            ordered_features.extend(group_features)
            group_positions.append((current_pos, current_pos + len(group_features), group_name))
            current_pos += len(group_features)
        
        # Create heatmap
        loadings_ordered = loadings_df.loc[ordered_features]
        
        sns.heatmap(loadings_ordered.T, cmap='RdBu_r', center=0,
                   cbar_kws={'label': 'Loading Strength'},
                   vmin=-0.8, vmax=0.8,
                   linewidths=0.5, linecolor='gray')
        
        # Add group separators and labels
        for start, end, name in group_positions:
            ax.axvline(x=start, color='black', linewidth=2)
            # Add group label
            ax.text((start + end) / 2, -0.5, name.replace('_', ' ').title(),
                   ha='center', va='top', rotation=0, fontsize=9)
        
        ax.axvline(x=current_pos, color='black', linewidth=2)
        
        plt.title('Component Loadings by Theoretical Construct')
        plt.xlabel('')
        plt.ylabel('Principal Components')
        plt.tight_layout()
        plt.savefig(os.path.join(self.figures_dir, 'theoretical_loadings.pdf'),
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_dimensional_space_plot(self):
        """Create dimensional space visualization"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # PC1 vs PC2 colored by phase
        ax = axes[0, 0]
        for phase in ['full_reasoning', 'light_reasoning', 'no_reasoning']:
            mask = self.all_data['phase'] == phase
            ax.scatter(self.X_pca[mask, 0], self.X_pca[mask, 1],
                      label=phase.replace('_', ' ').title(),
                      alpha=0.6, s=50)
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.set_title('Conversational Space by Model Type')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # PC1 vs PC2 colored by breakdown
        ax = axes[0, 1]
        breakdown_mask = self.all_data['breakdown_occurred'] == 1
        ax.scatter(self.X_pca[~breakdown_mask, 0], self.X_pca[~breakdown_mask, 1],
                  c='green', label='No Breakdown', alpha=0.6, s=50)
        ax.scatter(self.X_pca[breakdown_mask, 0], self.X_pca[breakdown_mask, 1],
                  c='red', label='Breakdown', alpha=0.6, s=50)
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.set_title('Conversational Space by Outcome')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Density plot for PC1
        ax = axes[1, 0]
        for phase in ['full_reasoning', 'light_reasoning', 'no_reasoning']:
            mask = self.all_data['phase'] == phase
            ax.hist(self.X_pca[mask, 0], bins=20, alpha=0.5,
                   label=phase.replace('_', ' ').title(), density=True)
        ax.set_xlabel('PC1 Value')
        ax.set_ylabel('Density')
        ax.set_title('PC1 Distribution by Model Type')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Component variance plot
        ax = axes[1, 1]
        var_exp = self.pca.explained_variance_ratio_
        cum_var = np.cumsum(var_exp)
        
        x = range(1, len(var_exp) + 1)
        ax.bar(x, var_exp, alpha=0.6, label='Individual')
        ax.plot(x, cum_var, 'ro-', label='Cumulative')
        ax.set_xlabel('Component')
        ax.set_ylabel('Variance Explained')
        ax.set_title('Variance Explained by Components')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.figures_dir, 'dimensional_space_analysis.pdf'),
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_model_separation_plot(self):
        """Statistical analysis of model separation in latent space"""
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Calculate separation metrics for each PC
        separation_metrics = []
        
        for i in range(min(5, self.pca.n_components_)):
            pc_values = self.X_pca[:, i]
            
            # Group by model
            groups = []
            for phase in ['full_reasoning', 'light_reasoning', 'no_reasoning']:
                mask = self.all_data['phase'] == phase
                groups.append(pc_values[mask])
            
            # ANOVA
            f_stat, p_val = stats.f_oneway(*groups)
            
            # Effect size (eta squared)
            ss_between = sum(len(g) * (np.mean(g) - np.mean(pc_values))**2 for g in groups)
            ss_total = np.sum((pc_values - np.mean(pc_values))**2)
            eta_squared = ss_between / ss_total
            
            separation_metrics.append({
                'PC': f'PC{i+1}',
                'F_statistic': f_stat,
                'p_value': p_val,
                'eta_squared': eta_squared
            })
        
        sep_df = pd.DataFrame(separation_metrics)
        
        # Plot
        x = range(len(sep_df))
        ax.bar(x, sep_df['eta_squared'], alpha=0.6)
        
        # Add significance stars
        for i, row in sep_df.iterrows():
            if row['p_value'] < 0.001:
                ax.text(i, row['eta_squared'] + 0.01, '***', ha='center')
            elif row['p_value'] < 0.01:
                ax.text(i, row['eta_squared'] + 0.01, '**', ha='center')
            elif row['p_value'] < 0.05:
                ax.text(i, row['eta_squared'] + 0.01, '*', ha='center')
        
        ax.set_xticks(x)
        ax.set_xticklabels(sep_df['PC'])
        ax.set_xlabel('Principal Component')
        ax.set_ylabel('Effect Size (η²)')
        ax.set_title('Model Type Separation by Component')
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.figures_dir, 'model_separation_effect_sizes.pdf'),
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save statistics
        sep_df.to_csv(os.path.join(self.data_dir, 'model_separation_statistics.csv'),
                     index=False)
    
    def diagnose_pca_issues(self, X):
        """Diagnose issues preventing stable PCA"""
        print("\n=== CRITICAL ISSUES DETECTED ===")
        
        # 1. Sample size issue
        n_samples, n_features = X.shape
        sample_ratio = n_samples / n_features
        print(f"\n1. SAMPLE SIZE PROBLEM:")
        print(f"   - You have {n_samples} samples for {n_features} features")
        print(f"   - Ratio: {sample_ratio:.2f} (minimum should be 5-10)")
        print(f"   - Recommendation: Either collect more data or reduce features")
        
        # 2. Multicollinearity analysis
        corr_matrix = np.corrcoef(X.T)
        print(f"\n2. MULTICOLLINEARITY PROBLEM:")
        
        # Find highly correlated feature pairs
        high_corr_pairs = []
        for i in range(n_features):
            for j in range(i+1, n_features):
                if abs(corr_matrix[i, j]) > 0.9:
                    high_corr_pairs.append((
                        self.features_for_pca[i], 
                        self.features_for_pca[j], 
                        corr_matrix[i, j]
                    ))
        
        if high_corr_pairs:
            print("   Highly correlated features (r > 0.9):")
            for feat1, feat2, corr in high_corr_pairs[:5]:  # Show top 5
                print(f"   - {feat1} <-> {feat2}: {corr:.3f}")
            print(f"   - Total pairs with |r| > 0.9: {len(high_corr_pairs)}")
        
        # 3. Feature variance analysis
        feature_vars = np.var(X, axis=0)
        low_var_features = [self.features_for_pca[i] for i, v in enumerate(feature_vars) if v < 0.01]
        
        print(f"\n3. LOW VARIANCE FEATURES:")
        if low_var_features:
            print(f"   - {len(low_var_features)} features have near-zero variance")
            print(f"   - Examples: {', '.join(low_var_features[:3])}")
        
        # 4. Recommendations
        print("\n=== RECOMMENDATIONS ===")
        print("Given these issues, consider:")
        print("1. Feature selection: Remove redundant/correlated features")
        print("2. Regularized PCA: Use sparse PCA or regularized factor analysis")
        print("3. Domain-specific dimensionality: Use theoretical groupings instead of data-driven PCA")
        print("4. Collect more data: You need at least 115 samples for stable 23-feature PCA")
        
        return high_corr_pairs, low_var_features
    
    def run_complete_analysis(self, phase1_path, phase2_path, phase3_path):
        """Run the complete rigorous PCA analysis"""
        
        # Load data
        self.load_and_prepare_data(phase1_path, phase2_path, phase3_path)
        
        # Define features
        self.define_theoretically_motivated_features()
        
        # Prepare feature matrix
        X = self.all_data[self.features_for_pca].copy()
        imputer = SimpleImputer(strategy='mean')
        X_imputed = imputer.fit_transform(X)
        
        # Use RobustScaler for outlier resistance
        scaler = RobustScaler()
        X_scaled = scaler.fit_transform(X_imputed)
        
        # Test assumptions
        self.test_assumptions(X_scaled)
        
        # Perform robust PCA
        self.perform_robust_pca(X_scaled)
        
        # Test rotations
        rotation_results = self.test_rotation_solutions(X_scaled)
        
        # Interpret components
        interpretations = self.interpret_components_theoretically()
        interpretations.to_csv(os.path.join(self.data_dir, 'component_interpretations.csv'),
                             index=False)
        
        # Create figures
        self.create_publication_figures()
        
        # Save processed data
        pc_df = pd.DataFrame(self.X_pca, 
                           columns=[f'PC{i+1}' for i in range(self.pca.n_components_)])
        pc_df['session_id'] = self.all_data['session_id'].values
        pc_df['phase'] = self.all_data['phase'].values
        pc_df['breakdown'] = self.all_data['breakdown_occurred'].values
        pc_df.to_csv(os.path.join(self.data_dir, 'principal_components.csv'), index=False)
        
        print("\n=== Analysis Complete ===")
        print(f"Results saved to: {self.output_dir}")
        
        return self.pca, self.X_pca, interpretations


# Usage
if __name__ == "__main__":
    analyzer = ConversationalSpacePCA()
    
    # Run analysis
    pca, X_pca, interpretations = analyzer.run_complete_analysis(
        phase1_path='/home/knots/git/the-academy/docs/paper/exp-data/phase-1-premium/n37/conversation_analysis_enhanced.csv',
        phase2_path='/home/knots/git/the-academy/docs/paper/exp-data/phase-2-efficient/n32/conversation_analysis_enhanced.csv',
        phase3_path='/home/knots/git/the-academy/docs/paper/exp-data/phase-3-no-reasoning/n30/conversation_analysis_enhanced.csv'
    )
    
    print("\nTop 5 Components:")
    print(interpretations.head())