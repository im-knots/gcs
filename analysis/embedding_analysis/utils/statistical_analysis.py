"""
Enhanced statistical analysis with power calculations and effect size interpretation.

This module provides rigorous statistical analysis including power calculations,
effect size interpretation, and confidence intervals for all key metrics.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from scipy import stats
from scipy.stats import bootstrap
import statsmodels.stats.power as smp
from statsmodels.stats.multitest import multipletests
import logging

logger = logging.getLogger(__name__)


class EnhancedStatisticalAnalyzer:
    """
    Provides comprehensive statistical analysis with proper effect sizes,
    power calculations, and confidence intervals.
    """
    
    def __init__(self, alpha: float = 0.05):
        """
        Initialize statistical analyzer.
        
        Args:
            alpha: Significance level for hypothesis tests
        """
        self.alpha = alpha
        
    def compute_effect_sizes(self, 
                           group1: np.ndarray, 
                           group2: np.ndarray,
                           paired: bool = False) -> Dict[str, float]:
        """
        Compute multiple effect size measures.
        
        Args:
            group1: First group of observations
            group2: Second group of observations
            paired: Whether observations are paired
            
        Returns:
            Dictionary of effect size measures
        """
        effect_sizes = {}
        
        # Cohen's d
        if paired:
            diff = group1 - group2
            d = np.mean(diff) / np.std(diff, ddof=1)
        else:
            pooled_std = np.sqrt(((len(group1) - 1) * np.var(group1, ddof=1) + 
                                 (len(group2) - 1) * np.var(group2, ddof=1)) / 
                                (len(group1) + len(group2) - 2))
            d = (np.mean(group1) - np.mean(group2)) / pooled_std
        
        effect_sizes['cohens_d'] = d
        effect_sizes['d_interpretation'] = self._interpret_cohens_d(d)
        
        # Hedges' g (bias-corrected Cohen's d)
        n1, n2 = len(group1), len(group2)
        df = n1 + n2 - 2
        correction = 1 - 3 / (4 * df - 1)
        effect_sizes['hedges_g'] = d * correction
        
        # Glass's delta (when variances are unequal)
        effect_sizes['glass_delta'] = (np.mean(group1) - np.mean(group2)) / np.std(group2, ddof=1)
        
        # Common language effect size (probability of superiority)
        if not paired:
            # Mann-Whitney U test
            u_stat, _ = stats.mannwhitneyu(group1, group2, alternative='two-sided')
            cles = u_stat / (n1 * n2)
            effect_sizes['cles'] = cles
            effect_sizes['cles_interpretation'] = f"{cles*100:.1f}% chance that random value from group 1 > group 2"
        
        # Rank-biserial correlation
        if not paired:
            u1 = u_stat
            u2 = n1 * n2 - u1
            r_rb = 1 - (2 * min(u1, u2)) / (n1 * n2)
            effect_sizes['rank_biserial'] = r_rb
        
        return effect_sizes
    
    def compute_correlation_effect_sizes(self,
                                       r1: float,
                                       r2: float,
                                       n1: int,
                                       n2: int) -> Dict[str, float]:
        """
        Compute effect sizes for correlation differences.
        
        Uses Cohen's q for correlation comparisons.
        """
        # Fisher's z transformation
        z1 = np.arctanh(r1)
        z2 = np.arctanh(r2)
        
        # Cohen's q
        q = z1 - z2
        
        # Standard error of difference
        se_diff = np.sqrt(1/(n1-3) + 1/(n2-3))
        
        # Confidence interval for difference
        ci_lower = q - 1.96 * se_diff
        ci_upper = q + 1.96 * se_diff
        
        return {
            'cohens_q': q,
            'q_interpretation': self._interpret_cohens_q(q),
            'ci_lower': np.tanh(ci_lower),
            'ci_upper': np.tanh(ci_upper),
            'se_difference': se_diff
        }
    
    def compute_power_analysis(self,
                             effect_size: float,
                             n_obs: int,
                             test_type: str = 'correlation',
                             alpha: Optional[float] = None) -> Dict[str, float]:
        """
        Compute statistical power for different test types.
        
        Args:
            effect_size: Effect size (Cohen's d, r, or q)
            n_obs: Number of observations
            test_type: Type of test ('correlation', 't-test', 'anova')
            alpha: Significance level (uses instance default if None)
            
        Returns:
            Power analysis results
        """
        if alpha is None:
            alpha = self.alpha
            
        results = {}
        
        if test_type == 'correlation':
            # Power for correlation test
            power = smp.ttest_power(effect_size, n_obs, alpha, alternative='two-sided')
            results['power'] = power
            
            # Required sample size for different power levels
            for target_power in [0.80, 0.90, 0.95]:
                n_required = smp.tt_solve_power(effect_size, power=target_power, 
                                              alpha=alpha, alternative='two-sided')
                results[f'n_for_power_{target_power}'] = int(np.ceil(n_required))
        
        elif test_type == 't-test':
            # Power for t-test
            power = smp.ttest_power(effect_size, n_obs, alpha)
            results['power'] = power
            
            # Required sample size
            n_required = smp.tt_solve_power(effect_size, power=0.80, alpha=alpha)
            results['n_for_power_0.80'] = int(np.ceil(n_required))
            
        elif test_type == 'anova':
            # Power for ANOVA (simplified for 3 groups)
            power = smp.FTestAnovaPower().solve_power(
                effect_size=effect_size,
                nobs=n_obs,
                alpha=alpha,
                k_groups=3
            )
            results['power'] = power
        
        # Interpret power
        results['power_interpretation'] = self._interpret_power(results.get('power', 0))
        results['is_adequate'] = results.get('power', 0) >= 0.80
        
        return results
    
    def compute_confidence_intervals(self,
                                   data: np.ndarray,
                                   statistic_func: callable,
                                   confidence_level: float = 0.95,
                                   n_bootstrap: int = 10000) -> Dict[str, float]:
        """
        Compute bootstrap confidence intervals for any statistic.
        
        Args:
            data: Input data
            statistic_func: Function to compute statistic
            confidence_level: Confidence level (default 0.95)
            n_bootstrap: Number of bootstrap samples
            
        Returns:
            Confidence interval results
        """
        # Compute point estimate
        point_estimate = statistic_func(data)
        
        # Bootstrap confidence interval
        rng = np.random.default_rng(42)  # For reproducibility
        
        # Handle both 1D and 2D data
        if data.ndim == 1:
            data_tuple = (data,)
        else:
            data_tuple = (data,)
        
        result = bootstrap(
            data_tuple,
            lambda x: statistic_func(x[0]),
            n_resamples=n_bootstrap,
            confidence_level=confidence_level,
            random_state=rng,
            method='percentile'
        )
        
        return {
            'point_estimate': point_estimate,
            'ci_lower': result.confidence_interval.low,
            'ci_upper': result.confidence_interval.high,
            'confidence_level': confidence_level,
            'se_bootstrap': result.standard_error
        }
    
    def analyze_invariance_scores(self,
                                scores: Dict[str, List[float]]) -> Dict[str, Dict]:
        """
        Comprehensive analysis of invariance scores with effect sizes and power.
        
        Args:
            scores: Dictionary mapping score types to lists of scores
            
        Returns:
            Comprehensive statistical analysis
        """
        results = {}
        
        for score_type, score_list in scores.items():
            if not score_list:
                continue
                
            score_array = np.array(score_list)
            
            analysis = {
                'n': len(score_array),
                'mean': np.mean(score_array),
                'std': np.std(score_array, ddof=1),
                'median': np.median(score_array),
                'iqr': np.percentile(score_array, 75) - np.percentile(score_array, 25)
            }
            
            # Test against null hypothesis (e.g., random correlation of 0.1)
            null_value = 0.1
            t_stat, p_value = stats.ttest_1samp(score_array, null_value)
            
            # Effect size against null
            d = (np.mean(score_array) - null_value) / np.std(score_array, ddof=1)
            analysis['effect_vs_null'] = {
                'cohens_d': d,
                'interpretation': self._interpret_cohens_d(d),
                'p_value': p_value,
                'significant': p_value < self.alpha
            }
            
            # Power analysis
            power_results = self.compute_power_analysis(
                effect_size=d,
                n_obs=len(score_array),
                test_type='t-test'
            )
            analysis['power_analysis'] = power_results
            
            # Confidence intervals
            ci_results = self.compute_confidence_intervals(
                score_array,
                np.mean,
                confidence_level=0.95
            )
            analysis['confidence_interval'] = ci_results
            
            # Distribution tests
            _, normality_p = stats.shapiro(score_array)
            analysis['normality_test'] = {
                'p_value': normality_p,
                'is_normal': normality_p > 0.05,
                'interpretation': 'Data appears normally distributed' if normality_p > 0.05 
                                else 'Data deviates from normality'
            }
            
            results[score_type] = analysis
            
        return results
    
    def analyze_scale_differences(self,
                                global_scores: List[float],
                                meso_scores: List[float],
                                local_scores: List[float]) -> Dict:
        """
        Analyze differences between scales with proper effect sizes.
        """
        results = {}
        
        # Pairwise comparisons
        comparisons = [
            ('global_vs_meso', global_scores, meso_scores),
            ('global_vs_local', global_scores, local_scores),
            ('meso_vs_local', meso_scores, local_scores)
        ]
        
        for comp_name, scores1, scores2 in comparisons:
            # Convert to arrays
            arr1 = np.array(scores1)
            arr2 = np.array(scores2)
            
            # T-test
            t_stat, p_value = stats.ttest_ind(arr1, arr2)
            
            # Effect sizes
            effect_sizes = self.compute_effect_sizes(arr1, arr2, paired=False)
            
            # Power analysis
            power = self.compute_power_analysis(
                effect_size=effect_sizes['cohens_d'],
                n_obs=min(len(arr1), len(arr2)),
                test_type='t-test'
            )
            
            results[comp_name] = {
                't_statistic': t_stat,
                'p_value': p_value,
                'effect_sizes': effect_sizes,
                'power_analysis': power,
                'significant': p_value < self.alpha
            }
        
        # Overall ANOVA
        f_stat, anova_p = stats.f_oneway(global_scores, meso_scores, local_scores)
        
        # Eta squared for ANOVA
        grand_mean = np.mean(np.concatenate([global_scores, meso_scores, local_scores]))
        ss_between = (len(global_scores) * (np.mean(global_scores) - grand_mean)**2 +
                     len(meso_scores) * (np.mean(meso_scores) - grand_mean)**2 +
                     len(local_scores) * (np.mean(local_scores) - grand_mean)**2)
        ss_total = np.sum([(x - grand_mean)**2 for x in 
                          np.concatenate([global_scores, meso_scores, local_scores])])
        eta_squared = ss_between / ss_total if ss_total > 0 else 0
        
        results['anova'] = {
            'f_statistic': f_stat,
            'p_value': anova_p,
            'eta_squared': eta_squared,
            'eta_squared_interpretation': self._interpret_eta_squared(eta_squared),
            'significant': anova_p < self.alpha
        }
        
        return results
    
    def _interpret_cohens_d(self, d: float) -> str:
        """Interpret Cohen's d effect size."""
        d_abs = abs(d)
        if d_abs < 0.2:
            return "negligible"
        elif d_abs < 0.5:
            return "small"
        elif d_abs < 0.8:
            return "medium"
        else:
            return "large"
    
    def _interpret_cohens_q(self, q: float) -> str:
        """Interpret Cohen's q for correlation differences."""
        q_abs = abs(q)
        if q_abs < 0.1:
            return "negligible"
        elif q_abs < 0.3:
            return "small"
        elif q_abs < 0.5:
            return "medium"
        else:
            return "large"
    
    def _interpret_eta_squared(self, eta2: float) -> str:
        """Interpret eta squared effect size."""
        if eta2 < 0.01:
            return "negligible"
        elif eta2 < 0.06:
            return "small"
        elif eta2 < 0.14:
            return "medium"
        else:
            return "large"
    
    def _interpret_power(self, power: float) -> str:
        """Interpret statistical power."""
        if power < 0.5:
            return "very low - high risk of Type II error"
        elif power < 0.7:
            return "low - moderate risk of Type II error"
        elif power < 0.8:
            return "acceptable - some risk of Type II error"
        elif power < 0.9:
            return "good - low risk of Type II error"
        else:
            return "excellent - very low risk of Type II error"


class BayesianAnalyzer:
    """
    Bayesian analysis for more nuanced interpretation of results.
    """
    
    def __init__(self):
        """Initialize Bayesian analyzer."""
        pass
    
    def compute_bayes_factor(self,
                           data: np.ndarray,
                           null_value: float = 0,
                           prior_scale: float = 1.0) -> Dict[str, float]:
        """
        Compute Bayes factor for testing against null hypothesis.
        
        Uses Jeffreys-Zellner-Siow (JZS) prior.
        """
        # Simplified Bayes factor calculation
        # In practice, use specialized packages like PyMC3
        
        n = len(data)
        t_stat = (np.mean(data) - null_value) / (np.std(data, ddof=1) / np.sqrt(n))
        
        # JZS Bayes factor approximation (Rouder et al., 2009)
        r = prior_scale
        df = n - 1
        
        # Numerical integration would be more accurate
        bf_10 = np.sqrt((df + t_stat**2) / df) * \
                ((1 + t_stat**2/df)**(-0.5*(df+1))) * \
                (r / np.sqrt(2))
        
        return {
            'bf_10': bf_10,
            'bf_01': 1 / bf_10,
            'log_bf_10': np.log10(bf_10),
            'interpretation': self._interpret_bayes_factor(bf_10)
        }
    
    def _interpret_bayes_factor(self, bf: float) -> str:
        """Interpret Bayes factor according to Jeffreys' scale."""
        if bf < 1/10:
            return "strong evidence for null"
        elif bf < 1/3:
            return "moderate evidence for null"
        elif bf < 1:
            return "anecdotal evidence for null"
        elif bf < 3:
            return "anecdotal evidence for alternative"
        elif bf < 10:
            return "moderate evidence for alternative"
        elif bf < 30:
            return "strong evidence for alternative"
        elif bf < 100:
            return "very strong evidence for alternative"
        else:
            return "extreme evidence for alternative"