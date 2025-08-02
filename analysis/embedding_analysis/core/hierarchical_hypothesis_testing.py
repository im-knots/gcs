"""
Hierarchical Hypothesis Testing Framework for Geometric Invariance

This module implements the revised hypothesis testing structure with proper
statistical methods including Fisher transformations, multiple testing corrections,
and effect size calculations.
"""

import numpy as np
import scipy.stats as stats
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import logging
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.power import tt_solve_power
import warnings

logger = logging.getLogger(__name__)


@dataclass
class HypothesisResult:
    """Result of a single hypothesis test."""
    name: str
    passed: bool
    p_value: float
    test_statistic: float
    effect_size: float
    confidence_interval: Tuple[float, float]
    power: float
    description: str


@dataclass
class TierResult:
    """Result of a tier of hypothesis tests."""
    tier_number: int
    tier_name: str
    passed: bool
    hypotheses: List[HypothesisResult]
    adjusted_alpha: float
    correction_method: str


class HierarchicalHypothesisTester:
    """
    Implements hierarchical hypothesis testing for geometric invariance.
    
    Tests proceed through tiers, with each tier only tested if the previous passes.
    """
    
    def __init__(self, alpha: float = 0.05):
        self.alpha = alpha
        self.results = {}
        
    def fisher_transform(self, r: float) -> float:
        """Apply Fisher's z-transformation to correlation."""
        # Handle edge cases
        if r >= 1.0:
            return np.inf
        elif r <= -1.0:
            return -np.inf
        return 0.5 * np.log((1 + r) / (1 - r))
    
    def inverse_fisher_transform(self, z: float) -> float:
        """Inverse Fisher transformation."""
        return np.tanh(z)
    
    def fisher_confidence_interval(self, r: float, n: int, confidence: float = 0.95) -> Tuple[float, float]:
        """
        Calculate confidence interval for correlation using Fisher transformation.
        
        Args:
            r: Correlation coefficient
            n: Sample size
            confidence: Confidence level
            
        Returns:
            (lower, upper) confidence bounds
        """
        z = self.fisher_transform(r)
        se = 1 / np.sqrt(n - 3)
        z_crit = stats.norm.ppf((1 + confidence) / 2)
        
        z_lower = z - z_crit * se
        z_upper = z + z_crit * se
        
        return (self.inverse_fisher_transform(z_lower), 
                self.inverse_fisher_transform(z_upper))
    
    def cohens_q(self, r1: float, r2: float) -> float:
        """
        Calculate Cohen's q effect size for difference between correlations.
        
        Args:
            r1: First correlation
            r2: Second correlation
            
        Returns:
            Cohen's q effect size
        """
        z1 = self.fisher_transform(r1)
        z2 = self.fisher_transform(r2)
        return abs(z1 - z2)
    
    def steiger_test_dependent(self, r12: float, r13: float, r23: float, n: int) -> Tuple[float, float]:
        """
        Steiger's test for comparing two dependent correlations.
        
        Tests if r12 != r13 given r23.
        
        Args:
            r12: Correlation between variables 1 and 2
            r13: Correlation between variables 1 and 3
            r23: Correlation between variables 2 and 3
            n: Sample size
            
        Returns:
            (z_statistic, p_value)
        """
        # Fisher transform
        z12 = self.fisher_transform(r12)
        z13 = self.fisher_transform(r13)
        
        # Calculate test statistic
        numerator = (z12 - z13) * np.sqrt(n - 3)
        denominator = np.sqrt(2 * (1 - r23))
        
        z = numerator / denominator
        p_value = 2 * (1 - stats.norm.cdf(abs(z)))
        
        return z, p_value
    
    def calculate_power(self, effect_size: float, n: int, alpha: float = 0.05) -> float:
        """
        Calculate statistical power for correlation test.
        
        Args:
            effect_size: Effect size (Cohen's q or similar)
            n: Sample size
            alpha: Significance level
            
        Returns:
            Statistical power
        """
        try:
            # Convert effect size to correlation for power calculation
            # Using approximation for correlation power
            power = tt_solve_power(effect_size=effect_size, nobs=n, alpha=alpha, 
                                 power=None, alternative='two-sided')
            return power
        except:
            # Fallback for edge cases
            return np.nan
    
    def test_tier1_within_paradigm(self, correlations: Dict[str, List[float]]) -> TierResult:
        """
        Test Tier 1: Within-paradigm invariance.
        
        Args:
            correlations: Dict with keys 'transformer_pairs', 'classical_pairs'
                         containing lists of pairwise correlations
                         
        Returns:
            TierResult with test outcomes
        """
        hypotheses = []
        
        # H1a: Transformer models show strong correlations
        transformer_corrs = correlations.get('transformer_pairs', [])
        if len(transformer_corrs) > 0:
            # Convert to Fisher z, calculate mean, then back-transform
            z_scores = [self.fisher_transform(r) for r in transformer_corrs]
            mean_z = np.mean(z_scores)
            mean_r = self.inverse_fisher_transform(mean_z)
            
            # Test against threshold
            n = len(transformer_corrs)
            se = 1 / np.sqrt(n - 3)
            z_threshold = self.fisher_transform(0.75)
            z_stat = (mean_z - z_threshold) / se
            p_value = 1 - stats.norm.cdf(z_stat)  # One-tailed test
            
            # Calculate power
            effect_size = self.cohens_q(mean_r, 0.75)
            power = self.calculate_power(effect_size, n, self.alpha/3)  # Bonferroni
            
            # Confidence interval
            ci = self.fisher_confidence_interval(mean_r, n)
            
            h1a = HypothesisResult(
                name="H1a",
                passed=p_value < self.alpha/3 and mean_r > 0.75,
                p_value=p_value,
                test_statistic=z_stat,
                effect_size=effect_size,
                confidence_interval=ci,
                power=power,
                description="Transformer models show strong correlations (ρ > 0.75)"
            )
            hypotheses.append(h1a)
        
        # H1b: Classical models show strong correlations
        classical_corrs = correlations.get('classical_pairs', [])
        if len(classical_corrs) > 0:
            z_scores = [self.fisher_transform(r) for r in classical_corrs]
            mean_z = np.mean(z_scores)
            mean_r = self.inverse_fisher_transform(mean_z)
            
            n = len(classical_corrs)
            se = 1 / np.sqrt(n - 3)
            z_threshold = self.fisher_transform(0.70)
            z_stat = (mean_z - z_threshold) / se
            p_value = 1 - stats.norm.cdf(z_stat)
            
            effect_size = self.cohens_q(mean_r, 0.70)
            power = self.calculate_power(effect_size, n, self.alpha/3)
            ci = self.fisher_confidence_interval(mean_r, n)
            
            h1b = HypothesisResult(
                name="H1b",
                passed=p_value < self.alpha/3 and mean_r > 0.70,
                p_value=p_value,
                test_statistic=z_stat,
                effect_size=effect_size,
                confidence_interval=ci,
                power=power,
                description="Classical models show strong correlations (ρ > 0.70)"
            )
            hypotheses.append(h1b)
        
        # H1c: Within-paradigm correlations exceed chance
        all_within = list(transformer_corrs) + list(classical_corrs)
        null_corrs = correlations.get('null_within_paradigm', [])
        
        if len(all_within) > 0 and len(null_corrs) > 0:
            # Mann-Whitney U test for difference
            u_stat, p_value = stats.mannwhitneyu(all_within, null_corrs, 
                                                alternative='greater')
            
            # Effect size (rank-biserial correlation)
            n1, n2 = len(all_within), len(null_corrs)
            effect_size = 1 - (2 * u_stat) / (n1 * n2)
            
            # Bootstrap confidence interval for median difference
            n_bootstrap = 1000
            diffs = []
            for _ in range(n_bootstrap):
                sample1 = np.random.choice(all_within, n1, replace=True)
                sample2 = np.random.choice(null_corrs, n2, replace=True)
                diffs.append(np.median(sample1) - np.median(sample2))
            ci = (np.percentile(diffs, 2.5), np.percentile(diffs, 97.5))
            
            h1c = HypothesisResult(
                name="H1c",
                passed=p_value < self.alpha/3,
                p_value=p_value,
                test_statistic=u_stat,
                effect_size=effect_size,
                confidence_interval=ci,
                power=np.nan,  # Power for non-parametric test
                description="Within-paradigm correlations exceed chance (p < 0.01)"
            )
            hypotheses.append(h1c)
        
        # Determine if tier passed
        tier_passed = all(h.passed for h in hypotheses)
        
        return TierResult(
            tier_number=1,
            tier_name="Within-Paradigm Invariance",
            passed=tier_passed,
            hypotheses=hypotheses,
            adjusted_alpha=self.alpha/3,  # Bonferroni
            correction_method="Bonferroni"
        )
    
    def test_tier2_cross_paradigm(self, correlations: Dict[str, List[float]]) -> TierResult:
        """
        Test Tier 2: Cross-paradigm invariance.
        
        Args:
            correlations: Dict with 'cross_paradigm_pairs' and comparison data
            
        Returns:
            TierResult with test outcomes
        """
        hypotheses = []
        
        # H2a: Transformer-to-Classical correlations are substantial
        cross_corrs = correlations.get('cross_paradigm_pairs', [])
        if len(cross_corrs) > 0:
            z_scores = [self.fisher_transform(r) for r in cross_corrs]
            mean_z = np.mean(z_scores)
            mean_r = self.inverse_fisher_transform(mean_z)
            
            n = len(cross_corrs)
            se = 1 / np.sqrt(n - 3)
            z_threshold = self.fisher_transform(0.50)
            z_stat = (mean_z - z_threshold) / se
            p_value = 1 - stats.norm.cdf(z_stat)
            
            effect_size = self.cohens_q(mean_r, 0.50)
            power = self.calculate_power(effect_size, n, self.alpha)
            ci = self.fisher_confidence_interval(mean_r, n)
            
            h2a = HypothesisResult(
                name="H2a",
                passed=p_value < self.alpha and mean_r > 0.50,
                p_value=p_value,
                test_statistic=z_stat,
                effect_size=effect_size,
                confidence_interval=ci,
                power=power,
                description="Cross-paradigm correlations are substantial (ρ > 0.50)"
            )
            hypotheses.append(h2a)
        
        # H2b: All cross-paradigm correlations are positive
        if len(cross_corrs) > 0:
            # Test if all correlations are significantly positive
            n_positive = sum(1 for r in cross_corrs if r > 0)
            # Binomial test
            p_value = stats.binom_test(n_positive, len(cross_corrs), 0.5, 
                                      alternative='greater')
            
            # Effect size (proportion above 0)
            effect_size = n_positive / len(cross_corrs) - 0.5
            
            h2b = HypothesisResult(
                name="H2b",
                passed=p_value < self.alpha,
                p_value=p_value,
                test_statistic=n_positive,
                effect_size=effect_size,
                confidence_interval=(min(cross_corrs), max(cross_corrs)),
                power=np.nan,
                description="All cross-paradigm correlations are positive"
            )
            hypotheses.append(h2b)
        
        # H2c: Cross-paradigm exceeds random embeddings
        random_corrs = correlations.get('random_embedding_pairs', [])
        if len(cross_corrs) > 0 and len(random_corrs) > 0:
            u_stat, p_value = stats.mannwhitneyu(cross_corrs, random_corrs,
                                                alternative='greater')
            
            n1, n2 = len(cross_corrs), len(random_corrs)
            effect_size = 1 - (2 * u_stat) / (n1 * n2)
            
            h2c = HypothesisResult(
                name="H2c",
                passed=p_value < self.alpha,
                p_value=p_value,
                test_statistic=u_stat,
                effect_size=effect_size,
                confidence_interval=(np.nan, np.nan),
                power=np.nan,
                description="Cross-paradigm exceeds random embeddings"
            )
            hypotheses.append(h2c)
        
        # Apply FDR correction
        if hypotheses:
            p_values = [h.p_value for h in hypotheses]
            rejected, adjusted_p, _, _ = multipletests(p_values, alpha=self.alpha, 
                                                      method='fdr_bh')
            
            # Update hypothesis results with FDR correction
            for i, h in enumerate(hypotheses):
                h.passed = rejected[i] and h.passed  # Must pass both original and FDR
        
        tier_passed = all(h.passed for h in hypotheses)
        
        return TierResult(
            tier_number=2,
            tier_name="Cross-Paradigm Invariance",
            passed=tier_passed,
            hypotheses=hypotheses,
            adjusted_alpha=self.alpha,
            correction_method="FDR (Benjamini-Hochberg)"
        )
    
    def test_tier3_hierarchy(self, correlations: Dict[str, List[float]], 
                            metrics: Dict[str, Dict]) -> TierResult:
        """
        Test Tier 3: Invariance hierarchy.
        
        Args:
            correlations: Correlation data from all paradigms
            metrics: Additional geometric metrics for testing
            
        Returns:
            TierResult with test outcomes
        """
        hypotheses = []
        
        # H3a: Within > Cross > Random ordering
        transformer_corrs = correlations.get('transformer_pairs', [])
        classical_corrs = correlations.get('classical_pairs', [])
        # Convert to lists and concatenate
        within_corrs = list(transformer_corrs) + list(classical_corrs)
        cross_corrs = correlations.get('cross_paradigm_pairs', [])
        random_corrs = correlations.get('random_embedding_pairs', [])
        
        if len(within_corrs) > 0 and len(cross_corrs) > 0 and len(random_corrs) > 0:
            # Kruskal-Wallis test for ordering
            h_stat, p_value = stats.kruskal(within_corrs, cross_corrs, random_corrs)
            
            # Post-hoc: Check specific ordering
            within_mean = np.mean([self.fisher_transform(r) for r in within_corrs])
            cross_mean = np.mean([self.fisher_transform(r) for r in cross_corrs])
            random_mean = np.mean([self.fisher_transform(r) for r in random_corrs])
            
            ordering_satisfied = within_mean > cross_mean > random_mean
            
            # Effect size (eta-squared)
            n_total = len(within_corrs) + len(cross_corrs) + len(random_corrs)
            effect_size = (h_stat - 2) / (n_total - 3)
            
            h3a = HypothesisResult(
                name="H3a",
                passed=p_value < self.alpha and ordering_satisfied,
                p_value=p_value,
                test_statistic=h_stat,
                effect_size=effect_size,
                confidence_interval=(np.nan, np.nan),
                power=np.nan,
                description="Within > Cross > Random hierarchy"
            )
            hypotheses.append(h3a)
        
        # H3b: Meaningful effect size differences
        if len(within_corrs) > 0 and len(cross_corrs) > 0:
            # Cohen's q between within and cross
            within_mean_r = self.inverse_fisher_transform(
                np.mean([self.fisher_transform(r) for r in within_corrs])
            )
            cross_mean_r = self.inverse_fisher_transform(
                np.mean([self.fisher_transform(r) for r in cross_corrs])
            )
            
            effect_size = self.cohens_q(within_mean_r, cross_mean_r)
            
            h3b = HypothesisResult(
                name="H3b",
                passed=effect_size > 0.3,
                p_value=np.nan,  # Effect size criterion, not p-value
                test_statistic=effect_size,
                effect_size=effect_size,
                confidence_interval=(np.nan, np.nan),
                power=np.nan,
                description="Effect size between levels is meaningful (q > 0.3)"
            )
            hypotheses.append(h3b)
        
        # H3c: Hierarchy persists across metrics
        if metrics:
            consistencies = []
            for metric_name, metric_data in metrics.items():
                if all(k in metric_data for k in ['within', 'cross', 'random']):
                    # Check ordering for this metric
                    ordering = (np.mean(metric_data['within']) > 
                              np.mean(metric_data['cross']) > 
                              np.mean(metric_data['random']))
                    consistencies.append(ordering)
            
            proportion_consistent = np.mean(consistencies) if consistencies else 0
            
            h3c = HypothesisResult(
                name="H3c",
                passed=proportion_consistent > 0.8,
                p_value=np.nan,
                test_statistic=proportion_consistent,
                effect_size=proportion_consistent - 0.5,
                confidence_interval=(0, 1),
                power=np.nan,
                description="Hierarchy persists across geometric metrics"
            )
            hypotheses.append(h3c)
        
        tier_passed = all(h.passed for h in hypotheses)
        
        return TierResult(
            tier_number=3,
            tier_name="Invariance Hierarchy",
            passed=tier_passed,
            hypotheses=hypotheses,
            adjusted_alpha=self.alpha,
            correction_method="None (effect size criteria)"
        )
    
    def test_controls(self, data: Dict) -> List[HypothesisResult]:
        """
        Test control hypotheses independently.
        
        Args:
            data: Dictionary with control test data
            
        Returns:
            List of HypothesisResult objects
        """
        controls = []
        
        # H4: Real > Scrambled conversations
        if 'real_scrambled_comparison' in data:
            real_corrs = data['real_scrambled_comparison']['real']
            scrambled_corrs = data['real_scrambled_comparison']['scrambled']
            
            # Check if we have data to test
            if len(real_corrs) == 0 or len(scrambled_corrs) == 0:
                # No data available - create a failed result
                h4 = HypothesisResult(
                    name="H4",
                    passed=False,
                    p_value=1.0,
                    test_statistic=np.nan,
                    effect_size=0.0,
                    confidence_interval=(np.nan, np.nan),
                    power=np.nan,
                    description="Real conversations > Scrambled conversations (insufficient data)"
                )
                controls.append(h4)
            else:
                u_stat, p_value = stats.mannwhitneyu(real_corrs, scrambled_corrs,
                                                    alternative='greater')
                
                n1, n2 = len(real_corrs), len(scrambled_corrs)
                effect_size = 1 - (2 * u_stat) / (n1 * n2)
                
                h4 = HypothesisResult(
                    name="H4",
                    passed=p_value < self.alpha,
                    p_value=p_value,
                    test_statistic=u_stat,
                    effect_size=effect_size,
                    confidence_interval=(np.nan, np.nan),
                    power=np.nan,
                    description="Real conversations > Scrambled conversations"
                )
                controls.append(h4)
        
        # H5: Patterns persist after controlling for length
        if 'length_controlled' in data:
            # Partial correlation controlling for message length
            partial_corr = data['length_controlled']['partial_correlation']
            n = data['length_controlled']['n_conversations']
            
            # Test if partial correlation is significant
            t_stat = partial_corr * np.sqrt((n - 2) / (1 - partial_corr**2))
            p_value = 2 * (1 - stats.t.cdf(abs(t_stat), n - 2))
            
            h5 = HypothesisResult(
                name="H5",
                passed=p_value < self.alpha and partial_corr > 0.5,
                p_value=p_value,
                test_statistic=t_stat,
                effect_size=partial_corr,
                confidence_interval=self.fisher_confidence_interval(partial_corr, n),
                power=self.calculate_power(partial_corr, n, self.alpha),
                description="Patterns persist after controlling for message length"
            )
            controls.append(h5)
        
        # H6: Results hold with normalized metrics
        if 'normalized_metrics' in data:
            norm_corrs = data['normalized_metrics']['correlations']
            threshold = 0.5
            
            z_scores = [self.fisher_transform(r) for r in norm_corrs]
            mean_z = np.mean(z_scores)
            mean_r = self.inverse_fisher_transform(mean_z)
            
            n = len(norm_corrs)
            se = 1 / np.sqrt(n - 3)
            z_threshold = self.fisher_transform(threshold)
            z_stat = (mean_z - z_threshold) / se
            p_value = 1 - stats.norm.cdf(z_stat)
            
            h6 = HypothesisResult(
                name="H6",
                passed=p_value < self.alpha and mean_r > threshold,
                p_value=p_value,
                test_statistic=z_stat,
                effect_size=self.cohens_q(mean_r, threshold),
                confidence_interval=self.fisher_confidence_interval(mean_r, n),
                power=self.calculate_power(self.cohens_q(mean_r, threshold), n, self.alpha),
                description="Results hold with dimension-normalized metrics"
            )
            controls.append(h6)
        
        # Apply FDR correction to control tests
        if controls:
            p_values = [h.p_value for h in controls if not np.isnan(h.p_value)]
            if p_values:
                rejected, _, _, _ = multipletests(p_values, alpha=self.alpha, 
                                                 method='fdr_bh')
                
                j = 0
                for h in controls:
                    if not np.isnan(h.p_value):
                        h.passed = rejected[j] and h.passed
                        j += 1
        
        return controls
    
    def run_hierarchical_testing(self, data: Dict) -> Dict:
        """
        Run the complete hierarchical hypothesis testing framework.
        
        Args:
            data: Dictionary containing all necessary correlation and metric data
            
        Returns:
            Dictionary with complete testing results
        """
        results = {
            'tiers': [],
            'controls': [],
            'summary': {}
        }
        
        # Tier 1: Within-paradigm
        logger.info("Testing Tier 1: Within-paradigm invariance")
        tier1 = self.test_tier1_within_paradigm(data['correlations'])
        results['tiers'].append(tier1)
        
        if not tier1.passed:
            logger.warning("Tier 1 failed - stopping hierarchical testing")
            results['summary']['conclusion'] = "Failed at Tier 1: Within-paradigm invariance not established"
            results['summary']['max_tier_passed'] = 0
        else:
            # Tier 2: Cross-paradigm
            logger.info("Testing Tier 2: Cross-paradigm invariance")
            tier2 = self.test_tier2_cross_paradigm(data['correlations'])
            results['tiers'].append(tier2)
            
            if not tier2.passed:
                logger.warning("Tier 2 failed - stopping hierarchical testing")
                results['summary']['conclusion'] = "Failed at Tier 2: Cross-paradigm invariance not established"
                results['summary']['max_tier_passed'] = 1
            else:
                # Tier 3: Hierarchy
                logger.info("Testing Tier 3: Invariance hierarchy")
                tier3 = self.test_tier3_hierarchy(data['correlations'], 
                                                 data.get('geometric_metrics', {}))
                results['tiers'].append(tier3)
                
                if tier3.passed:
                    results['summary']['conclusion'] = "All tiers passed: Complete geometric invariance established"
                    results['summary']['max_tier_passed'] = 3
                else:
                    results['summary']['conclusion'] = "Failed at Tier 3: Invariance hierarchy not established"
                    results['summary']['max_tier_passed'] = 2
        
        # Always run control tests
        logger.info("Running control hypothesis tests")
        results['controls'] = self.test_controls(data.get('control_data', {}))
        
        # Generate summary statistics
        all_hypotheses = []
        for tier in results['tiers']:
            all_hypotheses.extend(tier.hypotheses)
        all_hypotheses.extend(results['controls'])
        
        results['summary']['total_hypotheses'] = len(all_hypotheses)
        results['summary']['passed_hypotheses'] = sum(1 for h in all_hypotheses if h.passed)
        results['summary']['mean_effect_size'] = np.mean([h.effect_size for h in all_hypotheses 
                                                         if not np.isnan(h.effect_size)])
        results['summary']['min_p_value'] = min([h.p_value for h in all_hypotheses 
                                               if not np.isnan(h.p_value)])
        
        # Convert dataclasses to dictionaries for serialization
        results['tiers'] = [self._tier_to_dict(tier) for tier in results['tiers']]
        results['controls'] = [self._hypothesis_to_dict(h) for h in results['controls']]
        
        return results
    
    def _hypothesis_to_dict(self, hypothesis: HypothesisResult) -> Dict:
        """Convert HypothesisResult to dictionary."""
        return {
            'name': hypothesis.name,
            'passed': hypothesis.passed,
            'p_value': hypothesis.p_value,
            'test_statistic': hypothesis.test_statistic,
            'effect_size': hypothesis.effect_size,
            'confidence_interval': hypothesis.confidence_interval,
            'power': hypothesis.power,
            'description': hypothesis.description
        }
    
    def _tier_to_dict(self, tier: TierResult) -> Dict:
        """Convert TierResult to dictionary."""
        return {
            'tier_number': tier.tier_number,
            'tier_name': tier.tier_name,
            'passed': tier.passed,
            'hypotheses': [self._hypothesis_to_dict(h) for h in tier.hypotheses],
            'adjusted_alpha': tier.adjusted_alpha,
            'correction_method': tier.correction_method
        }