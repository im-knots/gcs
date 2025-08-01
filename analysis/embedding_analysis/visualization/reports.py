"""
Report generation functionality for conversation analysis.
"""

import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class ReportGenerator:
    """
    Generates analysis reports for conversation trajectory analysis.
    """
    
    def __init__(self, output_dir: Optional[Path] = None):
        """
        Initialize report generator.
        
        Args:
            output_dir: Directory to save reports
        """
        self.output_dir = Path(output_dir) if output_dir else Path('reports')
        self.output_dir.mkdir(exist_ok=True)
        
    def generate_summary_report(self,
                              analysis_results: Dict,
                              save_path: Optional[str] = None) -> str:
        """
        Generate comprehensive summary report.
        
        Args:
            analysis_results: Dictionary containing all analysis results
            save_path: Path to save report
            
        Returns:
            Report text
        """
        report = []
        report.append("CONVERSATION TRAJECTORY ANALYSIS SUMMARY")
        report.append("=" * 70)
        report.append(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Model information
        if 'ensemble_info' in analysis_results:
            report.append("ENSEMBLE MODELS")
            report.append("-" * 40)
            for model_name, info in analysis_results['ensemble_info'].items():
                report.append(f"  {model_name}:")
                report.append(f"    Model ID: {info['model_id']}")
                report.append(f"    Dimensions: {info['dimension']}")
            report.append("")
            
        # Tier analysis
        if 'tier_results' in analysis_results:
            report.append("TIER ANALYSIS")
            report.append("-" * 40)
            
            for tier_name, tier_data in analysis_results['tier_results'].items():
                n_conv = tier_data.get('n_conversations', 0)
                report.append(f"\n{tier_name}:")
                report.append(f"  Conversations analyzed: {n_conv}")
                
                if 'phase_metrics' in tier_data:
                    metrics = tier_data['phase_metrics']
                    report.append(f"  Average phases detected: {metrics['avg_phases']:.1f}")
                    report.append(f"  Phase detection precision: {metrics['precision']:.2%}")
                    report.append(f"  Phase detection recall: {metrics['recall']:.2%}")
                    
                if 'breakdown_metrics' in tier_data:
                    metrics = tier_data['breakdown_metrics']
                    report.append(f"  Breakdown rate: {metrics['breakdown_rate']:.2%}")
                    report.append(f"  Prediction accuracy: {metrics.get('accuracy', 0):.2%}")
                    
            report.append("")
            
        # Phase detection performance
        if 'phase_detection_summary' in analysis_results:
            report.append("PHASE DETECTION PERFORMANCE")
            report.append("-" * 40)
            summary = analysis_results['phase_detection_summary']
            report.append(f"Overall Precision: {summary['overall_precision']:.2%}")
            report.append(f"Overall Recall: {summary['overall_recall']:.2%}")
            report.append(f"Overall F1 Score: {summary['overall_f1']:.2%}")
            report.append(f"Mean distance to annotations: {summary.get('mean_distance', 0):.1f} turns")
            report.append("")
            
        # Breakdown prediction
        if 'breakdown_model_metrics' in analysis_results:
            report.append("BREAKDOWN PREDICTION MODEL")
            report.append("-" * 40)
            metrics = analysis_results['breakdown_model_metrics']
            report.append(f"Training samples: {metrics['n_samples']}")
            report.append(f"Breakdown rate: {metrics['breakdown_rate']:.2%}")
            if metrics.get('roc_auc') is not None:
                report.append(f"ROC-AUC: {metrics['roc_auc']:.3f}")
                report.append(f"PR-AUC: {metrics['pr_auc']:.3f}")
                
            # Top features
            if 'top_features' in metrics:
                report.append("\nTop Predictive Features:")
                for feature, importance in metrics['top_features'][:10]:
                    report.append(f"  - {feature}: {importance:.3f}")
                    
            report.append("")
            
        # Key findings
        report.append("KEY FINDINGS")
        report.append("-" * 40)
        findings = analysis_results.get('key_findings', [
            "Ensemble embeddings successfully capture trajectory patterns",
            "Blind phase detection shows promise for identifying transitions",
            "Trajectory features enable breakdown prediction",
            "Model tier differences are reflected in embedding dynamics"
        ])
        for i, finding in enumerate(findings, 1):
            report.append(f"{i}. {finding}")
            
        report_text = "\n".join(report)
        
        if save_path:
            save_path = Path(save_path)
            save_path.write_text(report_text)
            logger.info(f"Report saved to {save_path}")
            
        return report_text
        
    def generate_tier_comparison_report(self,
                                      tier_results: Dict,
                                      save_path: Optional[str] = None) -> str:
        """
        Generate detailed tier comparison report.
        
        Args:
            tier_results: Dictionary of results by tier
            save_path: Path to save report
            
        Returns:
            Report text
        """
        report = []
        report.append("TIER COMPARISON ANALYSIS")
        report.append("=" * 70)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Summary table
        report.append("SUMMARY METRICS")
        report.append("-" * 40)
        report.append(f"{'Tier':<20} {'Conv':<8} {'Phases':<10} {'Breakdown':<12} {'Agreement':<10}")
        report.append("-" * 60)
        
        for tier_name, data in tier_results.items():
            n_conv = data.get('n_conversations', 0)
            avg_phases = data.get('avg_phases_detected', 0)
            breakdown_rate = data.get('breakdown_rate', 0)
            ensemble_agreement = data.get('avg_ensemble_agreement', 0)
            
            report.append(
                f"{tier_name:<20} {n_conv:<8} {avg_phases:<10.1f} "
                f"{breakdown_rate:<12.2%} {ensemble_agreement:<10.3f}"
            )
            
        report.append("")
        
        # Detailed comparisons
        report.append("DETAILED ANALYSIS")
        report.append("-" * 40)
        
        # Trajectory metrics comparison
        report.append("\nTrajectory Metrics:")
        metrics_to_compare = ['velocity_mean', 'curvature_mean', 'efficiency']
        
        for metric in metrics_to_compare:
            report.append(f"\n  {metric}:")
            for tier_name, data in tier_results.items():
                if 'trajectory_stats' in data and metric in data['trajectory_stats']:
                    stats = data['trajectory_stats'][metric]
                    report.append(
                        f"    {tier_name}: mean={stats['mean']:.3f}, "
                        f"std={stats['std']:.3f}"
                    )
                    
        # Phase detection comparison
        report.append("\n\nPhase Detection Performance:")
        for tier_name, data in tier_results.items():
            if 'phase_detection_metrics' in data:
                metrics = data['phase_detection_metrics']
                report.append(
                    f"  {tier_name}: precision={metrics['precision']:.2%}, "
                    f"recall={metrics['recall']:.2%}, f1={metrics['f1']:.2%}"
                )
                
        # Statistical tests
        if 'statistical_tests' in tier_results:
            report.append("\n\nStatistical Comparisons:")
            tests = tier_results['statistical_tests']
            
            for test_name, results in tests.items():
                report.append(f"\n  {test_name}:")
                report.append(f"    Statistic: {results['statistic']:.3f}")
                report.append(f"    p-value: {results['p_value']:.4f}")
                report.append(f"    Significant: {'Yes' if results['p_value'] < 0.05 else 'No'}")
                
        report_text = "\n".join(report)
        
        if save_path:
            save_path = Path(save_path)
            save_path.write_text(report_text)
            logger.info(f"Tier comparison report saved to {save_path}")
            
        return report_text
        
    def generate_conversation_report(self,
                                   conversation: Dict,
                                   analysis_results: Dict,
                                   save_path: Optional[str] = None) -> str:
        """
        Generate detailed report for a single conversation.
        
        Args:
            conversation: Conversation data
            analysis_results: Analysis results for this conversation
            save_path: Path to save report
            
        Returns:
            Report text
        """
        report = []
        
        # Header
        session_id = conversation['metadata']['session_id']
        report.append(f"CONVERSATION ANALYSIS: {session_id}")
        report.append("=" * 70)
        report.append(f"Messages: {conversation['metadata']['n_messages']}")
        
        if 'outcome' in conversation['metadata']:
            report.append(f"Outcome: {conversation['metadata']['outcome']}")
            
        report.append("")
        
        # Trajectory metrics
        if 'trajectory_metrics' in analysis_results:
            report.append("TRAJECTORY METRICS")
            report.append("-" * 40)
            
            # Average across models
            avg_metrics = {}
            for model_name, metrics in analysis_results['trajectory_metrics'].items():
                if model_name != 'consistency':
                    for metric, value in metrics.items():
                        if isinstance(value, (int, float)):
                            if metric not in avg_metrics:
                                avg_metrics[metric] = []
                            avg_metrics[metric].append(value)
                            
            for metric, values in avg_metrics.items():
                report.append(f"{metric}: {np.mean(values):.3f}")
                
            report.append("")
            
        # Phase detection
        if 'phases' in analysis_results:
            report.append("DETECTED PHASES")
            report.append("-" * 40)
            
            phases = analysis_results['phases']['detected_phases']
            report.append(f"Total phases: {len(phases)}")
            
            for i, phase in enumerate(phases):
                report.append(
                    f"\nPhase {i+1}:"
                    f"\n  Turn: {phase['turn']}"
                    f"\n  Type: {phase.get('type', 'unknown')}"
                    f"\n  Confidence: {phase['confidence']:.2f}"
                    f"\n  Supporting models: {len(phase.get('support_models', []))}"
                )
                
            report.append("")
            
        # Breakdown prediction
        if 'breakdown_prediction' in analysis_results:
            report.append("BREAKDOWN PREDICTION")
            report.append("-" * 40)
            
            pred = analysis_results['breakdown_prediction']
            report.append(f"Probability: {pred['probability']:.2%}")
            report.append(f"Confidence: {pred['confidence']:.2%}")
            
            if 'contributing_factors' in pred:
                report.append("\nTop contributing factors:")
                for factor in pred['contributing_factors']:
                    report.append(
                        f"  - {factor['feature']}: {factor['value']:.3f} "
                        f"(importance: {factor['importance']:.3f})"
                    )
                    
        report_text = "\n".join(report)
        
        if save_path:
            save_path = Path(save_path)
            save_path.write_text(report_text)
            logger.info(f"Conversation report saved to {save_path}")
            
        return report_text
        
    def generate_hypothesis_test_report(self,
                                      hypothesis_results: Dict,
                                      save_path: Optional[str] = None) -> str:
        """
        Generate report for hypothesis test results.
        
        Args:
            hypothesis_results: Results from geometric invariance hypothesis testing
            save_path: Path to save report
            
        Returns:
            Report text
        """
        report = []
        report.append("GEOMETRIC INVARIANCE HYPOTHESIS TEST RESULTS")
        report.append("=" * 70)
        report.append(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Main hypothesis result
        if 'main_hypothesis' in hypothesis_results:
            main = hypothesis_results['main_hypothesis']
            report.append("MAIN HYPOTHESIS")
            report.append("-" * 40)
            report.append("H0: Conversations do NOT exhibit invariant geometric signatures")
            report.append("H1: Conversations exhibit invariant geometric signatures across models")
            report.append("")
            
            if main.get('hypothesis_supported'):
                report.append("✓ HYPOTHESIS SUPPORTED")
            else:
                report.append("✗ HYPOTHESIS NOT SUPPORTED")
                
            report.append("")
            report.append("Evidence Summary:")
            evidence = main.get('evidence', {})
            for key, value in evidence.items():
                report.append(f"  - {key.replace('_', ' ').title()}: {'✓' if value else '✗'}")
                
        # Geometric properties
        if 'geometric_properties' in hypothesis_results:
            report.append("\n\nGEOMETRIC PROPERTIES ANALYSIS")
            report.append("-" * 40)
            
            props = hypothesis_results['geometric_properties']
            
            # Distance matrices
            if 'distance_matrices' in props:
                dm = props['distance_matrices']
                report.append("\nDistance Matrix Correlations:")
                report.append(f"  Mean correlation: {dm['mean_correlation']:.3f}")
                report.append(f"  Min correlation: {dm['min_correlation']:.3f}")
                report.append(f"  Passes threshold (ρ > 0.8): {'Yes' if dm['passes_threshold'] else 'No'}")
                report.append(f"  Statistical test p-value: {dm['hypothesis_test']['p_value']:.4f}")
                
            # Curvature patterns
            if 'curvature_patterns' in props:
                cp = props['curvature_patterns']
                report.append("\nCurvature Pattern Consistency:")
                report.append(f"  Coefficient of variation: {cp['coefficient_of_variation']:.3f}")
                report.append(f"  Consistent across models: {'Yes' if cp['consistency_test']['consistent'] else 'No'}")
                
            # Phase transitions
            if 'phase_transitions' in props:
                pt = props['phase_transitions']
                report.append("\nPhase Transition Alignment:")
                report.append(f"  Mean agreement: {pt['mean_agreement']:.3f}")
                sig_test = pt.get('significance_test', {})
                if 'p_value' in sig_test:
                    report.append(f"  Statistical significance: p={sig_test['p_value']:.4f}")
                    
            # Velocity profiles
            if 'velocity_profiles' in props:
                vp = props['velocity_profiles']
                report.append("\nVelocity Profile Correlations:")
                report.append(f"  Mean correlation: {vp['mean_correlation']:.3f}")
                sig_test = vp.get('significance_test', {})
                if 'p_value' in sig_test:
                    report.append(f"  Statistical significance: p={sig_test['p_value']:.4f}")
                    
        # Control tests
        if 'controls' in hypothesis_results:
            report.append("\n\nCONTROL TESTS")
            report.append("-" * 40)
            
            controls = hypothesis_results['controls']
            
            # Non-transformer comparison
            if 'non_transformer' in controls:
                nt = controls['non_transformer']
                report.append("\nTransformer vs Non-Transformer Embeddings:")
                report.append(f"  Within-transformer correlation: {nt['within_transformer']:.3f}")
                report.append(f"  Transformer-to-control correlation: {nt['transformer_to_control']:.3f}")
                report.append(f"  Difference: {nt['difference']:.3f}")
                
            # Scrambled comparison
            if 'scrambled' in controls:
                sc = controls['scrambled']
                report.append("\nReal vs Scrambled Conversations:")
                report.append(f"  Real conversation correlation: {sc['real_correlation']:.3f}")
                report.append(f"  Scrambled correlation: {sc['scrambled_correlation']:.3f}")
                report.append(f"  Effect size (Cohen's d): {sc['effect_size']:.3f}")
                
        # Multiple testing correction
        if 'corrected' in hypothesis_results:
            corr = hypothesis_results['corrected']
            report.append("\n\nMULTIPLE TESTING CORRECTION")
            report.append("-" * 40)
            report.append(f"Number of tests: {corr.get('n_tests', 0)}")
            report.append(f"Significant after correction: {corr.get('n_significant', 0)}")
            
        # Effect sizes
        if 'effect_sizes' in hypothesis_results:
            report.append("\n\nEFFECT SIZES")
            report.append("-" * 40)
            
            for effect_name, effect_data in hypothesis_results['effect_sizes'].items():
                report.append(f"\n{effect_name.replace('_', ' ').title()}:")
                for key, value in effect_data.items():
                    if isinstance(value, (int, float)):
                        report.append(f"  {key}: {value:.3f}")
                    else:
                        report.append(f"  {key}: {value}")
                        
        # Summary
        if 'summary' in hypothesis_results:
            summary = hypothesis_results['summary']
            report.append("\n\nSUMMARY")
            report.append("-" * 40)
            
            if 'key_findings' in summary:
                report.append("\nKey Findings:")
                for finding in summary['key_findings']:
                    report.append(f"  • {finding}")
                    
            if 'limitations' in summary:
                report.append("\nLimitations:")
                for limitation in summary['limitations']:
                    report.append(f"  • {limitation}")
                    
        report_text = "\n".join(report)
        
        if save_path:
            save_path = Path(save_path)
            save_path.write_text(report_text)
            logger.info(f"Hypothesis test report saved to {save_path}")
            
        return report_text
    
    def generate_invariance_report(self,
                                 analysis_results: Dict,
                                 save_path: Optional[str] = None) -> str:
        """
        Generate report focused on geometric invariance findings.
        
        Args:
            analysis_results: Dictionary containing invariance analysis results
            save_path: Path to save report
            
        Returns:
            Report text
        """
        report = []
        report.append("GEOMETRIC INVARIANCE ANALYSIS REPORT")
        report.append("=" * 70)
        report.append(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Analysis overview
        report.append("ANALYSIS OVERVIEW")
        report.append("-" * 40)
        report.append(f"Analysis Type: {analysis_results.get('analysis_type', 'Geometric Invariance')}")
        report.append(f"Total Conversations Analyzed: {analysis_results.get('n_conversations', 0)}")
        report.append("")
        
        # Model information
        if 'model_info' in analysis_results:
            report.append("EMBEDDING MODELS")
            report.append("-" * 40)
            for model_name, info in analysis_results['model_info'].items():
                report.append(f"  {model_name}:")
                report.append(f"    Dimensions: {info['dimension']}")
            report.append("")
            
        # Invariance statistics
        if 'invariance_statistics' in analysis_results:
            stats = analysis_results['invariance_statistics']
            report.append("GEOMETRIC INVARIANCE STATISTICS")
            report.append("-" * 40)
            report.append(f"Overall Mean Invariance: {stats.get('mean_invariance', 0):.3f}")
            report.append(f"Standard Deviation: {stats.get('std_invariance', 0):.3f}")
            report.append(f"Median Invariance: {stats.get('median_invariance', 0):.3f}")
            report.append("")
            
            # Per-signature statistics
            if 'signature_type_stats' in stats:
                report.append("Invariance by Geometric Signature Type:")
                for sig_type, sig_stats in stats['signature_type_stats'].items():
                    report.append(f"  {sig_type.replace('_', ' ').title()}:")
                    report.append(f"    Mean: {sig_stats.get('mean', 0):.3f}")
                    report.append(f"    Std: {sig_stats.get('std', 0):.3f}")
                    report.append(f"    Range: [{sig_stats.get('min', 0):.3f}, {sig_stats.get('max', 0):.3f}]")
                report.append("")
                
        # Hypothesis test results
        if 'hypothesis_results' in analysis_results:
            hyp = analysis_results['hypothesis_results']
            report.append("HYPOTHESIS TEST RESULTS")
            report.append("-" * 40)
            report.append(f"Invariance Score: {hyp.get('invariance_score', 0):.3f}")
            report.append(f"95% Confidence Interval: {hyp.get('confidence_interval', (0, 0))}")
            report.append(f"p-value: {hyp.get('p_value', 1.0):.6f}")
            report.append(f"Effect Size (Cohen's d): {hyp.get('cohens_d', 0):.3f}")
            report.append("")
            
            if hyp.get('hypothesis_supported', False):
                report.append("✓ HYPOTHESIS SUPPORTED: Strong geometric invariance detected")
                report.append("  Conversations exhibit consistent geometric patterns across embedding models")
            else:
                report.append("✗ HYPOTHESIS NOT SUPPORTED: Weak geometric invariance")
                report.append("  Limited consistency in geometric patterns across models")
            report.append("")
            
            # Null model comparison if available
            if 'null_comparison' in hyp:
                report.append("NULL MODEL COMPARISON")
                report.append("-" * 40)
                for null_type, null_stats in hyp['null_comparison'].items():
                    if isinstance(null_stats, dict) and 'mean_invariance' in null_stats:
                        report.append(f"  {null_type.replace('_', ' ').title()}:")
                        report.append(f"    Mean Invariance: {null_stats.get('mean_invariance', 0):.3f}")
                report.append("")
                
        # Conclusions
        report.append("CONCLUSIONS")
        report.append("-" * 40)
        
        mean_inv = analysis_results.get('invariance_statistics', {}).get('mean_invariance', 0)
        if mean_inv > 0.7:
            report.append("Strong geometric invariance observed across transformer embedding models.")
            report.append("This suggests that conversations have intrinsic geometric properties")
            report.append("that are consistently captured by different embedding approaches.")
        elif mean_inv > 0.5:
            report.append("Moderate geometric invariance observed across embedding models.")
            report.append("Conversations show some consistent geometric patterns, but with")
            report.append("notable variations between different embedding approaches.")
        else:
            report.append("Weak geometric invariance observed across embedding models.")
            report.append("Different embeddings capture substantially different geometric")
            report.append("properties of conversations.")
            
        report.append("")
        report_text = "\n".join(report)
        
        # Save report
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(exist_ok=True)
            with open(save_path, 'w') as f:
                f.write(report_text)
            logger.info(f"Invariance report saved to {save_path}")
            
        return report_text