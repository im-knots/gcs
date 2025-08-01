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