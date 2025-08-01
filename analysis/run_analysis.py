#!/usr/bin/env python3
"""
Main script to run conversation trajectory analysis.

This script demonstrates the modular architecture and provides
a clean interface for running the complete analysis pipeline.
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional
from tqdm import tqdm
import pandas as pd

from embedding_analysis import (
    EnsembleEmbedder,
    TrajectoryAnalyzer,
    PhaseDetector,
    BreakdownPredictor,
    TrajectoryVisualizer
)
from embedding_analysis.core import ConversationLoader
from embedding_analysis.visualization import ReportGenerator, EnsembleVisualizer
from embedding_analysis.utils import CheckpointManager, setup_logging


class ConversationAnalysisPipeline:
    """
    Main pipeline for conversation analysis with trajectory-based breakdown prediction.
    """
    
    def __init__(self, 
                 output_dir: str = "analysis_output",
                 checkpoint_enabled: bool = True,
                 log_level: str = "INFO",
                 batch_size: int = 25,
                 outcome_csv_paths: Optional[Dict[str, str]] = None):
        """
        Initialize analysis pipeline.
        
        Args:
            output_dir: Directory for outputs
            checkpoint_enabled: Whether to enable checkpointing
            log_level: Logging level
            batch_size: Number of conversations to process in each GPU batch
            outcome_csv_paths: Optional dict mapping tier names to CSV paths
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Set up logging
        self.logger = setup_logging(
            log_level=log_level,
            log_dir=self.output_dir / "logs"
        )
        
        # Initialize components
        self.loader = ConversationLoader()
        self.embedder = EnsembleEmbedder()
        self.trajectory_analyzer = TrajectoryAnalyzer()
        self.phase_detector = PhaseDetector()
        self.breakdown_predictor = BreakdownPredictor()
        self.visualizer = TrajectoryVisualizer(self.output_dir / "figures")
        self.ensemble_visualizer = EnsembleVisualizer(self.output_dir / "figures" / "ensemble")
        self.report_generator = ReportGenerator(self.output_dir / "reports")
        
        # Checkpoint manager
        self.checkpoint_manager = None
        if checkpoint_enabled:
            self.checkpoint_manager = CheckpointManager(self.output_dir / "checkpoints")
            
        # Results storage
        self.tier_results = {}
        self.analysis_results = {}
        
        # Batch size for GPU processing
        self.batch_size = batch_size
        
        # Cache for outcome data
        self.outcome_data_cache = {}
        
        # Store CSV paths if provided, otherwise use defaults
        self.outcome_csv_paths = outcome_csv_paths or {}
        
        # Default CSV paths (highest n directories)
        self.default_csv_paths = {
            'full_reasoning': 'n67/conversation_analysis_enhanced.csv',
            'light_reasoning': 'n61/conversation_analysis_enhanced.csv',
            'non_reasoning': 'n100/conversation_analysis_enhanced.csv'
        }
        
    def _load_outcome_data(self, tier_name: str, tier_dir: Path) -> Optional[Dict[str, str]]:
        """
        Load conversation outcomes from CSV files.
        
        Args:
            tier_name: Name of the tier
            tier_dir: Directory containing tier data
            
        Returns:
            Dictionary mapping filename to conversation outcome, or None if no CSV available
        """
        # Check if we've already cached this data
        cache_key = f"{tier_name}_{tier_dir}"
        if cache_key in self.outcome_data_cache:
            return self.outcome_data_cache[cache_key]
            
        csv_path = None
        
        # Check if user provided a specific CSV path for this tier
        if tier_name in self.outcome_csv_paths:
            csv_path = Path(self.outcome_csv_paths[tier_name])
        else:
            # Use default path
            if tier_name in self.default_csv_paths:
                csv_path = tier_dir / self.default_csv_paths[tier_name]
                
        if not csv_path or not csv_path.exists():
            self.logger.info(f"No outcome CSV available for {tier_name}")
            return None
            
        outcome_map = {}
        
        try:
            df = pd.read_csv(csv_path)
            # Create mapping from filename to conversation_outcome
            for _, row in df.iterrows():
                filename = row['filename']
                outcome = row.get('conversation_outcome', 'unknown')
                outcome_map[filename] = outcome
                
            self.logger.info(f"Loaded {len(outcome_map)} outcomes from {csv_path}")
        except Exception as e:
            self.logger.warning(f"Failed to load outcomes from {csv_path}: {e}")
            return None
                
        # Cache the results
        self.outcome_data_cache[cache_key] = outcome_map
        return outcome_map
        
    def analyze_tier(self, 
                    tier_name: str,
                    tier_dir: Path,
                    max_conversations: Optional[int] = None) -> Dict:
        """
        Analyze conversations from a single tier.
        
        Args:
            tier_name: Name of the tier
            tier_dir: Directory containing conversations
            max_conversations: Maximum conversations to analyze
            
        Returns:
            Tier analysis results
        """
        self.logger.info(f"\nAnalyzing {tier_name} tier...")
        
        # Check for checkpoint
        if self.checkpoint_manager:
            checkpoint_data = self.checkpoint_manager.load_session_checkpoint(
                tier_name, "tier_analysis"
            )
            if checkpoint_data:
                self.logger.info(f"Loaded checkpoint for {tier_name}")
                return checkpoint_data
                
        # Load conversations
        conversations = self.loader.load_conversations_batch(
            tier_dir,
            max_conversations=max_conversations
        )
        
        # Filter conversations
        conversations = self.loader.filter_conversations(
            conversations,
            min_turns=20
        )
        
        self.logger.info(f"Processing {len(conversations)} conversations")
        
        # Load outcome data from CSV files
        outcome_map = self._load_outcome_data(tier_name, tier_dir)
        
        # Add outcome data to conversations if available
        if outcome_map:
            for conv in conversations:
                filename = conv['metadata'].get('filename', '')
                if filename in outcome_map:
                    conv['metadata']['annotated_outcome'] = outcome_map[filename]
        
        tier_results = {
            'tier_name': tier_name,
            'n_conversations': len(conversations),
            'conversations': [],
            'phase_metrics': [],
            'trajectory_metrics': [],
            'breakdown_predictions': []
        }
        
        # Process conversations in batches for GPU efficiency
        batch_size = self.batch_size
        n_batches = (len(conversations) + batch_size - 1) // batch_size
        
        self.logger.info(f"Processing {len(conversations)} conversations in {n_batches} batches of {batch_size}")
        
        for batch_idx in range(n_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, len(conversations))
            batch_conversations = conversations[start_idx:end_idx]
            
            # Collect all messages for batch embedding
            all_messages = []
            message_boundaries = [0]
            
            for conv in batch_conversations:
                messages = conv['messages']
                all_messages.extend(messages)
                message_boundaries.append(message_boundaries[-1] + len(messages))
            
            # Extract text content for embedding
            all_texts = [msg.get('content', '') for msg in all_messages]
            
            # Batch embed all messages at once
            self.logger.info(f"  Batch {batch_idx+1}/{n_batches}: Embedding {len(all_texts)} messages from {len(batch_conversations)} conversations")
            batch_embeddings = self.embedder.embed_texts(all_texts, show_progress=True)
            
            # Process each conversation in the batch
            for conv_idx, conv in enumerate(tqdm(batch_conversations, 
                                                desc=f"Processing batch {batch_idx+1}/{n_batches} ({tier_name})", 
                                                unit="conv")):
                
                session_id = conv['metadata']['session_id']
                
                # Extract embeddings for this conversation
                start = message_boundaries[conv_idx]
                end = message_boundaries[conv_idx + 1]
                
                embeddings = {}
                for model_name, all_emb in batch_embeddings.items():
                    embeddings[model_name] = all_emb[start:end]
                
                # Get number of messages in this conversation
                n_messages = len(conv['messages'])
                
                # Calculate trajectory metrics
                trajectory_metrics = self.trajectory_analyzer.calculate_ensemble_trajectories(embeddings)
                
                # Detect phases
                phase_results = self.phase_detector.detect_phases(embeddings)
                
                # Compare with annotations if available
                phase_comparison = None
                phase_statistics = None
                if 'phases' in conv:
                    phase_comparison = self.phase_detector.compare_with_annotations(
                        phase_results['detected_phases'],
                        conv['phases']
                    )
                    
                    # Run statistical tests
                    from embedding_analysis.utils.statistics import (
                        test_model_phase_agreement, 
                        test_phase_detection_accuracy
                    )
                    
                    # Test model agreement
                    model_agreement_stats = test_model_phase_agreement(
                        phase_results['model_phases'],
                        n_messages
                    )
                    
                    # Test accuracy against annotations
                    accuracy_stats = test_phase_detection_accuracy(
                        conv['phases'],
                        phase_results['model_phases'],
                        n_messages
                    )
                    
                    phase_statistics = {
                        'model_agreement': model_agreement_stats,
                        'detection_accuracy': accuracy_stats
                    }
                    
                # Store results
                conv_results = {
                    'session_id': session_id,
                    'embeddings': embeddings,
                    'trajectory_metrics': trajectory_metrics,
                    'phase_results': phase_results,
                    'phase_comparison': phase_comparison,
                    'phase_statistics': phase_statistics
                }
                
                tier_results['conversations'].append(conv_results)
                tier_results['phase_metrics'].append(phase_results['ensemble_agreement'])
                tier_results['trajectory_metrics'].append(trajectory_metrics)
                
                # Store for breakdown prediction
                conv['ensemble_embeddings'] = embeddings
                conv['trajectory_metrics'] = trajectory_metrics
                conv['phase_info'] = phase_results
                
                # Add tier information to conversation
                conv['tier'] = tier_name
                
                # Generate comprehensive ensemble visualization
                # The conversation dict already contains 'phases' if annotated phases exist
                self.ensemble_visualizer.create_comprehensive_ensemble_plot(
                    conv,
                    embeddings,
                    phase_results,  # This contains detected phases
                    save_path=self.output_dir / "figures" / "ensemble" / f"{tier_name}_{session_id}_ensemble.png"
                )
            
            # Clear GPU cache after each batch
            import torch
            torch.cuda.empty_cache()
            
        # Calculate tier statistics
        tier_results['summary'] = self._calculate_tier_summary(tier_results)
        
        # Save checkpoint
        if self.checkpoint_manager:
            self.checkpoint_manager.create_session_checkpoint(
                tier_name, "tier_analysis", tier_results
            )
            
        return tier_results
        
    def _calculate_tier_summary(self, tier_results: Dict) -> Dict:
        """Calculate summary statistics for a tier."""
        summary = {}
        
        # Phase detection statistics
        if tier_results['phase_metrics']:
            agreements = [m.get('mean_correlation', 0) for m in tier_results['phase_metrics']]
            summary['avg_ensemble_agreement'] = sum(agreements) / len(agreements)
            
        # Phase comparison statistics
        phase_comparisons = [c['phase_comparison'] for c in tier_results['conversations'] 
                           if c['phase_comparison'] is not None]
        
        if phase_comparisons:
            precisions = [pc['metrics']['precision'] for pc in phase_comparisons]
            recalls = [pc['metrics']['recall'] for pc in phase_comparisons]
            f1s = [pc['metrics']['f1'] for pc in phase_comparisons]
            
            summary['phase_detection'] = {
                'avg_precision': sum(precisions) / len(precisions),
                'avg_recall': sum(recalls) / len(recalls),
                'avg_f1': sum(f1s) / len(f1s)
            }
            
        # Phase statistical tests
        phase_stats = [c['phase_statistics'] for c in tier_results['conversations']
                      if c.get('phase_statistics') is not None]
        
        if phase_stats:
            # Aggregate model agreement stats
            fleiss_kappas = [ps['model_agreement']['fleiss_kappa'] for ps in phase_stats]
            kendall_ws = [ps['model_agreement']['kendall_w'] for ps in phase_stats]
            
            summary['phase_model_agreement'] = {
                'avg_fleiss_kappa': sum(fleiss_kappas) / len(fleiss_kappas),
                'avg_kendall_w': sum(kendall_ws) / len(kendall_ws),
                'interpretation': phase_stats[0]['model_agreement']['interpretation']['agreement_level']
            }
            
            # Aggregate detection accuracy stats
            model_precisions = {}
            model_recalls = {}
            for ps in phase_stats:
                for model, stats in ps['detection_accuracy']['per_model'].items():
                    if model not in model_precisions:
                        model_precisions[model] = []
                        model_recalls[model] = []
                    model_precisions[model].append(stats['precision'])
                    model_recalls[model].append(stats['recall'])
            
            summary['phase_detection_by_model'] = {
                model: {
                    'avg_precision': sum(precs) / len(precs),
                    'avg_recall': sum(recs) / len(recs)
                }
                for model, precs in model_precisions.items()
                for recs in [model_recalls[model]]
            }
            
        # Trajectory statistics
        if tier_results['trajectory_metrics']:
            # Extract key metrics across conversations
            velocity_means = []
            curvature_means = []
            
            for tm in tier_results['trajectory_metrics']:
                for model_name, metrics in tm.items():
                    if model_name != 'consistency' and 'velocity_mean' in metrics:
                        velocity_means.append(metrics['velocity_mean'])
                    if model_name != 'consistency' and 'curvature_mean' in metrics:
                        curvature_means.append(metrics['curvature_mean'])
                        
            if velocity_means:
                summary['avg_velocity'] = sum(velocity_means) / len(velocity_means)
            if curvature_means:
                summary['avg_curvature'] = sum(curvature_means) / len(curvature_means)
                
        return summary
        
    def train_breakdown_model(self, tier_results: Dict):
        """Train breakdown prediction model on tier data."""
        self.logger.info("\nTraining breakdown prediction model...")
        
        # Prepare training data
        training_data = []
        
        for tier_name, tier_data in tier_results.items():
            for conv in tier_data['conversations']:
                # Extract features
                features = self.breakdown_predictor.extract_features(
                    conv['embeddings'],
                    conv['trajectory_metrics'],
                    conv['phase_results']
                )
                
                # Get label (using synthetic labels for demo)
                # In real usage, these would come from annotations
                label = self._get_synthetic_label(conv)
                
                training_data.append((features, label))
                
        # Train model
        if training_data:
            metrics = self.breakdown_predictor.train(training_data)
            self.logger.info(f"Model trained with {len(training_data)} samples")
            return metrics
        else:
            self.logger.warning("No training data available")
            return None
            
    def _get_synthetic_label(self, conv_results: Dict) -> bool:
        """Generate synthetic breakdown label for demonstration."""
        # Simple heuristic based on trajectory metrics
        velocity_variance = 0
        
        for model_name, metrics in conv_results['trajectory_metrics'].items():
            if model_name != 'consistency' and 'velocity_std' in metrics:
                velocity_variance += metrics['velocity_std']
                
        # High variance suggests breakdown
        return velocity_variance > 0.5
        
    def generate_visualizations(self, tier_results: Dict):
        """Generate all visualizations."""
        self.logger.info("\nGenerating visualizations...")
        
        # Select representative conversations
        for tier_name, tier_data in tier_results.items():
            if tier_data['conversations']:
                # First conversation from each tier
                conv = tier_data['conversations'][0]
                
                # Ensemble trajectory plot
                self.visualizer.plot_ensemble_trajectories(
                    conv['embeddings'],
                    conv['phase_results']['detected_phases'],
                    title=f"{tier_name} - Ensemble Trajectories",
                    save_path=self.output_dir / "figures" / f"{tier_name}_trajectories.png"
                )
                
        # Breakdown predictions (if model trained)
        if hasattr(self.breakdown_predictor, 'is_trained') and self.breakdown_predictor.is_trained:
            # Collect predictions
            all_predictions = {}
            
            for tier_name, tier_data in tier_results.items():
                for conv in tier_data['conversations'][:5]:  # First 5 from each tier
                    features = self.breakdown_predictor.extract_features(
                        conv['embeddings'],
                        conv['trajectory_metrics'],
                        conv['phase_results']
                    )
                    
                    pred = self.breakdown_predictor.predict(features)
                    
                    # Multi-horizon predictions
                    lookahead = []
                    for n in range(5, 31, 5):
                        lookahead_pred = self.breakdown_predictor.predict(features)
                        lookahead_pred['n_turns_ahead'] = n
                        lookahead.append(lookahead_pred)
                        
                    all_predictions[conv['session_id']] = {
                        'tier': tier_name,
                        'immediate': pred,
                        'lookahead': lookahead
                    }
                    
            # Plot predictions
            self.visualizer.plot_breakdown_predictions(
                all_predictions,
                save_path=self.output_dir / "figures" / "breakdown_predictions.png"
            )
            
            # Feature importance
            feature_importance = self.breakdown_predictor.get_feature_importance()
            self.visualizer.plot_feature_importance(
                feature_importance,
                save_path=self.output_dir / "figures" / "feature_importance.png"
            )
            
    def generate_reports(self, tier_results: Dict, model_metrics: Optional[Dict] = None):
        """Generate analysis reports."""
        self.logger.info("\nGenerating reports...")
        
        # Prepare analysis results
        analysis_results = {
            'ensemble_info': self.embedder.get_model_info(),
            'tier_results': {}
        }
        
        # Summarize tier results
        for tier_name, tier_data in tier_results.items():
            analysis_results['tier_results'][tier_name] = {
                'n_conversations': tier_data['n_conversations'],
                'summary': tier_data.get('summary', {})
            }
            
        # Add model metrics
        if model_metrics:
            analysis_results['breakdown_model_metrics'] = model_metrics
            
        # Generate summary report
        self.report_generator.generate_summary_report(
            analysis_results,
            save_path=self.output_dir / "reports" / "summary_report.txt"
        )
        
        # Generate tier comparison report
        self.report_generator.generate_tier_comparison_report(
            analysis_results['tier_results'],
            save_path=self.output_dir / "reports" / "tier_comparison.txt"
        )
        
    def run_analysis(self, tier_directories: Dict[str, Path], max_conversations: Optional[int] = None):
        """
        Run complete analysis pipeline.
        
        Args:
            tier_directories: Dictionary mapping tier names to directories
            max_conversations: Maximum conversations per tier
        """
        self.logger.info("="*70)
        self.logger.info("CONVERSATION TRAJECTORY ANALYSIS")
        self.logger.info("="*70)
        
        # Analyze each tier
        tier_results = {}
        for tier_name, tier_dir in tier_directories.items():
            if tier_dir.exists():
                tier_results[tier_name] = self.analyze_tier(
                    tier_name, tier_dir, max_conversations
                )
            else:
                self.logger.warning(f"Tier directory not found: {tier_dir}")
                
        # Train breakdown model
        model_metrics = self.train_breakdown_model(tier_results)
        
        # Generate visualizations
        self.generate_visualizations(tier_results)
        
        # Generate reports
        self.generate_reports(tier_results, model_metrics)
        
        self.logger.info("\n" + "="*70)
        self.logger.info("ANALYSIS COMPLETE!")
        self.logger.info(f"Results saved to: {self.output_dir}")
        self.logger.info("="*70)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run conversation trajectory analysis"
    )
    
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("/home/knots/git/the-academy/docs/paper/exp-data"),
        help="Base directory containing tier data"
    )
    
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("analysis_output"),
        help="Output directory for results"
    )
    
    parser.add_argument(
        "--max-conversations",
        type=int,
        default=None,
        help="Maximum conversations per tier (for testing)"
    )
    
    parser.add_argument(
        "--no-checkpoint",
        action="store_true",
        help="Disable checkpointing"
    )
    
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level"
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=25,
        help="Number of conversations to process in each GPU batch (default: 25)"
    )
    
    parser.add_argument(
        "--phase1-csv",
        type=str,
        default=None,
        help="Path to CSV file with outcomes for phase-1-premium (default: phase-1-premium/n67/conversation_analysis_enhanced.csv)"
    )
    
    parser.add_argument(
        "--phase2-csv",
        type=str,
        default=None,
        help="Path to CSV file with outcomes for phase-2-efficient (default: phase-2-efficient/n61/conversation_analysis_enhanced.csv)"
    )
    
    parser.add_argument(
        "--phase3-csv",
        type=str,
        default=None,
        help="Path to CSV file with outcomes for phase-3-no-reasoning (default: phase-3-no-reasoning/n100/conversation_analysis_enhanced.csv)"
    )
    
    args = parser.parse_args()
    
    # Define tier directories
    tier_directories = {
        'full_reasoning': args.data_dir / 'phase-1-premium',
        'light_reasoning': args.data_dir / 'phase-2-efficient',
        'non_reasoning': args.data_dir / 'phase-3-no-reasoning'
    }
    
    # Build outcome CSV paths if provided
    outcome_csv_paths = {}
    if args.phase1_csv:
        outcome_csv_paths['full_reasoning'] = args.phase1_csv
    if args.phase2_csv:
        outcome_csv_paths['light_reasoning'] = args.phase2_csv
    if args.phase3_csv:
        outcome_csv_paths['non_reasoning'] = args.phase3_csv
    
    # Initialize pipeline
    pipeline = ConversationAnalysisPipeline(
        output_dir=str(args.output_dir),
        checkpoint_enabled=not args.no_checkpoint,
        log_level=args.log_level,
        batch_size=args.batch_size,
        outcome_csv_paths=outcome_csv_paths if outcome_csv_paths else None
    )
    
    # Run analysis
    pipeline.run_analysis(
        tier_directories,
        max_conversations=args.max_conversations
    )


if __name__ == "__main__":
    main()