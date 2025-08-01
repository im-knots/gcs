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
import numpy as np
import matplotlib.pyplot as plt

from embedding_analysis import (
    EnsembleEmbedder,
    TrajectoryAnalyzer,
    TrajectoryVisualizer
)
from embedding_analysis.core import ConversationLoader
from embedding_analysis.core.hypothesis_testing import GeometricInvarianceHypothesisTester
from embedding_analysis.core.geometric_invariance import (
    GeometricSignatureComputer,
    InvarianceAnalyzer,
    HypothesisTester,
    NullModelComparator
)
from embedding_analysis.models.ensemble_phase_detector import EnsemblePhaseDetector
from embedding_analysis.visualization import ReportGenerator, EnsembleVisualizer
from embedding_analysis.visualization.density_evolution import DensityEvolutionVisualizer
from embedding_analysis.utils import CheckpointManager, setup_logging


class ConversationAnalysisPipeline:
    """
    Main pipeline for conversation analysis focused on geometric invariance.
    """
    
    def __init__(self, 
                 output_dir: str = "analysis_output",
                 checkpoint_enabled: bool = True,
                 log_level: str = "INFO",
                 batch_size: int = 25,
                 figure_format: str = "both"):
        """
        Initialize analysis pipeline.
        
        Args:
            output_dir: Directory for outputs
            checkpoint_enabled: Whether to enable checkpointing
            log_level: Logging level
            batch_size: Number of conversations to process in each GPU batch
            figure_format: Format for saving figures ('png', 'pdf', or 'both')
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
        self.ensemble_phase_detector = EnsemblePhaseDetector()
        self.hypothesis_tester = GeometricInvarianceHypothesisTester()
        # Breakdown predictor removed - not needed for invariance analysis
        self.visualizer = TrajectoryVisualizer(self.output_dir / "figures")
        self.ensemble_visualizer = EnsembleVisualizer(self.output_dir / "figures" / "ensemble")
        self.density_visualizer = DensityEvolutionVisualizer()
        self.report_generator = ReportGenerator(self.output_dir / "reports")
        
        # Checkpoint manager
        self.checkpoint_manager = None
        if checkpoint_enabled:
            self.checkpoint_manager = CheckpointManager(self.output_dir / "checkpoints")
            
        # Initialize geometric invariance components
        self.signature_computer = GeometricSignatureComputer()
        self.invariance_analyzer = InvarianceAnalyzer()
        self.hypothesis_tester_new = HypothesisTester()
        self.null_comparator = NullModelComparator()
        
        # Results storage
        self.all_conversations = []
        self.invariance_results = {}
        self.analysis_results = {}
        
        # Batch size for GPU processing
        self.batch_size = batch_size
        
        # Figure format preference
        self.figure_format = figure_format
        
    def load_all_conversations(self, directories: List[Path], max_conversations: Optional[int] = None) -> List[Dict]:
        """
        Load all conversations from provided directories without tier categorization.
        
        Args:
            directories: List of directories to load conversations from
            max_conversations: Maximum total conversations to load
            
        Returns:
            List of conversation dictionaries
        """
        all_conversations = []
        
        for directory in directories:
            if not directory.exists():
                self.logger.warning(f"Directory not found: {directory}")
                continue
                
            self.logger.info(f"Loading conversations from {directory}")
            
            # Load conversations from this directory
            conversations = self.loader.load_conversations_batch(
                directory,
                max_conversations=max_conversations
            )
            
            # Filter conversations
            conversations = self.loader.filter_conversations(
                conversations,
                min_turns=20
            )
            
            all_conversations.extend(conversations)
            
            # Check if we've reached the limit
            if max_conversations and len(all_conversations) >= max_conversations:
                all_conversations = all_conversations[:max_conversations]
                break
                
        self.logger.info(f"Loaded {len(all_conversations)} conversations total")
        return all_conversations
        
    def analyze_conversations_for_invariance(self, conversations: List[Dict]) -> Dict:
        """
        Analyze conversations for geometric invariance across embedding models.
        
        Args:
            conversations: List of conversation dictionaries
            
        Returns:
            Invariance analysis results
        """
        self.logger.info(f"\nAnalyzing {len(conversations)} conversations for geometric invariance...")
        
        # Initialize results storage
        invariance_results = {
            'n_conversations': len(conversations),
            'conversation_results': [],
            'geometric_signatures': {},
            'invariance_scores': {},
            'aggregate_statistics': None
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
                                                desc=f"Processing batch {batch_idx+1}/{n_batches}", 
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
                
                # Calculate trajectory metrics with advanced methods
                trajectory_metrics = self.trajectory_analyzer.calculate_ensemble_trajectories(embeddings)
                
                # Add advanced trajectory analysis
                advanced_metrics = {}
                for model_name, emb in embeddings.items():
                    advanced_metrics[model_name] = self.trajectory_analyzer.analyze_trajectory_with_normalization(
                        emb, method='adaptive'
                    )
                trajectory_metrics['advanced'] = advanced_metrics
                
                # Calculate curvature with ensemble methods
                curvature_ensemble = {}
                for model_name, emb in embeddings.items():
                    curvature_ensemble[model_name] = self.trajectory_analyzer.calculate_curvature_ensemble(emb)
                trajectory_metrics['curvature_ensemble'] = curvature_ensemble
                
                # Detect phases using ensemble methods for each model separately
                annotated_phases = conv.get('phases', None)
                model_phase_results = {}
                
                # Apply ensemble phase detection to each model's embeddings
                for model_name, model_embeddings in embeddings.items():
                    # Create single-model dict for the detector
                    single_model_dict = {model_name: model_embeddings}
                    model_results = self.ensemble_phase_detector.detect_phases_ensemble(
                        single_model_dict, annotated_phases
                    )
                    model_phase_results[model_name] = model_results
                
                # Combine results across models
                phase_results = {
                    'model_phases': {},  # Phase detections for each model
                    'method_results': {},  # Combined method results across models
                    'ensemble_phases': [],  # Overall ensemble phases
                    'variance_by_model': {}  # Variance in phase detection across models
                }
                
                # Extract phases for each model
                for model_name, model_result in model_phase_results.items():
                    phase_results['model_phases'][model_name] = model_result.get('ensemble_phases', [])
                    
                    # Collect method results
                    for method_name, method_phases in model_result.get('method_results', {}).items():
                        if method_name not in phase_results['method_results']:
                            phase_results['method_results'][method_name] = {}
                        phase_results['method_results'][method_name][model_name] = method_phases
                
                # Calculate ensemble phases across all models
                all_model_phases = []
                for model_phases in phase_results['model_phases'].values():
                    all_model_phases.extend(model_phases)
                
                # Group similar phases and calculate variance
                if all_model_phases:
                    phase_results['ensemble_phases'] = self._combine_cross_model_phases(all_model_phases, n_messages)
                    phase_results['variance_by_model'] = self._calculate_phase_variance(phase_results['model_phases'], n_messages)
                
                # Compare with annotations if available
                phase_comparison = None
                phase_statistics = None
                if 'phases' in conv:
                    # Compare with annotations - use ensemble detected phases
                    detected_phases = phase_results.get('ensemble_phases', [])
                    phase_comparison = self._compare_phases_with_annotations(
                        detected_phases,
                        conv['phases']
                    )
                    
                    # Skip statistical tests for now - would need to refactor for ensemble detector
                    phase_statistics = None
                    
                # Compute geometric signatures for this conversation
                signatures_by_model = {}
                for model_name, emb in embeddings.items():
                    signatures = self.signature_computer.compute_all_signatures(
                        emb, f"{session_id}_{model_name}"
                    )
                    signatures_by_model[model_name] = signatures
                
                # Analyze invariance across models
                invariance_metrics = self.invariance_analyzer.compute_invariance_metrics(
                    signatures_by_model, session_id
                )
                
                # Store results
                conv_results = {
                    'session_id': session_id,
                    'embeddings': embeddings,
                    'trajectory_metrics': trajectory_metrics,
                    'phase_results': phase_results,
                    'phase_comparison': phase_comparison,
                    'phase_statistics': phase_statistics,
                    'geometric_signatures': signatures_by_model,
                    'invariance_metrics': invariance_metrics
                }
                
                invariance_results['conversation_results'].append(conv_results)
                invariance_results['geometric_signatures'][session_id] = signatures_by_model
                invariance_results['invariance_scores'][session_id] = invariance_metrics
                
                # Store for visualization
                conv['ensemble_embeddings'] = embeddings
                conv['trajectory_metrics'] = trajectory_metrics
                conv['phase_info'] = phase_results
                
                # Generate comprehensive ensemble visualization
                # The conversation dict already contains 'phases' if annotated phases exist
                base_filename = f"{session_id}_ensemble"
                
                # Determine which formats to save based on user preference
                png_path = None
                pdf_path = None
                
                if self.figure_format in ['png', 'both']:
                    png_dir = self.output_dir / "figures" / "ensemble" / "png"
                    png_dir.mkdir(parents=True, exist_ok=True)
                    png_path = png_dir / f"{base_filename}.png"
                
                if self.figure_format in ['pdf', 'both']:
                    pdf_dir = self.output_dir / "figures" / "ensemble" / "pdf"
                    pdf_dir.mkdir(parents=True, exist_ok=True)
                    pdf_path = pdf_dir / f"{base_filename}.pdf"
                
                # Create the plot with the specified formats
                self.ensemble_visualizer.create_comprehensive_ensemble_plot(
                    conv,
                    embeddings,
                    phase_results,  # This contains detected phases
                    save_path=png_path,
                    save_pdf=pdf_path
                )
                
                # Density evolution is now included in the comprehensive plot
                # No need for separate density plots
            
            # Clear GPU cache after each batch
            import torch
            torch.cuda.empty_cache()
            
        # Calculate aggregate invariance statistics
        invariance_results['aggregate_statistics'] = self.invariance_analyzer.aggregate_invariance_scores(
            invariance_results['invariance_scores']
        )
        
        # Log summary
        self.logger.info(f"Mean invariance score: {invariance_results['aggregate_statistics']['mean_invariance']:.3f}")
        self.logger.info(f"Std invariance score: {invariance_results['aggregate_statistics']['std_invariance']:.3f}")
        
        return invariance_results
        
    def _calculate_invariance_summary(self, invariance_results: Dict) -> Dict:
        """Calculate summary statistics for invariance analysis."""
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
        
            
        
        
        return summary
    
    def _combine_cross_model_phases(self, all_phases: List[Dict], n_messages: int) -> List[Dict]:
        """
        Combine phase detections from multiple models into consensus phases.
        
        Args:
            all_phases: List of all phase detections from all models
            n_messages: Total number of messages
            
        Returns:
            List of consensus phases with confidence scores
        """
        if not all_phases:
            return []
            
        # Group phases by proximity
        phase_groups = []
        window = 10  # turns
        
        for phase in all_phases:
            turn = phase.get('turn', phase.get('start_turn', 0))
            found_group = False
            
            for group in phase_groups:
                if any(abs(turn - p.get('turn', p.get('start_turn', 0))) <= window for p in group):
                    group.append(phase)
                    found_group = True
                    break
                    
            if not found_group:
                phase_groups.append([phase])
                
        # Create consensus phases
        consensus_phases = []
        for group in phase_groups:
            if len(group) >= 2:  # Require at least 2 models to agree
                turns = [p.get('turn', p.get('start_turn', 0)) for p in group]
                consensus_phase = {
                    'turn': int(np.median(turns)),
                    'confidence': len(group) / 5.0,  # Assuming 5 models
                    'std': np.std(turns),
                    'n_models': len(group),
                    'type': 'consensus'
                }
                consensus_phases.append(consensus_phase)
                
        return sorted(consensus_phases, key=lambda x: x['turn'])
    
    def _calculate_phase_variance(self, model_phases: Dict[str, List[Dict]], n_messages: int) -> Dict:
        """
        Calculate variance in phase detection across models.
        
        Args:
            model_phases: Dict mapping model names to their detected phases
            n_messages: Total number of messages
            
        Returns:
            Dictionary with variance statistics
        """
        # Collect all phase transitions
        all_transitions = []
        for phases in model_phases.values():
            transitions = [p.get('turn', p.get('start_turn', 0)) for p in phases]
            all_transitions.extend(transitions)
            
        if not all_transitions:
            return {'mean_std': 0, 'max_std': 0}
            
        # Calculate variance statistics
        unique_transitions = sorted(set(all_transitions))
        variances = []
        
        for trans in unique_transitions:
            # Find all transitions near this one
            window = 10
            nearby = [t for t in all_transitions if abs(t - trans) <= window]
            if len(nearby) > 1:
                variances.append(np.std(nearby))
                
        return {
            'mean_std': np.mean(variances) if variances else 0,
            'max_std': np.max(variances) if variances else 0,
            'by_phase': variances
        }
    
    def _compare_phases_with_annotations(self, detected_phases: List[Dict], 
                                        annotated_phases: List[Dict]) -> Dict:
        """
        Compare detected phases with annotations.
        
        Args:
            detected_phases: List of detected phase dictionaries
            annotated_phases: List of annotated phase dictionaries
            
        Returns:
            Comparison metrics
        """
        # Simple comparison based on phase transition points
        detected_turns = [p.get('turn', p.get('start_turn', 0)) for p in detected_phases]
        annotated_turns = [p.get('turn', p.get('start_turn', 0)) for p in annotated_phases]
        
        # Calculate matches within threshold
        threshold = 5  # turns
        matches = 0
        
        for det_turn in detected_turns:
            for ann_turn in annotated_turns:
                if abs(det_turn - ann_turn) <= threshold:
                    matches += 1
                    break
                    
        precision = matches / len(detected_turns) if detected_turns else 0
        recall = matches / len(annotated_turns) if annotated_turns else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            'metrics': {
                'precision': precision,
                'recall': recall,
                'f1': f1
            },
            'n_detected': len(detected_turns),
            'n_annotated': len(annotated_turns),
            'n_matches': matches
        }
        
    def run_analysis(self, directories: List[Path], max_conversations: Optional[int] = None):
        """
        Run complete geometric invariance analysis pipeline.
        
        Args:
            directories: List of directories containing conversations
            max_conversations: Maximum total conversations to analyze
        """
        self.logger.info("="*70)
        self.logger.info("GEOMETRIC INVARIANCE ANALYSIS")
        self.logger.info("="*70)
        
        # Load all conversations
        conversations = self.load_all_conversations(directories, max_conversations)
        
        if not conversations:
            self.logger.error("No conversations loaded!")
            return
        
        # Analyze conversations for invariance
        invariance_results = self.analyze_conversations_for_invariance(conversations)
        
        # Test geometric invariance hypothesis
        self.logger.info("\nTesting geometric invariance hypothesis...")
        hypothesis_results = self.hypothesis_tester_new.test_invariance_hypothesis(
            invariance_results['aggregate_statistics']
        )
        
        # Generate null models for comparison (optional)
        if len(conversations) > 0:
            self.logger.info("\nGenerating null model comparisons...")
            null_results = self.analyze_null_models(conversations[:5])  # Use first 5 conversations
            hypothesis_results['null_comparison'] = null_results
        
        # Generate invariance-specific visualizations
        self.generate_invariance_visualizations(invariance_results)
        
        # Generate reports
        self.generate_invariance_reports(invariance_results, hypothesis_results)
        
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
        help="Maximum total conversations to analyze (for testing)"
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
        "--directories",
        nargs="+",
        type=Path,
        help="Directories containing conversations to analyze"
    )
    
    parser.add_argument(
        "--figure-format",
        choices=["png", "pdf", "both"],
        default="both",
        help="Format for saving figures: png, pdf, or both (default: both)"
    )
    
    args = parser.parse_args()
    
    # Define directories to analyze
    if args.directories:
        directories = args.directories
    else:
        # Default: analyze all phase directories
        directories = [
            args.data_dir / 'phase-1-premium',
            args.data_dir / 'phase-2-efficient',
            args.data_dir / 'phase-3-no-reasoning'
        ]
    
    # Initialize pipeline
    pipeline = ConversationAnalysisPipeline(
        output_dir=str(args.output_dir),
        checkpoint_enabled=not args.no_checkpoint,
        log_level=args.log_level,
        batch_size=args.batch_size,
        figure_format=args.figure_format
    )
    
    # Run analysis
    pipeline.run_analysis(
        directories,
        max_conversations=args.max_conversations
    )


if __name__ == "__main__":
    main()