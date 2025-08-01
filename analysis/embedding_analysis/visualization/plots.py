"""
Visualization functionality for conversation trajectories.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class TrajectoryVisualizer:
    """
    Creates visualizations for conversation trajectories and analysis results.
    """
    
    def __init__(self, output_dir: Optional[Path] = None):
        """
        Initialize visualizer.
        
        Args:
            output_dir: Directory to save plots
        """
        self.output_dir = Path(output_dir) if output_dir else Path('visualizations')
        self.output_dir.mkdir(exist_ok=True)
        
        # Set style without grid
        plt.style.use('seaborn-v0_8-whitegrid')
        sns.set_style("white")  # No grid lines
        sns.set_palette("husl")
        
    def plot_ensemble_trajectories(self,
                                 ensemble_embeddings: Dict[str, np.ndarray],
                                 phases: List[Dict],
                                 title: str = "Ensemble Trajectories",
                                 save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot 2D trajectories from multiple models.
        
        Args:
            ensemble_embeddings: Embeddings from each model
            phases: Detected phase transitions
            title: Plot title
            save_path: Path to save figure
            
        Returns:
            Figure object
        """
        n_models = len(ensemble_embeddings)
        fig, axes = plt.subplots(1, n_models, figsize=(5*n_models, 5))
        
        if n_models == 1:
            axes = [axes]
            
        for idx, (model_name, embeddings) in enumerate(ensemble_embeddings.items()):
            ax = axes[idx]
            
            # PCA projection
            pca = PCA(n_components=2)
            trajectory_2d = pca.fit_transform(embeddings)
            
            # Plot trajectory
            ax.plot(trajectory_2d[:, 0], trajectory_2d[:, 1], 
                   'b-', alpha=0.7, linewidth=1.5)
                   
            # Mark start and end
            ax.scatter(trajectory_2d[0, 0], trajectory_2d[0, 1], 
                      color='green', s=100, marker='o', 
                      label='Start', zorder=5)
            ax.scatter(trajectory_2d[-1, 0], trajectory_2d[-1, 1], 
                      color='red', s=100, marker='s', 
                      label='End', zorder=5)
                      
            # Mark phases
            for phase in phases:
                turn = phase['turn']
                if turn < len(trajectory_2d):
                    ax.scatter(trajectory_2d[turn, 0], trajectory_2d[turn, 1],
                             color='orange', s=150, marker='*', 
                             edgecolor='black', linewidth=1, zorder=6)
                    
            ax.set_title(f'{model_name}')
            ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
            ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
            ax.grid(True, alpha=0.3)
            
            if idx == 0:
                ax.legend()
                
        fig.suptitle(title, fontsize=16)
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig
        
    def plot_breakdown_predictions(self,
                                 predictions: Dict[str, Dict],
                                 save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot breakdown prediction results.
        
        Args:
            predictions: Dictionary of predictions by conversation
            save_path: Path to save figure
            
        Returns:
            Figure object
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Breakdown Prediction Analysis', fontsize=16)
        
        # 1. Lookahead curves
        ax = axes[0, 0]
        for session_id, pred_data in list(predictions.items())[:10]:
            if 'lookahead' in pred_data:
                lookahead = pred_data['lookahead']
                turns = [p['n_turns_ahead'] for p in lookahead]
                probs = [p['probability'] for p in lookahead]
                
                tier = pred_data.get('tier', 'unknown')
                ax.plot(turns, probs, alpha=0.6, label=f"{tier[:4]}-{session_id[:8]}")
                
        ax.set_xlabel('Turns Ahead')
        ax.set_ylabel('Breakdown Probability')
        ax.set_title('Breakdown Probability vs Lookahead Distance')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, ncol=2)
        
        # 2. Confidence heatmap
        ax = axes[0, 1]
        confidence_matrix = []
        session_ids = []
        
        for session_id, pred_data in predictions.items():
            if 'lookahead' in pred_data:
                confs = [p.get('confidence', 0.5) for p in pred_data['lookahead']]
                confidence_matrix.append(confs)
                session_ids.append(session_id[:8])
                
        if confidence_matrix:
            sns.heatmap(
                confidence_matrix[:20],
                ax=ax,
                cmap='viridis',
                cbar_kws={'label': 'Confidence'},
                yticklabels=session_ids[:20]
            )
            ax.set_xlabel('Turns Ahead')
            ax.set_ylabel('Conversation')
            ax.set_title('Prediction Confidence Heatmap')
            
        # 3. Tier comparison
        ax = axes[1, 0]
        tier_probs = {}
        
        for pred_data in predictions.values():
            if 'immediate' in pred_data:
                tier = pred_data.get('tier', 'unknown')
                prob = pred_data['immediate']['probability']
                
                if tier not in tier_probs:
                    tier_probs[tier] = []
                tier_probs[tier].append(prob)
                
        if tier_probs:
            tier_names = list(tier_probs.keys())
            tier_means = [np.mean(tier_probs[t]) for t in tier_names]
            tier_stds = [np.std(tier_probs[t]) for t in tier_names]
            
            x = np.arange(len(tier_names))
            ax.bar(x, tier_means, yerr=tier_stds, capsize=5)
            ax.set_xticks(x)
            ax.set_xticklabels(tier_names)
            ax.set_ylabel('Breakdown Probability')
            ax.set_title('Breakdown Probability by Tier')
            ax.grid(True, alpha=0.3, axis='y')
            
        # 4. Probability distribution
        ax = axes[1, 1]
        all_probs = []
        
        for pred_data in predictions.values():
            if 'immediate' in pred_data:
                all_probs.append(pred_data['immediate']['probability'])
                
        if all_probs:
            ax.hist(all_probs, bins=20, alpha=0.7, edgecolor='black')
            ax.axvline(np.mean(all_probs), color='red', linestyle='--', 
                      label=f'Mean: {np.mean(all_probs):.2f}')
            ax.set_xlabel('Breakdown Probability')
            ax.set_ylabel('Count')
            ax.set_title('Distribution of Breakdown Probabilities')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig
        
    def plot_phase_comparison(self,
                            detected_phases: List[Dict],
                            annotated_phases: List[Dict],
                            n_messages: int,
                            save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot comparison of detected vs annotated phases.
        
        Args:
            detected_phases: List of detected phases
            annotated_phases: List of annotated phases
            n_messages: Total number of messages
            save_path: Path to save figure
            
        Returns:
            Figure object
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Timeline
        turns = np.arange(n_messages)
        
        # Plot detected phases
        for phase in detected_phases:
            turn = phase['turn']
            confidence = phase.get('confidence', 0.5)
            ax.axvline(turn, color='blue', alpha=confidence, 
                      linewidth=2, label='Detected' if phase == detected_phases[0] else "")
                      
        # Plot annotated phases
        for phase in annotated_phases:
            turn = phase['turn']
            ax.axvline(turn, color='red', alpha=0.8, linestyle='--',
                      linewidth=2, label='Annotated' if phase == annotated_phases[0] else "")
                      
        ax.set_xlim(0, n_messages)
        ax.set_xlabel('Turn')
        ax.set_ylabel('Phase Transition')
        ax.set_title('Phase Detection Comparison')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig
        
    def plot_feature_importance(self,
                              feature_importance: List[Tuple[str, float]],
                              top_n: int = 20,
                              save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot feature importance for breakdown prediction.
        
        Args:
            feature_importance: List of (feature_name, importance) tuples
            top_n: Number of top features to show
            save_path: Path to save figure
            
        Returns:
            Figure object
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Get top features
        top_features = feature_importance[:top_n]
        feature_names = [f[0] for f in top_features]
        importances = [f[1] for f in top_features]
        
        # Create horizontal bar plot
        y_pos = np.arange(len(feature_names))
        ax.barh(y_pos, importances, color='skyblue', edgecolor='navy')
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(feature_names, fontsize=10)
        ax.set_xlabel('Importance Score')
        ax.set_title(f'Top {top_n} Features for Breakdown Prediction')
        ax.grid(True, alpha=0.3, axis='x')
        
        # Add value labels
        for i, v in enumerate(importances):
            ax.text(v + 0.01, i, f'{v:.3f}', va='center', fontsize=8)
            
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig
        
    def plot_tier_trajectories(self,
                             tier_data: Dict[str, List[Dict]],
                             metric: str = 'velocity_mean',
                             save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot trajectory metrics across tiers.
        
        Args:
            tier_data: Dictionary mapping tier names to conversations
            metric: Which metric to plot
            save_path: Path to save figure
            
        Returns:
            Figure object
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Collect metric values by tier
        tier_metrics = {}
        
        for tier_name, conversations in tier_data.items():
            metrics = []
            for conv in conversations:
                if 'trajectory_metrics' in conv:
                    # Get average across models
                    model_metrics = []
                    for model_name, model_data in conv['trajectory_metrics'].items():
                        if model_name != 'consistency' and metric in model_data:
                            model_metrics.append(model_data[metric])
                    if model_metrics:
                        metrics.append(np.mean(model_metrics))
                        
            tier_metrics[tier_name] = metrics
            
        # Create violin plot
        if tier_metrics:
            data = []
            labels = []
            for tier, values in tier_metrics.items():
                data.extend(values)
                labels.extend([tier] * len(values))
                
            positions = range(len(tier_metrics))
            parts = ax.violinplot(
                [tier_metrics[tier] for tier in tier_metrics.keys()],
                positions=positions,
                showmeans=True,
                showmedians=True
            )
            
            ax.set_xticks(positions)
            ax.set_xticklabels(list(tier_metrics.keys()))
            ax.set_ylabel(metric)
            ax.set_title(f'{metric} Distribution by Tier')
            ax.grid(True, alpha=0.3, axis='y')
            
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig