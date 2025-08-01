"""
Comprehensive ensemble visualization for conversation analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle, Patch
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sns
from sklearn.decomposition import PCA
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class EnsembleVisualizer:
    """
    Creates comprehensive ensemble visualizations showing distance matrices,
    trajectories, and phase information across all models.
    """
    
    def __init__(self, output_dir: Optional[Path] = None):
        """Initialize ensemble visualizer."""
        self.output_dir = Path(output_dir) if output_dir else Path('ensemble_visualizations')
        self.output_dir.mkdir(exist_ok=True)
        
    def create_comprehensive_ensemble_plot(self,
                                         conversation: Dict,
                                         ensemble_embeddings: Dict[str, np.ndarray],
                                         phase_info: Optional[Dict] = None,
                                         save_path: Optional[str] = None) -> plt.Figure:
        """
        Create comprehensive ensemble visualization with distance matrices,
        self-similarity, recurrence plots, trajectories, and phase information.
        
        Args:
            conversation: Conversation data
            ensemble_embeddings: Embeddings from each model
            phase_info: Phase detection results
            save_path: Path to save figure
            
        Returns:
            Figure object
        """
        model_names = list(ensemble_embeddings.keys())
        n_models = len(model_names)
        n_messages = len(next(iter(ensemble_embeddings.values())))
        
        # Skip very large conversations
        if n_messages > 500:
            logger.warning(f"Skipping visualization for large conversation ({n_messages} messages)")
            return None
            
        # Calculate optimal figure size - increased width to prevent compression
        fig_width = 7 * n_models  # 7 per model
        fig_height = 28  # Slightly increased for better proportions
        logger.info(f"Creating figure ({fig_width}x{fig_height})...")
        fig = plt.figure(figsize=(fig_width, fig_height))
        
        # Create GridSpec for flexible layout
        # Total rows: 4 (section 1) + 1 (correlation tables) + 3 (phase diagram) = 8
        total_rows = 8
        gs = gridspec.GridSpec(total_rows, n_models, figure=fig, hspace=0.35, wspace=0.25)
        
        # Process phase information - separate detected and annotated
        detected_phases = []
        annotated_phases = []
        
        # Get detected phases from blind detection
        if phase_info and 'detected_phases' in phase_info:
            for phase in phase_info['detected_phases']:
                detected_phases.append({
                    'name': phase.get('type', f"Phase {len(detected_phases)+1}"),
                    'start_turn': phase['turn'],
                    'confidence': phase.get('confidence', 0.5),
                    'turn_range': (phase['turn'], phase.get('end_turn', n_messages))
                })
            detected_phases.sort(key=lambda x: x['start_turn'])
            
        # Get annotated phases if available
        if 'phases' in conversation and conversation['phases']:
            for phase in conversation['phases']:
                annotated_phases.append({
                    'name': phase.get('phase', phase.get('name', 'Unknown')),
                    'start_turn': phase['turn'],
                    'turn_range': (phase['turn'], phase.get('end_turn', n_messages))
                })
            annotated_phases.sort(key=lambda x: x['start_turn'])
            
        # Use detected phases for the main visualization
        phase_list = detected_phases
        
        # Standardize all embeddings first
        standardized_embeddings = {}
        for model_name, embeddings in ensemble_embeddings.items():
            standardized_embeddings[model_name] = self._standardize_embeddings(embeddings)
        
        # Calculate distance matrices and self-similarities
        distance_matrices = {}
        self_similarities = {}
        
        for model_name, embeddings in standardized_embeddings.items():
            distance_matrices[model_name] = self._calculate_distance_matrix(embeddings)
            self_similarities[model_name] = self._calculate_self_similarity(embeddings)
            
        # Add row labels on the left side
        row_labels = [
            'Euclidean\nDistance',
            'Self-Similarity\n(Cosine)',
            'Recurrence\nPlot',
            'Embedding\nTrajectory (PCA)'
        ]
        
        # Create visualizations for each model
        for idx, model_name in enumerate(model_names):
            embeddings = standardized_embeddings[model_name]
            
            # Create subplots for this model using GridSpec
            ax1 = fig.add_subplot(gs[0, idx])  # Row 0: Euclidean distances
            ax2 = fig.add_subplot(gs[1, idx])  # Row 1: Self-similarity
            ax3 = fig.add_subplot(gs[2, idx])  # Row 2: Recurrence plots
            ax4 = fig.add_subplot(gs[3, idx])  # Row 3: Trajectory
            
            # Plot euclidean distances
            im1 = ax1.imshow(distance_matrices[model_name], cmap='viridis', aspect='auto')
            ax1.set_title(f'{model_name}', fontsize=12)
            ax1.set_xlabel('Turn', fontsize=10)
            if idx == 0:
                ax1.set_ylabel('Turn', fontsize=10)
                # Add row label
                ax1.text(-0.3, 0.5, row_labels[0], transform=ax1.transAxes, 
                        fontsize=12, fontweight='bold', ha='right', va='center',
                        rotation=0)
            
            # Add colorbar for the last column only
            if idx == n_models - 1:
                divider = make_axes_locatable(ax1)
                cax1 = divider.append_axes("right", size="5%", pad=0.05)
                cbar1 = plt.colorbar(im1, cax=cax1)
                cbar1.ax.tick_params(labelsize=8)
                cbar1.set_label('Distance', fontsize=8)
            
            # Plot self-similarity
            im2 = ax2.imshow(self_similarities[model_name], cmap='RdBu_r', aspect='auto', vmin=0, vmax=1)
            ax2.set_xlabel('Turn', fontsize=10)
            if idx == 0:
                ax2.set_ylabel('Turn', fontsize=10)
                # Add row label
                ax2.text(-0.3, 0.5, row_labels[1], transform=ax2.transAxes,
                        fontsize=12, fontweight='bold', ha='right', va='center',
                        rotation=0)
            
            # Add colorbar for the last column only
            if idx == n_models - 1:
                divider = make_axes_locatable(ax2)
                cax2 = divider.append_axes("right", size="5%", pad=0.05)
                cbar2 = plt.colorbar(im2, cax=cax2)
                cbar2.ax.tick_params(labelsize=8)
                cbar2.set_label('Similarity', fontsize=8)
            
            # Add phase boundaries if available
            if phase_list:
                for phase in phase_list:
                    start_turn = phase['start_turn']
                    if start_turn < n_messages and start_turn > 0:
                        ax1.axhline(y=start_turn, color='red', linestyle='--', alpha=0.6, linewidth=1)
                        ax1.axvline(x=start_turn, color='red', linestyle='--', alpha=0.6, linewidth=1)
                        ax2.axhline(y=start_turn, color='red', linestyle='--', alpha=0.6, linewidth=1)
                        ax2.axvline(x=start_turn, color='red', linestyle='--', alpha=0.6, linewidth=1)
            
            # Create recurrence plot for third row
            sim_flat = self_similarities[model_name].flatten()
            # Use 85th percentile threshold
            model_threshold = np.percentile(sim_flat, 85)
            recurrence_matrix = self_similarities[model_name] > model_threshold
            
            # If still too few points, adjust threshold
            recurrence_rate = np.sum(recurrence_matrix) / (n_messages * n_messages)
            if recurrence_rate < 0.01:  # Less than 1% recurrence is too sparse
                model_threshold = np.percentile(sim_flat, 75)
                recurrence_matrix = self_similarities[model_name] > model_threshold
            
            # Plot recurrence
            im3 = ax3.imshow(recurrence_matrix, cmap='binary', aspect='auto')
            ax3.set_xlabel('Turn', fontsize=10)
            if idx == 0:
                ax3.set_ylabel('Turn', fontsize=10)
                # Add row label
                ax3.text(-0.3, 0.5, row_labels[2], transform=ax3.transAxes,
                        fontsize=12, fontweight='bold', ha='right', va='center',
                        rotation=0)
            
            # Add colorbar for the last column only
            if idx == n_models - 1:
                divider = make_axes_locatable(ax3)
                cax3 = divider.append_axes("right", size="5%", pad=0.05)
                cbar3 = plt.colorbar(im3, cax=cax3)
                cbar3.ax.tick_params(labelsize=8)
                cbar3.set_label('Recurrence', fontsize=8)
            
            # Add phase boundaries
            if phase_list:
                for phase in phase_list:
                    start_turn = phase['start_turn']
                    if start_turn < n_messages and start_turn > 0:
                        ax3.axhline(y=start_turn, color='red', linestyle='--', alpha=0.6, linewidth=1)
                        ax3.axvline(x=start_turn, color='red', linestyle='--', alpha=0.6, linewidth=1)
            
            # Row 4: Add trajectory visualization
            # Reduce to 2D for visualization
            pca = PCA(n_components=2)
            reduced_embeddings = pca.fit_transform(embeddings)
            
            # Create trajectory plot
            ax4.plot(reduced_embeddings[:, 0], reduced_embeddings[:, 1], 'b-', alpha=0.5, linewidth=1)
            
            # Color points by phase if available
            if phase_list:
                sorted_phases = sorted(phase_list, key=lambda x: x['start_turn'])
                distinct_colors = plt.cm.tab20(np.linspace(0, 1, 20))
                
                for phase_idx, phase in enumerate(sorted_phases):
                    start_turn = phase['start_turn']
                    if phase_idx < len(sorted_phases) - 1:
                        end_turn = sorted_phases[phase_idx + 1]['start_turn']
                    else:
                        end_turn = n_messages
                    
                    phase_color = distinct_colors[phase_idx % 20]
                    # Plot points in this phase
                    phase_points = reduced_embeddings[start_turn:end_turn]
                    if len(phase_points) > 0:
                        ax4.scatter(phase_points[:, 0], phase_points[:, 1], 
                                  c=[phase_color], s=20, alpha=0.7, edgecolors='black', linewidth=0.5)
            else:
                # Just show all points colored by time
                ax4.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], 
                          c=range(n_messages), cmap='viridis', s=20, alpha=0.7)
            
            # Mark start and end
            ax4.scatter(reduced_embeddings[0, 0], reduced_embeddings[0, 1], 
                      c='green', s=80, marker='^', edgecolors='black', linewidth=1.5, zorder=5)
            ax4.scatter(reduced_embeddings[-1, 0], reduced_embeddings[-1, 1], 
                      c='red', s=80, marker='v', edgecolors='black', linewidth=1.5, zorder=5)
            
            ax4.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})', fontsize=10)
            ax4.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})', fontsize=10)
            
            if idx == 0:
                # Add row label
                ax4.text(-0.3, 0.5, row_labels[3], transform=ax4.transAxes,
                        fontsize=12, fontweight='bold', ha='right', va='center',
                        rotation=0)
                # Add trajectory explanation
                ax4.text(-0.3, 0.2, '▲ = Start', transform=ax4.transAxes,
                        fontsize=10, ha='right', va='center', color='green')
                ax4.text(-0.3, 0.05, '▼ = End', transform=ax4.transAxes,
                        fontsize=10, ha='right', va='center', color='red')
                if phase_list:
                    ax4.text(-0.3, -0.1, 'Colors = Phases', transform=ax4.transAxes,
                            fontsize=9, ha='right', va='center', color='gray')
            
            # No grid lines - cleaner appearance
            ax4.set_aspect('equal', adjustable='box')
        
        # Section 2: Correlation Tables (Row 4)
        # Center 3 correlation tables with equal spacing
        if n_models >= 3:
            # Create a subgridspec for the correlation row to ensure proper centering
            from matplotlib.gridspec import GridSpecFromSubplotSpec
            corr_gs = GridSpecFromSubplotSpec(1, 3, subplot_spec=gs[4, :], 
                                             wspace=0.3, width_ratios=[1, 1, 1])
            ax_dist_corr = fig.add_subplot(corr_gs[0, 0])
            ax_vel_corr = fig.add_subplot(corr_gs[0, 1])
            ax_topo = fig.add_subplot(corr_gs[0, 2])
        elif n_models == 2:
            # For 2 models, center the available correlation plots
            ax_dist_corr = fig.add_subplot(gs[4, 0])
            ax_vel_corr = fig.add_subplot(gs[4, 1])
            ax_topo = None
        else:
            # For 1 model, just show distance correlation
            ax_dist_corr = fig.add_subplot(gs[4, 0])
            ax_vel_corr = None
            ax_topo = None
        
        # Calculate correlations
        dist_corr_matrix = self._calculate_matrix_correlations(distance_matrices)
        sns.heatmap(dist_corr_matrix, annot=True, fmt='.3f', xticklabels=model_names,
                   yticklabels=model_names, cmap='RdBu_r', center=0.5, ax=ax_dist_corr,
                   vmin=0, vmax=1, cbar_kws={'label': 'Correlation'})
        ax_dist_corr.set_title('Distance Matrix Correlations', fontsize=12)
        
        # Velocity correlations
        if ax_vel_corr is not None:
            vel_corr_matrix = self._calculate_trajectory_correlations(ensemble_embeddings)
            sns.heatmap(vel_corr_matrix, annot=True, fmt='.3f', xticklabels=model_names,
                       yticklabels=model_names, cmap='RdBu_r', center=0.5, ax=ax_vel_corr,
                       vmin=0, vmax=1, cbar_kws={'label': 'Correlation'})
            ax_vel_corr.set_title('Velocity Profile Correlations', fontsize=12)
        
        # Topology preservation
        if ax_topo is not None:
            topo_matrix = self._calculate_topology_preservation(ensemble_embeddings)
            sns.heatmap(topo_matrix, annot=True, fmt='.3f', xticklabels=model_names,
                       yticklabels=model_names, cmap='Greens', vmin=0, vmax=1,
                       cbar_kws={'label': 'Preservation'}, ax=ax_topo)
            ax_topo.set_title('Topology Preservation', fontsize=12)
        
        # Section 3: Phase transition timeline (Rows 5-7)
        # Create a subplot spanning all columns for the phase diagram
        ax_phase = fig.add_subplot(gs[5:8, :])
        
        # Create phase transition timeline
        if detected_phases or annotated_phases:
            # Determine timeline positions
            show_both = len(detected_phases) > 0 and len(annotated_phases) > 0
            annotated_y = 0.7 if show_both else 0.5
            detected_y = 0.3 if show_both else 0.5
            
            # Draw timeline base lines
            ax_phase.axhline(y=annotated_y, color='gray', linewidth=1, alpha=0.5)
            if show_both:
                ax_phase.axhline(y=detected_y, color='gray', linewidth=1, alpha=0.5)
            
            # Draw annotated phase transitions if available
            if annotated_phases:
                sorted_annotated = sorted(annotated_phases, key=lambda x: x['start_turn'])
                
                # Extract unique transitions (only when phase changes)
                unique_transitions = []
                last_phase = None
                
                for phase in sorted_annotated:
                    # Simplify phase name - extract main phase type
                    phase_name = phase['name'].lower()
                    
                    # If it's "x moving to y" or "x transitioning to y", use "y"
                    if ' moving to ' in phase_name:
                        phase_name = phase_name.split(' moving to ')[-1].strip()
                    elif ' transitioning to ' in phase_name:
                        phase_name = phase_name.split(' transitioning to ')[-1].strip()
                    elif ' into ' in phase_name:
                        phase_name = phase_name.split(' into ')[-1].strip()
                    
                    # Extract core phase name (before "/", "through", etc.)
                    if '/' in phase_name:
                        phase_name = phase_name.split('/')[0].strip()
                    if ' through ' in phase_name:
                        phase_name = phase_name.split(' through ')[0].strip()
                    
                    # Remove common prefixes and suffixes
                    phase_name = phase_name.replace('deep ', '').replace(' phase', '').strip()
                    
                    # Only add if it's different from the last phase
                    if phase_name != last_phase and phase_name:
                        unique_transitions.append({
                            'turn': phase['start_turn'],
                            'phase': phase_name,
                            'original': phase['name']
                        })
                        last_phase = phase_name
                
                # Draw vertical lines for each unique transition
                for phase_idx, transition in enumerate(unique_transitions):
                    turn = transition['turn']
                    phase_name = transition['phase'].title()
                    
                    if 0 <= turn <= n_messages:
                        # Draw vertical line
                        ax_phase.axvline(x=turn, ymin=annotated_y-0.15, ymax=annotated_y+0.15, 
                                       color='red', linewidth=2, alpha=0.8)
                        
                        # Add phase name with rotation to avoid overlap
                        ax_phase.text(turn, annotated_y+0.2, phase_name, 
                                    rotation=45, ha='left', va='bottom', fontsize=10,
                                    color='darkred', fontweight='bold')
                        
                        # Add turn number below the line
                        ax_phase.text(turn, annotated_y-0.2, f'T{turn}', 
                                    ha='center', va='top', fontsize=8,
                                    color='red', alpha=0.7)
                
                # Add label for annotated transitions
                ax_phase.text(-0.05, annotated_y, 'Annotated\nTransitions', 
                            transform=ax_phase.transAxes, ha='right', va='center',
                            fontsize=12, fontweight='bold', color='darkred')
            
            # Draw detected phase transitions if available
            if detected_phases:
                sorted_detected = sorted(detected_phases, key=lambda x: x['start_turn'])
                
                # Filter to only show actual transitions (skip turn 0 as it's not a transition)
                actual_transitions = [p for p in sorted_detected if p['start_turn'] > 0]
                
                # Draw vertical lines for each detected transition
                for phase_idx, phase in enumerate(actual_transitions):
                    turn = phase['start_turn']
                    confidence = phase.get('confidence', 0.5)
                    phase_type = phase.get('type', f'Phase {phase_idx+1}')
                    
                    if 0 < turn <= n_messages:  # Skip turn 0
                        # Draw vertical line with confidence-based styling
                        ax_phase.axvline(x=turn, ymin=detected_y-0.15, ymax=detected_y+0.15, 
                                       color='blue', linewidth=2, 
                                       alpha=0.3 + 0.7 * confidence,
                                       linestyle='--' if confidence < 0.7 else '-')
                        
                        # Add phase type and confidence above
                        label_text = f'{phase_type}\n({confidence:.2f})'
                        ax_phase.text(turn, detected_y-0.2, label_text, 
                                    rotation=45, ha='right', va='top', fontsize=9,
                                    color='darkblue', fontweight='bold' if confidence > 0.7 else 'normal',
                                    alpha=0.5 + 0.5 * confidence)
                        
                        # Add turn number below
                        ax_phase.text(turn, detected_y+0.15, f'T{turn}', 
                                    ha='center', va='bottom', fontsize=8,
                                    color='blue', alpha=0.5 + 0.5 * confidence)
                
                # Add label for detected transitions
                ax_phase.text(-0.05, detected_y, 'Detected\nTransitions', 
                            transform=ax_phase.transAxes, ha='right', va='center',
                            fontsize=12, fontweight='bold', color='darkblue')
            
            # Set up the axis
            ax_phase.set_xlim(-5, n_messages + 5)  # Add some padding
            ax_phase.set_ylim(0, 1)
            ax_phase.set_xlabel('Conversation Turn', fontsize=14)
            
            # Update title based on what we're showing
            if show_both:
                ax_phase.set_title('Phase Transitions: Annotated vs Detected', fontsize=16)
            elif detected_phases:
                ax_phase.set_title('Detected Phase Transitions', fontsize=16)
            else:
                ax_phase.set_title('Annotated Phase Transitions', fontsize=16)
            
            # Remove y-axis as it's not meaningful
            ax_phase.set_yticks([])
            ax_phase.set_ylabel('')
            
            # Add grid for x-axis only
            ax_phase.grid(True, axis='x', alpha=0.3)
            
            # Add annotations explaining the visualization
            if show_both:
                # Add legend explaining confidence visualization
                legend_text = 'Detected transitions: Line opacity & style indicate confidence\nSolid line = high confidence (>0.7), Dashed = lower confidence'
                ax_phase.text(0.98, 0.05, legend_text, 
                            transform=ax_phase.transAxes, ha='right', va='bottom',
                            fontsize=9, style='italic', color='darkblue',
                            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8))
        else:
            # If no phase info, just show a message
            ax_phase.text(0.5, 0.5, 'No phase information available', 
                        ha='center', va='center', fontsize=14)
            ax_phase.set_xlim(0, 1)
            ax_phase.set_ylim(0, 1)
            ax_phase.axis('off')
        
        # Overall title
        session_id = conversation.get('metadata', {}).get('session_id', 'Unknown')
        filename = conversation.get('metadata', {}).get('filename', 'unknown.json')
        plt.suptitle(f"Ensemble Analysis - {session_id[:12]}\n{filename}", 
                    fontsize=16, y=0.98)
        
        # Save figure
        if save_path:
            plt.tight_layout(rect=[0, 0.02, 1, 0.96])  # Leave space for suptitle and legend
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved ensemble visualization to {save_path}")
        else:
            session_id = conversation.get('metadata', {}).get('session_id', 'Unknown')
            tier = conversation.get('tier', 'unknown')
            filename = f"{tier}_{session_id[:12]}_ensemble_comprehensive.png"
            default_path = self.output_dir / filename
            plt.tight_layout(rect=[0, 0.02, 1, 0.96])
            plt.savefig(default_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved ensemble visualization to {default_path}")
            
        plt.close(fig)
        return fig
        
    def _standardize_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        """Standardize embeddings to zero mean and unit variance."""
        mean = np.mean(embeddings, axis=0)
        std = np.std(embeddings, axis=0)
        return (embeddings - mean) / (std + 1e-8)
        
    def _calculate_distance_matrix(self, embeddings: np.ndarray) -> np.ndarray:
        """Calculate pairwise Euclidean distance matrix."""
        n = len(embeddings)
        distances = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i+1, n):
                dist = np.linalg.norm(embeddings[i] - embeddings[j])
                distances[i, j] = dist
                distances[j, i] = dist
                
        return distances
        
    def _calculate_self_similarity(self, embeddings: np.ndarray) -> np.ndarray:
        """Calculate self-similarity matrix using cosine similarity."""
        n = len(embeddings)
        similarities = np.zeros((n, n))
        
        for i in range(n):
            for j in range(n):
                norm_i = np.linalg.norm(embeddings[i])
                norm_j = np.linalg.norm(embeddings[j])
                
                if norm_i > 1e-8 and norm_j > 1e-8:
                    cos_sim = np.dot(embeddings[i], embeddings[j]) / (norm_i * norm_j)
                    similarities[i, j] = (cos_sim + 1) / 2  # Normalize to [0, 1]
                else:
                    similarities[i, j] = 0
                    
        return similarities
        
    def _calculate_matrix_correlations(self, matrices: Dict[str, np.ndarray]) -> np.ndarray:
        """Calculate pairwise correlations between matrices."""
        model_names = list(matrices.keys())
        n_models = len(model_names)
        corr_matrix = np.ones((n_models, n_models))
        
        for i, model1 in enumerate(model_names):
            for j, model2 in enumerate(model_names):
                if i != j:
                    mat1 = matrices[model1].flatten()
                    mat2 = matrices[model2].flatten()
                    corr = np.corrcoef(mat1, mat2)[0, 1]
                    corr_matrix[i, j] = corr
                    
        return corr_matrix
        
    def _calculate_trajectory_correlations(self, embeddings: Dict[str, np.ndarray]) -> np.ndarray:
        """Calculate trajectory correlations based on velocity patterns."""
        model_names = list(embeddings.keys())
        n_models = len(model_names)
        corr_matrix = np.ones((n_models, n_models))
        
        # Calculate velocities for each model
        velocities = {}
        for model, emb in embeddings.items():
            vel = []
            for i in range(1, len(emb)):
                vel.append(np.linalg.norm(emb[i] - emb[i-1]))
            velocities[model] = np.array(vel)
            
        # Calculate correlations
        for i, model1 in enumerate(model_names):
            for j, model2 in enumerate(model_names):
                if i != j:
                    corr = np.corrcoef(velocities[model1], velocities[model2])[0, 1]
                    corr_matrix[i, j] = corr
                    
        return corr_matrix
        
    def _calculate_topology_preservation(self, embeddings: Dict[str, np.ndarray]) -> np.ndarray:
        """Calculate topology preservation between models."""
        model_names = list(embeddings.keys())
        n_models = len(model_names)
        topo_matrix = np.ones((n_models, n_models))
        
        # For topology preservation, we check if nearest neighbors are preserved
        k = min(10, len(next(iter(embeddings.values()))) // 5)
        
        for i, model1 in enumerate(model_names):
            for j, model2 in enumerate(model_names):
                if i != j:
                    # Calculate k-nearest neighbors for each point in both models
                    emb1 = embeddings[model1]
                    emb2 = embeddings[model2]
                    
                    preservation_scores = []
                    for idx in range(len(emb1)):
                        # Find k nearest neighbors in model1
                        dists1 = [np.linalg.norm(emb1[idx] - emb1[jdx]) 
                                 for jdx in range(len(emb1)) if idx != jdx]
                        nn1 = np.argsort(dists1)[:k]
                        
                        # Find k nearest neighbors in model2
                        dists2 = [np.linalg.norm(emb2[idx] - emb2[jdx]) 
                                 for jdx in range(len(emb2)) if idx != jdx]
                        nn2 = np.argsort(dists2)[:k]
                        
                        # Calculate overlap
                        overlap = len(set(nn1) & set(nn2)) / k
                        preservation_scores.append(overlap)
                        
                    topo_matrix[i, j] = np.mean(preservation_scores)
                    
        return topo_matrix