"""
Comprehensive ensemble visualization for conversation analysis.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for parallel processing
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle, Patch
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sns
from sklearn.decomposition import PCA
from scipy import stats
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import logging
from ..utils import calculate_phase_correlation, calculate_model_agreement

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
                                         save_path: Optional[str] = None,
                                         save_pdf: Optional[str] = None) -> plt.Figure:
        """
        Create comprehensive ensemble visualization with distance matrices,
        self-similarity, recurrence plots, trajectories, and phase information.
        
        Args:
            conversation: Conversation data
            ensemble_embeddings: Embeddings from each model
            phase_info: Phase detection results
            save_path: Path to save figure as PNG
            save_pdf: Path to save figure as PDF
            
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
        fig_height = 32  # Increased to accommodate statistics row
        logger.info(f"Creating figure ({fig_width}x{fig_height})...")
        fig = plt.figure(figsize=(fig_width, fig_height))
        
        # Create GridSpec for flexible layout
        # Total rows: 4 (section 1) + 1 (correlation tables) + 3 (phase diagram) + 1 (stats) = 9
        total_rows = 9
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
        
        # Import statistical functions
        from embedding_analysis.utils.statistics import (
            calculate_distance_matrix_correlations,
            calculate_velocity_profile_correlations,
            calculate_topology_preservation
        )
        
        # Calculate correlations using functions from stats module
        dist_corr_matrix = calculate_distance_matrix_correlations(distance_matrices)
        sns.heatmap(dist_corr_matrix, annot=True, fmt='.3f', xticklabels=model_names,
                   yticklabels=model_names, cmap='RdBu_r', center=0.5, ax=ax_dist_corr,
                   vmin=0, vmax=1, cbar_kws={'label': 'Correlation'})
        ax_dist_corr.set_title('Distance Matrix Correlations', fontsize=12)
        
        # Velocity correlations
        if ax_vel_corr is not None:
            vel_corr_matrix = calculate_velocity_profile_correlations(ensemble_embeddings)
            sns.heatmap(vel_corr_matrix, annot=True, fmt='.3f', xticklabels=model_names,
                       yticklabels=model_names, cmap='RdBu_r', center=0.5, ax=ax_vel_corr,
                       vmin=0, vmax=1, cbar_kws={'label': 'Correlation'})
            ax_vel_corr.set_title('Velocity Profile Correlations', fontsize=12)
        
        # Topology preservation
        if ax_topo is not None:
            topo_matrix = calculate_topology_preservation(ensemble_embeddings)
            sns.heatmap(topo_matrix, annot=True, fmt='.3f', xticklabels=model_names,
                       yticklabels=model_names, cmap='Greens', vmin=0, vmax=1,
                       cbar_kws={'label': 'Preservation'}, ax=ax_topo)
            ax_topo.set_title('Topology Preservation', fontsize=12)
        
        # Section 3: Phase transition timeline (Rows 5-7)
        # Create a subplot spanning all columns for the phase diagram
        ax_phase = fig.add_subplot(gs[5:8, :])
        
        # Get model-specific phase detections if available
        model_phases = {}
        if phase_info and 'model_phases' in phase_info:
            model_phases = phase_info['model_phases']
        
        # Create phase transition timeline
        if model_phases or annotated_phases:
            # Setup layout - annotated on top, then each model
            n_models_with_phases = len(model_phases)
            total_rows = 1 + n_models_with_phases  # 1 for annotated + n models
            
            # Calculate y positions
            y_positions = np.linspace(0.9, 0.1, total_rows)
            annotated_y = y_positions[0] if annotated_phases else None
            
            # Model colors
            model_colors = plt.cm.tab10(np.linspace(0, 1, 10))[:n_models_with_phases]
            
            # Draw base lines
            for i, y in enumerate(y_positions[:len(y_positions) if annotated_phases else n_models_with_phases]):
                ax_phase.axhline(y=y, color='gray', linewidth=1, alpha=0.3)
            
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
                    
                    # Check for core phase types in complex names
                    core_phases = ['exploration', 'synthesis', 'conclusion', 'introduction', 'opening']
                    for core_phase in core_phases:
                        if core_phase in phase_name:
                            phase_name = core_phase
                            break
                    
                    # Only add if it's different from the last phase
                    if phase_name != last_phase and phase_name:
                        unique_transitions.append({
                            'turn': phase['start_turn'],
                            'phase': phase_name,
                            'original': phase['name']
                        })
                        last_phase = phase_name
                
                # Draw points for each unique transition
                for phase_idx, transition in enumerate(unique_transitions):
                    turn = transition['turn']
                    phase_name = transition['phase'].title()
                    
                    if 0 <= turn <= n_messages:
                        # Draw point instead of vertical line
                        ax_phase.scatter(turn, annotated_y, color='red', s=100, 
                                       marker='o', zorder=5, alpha=0.8)
                        
                        # Add phase name diagonally below the point, touching the dot
                        # Calculate text position to touch the dot edge
                        text_offset = 0.05  # Small offset from dot edge
                        ax_phase.text(turn - 1, annotated_y - text_offset, phase_name, 
                                    ha='right', va='top', fontsize=8,
                                    color='darkred', fontweight='bold',
                                    rotation=45, rotation_mode='anchor',
                                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white', 
                                            edgecolor='none', alpha=0.7))
                
                # Add label for annotated transitions
                ax_phase.text(-0.05, annotated_y, 'Annotated', 
                            transform=ax_phase.transAxes, ha='right', va='center',
                            fontsize=12, fontweight='bold', color='darkred')
            
            # Draw model-specific phase transitions
            if model_phases:
                model_idx = 0
                for model_name, phases in model_phases.items():
                    if annotated_phases:
                        y_pos = y_positions[model_idx + 1]  # Skip first position for annotated
                    else:
                        y_pos = y_positions[model_idx]
                        
                    color = model_colors[model_idx]
                    
                    # Draw phase transitions for this model
                    for phase in phases:
                        turn = phase['turn']
                        confidence = phase.get('confidence', 0.5)
                        
                        if 0 < turn < n_messages:
                            # Calculate error bar size based on confidence
                            # Higher confidence = smaller error bars
                            error_size = (1 - confidence) * 5  # Max error of 5 turns when confidence is 0
                            
                            # Draw point with horizontal error bars
                            ax_phase.errorbar(turn, y_pos, xerr=error_size, 
                                            fmt='o', color=color, markersize=8,
                                            capsize=5, capthick=1.5,
                                            alpha=0.5 + 0.5 * confidence,
                                            zorder=4)
                    
                    # Add model label
                    ax_phase.text(-0.02, y_pos, model_name.replace('all-', '').replace('-v2', ''),
                                transform=ax_phase.transAxes, ha='right', va='center',
                                fontsize=10, color=color, fontweight='bold')
                    
                    model_idx += 1
            
            # Set up the axis
            ax_phase.set_xlim(-5, n_messages + 5)  # Add some padding
            ax_phase.set_ylim(0, 1)
            ax_phase.set_xlabel('Conversation Turn', fontsize=14)
            
            # Update title based on what we're showing
            if annotated_phases and model_phases:
                ax_phase.set_title('Phase Transitions: Annotated vs Model-Specific Detections', fontsize=16)
            elif model_phases:
                ax_phase.set_title('Model-Specific Phase Detections', fontsize=16)
            else:
                ax_phase.set_title('Annotated Phase Transitions', fontsize=16)
            
            # Remove y-axis as it's not meaningful
            ax_phase.set_yticks([])
            ax_phase.set_ylabel('')
            
            # Add grid for x-axis only
            ax_phase.grid(True, axis='x', alpha=0.3)
            
            # Add legend for model colors if showing model phases
            if model_phases:
                # Create color legend
                legend_elements = []
                model_idx = 0
                for model_name in model_phases.keys():
                    color = model_colors[model_idx]
                    legend_elements.append(Patch(facecolor=color, label=model_name.replace('all-', '').replace('-v2', '')))
                    model_idx += 1
                
                # Add note about error bars
                error_bar_legend = Line2D([0], [0], color='gray', marker='o', markersize=8,
                                        label='Error bars indicate confidence\n(smaller = higher confidence)',
                                        linestyle='', markerfacecolor='gray', alpha=0.7)
                legend_elements.append(error_bar_legend)
                    
                ax_phase.legend(handles=legend_elements, loc='upper right', 
                              bbox_to_anchor=(0.98, 0.98), ncol=1,
                              frameon=True, fancybox=True, shadow=True, fontsize=9)
        else:
            # If no phase info, just show a message
            ax_phase.text(0.5, 0.5, 'No phase information available', 
                        ha='center', va='center', fontsize=14)
            ax_phase.set_xlim(0, 1)
            ax_phase.set_ylim(0, 1)
            ax_phase.axis('off')
            
        # Section 4: Phase detection correlation matrices (Row 8)
        if model_phases:
            # Calculate phase agreement correlations between models
            phase_correlations = np.zeros((n_models, n_models))
            
            # Create binary phase arrays for each model
            model_phase_arrays = {}
            for i, (model_name, phases) in enumerate(model_phases.items()):
                phase_array = np.zeros(n_messages)
                for phase in phases:
                    turn = phase['turn']
                    if 0 <= turn < n_messages:
                        # Apply small window around phase
                        window = 3
                        start = max(0, turn - window)
                        end = min(n_messages, turn + window + 1)
                        phase_array[start:end] = 1
                model_phase_arrays[model_name] = phase_array
            
            # Calculate pairwise correlations
            for i, model1 in enumerate(model_names):
                for j, model2 in enumerate(model_names):
                    if i == j:
                        phase_correlations[i, j] = 1.0
                    else:
                        arr1 = model_phase_arrays[model1]
                        arr2 = model_phase_arrays[model2]
                        if np.std(arr1) > 0 and np.std(arr2) > 0:
                            corr, _ = stats.pearsonr(arr1, arr2)
                            phase_correlations[i, j] = corr
                        else:
                            phase_correlations[i, j] = 0
            
            # Create three correlation matrix plots
            # 1. Model-to-model phase agreement
            ax_phase_corr = fig.add_subplot(gs[8, :n_models//3 + 1])
            sns.heatmap(phase_correlations, 
                       xticklabels=[m.replace('all-', '').replace('-v2', '') for m in model_names],
                       yticklabels=[m.replace('all-', '').replace('-v2', '') for m in model_names],
                       cmap='coolwarm', center=0, vmin=-1, vmax=1,
                       square=True, cbar_kws={'label': 'Correlation'},
                       ax=ax_phase_corr, annot=True, fmt='.2f', annot_kws={'size': 8})
            ax_phase_corr.set_title('Phase Detection Agreement', fontsize=12)
            
            # 2. Model accuracy against annotations (if available)
            if annotated_phases:
                from embedding_analysis.utils.statistics import calculate_phase_correlation
                
                accuracy_matrix = np.zeros((n_models, 4))  # 4 metrics: correlation, precision, recall, F1
                
                for i, (model_name, phases) in enumerate(model_phases.items()):
                    phase_stats = calculate_phase_correlation(annotated_phases, phases, n_messages)
                    accuracy_matrix[i, 0] = phase_stats['correlation']
                    accuracy_matrix[i, 1] = phase_stats['precision']
                    accuracy_matrix[i, 2] = phase_stats['recall']
                    accuracy_matrix[i, 3] = phase_stats['f1_score']
                
                ax_accuracy = fig.add_subplot(gs[8, n_models//3 + 1:2*n_models//3 + 1])
                sns.heatmap(accuracy_matrix,
                           xticklabels=['Correlation', 'Precision', 'Recall', 'F1'],
                           yticklabels=[m.replace('all-', '').replace('-v2', '') for m in model_names],
                           cmap='Greens', vmin=0, vmax=1,
                           cbar_kws={'label': 'Score'}, ax=ax_accuracy,
                           annot=True, fmt='.2f', annot_kws={'size': 8})
                ax_accuracy.set_title('Phase Detection Accuracy', fontsize=12)
            
            # 3. Phase timing consistency
            if model_phases:
                # Create timing difference matrix
                timing_matrix = np.zeros((n_models, n_models))
                
                for i, model1 in enumerate(model_names):
                    phases1_turns = [p['turn'] for p in model_phases[model1]]
                    
                    for j, model2 in enumerate(model_names):
                        if i == j:
                            timing_matrix[i, j] = 0
                        else:
                            phases2_turns = [p['turn'] for p in model_phases[model2]]
                            
                            # Calculate mean timing difference
                            timing_diffs = []
                            for turn1 in phases1_turns:
                                if phases2_turns:
                                    min_diff = min(abs(turn1 - turn2) for turn2 in phases2_turns)
                                    if min_diff <= 10:  # Only count reasonable matches
                                        timing_diffs.append(min_diff)
                            
                            if timing_diffs:
                                timing_matrix[i, j] = np.mean(timing_diffs)
                            else:
                                timing_matrix[i, j] = np.nan
                
                ax_timing = fig.add_subplot(gs[8, 2*n_models//3 + 1:])
                
                # Mask NaN values for better visualization
                mask = np.isnan(timing_matrix)
                
                sns.heatmap(timing_matrix,
                           xticklabels=[m.replace('all-', '').replace('-v2', '') for m in model_names],
                           yticklabels=[m.replace('all-', '').replace('-v2', '') for m in model_names],
                           cmap='YlOrRd', vmin=0, vmax=5,
                           cbar_kws={'label': 'Mean Turn Difference'},
                           ax=ax_timing, mask=mask,
                           annot=True, fmt='.1f', annot_kws={'size': 8})
                ax_timing.set_title('Phase Timing Differences', fontsize=12)
        
        # Overall title
        session_id = conversation.get('metadata', {}).get('session_id', 'Unknown')
        filename = conversation.get('metadata', {}).get('filename', 'unknown.json')
        
        # Check if we have annotated outcome
        title_lines = [f"Ensemble Analysis - {session_id[:12]}", filename]
        if 'metadata' in conversation and 'annotated_outcome' in conversation['metadata']:
            outcome = conversation['metadata']['annotated_outcome']
            title_lines.append(f"Annotated Outcome: {outcome}")
            
        plt.suptitle('\n'.join(title_lines), fontsize=16, y=0.98)
        
        # Save figure
        if save_path or save_pdf:
            plt.tight_layout(rect=[0, 0.02, 1, 0.96])  # Leave space for suptitle and legend
            
            # Save as PNG if path provided
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Saved ensemble visualization to {save_path}")
            
            # Save as PDF if path provided
            if save_pdf:
                plt.savefig(save_pdf, format='pdf', bbox_inches='tight')
                logger.info(f"Saved ensemble visualization to {save_pdf}")
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
        
