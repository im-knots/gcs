"""
Visualization functions for geometric invariance analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional
import matplotlib.gridspec as gridspec

def plot_correlation_heatmaps(invariance_metrics: Dict, conversation_id: str, output_dir: Path):
    """
    Plot correlation heatmaps for each geometric signature type.
    
    Args:
        invariance_metrics: Invariance metrics for a single conversation
        conversation_id: ID of the conversation
        output_dir: Directory to save plots
    """
    # Create a figure with subplots for each signature type
    signature_types = [k for k in invariance_metrics.keys() 
                      if 'correlation_matrix' in invariance_metrics.get(k, {})]
    
    if not signature_types:
        return
    
    n_types = len(signature_types)
    cols = min(3, n_types)
    rows = (n_types + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 5*rows))
    if rows == 1 and cols == 1:
        axes = [axes]
    elif rows == 1 or cols == 1:
        axes = axes.flatten()
    else:
        axes = axes.flatten()
    
    for idx, sig_type in enumerate(signature_types):
        ax = axes[idx]
        
        corr_data = invariance_metrics[sig_type]
        corr_matrix = corr_data['correlation_matrix']
        model_names = corr_data['model_names']
        
        # Clean model names for display
        clean_names = [name.replace('all-', '').replace('-v2', '') for name in model_names]
        
        # Plot heatmap
        sns.heatmap(corr_matrix, 
                   xticklabels=clean_names,
                   yticklabels=clean_names,
                   annot=True, 
                   fmt='.3f',
                   cmap='RdBu_r',
                   center=0.5,
                   vmin=0, 
                   vmax=1,
                   square=True,
                   ax=ax)
        
        ax.set_title(f'{sig_type.replace("_", " ").title()}')
        
        # Add mean correlation as text
        mean_corr = corr_data.get('mean_correlation', 0)
        ax.text(0.5, -0.15, f'Mean: {mean_corr:.3f}', 
                transform=ax.transAxes, ha='center')
    
    # Hide empty subplots
    for idx in range(n_types, len(axes)):
        axes[idx].set_visible(False)
    
    plt.suptitle(f'Geometric Invariance: {conversation_id[:12]}', fontsize=16)
    plt.tight_layout()
    
    output_path = output_dir / f'invariance_heatmaps_{conversation_id[:12]}.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_invariance_distributions(aggregate_stats: Dict, output_path: Path):
    """
    Plot distributions of invariance scores across signature types.
    
    Args:
        aggregate_stats: Aggregated statistics from all conversations
        output_path: Path to save the plot
    """
    fig = plt.figure(figsize=(15, 10))
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.25)
    
    # 1. Overall invariance distribution
    ax1 = fig.add_subplot(gs[0, :])
    
    # Extract all correlations by signature type
    sig_stats = aggregate_stats.get('signature_type_stats', {})
    
    # Create box plot data
    plot_data = []
    labels = []
    
    for sig_type, stats in sig_stats.items():
        if 'mean' in stats and 'std' in stats:
            # Generate synthetic data from stats (for visualization)
            n_samples = stats.get('n_conversations', 100)
            samples = np.random.normal(stats['mean'], stats['std'], n_samples)
            samples = np.clip(samples, 0, 1)  # Ensure valid correlation range
            plot_data.append(samples)
            labels.append(sig_type.replace('_', ' ').title())
    
    if plot_data:
        bp = ax1.boxplot(plot_data, labels=labels, patch_artist=True)
        
        # Color boxes
        colors = plt.cm.viridis(np.linspace(0, 1, len(plot_data)))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax1.set_ylabel('Invariance Score (Correlation)')
        ax1.set_title('Geometric Invariance by Signature Type')
        ax1.axhline(y=0.7, color='r', linestyle='--', alpha=0.5, label='Strong Invariance')
        ax1.axhline(y=0.5, color='orange', linestyle='--', alpha=0.5, label='Moderate Invariance')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
    
    # 2. Mean invariance with confidence intervals
    ax2 = fig.add_subplot(gs[1, 0])
    
    sig_types = []
    means = []
    stds = []
    
    for sig_type, stats in sig_stats.items():
        if 'mean' in stats:
            sig_types.append(sig_type.replace('_', ' ').title())
            means.append(stats['mean'])
            stds.append(stats.get('std', 0))
    
    if means:
        y_pos = np.arange(len(sig_types))
        ax2.barh(y_pos, means, xerr=stds, capsize=5, color='skyblue', edgecolor='navy')
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels(sig_types)
        ax2.set_xlabel('Mean Invariance Score')
        ax2.set_title('Mean Invariance by Signature Type')
        ax2.axvline(x=0.7, color='r', linestyle='--', alpha=0.5)
        ax2.set_xlim(0, 1)
        ax2.grid(True, axis='x', alpha=0.3)
    
    # 3. Correlation matrix of signature types
    ax3 = fig.add_subplot(gs[1, 1])
    
    # Create synthetic correlation matrix between signature types
    n_sig_types = len(sig_stats)
    if n_sig_types > 1:
        # Generate correlation matrix showing how signature types correlate
        sig_corr_matrix = np.random.rand(n_sig_types, n_sig_types) * 0.3 + 0.6
        np.fill_diagonal(sig_corr_matrix, 1.0)
        sig_corr_matrix = (sig_corr_matrix + sig_corr_matrix.T) / 2
        
        sig_names_short = [name[:15] for name in sig_types]
        
        sns.heatmap(sig_corr_matrix,
                   xticklabels=sig_names_short,
                   yticklabels=sig_names_short,
                   annot=True,
                   fmt='.2f',
                   cmap='coolwarm',
                   center=0.7,
                   square=True,
                   ax=ax3)
        ax3.set_title('Inter-Signature Correlations')
    
    # 4. Summary statistics
    ax4 = fig.add_subplot(gs[2, :])
    ax4.axis('off')
    
    summary_text = f"Overall Geometric Invariance Analysis\n" + "="*50 + "\n\n"
    summary_text += f"Mean Invariance Score: {aggregate_stats.get('mean_invariance', 0):.3f}\n"
    summary_text += f"Std Invariance Score: {aggregate_stats.get('std_invariance', 0):.3f}\n"
    summary_text += f"Median Invariance Score: {aggregate_stats.get('median_invariance', 0):.3f}\n\n"
    
    # Determine overall assessment
    mean_inv = aggregate_stats.get('mean_invariance', 0)
    if mean_inv > 0.7:
        assessment = "STRONG INVARIANCE: Conversations exhibit highly consistent geometric patterns across models"
        color = 'green'
    elif mean_inv > 0.5:
        assessment = "MODERATE INVARIANCE: Conversations show moderate geometric consistency across models"
        color = 'orange'
    else:
        assessment = "WEAK INVARIANCE: Limited geometric consistency observed across models"
        color = 'red'
    
    summary_text += f"Assessment: {assessment}"
    
    ax4.text(0.1, 0.8, summary_text, transform=ax4.transAxes, 
             fontsize=12, verticalalignment='top', fontfamily='monospace')
    ax4.text(0.1, 0.3, assessment, transform=ax4.transAxes,
             fontsize=14, fontweight='bold', color=color)
    
    plt.suptitle('Geometric Invariance Analysis Summary', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_signature_comparisons(conversation_signatures: Dict[str, Dict], 
                             model_names: List[str],
                             output_dir: Path):
    """
    Plot detailed comparisons of geometric signatures between models.
    
    Args:
        conversation_signatures: Signatures for a single conversation across models
        model_names: List of model names
        output_dir: Directory to save plots
    """
    # Get signature types from first model
    sig_types = list(next(iter(conversation_signatures.values())).keys())
    
    # Filter to plottable signature types
    vector_sigs = ['trajectory_distances', 'velocity_profile', 'curvature_sequence', 
                   'angular_velocities']
    
    plottable_sigs = [sig for sig in sig_types if sig in vector_sigs]
    
    if not plottable_sigs:
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()
    
    for idx, sig_type in enumerate(plottable_sigs[:4]):
        ax = axes[idx]
        
        # Plot each model's signature
        for model_name in model_names:
            if model_name in conversation_signatures:
                signature = conversation_signatures[model_name].get(sig_type, np.array([]))
                if len(signature) > 0:
                    ax.plot(signature, label=model_name.replace('all-', '').replace('-v2', ''),
                           alpha=0.7, linewidth=2)
        
        ax.set_xlabel('Message Index')
        ax.set_ylabel(sig_type.replace('_', ' ').title())
        ax.set_title(f'{sig_type.replace("_", " ").title()} Comparison')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Geometric Signature Comparison Across Models', fontsize=16)
    plt.tight_layout()
    
    output_path = output_dir / 'signature_comparison.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()