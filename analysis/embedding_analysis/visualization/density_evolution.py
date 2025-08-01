"""
Embedding space density evolution visualizations.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Ellipse
import seaborn as sns
from scipy.stats import gaussian_kde, multivariate_normal
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class DensityEvolutionVisualizer:
    """
    Visualize how conversation density evolves in embedding space.
    """
    
    def __init__(self):
        """Initialize density evolution visualizer."""
        plt.style.use('seaborn-v0_8-whitegrid')
        
    def plot_density_evolution(self,
                             embeddings: np.ndarray,
                             window_size: int = 20,
                             step_size: int = 5,
                             save_path: Optional[str] = None,
                             phases: Optional[List[Dict]] = None) -> plt.Figure:
        """
        Plot how embedding density evolves over the conversation.
        
        Args:
            embeddings: Array of embeddings (n_messages, embedding_dim)
            window_size: Size of sliding window
            step_size: Step size for sliding window
            save_path: Path to save figure
            phases: Optional phase boundaries to mark
            
        Returns:
            Figure object
        """
        # Reduce to 2D for visualization
        if embeddings.shape[1] > 2:
            pca = PCA(n_components=2)
            embeddings_2d = pca.fit_transform(embeddings)
            variance_explained = pca.explained_variance_ratio_
        else:
            embeddings_2d = embeddings
            variance_explained = [1.0, 0.0]
            
        # Create figure with subplots
        fig = plt.figure(figsize=(15, 10))
        
        # Main density evolution plot
        ax_main = plt.subplot2grid((3, 3), (0, 0), colspan=2, rowspan=2)
        
        # Density metrics over time
        ax_metrics = plt.subplot2grid((3, 3), (2, 0), colspan=2)
        
        # Current window detail
        ax_detail = plt.subplot2grid((3, 3), (0, 2), rowspan=2)
        
        # Phase indicator
        ax_phase = plt.subplot2grid((3, 3), (2, 2))
        
        # Calculate density evolution
        windows = []
        densities = []
        centroids = []
        spreads = []
        
        for i in range(0, len(embeddings_2d) - window_size, step_size):
            window = embeddings_2d[i:i+window_size]
            windows.append((i, i+window_size))
            
            # Calculate density metrics
            centroid = np.mean(window, axis=0)
            centroids.append(centroid)
            
            # Covariance and spread
            if len(window) > 1:
                cov = np.cov(window.T)
                spread = np.sqrt(np.trace(cov))
                spreads.append(spread)
                
                # KDE density at centroid
                try:
                    kde = gaussian_kde(window.T)
                    density = kde(centroid.reshape(-1, 1))[0]
                    densities.append(density)
                except:
                    densities.append(0)
            else:
                spreads.append(0)
                densities.append(0)
                
        # Plot main trajectory with density coloring
        self._plot_density_trajectory(ax_main, embeddings_2d, windows, densities, phases)
        ax_main.set_title(f'Embedding Space Density Evolution\n(Variance explained: {variance_explained[0]:.1%} + {variance_explained[1]:.1%})')
        ax_main.set_xlabel('First Principal Component')
        ax_main.set_ylabel('Second Principal Component')
        
        # Plot density metrics
        window_centers = [w[0] + window_size//2 for w in windows]
        
        ax_metrics.plot(window_centers, densities, 'b-', label='Density at centroid')
        ax_metrics_twin = ax_metrics.twinx()
        ax_metrics_twin.plot(window_centers, spreads, 'r-', label='Spread', alpha=0.7)
        
        ax_metrics.set_xlabel('Conversation Turn')
        ax_metrics.set_ylabel('Density', color='b')
        ax_metrics_twin.set_ylabel('Spread', color='r')
        ax_metrics.set_title('Density Metrics Over Time')
        
        # Mark phases
        if phases:
            for phase in phases:
                turn = phase.get('turn', phase.get('start_turn', 0))
                ax_metrics.axvline(turn, color='gray', linestyle='--', alpha=0.5)
                
        # Add grid
        ax_metrics.grid(True, alpha=0.3)
        
        # Detail view of embedding space regions
        self._plot_density_regions(ax_detail, embeddings_2d, windows, densities)
        ax_detail.set_title('Density Regions')
        
        # Phase density distribution
        if phases:
            self._plot_phase_densities(ax_phase, embeddings_2d, phases, windows, densities)
        else:
            ax_phase.text(0.5, 0.5, 'No phase information', 
                         ha='center', va='center', transform=ax_phase.transAxes)
            ax_phase.axis('off')
            
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig
        
    def create_density_animation(self,
                               embeddings: np.ndarray,
                               window_size: int = 20,
                               save_path: Optional[str] = None,
                               fps: int = 10) -> animation.FuncAnimation:
        """
        Create an animation showing density evolution.
        
        Args:
            embeddings: Array of embeddings
            window_size: Size of sliding window
            save_path: Path to save animation (mp4)
            fps: Frames per second
            
        Returns:
            Animation object
        """
        # Reduce to 2D
        if embeddings.shape[1] > 2:
            pca = PCA(n_components=2)
            embeddings_2d = pca.fit_transform(embeddings)
        else:
            embeddings_2d = embeddings
            
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Set up plot limits
        margin = 0.1
        x_min, x_max = embeddings_2d[:, 0].min(), embeddings_2d[:, 0].max()
        y_min, y_max = embeddings_2d[:, 1].min(), embeddings_2d[:, 1].max()
        x_range = x_max - x_min
        y_range = y_max - y_min
        
        ax.set_xlim(x_min - margin * x_range, x_max + margin * x_range)
        ax.set_ylim(y_min - margin * y_range, y_max + margin * y_range)
        
        # Initialize plot elements
        trajectory_line, = ax.plot([], [], 'b-', alpha=0.3, linewidth=1)
        window_scatter = ax.scatter([], [], c='red', s=50, alpha=0.8)
        density_contour = None
        
        title = ax.set_title('')
        ax.set_xlabel('PC 1')
        ax.set_ylabel('PC 2')
        
        def init():
            trajectory_line.set_data([], [])
            window_scatter.set_offsets(np.empty((0, 2)))
            return trajectory_line, window_scatter
            
        def animate(frame):
            nonlocal density_contour
            
            # Clear previous contour
            if density_contour is not None:
                for coll in density_contour.collections:
                    coll.remove()
                    
            # Current window
            start = frame
            end = min(frame + window_size, len(embeddings_2d))
            
            if end <= start:
                return trajectory_line, window_scatter
                
            # Update trajectory
            trajectory_line.set_data(embeddings_2d[:end, 0], embeddings_2d[:end, 1])
            
            # Update current window
            window_data = embeddings_2d[start:end]
            window_scatter.set_offsets(window_data)
            
            # Calculate and plot density
            if len(window_data) > 3:
                try:
                    # Create density estimation
                    kde = gaussian_kde(window_data.T)
                    
                    # Create grid
                    x_grid = np.linspace(ax.get_xlim()[0], ax.get_xlim()[1], 50)
                    y_grid = np.linspace(ax.get_ylim()[0], ax.get_ylim()[1], 50)
                    X_grid, Y_grid = np.meshgrid(x_grid, y_grid)
                    positions = np.vstack([X_grid.ravel(), Y_grid.ravel()])
                    
                    # Evaluate density
                    Z = kde(positions).reshape(X_grid.shape)
                    
                    # Plot contour
                    density_contour = ax.contour(X_grid, Y_grid, Z, 
                                               colors='red', alpha=0.5, linewidths=1)
                except:
                    pass
                    
            # Update title
            title.set_text(f'Conversation Turns {start}-{end}')
            
            return trajectory_line, window_scatter
            
        # Create animation
        n_frames = len(embeddings_2d) - window_size + 1
        anim = animation.FuncAnimation(fig, animate, init_func=init,
                                     frames=n_frames, interval=1000/fps,
                                     blit=False)
        
        if save_path:
            writer = animation.FFMpegWriter(fps=fps, bitrate=1800)
            anim.save(save_path, writer=writer)
            
        return anim
        
    def plot_phase_density_comparison(self,
                                    embeddings: np.ndarray,
                                    phases: List[Dict],
                                    save_path: Optional[str] = None) -> plt.Figure:
        """
        Compare density characteristics across different phases.
        
        Args:
            embeddings: Array of embeddings
            phases: List of phase boundaries
            save_path: Path to save figure
            
        Returns:
            Figure object
        """
        # Sort phases by start turn
        sorted_phases = sorted(phases, key=lambda x: x.get('start_turn', x.get('turn', 0)))
        
        # Reduce to 2D
        if embeddings.shape[1] > 2:
            pca = PCA(n_components=2)
            embeddings_2d = pca.fit_transform(embeddings)
        else:
            embeddings_2d = embeddings
            
        # Create figure
        n_phases = len(sorted_phases)
        fig, axes = plt.subplots(2, (n_phases + 1) // 2, figsize=(15, 10))
        axes = axes.flatten()
        
        # Analyze each phase
        phase_stats = []
        
        for i, phase in enumerate(sorted_phases):
            start = phase.get('start_turn', phase.get('turn', 0))
            
            # Find end (next phase start or conversation end)
            if i < len(sorted_phases) - 1:
                end = sorted_phases[i + 1].get('start_turn', sorted_phases[i + 1].get('turn', 0))
            else:
                end = len(embeddings_2d)
                
            # Extract phase embeddings
            phase_emb = embeddings_2d[start:end]
            
            if len(phase_emb) < 2:
                axes[i].text(0.5, 0.5, f'Phase {i+1}: Too few points', 
                           ha='center', va='center', transform=axes[i].transAxes)
                axes[i].axis('off')
                continue
                
            # Plot phase trajectory and density
            ax = axes[i]
            
            # Trajectory
            ax.plot(phase_emb[:, 0], phase_emb[:, 1], 'b-', alpha=0.5, linewidth=1)
            ax.scatter(phase_emb[:, 0], phase_emb[:, 1], c=range(len(phase_emb)), 
                      cmap='viridis', s=30, alpha=0.8)
                      
            # Density estimation
            try:
                kde = gaussian_kde(phase_emb.T)
                
                # Create grid
                x_min, x_max = phase_emb[:, 0].min(), phase_emb[:, 0].max()
                y_min, y_max = phase_emb[:, 1].min(), phase_emb[:, 1].max()
                
                # Add margin
                margin = 0.2
                x_range = x_max - x_min
                y_range = y_max - y_min
                
                x_grid = np.linspace(x_min - margin * x_range, x_max + margin * x_range, 50)
                y_grid = np.linspace(y_min - margin * y_range, y_max + margin * y_range, 50)
                X_grid, Y_grid = np.meshgrid(x_grid, y_grid)
                positions = np.vstack([X_grid.ravel(), Y_grid.ravel()])
                
                # Evaluate density
                Z = kde(positions).reshape(X_grid.shape)
                
                # Plot density contours
                ax.contourf(X_grid, Y_grid, Z, alpha=0.3, cmap='Reds')
                
                # Calculate phase statistics
                centroid = np.mean(phase_emb, axis=0)
                spread = np.sqrt(np.trace(np.cov(phase_emb.T)))
                max_density = np.max(Z)
                
                phase_stats.append({
                    'phase': i + 1,
                    'centroid': centroid,
                    'spread': spread,
                    'max_density': max_density,
                    'n_points': len(phase_emb)
                })
                
            except Exception as e:
                logger.warning(f"Density estimation failed for phase {i+1}: {e}")
                
            # Labels
            phase_name = phase.get('name', f'Phase {i+1}')
            ax.set_title(f'{phase_name}\n(Turns {start}-{end})')
            ax.set_xlabel('PC 1')
            ax.set_ylabel('PC 2')
            
        # Hide unused subplots
        for i in range(n_phases, len(axes)):
            axes[i].axis('off')
            
        # Add summary statistics
        if phase_stats:
            summary_text = "Phase Statistics:\n"
            for stat in phase_stats:
                summary_text += f"Phase {stat['phase']}: "
                summary_text += f"Spread={stat['spread']:.2f}, "
                summary_text += f"Density={stat['max_density']:.2f}\n"
                
            fig.text(0.02, 0.02, summary_text, fontsize=10, 
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
        plt.suptitle('Density Characteristics by Phase', fontsize=16)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig
        
    def _plot_density_trajectory(self, ax, embeddings_2d, windows, densities, phases):
        """Helper to plot trajectory with density coloring."""
        # Normalize densities for coloring
        if densities:
            norm_densities = (np.array(densities) - np.min(densities)) / (np.max(densities) - np.min(densities) + 1e-8)
        else:
            norm_densities = []
            
        # Plot trajectory
        ax.plot(embeddings_2d[:, 0], embeddings_2d[:, 1], 'gray', alpha=0.3, linewidth=0.5)
        
        # Color windows by density
        cmap = plt.cm.hot
        
        for i, ((start, end), density) in enumerate(zip(windows, norm_densities)):
            window_points = embeddings_2d[start:end]
            color = cmap(density)
            
            ax.scatter(window_points[:, 0], window_points[:, 1], 
                      c=[color], s=30, alpha=0.6)
                      
        # Mark start and end
        ax.scatter(embeddings_2d[0, 0], embeddings_2d[0, 1], 
                  color='green', s=100, marker='o', label='Start', zorder=5)
        ax.scatter(embeddings_2d[-1, 0], embeddings_2d[-1, 1], 
                  color='red', s=100, marker='s', label='End', zorder=5)
                  
        # Mark phases
        if phases:
            for phase in phases:
                turn = phase.get('turn', phase.get('start_turn', 0))
                if turn < len(embeddings_2d):
                    ax.scatter(embeddings_2d[turn, 0], embeddings_2d[turn, 1],
                             color='blue', s=150, marker='*', edgecolor='black',
                             linewidth=1, zorder=6)
                             
        ax.legend()
        
    def _plot_density_regions(self, ax, embeddings_2d, windows, densities):
        """Helper to plot density regions."""
        # Find high and low density regions
        if not densities:
            return
            
        density_array = np.array(densities)
        high_threshold = np.percentile(density_array, 75)
        low_threshold = np.percentile(density_array, 25)
        
        # Collect points by density level
        high_density_points = []
        low_density_points = []
        
        for (start, end), density in zip(windows, densities):
            window_points = embeddings_2d[start:end]
            if density > high_threshold:
                high_density_points.extend(window_points)
            elif density < low_threshold:
                low_density_points.extend(window_points)
                
        # Plot regions
        if high_density_points:
            high_density_points = np.array(high_density_points)
            ax.scatter(high_density_points[:, 0], high_density_points[:, 1],
                      c='red', alpha=0.3, s=20, label='High density')
                      
        if low_density_points:
            low_density_points = np.array(low_density_points)
            ax.scatter(low_density_points[:, 0], low_density_points[:, 1],
                      c='blue', alpha=0.3, s=20, label='Low density')
                      
        ax.set_xlabel('PC 1')
        ax.set_ylabel('PC 2')
        ax.legend()
        
    def _plot_phase_densities(self, ax, embeddings_2d, phases, windows, densities):
        """Helper to plot density distribution by phase."""
        # Map windows to phases
        phase_densities = {}
        
        for i, phase in enumerate(phases):
            phase_turn = phase.get('turn', phase.get('start_turn', 0))
            phase_name = phase.get('name', f'Phase {i+1}')
            phase_densities[phase_name] = []
            
            # Find windows in this phase
            for (start, end), density in zip(windows, densities):
                window_center = (start + end) // 2
                
                # Determine which phase this window belongs to
                if i == 0 and window_center < phase_turn:
                    phase_densities[phase_name].append(density)
                elif i < len(phases) - 1:
                    next_phase_turn = phases[i+1].get('turn', phases[i+1].get('start_turn', 0))
                    if phase_turn <= window_center < next_phase_turn:
                        phase_densities[phase_name].append(density)
                elif window_center >= phase_turn:
                    phase_densities[phase_name].append(density)
                    
        # Plot distributions
        data = []
        labels = []
        
        for phase_name, dens in phase_densities.items():
            if dens:
                data.append(dens)
                labels.append(phase_name)
                
        if data:
            ax.violinplot(data, showmeans=True)
            ax.set_xticks(range(1, len(labels) + 1))
            ax.set_xticklabels(labels, rotation=45, ha='right')
            ax.set_ylabel('Density')
            ax.set_title('Density by Phase')
            ax.grid(True, alpha=0.3)