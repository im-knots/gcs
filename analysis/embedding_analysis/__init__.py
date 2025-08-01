"""
Embedding Analysis Package for Conversation Trajectory Analysis

This package provides tools for analyzing conversation dynamics through
embedding space trajectories, phase detection, and breakdown prediction.
"""

__version__ = "0.1.0"

from .core.embedder import EnsembleEmbedder
from .core.trajectory import TrajectoryAnalyzer
from .models.breakdown import BreakdownPredictor
from .models.phase_detector import PhaseDetector
from .visualization.plots import TrajectoryVisualizer

__all__ = [
    'EnsembleEmbedder',
    'TrajectoryAnalyzer',
    'BreakdownPredictor',
    'PhaseDetector',
    'TrajectoryVisualizer'
]