"""Core functionality for embedding analysis."""

from .embedder import EnsembleEmbedder
from .trajectory import TrajectoryAnalyzer
from .conversation import ConversationLoader
from .geometric_invariance import (
    GeometricSignatureComputer,
    InvarianceAnalyzer,
    HypothesisTester,
    NullModelComparator
)
from .hierarchical_hypothesis_testing import (
    HierarchicalHypothesisTester,
    HypothesisResult,
    TierResult
)
from .paradigm_null_models import (
    ParadigmSpecificNullModels,
    MessageLevelNullModels
)
from .control_analyses import ControlAnalyses

__all__ = [
    'EnsembleEmbedder', 
    'TrajectoryAnalyzer', 
    'ConversationLoader',
    'GeometricSignatureComputer',
    'InvarianceAnalyzer',
    'HypothesisTester',
    'NullModelComparator',
    'HierarchicalHypothesisTester',
    'HypothesisResult',
    'TierResult',
    'ParadigmSpecificNullModels',
    'MessageLevelNullModels',
    'ControlAnalyses'
]