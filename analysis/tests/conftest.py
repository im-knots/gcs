"""
Shared fixtures and configuration for tests.
"""

import pytest
import numpy as np
import tempfile
import shutil
from pathlib import Path


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test outputs."""
    temp_path = tempfile.mkdtemp()
    yield Path(temp_path)
    shutil.rmtree(temp_path)


@pytest.fixture
def sample_embedding_data():
    """Create sample embedding data for testing."""
    n_messages = 50
    models = {
        'all-MiniLM-L6-v2': (384,),
        'all-mpnet-base-v2': (768,),
        'all-distilroberta-v1': (768,),
        'word2vec': (300,),
        'glove': (300,)
    }
    
    embeddings = {}
    for model, (dim,) in models.items():
        embeddings[model] = np.random.randn(n_messages, dim)
    
    return embeddings


@pytest.fixture
def mock_gpu_available(monkeypatch):
    """Mock GPU availability for testing."""
    monkeypatch.setattr('torch.cuda.is_available', lambda: True)
    monkeypatch.setattr('torch.cuda.device_count', lambda: 1)


@pytest.fixture(autouse=True)
def set_random_seed():
    """Set random seed for reproducible tests."""
    np.random.seed(42)