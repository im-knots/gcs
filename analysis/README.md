# Conversation Trajectory Analysis

A GPU-accelerated Python package for analyzing conversation dynamics through embedding space trajectories, phase detection, and breakdown prediction.

## Overview

This package provides tools for:
- **Ensemble Embedding Analysis**: Using multiple embedding models to capture "shadows" of higher-dimensional conversational structure
- **Trajectory Analysis**: Tracking conversation dynamics through embedding space
- **Phase Detection**: Blindly detecting conversation phase transitions
- **Breakdown Prediction**: Predicting conversation breakdown using trajectory features

## Requirements

- **CUDA-capable GPU** (required)
- CUDA 11.0 or higher
- Python 3.8+

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/gcs.git
cd gcs/analysis

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install package with GPU support
pip install -e .

# For development
pip install -e ".[dev]"
```

### Verify GPU Installation

```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
```

## Quick Start

```python
from embedding_analysis import (
    EnsembleEmbedder,
    TrajectoryAnalyzer,
    PhaseDetector,
    BreakdownPredictor
)

# Initialize components
embedder = EnsembleEmbedder()
analyzer = TrajectoryAnalyzer()
detector = PhaseDetector()
predictor = BreakdownPredictor()

# Load conversation
conversation = {...}  # Your conversation data

# Generate ensemble embeddings
embeddings = embedder.embed_conversation(conversation['messages'])

# Analyze trajectory
metrics = analyzer.calculate_ensemble_trajectories(embeddings)

# Detect phases
phases = detector.detect_phases(embeddings)

# Extract features and predict breakdown
features = predictor.extract_features(embeddings, metrics, phases)
prediction = predictor.predict(features)
```

## Command Line Usage

Run the complete analysis pipeline:

```bash
# Analyze conversations with default settings
analyze-conversations

# Specify custom directories
analyze-conversations --data-dir /path/to/data --output-dir results/

# Limit conversations for testing
analyze-conversations --max-conversations 10

# Disable checkpointing
analyze-conversations --no-checkpoint

# Set logging level
analyze-conversations --log-level DEBUG
```

## Package Structure

```
embedding_analysis/
├── core/               # Core functionality
│   ├── embedder.py    # Ensemble embedding generation
│   ├── trajectory.py  # Trajectory analysis
│   └── conversation.py # Conversation loading
├── models/            # Analysis models
│   ├── phase_detector.py    # Phase detection
│   └── breakdown.py         # Breakdown prediction
├── visualization/     # Plotting and reports
│   ├── plots.py      # Visualization functions
│   └── reports.py    # Report generation
└── utils/            # Utilities
    ├── checkpoint.py # Checkpoint management
    └── logging.py    # Logging configuration
```

## Key Features

### Ensemble Embeddings
- Multiple embedding models capture different aspects of conversation
- Automatic agreement calculation between models
- Support for custom model configurations

### Trajectory Analysis
- Velocity, acceleration, and curvature metrics
- Cross-model consistency measures
- Anomaly detection in trajectories

### Phase Detection
- Blind detection using embedding shifts
- Consensus across multiple models
- Comparison with annotated ground truth

### Breakdown Prediction
- Feature extraction from trajectories and phases
- N-turn lookahead predictions
- Confidence scoring

## Output Files

The analysis generates:
- `figures/`: Visualization plots
  - Ensemble trajectories
  - Breakdown predictions
  - Feature importance
- `reports/`: Text reports
  - Summary statistics
  - Tier comparisons
  - Individual conversation reports
- `checkpoints/`: Resumable analysis state
- `logs/`: Detailed execution logs

## Citation

If you use this code in your research, please cite:

```bibtex
@article{yourname2024,
  title={Analyzing Conversation Dynamics through Embedding Space Trajectories},
  author={Your Name},
  journal={arXiv preprint},
  year={2024}
}
```

## License

MIT License - see LICENSE file for details.