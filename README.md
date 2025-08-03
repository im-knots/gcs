# Geometry of Conversation Space (GCS)

A research framework for analyzing the geometric properties of AI conversations through embedding trajectory analysis, phase detection, and statistical invariance testing.

## Overview

This project investigates whether conversations with AI systems exhibit consistent geometric patterns across different embedding models. We analyze conversation trajectories in high-dimensional embedding spaces to identify:

- **Geometric invariance** across different embedding models (transformer-based and classical)
- **Phase transitions** in conversation dynamics (exploration → insight → synthesis)
- **Transport-theoretic measures** of conversation structure
- **Statistical validation** through hierarchical hypothesis testing


## Installation

### Prerequisites
- Python 3.12+
- CUDA-capable GPU (required for embedding generation)
- LaTeX distribution (for paper compilation)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/im-knots/gcs.git
cd gcs
```

2. Create and activate a virtual environment:
```bash
cd analysis
python -m venv conversation_analysis_env
source conversation_analysis_env/bin/activate  # On Windows: conversation_analysis_env\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Running the Analysis

Basic usage:
```bash
python run_analysis.py
```

With options:
```bash
python run_analysis.py \
    --data-dir /path/to/conversations \
    --output-dir results/ \
    --max-conversations 100 \
    --batch-size 25 \
    --figure-format both \
    --log-level INFO
```

Options:
- `--data-dir`: Directory containing conversation JSON files
- `--output-dir`: Output directory for results (default: `analysis_output`)
- `--max-conversations`: Limit number of conversations to analyze
- `--batch-size`: GPU batch size for embedding generation (default: 25)
- `--figure-format`: Output format for figures: `png`, `pdf`, or `both`
- `--no-checkpoint`: Disable checkpointing
- `--clear-checkpoints`: Clear existing checkpoints before starting

### Conversation Data Format

Conversations should be in JSON format:
```json
{
  "metadata": {
    "session_id": "unique-id",
    "timestamp": "2024-01-01T00:00:00Z"
  },
  "messages": [
    {
      "role": "user",
      "content": "Message content",
      "timestamp": "2024-01-01T00:00:00Z"
    },
    {
      "role": "assistant",
      "content": "Response content",
      "timestamp": "2024-01-01T00:00:01Z"
    }
  ],
  "phases": [  // Optional: annotated phase transitions
    {
      "turn": 5,
      "phase": "exploration"
    }
  ]
}
```

## Analysis Components

### 1. Embedding Generation
- **Ensemble approach**: 5 models (3 transformers, 2 classical)
- **GPU-accelerated** batch processing
- **Dimension handling**: Automatic alignment for cross-model comparison

### 2. Trajectory Analysis
- **Velocity & acceleration** in embedding space
- **Curvature** with ensemble methods
- **Adaptive normalization** for numerical stability

### 3. Phase Detection
- **Multiple algorithms**: HMM, spectral clustering, change point detection
- **Ensemble consensus** across methods
- **Comparison** with human annotations when available

### 4. Transport Metrics
- **Wasserstein distance** for trajectory comparison
- **Sinkhorn divergence** for high-dimensional robustness
- **Gromov-Wasserstein** for isometry-invariant comparison
- **GPU acceleration** via PyTorch backend

### 5. Statistical Validation
- **Hierarchical hypothesis testing** (3 tiers)
- **Bootstrap confidence intervals** (10,000 iterations)
- **Null model comparison** (phase-scrambled baselines)
- **Multiple testing correction** (Bonferroni)

## Visualization

Each conversation generates a comprehensive visualization including:
- Distance matrices and self-similarity plots
- Recurrence plots showing temporal patterns
- PCA trajectory projections
- Density evolution over time
- Phase detection results across models
- Transport metric comparisons
- Statistical summaries

## Testing

Run the test suite:
```bash
# All tests
python -m pytest tests/ -v

# Specific test categories
python -m pytest tests/test_transport_metrics.py -v
python -m pytest tests/test_pipeline_integration.py -v

# Skip slow tests
python -m pytest tests/ -m "not slow" -v
```

## Paper Compilation

To compile the research paper:
```bash
cd paper
pdflatex paper.tex
bibtex paper
pdflatex paper.tex
pdflatex paper.tex
```

Or using latexmk:
```bash
latexmk -pdf paper.tex
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.
