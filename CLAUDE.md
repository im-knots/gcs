# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This is a research project analyzing AI conversation dynamics through geometric and embedding trajectory analysis. It combines Python-based data analysis with an academic LaTeX paper documenting the findings. Sample conversation data can be found in `analysis/sample-json/`.

## Available Agents

The `.claude/agents` directory contains specialized agents that can be invoked using the Task tool:

### Research & Development
- **ai-researcher**: Multi-agent systems, emergent behaviors, AI social dynamics
- **critical-reviewer**: Research methodology, mathematical rigor, logical consistency
- **math-genius**: Mathematical proofs, theoretical foundations, statistical analysis
- **open-source-developer**: Production-ready implementations, collaborative development

### Engineering
- **go-performance-engineer**: Go optimization, zero-allocation design, concurrent systems
- **frontend-architect**: React/Next.js development, performance optimization, accessibility
- **kubernetes-wizard**: Container orchestration, cloud-native architecture, scaling

### Infrastructure & Operations
- **project-orchestrator**: Project planning, task decomposition, multi-agent coordination
- **infrastructure-maintainer**: DevOps, system reliability, production deployment
- **consensus-facilitator**: Collective decision-making, conflict resolution
- **privacy-advocate**: Privacy-preserving tech, anti-surveillance, data sovereignty

### Community & Culture
- **community-builder**: IRC channels, open source communities, knowledge sharing
- **ethics-guardian**: AI ethics, human rights, value alignment
- **knowledge-curator**: Documentation, learning resources, information organization

## Key Commands

### Python Analysis Environment

```bash
# Activate virtual environment
source analysis/conversation_analysis_env/bin/activate

# Install dependencies
pip install -r analysis/requirements.txt

# Run the main analysis pipeline
python analysis/run_analysis.py

# Run with specific options
python analysis/run_analysis.py --data-dir /path/to/data --output-dir results/ --max-conversations 10

# Additional options:
# --no-checkpoint: Disable checkpointing
# --clear-checkpoints: Clear existing checkpoints before starting
# --batch-size 25: Number of conversations per GPU batch
# --figure-format both: Save figures as 'png', 'pdf', or 'both'
# --log-level INFO: Set logging level (DEBUG, INFO, WARNING, ERROR)

# Run tests
python analysis/run_tests.py

# Run specific test
python -m pytest analysis/tests/test_pipeline.py -v

# Run tests excluding slow ones
python -m pytest analysis/tests -m "not slow" -v
```

### LaTeX Paper Compilation

```bash
# Build the paper (from paper/ directory)
cd paper
pdflatex paper.tex
bibtex paper
pdflatex paper.tex
pdflatex paper.tex

# Or use latexmk for automated builds
latexmk -pdf paper.tex

# Clean build files
latexmk -c paper.tex
```

## Architecture Overview

### Core Analysis Pipeline (`analysis/run_analysis.py`)

The `ConversationAnalysisPipeline` class orchestrates the entire analysis:

1. **Embedding Generation** (`embedding_analysis/core/embedder.py`)
   - `EnsembleEmbedder`: Uses multiple embedding models in ensemble
   - Models: MiniLM-L6 (384d), MPNet (768d), MiniLM-L12 (384d), Word2Vec (300d), GloVe (300d)
   - **GPU Required**: CUDA-capable GPU is mandatory for embedding generation

2. **Trajectory Analysis** (`embedding_analysis/core/trajectory.py`)
   - `TrajectoryAnalyzer`: Calculates velocity, acceleration, curvature in embedding space
   - Advanced metrics: adaptive normalization, ensemble curvature methods
   - Cross-model consistency measures

3. **Phase Detection** (`embedding_analysis/models/ensemble_phase_detector.py`)
   - `EnsemblePhaseDetector`: Detects conversation phase transitions
   - Consensus-based detection across multiple models
   - Comparison with annotated ground truth when available

4. **Geometric Invariance Analysis** (`embedding_analysis/core/geometric_invariance.py`)
   - `GeometricSignatureComputer`: Computes embedding-independent signatures
   - `InvarianceAnalyzer`: Tests patterns invariant across models
   - `HypothesisTester`: Statistical validation of invariance

5. **Advanced Components**
   - `HierarchicalHypothesisTester` (`core/hierarchical_hypothesis_testing.py`): Tiered statistical testing
   - `ParadigmSpecificNullModels` (`core/paradigm_null_models.py`): Null model generation
   - `MultiScaleAnalyzer` (`core/multiscale_analysis.py`): Multi-scale trajectory analysis
   - `TransportMetrics` (`core/transport_metrics.py`): Optimal transport-based metrics
   - `ControlAnalyses` (`core/control_analyses.py`): Control for confounding variables

### Data Flow

1. **Input**: JSON conversations from `/home/knots/git/the-academy/docs/paper/exp-data/`
   - Directories: `phase-1-premium`, `phase-2-efficient`, `phase-3-no-reasoning`
   
2. **Processing**:
   - Batch GPU processing (default batch size: 25 conversations)
   - Automatic checkpointing after each conversation
   - Resume capability from checkpoints
   
3. **Output** (`analysis_output/`):
   ```
   ├── checkpoints/         # Resumable analysis state (.pkl files)
   ├── figures/            # Visualizations
   │   ├── ensemble/       # Per-conversation ensemble plots
   │   ├── invariance/     # Cross-model correlation analysis
   │   └── multiscale/     # Multi-scale analysis figures
   ├── reports/            # Text analysis reports
   │   ├── invariance_analysis.txt
   │   ├── hypothesis_test_results.txt
   │   └── analysis_summary.txt
   └── logs/              # Detailed execution logs
   ```

### Testing Infrastructure

- **Test Files** (`analysis/tests/`):
  - `test_pipeline.py`: Integration tests
  - `test_hierarchical_hypothesis.py`: Statistical testing
  - `test_paradigm_null_models.py`: Null model validation
  - `test_invariance_analyzer.py`: Invariance metrics
  - `test_control_analyses.py`: Control analysis tests

- **Pytest Configuration** (`analysis/pytest.ini`):
  - Markers: `@pytest.mark.slow`, `@pytest.mark.integration`, `@pytest.mark.gpu`
  - Run specific markers: `pytest -m "not slow"`

## Important Considerations

- **GPU Requirement**: The analysis will fail without a CUDA-capable GPU
- **Memory Usage**: Large conversation sets are processed in batches to manage GPU memory
- **Checkpoint System**:
  - Saves after each conversation (including figure generation)
  - Contains embeddings, metrics, phase results, and invariance scores
  - Use `--clear-checkpoints` to start fresh
  - Use `--no-checkpoint` to disable
- **Figure Formats**: Supports PNG, PDF, or both via `--figure-format`
- **Embedding Models**: 
  - 3 transformer models (sentence-transformers)
  - 2 classical models (Word2Vec, GloVe via gensim)