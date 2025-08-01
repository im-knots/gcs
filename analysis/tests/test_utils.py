"""
Test utilities and fixtures for conversation analysis tests.

Provides realistic data generation and common test helpers.
"""

import numpy as np
from typing import Dict, List, Tuple
import json
from pathlib import Path


class TestDataGenerator:
    """Generate realistic test data for various components."""
    
    def __init__(self, seed: int = 42):
        self.rng = np.random.RandomState(seed)
        
    def generate_conversation_embeddings(self, n_messages: int = 50, 
                                       embedding_dim: int = 384) -> np.ndarray:
        """
        Generate realistic conversation embeddings with temporal structure.
        
        Creates embeddings that simulate a conversation trajectory with:
        - Smooth transitions between messages
        - Occasional topic shifts
        - Realistic velocity and curvature patterns
        """
        # Start with random initial point
        embeddings = np.zeros((n_messages, embedding_dim))
        embeddings[0] = self.rng.randn(embedding_dim)
        
        # Generate trajectory with momentum
        velocity = self.rng.randn(embedding_dim) * 0.1
        
        for i in range(1, n_messages):
            # Add some acceleration (topic drift)
            acceleration = self.rng.randn(embedding_dim) * 0.02
            velocity += acceleration
            
            # Occasionally have topic shifts
            if self.rng.rand() < 0.1:  # 10% chance of topic shift
                velocity += self.rng.randn(embedding_dim) * 0.3
            
            # Update position
            embeddings[i] = embeddings[i-1] + velocity
            
            # Add some noise
            embeddings[i] += self.rng.randn(embedding_dim) * 0.05
            
            # Damping to prevent explosion
            velocity *= 0.95
        
        return embeddings
    
    def generate_model_ensemble_embeddings(self, n_messages: int = 50) -> Dict[str, np.ndarray]:
        """
        Generate embeddings for multiple models with realistic correlations.
        
        Returns embeddings for transformer and classical models with:
        - High within-paradigm correlation
        - Moderate cross-paradigm correlation
        - Appropriate dimensionalities
        """
        # Base trajectory
        base_trajectory = self.generate_conversation_embeddings(n_messages, 100)
        
        # Transformer models (384/768 dims)
        transformer_base = self.generate_conversation_embeddings(n_messages, 384)
        
        embeddings = {
            # Transformer models - high correlation
            'all-MiniLM-L6-v2': transformer_base + self.rng.randn(n_messages, 384) * 0.1,
            'all-mpnet-base-v2': np.hstack([
                transformer_base + self.rng.randn(n_messages, 384) * 0.12,
                self.rng.randn(n_messages, 384)  # 768 total
            ]),
            'all-distilroberta-v1': np.hstack([
                transformer_base + self.rng.randn(n_messages, 384) * 0.11,
                self.rng.randn(n_messages, 384)
            ]),
            
            # Classical models - high correlation within, moderate with transformers
            'word2vec': self._project_and_perturb(transformer_base, 300, noise_scale=0.3),
            'glove': self._project_and_perturb(transformer_base, 300, noise_scale=0.32)
        }
        
        return embeddings
    
    def _project_and_perturb(self, embeddings: np.ndarray, target_dim: int, 
                           noise_scale: float = 0.2) -> np.ndarray:
        """Project embeddings to different dimension with noise."""
        n_messages, source_dim = embeddings.shape
        
        # Random projection matrix
        projection = self.rng.randn(source_dim, target_dim)
        projection /= np.linalg.norm(projection, axis=0)
        
        # Project and add noise
        projected = embeddings @ projection
        projected += self.rng.randn(n_messages, target_dim) * noise_scale
        
        return projected
    
    def generate_conversation_data(self, n_messages: int = 50, 
                                 include_phases: bool = True) -> Dict:
        """Generate complete conversation data with metadata."""
        messages = []
        
        for i in range(n_messages):
            role = 'user' if i % 2 == 0 else 'assistant'
            content = f"Message {i} from {role}" + " ".join([
                f"word{j}" for j in range(self.rng.randint(10, 50))
            ])
            messages.append({
                'role': role,
                'content': content
            })
        
        conversation = {
            'metadata': {
                'session_id': f'test_session_{self.rng.randint(1000, 9999)}',
                'timestamp': 1234567890 + self.rng.randint(0, 86400),
                'type': self.rng.choice(['full_reasoning', 'light_reasoning', 'no_reasoning'])
            },
            'messages': messages
        }
        
        if include_phases:
            # Add realistic phase annotations
            phases = []
            phase_names = ['greeting', 'problem_statement', 'exploration', 'solution', 'conclusion']
            phase_points = sorted(self.rng.choice(range(5, n_messages-5), 
                                                size=min(4, n_messages//10), 
                                                replace=False))
            
            phases.append({'turn': 0, 'phase': phase_names[0]})
            for i, turn in enumerate(phase_points):
                phases.append({
                    'turn': turn,
                    'phase': phase_names[min(i+1, len(phase_names)-1)]
                })
            
            conversation['phases'] = phases
        
        return conversation
    
    def generate_invariance_results(self, n_conversations: int = 50) -> Dict:
        """Generate realistic invariance analysis results."""
        conversation_results = []
        
        # Generate correlations matching paper's findings
        for i in range(n_conversations):
            # Within-paradigm: high (0.80-0.95)
            transformer_corr = self.rng.uniform(0.80, 0.95)
            classical_corr = self.rng.uniform(0.85, 0.98)
            
            # Cross-paradigm: moderate (0.55-0.75)
            cross_corr = self.rng.uniform(0.55, 0.75)
            
            conv_result = {
                'session_id': f'session_{i}',
                'invariance_metrics': {
                    'mean_correlation': np.mean([transformer_corr, classical_corr, cross_corr]),
                    'pairwise_correlations': {
                        'all-MiniLM-L6-v2-all-mpnet-base-v2': transformer_corr,
                        'all-MiniLM-L6-v2-all-distilroberta-v1': transformer_corr - 0.02,
                        'all-mpnet-base-v2-all-distilroberta-v1': transformer_corr - 0.01,
                        'word2vec-glove': classical_corr,
                        'all-MiniLM-L6-v2-word2vec': cross_corr,
                        'all-mpnet-base-v2-glove': cross_corr + 0.02,
                    },
                    'invariance_score': np.mean([transformer_corr, classical_corr]) * 0.9
                },
                'trajectory_metrics': self._generate_trajectory_metrics()
            }
            conversation_results.append(conv_result)
        
        # Aggregate statistics
        all_scores = [r['invariance_metrics']['invariance_score'] for r in conversation_results]
        
        return {
            'conversation_results': conversation_results,
            'geometric_signatures': {f'session_{i}': {} for i in range(n_conversations)},
            'invariance_scores': {f'session_{i}': s for i, s in enumerate(all_scores)},
            'aggregate_statistics': {
                'mean_invariance': np.mean(all_scores),
                'std_invariance': np.std(all_scores),
                'median_invariance': np.median(all_scores),
                'confidence_interval': {
                    'lower': np.percentile(all_scores, 2.5),
                    'upper': np.percentile(all_scores, 97.5),
                    'confidence_level': 0.95
                },
                'n_conversations': n_conversations
            }
        }
    
    def _generate_trajectory_metrics(self) -> Dict:
        """Generate realistic trajectory metrics."""
        n_points = 50
        
        # Generate correlated velocity profiles
        base_velocity = np.abs(self.rng.randn(n_points)) * 0.5 + 0.3
        
        metrics = {
            'advanced': {
                'all-MiniLM-L6-v2': {
                    'velocities': base_velocity + self.rng.randn(n_points) * 0.05,
                    'curvatures': np.abs(self.rng.randn(n_points)) * 0.2
                },
                'all-mpnet-base-v2': {
                    'velocities': base_velocity + self.rng.randn(n_points) * 0.06,
                    'curvatures': np.abs(self.rng.randn(n_points)) * 0.2
                },
                'word2vec': {
                    'velocities': base_velocity + self.rng.randn(n_points) * 0.1,
                    'curvatures': np.abs(self.rng.randn(n_points)) * 0.25
                }
            },
            'curvature_ensemble': {
                'all-MiniLM-L6-v2': list(np.abs(self.rng.randn(n_points)) * 0.2),
                'all-mpnet-base-v2': list(np.abs(self.rng.randn(n_points)) * 0.2),
                'word2vec': list(np.abs(self.rng.randn(n_points)) * 0.25)
            },
            'consistency': {
                'distance_correlation': self.rng.uniform(0.7, 0.9)
            }
        }
        
        return metrics


class MockComponents:
    """Mock versions of analysis components for testing."""
    
    @staticmethod
    def mock_embedder():
        """Create a mock embedder that returns predictable embeddings."""
        class MockEmbedder:
            def embed_texts(self, texts, show_progress=False):
                n_texts = len(texts)
                return {
                    'all-MiniLM-L6-v2': np.random.randn(n_texts, 384),
                    'all-mpnet-base-v2': np.random.randn(n_texts, 768),
                    'word2vec': np.random.randn(n_texts, 300)
                }
        return MockEmbedder()
    
    @staticmethod
    def mock_trajectory_analyzer():
        """Create a mock trajectory analyzer."""
        class MockAnalyzer:
            def calculate_trajectory_metrics(self, embeddings):
                n_points = len(embeddings) - 1
                return {
                    'velocities': list(np.random.rand(n_points) * 0.5 + 0.2),
                    'curvatures': list(np.random.rand(n_points) * 0.3),
                    'total_distance': np.random.uniform(10, 50),
                    'mean_velocity': np.random.uniform(0.3, 0.8)
                }
            
            def calculate_ensemble_trajectories(self, embeddings):
                return {'consistency': {'mean_correlation': 0.85}}
            
            def analyze_trajectory_with_normalization(self, embeddings, method='adaptive'):
                return self.calculate_trajectory_metrics(embeddings)
            
            def calculate_curvature_ensemble(self, embeddings):
                return list(np.random.rand(len(embeddings)-2) * 0.3)
                
        return MockAnalyzer()


def assert_correlation_in_range(corr: float, min_val: float, max_val: float, 
                               name: str = "correlation"):
    """Assert a correlation is in expected range."""
    assert min_val <= corr <= max_val, \
        f"{name} {corr:.3f} not in range [{min_val}, {max_val}]"


def assert_hypothesis_result_valid(result):
    """Assert a hypothesis result has all required fields."""
    required_fields = ['name', 'passed', 'p_value', 'test_statistic', 
                      'effect_size', 'confidence_interval', 'power', 'description']
    
    for field in required_fields:
        assert field in result.__dict__, f"Missing field: {field}"
    
    # Check types
    assert isinstance(result.passed, bool)
    assert isinstance(result.p_value, (float, type(np.nan)))
    assert isinstance(result.confidence_interval, tuple)
    assert len(result.confidence_interval) == 2