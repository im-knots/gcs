#!/usr/bin/env python3
"""
Test checkpoint functionality for conversation analysis pipeline.
"""

import pytest
import numpy as np
from pathlib import Path
import shutil
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from embedding_analysis.utils.checkpoint import CheckpointManager


class TestCheckpointFunctionality:
    """Test checkpoint saving and loading."""
    
    @pytest.fixture
    def checkpoint_dir(self, tmp_path):
        """Create temporary checkpoint directory."""
        checkpoint_dir = tmp_path / "test_checkpoints"
        checkpoint_dir.mkdir()
        yield checkpoint_dir
        # Cleanup
        if checkpoint_dir.exists():
            shutil.rmtree(checkpoint_dir)
    
    @pytest.fixture
    def checkpoint_manager(self, checkpoint_dir):
        """Create checkpoint manager."""
        return CheckpointManager(checkpoint_dir)
    
    def test_save_and_load_conversation_checkpoint(self, checkpoint_manager):
        """Test saving and loading conversation checkpoint."""
        # Create sample checkpoint data
        checkpoint_data = {
            'session_id': 'test_session_001',
            'embeddings': {
                'model1': np.random.randn(50, 384),
                'model2': np.random.randn(50, 768)
            },
            'trajectory_metrics': {
                'velocity_mean': 1.2,
                'curvature_mean': 0.45
            },
            'phase_results': {
                'detected_phases': [
                    {'turn': 10, 'confidence': 0.85},
                    {'turn': 25, 'confidence': 0.92}
                ]
            },
            'invariance_metrics': {
                'pairwise_correlations': {
                    'model1-model2': 0.87
                }
            }
        }
        
        # Save checkpoint
        checkpoint_name = "conv_test_session_001"
        checkpoint_manager.save(
            checkpoint_data,
            checkpoint_name,
            metadata={
                'session_id': 'test_session_001',
                'n_messages': 50,
                'models': ['model1', 'model2']
            }
        )
        
        # Verify checkpoint exists
        assert checkpoint_manager.exists(checkpoint_name)
        
        # Load checkpoint
        loaded_data = checkpoint_manager.load(checkpoint_name)
        
        # Verify data integrity
        assert loaded_data is not None
        assert loaded_data['session_id'] == checkpoint_data['session_id']
        assert 'embeddings' in loaded_data
        assert 'model1' in loaded_data['embeddings']
        assert loaded_data['embeddings']['model1'].shape == (50, 384)
        assert loaded_data['trajectory_metrics']['velocity_mean'] == 1.2
        assert len(loaded_data['phase_results']['detected_phases']) == 2
        assert loaded_data['invariance_metrics']['pairwise_correlations']['model1-model2'] == 0.87
    
    def test_list_checkpoints(self, checkpoint_manager):
        """Test listing checkpoints."""
        # Save multiple checkpoints
        for i in range(3):
            checkpoint_manager.save(
                {'data': f'checkpoint_{i}'},
                f'conv_session_{i:03d}',
                metadata={'session_id': f'session_{i:03d}'}
            )
        
        # List checkpoints
        checkpoints = checkpoint_manager.list_checkpoints()
        
        # Verify
        assert len(checkpoints) == 3
        checkpoint_names = [cp['name'] for cp in checkpoints]
        assert 'conv_session_000' in checkpoint_names
        assert 'conv_session_001' in checkpoint_names
        assert 'conv_session_002' in checkpoint_names
    
    def test_checkpoint_metadata(self, checkpoint_manager):
        """Test checkpoint metadata handling."""
        checkpoint_name = "conv_test_metadata"
        metadata = {
            'session_id': 'test_123',
            'n_messages': 42,
            'models': ['all-MiniLM-L6-v2', 'all-mpnet-base-v2'],
            'phase_detected': True,
            'figures_generated': {'png': True, 'pdf': False}
        }
        
        # Save with metadata
        checkpoint_manager.save(
            {'test': 'data'},
            checkpoint_name,
            metadata=metadata
        )
        
        # Get metadata
        loaded_metadata = checkpoint_manager.get_metadata(checkpoint_name)
        
        # Verify
        assert loaded_metadata is not None
        assert loaded_metadata['session_id'] == 'test_123'
        assert loaded_metadata['n_messages'] == 42
        assert loaded_metadata['phase_detected'] is True
        assert loaded_metadata['figures_generated']['png'] is True
        assert loaded_metadata['figures_generated']['pdf'] is False
        assert 'timestamp' in loaded_metadata
        assert 'data_hash' in loaded_metadata
    
    def test_resume_analysis_simulation(self, checkpoint_manager):
        """Simulate resuming analysis from checkpoints."""
        # Simulate processing 5 conversations, checkpoint after each
        conversation_ids = ['conv_001', 'conv_002', 'conv_003', 'conv_004', 'conv_005']
        processed_results = []
        
        # Process first 3 conversations
        for i, conv_id in enumerate(conversation_ids[:3]):
            result = {
                'session_id': conv_id,
                'invariance_score': 0.8 + i * 0.02,
                'n_phases': i + 1
            }
            processed_results.append(result)
            
            # Save checkpoint
            checkpoint_manager.save(
                result,
                f'conv_{conv_id}',
                metadata={'session_id': conv_id, 'processed': True}
            )
        
        # Simulate restart - load existing checkpoints
        loaded_results = []
        already_processed = set()
        
        checkpoints = checkpoint_manager.list_checkpoints()
        for checkpoint in checkpoints:
            if checkpoint['name'].startswith('conv_'):
                # Extract session ID from checkpoint name
                # checkpoint['name'] is like 'conv_conv_001', so we need the last part
                parts = checkpoint['name'].split('_')
                if len(parts) >= 2:
                    session_id = '_'.join(parts[1:])  # rejoin in case session_id has underscores
                    already_processed.add(session_id)
                
                # Load checkpoint data
                data = checkpoint_manager.load(checkpoint['name'])
                if data:
                    loaded_results.append(data)
        
        # Verify we loaded the correct checkpoints
        assert len(already_processed) == 3
        assert 'conv_001' in already_processed
        assert 'conv_002' in already_processed
        assert 'conv_003' in already_processed
        
        # Process remaining conversations
        remaining = [cid for cid in conversation_ids if cid not in already_processed]
        assert remaining == ['conv_004', 'conv_005']
        
        # Verify loaded data matches original
        loaded_results.sort(key=lambda x: x['session_id'])
        processed_results.sort(key=lambda x: x['session_id'])
        
        for loaded, original in zip(loaded_results, processed_results):
            assert loaded['session_id'] == original['session_id']
            assert loaded['invariance_score'] == original['invariance_score']
            assert loaded['n_phases'] == original['n_phases']


if __name__ == "__main__":
    pytest.main([__file__, "-v"])