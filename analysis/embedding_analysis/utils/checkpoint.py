"""
Checkpoint management for analysis workflows.
"""

import json
import pickle
from pathlib import Path
from typing import Any, Dict, Optional, List
import hashlib
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class CheckpointManager:
    """
    Manages checkpoints for resumable analysis workflows.
    """
    
    def __init__(self, checkpoint_dir: Optional[Path] = None):
        """
        Initialize checkpoint manager.
        
        Args:
            checkpoint_dir: Directory to store checkpoints
        """
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else Path('checkpoints')
        self.checkpoint_dir.mkdir(exist_ok=True)
        
    def save(self, 
             data: Any,
             checkpoint_name: str,
             metadata: Optional[Dict] = None) -> Path:
        """
        Save checkpoint data.
        
        Args:
            data: Data to checkpoint
            checkpoint_name: Name for checkpoint
            metadata: Optional metadata
            
        Returns:
            Path to saved checkpoint
        """
        checkpoint_path = self.checkpoint_dir / f"{checkpoint_name}.pkl"
        metadata_path = self.checkpoint_dir / f"{checkpoint_name}_metadata.json"
        
        # Save data
        with open(checkpoint_path, 'wb') as f:
            pickle.dump(data, f)
            
        # Save metadata
        if metadata is None:
            metadata = {}
            
        metadata.update({
            'checkpoint_name': checkpoint_name,
            'timestamp': datetime.now().isoformat(),
            'data_hash': self._compute_hash(checkpoint_path)
        })
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
            
        logger.info(f"Saved checkpoint: {checkpoint_name}")
        return checkpoint_path
        
    def load(self, checkpoint_name: str) -> Optional[Any]:
        """
        Load checkpoint data.
        
        Args:
            checkpoint_name: Name of checkpoint to load
            
        Returns:
            Loaded data or None if not found
        """
        checkpoint_path = self.checkpoint_dir / f"{checkpoint_name}.pkl"
        
        if not checkpoint_path.exists():
            logger.warning(f"Checkpoint not found: {checkpoint_name}")
            return None
            
        try:
            with open(checkpoint_path, 'rb') as f:
                data = pickle.load(f)
                
            logger.info(f"Loaded checkpoint: {checkpoint_name}")
            return data
            
        except Exception as e:
            logger.error(f"Error loading checkpoint {checkpoint_name}: {e}")
            return None
            
    def exists(self, checkpoint_name: str) -> bool:
        """Check if checkpoint exists."""
        checkpoint_path = self.checkpoint_dir / f"{checkpoint_name}.pkl"
        return checkpoint_path.exists()
        
    def get_metadata(self, checkpoint_name: str) -> Optional[Dict]:
        """Get checkpoint metadata."""
        metadata_path = self.checkpoint_dir / f"{checkpoint_name}_metadata.json"
        
        if not metadata_path.exists():
            return None
            
        try:
            with open(metadata_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading metadata for {checkpoint_name}: {e}")
            return None
            
    def list_checkpoints(self) -> List[Dict]:
        """List all available checkpoints."""
        checkpoints = []
        
        for pkl_file in self.checkpoint_dir.glob("*.pkl"):
            checkpoint_name = pkl_file.stem
            metadata = self.get_metadata(checkpoint_name)
            
            checkpoints.append({
                'name': checkpoint_name,
                'path': str(pkl_file),
                'metadata': metadata
            })
            
        return sorted(checkpoints, key=lambda x: x['name'])
        
    def cleanup(self, keep_latest: int = 5):
        """
        Clean up old checkpoints.
        
        Args:
            keep_latest: Number of latest checkpoints to keep
        """
        checkpoints = self.list_checkpoints()
        
        # Sort by timestamp if available
        checkpoints_with_time = []
        for cp in checkpoints:
            if cp['metadata'] and 'timestamp' in cp['metadata']:
                checkpoints_with_time.append(cp)
                
        checkpoints_with_time.sort(
            key=lambda x: x['metadata']['timestamp'],
            reverse=True
        )
        
        # Remove old checkpoints
        to_remove = checkpoints_with_time[keep_latest:]
        
        for cp in to_remove:
            checkpoint_path = Path(cp['path'])
            metadata_path = checkpoint_path.with_suffix('') + '_metadata.json'
            
            try:
                checkpoint_path.unlink()
                if metadata_path.exists():
                    metadata_path.unlink()
                logger.info(f"Removed old checkpoint: {cp['name']}")
            except Exception as e:
                logger.error(f"Error removing checkpoint {cp['name']}: {e}")
                
    def _compute_hash(self, file_path: Path) -> str:
        """Compute hash of file."""
        hash_md5 = hashlib.md5()
        
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
                
        return hash_md5.hexdigest()
        
    def create_session_checkpoint(self, 
                                session_id: str,
                                step: str,
                                data: Any) -> Path:
        """
        Create checkpoint for specific analysis session and step.
        
        Args:
            session_id: Unique session identifier
            step: Analysis step name
            data: Data to checkpoint
            
        Returns:
            Path to checkpoint
        """
        checkpoint_name = f"{session_id}_{step}"
        metadata = {
            'session_id': session_id,
            'step': step
        }
        
        return self.save(data, checkpoint_name, metadata)
        
    def load_session_checkpoint(self,
                              session_id: str,
                              step: str) -> Optional[Any]:
        """Load checkpoint for specific session and step."""
        checkpoint_name = f"{session_id}_{step}"
        return self.load(checkpoint_name)
        
    def get_session_progress(self, session_id: str) -> List[str]:
        """Get completed steps for a session."""
        completed_steps = []
        
        for cp in self.list_checkpoints():
            if cp['metadata'] and cp['metadata'].get('session_id') == session_id:
                step = cp['metadata'].get('step')
                if step:
                    completed_steps.append(step)
                    
        return completed_steps