#!/usr/bin/env python3
"""
Embedding Space Trajectory Analysis for AI Conversations

IMPORTANT: This analysis operates in EMBEDDING SPACE, not true semantic space.
Embedding models project text into high-dimensional vectors that approximate
semantic relationships, but these are learned representations with specific
biases and limitations. Results should be interpreted as patterns in the
embedding space, which may or may not reflect true semantic relationships.

Analyzes conversation trajectories through high-dimensional embedding space
to understand breakdown patterns and social dynamics.
Includes ensemble mode for invariant pattern detection across multiple embedding models.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy.spatial.distance import cosine, euclidean
from scipy.stats import pearsonr, spearmanr, f_oneway, kruskal, ttest_ind, mannwhitneyu
from scipy.stats import chi2_contingency, fisher_exact
from tqdm import tqdm
import warnings
import pickle
import hashlib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import LabelEncoder
import statsmodels.api as sm
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.power import FTestAnovaPower
warnings.filterwarnings('ignore')


class EmbeddingTrajectoryAnalyzer:
    """Analyze conversation trajectories through embedding space (not true semantic space)"""
    
    def __init__(self, model_name='all-MiniLM-L6-v2', output_dir='embedding_analysis', cache_dir=None, ensemble_models=None, checkpoint_dir='checkpoints'):
        """
        Initialize the analyzer with a sentence transformer model.
        
        Args:
            model_name: Name of the primary sentence transformer model to use
                       Options: 'all-MiniLM-L6-v2' (faster, 384 dim)
                               'all-mpnet-base-v2' (better, 768 dim) 
                               'all-MiniLM-L12-v2' (balanced, 384 dim)
            output_dir: Directory to save analysis outputs
            cache_dir: Directory to cache models (helps with download issues)
            ensemble_models: List of additional models for ensemble analysis
                           Format: [{'name': str, 'model_id': str, 'dim': int}, ...]
            checkpoint_dir: Directory to save checkpoints for resuming interrupted runs
        """
        # Load primary model
        print(f"Loading primary embedding model: {model_name}")
        print("This may take a few minutes on first run to download the model...")
        
        try:
            # Set cache directory if provided
            if cache_dir:
                import os
                os.environ['SENTENCE_TRANSFORMERS_HOME'] = cache_dir
            
            # Load model with GPU support
            import torch
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.encoder = SentenceTransformer(model_name, device=device)
            
            if device == 'cuda':
                print(f"✓ GPU detected! Using {torch.cuda.get_device_name(0)} for acceleration")
                # Set optimal batch size for RTX 4070 Super (12GB VRAM)
                self.gpu_batch_size = 256  # RTX 4070 Super can handle large batches
            else:
                print("⚠️  No GPU detected, using CPU (this will be slower)")
                self.gpu_batch_size = 32
            
        except Exception as e:
            print(f"Error loading {model_name}: {e}")
            print("Trying alternative model 'all-MiniLM-L6-v2'...")
            
            # Fallback to a smaller, more reliable model
            try:
                import torch
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
                self.encoder = SentenceTransformer('all-MiniLM-L6-v2', device=device)
                model_name = 'all-MiniLM-L6-v2'
                if device == 'cuda':
                    print(f"✓ Fallback model loaded on GPU")
                    self.gpu_batch_size = 256
                else:
                    self.gpu_batch_size = 32
            except Exception as e2:
                print(f"Failed to load fallback model: {e2}")
                print("\nTroubleshooting steps:")
                print("1. Check your internet connection")
                print("2. Try running with a specific cache directory:")
                print("   analyzer = EmbeddingTrajectoryAnalyzer(cache_dir='./model_cache')")
                print("3. Or manually download the model first:")
                print("   from sentence_transformers import SentenceTransformer")
                print("   model = SentenceTransformer('all-MiniLM-L6-v2')")
                print("   model.save('./local_model')")
                print("   Then use: analyzer = EmbeddingTrajectoryAnalyzer(model_name='./local_model')")
                raise
        
        self.model_name = model_name
        self.embedding_dim = self.encoder.get_sentence_embedding_dimension()
        print(f"Successfully loaded {model_name} with {self.embedding_dim} dimensions")
        
        # Load ensemble models if specified
        self.ensemble_mode = ensemble_models is not None
        self.ensemble_models = {}
        
        if self.ensemble_mode:
            print("\nLoading ensemble models for invariant pattern detection...")
            # Add primary model to ensemble
            self.ensemble_models[model_name] = self.encoder
            
            # Load additional models
            for config in ensemble_models:
                print(f"  Loading {config['name']} ({config['model_id']})...")
                try:
                    import torch
                    device = 'cuda' if torch.cuda.is_available() else 'cpu'
                    model = SentenceTransformer(config['model_id'], device=device)
                    self.ensemble_models[config['name']] = model
                    print(f"    ✓ Loaded with {config['dim']} dimensions on {device.upper()}")
                except Exception as e:
                    print(f"    ✗ Failed to load: {e}")
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Setup checkpoint directory
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        # Storage for analysis results
        self.conversations = []
        self.all_embeddings = []
        self.all_metadata = []
        self.ensemble_embeddings = {} if self.ensemble_mode else None
    
    def _safe_euclidean(self, vec1, vec2):
        """Compute euclidean distance with proper shape handling"""
        vec1 = np.asarray(vec1).flatten()
        vec2 = np.asarray(vec2).flatten()
        return euclidean(vec1, vec2)
    
    def _safe_cosine(self, vec1, vec2):
        """Compute cosine distance with proper shape handling"""
        vec1 = np.asarray(vec1).flatten()
        vec2 = np.asarray(vec2).flatten()
        return cosine(vec1, vec2)
        
    def load_conversation(self, json_path, verbose=False):
        """Load a conversation from The Academy export format"""
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        # Debug: print available keys only if verbose
        if verbose:
            print(f"\nLoading {json_path}")
            print(f"Available keys: {list(data.keys())[:10]}...")  # Show first 10 keys
        
        # The JSON has 'session' at the root level
        if 'session' in data:
            session_data = data['session']
            messages = session_data.get('messages', [])
            session_id = session_data.get('id', Path(json_path).stem)
            
            # Also get analysisHistory from the root level
            analysis_history = data.get('analysisHistory', [])
        else:
            # Fallback for other formats
            messages = data.get('messages', [])
            session_id = data.get('sessionId', Path(json_path).stem)
            analysis_history = data.get('analysisHistory', [])
        
        # Extract conversation metadata
        conv_metadata = {
            'session_id': session_id,
            'export_date': data.get('exportedAt', datetime.now().isoformat()),
            'message_count': len(messages),
        }
        
        # Extract and process messages
        processed_messages = []
        for i, msg in enumerate(messages):
            content = msg.get('content', '')
            speaker = msg.get('participantName', f'Speaker_{i}')
            timestamp = msg.get('timestamp', '')
            
            processed_messages.append({
                'content': content,
                'speaker': speaker,
                'timestamp': timestamp,
                'turn': i,
                'session_id': session_id
            })
        
        # Extract conversation phases from analysis history
        phases = []
        phase_set = set()  # To avoid duplicates
        
        if analysis_history and processed_messages:
            # Create a list of message timestamps for mapping
            message_timestamps = [(i, msg['timestamp']) for i, msg in enumerate(processed_messages) if msg.get('timestamp')]
            
            for analysis in analysis_history:
                # Get the phase from either direct attribute or nested in analysis
                phase = None
                analysis_timestamp = analysis.get('timestamp')
                
                # Check direct conversationPhase attribute first
                if 'conversationPhase' in analysis:
                    phase = analysis['conversationPhase']
                # Then check nested in analysis object
                elif 'analysis' in analysis and 'conversationPhase' in analysis['analysis']:
                    phase = analysis['analysis']['conversationPhase']
                
                if phase and analysis_timestamp:
                    # Find the closest message turn to this analysis timestamp
                    # Parse timestamps
                    import datetime as dt
                    try:
                        analysis_dt = dt.datetime.fromisoformat(analysis_timestamp.replace('Z', '+00:00'))
                        
                        best_turn = 0
                        min_diff = float('inf')
                        
                        for turn, msg_timestamp in message_timestamps:
                            if msg_timestamp:
                                msg_dt = dt.datetime.fromisoformat(msg_timestamp.replace('Z', '+00:00'))
                                diff = abs((analysis_dt - msg_dt).total_seconds())
                                if diff < min_diff:
                                    min_diff = diff
                                    best_turn = turn
                        
                        # Create unique key to avoid duplicates at same turn
                        phase_key = f"{best_turn}:{phase}"
                        
                        if phase_key not in phase_set:
                            phases.append({
                                'turn': best_turn,
                                'phase': phase,
                                'analysis_time': analysis_timestamp
                            })
                            phase_set.add(phase_key)
                    except:
                        # Fallback to messageCountAtAnalysis if timestamp parsing fails
                        turn = analysis.get('messageCountAtAnalysis', 0)
                        phase_key = f"{turn}:{phase}"
                        if turn > 0 and phase_key not in phase_set:
                            phases.append({
                                'turn': turn,
                                'phase': phase
                            })
                            phase_set.add(phase_key)
        
        # Sort phases by turn number
        phases = sorted(phases, key=lambda x: x['turn'])
        
        if verbose:
            print(f"Loaded {len(processed_messages)} messages from {session_id[:20]}...")
            if phases:
                print(f"Found {len(phases)} conversation phase markers")
        
        return {
            'metadata': conv_metadata,
            'messages': processed_messages,
            'phases': phases
        }
    
    def _determine_outcome(self, data):
        """Placeholder for compatibility - not used in embedding analysis"""
        return 'conversation'
    
    def save_checkpoint(self, checkpoint_name, data):
        """Save checkpoint data to disk for resuming interrupted runs"""
        checkpoint_path = self.checkpoint_dir / f"{checkpoint_name}.pkl"
        temp_path = checkpoint_path.with_suffix('.pkl.tmp')
        
        try:
            # Add metadata to checkpoint
            checkpoint_data = {
                'timestamp': datetime.now().isoformat(),
                'version': '2.0',  # Checkpoint format version
                'data': data
            }
            
            # Save to temporary file first
            with open(temp_path, 'wb') as f:
                pickle.dump(checkpoint_data, f)
            
            # Atomically move to final location
            temp_path.replace(checkpoint_path)
            # Silently save checkpoints during processing
        except Exception as e:
            print(f"Failed to save checkpoint {checkpoint_name}: {e}")
            if temp_path.exists():
                temp_path.unlink()
    
    def load_checkpoint(self, checkpoint_name, verbose=True):
        """Load checkpoint data from disk"""
        checkpoint_path = self.checkpoint_dir / f"{checkpoint_name}.pkl"
        
        if checkpoint_path.exists():
            try:
                with open(checkpoint_path, 'rb') as f:
                    checkpoint_data = pickle.load(f)
                
                # Only handle new checkpoint format
                if isinstance(checkpoint_data, dict) and checkpoint_data.get('version') == '2.0':
                    if verbose:
                        print(f"Checkpoint loaded: {checkpoint_name} (v{checkpoint_data['version']})")
                    return checkpoint_data['data']
                else:
                    if verbose:
                        print(f"Incompatible checkpoint format, starting fresh")
                    return None
            except Exception as e:
                if verbose:
                    print(f"Failed to load checkpoint {checkpoint_name}: {e}")
                return None
        return None
    
    def get_conversation_hash(self, conversation_path):
        """Generate unique hash for a conversation file to track processing"""
        # Hash based on file path and modification time
        path_str = str(conversation_path.absolute())
        mtime = conversation_path.stat().st_mtime
        hash_input = f"{path_str}:{mtime}".encode()
        return hashlib.md5(hash_input).hexdigest()
    
    def save_conversation_checkpoint(self, conversation, analysis_step=None):
        """Save checkpoint for an individual conversation analysis"""
        session_id = conversation['metadata']['session_id']
        checkpoint_name = f"conv_{session_id[:12]}"
        
        # Track which analysis steps are completed
        if 'analysis_completed' not in conversation:
            conversation['analysis_completed'] = set()
        
        if analysis_step:
            conversation['analysis_completed'].add(analysis_step)
        
        # Update cache if it exists
        if hasattr(self, '_checkpoint_cache'):
            self._checkpoint_cache[session_id] = conversation.copy()
        
        # Save the conversation state
        self.save_checkpoint(checkpoint_name, conversation)
    
    def load_conversation_checkpoint(self, session_id):
        """Load checkpoint for a specific conversation"""
        # Check cache first
        if hasattr(self, '_checkpoint_cache') and session_id in self._checkpoint_cache:
            return self._checkpoint_cache[session_id]
        
        checkpoint_name = f"conv_{session_id[:12]}"
        return self.load_checkpoint(checkpoint_name, verbose=False)
    
    def load_all_conversation_checkpoints(self):
        """Load all conversation checkpoints into memory cache"""
        print("Loading conversation checkpoints into memory...")
        self._checkpoint_cache = {}
        
        conv_checkpoints = list(self.checkpoint_dir.glob("conv_*.pkl"))
        
        if conv_checkpoints:
            print(f"Found {len(conv_checkpoints)} conversation checkpoints")
            
            for checkpoint_path in tqdm(conv_checkpoints, desc="Loading checkpoints", unit="file"):
                try:
                    with open(checkpoint_path, 'rb') as f:
                        checkpoint_data = pickle.load(f)
                    
                    if isinstance(checkpoint_data, dict) and checkpoint_data.get('version') == '2.0':
                        conv_data = checkpoint_data['data']
                        if 'metadata' in conv_data and 'session_id' in conv_data['metadata']:
                            session_id = conv_data['metadata']['session_id']
                            self._checkpoint_cache[session_id] = conv_data
                except Exception as e:
                    print(f"Failed to load {checkpoint_path.name}: {e}")
            
            print(f"Loaded {len(self._checkpoint_cache)} conversation checkpoints into cache")
        else:
            print("No conversation checkpoints found")
        
        return self._checkpoint_cache
    
    def get_completed_analysis_steps(self, conversation):
        """Get list of completed analysis steps for a conversation"""
        if 'analysis_completed' in conversation:
            return conversation['analysis_completed']
        return set()
    
    def cleanup_old_checkpoints(self, keep_latest=5):
        """Remove old checkpoint files, keeping only the most recent ones"""
        # Clean up main analysis checkpoints
        main_checkpoints = list(self.checkpoint_dir.glob("analysis_state*.pkl"))
        if len(main_checkpoints) > keep_latest:
            # Sort by modification time
            main_checkpoints.sort(key=lambda p: p.stat().st_mtime)
            # Remove oldest
            for checkpoint in main_checkpoints[:-keep_latest]:
                checkpoint.unlink()
                print(f"Removed old checkpoint: {checkpoint.name}")
    
    def cleanup_conversation_checkpoints(self, processed_conversations=None):
        """Clean up individual conversation checkpoints after successful processing"""
        conv_checkpoints = list(self.checkpoint_dir.glob("conv_*.pkl"))
        
        if processed_conversations:
            # Remove checkpoints for successfully processed conversations
            for conv in processed_conversations:
                session_id = conv['metadata']['session_id']
                checkpoint_name = f"conv_{session_id[:12]}.pkl"
                checkpoint_path = self.checkpoint_dir / checkpoint_name
                if checkpoint_path.exists():
                    checkpoint_path.unlink()
        else:
            # Remove all conversation checkpoints
            for checkpoint in conv_checkpoints:
                checkpoint.unlink()
        
        print(f"Cleaned up {len(conv_checkpoints)} conversation checkpoints")
    
    def embed_conversations_batch(self, conversations, verbose=False):
        """Batch process multiple conversations for GPU efficiency"""
        if not conversations:
            return conversations
        
        # Collect all messages across conversations
        all_contents = []
        conversation_boundaries = [0]
        
        for conv in conversations:
            messages = conv['messages']
            all_contents.extend([msg['content'] for msg in messages])
            conversation_boundaries.append(conversation_boundaries[-1] + len(messages))
        
        if verbose:
            print(f"Batch embedding {len(all_contents)} messages from {len(conversations)} conversations...")
        
        # Batch encode all messages at once
        batch_size = getattr(self, 'gpu_batch_size', 32)
        
        # Show progress only if verbose
        import torch
        if torch.cuda.is_available() and verbose:
            print(f"Using GPU batch size: {batch_size}")
        
        all_embeddings = self.encoder.encode(all_contents, 
                                           batch_size=batch_size,
                                           show_progress_bar=verbose,
                                           convert_to_numpy=True)
        
        # Distribute embeddings back to conversations
        for i, conv in enumerate(conversations):
            start_idx = conversation_boundaries[i]
            end_idx = conversation_boundaries[i + 1]
            
            conv_embeddings = all_embeddings[start_idx:end_idx]
            messages = conv['messages']
            
            # Store embeddings with metadata
            embedded_messages = []
            for j, (msg, embedding) in enumerate(zip(messages, conv_embeddings)):
                embedded_msg = msg.copy()
                embedded_msg['embedding'] = embedding
                embedded_msg['embedding_norm'] = np.linalg.norm(embedding)
                embedded_messages.append(embedded_msg)
                
                # Add to global collections
                self.all_embeddings.append(embedding)
                self.all_metadata.append({
                    'session_id': msg['session_id'],
                    'turn': msg['turn'],
                    'speaker': msg['speaker']
                })
            
            conv['embedded_messages'] = embedded_messages
            
            # Handle ensemble models if needed
            if self.ensemble_mode:
                ensemble_embeddings = {}
                # Store primary model embeddings
                ensemble_embeddings[self.model_name] = conv_embeddings
                
                # Process other models (we'll batch these too)
                for model_name, model in self.ensemble_models.items():
                    if model_name != self.model_name:
                        contents = [msg['content'] for msg in messages]
                        model_embeddings = model.encode(contents, 
                                                       batch_size=batch_size,
                                                       show_progress_bar=False,
                                                       convert_to_numpy=True)
                        ensemble_embeddings[model_name] = model_embeddings
                
                conv['ensemble_embeddings'] = ensemble_embeddings
        
        return conversations
    
    def embed_conversation(self, conversation, verbose=False):
        """Generate embeddings for all messages in a conversation"""
        messages = conversation['messages']
        
        if verbose:
            print(f"Embedding {len(messages)} messages from session {conversation['metadata']['session_id'][:20]}...")
        
        # Batch encode with GPU optimization
        contents = [msg['content'] for msg in messages]
        
        # Use larger batch size for GPU if available
        batch_size = getattr(self, 'gpu_batch_size', 32)
        embeddings = self.encoder.encode(contents, 
                                        batch_size=batch_size,
                                        show_progress_bar=verbose,
                                        convert_to_numpy=True)
        
        # Store embeddings with metadata
        embedded_messages = []
        for i, (msg, embedding) in enumerate(zip(messages, embeddings)):
            embedded_msg = msg.copy()
            embedded_msg['embedding'] = embedding
            embedded_msg['embedding_norm'] = np.linalg.norm(embedding)
            embedded_messages.append(embedded_msg)
            
            # Add to global collections
            self.all_embeddings.append(embedding)
            self.all_metadata.append({
                'session_id': msg['session_id'],
                'turn': msg['turn'],
                'speaker': msg['speaker']
            })
        
        conversation['embedded_messages'] = embedded_messages
        
        # If ensemble mode, generate embeddings for all models
        if self.ensemble_mode:
            ensemble_embeddings = {}
            
            for model_name, model in self.ensemble_models.items():
                if model_name != self.model_name:  # Skip primary model (already done)
                    model_embeddings = model.encode(contents, 
                                                   batch_size=batch_size,
                                                   show_progress_bar=False,
                                                   convert_to_numpy=True)
                    ensemble_embeddings[model_name] = model_embeddings
                else:
                    ensemble_embeddings[model_name] = embeddings
            
            conversation['ensemble_embeddings'] = ensemble_embeddings
        
        return conversation
    
    def calculate_trajectory_metrics(self, conversation):
        """Calculate embedding trajectory metrics for a conversation"""
        messages = conversation['embedded_messages']
        metrics = []
        
        
        for i in range(1, len(messages)):
            prev = messages[i-1]
            curr = messages[i]
            
            # Embedding distance (Euclidean)
            euclidean_dist = self._safe_euclidean(prev['embedding'], curr['embedding'])
            
            # Cosine distance (1 - cosine similarity)
            cosine_dist = self._safe_cosine(prev['embedding'], curr['embedding'])
            
            # Direction of movement
            # Ensure embeddings are numpy arrays
            curr_emb = np.asarray(curr['embedding'])
            prev_emb = np.asarray(prev['embedding'])
            movement_vector = curr_emb - prev_emb
            
            # Embedding velocity (normalized by time if timestamps available)
            velocity = euclidean_dist  # Could divide by time delta
            
            metrics.append({
                'turn': curr['turn'],
                'speaker': curr['speaker'],
                'euclidean_distance': euclidean_dist,
                'cosine_distance': cosine_dist,
                'embedding_velocity': velocity,
                'movement_vector': movement_vector
            })
        
        conversation['trajectory_metrics'] = metrics
        
        # Calculate aggregate statistics with length normalization
        if metrics:
            distances = [m['euclidean_distance'] for m in metrics]
            n_messages = len(conversation['embedded_messages'])
            
            # Length normalization factors
            # Sqrt normalization for extensive properties that grow with sqrt(n) due to random walk behavior
            sqrt_norm = np.sqrt(n_messages / 100.0)  # Normalize to 100-turn baseline
            linear_norm = n_messages / 100.0  # For properties that scale linearly
            
            # Validate normalization is appropriate
            if n_messages < 20:
                print(f"Warning: Conversation has only {n_messages} messages. Length normalization may be unreliable.")
            
            conversation['trajectory_stats'] = {
                'mean_distance': np.mean(distances),
                'std_distance': np.std(distances),
                'max_distance': np.max(distances),
                'total_distance': np.sum(distances),
                'total_distance_normalized': np.sum(distances) / sqrt_norm,  # Normalize by sqrt(n)
                'distance_acceleration': np.diff(distances).mean() if len(distances) > 1 else 0,
                'conversation_length': n_messages,
                'length_norm_sqrt': sqrt_norm,
                'length_norm_linear': linear_norm
            }
        
        return conversation
    
    def calculate_phase_transition_metrics(self, conversation):
        """Calculate metrics around conversation phase transitions with confidence intervals"""
        messages = conversation['embedded_messages']
        phases = conversation.get('phases', [])
        
        if not phases:
            return None
        
        # Sort phases by turn number
        phases_sorted = sorted(phases, key=lambda x: x['turn'])
        
        # Map messages to phases
        phase_embeddings = {}
        for i, phase in enumerate(phases_sorted):
            phase_name = phase['phase']
            phase_turn = phase['turn']
            
            # Get messages in this phase (until next phase or end)
            next_phase_turn = float('inf')
            if i + 1 < len(phases_sorted):
                next_phase_turn = phases_sorted[i + 1]['turn']
                
            phase_messages = [m for m in messages 
                             if phase_turn <= m['turn'] < next_phase_turn]
            
            if phase_messages:
                # Calculate phase centroid
                embeddings = [m['embedding'] for m in phase_messages]
                phase_embeddings[phase_name] = {
                    'centroid': np.mean(embeddings, axis=0),
                    'spread': np.std(embeddings, axis=0).mean(),
                    'message_count': len(phase_messages),
                    'turn_range': (phase_turn, min(next_phase_turn - 1, messages[-1]['turn'])),
                    'start_turn': phase_turn,
                    'embeddings': embeddings  # Store for visualization
                }
        
        # Calculate inter-phase distances with bootstrap confidence intervals
        phase_transitions = []
        phase_names = list(phase_embeddings.keys())
        
        for i in range(len(phase_names)-1):
            curr_phase = phase_names[i]
            next_phase = phase_names[i+1]
            
            # Get embeddings for both phases
            curr_embeddings = phase_embeddings[curr_phase]['embeddings']
            next_embeddings = phase_embeddings[next_phase]['embeddings']
            
            # Calculate transition distance
            transition_distance = self._safe_euclidean(
                phase_embeddings[curr_phase]['centroid'],
                phase_embeddings[next_phase]['centroid']
            )
            
            # Bootstrap for confidence intervals
            n_bootstrap = 1000
            bootstrap_distances = []
            
            for _ in range(n_bootstrap):
                # Resample embeddings from each phase
                curr_sample = np.random.choice(len(curr_embeddings), len(curr_embeddings), replace=True)
                next_sample = np.random.choice(len(next_embeddings), len(next_embeddings), replace=True)
                
                curr_centroid_boot = np.mean([curr_embeddings[j] for j in curr_sample], axis=0)
                next_centroid_boot = np.mean([next_embeddings[j] for j in next_sample], axis=0)
                
                boot_distance = self._safe_euclidean(curr_centroid_boot, next_centroid_boot)
                bootstrap_distances.append(boot_distance)
            
            # Calculate confidence intervals
            ci_lower = np.percentile(bootstrap_distances, 2.5)
            ci_upper = np.percentile(bootstrap_distances, 97.5)
            
            phase_transitions.append({
                'from': curr_phase,
                'to': next_phase,
                'distance': transition_distance,
                'distance_ci_lower': ci_lower,
                'distance_ci_upper': ci_upper,
                'spread_change': phase_embeddings[next_phase]['spread'] - phase_embeddings[curr_phase]['spread'],
                'turn_boundary': phase_embeddings[next_phase]['start_turn']
            })
        
        return {
            'phase_embeddings': phase_embeddings,
            'transitions': phase_transitions,
            'phases_sorted': phases_sorted
        }
    
    def calculate_trajectory_curvature(self, conversation):
        """Calculate curvature and acceleration of embedding trajectory with confidence intervals"""
        messages = conversation['embedded_messages']
        n_messages = len(messages)
        
        if n_messages < 3:
            return None
        
        curvatures = []
        accelerations = []
        
        for i in range(2, n_messages):
            # Get three consecutive points
            p1 = messages[i-2]['embedding']
            p2 = messages[i-1]['embedding']
            p3 = messages[i]['embedding']
            
            # Calculate vectors
            v1 = p2 - p1
            v2 = p3 - p2
            
            # Curvature: angle between consecutive movement vectors
            v1_norm = np.linalg.norm(v1)
            v2_norm = np.linalg.norm(v2)
            
            if v1_norm > 0 and v2_norm > 0:
                cos_angle = np.dot(v1, v2) / (v1_norm * v2_norm)
                angle = np.arccos(np.clip(cos_angle, -1, 1))
                curvatures.append(angle)
                
                # Acceleration: change in velocity
                acceleration = v2_norm - v1_norm
                accelerations.append(acceleration)
        
        if not curvatures:
            return None
        
        # Bootstrap for confidence intervals
        n_bootstrap = 1000
        bootstrap_means = []
        bootstrap_maxs = []
        
        for _ in range(n_bootstrap):
            # Resample curvatures with replacement
            indices = np.random.choice(len(curvatures), len(curvatures), replace=True)
            boot_curvatures = [curvatures[i] for i in indices]
            
            bootstrap_means.append(np.mean(boot_curvatures))
            bootstrap_maxs.append(np.max(boot_curvatures))
        
        # Length normalization factor
        # Normalize by sqrt(n) for extensive properties (total curvature grows with length)
        # Keep intensive properties (mean curvature) as is
        length_norm_factor = np.sqrt(n_messages / 100.0)  # Normalize to 100-turn baseline
            
        return {
            'mean_curvature': np.mean(curvatures),
            'mean_curvature_ci_lower': np.percentile(bootstrap_means, 2.5),
            'mean_curvature_ci_upper': np.percentile(bootstrap_means, 97.5),
            'mean_curvature_normalized': np.mean(curvatures) / length_norm_factor,
            'max_curvature': np.max(curvatures),
            'max_curvature_ci_lower': np.percentile(bootstrap_maxs, 2.5),
            'max_curvature_ci_upper': np.percentile(bootstrap_maxs, 97.5),
            'curvature_variance': np.var(curvatures),
            'mean_acceleration': np.mean(accelerations),
            'acceleration_variance': np.var(accelerations),
            'n_turns': n_messages,
            'length_normalization_factor': length_norm_factor
        }
    
    def calculate_embedding_coherence_windows(self, conversation, window_size=5):
        """Calculate embedding coherence over sliding windows"""
        messages = conversation['embedded_messages']
        
        if len(messages) < window_size:
            return []
        
        coherence_scores = []
        for i in range(len(messages) - window_size + 1):
            window = messages[i:i+window_size]
            window_embeddings = [m['embedding'] for m in window]
            
            # Calculate pairwise similarities within window
            similarities = []
            for j in range(len(window_embeddings)):
                for k in range(j+1, len(window_embeddings)):
                    sim = 1 - self._safe_cosine(window_embeddings[j], window_embeddings[k])
                    similarities.append(sim)
            
            coherence_scores.append({
                'window_start': i,
                'window_end': i + window_size - 1,
                'mean_similarity': np.mean(similarities),
                'min_similarity': np.min(similarities),
                'similarity_variance': np.var(similarities)
            })
        
        return coherence_scores
    
    def detect_convergence_with_confidence(self, conversation):
        """Detect convergence regions in trajectory with confidence scores"""
        messages = conversation['embedded_messages']
        
        if len(messages) < 5:
            return None
        
        embeddings = np.array([m['embedding'] for m in messages])
        
        # Calculate velocities
        velocities = []
        for i in range(len(embeddings)-1):
            velocity = np.linalg.norm(embeddings[i+1] - embeddings[i])
            velocities.append(velocity)
        velocities = np.array(velocities)
        
        # Detect convergence using sliding window
        window_size = 5
        convergence_scores = []
        
        for i in range(len(velocities) - window_size + 1):
            window_velocities = velocities[i:i+window_size]
            
            # Convergence score: inverse of mean velocity in window
            mean_velocity = np.mean(window_velocities)
            velocity_variance = np.var(window_velocities)
            
            # Score is high when velocity is low and consistent
            if mean_velocity > 0:
                score = 1.0 / (mean_velocity * (1 + velocity_variance))
            else:
                score = float('inf')
            
            # Bootstrap for confidence interval
            n_bootstrap = 100
            bootstrap_scores = []
            
            for _ in range(n_bootstrap):
                # Resample velocities in window
                boot_indices = np.random.choice(window_size, window_size, replace=True)
                boot_velocities = window_velocities[boot_indices]
                
                boot_mean = np.mean(boot_velocities)
                boot_var = np.var(boot_velocities)
                
                if boot_mean > 0:
                    boot_score = 1.0 / (boot_mean * (1 + boot_var))
                    bootstrap_scores.append(boot_score)
            
            if bootstrap_scores:
                ci_lower = np.percentile(bootstrap_scores, 2.5)
                ci_upper = np.percentile(bootstrap_scores, 97.5)
            else:
                ci_lower = ci_upper = score
            
            convergence_scores.append({
                'turn': i + window_size // 2,  # Center of window
                'score': score,
                'score_ci_lower': ci_lower,
                'score_ci_upper': ci_upper,
                'mean_velocity': mean_velocity,
                'velocity_variance': velocity_variance
            })
        
        # Find significant convergence regions
        threshold = np.percentile([s['score'] for s in convergence_scores], 75)
        convergent_regions = [s for s in convergence_scores if s['score'] > threshold]
        
        # Length normalization for convergence detection
        n_messages = len(messages)
        expected_convergences = n_messages / 50  # Expect one convergence per 50 turns
        normalized_convergence_rate = len(convergent_regions) / expected_convergences
        
        return {
            'convergence_scores': convergence_scores,
            'convergent_regions': convergent_regions,
            'max_convergence_score': max(s['score'] for s in convergence_scores),
            'n_convergent_regions': len(convergent_regions),
            'normalized_convergence_rate': normalized_convergence_rate,
            'conversation_length': n_messages
        }
    
    def analyze_speaker_dynamics(self, conversation):
        """Analyze speaker interaction patterns beyond convergence"""
        messages = conversation['embedded_messages']
        
        # Speaker statistics
        speaker_stats = {}
        for msg in messages:
            speaker = msg['speaker']
            if speaker not in speaker_stats:
                speaker_stats[speaker] = {
                    'message_count': 0,
                    'total_movement': 0,
                    'embeddings': [],
                    'turns': []
                }
            speaker_stats[speaker]['message_count'] += 1
            speaker_stats[speaker]['embeddings'].append(msg['embedding'])
            speaker_stats[speaker]['turns'].append(msg['turn'])
        
        # Calculate embedding territory for each speaker
        for speaker, stats in speaker_stats.items():
            embeddings = np.array(stats['embeddings'])
            stats['embedding_centroid'] = embeddings.mean(axis=0)
            stats['embedding_spread'] = embeddings.std(axis=0).mean()
            
            # Calculate total trajectory length for this speaker
            if len(embeddings) > 1:
                distances = [self._safe_euclidean(embeddings[i], embeddings[i+1]) 
                            for i in range(len(embeddings)-1)]
                stats['total_movement'] = sum(distances)
                stats['mean_step_size'] = np.mean(distances)
        
        # Turn-taking patterns
        turn_gaps = []
        for speaker in speaker_stats:
            turns = speaker_stats[speaker]['turns']
            if len(turns) > 1:
                gaps = [turns[i+1] - turns[i] for i in range(len(turns)-1)]
                turn_gaps.extend(gaps)
                speaker_stats[speaker]['mean_turn_gap'] = np.mean(gaps)
        
        # Calculate embedding overlap between speakers with bootstrap CIs
        speakers = list(speaker_stats.keys())
        speaker_overlaps = {}
        
        for i in range(len(speakers)):
            for j in range(i+1, len(speakers)):
                s1, s2 = speakers[i], speakers[j]
                
                # Bootstrap for confidence intervals
                n_bootstrap = 1000
                boot_distances = []
                boot_spread_ratios = []
                
                embeds1 = np.array(speaker_stats[s1]['embeddings'])
                embeds2 = np.array(speaker_stats[s2]['embeddings'])
                
                for _ in range(n_bootstrap):
                    # Resample embeddings  
                    idx1 = np.random.choice(len(embeds1), len(embeds1), replace=True)
                    idx2 = np.random.choice(len(embeds2), len(embeds2), replace=True)
                    
                    cent1 = np.mean(embeds1[idx1], axis=0)
                    cent2 = np.mean(embeds2[idx2], axis=0)
                    
                    boot_distances.append(self._safe_euclidean(cent1, cent2))
                    
                    spread1 = np.std(embeds1[idx1], axis=0).mean()
                    spread2 = np.std(embeds2[idx2], axis=0).mean()
                    boot_spread_ratios.append(spread1 / (spread2 + 1e-8))
                
                centroid_distance = self._safe_euclidean(
                    speaker_stats[s1]['embedding_centroid'],
                    speaker_stats[s2]['embedding_centroid']
                )
                
                speaker_overlaps[f"{s1}-{s2}"] = {
                    'centroid_distance': centroid_distance,
                    'distance_ci_lower': np.percentile(boot_distances, 2.5),
                    'distance_ci_upper': np.percentile(boot_distances, 97.5),
                    'spread_ratio': speaker_stats[s1]['embedding_spread'] / (speaker_stats[s2]['embedding_spread'] + 1e-8),
                    'spread_ratio_ci_lower': np.percentile(boot_spread_ratios, 2.5),
                    'spread_ratio_ci_upper': np.percentile(boot_spread_ratios, 97.5)
                }
        
        return {
            'speaker_stats': speaker_stats,
            'mean_turn_gap': np.mean(turn_gaps) if turn_gaps else None,
            'turn_gap_variance': np.var(turn_gaps) if turn_gaps else None,
            'speaker_overlaps': speaker_overlaps
        }
    
    def calculate_information_flow(self, conversation, k=5):
        """Calculate information flow and entropy metrics"""
        messages = conversation['embedded_messages']
        embeddings = np.array([m['embedding'] for m in messages])
        
        # Local entropy using k-NN distances
        entropies = []
        for i, embedding in enumerate(embeddings):
            # Find k nearest neighbors in conversation so far
            if i < k:
                continue
                
            past_embeddings = embeddings[:i]
            nbrs = NearestNeighbors(n_neighbors=min(k, len(past_embeddings)))
            nbrs.fit(past_embeddings)
            
            distances, indices = nbrs.kneighbors([embedding])
            
            # Entropy estimate based on average distance to neighbors
            entropy = np.log(distances.mean() + 1e-8)
            
            # Also track which past messages are most similar
            most_similar_turns = [indices[0][j] for j in range(min(3, len(indices[0])))]
            
            entropies.append({
                'turn': i,
                'entropy': entropy,
                'mean_nn_distance': distances.mean(),
                'most_similar_past_turns': most_similar_turns
            })
        
        if entropies:
            # Calculate entropy trend
            turns = [e['turn'] for e in entropies]
            entropy_values = [e['entropy'] for e in entropies]
            entropy_trend = np.polyfit(turns, entropy_values, 1)[0] if len(entropies) > 1 else 0
        else:
            entropy_trend = 0
        
        return {
            'entropy_trajectory': entropies,
            'mean_entropy': np.mean([e['entropy'] for e in entropies]) if entropies else 0,
            'entropy_trend': entropy_trend,
            'final_entropy': entropies[-1]['entropy'] if entropies else 0
        }
    
    def find_embedding_clusters(self, eps=0.5, min_samples=5):
        """Find embedding clusters (potential topics/themes) in the embedding space"""
        print(f"\nFinding embedding clusters with DBSCAN (eps={eps}, min_samples={min_samples})...")
        
        # Convert to numpy array
        embeddings_array = np.array(self.all_embeddings)
        
        # Standardize embeddings
        scaler = StandardScaler()
        embeddings_scaled = scaler.fit_transform(embeddings_array)
        
        # Cluster
        clustering = DBSCAN(eps=eps, min_samples=min_samples, metric='cosine')
        clusters = clustering.fit_predict(embeddings_scaled)
        
        # Analyze clusters
        unique_clusters = set(clusters) - {-1}  # Exclude noise
        print(f"Found {len(unique_clusters)} embedding clusters (plus {sum(clusters == -1)} unclustered points)")
        
        # Map clusters back to messages
        for i, cluster in enumerate(clusters):
            self.all_metadata[i]['embedding_cluster'] = cluster
        
        # Analyze cluster composition
        cluster_analysis = {}
        for cluster_id in unique_clusters:
            cluster_mask = clusters == cluster_id
            cluster_metadata = [self.all_metadata[i] for i in range(len(clusters)) if cluster_mask[i]]
            
            # Speaker distribution
            speakers = [m['speaker'] for m in cluster_metadata]
            speaker_dist = pd.Series(speakers).value_counts(normalize=True).to_dict()
            
            # Turn distribution (to see if clusters are temporal)
            turns = [m['turn'] for m in cluster_metadata]
            
            cluster_analysis[cluster_id] = {
                'size': len(cluster_metadata),
                'speaker_distribution': speaker_dist,
                'mean_turn': np.mean(turns),
                'turn_range': (min(turns), max(turns))
            }
            
            print(f"\nCluster {cluster_id}:")
            print(f"  Size: {len(cluster_metadata)} messages")
            print(f"  Turn range: {min(turns)}-{max(turns)} (mean: {np.mean(turns):.1f})")
            print(f"  Speakers: {', '.join([f'{s}: {p:.1%}' for s, p in speaker_dist.items()])}")
        
        return clusters, cluster_analysis
    
    def analyze_embedding_structure(self):
        """Analyze the embedding structure of conversations"""
        print("\nAnalyzing embedding structure...")
        
        # Analyze trajectory statistics per conversation
        for conv in self.conversations:
            if 'trajectory_stats' in conv:
                stats = conv['trajectory_stats']
                print(f"\nSession {conv['metadata']['session_id'][:20]}:")
                print(f"  Mean embedding distance between turns: {stats['mean_distance']:.3f}")
                print(f"  Std deviation: {stats['std_distance']:.3f}")
                print(f"  Max jump: {stats['max_distance']:.3f}")
                print(f"  Total trajectory length: {stats['total_distance']:.3f}")
            
            # Print phase transition metrics
            if 'phase_metrics' in conv:
                print(f"\nPhase transitions:")
                for transition in conv['phase_metrics']['transitions']:
                    print(f"  {transition['from']} → {transition['to']}: distance={transition['distance']:.3f}, at turn {transition['turn_boundary']}")
                print(f"\nPhase regions:")
                for phase_name, info in conv['phase_metrics']['phase_embeddings'].items():
                    print(f"  {phase_name}: turns {info['turn_range'][0]}-{info['turn_range'][1]}, "
                          f"{info['message_count']} messages, spread={info['spread']:.3f}")
            
            # Print curvature metrics
            if 'curvature_metrics' in conv:
                curv = conv['curvature_metrics']
                print(f"\nTrajectory curvature:")
                print(f"  Mean curvature: {curv['mean_curvature']:.3f}")
                print(f"  Max curvature: {curv['max_curvature']:.3f}")
                print(f"  Mean acceleration: {curv['mean_acceleration']:.3f}")
            
            # Print speaker dynamics
            if 'speaker_dynamics' in conv:
                dynamics = conv['speaker_dynamics']
                print(f"\nSpeaker dynamics:")
                for speaker, stats in dynamics['speaker_stats'].items():
                    print(f"  {speaker}: {stats['message_count']} messages, "
                          f"spread={stats['embedding_spread']:.3f}, "
                          f"movement={stats['total_movement']:.3f}")
            
            # Print information flow
            if 'information_flow' in conv:
                flow = conv['information_flow']
                print(f"\nInformation flow:")
                print(f"  Mean entropy: {flow['mean_entropy']:.3f}")
                print(f"  Entropy trend: {flow['entropy_trend']:.3f}")
                print(f"  Final entropy: {flow['final_entropy']:.3f}")
        
        # Analyze phases if available
        for conv in self.conversations:
            if conv.get('phases'):
                print(f"\nConversation phases for {conv['metadata']['session_id'][:20]}:")
                for phase in conv['phases']:
                    print(f"  Turn {phase['turn']}: {phase['phase']}")
        
        return True
    
    def visualize_trajectories_3d(self, method='pca', n_conversations=10):
        """Visualize conversation trajectories in 3D reduced space"""
        print(f"\nCreating 3D trajectory visualization using {method.upper()}...")
        
        # Convert embeddings list to numpy array
        embeddings_array = np.array(self.all_embeddings)
        
        # Reduce dimensions
        if method == 'pca':
            reducer = PCA(n_components=3)
        elif method == 'tsne':
            # Adjust perplexity based on number of samples
            n_samples = len(embeddings_array)
            perplexity = min(30, max(5, n_samples // 4))
            print(f"Using perplexity={perplexity} for {n_samples} samples")
            reducer = TSNE(n_components=3, random_state=42, perplexity=perplexity)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        embeddings_3d = reducer.fit_transform(embeddings_array)
        
        # Create plotly figure
        fig = go.Figure()
        
        # Plot trajectories for each conversation
        for conv in self.conversations[:n_conversations]:
            session_id = conv['metadata']['session_id']
            
            # Get indices for this conversation
            indices = [i for i, m in enumerate(self.all_metadata) if m['session_id'] == session_id]
            
            if len(indices) < 2:
                continue
            
            # Extract 3D coordinates
            trajectory_3d = embeddings_3d[indices]
            x, y, z = trajectory_3d[:, 0], trajectory_3d[:, 1], trajectory_3d[:, 2]
            
            # Get speakers for coloring
            speakers = [self.all_metadata[i]['speaker'] for i in indices]
            unique_speakers = list(set(speakers))
            speaker_colors = {s: f'hsl({i*360/len(unique_speakers)}, 70%, 50%)' 
                            for i, s in enumerate(unique_speakers)}
            
            # Plot trajectory with color gradient
            for i in range(len(x)-1):
                fig.add_trace(go.Scatter3d(
                    x=[x[i], x[i+1]], 
                    y=[y[i], y[i+1]], 
                    z=[z[i], z[i+1]],
                    mode='lines',
                    line=dict(
                        color=speaker_colors[speakers[i]], 
                        width=4
                    ),
                    showlegend=False,
                    hovertext=f"{speakers[i]} (turn {i})"
                ))
            
            # Add markers for each message
            for speaker in unique_speakers:
                speaker_indices = [i for i in range(len(indices)) if speakers[i] == speaker]
                speaker_x = [x[i] for i in speaker_indices]
                speaker_y = [y[i] for i in speaker_indices]
                speaker_z = [z[i] for i in speaker_indices]
                
                fig.add_trace(go.Scatter3d(
                    x=speaker_x, y=speaker_y, z=speaker_z,
                    mode='markers',
                    name=speaker,
                    marker=dict(
                        size=6,
                        color=speaker_colors[speaker],
                        symbol='circle'
                    ),
                    text=[f"Turn {indices[i]}" for i in speaker_indices],
                    hoverinfo='text+name'
                ))
            
            # Add phase markers if available
            if 'phases' in conv and len(conv['phases']) > 0:
                # Create a mapping of turn numbers to phase names
                phase_map = {}
                phases_sorted = sorted(conv['phases'], key=lambda x: x['turn'])
                
                for i, phase in enumerate(phases_sorted):
                    start_turn = phase['turn']
                    # Find end turn (next phase start or last message)
                    if i + 1 < len(phases_sorted):
                        end_turn = phases_sorted[i + 1]['turn'] - 1
                    else:
                        end_turn = conv['metadata']['message_count'] - 1
                    
                    # Map all turns in this range to the phase
                    for turn in range(start_turn, end_turn + 1):
                        phase_map[turn] = phase['phase']
                
                # Now add phase transition markers
                added_phases = set()
                for i, phase in enumerate(phases_sorted):
                    phase_name = phase['phase']
                    phase_turn = phase['turn']
                    
                    # Find the index in our global metadata
                    phase_indices = [idx for idx, m in enumerate(self.all_metadata) 
                                   if m['session_id'] == session_id and m['turn'] == phase_turn]
                    
                    if phase_indices:
                        phase_idx = phase_indices[0]
                        # Add a vertical line to show phase transition
                        if i > 0:  # Not the first phase
                            prev_idx = phase_indices[0] - 1 if phase_indices[0] > 0 else 0
                            fig.add_trace(go.Scatter3d(
                                x=[embeddings_3d[prev_idx, 0], embeddings_3d[phase_idx, 0]],
                                y=[embeddings_3d[prev_idx, 1], embeddings_3d[phase_idx, 1]],
                                z=[embeddings_3d[prev_idx, 2], embeddings_3d[phase_idx, 2]],
                                mode='lines',
                                line=dict(color='gray', width=2, dash='dash'),
                                showlegend=False,
                                hovertext=f'Phase transition to: {phase_name}'
                            ))
                        
                        # Add phase label at transition point
                        fig.add_trace(go.Scatter3d(
                            x=[embeddings_3d[phase_idx, 0]],
                            y=[embeddings_3d[phase_idx, 1]],
                            z=[embeddings_3d[phase_idx, 2]],
                            mode='markers+text',
                            marker=dict(size=15, symbol='diamond', color='purple', 
                                      line=dict(color='black', width=2)),
                            text=[f'{phase_name}'],
                            textposition='top center',
                            textfont=dict(size=12, color='black'),
                            showlegend=False,
                            hovertext=f'Phase: {phase_name} (starts turn {phase_turn})'
                        ))
                        added_phases.add(phase_name)
                
                # Add phase regions with semi-transparent coloring
                if 'phase_metrics' in conv:
                    phase_data = conv['phase_metrics']['phase_embeddings']
                    colors = ['rgba(255,0,0,0.1)', 'rgba(0,255,0,0.1)', 'rgba(0,0,255,0.1)', 
                             'rgba(255,255,0,0.1)', 'rgba(255,0,255,0.1)', 'rgba(0,255,255,0.1)']
                    
                    for idx, (phase_name, phase_info) in enumerate(phase_data.items()):
                        start_turn, end_turn = phase_info['turn_range']
                        phase_indices = [i for i, m in enumerate(self.all_metadata) 
                                       if m['session_id'] == session_id and 
                                       start_turn <= m['turn'] <= end_turn]
                        
                        if len(phase_indices) >= 2:
                            # Create a mesh to show the phase region
                            phase_points = embeddings_3d[phase_indices]
                            color = colors[idx % len(colors)]
                            
                            # Add convex hull or simplified boundary
                            for j in range(len(phase_indices) - 1):
                                fig.add_trace(go.Scatter3d(
                                    x=[phase_points[j, 0], phase_points[j+1, 0]],
                                    y=[phase_points[j, 1], phase_points[j+1, 1]],
                                    z=[phase_points[j, 2], phase_points[j+1, 2]],
                                    mode='lines',
                                    line=dict(color=color.replace('0.1', '0.3'), width=6),
                                    showlegend=False,
                                    hovertext=f'Phase region: {phase_name}'
                                ))
            
            # Add start and end markers
            fig.add_trace(go.Scatter3d(
                x=[x[0]], y=[y[0]], z=[z[0]],
                mode='markers+text',
                marker=dict(size=10, color='green', symbol='diamond'),
                text=['START'],
                textposition='top center',
                showlegend=False,
                hovertext=f'Conversation Start'
            ))
            
            fig.add_trace(go.Scatter3d(
                x=[x[-1]], y=[y[-1]], z=[z[-1]],
                mode='markers+text',
                marker=dict(size=10, color='red', symbol='square'),
                text=['END'],
                textposition='top center',
                showlegend=False,
                hovertext=f'Conversation End'
            ))
        
        fig.update_layout(
            title=f'Conversation Trajectory in Embedding Space ({method.upper()})',
            scene=dict(
                xaxis_title='Embedding Dimension 1',
                yaxis_title='Embedding Dimension 2',
                zaxis_title='Embedding Dimension 3',
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.5)
                )
            ),
            showlegend=True,
            height=800
        )
        
        # Save
        output_path = self.output_dir / f'trajectories_3d_{method}.html'
        fig.write_html(str(output_path))
        print(f"Saved 3D visualization to {output_path}")
        
        return fig
    
    def analyze_peer_pressure_convergence(self):
        """Analyze embedding convergence patterns between speakers"""
        print("\nAnalyzing embedding convergence between speakers...")
        
        convergence_results = []
        
        for conv in self.conversations:
            messages = conv['embedded_messages']
            
            # Group by speaker
            speaker_messages = {}
            for msg in messages:
                speaker = msg['speaker']
                if speaker not in speaker_messages:
                    speaker_messages[speaker] = []
                speaker_messages[speaker].append(msg)
            
            # Calculate pairwise convergence over time
            speakers = list(speaker_messages.keys())
            if len(speakers) < 2:
                continue
            
            # Track convergence between each pair
            for i in range(len(speakers)):
                for j in range(i+1, len(speakers)):
                    speaker1, speaker2 = speakers[i], speakers[j]
                    
                    # Get embeddings in chronological order
                    msgs1 = sorted(speaker_messages[speaker1], key=lambda x: x['turn'])
                    msgs2 = sorted(speaker_messages[speaker2], key=lambda x: x['turn'])
                    
                    # Calculate similarity at different phases
                    if len(msgs1) >= 3 and len(msgs2) >= 3:
                        # Early phase (first third)
                        early1 = np.mean([m['embedding'] for m in msgs1[:len(msgs1)//3]], axis=0)
                        early2 = np.mean([m['embedding'] for m in msgs2[:len(msgs2)//3]], axis=0)
                        early_similarity = 1 - self._safe_cosine(early1, early2)
                        
                        # Late phase (last third)
                        late1 = np.mean([m['embedding'] for m in msgs1[-len(msgs1)//3:]], axis=0)
                        late2 = np.mean([m['embedding'] for m in msgs2[-len(msgs2)//3:]], axis=0)
                        late_similarity = 1 - self._safe_cosine(late1, late2)
                        
                        # Convergence score
                        convergence = late_similarity - early_similarity
                        
                        convergence_results.append({
                            'session_id': conv['metadata']['session_id'],
                            'speaker_pair': f"{speaker1}-{speaker2}",
                            'early_similarity': early_similarity,
                            'late_similarity': late_similarity,
                            'convergence': convergence,
                            'messages_speaker1': len(msgs1),
                            'messages_speaker2': len(msgs2)
                        })
        
        # Analyze results
        convergence_df = pd.DataFrame(convergence_results)
        
        if not convergence_df.empty:
            print("\nConvergence Statistics:")
            print(convergence_df[['speaker_pair', 'early_similarity', 'late_similarity', 'convergence']].describe())
            
            # Save results
            convergence_df.to_csv(self.output_dir / 'embedding_convergence.csv', index=False)
        
        return convergence_df
    
    def identify_attractor_regions(self, threshold_percentile=90):
        """Identify embedding attractor regions where conversations tend to converge"""
        print(f"\nIdentifying high-density embedding regions (top {100-threshold_percentile}% density)...")
        
        # Calculate local density for each point
        # Convert to numpy array
        embeddings_array = np.array(self.all_embeddings)
        
        # Use k-nearest neighbors to estimate local density
        k = min(20, len(embeddings_array) // 10)
        nbrs = NearestNeighbors(n_neighbors=k, metric='cosine')
        nbrs.fit(embeddings_array)
        
        distances, indices = nbrs.kneighbors(embeddings_array)
        
        # Density is inverse of average distance to k nearest neighbors
        densities = 1 / (distances.mean(axis=1) + 1e-8)
        
        # Find high-density regions
        threshold = np.percentile(densities, threshold_percentile)
        high_density_mask = densities > threshold
        
        # Analyze what's in high-density regions
        high_density_metadata = [self.all_metadata[i] for i in range(len(densities)) if high_density_mask[i]]
        
        print(f"\nFound {sum(high_density_mask)} messages in high-density regions")
        
        # Speaker distribution in attractors
        speakers = [m['speaker'] for m in high_density_metadata]
        speaker_dist = pd.Series(speakers).value_counts()
        print("\nSpeaker distribution in high-density regions:")
        for speaker, count in speaker_dist.items():
            print(f"  {speaker}: {count} messages")
        
        # Turn distribution
        turns = [m['turn'] for m in high_density_metadata]
        print(f"\nTurn range in high-density regions: {min(turns)}-{max(turns)}")
        print(f"Mean turn: {np.mean(turns):.1f}")
        
        # Find representative messages from attractors
        attractor_indices = np.where(high_density_mask)[0]
        if len(attractor_indices) > 0:
            # Use clustering to find distinct attractors
            attractor_embeddings = embeddings_array[attractor_indices]
            attractor_clusters = DBSCAN(eps=0.3, min_samples=5).fit_predict(attractor_embeddings)
            
            n_distinct = len(set(attractor_clusters)) - (1 if -1 in attractor_clusters else 0)
            print(f"\nFound {n_distinct} distinct high-density regions")
        
        return high_density_mask, densities
    
    def calculate_metric_reliability(self, conversation):
        """Calculate reliability scores for each metric"""
        reliability = {}
        n_messages = len(conversation['embedded_messages'])
        
        # Intrinsic dimension reliability
        if 'full_dimensional_analysis' in conversation:
            dim_analysis = conversation['full_dimensional_analysis']
            if 'intrinsic_dimensionality' in dim_analysis:
                id_data = dim_analysis['intrinsic_dimensionality']
                
                # Check convergence of estimates
                if id_data.get('mle_dimension') and id_data.get('correlation_dimension'):
                    relative_diff = abs(id_data['mle_dimension'] - id_data['correlation_dimension']) / id_data['mle_dimension']
                    reliability['intrinsic_dim'] = 1.0 - min(relative_diff, 1.0)
                elif id_data.get('consensus_dimension'):
                    # Use consensus if available
                    reliability['intrinsic_dim'] = 0.7
                else:
                    reliability['intrinsic_dim'] = 0.5
                
                # Boost reliability based on estimation quality
                if id_data.get('estimation_quality') == 'high':
                    reliability['intrinsic_dim'] = min(1.0, reliability['intrinsic_dim'] * 1.2)
        
        # Phase detection reliability (based on confidence of shifts)
        if 'phase_metrics' in conversation:
            transitions = conversation['phase_metrics'].get('transitions', [])
            if transitions:
                # High reliability if phase shifts are well-separated from background
                shift_distances = [t['distance'] for t in transitions]
                if len(shift_distances) > 1:
                    mean_shift = np.mean(shift_distances)
                    std_shift = np.std(shift_distances)
                    z_scores = [(s - mean_shift) / std_shift for s in shift_distances] if std_shift > 0 else [0]
                    reliability['phase_detection'] = min(np.mean(np.abs(z_scores)) / 2, 1.0)
                else:
                    reliability['phase_detection'] = 0.6
            else:
                reliability['phase_detection'] = 0.3
        
        # Distance metrics reliability (based on sample size)
        if 'distance_analysis' in conversation:
            # Reliability increases with conversation length
            reliability['distance_metrics'] = min(1.0, n_messages / 200.0)
        
        # Trajectory metrics reliability
        if 'trajectory_stats' in conversation:
            # Check if trajectory is smooth vs erratic
            traj = conversation['trajectory_stats']
            if 'step_distances' in traj:
                step_var = np.var(traj['step_distances'])
                mean_step = np.mean(traj['step_distances'])
                cv = np.sqrt(step_var) / mean_step if mean_step > 0 else 1.0
                reliability['trajectory'] = max(0.3, 1.0 - min(cv, 1.0))
        
        # Add to conversation data
        conversation['metric_reliability'] = reliability
        
        return reliability
    
    def analyze_conversation_comprehensive(self, conversation, verbose=False, use_checkpoint=True):
        """Run comprehensive analysis on a single conversation with checkpoint support"""
        session_id = conversation['metadata']['session_id']
        
        # Try to load existing checkpoint if enabled
        if use_checkpoint:
            checkpoint_conv = self.load_conversation_checkpoint(session_id)
            if checkpoint_conv:
                # Merge loaded data with current conversation
                conversation.update(checkpoint_conv)
                completed_steps = self.get_completed_analysis_steps(conversation)
                if verbose:
                    print(f"Loaded checkpoint with {len(completed_steps)} completed steps")
            else:
                completed_steps = set()
        else:
            completed_steps = set()
        
        # Define analysis steps in order
        analysis_steps = [
            ('embedding', lambda c: self.embed_conversation(c, verbose=verbose)),
            ('trajectory_metrics', lambda c: self.calculate_trajectory_metrics(c)),
            ('phase_metrics', lambda c: self._add_phase_metrics(c)),
            ('curvature_metrics', lambda c: self._add_curvature_metrics(c)),
            ('coherence_windows', lambda c: self._add_coherence_windows(c)),
            ('convergence_analysis', lambda c: self._add_convergence_analysis(c)),
            ('speaker_dynamics', lambda c: self._add_speaker_dynamics(c)),
            ('information_flow', lambda c: self._add_information_flow(c)),
            ('full_dimensional_analysis', lambda c: self._add_full_dimensional_analysis(c)),
            ('distance_analysis', lambda c: self._add_distance_analysis(c)),
            ('ensemble_invariants', lambda c: self._add_ensemble_invariants(c) if self.ensemble_mode else c),
            ('metric_reliability', lambda c: self._add_metric_reliability(c))
        ]
        
        # Run only the steps that haven't been completed
        for step_name, step_func in analysis_steps:
            if step_name not in completed_steps:
                try:
                    if verbose:
                        print(f"Running {step_name}...")
                    conversation = step_func(conversation)
                    
                    # Save checkpoint after each step
                    if use_checkpoint:
                        self.save_conversation_checkpoint(conversation, step_name)
                        
                except Exception as e:
                    print(f"Error in {step_name}: {e}")
                    # Save checkpoint even on error
                    if use_checkpoint:
                        self.save_conversation_checkpoint(conversation)
                    raise
            elif verbose:
                print(f"Skipping {step_name} (already completed)")
        
        return conversation
    
    # Helper methods to wrap analysis functions
    def _add_phase_metrics(self, conversation):
        phase_metrics = self.calculate_phase_transition_metrics(conversation)
        if phase_metrics:
            conversation['phase_metrics'] = phase_metrics
        return conversation
    
    def _add_curvature_metrics(self, conversation):
        curvature_metrics = self.calculate_trajectory_curvature(conversation)
        if curvature_metrics:
            conversation['curvature_metrics'] = curvature_metrics
        return conversation
    
    def _add_coherence_windows(self, conversation):
        coherence_windows = self.calculate_embedding_coherence_windows(conversation)
        conversation['coherence_windows'] = coherence_windows
        return conversation
    
    def _add_convergence_analysis(self, conversation):
        convergence_analysis = self.detect_convergence_with_confidence(conversation)
        if convergence_analysis:
            conversation['convergence_analysis'] = convergence_analysis
        return conversation
    
    def _add_speaker_dynamics(self, conversation):
        speaker_dynamics = self.analyze_speaker_dynamics(conversation)
        conversation['speaker_dynamics'] = speaker_dynamics
        return conversation
    
    def _add_information_flow(self, conversation):
        information_flow = self.calculate_information_flow(conversation)
        conversation['information_flow'] = information_flow
        return conversation
    
    def _add_full_dimensional_analysis(self, conversation):
        full_dimensional = self.analyze_full_dimensional_structure(conversation)
        conversation['full_dimensional_analysis'] = full_dimensional
        return conversation
    
    def _add_distance_analysis(self, conversation):
        distance_analysis = self.create_distance_matrices(conversation)
        conversation['distance_analysis'] = distance_analysis
        return conversation
    
    def _add_ensemble_invariants(self, conversation):
        invariant_patterns = self.find_ensemble_invariants(conversation)
        conversation['invariant_patterns'] = invariant_patterns
        # Skip separate ensemble visualization - it's now included in the comprehensive figure
        # self.visualize_ensemble_analysis(conversation)
        return conversation
    
    def _add_metric_reliability(self, conversation):
        reliability_scores = self.calculate_metric_reliability(conversation)
        conversation['metric_reliability'] = reliability_scores
        return conversation
    
    def extract_feature_vector(self, conversation):
        """Extract a comprehensive feature vector for ML applications"""
        features = {}
        
        # Basic metadata
        features['message_count'] = conversation['metadata']['message_count']
        features['speaker_count'] = len(set([m['speaker'] for m in conversation['messages']]))
        
        # Trajectory features
        if 'trajectory_stats' in conversation:
            stats = conversation['trajectory_stats']
            features['traj_mean_distance'] = stats['mean_distance']
            features['traj_std_distance'] = stats['std_distance']
            features['traj_max_distance'] = stats['max_distance']
            features['traj_total_distance'] = stats['total_distance']
            features['traj_acceleration'] = stats['distance_acceleration']
        
        # Phase features with confidence intervals
        if 'phase_metrics' in conversation:
            phase_data = conversation['phase_metrics']
            features['num_phases'] = len(phase_data['phase_embeddings'])
            if phase_data['transitions']:
                features['phase_mean_transition_dist'] = np.mean([t['distance'] for t in phase_data['transitions']])
                features['phase_max_transition_dist'] = np.max([t['distance'] for t in phase_data['transitions']])
                # Include confidence intervals if available
                if 'distance_ci_lower' in phase_data['transitions'][0]:
                    features['phase_mean_dist_ci_width'] = np.mean([
                        t['distance_ci_upper'] - t['distance_ci_lower'] 
                        for t in phase_data['transitions']
                    ])
        
        # Curvature features with confidence intervals and normalization
        if 'curvature_metrics' in conversation:
            curv = conversation['curvature_metrics']
            features['curv_mean'] = curv['mean_curvature']
            features['curv_mean_normalized'] = curv.get('mean_curvature_normalized', curv['mean_curvature'])
            features['curv_max'] = curv['max_curvature']
            features['curv_variance'] = curv['curvature_variance']
            features['accel_mean'] = curv['mean_acceleration']
            features['accel_variance'] = curv['acceleration_variance']
            # Confidence interval widths
            if 'mean_curvature_ci_lower' in curv:
                features['curv_mean_ci_width'] = curv['mean_curvature_ci_upper'] - curv['mean_curvature_ci_lower']
                features['curv_max_ci_width'] = curv['max_curvature_ci_upper'] - curv['max_curvature_ci_lower']
        
        # Coherence features
        if 'coherence_windows' in conversation:
            coherence = conversation['coherence_windows']
            if coherence:
                features['coherence_mean'] = np.mean([c['mean_similarity'] for c in coherence])
                features['coherence_min'] = np.min([c['min_similarity'] for c in coherence])
                features['coherence_variance'] = np.mean([c['similarity_variance'] for c in coherence])
        
        # Speaker dynamics features
        if 'speaker_dynamics' in conversation:
            dynamics = conversation['speaker_dynamics']
            features['turn_gap_mean'] = dynamics['mean_turn_gap'] or 0
            features['turn_gap_variance'] = dynamics['turn_gap_variance'] or 0
            
            # Average embedding spread across speakers
            spreads = [s['embedding_spread'] for s in dynamics['speaker_stats'].values()]
            features['speaker_spread_mean'] = np.mean(spreads)
        
        # Convergence features with confidence
        if 'convergence_analysis' in conversation:
            conv_analysis = conversation['convergence_analysis']
            features['max_convergence_score'] = conv_analysis['max_convergence_score']
            features['n_convergent_regions'] = conv_analysis['n_convergent_regions']
            features['normalized_convergence_rate'] = conv_analysis['normalized_convergence_rate']
            
            # Average confidence interval width for convergence scores
            if conv_analysis['convergence_scores']:
                ci_widths = [s['score_ci_upper'] - s['score_ci_lower'] 
                           for s in conv_analysis['convergence_scores'] 
                           if s['score_ci_upper'] < float('inf')]
                if ci_widths:
                    features['convergence_ci_width_mean'] = np.mean(ci_widths)
        
        # Information flow features
        if 'information_flow' in conversation:
            flow = conversation['information_flow']
            features['entropy_mean'] = flow['mean_entropy']
            features['entropy_trend'] = flow['entropy_trend']
            features['entropy_final'] = flow['final_entropy']
        
        # Full dimensional features
        if 'full_dimensional_analysis' in conversation:
            full_dim = conversation['full_dimensional_analysis']
            
            if 'intrinsic_dimensionality' in full_dim:
                intrinsic = full_dim['intrinsic_dimensionality']
                features['intrinsic_dim_mle'] = intrinsic['mle_dimension'] or 0
                features['effective_rank'] = intrinsic['effective_rank']
            
            if 'dimensional_utilization' in full_dim:
                dim_util = full_dim['dimensional_utilization']
                features['participation_ratio'] = dim_util['participation_ratio']
                features['dormant_dimensions'] = dim_util['dormant_dimensions']
                features['dimension_entropy'] = dim_util['dimension_entropy']
            
            if 'trajectory_smoothness' in full_dim:
                smooth = full_dim['trajectory_smoothness']
                features['smoothness_index'] = smooth['smoothness_index']
                features['spectral_entropy'] = smooth['spectral_entropy'] or 0
            
            if 'phase_separability' in full_dim and full_dim['phase_separability']:
                phase_sep = full_dim['phase_separability']
                features['phase_silhouette'] = phase_sep['silhouette_coefficient'] or 0
                features['davies_bouldin'] = phase_sep['davies_bouldin_index'] or 0
        
        # Distance analysis features
        if 'distance_analysis' in conversation:
            dist = conversation['distance_analysis']
            features['embedding_loops'] = dist['loop_count']
            features['recurrence_rate'] = dist['recurrence_stats']['recurrence_rate']
            features['determinism'] = dist['recurrence_stats']['determinism']
        
        # Ensemble invariant features
        if self.ensemble_mode and 'invariant_patterns' in conversation:
            invariants = conversation['invariant_patterns']['summary']
            features['ensemble_distance_corr'] = invariants['mean_distance_correlation']
            features['ensemble_velocity_corr'] = invariants['mean_velocity_correlation']
            features['ensemble_topology_pres'] = invariants['mean_topology_preservation']
            features['ensemble_phase_consensus'] = invariants['phase_consensus_rate']
            features['ensemble_curvature_agree'] = invariants['curvature_agreement_rate']
            features['ensemble_n_convergences'] = invariants['n_consensus_convergences']
        
        return features
    
    def save_comprehensive_metadata(self):
        """Save comprehensive metadata including all computed metrics and phase information"""
        metadata_records = []
        
        for conv in self.conversations:
            record = {
                'session_id': conv['metadata']['session_id'],
                'tier': conv.get('tier', 'unknown'),
                'message_count': conv['metadata']['message_count'],
                'speaker_count': len(set([m['speaker'] for m in conv['messages']])),
                'phase_count': len(conv.get('phases', [])),
                'selected_for_visualization': conv.get('selected_for_visualization', False)
            }
            
            # Add phase names and transitions
            if 'phases' in conv:
                phase_names = [p['phase'] for p in conv['phases']]
                record['phase_sequence'] = ' -> '.join(phase_names)
                record['first_phase'] = phase_names[0] if phase_names else None
                record['last_phase'] = phase_names[-1] if phase_names else None
            
            # Add key metrics if available
            if 'trajectory_stats' in conv:
                record['total_trajectory_distance'] = conv['trajectory_stats']['total_distance']
                record['mean_turn_distance'] = conv['trajectory_stats']['mean_distance']
            
            if 'curvature_metrics' in conv:
                record['mean_curvature'] = conv['curvature_metrics']['mean_curvature']
                record['max_curvature'] = conv['curvature_metrics']['max_curvature']
            
            if 'full_dimensional_analysis' in conv:
                fa = conv['full_dimensional_analysis']
                if 'intrinsic_dimensionality' in fa and fa['intrinsic_dimensionality']['mle_dimension']:
                    record['intrinsic_dimension'] = fa['intrinsic_dimensionality']['mle_dimension']
                if 'dimensional_utilization' in fa:
                    record['participation_ratio'] = fa['dimensional_utilization']['participation_ratio']
            
            if 'distance_matrix_stats' in conv:
                dm = conv['distance_matrix_stats']
                record['recurrence_rate'] = dm['recurrence_stats']['recurrence_rate']
                record['embedding_loop_count'] = dm['loop_count']
            
            metadata_records.append(record)
        
        if metadata_records:
            metadata_df = pd.DataFrame(metadata_records)
            metadata_df.to_csv(self.output_dir / 'conversation_metadata.csv', index=False)
            print(f"\nSaved comprehensive metadata for {len(metadata_df)} conversations")
            
            # Print summary statistics
            print(f"  - Conversations with phases: {sum(1 for r in metadata_records if r['phase_count'] > 0)}")
            print(f"  - Average phases per conversation: {metadata_df['phase_count'].mean():.1f}")
            if 'intrinsic_dimension' in metadata_df.columns:
                print(f"  - Average intrinsic dimension: {metadata_df['intrinsic_dimension'].mean():.2f}")
    
    def _calculate_adaptive_similarity_threshold(self, embeddings, percentile=90):
        """Calculate model-specific similarity threshold"""
        # Sample pairwise similarities
        n = len(embeddings)
        n_samples = min(1000, n * (n - 1) // 2)
        
        similarities = []
        for _ in range(n_samples):
            i, j = np.random.choice(n, 2, replace=False)
            sim = 1 - self._safe_cosine(embeddings[i], embeddings[j])
            similarities.append(sim)
        
        # Use percentile-based threshold
        threshold = np.percentile(similarities, percentile)
        
        # Store model-specific statistics
        self.similarity_stats = {
            'mean': np.mean(similarities),
            'std': np.std(similarities),
            'percentiles': np.percentile(similarities, [10, 25, 50, 75, 90, 95, 99]),
            'threshold': threshold
        }
        
        return threshold
    
    def _calculate_ensemble_correlations(self, distance_matrices, self_similarities, model_names, n_messages):
        """Calculate correlations between distance matrices"""
        correlations = {}
        for i in range(len(model_names)):
            for j in range(i+1, len(model_names)):
                model1, model2 = model_names[i], model_names[j]
                triu_indices = np.triu_indices(n_messages, k=1)
                dist1 = distance_matrices[model1][triu_indices]
                dist2 = distance_matrices[model2][triu_indices]
                
                from scipy.stats import spearmanr
                corr, p_value = spearmanr(dist1, dist2)
                correlations[f'{model1}-{model2}'] = {
                    'correlation': corr,
                    'p_value': p_value
                }
        return correlations
    
    def _calculate_velocity_correlations(self, ensemble_embeddings, model_names):
        """Calculate velocity profile correlations across models"""
        velocity_profiles = {}
        for model_name, embeddings in ensemble_embeddings.items():
            if isinstance(embeddings, list) and embeddings and isinstance(embeddings[0], dict):
                embeddings = [m['embedding'] for m in embeddings]
            embeddings = np.array(embeddings)
            
            velocities = []
            for i in range(1, len(embeddings)):
                v = np.linalg.norm(embeddings[i] - embeddings[i-1])
                velocities.append(v)
            velocity_profiles[model_name] = velocities
        
        correlations = {}
        for i in range(len(model_names)):
            for j in range(i+1, len(model_names)):
                model1, model2 = model_names[i], model_names[j]
                from scipy.stats import pearsonr
                corr, p_value = pearsonr(velocity_profiles[model1], velocity_profiles[model2])
                correlations[f'{model1}-{model2}'] = {
                    'correlation': corr,
                    'p_value': p_value
                }
        return correlations
    
    def _calculate_topology_preservation(self, ensemble_embeddings, model_names):
        """Calculate topology preservation between models"""
        k = 10  # k-nearest neighbors
        scores = {}
        
        # Convert embeddings to arrays
        embeddings_arrays = {}
        for model_name, embeddings in ensemble_embeddings.items():
            if isinstance(embeddings, list) and embeddings and isinstance(embeddings[0], dict):
                embeddings = [m['embedding'] for m in embeddings]
            embeddings_arrays[model_name] = np.array(embeddings)
        
        for i in range(len(model_names)):
            for j in range(i+1, len(model_names)):
                model1, model2 = model_names[i], model_names[j]
                
                from sklearn.neighbors import NearestNeighbors
                nbrs1 = NearestNeighbors(n_neighbors=min(k, len(embeddings_arrays[model1])))
                nbrs2 = NearestNeighbors(n_neighbors=min(k, len(embeddings_arrays[model2])))
                
                nbrs1.fit(embeddings_arrays[model1])
                nbrs2.fit(embeddings_arrays[model2])
                
                _, indices1 = nbrs1.kneighbors(embeddings_arrays[model1])
                _, indices2 = nbrs2.kneighbors(embeddings_arrays[model2])
                
                preservation_scores = []
                for idx in range(len(embeddings_arrays[model1])):
                    neighbors1 = set(indices1[idx])
                    neighbors2 = set(indices2[idx])
                    preservation = len(neighbors1.intersection(neighbors2)) / min(k, len(embeddings_arrays[model1]))
                    preservation_scores.append(preservation)
                
                scores[f'{model1}-{model2}'] = {
                    'mean_preservation': np.mean(preservation_scores),
                    'std_preservation': np.std(preservation_scores)
                }
        return scores
    
    def _create_correlation_matrix(self, correlations, model_names, metric_type):
        """Create a symmetric correlation matrix from pairwise correlations"""
        n = len(model_names)
        matrix = np.eye(n)
        
        for i, m1 in enumerate(model_names):
            for j, m2 in enumerate(model_names):
                if i < j:
                    key = f'{m1}-{m2}' if f'{m1}-{m2}' in correlations else f'{m2}-{m1}'
                    if key in correlations:
                        if metric_type == 'topology':
                            value = correlations[key]['mean_preservation']
                        else:
                            value = correlations[key]['correlation']
                        matrix[i, j] = matrix[j, i] = value
        return matrix
    
    def _create_ensemble_summary_text(self, distance_corrs, velocity_corrs, topo_scores):
        """Create summary text for ensemble analysis"""
        all_dist_corrs = [v['correlation'] for v in distance_corrs.values()]
        all_vel_corrs = [v['correlation'] for v in velocity_corrs.values()]
        all_topo = [v['mean_preservation'] for v in topo_scores.values()]
        
        text = "Ensemble Summary\n" + "="*20 + "\n\n"
        text += f"Distance Corr:\n  μ={np.mean(all_dist_corrs):.3f}\n  σ={np.std(all_dist_corrs):.3f}\n\n"
        text += f"Velocity Corr:\n  μ={np.mean(all_vel_corrs):.3f}\n  σ={np.std(all_vel_corrs):.3f}\n\n"
        text += f"Topology Pres:\n  μ={np.mean(all_topo):.3f}\n  σ={np.std(all_topo):.3f}\n"
        return text
    
    def _add_phase_annotations_smart(self, ax, phase_info, n_messages):
        """Add phase annotations with smart positioning to avoid overlap"""
        if not phase_info:
            return
        
        # Sort phases by start turn
        sorted_phases = sorted(phase_info, key=lambda x: x['start_turn'])
        
        # Group phases that are close together
        phase_groups = []
        current_group = [sorted_phases[0]]
        
        for i in range(1, len(sorted_phases)):
            # If phases are close (within 20 turns), group them
            if sorted_phases[i]['start_turn'] - current_group[-1]['start_turn'] < 20:
                current_group.append(sorted_phases[i])
            else:
                phase_groups.append(current_group)
                current_group = [sorted_phases[i]]
        phase_groups.append(current_group)
        
        # Add annotations for each group
        for group_idx, group in enumerate(phase_groups):
            # Calculate vertical positions for this group
            group_y_start = n_messages - 10 - (group_idx * 15)
            
            for phase_idx, phase in enumerate(group):
                start_turn = phase['start_turn']
                if isinstance(phase['turn_range'], (list, tuple)) and len(phase['turn_range']) > 1:
                    end_turn = min(phase['turn_range'][1], n_messages - 1)
                else:
                    end_turn = min(start_turn + 10, n_messages - 1)
                
                mid_turn = (start_turn + end_turn) // 2
                phase_name = phase['name'].replace('_', ' ').title()
                
                # Alternate left/right positioning for adjacent phases
                if phase_idx % 2 == 0:
                    h_offset = -10
                else:
                    h_offset = 10
                
                annotation_x = mid_turn + h_offset
                annotation_y = group_y_start - (phase_idx * 3)
                
                # Ensure annotation stays within plot bounds
                annotation_x = max(5, min(annotation_x, n_messages - 5))
                annotation_y = max(5, min(annotation_y, n_messages - 5))
                
                # Add connecting line
                ax.plot([annotation_x, mid_turn], [annotation_y, start_turn],
                       color='gray', linestyle=':', alpha=0.5, linewidth=1)
                
                # Add text annotation
                ax.text(annotation_x, annotation_y, phase_name,
                       ha='center', va='center', fontsize=10,
                       bbox=dict(boxstyle="round,pad=0.3", facecolor='yellow', alpha=0.8),
                       rotation=0, weight='bold')
                
                # Add colored bar to show phase extent
                from matplotlib.patches import Rectangle
                ax.add_patch(Rectangle((start_turn, start_turn),
                                     end_turn - start_turn, 2,
                                     facecolor=plt.cm.tab10(phase_idx % 10), alpha=0.7))
    
    def _add_phase_annotations_edges(self, ax, phase_info, n_messages):
        """Add phase annotations on the edges (top and left) of the plot"""
        if not phase_info:
            return
        
        # Sort phases by start turn
        sorted_phases = sorted(phase_info, key=lambda x: x['start_turn'])
        
        # Colors for phases
        colors = plt.cm.tab10(np.linspace(0, 1, 10))
        
        for phase_idx, phase in enumerate(sorted_phases):
            start_turn = phase['start_turn']
            if start_turn >= n_messages or start_turn < 0:
                continue
                
            phase_name = phase['name'].replace('_', ' ').title()
            color = colors[phase_idx % 10]
            
            # Calculate phase end
            if phase_idx < len(sorted_phases) - 1:
                end_turn = sorted_phases[phase_idx + 1]['start_turn']
            else:
                end_turn = n_messages
            
            mid_turn = (start_turn + end_turn) / 2
            
            # Add annotation on top edge
            ax.text(mid_turn, n_messages + 2, phase_name,
                   ha='center', va='bottom', fontsize=10,
                   bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.8),
                   rotation=45, weight='bold')
            
            # Add annotation on left edge
            ax.text(-2, mid_turn, phase_name,
                   ha='right', va='center', fontsize=10,
                   bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.8),
                   rotation=0, weight='bold')
            
            # Add colored bars on the edges to show phase extent
            # Top edge bar
            from matplotlib.patches import Rectangle
            ax.add_patch(Rectangle((start_turn, n_messages - 1),
                                 end_turn - start_turn, 2,
                                 facecolor=color, alpha=0.7,
                                 transform=ax.transData))
            # Left edge bar
            ax.add_patch(Rectangle((-2, start_turn),
                                 2, end_turn - start_turn,
                                 facecolor=color, alpha=0.7,
                                 transform=ax.transData))
    
    def create_ensemble_distance_matrices(self, conversation, save_individual=True):
        """Create distance matrices for all ensemble models"""
        ensemble_embeddings = conversation['ensemble_embeddings']
        model_names = list(ensemble_embeddings.keys())
        n_models = len(model_names)
        n_messages = len(conversation['messages'])
        
        # Create a comprehensive figure with multiple sections
        # Section 1: Distance matrices and trajectories (4 rows)
        # Section 2: Ensemble analysis (1 row) - removed summary
        # Section 3: Large annotated recurrence plot (4 rows - much larger)
        total_rows = 9
        fig = plt.figure(figsize=(5*n_models, 45))
        
        # Create GridSpec for flexible layout
        from matplotlib.gridspec import GridSpec
        gs = GridSpec(total_rows, n_models + 1, figure=fig, hspace=0.4, wspace=0.3)  # Added column for legend, more spacing
        
        # Store distance matrices for correlation analysis
        distance_matrices = {}
        self_similarities = {}
        
        # Get phase information if available
        phase_info = []
        if 'phase_metrics' in conversation and 'phase_embeddings' in conversation['phase_metrics']:
            phase_embeddings = conversation['phase_metrics']['phase_embeddings']
            for phase_name, phase_data in phase_embeddings.items():
                phase_info.append({
                    'name': phase_name,
                    'start_turn': phase_data['start_turn'],
                    'turn_range': phase_data['turn_range']
                })
            phase_info.sort(key=lambda x: x['start_turn'])
        
        # Create distance matrices for each model
        for idx, (model_name, embeddings) in enumerate(ensemble_embeddings.items()):
            # Handle different possible formats
            if isinstance(embeddings, dict):
                # If it's a dict, it might be the full embedded_messages structure
                # Try to extract just the embeddings
                if 'embedded_messages' in embeddings:
                    embeddings = [m['embedding'] for m in embeddings['embedded_messages']]
                else:
                    # Unknown dict structure
                    raise ValueError(f"Unexpected dict structure for {model_name} embeddings")
            elif isinstance(embeddings, list) and embeddings and isinstance(embeddings[0], dict):
                # List of embedded message dicts
                embeddings = [m['embedding'] for m in embeddings]
            
            # Ensure embeddings is a proper 2D numpy array
            embeddings = np.array(embeddings)
            if embeddings.ndim == 3:
                # If there's an extra dimension, squeeze it
                embeddings = embeddings.squeeze()
            # Ensure it's 2D (n_messages, embedding_dim)
            if embeddings.ndim == 1:
                embeddings = embeddings.reshape(1, -1)
            
            # Compute distances with GPU if available
            try:
                import torch
                if torch.cuda.is_available():
                    device = torch.device('cuda')
                    try:
                        embeddings_tensor = torch.from_numpy(embeddings).float().to(device)
                    except Exception as e:
                        print(f"Error converting embeddings to tensor for {model_name}: {e}")
                        print(f"Embeddings type: {type(embeddings)}, shape: {embeddings.shape if hasattr(embeddings, 'shape') else 'no shape'}")
                        raise
                    
                    # Euclidean distances
                    diff = embeddings_tensor.unsqueeze(0) - embeddings_tensor.unsqueeze(1)
                    euclidean_distances = torch.norm(diff, dim=2).cpu().numpy()
                    
                    # Cosine distances
                    norms = torch.norm(embeddings_tensor, dim=1, keepdim=True)
                    # Avoid division by zero
                    norms = torch.clamp(norms, min=1e-8)
                    normalized = embeddings_tensor / norms
                    try:
                        cosine_sim = torch.mm(normalized, normalized.t())
                    except Exception as e:
                        print(f"Error in torch.mm for {model_name}: {e}")
                        print(f"normalized type: {type(normalized)}, shape: {normalized.shape}")
                        raise
                    cosine_distances = (1 - cosine_sim).cpu().numpy()
                else:
                    raise RuntimeError("No GPU")
            except Exception as e:
                if "dict" in str(e):
                    print(f"Dict error caught in GPU path for {model_name}")
                    raise
                # CPU fallback
                euclidean_distances = np.zeros((n_messages, n_messages))
                cosine_distances = np.zeros((n_messages, n_messages))
                for i in range(n_messages):
                    for j in range(n_messages):
                        # Ensure vectors are 1-D for scipy distance functions
                        vec_i = embeddings[i].flatten()
                        vec_j = embeddings[j].flatten()
                        euclidean_distances[i, j] = self._safe_euclidean(vec_i, vec_j)
                        cosine_distances[i, j] = self._safe_cosine(vec_i, vec_j)
            
            # Store for correlation analysis
            distance_matrices[model_name] = euclidean_distances
            self_similarities[model_name] = 1 - cosine_distances
            
            # Create subplots for this model using GridSpec
            ax1 = fig.add_subplot(gs[0, idx])  # Row 0: Euclidean distances
            ax2 = fig.add_subplot(gs[1, idx])  # Row 1: Self-similarity
            ax3 = fig.add_subplot(gs[2, idx])  # Row 2: Recurrence plots
            ax4 = fig.add_subplot(gs[3, idx])  # Row 3: Trajectory
            
            # Plot euclidean distances
            im1 = ax1.imshow(euclidean_distances, cmap='viridis', aspect='auto')
            ax1.set_title(f'{model_name}\nEuclidean Distance')
            ax1.set_xlabel('Turn')
            if idx == 0:
                ax1.set_ylabel('Turn')
            
            # Plot self-similarity
            im2 = ax2.imshow(self_similarities[model_name], cmap='RdBu_r', aspect='auto', vmin=0, vmax=1)
            ax2.set_title(f'Self-Similarity')
            ax2.set_xlabel('Turn')
            if idx == 0:
                ax2.set_ylabel('Turn')
            
            # Add phase boundaries if available
            if phase_info:
                for phase in phase_info:
                    start_turn = phase['start_turn']
                    if start_turn < n_messages and start_turn > 0:
                        ax1.axhline(y=start_turn, color='red', linestyle='--', alpha=0.6, linewidth=1)
                        ax1.axvline(x=start_turn, color='red', linestyle='--', alpha=0.6, linewidth=1)
                        ax2.axhline(y=start_turn, color='red', linestyle='--', alpha=0.6, linewidth=1)
                        ax2.axvline(x=start_turn, color='red', linestyle='--', alpha=0.6, linewidth=1)
            
            # Create recurrence plot for third row
            # Calculate adaptive threshold for this model
            # Use a more permissive threshold to ensure we see patterns
            sim_flat = self_similarities[model_name].flatten()
            # Use 85th percentile instead of mean + 2*std which might be too strict
            model_threshold = np.percentile(sim_flat, 85)
            recurrence_matrix = self_similarities[model_name] > model_threshold
            
            # If still too few points, adjust threshold
            recurrence_rate = np.sum(recurrence_matrix) / (n_messages * n_messages)
            if recurrence_rate < 0.01:  # Less than 1% recurrence is too sparse
                # Use 75th percentile instead
                model_threshold = np.percentile(sim_flat, 75)
                recurrence_matrix = self_similarities[model_name] > model_threshold
            
            # Plot recurrence
            ax3.imshow(recurrence_matrix, cmap='binary', aspect='auto')
            ax3.set_title(f'Recurrence Plot')
            ax3.set_xlabel('Turn')
            if idx == 0:
                ax3.set_ylabel('Turn')
            
            # Phase boundaries omitted - shown in detail in the large bottom panel
            
            # Row 4: Add trajectory visualization
            # Reduce to 2D for visualization
            from sklearn.decomposition import PCA
            pca = PCA(n_components=2)
            reduced_embeddings = pca.fit_transform(embeddings)
            
            # Create trajectory plot
            ax4.plot(reduced_embeddings[:, 0], reduced_embeddings[:, 1], 'b-', alpha=0.5, linewidth=1)
            
            # Color points by phase if available
            if phase_info:
                colors_traj = plt.cm.tab10(np.linspace(0, 1, 10))
                sorted_phases_traj = sorted(phase_info, key=lambda x: x['start_turn'])
                
                for phase_idx, phase in enumerate(sorted_phases_traj):
                    start_turn = phase['start_turn']
                    if phase_idx < len(sorted_phases_traj) - 1:
                        end_turn = sorted_phases_traj[phase_idx + 1]['start_turn']
                    else:
                        end_turn = n_messages
                    
                    # Use same distinct colors as the bottom plot
                    distinct_colors_traj = plt.cm.tab20(np.linspace(0, 1, 20))
                    phase_color = distinct_colors_traj[phase_idx % 20]
                    # Plot points in this phase
                    phase_points = reduced_embeddings[start_turn:end_turn]
                    if len(phase_points) > 0:
                        ax4.scatter(phase_points[:, 0], phase_points[:, 1], 
                                  c=[phase_color], s=20, alpha=0.7, edgecolors='black', linewidth=0.5)
            else:
                # Just show all points colored by time
                ax4.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], 
                          c=range(n_messages), cmap='viridis', s=20, alpha=0.7)
            
            # Mark start and end
            ax4.scatter(reduced_embeddings[0, 0], reduced_embeddings[0, 1], 
                      c='green', s=80, marker='^', edgecolors='black', linewidth=1.5, zorder=5)
            ax4.scatter(reduced_embeddings[-1, 0], reduced_embeddings[-1, 1], 
                      c='red', s=80, marker='v', edgecolors='black', linewidth=1.5, zorder=5)
            
            ax4.set_title(f'Trajectory (PCA)')
            ax4.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.0%})')
            if idx == 0:
                ax4.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.0%})')
            
            # Add grid for better readability
            ax4.grid(True, alpha=0.3)
            ax4.set_aspect('equal', adjustable='box')
        
        # Section 2: Ensemble Analysis Visualizations
        # Row 4: Correlation matrices - make them all the same size
        # Calculate positions to center the 3 correlation plots
        corr_start_col = max(0, (n_models - 3) // 2)
        ax_dist_corr = fig.add_subplot(gs[4, corr_start_col])
        ax_vel_corr = fig.add_subplot(gs[4, corr_start_col + 1]) if n_models >= 2 else None
        ax_topo = fig.add_subplot(gs[4, corr_start_col + 2]) if n_models >= 3 else None
        
        # Calculate and visualize ensemble metrics
        correlations = self._calculate_ensemble_correlations(distance_matrices, self_similarities, 
                                                           model_names, n_messages)
        
        # Distance correlations heatmap
        corr_matrix = self._create_correlation_matrix(correlations, model_names, 'distance')
        sns.heatmap(corr_matrix, annot=True, fmt='.3f', xticklabels=model_names,
                   yticklabels=model_names, cmap='RdBu_r', center=0.5, ax=ax_dist_corr,
                   vmin=0, vmax=1, cbar_kws={'label': 'Correlation'})
        ax_dist_corr.set_title('Distance Matrix Correlations')
        
        # Velocity correlations
        if ax_vel_corr is not None:
            velocity_corrs = self._calculate_velocity_correlations(ensemble_embeddings, model_names)
            vel_corr_matrix = self._create_correlation_matrix(velocity_corrs, model_names, 'velocity')
            sns.heatmap(vel_corr_matrix, annot=True, fmt='.3f', xticklabels=model_names,
                       yticklabels=model_names, cmap='RdBu_r', center=0.5, ax=ax_vel_corr,
                       vmin=0, vmax=1, cbar_kws={'label': 'Correlation'})
            ax_vel_corr.set_title('Velocity Profile Correlations')
        
        # Topology preservation
        if ax_topo is not None:
            topo_scores = self._calculate_topology_preservation(ensemble_embeddings, model_names)
            topo_matrix = self._create_correlation_matrix(topo_scores, model_names, 'topology')
            sns.heatmap(topo_matrix, annot=True, fmt='.3f', xticklabels=model_names,
                       yticklabels=model_names, cmap='Greens', vmin=0, vmax=1,
                       cbar_kws={'label': 'Preservation'}, ax=ax_topo)
            ax_topo.set_title('Topology Preservation')
        
        # Summary statistics removed to avoid overlap
        
        # Section 3: Large annotated recurrence plot for primary model
        # Use most of the remaining space (4 rows) for a large plot
        # Leave last column for color key
        ax_large = fig.add_subplot(gs[5:9, :n_models])  # Rows 5-8, all model columns
        
        # Get primary model data
        primary_self_sim = self_similarities[self.model_name]
        sim_flat = primary_self_sim.flatten()
        threshold = np.percentile(sim_flat, 85)
        recurrence_matrix = primary_self_sim > threshold
        
        # First add phase shading if available
        if phase_info:
            # Use more distinct colors for phases
            distinct_colors = plt.cm.tab20(np.linspace(0, 1, 20))
            
            # Sort phases by start turn
            sorted_phases = sorted(phase_info, key=lambda x: x['start_turn'])
            
            for phase_idx, phase in enumerate(sorted_phases):
                start_turn = phase['start_turn']
                if start_turn >= n_messages or start_turn < 0:
                    continue
                    
                # Calculate phase end
                if phase_idx < len(sorted_phases) - 1:
                    end_turn = sorted_phases[phase_idx + 1]['start_turn']
                else:
                    end_turn = n_messages
                
                color = distinct_colors[phase_idx % 20]
                
                # Add shaded regions (rectangles) with higher opacity for visibility
                from matplotlib.patches import Rectangle
                # Horizontal band
                ax_large.add_patch(Rectangle((0, start_turn), n_messages, end_turn - start_turn,
                                           facecolor=color, alpha=0.35, zorder=1))
                # Vertical band  
                ax_large.add_patch(Rectangle((start_turn, 0), end_turn - start_turn, n_messages,
                                           facecolor=color, alpha=0.35, zorder=1))
        
        # Now plot the recurrence matrix on top
        ax_large.imshow(recurrence_matrix, cmap='binary', aspect='equal', interpolation='nearest',
                       extent=[0, n_messages, n_messages, 0], zorder=2)
        
        # Add phase boundaries
        if phase_info:
            for phase in phase_info:
                start_turn = phase['start_turn']
                if start_turn < n_messages and start_turn > 0:
                    ax_large.axhline(y=start_turn, color='red', linestyle='--', alpha=0.8, linewidth=2, zorder=3)
                    ax_large.axvline(x=start_turn, color='red', linestyle='--', alpha=0.8, linewidth=2, zorder=3)
        
        ax_large.set_title(f'Detailed Recurrence Plot with Phase Regions ({self.model_name})', fontsize=16)
        ax_large.set_xlabel('Turn', fontsize=14) 
        ax_large.set_ylabel('Turn', fontsize=14)
        ax_large.set_xlim(0, n_messages)
        ax_large.set_ylim(0, n_messages)
        ax_large.invert_yaxis()  # Match typical recurrence plot orientation
        
        # Add color key legend on the right
        if phase_info:
            ax_legend = fig.add_subplot(gs[5:9, n_models])  # Last column, rows 5-8
            ax_legend.axis('off')
            
            # Create legend entries
            from matplotlib.patches import Patch
            legend_elements = []
            sorted_phases = sorted(phase_info, key=lambda x: x['start_turn'])
            
            for phase_idx, phase in enumerate(sorted_phases):
                phase_name = phase['name'].replace('_', ' ').title()
                color = distinct_colors[phase_idx % 20]
                legend_elements.append(Patch(facecolor=color, alpha=0.8, label=phase_name))
            
            # Add the legend
            legend = ax_legend.legend(handles=legend_elements, loc='center', 
                                    title='Conversation Phases', fontsize=12,
                                    title_fontsize=14, frameon=True, fancybox=True,
                                    shadow=True)
            legend.get_frame().set_alpha(0.9)
            legend.get_frame().set_facecolor('white')
            legend.get_frame().set_edgecolor('gray')
        
        # Overall title
        plt.suptitle(f"Comprehensive Ensemble Analysis - {conversation['metadata']['session_id'][:12]}", 
                    fontsize=16, y=0.99)
        
        if save_individual:
            distance_dir = self.output_dir / 'distance_matrices'
            distance_dir.mkdir(exist_ok=True)
            
            session_id = conversation['metadata']['session_id']
            tier = conversation.get('tier', 'unknown')
            filename = f"{tier}_{session_id[:12]}_ensemble_comprehensive.png"
            output_path = distance_dir / filename
            
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            conversation['ensemble_distance_matrix_path'] = str(output_path)
        
        plt.close()
        
        # Calculate cross-model correlations
        correlations = {}
        model_pairs = [(model_names[i], model_names[j]) 
                      for i in range(n_models) for j in range(i+1, n_models)]
        
        for model1, model2 in model_pairs:
            # Compare distance matrices
            triu_indices = np.triu_indices(n_messages, k=1)
            dist1 = distance_matrices[model1][triu_indices]
            dist2 = distance_matrices[model2][triu_indices]
            
            from scipy.stats import spearmanr
            corr, p_value = spearmanr(dist1, dist2)
            correlations[f'{model1}-{model2}'] = {
                'correlation': corr,
                'p_value': p_value
            }
        
        # Analyze loops and patterns (using primary model for consistency)
        primary_embeddings = np.array([m['embedding'] for m in conversation['embedded_messages']])
        primary_self_sim = self_similarities[self.model_name]
        
        # Find loops with adaptive threshold
        threshold = self._calculate_adaptive_similarity_threshold(primary_embeddings)
        loops = []
        min_loop_size = max(10, n_messages // 20)
        
        for i in range(n_messages):
            for j in range(i + min_loop_size, n_messages):
                if primary_self_sim[i, j] > threshold:
                    loops.append({
                        'start': i,
                        'end': j,
                        'similarity': primary_self_sim[i, j],
                        'loop_size': j - i,
                        'z_score': (primary_self_sim[i, j] - self.similarity_stats['mean']) / self.similarity_stats['std']
                    })
        
        # Return comprehensive results
        return {
            'distance_matrices': distance_matrices,
            'self_similarities': self_similarities,
            'cross_model_correlations': correlations,
            'mean_correlation': np.mean([c['correlation'] for c in correlations.values()]),
            'embedding_loops': loops[:10],
            'loop_count': len(loops),
            'phase_info': phase_info
        }
    
    def create_distance_matrices(self, conversation, save_individual=True):
            """Create and analyze distance matrices in full embedding space"""
            messages = conversation['embedded_messages']
            embeddings = np.array([m['embedding'] for m in messages])
            n = len(embeddings)
            
            # If ensemble mode, create distance matrices for all models
            if self.ensemble_mode and 'ensemble_embeddings' in conversation:
                return self.create_ensemble_distance_matrices(conversation, save_individual)
            
            # Otherwise continue with single model analysis
            
            # Get phase information if available
            phase_info = []
            if 'phase_metrics' in conversation and 'phase_embeddings' in conversation['phase_metrics']:
                phase_embeddings = conversation['phase_metrics']['phase_embeddings']
                for phase_name, phase_data in phase_embeddings.items():
                    phase_info.append({
                        'name': phase_name,
                        'start_turn': phase_data['start_turn'],
                        'turn_range': phase_data['turn_range']
                    })
                # Sort by start turn
                phase_info.sort(key=lambda x: x['start_turn'])
            
            # Compute full distance matrices with GPU acceleration if available
            try:
                import torch
                if torch.cuda.is_available():
                    # GPU-accelerated distance computation
                    device = torch.device('cuda')
                    embeddings_tensor = torch.from_numpy(embeddings).float().to(device)
                    
                    # Euclidean distances using broadcasting
                    diff = embeddings_tensor.unsqueeze(0) - embeddings_tensor.unsqueeze(1)
                    euclidean_distances = torch.norm(diff, dim=2).cpu().numpy()
                    
                    # Cosine distances
                    # Normalize embeddings
                    normalized = embeddings_tensor / torch.norm(embeddings_tensor, dim=1, keepdim=True)
                    # Compute cosine similarity matrix
                    cosine_sim = torch.mm(normalized, normalized.t())
                    cosine_distances = (1 - cosine_sim).cpu().numpy()
                else:
                    raise RuntimeError("No GPU available")
                    
            except Exception as e:
                # Fallback to CPU computation
                euclidean_distances = np.zeros((n, n))
                cosine_distances = np.zeros((n, n))
                
                for i in range(n):
                    for j in range(n):
                        # Ensure vectors are 1-D for scipy distance functions
                        vec_i = embeddings[i].flatten()
                        vec_j = embeddings[j].flatten()
                        euclidean_distances[i, j] = self._safe_euclidean(vec_i, vec_j)
                        cosine_distances[i, j] = self._safe_cosine(vec_i, vec_j)
            
            # Analyze distance patterns
            # 1. Self-similarity matrix (for detecting returns to topics)
            self_similarity = 1 - cosine_distances
            
            # 2. Recurrence analysis using FAN (Fixed Amount of Neighbors) method
            # FAN method: each point has a fixed number of neighbors
            # This is more robust than arbitrary threshold selection
            k_neighbors = max(2, int(0.05 * n))  # 5% of points as neighbors, min 2
            recurrence_matrix = np.zeros((n, n), dtype=bool)
            
            for i in range(n):
                # Find k nearest neighbors for each point
                distances_from_i = euclidean_distances[i, :]
                # Exclude self (distance 0)
                neighbor_indices = np.argsort(distances_from_i)[1:k_neighbors+1]
                recurrence_matrix[i, neighbor_indices] = True
                recurrence_matrix[neighbor_indices, i] = True  # Symmetric
            
            # 3. Find embedding loops with adaptive threshold
            # Calculate adaptive threshold
            threshold = self._calculate_adaptive_similarity_threshold(embeddings)
            
            loops = []
            min_loop_size = max(10, n // 20)  # Dynamic min loop size
            
            for i in range(n):
                for j in range(i + min_loop_size, n):
                    if self_similarity[i, j] > threshold:
                        loops.append({
                            'start': i,
                            'end': j,
                            'similarity': self_similarity[i, j],
                            'loop_size': j - i,
                            'z_score': (self_similarity[i, j] - self.similarity_stats['mean']) / self.similarity_stats['std']
                        })
            
            # 4. Analyze distance distribution
            upper_triangle = euclidean_distances[np.triu_indices(n, k=1)]
            
            # 5. Create heatmap visualization
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            # Euclidean distance heatmap
            im1 = axes[0].imshow(euclidean_distances, cmap='viridis', aspect='auto')
            axes[0].set_title('Euclidean Distance Matrix')
            axes[0].set_xlabel('Turn')
            axes[0].set_ylabel('Turn')
            plt.colorbar(im1, ax=axes[0])
            
            # Self-similarity heatmap
            im2 = axes[1].imshow(self_similarity, cmap='RdBu_r', aspect='auto', vmin=0, vmax=1)
            axes[1].set_title('Self-Similarity Matrix')
            axes[1].set_xlabel('Turn')
            axes[1].set_ylabel('Turn')
            plt.colorbar(im2, ax=axes[1])
            
            # Recurrence plot
            axes[2].imshow(recurrence_matrix, cmap='binary', aspect='auto')
            axes[2].set_title('Recurrence Plot')
            axes[2].set_xlabel('Turn')
            axes[2].set_ylabel('Turn')
            
            # Add phase annotations only to recurrence plot
            if phase_info and len(axes) > 2:
                ax = axes[2]  # Recurrence plot only
                
                for i, phase in enumerate(phase_info):
                    start_turn = phase['start_turn']
                    phase_name = phase['name']
                    
                    # Clean phase name - extract just the current state
                    # Handle patterns like "exploration transitioning to synthesis"
                    if 'transitioning to' in phase_name:
                        clean_name = phase_name.split('transitioning to')[0].strip()
                    else:
                        clean_name = phase_name
                    
                    # Further cleanup - remove underscores and capitalize
                    clean_name = clean_name.replace('_', ' ').title()
                    
                    # Add vertical and horizontal lines at phase boundaries
                    if start_turn < n and i > 0:  # Don't draw line at turn 0
                        ax.axhline(y=start_turn, color='red', linestyle='--', alpha=0.6, linewidth=1)
                        ax.axvline(x=start_turn, color='red', linestyle='--', alpha=0.6, linewidth=1)
                    
                    # Determine end turn for this phase
                    if i < len(phase_info) - 1:
                        end_turn = phase_info[i + 1]['start_turn']
                    else:
                        end_turn = n
                    
                    # Add phase label on the right side of the plot
                    mid_turn = (start_turn + end_turn) / 2
                    if mid_turn < n:
                        ax.text(n + 1, mid_turn, clean_name, 
                               fontsize=9, color='red', 
                               horizontalalignment='left', 
                               verticalalignment='center',
                               rotation=0, alpha=0.9)
            
            plt.tight_layout()
            
            # Save individual distance matrices if requested
            if save_individual:
                # Create subfolder for distance matrices
                distance_dir = self.output_dir / 'distance_matrices'
                distance_dir.mkdir(exist_ok=True)
                
                # Create unique filename using session_id and tier
                session_id = conversation['metadata']['session_id']
                tier = conversation.get('tier', 'unknown')
                filename = f"{tier}_{session_id[:12]}_distance_matrices.png"
                output_path = distance_dir / filename
                
                plt.savefig(output_path, dpi=300, bbox_inches='tight')
                
                # Store the path for later use
                conversation['distance_matrix_path'] = str(output_path)
            
            plt.close()
            
            return {
                'distance_stats': {
                    'mean_distance': np.mean(upper_triangle),
                    'std_distance': np.std(upper_triangle),
                    'min_distance': np.min(upper_triangle),
                    'max_distance': np.max(upper_triangle),
                    'distance_quartiles': np.percentile(upper_triangle, [25, 50, 75]).tolist()
                },
                'recurrence_stats': {
                    'recurrence_rate': np.sum(recurrence_matrix[np.triu_indices(n, k=1)]) / (n * (n-1) / 2),
                    'determinism': self._calculate_determinism(recurrence_matrix),
                    'max_diagonal_length': self._max_diagonal_length(recurrence_matrix)
                },
                'embedding_loops': loops[:10],  # Top 10 loops
                'loop_count': len(loops)
            }
    
    def _calculate_determinism(self, recurrence_matrix):
        """Calculate determinism from recurrence matrix"""
        n = len(recurrence_matrix)
        diagonal_lengths = []
        
        # Find all diagonal lines
        for k in range(1, n):
            i, j = 0, k
            current_length = 0
            while i < n and j < n:
                if recurrence_matrix[i, j]:
                    current_length += 1
                else:
                    if current_length > 2:  # Minimum diagonal length
                        diagonal_lengths.append(current_length)
                    current_length = 0
                i += 1
                j += 1
            if current_length > 2:
                diagonal_lengths.append(current_length)
        
        if not diagonal_lengths:
            return 0
        
        # Determinism = ratio of recurrence points in diagonals to all recurrence points
        points_in_diagonals = sum(diagonal_lengths)
        total_recurrence_points = np.sum(recurrence_matrix[np.triu_indices(n, k=1)])
        
        return points_in_diagonals / total_recurrence_points if total_recurrence_points > 0 else 0
    
    def _max_diagonal_length(self, recurrence_matrix):
        """Find maximum diagonal length in recurrence matrix"""
        n = len(recurrence_matrix)
        max_length = 0
        
        for k in range(1, n):
            i, j = 0, k
            current_length = 0
            while i < n and j < n:
                if recurrence_matrix[i, j]:
                    current_length += 1
                    max_length = max(max_length, current_length)
                else:
                    current_length = 0
                i += 1
                j += 1
        
        return max_length
    
    def compare_projection_methods(self, conversation):
        """Compare different dimensionality reduction methods"""
        messages = conversation['embedded_messages']
        embeddings = np.array([m['embedding'] for m in messages])
        
        # Standardize embeddings
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        embeddings_scaled = scaler.fit_transform(embeddings)
        
        # Different projection methods
        projections = {}
        
        # 1. PCA
        from sklearn.decomposition import PCA
        pca = PCA(n_components=3)
        projections['pca'] = pca.fit_transform(embeddings_scaled)
        
        # 2. t-SNE with different perplexities
        from sklearn.manifold import TSNE
        for perp in [5, 30, 50]:
            tsne = TSNE(n_components=3, perplexity=min(perp, len(embeddings)-1), 
                       random_state=42, n_iter=1000)
            projections[f'tsne_perp{perp}'] = tsne.fit_transform(embeddings_scaled)
        
        # 3. UMAP (if available)
        try:
            import umap
            reducer = umap.UMAP(n_components=3, random_state=42)
            projections['umap'] = reducer.fit_transform(embeddings_scaled)
        except ImportError:
            print("UMAP not available. Install with: pip install umap-learn")
        
        # Calculate preservation metrics
        preservation_metrics = {}
        
        # Original distances (sample for efficiency)
        n_samples = min(1000, len(embeddings) * (len(embeddings) - 1) // 2)
        sample_pairs = np.random.choice(len(embeddings), size=(n_samples, 2), replace=True)
        sample_pairs = sample_pairs[sample_pairs[:, 0] != sample_pairs[:, 1]]  # Remove self-pairs
        
        original_distances = []
        for i, j in sample_pairs:
            original_distances.append(self._safe_euclidean(embeddings[i], embeddings[j]))
        original_distances = np.array(original_distances)
        
        # Calculate preservation for each method
        for method, projection in projections.items():
            projected_distances = []
            for i, j in sample_pairs:
                projected_distances.append(self._safe_euclidean(projection[i], projection[j]))
            projected_distances = np.array(projected_distances)
            
            # Spearman correlation
            from scipy.stats import spearmanr
            correlation, _ = spearmanr(original_distances, projected_distances)
            
            # Trustworthiness and continuity
            from sklearn.manifold import trustworthiness
            trust = trustworthiness(embeddings, projection, n_neighbors=12)
            
            preservation_metrics[method] = {
                'spearman_correlation': correlation,
                'trustworthiness': trust
            }
        
        # Visualize comparison
        n_methods = len(projections)
        fig = plt.figure(figsize=(5 * n_methods, 5))
        
        for idx, (method, projection) in enumerate(projections.items()):
            ax = fig.add_subplot(1, n_methods, idx + 1, projection='3d')
            
            # Plot trajectory
            ax.plot(projection[:, 0], projection[:, 1], projection[:, 2], 'b-', alpha=0.5)
            
            # Color by turn
            scatter = ax.scatter(projection[:, 0], projection[:, 1], projection[:, 2], 
                               c=range(len(projection)), cmap='viridis', s=50)
            
            ax.set_title(f'{method.upper()}\nCorr: {preservation_metrics[method]["spearman_correlation"]:.3f}')
            ax.set_xlabel('Dim 1')
            ax.set_ylabel('Dim 2')
            ax.set_zlabel('Dim 3')
        
        plt.tight_layout()
        output_path = self.output_dir / 'projection_comparison.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return preservation_metrics    
        
    def analyze_full_dimensional_structure(self, conversation):
        """Analyze embedding structure in full embedding space (no dimensionality reduction)"""
        messages = conversation['embedded_messages']
        embeddings = np.array([m['embedding'] for m in messages])
        
        results = {
            'intrinsic_dimensionality': self._estimate_intrinsic_dimensionality(embeddings),
            'embedding_axes': self._identify_embedding_axes(embeddings),
            'trajectory_smoothness': self._calculate_trajectory_smoothness(embeddings),
            'embedding_velocity_profile': self._analyze_velocity_profile(embeddings),
            'phase_separability': self._analyze_phase_separability(conversation),
            'dimensional_utilization': self._analyze_dimensional_utilization(embeddings)
        }
        
        return results
    
    def _estimate_intrinsic_dimensionality(self, embeddings):
        """Estimate the intrinsic dimensionality with confidence intervals"""
        from sklearn.neighbors import NearestNeighbors
        from scipy import stats
        
        n_samples = len(embeddings)
        
        # More robust k selection for 150-250 message conversations
        k = min(50, n_samples // 3)  # Increased from 20 to 50
        
        # Add sample size validation
        if n_samples < 100:
            print(f"Warning: Sample size {n_samples} may be too small for reliable dimensionality estimation")
        
        nbrs = NearestNeighbors(n_neighbors=k)
        nbrs.fit(embeddings)
        distances, _ = nbrs.kneighbors(embeddings)
        
        # Use TWO-NN estimator (Facco et al. 2017) - more robust
        distances_excl_self = distances[:, 1:]  # Exclude self
        ratios = distances[:, 2] / distances[:, 1]  # ratio of 2nd to 1st neighbor
        ratios = ratios[ratios > 1]  # Remove any numerical errors
        
        # MLE dimension estimate
        if len(ratios) > 0:
            mu = np.mean(np.log(ratios))
            mle_dimension_twonn = 2 / mu
        else:
            mle_dimension_twonn = None
        
        # Original Levina-Bickel for comparison
        log_distances = np.log(distances_excl_self + 1e-10)
        mle_dims = []
        
        for i in range(n_samples):
            if np.max(log_distances[i]) > np.min(log_distances[i]):
                dim_estimate = (k - 1) / np.sum(log_distances[i] - np.min(log_distances[i]))
                if 0 < dim_estimate < self.embedding_dim:  # Sanity check
                    mle_dims.append(dim_estimate)
        
        # Bootstrap for confidence intervals
        n_bootstrap = 1000
        bootstrap_estimates = []
        
        if mle_dims:
            for _ in range(n_bootstrap):
                # Resample with replacement
                bootstrap_indices = np.random.choice(len(mle_dims), len(mle_dims), replace=True)
                bootstrap_sample = [mle_dims[i] for i in bootstrap_indices]
                bootstrap_estimates.append(np.median(bootstrap_sample))
            
            mle_median = np.median(mle_dims)
            mle_ci_lower = np.percentile(bootstrap_estimates, 2.5)
            mle_ci_upper = np.percentile(bootstrap_estimates, 97.5)
        else:
            mle_median = None
            mle_ci_lower = None
            mle_ci_upper = None
        
        # Use more robust correlation dimension estimation
        # Increase sample size and points
        n_pairs = min(5000, n_samples * (n_samples - 1) // 2)  # Increased from 1000
        sampled_distances = []
        
        for _ in range(n_pairs):
            i, j = np.random.choice(n_samples, 2, replace=False)
            dist = np.linalg.norm(embeddings[i] - embeddings[j])
            if dist > 0:
                sampled_distances.append(dist)
        
        sampled_distances = np.array(sampled_distances)
        
        if len(sampled_distances) > 10:
            # Find linear region in log-log plot
            r_min, r_max = np.percentile(sampled_distances, [10, 90])
            r_values = np.logspace(np.log10(r_min), np.log10(r_max), 15)
            
            correlation_counts = []
            for r in r_values:
                count = np.sum(sampled_distances < r) / len(sampled_distances)
                correlation_counts.append(count)
            
            # Fit in linear region only
            log_r = np.log(r_values)
            log_counts = np.log(np.array(correlation_counts) + 1e-10)
            
            # Find region with best linear fit
            valid_mask = (np.array(correlation_counts) > 0.1) & (np.array(correlation_counts) < 0.9)
            
            if np.sum(valid_mask) > 3:
                slope, intercept, r_value, p_value, std_err = stats.linregress(
                    log_r[valid_mask], log_counts[valid_mask]
                )
                
                # 95% confidence interval
                t_value = stats.t.ppf(0.975, np.sum(valid_mask) - 2)
                correlation_dim = slope
                correlation_ci_lower = slope - t_value * std_err
                correlation_ci_upper = slope + t_value * std_err
            else:
                correlation_dim = None
                correlation_ci_lower = None
                correlation_ci_upper = None
        else:
            correlation_dim = None
            correlation_ci_lower = None
            correlation_ci_upper = None
        
        # Add consensus estimate
        estimates = []
        if mle_dimension_twonn and mle_dimension_twonn < self.embedding_dim:
            estimates.append(mle_dimension_twonn)
        if mle_median:
            estimates.append(mle_median)
        if correlation_dim and correlation_dim < self.embedding_dim:
            estimates.append(correlation_dim)
        
        consensus_estimate = np.median(estimates) if estimates else None
        
        return {
            'mle_dimension': mle_median,
            'mle_dimension_twonn': mle_dimension_twonn,  # New, more robust
            'consensus_dimension': consensus_estimate,    # New
            'estimation_quality': 'high' if n_samples > 150 else 'moderate',
            'mle_ci_lower': mle_ci_lower,
            'mle_ci_upper': mle_ci_upper,
            'correlation_dimension': correlation_dim,
            'correlation_ci_lower': correlation_ci_lower,
            'correlation_ci_upper': correlation_ci_upper,
            'effective_rank': np.linalg.matrix_rank(embeddings - embeddings.mean(axis=0)),
            'n_samples': n_samples
        }
    
    def _identify_embedding_axes(self, embeddings):
        """Identify primary embedding axes using PCA in full space"""
        from sklearn.decomposition import PCA
        
        # Center the embeddings
        centered = embeddings - embeddings.mean(axis=0)
        
        # Full PCA
        pca = PCA()
        pca.fit(centered)
        
        # Find number of components for different variance thresholds
        cumsum = np.cumsum(pca.explained_variance_ratio_)
        n_components_50 = np.argmax(cumsum >= 0.5) + 1
        n_components_90 = np.argmax(cumsum >= 0.9) + 1
        n_components_95 = np.argmax(cumsum >= 0.95) + 1
        
        # Analyze top components
        top_components = []
        for i in range(min(10, len(pca.components_))):
            # Find dimensions with highest loadings
            component = pca.components_[i]
            top_dims = np.argsort(np.abs(component))[-5:]
            
            top_components.append({
                'component': i,
                'variance_explained': pca.explained_variance_ratio_[i],
                'top_dimensions': top_dims.tolist(),
                'max_loading': np.max(np.abs(component))
            })
        
        return {
            'n_components_50_variance': n_components_50,
            'n_components_90_variance': n_components_90,
            'n_components_95_variance': n_components_95,
            'top_components': top_components,
            'total_variance': pca.explained_variance_.sum()
        }
    
    def _calculate_trajectory_smoothness(self, embeddings):
        """Calculate smoothness metrics in full dimensional space"""
        if len(embeddings) < 3:
            return None
        
        # First derivatives (velocities)
        velocities = np.diff(embeddings, axis=0)
        velocity_norms = np.linalg.norm(velocities, axis=1)
        
        # Second derivatives (accelerations)
        accelerations = np.diff(velocities, axis=0)
        acceleration_norms = np.linalg.norm(accelerations, axis=1)
        
        # Jerk (third derivative)
        if len(embeddings) > 3:
            jerks = np.diff(accelerations, axis=0)
            jerk_norms = np.linalg.norm(jerks, axis=1).tolist()  # Convert to list
        else:
            jerk_norms = []
        
        # Spectral analysis of trajectory
        from scipy.fft import fft
        
        # Analyze frequency content of each dimension
        freq_powers = []
        for dim in range(embeddings.shape[1]):
            dim_trajectory = embeddings[:, dim]
            if len(dim_trajectory) > 10:
                spectrum = np.abs(fft(dim_trajectory - dim_trajectory.mean()))
                # Normalize and take only positive frequencies
                spectrum = spectrum[:len(spectrum)//2]
                if spectrum.sum() > 0:
                    spectrum = spectrum / spectrum.sum()
                    # Calculate spectral entropy
                    spectral_entropy = -np.sum(spectrum * np.log(spectrum + 1e-10))
                    freq_powers.append(spectral_entropy)
        
        return {
            'mean_velocity': np.mean(velocity_norms),
            'velocity_variance': np.var(velocity_norms),
            'mean_acceleration': np.mean(acceleration_norms),
            'acceleration_variance': np.var(acceleration_norms),
            'mean_jerk': np.mean(jerk_norms) if len(jerk_norms) > 0 else None,
            'smoothness_index': np.mean(velocity_norms) / (np.std(velocity_norms) + 1e-8),
            'spectral_entropy': np.mean(freq_powers) if freq_powers else None
        }
    
    def _analyze_velocity_profile(self, embeddings):
        """Analyze how embedding velocity changes over conversation"""
        if len(embeddings) < 2:
            return None
        
        velocities = []
        for i in range(1, len(embeddings)):
            velocity = np.linalg.norm(embeddings[i] - embeddings[i-1])
            velocities.append(velocity)
        
        # Segment conversation into quarters
        quarter_size = len(velocities) // 4
        quarters = []
        for i in range(4):
            start = i * quarter_size
            end = (i + 1) * quarter_size if i < 3 else len(velocities)
            if start < end:
                quarters.append({
                    'quarter': i + 1,
                    'mean_velocity': np.mean(velocities[start:end]),
                    'std_velocity': np.std(velocities[start:end])
                })
        
        # Detect velocity peaks (topic shifts)
        velocity_array = np.array(velocities)
        mean_vel = np.mean(velocity_array)
        std_vel = np.std(velocity_array)
        peaks = np.where(velocity_array > mean_vel + 2 * std_vel)[0]
        
        return {
            'velocity_quarters': quarters,
            'peak_turns': peaks.tolist(),
            'peak_count': len(peaks),
            'velocity_trend': np.polyfit(range(len(velocities)), velocities, 1)[0]
        }
    
    def _analyze_phase_separability(self, conversation):
        """Analyze how well phases are separated in embedding space"""
        if 'phase_metrics' not in conversation:
            return None
        
        phase_data = conversation['phase_metrics']['phase_embeddings']
        if len(phase_data) < 2:
            return None
        
        # Calculate inter-phase distances
        phase_names = list(phase_data.keys())
        inter_phase_distances = {}
        
        for i in range(len(phase_names)):
            for j in range(i + 1, len(phase_names)):
                p1, p2 = phase_names[i], phase_names[j]
                dist = np.linalg.norm(phase_data[p1]['centroid'] - phase_data[p2]['centroid'])
                inter_phase_distances[f"{p1}-{p2}"] = dist
        
        # Calculate silhouette coefficient if we have enough data
        messages = conversation['embedded_messages']
        phase_labels = []
        phase_embeddings = []
        
        # Map messages to phases
        for msg in messages:
            turn = msg['turn']
            for phase_name, info in phase_data.items():
                if info['turn_range'][0] <= turn <= info['turn_range'][1]:
                    phase_labels.append(phase_name)
                    phase_embeddings.append(msg['embedding'])
                    break
        
        silhouette = None
        if len(set(phase_labels)) > 1 and len(phase_labels) > len(set(phase_labels)):
            from sklearn.metrics import silhouette_score
            try:
                silhouette = silhouette_score(np.array(phase_embeddings), phase_labels)
            except:
                silhouette = None
        
        # Davies-Bouldin Index (lower is better)
        davies_bouldin = None
        if len(phase_data) > 1:
            db_scores = []
            for i, (p1, d1) in enumerate(phase_data.items()):
                max_ratio = 0
                for j, (p2, d2) in enumerate(phase_data.items()):
                    if i != j:
                        within_cluster = d1['spread'] + d2['spread']
                        between_cluster = np.linalg.norm(d1['centroid'] - d2['centroid'])
                        if between_cluster > 0:
                            ratio = within_cluster / between_cluster
                            max_ratio = max(max_ratio, ratio)
                db_scores.append(max_ratio)
            davies_bouldin = np.mean(db_scores)
        
        return {
            'inter_phase_distances': inter_phase_distances,
            'mean_inter_phase_distance': np.mean(list(inter_phase_distances.values())),
            'silhouette_coefficient': silhouette,
            'davies_bouldin_index': davies_bouldin,
            'phase_spreads': {p: d['spread'] for p, d in phase_data.items()}
        }
    
    def _analyze_dimensional_utilization(self, embeddings):
        """Analyze how different dimensions are utilized"""
        # Variance per dimension
        dim_variances = np.var(embeddings, axis=0)
        
        # Dimensions sorted by variance
        sorted_dims = np.argsort(dim_variances)[::-1]
        
        # Participation ratio (effective number of dimensions)
        normalized_var = dim_variances / dim_variances.sum()
        participation_ratio = 1 / np.sum(normalized_var ** 2)
        
        # Kurtosis per dimension (to find sparse dimensions)
        from scipy.stats import kurtosis
        dim_kurtosis = kurtosis(embeddings, axis=0)
        
        # Find dormant dimensions (low variance)
        dormant_threshold = np.percentile(dim_variances, 10)
        dormant_dims = np.where(dim_variances < dormant_threshold)[0]
        
        # Find active dimensions (high variance)
        active_threshold = np.percentile(dim_variances, 90)
        active_dims = np.where(dim_variances > active_threshold)[0]
        
        return {
            'participation_ratio': participation_ratio,
            'top_10_dims': sorted_dims[:10].tolist(),
            'top_10_variances': dim_variances[sorted_dims[:10]].tolist(),
            'dormant_dimensions': len(dormant_dims),
            'active_dimensions': len(active_dims),
            'mean_kurtosis': np.mean(dim_kurtosis),
            'dimension_entropy': -np.sum(normalized_var * np.log(normalized_var + 1e-10))
        }
    
    def visualize_embedding_analysis(self, conversation):
        """Create comprehensive visualizations of embedding space analysis"""
        import matplotlib.pyplot as plt
        
        # Run full dimensional analysis
        full_analysis = self.analyze_full_dimensional_structure(conversation)
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Full Dimensional Embedding Space Analysis', fontsize=16)
        
        # 1. Variance per dimension
        if 'dimensional_utilization' in full_analysis:
            dim_util = full_analysis['dimensional_utilization']
            ax = axes[0, 0]
            ax.bar(range(10), dim_util['top_10_variances'])
            ax.set_xlabel('Top 10 Dimensions')
            ax.set_ylabel('Variance')
            ax.set_title('Dimensional Variance Distribution')
        
        # 2. Velocity profile
        if 'embedding_velocity_profile' in full_analysis:
            vel_profile = full_analysis['embedding_velocity_profile']
            ax = axes[0, 1]
            quarters = vel_profile['velocity_quarters']
            x = [q['quarter'] for q in quarters]
            y = [q['mean_velocity'] for q in quarters]
            err = [q['std_velocity'] for q in quarters]
            ax.errorbar(x, y, yerr=err, marker='o')
            ax.set_xlabel('Conversation Quarter')
            ax.set_ylabel('Mean Embedding Velocity')
            ax.set_title('Velocity Profile Over Time')
        
        # 3. Phase separability
        if 'phase_separability' in full_analysis and full_analysis['phase_separability']:
            phase_sep = full_analysis['phase_separability']
            ax = axes[0, 2]
            if phase_sep['phase_spreads']:
                phases = list(phase_sep['phase_spreads'].keys())
                spreads = list(phase_sep['phase_spreads'].values())
                ax.bar(phases, spreads)
                ax.set_xlabel('Phase')
                ax.set_ylabel('Spread')
                ax.set_title('Phase Spread in Embedding Space')
                ax.tick_params(axis='x', rotation=45)
        
        # 4. Intrinsic dimensionality
        if 'intrinsic_dimensionality' in full_analysis:
            intrinsic = full_analysis['intrinsic_dimensionality']
            ax = axes[1, 0]
            dims = []
            labels = []
            if intrinsic['mle_dimension']:
                dims.append(intrinsic['mle_dimension'])
                labels.append('MLE')
            if intrinsic['correlation_dimension']:
                dims.append(intrinsic['correlation_dimension'])
                labels.append('Correlation')
            if intrinsic['effective_rank']:
                dims.append(intrinsic['effective_rank'])
                labels.append('Effective Rank')
            
            ax.bar(labels, dims)
            ax.set_ylabel('Dimensionality')
            ax.set_title('Intrinsic Dimensionality Estimates')
            ax.axhline(y=self.embedding_dim, color='r', linestyle='--', label=f'Full dim ({self.embedding_dim})')
            ax.legend()
        
        # 5. Explained variance curve
        if 'embedding_axes' in full_analysis:
            axes_data = full_analysis['embedding_axes']
            ax = axes[1, 1]
            components = [c['component'] for c in axes_data['top_components']]
            variance = [c['variance_explained'] for c in axes_data['top_components']]
            cumvar = np.cumsum(variance)
            
            ax.plot(components, cumvar, 'b-', label='Cumulative')
            ax.bar(components, variance, alpha=0.5, label='Individual')
            ax.axhline(y=0.9, color='r', linestyle='--', label='90% threshold')
            ax.set_xlabel('Component')
            ax.set_ylabel('Variance Explained')
            ax.set_title('PCA Variance Explained')
            ax.legend()
        
        # 6. Smoothness metrics
        if 'trajectory_smoothness' in full_analysis:
            smooth = full_analysis['trajectory_smoothness']
            ax = axes[1, 2]
            metrics = ['Velocity', 'Acceleration', 'Jerk']
            means = [smooth['mean_velocity'], smooth['mean_acceleration'], 
                    smooth['mean_jerk'] if smooth['mean_jerk'] else 0]
            
            ax.bar(metrics, means)
            ax.set_ylabel('Mean Value')
            ax.set_title('Trajectory Smoothness Metrics')
        
        plt.tight_layout()
        
        # Save
        output_path = self.output_dir / 'full_dimensional_analysis.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return full_analysis
    
    def create_synthetic_validation(self):
        """Create synthetic conversations with known properties for validation"""
        print("\n" + "="*70)
        print("SYNTHETIC VALIDATION")
        print("="*70)
        
        synthetic_conversations = []
        
        # 1. Linear trajectory (no breakdown)
        print("Creating synthetic linear trajectory...")
        linear_points = []
        direction = np.random.randn(self.embedding_dim)
        direction = direction / np.linalg.norm(direction)
        
        for i in range(50):
            point = i * 0.1 * direction + np.random.randn(self.embedding_dim) * 0.01
            linear_points.append(point)
        
        synthetic_conversations.append({
            'type': 'linear',
            'embeddings': np.array(linear_points),
            'expected_curvature': 'low',
            'expected_phases': 1,
            'expected_dimension': 1,
            'expected_convergence': False
        })
        
        # 2. Spiral trajectory (gradual drift)
        print("Creating synthetic spiral trajectory...")
        spiral_points = []
        for i in range(50):
            t = i * 0.2
            # Spiral in 3D subspace
            point = np.zeros(self.embedding_dim)
            point[0] = t * np.cos(t) * 0.1
            point[1] = t * np.sin(t) * 0.1
            point[2] = t * 0.05
            point += np.random.randn(self.embedding_dim) * 0.01
            spiral_points.append(point)
        
        synthetic_conversations.append({
            'type': 'spiral',
            'embeddings': np.array(spiral_points),
            'expected_curvature': 'medium',
            'expected_phases': 2,
            'expected_dimension': 3,
            'expected_convergence': False
        })
        
        # 3. Convergent trajectory (attractor)
        print("Creating synthetic convergent trajectory...")
        convergent_points = []
        attractor = np.random.randn(self.embedding_dim)
        
        for i in range(50):
            if i < 20:
                # Random walk phase
                if i == 0:
                    point = np.random.randn(self.embedding_dim)
                else:
                    point = convergent_points[-1] + np.random.randn(self.embedding_dim) * 0.1
            else:
                # Convergence phase
                alpha = 0.9  # Convergence rate
                point = alpha * convergent_points[-1] + (1 - alpha) * attractor
                point += np.random.randn(self.embedding_dim) * 0.01
            convergent_points.append(point)
        
        synthetic_conversations.append({
            'type': 'convergent',
            'embeddings': np.array(convergent_points),
            'expected_curvature': 'high',
            'expected_phases': 2,
            'expected_dimension': 2,
            'expected_convergence': True
        })
        
        # 4. Phase transition trajectory
        print("Creating synthetic phase transition trajectory...")
        phase_points = []
        phase_centers = [np.random.randn(self.embedding_dim) for _ in range(3)]
        
        for i in range(50):
            if i < 17:
                center = phase_centers[0]
            elif i < 34:
                center = phase_centers[1]
            else:
                center = phase_centers[2]
            
            point = center + np.random.randn(self.embedding_dim) * 0.05
            phase_points.append(point)
        
        synthetic_conversations.append({
            'type': 'phase_transition',
            'embeddings': np.array(phase_points),
            'expected_curvature': 'low',
            'expected_phases': 3,
            'expected_dimension': 2,
            'expected_convergence': False
        })
        
        # Analyze synthetic conversations
        print("\nAnalyzing synthetic conversations...")
        validation_results = []
        
        for synth_conv in synthetic_conversations:
            # Create conversation-like structure
            conv = {
                'embedded_messages': [{'embedding': emb, 'turn': i, 'speaker': 'synthetic'} 
                                    for i, emb in enumerate(synth_conv['embeddings'])],
                'metadata': {'session_id': f"synthetic_{synth_conv['type']}", 'message_count': len(synth_conv['embeddings'])}
            }
            
            # Run analyses
            conv = self.calculate_trajectory_metrics(conv)
            conv = self.calculate_trajectory_curvature(conv)
            
            # Simplified phase detection
            detected_phases = self.detect_embedding_phases(conv)
            
            # Extract metrics
            actual_curvature = conv['curvature_metrics']['mean_curvature']
            actual_phases = len(detected_phases)
            
            # Compare with expected
            result = {
                'type': synth_conv['type'],
                'expected_curvature': synth_conv['expected_curvature'],
                'actual_curvature': actual_curvature,
                'curvature_category': 'low' if actual_curvature < 0.5 else 'medium' if actual_curvature < 1.0 else 'high',
                'expected_phases': synth_conv['expected_phases'],
                'actual_phases': actual_phases,
                'phase_detection_accurate': abs(actual_phases - synth_conv['expected_phases']) <= 1
            }
            
            validation_results.append(result)
            
        # Report results
        print("\nSynthetic Validation Results:")
        print("-" * 50)
        for result in validation_results:
            print(f"\n{result['type'].upper()}:")
            print(f"  Curvature: expected={result['expected_curvature']}, actual={result['curvature_category']} ({result['actual_curvature']:.3f})")
            print(f"  Phases: expected={result['expected_phases']}, actual={result['actual_phases']}, accurate={result['phase_detection_accurate']}")
        
        # Save results
        pd.DataFrame(validation_results).to_csv(self.output_dir / 'synthetic_validation_results.csv', index=False)
        print("\nSaved synthetic validation results")
        
        return validation_results
    
    def generate_null_model_conversations(self, n_conversations=100, conversation_length=100):
        """Generate null model conversations for hypothesis testing
        
        Null models test whether observed patterns could arise by chance.
        We create several types of null models:
        1. Random walk in embedding space
        2. Shuffled real embeddings (preserves distribution)
        3. Markov chain based on transition probabilities
        """
        null_conversations = []
        
        # Get distribution of real embeddings if available
        real_embeddings = []
        if hasattr(self, 'conversations') and self.conversations:
            for conv in self.conversations[:50]:  # Sample from first 50
                if 'embedded_messages' in conv:
                    real_embeddings.extend([m['embedding'] for m in conv['embedded_messages']])
        real_embeddings = np.array(real_embeddings) if real_embeddings else None
        
        for i in range(n_conversations):
            null_type = i % 3  # Cycle through null model types
            
            if null_type == 0:
                # Type 1: Random walk in embedding space
                embeddings = []
                current_point = np.random.randn(self.embedding_dim)
                
                for _ in range(conversation_length):
                    # Random step with controlled magnitude
                    step = np.random.randn(self.embedding_dim) * 0.1
                    current_point = current_point + step
                    # Normalize to unit sphere (like many embedding models)
                    current_point = current_point / np.linalg.norm(current_point)
                    embeddings.append(current_point.copy())
                    
                conv_type = 'random_walk'
                
            elif null_type == 1 and real_embeddings is not None:
                # Type 2: Shuffled real embeddings
                indices = np.random.choice(len(real_embeddings), conversation_length, replace=True)
                embeddings = [real_embeddings[idx].copy() for idx in indices]
                np.random.shuffle(embeddings)  # Break temporal structure
                conv_type = 'shuffled_real'
                
            else:
                # Type 3: Markov chain (simple transition model)
                embeddings = []
                
                if real_embeddings is not None and len(real_embeddings) > 10:
                    # Create simple clusters
                    from sklearn.cluster import KMeans
                    n_states = min(5, len(real_embeddings) // 20)
                    kmeans = KMeans(n_clusters=n_states, n_init=3)
                    kmeans.fit(real_embeddings[:1000])  # Fit on subset
                    
                    # Transition matrix (random)
                    transition_matrix = np.random.dirichlet(np.ones(n_states), size=n_states)
                    
                    # Generate sequence
                    current_state = np.random.randint(n_states)
                    for _ in range(conversation_length):
                        # Sample from current cluster with noise
                        center = kmeans.cluster_centers_[current_state]
                        noise = np.random.randn(self.embedding_dim) * 0.1
                        embeddings.append(center + noise)
                        
                        # Transition to next state
                        current_state = np.random.choice(n_states, p=transition_matrix[current_state])
                else:
                    # Fallback to random walk
                    current_point = np.random.randn(self.embedding_dim)
                    for _ in range(conversation_length):
                        step = np.random.randn(self.embedding_dim) * 0.1
                        current_point = current_point + step
                        embeddings.append(current_point.copy())
                
                conv_type = 'markov_chain'
            
            # Create conversation structure
            null_conv = {
                'metadata': {
                    'session_id': f'null_{conv_type}_{i}',
                    'message_count': conversation_length,
                    'null_model_type': conv_type
                },
                'messages': [{'content': f'null_msg_{j}', 'speaker': 'null'} 
                           for j in range(conversation_length)],
                'embedded_messages': [{'embedding': emb, 'turn': j} 
                                    for j, emb in enumerate(embeddings)],
                'is_null_model': True
            }
            
            null_conversations.append(null_conv)
        
        return null_conversations
    
    def validate_length_normalization(self):
        """Validate that length normalization removes conversation length bias"""
        print("\n" + "="*70)
        print("VALIDATING LENGTH NORMALIZATION")
        print("="*70)
        
        # Group conversations by length bins
        length_bins = [(0, 50), (50, 100), (100, 150), (150, 200), (200, float('inf'))]
        bin_stats = {f"{low}-{high}": [] for low, high in length_bins}
        
        for conv in self.conversations:
            if 'trajectory_stats' not in conv:
                continue
                
            length = conv['trajectory_stats']['conversation_length']
            total_dist = conv['trajectory_stats']['total_distance']
            total_dist_norm = conv['trajectory_stats']['total_distance_normalized']
            
            # Find appropriate bin
            for low, high in length_bins:
                if low <= length < high:
                    bin_name = f"{low}-{high if high != float('inf') else '+'}"
                    bin_stats[bin_name].append({
                        'length': length,
                        'total_distance': total_dist,
                        'normalized_distance': total_dist_norm
                    })
                    break
        
        # Calculate correlations
        all_lengths = []
        all_distances = []
        all_normalized = []
        
        for bin_data in bin_stats.values():
            for item in bin_data:
                all_lengths.append(item['length'])
                all_distances.append(item['total_distance'])
                all_normalized.append(item['normalized_distance'])
        
        if len(all_lengths) > 10:
            # Correlation before normalization
            corr_before, p_before = pearsonr(all_lengths, all_distances)
            # Correlation after normalization
            corr_after, p_after = pearsonr(all_lengths, all_normalized)
            
            print(f"\nLength-Distance Correlation:")
            print(f"  Before normalization: r={corr_before:.3f} (p={p_before:.6f})")
            print(f"  After normalization: r={corr_after:.3f} (p={p_after:.6f})")
            
            if abs(corr_after) < abs(corr_before) * 0.5:
                print("✓ Normalization successfully reduces length bias")
            else:
                print("⚠️  WARNING: Normalization may not be adequately removing length bias")
            
            # Print bin statistics
            print("\nMean normalized distance by conversation length:")
            for bin_name, bin_data in bin_stats.items():
                if bin_data:
                    mean_norm = np.mean([d['normalized_distance'] for d in bin_data])
                    std_norm = np.std([d['normalized_distance'] for d in bin_data])
                    n = len(bin_data)
                    print(f"  {bin_name} messages (n={n}): {mean_norm:.3f} ± {std_norm:.3f}")
    
    def compare_to_null_models(self, real_conversations, n_null=100):
        """Compare real conversation metrics to null models"""
        print("\n" + "="*70)
        print("NULL MODEL COMPARISON")
        print("="*70)
        
        # Generate null conversations
        avg_length = int(np.mean([len(c['embedded_messages']) for c in real_conversations[:20]]))
        null_conversations = self.generate_null_model_conversations(n_null, avg_length)
        
        # Analyze null conversations
        print("Analyzing null model conversations...")
        null_analyzed = []
        for null_conv in tqdm(null_conversations, desc="Processing null models"):
            analyzed = self.embed_conversation(null_conv)
            analyzed = self.calculate_trajectory_metrics(analyzed)
            analyzed = self.calculate_trajectory_curvature(analyzed)
            analyzed = self.detect_convergence_with_confidence(analyzed)
            null_analyzed.append(analyzed)
        
        # Extract metrics for comparison
        metrics_to_compare = [
            ('mean_curvature', 'curvature_metrics'),
            ('total_distance_normalized', 'trajectory_stats'),
            ('normalized_convergence_rate', 'convergence_analysis')
        ]
        
        comparison_results = {}
        
        for metric_name, metric_location in metrics_to_compare:
            # Extract real values
            real_values = []
            for conv in real_conversations:
                if metric_location in conv and conv[metric_location]:
                    if metric_name in conv[metric_location]:
                        real_values.append(conv[metric_location][metric_name])
            
            # Extract null values
            null_values = []
            for conv in null_analyzed:
                if metric_location in conv and conv[metric_location]:
                    if metric_name in conv[metric_location]:
                        null_values.append(conv[metric_location][metric_name])
            
            if real_values and null_values:
                # Statistical comparison
                from scipy.stats import mannwhitneyu, ks_2samp
                
                # Mann-Whitney U test
                u_stat, p_mw = mannwhitneyu(real_values, null_values, alternative='two-sided')
                
                # Kolmogorov-Smirnov test
                ks_stat, p_ks = ks_2samp(real_values, null_values)
                
                # Effect size (Cohen's d)
                mean_real = np.mean(real_values)
                mean_null = np.mean(null_values)
                std_pooled = np.sqrt((np.var(real_values) + np.var(null_values)) / 2)
                cohens_d = (mean_real - mean_null) / std_pooled if std_pooled > 0 else 0
                
                comparison_results[metric_name] = {
                    'mean_real': mean_real,
                    'mean_null': mean_null,
                    'std_real': np.std(real_values),
                    'std_null': np.std(null_values),
                    'mann_whitney_u': u_stat,
                    'p_value_mw': p_mw,
                    'ks_statistic': ks_stat,
                    'p_value_ks': p_ks,
                    'cohens_d': cohens_d,
                    'n_real': len(real_values),
                    'n_null': len(null_values),
                    'significantly_different': p_mw < 0.05
                }
                
                print(f"\n{metric_name}:")
                print(f"  Real: {mean_real:.3f} ± {np.std(real_values):.3f}")
                print(f"  Null: {mean_null:.3f} ± {np.std(null_values):.3f}")
                print(f"  Mann-Whitney p-value: {p_mw:.6f}")
                print(f"  Effect size (Cohen's d): {cohens_d:.3f}")
                print(f"  Significantly different from null: {p_mw < 0.05}")
        
        # Save results
        comparison_df = pd.DataFrame(comparison_results).T
        comparison_df.to_csv(self.output_dir / 'null_model_comparison.csv')
        print("\nSaved null model comparison results")
        
        return comparison_results, null_analyzed
    
    def perturbation_analysis(self, conversation, n_perturbations=10):
        """Analyze trajectory stability under message perturbations"""
        messages = conversation['messages']
        original_embeddings = np.array([m['embedding'] for m in conversation['embedded_messages']])
        
        perturbation_results = []
        
        # Different perturbation types
        perturbation_types = [
            ('paraphrase', 0.1),  # Small embedding shift
            ('noise', 0.05),      # Random noise
            ('synonym', 0.15),    # Larger embedding shift
        ]
        
        for perturb_type, perturb_strength in perturbation_types:
            for _ in range(n_perturbations):
                # Create perturbed embeddings
                perturbed_embeddings = original_embeddings.copy()
                
                if perturb_type == 'noise':
                    # Add Gaussian noise
                    noise = np.random.randn(*original_embeddings.shape) * perturb_strength
                    perturbed_embeddings += noise
                    
                elif perturb_type == 'paraphrase':
                    # Simulate paraphrase by small directed perturbation
                    for i in range(len(perturbed_embeddings)):
                        direction = np.random.randn(self.embedding_dim)
                        direction = direction / np.linalg.norm(direction)
                        perturbed_embeddings[i] += direction * perturb_strength
                        
                elif perturb_type == 'synonym':
                    # Simulate synonym replacement by larger perturbation
                    # Randomly perturb 20% of messages
                    n_perturb = int(0.2 * len(perturbed_embeddings))
                    indices = np.random.choice(len(perturbed_embeddings), n_perturb, replace=False)
                    for idx in indices:
                        direction = np.random.randn(self.embedding_dim)
                        direction = direction / np.linalg.norm(direction)
                        perturbed_embeddings[idx] += direction * perturb_strength
                
                # Calculate trajectory differences
                # 1. Point-wise distances
                point_distances = np.array([np.linalg.norm(original_embeddings[i] - perturbed_embeddings[i]) 
                                          for i in range(len(original_embeddings))])
                
                # 2. Trajectory shape similarity (Procrustes distance)
                # Center both trajectories
                orig_centered = original_embeddings - original_embeddings.mean(axis=0)
                pert_centered = perturbed_embeddings - perturbed_embeddings.mean(axis=0)
                
                # Optimal rotation via SVD
                H = orig_centered.T @ pert_centered
                U, _, Vt = np.linalg.svd(H)
                R = Vt.T @ U.T
                
                # Apply rotation and calculate residual
                pert_aligned = pert_centered @ R
                procrustes_dist = np.linalg.norm(orig_centered - pert_aligned) / np.linalg.norm(orig_centered)
                
                # 3. Phase preservation
                orig_phases = self.detect_embedding_phases({'embedded_messages': [{'embedding': e} for e in original_embeddings]})
                pert_phases = self.detect_embedding_phases({'embedded_messages': [{'embedding': e} for e in perturbed_embeddings]})
                
                phase_preservation = 1.0 - abs(len(orig_phases) - len(pert_phases)) / max(len(orig_phases), 1)
                
                perturbation_results.append({
                    'perturbation_type': perturb_type,
                    'perturbation_strength': perturb_strength,
                    'mean_point_distance': np.mean(point_distances),
                    'max_point_distance': np.max(point_distances),
                    'procrustes_distance': procrustes_dist,
                    'phase_preservation': phase_preservation,
                    'n_original_phases': len(orig_phases),
                    'n_perturbed_phases': len(pert_phases)
                })
        
        return pd.DataFrame(perturbation_results)

    def phase_detection_sensitivity_analysis(self, conversation):
        """Analyze sensitivity of phase detection to parameter choices"""
        messages = conversation['embedded_messages']
        manual_phases = conversation.get('phases', [])
        
        # Use dynamic tolerance based on conversation length
        conversation_length = len(messages)
        tolerance = max(5, int(conversation_length * 0.03))  # 3% of conversation length
        
        # Test different parameter combinations
        window_sizes = [10, 15, 20, 25]  # Added 25 for longer conversations
        threshold_percentiles = [60, 70, 75, 80, 85, 90]
        
        results = []
        
        for window_size in window_sizes:
            for threshold_percentile in threshold_percentiles:
                detected_phases = self.detect_embedding_phases(
                    conversation, 
                    window_size=window_size,
                    threshold_percentile=threshold_percentile
                )
                
                # Improved phase matching with Hungarian algorithm
                if manual_phases and detected_phases:
                    from scipy.optimize import linear_sum_assignment
                    
                    # Create cost matrix based on turn distances
                    manual_turns = [p['turn'] for p in manual_phases]
                    detected_turns = [p['turn'] for p in detected_phases]
                    
                    cost_matrix = np.zeros((len(manual_turns), len(detected_turns)))
                    for i, mt in enumerate(manual_turns):
                        for j, dt in enumerate(detected_turns):
                            cost_matrix[i, j] = abs(mt - dt)
                    
                    # Find optimal assignment
                    row_ind, col_ind = linear_sum_assignment(cost_matrix)
                    
                    # Count matches within tolerance
                    matches = 0
                    for i, j in zip(row_ind, col_ind):
                        if cost_matrix[i, j] <= tolerance:
                            matches += 1
                    
                    # Calculate metrics based on optimal matching
                    precision = matches / len(detected_turns) if detected_turns else 0
                    recall = matches / len(manual_turns) if manual_turns else 0
                    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                    
                    # Add temporal accuracy metric
                    if matches > 0:
                        matched_distances = [cost_matrix[i, j] for i, j in zip(row_ind, col_ind) 
                                           if cost_matrix[i, j] <= tolerance]
                        avg_temporal_error = np.mean(matched_distances)
                    else:
                        avg_temporal_error = conversation_length  # Max penalty
                    
                    results.append({
                        'window_size': window_size,
                        'threshold_percentile': threshold_percentile,
                        'n_detected': len(detected_phases),
                        'n_manual': len(manual_phases),
                        'precision': precision,
                        'recall': recall,
                        'f1_score': f1_score,
                        'avg_temporal_error': avg_temporal_error,
                        'tolerance_used': tolerance
                    })
                else:
                    results.append({
                        'window_size': window_size,
                        'threshold_percentile': threshold_percentile,
                        'n_detected': len(detected_phases),
                        'n_manual': len(manual_phases),
                        'precision': 0,
                        'recall': 0,
                        'f1_score': 0,
                        'avg_temporal_error': conversation_length,
                        'tolerance_used': tolerance
                    })
        
        return pd.DataFrame(results)
    
    def detect_embedding_phases(self, conversation, window_size=10, threshold_percentile=75):
        """Automatically detect conversation phases based on embedding shifts"""
        messages = conversation['embedded_messages']
        
        if len(messages) < window_size * 2:
            return []
        
        # Calculate embedding shift at each point
        embedding_shifts = []
        for i in range(window_size, len(messages) - window_size):
            # Compare windows before and after this point
            before_window = [m['embedding'] for m in messages[i-window_size:i]]
            after_window = [m['embedding'] for m in messages[i:i+window_size]]
            
            # Calculate centroid of each window
            before_centroid = np.mean(before_window, axis=0)
            after_centroid = np.mean(after_window, axis=0)
            
            # Measure shift
            shift = self._safe_euclidean(before_centroid, after_centroid)
            embedding_shifts.append({
                'turn': i,
                'shift': shift
            })
        
        # Find significant shifts (peaks above threshold)
        if embedding_shifts:
            shifts_array = np.array([s['shift'] for s in embedding_shifts])
            threshold = np.percentile(shifts_array, threshold_percentile)
            
            # Find local maxima above threshold
            detected_phases = []
            for i in range(1, len(embedding_shifts) - 1):
                if (embedding_shifts[i]['shift'] > threshold and
                    embedding_shifts[i]['shift'] > embedding_shifts[i-1]['shift'] and
                    embedding_shifts[i]['shift'] > embedding_shifts[i+1]['shift']):
                    
                    detected_phases.append({
                        'turn': embedding_shifts[i]['turn'],
                        'phase': f'auto_phase_{len(detected_phases)+1}',
                        'shift_magnitude': embedding_shifts[i]['shift']
                    })
            
            # Add labels based on conversation position
            for i, phase in enumerate(detected_phases):
                if i == 0:
                    phase['phase'] = 'exploration'
                elif i == len(detected_phases) - 1:
                    phase['phase'] = 'conclusion'
                else:
                    phase['phase'] = f'development_{i}'
            
            return detected_phases
        
        return []

    def find_ensemble_invariants(self, conversation):
        """Find invariant patterns across multiple embedding models"""
        if not self.ensemble_mode or 'ensemble_embeddings' not in conversation:
            return None
        
        embeddings_dict = conversation['ensemble_embeddings']
        model_names = list(embeddings_dict.keys())
        n_models = len(model_names)
        n_messages = len(conversation['messages'])
        
        invariants = {
            'distance_correlations': {},
            'velocity_correlations': {},
            'phase_consensus': {},
            'topology_preservation': {},
            'curvature_agreement': {},
            'convergence_patterns': {}
        }
        
        # 1. Distance Matrix Correlations
        distance_matrices = {}
        for model_name, embeddings in embeddings_dict.items():
            dist_matrix = np.zeros((n_messages, n_messages))
            for i in range(n_messages):
                for j in range(n_messages):
                    dist_matrix[i, j] = self._safe_euclidean(embeddings[i], embeddings[j])
            distance_matrices[model_name] = dist_matrix
        
        # Compare all pairs with baseline random correlations
        from scipy.stats import spearmanr, permutation_test
        
        # Generate random baseline by shuffling embeddings
        def generate_random_baseline(embeddings, n_permutations=100):
            """Generate stronger baseline that preserves some temporal structure"""
            n = len(embeddings)
            random_correlations = []
            
            for _ in range(n_permutations):
                # Method 1: Block shuffle (preserves local structure)
                block_size = 10
                n_blocks = n // block_size
                blocks = [embeddings[i*block_size:(i+1)*block_size] for i in range(n_blocks)]
                if n % block_size > 0:
                    blocks.append(embeddings[n_blocks*block_size:])
                
                np.random.shuffle(blocks)
                block_shuffled = np.vstack(blocks)
                
                # Method 2: AR(1) surrogate (preserves autocorrelation)
                # Generate surrogate that matches mean, std, and autocorrelation
                mean = np.mean(embeddings, axis=0)
                std = np.std(embeddings, axis=0)
                
                # Estimate AR(1) coefficient
                autocorr = np.mean([np.corrcoef(embeddings[:-1, i], embeddings[1:, i])[0, 1] 
                                   for i in range(embeddings.shape[1]) 
                                   if np.std(embeddings[:, i]) > 0])
                
                # Generate AR(1) process
                ar1_surrogate = np.zeros_like(embeddings)
                ar1_surrogate[0] = np.random.randn(embeddings.shape[1]) * std + mean
                
                for t in range(1, n):
                    innovation = np.random.randn(embeddings.shape[1]) * std * np.sqrt(max(0, 1 - autocorr**2))
                    ar1_surrogate[t] = autocorr * (ar1_surrogate[t-1] - mean) + mean + innovation
                
                # Use both baselines
                for surrogate in [block_shuffled, ar1_surrogate]:
                    dist_surrogate = np.zeros((n, n))
                    for i in range(n):
                        for j in range(n):
                            dist_surrogate[i, j] = self._safe_euclidean(surrogate[i], surrogate[j])
                    
                    triu_indices = np.triu_indices(n, k=1)
                    original_dist = distance_matrices[model_names[0]][triu_indices]
                    surrogate_dist = dist_surrogate[triu_indices]
                    corr, _ = spearmanr(original_dist, surrogate_dist)
                    random_correlations.append(corr)
            
            return np.array(random_correlations)
        
        # Get baseline correlations
        baseline_embeddings = list(embeddings_dict.values())[0]
        baseline_correlations = generate_random_baseline(baseline_embeddings, n_permutations=100)
        baseline_mean = np.mean(baseline_correlations)
        baseline_std = np.std(baseline_correlations)
        
        for i in range(n_models):
            for j in range(i+1, n_models):
                model1, model2 = model_names[i], model_names[j]
                triu_indices = np.triu_indices(n_messages, k=1)
                dist1 = distance_matrices[model1][triu_indices]
                dist2 = distance_matrices[model2][triu_indices]
                corr, p_value = spearmanr(dist1, dist2)
                
                # Normalize by baseline
                normalized_corr = (corr - baseline_mean) / baseline_std
                
                # Test if correlation is significantly different from random baseline
                z_score = (corr - baseline_mean) / baseline_std
                from scipy.stats import norm
                significance = 2 * (1 - norm.cdf(abs(z_score)))
                
                invariants['distance_correlations'][f'{model1}-{model2}'] = {
                    'correlation': corr,
                    'normalized_correlation': normalized_corr,
                    'p_value': p_value,
                    'baseline_mean': baseline_mean,
                    'baseline_std': baseline_std,
                    'z_score': z_score,
                    'significant_vs_baseline': significance < 0.05
                }
        
        # 2. Velocity Profile Correlations
        velocity_profiles = {}
        for model_name, embeddings in embeddings_dict.items():
            velocities = []
            for i in range(1, n_messages):
                v = np.linalg.norm(embeddings[i] - embeddings[i-1])
                velocities.append(v)
            velocity_profiles[model_name] = velocities
        
        for i in range(n_models):
            for j in range(i+1, n_models):
                model1, model2 = model_names[i], model_names[j]
                corr, p_value = pearsonr(velocity_profiles[model1], velocity_profiles[model2])
                invariants['velocity_correlations'][f'{model1}-{model2}'] = {
                    'correlation': corr,
                    'p_value': p_value
                }
        
        # 3. Phase Detection Consensus
        phase_boundaries_by_model = {}
        for model_name, embeddings in embeddings_dict.items():
            # Detect phases using velocity peaks
            velocities = velocity_profiles[model_name]
            mean_v = np.mean(velocities)
            std_v = np.std(velocities)
            boundaries = [i+1 for i, v in enumerate(velocities) if v > mean_v + 1.5 * std_v]
            phase_boundaries_by_model[model_name] = boundaries
        
        # Find consensus boundaries
        all_boundaries = []
        for boundaries in phase_boundaries_by_model.values():
            all_boundaries.extend(boundaries)
        
        from collections import Counter
        boundary_counts = Counter(all_boundaries)
        consensus_threshold = n_models * 0.6
        consensus_boundaries = [b for b, count in boundary_counts.items() if count >= consensus_threshold]
        
        invariants['phase_consensus'] = {
            'model_boundaries': phase_boundaries_by_model,
            'consensus_boundaries': sorted(consensus_boundaries),
            'consensus_rate': len(consensus_boundaries) / max(1, len(set(all_boundaries)))
        }
        
        # 4. Topology Preservation
        k = min(10, n_messages // 5)
        topology_scores = {}
        
        for i in range(n_models):
            for j in range(i+1, n_models):
                model1, model2 = model_names[i], model_names[j]
                
                # k-NN for each point
                from sklearn.neighbors import NearestNeighbors
                nbrs1 = NearestNeighbors(n_neighbors=k).fit(embeddings_dict[model1])
                nbrs2 = NearestNeighbors(n_neighbors=k).fit(embeddings_dict[model2])
                
                _, indices1 = nbrs1.kneighbors(embeddings_dict[model1])
                _, indices2 = nbrs2.kneighbors(embeddings_dict[model2])
                
                # Calculate preservation
                preservation_scores = []
                for idx in range(n_messages):
                    neighbors1 = set(indices1[idx])
                    neighbors2 = set(indices2[idx])
                    preservation = len(neighbors1.intersection(neighbors2)) / k
                    preservation_scores.append(preservation)
                
                topology_scores[f'{model1}-{model2}'] = {
                    'mean_preservation': np.mean(preservation_scores),
                    'std_preservation': np.std(preservation_scores)
                }
        
        invariants['topology_preservation'] = topology_scores
        
        # 5. Curvature Agreement
        curvature_profiles = {}
        for model_name, embeddings in embeddings_dict.items():
            curvatures = []
            for i in range(2, n_messages):
                v1 = embeddings[i-1] - embeddings[i-2]
                v2 = embeddings[i] - embeddings[i-1]
                
                v1_norm = np.linalg.norm(v1)
                v2_norm = np.linalg.norm(v2)
                
                if v1_norm > 0 and v2_norm > 0:
                    cos_angle = np.dot(v1, v2) / (v1_norm * v2_norm)
                    angle = np.arccos(np.clip(cos_angle, -1, 1))
                    curvatures.append(angle)
                else:
                    curvatures.append(0)
            curvature_profiles[model_name] = curvatures
        
        # Find high curvature consensus
        high_curvature_turns = []
        for turn in range(len(curvatures)):
            high_curv_count = 0
            for model_name in model_names:
                if turn < len(curvature_profiles[model_name]):
                    model_curvatures = curvature_profiles[model_name]
                    threshold = np.mean(model_curvatures) + np.std(model_curvatures)
                    if model_curvatures[turn] > threshold:
                        high_curv_count += 1
            
            if high_curv_count >= n_models * 0.6:
                high_curvature_turns.append(turn + 2)
        
        invariants['curvature_agreement'] = {
            'high_curvature_consensus': high_curvature_turns,
            'agreement_rate': len(high_curvature_turns) / max(1, n_messages - 2)
        }
        
        # 6. Speaker Convergence Patterns
        convergence_by_model = {}
        
        for model_name, embeddings in embeddings_dict.items():
            # Group by speaker
            speaker_trajectories = {}
            for i, msg in enumerate(conversation['messages']):
                speaker = msg['speaker']
                if speaker not in speaker_trajectories:
                    speaker_trajectories[speaker] = []
                speaker_trajectories[speaker].append(embeddings[i])
            
            # Calculate convergence for each speaker pair
            speakers = list(speaker_trajectories.keys())
            model_convergence = {}
            
            for i in range(len(speakers)):
                for j in range(i+1, len(speakers)):
                    s1, s2 = speakers[i], speakers[j]
                    traj1 = speaker_trajectories[s1]
                    traj2 = speaker_trajectories[s2]
                    
                    if len(traj1) >= 3 and len(traj2) >= 3:
                        # Early vs late similarity
                        early1 = np.mean(traj1[:len(traj1)//3], axis=0)
                        early2 = np.mean(traj2[:len(traj2)//3], axis=0)
                        late1 = np.mean(traj1[-len(traj1)//3:], axis=0)
                        late2 = np.mean(traj2[-len(traj2)//3:], axis=0)
                        
                        early_sim = 1 - self._safe_cosine(early1, early2)
                        late_sim = 1 - self._safe_cosine(late1, late2)
                        convergence = late_sim - early_sim
                        
                        model_convergence[f'{s1}-{s2}'] = convergence
            
            convergence_by_model[model_name] = model_convergence
        
        # Find consensus convergence patterns
        speaker_pairs = list(convergence_by_model[model_names[0]].keys())
        consensus_convergence = {}
        
        for pair in speaker_pairs:
            convergences = [convergence_by_model[m].get(pair, 0) for m in model_names]
            if all(c > 0 for c in convergences) or all(c < 0 for c in convergences):
                consensus_convergence[pair] = {
                    'mean': np.mean(convergences),
                    'std': np.std(convergences),
                    'direction': 'convergent' if np.mean(convergences) > 0 else 'divergent'
                }
        
        invariants['convergence_patterns'] = {
            'by_model': convergence_by_model,
            'consensus': consensus_convergence
        }
        
        # Calculate summary statistics
        invariants['summary'] = {
            'mean_distance_correlation': np.mean([v['correlation'] for v in invariants['distance_correlations'].values()]),
            'mean_velocity_correlation': np.mean([v['correlation'] for v in invariants['velocity_correlations'].values()]),
            'mean_topology_preservation': np.mean([v['mean_preservation'] for v in invariants['topology_preservation'].values()]),
            'phase_consensus_rate': invariants['phase_consensus']['consensus_rate'],
            'curvature_agreement_rate': invariants['curvature_agreement']['agreement_rate'],
            'n_consensus_convergences': len(invariants['convergence_patterns']['consensus'])
        }
        
        return invariants

    def visualize_ensemble_analysis(self, conversation):
        """Create visualizations for ensemble analysis"""
        if not self.ensemble_mode or 'invariant_patterns' not in conversation:
            return
        
        invariants = conversation['invariant_patterns']
        model_names = list(conversation['ensemble_embeddings'].keys())
        
        # Create correlation heatmap
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. Distance correlations
        ax = axes[0, 0]
        corr_matrix = np.eye(len(model_names))
        for i, m1 in enumerate(model_names):
            for j, m2 in enumerate(model_names):
                if i < j:
                    key = f'{m1}-{m2}' if f'{m1}-{m2}' in invariants['distance_correlations'] else f'{m2}-{m1}'
                    if key in invariants['distance_correlations']:
                        corr_matrix[i, j] = corr_matrix[j, i] = invariants['distance_correlations'][key]['correlation']
        
        sns.heatmap(corr_matrix, annot=True, fmt='.3f', xticklabels=model_names,
                   yticklabels=model_names, cmap='RdBu_r', center=0.5, ax=ax, vmin=0, vmax=1)
        ax.set_title('Distance Matrix Correlations')
        
        # 2. Velocity correlations
        ax = axes[0, 1]
        corr_matrix = np.eye(len(model_names))
        for i, m1 in enumerate(model_names):
            for j, m2 in enumerate(model_names):
                if i < j:
                    key = f'{m1}-{m2}' if f'{m1}-{m2}' in invariants['velocity_correlations'] else f'{m2}-{m1}'
                    if key in invariants['velocity_correlations']:
                        corr_matrix[i, j] = corr_matrix[j, i] = invariants['velocity_correlations'][key]['correlation']
        
        sns.heatmap(corr_matrix, annot=True, fmt='.3f', xticklabels=model_names,
                   yticklabels=model_names, cmap='RdBu_r', center=0.5, ax=ax, vmin=0, vmax=1)
        ax.set_title('Velocity Profile Correlations')
        
        # 3. Topology preservation
        ax = axes[0, 2]
        topo_matrix = np.eye(len(model_names))
        for i, m1 in enumerate(model_names):
            for j, m2 in enumerate(model_names):
                if i < j:
                    key = f'{m1}-{m2}' if f'{m1}-{m2}' in invariants['topology_preservation'] else f'{m2}-{m1}'
                    if key in invariants['topology_preservation']:
                        topo_matrix[i, j] = topo_matrix[j, i] = invariants['topology_preservation'][key]['mean_preservation']
        
        sns.heatmap(topo_matrix, annot=True, fmt='.3f', xticklabels=model_names,
                   yticklabels=model_names, cmap='Greens', vmin=0, vmax=1, ax=ax)
        ax.set_title('Topology Preservation')
        
        # 4. Phase boundaries comparison
        ax = axes[1, 0]
        phase_data = invariants['phase_consensus']['model_boundaries']
        consensus = invariants['phase_consensus']['consensus_boundaries']
        
        y_pos = 0
        for model_name, boundaries in phase_data.items():
            ax.scatter(boundaries, [y_pos] * len(boundaries), label=model_name, s=50)
            y_pos += 1
        
        # Mark consensus boundaries
        for cb in consensus:
            ax.axvline(x=cb, color='red', linestyle='--', alpha=0.5)
        
        ax.set_yticks(range(len(model_names)))
        ax.set_yticklabels(model_names)
        ax.set_xlabel('Turn Number')
        ax.set_title('Phase Boundary Detection')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # 5. Summary statistics
        ax = axes[1, 1]
        ax.axis('off')
        summary = invariants['summary']
        summary_text = "Invariant Pattern Summary\n" + "="*30 + "\n\n"
        summary_text += f"Distance Correlation: {summary['mean_distance_correlation']:.3f}\n"
        summary_text += f"Velocity Correlation: {summary['mean_velocity_correlation']:.3f}\n"
        summary_text += f"Topology Preservation: {summary['mean_topology_preservation']:.3f}\n"
        summary_text += f"Phase Consensus Rate: {summary['phase_consensus_rate']:.1%}\n"
        summary_text += f"Curvature Agreement: {summary['curvature_agreement_rate']:.1%}\n"
        summary_text += f"Consensus Convergences: {summary['n_consensus_convergences']}\n"
        
        ax.text(0.1, 0.9, summary_text, transform=ax.transAxes,
                fontsize=12, verticalalignment='top', fontfamily='monospace')
        
        # 6. Convergence patterns
        ax = axes[1, 2]
        convergence_consensus = invariants['convergence_patterns']['consensus']
        if convergence_consensus:
            pairs = list(convergence_consensus.keys())
            means = [v['mean'] for v in convergence_consensus.values()]
            stds = [v['std'] for v in convergence_consensus.values()]
            colors = ['green' if v['direction'] == 'convergent' else 'red' for v in convergence_consensus.values()]
            
            y_pos = np.arange(len(pairs))
            ax.barh(y_pos, means, xerr=stds, color=colors, alpha=0.7)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(pairs)
            ax.set_xlabel('Convergence Score')
            ax.set_title('Speaker Convergence Consensus')
            ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
        
        plt.tight_layout()
        output_path = self.output_dir / 'ensemble_analysis.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"\nSaved ensemble analysis visualization to {output_path}")

    def generate_report(self):
        """Generate a comprehensive analysis report"""
        report_path = self.output_dir / 'embedding_trajectory_report.txt'
        
        with open(report_path, 'w') as f:
            f.write("EMBEDDING TRAJECTORY ANALYSIS REPORT\n")
            f.write("="*50 + "\n\n")
            
            f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total Conversations: {len(self.conversations)}\n")
            f.write(f"Total Messages: {len(self.all_embeddings)}\n")
            f.write(f"Primary Embedding Model: {self.model_name}\n")
            f.write(f"Embedding Dimensions: {self.embedding_dim}\n")
            
            if self.ensemble_mode:
                f.write(f"Ensemble Models: {', '.join(self.ensemble_models.keys())}\n")
            
            f.write("\n")
            
            # Conversation details
            f.write("Conversations Analyzed:\n")
            for conv in self.conversations:
                f.write(f"  - Session {conv['metadata']['session_id'][:30]}...\n")
                f.write(f"    Messages: {conv['metadata']['message_count']}\n")
                if 'trajectory_stats' in conv:
                    stats = conv['trajectory_stats']
                    f.write(f"    Mean embedding distance: {stats['mean_distance']:.3f}\n")
                    f.write(f"    Total trajectory length: {stats['total_distance']:.3f}\n")
                if 'curvature_metrics' in conv:
                    curv = conv['curvature_metrics']
                    f.write(f"    Mean curvature: {curv['mean_curvature']:.3f}\n")
                if 'information_flow' in conv:
                    flow = conv['information_flow']
                    f.write(f"    Mean entropy: {flow['mean_entropy']:.3f}\n")
            
            f.write("\n" + "="*50 + "\n\n")
            
            # Trajectory statistics
            f.write("TRAJECTORY STATISTICS\n")
            f.write("-"*30 + "\n")
            
            all_stats = [c['trajectory_stats'] for c in self.conversations if 'trajectory_stats' in c]
            if all_stats:
                for metric in ['mean_distance', 'std_distance', 'total_distance']:
                    values = [s[metric] for s in all_stats]
                    f.write(f"\n{metric}:\n")
                    f.write(f"  Mean: {np.mean(values):.3f}\n")
                    f.write(f"  Std: {np.std(values):.3f}\n")
                    f.write(f"  Range: [{np.min(values):.3f}, {np.max(values):.3f}]\n")
            
            # New metrics summary
            f.write("\n" + "="*50 + "\n")
            f.write("ADVANCED METRICS SUMMARY\n")
            f.write("-"*30 + "\n")
            
            # Phase transitions
            phase_conversations = [c for c in self.conversations if 'phase_metrics' in c]
            if phase_conversations:
                f.write(f"\nPhase Analysis:\n")
                f.write(f"  Conversations with phases: {len(phase_conversations)}\n")
                all_transitions = []
                for c in phase_conversations:
                    all_transitions.extend([t['distance'] for t in c['phase_metrics']['transitions']])
                if all_transitions:
                    f.write(f"  Mean phase transition distance: {np.mean(all_transitions):.3f}\n")
            
            # Information flow
            entropy_conversations = [c for c in self.conversations if 'information_flow' in c]
            if entropy_conversations:
                f.write(f"\nInformation Flow:\n")
                entropies = [c['information_flow']['mean_entropy'] for c in entropy_conversations]
                trends = [c['information_flow']['entropy_trend'] for c in entropy_conversations]
                f.write(f"  Mean entropy across conversations: {np.mean(entropies):.3f}\n")
                f.write(f"  Mean entropy trend: {np.mean(trends):.3f}\n")
            
            # Ensemble analysis summary
            if self.ensemble_mode:
                f.write("\n" + "="*50 + "\n")
                f.write("ENSEMBLE INVARIANT ANALYSIS\n")
                f.write("-"*30 + "\n")
                
                ensemble_conversations = [c for c in self.conversations if 'invariant_patterns' in c]
                if ensemble_conversations:
                    # Aggregate invariant metrics
                    all_dist_corrs = []
                    all_vel_corrs = []
                    all_topo_pres = []
                    all_phase_consensus = []
                    all_curv_agree = []
                    
                    for conv in ensemble_conversations:
                        summary = conv['invariant_patterns']['summary']
                        all_dist_corrs.append(summary['mean_distance_correlation'])
                        all_vel_corrs.append(summary['mean_velocity_correlation'])
                        all_topo_pres.append(summary['mean_topology_preservation'])
                        all_phase_consensus.append(summary['phase_consensus_rate'])
                        all_curv_agree.append(summary['curvature_agreement_rate'])
                    
                    f.write(f"\nCross-Model Agreement Metrics:\n")
                    f.write(f"  Distance Correlation: {np.mean(all_dist_corrs):.3f} (±{np.std(all_dist_corrs):.3f})\n")
                    f.write(f"  Velocity Correlation: {np.mean(all_vel_corrs):.3f} (±{np.std(all_vel_corrs):.3f})\n")
                    f.write(f"  Topology Preservation: {np.mean(all_topo_pres):.3f} (±{np.std(all_topo_pres):.3f})\n")
                    f.write(f"  Phase Consensus Rate: {np.mean(all_phase_consensus):.1%}\n")
                    f.write(f"  Curvature Agreement: {np.mean(all_curv_agree):.1%}\n")
                    
                    # Interpretation
                    f.write("\nInterpretation:\n")
                    avg_corr = np.mean(all_dist_corrs)
                    if avg_corr > 0.8:
                        f.write("  • HIGH INVARIANCE: Strong consensus across models (>0.8)\n")
                        f.write("    → Detected patterns likely represent true embedding structure\n")
                    elif avg_corr > 0.6:
                        f.write("  • MODERATE INVARIANCE: Fair consensus across models (0.6-0.8)\n")
                        f.write("    → Core embedding patterns preserved, some model-specific artifacts\n")
                    else:
                        f.write("  • LOW INVARIANCE: Weak consensus across models (<0.6)\n")
                        f.write("    → High model dependence, interpret with caution\n")
            
            f.write("\n" + "="*50 + "\n")
            f.write("\nVisualization files generated:\n")
            f.write(f"  - trajectories_3d_pca.html\n")
            f.write(f"  - trajectories_3d_tsne.html\n")
            f.write(f"  - embedding_convergence.csv\n")
            f.write(f"  - trajectory_comparison.png\n")
            f.write(f"  - dimensional_analysis_summary.png\n")
            f.write(f"  - projection_comparison_all.png\n")
            f.write(f"  - distance_matrices/ (folder with individual conversation matrices)\n")
            f.write(f"  - distance_matrices_tier_comparison.png\n")
            f.write(f"  - distance_statistics_summary.png\n")
            if self.ensemble_mode:
                f.write(f"  - ensemble_comparison_summary.png\n")
            
            # Statistical Analysis Results
            if hasattr(self, 'statistical_results') and self.statistical_results:
                f.write("\n" + "="*50 + "\n")
                f.write("STATISTICAL SIGNIFICANCE TESTS\n")
                f.write("-"*30 + "\n")
                
                significant_metrics = []
                for metric_key, results in self.statistical_results.items():
                    if results['significant']:
                        significant_metrics.append(metric_key)
                
                f.write(f"Significant differences found in {len(significant_metrics)} out of {len(self.statistical_results)} metrics\n")
                f.write("\nSignificant metrics (p < 0.05):\n")
                for metric_key in significant_metrics:
                    results = self.statistical_results[metric_key]
                    f.write(f"\n{metric_key.replace('_', ' ').title()}:\n")
                    f.write(f"  Test: {results['test_name']}\n")
                    f.write(f"  p-value: {results['p_value']:.6f}\n")
                    f.write(f"  Tier means: ")
                    for tier, mean in results['tier_means'].items():
                        f.write(f"{tier}={mean:.3f} ")
                    f.write("\n")
                    if 'effect_size' in results:
                        f.write(f"  Effect size (Cohen's d): {results['effect_size']:.3f}\n")
            
            # Null Model Comparison Results
            if hasattr(self, 'null_model_results') and self.null_model_results:
                f.write("\n" + "="*50 + "\n")
                f.write("NULL MODEL COMPARISON\n")
                f.write("-"*30 + "\n")
                
                f.write("Metrics significantly different from random null models:\n\n")
                for metric, results in self.null_model_results.items():
                    if results['significantly_different']:
                        f.write(f"{metric}:\n")
                        f.write(f"  Real conversations: {results['mean_real']:.3f} ± {results['std_real']:.3f}\n")
                        f.write(f"  Null models: {results['mean_null']:.3f} ± {results['std_null']:.3f}\n")
                        f.write(f"  Effect size (Cohen's d): {results['cohens_d']:.3f}\n")
                        f.write(f"  p-value: {results['p_value_mw']:.6f}\n\n")
            
            # Topic Analysis Results
            if hasattr(self, 'topic_analysis') and self.topic_analysis:
                f.write("\n" + "="*50 + "\n")
                f.write("TOPIC-ATTRACTOR ANALYSIS\n")
                f.write("-"*30 + "\n")
                
                topics = self.topic_analysis['topics']
                f.write(f"Identified {len(topics)} conversation topics\n\n")
                for i, topic_words in enumerate(topics[:5]):  # Show top 5 topics
                    f.write(f"Topic {i}: {', '.join(topic_words[:5])}\n")
                
                if 'topic_attractor_distances' in self.topic_analysis:
                    distances = self.topic_analysis['topic_attractor_distances']
                    if distances:
                        f.write(f"\nAverage topic distance to attractors: {np.mean(distances):.3f}\n")
                        f.write("Topics appear to cluster around embedding attractors\n")
            
            # Predictive Model Results
            if hasattr(self, 'predictive_results') and self.predictive_results:
                f.write("\n" + "="*50 + "\n")
                f.write("PREDICTIVE VALIDATION\n")
                f.write("-"*30 + "\n")
                
                pr = self.predictive_results
                f.write(f"Dataset: {pr['n_samples']} conversations\n")
                f.write(f"Breakdown rate: {pr['breakdown_rate']:.2%}\n")
                f.write(f"Cross-validation ROC-AUC: {np.mean(pr['cv_scores']):.3f} (±{np.std(pr['cv_scores']):.3f})\n")
                
                f.write("\nTop 5 predictive features:\n")
                for idx, row in pr['feature_importance'].head(5).iterrows():
                    f.write(f"  {row['feature']}: coefficient={row['coefficient']:.3f}\n")
                
                f.write("\nStatistically significant features:\n")
                sig_features = pr['significant_features'][pr['significant_features']['significant']]
                for idx, row in sig_features.head(5).iterrows():
                    f.write(f"  {row['feature']}: p={row['p_value']:.6f}\n")
            
            # Feature extraction summary
            if self.conversations:
                f.write("\n" + "="*50 + "\n")
                f.write("FEATURE EXTRACTION SUMMARY\n")
                f.write("-"*30 + "\n")
                features = self.extract_feature_vector(self.conversations[0])
                f.write(f"Features extracted per conversation: {len(features)}\n")
                f.write("Feature categories:\n")
                f.write("  - Basic metadata (2 features)\n")
                f.write("  - Trajectory dynamics (5 features)\n")
                f.write("  - Phase transitions (2 features)\n")
                f.write("  - Curvature metrics (5 features)\n")
                f.write("  - Coherence windows (3 features)\n")
                f.write("  - Speaker dynamics (4 features)\n")
                f.write("  - Information flow (3 features)\n")
                f.write("  - Full dimensional analysis (13 features)\n")
                if self.ensemble_mode:
                    f.write("  - Ensemble invariants (6 features)\n")
            
            f.write("\n" + "="*50 + "\n")
            f.write("\nReport saved to: " + str(report_path))
        
        print(f"\nGenerated report: {report_path}")
        
        # Generate ensemble-specific report if enabled
        if self.ensemble_mode:
            self.generate_ensemble_report()
    
    def generate_ensemble_report(self):
        """Generate detailed report on ensemble invariant patterns"""
        report_path = self.output_dir / 'ensemble_invariant_report.txt'
        
        with open(report_path, 'w') as f:
            f.write("ENSEMBLE INVARIANT PATTERN ANALYSIS\n")
            f.write("="*50 + "\n\n")
            
            f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Ensemble Models: {', '.join(self.ensemble_models.keys())}\n")
            f.write(f"Total Conversations: {len(self.conversations)}\n\n")
            
            for conv in self.conversations:
                if 'invariant_patterns' not in conv:
                    continue
                
                f.write(f"\nSession: {conv['metadata']['session_id'][:30]}...\n")
                f.write("-"*50 + "\n")
                
                invariants = conv['invariant_patterns']
                
                # Distance correlations
                f.write("\n1. Distance Matrix Correlations:\n")
                for pair, data in invariants['distance_correlations'].items():
                    f.write(f"   {pair}: r={data['correlation']:.3f} (p={data['p_value']:.4f})\n")
                
                # Velocity correlations
                f.write("\n2. Velocity Profile Correlations:\n")
                for pair, data in invariants['velocity_correlations'].items():
                    f.write(f"   {pair}: r={data['correlation']:.3f} (p={data['p_value']:.4f})\n")
                
                # Phase consensus
                f.write("\n3. Phase Detection Consensus:\n")
                f.write(f"   Consensus boundaries: {invariants['phase_consensus']['consensus_boundaries']}\n")
                f.write(f"   Consensus rate: {invariants['phase_consensus']['consensus_rate']:.1%}\n")
                
                # Topology preservation
                f.write("\n4. Topology Preservation:\n")
                for pair, data in invariants['topology_preservation'].items():
                    f.write(f"   {pair}: {data['mean_preservation']:.3f} (±{data['std_preservation']:.3f})\n")
                
                # Curvature agreement
                f.write("\n5. Curvature Agreement:\n")
                f.write(f"   High curvature turns: {invariants['curvature_agreement']['high_curvature_consensus']}\n")
                f.write(f"   Agreement rate: {invariants['curvature_agreement']['agreement_rate']:.1%}\n")
                
                # Convergence patterns
                f.write("\n6. Speaker Convergence Consensus:\n")
                consensus = invariants['convergence_patterns']['consensus']
                for pair, data in consensus.items():
                    f.write(f"   {pair}: {data['mean']:.3f} ({data['direction']})\n")
            
            f.write("\n" + "="*50 + "\n")
            f.write("IMPLICATIONS FOR EMBEDDING VS FEATURE SPACE\n")
            f.write("-"*30 + "\n")
            f.write("\nHigh correlations (>0.8) across models suggest we are capturing\n")
            f.write("genuine embedding structure rather than model-specific artifacts.\n")
            f.write("\nPatterns that show consensus across diverse embedding models\n")
            f.write("(trained on different objectives) are more likely to represent\n")
            f.write("true conversational dynamics in embedding space.\n")
        
        print(f"Generated ensemble report: {report_path}")
    
    def create_ensemble_summary_visualization(self):
        """Create a comprehensive visualization summarizing ensemble analysis across all conversations"""
        if not self.ensemble_mode:
            return
        
        # Collect all conversations with ensemble analysis
        ensemble_conversations = [c for c in self.conversations if 'invariant_patterns' in c]
        if not ensemble_conversations:
            print("No conversations with ensemble analysis found.")
            return
        
        print(f"\nCreating ensemble summary visualization for {len(ensemble_conversations)} conversations...")
        
        # Aggregate metrics across all conversations
        all_dist_corrs = []
        all_vel_corrs = []
        all_topo_pres = []
        all_phase_consensus = []
        all_curv_agree = []
        all_convergences = []
        
        # Collect per-conversation metrics
        conv_metrics = []
        
        for conv in ensemble_conversations:
            summary = conv['invariant_patterns']['summary']
            all_dist_corrs.append(summary['mean_distance_correlation'])
            all_vel_corrs.append(summary['mean_velocity_correlation'])
            all_topo_pres.append(summary['mean_topology_preservation'])
            all_phase_consensus.append(summary['phase_consensus_rate'])
            all_curv_agree.append(summary['curvature_agreement_rate'])
            all_convergences.append(summary['n_consensus_convergences'])
            
            # Store per-conversation metrics
            conv_metrics.append({
                'session_id': conv['metadata']['session_id'][:12],
                'tier': conv.get('tier', 'unknown'),
                'distance_corr': summary['mean_distance_correlation'],
                'velocity_corr': summary['mean_velocity_correlation'],
                'topology_pres': summary['mean_topology_preservation'],
                'phase_consensus': summary['phase_consensus_rate'],
                'curvature_agree': summary['curvature_agreement_rate']
            })
        
        # Create comprehensive figure
        fig = plt.figure(figsize=(20, 16))
        
        # 1. Distribution of correlations across conversations
        ax1 = plt.subplot(3, 3, 1)
        data = [all_dist_corrs, all_vel_corrs, all_topo_pres]
        labels = ['Distance\nCorrelation', 'Velocity\nCorrelation', 'Topology\nPreservation']
        bp = ax1.boxplot(data, labels=labels, patch_artist=True)
        colors = ['lightblue', 'lightgreen', 'lightcoral']
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
        ax1.set_ylabel('Correlation')
        ax1.set_title('Cross-Model Agreement Distributions')
        ax1.axhline(y=0.8, color='red', linestyle='--', alpha=0.5, label='High agreement threshold')
        ax1.legend()
        
        # 2. Consensus rates
        ax2 = plt.subplot(3, 3, 2)
        consensus_data = [all_phase_consensus, all_curv_agree]
        consensus_labels = ['Phase\nBoundaries', 'High\nCurvature']
        bp2 = ax2.boxplot(consensus_data, labels=consensus_labels, patch_artist=True)
        for patch in bp2['boxes']:
            patch.set_facecolor('lightyellow')
        ax2.set_ylabel('Consensus Rate')
        ax2.set_title('Multi-Model Consensus Rates')
        ax2.set_ylim(0, 1)
        
        # 3. Correlation by tier (if tier information available)
        ax3 = plt.subplot(3, 3, 3)
        tiers = set(m['tier'] for m in conv_metrics if m['tier'] != 'unknown')
        if tiers:
            tier_data = {tier: [] for tier in tiers}
            for m in conv_metrics:
                if m['tier'] != 'unknown':
                    tier_data[m['tier']].append(m['distance_corr'])
            
            positions = range(1, len(tiers) + 1)
            tier_names = list(tier_data.keys())
            tier_values = [tier_data[t] for t in tier_names]
            
            bp3 = ax3.boxplot(tier_values, positions=positions, labels=tier_names, patch_artist=True)
            tier_colors = {'full_reasoning': 'darkblue', 'light_reasoning': 'blue', 'non_reasoning': 'lightblue'}
            for patch, tier in zip(bp3['boxes'], tier_names):
                patch.set_facecolor(tier_colors.get(tier, 'gray'))
            
            ax3.set_ylabel('Distance Correlation')
            ax3.set_title('Distance Correlations by Model Tier')
        else:
            ax3.text(0.5, 0.5, 'No tier information available', 
                    transform=ax3.transAxes, ha='center', va='center')
            ax3.set_xlim(0, 1)
            ax3.set_ylim(0, 1)
        
        # 4. Scatter plot: Distance vs Velocity correlations
        ax4 = plt.subplot(3, 3, 4)
        scatter = ax4.scatter(all_dist_corrs, all_vel_corrs, 
                             c=[m['topology_pres'] for m in conv_metrics],
                             cmap='viridis', s=100, alpha=0.6, edgecolors='black')
        ax4.set_xlabel('Distance Correlation')
        ax4.set_ylabel('Velocity Correlation')
        ax4.set_title('Distance vs Velocity Correlations')
        ax4.axhline(y=0.8, color='red', linestyle='--', alpha=0.3)
        ax4.axvline(x=0.8, color='red', linestyle='--', alpha=0.3)
        cbar = plt.colorbar(scatter, ax=ax4)
        cbar.set_label('Topology Preservation')
        
        # 5. Histogram of all correlation values
        ax5 = plt.subplot(3, 3, 5)
        all_corrs = all_dist_corrs + all_vel_corrs + all_topo_pres
        ax5.hist(all_corrs, bins=20, alpha=0.7, color='purple', edgecolor='black')
        ax5.axvline(x=np.mean(all_corrs), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(all_corrs):.3f}')
        ax5.axvline(x=0.8, color='green', linestyle='--', alpha=0.5, label='High agreement')
        ax5.set_xlabel('Correlation Value')
        ax5.set_ylabel('Frequency')
        ax5.set_title('Distribution of All Correlation Values')
        ax5.legend()
        
        # 6. Phase consensus vs curvature agreement
        ax6 = plt.subplot(3, 3, 6)
        ax6.scatter(all_phase_consensus, all_curv_agree, s=100, alpha=0.6, edgecolors='black')
        ax6.set_xlabel('Phase Consensus Rate')
        ax6.set_ylabel('Curvature Agreement Rate')
        ax6.set_title('Phase vs Curvature Consensus')
        ax6.set_xlim(0, 1)
        ax6.set_ylim(0, 1)
        # Add diagonal line
        ax6.plot([0, 1], [0, 1], 'k--', alpha=0.3)
        
        # 7. Summary statistics table
        ax7 = plt.subplot(3, 3, 7)
        ax7.axis('off')
        summary_text = "ENSEMBLE INVARIANT SUMMARY\n" + "="*35 + "\n\n"
        summary_text += f"Total Conversations: {len(ensemble_conversations)}\n\n"
        summary_text += "Mean Correlations:\n"
        summary_text += f"  Distance: {np.mean(all_dist_corrs):.3f} (±{np.std(all_dist_corrs):.3f})\n"
        summary_text += f"  Velocity: {np.mean(all_vel_corrs):.3f} (±{np.std(all_vel_corrs):.3f})\n"
        summary_text += f"  Topology: {np.mean(all_topo_pres):.3f} (±{np.std(all_topo_pres):.3f})\n\n"
        summary_text += "Consensus Rates:\n"
        summary_text += f"  Phase Boundaries: {np.mean(all_phase_consensus):.1%}\n"
        summary_text += f"  Curvature Points: {np.mean(all_curv_agree):.1%}\n\n"
        
        # Interpretation
        avg_corr = np.mean(all_dist_corrs)
        if avg_corr > 0.8:
            interpretation = "HIGH INVARIANCE (>0.8)\nStrong cross-model consensus"
        elif avg_corr > 0.6:
            interpretation = "MODERATE INVARIANCE (0.6-0.8)\nCore patterns preserved"
        else:
            interpretation = "LOW INVARIANCE (<0.6)\nModel-dependent patterns"
        summary_text += f"Overall Assessment:\n{interpretation}"
        
        ax7.text(0.1, 0.9, summary_text, transform=ax7.transAxes,
                fontsize=11, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.5))
        
        # 8. Individual conversation performance
        ax8 = plt.subplot(3, 3, 8)
        conv_ids = [m['session_id'] for m in conv_metrics[:20]]  # Show first 20
        conv_dist_corrs = [m['distance_corr'] for m in conv_metrics[:20]]
        
        y_pos = np.arange(len(conv_ids))
        bars = ax8.barh(y_pos, conv_dist_corrs)
        
        # Color bars by value
        for i, bar in enumerate(bars):
            if conv_dist_corrs[i] > 0.8:
                bar.set_color('darkgreen')
            elif conv_dist_corrs[i] > 0.6:
                bar.set_color('orange')
            else:
                bar.set_color('red')
        
        ax8.set_yticks(y_pos)
        ax8.set_yticklabels(conv_ids, fontsize=8)
        ax8.set_xlabel('Distance Correlation')
        ax8.set_title('Per-Conversation Distance Correlations')
        ax8.axvline(x=0.8, color='green', linestyle='--', alpha=0.5)
        ax8.axvline(x=0.6, color='orange', linestyle='--', alpha=0.5)
        ax8.set_xlim(0, 1)
        
        # 9. Model comparison matrix (if we have detailed model data)
        ax9 = plt.subplot(3, 3, 9)
        if ensemble_conversations and 'invariant_patterns' in ensemble_conversations[0]:
            # Get model names from first conversation
            first_conv = ensemble_conversations[0]
            model_names = list(first_conv['ensemble_embeddings'].keys())
            
            # Aggregate distance correlations across all conversations
            aggregated_correlations = {}
            for i, m1 in enumerate(model_names):
                for j, m2 in enumerate(model_names):
                    if i < j:
                        key = f'{m1}-{m2}'
                        corrs = []
                        for conv in ensemble_conversations:
                            inv = conv['invariant_patterns']
                            if key in inv['distance_correlations']:
                                corrs.append(inv['distance_correlations'][key]['correlation'])
                            elif f'{m2}-{m1}' in inv['distance_correlations']:
                                corrs.append(inv['distance_correlations'][f'{m2}-{m1}']['correlation'])
                        if corrs:
                            aggregated_correlations[key] = np.mean(corrs)
            
            # Create correlation matrix
            n_models = len(model_names)
            corr_matrix = np.eye(n_models)
            for i, m1 in enumerate(model_names):
                for j, m2 in enumerate(model_names):
                    if i < j:
                        key = f'{m1}-{m2}' if f'{m1}-{m2}' in aggregated_correlations else f'{m2}-{m1}'
                        if key in aggregated_correlations:
                            corr_matrix[i, j] = corr_matrix[j, i] = aggregated_correlations[key]
            
            sns.heatmap(corr_matrix, annot=True, fmt='.3f', xticklabels=model_names,
                       yticklabels=model_names, cmap='RdBu_r', center=0.5, ax=ax9, vmin=0, vmax=1)
            ax9.set_title('Average Model-to-Model Correlations')
        else:
            ax9.text(0.5, 0.5, 'No detailed model data available', 
                    transform=ax9.transAxes, ha='center', va='center')
        
        plt.suptitle('Ensemble Analysis Summary - Cross-Model Invariant Patterns', fontsize=16)
        plt.tight_layout()
        
        output_path = self.output_dir / 'ensemble_summary_visualization.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved ensemble summary visualization to {output_path}")
        
        # Also save detailed metrics to CSV
        import pandas as pd
        metrics_df = pd.DataFrame(conv_metrics)
        csv_path = self.output_dir / 'ensemble_metrics_summary.csv'
        metrics_df.to_csv(csv_path, index=False)
        print(f"Saved detailed metrics to {csv_path}")
    
    def analyze_tier_differences(self):
        """Analyze differences in trajectory patterns across model tiers"""
        if not self.tier_results:
            return
        
        print("\nAnalyzing tier-specific patterns...")
        
        tier_metrics = {}
        
        for tier_name, tier_data in self.tier_results.items():
            conversations = tier_data['conversations']
            
            # Aggregate metrics for this tier
            metrics = {
                'n_conversations': len(conversations),
                'trajectory_lengths': [],
                'mean_distances': [],
                'curvatures': [],
                'intrinsic_dims': [],
                'participation_ratios': [],
                'entropy_means': [],
                'convergence_rates': [],
                'phase_counts': [],
                'embedding_spreads': []
            }
            
            for conv in conversations:
                if 'trajectory_stats' in conv:
                    metrics['trajectory_lengths'].append(conv['trajectory_stats']['total_distance'])
                    metrics['mean_distances'].append(conv['trajectory_stats']['mean_distance'])
                
                if 'curvature_metrics' in conv:
                    metrics['curvatures'].append(conv['curvature_metrics']['mean_curvature'])
                
                if 'full_dimensional_analysis' in conv:
                    full_dim = conv['full_dimensional_analysis']
                    if 'intrinsic_dimensionality' in full_dim and full_dim['intrinsic_dimensionality']['mle_dimension']:
                        metrics['intrinsic_dims'].append(full_dim['intrinsic_dimensionality']['mle_dimension'])
                    if 'dimensional_utilization' in full_dim:
                        metrics['participation_ratios'].append(full_dim['dimensional_utilization']['participation_ratio'])
                
                if 'information_flow' in conv:
                    metrics['entropy_means'].append(conv['information_flow']['mean_entropy'])
                
                if 'phase_metrics' in conv:
                    metrics['phase_counts'].append(len(conv['phase_metrics']['phase_embeddings']))
                
                if 'speaker_dynamics' in conv:
                    spreads = [s['embedding_spread'] for s in conv['speaker_dynamics']['speaker_stats'].values()]
                    metrics['embedding_spreads'].append(np.mean(spreads))
            
            # Calculate tier statistics with bootstrap confidence intervals
            tier_data = {
                'n_conversations': metrics['n_conversations'],
                'mean_trajectory_length': np.mean(metrics['trajectory_lengths']) if metrics['trajectory_lengths'] else 0,
                'std_trajectory_length': np.std(metrics['trajectory_lengths']) if metrics['trajectory_lengths'] else 0,
                'mean_distance': np.mean(metrics['mean_distances']) if metrics['mean_distances'] else 0,
                'mean_curvature': np.mean(metrics['curvatures']) if metrics['curvatures'] else 0,
                'mean_intrinsic_dim': np.mean(metrics['intrinsic_dims']) if metrics['intrinsic_dims'] else 0,
                'mean_participation_ratio': np.mean(metrics['participation_ratios']) if metrics['participation_ratios'] else 0,
                'mean_entropy': np.mean(metrics['entropy_means']) if metrics['entropy_means'] else 0,
                'mean_phase_count': np.mean(metrics['phase_counts']) if metrics['phase_counts'] else 0,
                'mean_embedding_spread': np.mean(metrics['embedding_spreads']) if metrics['embedding_spreads'] else 0
            }
            
            # Add bootstrap confidence intervals for speaker dynamics
            if metrics['embedding_spreads']:
                n_bootstrap = 1000
                bootstrap_spreads = []
                data = metrics['embedding_spreads']
                
                for _ in range(n_bootstrap):
                    bootstrap_sample = np.random.choice(data, len(data), replace=True)
                    bootstrap_spreads.append(np.mean(bootstrap_sample))
                
                tier_data['embedding_spread_ci_lower'] = np.percentile(bootstrap_spreads, 2.5)
                tier_data['embedding_spread_ci_upper'] = np.percentile(bootstrap_spreads, 97.5)
                tier_data['embedding_spread_std'] = np.std(bootstrap_spreads)
            
            tier_metrics[tier_name] = tier_data
        
        # Store for visualization
        self.tier_metrics = tier_metrics
        
        # Print comparison
        print("\nTIER COMPARISON:")
        print("-" * 80)
        
        # Create comparison table
        metric_names = ['n_conversations', 'mean_trajectory_length', 'mean_distance', 
                       'mean_curvature', 'mean_intrinsic_dim', 'mean_participation_ratio',
                       'mean_entropy', 'mean_phase_count', 'mean_embedding_spread']
        
        for metric in metric_names:
            print(f"\n{metric}:")
            for tier_name, metrics in tier_metrics.items():
                value = metrics.get(metric, 0)
                # Special handling for embedding spread to show CI
                if metric == 'mean_embedding_spread' and 'embedding_spread_ci_lower' in metrics:
                    ci_lower = metrics['embedding_spread_ci_lower']
                    ci_upper = metrics['embedding_spread_ci_upper']
                    print(f"  {tier_name}: {value:.3f} (95% CI: [{ci_lower:.3f}, {ci_upper:.3f}])")
                else:
                    print(f"  {tier_name}: {value:.3f}" if isinstance(value, float) else f"  {tier_name}: {value}")
    
    def visualize_tier_trajectories(self):
        """Create visualizations comparing trajectories across tiers"""
        if not self.tier_results:
            return
        
        # Create tier comparison visualization
        fig, axes = plt.subplots(3, 3, figsize=(18, 15))
        fig.suptitle('Trajectory Characteristics Across Model Tiers', fontsize=16)
        
        tier_names = list(self.tier_metrics.keys())
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Blue, Orange, Green
        
        # 1. Trajectory Length Distribution
        ax = axes[0, 0]
        for i, (tier_name, tier_data) in enumerate(self.tier_results.items()):
            lengths = []
            for conv in tier_data['conversations']:
                if 'trajectory_stats' in conv:
                    lengths.append(conv['trajectory_stats']['total_distance'])
            if lengths:
                ax.hist(lengths, bins=20, alpha=0.5, label=tier_name, color=colors[i % len(colors)])
        ax.set_xlabel('Total Trajectory Length')
        ax.set_ylabel('Count')
        ax.set_title('Trajectory Length Distribution')
        ax.legend()
        
        # 2. Mean Distance by Tier
        ax = axes[0, 1]
        means = [self.tier_metrics[t]['mean_distance'] for t in tier_names]
        x_pos = np.arange(len(tier_names))
        ax.bar(x_pos, means, color=colors[:len(tier_names)])
        ax.set_xticks(x_pos)
        ax.set_xticklabels(tier_names)
        ax.set_ylabel('Mean Distance Between Turns')
        ax.set_title('Average Embedding Step Size')
        
        # 3. Intrinsic Dimensionality
        ax = axes[0, 2]
        dims = [self.tier_metrics[t]['mean_intrinsic_dim'] for t in tier_names]
        ax.bar(x_pos, dims, color=colors[:len(tier_names)])
        ax.set_xticks(x_pos)
        ax.set_xticklabels(tier_names)
        ax.set_ylabel('Intrinsic Dimensionality')
        ax.set_title('Embedding Space Complexity')
        
        # 4. Curvature Distribution
        ax = axes[1, 0]
        for i, (tier_name, tier_data) in enumerate(self.tier_results.items()):
            curvatures = []
            for conv in tier_data['conversations']:
                if 'curvature_metrics' in conv:
                    curvatures.append(conv['curvature_metrics']['mean_curvature'])
            if curvatures:
                ax.hist(curvatures, bins=20, alpha=0.5, label=tier_name, color=colors[i % len(colors)])
        ax.set_xlabel('Mean Curvature')
        ax.set_ylabel('Count')
        ax.set_title('Trajectory Curvature Distribution')
        ax.legend()
        
        # 5. Participation Ratio
        ax = axes[1, 1]
        ratios = [self.tier_metrics[t]['mean_participation_ratio'] for t in tier_names]
        ax.bar(x_pos, ratios, color=colors[:len(tier_names)])
        ax.set_xticks(x_pos)
        ax.set_xticklabels(tier_names)
        ax.set_ylabel('Participation Ratio')
        ax.set_title('Dimensional Utilization')
        
        # 6. Information Entropy
        ax = axes[1, 2]
        entropies = [self.tier_metrics[t]['mean_entropy'] for t in tier_names]
        ax.bar(x_pos, entropies, color=colors[:len(tier_names)])
        ax.set_xticks(x_pos)
        ax.set_xticklabels(tier_names)
        ax.set_ylabel('Mean Entropy')
        ax.set_title('Information Flow Complexity')
        
        # 7. Phase Count
        ax = axes[2, 0]
        phases = [self.tier_metrics[t]['mean_phase_count'] for t in tier_names]
        ax.bar(x_pos, phases, color=colors[:len(tier_names)])
        ax.set_xticks(x_pos)
        ax.set_xticklabels(tier_names)
        ax.set_ylabel('Average Phase Count')
        ax.set_title('Conversation Structure Complexity')
        
        # 8. Embedding Spread
        ax = axes[2, 1]
        spreads = [self.tier_metrics[t]['mean_embedding_spread'] for t in tier_names]
        ax.bar(x_pos, spreads, color=colors[:len(tier_names)])
        ax.set_xticks(x_pos)
        ax.set_xticklabels(tier_names)
        ax.set_ylabel('Embedding Spread')
        ax.set_title('Speaker Territory Size')
        
        # 9. Summary Statistics
        ax = axes[2, 2]
        ax.axis('off')
        summary_text = "Key Findings:\n" + "="*30 + "\n\n"
        
        # Find tier with highest/lowest values
        for metric in ['mean_intrinsic_dim', 'mean_trajectory_length', 'mean_curvature']:
            values = {t: self.tier_metrics[t][metric] for t in tier_names}
            max_tier = max(values, key=values.get)
            min_tier = min(values, key=values.get)
            
            metric_name = metric.replace('mean_', '').replace('_', ' ').title()
            summary_text += f"{metric_name}:\n"
            summary_text += f"  Highest: {max_tier} ({values[max_tier]:.3f})\n"
            summary_text += f"  Lowest: {min_tier} ({values[min_tier]:.3f})\n\n"
        
        ax.text(0.1, 0.9, summary_text, transform=ax.transAxes,
                fontsize=11, verticalalignment='top', fontfamily='monospace')
        
        plt.tight_layout()
        output_path = self.output_dir / 'tier_comparison_analysis.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"\nSaved tier comparison visualization to {output_path}")
    
    def create_conversation_trajectory_space(self):
        """Create a unified visualization of all conversations in trajectory space"""
        if not self.tier_results:
            return
        
        print("\nCreating conversation trajectory space visualization...")
        
        # Extract summary features for each conversation
        conversation_summaries = []
        
        for conv in self.conversations:
            summary = {
                'session_id': conv['metadata']['session_id'][:20],
                'tier': conv.get('tier', 'unknown'),
                'message_count': conv['metadata']['message_count']
            }
            
            # Trajectory features
            if 'trajectory_stats' in conv:
                summary['total_distance'] = conv['trajectory_stats']['total_distance']
                summary['mean_distance'] = conv['trajectory_stats']['mean_distance']
                summary['max_distance'] = conv['trajectory_stats']['max_distance']
            
            # Complexity features
            if 'full_dimensional_analysis' in conv:
                full_dim = conv['full_dimensional_analysis']
                if 'intrinsic_dimensionality' in full_dim and full_dim['intrinsic_dimensionality']['mle_dimension']:
                    summary['intrinsic_dim'] = full_dim['intrinsic_dimensionality']['mle_dimension']
                if 'dimensional_utilization' in full_dim:
                    summary['participation_ratio'] = full_dim['dimensional_utilization']['participation_ratio']
            
            # Dynamic features
            if 'curvature_metrics' in conv:
                summary['mean_curvature'] = conv['curvature_metrics']['mean_curvature']
                summary['max_curvature'] = conv['curvature_metrics']['max_curvature']
            
            if 'information_flow' in conv:
                summary['entropy_mean'] = conv['information_flow']['mean_entropy']
                summary['entropy_trend'] = conv['information_flow']['entropy_trend']
            
            conversation_summaries.append(summary)
        
        # Convert to DataFrame
        summary_df = pd.DataFrame(conversation_summaries)
        
        # Select features for visualization
        feature_cols = ['total_distance', 'mean_distance', 'intrinsic_dim', 
                       'mean_curvature', 'entropy_mean', 'participation_ratio']
        
        # Filter to available features
        available_features = [col for col in feature_cols if col in summary_df.columns]
        
        if len(available_features) < 3:
            print("Not enough features available for trajectory space visualization")
            return
        
        # Standardize features
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        feature_matrix = summary_df[available_features].fillna(0).values
        feature_matrix_scaled = scaler.fit_transform(feature_matrix)
        
        # Reduce to 3D for visualization
        from sklearn.decomposition import PCA
        pca = PCA(n_components=3)
        trajectory_space_3d = pca.fit_transform(feature_matrix_scaled)
        
        # Create interactive 3D visualization
        fig = go.Figure()
        
        # Plot each tier with different color
        tier_colors = {
            'full_reasoning': 'blue',
            'light_reasoning': 'orange',
            'non_reasoning': 'green'
        }
        
        for tier_name in self.tier_results.keys():
            tier_mask = summary_df['tier'] == tier_name
            tier_points = trajectory_space_3d[tier_mask]
            
            if len(tier_points) > 0:
                fig.add_trace(go.Scatter3d(
                    x=tier_points[:, 0],
                    y=tier_points[:, 1],
                    z=tier_points[:, 2],
                    mode='markers',
                    name=tier_name,
                    marker=dict(
                        size=8,
                        color=tier_colors.get(tier_name, 'gray'),
                        opacity=0.8,
                        line=dict(width=1, color='DarkSlateGrey')
                    ),
                    text=[f"Session: {sid}<br>Messages: {mc}" 
                         for sid, mc in zip(summary_df[tier_mask]['session_id'], 
                                          summary_df[tier_mask]['message_count'])],
                    hoverinfo='text+name'
                ))
        
        # Add centroids for each tier
        for tier_name in self.tier_results.keys():
            tier_mask = summary_df['tier'] == tier_name
            tier_points = trajectory_space_3d[tier_mask]
            
            if len(tier_points) > 0:
                centroid = np.mean(tier_points, axis=0)
                fig.add_trace(go.Scatter3d(
                    x=[centroid[0]],
                    y=[centroid[1]],
                    z=[centroid[2]],
                    mode='markers+text',
                    name=f'{tier_name} centroid',
                    marker=dict(
                        size=15,
                        color=tier_colors.get(tier_name, 'gray'),
                        symbol='diamond',
                        line=dict(width=2, color='black')
                    ),
                    text=[f'{tier_name}<br>centroid'],
                    textposition='top center',
                    showlegend=False
                ))
        
        fig.update_layout(
            title='Conversation Trajectory Space<br>(Each point is a complete conversation)',
            scene=dict(
                xaxis_title=f'PC1 ({pca.explained_variance_ratio_[0]:.1%} var)',
                yaxis_title=f'PC2 ({pca.explained_variance_ratio_[1]:.1%} var)',
                zaxis_title=f'PC3 ({pca.explained_variance_ratio_[2]:.1%} var)',
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.5)
                )
            ),
            showlegend=True,
            height=800
        )
        
        # Save
        output_path = self.output_dir / 'conversation_trajectory_space.html'
        fig.write_html(str(output_path))
        print(f"Saved conversation trajectory space to {output_path}")
        
        # Also create a 2D projection comparison
        self.create_tier_trajectory_comparison()
    
    def create_tier_trajectory_comparison(self):
        """Create a comparison of actual conversation trajectories by tier"""
        if not self.tier_results:
            return
        
        print("\nCreating tier trajectory comparison...")
        
        # Sample conversations from each tier
        fig = make_subplots(
            rows=1, cols=len(self.tier_results),
            subplot_titles=list(self.tier_results.keys()),
            specs=[[{'type': 'scatter3d'}] * len(self.tier_results)]
        )
        
        for idx, (tier_name, tier_data) in enumerate(self.tier_results.items()):
            # Sample up to 3 conversations from this tier
            sample_size = min(3, len(tier_data['conversations']))
            sample_convs = np.random.choice(tier_data['conversations'], sample_size, replace=False)
            
            colors = ['red', 'blue', 'green']
            
            for conv_idx, conv in enumerate(sample_convs):
                if 'embedded_messages' not in conv:
                    continue
                
                # Get embeddings
                embeddings = np.array([m['embedding'] for m in conv['embedded_messages']])
                
                # Reduce to 3D
                if len(embeddings) > 3:
                    pca = PCA(n_components=3)
                    trajectory_3d = pca.fit_transform(embeddings)
                    
                    # Plot trajectory
                    fig.add_trace(
                        go.Scatter3d(
                            x=trajectory_3d[:, 0],
                            y=trajectory_3d[:, 1],
                            z=trajectory_3d[:, 2],
                            mode='lines+markers',
                            line=dict(color=colors[conv_idx % len(colors)], width=3),
                            marker=dict(size=4, color=colors[conv_idx % len(colors)]),
                            name=f'Conv {conv_idx+1}',
                            showlegend=(idx == 0)  # Only show legend for first subplot
                        ),
                        row=1, col=idx+1
                    )
                    
                    # Add start and end markers
                    fig.add_trace(
                        go.Scatter3d(
                            x=[trajectory_3d[0, 0]],
                            y=[trajectory_3d[0, 1]],
                            z=[trajectory_3d[0, 2]],
                            mode='markers',
                            marker=dict(size=10, color='green', symbol='diamond'),
                            showlegend=False
                        ),
                        row=1, col=idx+1
                    )
                    
                    fig.add_trace(
                        go.Scatter3d(
                            x=[trajectory_3d[-1, 0]],
                            y=[trajectory_3d[-1, 1]],
                            z=[trajectory_3d[-1, 2]],
                            mode='markers',
                            marker=dict(size=10, color='red', symbol='square'),
                            showlegend=False
                        ),
                        row=1, col=idx+1
                    )
        
        fig.update_layout(
            title='Sample Conversation Trajectories by Model Tier',
            height=600,
            showlegend=True
        )
        
        # Save
        output_path = self.output_dir / 'tier_trajectory_comparison.html'
        fig.write_html(str(output_path))
        print(f"Saved tier trajectory comparison to {output_path}")
    
    def perform_statistical_tests(self):
        """Perform statistical significance tests on tier comparisons"""
        if not self.tier_results or len(self.tier_results) < 2:
            print("Need at least 2 tiers for statistical comparison")
            return None
        
        print("\n" + "="*70)
        print("STATISTICAL SIGNIFICANCE TESTS")
        print("="*70)
        
        # Prepare data for tests
        tier_names = list(self.tier_results.keys())
        test_results = {}
        
        # Key metrics to test
        metrics_to_test = [
            ('intrinsic_dim', 'Intrinsic Dimensionality'),
            ('mean_distance', 'Mean Step Distance'),
            ('total_distance', 'Total Trajectory Distance'),
            ('mean_curvature', 'Mean Curvature'),
            ('phase_count', 'Number of Phases'),
            ('entropy_mean', 'Mean Entropy'),
            ('participation_ratio', 'Participation Ratio'),
            ('embedding_loops', 'Embedding Loop Count')
        ]
        
        # Store all p-values for multiple comparison correction
        all_p_values = []
        all_test_info = []
        
        for metric_key, metric_name in metrics_to_test:
            print(f"\n{metric_name}:")
            print("-" * 50)
            
            # Extract data for each tier
            tier_data = {}
            for tier_name in tier_names:
                values = []
                for conv in self.tier_results[tier_name]['conversations']:
                    # Extract metric value based on where it's stored
                    value = None
                    
                    if metric_key in ['mean_distance', 'total_distance']:
                        if 'trajectory_stats' in conv:
                            value = conv['trajectory_stats'].get(metric_key)
                    elif metric_key == 'intrinsic_dim':
                        if 'full_dimensional_analysis' in conv:
                            fa = conv['full_dimensional_analysis']
                            if 'intrinsic_dimensionality' in fa:
                                value = fa['intrinsic_dimensionality'].get('mle_dimension')
                    elif metric_key == 'mean_curvature':
                        if 'curvature_metrics' in conv:
                            value = conv['curvature_metrics'].get('mean_curvature')
                    elif metric_key == 'phase_count':
                        if 'phase_metrics' in conv:
                            value = len(conv['phase_metrics'].get('phase_embeddings', []))
                    elif metric_key == 'entropy_mean':
                        if 'information_flow' in conv:
                            value = conv['information_flow'].get('mean_entropy')
                    elif metric_key == 'participation_ratio':
                        if 'full_dimensional_analysis' in conv:
                            fa = conv['full_dimensional_analysis']
                            if 'dimensional_utilization' in fa:
                                value = fa['dimensional_utilization'].get('participation_ratio')
                    elif metric_key == 'embedding_loops':
                        if 'distance_analysis' in conv:
                            value = conv['distance_analysis'].get('loop_count')
                    
                    if value is not None and not np.isnan(value):
                        values.append(value)
                
                if values:
                    tier_data[tier_name] = np.array(values)
                    print(f"  {tier_name}: n={len(values)}, mean={np.mean(values):.3f}, std={np.std(values):.3f}")
            
            if len(tier_data) >= 2:
                # Prepare data for ANOVA/Kruskal-Wallis
                all_groups = []
                group_labels = []
                for tier_name, values in tier_data.items():
                    all_groups.extend(values)
                    group_labels.extend([tier_name] * len(values))
                
                # Check normality with Shapiro-Wilk test
                normality_passed = True
                from scipy.stats import shapiro
                for tier_name, values in tier_data.items():
                    if len(values) >= 3:
                        from scipy.stats import shapiro
                        _, p_norm = shapiro(values)
                        if p_norm < 0.05:
                            normality_passed = False
                
                # Perform appropriate test
                if normality_passed and len(tier_data) > 2:
                    # Use ANOVA for normally distributed data with >2 groups
                    f_stat, p_value = f_oneway(*tier_data.values())
                    test_name = "One-way ANOVA"
                    print(f"\n  {test_name}: F={f_stat:.3f}, p={p_value:.6f}")
                    
                    # If significant, perform post-hoc Tukey HSD
                    if p_value < 0.05:
                        print("\n  Post-hoc Tukey HSD:")
                        # Create DataFrame for statsmodels
                        import pandas as pd
                        df = pd.DataFrame({
                            'value': all_groups,
                            'tier': group_labels
                        })
                        tukey = pairwise_tukeyhsd(df['value'], df['tier'], alpha=0.05)
                        print(tukey)
                else:
                    # Use Kruskal-Wallis for non-normal data or as default
                    h_stat, p_value = kruskal(*tier_data.values())
                    test_name = "Kruskal-Wallis H-test"
                    print(f"\n  {test_name}: H={h_stat:.3f}, p={p_value:.6f}")
                    
                    # If significant and only 2 groups, perform Mann-Whitney U
                    if p_value < 0.05 and len(tier_data) == 2:
                        tier_list = list(tier_data.keys())
                        u_stat, p_mw = mannwhitneyu(tier_data[tier_list[0]], 
                                                   tier_data[tier_list[1]], 
                                                   alternative='two-sided')
                        print(f"  Mann-Whitney U: U={u_stat:.3f}, p={p_mw:.6f}")
                
                # Store results
                test_results[metric_key] = {
                    'test_name': test_name,
                    'statistic': f_stat if normality_passed else h_stat,
                    'p_value': p_value,
                    'significant': p_value < 0.05,  # Will be updated after correction
                    'tier_means': {t: np.mean(v) for t, v in tier_data.items()},
                    'tier_stds': {t: np.std(v) for t, v in tier_data.items()},
                    'tier_ns': {t: len(v) for t, v in tier_data.items()}
                }
                
                # Store for multiple comparison correction
                all_p_values.append(p_value)
                all_test_info.append((metric_key, metric_name))
                
                # Effect size calculation
                if len(tier_data) == 2:
                    # Cohen's d for two groups
                    tier_list = list(tier_data.keys())
                    mean1 = np.mean(tier_data[tier_list[0]])
                    mean2 = np.mean(tier_data[tier_list[1]])
                    std1 = np.std(tier_data[tier_list[0]])
                    std2 = np.std(tier_data[tier_list[1]])
                    n1 = len(tier_data[tier_list[0]])
                    n2 = len(tier_data[tier_list[1]])
                    
                    pooled_std = np.sqrt(((n1-1)*std1**2 + (n2-1)*std2**2) / (n1+n2-2))
                    cohens_d = (mean1 - mean2) / pooled_std
                    
                    test_results[metric_key]['effect_size'] = abs(cohens_d)
                    print(f"  Cohen's d: {abs(cohens_d):.3f} ({'small' if abs(cohens_d) < 0.5 else 'medium' if abs(cohens_d) < 0.8 else 'large'})")
        
        # Power analysis
        print("\n\nPOWER ANALYSIS")
        print("-" * 50)
        power_analyzer = FTestAnovaPower()
        
        # Calculate average effect size across metrics
        effect_sizes = []
        for metric_key, results in test_results.items():
            if 'effect_size' in results:
                effect_sizes.append(results['effect_size'])
        
        if effect_sizes:
            avg_effect_size = np.mean(effect_sizes)
            n_groups = len(tier_names)
            avg_n_per_group = np.mean([len(self.tier_results[t]['conversations']) for t in tier_names])
            
            # Calculate achieved power
            power = power_analyzer.solve_power(effect_size=avg_effect_size, 
                                             nobs=avg_n_per_group * n_groups, 
                                             alpha=0.05, 
                                             k_groups=n_groups)
            print(f"Average effect size: {avg_effect_size:.3f}")
            print(f"Achieved power: {power:.3f}")
            
            # Required sample size for 0.8 power
            required_n = power_analyzer.solve_power(effect_size=avg_effect_size, 
                                                   power=0.8, 
                                                   alpha=0.05, 
                                                   k_groups=n_groups)
            print(f"Required total n for 0.8 power: {required_n:.0f} ({required_n/n_groups:.0f} per tier)")
            
            # Power warning
            if power < 0.8:
                print(f"\n⚠️  WARNING: Statistical power ({power:.3f}) is below recommended threshold of 0.8")
                print(f"   This means there is a {(1-power)*100:.1f}% chance of missing true effects")
                print(f"   Consider collecting more data or focusing on larger effect sizes")
            
            # Store power analysis results
            self.power_analysis = {
                'achieved_power': power,
                'average_effect_size': avg_effect_size,
                'total_n': avg_n_per_group * n_groups,
                'n_per_group': avg_n_per_group,
                'required_n_for_80_power': required_n,
                'alpha': 0.05,
                'n_groups': n_groups
            }
        
        # Apply hierarchical FDR correction
        print("\n\nHIERARCHICAL MULTIPLE COMPARISON CORRECTION")
        print("-" * 50)
        
        if all_p_values:
            from statsmodels.stats.multitest import multipletests
            
            # Group p-values by metric type
            grouped_tests = {
                'trajectory': ['mean_distance', 'total_distance'],
                'complexity': ['intrinsic_dim', 'participation_ratio'],
                'dynamics': ['mean_curvature', 'entropy_mean'],
                'structure': ['phase_count', 'embedding_loops']
            }
            
            # Apply two-stage FDR
            # Stage 1: Correct within groups
            stage1_results = {}
            for group_name, metrics in grouped_tests.items():
                group_p_values = []
                group_indices = []
                for i, (metric_key, metric_name) in enumerate(all_test_info):
                    if metric_key in metrics:
                        group_p_values.append(all_p_values[i])
                        group_indices.append(i)
                
                if group_p_values:
                    rejected, corrected_p, _, _ = multipletests(group_p_values, alpha=0.05, method='fdr_bh')
                    for j, idx in enumerate(group_indices):
                        stage1_results[idx] = corrected_p[j]
            
            # Stage 2: Correct across groups
            if stage1_results:
                all_corrected_p = list(stage1_results.values())
                rejected_final, corrected_final, _, _ = multipletests(all_corrected_p, alpha=0.05, method='fdr_bh')
                
                # Update results with hierarchical correction
                idx_to_final = dict(zip(stage1_results.keys(), range(len(stage1_results))))
                
                for i, (metric_key, metric_name) in enumerate(all_test_info):
                    if metric_key in test_results and i in stage1_results:
                        final_idx = idx_to_final[i]
                        test_results[metric_key]['p_value_hierarchical'] = corrected_final[final_idx]
                        test_results[metric_key]['significant_hierarchical'] = rejected_final[final_idx]
                        test_results[metric_key]['p_value_stage1'] = stage1_results[i]
            
            # Also apply standard FDR for comparison
            rejected_standard, corrected_standard, _, _ = multipletests(
                all_p_values, alpha=0.05, method='fdr_bh'
            )
            
            # Update test results with both corrections
            print("\nHierarchical FDR Results:")
            print("Group -> Metric: Original p -> Stage 1 p -> Final p (Significant?)")
            print("-" * 70)
            
            for group_name, metrics in grouped_tests.items():
                for i, (metric_key, metric_name) in enumerate(all_test_info):
                    if metric_key in metrics and metric_key in test_results:
                        test_results[metric_key]['p_value_corrected'] = corrected_standard[i]
                        test_results[metric_key]['significant_corrected'] = rejected_standard[i]
                        
                        orig_p = all_p_values[i]
                        stage1_p = test_results[metric_key].get('p_value_stage1', orig_p)
                        hier_p = test_results[metric_key].get('p_value_hierarchical', orig_p)
                        sig = test_results[metric_key].get('significant_hierarchical', False)
                        
                        print(f"{group_name} -> {metric_name}: {orig_p:.4f} -> {stage1_p:.4f} -> {hier_p:.4f} ({'*' if sig else ' '})")
            
            # Summary
            n_sig_original = sum(1 for p in all_p_values if p < 0.05)
            n_sig_standard = sum(rejected_standard)
            n_sig_hierarchical = sum(1 for r in test_results.values() if r.get('significant_hierarchical', False))
            
            print(f"\nSummary:")
            print(f"  Significant before correction: {n_sig_original}/{len(all_p_values)}")
            print(f"  Significant after standard FDR: {n_sig_standard}/{len(all_p_values)}")
            print(f"  Significant after hierarchical FDR: {n_sig_hierarchical}/{len(all_p_values)}")
            print(f"  Bonferroni alpha would be: {0.05/len(all_p_values):.6f}")
        
        # Interpret effect sizes
        self.interpret_effect_sizes(test_results)
        
        # Create visualization
        self.visualize_statistical_results(test_results)
        
        return test_results
    
    def interpret_effect_sizes(self, test_results):
        """Add practical interpretation to effect sizes"""
        
        print("\n\nEFFECT SIZE INTERPRETATION")
        print("-" * 50)
        
        interpretations = {
            'small': (0, 0.2),
            'medium': (0.2, 0.5),
            'large': (0.5, 0.8),
            'very_large': (0.8, float('inf'))
        }
        
        # Add domain-specific interpretations
        for metric, results in test_results.items():
            if 'effect_size' in results:
                d = results['effect_size']
                
                # Find category
                for category, (low, high) in interpretations.items():
                    if low <= d < high:
                        results['effect_category'] = category
                        break
                
                # Add practical significance based on metric type
                if metric == 'intrinsic_dim':
                    # Even small differences in dimensionality matter
                    results['practical_significance'] = d > 0.3
                    interpretation = (
                        "Intrinsic dimensionality differences indicate fundamentally different "
                        "conversation structures. Even small effects (d>0.3) are practically meaningful."
                    )
                elif metric in ['mean_distance', 'total_distance']:
                    # Need medium effect for trajectory differences
                    results['practical_significance'] = d > 0.5
                    interpretation = (
                        "Trajectory distance differences require medium effects (d>0.5) to be "
                        "practically meaningful, as small variations are expected."
                    )
                elif metric == 'embedding_loops':
                    # Loop counts need larger differences
                    results['practical_significance'] = d > 0.6
                    interpretation = (
                        "Embedding loop differences need to be substantial (d>0.6) to indicate "
                        "different conversational patterns."
                    )
                elif metric == 'phase_count':
                    # Phase differences are meaningful at medium effect sizes
                    results['practical_significance'] = d > 0.4
                    interpretation = (
                        "Phase count differences at medium effect sizes (d>0.4) suggest "
                        "different conversation structures."
                    )
                else:
                    # Default threshold
                    results['practical_significance'] = d > 0.4
                    interpretation = "Standard interpretation applies."
                
                # Print interpretation
                print(f"\n{metric}:")
                print(f"  Effect size (Cohen's d): {d:.3f} ({results['effect_category']})")
                print(f"  Practically significant: {'Yes' if results['practical_significance'] else 'No'}")
                print(f"  Interpretation: {interpretation}")
                
                # Add warning if statistically significant but not practically significant
                if results.get('significant_hierarchical', False) and not results['practical_significance']:
                    print(f"  ⚠️  Warning: Statistically significant but may not be practically meaningful")
        
        # Summary of practical significance
        n_practical = sum(1 for r in test_results.values() 
                         if r.get('practical_significance', False))
        n_statistical = sum(1 for r in test_results.values() 
                           if r.get('significant_hierarchical', False))
        
        print(f"\n\nSUMMARY:")
        print(f"  Statistically significant (after correction): {n_statistical}/{len(test_results)}")
        print(f"  Practically significant: {n_practical}/{len(test_results)}")
        
        if n_statistical > n_practical:
            print("\n⚠️  Some results are statistically significant but may not be practically meaningful.")
            print("   Consider focusing on metrics with both statistical and practical significance.")
    
    def visualize_statistical_results(self, test_results):
        """Create visualization of statistical test results"""
        if not test_results:
            return
        
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # Create figure with subplots
        n_metrics = len(test_results)
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        # 1. P-value heatmap
        ax = axes[0]
        metrics = list(test_results.keys())
        p_values = [test_results[m]['p_value'] for m in metrics]
        
        # Create color map (red for significant, blue for not)
        colors = ['red' if p < 0.05 else 'blue' for p in p_values]
        bars = ax.barh(metrics, p_values, color=colors)
        ax.axvline(x=0.05, color='black', linestyle='--', label='p=0.05')
        ax.set_xlabel('p-value')
        ax.set_title('Statistical Significance by Metric')
        ax.legend()
        
        # Add p-value labels
        for i, (bar, p) in enumerate(zip(bars, p_values)):
            ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                   f'{p:.4f}', va='center')
        
        # 2. Effect sizes (Cohen's d)
        ax = axes[1]
        effect_sizes = []
        effect_metrics = []
        for metric, results in test_results.items():
            if 'effect_size' in results:
                effect_sizes.append(results['effect_size'])
                effect_metrics.append(metric)
        
        if effect_sizes:
            bars = ax.bar(range(len(effect_sizes)), effect_sizes)
            ax.set_xticks(range(len(effect_sizes)))
            ax.set_xticklabels(effect_metrics, rotation=45, ha='right')
            ax.set_ylabel("Cohen's d")
            ax.set_title('Effect Sizes')
            
            # Add reference lines
            ax.axhline(y=0.2, color='gray', linestyle='--', alpha=0.5, label='Small')
            ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Medium')
            ax.axhline(y=0.8, color='gray', linestyle='--', alpha=0.5, label='Large')
            ax.legend()
        
        # 3. Tier means comparison
        ax = axes[2]
        # Get tier names from first metric
        first_metric_results = next(iter(test_results.values()))
        tier_names = list(first_metric_results['tier_means'].keys())
        
        # Select top significant metrics
        sig_metrics = [m for m, r in test_results.items() if r['significant']][:5]
        
        if sig_metrics:
            x = np.arange(len(tier_names))
            width = 0.8 / len(sig_metrics)
            
            for i, metric in enumerate(sig_metrics):
                means = [test_results[metric]['tier_means'][tier] for tier in tier_names]
                ax.bar(x + i * width, means, width, label=metric.replace('_', ' '))
            
            ax.set_xlabel('Tier')
            ax.set_ylabel('Mean Value')
            ax.set_title('Tier Comparisons for Significant Metrics')
            ax.set_xticks(x + width * (len(sig_metrics) - 1) / 2)
            ax.set_xticklabels(tier_names)
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # 4. Summary statistics
        ax = axes[3]
        ax.axis('off')
        
        summary_text = "Statistical Summary\n" + "="*30 + "\n\n"
        summary_text += f"Total metrics tested: {len(test_results)}\n"
        summary_text += f"Significant differences: {len([r for r in test_results.values() if r['significant']])}\n"
        
        if effect_sizes:
            summary_text += f"\nAverage effect size: {np.mean(effect_sizes):.3f}\n"
            summary_text += f"Largest effect: {max(effect_sizes):.3f} ({effect_metrics[effect_sizes.index(max(effect_sizes))]})\n"
        
        # Power analysis summary if available
        if hasattr(self, 'power_analysis'):
            summary_text += f"\n\nPower Analysis:\n"
            summary_text += f"Achieved power: {self.power_analysis['achieved_power']:.3f}\n"
            if self.power_analysis['achieved_power'] < 0.8:
                summary_text += "⚠️  WARNING: Power is below 0.8 - results may miss true effects\n"
            summary_text += f"Sample size: {self.power_analysis['total_n']:.0f} total ({self.power_analysis['n_per_group']:.0f} per group)\n"
            summary_text += f"Required n for 0.8 power: {self.power_analysis['required_n_for_80_power']:.0f}\n"
        
        ax.text(0.1, 0.9, summary_text, transform=ax.transAxes,
               fontsize=12, verticalalignment='top', fontfamily='monospace')
        
        plt.tight_layout()
        output_path = self.output_dir / 'statistical_analysis.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved statistical analysis visualization to {output_path}")
    
    def create_aggregated_dimensional_visualizations(self, selected_conversations):
        """Create aggregated visualizations for selected conversations"""
        if not selected_conversations:
            return
        
        print(f"\nCreating visualizations for {len(selected_conversations)} selected conversations...")
        
        # Organize by tier if available
        tier_groups = {}
        if self.tier_results:
            for conv in selected_conversations:
                tier = conv.get('tier', 'unknown')
                if tier not in tier_groups:
                    tier_groups[tier] = []
                tier_groups[tier].append(conv)
        else:
            tier_groups['all'] = selected_conversations
        
        # Create one figure per analysis type
        self.create_trajectory_comparison_figure(tier_groups)
        self.create_dimensional_analysis_figure(tier_groups)
        self.create_projection_comparison_figure(tier_groups)
        
        if self.ensemble_mode:
            self.create_ensemble_comparison_figure(tier_groups)
        
        # Create distance matrices comparison
        self.create_distance_matrices_comparison_figure(tier_groups)
    
    def create_trajectory_comparison_figure(self, tier_groups):
        """Create figure comparing trajectories across selected conversations"""
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec
        
        n_tiers = len(tier_groups)
        max_per_tier = max(len(convs) for convs in tier_groups.values())
        
        fig = plt.figure(figsize=(20, 4 * n_tiers))
        gs = GridSpec(n_tiers, max_per_tier, figure=fig, hspace=0.3, wspace=0.3)
        
        tier_colors = {'full_reasoning': 'blue', 'light_reasoning': 'orange', 'non_reasoning': 'green'}
        
        for tier_idx, (tier_name, conversations) in enumerate(tier_groups.items()):
            for conv_idx, conv in enumerate(conversations[:8]):  # Max 8 per tier
                ax = fig.add_subplot(gs[tier_idx, conv_idx])
                
                # Plot 2D PCA projection of trajectory
                if 'embedded_messages' in conv:
                    embeddings = np.array([m['embedding'] for m in conv['embedded_messages']])
                    
                    # Quick PCA for visualization
                    from sklearn.decomposition import PCA
                    pca = PCA(n_components=2)
                    coords_2d = pca.fit_transform(embeddings)
                    
                    # Plot trajectory
                    ax.plot(coords_2d[:, 0], coords_2d[:, 1], 
                           color=tier_colors.get(tier_name, 'gray'), 
                           alpha=0.7, linewidth=1)
                    ax.scatter(coords_2d[0, 0], coords_2d[0, 1], 
                              color='green', s=50, marker='o', label='Start')
                    ax.scatter(coords_2d[-1, 0], coords_2d[-1, 1], 
                              color='red', s=50, marker='s', label='End')
                    
                    # Add title with key metrics
                    title = f"{conv['metadata']['session_id'][:8]}..."
                    if 'full_dimensional_analysis' in conv:
                        fa = conv['full_dimensional_analysis']
                        if 'intrinsic_dimensionality' in fa and fa['intrinsic_dimensionality']['mle_dimension']:
                            title += f"\nDim: {fa['intrinsic_dimensionality']['mle_dimension']:.1f}"
                    ax.set_title(title, fontsize=10)
                    ax.set_xticks([])
                    ax.set_yticks([])
                    
                    if conv_idx == 0:
                        ax.set_ylabel(tier_name.replace('_', ' ').title(), fontsize=12)
        
        plt.suptitle('Embedding Trajectory Comparison Across Tiers', fontsize=16)
        output_path = self.output_dir / 'trajectory_comparison.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved trajectory comparison to {output_path}")
    
    def create_dimensional_analysis_figure(self, tier_groups):
        """Create figure showing dimensional analysis across conversations"""
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Collect data across all conversations
        all_data = {
            'intrinsic_dims': [],
            'participation_ratios': [],
            'smoothness_indices': [],
            'tier_labels': []
        }
        
        for tier_name, conversations in tier_groups.items():
            for conv in conversations:
                if 'full_dimensional_analysis' in conv:
                    fa = conv['full_dimensional_analysis']
                    
                    if 'intrinsic_dimensionality' in fa and fa['intrinsic_dimensionality']['mle_dimension']:
                        all_data['intrinsic_dims'].append(fa['intrinsic_dimensionality']['mle_dimension'])
                        all_data['tier_labels'].append(tier_name)
                    
                    if 'dimensional_utilization' in fa:
                        all_data['participation_ratios'].append(fa['dimensional_utilization']['participation_ratio'])
                    
                    if 'trajectory_smoothness' in fa:
                        all_data['smoothness_indices'].append(fa['trajectory_smoothness']['smoothness_index'])
        
        # 1. Intrinsic dimensions by tier
        ax = axes[0, 0]
        if all_data['intrinsic_dims']:
            tier_names = sorted(set(all_data['tier_labels']))
            tier_dims = {tier: [] for tier in tier_names}
            
            for dim, tier in zip(all_data['intrinsic_dims'], all_data['tier_labels']):
                tier_dims[tier].append(dim)
            
            positions = []
            labels = []
            for i, (tier, dims) in enumerate(tier_dims.items()):
                if dims:
                    positions.append(dims)
                    labels.append(tier.replace('_', ' ').title())
            
            bp = ax.boxplot(positions, labels=labels, patch_artist=True)
            for patch, tier in zip(bp['boxes'], tier_names):
                patch.set_facecolor(plt.cm.viridis(tier_names.index(tier) / len(tier_names)))
            
            ax.set_ylabel('Intrinsic Dimensionality')
            ax.set_title('Intrinsic Dimensions by Tier')
        
        # 2. Participation ratio distribution
        ax = axes[0, 1]
        if all_data['participation_ratios']:
            ax.hist(all_data['participation_ratios'], bins=20, alpha=0.7, edgecolor='black')
            ax.axvline(np.mean(all_data['participation_ratios']), 
                      color='red', linestyle='--', 
                      label=f'Mean: {np.mean(all_data["participation_ratios"]):.1f}')
            ax.set_xlabel('Participation Ratio')
            ax.set_ylabel('Count')
            ax.set_title('Distribution of Dimensional Participation')
            ax.legend()
        
        # 3. Smoothness index scatter
        ax = axes[1, 0]
        if all_data['smoothness_indices'] and all_data['intrinsic_dims']:
            # Match lengths
            min_len = min(len(all_data['smoothness_indices']), len(all_data['intrinsic_dims']))
            ax.scatter(all_data['intrinsic_dims'][:min_len], 
                      all_data['smoothness_indices'][:min_len],
                      alpha=0.6)
            ax.set_xlabel('Intrinsic Dimensionality')
            ax.set_ylabel('Smoothness Index')
            ax.set_title('Trajectory Smoothness vs Complexity')
            
            # Add correlation
            if min_len > 2:
                from scipy.stats import pearsonr
                corr, p_val = pearsonr(all_data['intrinsic_dims'][:min_len], 
                                     all_data['smoothness_indices'][:min_len])
                ax.text(0.05, 0.95, f'r = {corr:.3f}, p = {p_val:.3f}', 
                       transform=ax.transAxes, verticalalignment='top')
        
        # 4. Summary statistics
        ax = axes[1, 1]
        ax.axis('off')
        
        summary_text = "Dimensional Analysis Summary\n" + "="*30 + "\n\n"
        
        if all_data['intrinsic_dims']:
            summary_text += f"Intrinsic Dimensions:\n"
            summary_text += f"  Mean: {np.mean(all_data['intrinsic_dims']):.2f}\n"
            summary_text += f"  Std: {np.std(all_data['intrinsic_dims']):.2f}\n"
            summary_text += f"  Range: [{min(all_data['intrinsic_dims']):.1f}, {max(all_data['intrinsic_dims']):.1f}]\n\n"
        
        if all_data['participation_ratios']:
            summary_text += f"Participation Ratios:\n"
            summary_text += f"  Mean: {np.mean(all_data['participation_ratios']):.1f}\n"
            summary_text += f"  Std: {np.std(all_data['participation_ratios']):.1f}\n\n"
        
        total_conversations = sum(len(convs) for convs in tier_groups.values())
        summary_text += f"Total conversations analyzed: {total_conversations}"
        
        ax.text(0.1, 0.9, summary_text, transform=ax.transAxes,
               fontsize=12, verticalalignment='top', fontfamily='monospace')
        
        plt.suptitle('Dimensional Analysis Summary', fontsize=16)
        plt.tight_layout()
        
        output_path = self.output_dir / 'dimensional_analysis_summary.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved dimensional analysis summary to {output_path}")
    
    def create_projection_comparison_figure(self, tier_groups):
        """Create figure comparing different projection methods"""
        import matplotlib.pyplot as plt
        
        # Select one representative conversation per tier
        representative_convs = []
        for tier_name, conversations in tier_groups.items():
            if conversations:
                # Pick the one closest to median intrinsic dimension
                dims = []
                for conv in conversations:
                    if 'full_dimensional_analysis' in conv:
                        fa = conv['full_dimensional_analysis']
                        if 'intrinsic_dimensionality' in fa and fa['intrinsic_dimensionality']['mle_dimension']:
                            dims.append((conv, fa['intrinsic_dimensionality']['mle_dimension']))
                
                if dims:
                    dims.sort(key=lambda x: x[1])
                    median_idx = len(dims) // 2
                    representative_convs.append((tier_name, dims[median_idx][0]))
        
        if not representative_convs:
            return
        
        n_methods = 4  # PCA, t-SNE, UMAP, MDS
        n_tiers = len(representative_convs)
        
        fig, axes = plt.subplots(n_tiers, n_methods, figsize=(16, 4 * n_tiers))
        if n_tiers == 1:
            axes = axes.reshape(1, -1)
        
        for tier_idx, (tier_name, conv) in enumerate(representative_convs):
            if 'embedded_messages' not in conv:
                continue
                
            embeddings = np.array([m['embedding'] for m in conv['embedded_messages']])
            
            # Standardize
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            embeddings_scaled = scaler.fit_transform(embeddings)
            
            # 1. PCA
            ax = axes[tier_idx, 0]
            from sklearn.decomposition import PCA
            pca = PCA(n_components=2)
            coords_pca = pca.fit_transform(embeddings_scaled)
            ax.scatter(coords_pca[:, 0], coords_pca[:, 1], c=range(len(coords_pca)), 
                      cmap='viridis', s=30, alpha=0.7)
            ax.set_title(f'PCA ({tier_name})')
            ax.set_xticks([])
            ax.set_yticks([])
            
            # 2. t-SNE
            ax = axes[tier_idx, 1]
            from sklearn.manifold import TSNE
            # Use smaller perplexity for faster computation
            perp = min(30, len(embeddings) - 1)
            tsne = TSNE(n_components=2, perplexity=perp, n_iter=500, random_state=42)
            coords_tsne = tsne.fit_transform(embeddings_scaled[:min(100, len(embeddings_scaled))])  # Limit points
            ax.scatter(coords_tsne[:, 0], coords_tsne[:, 1], c=range(len(coords_tsne)), 
                      cmap='viridis', s=30, alpha=0.7)
            ax.set_title(f't-SNE ({tier_name})')
            ax.set_xticks([])
            ax.set_yticks([])
            
            # 3. UMAP (if available)
            ax = axes[tier_idx, 2]
            try:
                import umap
                reducer = umap.UMAP(n_components=2, n_neighbors=min(15, len(embeddings)-1), 
                                  min_dist=0.1, random_state=42)
                coords_umap = reducer.fit_transform(embeddings_scaled[:min(100, len(embeddings_scaled))])
                ax.scatter(coords_umap[:, 0], coords_umap[:, 1], c=range(len(coords_umap)), 
                          cmap='viridis', s=30, alpha=0.7)
                ax.set_title(f'UMAP ({tier_name})')
            except:
                ax.text(0.5, 0.5, 'UMAP not available', ha='center', va='center', 
                       transform=ax.transAxes)
                ax.set_title(f'UMAP ({tier_name})')
            ax.set_xticks([])
            ax.set_yticks([])
            
            # 4. MDS
            ax = axes[tier_idx, 3]
            from sklearn.manifold import MDS
            mds = MDS(n_components=2, random_state=42, max_iter=100)
            coords_mds = mds.fit_transform(embeddings_scaled[:min(50, len(embeddings_scaled))])  # Very limited
            ax.scatter(coords_mds[:, 0], coords_mds[:, 1], c=range(len(coords_mds)), 
                      cmap='viridis', s=30, alpha=0.7)
            ax.set_title(f'MDS ({tier_name})')
            ax.set_xticks([])
            ax.set_yticks([])
        
        plt.suptitle('Projection Method Comparison Across Tiers', fontsize=16)
        plt.tight_layout()
        
        output_path = self.output_dir / 'projection_comparison_all.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved projection comparison to {output_path}")
    
    def create_ensemble_comparison_figure(self, tier_groups):
        """Create figure showing ensemble model agreement"""
        import matplotlib.pyplot as plt
        
        # Collect ensemble statistics
        ensemble_stats = {
            'distance_corrs': [],
            'velocity_corrs': [],
            'topology_pres': [],
            'tier_labels': []
        }
        
        for tier_name, conversations in tier_groups.items():
            for conv in conversations:
                if 'invariant_patterns' in conv and 'summary' in conv['invariant_patterns']:
                    summary = conv['invariant_patterns']['summary']
                    ensemble_stats['distance_corrs'].append(summary['mean_distance_correlation'])
                    ensemble_stats['velocity_corrs'].append(summary['mean_velocity_correlation'])
                    ensemble_stats['topology_pres'].append(summary['mean_topology_preservation'])
                    ensemble_stats['tier_labels'].append(tier_name)
        
        if not ensemble_stats['distance_corrs']:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 1. Distance correlation by tier
        ax = axes[0, 0]
        tier_names = sorted(set(ensemble_stats['tier_labels']))
        tier_data = {tier: [] for tier in tier_names}
        
        for corr, tier in zip(ensemble_stats['distance_corrs'], ensemble_stats['tier_labels']):
            tier_data[tier].append(corr)
        
        positions = []
        labels = []
        for tier, data in tier_data.items():
            if data:
                positions.append(data)
                labels.append(tier.replace('_', ' ').title())
        
        if positions:
            bp = ax.boxplot(positions, labels=labels, patch_artist=True)
            ax.set_ylabel('Distance Correlation')
            ax.set_title('Cross-Model Distance Agreement by Tier')
            ax.axhline(y=0.8, color='r', linestyle='--', alpha=0.5, label='High agreement')
            ax.legend()
        
        # 2. Velocity vs Topology preservation
        ax = axes[0, 1]
        ax.scatter(ensemble_stats['velocity_corrs'], ensemble_stats['topology_pres'], alpha=0.6)
        ax.set_xlabel('Velocity Correlation')
        ax.set_ylabel('Topology Preservation')
        ax.set_title('Ensemble Agreement Patterns')
        
        # 3. Distribution of all metrics
        ax = axes[1, 0]
        metrics = ['Distance', 'Velocity', 'Topology']
        values = [ensemble_stats['distance_corrs'], 
                 ensemble_stats['velocity_corrs'], 
                 ensemble_stats['topology_pres']]
        
        bp = ax.boxplot(values, labels=metrics, patch_artist=True)
        ax.set_ylabel('Cross-Model Correlation')
        ax.set_title('Distribution of Ensemble Agreement Metrics')
        ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
        
        # 4. Summary
        ax = axes[1, 1]
        ax.axis('off')
        
        summary_text = "Ensemble Analysis Summary\n" + "="*30 + "\n\n"
        summary_text += f"Number of models: {len(self.ensemble_models)}\n\n"
        
        for metric, values in zip(['Distance', 'Velocity', 'Topology'], 
                                 [ensemble_stats['distance_corrs'], 
                                  ensemble_stats['velocity_corrs'], 
                                  ensemble_stats['topology_pres']]):
            if values:
                summary_text += f"{metric} Correlation:\n"
                summary_text += f"  Mean: {np.mean(values):.3f}\n"
                summary_text += f"  Std: {np.std(values):.3f}\n\n"
        
        ax.text(0.1, 0.9, summary_text, transform=ax.transAxes,
               fontsize=12, verticalalignment='top', fontfamily='monospace')
        
        plt.suptitle('Ensemble Model Agreement Analysis', fontsize=16)
        plt.tight_layout()
        
        output_path = self.output_dir / 'ensemble_comparison_summary.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved ensemble comparison to {output_path}")
    
    def create_distance_matrices_comparison_figure(self, tier_groups):
        """Create aggregated figure showing saved distance matrices for selected conversations"""
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec
        from PIL import Image
        import numpy as np
        
        print("\nCreating distance matrices comparison figure...")
        
        # Select 5 diverse conversations per tier based on different metrics
        selected_convs = []
        
        for tier_name, conversations in tier_groups.items():
            # Filter conversations that have distance matrix images saved
            convs_with_images = [c for c in conversations 
                               if 'distance_matrix_path' in c and Path(c['distance_matrix_path']).exists()]
            
            if not convs_with_images:
                print(f"No distance matrix images found for {tier_name}")
                continue
            
            # Select 5 diverse conversations based on various metrics
            tier_selected = []
            
            # 1. Sort by intrinsic dimension to get diversity in complexity
            if len(convs_with_images) >= 5:
                # Get conversations with various characteristics
                metrics = []
                for conv in convs_with_images:
                    metric_dict = {
                        'conv': conv,
                        'message_count': conv['metadata'].get('message_count', 0),
                        'intrinsic_dim': 0,
                        'recurrence_rate': 0,
                        'mean_distance': 0,
                        'loop_count': 0
                    }
                    
                    # Extract metrics
                    if 'full_dimensional_analysis' in conv:
                        fa = conv['full_dimensional_analysis']
                        if 'intrinsic_dimensionality' in fa and fa['intrinsic_dimensionality']['mle_dimension']:
                            metric_dict['intrinsic_dim'] = fa['intrinsic_dimensionality']['mle_dimension']
                    
                    if 'distance_analysis' in conv:
                        da = conv['distance_analysis']
                        metric_dict['recurrence_rate'] = da['recurrence_stats']['recurrence_rate']
                        metric_dict['mean_distance'] = da['distance_stats']['mean_distance']
                        metric_dict['loop_count'] = da['loop_count']
                    
                    metrics.append(metric_dict)
                
                # Select diverse conversations:
                # 1. Lowest intrinsic dimension
                sorted_by_dim = sorted(metrics, key=lambda x: x['intrinsic_dim'])
                if sorted_by_dim[0]['intrinsic_dim'] > 0:
                    tier_selected.append(sorted_by_dim[0]['conv'])
                
                # 2. Highest intrinsic dimension
                if sorted_by_dim[-1]['intrinsic_dim'] > 0 and sorted_by_dim[-1]['conv'] not in tier_selected:
                    tier_selected.append(sorted_by_dim[-1]['conv'])
                
                # 3. Highest recurrence rate
                sorted_by_rr = sorted(metrics, key=lambda x: x['recurrence_rate'], reverse=True)
                if sorted_by_rr[0]['conv'] not in tier_selected:
                    tier_selected.append(sorted_by_rr[0]['conv'])
                
                # 4. Most embedding loops
                sorted_by_loops = sorted(metrics, key=lambda x: x['loop_count'], reverse=True)
                if sorted_by_loops[0]['conv'] not in tier_selected:
                    tier_selected.append(sorted_by_loops[0]['conv'])
                
                # 5. Fill remaining with random selection
                remaining = [c for c in convs_with_images if c not in tier_selected]
                while len(tier_selected) < 5 and remaining:
                    idx = np.random.randint(len(remaining))
                    tier_selected.append(remaining.pop(idx))
            else:
                # If less than 5, take all available
                tier_selected = convs_with_images[:5]
            
            selected_convs.append((tier_name, tier_selected))
        
        if not selected_convs:
            print("No suitable conversations for distance matrix visualization")
            return
        
        # Create figure showing the saved distance matrix images
        n_tiers = len(selected_convs)
        n_cols = 5  # 5 conversations per tier
        
        # Larger figure size for better visibility
        fig_width = 40  # 8 inches per column
        fig_height = 12 * n_tiers  # 12 inches per row for 3 tiers = 36 inches total
        
        fig = plt.figure(figsize=(fig_width, fig_height))
        gs = GridSpec(n_tiers, n_cols, figure=fig, hspace=0.4, wspace=0.1)
        
        tier_colors = {'full_reasoning': 'blue', 'light_reasoning': 'orange', 'non_reasoning': 'green'}
        
        # Load and display saved distance matrix images
        for tier_idx, (tier_name, tier_convs) in enumerate(selected_convs):
            # Add tier label directly on the figure
            tier_y_position = 1 - (tier_idx + 0.5) / n_tiers
            fig.text(0.02, tier_y_position, tier_name.replace('_', ' ').title(), 
                    fontsize=28, fontweight='bold',
                    verticalalignment='center',
                    horizontalalignment='left',
                    color=tier_colors.get(tier_name, 'gray'),
                    rotation=90)
            
            for conv_idx, conv in enumerate(tier_convs[:5]):  # Max 5 per tier
                ax = fig.add_subplot(gs[tier_idx, conv_idx])
                
                # Load the saved image
                img_path = conv['distance_matrix_path']
                try:
                    img = Image.open(img_path)
                    ax.imshow(img, aspect='auto')
                    ax.axis('off')
                    
                    # Add metrics as subtitle with larger font
                    metrics_text = []
                    
                    if 'full_dimensional_analysis' in conv:
                        fa = conv['full_dimensional_analysis']
                        if 'intrinsic_dimensionality' in fa and fa['intrinsic_dimensionality']['mle_dimension']:
                            metrics_text.append(f"Dim: {fa['intrinsic_dimensionality']['mle_dimension']:.1f}")
                    
                    if 'distance_matrix_stats' in conv:
                        dm = conv['distance_matrix_stats']
                        metrics_text.append(f"RR: {dm['recurrence_stats']['recurrence_rate']:.2f}")
                        metrics_text.append(f"Loops: {dm['loop_count']}")
                    
                    # Place metrics below the image
                    metrics_str = " | ".join(metrics_text)
                    ax.text(0.5, -0.05, metrics_str, 
                           transform=ax.transAxes,
                           fontsize=14, 
                           horizontalalignment='center',
                           verticalalignment='top')
                    
                    # Add colored border to indicate tier
                    color = tier_colors.get(tier_name, 'gray')
                    for spine in ax.spines.values():
                        spine.set_edgecolor(color)
                        spine.set_linewidth(4)
                    
                except Exception as e:
                    print(f"Error loading image {img_path}: {e}")
                    ax.text(0.5, 0.5, 'Image\nNot Found', ha='center', va='center',
                           transform=ax.transAxes, fontsize=12)
                    ax.axis('off')
        
        plt.suptitle('Distance Matrices Comparison: Showing Variance Within Each Tier', 
                    fontsize=28, fontweight='bold', y=0.98)
        
        # Add explanation text at bottom with larger font
        fig.text(0.5, 0.02, 
                'Selected conversations show diversity in: intrinsic dimensionality (Dim), recurrence rate (RR), and embedding loops',
                ha='center', fontsize=18, style='italic')
        
        # Add legend for the three plots in each image
        fig.text(0.5, 0.04,
                'Each image shows: Euclidean Distance (left), Self-Similarity (center), Recurrence Plot with phase annotations (right)',
                ha='center', fontsize=16)
        
        output_path = self.output_dir / 'distance_matrices_tier_comparison.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved distance matrices tier comparison to {output_path}")
        
        # Also create a summary statistics figure
        self.create_distance_statistics_summary(tier_groups)
    
    def create_distance_statistics_summary(self, tier_groups):
        """Create summary statistics for distance matrix analysis"""
        import matplotlib.pyplot as plt
        
        # Collect statistics for all conversations
        stats_by_tier = {tier: {'recurrence_rates': [], 'mean_distances': [], 
                               'loop_counts': [], 'coherence_scores': []} 
                        for tier in tier_groups.keys()}
        
        for tier_name, conversations in tier_groups.items():
            for conv in conversations:
                if 'distance_analysis' in conv:
                    dist = conv['distance_analysis']
                    stats_by_tier[tier_name]['recurrence_rates'].append(
                        dist['recurrence_stats']['recurrence_rate'])
                    stats_by_tier[tier_name]['mean_distances'].append(
                        dist['distance_stats']['mean_distance'])
                    stats_by_tier[tier_name]['loop_counts'].append(
                        dist['loop_count'])
                    
                if 'coherence_windows' in conv and conv['coherence_windows']:
                    mean_coherence = np.mean([w['mean_similarity'] 
                                            for w in conv['coherence_windows']])
                    stats_by_tier[tier_name]['coherence_scores'].append(mean_coherence)
        
        # Create summary figure
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        tier_names = list(tier_groups.keys())
        colors = ['blue', 'orange', 'green'][:len(tier_names)]
        
        # 1. Recurrence rates by tier
        ax = axes[0, 0]
        data = [stats_by_tier[tier]['recurrence_rates'] for tier in tier_names]
        bp = ax.boxplot(data, labels=[t.replace('_', ' ').title() for t in tier_names], 
                       patch_artist=True)
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        ax.set_ylabel('Recurrence Rate')
        ax.set_title('Conversation Recurrence Patterns')
        
        # 2. Mean distances
        ax = axes[0, 1]
        data = [stats_by_tier[tier]['mean_distances'] for tier in tier_names]
        bp = ax.boxplot(data, labels=[t.replace('_', ' ').title() for t in tier_names], 
                       patch_artist=True)
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        ax.set_ylabel('Mean Pairwise Distance')
        ax.set_title('Embedding Distances')
        
        # 3. Loop counts
        ax = axes[1, 0]
        for i, tier in enumerate(tier_names):
            if stats_by_tier[tier]['loop_counts']:
                ax.hist(stats_by_tier[tier]['loop_counts'], bins=10, alpha=0.6, 
                       label=tier.replace('_', ' ').title(), color=colors[i])
        ax.set_xlabel('Number of Embedding Loops')
        ax.set_ylabel('Count')
        ax.set_title('Distribution of Embedding Returns')
        ax.legend()
        
        # 4. Summary statistics
        ax = axes[1, 1]
        ax.axis('off')
        
        summary_text = "Distance Analysis Summary\n" + "="*30 + "\n\n"
        
        for tier in tier_names:
            summary_text += f"{tier.replace('_', ' ').title()}:\n"
            
            if stats_by_tier[tier]['recurrence_rates']:
                rr = stats_by_tier[tier]['recurrence_rates']
                summary_text += f"  Recurrence Rate: {np.mean(rr):.3f} (±{np.std(rr):.3f})\n"
            
            if stats_by_tier[tier]['loop_counts']:
                lc = stats_by_tier[tier]['loop_counts']
                summary_text += f"  Avg Loops: {np.mean(lc):.1f} (±{np.std(lc):.1f})\n"
            
            summary_text += "\n"
        
        ax.text(0.1, 0.9, summary_text, transform=ax.transAxes,
               fontsize=11, verticalalignment='top', fontfamily='monospace')
        
        plt.suptitle('Distance Matrix Statistics Summary', fontsize=16)
        plt.tight_layout()
        
        output_path = self.output_dir / 'distance_statistics_summary.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved distance statistics summary to {output_path}")
    
    def perform_dimensional_statistical_tests(self, dimensional_stats):
        """Perform statistical tests on dimensional analysis results"""
        print("\n" + "="*70)
        print("DIMENSIONAL ANALYSIS STATISTICAL TESTS")
        print("="*70)
        
        if not dimensional_stats['intrinsic_dims']:
            print("Insufficient data for statistical tests")
            return
        
        # Convert to DataFrame for easier analysis
        import pandas as pd
        df = pd.DataFrame({
            'intrinsic_dim': dimensional_stats['intrinsic_dims'][:len(dimensional_stats['tier_labels'])],
            'participation_ratio': dimensional_stats['participation_ratios'][:len(dimensional_stats['tier_labels'])],
            'active_dims': dimensional_stats['active_dimensions'][:len(dimensional_stats['tier_labels'])],
            'dormant_dims': dimensional_stats['dormant_dimensions'][:len(dimensional_stats['tier_labels'])],
            'tier': dimensional_stats['tier_labels']
        })
        
        # Group by tier
        tier_groups = df.groupby('tier')
        
        print("\nDimensional Statistics by Tier:")
        print("-" * 50)
        for tier, group in tier_groups:
            print(f"\n{tier}:")
            print(f"  Intrinsic dimension: {group['intrinsic_dim'].mean():.2f} (±{group['intrinsic_dim'].std():.2f})")
            print(f"  Participation ratio: {group['participation_ratio'].mean():.1f} (±{group['participation_ratio'].std():.1f})")
            print(f"  Active dimensions: {group['active_dims'].mean():.1f} (±{group['active_dims'].std():.1f})")
        
        # Statistical tests if multiple tiers
        if len(tier_groups) > 1:
            print("\nStatistical Comparisons:")
            print("-" * 50)
            
            # Test intrinsic dimensions
            tier_dims = [group['intrinsic_dim'].values for _, group in tier_groups]
            h_stat, p_value = kruskal(*tier_dims)
            print(f"\nIntrinsic Dimensions (Kruskal-Wallis): H={h_stat:.3f}, p={p_value:.6f}")
            
            # Test participation ratios
            tier_ratios = [group['participation_ratio'].values for _, group in tier_groups]
            h_stat, p_value = kruskal(*tier_ratios)
            print(f"Participation Ratios (Kruskal-Wallis): H={h_stat:.3f}, p={p_value:.6f}")
            
            # Correlation analysis
            print("\nCorrelation Analysis:")
            print("-" * 50)
            
            # Intrinsic dim vs participation ratio
            corr, p_val = pearsonr(df['intrinsic_dim'], df['participation_ratio'])
            print(f"Intrinsic Dim vs Participation Ratio: r={corr:.3f}, p={p_val:.6f}")
            
            # Intrinsic dim vs active dims
            corr, p_val = pearsonr(df['intrinsic_dim'], df['active_dims'])
            print(f"Intrinsic Dim vs Active Dimensions: r={corr:.3f}, p={p_val:.6f}")
    
    def identify_conversation_topics(self):
        """Identify conversation topics and map to attractor states"""
        print("\n" + "="*70)
        print("TOPIC IDENTIFICATION AND ATTRACTOR MAPPING")
        print("="*70)
        
        # Use LDA or simple clustering on message content
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.decomposition import LatentDirichletAllocation
        
        # Collect all messages with their embeddings
        all_messages = []
        message_embeddings = []
        conversation_ids = []
        
        for conv_idx, conv in enumerate(self.conversations):
            if 'embedded_messages' in conv:
                for msg in conv['embedded_messages']:
                    all_messages.append(msg['content'])
                    message_embeddings.append(msg['embedding'])
                    conversation_ids.append(conv_idx)
        
        if not all_messages:
            print("No messages found for topic analysis")
            return None
        
        print(f"  Analyzing {len(all_messages)} messages...")
        
        # TF-IDF vectorization
        vectorizer = TfidfVectorizer(max_features=100, stop_words='english', 
                                   max_df=0.9, min_df=5)
        try:
            tfidf_matrix = vectorizer.fit_transform(all_messages)
        except:
            print("Not enough messages for topic modeling")
            return None
        
        # LDA for topic modeling
        n_topics = min(10, len(all_messages) // 50)  # Adaptive number of topics
        lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
        topic_distributions = lda.fit_transform(tfidf_matrix)
        
        # Get topic words
        feature_names = vectorizer.get_feature_names_out()
        topics = []
        for topic_idx, topic in enumerate(lda.components_):
            top_indices = topic.argsort()[-10:][::-1]
            top_words = [feature_names[i] for i in top_indices]
            topics.append(top_words)
            print(f"\nTopic {topic_idx}: {', '.join(top_words[:5])}")
        
        # Map topics to embedding space (vectorized for performance)
        print("\n  Computing topic embeddings...")
        message_embeddings_array = np.array(message_embeddings)
        
        # Calculate topic embeddings as weighted averages
        topic_embeddings = []
        for topic_idx in range(n_topics):
            # Get topic probabilities for all messages
            topic_probs = topic_distributions[:, topic_idx]
            
            # Only consider messages with strong topic association
            strong_mask = topic_probs > 0.1
            if np.any(strong_mask):
                # Weighted average of embeddings
                weights = topic_probs[strong_mask]
                embeddings = message_embeddings_array[strong_mask]
                topic_embedding = np.average(embeddings, axis=0, weights=weights)
                topic_embeddings.append(topic_embedding)
            else:
                # Fallback to mean of all embeddings
                topic_embeddings.append(np.mean(message_embeddings_array, axis=0))
        
        # Find attractor states using convergent dynamics
        print("  Identifying attractor states through convergent dynamics...")
        
        # True attractors show convergent dynamics - trajectories that slow down and stabilize
        # We'll identify these by looking for regions where consecutive embeddings have low velocity
        
        attractors = []
        convergence_regions = []
        
        # Analyze each conversation's trajectory
        for conv_idx in conversation_ids[:100]:  # Sample for efficiency
            conv = self.conversations[conv_idx]
            if 'embedded_messages' not in conv:
                continue
                
            embeddings = np.array([m['embedding'] for m in conv['embedded_messages']])
            if len(embeddings) < 5:
                continue
            
            # Calculate velocities (changes between consecutive points)
            velocities = np.array([np.linalg.norm(embeddings[i+1] - embeddings[i]) 
                                  for i in range(len(embeddings)-1)])
            
            # Find regions of low velocity (potential convergence)
            # Use adaptive threshold based on velocity distribution
            velocity_threshold = np.percentile(velocities, 25)
            
            # Find sustained low-velocity regions (at least 3 consecutive low velocities)
            for i in range(len(velocities)-2):
                if all(velocities[i:i+3] < velocity_threshold):
                    # This is a convergence region
                    convergence_point = np.mean(embeddings[i:i+4], axis=0)
                    convergence_regions.append({
                        'point': convergence_point,
                        'velocity': np.mean(velocities[i:i+3]),
                        'duration': 3  # Could extend to find longer convergences
                    })
        
        if convergence_regions:
            # Cluster convergence points to find attractor basins
            from sklearn.cluster import DBSCAN
            convergence_points = np.array([cr['point'] for cr in convergence_regions])
            
            # Use DBSCAN to find clusters of convergence points
            clustering = DBSCAN(eps=0.5, min_samples=3)
            labels = clustering.fit_predict(convergence_points)
            
            # Extract attractor centers from clusters
            unique_labels = set(labels) - {-1}  # Exclude noise
            for label in unique_labels:
                cluster_mask = labels == label
                cluster_points = convergence_points[cluster_mask]
                attractor_center = np.mean(cluster_points, axis=0)
                attractors.append(attractor_center)
            
            attractors = np.array(attractors)
            print(f"  Found {len(attractors)} attractor basins through convergent dynamics")
        
        # Correlate topics with attractors
        print("\n\nTOPIC-ATTRACTOR CORRELATIONS")
        print("-" * 50)
        
        topic_attractor_distances = []
        for topic_idx, topic_words in enumerate(topics):
            # Find messages strongly associated with this topic (vectorized)
            topic_probs = topic_distributions[:, topic_idx]
            strong_mask = topic_probs > 0.3
            
            if np.any(strong_mask) and len(attractors) > 0:
                # Calculate topic center from strongly associated messages
                topic_messages = message_embeddings_array[strong_mask]
                topic_center = np.mean(topic_messages, axis=0)
                
                # Calculate distances to all attractors vectorized
                distances = np.array([np.linalg.norm(topic_center - att) for att in attractors])
                min_distance = np.min(distances)
                topic_attractor_distances.append(min_distance)
                
                print(f"Topic {topic_idx} ({', '.join(topic_words[:3])}): "
                      f"distance to nearest attractor = {min_distance:.3f}")
        
        # Statistical test: are certain topics more likely to be attractors?
        if topic_attractor_distances:
            # Compare to random baseline (vectorized)
            print("\n  Computing random baseline...")
            random_indices = np.random.choice(len(message_embeddings_array), 100, replace=True)
            random_points = message_embeddings_array[random_indices]
            
            random_distances = []
            if len(attractors) > 0:
                for random_point in random_points:
                    distances = np.array([np.linalg.norm(random_point - att) for att in attractors])
                    random_distances.append(np.min(distances))
            
            if random_distances:
                u_stat, p_value = mannwhitneyu(topic_attractor_distances, 
                                              random_distances, 
                                              alternative='less')
                print(f"\nTopics vs Random attractor proximity: U={u_stat:.3f}, p={p_value:.6f}")
                
                if p_value < 0.05:
                    print("Topics are significantly closer to attractors than random points!")
        
        return {
            'topics': topics,
            'topic_distributions': topic_distributions,
            'topic_attractor_distances': topic_attractor_distances,
            'attractors': attractors if 'attractors' in locals() else None
        }
    
    def build_predictive_model(self):
        """Demonstrate predictive modeling framework for conversation outcomes
        
        This method shows how embedding space features could be used for outcome
        prediction IF appropriate labels were available. The Academy dataset does
        not include outcome labels, as these would require:
        
        1. Human annotation of conversation quality/success
        2. Objective task completion metrics  
        3. User satisfaction ratings
        4. Validated behavioral indicators
        
        The geometric features extracted from embedding trajectories could
        potentially predict outcomes such as:
        - Conversation coherence and flow
        - Topic drift and recovery patterns
        - Collaborative vs adversarial dynamics
        - Information exchange efficiency
        - Task completion likelihood
        
        Returns:
            None if no labels found (expected for Academy data)
            dict with model results if labels are present
        """
        print("\n" + "="*70)
        print("PREDICTIVE FRAMEWORK DEMONSTRATION")
        print("="*70)
        print("\nNOTE: This is a conceptual demonstration of how geometric features")
        print("could be used for outcome prediction with properly labeled data.")
        
        # Extract features and labels
        feature_matrix = []
        labels = []
        session_ids = []
        
        # Check if outcome labels are available in the data
        has_labels = False
        label_key = None
        
        # Check various possible label locations
        for conv in self.conversations[:10]:  # Check first 10 conversations
            if 'outcome' in conv['metadata']:
                has_labels = True
                label_key = 'outcome'
                break
            elif 'breakdown' in conv['metadata']:
                has_labels = True
                label_key = 'breakdown'
                break
            elif 'label' in conv['metadata']:
                has_labels = True
                label_key = 'label'
                break
        
        if not has_labels:
            print("\nNo outcome labels found in conversation metadata.")
            print("This is expected - proper outcome labels would require:")
            print("  1. Human annotation of conversation quality/success")
            print("  2. Objective criteria (e.g., task completion, user satisfaction)")
            print("  3. Validated behavioral indicators")
            print("\nThe geometric features extracted could predict outcomes such as:")
            print("  - Conversation coherence and flow")
            print("  - Topic drift and recovery patterns")
            print("  - Collaborative vs. adversarial dynamics")
            print("  - Information exchange efficiency")
            
            # Show what features would be available
            if self.conversations:
                sample_features = self.extract_feature_vector(self.conversations[0])
                print(f"\nAvailable geometric features ({len(sample_features)} total):")
                for key in list(sample_features.keys())[:10]:
                    print(f"  - {key}")
                print("  ...")
            
            return None
        
        # Extract features and actual labels
        for conv in self.conversations:
            features = self.extract_feature_vector(conv)
            
            # Get actual label from metadata
            if label_key in conv['metadata']:
                label = conv['metadata'][label_key]
                # Convert to binary if needed
                if isinstance(label, str):
                    label = label.lower() in ['true', 'yes', 'breakdown', 'failed', '1']
                elif isinstance(label, (int, float)):
                    label = bool(label)
                
                feature_matrix.append(features)
                labels.append(label)
                session_ids.append(conv['metadata']['session_id'])
        
        if not feature_matrix:
            print("No features extracted for predictive modeling")
            return None
        
        # Convert to arrays
        X = pd.DataFrame(feature_matrix).fillna(0).values
        y = np.array(labels)
        
        print(f"Dataset: {len(X)} conversations, {X.shape[1]} features")
        print(f"Breakdown rate: {np.mean(y):.2%}")
        
        # Feature selection - keep only non-constant features
        feature_std = np.std(X, axis=0)
        valid_features = feature_std > 0
        X = X[:, valid_features]
        feature_names = [col for col, valid in zip(pd.DataFrame(feature_matrix).columns, valid_features) if valid]
        
        print(f"After filtering: {X.shape[1]} valid features")
        
        if len(np.unique(y)) < 2:
            print("Not enough variation in outcomes for prediction")
            return None
        
        # Split by tier if available
        tier_labels = [conv.get('tier', 'unknown') for conv in self.conversations]
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Logistic Regression with cross-validation
        print("\nLogistic Regression Results:")
        print("-" * 50)
        
        lr = LogisticRegression(random_state=42, max_iter=1000)
        
        # Stratified K-fold cross-validation
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(lr, X_scaled, y, cv=skf, scoring='roc_auc')
        
        print(f"Cross-validation ROC-AUC: {np.mean(cv_scores):.3f} (+/- {np.std(cv_scores):.3f})")
        
        # Fit on full data for feature importance
        lr.fit(X_scaled, y)
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': feature_names,
            'coefficient': lr.coef_[0],
            'abs_coefficient': np.abs(lr.coef_[0])
        }).sort_values('abs_coefficient', ascending=False)
        
        print("\nTop 10 Most Important Features:")
        for idx, row in feature_importance.head(10).iterrows():
            print(f"  {row['feature']}: {row['coefficient']:.3f}")
        
        # Per-tier analysis if tiers available
        if self.tier_results:
            print("\nPer-Tier Predictive Performance:")
            print("-" * 50)
            
            for tier in set(tier_labels):
                if tier != 'unknown':
                    tier_mask = np.array(tier_labels) == tier
                    if np.sum(tier_mask) > 10:  # Need enough samples
                        X_tier = X_scaled[tier_mask]
                        y_tier = y[tier_mask]
                        
                        if len(np.unique(y_tier)) > 1:
                            scores = cross_val_score(lr, X_tier, y_tier, cv=3, scoring='roc_auc')
                            print(f"  {tier}: ROC-AUC = {np.mean(scores):.3f}, "
                                  f"breakdown rate = {np.mean(y_tier):.2%}")
        
        # Statistical significance of individual features
        print("\nFeature Statistical Significance:")
        print("-" * 50)
        
        # Add intercept for statsmodels
        X_with_intercept = sm.add_constant(X_scaled)
        
        # Logistic regression with statsmodels for p-values
        logit_model = sm.Logit(y, X_with_intercept)
        result = logit_model.fit(disp=0)
        
        # Get p-values for features
        p_values = result.pvalues[1:]  # Exclude intercept
        
        significant_features = pd.DataFrame({
            'feature': feature_names,
            'coefficient': result.params[1:],
            'p_value': p_values,
            'significant': p_values < 0.05
        }).sort_values('p_value')
        
        print("Statistically Significant Features (p < 0.05):")
        for idx, row in significant_features[significant_features['significant']].iterrows():
            print(f"  {row['feature']}: coef={row['coefficient']:.3f}, p={row['p_value']:.6f}")
        
        return {
            'model': lr,
            'scaler': scaler,
            'feature_names': feature_names,
            'cv_scores': cv_scores,
            'feature_importance': feature_importance,
            'significant_features': significant_features,
            'breakdown_rate': np.mean(y),
            'n_samples': len(y)
        }
    
    def run_full_analysis(self, tier_directories, resume_from_checkpoint=True):
        """
        Run complete embedding trajectory analysis on conversations by tier.
        
        Args:
            tier_directories: Dict mapping tier names to directories containing conversations
                             e.g., {'full_reasoning': '/path/to/full/', 
                                    'light_reasoning': '/path/to/light/',
                                    'non_reasoning': '/path/to/non/'}
            resume_from_checkpoint: Whether to attempt resuming from a checkpoint
        """
        print("="*70)
        print("EMBEDDING TRAJECTORY ANALYSIS")
        if self.ensemble_mode:
            print(f"WITH ENSEMBLE INVARIANT DETECTION ({len(self.ensemble_models)} models)")
        print(f"ANALYZING {len(tier_directories)} MODEL TIERS")
        print("="*70)
        
        # Check for checkpoint
        checkpoint_data = None
        if resume_from_checkpoint:
            checkpoint_data = self.load_checkpoint('analysis_state')
            if checkpoint_data:
                print("\nResuming from checkpoint...")
                # Restore state
                self.conversations = checkpoint_data.get('conversations', [])
                self.tier_results = checkpoint_data.get('tier_results', {})
                processed_files = set(checkpoint_data.get('processed_files', []))
                print(f"Found {len(self.conversations)} already processed conversations")
            else:
                processed_files = set()
            
            # Load all conversation checkpoints into memory cache
            self.load_all_conversation_checkpoints()
        else:
            processed_files = set()
            self._checkpoint_cache = {}
        
        # Store tier information
        if not checkpoint_data:
            self.tier_results = {}
        
        # Process by tier
        all_conversation_files = []
        
        # Display tier summary
        print("\nTier Processing Summary:")
        for tier_name, tier_dir in tier_directories.items():
            tier_path = Path(tier_dir)
            tier_files = list(tier_path.glob('*.json'))
            tier_files_to_process = [f for f in tier_files if str(f) not in processed_files]
            already_processed = len(tier_files) - len(tier_files_to_process)
            print(f"  {tier_name}: {len(tier_files)} total ({already_processed} already processed)")
        print("")
        
        for tier_name, tier_dir in tier_directories.items():
            # Find all JSON files in tier directory
            tier_path = Path(tier_dir)
            tier_files = list(tier_path.glob('*.json'))
            
            # Filter out already processed files
            tier_files_to_process = [f for f in tier_files if str(f) not in processed_files]
            
            # Store tier information for each conversation
            # If resuming, get existing tier conversations
            if checkpoint_data and tier_name in self.tier_results:
                tier_conversations = self.tier_results[tier_name]['conversations']
            else:
                tier_conversations = []
                self.tier_results[tier_name] = {
                    'conversations': tier_conversations,
                    'n_conversations': 0
                }
            
            # Process conversations in batches for GPU efficiency
            batch_size = 10  # Process 10 conversations at a time
            checkpoint_counter = 0
            
            # Use a cleaner progress bar
            with tqdm(total=len(tier_files_to_process), 
                     desc=f"{tier_name.replace('_', ' ').title()}", 
                     unit="conv",
                     bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]') as pbar:
                
                # Process in batches
                for i in range(0, len(tier_files_to_process), batch_size):
                    batch_files = tier_files_to_process[i:i + batch_size]
                    batch_conversations = []
                    
                    # Load batch of conversations
                    for conv_file in batch_files:
                        try:
                            conv = self.load_conversation(conv_file, verbose=False)
                            conv['tier'] = tier_name
                            batch_conversations.append(conv)
                        except Exception as e:
                            pbar.write(f"Error loading {conv_file.name}: {e}")
                            pbar.update(1)
                            continue
                    
                    # Process conversations
                    if batch_conversations:
                        # First check if any need embedding
                        need_embedding = []
                        already_embedded = []
                        
                        for conv in batch_conversations:
                            session_id = conv['metadata']['session_id']
                            checkpoint_conv = self.load_conversation_checkpoint(session_id)
                            
                            if checkpoint_conv and 'embedding' in checkpoint_conv.get('analysis_completed', set()):
                                # Use checkpoint data
                                conv.update(checkpoint_conv)
                                already_embedded.append(conv)
                            else:
                                need_embedding.append(conv)
                        
                        # Batch embed only those that need it
                        if need_embedding:
                            need_embedding = self.embed_conversations_batch(need_embedding, verbose=False)
                            # Mark embedding as complete for these
                            for conv in need_embedding:
                                self.save_conversation_checkpoint(conv, 'embedding')
                        
                        # Combine all conversations
                        all_convs = already_embedded + need_embedding
                        
                        # Now analyze each conversation individually
                        for j, conv in enumerate(all_convs):
                            try:
                                # Use checkpoint-aware comprehensive analysis
                                # Skip re-embedding since we handled it above
                                completed_steps = self.get_completed_analysis_steps(conv)
                                if 'embedding' not in completed_steps:
                                    completed_steps.add('embedding')
                                    if 'analysis_completed' not in conv:
                                        conv['analysis_completed'] = set()
                                    conv['analysis_completed'].add('embedding')
                                
                                # Run comprehensive analysis with checkpointing
                                conv = self.analyze_conversation_comprehensive(conv, verbose=False, use_checkpoint=True)
                                
                                self.conversations.append(conv)
                                tier_conversations.append(conv)
                                all_conversation_files.append(batch_files[j])
                                processed_files.add(str(batch_files[j]))
                                
                            except Exception as e:
                                pbar.write(f"Error analyzing conversation: {e}")
                            
                            pbar.update(1)
                        
                        # Save checkpoint after each batch
                        checkpoint_counter += len(batch_conversations)
                        checkpoint_state = {
                            'conversations': self.conversations,
                            'tier_results': self.tier_results,
                            'processed_files': list(processed_files)
                        }
                        self.save_checkpoint('analysis_state', checkpoint_state)
            
            # Update tier results count
            self.tier_results[tier_name]['n_conversations'] = len(tier_conversations)
            
            # Save checkpoint after each tier
            checkpoint_state = {
                'conversations': self.conversations,
                'tier_results': self.tier_results,
                'processed_files': list(processed_files)
            }
            self.save_checkpoint('analysis_state', checkpoint_state)
            
        
        # Analyze patterns
        print("\n" + "="*70)
        print("ANALYZING EMBEDDING PATTERNS")
        print("="*70)
        
        # Find embedding clusters
        clusters, cluster_analysis = self.find_embedding_clusters()
        
        # Analyze embedding structure
        self.analyze_embedding_structure()
        
        # Analyze peer pressure convergence
        convergence_df = self.analyze_peer_pressure_convergence()
        
        # Identify high-density regions
        attractor_mask, densities = self.identify_attractor_regions()
        
        # Tier-specific analysis if applicable
        if self.tier_results:
            print("\n" + "="*70)
            print("TIER COMPARISON ANALYSIS")
            print("="*70)
            self.analyze_tier_differences()
            self.visualize_tier_trajectories()
            self.create_conversation_trajectory_space()
        
        # Generate visualizations
        print("\n" + "="*70)
        print("GENERATING VISUALIZATIONS")
        print("="*70)
        
        # 3D trajectory visualization
        self.visualize_trajectories_3d(method='pca', n_conversations=10)
        self.visualize_trajectories_3d(method='tsne', n_conversations=10)
        
        # Extract features for all conversations (for ML applications)
        feature_matrix = []
        phase_details = []  # Separate list for detailed phase information
        
        for conv in self.conversations:
            features = self.extract_feature_vector(conv)
            # Add tier information if available
            if 'tier' in conv:
                features['tier'] = conv['tier']
            
            # Add session ID for joining datasets
            features['session_id'] = conv['metadata']['session_id']
            feature_matrix.append(features)
            
            # Extract detailed phase information
            if 'phases' in conv:
                for phase in conv['phases']:
                    phase_detail = {
                        'session_id': conv['metadata']['session_id'],
                        'tier': conv.get('tier', 'unknown'),
                        'phase_turn': phase['turn'],
                        'phase_name': phase['phase'],
                        'analysis_time': phase.get('analysis_time', None)
                    }
                    phase_details.append(phase_detail)
        
        if feature_matrix:
            # Save feature matrix
            feature_df = pd.DataFrame(feature_matrix)
            feature_df.to_csv(self.output_dir / 'conversation_features.csv', index=False)
            print(f"\nSaved feature matrix ({len(feature_df)} conversations, {len(feature_df.columns)} features)")
            
            # Save detailed phase information
            if phase_details:
                phase_df = pd.DataFrame(phase_details)
                phase_df.to_csv(self.output_dir / 'conversation_phases.csv', index=False)
                print(f"Saved phase details ({len(phase_df)} phase transitions across {phase_df['session_id'].nunique()} conversations)")
        
        # Save comprehensive conversation metadata
        self.save_comprehensive_metadata()
        
        # Validate phase detection against manual annotations
        print("\n" + "="*70)
        print("PHASE DETECTION VALIDATION")
        print("="*70)
        
        # Run phase detection validation on conversations with manual phases
        conversations_with_phases = [c for c in self.conversations if c.get('phases')]
        
        if conversations_with_phases:
            print(f"Found {len(conversations_with_phases)} conversations with manual phase annotations")
            
            # Run sensitivity analysis on a sample
            sample_size = min(20, len(conversations_with_phases))
            sample_convs = np.random.choice(conversations_with_phases, sample_size, replace=False)
            
            all_results = []
            for conv in tqdm(sample_convs, desc="Validating phase detection"):
                sensitivity_results = self.phase_detection_sensitivity_analysis(conv)
                sensitivity_results['session_id'] = conv['metadata']['session_id']
                all_results.append(sensitivity_results)
            
            # Aggregate results
            combined_results = pd.concat(all_results)
            
            # Find best parameters
            best_params = combined_results.groupby(['window_size', 'threshold_percentile'])['f1_score'].mean()
            best_idx = best_params.idxmax()
            best_window, best_threshold = best_idx
            
            print(f"\nBest parameters:")
            print(f"  Window size: {best_window}")
            print(f"  Threshold percentile: {best_threshold}")
            print(f"  Average F1 score: {best_params.max():.3f}")
            
            # Save validation results
            validation_summary = combined_results.groupby(['window_size', 'threshold_percentile']).agg({
                'f1_score': ['mean', 'std'],
                'precision': 'mean',
                'recall': 'mean',
                'n_detected': 'mean',
                'n_manual': 'mean'
            }).round(3)
            
            validation_summary.to_csv(self.output_dir / 'phase_detection_validation.csv')
            print("\nSaved phase detection validation results to phase_detection_validation.csv")
        else:
            print("No conversations with manual phase annotations found for validation")
        
        # Run synthetic validation
        synthetic_results = self.create_synthetic_validation()
        
        # Run perturbation analysis on a sample of conversations
        print("\n" + "="*70)
        print("PERTURBATION ANALYSIS")
        print("="*70)
        
        sample_size = min(10, len(self.conversations))
        sample_convs = np.random.choice(self.conversations, sample_size, replace=False)
        
        all_perturbation_results = []
        for conv in tqdm(sample_convs, desc="Running perturbation analysis"):
            pert_results = self.perturbation_analysis(conv, n_perturbations=5)
            pert_results['session_id'] = conv['metadata']['session_id']
            all_perturbation_results.append(pert_results)
        
        if all_perturbation_results:
            combined_pert_results = pd.concat(all_perturbation_results)
            
            # Summarize by perturbation type
            summary = combined_pert_results.groupby('perturbation_type').agg({
                'procrustes_distance': ['mean', 'std'],
                'phase_preservation': ['mean', 'std'],
                'mean_point_distance': 'mean'
            }).round(3)
            
            print("\nPerturbation Analysis Summary:")
            print(summary)
            
            # Save results
            combined_pert_results.to_csv(self.output_dir / 'perturbation_analysis.csv', index=False)
            print("\nSaved perturbation analysis results")
        
        # Generate additional analyses
        print("\n" + "="*70)
        print("FULL DIMENSIONAL ANALYSIS")
        print("="*70)
        
        # First, analyze ALL conversations to get complete statistics
        print("\nAnalyzing all conversations for dimensional statistics...")
        
        dimensional_stats = {
            'intrinsic_dims': [],
            'participation_ratios': [],
            'active_dimensions': [],
            'dormant_dimensions': [],
            'smoothness_indices': [],
            'spectral_entropies': [],
            'tier_labels': []
        }
        
        # Process all conversations with progress bar
        with tqdm(total=len(self.conversations), 
                 desc="Dimensional analysis", 
                 unit="conv") as pbar:
            for conv in self.conversations:
                try:
                    # Run lightweight dimensional analysis (no visualization)
                    full_analysis = self.analyze_full_dimensional_structure(conv)
                    
                    # Collect statistics
                    if 'intrinsic_dimensionality' in full_analysis:
                        intrinsic = full_analysis['intrinsic_dimensionality']
                        if intrinsic['mle_dimension']:
                            dimensional_stats['intrinsic_dims'].append(intrinsic['mle_dimension'])
                    
                    if 'dimensional_utilization' in full_analysis:
                        dim_util = full_analysis['dimensional_utilization']
                        dimensional_stats['participation_ratios'].append(dim_util['participation_ratio'])
                        dimensional_stats['active_dimensions'].append(dim_util['active_dimensions'])
                        dimensional_stats['dormant_dimensions'].append(dim_util['dormant_dimensions'])
                    
                    if 'trajectory_smoothness' in full_analysis:
                        smooth = full_analysis['trajectory_smoothness']
                        dimensional_stats['smoothness_indices'].append(smooth['smoothness_index'])
                        if smooth['spectral_entropy']:
                            dimensional_stats['spectral_entropies'].append(smooth['spectral_entropy'])
                    
                    dimensional_stats['tier_labels'].append(conv.get('tier', 'unknown'))
                    
                    pbar.update(1)
                except Exception as e:
                    pbar.write(f"Error analyzing conversation: {e}")
                    pbar.update(1)
                    continue
        
        # Now intelligently select conversations for detailed visualization
        print("\nSelecting representative conversations for visualization...")
        
        selected_conversations = []
        visualizations_per_tier = 8
        
        if self.tier_results:
            # Select up to 8 conversations from each tier
            for tier_name in self.tier_results.keys():
                tier_convs = [c for c in self.conversations if c.get('tier') == tier_name]
                
                if len(tier_convs) > 0:
                    # Sort by intrinsic dimension to get diverse examples
                    tier_convs_with_dim = []
                    for conv in tier_convs:
                        if 'full_dimensional_analysis' in conv:
                            fa = conv['full_dimensional_analysis']
                            if 'intrinsic_dimensionality' in fa and fa['intrinsic_dimensionality']['mle_dimension']:
                                dim = fa['intrinsic_dimensionality']['mle_dimension']
                                tier_convs_with_dim.append((conv, dim))
                    
                    if tier_convs_with_dim:
                        # Sort by dimension and select evenly spaced
                        tier_convs_with_dim.sort(key=lambda x: x[1])
                        indices = np.linspace(0, len(tier_convs_with_dim)-1, 
                                            min(visualizations_per_tier, len(tier_convs_with_dim)), 
                                            dtype=int)
                        for idx in indices:
                            selected_conversations.append(tier_convs_with_dim[idx][0])
                    else:
                        # Random selection if no dimension data
                        n_select = min(visualizations_per_tier, len(tier_convs))
                        selected_conversations.extend(np.random.choice(tier_convs, n_select, replace=False))
        else:
            # No tiers, just select diverse sample
            n_select = min(24, len(self.conversations))  # 24 total if no tiers
            selected_conversations = np.random.choice(self.conversations, n_select, replace=False)
        
        print(f"Selected {len(selected_conversations)} conversations for detailed visualization")
        
        # Create detailed visualizations for selected conversations
        self.create_aggregated_dimensional_visualizations(selected_conversations)
        
        # Perform statistical tests on dimensional statistics
        self.perform_dimensional_statistical_tests(dimensional_stats)
        
        # Run statistical tests if we have tiers
        statistical_results = None
        if self.tier_results:
            statistical_results = self.perform_statistical_tests()
        
        # Validate length normalization
        self.validate_length_normalization()
        
        # Run null model comparison on a sample
        print("\n" + "="*70)
        print("NULL MODEL COMPARISON") 
        print("="*70)
        
        sample_size = min(50, len(self.conversations))
        sample_convs = np.random.choice(self.conversations, sample_size, replace=False).tolist()
        null_results = self.compare_to_null_models(sample_convs, n_null=100)
        
        # Store null model results
        self.null_model_results = null_results
        
        # Identify conversation topics and map to attractors
        topic_analysis = self.identify_conversation_topics()
        
        # Build predictive model (only if outcome labels are available)
        print("\n" + "="*70)
        print("CHECKING FOR PREDICTIVE MODELING CAPABILITY")
        print("="*70)
        
        predictive_results = self.build_predictive_model()
        
        if predictive_results is None:
            print("\nNOTE: Predictive modeling was skipped because no outcome labels were found.")
            print("To enable predictive modeling, ensure your conversation data includes")
            print("outcome labels in the metadata (e.g., 'outcome', 'breakdown', 'label').")
        
        # Store results for report
        self.statistical_results = statistical_results
        self.topic_analysis = topic_analysis
        self.predictive_results = predictive_results
        
        # Generate report
        self.generate_report()
        
        # Create ensemble summary visualization if in ensemble mode
        if self.ensemble_mode:
            self.create_ensemble_summary_visualization()
        
        print("\n" + "="*70)
        print("ANALYSIS COMPLETE")
        print("="*70)
        print(f"Results saved to: {self.output_dir}")
        print("\nNew files generated:")
        print("  - trajectory_comparison.png")
        print("  - dimensional_analysis_summary.png")
        print("  - projection_comparison_all.png")
        print("  - distance_matrices/ (folder with individual matrices)")  
        print("  - distance_matrices_tier_comparison.png")
        print("  - distance_statistics_summary.png")
        print("  - statistical_analysis.png")
        if self.ensemble_mode:
            print("  - ensemble_comparison_summary.png")
            print("  - ensemble_invariant_report.txt")
        if self.tier_results:
            print("  - tier_comparison_analysis.png")
            print("  - conversation_trajectory_space.html")
            print("  - tier_trajectory_comparison.html")
        
        # Clean up checkpoints on successful completion
        print("\nCleaning up checkpoints...")
        
        # Remove individual conversation checkpoints
        self.cleanup_conversation_checkpoints(self.conversations)
        
        # Remove main analysis checkpoint
        checkpoint_path = self.checkpoint_dir / 'analysis_state.pkl'
        if checkpoint_path.exists():
            checkpoint_path.unlink()
            print("Analysis completed successfully - main checkpoint removed")
        
        # Save final analysis state for future reference
        final_checkpoint = {
            'conversations': self.conversations,
            'tier_results': self.tier_results,
            'analysis_complete': True,
            'completion_time': datetime.now().isoformat(),
            'total_conversations': len(self.conversations)
        }
        self.save_checkpoint('analysis_state_final', final_checkpoint)
        
        # Clean up old checkpoints
        self.cleanup_old_checkpoints(keep_latest=3)
        
        return {
            'conversations': self.conversations,
            'clusters': cluster_analysis,
            'convergence': convergence_df,
            'attractors': (attractor_mask, densities),
            'features': feature_matrix,
            'tier_results': self.tier_results if self.tier_results else None
        }


# Example usage
if __name__ == "__main__":
    # Initialize analyzer with ensemble mode for invariant detection
    ensemble_models = [
        {'name': 'MPNet', 'model_id': 'all-mpnet-base-v2', 'dim': 768},
        {'name': 'MiniLM-L12', 'model_id': 'all-MiniLM-L12-v2', 'dim': 384},
        {'name': 'DistilBERT', 'model_id': 'all-distilroberta-v1', 'dim': 768},
    ]
    
    # Standard analysis for single conversation
    # analyzer = EmbeddingTrajectoryAnalyzer()
    
    # Ensemble analysis for invariant patterns
    analyzer = EmbeddingTrajectoryAnalyzer(
        model_name='all-MiniLM-L6-v2',
        ensemble_models=ensemble_models
    )
    
    # Option 1: Analyze by model tiers (NEW!)
    tier_directories = {
        'full_reasoning': '/home/knots/git/the-academy/docs/paper/exp-data/phase-1-premium/',
        'light_reasoning': '/home/knots/git/the-academy/docs/paper/exp-data/phase-2-efficient/',
        'non_reasoning': '/home/knots/git/the-academy/docs/paper/exp-data/phase-3-no-reasoning/'
    }
    
    # Run tier-based analysis
    # The analyzer will automatically save checkpoints every 10 conversations
    # If the script is interrupted, it will resume from the last checkpoint
    # To force a fresh start, use: resume_from_checkpoint=False
    results = analyzer.run_full_analysis(tier_directories)
    
    print("\nAnalysis complete! Check the 'embedding_analysis' directory for results.")
    
    # Access tier-specific results
    if results.get('tier_results'):
        print("\nTier Analysis Summary:")
        for tier_name, tier_data in results['tier_results'].items():
            print(f"  {tier_name}: {tier_data['n_conversations']} conversations analyzed")
    
    # Example of accessing extracted features for ML
    if results['features']:
        print(f"\nExtracted {len(results['features'][0])} features per conversation")
        print("Features include tier labels for downstream analysis!")
        
        # Convert to DataFrame for easy analysis
        import pandas as pd
        feature_df = pd.DataFrame(results['features'])
        
        # Group by tier if available
        if 'tier' in feature_df.columns:
            print("\nFeature means by tier:")
            print(feature_df.groupby('tier').mean()[['traj_mean_distance', 'intrinsic_dim_mle', 'participation_ratio']])