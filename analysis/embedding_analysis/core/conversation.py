"""
Conversation loading and preprocessing functionality.
"""

import json
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Union
import logging

logger = logging.getLogger(__name__)


class ConversationLoader:
    """
    Handles loading and preprocessing of conversation data.
    """
    
    def __init__(self):
        """Initialize conversation loader."""
        self.loaded_conversations = {}
        
    def load_conversation(self, 
                         json_path: Union[str, Path],
                         validate: bool = True) -> Optional[Dict]:
        """
        Load a conversation from JSON file.
        
        Args:
            json_path: Path to JSON file
            validate: Whether to validate conversation structure
            
        Returns:
            Conversation dictionary or None if invalid
        """
        json_path = Path(json_path)
        
        if not json_path.exists():
            logger.error(f"File not found: {json_path}")
            return None
            
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
                
            if validate and not self._validate_conversation(data):
                return None
                
            # Add metadata
            conversation = self._preprocess_conversation(data, json_path)
            
            # Cache conversation
            conv_hash = self._get_conversation_hash(json_path)
            self.loaded_conversations[conv_hash] = conversation
            
            return conversation
            
        except Exception as e:
            logger.error(f"Error loading {json_path}: {e}")
            return None
            
    def load_conversations_batch(self, 
                               directory: Union[str, Path],
                               pattern: str = "*.json",
                               max_conversations: Optional[int] = None) -> List[Dict]:
        """
        Load multiple conversations from a directory.
        
        Args:
            directory: Directory containing JSON files
            pattern: Glob pattern for JSON files
            max_conversations: Maximum number to load
            
        Returns:
            List of conversation dictionaries
        """
        directory = Path(directory)
        conversations = []
        
        json_files = list(directory.glob(pattern))
        
        if max_conversations:
            json_files = json_files[:max_conversations]
            
        for json_path in json_files:
            conv = self.load_conversation(json_path)
            if conv:
                conversations.append(conv)
                
        logger.info(f"Loaded {len(conversations)} conversations from {directory}")
        return conversations
        
    def _validate_conversation(self, data: Dict) -> bool:
        """Validate conversation structure."""
        # Check for required fields
        if 'messages' not in data:
            # Check if messages are nested under 'session'
            if 'session' in data and 'messages' in data['session']:
                messages = data['session']['messages']
            else:
                logger.warning("Conversation missing 'messages' field")
                return False
        else:
            messages = data['messages']
        
        if not isinstance(messages, list) or len(messages) == 0:
            logger.warning("Invalid or empty messages list")
            return False
            
        # Validate message structure
        for i, msg in enumerate(messages):
            if not isinstance(msg, dict):
                logger.warning(f"Message {i} is not a dictionary")
                return False
                
            if 'content' not in msg and 'text' not in msg:
                logger.warning(f"Message {i} missing content/text field")
                return False
                
        return True
        
    def _preprocess_conversation(self, data: Dict, source_path: Path) -> Dict:
        """Preprocess and enrich conversation data."""
        conversation = data.copy()
        
        # Handle nested structure
        if 'session' in data and 'messages' in data['session']:
            conversation['messages'] = data['session']['messages']
            # Copy session metadata
            if 'id' in data['session']:
                conversation['session_id'] = data['session']['id']
        
        # Standardize message format
        messages = []
        for i, msg in enumerate(conversation['messages']):
            std_msg = {
                'turn': i,
                'content': msg.get('content') or msg.get('text', ''),
                'speaker': msg.get('speaker') or msg.get('role', f'speaker_{i%2}'),
                'original': msg
            }
            
            # Add any additional fields
            for key, value in msg.items():
                if key not in ['content', 'text', 'speaker', 'role']:
                    std_msg[key] = value
                    
            messages.append(std_msg)
            
        conversation['messages'] = messages
        
        # Add metadata
        if 'metadata' not in conversation:
            conversation['metadata'] = {}
            
        conversation['metadata'].update({
            'source_path': str(source_path),
            'filename': source_path.name,
            'session_id': source_path.stem,
            'n_messages': len(messages),
            'n_turns': len(messages)
        })
        
        # Extract phases if present
        if 'phases' in data:
            conversation['phases'] = self._standardize_phases(data['phases'])
        elif 'analysisHistory' in data:
            # Extract phases from analysis history like analysis.py does
            conversation['phases'] = self._extract_phases_from_analysis_history(
                data.get('analysisHistory', []), 
                messages
            )
        elif 'session' in data and 'analysisHistory' in data['session']:
            # Check if analysisHistory is under session
            conversation['phases'] = self._extract_phases_from_analysis_history(
                data['session'].get('analysisHistory', []), 
                messages
            )
            
        # Determine outcome if present
        if 'outcome' in data:
            conversation['metadata']['outcome'] = data['outcome']
        elif 'breakdown' in data:
            conversation['metadata']['outcome'] = 'breakdown' if data['breakdown'] else 'successful'
            
        return conversation
        
    def _standardize_phases(self, phases: Union[List, Dict]) -> List[Dict]:
        """Standardize phase annotations."""
        if isinstance(phases, dict):
            # Convert dict to list format
            phase_list = []
            for phase_name, turn in phases.items():
                phase_list.append({
                    'phase': phase_name,
                    'turn': turn
                })
            return sorted(phase_list, key=lambda x: x['turn'])
            
        elif isinstance(phases, list):
            # Ensure consistent format
            standardized = []
            for phase in phases:
                if isinstance(phase, dict) and 'turn' in phase:
                    standardized.append({
                        'phase': phase.get('phase', f'phase_{len(standardized)}'),
                        'turn': phase['turn']
                    })
            return sorted(standardized, key=lambda x: x['turn'])
            
        return []
        
    def _extract_phases_from_analysis_history(self, analysis_history: List[Dict], messages: List[Dict]) -> List[Dict]:
        """Extract conversation phases from analysis history."""
        phases = []
        phase_set = set()  # To avoid duplicates
        
        if not analysis_history or not messages:
            return phases
            
        # Create a list of message timestamps for mapping
        message_timestamps = [(i, msg.get('timestamp') or msg.get('original', {}).get('timestamp')) 
                            for i, msg in enumerate(messages) if msg.get('timestamp') or msg.get('original', {}).get('timestamp')]
        
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
                            'phase': phase
                        })
                        phase_set.add(phase_key)
                except:
                    # Fallback to messageCountAtAnalysis if timestamp parsing fails
                    turn = analysis.get('messageCountAtAnalysis', 0)
                    phase_key = f"{turn}:{phase}"
                    
                    if phase_key not in phase_set:
                        phases.append({
                            'turn': turn,
                            'phase': phase
                        })
                        phase_set.add(phase_key)
                        
        # Sort phases by turn
        phases.sort(key=lambda x: x['turn'])
        return phases
        
    def _get_conversation_hash(self, json_path: Path) -> str:
        """Generate hash for conversation file."""
        return hashlib.md5(str(json_path).encode()).hexdigest()
        
    def filter_conversations(self,
                           conversations: List[Dict],
                           min_turns: int = 20,
                           max_turns: Optional[int] = None,
                           require_outcome: bool = False) -> List[Dict]:
        """
        Filter conversations based on criteria.
        
        Args:
            conversations: List of conversations
            min_turns: Minimum number of turns
            max_turns: Maximum number of turns
            require_outcome: Whether to require outcome labels
            
        Returns:
            Filtered list of conversations
        """
        filtered = []
        
        for conv in conversations:
            n_turns = conv['metadata']['n_turns']
            
            # Check turn count
            if n_turns < min_turns:
                continue
            if max_turns and n_turns > max_turns:
                continue
                
            # Check outcome requirement
            if require_outcome and 'outcome' not in conv['metadata']:
                continue
                
            filtered.append(conv)
            
        logger.info(f"Filtered {len(conversations)} to {len(filtered)} conversations")
        return filtered