import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from typing import List, Tuple, Dict, Optional
import logging
from collections import defaultdict
import pickle


class SequentialDataset(Dataset):
    """
    Dataset for sequential recommendation that handles user interaction sequences.
    """
    
    def __init__(self, sequences: List[List[int]], targets: List[int], 
                 max_seq_len: int = 50, item_encoder: Optional[LabelEncoder] = None):
        """
        Args:
            sequences: List of item sequences for each user
            targets: List of target items (next item in sequence)
            max_seq_len: Maximum sequence length for padding/truncation
            item_encoder: Label encoder for items
        """
        self.sequences = sequences
        self.targets = targets
        self.max_seq_len = max_seq_len
        self.item_encoder = item_encoder
        
        # Pad/truncate sequences
        self.padded_sequences = []
        self.sequence_lengths = []
        
        for seq in sequences:
            if len(seq) > max_seq_len:
                # Take the most recent items
                padded_seq = seq[-max_seq_len:]
                seq_len = max_seq_len
            else:
                # Pad with zeros
                padded_seq = seq + [0] * (max_seq_len - len(seq))
                seq_len = len(seq)
            
            self.padded_sequences.append(padded_seq)
            self.sequence_lengths.append(seq_len)
        
        # Convert to tensors
        self.sequences_tensor = torch.LongTensor(self.padded_sequences)
        self.targets_tensor = torch.LongTensor(targets)
        self.lengths_tensor = torch.LongTensor(self.sequence_lengths)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return {
            'sequence': self.sequences_tensor[idx],
            'target': self.targets_tensor[idx],
            'length': self.lengths_tensor[idx]
        }


class SessionDataset(Dataset):
    """
    Dataset for session-based recommendation where each sample is a session.
    """
    
    def __init__(self, sessions: List[List[int]], max_session_len: int = 20):
        """
        Args:
            sessions: List of sessions, each containing item IDs
            max_session_len: Maximum session length
        """
        self.sessions = sessions
        self.max_session_len = max_session_len
        
        # Process sessions for next-item prediction
        self.input_sequences = []
        self.target_items = []
        self.session_lengths = []
        
        for session in sessions:
            if len(session) < 2:  # Need at least 2 items
                continue
            
            # Create multiple training samples from each session
            for i in range(1, len(session)):
                input_seq = session[:i]
                target_item = session[i]
                
                # Pad/truncate input sequence
                if len(input_seq) > max_session_len:
                    input_seq = input_seq[-max_session_len:]
                    seq_len = max_session_len
                else:
                    seq_len = len(input_seq)
                    input_seq = input_seq + [0] * (max_session_len - len(input_seq))
                
                self.input_sequences.append(input_seq)
                self.target_items.append(target_item)
                self.session_lengths.append(seq_len)
        
        # Convert to tensors
        self.sequences_tensor = torch.LongTensor(self.input_sequences)
        self.targets_tensor = torch.LongTensor(self.target_items)
        self.lengths_tensor = torch.LongTensor(self.session_lengths)
    
    def __len__(self):
        return len(self.input_sequences)
    
    def __getitem__(self, idx):
        return {
            'sequence': self.sequences_tensor[idx],
            'target': self.targets_tensor[idx],
            'length': self.lengths_tensor[idx]
        }


class SequentialDataLoader:
    """
    Data loader for sequential recommendation models with various preprocessing options.
    """
    
    def __init__(self, ratings_path: str, min_interactions: int = 10, 
                 min_seq_length: int = 3, max_seq_length: int = 50):
        """
        Args:
            ratings_path: Path to ratings CSV file
            min_interactions: Minimum interactions per user
            min_seq_length: Minimum sequence length to include
            max_seq_length: Maximum sequence length for training
        """
        self.ratings_path = ratings_path
        self.min_interactions = min_interactions
        self.min_seq_length = min_seq_length
        self.max_seq_length = max_seq_length
        
        self.logger = logging.getLogger(__name__)
        
        # Initialize encoders
        self.user_encoder = LabelEncoder()
        self.item_encoder = LabelEncoder()
        
        # Load and process data
        self._load_data()
        self._create_sequences()
    
    def _load_data(self):
        """Load and preprocess ratings data"""
        self.logger.info(f"Loading ratings from {self.ratings_path}")
        
        # Load ratings
        self.ratings_df = pd.read_csv(self.ratings_path)
        
        # Sort by user and timestamp for sequential order
        self.ratings_df = self.ratings_df.sort_values(['userId', 'timestamp'])
        
        # Filter users with minimum interactions
        user_counts = self.ratings_df['userId'].value_counts()
        valid_users = user_counts[user_counts >= self.min_interactions].index
        self.ratings_df = self.ratings_df[self.ratings_df['userId'].isin(valid_users)]
        
        self.logger.info(f"Filtered to {len(valid_users)} users with {len(self.ratings_df)} interactions")
        
        # Encode users and items
        self.ratings_df['user_encoded'] = self.user_encoder.fit_transform(self.ratings_df['userId'])
        self.ratings_df['item_encoded'] = self.item_encoder.fit_transform(self.ratings_df['movieId'])
        
        # Add 1 to item encodings to reserve 0 for padding
        self.ratings_df['item_encoded'] += 1
    
    def _create_sequences(self):
        """Create user interaction sequences"""
        self.user_sequences = defaultdict(list)
        
        # Group by user and create sequences
        for _, row in self.ratings_df.iterrows():
            user_id = row['user_encoded']
            item_id = row['item_encoded']
            
            self.user_sequences[user_id].append(item_id)
        
        # Filter sequences by minimum length
        filtered_sequences = {
            user_id: seq for user_id, seq in self.user_sequences.items()
            if len(seq) >= self.min_seq_length
        }
        
        self.user_sequences = filtered_sequences
        self.logger.info(f"Created {len(self.user_sequences)} user sequences")
    
    def get_sequential_splits(self, train_ratio: float = 0.8, val_ratio: float = 0.1) -> Tuple[List, List, List]:
        """
        Create train/val/test splits for sequential recommendation.
        Uses temporal splitting to maintain chronological order.
        
        Returns:
            Tuple of (train_sequences, val_sequences, test_sequences)
        """
        train_sequences = []
        val_sequences = []
        test_sequences = []
        
        for user_id, sequence in self.user_sequences.items():
            seq_len = len(sequence)
            
            # Calculate split points
            train_end = int(seq_len * train_ratio)
            val_end = int(seq_len * (train_ratio + val_ratio))
            
            # Ensure minimum sequence lengths
            if train_end < self.min_seq_length:
                continue
            
            # Create training samples (all possible subsequences)
            train_seq = sequence[:train_end]
            for i in range(self.min_seq_length, len(train_seq)):
                input_seq = train_seq[:i]
                target = train_seq[i]
                train_sequences.append((input_seq, target))
            
            # Validation sample
            if val_end > train_end:
                val_input = sequence[:val_end-1]
                val_target = sequence[val_end-1]
                val_sequences.append((val_input, val_target))
            
            # Test sample
            if seq_len > val_end:
                test_input = sequence[:seq_len-1]
                test_target = sequence[seq_len-1]
                test_sequences.append((test_input, test_target))
        
        self.logger.info(f"Created splits: train={len(train_sequences)}, "
                        f"val={len(val_sequences)}, test={len(test_sequences)}")
        
        return train_sequences, val_sequences, test_sequences
    
    def get_session_splits(self, session_threshold: int = 3600) -> Tuple[List, List, List]:
        """
        Create session-based splits where sessions are defined by time gaps.
        
        Args:
            session_threshold: Time gap (seconds) to define new session
            
        Returns:
            Tuple of (train_sessions, val_sessions, test_sessions)
        """
        all_sessions = []
        
        # Create sessions for each user
        for user_id, sequence in self.user_sequences.items():
            # Get user's ratings with timestamps
            user_ratings = self.ratings_df[self.ratings_df['user_encoded'] == user_id].copy()
            user_ratings = user_ratings.sort_values('timestamp')
            
            # Split into sessions based on time gaps
            sessions = []
            current_session = []
            prev_timestamp = None
            
            for _, row in user_ratings.iterrows():
                timestamp = row['timestamp']
                item_id = row['item_encoded']
                
                if prev_timestamp is None or (timestamp - prev_timestamp) <= session_threshold:
                    current_session.append(item_id)
                else:
                    if len(current_session) >= 2:
                        sessions.append(current_session)
                    current_session = [item_id]
                
                prev_timestamp = timestamp
            
            # Add final session
            if len(current_session) >= 2:
                sessions.append(current_session)
            
            all_sessions.extend(sessions)
        
        # Split sessions into train/val/test
        np.random.shuffle(all_sessions)
        
        train_end = int(len(all_sessions) * 0.8)
        val_end = int(len(all_sessions) * 0.9)
        
        train_sessions = all_sessions[:train_end]
        val_sessions = all_sessions[train_end:val_end]
        test_sessions = all_sessions[val_end:]
        
        self.logger.info(f"Created session splits: train={len(train_sessions)}, "
                        f"val={len(val_sessions)}, test={len(test_sessions)}")
        
        return train_sessions, val_sessions, test_sessions
    
    def create_data_loaders(self, data_type: str = 'sequential', batch_size: int = 256, 
                          num_workers: int = 4) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Create PyTorch data loaders for training.
        
        Args:
            data_type: 'sequential' or 'session'
            batch_size: Batch size for data loaders
            num_workers: Number of worker processes
            
        Returns:
            Tuple of (train_loader, val_loader, test_loader)
        """
        if data_type == 'sequential':
            train_data, val_data, test_data = self.get_sequential_splits()
            
            # Extract sequences and targets
            train_sequences = [seq for seq, _ in train_data]
            train_targets = [target for _, target in train_data]
            
            val_sequences = [seq for seq, _ in val_data]
            val_targets = [target for _, target in val_data]
            
            test_sequences = [seq for seq, _ in test_data]
            test_targets = [target for _, target in test_data]
            
            # Create datasets
            train_dataset = SequentialDataset(
                train_sequences, train_targets, self.max_seq_length, self.item_encoder
            )
            val_dataset = SequentialDataset(
                val_sequences, val_targets, self.max_seq_length, self.item_encoder
            )
            test_dataset = SequentialDataset(
                test_sequences, test_targets, self.max_seq_length, self.item_encoder
            )
            
        elif data_type == 'session':
            train_sessions, val_sessions, test_sessions = self.get_session_splits()
            
            # Create datasets
            train_dataset = SessionDataset(train_sessions)
            val_dataset = SessionDataset(val_sessions)
            test_dataset = SessionDataset(test_sessions)
            
        else:
            raise ValueError(f"Unknown data_type: {data_type}")
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, 
            num_workers=num_workers, pin_memory=True
        )
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True
        )
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True
        )
        
        return train_loader, val_loader, test_loader
    
    def get_model_config(self) -> Dict:
        """Get configuration for model initialization"""
        return {
            'num_items': len(self.item_encoder.classes_) + 1,  # +1 for padding
            'num_users': len(self.user_encoder.classes_),
            'max_seq_length': self.max_seq_length
        }
    
    def save_encoders(self, save_path: str):
        """Save encoders for inference"""
        encoders = {
            'user_encoder': self.user_encoder,
            'item_encoder': self.item_encoder,
            'user_sequences': dict(self.user_sequences)
        }
        
        with open(save_path, 'wb') as f:
            pickle.dump(encoders, f)
        
        self.logger.info(f"Saved encoders to {save_path}")
    
    @classmethod
    def load_encoders(cls, load_path: str) -> Dict:
        """Load encoders for inference"""
        with open(load_path, 'rb') as f:
            encoders = pickle.load(f)
        
        return encoders


class HierarchicalDataset(Dataset):
    """
    Dataset for hierarchical sequential models that need both short and long sequences.
    """
    
    def __init__(self, user_sequences: Dict, short_len: int = 10, long_len: int = 50):
        """
        Args:
            user_sequences: Dictionary mapping user_id to sequence
            short_len: Length of short-term sequences
            long_len: Length of long-term sequences
        """
        self.user_sequences = user_sequences
        self.short_len = short_len
        self.long_len = long_len
        
        # Create training samples
        self.samples = []
        
        for user_id, sequence in user_sequences.items():
            if len(sequence) < short_len + 1:
                continue
            
            # Create samples at different time points
            for i in range(short_len, len(sequence)):
                # Target is the current item
                target = sequence[i]
                
                # Short sequence (recent items)
                short_end = i
                short_start = max(0, short_end - short_len)
                short_seq = sequence[short_start:short_end]
                
                # Long sequence (full history up to this point)
                long_seq = sequence[:i]
                if len(long_seq) > long_len:
                    long_seq = long_seq[-long_len:]
                
                # Pad sequences
                short_padded = short_seq + [0] * (short_len - len(short_seq))
                long_padded = long_seq + [0] * (long_len - len(long_seq))
                
                self.samples.append({
                    'short_sequence': short_padded,
                    'long_sequence': long_padded,
                    'short_length': len(short_seq),
                    'long_length': len(long_seq),
                    'target': target
                })
        
        # Convert to tensors
        self.short_sequences = torch.LongTensor([s['short_sequence'] for s in self.samples])
        self.long_sequences = torch.LongTensor([s['long_sequence'] for s in self.samples])
        self.short_lengths = torch.LongTensor([s['short_length'] for s in self.samples])
        self.long_lengths = torch.LongTensor([s['long_length'] for s in self.samples])
        self.targets = torch.LongTensor([s['target'] for s in self.samples])
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return {
            'short_sequence': self.short_sequences[idx],
            'long_sequence': self.long_sequences[idx],
            'short_length': self.short_lengths[idx],
            'long_length': self.long_lengths[idx],
            'target': self.targets[idx]
        }