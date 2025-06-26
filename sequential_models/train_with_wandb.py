#!/usr/bin/env python3
"""
Enhanced Sequential Recommendation Model Training with Full Wandb Integration
Production-ready training script with comprehensive monitoring
"""

import argparse
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle
import os
import sys
from pathlib import Path
from datetime import datetime
import math

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

# Import wandb utilities
from wandb_config import init_wandb_for_training, WandbManager
from wandb_training_integration import WandbTrainingLogger

logger = logging.getLogger(__name__)


class SequentialDataset(Dataset):
    """Dataset for Sequential recommendation training"""
    
    def __init__(self, sequences, targets, max_seq_len=50):
        self.sequences = sequences
        self.targets = targets
        self.max_seq_len = max_seq_len
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        seq = self.sequences[idx]
        target = self.targets[idx]
        
        # Pad or truncate sequence
        if len(seq) > self.max_seq_len:
            seq = seq[-self.max_seq_len:]
        else:
            seq = [0] * (self.max_seq_len - len(seq)) + seq
        
        return torch.LongTensor(seq), torch.LongTensor([target])


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer-based models"""
    
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:x.size(0), :]


class TransformerBlock(nn.Module):
    """Transformer block for sequential modeling"""
    
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(TransformerBlock, self).__init__()
        
        self.self_attention = nn.MultiheadAttention(d_model, num_heads, dropout=dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        # Self-attention
        attn_output, _ = self.self_attention(x, x, x, attn_mask=mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed-forward
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x


class AttentionalSequentialRecommender(nn.Module):
    """
    Attentional Sequential Recommender with Transformer architecture
    """
    
    def __init__(self, num_items, embedding_dim=256, num_heads=8, num_layers=4, 
                 d_ff=512, max_seq_len=50, dropout=0.1):
        super(AttentionalSequentialRecommender, self).__init__()
        
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.max_seq_len = max_seq_len
        
        # Item embedding (with padding token at index 0)
        self.item_embedding = nn.Embedding(num_items + 1, embedding_dim, padding_idx=0)
        self.pos_encoding = PositionalEncoding(embedding_dim, max_seq_len)
        
        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(embedding_dim, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        # Output projection (should match embedding vocab size minus padding)
        self.output_projection = nn.Linear(embedding_dim, num_items)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights"""
        for module in self.modules():
            if isinstance(module, nn.Embedding):
                nn.init.xavier_normal_(module.weight)
                if hasattr(module, 'padding_idx') and module.padding_idx is not None:
                    with torch.no_grad():
                        module.weight[module.padding_idx].fill_(0)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def create_padding_mask(self, seq):
        """Create padding mask for sequences"""
        return (seq == 0).transpose(0, 1)
    
    def create_causal_mask(self, seq_len):
        """Create causal mask for autoregressive generation"""
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask
    
    def forward(self, sequences):
        batch_size, seq_len = sequences.shape
        
        # Embed items
        x = self.item_embedding(sequences)  # [batch, seq_len, embed_dim]
        x = x * math.sqrt(self.embedding_dim)  # Scale embeddings
        
        # Add positional encoding
        x = x.transpose(0, 1)  # [seq_len, batch, embed_dim]
        x = self.pos_encoding(x)
        x = self.dropout(x)
        
        # Create masks
        padding_mask = self.create_padding_mask(sequences)
        causal_mask = self.create_causal_mask(seq_len).to(sequences.device)
        
        # Apply transformer blocks
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x, mask=causal_mask)
        
        # Get the last non-padding item representation
        # For simplicity, we'll use the last item in the sequence
        x = x[-1]  # [batch, embed_dim]
        
        # Project to item space
        logits = self.output_projection(x)  # [batch, num_items]
        
        return logits


def setup_logging():
    """Minimal logging setup - Wandb handles most logging"""
    logging.basicConfig(
        level=logging.WARNING,
        format='%(levelname)s - %(message)s'
    )


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train Sequential Model with Wandb')
    
    # Data arguments
    # Try multiple possible paths for the data files
    possible_ratings_paths = [
        str(Path(__file__).parent.parent / 'movies' / 'cinesync' / 'ml-32m' / 'ratings.csv'),
        '../movies/cinesync/ml-32m/ratings.csv',
        '../../movies/cinesync/ml-32m/ratings.csv',
        'movies/cinesync/ml-32m/ratings.csv',
        '/Users/timmy/workspace/ai-apps/cine-sync-v2/movies/cinesync/ml-32m/ratings.csv'
    ]
    
    possible_movies_paths = [
        str(Path(__file__).parent.parent / 'movies' / 'cinesync' / 'ml-32m' / 'movies.csv'),
        '../movies/cinesync/ml-32m/movies.csv',
        '../../movies/cinesync/ml-32m/movies.csv',
        'movies/cinesync/ml-32m/movies.csv',
        '/Users/timmy/workspace/ai-apps/cine-sync-v2/movies/cinesync/ml-32m/movies.csv'
    ]
    
    # Find first existing path
    default_ratings_path = None
    for path in possible_ratings_paths:
        if os.path.exists(path):
            default_ratings_path = path
            break
    
    default_movies_path = None
    for path in possible_movies_paths:
        if os.path.exists(path):
            default_movies_path = path
            break
    
    parser.add_argument('--ratings-path', type=str, default=default_ratings_path,
                       help='Path to ratings CSV file')
    parser.add_argument('--movies-path', type=str, default=default_movies_path,
                       help='Path to movies CSV file')
    parser.add_argument('--min-interactions', type=int, default=20,
                       help='Minimum interactions per user')
    parser.add_argument('--min-seq-length', type=int, default=5,
                       help='Minimum sequence length')
    parser.add_argument('--max-seq-length', type=int, default=50,
                       help='Maximum sequence length')
    
    # Model arguments
    parser.add_argument('--embedding-dim', type=int, default=256,
                       help='Embedding dimension')
    parser.add_argument('--num-heads', type=int, default=8,
                       help='Number of attention heads')
    parser.add_argument('--num-layers', type=int, default=4,
                       help='Number of transformer layers')
    parser.add_argument('--d-ff', type=int, default=512,
                       help='Feed-forward dimension')
    parser.add_argument('--dropout', type=float, default=0.1,
                       help='Dropout rate')
    
    # Training arguments
    parser.add_argument('--learning-rate', type=float, default=0.0001,
                       help='Learning rate')
    parser.add_argument('--batch-size', type=int, default=64,
                       help='Batch size')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of epochs')
    parser.add_argument('--early-stopping-patience', type=int, default=10,
                       help='Early stopping patience')
    parser.add_argument('--warmup-steps', type=int, default=4000,
                       help='Warmup steps for learning rate schedule')
    
    # Wandb arguments
    parser.add_argument('--wandb-project', type=str, default='cinesync-v2-sequential',
                       help='Wandb project name')
    parser.add_argument('--wandb-entity', type=str, default=None,
                       help='Wandb entity')
    parser.add_argument('--wandb-name', type=str, default=None,
                       help='Wandb run name')
    parser.add_argument('--wandb-tags', type=str, nargs='+', 
                       default=['sequential', 'transformer', 'production'],
                       help='Wandb tags')
    parser.add_argument('--wandb-offline', action='store_true',
                       help='Run wandb in offline mode')
    
    # System arguments
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cpu', 'cuda'],
                       help='Device to use for training')
    parser.add_argument('--num-workers', type=int, default=4,
                       help='Number of data loader workers')
    
    return parser.parse_args()


def prepare_sequential_data(ratings_path, min_interactions=5, min_seq_length=3, 
                          max_seq_length=50, test_size=0.2, val_size=0.1):
    """
    Prepare sequential data for training
    
    Returns:
        train_loader, val_loader, test_loader, metadata
    """
    
    # Load ratings with chunked loading for large files
    file_size = os.path.getsize(ratings_path) / (1024 * 1024)  # Size in MB
    
    if file_size > 200:  # Use chunked loading for files > 200MB
        chunks = []
        chunk_size = 25000  # Smaller chunks to manage memory
        for i, chunk in enumerate(pd.read_csv(ratings_path, chunksize=chunk_size)):
            chunks.append(chunk)
        ratings_df = pd.concat(chunks, ignore_index=True)
        del chunks  # Free memory
    else:
        ratings_df = pd.read_csv(ratings_path)
    
    # Sort by user (no timestamp available in this dataset)
    ratings_df = ratings_df.sort_values(['uid'])
    
    # Filter users with minimum interactions
    user_counts = ratings_df['uid'].value_counts()
    print(f"Total users: {len(user_counts)}")
    print(f"Users with >= {min_interactions} interactions: {len(user_counts[user_counts >= min_interactions])}")
    valid_users = user_counts[user_counts >= min_interactions].index
    ratings_df = ratings_df[ratings_df['uid'].isin(valid_users)]
    print(f"Filtered ratings: {len(ratings_df)}")
    
    
    # Create item encoder
    item_encoder = LabelEncoder()
    ratings_df['item_idx'] = item_encoder.fit_transform(ratings_df['anime_uid']) + 1  # +1 for padding
    
    # Group by user to create sequences (memory-efficient)
    user_sequences = []
    user_targets = []
    max_sequences_per_user = 5  # Further limit sequences per user to prevent memory explosion
    
    grouped = list(ratings_df.groupby('uid'))
    
    for idx, (user_id, group) in enumerate(grouped):
        items = group['item_idx'].tolist()
        
        if len(items) < min_seq_length + 1:
            continue
            
        # Create fewer sequences per user to manage memory
        # Use only the last few sequences (most recent behavior)
        start_idx = max(min_seq_length, len(items) - max_sequences_per_user)
        
        for i in range(start_idx, len(items)):
            # Use sliding window approach with maximum sequence length
            seq_start = max(0, i - max_seq_length)
            seq = items[seq_start:i]
            target = items[i] - 1  # Adjust target back to 0-based indexing for output layer
            
            if len(seq) >= min_seq_length and target >= 0:  # Ensure valid target
                user_sequences.append(seq)
                user_targets.append(target)
                
        # Clear items list to free memory
        del items
    
    del grouped  # Free memory
    
    print(f"Total sequences created: {len(user_sequences)}")
    
    # Split data
    train_val_seq, test_seq, train_val_targets, test_targets = train_test_split(
        user_sequences, user_targets, test_size=test_size, random_state=42
    )
    train_seq, val_seq, train_targets, val_targets = train_test_split(
        train_val_seq, train_val_targets, test_size=val_size/(1-test_size), random_state=42
    )
    
    
    # Create datasets
    train_dataset = SequentialDataset(train_seq, train_targets, max_seq_length)
    val_dataset = SequentialDataset(val_seq, val_targets, max_seq_length)
    test_dataset = SequentialDataset(test_seq, test_targets, max_seq_length)
    
    # Create memory-efficient data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=128,  # Smaller batch for memory efficiency
        shuffle=True, 
        num_workers=4,   # Fewer workers to reduce memory overhead
        pin_memory=False,  # Disable pin_memory to save GPU memory
        persistent_workers=False
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=256,  # Smaller validation batch
        shuffle=False, 
        num_workers=4,
        pin_memory=False,
        persistent_workers=False
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=256,
        shuffle=False, 
        num_workers=4,
        pin_memory=False,
        persistent_workers=False
    )
    
    # Metadata
    metadata = {
        'num_items': len(item_encoder.classes_),
        'num_sequences': len(user_sequences),
        'max_seq_length': max_seq_length,
        'min_seq_length': min_seq_length,
        'item_encoder': item_encoder,
        'train_size': len(train_seq),
        'val_size': len(val_seq),
        'test_size': len(test_seq),
        'vocab_size': len(item_encoder.classes_) + 1  # +1 for padding
    }
    
    return train_loader, val_loader, test_loader, metadata


class WarmupScheduler:
    """Warmup learning rate scheduler"""
    
    def __init__(self, optimizer, d_model, warmup_steps=4000):
        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.step_num = 0
    
    def step(self):
        self.step_num += 1
        lr = self.d_model ** (-0.5) * min(
            self.step_num ** (-0.5),
            self.step_num * self.warmup_steps ** (-1.5)
        )
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        return lr


def train_sequential_with_wandb(args):
    """Main training function with wandb integration"""
    
    # Prepare training configuration
    config = {
        'model_type': 'sequential',
        'embedding_dim': args.embedding_dim,
        'num_heads': args.num_heads,
        'num_layers': args.num_layers,
        'd_ff': args.d_ff,
        'dropout': args.dropout,
        'learning_rate': args.learning_rate,
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'early_stopping_patience': args.early_stopping_patience,
        'warmup_steps': args.warmup_steps,
        'min_interactions': args.min_interactions,
        'min_seq_length': args.min_seq_length,
        'max_seq_length': args.max_seq_length,
        'ratings_path': args.ratings_path,
        'movies_path': args.movies_path
    }
    
    # Initialize wandb with error handling
    try:
        wandb_manager = init_wandb_for_training(
            'sequential', 
            config,
            resume=False
        )
    except Exception as e:
        wandb_manager = None
    
    # Override wandb config if provided and wandb_manager exists
    if wandb_manager:
        if args.wandb_project:
            wandb_manager.config.project = args.wandb_project
        if args.wandb_entity:
            wandb_manager.config.entity = args.wandb_entity
        if args.wandb_name:
            wandb_manager.config.name = args.wandb_name
        if args.wandb_tags:
            wandb_manager.config.tags = args.wandb_tags
        if args.wandb_offline:
            wandb_manager.config.mode = 'offline'
    
    try:
        # Prepare data
        train_loader, val_loader, test_loader, metadata = prepare_sequential_data(
            args.ratings_path, args.min_interactions, args.min_seq_length, args.max_seq_length
        )
        
        # Log dataset information
        if wandb_manager:
            wandb_manager.log_dataset_info(metadata)
        
        # Create model
        num_items = metadata['num_items']
        
        model = AttentionalSequentialRecommender(
            num_items=num_items,
            embedding_dim=args.embedding_dim,
            num_heads=args.num_heads,
            num_layers=args.num_layers,
            d_ff=args.d_ff,
            max_seq_len=args.max_seq_length,
            dropout=args.dropout
        )
        
        # Validate data ranges
        sample_batch = next(iter(train_loader))
        sequences, targets = sample_batch
        
        # Check for any out-of-bounds indices
        if sequences.max().item() > num_items:
            raise ValueError(f"Sequence contains index {sequences.max().item()} > {num_items}")
        if targets.max().item() >= num_items:
            raise ValueError(f"Target contains index {targets.max().item()} >= {num_items}")
        if targets.min().item() < 0:
            raise ValueError(f"Target contains negative index {targets.min().item()}")
        
        # Log model architecture
        if wandb_manager:
            wandb_manager.log_model_architecture(model, 'sequential')
        
        # Setup device
        if args.device == 'auto':
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            device = torch.device(args.device)
        
        # Enable CUDA debugging if using CUDA
        if device.type == 'cuda':
            import os
            os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
        
        model = model.to(device)
        
        # Setup training components
        criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding tokens
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.98), eps=1e-9)
        
        # Warmup scheduler
        scheduler = WarmupScheduler(optimizer, args.embedding_dim, args.warmup_steps)
        
        # Training logger
        training_logger = WandbTrainingLogger(wandb_manager, 'sequential') if wandb_manager else None
        
        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(args.epochs):
            # Log epoch start
            current_lr = optimizer.param_groups[0]['lr']
            training_logger.log_epoch_start(epoch, args.epochs, current_lr)
            
            # Training phase
            model.train()
            train_losses = []
            train_accuracies = []
            
            for batch_idx, (sequences, targets) in enumerate(train_loader):
                sequences = sequences.to(device)
                targets = targets.to(device).squeeze()
                
                optimizer.zero_grad()
                
                logits = model(sequences)
                loss = criterion(logits, targets)
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                scheduler.step()
                
                # Calculate metrics
                batch_loss = loss.item()
                _, predicted = torch.max(logits, 1)
                accuracy = (predicted == targets).float().mean().item()
                
                train_losses.append(batch_loss)
                train_accuracies.append(accuracy)
                
                # Log batch metrics
                if batch_idx % 50 == 0:
                    # Calculate gradient norm before clipping for monitoring
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float('inf')).item()
                    
                    additional_metrics = {
                        'accuracy': accuracy,
                        'perplexity': torch.exp(loss).item(),
                        'gradient_norm': grad_norm,
                        'learning_rate': scheduler.optimizer.param_groups[0]['lr']
                    }
                    
                    training_logger.log_batch(
                        epoch, batch_idx, len(train_loader),
                        batch_loss, len(sequences), additional_metrics
                    )
            
            # Validation phase
            model.eval()
            val_losses = []
            val_accuracies = []
            
            with torch.no_grad():
                for sequences, targets in val_loader:
                    sequences = sequences.to(device)
                    targets = targets.to(device).squeeze()
                    
                    logits = model(sequences)
                    loss = criterion(logits, targets)
                    
                    _, predicted = torch.max(logits, 1)
                    accuracy = (predicted == targets).float().mean().item()
                    
                    val_losses.append(loss.item())
                    val_accuracies.append(accuracy)
            
            # Calculate epoch metrics
            train_loss = np.mean(train_losses)
            val_loss = np.mean(val_losses)
            train_accuracy = np.mean(train_accuracies)
            val_accuracy = np.mean(val_accuracies)
            
            # Log epoch summary
            train_metrics = {
                'accuracy': train_accuracy,
                'perplexity': np.exp(train_loss)
            }
            val_metrics = {
                'accuracy': val_accuracy,
                'perplexity': np.exp(val_loss)
            }
            training_logger.log_epoch_end(epoch, train_loss, val_loss, train_metrics, val_metrics)
            
            # Early stopping and model saving
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                
                # Save best model
                os.makedirs("models", exist_ok=True)
                best_model_path = "models/best_sequential_model.pt"
                
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                    'val_accuracy': val_accuracy,
                    'metadata': metadata,
                    'config': config
                }, best_model_path)
                
                # Save metadata
                with open("models/sequential_metadata.pkl", 'wb') as f:
                    pickle.dump(metadata, f)
                
                # Save model locally instead of WandB artifact
                wandb_manager.save_model_locally(
                    best_model_path,
                    'sequential',
                    metadata={
                        'epoch': epoch,
                        'val_loss': val_loss,
                        'val_accuracy': val_accuracy,
                        'train_loss': train_loss,
                        'train_accuracy': train_accuracy
                    }
                )
                
            else:
                patience_counter += 1
                if patience_counter >= args.early_stopping_patience:
                    break
        
        # Test evaluation
        model.eval()
        test_losses = []
        test_accuracies = []
        
        with torch.no_grad():
            for sequences, targets in test_loader:
                sequences = sequences.to(device)
                targets = targets.to(device).squeeze()
                
                logits = model(sequences)
                loss = criterion(logits, targets)
                
                _, predicted = torch.max(logits, 1)
                accuracy = (predicted == targets).float().mean().item()
                
                test_losses.append(loss.item())
                test_accuracies.append(accuracy)
        
        test_loss = np.mean(test_losses)
        test_accuracy = np.mean(test_accuracies)
        
        # Log final metrics
        final_metrics = {
            'test_loss': test_loss,
            'test_accuracy': test_accuracy,
            'test_perplexity': np.exp(test_loss),
            'best_val_loss': best_val_loss,
            'total_epochs': epoch + 1,
            'total_parameters': sum(p.numel() for p in model.parameters()),
            'model_size_mb': sum(p.numel() * p.element_size() for p in model.parameters()) / (1024**2)
        }
        
        if wandb_manager:
            wandb_manager.log_metrics({f'final/{k}': v for k, v in final_metrics.items()})
        
        
        return model, final_metrics
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise
    finally:
        # Always finish wandb run with error handling
        if wandb_manager:
            try:
                wandb_manager.finish()
            except Exception:
                pass
        
        # Also try to finish regular wandb
        try:
            import wandb
            if wandb.run is not None:
                wandb.finish()
        except Exception:
            pass


def main():
    """Main function"""
    setup_logging()
    args = parse_args()
    
    
    # Check if datasets exist
    if not os.path.exists(args.ratings_path):
        logger.error(f"Ratings file not found: {args.ratings_path}")
        return
    
    # Train model
    model, metrics = train_sequential_with_wandb(args)
    


if __name__ == "__main__":
    main()