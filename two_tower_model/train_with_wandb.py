#!/usr/bin/env python3
"""
Enhanced Two-Tower Model Training with Full Wandb Integration
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


class TwoTowerDataset(Dataset):
    """Dataset for Two-Tower model training"""
    
    def __init__(self, users, items, ratings):
        self.users = torch.LongTensor(users)
        self.items = torch.LongTensor(items)
        self.ratings = torch.FloatTensor(ratings)
    
    def __len__(self):
        return len(self.users)
    
    def __getitem__(self, idx):
        return self.users[idx], self.items[idx], self.ratings[idx]


class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism"""
    
    def __init__(self, d_model, num_heads=8, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        
        self._init_weights()
    
    def _init_weights(self):
        for module in [self.w_q, self.w_k, self.w_v, self.w_o]:
            nn.init.xavier_uniform_(module.weight)
    
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        # Apply linear transformations
        Q = self.w_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        attn_output = torch.matmul(attn_weights, V)
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model
        )
        
        output = self.w_o(attn_output)
        
        # Residual connection and layer norm
        return self.layer_norm(query + self.dropout(output))


class EnhancedTwoTowerModel(nn.Module):
    """
    Enhanced Two-Tower model with cross-attention and multi-task learning
    """
    
    def __init__(self, num_users, num_items, embedding_dim=256, hidden_dim=512, 
                 num_heads=8, num_layers=4, dropout=0.1):
        super(EnhancedTwoTowerModel, self).__init__()
        
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        
        # User tower
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.user_tower = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embedding_dim)
        )
        
        # Item tower
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        self.item_tower = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embedding_dim)
        )
        
        # Cross-attention layers
        self.cross_attention_layers = nn.ModuleList([
            MultiHeadAttention(embedding_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])
        
        # Final prediction layers
        self.prediction_head = nn.Sequential(
            nn.Linear(embedding_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Temperature parameter for calibration
        self.temperature = nn.Parameter(torch.ones(1))
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights"""
        for module in self.modules():
            if isinstance(module, nn.Embedding):
                nn.init.xavier_normal_(module.weight)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, user_ids, item_ids):
        # Get embeddings
        user_emb = self.user_embedding(user_ids)
        item_emb = self.item_embedding(item_ids)
        
        # Process through towers
        user_repr = self.user_tower(user_emb)
        item_repr = self.item_tower(item_emb)
        
        # Add sequence dimension for attention
        user_repr = user_repr.unsqueeze(1)  # [batch, 1, dim]
        item_repr = item_repr.unsqueeze(1)  # [batch, 1, dim]
        
        # Apply cross-attention layers
        for attn_layer in self.cross_attention_layers:
            # User attends to item
            user_repr = attn_layer(user_repr, item_repr, item_repr)
            # Item attends to user
            item_repr = attn_layer(item_repr, user_repr, user_repr)
        
        # Remove sequence dimension
        user_repr = user_repr.squeeze(1)
        item_repr = item_repr.squeeze(1)
        
        # Combine representations
        combined = torch.cat([user_repr, item_repr], dim=-1)
        
        # Final prediction
        logits = self.prediction_head(combined)
        
        # Apply temperature scaling and sigmoid
        prediction = torch.sigmoid(logits / self.temperature)
        
        return prediction.squeeze() * 5.0  # Scale to rating range


def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f"two_tower_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
            logging.StreamHandler(sys.stdout)
        ]
    )


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train Two-Tower Model with Wandb')
    
    # Data arguments
    parser.add_argument('--ratings-path', type=str, default='../movies/cinesync/ml-32m/ratings.csv',
                       help='Path to ratings CSV file')
    parser.add_argument('--min-interactions', type=int, default=20,
                       help='Minimum interactions per user/item')
    
    # Model arguments
    parser.add_argument('--embedding-dim', type=int, default=256,
                       help='Embedding dimension')
    parser.add_argument('--hidden-dim', type=int, default=512,
                       help='Hidden layer dimension')
    parser.add_argument('--num-heads', type=int, default=8,
                       help='Number of attention heads')
    parser.add_argument('--num-layers', type=int, default=4,
                       help='Number of attention layers')
    parser.add_argument('--dropout', type=float, default=0.1,
                       help='Dropout rate')
    
    # Training arguments
    parser.add_argument('--learning-rate', type=float, default=0.0001,
                       help='Learning rate')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of epochs')
    parser.add_argument('--early-stopping-patience', type=int, default=10,
                       help='Early stopping patience')
    parser.add_argument('--gradient-clipping', action='store_true',
                       help='Enable gradient clipping')
    
    # Wandb arguments
    parser.add_argument('--wandb-project', type=str, default='cinesync-v2-two-tower',
                       help='Wandb project name')
    parser.add_argument('--wandb-entity', type=str, default=None,
                       help='Wandb entity')
    parser.add_argument('--wandb-name', type=str, default=None,
                       help='Wandb run name')
    parser.add_argument('--wandb-tags', type=str, nargs='+', 
                       default=['two-tower', 'attention', 'production'],
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


def prepare_data(ratings_path, min_interactions=20, test_size=0.2, val_size=0.1):
    """
    Prepare data for Two-Tower training
    
    Returns:
        train_loader, val_loader, test_loader, metadata
    """
    logger.info("Loading and preparing Two-Tower data...")
    
    # Load ratings
    ratings_df = pd.read_csv(ratings_path)
    logger.info(f"Loaded {len(ratings_df)} ratings")
    
    # Filter users and items with minimum interactions
    user_counts = ratings_df['userId'].value_counts()
    item_counts = ratings_df['movieId'].value_counts()
    
    valid_users = user_counts[user_counts >= min_interactions].index
    valid_items = item_counts[item_counts >= min_interactions].index
    
    ratings_df = ratings_df[
        (ratings_df['userId'].isin(valid_users)) & 
        (ratings_df['movieId'].isin(valid_items))
    ]
    
    logger.info(f"After filtering: {len(ratings_df)} ratings, "
                f"{ratings_df['userId'].nunique()} users, "
                f"{ratings_df['movieId'].nunique()} items")
    
    # Create label encoders for contiguous indices
    user_encoder = LabelEncoder()
    item_encoder = LabelEncoder()
    
    ratings_df['user_idx'] = user_encoder.fit_transform(ratings_df['userId'])
    ratings_df['item_idx'] = item_encoder.fit_transform(ratings_df['movieId'])
    
    # Normalize ratings to 0-1 range
    min_rating = ratings_df['rating'].min()
    max_rating = ratings_df['rating'].max()
    ratings_df['rating_norm'] = (ratings_df['rating'] - min_rating) / (max_rating - min_rating)
    
    # Split data
    train_val_df, test_df = train_test_split(ratings_df, test_size=test_size, random_state=42)
    train_df, val_df = train_test_split(train_val_df, test_size=val_size/(1-test_size), random_state=42)
    
    logger.info(f"Split: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
    
    # Create datasets
    train_dataset = TwoTowerDataset(
        train_df['user_idx'].values,
        train_df['item_idx'].values,
        train_df['rating_norm'].values
    )
    val_dataset = TwoTowerDataset(
        val_df['user_idx'].values,
        val_df['item_idx'].values,
        val_df['rating_norm'].values
    )
    test_dataset = TwoTowerDataset(
        test_df['user_idx'].values,
        test_df['item_idx'].values,
        test_df['rating_norm'].values
    )
    
    # Create data loaders (smaller batch size for Two-Tower due to complexity)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4)
    
    # Metadata
    metadata = {
        'num_users': len(user_encoder.classes_),
        'num_items': len(item_encoder.classes_),
        'num_ratings': len(ratings_df),
        'rating_range': (min_rating, max_rating),
        'sparsity': 1 - (len(ratings_df) / (len(user_encoder.classes_) * len(item_encoder.classes_))),
        'user_encoder': user_encoder,
        'item_encoder': item_encoder,
        'train_size': len(train_df),
        'val_size': len(val_df),
        'test_size': len(test_df)
    }
    
    return train_loader, val_loader, test_loader, metadata


def train_two_tower_with_wandb(args):
    """Main training function with wandb integration"""
    
    # Prepare training configuration
    config = {
        'model_type': 'two_tower',
        'embedding_dim': args.embedding_dim,
        'hidden_dim': args.hidden_dim,
        'num_heads': args.num_heads,
        'num_layers': args.num_layers,
        'dropout': args.dropout,
        'learning_rate': args.learning_rate,
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'early_stopping_patience': args.early_stopping_patience,
        'gradient_clipping': args.gradient_clipping,
        'min_interactions': args.min_interactions,
        'ratings_path': args.ratings_path
    }
    
    # Initialize wandb
    wandb_manager = init_wandb_for_training(
        'two_tower', 
        config,
        resume=False
    )
    
    # Override wandb config if provided
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
        train_loader, val_loader, test_loader, metadata = prepare_data(
            args.ratings_path, args.min_interactions
        )
        
        # Log dataset information
        wandb_manager.log_dataset_info(metadata)
        
        # Create model
        model = EnhancedTwoTowerModel(
            num_users=metadata['num_users'],
            num_items=metadata['num_items'],
            embedding_dim=args.embedding_dim,
            hidden_dim=args.hidden_dim,
            num_heads=args.num_heads,
            num_layers=args.num_layers,
            dropout=args.dropout
        )
        
        # Log model architecture
        wandb_manager.log_model_architecture(model, 'two_tower')
        
        # Setup device
        if args.device == 'auto':
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            device = torch.device(args.device)
        
        logger.info(f"Using device: {device}")
        model = model.to(device)
        
        # Setup training components
        criterion = nn.MSELoss()
        optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
        
        # Training logger
        training_logger = WandbTrainingLogger(wandb_manager, 'two_tower')
        
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
            train_rmses = []
            
            for batch_idx, (user_ids, item_ids, ratings) in enumerate(train_loader):
                user_ids = user_ids.to(device)
                item_ids = item_ids.to(device)
                ratings = ratings.to(device)
                
                optimizer.zero_grad()
                
                predictions = model(user_ids, item_ids)
                loss = criterion(predictions, ratings)
                loss.backward()
                
                # Gradient clipping
                if args.gradient_clipping:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                # Calculate metrics
                batch_loss = loss.item()
                batch_rmse = torch.sqrt(loss).item()
                
                train_losses.append(batch_loss)
                train_rmses.append(batch_rmse)
                
                # Log batch metrics
                if batch_idx % 50 == 0:  # More frequent logging for complex model
                    additional_metrics = {
                        'rmse': batch_rmse,
                        'temperature': model.temperature.item(),
                        'gradient_norm': torch.nn.utils.clip_grad_norm_(model.parameters(), float('inf')).item()
                    }
                    
                    training_logger.log_batch(
                        epoch, batch_idx, len(train_loader),
                        batch_loss, len(user_ids), additional_metrics
                    )
            
            # Validation phase
            model.eval()
            val_losses = []
            val_rmses = []
            
            with torch.no_grad():
                for user_ids, item_ids, ratings in val_loader:
                    user_ids = user_ids.to(device)
                    item_ids = item_ids.to(device)
                    ratings = ratings.to(device)
                    
                    predictions = model(user_ids, item_ids)
                    loss = criterion(predictions, ratings)
                    
                    val_losses.append(loss.item())
                    val_rmses.append(torch.sqrt(loss).item())
            
            # Calculate epoch metrics
            train_loss = np.mean(train_losses)
            val_loss = np.mean(val_losses)
            train_rmse = np.mean(train_rmses)
            val_rmse = np.mean(val_rmses)
            
            # Log epoch summary
            train_metrics = {'rmse': train_rmse, 'temperature': model.temperature.item()}
            val_metrics = {'rmse': val_rmse}
            training_logger.log_epoch_end(epoch, train_loss, val_loss, train_metrics, val_metrics)
            
            # Learning rate scheduling
            scheduler.step(val_loss)
            
            # Early stopping and model saving
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                
                # Save best model
                os.makedirs("models", exist_ok=True)
                best_model_path = "models/best_two_tower_model.pt"
                
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                    'val_rmse': val_rmse,
                    'metadata': metadata,
                    'config': config
                }, best_model_path)
                
                # Save metadata
                with open("models/two_tower_metadata.pkl", 'wb') as f:
                    pickle.dump(metadata, f)
                
                # Save as wandb artifact
                wandb_manager.save_model_artifact(
                    best_model_path,
                    'two_tower',
                    metadata={
                        'epoch': epoch,
                        'val_loss': val_loss,
                        'val_rmse': val_rmse,
                        'train_loss': train_loss,
                        'train_rmse': train_rmse,
                        'temperature': model.temperature.item()
                    }
                )
                
            else:
                patience_counter += 1
                if patience_counter >= args.early_stopping_patience:
                    logger.info(f"Early stopping triggered after {epoch + 1} epochs")
                    break
        
        # Test evaluation
        logger.info("Evaluating on test set...")
        model.eval()
        test_losses = []
        test_rmses = []
        
        with torch.no_grad():
            for user_ids, item_ids, ratings in test_loader:
                user_ids = user_ids.to(device)
                item_ids = item_ids.to(device)
                ratings = ratings.to(device)
                
                predictions = model(user_ids, item_ids)
                loss = criterion(predictions, ratings)
                
                test_losses.append(loss.item())
                test_rmses.append(torch.sqrt(loss).item())
        
        test_loss = np.mean(test_losses)
        test_rmse = np.mean(test_rmses)
        
        # Log final metrics
        final_metrics = {
            'test_loss': test_loss,
            'test_rmse': test_rmse,
            'best_val_loss': best_val_loss,
            'total_epochs': epoch + 1,
            'total_parameters': sum(p.numel() for p in model.parameters()),
            'model_size_mb': sum(p.numel() * p.element_size() for p in model.parameters()) / (1024**2),
            'final_temperature': model.temperature.item()
        }
        
        wandb_manager.log_metrics({f'final/{k}': v for k, v in final_metrics.items()})
        
        logger.info(f"Training completed! Test RMSE: {test_rmse:.4f}")
        
        return model, final_metrics
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise
    finally:
        # Always finish wandb run
        wandb_manager.finish()


def main():
    """Main function"""
    setup_logging()
    args = parse_args()
    
    logger.info("Starting Two-Tower model training with Wandb integration")
    logger.info(f"Configuration: {vars(args)}")
    
    # Check if datasets exist
    if not os.path.exists(args.ratings_path):
        logger.error(f"Ratings file not found: {args.ratings_path}")
        return
    
    # Train model
    model, metrics = train_two_tower_with_wandb(args)
    
    logger.info("Training completed successfully!")
    logger.info(f"Final metrics: {metrics}")


if __name__ == "__main__":
    main()