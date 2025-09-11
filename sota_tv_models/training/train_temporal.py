"""
Training script for Temporal Attention TV Model (TAT-TV)
Optimized for RTX 4090 with temporal pattern learning
"""

import os
import sys
import argparse
import logging
from pathlib import Path
import json
import time
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
import wandb
import numpy as np
from transformers import get_cosine_schedule_with_warmup
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tqdm.auto import tqdm
from datetime import datetime, timedelta
import random

# Add parent directory to path to import models
sys.path.append(str(Path(__file__).parent.parent))
from models.temporal_attention import TemporalAttentionTVModel, TemporalLoss, get_temporal_config

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TemporalTVDataset(Dataset):
    """Dataset for temporal TV model training"""
    
    def __init__(self, 
                 data_file: str,
                 max_seq_length: int = 100,
                 temporal_window_days: int = 365):
        
        with open(data_file) as f:
            self.data = json.load(f)
        
        self.max_seq_length = max_seq_length
        self.temporal_window_days = temporal_window_days
        
        # Create temporal sequences for each show
        self.temporal_sequences = self._create_temporal_sequences()
        
        logger.info(f"Created {len(self.temporal_sequences)} temporal sequences")
    
    def _create_temporal_sequences(self):
        """Create temporal sequences from show data"""
        sequences = []
        
        for show in self.data:
            # Create synthetic temporal data (in practice, this would come from real data)
            show_id = show['id']
            
            # Generate temporal sequence (e.g., popularity over time)
            base_date = datetime(2020, 1, 1)
            
            for seq_start in range(0, 365, 30):  # Monthly sequences
                sequence = {
                    'show_id': show_id,
                    'show_data': show,
                    'timestamps': [],
                    'popularity_scores': [],
                    'seasonal_patterns': [],
                    'trend_values': []
                }
                
                # Generate sequence data
                seq_length = min(self.max_seq_length, 30)  # 30 days per sequence
                
                for day in range(seq_length):
                    timestamp = base_date + timedelta(days=seq_start + day)
                    timestamp_unix = timestamp.timestamp()
                    
                    # Synthetic popularity with seasonal patterns
                    day_of_year = timestamp.timetuple().tm_yday
                    seasonal_score = 0.5 + 0.3 * np.sin(2 * np.pi * day_of_year / 365)
                    trend_score = 0.1 * (day / seq_length)  # Slight upward trend
                    
                    # Add show-specific factors
                    base_popularity = show['numerical_features'][0] if show['numerical_features'] else 0.5
                    noise = np.random.normal(0, 0.1)
                    
                    popularity = base_popularity + seasonal_score + trend_score + noise
                    popularity = np.clip(popularity, 0, 1)
                    
                    sequence['timestamps'].append(timestamp_unix)
                    sequence['popularity_scores'].append(popularity)
                    sequence['seasonal_patterns'].append(seasonal_score)
                    sequence['trend_values'].append(trend_score)
                
                if len(sequence['timestamps']) > 0:
                    sequences.append(sequence)
        
        return sequences
    
    def __len__(self):
        return len(self.temporal_sequences)
    
    def __getitem__(self, idx):
        sequence = self.temporal_sequences[idx]
        show_data = sequence['show_data']
        
        # Prepare inputs
        show_ids = torch.tensor([sequence['show_id']], dtype=torch.long)
        timestamps = torch.tensor(sequence['timestamps'], dtype=torch.float)
        
        # Genre and network IDs
        genre_ids = torch.tensor(
            show_data['categorical_features'].get('genres', [0])[:5], 
            dtype=torch.long
        )
        if len(genre_ids) < 5:
            genre_ids = torch.cat([
                genre_ids, 
                torch.zeros(5 - len(genre_ids), dtype=torch.long)
            ])
        
        network_ids = torch.tensor(
            show_data['categorical_features'].get('networks', [0])[:3], 
            dtype=torch.long
        )
        if len(network_ids) < 3:
            network_ids = torch.cat([
                network_ids, 
                torch.zeros(3 - len(network_ids), dtype=torch.long)
            ])
        
        # Targets
        popularity_targets = torch.tensor(sequence['popularity_scores'], dtype=torch.float)
        trend_targets = torch.tensor(sequence['trend_values'], dtype=torch.float)
        
        # Seasonal targets (simplified)
        seasonal_7_targets = torch.tensor(sequence['seasonal_patterns'], dtype=torch.float)
        seasonal_30_targets = torch.tensor(sequence['seasonal_patterns'], dtype=torch.float)
        seasonal_365_targets = torch.tensor(sequence['seasonal_patterns'], dtype=torch.float)
        
        # Padding/truncation to max_seq_length
        seq_len = len(timestamps)
        if seq_len < self.max_seq_length:
            pad_len = self.max_seq_length - seq_len
            timestamps = torch.cat([timestamps, torch.zeros(pad_len)])
            popularity_targets = torch.cat([popularity_targets, torch.zeros(pad_len)])
            trend_targets = torch.cat([trend_targets, torch.zeros(pad_len)])
            seasonal_7_targets = torch.cat([seasonal_7_targets, torch.zeros(pad_len)])
            seasonal_30_targets = torch.cat([seasonal_30_targets, torch.zeros(pad_len)])
            seasonal_365_targets = torch.cat([seasonal_365_targets, torch.zeros(pad_len)])
            
            # Create mask
            mask = torch.cat([torch.ones(seq_len), torch.zeros(pad_len)])
        else:
            timestamps = timestamps[:self.max_seq_length]
            popularity_targets = popularity_targets[:self.max_seq_length]
            trend_targets = trend_targets[:self.max_seq_length]
            seasonal_7_targets = seasonal_7_targets[:self.max_seq_length]
            seasonal_30_targets = seasonal_30_targets[:self.max_seq_length]
            seasonal_365_targets = seasonal_365_targets[:self.max_seq_length]
            mask = torch.ones(self.max_seq_length)
        
        return {
            'show_ids': show_ids,
            'timestamps': timestamps.unsqueeze(0),  # Add batch dim
            'genre_ids': genre_ids.unsqueeze(0),  # Add batch dim
            'network_ids': network_ids.unsqueeze(0),  # Add batch dim
            'mask': mask.unsqueeze(0),  # Add batch dim
            'popularity_targets': popularity_targets.mean(),  # Sequence-level target
            'trend_targets': trend_targets.mean(),
            'seasonal_7_targets': seasonal_7_targets.mean(),
            'seasonal_30_targets': seasonal_30_targets.mean(),
            'seasonal_365_targets': seasonal_365_targets.mean(),
        }

def collate_temporal_fn(batch):
    """Custom collate function for temporal data"""
    
    collated = {}
    
    # Stack tensors
    for key in batch[0].keys():
        if key.endswith('_targets'):
            collated[key] = torch.stack([item[key] for item in batch])
        else:
            collated[key] = torch.cat([item[key] for item in batch], dim=0)
    
    return collated

class TemporalTrainer:
    """Trainer for temporal attention TV model"""
    
    def __init__(self, 
                 model: TemporalAttentionTVModel,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 config: dict,
                 device: torch.device):
        
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        
        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config.get('weight_decay', 1e-5),
            eps=1e-8
        )
        
        # Scheduler
        total_steps = len(train_loader) * config['epochs'] // config.get('gradient_accumulation_steps', 1)
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=config.get('warmup_steps', 1000),
            num_training_steps=total_steps
        )
        
        # Loss function
        self.criterion = TemporalLoss(
            popularity_weight=config.get('popularity_weight', 1.0),
            trend_weight=config.get('trend_weight', 0.5),
            seasonal_weight=config.get('seasonal_weight', 0.3),
            forecast_weight=config.get('forecast_weight', 0.8)
        )
        
        # Mixed precision
        self.scaler = GradScaler() if config.get('use_mixed_precision', True) else None
        
        # Metrics
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        progress_bar = tqdm(self.train_loader, desc="Training Temporal")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move to device
            for key, value in batch.items():
                batch[key] = value.to(self.device)
            
            # Forward pass
            if self.scaler:
                with autocast():
                    outputs = self.model(
                        show_ids=batch['show_ids'],
                        timestamps=batch['timestamps'],
                        genre_ids=batch['genre_ids'],
                        network_ids=batch['network_ids'],
                        mask=batch['mask']
                    )
                    
                    # Prepare targets
                    targets = {
                        'popularity': batch['popularity_targets'],
                        'trend': batch['trend_targets'],
                        'seasonal_7': batch['seasonal_7_targets'],
                        'seasonal_30': batch['seasonal_30_targets'],
                        'seasonal_365': batch['seasonal_365_targets']
                    }
                    
                    loss_dict = self.criterion(outputs, targets)
                    loss = loss_dict['total_loss']
                
                # Backward pass
                self.scaler.scale(loss).backward()
                
                if (batch_idx + 1) % self.config.get('gradient_accumulation_steps', 1) == 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.get('max_grad_norm', 1.0))
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
                    self.scheduler.step()
            
            else:
                outputs = self.model(
                    show_ids=batch['show_ids'],
                    timestamps=batch['timestamps'],
                    genre_ids=batch['genre_ids'],
                    network_ids=batch['network_ids'],
                    mask=batch['mask']
                )
                
                targets = {
                    'popularity': batch['popularity_targets'],
                    'trend': batch['trend_targets'],
                    'seasonal_7': batch['seasonal_7_targets'],
                    'seasonal_30': batch['seasonal_30_targets'],
                    'seasonal_365': batch['seasonal_365_targets']
                }
                
                loss_dict = self.criterion(outputs, targets)
                loss = loss_dict['total_loss']
                
                loss.backward()
                
                if (batch_idx + 1) % self.config.get('gradient_accumulation_steps', 1) == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.get('max_grad_norm', 1.0))
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    self.scheduler.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'avg_loss': f"{total_loss/num_batches:.4f}"
            })
            
            # Log to wandb
            if batch_idx % 20 == 0:
                log_dict = {
                    'temporal_train_loss': loss.item(),
                    'learning_rate': self.scheduler.get_last_lr()[0],
                }
                
                # Add individual loss components
                for key, value in loss_dict.items():
                    if key != 'total_loss':
                        log_dict[f'temporal_{key}'] = value.item()
                
                wandb.log(log_dict)
        
        return total_loss / num_batches
    
    def validate(self):
        """Validate the model"""
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                # Move to device
                for key, value in batch.items():
                    batch[key] = value.to(self.device)
                
                outputs = self.model(
                    show_ids=batch['show_ids'],
                    timestamps=batch['timestamps'],
                    genre_ids=batch['genre_ids'],
                    network_ids=batch['network_ids'],
                    mask=batch['mask']
                )
                
                targets = {
                    'popularity': batch['popularity_targets'],
                    'trend': batch['trend_targets'],
                    'seasonal_7': batch['seasonal_7_targets'],
                    'seasonal_30': batch['seasonal_30_targets'],
                    'seasonal_365': batch['seasonal_365_targets']
                }
                
                loss_dict = self.criterion(outputs, targets)
                loss = loss_dict['total_loss']
                
                total_loss += loss.item()
                num_batches += 1
                
                # Collect predictions for metrics
                if 'popularity_prediction' in outputs:
                    all_predictions.extend(outputs['popularity_prediction'].cpu().numpy())
                    all_targets.extend(batch['popularity_targets'].cpu().numpy())
        
        avg_val_loss = total_loss / num_batches
        
        # Calculate additional metrics
        if all_predictions and all_targets:
            mse = mean_squared_error(all_targets, all_predictions)
            mae = mean_absolute_error(all_targets, all_predictions)
        else:
            mse, mae = 0, 0
        
        return avg_val_loss, mse, mae
    
    def train(self):
        """Main training loop"""
        logger.info("Starting temporal model training...")
        
        for epoch in range(self.config['epochs']):
            logger.info(f"Epoch {epoch + 1}/{self.config['epochs']}")
            
            # Train
            train_loss = self.train_epoch()
            
            # Validate
            val_loss, val_mse, val_mae = self.validate()
            
            logger.info(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            logger.info(f"Val MSE: {val_mse:.4f}, Val MAE: {val_mae:.4f}")
            
            # Log to wandb
            wandb.log({
                'temporal_epoch': epoch,
                'temporal_train_loss_epoch': train_loss,
                'temporal_val_loss_epoch': val_loss,
                'temporal_val_mse': val_mse,
                'temporal_val_mae': val_mae
            })
            
            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                self.save_model(f"best_temporal_tv_model.pt")
                logger.info(f"New best model saved with val_loss: {val_loss:.4f}")
            else:
                self.patience_counter += 1
            
            # Early stopping
            if self.patience_counter >= self.config.get('patience', 15):
                logger.info(f"Early stopping after {epoch + 1} epochs")
                break
        
        logger.info("Temporal model training completed!")
    
    def save_model(self, filename):
        """Save model checkpoint"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': self.config,
            'best_val_loss': self.best_val_loss
        }
        torch.save(checkpoint, filename)

def main():
    parser = argparse.ArgumentParser(description='Train Temporal Attention TV Model')
    parser.add_argument('--data_path', type=str, default='./processed_data', help='Path to processed data')
    parser.add_argument('--output_dir', type=str, default='./models', help='Output directory for models')
    parser.add_argument('--config_file', type=str, help='Config file path')
    parser.add_argument('--wandb_project', type=str, default='sota-tv-models', help='Wandb project name')
    parser.add_argument('--wandb_run_name', type=str, default='temporal-attention', help='Wandb run name')
    parser.add_argument('--resume_from', type=str, help='Resume training from checkpoint')
    
    args = parser.parse_args()
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Load config
    config = get_temporal_config()
    if args.config_file and os.path.exists(args.config_file):
        with open(args.config_file) as f:
            config.update(json.load(f).get('temporal_config', {}))
    
    # Add training specific config
    config.update({
        'popularity_weight': 1.0,
        'trend_weight': 0.5,
        'seasonal_weight': 0.3,
        'forecast_weight': 0.8
    })
    
    # Initialize wandb
    wandb.init(
        project=args.wandb_project,
        name=args.wandb_run_name,
        config=config
    )
    
    # Create datasets
    train_dataset = TemporalTVDataset(
        data_file=os.path.join(args.data_path, 'train_data.json'),
        max_seq_length=config['max_seq_length'],
        temporal_window_days=365
    )
    
    val_dataset = TemporalTVDataset(
        data_file=os.path.join(args.data_path, 'val_data.json'),
        max_seq_length=config['max_seq_length'],
        temporal_window_days=365
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=4,
        collate_fn=collate_temporal_fn,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=4,
        collate_fn=collate_temporal_fn,
        pin_memory=True
    )
    
    # Load vocabulary sizes
    with open(os.path.join(args.data_path, 'metadata.json')) as f:
        metadata = json.load(f)
    
    vocab_sizes = metadata['vocab_sizes']
    vocab_sizes['shows'] = metadata['num_samples']['train']  # Number of unique shows
    
    # Create model
    model = TemporalAttentionTVModel(
        vocab_sizes=vocab_sizes,
        d_model=config['d_model'],
        n_heads=config['n_heads'],
        num_layers=config['num_layers'],
        d_ff=config['d_ff'],
        max_seq_length=config['max_seq_length'],
        dropout=config['dropout'],
        seasonal_periods=config['seasonal_periods'],
        use_seasonal_decomposition=config['use_seasonal_decomposition']
    )
    
    logger.info(f"Temporal Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Resume from checkpoint if specified
    if args.resume_from and os.path.exists(args.resume_from):
        checkpoint = torch.load(args.resume_from, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Resumed from checkpoint: {args.resume_from}")
    
    # Create trainer and train
    trainer = TemporalTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device
    )
    
    trainer.train()
    
    # Save final model
    os.makedirs(args.output_dir, exist_ok=True)
    trainer.save_model(os.path.join(args.output_dir, 'final_temporal_tv_model.pt'))
    
    wandb.finish()

if __name__ == "__main__":
    main()