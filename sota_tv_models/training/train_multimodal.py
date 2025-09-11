"""
Training script for Multimodal Transformer TV Model (MTTV)
Optimized for RTX 4090 with mixed precision and gradient checkpointing
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
from transformers import RobertaTokenizer, get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from tqdm.auto import tqdm

# Add parent directory to path to import models
sys.path.append(str(Path(__file__).parent.parent))
from models.multimodal_transformer import MultimodalTransformerTV, TVRecommendationLoss, get_model_config

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TVShowDataset(Dataset):
    """Dataset for TV show multimodal training"""
    
    def __init__(self, 
                 data_file: str,
                 tokenizer: RobertaTokenizer,
                 max_length: int = 512,
                 augment_data: bool = True):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.augment_data = augment_data
        
        # Load data (would be replaced with actual data loading)
        self.data = self._load_data(data_file)
        
        logger.info(f"Loaded {len(self.data)} TV show samples")
    
    def _load_data(self, data_file: str):
        """Load and preprocess TV show data"""
        # This would load actual TV show data from your datasets
        # For now, creating dummy data structure
        dummy_data = []
        for i in range(1000):  # Dummy 1000 shows
            dummy_data.append({
                'id': i,
                'title': f"Show {i}",
                'overview': f"This is an amazing TV show number {i} with great storyline and characters.",
                'genres': [1, 2, 3],  # Genre IDs
                'actors': [10, 20, 30, 40],  # Actor IDs
                'network': 5,  # Network ID
                'creators': [100, 101],  # Creator IDs
                'vote_average': 7.5 + (i % 3),
                'vote_count': 1000 + i * 10,
                'num_seasons': (i % 5) + 1,
                'num_episodes': ((i % 5) + 1) * 10,
                'first_air_date': f"2020-{(i % 12) + 1:02d}-01",
                'status': i % 3,  # 0: ended, 1: ongoing, 2: cancelled
                'runtime': 45 + (i % 30),
                'popularity': 50.0 + (i % 100),
                'language': i % 5,
            })
        return dummy_data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Text features
        text = f"{item['title']} [SEP] {item['overview']}"
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Categorical features
        categorical_features = {
            'genres': torch.tensor(item['genres'][:5], dtype=torch.long),  # Limit to 5 genres
            'actors': torch.tensor(item['actors'][:10], dtype=torch.long),  # Limit to 10 actors
            'network': torch.tensor([item['network']], dtype=torch.long),
            'creators': torch.tensor(item['creators'][:3], dtype=torch.long),  # Limit to 3 creators
            'status': torch.tensor([item['status']], dtype=torch.long),
            'language': torch.tensor([item['language']], dtype=torch.long),
        }
        
        # Numerical features
        numerical_features = torch.tensor([
            item['vote_average'],
            item['vote_count'] / 10000.0,  # Normalize
            item['num_seasons'],
            item['num_episodes'] / 100.0,  # Normalize
            item['runtime'] / 60.0,  # Normalize to hours
            item['popularity'] / 100.0,  # Normalize
            2024 - int(item['first_air_date'][:4]),  # Years since first aired
            len(item['genres']),
            len(item['actors']),
            len(item['creators'])
        ], dtype=torch.float)
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'categorical_features': categorical_features,
            'numerical_features': numerical_features,
            'show_id': item['id']
        }

def collate_fn(batch):
    """Custom collate function for batching"""
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    numerical_features = torch.stack([item['numerical_features'] for item in batch])
    show_ids = torch.tensor([item['show_id'] for item in batch])
    
    # Handle categorical features with padding
    categorical_features = {}
    for key in batch[0]['categorical_features'].keys():
        max_len = max(len(item['categorical_features'][key]) for item in batch)
        padded_features = []
        for item in batch:
            feature = item['categorical_features'][key]
            if len(feature) < max_len:
                # Pad with zeros
                padding = torch.zeros(max_len - len(feature), dtype=torch.long)
                feature = torch.cat([feature, padding])
            padded_features.append(feature)
        categorical_features[key] = torch.stack(padded_features)
    
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'categorical_features': categorical_features,
        'numerical_features': numerical_features,
        'show_ids': show_ids
    }

class MultimodalTrainer:
    """Trainer for multimodal TV transformer"""
    
    def __init__(self, 
                 model: MultimodalTransformerTV,
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
        total_steps = len(train_loader) * config['epochs'] // config.get('accumulation_steps', 1)
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=config.get('warmup_steps', 1000),
            num_training_steps=total_steps
        )
        
        # Loss function
        self.criterion = TVRecommendationLoss(
            contrastive_weight=config.get('contrastive_weight', 1.0),
            recommendation_weight=config.get('recommendation_weight', 1.0)
        )
        
        # Mixed precision
        self.scaler = GradScaler() if config.get('use_mixed_precision', True) else None
        
        # Metrics
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        
    def create_training_pairs(self, batch):
        """Create positive and negative pairs for contrastive learning"""
        batch_size = batch['input_ids'].size(0)
        
        # For simplicity, use in-batch negatives
        # In practice, you'd want more sophisticated negative sampling
        
        # Positive pairs: same show with different augmentations (simulated here)
        positive_indices = torch.arange(batch_size)
        
        # Negative pairs: different shows in the batch
        negative_indices = torch.roll(torch.arange(batch_size), 1)
        
        # Create labels (1 for positive pairs, 0 for negative pairs)
        positive_labels = torch.ones(batch_size)
        negative_labels = torch.zeros(batch_size)
        
        return {
            'positive_indices': positive_indices,
            'negative_indices': negative_indices,
            'positive_labels': positive_labels,
            'negative_labels': negative_labels
        }
    
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        progress_bar = tqdm(self.train_loader, desc="Training")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move to device
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            numerical_features = batch['numerical_features'].to(self.device)
            
            categorical_features = {}
            for key, value in batch['categorical_features'].items():
                categorical_features[key] = value.to(self.device)
            
            # Create training pairs
            pairs = self.create_training_pairs(batch)
            
            # Forward pass with mixed precision
            if self.scaler:
                with autocast():
                    # Anchor samples
                    anchor_outputs = self.model(
                        input_ids, attention_mask,
                        categorical_features, numerical_features
                    )
                    
                    # Positive samples (same as anchor for now)
                    positive_outputs = self.model(
                        input_ids, attention_mask,
                        categorical_features, numerical_features
                    )
                    
                    # Negative samples (rolled batch)
                    neg_indices = pairs['negative_indices']
                    negative_outputs = self.model(
                        input_ids[neg_indices], attention_mask[neg_indices],
                        {k: v[neg_indices] for k, v in categorical_features.items()},
                        numerical_features[neg_indices]
                    )
                    
                    # Compute loss
                    outputs = {
                        'query_embedding': anchor_outputs['query_embedding'],
                        'target_embedding': positive_outputs['query_embedding'],
                        'recommendation_score': torch.cosine_similarity(
                            anchor_outputs['query_embedding'],
                            positive_outputs['query_embedding'],
                            dim=1
                        ).unsqueeze(1)
                    }
                    
                    labels = pairs['positive_labels'].to(self.device)
                    negative_embeddings = negative_outputs['query_embedding']
                    
                    loss_dict = self.criterion(outputs, labels, negative_embeddings)
                    loss = loss_dict['total_loss']
                
                # Backward pass
                self.scaler.scale(loss).backward()
                
                if (batch_idx + 1) % self.config.get('accumulation_steps', 1) == 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.get('max_grad_norm', 1.0))
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
                    self.scheduler.step()
            
            else:
                # Regular training without mixed precision
                anchor_outputs = self.model(
                    input_ids, attention_mask,
                    categorical_features, numerical_features
                )
                
                positive_outputs = self.model(
                    input_ids, attention_mask,
                    categorical_features, numerical_features
                )
                
                neg_indices = pairs['negative_indices']
                negative_outputs = self.model(
                    input_ids[neg_indices], attention_mask[neg_indices],
                    {k: v[neg_indices] for k, v in categorical_features.items()},
                    numerical_features[neg_indices]
                )
                
                outputs = {
                    'query_embedding': anchor_outputs['query_embedding'],
                    'target_embedding': positive_outputs['query_embedding'],
                    'recommendation_score': torch.cosine_similarity(
                        anchor_outputs['query_embedding'],
                        positive_outputs['query_embedding'],
                        dim=1
                    ).unsqueeze(1)
                }
                
                labels = pairs['positive_labels'].to(self.device)
                negative_embeddings = negative_outputs['query_embedding']
                
                loss_dict = self.criterion(outputs, labels, negative_embeddings)
                loss = loss_dict['total_loss']
                
                loss.backward()
                
                if (batch_idx + 1) % self.config.get('accumulation_steps', 1) == 0:
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
            if batch_idx % 50 == 0:
                wandb.log({
                    'train_loss': loss.item(),
                    'learning_rate': self.scheduler.get_last_lr()[0],
                    'step': batch_idx + len(self.train_loader) * (wandb.run.step // len(self.train_loader))
                })
        
        return total_loss / num_batches
    
    def validate(self):
        """Validate the model"""
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                # Move to device
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                numerical_features = batch['numerical_features'].to(self.device)
                
                categorical_features = {}
                for key, value in batch['categorical_features'].items():
                    categorical_features[key] = value.to(self.device)
                
                # Create validation pairs
                pairs = self.create_training_pairs(batch)
                
                # Forward pass
                anchor_outputs = self.model(
                    input_ids, attention_mask,
                    categorical_features, numerical_features
                )
                
                positive_outputs = self.model(
                    input_ids, attention_mask,
                    categorical_features, numerical_features
                )
                
                neg_indices = pairs['negative_indices']
                negative_outputs = self.model(
                    input_ids[neg_indices], attention_mask[neg_indices],
                    {k: v[neg_indices] for k, v in categorical_features.items()},
                    numerical_features[neg_indices]
                )
                
                outputs = {
                    'query_embedding': anchor_outputs['query_embedding'],
                    'target_embedding': positive_outputs['query_embedding'],
                    'recommendation_score': torch.cosine_similarity(
                        anchor_outputs['query_embedding'],
                        positive_outputs['query_embedding'],
                        dim=1
                    ).unsqueeze(1)
                }
                
                labels = pairs['positive_labels'].to(self.device)
                negative_embeddings = negative_outputs['query_embedding']
                
                loss_dict = self.criterion(outputs, labels, negative_embeddings)
                loss = loss_dict['total_loss']
                
                total_loss += loss.item()
                num_batches += 1
        
        avg_val_loss = total_loss / num_batches
        return avg_val_loss
    
    def train(self):
        """Main training loop"""
        logger.info("Starting training...")
        
        for epoch in range(self.config['epochs']):
            logger.info(f"Epoch {epoch + 1}/{self.config['epochs']}")
            
            # Train
            train_loss = self.train_epoch()
            
            # Validate
            val_loss = self.validate()
            
            logger.info(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            
            # Log to wandb
            wandb.log({
                'epoch': epoch,
                'train_loss_epoch': train_loss,
                'val_loss_epoch': val_loss
            })
            
            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                self.save_model(f"best_multimodal_tv_model.pt")
                logger.info(f"New best model saved with val_loss: {val_loss:.4f}")
            else:
                self.patience_counter += 1
            
            # Early stopping
            if self.patience_counter >= self.config.get('patience', 10):
                logger.info(f"Early stopping after {epoch + 1} epochs")
                break
        
        logger.info("Training completed!")
    
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
    parser = argparse.ArgumentParser(description='Train Multimodal TV Transformer')
    parser.add_argument('--data_path', type=str, default='./data', help='Path to training data')
    parser.add_argument('--output_dir', type=str, default='./models', help='Output directory for models')
    parser.add_argument('--config_file', type=str, help='Config file path')
    parser.add_argument('--wandb_project', type=str, default='sota-tv-models', help='Wandb project name')
    parser.add_argument('--wandb_run_name', type=str, default='multimodal-transformer', help='Wandb run name')
    parser.add_argument('--resume_from', type=str, help='Resume training from checkpoint')
    
    args = parser.parse_args()
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name()}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(device).total_memory / 1e9:.1f} GB")
    
    # Load config
    config = get_model_config()
    if args.config_file and os.path.exists(args.config_file):
        with open(args.config_file) as f:
            config.update(json.load(f))
    
    # Add training specific config
    config.update({
        'epochs': 50,
        'patience': 10,
        'accumulation_steps': 2,
        'warmup_steps': 1000,
        'weight_decay': 1e-5,
        'contrastive_weight': 1.0,
        'recommendation_weight': 1.0,
        'use_mixed_precision': True,
        'max_grad_norm': 1.0
    })
    
    # Initialize wandb
    wandb.init(
        project=args.wandb_project,
        name=args.wandb_run_name,
        config=config
    )
    
    # Setup tokenizer
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    
    # Create datasets
    train_dataset = TVShowDataset(
        data_file=os.path.join(args.data_path, 'train.json'),
        tokenizer=tokenizer,
        max_length=512,
        augment_data=True
    )
    
    val_dataset = TVShowDataset(
        data_file=os.path.join(args.data_path, 'val.json'),
        tokenizer=tokenizer,
        max_length=512,
        augment_data=False
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    # Create model
    vocab_sizes = {
        'genres': 50,
        'actors': 10000,
        'network': 100,
        'creators': 5000,
        'status': 5,
        'language': 20
    }
    
    model = MultimodalTransformerTV(
        vocab_sizes=vocab_sizes,
        num_shows=10000,
        embed_dim=config['embed_dim'],
        hidden_dim=config['hidden_dim'],
        num_layers=config['num_layers'],
        num_heads=config['num_heads'],
        dropout=config['dropout'],
        use_gradient_checkpointing=config['use_gradient_checkpointing']
    )
    
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Resume from checkpoint if specified
    if args.resume_from and os.path.exists(args.resume_from):
        checkpoint = torch.load(args.resume_from, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Resumed from checkpoint: {args.resume_from}")
    
    # Create trainer and train
    trainer = MultimodalTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device
    )
    
    trainer.train()
    
    # Save final model
    os.makedirs(args.output_dir, exist_ok=True)
    trainer.save_model(os.path.join(args.output_dir, 'final_multimodal_tv_model.pt'))
    
    wandb.finish()

if __name__ == "__main__":
    main()