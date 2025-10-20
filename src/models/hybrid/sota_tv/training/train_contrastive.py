"""
Training script for Contrastive Learning TV Encoder (CL-TV)
Optimized for RTX 4090 with large batch sizes and hard negative mining
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
from transformers import RobertaTokenizer, get_cosine_schedule_with_warmup
from sklearn.metrics import accuracy_score
from tqdm.auto import tqdm
import random

# Add parent directory to path to import models
sys.path.append(str(Path(__file__).parent.parent))
from models.contrastive_learning import (
    ContrastiveTVModel, DataAugmentation, MultiPositiveContrastiveLoss, 
    get_contrastive_config
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ContrastiveTVDataset(Dataset):
    """Dataset for contrastive learning with TV shows"""
    
    def __init__(self, 
                 data_file: str,
                 tokenizer: RobertaTokenizer,
                 max_length: int = 512,
                 num_negatives: int = 63,
                 hard_negative_ratio: float = 0.3):
        
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.num_negatives = num_negatives
        self.hard_negative_ratio = hard_negative_ratio
        
        # Load data
        with open(data_file) as f:
            self.data = json.load(f)
        
        # Data augmentation
        self.augmentation = DataAugmentation(
            text_dropout_prob=0.1,
            genre_dropout_prob=0.2,
            actor_dropout_prob=0.3,
            temporal_noise_std=0.1
        )
        
        # Build genre similarity index for hard negatives
        self.build_similarity_index()
        
        logger.info(f"Loaded {len(self.data)} TV show samples for contrastive learning")
    
    def build_similarity_index(self):
        """Build index for finding hard negatives based on genre similarity"""
        self.genre_to_shows = {}
        
        for idx, show in enumerate(self.data):
            genres = show['categorical_features']['genres']
            for genre_id in genres:
                if genre_id not in self.genre_to_shows:
                    self.genre_to_shows[genre_id] = []
                self.genre_to_shows[genre_id].append(idx)
        
        logger.info(f"Built similarity index with {len(self.genre_to_shows)} genres")
    
    def get_hard_negatives(self, anchor_idx: int, num_hard: int) -> list:
        """Get hard negatives based on genre similarity"""
        anchor_genres = set(self.data[anchor_idx]['categorical_features']['genres'])
        
        # Find shows that share some genres (but are not identical)
        candidate_negatives = []
        for genre_id in anchor_genres:
            if genre_id in self.genre_to_shows:
                for show_idx in self.genre_to_shows[genre_id]:
                    if show_idx != anchor_idx:
                        candidate_negatives.append(show_idx)
        
        # Remove duplicates and limit
        candidate_negatives = list(set(candidate_negatives))
        
        if len(candidate_negatives) >= num_hard:
            return random.sample(candidate_negatives, num_hard)
        else:
            # Fill remaining with random negatives
            remaining = num_hard - len(candidate_negatives)
            random_negatives = random.sample(
                [i for i in range(len(self.data)) if i != anchor_idx], 
                remaining
            )
            return candidate_negatives + random_negatives
    
    def tokenize_text(self, text: str):
        """Tokenize text with the tokenizer"""
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return encoding['input_ids'].squeeze(), encoding['attention_mask'].squeeze()
    
    def prepare_features(self, show_data: dict, augment: bool = False):
        """Prepare features for a show"""
        text = show_data['text']
        categorical_features = show_data['categorical_features'].copy()
        numerical_features = torch.tensor(show_data['numerical_features'], dtype=torch.float)
        
        if augment:
            # Apply augmentations
            text, genres, networks, numerical_features = self.augmentation.create_positive_pair(
                text,
                categorical_features['genres'],
                categorical_features.get('networks', []),
                numerical_features
            )
            categorical_features['genres'] = genres
            categorical_features['networks'] = networks
        
        # Tokenize text
        input_ids, attention_mask = self.tokenize_text(text)
        
        # Convert categorical features to tensors with padding
        max_lengths = {'genres': 5, 'networks': 3, 'creators': 5, 'status': 1, 'language': 1}
        
        for key, values in categorical_features.items():
            if isinstance(values, list):
                # Pad or truncate to max length
                max_len = max_lengths.get(key, 5)
                if len(values) > max_len:
                    values = values[:max_len]
                elif len(values) < max_len:
                    values = values + [0] * (max_len - len(values))
                categorical_features[key] = torch.tensor(values, dtype=torch.long)
            else:
                categorical_features[key] = torch.tensor([values], dtype=torch.long)
        
        return input_ids, attention_mask, categorical_features, numerical_features
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        anchor_show = self.data[idx]
        
        # Anchor features (original)
        anchor_input_ids, anchor_attention_mask, anchor_categorical, anchor_numerical = \
            self.prepare_features(anchor_show, augment=False)
        
        # Positive features (augmented version of same show)
        positive_input_ids, positive_attention_mask, positive_categorical, positive_numerical = \
            self.prepare_features(anchor_show, augment=True)
        
        # Negative features
        num_hard_negatives = int(self.num_negatives * self.hard_negative_ratio)
        num_random_negatives = self.num_negatives - num_hard_negatives
        
        # Get hard negatives
        hard_negative_indices = self.get_hard_negatives(idx, num_hard_negatives)
        
        # Get random negatives
        random_negative_indices = random.sample(
            [i for i in range(len(self.data)) if i != idx and i not in hard_negative_indices],
            num_random_negatives
        )
        
        negative_indices = hard_negative_indices + random_negative_indices
        
        # Prepare negative features
        negative_input_ids = []
        negative_attention_mask = []
        negative_categorical = {key: [] for key in anchor_categorical.keys()}
        negative_numerical = []
        
        for neg_idx in negative_indices:
            neg_show = self.data[neg_idx]
            neg_input_ids, neg_attention_mask, neg_categorical, neg_numerical = \
                self.prepare_features(neg_show, augment=False)
            
            negative_input_ids.append(neg_input_ids)
            negative_attention_mask.append(neg_attention_mask)
            negative_numerical.append(neg_numerical)
            
            for key, value in neg_categorical.items():
                negative_categorical[key].append(value)
        
        # Stack negative features
        negative_input_ids = torch.stack(negative_input_ids)
        negative_attention_mask = torch.stack(negative_attention_mask)
        negative_numerical = torch.stack(negative_numerical)
        
        for key in negative_categorical.keys():
            negative_categorical[key] = torch.stack(negative_categorical[key])
        
        return {
            'anchor_input_ids': anchor_input_ids,
            'anchor_attention_mask': anchor_attention_mask,
            'anchor_categorical': anchor_categorical,
            'anchor_numerical': anchor_numerical,
            'positive_input_ids': positive_input_ids,
            'positive_attention_mask': positive_attention_mask,
            'positive_categorical': positive_categorical,
            'positive_numerical': positive_numerical,
            'negative_input_ids': negative_input_ids,
            'negative_attention_mask': negative_attention_mask,
            'negative_categorical': negative_categorical,
            'negative_numerical': negative_numerical,
            'anchor_id': anchor_show['id']
        }

def collate_contrastive_fn(batch):
    """Custom collate function for contrastive learning batch"""
    # This is a simplified collate function
    # In practice, you'd handle variable-length sequences
    return {
        key: torch.stack([item[key] if not isinstance(item[key], dict) else 
                         {k: torch.stack([item[key][k] for item in batch]) 
                          for k in item[key].keys()}[0] if key.endswith('_categorical') else item[key]
                         for item in batch]) if not key.endswith('_categorical') else
        {k: torch.stack([item[key][k] for item in batch]) for k in batch[0][key].keys()}
        for key in batch[0].keys()
    }

class ContrastiveTrainer:
    """Trainer for contrastive TV model"""
    
    def __init__(self, 
                 model: ContrastiveTVModel,
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
            num_warmup_steps=config.get('warmup_steps', 2000),
            num_training_steps=total_steps
        )
        
        # Mixed precision
        self.scaler = GradScaler() if config.get('use_mixed_precision', True) else None
        
        # Advanced loss function
        self.advanced_loss = MultiPositiveContrastiveLoss(
            temperature=config['temperature'],
            alpha=0.1
        )
        
        # Metrics
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        progress_bar = tqdm(self.train_loader, desc="Training Contrastive")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move to device
            for key in batch.keys():
                if isinstance(batch[key], dict):
                    for subkey in batch[key].keys():
                        batch[key][subkey] = batch[key][subkey].to(self.device)
                else:
                    batch[key] = batch[key].to(self.device)
            
            # Forward pass with mixed precision
            if self.scaler:
                with autocast():
                    outputs = self.model(
                        anchor_input_ids=batch['anchor_input_ids'],
                        anchor_attention_mask=batch['anchor_attention_mask'],
                        anchor_categorical=batch['anchor_categorical'],
                        anchor_numerical=batch['anchor_numerical'],
                        positive_input_ids=batch['positive_input_ids'],
                        positive_attention_mask=batch['positive_attention_mask'],
                        positive_categorical=batch['positive_categorical'],
                        positive_numerical=batch['positive_numerical'],
                        negative_input_ids=batch['negative_input_ids'].view(-1, batch['negative_input_ids'].size(-1)),
                        negative_attention_mask=batch['negative_attention_mask'].view(-1, batch['negative_attention_mask'].size(-1)),
                        negative_categorical={k: v.view(-1, v.size(-1)) for k, v in batch['negative_categorical'].items()},
                        negative_numerical=batch['negative_numerical'].view(-1, batch['negative_numerical'].size(-1))
                    )
                    
                    loss = outputs['contrastive_loss']
                
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
                    anchor_input_ids=batch['anchor_input_ids'],
                    anchor_attention_mask=batch['anchor_attention_mask'],
                    anchor_categorical=batch['anchor_categorical'],
                    anchor_numerical=batch['anchor_numerical'],
                    positive_input_ids=batch['positive_input_ids'],
                    positive_attention_mask=batch['positive_attention_mask'],
                    positive_categorical=batch['positive_categorical'],
                    positive_numerical=batch['positive_numerical'],
                    negative_input_ids=batch['negative_input_ids'].view(-1, batch['negative_input_ids'].size(-1)),
                    negative_attention_mask=batch['negative_attention_mask'].view(-1, batch['negative_attention_mask'].size(-1)),
                    negative_categorical={k: v.view(-1, v.size(-1)) for k, v in batch['negative_categorical'].items()},
                    negative_numerical=batch['negative_numerical'].view(-1, batch['negative_numerical'].size(-1))
                )
                
                loss = outputs['contrastive_loss']
                
                loss.backward()
                
                if (batch_idx + 1) % self.config.get('gradient_accumulation_steps', 1) == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.get('max_grad_norm', 1.0))
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    self.scheduler.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            # Calculate similarities for monitoring
            pos_sim = outputs['anchor_pos_similarity'].mean().item()
            neg_sim = outputs['anchor_neg_similarity'].mean().item()
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'pos_sim': f"{pos_sim:.3f}",
                'neg_sim': f"{neg_sim:.3f}",
                'ratio': f"{pos_sim/max(neg_sim, 1e-6):.2f}"
            })
            
            # Log to wandb
            if batch_idx % 20 == 0:
                wandb.log({
                    'contrastive_train_loss': loss.item(),
                    'pos_similarity': pos_sim,
                    'neg_similarity': neg_sim,
                    'sim_ratio': pos_sim / max(neg_sim, 1e-6),
                    'learning_rate': self.scheduler.get_last_lr()[0],
                    'temperature': self.model.temperature
                })
        
        return total_loss / num_batches
    
    def validate(self):
        """Validate the model"""
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        all_pos_sims = []
        all_neg_sims = []
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                # Move to device
                for key in batch.keys():
                    if isinstance(batch[key], dict):
                        for subkey in batch[key].keys():
                            batch[key][subkey] = batch[key][subkey].to(self.device)
                    else:
                        batch[key] = batch[key].to(self.device)
                
                outputs = self.model(
                    anchor_input_ids=batch['anchor_input_ids'],
                    anchor_attention_mask=batch['anchor_attention_mask'],
                    anchor_categorical=batch['anchor_categorical'],
                    anchor_numerical=batch['anchor_numerical'],
                    positive_input_ids=batch['positive_input_ids'],
                    positive_attention_mask=batch['positive_attention_mask'],
                    positive_categorical=batch['positive_categorical'],
                    positive_numerical=batch['positive_numerical'],
                    negative_input_ids=batch['negative_input_ids'].view(-1, batch['negative_input_ids'].size(-1)),
                    negative_attention_mask=batch['negative_attention_mask'].view(-1, batch['negative_attention_mask'].size(-1)),
                    negative_categorical={k: v.view(-1, v.size(-1)) for k, v in batch['negative_categorical'].items()},
                    negative_numerical=batch['negative_numerical'].view(-1, batch['negative_numerical'].size(-1))
                )
                
                loss = outputs['contrastive_loss']
                total_loss += loss.item()
                num_batches += 1
                
                # Collect similarities
                all_pos_sims.extend(outputs['anchor_pos_similarity'].cpu().tolist())
                all_neg_sims.extend(outputs['anchor_neg_similarity'].cpu().tolist())
        
        avg_val_loss = total_loss / num_batches
        avg_pos_sim = np.mean(all_pos_sims)
        avg_neg_sim = np.mean(all_neg_sims)
        
        return avg_val_loss, avg_pos_sim, avg_neg_sim
    
    def train(self):
        """Main training loop"""
        logger.info("Starting contrastive training...")
        
        for epoch in range(self.config['epochs']):
            logger.info(f"Epoch {epoch + 1}/{self.config['epochs']}")
            
            # Train
            train_loss = self.train_epoch()
            
            # Validate
            val_loss, val_pos_sim, val_neg_sim = self.validate()
            
            logger.info(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            logger.info(f"Val Pos Sim: {val_pos_sim:.3f}, Val Neg Sim: {val_neg_sim:.3f}")
            
            # Log to wandb
            wandb.log({
                'contrastive_epoch': epoch,
                'contrastive_train_loss_epoch': train_loss,
                'contrastive_val_loss_epoch': val_loss,
                'val_pos_similarity': val_pos_sim,
                'val_neg_similarity': val_neg_sim,
                'val_sim_ratio': val_pos_sim / max(val_neg_sim, 1e-6)
            })
            
            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                self.save_model(f"best_contrastive_tv_model.pt")
                logger.info(f"New best model saved with val_loss: {val_loss:.4f}")
            else:
                self.patience_counter += 1
            
            # Early stopping
            if self.patience_counter >= self.config.get('patience', 10):
                logger.info(f"Early stopping after {epoch + 1} epochs")
                break
        
        logger.info("Contrastive training completed!")
    
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
    parser = argparse.ArgumentParser(description='Train Contrastive TV Model')
    parser.add_argument('--data_path', type=str, default='./processed_data', help='Path to processed data')
    parser.add_argument('--output_dir', type=str, default='./models', help='Output directory for models')
    parser.add_argument('--config_file', type=str, help='Config file path')
    parser.add_argument('--wandb_project', type=str, default='sota-tv-models', help='Wandb project name')
    parser.add_argument('--wandb_run_name', type=str, default='contrastive-tv', help='Wandb run name')
    parser.add_argument('--resume_from', type=str, help='Resume training from checkpoint')
    
    args = parser.parse_args()
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Load config
    config = get_contrastive_config()
    if args.config_file and os.path.exists(args.config_file):
        with open(args.config_file) as f:
            config.update(json.load(f).get('contrastive_config', {}))
    
    # Add training specific config
    config.update({
        'epochs': 25,
        'patience': 10,
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
    train_dataset = ContrastiveTVDataset(
        data_file=os.path.join(args.data_path, 'train_data.json'),
        tokenizer=tokenizer,
        max_length=512,
        num_negatives=config['num_negatives'],
        hard_negative_ratio=config['hard_negative_ratio']
    )
    
    val_dataset = ContrastiveTVDataset(
        data_file=os.path.join(args.data_path, 'val_data.json'),
        tokenizer=tokenizer,
        max_length=512,
        num_negatives=config['num_negatives'],
        hard_negative_ratio=config['hard_negative_ratio']
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True  # Important for contrastive learning
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )
    
    # Load vocabulary sizes
    with open(os.path.join(args.data_path, 'metadata.json')) as f:
        metadata = json.load(f)
    
    vocab_sizes = metadata['vocab_sizes']
    
    # Create model
    model = ContrastiveTVModel(
        vocab_sizes=vocab_sizes,
        embed_dim=config['embed_dim'],
        hidden_dim=config['hidden_dim'],
        num_layers=config['num_layers'],
        dropout=config['dropout'],
        temperature=config['temperature'],
        use_hard_negatives=config['use_hard_negatives']
    )
    
    logger.info(f"Contrastive Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Resume from checkpoint if specified
    if args.resume_from and os.path.exists(args.resume_from):
        checkpoint = torch.load(args.resume_from, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Resumed from checkpoint: {args.resume_from}")
    
    # Create trainer and train
    trainer = ContrastiveTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device
    )
    
    trainer.train()
    
    # Save final model
    os.makedirs(args.output_dir, exist_ok=True)
    trainer.save_model(os.path.join(args.output_dir, 'final_contrastive_tv_model.pt'))
    
    wandb.finish()

if __name__ == "__main__":
    main()