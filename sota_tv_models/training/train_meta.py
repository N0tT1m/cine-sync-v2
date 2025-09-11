"""
Training script for Meta-Learning TV Adapter (MLTA)
Optimized for RTX 4090 with MAML and few-shot learning
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
from sklearn.metrics import accuracy_score
from tqdm.auto import tqdm
import random
from collections import defaultdict

# Add parent directory to path to import models
sys.path.append(str(Path(__file__).parent.parent))
from models.meta_learning import MetaLearningTVAdapter, MAMLTrainer, get_meta_learning_config

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MetaTaskDataset(Dataset):
    """Dataset for meta-learning tasks"""
    
    def __init__(self, 
                 data_file: str,
                 support_size: int = 5,
                 query_size: int = 15,
                 num_tasks_per_episode: int = 8):
        
        with open(data_file) as f:
            self.data = json.load(f)
        
        self.support_size = support_size
        self.query_size = query_size
        self.num_tasks_per_episode = num_tasks_per_episode
        
        # Group shows by genres for task creation
        self.genre_to_shows = defaultdict(list)
        self.platform_to_shows = defaultdict(list)
        
        for show in self.data:
            genres = show['categorical_features'].get('genres', [])
            for genre in genres:
                if genre > 0:  # Skip padding tokens
                    self.genre_to_shows[genre].append(show)
            
            networks = show['categorical_features'].get('networks', [])
            for network in networks:
                if network > 0:  # Skip padding tokens
                    self.platform_to_shows[network].append(show)
        
        # Filter genres/platforms with sufficient shows
        self.valid_genres = [g for g, shows in self.genre_to_shows.items() 
                           if len(shows) >= support_size + query_size]
        self.valid_platforms = [p for p, shows in self.platform_to_shows.items() 
                              if len(shows) >= support_size + query_size]
        
        logger.info(f"Valid genres for meta-learning: {len(self.valid_genres)}")
        logger.info(f"Valid platforms for meta-learning: {len(self.valid_platforms)}")
        
        # Create base feature extractor (simplified)
        self.base_features = self._extract_base_features()
    
    def _extract_base_features(self):
        """Extract base features from all shows"""
        features = {}
        for show in self.data:
            show_id = show['id']
            
            # Combine categorical and numerical features
            categorical_features = show['categorical_features']
            numerical_features = show['numerical_features']
            
            # Create feature vector (simplified)
            feature_vector = []
            
            # Add genre features (one-hot like)
            genre_vector = [0] * 50  # Assume max 50 genres
            for genre in categorical_features.get('genres', [])[:5]:
                if 0 < genre < 50:
                    genre_vector[genre] = 1
            feature_vector.extend(genre_vector)
            
            # Add network features
            network_vector = [0] * 20  # Assume max 20 networks
            for network in categorical_features.get('networks', [])[:3]:
                if 0 < network < 20:
                    network_vector[network] = 1
            feature_vector.extend(network_vector)
            
            # Add numerical features
            feature_vector.extend(numerical_features[:10])  # Take first 10
            
            # Pad to fixed size
            while len(feature_vector) < 1024:
                feature_vector.append(0.0)
            
            features[show_id] = torch.tensor(feature_vector[:1024], dtype=torch.float)
        
        return features
    
    def create_genre_task(self):
        """Create a few-shot genre classification task"""
        
        # Sample a genre
        genre = random.choice(self.valid_genres)
        genre_shows = self.genre_to_shows[genre]
        
        # Sample support and query sets
        sampled_shows = random.sample(genre_shows, self.support_size + self.query_size)
        support_shows = sampled_shows[:self.support_size]
        query_shows = sampled_shows[self.support_size:]
        
        # Create features and labels
        support_features = torch.stack([self.base_features[show['id']] for show in support_shows])
        query_features = torch.stack([self.base_features[show['id']] for show in query_shows])
        
        # Labels (1 for target genre, 0 for others)
        support_labels = torch.ones(self.support_size)
        query_labels = torch.ones(self.query_size)
        
        # Add negative examples (other genres)
        other_genres = [g for g in self.valid_genres if g != genre]
        if other_genres:
            other_genre = random.choice(other_genres)
            other_shows = self.genre_to_shows[other_genre]
            
            if len(other_shows) >= self.support_size + self.query_size:
                neg_sampled = random.sample(other_shows, self.support_size + self.query_size)
                neg_support = neg_sampled[:self.support_size]
                neg_query = neg_sampled[self.support_size:]
                
                neg_support_features = torch.stack([self.base_features[show['id']] for show in neg_support])
                neg_query_features = torch.stack([self.base_features[show['id']] for show in neg_query])
                
                # Combine positive and negative
                support_features = torch.cat([support_features, neg_support_features])
                query_features = torch.cat([query_features, neg_query_features])
                support_labels = torch.cat([support_labels, torch.zeros(self.support_size)])
                query_labels = torch.cat([query_labels, torch.zeros(self.query_size)])
        
        return {
            'task_type': 'genre',
            'task_id': genre,
            'support': {
                'features': support_features,
                'targets': support_labels
            },
            'query': {
                'features': query_features,
                'targets': query_labels
            }
        }
    
    def create_platform_task(self):
        """Create a few-shot platform adaptation task"""
        
        # Sample a platform
        platform = random.choice(self.valid_platforms)
        platform_shows = self.platform_to_shows[platform]
        
        # Sample support and query sets
        sampled_shows = random.sample(platform_shows, self.support_size + self.query_size)
        support_shows = sampled_shows[:self.support_size]
        query_shows = sampled_shows[self.support_size:]
        
        support_features = torch.stack([self.base_features[show['id']] for show in support_shows])
        query_features = torch.stack([self.base_features[show['id']] for show in query_shows])
        
        # Create platform-specific targets (simplified as rating prediction)
        support_targets = torch.tensor([
            show['numerical_features'][0] if show['numerical_features'] else 0.5 
            for show in support_shows
        ], dtype=torch.float)
        
        query_targets = torch.tensor([
            show['numerical_features'][0] if show['numerical_features'] else 0.5 
            for show in query_shows
        ], dtype=torch.float)
        
        return {
            'task_type': 'platform',
            'task_id': platform,
            'support': {
                'features': support_features,
                'targets': support_targets
            },
            'query': {
                'features': query_features,
                'targets': query_targets
            }
        }
    
    def __len__(self):
        # Number of episodes
        return 1000  # Arbitrary large number for meta-learning
    
    def __getitem__(self, idx):
        """Create a batch of meta-learning tasks"""
        
        tasks = []
        
        for _ in range(self.num_tasks_per_episode):
            # Randomly choose task type
            if random.random() < 0.5 and self.valid_genres:
                task = self.create_genre_task()
            elif self.valid_platforms:
                task = self.create_platform_task()
            else:
                # Fallback to genre if no platforms
                task = self.create_genre_task()
            
            tasks.append(task)
        
        return tasks

class MetaLearningTrainer:
    """Trainer for meta-learning TV adapter"""
    
    def __init__(self, 
                 model: MetaLearningTVAdapter,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 config: dict,
                 device: torch.device):
        
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        
        # MAML trainer
        self.maml_trainer = MAMLTrainer(
            model=model,
            meta_lr=config['meta_lr'],
            inner_lr=config['inner_lr']
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
        
        progress_bar = tqdm(self.train_loader, desc="Training Meta-Learning")
        
        for batch_idx, task_batch in enumerate(progress_bar):
            # task_batch is a list of tasks
            
            if self.scaler:
                with autocast():
                    meta_loss = self.maml_trainer.meta_train_step(task_batch)
            else:
                meta_loss = self.maml_trainer.meta_train_step(task_batch)
            
            total_loss += meta_loss
            num_batches += 1
            
            # Update progress bar
            progress_bar.set_postfix({
                'meta_loss': f"{meta_loss:.4f}",
                'avg_loss': f"{total_loss/num_batches:.4f}"
            })
            
            # Log to wandb
            if batch_idx % 10 == 0:
                wandb.log({
                    'meta_train_loss': meta_loss,
                    'meta_learning_rate': self.config['meta_lr'],
                    'inner_learning_rate': self.config['inner_lr']
                })
        
        return total_loss / num_batches
    
    def validate(self):
        """Validate the model"""
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        adaptation_accuracies = []
        
        with torch.no_grad():
            for task_batch in tqdm(self.val_loader, desc="Validation"):
                
                batch_loss = 0
                for task in task_batch:
                    # Test adaptation on each task
                    support_data = task['support']
                    query_data = task['query']
                    
                    # Move to device
                    support_features = support_data['features'].to(self.device)
                    support_targets = support_data['targets'].to(self.device)
                    query_features = query_data['features'].to(self.device)
                    query_targets = query_data['targets'].to(self.device)
                    
                    # Fast adaptation
                    adapted_features = self.model.fast_adapt(
                        support_data={'features': support_features, 'targets': support_targets},
                        query_data={'features': query_features},
                        adaptation_type=task['task_type']
                    )
                    
                    # Compute loss (simplified)
                    if task['task_type'] == 'genre':
                        # Classification loss
                        logits = torch.matmul(adapted_features, support_features.T).mean(dim=1)
                        loss = nn.functional.binary_cross_entropy_with_logits(
                            logits, query_targets
                        )
                        
                        # Accuracy
                        preds = (torch.sigmoid(logits) > 0.5).float()
                        accuracy = (preds == query_targets).float().mean()
                        adaptation_accuracies.append(accuracy.item())
                        
                    else:
                        # Regression loss
                        pred_ratings = torch.matmul(adapted_features, support_features.T).mean(dim=1)
                        loss = nn.functional.mse_loss(pred_ratings, query_targets)
                    
                    batch_loss += loss.item()
                
                total_loss += batch_loss / len(task_batch)
                num_batches += 1
        
        avg_val_loss = total_loss / num_batches
        avg_adaptation_accuracy = np.mean(adaptation_accuracies) if adaptation_accuracies else 0
        
        return avg_val_loss, avg_adaptation_accuracy
    
    def train(self):
        """Main training loop"""
        logger.info("Starting meta-learning training...")
        
        for epoch in range(self.config['epochs']):
            logger.info(f"Epoch {epoch + 1}/{self.config['epochs']}")
            
            # Train
            train_loss = self.train_epoch()
            
            # Validate
            val_loss, val_accuracy = self.validate()
            
            logger.info(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            logger.info(f"Val Adaptation Accuracy: {val_accuracy:.4f}")
            
            # Log to wandb
            wandb.log({
                'meta_epoch': epoch,
                'meta_train_loss_epoch': train_loss,
                'meta_val_loss_epoch': val_loss,
                'meta_val_adaptation_accuracy': val_accuracy
            })
            
            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                self.save_model(f"best_meta_learning_tv_model.pt")
                logger.info(f"New best model saved with val_loss: {val_loss:.4f}")
            else:
                self.patience_counter += 1
            
            # Early stopping
            if self.patience_counter >= self.config.get('patience', 20):
                logger.info(f"Early stopping after {epoch + 1} epochs")
                break
        
        logger.info("Meta-learning training completed!")
    
    def save_model(self, filename):
        """Save model checkpoint"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'maml_optimizer_state_dict': self.maml_trainer.meta_optimizer.state_dict(),
            'config': self.config,
            'best_val_loss': self.best_val_loss
        }
        torch.save(checkpoint, filename)

def collate_meta_fn(batch):
    """Custom collate function for meta-learning"""
    # batch is a list of task lists
    # Flatten to get all tasks
    all_tasks = []
    for task_list in batch:
        all_tasks.extend(task_list)
    
    return all_tasks

def main():
    parser = argparse.ArgumentParser(description='Train Meta-Learning TV Adapter')
    parser.add_argument('--data_path', type=str, default='./processed_data', help='Path to processed data')
    parser.add_argument('--output_dir', type=str, default='./models', help='Output directory for models')
    parser.add_argument('--config_file', type=str, help='Config file path')
    parser.add_argument('--wandb_project', type=str, default='sota-tv-models', help='Wandb project name')
    parser.add_argument('--wandb_run_name', type=str, default='meta-learning', help='Wandb run name')
    parser.add_argument('--resume_from', type=str, help='Resume training from checkpoint')
    
    args = parser.parse_args()
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Load config
    config = get_meta_learning_config()
    if args.config_file and os.path.exists(args.config_file):
        with open(args.config_file) as f:
            config.update(json.load(f).get('meta_learning_config', {}))
    
    # Initialize wandb
    wandb.init(
        project=args.wandb_project,
        name=args.wandb_run_name,
        config=config
    )
    
    # Create datasets
    train_dataset = MetaTaskDataset(
        data_file=os.path.join(args.data_path, 'train_data.json'),
        support_size=config['support_size'],
        query_size=config['query_size'],
        num_tasks_per_episode=config['meta_batch_size']
    )
    
    val_dataset = MetaTaskDataset(
        data_file=os.path.join(args.data_path, 'val_data.json'),
        support_size=config['support_size'],
        query_size=config['query_size'],
        num_tasks_per_episode=config['meta_batch_size']
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=1,  # Each item is already a batch of tasks
        shuffle=True,
        num_workers=2,
        collate_fn=collate_meta_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=2,
        collate_fn=collate_meta_fn
    )
    
    # Load vocabulary sizes
    with open(os.path.join(args.data_path, 'metadata.json')) as f:
        metadata = json.load(f)
    
    vocab_sizes = metadata['vocab_sizes']
    
    # Create model
    model = MetaLearningTVAdapter(
        base_feature_dim=config['base_feature_dim'],
        adapter_dim=config['adapter_dim'],
        num_genres=vocab_sizes.get('genres', 50),
        num_platforms=vocab_sizes.get('networks', 20),
        num_adaptation_steps=config['num_adaptation_steps'],
        meta_lr=config['meta_lr']
    )
    
    logger.info(f"Meta-Learning Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Resume from checkpoint if specified
    if args.resume_from and os.path.exists(args.resume_from):
        checkpoint = torch.load(args.resume_from, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Resumed from checkpoint: {args.resume_from}")
    
    # Create trainer and train
    trainer = MetaLearningTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device
    )
    
    trainer.train()
    
    # Save final model
    os.makedirs(args.output_dir, exist_ok=True)
    trainer.save_model(os.path.join(args.output_dir, 'final_meta_learning_tv_model.pt'))
    
    wandb.finish()

if __name__ == "__main__":
    main()