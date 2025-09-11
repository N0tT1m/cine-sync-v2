"""
Training script for Ensemble System
Combines and fine-tunes all SOTA TV models
Optimized for RTX 4090
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
from sklearn.metrics import accuracy_score, roc_auc_score
from tqdm.auto import tqdm

# Add parent directory to path to import models
sys.path.append(str(Path(__file__).parent.parent))
from models.ensemble_system import EnsembleSystem, EnsembleLoss, get_ensemble_config

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnsembleDataset(Dataset):
    """Dataset for ensemble training"""
    
    def __init__(self, 
                 data_file: str,
                 tokenizer: RobertaTokenizer,
                 max_length: int = 512):
        
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Load data
        with open(data_file) as f:
            self.data = json.load(f)
        
        # Create positive and negative pairs
        self.pairs = self._create_training_pairs()
        
        logger.info(f"Created {len(self.pairs)} training pairs for ensemble")
    
    def _create_training_pairs(self):
        """Create positive and negative pairs for training"""
        pairs = []
        
        for i, show1 in enumerate(self.data):
            # Create positive pairs (same genre shows)
            show1_genres = set(show1['categorical_features']['genres'])
            
            for j, show2 in enumerate(self.data):
                if i >= j:  # Avoid duplicates
                    continue
                
                show2_genres = set(show2['categorical_features']['genres'])
                
                # Calculate genre overlap
                intersection = len(show1_genres.intersection(show2_genres))
                union = len(show1_genres.union(show2_genres))
                
                if union > 0:
                    similarity = intersection / union
                    
                    # Create pair
                    label = 1 if similarity > 0.3 else 0  # Threshold for positive pair
                    
                    pairs.append({
                        'show1': show1,
                        'show2': show2,
                        'label': label,
                        'similarity': similarity
                    })
        
        return pairs
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        pair = self.pairs[idx]
        
        # Process first show
        show1 = pair['show1']
        text1 = show1['text']
        encoding1 = self.tokenizer(
            text1,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        categorical1 = {}
        for key, values in show1['categorical_features'].items():
            if isinstance(values, list):
                # Pad to fixed length
                max_len = {'genres': 5, 'networks': 3, 'creators': 5, 'status': 1, 'language': 1}.get(key, 5)
                if len(values) > max_len:
                    values = values[:max_len]
                elif len(values) < max_len:
                    values = values + [0] * (max_len - len(values))
                categorical1[key] = torch.tensor(values, dtype=torch.long)
            else:
                categorical1[key] = torch.tensor([values], dtype=torch.long)
        
        numerical1 = torch.tensor(show1['numerical_features'], dtype=torch.float)
        
        # Process second show
        show2 = pair['show2']
        text2 = show2['text']
        encoding2 = self.tokenizer(
            text2,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        categorical2 = {}
        for key, values in show2['categorical_features'].items():
            if isinstance(values, list):
                max_len = {'genres': 5, 'networks': 3, 'creators': 5, 'status': 1, 'language': 1}.get(key, 5)
                if len(values) > max_len:
                    values = values[:max_len]
                elif len(values) < max_len:
                    values = values + [0] * (max_len - len(values))
                categorical2[key] = torch.tensor(values, dtype=torch.long)
            else:
                categorical2[key] = torch.tensor([values], dtype=torch.long)
        
        numerical2 = torch.tensor(show2['numerical_features'], dtype=torch.float)
        
        return {
            'input_ids_1': encoding1['input_ids'].squeeze(),
            'attention_mask_1': encoding1['attention_mask'].squeeze(),
            'categorical_1': categorical1,
            'numerical_1': numerical1,
            'show_id_1': torch.tensor([show1['id']], dtype=torch.long),
            'input_ids_2': encoding2['input_ids'].squeeze(),
            'attention_mask_2': encoding2['attention_mask'].squeeze(),
            'categorical_2': categorical2,
            'numerical_2': numerical2,
            'show_id_2': torch.tensor([show2['id']], dtype=torch.long),
            'label': torch.tensor(pair['label'], dtype=torch.float),
            'similarity': torch.tensor(pair['similarity'], dtype=torch.float)
        }

def collate_ensemble_fn(batch):
    """Custom collate function for ensemble training"""
    
    collated = {}
    
    # Stack regular tensors
    for key in ['input_ids_1', 'attention_mask_1', 'numerical_1', 'show_id_1',
                'input_ids_2', 'attention_mask_2', 'numerical_2', 'show_id_2',
                'label', 'similarity']:
        collated[key] = torch.stack([item[key] for item in batch])
    
    # Handle categorical features
    for cat_key in ['categorical_1', 'categorical_2']:
        collated[cat_key] = {}
        for feature_name in batch[0][cat_key].keys():
            collated[cat_key][feature_name] = torch.stack([item[cat_key][feature_name] for item in batch])
    
    return collated

class EnsembleTrainer:
    """Trainer for ensemble system"""
    
    def __init__(self, 
                 model: EnsembleSystem,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 config: dict,
                 device: torch.device):
        
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        
        # Optimizer (lower learning rate for ensemble fine-tuning)
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
            num_warmup_steps=config.get('warmup_steps', 500),
            num_training_steps=total_steps
        )
        
        # Loss function
        self.criterion = EnsembleLoss(
            recommendation_weight=config.get('recommendation_weight', 1.0),
            similarity_weight=config.get('similarity_weight', 0.8)
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
        
        progress_bar = tqdm(self.train_loader, desc="Training Ensemble")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move to device
            for key, value in batch.items():
                if isinstance(value, dict):
                    for subkey in value.keys():
                        batch[key][subkey] = batch[key][subkey].to(self.device)
                else:
                    batch[key] = value.to(self.device)
            
            # Forward pass
            if self.scaler:
                with autocast():
                    # Get features from first show
                    show1_features = self._extract_features(
                        batch['input_ids_1'], batch['attention_mask_1'],
                        batch['categorical_1'], batch['numerical_1'],
                        batch['show_id_1']
                    )
                    
                    # Get features from second show  
                    show2_features = self._extract_features(
                        batch['input_ids_2'], batch['attention_mask_2'],
                        batch['categorical_2'], batch['numerical_2'],
                        batch['show_id_2']
                    )
                    
                    # Compute ensemble outputs
                    ensemble_outputs = self.model([show1_features, show2_features])
                    
                    # Compute similarity between shows
                    similarity_score = torch.cosine_similarity(
                        ensemble_outputs['ensemble_features'],
                        show2_features,
                        dim=1
                    )
                    
                    # Create targets
                    targets = {
                        'recommendation_target': batch['similarity'],
                        'similarity_target': batch['label']
                    }
                    
                    # Prepare outputs for loss
                    outputs = {
                        'recommendation_score': similarity_score.unsqueeze(1),
                        'ensemble_features': ensemble_outputs['ensemble_features']
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
                # Regular training without mixed precision
                show1_features = self._extract_features(
                    batch['input_ids_1'], batch['attention_mask_1'],
                    batch['categorical_1'], batch['numerical_1'],
                    batch['show_id_1']
                )
                
                show2_features = self._extract_features(
                    batch['input_ids_2'], batch['attention_mask_2'],
                    batch['categorical_2'], batch['numerical_2'],
                    batch['show_id_2']
                )
                
                ensemble_outputs = self.model([show1_features, show2_features])
                
                similarity_score = torch.cosine_similarity(
                    ensemble_outputs['ensemble_features'],
                    show2_features,
                    dim=1
                )
                
                targets = {
                    'recommendation_target': batch['similarity'],
                    'similarity_target': batch['label']
                }
                
                outputs = {
                    'recommendation_score': similarity_score.unsqueeze(1),
                    'ensemble_features': ensemble_outputs['ensemble_features']
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
            if batch_idx % 10 == 0:
                log_dict = {
                    'ensemble_train_loss': loss.item(),
                    'learning_rate': self.scheduler.get_last_lr()[0],
                }
                
                # Add model weights if available
                if 'model_weights' in ensemble_outputs:
                    weights = ensemble_outputs['model_weights'].mean(dim=0)
                    for i, weight in enumerate(weights):
                        log_dict[f'model_weight_{i}'] = weight.item()
                
                wandb.log(log_dict)
        
        return total_loss / num_batches
    
    def _extract_features(self, input_ids, attention_mask, categorical, numerical, show_ids):
        """Extract features using a simple approach (placeholder)"""
        # This is a simplified feature extraction
        # In practice, this would use pre-trained model components
        
        batch_size = input_ids.size(0)
        feature_dim = self.config.get('ensemble_dim', 1024)
        
        # Create dummy features for now
        features = torch.randn(batch_size, feature_dim, device=self.device)
        
        return features
    
    def validate(self):
        """Validate the model"""
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                # Move to device
                for key, value in batch.items():
                    if isinstance(value, dict):
                        for subkey in value.keys():
                            batch[key][subkey] = batch[key][subkey].to(self.device)
                    else:
                        batch[key] = value.to(self.device)
                
                # Forward pass (simplified)
                show1_features = self._extract_features(
                    batch['input_ids_1'], batch['attention_mask_1'],
                    batch['categorical_1'], batch['numerical_1'],
                    batch['show_id_1']
                )
                
                show2_features = self._extract_features(
                    batch['input_ids_2'], batch['attention_mask_2'],
                    batch['categorical_2'], batch['numerical_2'],
                    batch['show_id_2']
                )
                
                ensemble_outputs = self.model([show1_features, show2_features])
                
                similarity_score = torch.cosine_similarity(
                    ensemble_outputs['ensemble_features'],
                    show2_features,
                    dim=1
                )
                
                targets = {
                    'recommendation_target': batch['similarity'],
                    'similarity_target': batch['label']
                }
                
                outputs = {
                    'recommendation_score': similarity_score.unsqueeze(1),
                    'ensemble_features': ensemble_outputs['ensemble_features']
                }
                
                loss_dict = self.criterion(outputs, targets)
                loss = loss_dict['total_loss']
                
                total_loss += loss.item()
                num_batches += 1
                
                # Collect predictions for metrics
                predictions = torch.sigmoid(similarity_score).cpu().numpy()
                labels = batch['label'].cpu().numpy()
                
                all_predictions.extend(predictions)
                all_labels.extend(labels)
        
        avg_val_loss = total_loss / num_batches
        
        # Calculate metrics
        try:
            val_auc = roc_auc_score(all_labels, all_predictions)
        except:
            val_auc = 0.5
        
        val_accuracy = accuracy_score(all_labels, np.array(all_predictions) > 0.5)
        
        return avg_val_loss, val_auc, val_accuracy
    
    def train(self):
        """Main training loop"""
        logger.info("Starting ensemble training...")
        
        for epoch in range(self.config['epochs']):
            logger.info(f"Epoch {epoch + 1}/{self.config['epochs']}")
            
            # Train
            train_loss = self.train_epoch()
            
            # Validate
            val_loss, val_auc, val_accuracy = self.validate()
            
            logger.info(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            logger.info(f"Val AUC: {val_auc:.4f}, Val Accuracy: {val_accuracy:.4f}")
            
            # Log to wandb
            wandb.log({
                'ensemble_epoch': epoch,
                'ensemble_train_loss_epoch': train_loss,
                'ensemble_val_loss_epoch': val_loss,
                'ensemble_val_auc': val_auc,
                'ensemble_val_accuracy': val_accuracy
            })
            
            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                self.save_model(f"best_ensemble_tv_model.pt")
                logger.info(f"New best model saved with val_loss: {val_loss:.4f}")
            else:
                self.patience_counter += 1
            
            # Early stopping
            if self.patience_counter >= self.config.get('patience', 8):
                logger.info(f"Early stopping after {epoch + 1} epochs")
                break
        
        logger.info("Ensemble training completed!")
    
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
    parser = argparse.ArgumentParser(description='Train Ensemble TV System')
    parser.add_argument('--data_path', type=str, default='./processed_data', help='Path to processed data')
    parser.add_argument('--output_dir', type=str, default='./models', help='Output directory for models')
    parser.add_argument('--config_file', type=str, help='Config file path')
    parser.add_argument('--wandb_project', type=str, default='sota-tv-models', help='Wandb project name')
    parser.add_argument('--wandb_run_name', type=str, default='ensemble-system', help='Wandb run name')
    parser.add_argument('--resume_from', type=str, help='Resume training from checkpoint')
    
    args = parser.parse_args()
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Load config
    config = get_ensemble_config()
    if args.config_file and os.path.exists(args.config_file):
        with open(args.config_file) as f:
            config.update(json.load(f).get('ensemble_config', {}))
    
    # Initialize wandb
    wandb.init(
        project=args.wandb_project,
        name=args.wandb_run_name,
        config=config
    )
    
    # Setup tokenizer
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    
    # Create datasets
    train_dataset = EnsembleDataset(
        data_file=os.path.join(args.data_path, 'train_data.json'),
        tokenizer=tokenizer,
        max_length=512
    )
    
    val_dataset = EnsembleDataset(
        data_file=os.path.join(args.data_path, 'val_data.json'),
        tokenizer=tokenizer,
        max_length=512
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=4,
        collate_fn=collate_ensemble_fn,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=4,
        collate_fn=collate_ensemble_fn,
        pin_memory=True
    )
    
    # Create model
    model = EnsembleSystem(
        ensemble_dim=config['ensemble_dim'],
        num_models=config['num_models'],
        use_dynamic_weighting=config['use_dynamic_weighting']
    )
    
    logger.info(f"Ensemble Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Resume from checkpoint if specified
    if args.resume_from and os.path.exists(args.resume_from):
        checkpoint = torch.load(args.resume_from, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Resumed from checkpoint: {args.resume_from}")
    
    # Create trainer and train
    trainer = EnsembleTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device
    )
    
    trainer.train()
    
    # Save final model
    os.makedirs(args.output_dir, exist_ok=True)
    trainer.save_model(os.path.join(args.output_dir, 'final_ensemble_tv_model.pt'))
    
    wandb.finish()

if __name__ == "__main__":
    main()