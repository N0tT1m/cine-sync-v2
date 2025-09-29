"""
Training script for Graph Neural Network TV Recommender (GNN-TV)
Optimized for RTX 4090 with PyTorch Geometric
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
from torch_geometric.data import HeteroData
from torch_geometric.loader import NeighborLoader
from torch.cuda.amp import GradScaler, autocast
import wandb
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from tqdm.auto import tqdm

# Add parent directory to path to import models
sys.path.append(str(Path(__file__).parent.parent))
from models.graph_neural_network import TVGraphRecommender, GraphLoss, get_gnn_config

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TVGraphDataset:
    """Dataset wrapper for TV Graph data"""
    
    def __init__(self, processed_data_dir: str, device: str = 'cuda'):
        self.processed_data_dir = Path(processed_data_dir)
        self.device = device
        
        # Check if graph data exists, if not run preprocessing
        graph_data_path = self.processed_data_dir / 'graph_data.pt'
        if not graph_data_path.exists():
            logger.error(f"Graph data file not found: {graph_data_path}")
            logger.info("Please run data preprocessing first:")
            logger.info("python sota_tv_models/data/tv_preprocessor.py --output_dir sota_tv_outputs/processed_data")
            raise FileNotFoundError(f"Graph data file not found: {graph_data_path}. Please run preprocessing first.")
        
        # Load graph data
        self.graph_data = torch.load(graph_data_path, map_location=device)
        
        # Check and load metadata
        metadata_path = self.processed_data_dir / 'metadata.json'
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}. Please run preprocessing first.")
        
        with open(metadata_path) as f:
            self.metadata = json.load(f)
        
        # Check and load training data for node features
        train_data_path = self.processed_data_dir / 'train_data.json'
        if not train_data_path.exists():
            raise FileNotFoundError(f"Training data file not found: {train_data_path}. Please run preprocessing first.")
            
        with open(train_data_path) as f:
            self.train_data = json.load(f)
        
        val_data_path = self.processed_data_dir / 'val_data.json'
        if not val_data_path.exists():
            raise FileNotFoundError(f"Validation data file not found: {val_data_path}. Please run preprocessing first.")
            
        with open(val_data_path) as f:
            self.val_data = json.load(f)
        
        logger.info(f"Loaded graph with {self.metadata['vocab_sizes']} vocabulary sizes")
        
    def create_hetero_data(self) -> HeteroData:
        """Create HeteroData object for PyTorch Geometric"""
        
        data = HeteroData()
        
        # Node features (using simple node indices for now)
        data['show'].x = torch.arange(self.graph_data['num_shows'], device=self.device)
        data['genre'].x = torch.arange(self.metadata['vocab_sizes']['genres'], device=self.device)
        data['network'].x = torch.arange(self.metadata['vocab_sizes']['networks'], device=self.device) 
        data['creator'].x = torch.arange(self.metadata['vocab_sizes']['creators'], device=self.device)
        
        # Add edge indices
        for edge_type, edge_index in self.graph_data['edge_index_dict'].items():
            data[edge_type].edge_index = edge_index.to(self.device)
        
        return data
    
    def create_training_pairs(self, num_pairs: int = 10000):
        """Create positive and negative pairs for training"""
        
        num_shows = self.graph_data['num_shows']
        
        # Create positive pairs (similar shows based on genres)
        positive_pairs = []
        labels = []
        
        # Get genre similarity edges
        if ('show', 'similar_to', 'show') in self.graph_data['edge_index_dict']:
            similar_edges = self.graph_data['edge_index_dict'][('show', 'similar_to', 'show')]
            for i in range(min(num_pairs // 2, similar_edges.size(1))):
                src, dst = similar_edges[:, i]
                positive_pairs.append([src.item(), dst.item()])
                labels.append(1)
        
        # Create negative pairs (random)
        negative_pairs = []
        for _ in range(num_pairs // 2):
            src = np.random.randint(0, num_shows)
            dst = np.random.randint(0, num_shows)
            if src != dst:
                negative_pairs.append([src, dst])
                labels.append(0)
        
        all_pairs = positive_pairs + negative_pairs
        all_labels = labels
        
        # Shuffle
        indices = np.random.permutation(len(all_pairs))
        all_pairs = [all_pairs[i] for i in indices]
        all_labels = [all_labels[i] for i in indices]
        
        return torch.tensor(all_pairs), torch.tensor(all_labels, dtype=torch.float)

class GNNTrainer:
    """Trainer for TV Graph Neural Network"""
    
    def __init__(self, 
                 model: TVGraphRecommender,
                 dataset: TVGraphDataset,
                 config: dict,
                 device: torch.device):
        
        self.model = model.to(device)
        self.dataset = dataset
        self.config = config
        self.device = device
        
        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config.get('weight_decay', 1e-5)
        )
        
        # Scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=10, T_mult=2
        )
        
        # Loss function
        self.criterion = GraphLoss(
            similarity_weight=config.get('similarity_weight', 1.0),
            contrastive_weight=config.get('contrastive_weight', 0.5),
            genre_weight=config.get('genre_weight', 0.3),
            network_weight=config.get('network_weight', 0.2)
        )
        
        # Mixed precision
        self.scaler = GradScaler() if config.get('use_mixed_precision', True) else None
        
        # Create HeteroData
        self.hetero_data = dataset.create_hetero_data()
        
        # Metrics
        self.best_val_auc = 0
        self.patience_counter = 0
        
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        
        total_loss = 0
        num_batches = 50  # Number of mini-batches per epoch
        
        progress_bar = tqdm(range(num_batches), desc="Training GNN")
        
        for batch_idx in progress_bar:
            # Create training pairs for this batch
            show_pairs, similarity_labels = self.dataset.create_training_pairs(
                num_pairs=self.config['batch_size']
            )
            show_pairs = show_pairs.to(self.device)
            similarity_labels = similarity_labels.to(self.device)
            
            # Create auxiliary labels (genre and network prediction)
            batch_size = show_pairs.size(0)
            show_indices = show_pairs[:, 0]  # Use first show in each pair
            
            # Get genre labels for shows (multi-hot encoding)
            genre_labels = torch.zeros(batch_size, self.dataset.metadata['vocab_sizes']['genres'])
            
            # For simplicity, create dummy genre labels based on show index
            for i, show_idx in enumerate(show_indices):
                if show_idx < len(self.dataset.train_data):
                    try:
                        show_genres = self.dataset.train_data[show_idx]['categorical_features']['genres']
                        for genre_id in show_genres[:5]:  # Limit to 5 genres
                            if 0 < genre_id < genre_labels.size(1):
                                genre_labels[i, genre_id] = 1
                    except:
                        pass
            
            genre_labels = genre_labels.to(self.device)
            
            # Network labels (single label)
            network_labels = torch.zeros(batch_size, dtype=torch.long)
            for i, show_idx in enumerate(show_indices):
                if show_idx < len(self.dataset.train_data):
                    try:
                        show_networks = self.dataset.train_data[show_idx]['categorical_features']['networks']
                        if show_networks and show_networks[0] > 0:
                            network_labels[i] = show_networks[0]
                    except:
                        pass
            
            network_labels = network_labels.to(self.device)
            
            # Forward pass
            if self.scaler:
                with autocast():
                    outputs = self.model(
                        x_dict=self.hetero_data.x_dict,
                        edge_index_dict=self.hetero_data.edge_index_dict,
                        show_pairs=show_pairs,
                        target_genres=genre_labels,
                        target_networks=network_labels
                    )
                    
                    # Create negative samples for contrastive loss
                    negative_logits = torch.randn(batch_size, 5, device=self.device)  # Dummy negatives
                    
                    loss_dict = self.criterion(
                        outputs=outputs,
                        similarity_labels=similarity_labels,
                        genre_labels=genre_labels,
                        network_labels=network_labels,
                        negative_samples=negative_logits
                    )
                    
                    loss = loss_dict['total_loss']
                
                # Backward pass
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.get('gradient_clip', 1.0))
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
            
            else:
                outputs = self.model(
                    x_dict=self.hetero_data.x_dict,
                    edge_index_dict=self.hetero_data.edge_index_dict,
                    show_pairs=show_pairs,
                    target_genres=genre_labels,
                    target_networks=network_labels
                )
                
                negative_logits = torch.randn(batch_size, 5, device=self.device)
                
                loss_dict = self.criterion(
                    outputs=outputs,
                    similarity_labels=similarity_labels,
                    genre_labels=genre_labels,
                    network_labels=network_labels,
                    negative_samples=negative_logits
                )
                
                loss = loss_dict['total_loss']
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.get('gradient_clip', 1.0))
                self.optimizer.step()
                self.optimizer.zero_grad()
            
            total_loss += loss.item()
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'avg_loss': f"{total_loss/(batch_idx+1):.4f}"
            })
            
            # Log to wandb
            if batch_idx % 10 == 0:
                wandb.log({
                    'gnn_train_loss': loss.item(),
                    'learning_rate': self.scheduler.get_last_lr()[0],
                })
        
        self.scheduler.step()
        return total_loss / num_batches
    
    def validate(self):
        """Validate the model"""
        self.model.eval()
        
        val_pairs, val_labels = self.dataset.create_training_pairs(num_pairs=1000)
        val_pairs = val_pairs.to(self.device)
        val_labels = val_labels.to(self.device)
        
        with torch.no_grad():
            outputs = self.model(
                x_dict=self.hetero_data.x_dict,
                edge_index_dict=self.hetero_data.edge_index_dict,
                show_pairs=val_pairs
            )
            
            similarities = outputs['cosine_similarity']
            
            # Calculate AUC
            try:
                auc = roc_auc_score(val_labels.cpu().numpy(), similarities.cpu().numpy())
            except:
                auc = 0.5
            
            # Calculate accuracy using threshold
            predictions = (similarities > 0.5).float()
            accuracy = (predictions == val_labels).float().mean()
        
        return auc, accuracy.item()
    
    def train(self):
        """Main training loop"""
        logger.info("Starting GNN training...")
        
        for epoch in range(self.config['epochs']):
            logger.info(f"Epoch {epoch + 1}/{self.config['epochs']}")
            
            # Train
            train_loss = self.train_epoch()
            
            # Validate
            val_auc, val_accuracy = self.validate()
            
            logger.info(f"Train Loss: {train_loss:.4f}, Val AUC: {val_auc:.4f}, Val Acc: {val_accuracy:.4f}")
            
            # Log to wandb
            wandb.log({
                'gnn_epoch': epoch,
                'gnn_train_loss_epoch': train_loss,
                'gnn_val_auc': val_auc,
                'gnn_val_accuracy': val_accuracy
            })
            
            # Save best model
            if val_auc > self.best_val_auc:
                self.best_val_auc = val_auc
                self.patience_counter = 0
                self.save_model(f"best_gnn_tv_model.pt")
                logger.info(f"New best model saved with val_auc: {val_auc:.4f}")
            else:
                self.patience_counter += 1
            
            # Early stopping
            if self.patience_counter >= self.config.get('patience', 15):
                logger.info(f"Early stopping after {epoch + 1} epochs")
                break
        
        logger.info("GNN training completed!")
    
    def save_model(self, filename):
        """Save model checkpoint"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': self.config,
            'best_val_auc': self.best_val_auc
        }
        torch.save(checkpoint, filename)

def main():
    parser = argparse.ArgumentParser(description='Train Graph Neural Network TV Model')
    parser.add_argument('--data_path', type=str, default='./processed_data', help='Path to processed data')
    parser.add_argument('--output_dir', type=str, default='./models', help='Output directory for models')
    parser.add_argument('--config_file', type=str, help='Config file path')
    parser.add_argument('--wandb_project', type=str, default='sota-tv-models', help='Wandb project name')
    parser.add_argument('--wandb_run_name', type=str, default='gnn-tv', help='Wandb run name')
    parser.add_argument('--resume_from', type=str, help='Resume training from checkpoint')
    
    args = parser.parse_args()
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Load config
    config = get_gnn_config()
    if args.config_file and os.path.exists(args.config_file):
        with open(args.config_file) as f:
            config.update(json.load(f).get('gnn_config', {}))
    
    # Add training specific config
    config.update({
        'epochs': 30,
        'patience': 15,
        'similarity_weight': 1.0,
        'contrastive_weight': 0.5,
        'genre_weight': 0.3,
        'network_weight': 0.2
    })
    
    # Initialize wandb
    wandb.init(
        project=args.wandb_project,
        name=args.wandb_run_name,
        config=config
    )
    
    # Create dataset
    dataset = TVGraphDataset(args.data_path, device=device)
    
    # Create model
    model = TVGraphRecommender(
        num_shows=dataset.graph_data['num_shows'],
        num_actors=5000,  # Placeholder
        num_genres=dataset.metadata['vocab_sizes']['genres'],
        num_networks=dataset.metadata['vocab_sizes']['networks'],
        num_creators=dataset.metadata['vocab_sizes']['creators'],
        hidden_dim=config['hidden_dim'],
        num_gnn_layers=config['num_gnn_layers'],
        dropout=config['dropout'],
        use_metapath=config['use_metapath']
    )
    
    logger.info(f"GNN Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Resume from checkpoint if specified
    if args.resume_from and os.path.exists(args.resume_from):
        checkpoint = torch.load(args.resume_from, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Resumed from checkpoint: {args.resume_from}")
    
    # Create trainer and train
    trainer = GNNTrainer(
        model=model,
        dataset=dataset,
        config=config,
        device=device
    )
    
    trainer.train()
    
    # Save final model
    os.makedirs(args.output_dir, exist_ok=True)
    trainer.save_model(os.path.join(args.output_dir, 'final_gnn_tv_model.pt'))
    
    wandb.finish()

if __name__ == "__main__":
    main()