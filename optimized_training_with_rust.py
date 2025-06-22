#!/usr/bin/env python3
"""
Optimized Training Pipeline for CineSync v2 with Rust Data Loading and Discord Notifications
Combines the high-performance patterns from chef-genius with CineSync's recommendation models
"""

import os
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
import threading
import gc
import json
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import logging
from datetime import datetime

# Import CineSync specific modules
sys.path.append('.')
sys.path.append('./hybrid_recommendation_movie')
sys.path.append('./hybrid_recommendation_tv')

from fast_dataloader import CineSyncFastDataLoader, DataLoaderConfig, create_optimized_dataloader
from discord_notifications import CineSyncDiscordAlerter, TrainingConfig

# Import monitoring dependencies
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("⚠️  W&B not available. Install with: pip install wandb")

try:
    import psutil
    import GPUtil
    SYSTEM_MONITORING_AVAILABLE = True
except ImportError:
    SYSTEM_MONITORING_AVAILABLE = False
    print("⚠️  System monitoring not available. Install with: pip install psutil gputil")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class CineSyncHardwareConfig:
    """Hardware optimization configuration for CineSync training."""
    # RTX 4090 optimizations
    batch_size: int = 256
    gradient_accumulation_steps: int = 4
    mixed_precision: bool = True
    
    # CPU optimizations (assumes high-end CPU)
    cpu_threads: int = 16
    num_workers: int = 8
    
    # Memory optimizations
    pin_memory: bool = True
    prefetch_factor: int = 4
    
    # Training optimizations
    compile_model: bool = True
    use_fused_optimizer: bool = True

class TrainingLogger:
    """W&B logging with step management for CineSync."""
    
    def __init__(self, use_wandb: bool = True, project_name: str = "cine-sync-v2-optimized"):
        self.use_wandb = use_wandb and WANDB_AVAILABLE
        self.current_step = 0
        self._step_lock = threading.Lock()
        
        if self.use_wandb:
            try:
                run_name = f"cinesync-training-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
                
                wandb.init(
                    project=project_name,
                    name=run_name,
                    mode="online",
                    settings=wandb.Settings(console='off', start_method='thread'),
                    tags=["recommendation", "hybrid-model", "rust-dataloader", "optimized"]
                )
                print("📊 W&B logging initialized")
            except Exception as e:
                print(f"⚠️  W&B initialization failed: {e}")
                self.use_wandb = False
    
    def set_step(self, step: int):
        """Set the current training step."""
        with self._step_lock:
            self.current_step = step
    
    def log_metrics(self, metrics: Dict, step: int = None, commit: bool = True):
        """Log metrics with proper step management."""
        if not self.use_wandb or not metrics:
            return
            
        with self._step_lock:
            log_step = step if step is not None else self.current_step
            
            try:
                wandb.log(metrics, step=log_step, commit=commit)
            except Exception as e:
                print(f"⚠️  W&B logging failed: {e}")

class SystemMonitor:
    """System monitoring for hardware metrics."""
    
    def __init__(self, training_logger=None):
        self.training_logger = training_logger
        self.monitoring_available = SYSTEM_MONITORING_AVAILABLE
        self.last_metrics = {}
        
    def get_system_metrics(self) -> Dict[str, float]:
        """Get current system metrics."""
        if not self.monitoring_available:
            return {}
        
        metrics = {}
        
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            
            metrics.update({
                'system/cpu_percent': cpu_percent,
                'system/memory_percent': memory.percent,
                'system/memory_available_gb': memory.available / (1024**3),
            })
            
            # GPU metrics
            if torch.cuda.is_available():
                gpu_memory_allocated = torch.cuda.memory_allocated() / (1024**3)
                gpu_memory_reserved = torch.cuda.memory_reserved() / (1024**3)
                gpu_max_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                
                metrics.update({
                    'system/gpu_memory_allocated_gb': gpu_memory_allocated,
                    'system/gpu_memory_reserved_gb': gpu_memory_reserved,
                    'system/gpu_memory_percent': (gpu_memory_allocated / gpu_max_memory) * 100,
                })
                
                # GPU utilization
                try:
                    gpus = GPUtil.getGPUs()
                    if gpus:
                        gpu = gpus[0]  # First GPU
                        metrics.update({
                            'system/gpu_utilization': gpu.load * 100,
                            'system/gpu_temperature': gpu.temperature,
                        })
                except:
                    pass
            
        except Exception as e:
            print(f"⚠️  System monitoring error: {e}")
        
        self.last_metrics = metrics
        return metrics

class HybridRecommenderModel(nn.Module):
    """Simplified hybrid recommender for demonstration."""
    
    def __init__(self, num_users: int, num_items: int, embedding_dim: int = 128, hidden_dim: int = 256):
        super().__init__()
        
        # User and item embeddings
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        
        # Content features
        self.content_mlp = nn.Sequential(
            nn.Linear(4, hidden_dim),  # content_type, year, genre_count, extra_features
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, embedding_dim)
        )
        
        # Final prediction layers
        self.prediction_mlp = nn.Sequential(
            nn.Linear(embedding_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # Initialize embeddings
        nn.init.normal_(self.user_embedding.weight, 0, 0.1)
        nn.init.normal_(self.item_embedding.weight, 0, 0.1)
    
    def forward(self, user_ids, item_ids, content_features):
        """Forward pass through the hybrid model."""
        # Get embeddings
        user_emb = self.user_embedding(user_ids)
        item_emb = self.item_embedding(item_ids)
        
        # Process content features
        content_emb = self.content_mlp(content_features)
        
        # Combine all features
        combined = torch.cat([user_emb, item_emb, content_emb], dim=1)
        
        # Predict rating (scaled to 1-5 range)
        prediction = self.prediction_mlp(combined) * 4 + 1
        
        return prediction.squeeze()

class CineSyncOptimizedTrainer:
    """
    Complete training pipeline for CineSync with:
    - Rust data loading for maximum performance
    - RTX 4090 + high-end CPU optimizations
    - Discord + SMS notifications
    - W&B monitoring
    - System monitoring
    - Memory management
    """
    
    def __init__(self, 
                 data_paths: Dict[str, str],
                 output_dir: str = "./models/cinesync_optimized",
                 discord_webhook: str = None,
                 alert_phone: str = None,
                 wandb_project: str = "cine-sync-v2-optimized",
                 use_wandb: bool = True):
        
        self.data_paths = data_paths
        self.output_dir = output_dir
        
        # Initialize configurations
        self.hw_config = CineSyncHardwareConfig()
        
        # Initialize monitoring
        self.training_logger = TrainingLogger(use_wandb=use_wandb, project_name=wandb_project)
        self.system_monitor = SystemMonitor(self.training_logger)
        self.alerter = CineSyncDiscordAlerter(webhook_url=discord_webhook, phone_number=alert_phone)
        
        # Training state
        self.start_time = None
        self.epoch_losses = []
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        
        # Model components (will be initialized after data loading)
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.scaler = None
        
        # Setup hardware optimizations
        self.setup_hardware_optimizations()
        
        # Create data loader
        self.data_loader = self.create_data_loader()
        
    def setup_hardware_optimizations(self):
        """Setup hardware optimizations for high-performance training."""
        
        # CUDA optimizations
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
            torch.cuda.empty_cache()
            
            # Memory optimization settings
            os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True,max_split_size_mb:128'
            
        # CPU optimizations
        torch.set_num_threads(self.hw_config.cpu_threads)
        os.environ['OMP_NUM_THREADS'] = str(self.hw_config.cpu_threads)
        os.environ['MKL_NUM_THREADS'] = str(self.hw_config.cpu_threads)
        
        print(f"🔧 Hardware optimizations enabled:")
        print(f"   CPU: {self.hw_config.cpu_threads} threads")
        print(f"   GPU: RTX 4090 with TF32 + mixed precision")
        print(f"   Batch size: {self.hw_config.batch_size}")
        print(f"   Memory: Optimized allocation + expandable segments")
    
    def create_data_loader(self):
        """Create the optimized data loader using Rust backend."""
        
        config = DataLoaderConfig(
            batch_size=self.hw_config.batch_size,
            shuffle=True,
            use_rust=True,
            buffer_size=self.hw_config.batch_size * 4,
            num_prefetch_threads=self.hw_config.num_workers
        )
        
        loader = CineSyncFastDataLoader(config)
        load_counts = loader.load_datasets(self.data_paths)
        
        print(f"📊 Data loaded: {load_counts}")
        print(f"🚀 Using {'Rust' if loader.using_rust else 'Python'} backend")
        
        return loader
        
    def initialize_model(self):
        """Initialize the model based on data statistics."""
        
        # Get data summary to determine model dimensions
        data_summary = self.data_loader.get_data_summary()
        
        # Calculate max user and item IDs (simplified approach)
        max_user_id = 100000  # You would calculate this from your data
        max_item_id = 200000  # You would calculate this from your data
        
        # Create model
        self.model = HybridRecommenderModel(
            num_users=max_user_id + 1,
            num_items=max_item_id + 1,
            embedding_dim=128,
            hidden_dim=256
        )
        
        # Move to GPU
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(device)
        
        # Compile model for better performance (PyTorch 2.0+)
        if self.hw_config.compile_model and hasattr(torch, 'compile'):
            try:
                self.model = torch.compile(self.model)
                print("   Model compiled for better performance")
            except:
                print("   Model compilation not available")
        
        # Setup optimizer
        if self.hw_config.use_fused_optimizer:
            try:
                self.optimizer = torch.optim.AdamW(
                    self.model.parameters(),
                    lr=0.001,
                    weight_decay=0.01,
                    fused=True
                )
                print("   Using fused AdamW optimizer")
            except:
                self.optimizer = torch.optim.AdamW(
                    self.model.parameters(),
                    lr=0.001,
                    weight_decay=0.01
                )
        else:
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=0.001,
                weight_decay=0.01
            )
        
        # Setup scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
        
        # Setup mixed precision scaler
        if self.hw_config.mixed_precision:
            self.scaler = torch.amp.GradScaler()
        
        print(f"📱 Model initialized: {sum(p.numel() for p in self.model.parameters()):,} parameters")
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        epoch_losses = []
        epoch_start_time = time.time()
        
        # Create batches for this epoch
        batches = self.data_loader.create_batches()
        total_batches = len(batches)
        
        self.optimizer.zero_grad()
        
        for batch_idx, batch in enumerate(batches):
            # Convert batch to tensors
            batch_tensor = torch.tensor(batch, dtype=torch.float32)
            
            if torch.cuda.is_available():
                batch_tensor = batch_tensor.cuda()
            
            # Extract features
            user_ids = batch_tensor[:, 0].long()
            item_ids = batch_tensor[:, 1].long()
            ratings = batch_tensor[:, 2]
            content_features = batch_tensor[:, 3:]  # content_type, year, genre_count
            
            # Forward pass with mixed precision
            if self.hw_config.mixed_precision and self.scaler:
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    predictions = self.model(user_ids, item_ids, content_features)
                    loss = nn.MSELoss()(predictions, ratings)
                    loss = loss / self.hw_config.gradient_accumulation_steps
                
                # Backward pass
                self.scaler.scale(loss).backward()
                
                if (batch_idx + 1) % self.hw_config.gradient_accumulation_steps == 0:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
            else:
                predictions = self.model(user_ids, item_ids, content_features)
                loss = nn.MSELoss()(predictions, ratings)
                loss = loss / self.hw_config.gradient_accumulation_steps
                
                loss.backward()
                
                if (batch_idx + 1) % self.hw_config.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()
                    self.optimizer.zero_grad()
            
            epoch_losses.append(loss.item() * self.hw_config.gradient_accumulation_steps)
            
            # Log progress
            if batch_idx % 100 == 0 and batch_idx > 0:
                avg_loss = np.mean(epoch_losses[-100:])
                print(f"Epoch {epoch}, Batch {batch_idx}/{total_batches}, Loss: {avg_loss:.4f}")
                
                # Log to W&B
                if self.training_logger.use_wandb:
                    step = epoch * total_batches + batch_idx
                    metrics = {
                        'train/batch_loss': avg_loss,
                        'train/learning_rate': self.optimizer.param_groups[0]['lr'],
                        'train/epoch_progress': batch_idx / total_batches
                    }
                    
                    # Add system metrics
                    system_metrics = self.system_monitor.get_system_metrics()
                    metrics.update(system_metrics)
                    
                    self.training_logger.log_metrics(metrics, step=step)
        
        # Calculate epoch metrics
        epoch_time = time.time() - epoch_start_time
        avg_epoch_loss = np.mean(epoch_losses)
        
        # Reset data loader for next epoch
        self.data_loader.reset_epoch()
        
        return {
            'train_loss': avg_epoch_loss,
            'epoch_time': epoch_time,
            'samples_per_sec': len(batches) * self.hw_config.batch_size / epoch_time
        }
    
    def train_complete_optimized(self, epochs: int = 50, patience: int = 10):
        """Run complete optimized training with full monitoring."""
        
        self.start_time = time.time()
        
        # Initialize model
        self.initialize_model()
        
        # Get initial data statistics
        data_stats = self.data_loader.get_performance_stats()
        data_summary = self.data_loader.get_data_summary()
        
        # Create training configuration for notifications
        training_config = TrainingConfig(
            model_name="HybridRecommender",
            dataset_name="CineSync Multi-Dataset",
            epochs=epochs,
            batch_size=self.hw_config.batch_size,
            learning_rate=0.001,
            optimizer="AdamW"
        )
        
        # Log initial configuration to W&B
        if self.training_logger.use_wandb:
            config_metrics = {
                'config/batch_size': self.hw_config.batch_size,
                'config/gradient_accumulation_steps': self.hw_config.gradient_accumulation_steps,
                'config/effective_batch_size': self.hw_config.batch_size * self.hw_config.gradient_accumulation_steps,
                'config/cpu_threads': self.hw_config.cpu_threads,
                'config/total_samples': data_stats.get('total_samples', 0),
                'config/model_parameters': sum(p.numel() for p in self.model.parameters()),
                'config/mixed_precision': self.hw_config.mixed_precision,
                'config/rust_dataloader': self.data_loader.using_rust,
            }
            self.training_logger.log_metrics(config_metrics, step=0)
        
        # Send training started notification
        self.alerter.training_started(training_config, data_stats)
        
        # Training loop
        try:
            for epoch in range(epochs):
                self.current_epoch = epoch + 1
                
                # Train epoch
                epoch_metrics = self.train_epoch(epoch + 1)
                
                # Update learning rate scheduler
                self.scheduler.step(epoch_metrics['train_loss'])
                
                # Track losses
                self.epoch_losses.append(epoch_metrics['train_loss'])
                
                print(f"Epoch {self.current_epoch}/{epochs}: "
                      f"Loss {epoch_metrics['train_loss']:.4f}, "
                      f"Time {epoch_metrics['epoch_time']:.1f}s, "
                      f"Speed {epoch_metrics['samples_per_sec']:.1f} samples/sec")
                
                # Log epoch metrics to W&B
                if self.training_logger.use_wandb:
                    wandb_metrics = {
                        'train/epoch_loss': epoch_metrics['train_loss'],
                        'train/epoch_time': epoch_metrics['epoch_time'],
                        'train/samples_per_sec': epoch_metrics['samples_per_sec'],
                        'train/epoch': self.current_epoch,
                    }
                    
                    # Add system metrics
                    system_metrics = self.system_monitor.get_system_metrics()
                    wandb_metrics.update(system_metrics)
                    
                    self.training_logger.log_metrics(wandb_metrics, step=self.current_epoch)
                
                # Send progress notification every 5 epochs
                if self.current_epoch % 5 == 0:
                    progress_metrics = {
                        'train_loss': epoch_metrics['train_loss'],
                        'samples_per_sec': epoch_metrics['samples_per_sec']
                    }
                    
                    self.alerter.training_progress(
                        epoch=self.current_epoch,
                        total_epochs=epochs,
                        metrics=progress_metrics,
                        dataset_stats={'samples_per_sec': epoch_metrics['samples_per_sec']}
                    )
                
                # Save checkpoint
                if self.current_epoch % 10 == 0:
                    self.save_checkpoint(self.current_epoch, epoch_metrics)
                
                # Early stopping check
                if epoch_metrics['train_loss'] < self.best_val_loss:
                    self.best_val_loss = epoch_metrics['train_loss']
                    self.patience_counter = 0
                    
                    # Save best model
                    best_path = f"{self.output_dir}/best_model"
                    os.makedirs(best_path, exist_ok=True)
                    torch.save(self.model.state_dict(), f"{best_path}/model.pt")
                    
                else:
                    self.patience_counter += 1
                    
                    if self.patience_counter >= patience:
                        print(f"Early stopping triggered after {patience} epochs without improvement")
                        self.alerter.early_stopping_triggered(
                            epoch=self.current_epoch,
                            patience=patience,
                            best_metric=self.best_val_loss,
                            metric_name="train_loss"
                        )
                        break
                
                # Cleanup GPU memory
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
                gc.collect()
        
        except Exception as e:
            error_msg = f"Training failed: {str(e)}"
            print(f"❌ {error_msg}")
            self.alerter.training_error(error_msg, epoch=self.current_epoch)
            raise
        
        # Training completed
        total_time = time.time() - self.start_time
        total_hours = total_time / 3600
        
        print(f"Training complete: {total_hours:.2f}h, saved to {self.output_dir}")
        
        # Save final model
        os.makedirs(self.output_dir, exist_ok=True)
        torch.save(self.model.state_dict(), f"{self.output_dir}/final_model.pt")
        
        # Save training metadata
        metadata = {
            'epochs_trained': self.current_epoch,
            'best_loss': self.best_val_loss,
            'total_hours': total_hours,
            'hardware_config': self.hw_config.__dict__,
            'data_stats': data_stats,
        }
        
        with open(f"{self.output_dir}/training_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Final metrics
        final_metrics = {
            'final_train_loss': self.epoch_losses[-1] if self.epoch_losses else 0,
            'best_train_loss': self.best_val_loss,
            'total_epochs': self.current_epoch,
            'total_hours': total_hours,
        }
        
        # Send completion notification
        self.alerter.training_completed(
            duration_hours=total_hours,
            final_metrics=final_metrics,
            model_path=self.output_dir,
            best_metrics={'best_val_loss': self.best_val_loss}
        )
        
        # Final W&B log
        if self.training_logger.use_wandb:
            final_wandb_metrics = {
                'final/total_time_hours': total_hours,
                'final/final_loss': final_metrics['final_train_loss'],
                'final/best_loss': self.best_val_loss,
                'final/total_epochs': self.current_epoch,
            }
            self.training_logger.log_metrics(final_wandb_metrics, step=self.current_epoch)
            
            try:
                wandb.finish()
                print("📊 W&B session completed")
            except:
                pass
    
    def save_checkpoint(self, epoch: int, metrics: Dict[str, float]):
        """Save training checkpoint."""
        checkpoint_dir = f"{self.output_dir}/checkpoint-{epoch}"
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metrics': metrics,
            'best_val_loss': self.best_val_loss,
        }
        
        if self.scaler:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        torch.save(checkpoint, f"{checkpoint_dir}/checkpoint.pt")
        
        # Notify checkpoint saved
        self.alerter.model_checkpoint_saved(
            epoch=epoch,
            metrics=metrics,
            checkpoint_path=checkpoint_dir
        )

def main():
    """Main training function with argument parsing."""
    import argparse
    
    parser = argparse.ArgumentParser(description='CineSync v2 Optimized Training Pipeline')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=256, help='Training batch size')
    parser.add_argument('--model-output', type=str, default='./models/cinesync_optimized', help='Output directory for model')
    parser.add_argument('--movies-path', type=str, help='Path to movie ratings CSV')
    parser.add_argument('--tv-path', type=str, help='Path to TV show ratings CSV')
    parser.add_argument('--alert-phone', type=str, help='Phone number for SMS alerts')
    parser.add_argument('--discord-webhook', type=str, help='Discord webhook URL')
    parser.add_argument('--patience', type=int, default=10, help='Early stopping patience')
    
    args = parser.parse_args()
    
    # Configure data paths
    data_paths = {}
    if args.movies_path:
        data_paths['movies'] = args.movies_path
    if args.tv_path:
        data_paths['tv_shows'] = args.tv_path
    
    if not data_paths:
        # Use default paths
        data_paths = {
            'movies': 'movies/ml-32m/ratings.csv',
            'tv_shows': 'tv/tmdb/TMDB_tv_dataset_v3.csv'
        }
    
    print(f"🎬 Starting CineSync v2 optimized training")
    print(f"📁 Data paths: {data_paths}")
    
    # Create trainer
    trainer = CineSyncOptimizedTrainer(
        data_paths=data_paths,
        output_dir=args.model_output,
        discord_webhook=args.discord_webhook,
        alert_phone=args.alert_phone,
        wandb_project="cine-sync-v2-optimized",
        use_wandb=True
    )
    
    # Start training
    trainer.train_complete_optimized(epochs=args.epochs, patience=args.patience)

if __name__ == "__main__":
    main()