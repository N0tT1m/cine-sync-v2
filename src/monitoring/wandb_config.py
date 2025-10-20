#!/usr/bin/env python3
"""
Weights & Biases (wandb) Configuration and Utilities for CineSync v2
Production-ready wandb integration for all recommendation models
"""

import os
import wandb
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import json
import logging
from datetime import datetime
from dataclasses import dataclass, asdict
import psutil
import platform

logger = logging.getLogger(__name__)


@dataclass
class WandbConfig:
    """Configuration for wandb integration"""
    project: str = "cinesync-v2"
    entity: Optional[str] = None
    name: Optional[str] = None
    tags: List[str] = None
    notes: Optional[str] = None
    config: Dict[str, Any] = None
    mode: str = "online"  # online, offline, disabled
    dir: str = "./wandb"
    group: Optional[str] = None
    job_type: Optional[str] = None
    resume: Optional[str] = None


class WandbManager:
    """
    Production-ready Weights & Biases integration for CineSync v2
    
    Features:
    - Model training tracking
    - Real-time metrics logging
    - Model artifact management
    - Hyperparameter sweeps
    - Production monitoring
    - Cross-model comparison
    """
    
    def __init__(self, config: Optional[WandbConfig] = None):
        """Initialize wandb manager with configuration"""
        self.config = config or WandbConfig()
        self.run = None
        self.is_initialized = False
        self.logger = logging.getLogger(__name__)
        
        # Production monitoring
        self.start_time = None
        self.system_metrics = {}
        
    def init(self, **kwargs):
        """
        Initialize wandb run with comprehensive configuration
        
        Args:
            **kwargs: Additional wandb.init parameters
            
        Returns:
            wandb Run object
        """
        try:
            # Merge config with kwargs
            init_config = {
                'project': self.config.project,
                'entity': self.config.entity,
                'name': self.config.name,
                'tags': self.config.tags or [],
                'notes': self.config.notes,
                'config': self.config.config or {},
                'mode': self.config.mode,
                'dir': self.config.dir,
                'group': self.config.group,
                'job_type': self.config.job_type,
                'resume': self.config.resume
            }
            
            # Remove None values
            init_config = {k: v for k, v in init_config.items() if v is not None}
            init_config.update(kwargs)
            
            # Initialize wandb
            self.run = wandb.init(**init_config)
            self.is_initialized = True
            self.start_time = datetime.now()
            
            # Log system information
            self._log_system_info()
            
            self.logger.info(f"Wandb initialized: {self.run.name} ({self.run.id})")
            return self.run
            
        except Exception as e:
            self.logger.error(f"Failed to initialize wandb: {e}")
            self.is_initialized = False
            return None
    
    def _log_system_info(self):
        """Log system and environment information"""
        try:
            system_info = {
                'system/platform': platform.platform(),
                'system/python_version': platform.python_version(),
                'system/cpu_count': psutil.cpu_count(),
                'system/memory_gb': round(psutil.virtual_memory().total / (1024**3), 2),
                'system/gpu_available': torch.cuda.is_available(),
                'system/gpu_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
                'system/pytorch_version': torch.__version__,
                'system/hostname': platform.node()
            }
            
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    gpu_props = torch.cuda.get_device_properties(i)
                    system_info[f'system/gpu_{i}_name'] = gpu_props.name
                    system_info[f'system/gpu_{i}_memory_gb'] = round(gpu_props.total_memory / (1024**3), 2)
            
            self.system_metrics = system_info
            if self.run:
                self.run.config.update(system_info)
                
        except Exception as e:
            self.logger.warning(f"Failed to log system info: {e}")
    
    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None, commit: bool = True):
        """Log metrics to wandb with step monotonicity handling"""
        if not self.is_initialized:
            return
            
        try:
            # Check for step monotonicity issues
            if step is not None and hasattr(wandb.run, 'step') and wandb.run.step is not None:
                current_wandb_step = wandb.run.step
                if step <= current_wandb_step:
                    # If provided step is not greater than current, use automatic stepping
                    self.logger.warning(f"Step {step} <= current wandb step {current_wandb_step}. Using automatic stepping.")
                    wandb.log(metrics, commit=commit)  # Let wandb auto-increment
                    return
            
            # Log with provided step or auto-increment
            if step is not None:
                wandb.log(metrics, step=step, commit=commit)
            else:
                wandb.log(metrics, commit=commit)  # Auto-increment
                
        except Exception as e:
            self.logger.error(f"Failed to log metrics: {e}")
            # Fallback: try logging without step parameter
            try:
                wandb.log(metrics, commit=commit)
            except Exception as fallback_e:
                self.logger.error(f"Fallback logging also failed: {fallback_e}")
    
    def log_model_architecture(self, model: torch.nn.Module, model_name: str = "model"):
        """Log model architecture and parameters"""
        if not self.is_initialized:
            return
            
        try:
            # Model summary
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            model_info = {
                f'{model_name}/total_parameters': total_params,
                f'{model_name}/trainable_parameters': trainable_params,
                f'{model_name}/model_size_mb': sum(p.numel() * p.element_size() for p in model.parameters()) / (1024**2),
                f'{model_name}/architecture': str(model)
            }
            
            self.run.config.update(model_info)
            self.logger.info(f"Logged {model_name} architecture: {total_params:,} parameters")
            
        except Exception as e:
            self.logger.error(f"Failed to log model architecture: {e}")
    
    def log_dataset_info(self, dataset_info: Dict[str, Any]):
        """Log dataset information"""
        if not self.is_initialized:
            return
            
        try:
            dataset_metrics = {}
            for key, value in dataset_info.items():
                if isinstance(value, (int, float, str)):
                    dataset_metrics[f'dataset/{key}'] = value
                elif isinstance(value, (list, tuple)):
                    dataset_metrics[f'dataset/{key}_count'] = len(value)
                elif isinstance(value, pd.DataFrame):
                    dataset_metrics[f'dataset/{key}_shape'] = f"{value.shape[0]}x{value.shape[1]}"
                    dataset_metrics[f'dataset/{key}_memory_mb'] = round(value.memory_usage(deep=True).sum() / (1024**2), 2)
            
            self.run.config.update(dataset_metrics)
            self.logger.info("Logged dataset information")
            
        except Exception as e:
            self.logger.error(f"Failed to log dataset info: {e}")
    
    def log_hyperparameters(self, hyperparams: Dict[str, Any]):
        """Log hyperparameters"""
        if not self.is_initialized:
            return
            
        try:
            hp_dict = {}
            for key, value in hyperparams.items():
                if isinstance(value, (int, float, str, bool, list, dict)):
                    hp_dict[f'hyperparameters/{key}'] = value
                else:
                    hp_dict[f'hyperparameters/{key}'] = str(value)
            
            self.run.config.update(hp_dict)
            self.logger.info(f"Logged {len(hp_dict)} hyperparameters")
            
        except Exception as e:
            self.logger.error(f"Failed to log hyperparameters: {e}")
    
    def save_model_locally(self, model_path: str, model_name: str, metadata: Optional[Dict] = None):
        """Save model locally to models/ folder instead of WandB artifacts"""
        try:
            # Create models directory if it doesn't exist
            models_dir = Path("models")
            models_dir.mkdir(exist_ok=True)
            
            # Create subdirectory for this model type
            model_type_dir = models_dir / model_name
            model_type_dir.mkdir(exist_ok=True)
            
            # Generate timestamped filename
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            model_filename = f"{model_name}_{timestamp}.pt"
            local_model_path = model_type_dir / model_filename
            
            # Copy model file to local models directory
            import shutil
            shutil.copy2(model_path, local_model_path)
            
            # Copy related files if they exist
            for ext in ['.pkl', '.json', '_metadata.pkl', '_encoders.pkl']:
                related_file = str(model_path).replace('.pt', ext)
                if Path(related_file).exists():
                    local_related_file = str(local_model_path).replace('.pt', ext)
                    shutil.copy2(related_file, local_related_file)
            
            # Save metadata to JSON file
            if metadata:
                metadata_file = str(local_model_path).replace('.pt', '_metadata.json')
                with open(metadata_file, 'w') as f:
                    json.dump(metadata, f, indent=2, default=str)
            
            self.logger.info(f"Saved model locally: {local_model_path}")
            
            # Log the local path to WandB for reference (without uploading the file)
            if self.is_initialized:
                self.log_metrics({
                    f'{model_name}/local_model_path': str(local_model_path),
                    f'{model_name}/model_saved_timestamp': timestamp
                })
            
            return str(local_model_path)
            
        except Exception as e:
            self.logger.error(f"Failed to save model locally: {e}")
            return None
    
    def load_model_artifact(self, artifact_name: str, version: str = "latest") -> Optional[str]:
        """Load model from wandb artifact"""
        if not self.is_initialized:
            return None
            
        try:
            artifact = self.run.use_artifact(f"{artifact_name}:{version}")
            artifact_dir = artifact.download()
            
            # Find model file
            model_files = list(Path(artifact_dir).glob("*.pt"))
            if model_files:
                model_path = str(model_files[0])
                self.logger.info(f"Loaded model artifact: {artifact_name}")
                return model_path
            else:
                self.logger.error(f"No .pt file found in artifact {artifact_name}")
                return None
                
        except Exception as e:
            self.logger.error(f"Failed to load model artifact: {e}")
            return None
    
    def log_training_progress(self, epoch: int, batch: int, loss: float, 
                            metrics: Dict[str, float], lr: float = None):
        """Log training progress with detailed metrics"""
        if not self.is_initialized:
            return
            
        try:
            log_dict = {
                'epoch': epoch,
                'batch': batch,
                'train/loss': loss,
                'train/step': epoch * 1000 + batch  # Approximate step
            }
            
            # Add learning rate
            if lr is not None:
                log_dict['train/learning_rate'] = lr
            
            # Add additional metrics
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    log_dict[f'train/{key}'] = value
            
            # Add system metrics periodically
            if batch % 100 == 0:  # Every 100 batches
                log_dict.update(self._get_system_metrics())
            
            self.log_metrics(log_dict)
            
        except Exception as e:
            self.logger.error(f"Failed to log training progress: {e}")
    
    def log_validation_results(self, epoch: int, val_loss: float, 
                             val_metrics: Dict[str, float]):
        """Log validation results"""
        if not self.is_initialized:
            return
            
        try:
            log_dict = {
                'epoch': epoch,
                'validation/loss': val_loss
            }
            
            for key, value in val_metrics.items():
                if isinstance(value, (int, float)):
                    log_dict[f'validation/{key}'] = value
            
            self.log_metrics(log_dict)
            
        except Exception as e:
            self.logger.error(f"Failed to log validation results: {e}")
    
    def _get_system_metrics(self) -> Dict[str, float]:
        """Get current system metrics"""
        try:
            metrics = {}
            
            # CPU and Memory
            metrics['system/cpu_percent'] = psutil.cpu_percent()
            memory = psutil.virtual_memory()
            metrics['system/memory_percent'] = memory.percent
            metrics['system/memory_used_gb'] = round((memory.total - memory.available) / (1024**3), 2)
            
            # GPU metrics
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    metrics[f'system/gpu_{i}_memory_allocated_gb'] = round(
                        torch.cuda.memory_allocated(i) / (1024**3), 2
                    )
                    metrics[f'system/gpu_{i}_memory_reserved_gb'] = round(
                        torch.cuda.memory_reserved(i) / (1024**3), 2
                    )
            
            # Training time
            if self.start_time:
                elapsed = (datetime.now() - self.start_time).total_seconds()
                metrics['system/training_time_hours'] = round(elapsed / 3600, 2)
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Failed to get system metrics: {e}")
            return {}
    
    def log_inference_metrics(self, model_name: str, inference_time: float, 
                            batch_size: int, accuracy_metrics: Dict[str, float]):
        """Log inference/production metrics"""
        if not self.is_initialized:
            return
            
        try:
            log_dict = {
                f'inference/{model_name}/latency_ms': inference_time * 1000,
                f'inference/{model_name}/throughput_per_sec': batch_size / inference_time,
                f'inference/{model_name}/batch_size': batch_size
            }
            
            for key, value in accuracy_metrics.items():
                log_dict[f'inference/{model_name}/{key}'] = value
            
            self.log_metrics(log_dict)
            
        except Exception as e:
            self.logger.error(f"Failed to log inference metrics: {e}")
    
    def create_model_comparison_table(self, model_results: Dict[str, Dict[str, float]]):
        """Create comparison table for multiple models"""
        if not self.is_initialized:
            return
            
        try:
            # Create table data
            table_data = []
            for model_name, metrics in model_results.items():
                row = {'Model': model_name}
                row.update(metrics)
                table_data.append(row)
            
            # Create wandb table
            table = wandb.Table(
                columns=['Model'] + list(next(iter(model_results.values())).keys()),
                data=[[row[col] for col in table.columns] for row in table_data]
            )
            
            self.log_metrics({'model_comparison': table})
            self.logger.info("Created model comparison table")
            
        except Exception as e:
            self.logger.error(f"Failed to create model comparison table: {e}")
    
    def finish(self):
        """Finish wandb run and cleanup"""
        try:
            if self.is_initialized and self.run:
                # Log final system metrics
                final_metrics = self._get_system_metrics()
                if final_metrics:
                    self.log_metrics(final_metrics)
                
                # Finish run
                wandb.finish()
                self.logger.info("Wandb run finished successfully")
                
        except Exception as e:
            self.logger.error(f"Error finishing wandb run: {e}")
        finally:
            self.is_initialized = False
            self.run = None


def create_sweep_config(model_type: str) -> Dict[str, Any]:
    """Create hyperparameter sweep configuration for different models"""
    
    base_config = {
        'method': 'bayes',  # bayes, grid, random
        'metric': {
            'name': 'validation/rmse',
            'goal': 'minimize'
        },
        'early_terminate': {
            'type': 'hyperband',
            'min_iter': 5
        }
    }
    
    if model_type == 'ncf':
        base_config['parameters'] = {
            'embedding_dim': {'values': [32, 64, 128, 256]},
            'hidden_layers': {'values': [[64], [128, 64], [256, 128, 64], [512, 256, 128]]},
            'dropout': {'min': 0.1, 'max': 0.5},
            'learning_rate': {'min': 0.0001, 'max': 0.01},
            'batch_size': {'values': [32, 64, 128, 256]},
            'alpha': {'min': 0.3, 'max': 0.7}
        }
    
    elif model_type == 'two_tower':
        base_config['parameters'] = {
            'embedding_dim': {'values': [128, 256, 512]},
            'hidden_dim': {'values': [256, 512, 1024]},
            'num_heads': {'values': [4, 8, 16]},
            'num_layers': {'values': [2, 4, 6]},
            'dropout': {'min': 0.1, 'max': 0.4},
            'learning_rate': {'min': 0.0001, 'max': 0.005},
            'batch_size': {'values': [16, 32, 64]}
        }
    
    elif model_type == 'sequential':
        base_config['parameters'] = {
            'embedding_dim': {'values': [64, 128, 256]},
            'hidden_dim': {'values': [128, 256, 512]},
            'num_layers': {'values': [2, 4, 6]},
            'num_heads': {'values': [4, 8]},
            'max_seq_len': {'values': [50, 100, 200]},
            'dropout': {'min': 0.1, 'max': 0.4},
            'learning_rate': {'min': 0.0001, 'max': 0.005},
            'batch_size': {'values': [32, 64, 128]}
        }
    
    else:  # hybrid
        base_config['parameters'] = {
            'embedding_dim': {'values': [32, 64, 128]},
            'hidden_dim': {'values': [64, 128, 256]},
            'dropout': {'min': 0.1, 'max': 0.4},
            'learning_rate': {'min': 0.0001, 'max': 0.01},
            'batch_size': {'values': [32, 64, 128, 256]}
        }
    
    return base_config


def run_sweep(model_type: str, train_function: callable, count: int = 20):
    """Run hyperparameter sweep"""
    try:
        # Create sweep configuration
        sweep_config = create_sweep_config(model_type)
        
        # Initialize sweep
        sweep_id = wandb.sweep(
            sweep_config, 
            project=f"cinesync-v2-{model_type}-sweep"
        )
        
        # Start sweep agent
        wandb.agent(sweep_id, train_function, count=count)
        
        logger.info(f"Completed sweep for {model_type}: {sweep_id}")
        return sweep_id
        
    except Exception as e:
        logger.error(f"Failed to run sweep: {e}")
        return None


# Utility functions for easy integration
def init_wandb_for_training(model_name: str, config: Dict[str, Any], 
                          resume: bool = False) -> WandbManager:
    """Initialize wandb for model training"""
    
    wandb_config = WandbConfig(
        project=f"cinesync-v2-{model_name}",
        name=f"{model_name}-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
        tags=[model_name, "training"],
        job_type="train",
        config=config,
        resume="auto" if resume else None
    )
    
    manager = WandbManager(wandb_config)
    manager.init()
    
    return manager


def init_wandb_for_inference(model_name: str) -> WandbManager:
    """Initialize wandb for inference monitoring"""
    
    wandb_config = WandbConfig(
        project=f"cinesync-v2-production",
        name=f"{model_name}-inference-{datetime.now().strftime('%Y%m%d')}",
        tags=[model_name, "inference", "production"],
        job_type="inference",
        mode="online"
    )
    
    manager = WandbManager(wandb_config)
    manager.init()
    
    return manager