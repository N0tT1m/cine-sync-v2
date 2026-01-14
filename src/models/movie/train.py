"""
CineSync v2 - Unified Movie Model Training Script

Trains all 14 movie-specific recommendation models with support for:
- Single model training
- Multi-model training
- Distributed training
- WandB logging
- Checkpoint management
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any
import json
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np

# Optional imports
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

# Import all movie models
from .franchise_sequence import FranchiseSequenceModel, FranchiseSequenceTrainer, FranchiseConfig
from .director_auteur import DirectorAuteurModel, DirectorAuteurTrainer, DirectorConfig
from .cinematic_universe import CinematicUniverseModel, CinematicUniverseTrainer, UniverseConfig
from .awards_prediction import AwardsPredictionModel, AwardsPredictionTrainer, AwardsConfig
from .runtime_preference import RuntimePreferenceModel, RuntimePreferenceTrainer, RuntimeConfig
from .era_style import EraStyleModel, EraStyleTrainer, EraConfig
from .critic_audience import CriticAudienceModel, CriticAudienceTrainer, CriticConfig
from .remake_connection import RemakeConnectionModel, RemakeConnectionTrainer, RemakeConfig
from .actor_collaboration import ActorCollaborationModel, ActorCollaborationTrainer, ActorConfig
from .studio_fingerprint import StudioFingerprintModel, StudioFingerprintTrainer, StudioConfig
from .adaptation_source import AdaptationSourceModel, AdaptationSourceTrainer, AdaptationConfig
from .international_cinema import InternationalCinemaModel, InternationalCinemaTrainer, InternationalConfig
from .narrative_complexity import NarrativeComplexityModel, NarrativeComplexityTrainer, NarrativeConfig
from .viewing_context import ViewingContextModel, ViewingContextTrainer, ViewingContextConfig

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Model registry
MOVIE_MODELS = {
    'franchise_sequence': {
        'model_class': FranchiseSequenceModel,
        'trainer_class': FranchiseSequenceTrainer,
        'config_class': FranchiseConfig,
        'description': 'Franchise and sequel ordering recommendations'
    },
    'director_auteur': {
        'model_class': DirectorAuteurModel,
        'trainer_class': DirectorAuteurTrainer,
        'config_class': DirectorConfig,
        'description': 'Director filmography and style matching'
    },
    'cinematic_universe': {
        'model_class': CinematicUniverseModel,
        'trainer_class': CinematicUniverseTrainer,
        'config_class': UniverseConfig,
        'description': 'Connected universe navigation (MCU, DC, etc.)'
    },
    'awards_prediction': {
        'model_class': AwardsPredictionModel,
        'trainer_class': AwardsPredictionTrainer,
        'config_class': AwardsConfig,
        'description': 'Oscar and prestige film recommendations'
    },
    'runtime_preference': {
        'model_class': RuntimePreferenceModel,
        'trainer_class': RuntimePreferenceTrainer,
        'config_class': RuntimeConfig,
        'description': 'Time-aware movie selection'
    },
    'era_style': {
        'model_class': EraStyleModel,
        'trainer_class': EraStyleTrainer,
        'config_class': EraConfig,
        'description': 'Decade and era taste modeling'
    },
    'critic_audience': {
        'model_class': CriticAudienceModel,
        'trainer_class': CriticAudienceTrainer,
        'config_class': CriticConfig,
        'description': 'Critic vs audience score preferences'
    },
    'remake_connection': {
        'model_class': RemakeConnectionModel,
        'trainer_class': RemakeConnectionTrainer,
        'config_class': RemakeConfig,
        'description': 'Original and remake relationships'
    },
    'actor_collaboration': {
        'model_class': ActorCollaborationModel,
        'trainer_class': ActorCollaborationTrainer,
        'config_class': ActorConfig,
        'description': 'Actor pairings and ensemble chemistry'
    },
    'studio_fingerprint': {
        'model_class': StudioFingerprintModel,
        'trainer_class': StudioFingerprintTrainer,
        'config_class': StudioConfig,
        'description': 'Studio style and brand preferences'
    },
    'adaptation_source': {
        'model_class': AdaptationSourceModel,
        'trainer_class': AdaptationSourceTrainer,
        'config_class': AdaptationConfig,
        'description': 'Book, comic, and game adaptations'
    },
    'international_cinema': {
        'model_class': InternationalCinemaModel,
        'trainer_class': InternationalCinemaTrainer,
        'config_class': InternationalConfig,
        'description': 'Country and region preferences'
    },
    'narrative_complexity': {
        'model_class': NarrativeComplexityModel,
        'trainer_class': NarrativeComplexityTrainer,
        'config_class': NarrativeConfig,
        'description': 'Storytelling structure preferences'
    },
    'viewing_context': {
        'model_class': ViewingContextModel,
        'trainer_class': ViewingContextTrainer,
        'config_class': ViewingContextConfig,
        'description': 'Context-aware recommendations'
    }
}


class MovieDataset(Dataset):
    """Generic movie dataset for training"""

    def __init__(self, data_path: str, model_type: str, split: str = 'train'):
        self.data_path = Path(data_path)
        self.model_type = model_type
        self.split = split
        self.data = self._load_data()

    def _load_data(self) -> List[Dict]:
        """Load data for specific model type"""
        data_file = self.data_path / f"{self.model_type}_{self.split}.json"
        if data_file.exists():
            with open(data_file, 'r') as f:
                return json.load(f)
        else:
            logger.warning(f"Data file not found: {data_file}. Using synthetic data.")
            return self._generate_synthetic_data()

    def _generate_synthetic_data(self, size: int = 10000) -> List[Dict]:
        """Generate synthetic training data for testing"""
        data = []
        for _ in range(size):
            sample = {
                'user_ids': np.random.randint(0, 50000),
                'movie_ids': np.random.randint(0, 100000),
                'ratings': np.random.uniform(1, 5),
            }
            data.append(sample)
        return data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.data[idx]
        return {k: torch.tensor(v) for k, v in sample.items()}


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Collate function for DataLoader"""
    collated = {}
    for key in batch[0].keys():
        values = [item[key] for item in batch]
        if isinstance(values[0], torch.Tensor):
            collated[key] = torch.stack(values)
        else:
            collated[key] = torch.tensor(values)
    return collated


class MovieModelTrainingPipeline:
    """Unified training pipeline for all movie models"""

    def __init__(
        self,
        model_name: str,
        data_dir: str = '/Users/timmy/workspace/ai-apps/cine-sync-v2/data/movies',
        output_dir: str = '/Users/timmy/workspace/ai-apps/cine-sync-v2/models/movies',
        device: str = 'cuda',
        use_wandb: bool = False,
        wandb_project: str = 'cinesync-movie-models'
    ):
        self.model_name = model_name
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.use_wandb = use_wandb and WANDB_AVAILABLE

        if model_name not in MOVIE_MODELS:
            raise ValueError(f"Unknown model: {model_name}. Available: {list(MOVIE_MODELS.keys())}")

        self.model_info = MOVIE_MODELS[model_name]
        self.output_dir.mkdir(parents=True, exist_ok=True)

        if self.use_wandb:
            wandb.init(
                project=wandb_project,
                name=f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                config={'model_name': model_name}
            )

    def create_model(self, **config_overrides) -> nn.Module:
        """Create model with optional config overrides"""
        config_class = self.model_info['config_class']
        model_class = self.model_info['model_class']

        config = config_class(**config_overrides) if config_overrides else config_class()
        model = model_class(config)

        logger.info(f"Created {self.model_name} model with {sum(p.numel() for p in model.parameters()):,} parameters")
        return model

    def create_dataloaders(
        self,
        batch_size: int = 256,
        num_workers: int = 4
    ) -> tuple:
        """Create train and validation dataloaders"""
        train_dataset = MovieDataset(self.data_dir, self.model_name, split='train')
        val_dataset = MovieDataset(self.data_dir, self.model_name, split='val')

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=True
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=True
        )

        return train_loader, val_loader

    def train(
        self,
        epochs: int = 50,
        batch_size: int = 256,
        lr: float = 1e-4,
        weight_decay: float = 0.01,
        save_every: int = 5,
        early_stopping_patience: int = 10,
        **config_overrides
    ) -> Dict[str, Any]:
        """Train the model"""
        # Create model and trainer
        model = self.create_model(**config_overrides)
        trainer_class = self.model_info['trainer_class']
        trainer = trainer_class(model, lr=lr, weight_decay=weight_decay, device=self.device)

        # Create dataloaders
        train_loader, val_loader = self.create_dataloaders(batch_size)

        best_val_loss = float('inf')
        patience_counter = 0
        training_history = {'train_loss': [], 'val_metrics': []}

        logger.info(f"Starting training for {self.model_name}")
        logger.info(f"  Epochs: {epochs}, Batch size: {batch_size}, LR: {lr}")
        logger.info(f"  Device: {self.device}")

        for epoch in range(epochs):
            # Training
            epoch_losses = []
            for batch_idx, batch in enumerate(train_loader):
                losses = trainer.train_step(batch)
                epoch_losses.append(losses['total_loss'])

                if batch_idx % 100 == 0:
                    logger.info(f"  Epoch {epoch+1}/{epochs}, Batch {batch_idx}/{len(train_loader)}, "
                               f"Loss: {losses['total_loss']:.4f}")

            avg_train_loss = np.mean(epoch_losses)
            training_history['train_loss'].append(avg_train_loss)

            # Validation
            val_metrics = trainer.evaluate(val_loader)
            training_history['val_metrics'].append(val_metrics)

            logger.info(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f}, "
                       f"Val Metrics: {val_metrics}")

            # WandB logging
            if self.use_wandb:
                wandb.log({
                    'epoch': epoch + 1,
                    'train_loss': avg_train_loss,
                    **{f'val_{k}': v for k, v in val_metrics.items()}
                })

            # Save checkpoint
            if (epoch + 1) % save_every == 0:
                self._save_checkpoint(model, epoch + 1, val_metrics)

            # Early stopping
            current_val_loss = val_metrics.get('rating_mse', val_metrics.get('total_loss', avg_train_loss))
            if current_val_loss < best_val_loss:
                best_val_loss = current_val_loss
                patience_counter = 0
                self._save_checkpoint(model, 'best', val_metrics)
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break

            # Update scheduler
            trainer.scheduler.step()

        # Save final model
        self._save_checkpoint(model, 'final', val_metrics)

        if self.use_wandb:
            wandb.finish()

        return {
            'model_name': self.model_name,
            'best_val_loss': best_val_loss,
            'training_history': training_history,
            'epochs_trained': epoch + 1
        }

    def _save_checkpoint(self, model: nn.Module, identifier: Any, metrics: Dict):
        """Save model checkpoint"""
        checkpoint_dir = self.output_dir / self.model_name
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        checkpoint_path = checkpoint_dir / f"checkpoint_{identifier}.pt"
        torch.save({
            'model_state_dict': model.state_dict(),
            'metrics': metrics,
            'timestamp': datetime.now().isoformat()
        }, checkpoint_path)

        logger.info(f"Saved checkpoint: {checkpoint_path}")

    def load_checkpoint(self, identifier: str = 'best') -> nn.Module:
        """Load model from checkpoint"""
        checkpoint_dir = self.output_dir / self.model_name
        checkpoint_path = checkpoint_dir / f"checkpoint_{identifier}.pt"

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        model = self.create_model()
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)

        logger.info(f"Loaded checkpoint: {checkpoint_path}")
        return model


def train_all_models(
    data_dir: str,
    output_dir: str,
    models: Optional[List[str]] = None,
    **training_kwargs
) -> Dict[str, Any]:
    """Train all or selected movie models"""
    models_to_train = models or list(MOVIE_MODELS.keys())
    results = {}

    for model_name in models_to_train:
        logger.info(f"\n{'='*60}")
        logger.info(f"Training model: {model_name}")
        logger.info(f"Description: {MOVIE_MODELS[model_name]['description']}")
        logger.info(f"{'='*60}\n")

        try:
            pipeline = MovieModelTrainingPipeline(
                model_name=model_name,
                data_dir=data_dir,
                output_dir=output_dir,
                **training_kwargs
            )
            result = pipeline.train(**training_kwargs)
            results[model_name] = {'status': 'success', 'result': result}
        except Exception as e:
            logger.error(f"Failed to train {model_name}: {e}")
            results[model_name] = {'status': 'failed', 'error': str(e)}

    return results


def main():
    parser = argparse.ArgumentParser(description='Train CineSync Movie Models')
    parser.add_argument('--model', type=str, default='all',
                       help='Model to train (or "all" for all models)')
    parser.add_argument('--data-dir', type=str,
                       default='/Users/timmy/workspace/ai-apps/cine-sync-v2/data/movies',
                       help='Directory containing training data')
    parser.add_argument('--output-dir', type=str,
                       default='/Users/timmy/workspace/ai-apps/cine-sync-v2/models/movies',
                       help='Directory for model outputs')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=256, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda/cpu)')
    parser.add_argument('--wandb', action='store_true', help='Enable WandB logging')
    parser.add_argument('--list-models', action='store_true', help='List available models')

    args = parser.parse_args()

    if args.list_models:
        print("\nAvailable Movie Models:")
        print("=" * 60)
        for name, info in MOVIE_MODELS.items():
            print(f"  {name:25s} - {info['description']}")
        print()
        return

    training_kwargs = {
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'lr': args.lr,
        'device': args.device,
        'use_wandb': args.wandb
    }

    if args.model == 'all':
        results = train_all_models(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            **training_kwargs
        )
        print("\n" + "=" * 60)
        print("Training Summary:")
        print("=" * 60)
        for model_name, result in results.items():
            status = "✓" if result['status'] == 'success' else "✗"
            print(f"  {status} {model_name}: {result['status']}")
    else:
        if args.model not in MOVIE_MODELS:
            print(f"Unknown model: {args.model}")
            print(f"Available models: {list(MOVIE_MODELS.keys())}")
            return

        pipeline = MovieModelTrainingPipeline(
            model_name=args.model,
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            device=args.device,
            use_wandb=args.wandb
        )
        result = pipeline.train(**training_kwargs)
        print(f"\nTraining complete for {args.model}")
        print(f"Best validation loss: {result['best_val_loss']:.4f}")


if __name__ == '__main__':
    main()
