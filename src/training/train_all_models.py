"""
CineSync v2 - Master Training Script

Unified training script for ALL recommendation models:
- 14 Movie-specific models
- 14 TV-specific models
- 25+ Content-agnostic models (work for both movies and TV)
- 3 Unified cross-domain models

Supports:
- Single model training
- Category-based training (movie/tv/both/unified)
- Full system training
- Distributed training
- WandB logging
- Checkpoint management
"""

import argparse
import logging
import os
import sys
from pathlib import Path

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from typing import Dict, List, Optional, Any, Union
import json
from datetime import datetime
from enum import Enum

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

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ModelCategory(Enum):
    """Model category enumeration"""
    MOVIE_SPECIFIC = "movie"
    TV_SPECIFIC = "tv"
    CONTENT_AGNOSTIC = "both"
    UNIFIED = "unified"


# ============================================================================
# MODEL REGISTRY - All Models Organized by Category
# ============================================================================

MOVIE_SPECIFIC_MODELS = {
    'movie_franchise_sequence': {
        'module': 'src.models.movie.franchise_sequence',
        'model_class': 'FranchiseSequenceModel',
        'trainer_class': 'FranchiseSequenceTrainer',
        'config_class': 'FranchiseConfig',
        'description': 'Franchise and sequel ordering',
        'priority': 5
    },
    'movie_director_auteur': {
        'module': 'src.models.movie.director_auteur',
        'model_class': 'DirectorAuteurModel',
        'trainer_class': 'DirectorAuteurTrainer',
        'config_class': 'DirectorConfig',
        'description': 'Director filmography matching',
        'priority': 4
    },
    'movie_cinematic_universe': {
        'module': 'src.models.movie.cinematic_universe',
        'model_class': 'CinematicUniverseModel',
        'trainer_class': 'CinematicUniverseTrainer',
        'config_class': 'UniverseConfig',
        'description': 'Connected universe navigation',
        'priority': 4
    },
    'movie_awards_prediction': {
        'module': 'src.models.movie.awards_prediction',
        'model_class': 'AwardsPredictionModel',
        'trainer_class': 'AwardsPredictionTrainer',
        'config_class': 'AwardsConfig',
        'description': 'Oscar/prestige recommendations',
        'priority': 3
    },
    'movie_runtime_preference': {
        'module': 'src.models.movie.runtime_preference',
        'model_class': 'RuntimePreferenceModel',
        'trainer_class': 'RuntimePreferenceTrainer',
        'config_class': 'RuntimeConfig',
        'description': 'Time-aware movie selection',
        'priority': 3
    },
    'movie_era_style': {
        'module': 'src.models.movie.era_style',
        'model_class': 'EraStyleModel',
        'trainer_class': 'EraStyleTrainer',
        'config_class': 'EraConfig',
        'description': 'Decade/era taste modeling',
        'priority': 3
    },
    'movie_critic_audience': {
        'module': 'src.models.movie.critic_audience',
        'model_class': 'CriticAudienceModel',
        'trainer_class': 'CriticAudienceTrainer',
        'config_class': 'CriticConfig',
        'description': 'Score alignment preferences',
        'priority': 4
    },
    'movie_remake_connection': {
        'module': 'src.models.movie.remake_connection',
        'model_class': 'RemakeConnectionModel',
        'trainer_class': 'RemakeConnectionTrainer',
        'config_class': 'RemakeConfig',
        'description': 'Original/remake relationships',
        'priority': 3
    },
    'movie_actor_collaboration': {
        'module': 'src.models.movie.actor_collaboration',
        'model_class': 'ActorCollaborationModel',
        'trainer_class': 'ActorCollaborationTrainer',
        'config_class': 'ActorConfig',
        'description': 'Actor pairings and chemistry',
        'priority': 4
    },
    'movie_studio_fingerprint': {
        'module': 'src.models.movie.studio_fingerprint',
        'model_class': 'StudioFingerprintModel',
        'trainer_class': 'StudioFingerprintTrainer',
        'config_class': 'StudioConfig',
        'description': 'Studio style preferences',
        'priority': 3
    },
    'movie_adaptation_source': {
        'module': 'src.models.movie.adaptation_source',
        'model_class': 'AdaptationSourceModel',
        'trainer_class': 'AdaptationSourceTrainer',
        'config_class': 'AdaptationConfig',
        'description': 'Book/comic/game adaptations',
        'priority': 3
    },
    'movie_international': {
        'module': 'src.models.movie.international_cinema',
        'model_class': 'InternationalCinemaModel',
        'trainer_class': 'InternationalCinemaTrainer',
        'config_class': 'InternationalConfig',
        'description': 'Country/region preferences',
        'priority': 3
    },
    'movie_narrative_complexity': {
        'module': 'src.models.movie.narrative_complexity',
        'model_class': 'NarrativeComplexityModel',
        'trainer_class': 'NarrativeComplexityTrainer',
        'config_class': 'NarrativeConfig',
        'description': 'Storytelling structure prefs',
        'priority': 3
    },
    'movie_viewing_context': {
        'module': 'src.models.movie.viewing_context',
        'model_class': 'ViewingContextModel',
        'trainer_class': 'ViewingContextTrainer',
        'config_class': 'ViewingContextConfig',
        'description': 'Context-aware recommendations',
        'priority': 4
    },
}

TV_SPECIFIC_MODELS = {
    'tv_temporal_attention': {
        'module': 'src.models.hybrid.sota_tv.models.temporal_attention',
        'model_class': 'TemporalAttentionTVModel',
        'trainer_class': None,
        'config_class': None,
        'description': 'Temporal viewing patterns',
        'priority': 5
    },
    'tv_graph_neural': {
        'module': 'src.models.hybrid.sota_tv.models.graph_neural_network',
        'model_class': 'TVGraphNeuralNetwork',
        'trainer_class': None,
        'config_class': None,
        'description': 'Graph-based TV recommendations',
        'priority': 4
    },
    'tv_contrastive': {
        'module': 'src.models.hybrid.sota_tv.models.contrastive_learning',
        'model_class': 'ContrastiveTVLearning',
        'trainer_class': None,
        'config_class': None,
        'description': 'Self-supervised TV learning',
        'priority': 4
    },
    'tv_meta_learning': {
        'module': 'src.models.hybrid.sota_tv.models.meta_learning',
        'model_class': 'MetaLearningTVModel',
        'trainer_class': None,
        'config_class': None,
        'description': 'Few-shot adaptation',
        'priority': 4
    },
    'tv_ensemble': {
        'module': 'src.models.hybrid.sota_tv.models.ensemble_system',
        'model_class': 'TVEnsembleSystem',
        'trainer_class': None,
        'config_class': None,
        'description': 'Ensemble of TV models',
        'priority': 5
    },
    'tv_multimodal': {
        'module': 'src.models.hybrid.sota_tv.models.multimodal_transformer',
        'model_class': 'MultimodalTVTransformer',
        'trainer_class': None,
        'config_class': None,
        'description': 'Multimodal TV features',
        'priority': 4
    },
    'tv_episode_sequence': {
        'module': 'src.models.hybrid.sota_tv.models.episode_sequence',
        'model_class': 'EpisodeSequenceModel',
        'trainer_class': 'EpisodeSequenceTrainer',
        'config_class': 'EpisodeSequenceConfig',
        'description': 'Episode-level sequences',
        'priority': 5
    },
    'tv_binge_prediction': {
        'module': 'src.models.hybrid.sota_tv.models.binge_prediction',
        'model_class': 'BingePredictionModel',
        'trainer_class': 'BingePredictionTrainer',
        'config_class': 'BingePredictionConfig',
        'description': 'Binge-watching prediction',
        'priority': 4
    },
    'tv_series_completion': {
        'module': 'src.models.hybrid.sota_tv.models.series_completion',
        'model_class': 'SeriesCompletionModel',
        'trainer_class': 'SeriesCompletionTrainer',
        'config_class': 'SeriesCompletionConfig',
        'description': 'Series completion prediction',
        'priority': 4
    },
    'tv_season_quality': {
        'module': 'src.models.hybrid.sota_tv.models.season_quality',
        'model_class': 'SeasonQualityModel',
        'trainer_class': 'SeasonQualityTrainer',
        'config_class': 'SeasonQualityConfig',
        'description': 'Season quality variance',
        'priority': 3
    },
    'tv_platform_availability': {
        'module': 'src.models.hybrid.sota_tv.models.platform_availability',
        'model_class': 'PlatformAvailabilityModel',
        'trainer_class': 'PlatformAvailabilityTrainer',
        'config_class': 'PlatformConfig',
        'description': 'Streaming platform awareness',
        'priority': 4
    },
    'tv_watch_pattern': {
        'module': 'src.models.hybrid.sota_tv.models.watch_pattern',
        'model_class': 'WatchPatternModel',
        'trainer_class': 'WatchPatternTrainer',
        'config_class': 'WatchPatternConfig',
        'description': 'Viewing pattern modeling',
        'priority': 4
    },
    'tv_series_lifecycle': {
        'module': 'src.models.hybrid.sota_tv.models.series_lifecycle',
        'model_class': 'SeriesLifecycleModel',
        'trainer_class': 'SeriesLifecycleTrainer',
        'config_class': 'LifecycleConfig',
        'description': 'Series lifecycle stages',
        'priority': 3
    },
    'tv_cast_migration': {
        'module': 'src.models.hybrid.sota_tv.models.cast_migration',
        'model_class': 'CastMigrationModel',
        'trainer_class': 'CastMigrationTrainer',
        'config_class': 'CastMigrationConfig',
        'description': 'Cast changes across seasons',
        'priority': 3
    },
}

CONTENT_AGNOSTIC_MODELS = {
    # Collaborative Filtering
    'ncf': {
        'module': 'src.models.collaborative.src.model',
        'model_class': 'NeuralCollaborativeFiltering',
        'trainer_class': None,
        'config_class': None,
        'description': 'Neural Collaborative Filtering',
        'priority': 5
    },
    # Sequential Models
    'sequential_recommender': {
        'module': 'src.models.sequential.src.model',
        'model_class': 'SequentialRecommender',
        'trainer_class': None,
        'config_class': None,
        'description': 'Sequential recommendations',
        'priority': 4
    },
    # Two-Tower Models
    'two_tower': {
        'module': 'src.models.two_tower.src.model',
        'model_class': 'TwoTowerModel',
        'trainer_class': None,
        'config_class': None,
        'description': 'Two-tower retrieval model',
        'priority': 5
    },
    # Advanced Models
    'bert4rec': {
        'module': 'src.models.advanced.bert4rec_recommender',
        'model_class': 'BERT4Rec',
        'trainer_class': None,
        'config_class': None,
        'description': 'BERT-based sequential rec',
        'priority': 5
    },
    'graphsage': {
        'module': 'src.models.advanced.graphsage_recommender',
        'model_class': 'GraphSAGERecommender',
        'trainer_class': None,
        'config_class': None,
        'description': 'GraphSAGE recommendations',
        'priority': 4
    },
    'transformer_recommender': {
        'module': 'src.models.advanced.transformer_recommender',
        'model_class': 'TransformerRecommender',
        'trainer_class': None,
        'config_class': None,
        'description': 'Transformer-based rec',
        'priority': 5
    },
    'vae_recommender': {
        'module': 'src.models.advanced.variational_autoencoder',
        'model_class': 'VAERecommender',
        'trainer_class': None,
        'config_class': None,
        'description': 'Variational autoencoder rec',
        'priority': 4
    },
    'gnn_recommender': {
        'module': 'src.models.advanced.graph_neural_network',
        'model_class': 'GNNRecommender',
        'trainer_class': None,
        'config_class': None,
        'description': 'Graph neural network rec',
        'priority': 4
    },
    'enhanced_two_tower': {
        'module': 'src.models.advanced.enhanced_two_tower',
        'model_class': 'EnhancedTwoTower',
        'trainer_class': None,
        'config_class': None,
        'description': 'Enhanced two-tower model',
        'priority': 4
    },
    'sentence_bert_two_tower': {
        'module': 'src.models.advanced.sentence_bert_two_tower',
        'model_class': 'SentenceBERTTwoTower',
        'trainer_class': None,
        'config_class': None,
        'description': 'Sentence-BERT two-tower',
        'priority': 4
    },
    't5_hybrid': {
        'module': 'src.models.advanced.t5_hybrid_recommender',
        'model_class': 'T5HybridRecommender',
        'trainer_class': None,
        'config_class': None,
        'description': 'T5-based hybrid recommender',
        'priority': 3
    },
    # Hybrid/Unified Content Model
    'unified_content': {
        'module': 'src.models.hybrid.content_recommender',
        'model_class': 'UnifiedContentRecommender',
        'trainer_class': None,
        'config_class': None,
        'description': 'Unified content recommender',
        'priority': 5
    },
}

UNIFIED_MODELS = {
    'cross_domain_embeddings': {
        'module': 'src.models.unified.cross_domain_embeddings',
        'model_class': 'CrossDomainEmbeddings',
        'trainer_class': None,
        'config_class': None,
        'description': 'Cross-domain embeddings',
        'priority': 5
    },
    'movie_ensemble': {
        'module': 'src.models.unified.movie_ensemble_system',
        'model_class': 'MovieEnsembleSystem',
        'trainer_class': None,
        'config_class': None,
        'description': 'Movie ensemble system',
        'priority': 5
    },
    'unified_contrastive': {
        'module': 'src.models.unified.contrastive_learning',
        'model_class': 'UnifiedContrastiveLearning',
        'trainer_class': None,
        'config_class': None,
        'description': 'Unified contrastive learning',
        'priority': 4
    },
    'multimodal_features': {
        'module': 'src.models.unified.multimodal_features',
        'model_class': 'MultimodalFeatures',
        'trainer_class': None,
        'config_class': None,
        'description': 'Multimodal feature extraction',
        'priority': 4
    },
    'context_aware': {
        'module': 'src.models.unified.context_aware',
        'model_class': 'ContextAwareRecommender',
        'trainer_class': None,
        'config_class': None,
        'description': 'Context-aware recommendations',
        'priority': 4
    },
}

# Combined registry
ALL_MODELS = {
    **{k: {**v, 'category': ModelCategory.MOVIE_SPECIFIC} for k, v in MOVIE_SPECIFIC_MODELS.items()},
    **{k: {**v, 'category': ModelCategory.TV_SPECIFIC} for k, v in TV_SPECIFIC_MODELS.items()},
    **{k: {**v, 'category': ModelCategory.CONTENT_AGNOSTIC} for k, v in CONTENT_AGNOSTIC_MODELS.items()},
    **{k: {**v, 'category': ModelCategory.UNIFIED} for k, v in UNIFIED_MODELS.items()},
}


# ============================================================================
# DATASET CLASSES
# ============================================================================

class UnifiedDataset(Dataset):
    """Unified dataset that handles both movie and TV data"""

    def __init__(self, data_dir: str, content_type: str, model_name: str, split: str = 'train'):
        self.data_dir = Path(data_dir)
        self.content_type = content_type
        self.model_name = model_name
        self.split = split
        self.data = self._load_data()

    def _load_data(self) -> List[Dict]:
        """Load data based on content type"""
        if self.content_type == 'movie':
            data_path = self.data_dir / 'movies'
        elif self.content_type == 'tv':
            data_path = self.data_dir / 'tv'
        else:
            # Both - combine movie and TV data
            movie_data = self._load_from_path(self.data_dir / 'movies')
            tv_data = self._load_from_path(self.data_dir / 'tv')
            return movie_data + tv_data

        return self._load_from_path(data_path)

    def _load_from_path(self, data_path: Path) -> List[Dict]:
        """Load data from specific path"""
        data_file = data_path / f"{self.model_name}_{self.split}.json"
        if data_file.exists():
            with open(data_file, 'r') as f:
                return json.load(f)

        # Try generic data file
        generic_file = data_path / f"ratings_{self.split}.json"
        if generic_file.exists():
            with open(generic_file, 'r') as f:
                return json.load(f)

        logger.warning(f"Data file not found: {data_file}. Using synthetic data.")
        return self._generate_synthetic_data()

    def _generate_synthetic_data(self, size: int = 10000) -> List[Dict]:
        """Generate synthetic training data"""
        data = []
        for _ in range(size):
            sample = {
                'user_ids': np.random.randint(0, 50000),
                'item_ids': np.random.randint(0, 100000),
                'ratings': np.random.uniform(1, 5),
            }
            data.append(sample)
        return data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.data[idx]
        return {k: torch.tensor(v) if not isinstance(v, torch.Tensor) else v
                for k, v in sample.items()}


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


# ============================================================================
# TRAINING PIPELINE
# ============================================================================

class UnifiedTrainingPipeline:
    """Unified training pipeline for all model types"""

    def __init__(
        self,
        model_name: str,
        data_dir: str = '/Users/timmy/workspace/ai-apps/cine-sync-v2/data',
        output_dir: str = '/Users/timmy/workspace/ai-apps/cine-sync-v2/models',
        device: str = 'cuda',
        use_wandb: bool = False,
        wandb_project: str = 'cinesync-models'
    ):
        self.model_name = model_name
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.use_wandb = use_wandb and WANDB_AVAILABLE

        if model_name not in ALL_MODELS:
            raise ValueError(f"Unknown model: {model_name}. Use --list-models to see available models.")

        self.model_info = ALL_MODELS[model_name]
        self.category = self.model_info['category']

        # Set output directory based on category
        if self.category == ModelCategory.MOVIE_SPECIFIC:
            self.model_output_dir = self.output_dir / 'movies' / model_name
        elif self.category == ModelCategory.TV_SPECIFIC:
            self.model_output_dir = self.output_dir / 'tv' / model_name
        else:
            self.model_output_dir = self.output_dir / 'unified' / model_name

        self.model_output_dir.mkdir(parents=True, exist_ok=True)

        if self.use_wandb:
            wandb.init(
                project=wandb_project,
                name=f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                config={
                    'model_name': model_name,
                    'category': self.category.value
                }
            )

    def _load_model_class(self):
        """Dynamically load model class"""
        import importlib
        module = importlib.import_module(self.model_info['module'])
        model_class = getattr(module, self.model_info['model_class'])

        trainer_class = None
        config_class = None

        if self.model_info.get('trainer_class'):
            trainer_class = getattr(module, self.model_info['trainer_class'])
        if self.model_info.get('config_class'):
            config_class = getattr(module, self.model_info['config_class'])

        return model_class, trainer_class, config_class

    def create_model(self, **config_overrides) -> nn.Module:
        """Create model instance"""
        model_class, _, config_class = self._load_model_class()

        if config_class is not None:
            config = config_class(**config_overrides) if config_overrides else config_class()
            model = model_class(config)
        else:
            model = model_class()

        param_count = sum(p.numel() for p in model.parameters())
        logger.info(f"Created {self.model_name} with {param_count:,} parameters")
        return model

    def create_dataloaders(self, batch_size: int = 256, num_workers: int = 4) -> tuple:
        """Create train and validation dataloaders"""
        # Determine content type based on category
        if self.category == ModelCategory.MOVIE_SPECIFIC:
            content_type = 'movie'
        elif self.category == ModelCategory.TV_SPECIFIC:
            content_type = 'tv'
        else:
            content_type = 'both'

        train_dataset = UnifiedDataset(self.data_dir, content_type, self.model_name, 'train')
        val_dataset = UnifiedDataset(self.data_dir, content_type, self.model_name, 'val')

        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, collate_fn=collate_fn, pin_memory=True
        )
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, collate_fn=collate_fn, pin_memory=True
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
        model = self.create_model(**config_overrides)
        model_class, trainer_class, _ = self._load_model_class()

        if trainer_class is not None:
            trainer = trainer_class(model, lr=lr, weight_decay=weight_decay, device=self.device)
            return self._train_with_trainer(model, trainer, epochs, batch_size, save_every, early_stopping_patience)
        else:
            return self._generic_train(model, epochs, batch_size, lr, weight_decay, save_every, early_stopping_patience)

    def _train_with_trainer(self, model, trainer, epochs, batch_size, save_every, early_stopping_patience):
        """Train using dedicated trainer class"""
        train_loader, val_loader = self.create_dataloaders(batch_size)

        best_val_loss = float('inf')
        patience_counter = 0
        history = {'train_loss': [], 'val_metrics': []}

        logger.info(f"Training {self.model_name} ({self.category.value})")
        logger.info(f"  Epochs: {epochs}, Batch size: {batch_size}, Device: {self.device}")

        for epoch in range(epochs):
            epoch_losses = []
            for batch_idx, batch in enumerate(train_loader):
                losses = trainer.train_step(batch)
                epoch_losses.append(losses['total_loss'])

                if batch_idx % 100 == 0:
                    logger.info(f"  Epoch {epoch+1}/{epochs}, Batch {batch_idx}/{len(train_loader)}, "
                               f"Loss: {losses['total_loss']:.4f}")

            avg_train_loss = np.mean(epoch_losses)
            history['train_loss'].append(avg_train_loss)

            val_metrics = trainer.evaluate(val_loader)
            history['val_metrics'].append(val_metrics)

            logger.info(f"Epoch {epoch+1}/{epochs} - Train: {avg_train_loss:.4f}, Val: {val_metrics}")

            if self.use_wandb:
                wandb.log({'epoch': epoch + 1, 'train_loss': avg_train_loss,
                          **{f'val_{k}': v for k, v in val_metrics.items()}})

            if (epoch + 1) % save_every == 0:
                self._save_checkpoint(model, epoch + 1, val_metrics)

            current_val = val_metrics.get('rating_mse', avg_train_loss)
            if current_val < best_val_loss:
                best_val_loss = current_val
                patience_counter = 0
                self._save_checkpoint(model, 'best', val_metrics)
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break

            trainer.scheduler.step()

        self._save_checkpoint(model, 'final', val_metrics)

        if self.use_wandb:
            wandb.finish()

        return {'model_name': self.model_name, 'best_val_loss': best_val_loss,
                'history': history, 'epochs_trained': epoch + 1}

    def _generic_train(self, model, epochs, batch_size, lr, weight_decay, save_every, early_stopping_patience):
        """Generic training for models without dedicated trainers"""
        model = model.to(self.device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
        criterion = nn.MSELoss()

        train_loader, val_loader = self.create_dataloaders(batch_size)

        logger.info(f"Training {self.model_name} with generic trainer")

        best_val_loss = float('inf')
        for epoch in range(epochs):
            model.train()
            epoch_loss = 0
            for batch in train_loader:
                optimizer.zero_grad()
                # Generic forward - assumes model takes user_ids, item_ids
                try:
                    user_ids = batch.get('user_ids', batch.get('user_id')).to(self.device)
                    item_ids = batch.get('item_ids', batch.get('movie_ids', batch.get('show_ids'))).to(self.device)
                    ratings = batch.get('ratings', batch.get('rating')).to(self.device)

                    outputs = model(user_ids, item_ids)
                    if isinstance(outputs, dict):
                        pred = outputs.get('rating_pred', outputs.get('predictions', list(outputs.values())[0]))
                    else:
                        pred = outputs

                    loss = criterion(pred.squeeze(), ratings.float())
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()
                except Exception as e:
                    logger.warning(f"Batch failed: {e}")
                    continue

            scheduler.step()

            if (epoch + 1) % save_every == 0:
                self._save_checkpoint(model, epoch + 1, {'loss': epoch_loss})

            logger.info(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss/len(train_loader):.4f}")

        self._save_checkpoint(model, 'final', {})
        return {'model_name': self.model_name, 'epochs_trained': epochs}

    def _save_checkpoint(self, model: nn.Module, identifier: Any, metrics: Dict):
        """Save model checkpoint"""
        checkpoint_path = self.model_output_dir / f"checkpoint_{identifier}.pt"
        torch.save({
            'model_state_dict': model.state_dict(),
            'metrics': metrics,
            'timestamp': datetime.now().isoformat(),
            'category': self.category.value
        }, checkpoint_path)
        logger.info(f"Saved: {checkpoint_path}")


# ============================================================================
# MAIN FUNCTIONS
# ============================================================================

def train_category(
    category: ModelCategory,
    data_dir: str,
    output_dir: str,
    **training_kwargs
) -> Dict[str, Any]:
    """Train all models in a category"""
    if category == ModelCategory.MOVIE_SPECIFIC:
        models = MOVIE_SPECIFIC_MODELS
    elif category == ModelCategory.TV_SPECIFIC:
        models = TV_SPECIFIC_MODELS
    elif category == ModelCategory.CONTENT_AGNOSTIC:
        models = CONTENT_AGNOSTIC_MODELS
    else:
        models = UNIFIED_MODELS

    results = {}
    # Sort by priority (higher priority first)
    sorted_models = sorted(models.items(), key=lambda x: x[1].get('priority', 0), reverse=True)

    for model_name, info in sorted_models:
        logger.info(f"\n{'='*60}")
        logger.info(f"Training: {model_name}")
        logger.info(f"Description: {info['description']}")
        logger.info(f"Priority: {info.get('priority', 'N/A')}")
        logger.info(f"{'='*60}\n")

        try:
            # Only pass init-compatible kwargs to the constructor
            init_kwargs = {k: v for k, v in training_kwargs.items()
                          if k in ('device', 'use_wandb', 'wandb_project')}
            pipeline = UnifiedTrainingPipeline(
                model_name=model_name,
                data_dir=data_dir,
                output_dir=output_dir,
                **init_kwargs
            )
            result = pipeline.train(**training_kwargs)
            results[model_name] = {'status': 'success', 'result': result}
        except Exception as e:
            logger.error(f"Failed: {model_name} - {e}")
            results[model_name] = {'status': 'failed', 'error': str(e)}

    return results


def train_all(data_dir: str, output_dir: str, **training_kwargs) -> Dict[str, Any]:
    """Train all models in all categories"""
    all_results = {}

    for category in ModelCategory:
        logger.info(f"\n{'#'*60}")
        logger.info(f"CATEGORY: {category.value.upper()}")
        logger.info(f"{'#'*60}\n")

        results = train_category(category, data_dir, output_dir, **training_kwargs)
        all_results[category.value] = results

    return all_results


def list_models():
    """List all available models"""
    print("\n" + "="*80)
    print("CINESYNC V2 - ALL AVAILABLE MODELS")
    print("="*80)

    categories = [
        ("MOVIE-SPECIFIC (14 models)", MOVIE_SPECIFIC_MODELS),
        ("TV-SPECIFIC (14 models)", TV_SPECIFIC_MODELS),
        ("CONTENT-AGNOSTIC (12+ models)", CONTENT_AGNOSTIC_MODELS),
        ("UNIFIED (5 models)", UNIFIED_MODELS),
    ]

    for cat_name, models in categories:
        print(f"\n{cat_name}")
        print("-" * 60)
        sorted_models = sorted(models.items(), key=lambda x: x[1].get('priority', 0), reverse=True)
        for name, info in sorted_models:
            priority = info.get('priority', 'N/A')
            print(f"  [{priority}] {name:30s} - {info['description']}")

    print(f"\nTotal: {len(ALL_MODELS)} models")
    print()


def main():
    parser = argparse.ArgumentParser(
        description='CineSync v2 Master Training Script',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train a specific model
  python train_all_models.py --model movie_franchise_sequence

  # Train all movie models
  python train_all_models.py --category movie

  # Train all TV models
  python train_all_models.py --category tv

  # Train content-agnostic models
  python train_all_models.py --category both

  # Train everything
  python train_all_models.py --all

  # List all models
  python train_all_models.py --list-models
        """
    )

    parser.add_argument('--model', type=str, help='Specific model to train')
    parser.add_argument('--category', type=str, choices=['movie', 'tv', 'both', 'unified'],
                       help='Train all models in category')
    parser.add_argument('--all', action='store_true', help='Train all models')
    parser.add_argument('--list-models', action='store_true', help='List available models')

    parser.add_argument('--data-dir', type=str,
                       default='/Users/timmy/workspace/ai-apps/cine-sync-v2/data',
                       help='Data directory')
    parser.add_argument('--output-dir', type=str,
                       default='/Users/timmy/workspace/ai-apps/cine-sync-v2/models',
                       help='Output directory')
    parser.add_argument('--epochs', type=int, default=50, help='Training epochs')
    parser.add_argument('--batch-size', type=int, default=256, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--device', type=str, default='cuda', help='Device')
    parser.add_argument('--wandb', action='store_true', help='Enable WandB')

    args = parser.parse_args()

    if args.list_models:
        list_models()
        return

    training_kwargs = {
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'lr': args.lr,
        'device': args.device,
        'use_wandb': args.wandb
    }

    if args.all:
        results = train_all(args.data_dir, args.output_dir, **training_kwargs)
        print("\n" + "="*60)
        print("FULL TRAINING SUMMARY")
        print("="*60)
        for category, cat_results in results.items():
            print(f"\n{category.upper()}:")
            for model, result in cat_results.items():
                status = "✓" if result['status'] == 'success' else "✗"
                print(f"  {status} {model}")

    elif args.category:
        category_map = {
            'movie': ModelCategory.MOVIE_SPECIFIC,
            'tv': ModelCategory.TV_SPECIFIC,
            'both': ModelCategory.CONTENT_AGNOSTIC,
            'unified': ModelCategory.UNIFIED
        }
        results = train_category(
            category_map[args.category],
            args.data_dir, args.output_dir, **training_kwargs
        )
        print("\n" + "="*60)
        print(f"TRAINING SUMMARY ({args.category.upper()})")
        print("="*60)
        for model, result in results.items():
            status = "✓" if result['status'] == 'success' else "✗"
            print(f"  {status} {model}")

    elif args.model:
        if args.model not in ALL_MODELS:
            print(f"Unknown model: {args.model}")
            print("Use --list-models to see available models")
            return

        pipeline = UnifiedTrainingPipeline(
            model_name=args.model,
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            device=args.device,
            use_wandb=args.wandb
        )
        result = pipeline.train(**training_kwargs)
        print(f"\nTraining complete for {args.model}")
        print(f"Best validation loss: {result.get('best_val_loss', 'N/A')}")

    else:
        parser.print_help()


if __name__ == '__main__':
    main()
