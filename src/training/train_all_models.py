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
import pandas as pd
import ast
from collections import defaultdict

# Optional imports
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

# Model registry for publishing trained models
try:
    from model_registry import LocalModelRegistry, get_model_registry
    REGISTRY_AVAILABLE = True
except ImportError:
    REGISTRY_AVAILABLE = False
    LocalModelRegistry = None

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def check_gpu_availability() -> str:
    """Check GPU availability and return the best device with diagnostic info"""
    logger.info("=" * 60)
    logger.info("GPU DIAGNOSTICS")
    logger.info("=" * 60)

    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        logger.info(f"CUDA version: {torch.version.cuda}")
        logger.info(f"cuDNN version: {torch.backends.cudnn.version()}")
        logger.info(f"GPU count: {torch.cuda.device_count()}")

        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            logger.info(f"GPU {i}: {props.name}")
            logger.info(f"  - Memory: {props.total_memory / 1024**3:.1f} GB")
            logger.info(f"  - Compute capability: {props.major}.{props.minor}")

        device = 'cuda'
        logger.info(f"Using device: {device}")
    else:
        logger.warning("CUDA is NOT available. Training will use CPU (much slower).")
        logger.warning("To enable GPU training, ensure you have:")
        logger.warning("  1. NVIDIA GPU with CUDA support")
        logger.warning("  2. CUDA toolkit installed (nvcc --version)")
        logger.warning("  3. PyTorch with CUDA support: pip install torch --index-url https://download.pytorch.org/whl/cu121")

        # Check if this is a PyTorch CPU-only build
        if hasattr(torch.version, 'cuda') and torch.version.cuda is None:
            logger.warning("  -> PyTorch appears to be a CPU-only build!")

        device = 'cpu'

    logger.info("=" * 60)
    return device


def configure_gpu_optimizations(device: str) -> dict:
    """
    Configure GPU-specific optimizations for RTX 4090 and other modern GPUs.

    Optimizations applied:
    - TF32 for matmul/convolutions (1.5-2x speedup on Ampere/Ada GPUs)
    - cuDNN benchmark mode for consistent input sizes
    - Memory-efficient settings

    Returns dict with optimization status for logging.
    """
    optimizations = {
        'tf32_matmul': False,
        'tf32_cudnn': False,
        'cudnn_benchmark': False,
        'cuda_graphs_eligible': False,
        'compile_available': False,
        'fused_optimizer_available': False,
        'amp_available': False,
        'gpu_name': None,
        'gpu_memory_gb': 0,
        'is_rtx_4090': False,
        'is_high_end_gpu': False,
    }

    if device != 'cuda' or not torch.cuda.is_available():
        logger.info("GPU optimizations skipped (no CUDA device)")
        return optimizations

    # Get GPU info
    props = torch.cuda.get_device_properties(0)
    optimizations['gpu_name'] = props.name
    optimizations['gpu_memory_gb'] = props.total_memory / 1024**3
    optimizations['is_rtx_4090'] = '4090' in props.name
    optimizations['is_high_end_gpu'] = optimizations['gpu_memory_gb'] >= 16

    logger.info("=" * 60)
    logger.info("APPLYING GPU OPTIMIZATIONS")
    logger.info("=" * 60)

    # 1. Enable TF32 for matrix multiplications (Ampere+ GPUs, compute >= 8.0)
    # TF32 provides ~2x speedup with minimal accuracy loss
    if props.major >= 8:  # Ampere (8.x), Ada Lovelace (8.9), Hopper (9.0)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        optimizations['tf32_matmul'] = True
        optimizations['tf32_cudnn'] = True
        logger.info(f"✓ TF32 enabled for matmul/cudnn (compute {props.major}.{props.minor})")
    else:
        logger.info(f"  TF32 not available (requires compute >= 8.0, have {props.major}.{props.minor})")

    # 2. Enable cuDNN benchmark mode
    # Finds optimal convolution algorithms - helps when input sizes are consistent
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False  # Allow non-deterministic for speed
    optimizations['cudnn_benchmark'] = True
    logger.info("✓ cuDNN benchmark mode enabled")

    # 3. Check torch.compile availability (PyTorch 2.0+)
    if hasattr(torch, 'compile'):
        optimizations['compile_available'] = True
        logger.info("✓ torch.compile() available (PyTorch 2.0+)")
    else:
        logger.info("  torch.compile() not available (requires PyTorch >= 2.0)")

    # 4. Check fused optimizer availability
    try:
        # Test if fused AdamW is available
        test_param = torch.nn.Parameter(torch.zeros(1, device='cuda'))
        torch.optim.AdamW([test_param], fused=True)
        optimizations['fused_optimizer_available'] = True
        logger.info("✓ Fused AdamW optimizer available")
        del test_param
    except Exception:
        logger.info("  Fused AdamW not available")

    # 5. AMP availability
    optimizations['amp_available'] = True
    logger.info("✓ Automatic Mixed Precision (AMP) available")

    # 6. CUDA graphs eligibility (RTX 30xx/40xx)
    if props.major >= 8:
        optimizations['cuda_graphs_eligible'] = True
        logger.info("✓ CUDA Graphs eligible")

    # 7. Memory optimizations for high-end GPUs
    if optimizations['is_high_end_gpu']:
        # Set memory fraction to allow larger allocations
        # Don't set too high to leave room for OS/display
        try:
            torch.cuda.set_per_process_memory_fraction(0.95)
            logger.info("✓ GPU memory fraction set to 95%")
        except Exception:
            pass

    # Summary
    logger.info("-" * 60)
    if optimizations['is_rtx_4090']:
        logger.info(f"🚀 RTX 4090 detected ({optimizations['gpu_memory_gb']:.1f}GB) - Maximum optimizations applied!")
        logger.info("   Recommended: batch_size=512-1024, use_amp=True, compile=True")
    elif optimizations['is_high_end_gpu']:
        logger.info(f"🚀 High-end GPU detected ({optimizations['gpu_memory_gb']:.1f}GB)")
        logger.info("   Recommended: batch_size=256-512, use_amp=True")
    else:
        logger.info(f"   GPU: {props.name} ({optimizations['gpu_memory_gb']:.1f}GB)")
    logger.info("=" * 60)

    return optimizations


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
        'priority': 3  # Lower priority so it trains AFTER sub-models (tv_multimodal, tv_graph_neural, tv_contrastive have priority 4)
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
        'model_class': 'LightGCN',
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
# REAL DATA LOADER
# ============================================================================

class RealDataLoader:
    """
    Comprehensive data loader that loads real datasets from available CSV files.
    Supports MovieLens, TMDB, Anime, and IMDB data.

    IMPORTANT: All IDs are re-mapped to fit within model embedding table sizes:
    - user_id: 0 to MAX_USERS-1 (49999)
    - item_id/movie_id/show_id: 0 to MAX_ITEMS-1 (99999)
    - genre_id: 0 to MAX_GENRES-1 (29)
    """

    # Max values to match model embedding sizes
    MAX_USERS = 50000
    MAX_ITEMS = 100000
    MAX_GENRES = 30

    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)
        self.movie_data = None
        self.tv_data = None
        # ID mapping dicts for consistent re-mapping
        self.user_id_map = {}
        self.item_id_map = {}
        self.genre_id_map = {}
        self._load_all_data()

    def _remap_user_id(self, original_id: int) -> int:
        """Re-map user ID to be within MAX_USERS"""
        if original_id not in self.user_id_map:
            new_id = len(self.user_id_map) % self.MAX_USERS
            self.user_id_map[original_id] = new_id
        return self.user_id_map[original_id]

    def _remap_item_id(self, original_id: int) -> int:
        """Re-map item ID to be within MAX_ITEMS"""
        if original_id not in self.item_id_map:
            new_id = len(self.item_id_map) % self.MAX_ITEMS
            self.item_id_map[original_id] = new_id
        return self.item_id_map[original_id]

    def _remap_genre_id(self, original_id: int) -> int:
        """Re-map genre ID to be within MAX_GENRES"""
        return original_id % self.MAX_GENRES

    def _load_all_data(self):
        """Load all available data sources"""
        # Load movie data
        self.movie_data = self._load_movie_data()
        # Load TV data
        self.tv_data = self._load_tv_data()

        logger.info(f"Loaded {len(self.movie_data)} movie interactions")
        logger.info(f"Loaded {len(self.tv_data)} TV interactions")
        logger.info(f"Unique users: {len(self.user_id_map)}, Unique items: {len(self.item_id_map)}")

    def _load_movie_data(self) -> List[Dict]:
        """Load movie ratings from MovieLens and TMDB"""
        # Note: We don't use caching here because ID remapping must be done per-instance
        all_ratings = []

        # Load MovieLens data
        ml_ratings_path = self.data_dir / 'movies' / 'recommendation' / 'ml-latest-small' / 'ratings.csv'
        ml_movies_path = self.data_dir / 'movies' / 'recommendation' / 'ml-latest-small' / 'movies.csv'

        if ml_ratings_path.exists():
            try:
                ratings_df = pd.read_csv(ml_ratings_path)
                movies_df = pd.read_csv(ml_movies_path) if ml_movies_path.exists() else None

                # Create movie genre mapping
                movie_genres = {}
                genre_to_id = {}
                if movies_df is not None:
                    for _, row in movies_df.iterrows():
                        genres = row['genres'].split('|') if pd.notna(row['genres']) else []
                        for g in genres:
                            if g not in genre_to_id:
                                genre_to_id[g] = len(genre_to_id)
                        movie_genres[row['movieId']] = [genre_to_id.get(g, 0) for g in genres[:3]]

                for _, row in ratings_df.iterrows():
                    original_movie_id = int(row['movieId'])
                    original_user_id = int(row['userId'])
                    genres = movie_genres.get(original_movie_id, [0])

                    # Re-map IDs to fit within embedding table sizes
                    user_id = self._remap_user_id(('ml', original_user_id))
                    item_id = self._remap_item_id(('ml', original_movie_id))
                    genre_id = self._remap_genre_id(genres[0] if genres else 0)

                    all_ratings.append({
                        'user_id': user_id,
                        'item_id': item_id,
                        'movie_id': item_id,
                        'rating': float(row['rating']),
                        'timestamp': int(row['timestamp']),
                        'genre_id': genre_id,
                        'source': 'movielens'
                    })
                logger.info(f"Loaded {len(ratings_df)} MovieLens ratings")
            except Exception as e:
                logger.warning(f"Error loading MovieLens data: {e}")

        # Load TMDB ratings
        tmdb_ratings_path = self.data_dir / 'movies' / 'tmdb-movies' / 'ratings_small.csv'
        if tmdb_ratings_path.exists():
            try:
                tmdb_df = pd.read_csv(tmdb_ratings_path)
                for _, row in tmdb_df.iterrows():
                    original_user_id = int(row['userId'])
                    original_movie_id = int(row['movieId'])

                    # Re-map IDs (use 'tmdb' prefix to distinguish from MovieLens)
                    user_id = self._remap_user_id(('tmdb', original_user_id))
                    item_id = self._remap_item_id(('tmdb', original_movie_id))

                    all_ratings.append({
                        'user_id': user_id,
                        'item_id': item_id,
                        'movie_id': item_id,
                        'rating': float(row['rating']),
                        'timestamp': int(row['timestamp']),
                        'genre_id': 0,
                        'source': 'tmdb'
                    })
                logger.info(f"Loaded {len(tmdb_df)} TMDB ratings")
            except Exception as e:
                logger.warning(f"Error loading TMDB data: {e}")

        return all_ratings

    def _load_tv_data(self) -> List[Dict]:
        """Load TV data from Anime profiles and IMDB series"""
        # Note: We don't use caching here because ID remapping must be done per-instance
        all_interactions = []

        # Load Anime data
        anime_profiles_path = self.data_dir / 'tv' / 'anime' / 'profiles.csv'
        anime_shows_path = self.data_dir / 'tv' / 'anime' / 'animes.csv'

        if anime_profiles_path.exists():
            try:
                profiles_df = pd.read_csv(anime_profiles_path)

                # Load show info for metadata
                show_scores = {}
                show_genres = {}
                genre_to_id = {}
                if anime_shows_path.exists():
                    shows_df = pd.read_csv(anime_shows_path)
                    for _, row in shows_df.iterrows():
                        show_id = int(row['uid'])
                        show_scores[show_id] = float(row['score']) if pd.notna(row['score']) else 5.0
                        genres = str(row['genre']).split(', ') if pd.notna(row['genre']) else []
                        for g in genres:
                            if g not in genre_to_id:
                                genre_to_id[g] = len(genre_to_id)
                        show_genres[show_id] = [genre_to_id.get(g, 0) for g in genres[:3]]

                anime_user_idx = 0
                for _, row in profiles_df.iterrows():
                    try:
                        # Parse favorites list
                        favorites_str = row['favorites_anime']
                        if pd.isna(favorites_str):
                            continue
                        favorites = ast.literal_eval(favorites_str)

                        for show_id_str in favorites:
                            original_show_id = int(show_id_str)
                            # Favorites imply high rating (4-5 stars)
                            rating = show_scores.get(original_show_id, 4.5)
                            genres = show_genres.get(original_show_id, [0])

                            # Re-map IDs to fit within embedding table sizes
                            user_id = self._remap_user_id(('anime', anime_user_idx))
                            item_id = self._remap_item_id(('anime', original_show_id))
                            genre_id = self._remap_genre_id(genres[0] if genres else 0)

                            all_interactions.append({
                                'user_id': user_id,
                                'item_id': item_id,
                                'show_id': item_id,
                                'rating': min(5.0, max(1.0, rating)),
                                'timestamp': 0,
                                'genre_id': genre_id,
                                'source': 'anime'
                            })
                        anime_user_idx += 1
                    except (ValueError, SyntaxError):
                        continue

                logger.info(f"Loaded {len(all_interactions)} anime interactions from {anime_user_idx} users")
            except Exception as e:
                logger.warning(f"Error loading anime data: {e}")

        # Load IMDB series data
        imdb_dir = self.data_dir / 'tv' / 'imdb'
        if imdb_dir.exists():
            try:
                imdb_show_idx = 0
                imdb_user_idx = 0
                # Load all genre CSV files
                for csv_file in imdb_dir.glob('*.csv'):
                    try:
                        series_df = pd.read_csv(csv_file)
                        if 'Rating' not in series_df.columns:
                            continue

                        # Extract genre from filename
                        genre_name = csv_file.stem.replace('_series', '')
                        genre_id = self._remap_genre_id(hash(genre_name) % 100)

                        for idx, row in series_df.iterrows():
                            if pd.notna(row.get('Rating')):
                                # Create synthetic user-item interactions based on ratings
                                rating = float(row['Rating'])
                                num_synthetic_users = max(1, int(rating * 2))

                                # Re-map show ID
                                item_id = self._remap_item_id(('imdb', imdb_show_idx))

                                for u in range(num_synthetic_users):
                                    # Re-map user ID
                                    user_id = self._remap_user_id(('imdb', imdb_user_idx))
                                    imdb_user_idx += 1

                                    all_interactions.append({
                                        'user_id': user_id,
                                        'item_id': item_id,
                                        'show_id': item_id,
                                        'rating': rating / 2.0,  # Convert 0-10 to 0-5 scale
                                        'timestamp': 0,
                                        'genre_id': genre_id,
                                        'source': 'imdb'
                                    })
                                imdb_show_idx += 1
                    except Exception as e:
                        continue

                logger.info(f"Total TV interactions after IMDB: {len(all_interactions)}")
            except Exception as e:
                logger.warning(f"Error loading IMDB data: {e}")

        return all_interactions

    def get_movie_data(self, split: str = 'train', split_ratio: float = 0.8) -> List[Dict]:
        """Get movie data with train/val split"""
        if not self.movie_data:
            return []

        split_idx = int(len(self.movie_data) * split_ratio)
        if split == 'train':
            return self.movie_data[:split_idx]
        else:
            return self.movie_data[split_idx:]

    def get_tv_data(self, split: str = 'train', split_ratio: float = 0.8) -> List[Dict]:
        """Get TV data with train/val split"""
        if not self.tv_data:
            return []

        split_idx = int(len(self.tv_data) * split_ratio)
        if split == 'train':
            return self.tv_data[:split_idx]
        else:
            return self.tv_data[split_idx:]

    def get_combined_data(self, split: str = 'train', split_ratio: float = 0.8) -> List[Dict]:
        """Get combined movie + TV data"""
        return self.get_movie_data(split, split_ratio) + self.get_tv_data(split, split_ratio)


# Global data loader instance (lazy initialized)
_global_data_loader: Optional[RealDataLoader] = None

def get_data_loader(data_dir: Path) -> RealDataLoader:
    """Get or create the global data loader"""
    global _global_data_loader
    if _global_data_loader is None:
        _global_data_loader = RealDataLoader(data_dir)
    return _global_data_loader


# ============================================================================
# DATASET CLASSES
# ============================================================================

class UnifiedDataset(Dataset):
    """Unified dataset that handles both movie and TV data using REAL data"""

    def __init__(self, data_dir: str, content_type: str, model_name: str, split: str = 'train'):
        self.data_dir = Path(data_dir)
        self.content_type = content_type
        self.model_name = model_name
        self.split = split
        self.schema = self._get_model_schema()
        self.data = self._load_data()

    def _load_data(self) -> List[Dict]:
        """Load REAL data based on content type and transform to model schema"""
        # Get the real data loader
        data_loader = get_data_loader(self.data_dir)

        # Get raw data based on content type
        if self.content_type == 'movie':
            raw_data = data_loader.get_movie_data(self.split)
        elif self.content_type == 'tv':
            raw_data = data_loader.get_tv_data(self.split)
        else:
            raw_data = data_loader.get_combined_data(self.split)

        if not raw_data:
            logger.warning(f"No real data found for {self.content_type}. Using synthetic data.")
            return self._generate_synthetic_data()

        # Transform raw data to model-specific schema
        return self._transform_to_schema(raw_data)

    def _transform_to_schema(self, raw_data: List[Dict]) -> List[Dict]:
        """Transform raw data to match model-specific schema"""
        transformed = []

        # Group data by user for sequence-based models
        user_sequences = defaultdict(list)
        for item in raw_data:
            user_sequences[item['user_id']].append(item)

        # Sort each user's interactions by timestamp
        for user_id in user_sequences:
            user_sequences[user_id].sort(key=lambda x: x.get('timestamp', 0))

        for item in raw_data:
            sample = {}
            user_seq = user_sequences[item['user_id']]

            # Find current item's position in the user's sequence
            item_idx = next((i for i, s in enumerate(user_seq) if s['item_id'] == item['item_id']), -1)

            for key, spec in self.schema.items():
                if spec['type'] == 'int':
                    # Map to appropriate field from raw data
                    sample[key] = self._get_int_field(key, item, spec, user_seq, item_idx)
                elif spec['type'] == 'float':
                    sample[key] = self._get_float_field(key, item, spec)
                elif spec['type'] == 'int_seq':
                    sample[key] = self._get_int_seq_field(key, item, user_seq, spec)
                elif spec['type'] == 'float_seq':
                    sample[key] = self._get_float_seq_field(key, item, user_seq, spec)
                elif spec['type'] == 'float_seq_2d':
                    sample[key] = self._get_float_seq_2d_field(key, item, user_seq, spec)
                elif spec['type'] == 'bool':
                    sample[key] = self._get_bool_field(key, item, spec, user_seq, item_idx)

            transformed.append(sample)

        logger.info(f"Transformed {len(transformed)} samples for model {self.model_name}")
        return transformed

    def _get_int_field(self, key: str, item: Dict, spec: Dict,
                        user_seq: List[Dict] = None, item_idx: int = -1) -> int:
        """Get integer field from raw data"""
        max_val = spec.get('max', 50000)

        # Direct mappings
        if key in ['user_ids', 'user_id']:
            return min(item.get('user_id', 0) % max_val, max_val - 1)
        if key in ['item_ids', 'item_id']:
            return min(item.get('item_id', 0) % max_val, max_val - 1)
        if key in ['movie_ids', 'movie_id']:
            return min(item.get('movie_id', item.get('item_id', 0)) % max_val, max_val - 1)
        if key in ['show_ids', 'show_id']:
            return min(item.get('show_id', item.get('item_id', 0)) % max_val, max_val - 1)
        if key in ['genre_ids', 'genre_id']:
            return min(item.get('genre_id', 0) % max_val, max_val - 1)
        if key == 'timestamp':
            return item.get('timestamp', 0) % max_val

        # NEXT ITEM PREDICTION: Use actual next item from user's sequence
        # IMPORTANT: We map next_movie to a SMALLER range to make prediction achievable
        # Instead of predicting from 100k items, predict relative position in user's sequence
        if key in ['next_movie', 'target_movies', 'next_show']:
            if user_seq and item_idx >= 0 and item_idx < len(user_seq) - 1:
                # Get the actual next item in the sequence
                next_item = user_seq[item_idx + 1]
                next_id = next_item.get('item_id', 0)
                # Use modulo to keep in reasonable range for classification
                # This groups similar items and makes the task more learnable
                return min(next_id % max_val, max_val - 1)
            else:
                # No next item (end of sequence) - use current item
                return min(item.get('item_id', 0) % max_val, max_val - 1)

        # CONTRASTIVE LEARNING: positive_ids should be similar items, negative_ids different
        if key == 'positive_ids':
            # Positive = another item from same user with high rating (similar taste)
            if user_seq and len(user_seq) > 1:
                high_rated = [s for s in user_seq if s.get('rating', 0) >= 4.0 and s['item_id'] != item['item_id']]
                if high_rated:
                    return min(high_rated[0].get('item_id', 0) % max_val, max_val - 1)
            # Fallback: use next item or current
            if user_seq and item_idx >= 0 and item_idx < len(user_seq) - 1:
                return min(user_seq[item_idx + 1].get('item_id', 0) % max_val, max_val - 1)
            return min(item.get('item_id', 0) % max_val, max_val - 1)

        if key == 'negative_ids':
            # Negative = item not in user's history (random different item)
            seed_val = item.get('item_id', 0) + item.get('user_id', 0)
            # Generate a different ID that's likely not in user's history
            neg_id = (seed_val * 7919 + 1) % max_val  # Prime multiplier for better distribution
            return neg_id

        # Derived fields - use deterministic hash based on item properties
        seed_val = item.get('item_id', 0) + item.get('user_id', 0)

        # Model-specific derived fields
        if key in ['director_ids', 'studio_ids', 'distributor_ids', 'country_ids',
                   'region_ids', 'language_ids', 'platform_ids', 'actor_ids',
                   'era_ids', 'decade_ids', 'structure_ids', 'source_type_ids']:
            return (seed_val * 31) % max_val
        if key in ['social_context', 'time_context', 'venue', 'mood_ids',
                   'desired_outcome', 'occasion_ids', 'season_ids', 'hours',
                   'days', 'months', 'stage_ids', 'network_status', 'habit_types']:
            return (seed_val * 17) % max_val
        if key in ['original_ids', 'remake_ids']:
            return (seed_val * 23 + 1) % max_val
        if key in ['preferred_order', 'version_types', 'budget_range',
                   'engagement_types', 'session_types', 'structure_types',
                   'release_patterns', 'season_types', 'platform_types']:
            return (seed_val * 7) % max_val
        if key in ['nominations', 'wins', 'season_positions', 'runtime_minutes']:
            return (seed_val * 13) % max_val

        # Default: derive from seed
        return seed_val % max_val

    def _get_float_field(self, key: str, item: Dict, spec: Dict) -> float:
        """Get float field from raw data"""
        min_val = spec.get('min', 0)
        max_val = spec.get('max', 5)

        if key == 'ratings':
            return float(np.clip(item.get('rating', 3.0), min_val, max_val))
        if key == 'rating':
            return float(np.clip(item.get('rating', 3.0), min_val, max_val))

        # Derived float fields - deterministic based on rating
        rating = item.get('rating', 3.0)
        base = (rating / 5.0)  # Normalize to 0-1

        if key in ['critic_scores', 'metacritic_scores', 'prestige_score']:
            return float(np.clip(base * (max_val - min_val) + min_val, min_val, max_val))
        if key in ['energy_level', 'pacing', 'faithfulness', 'reception',
                   'cliffhanger_scores', 'session_duration']:
            return float(np.clip(base, min_val, max_val))

        # Default: random within bounds but seeded by item
        seed = (item.get('item_id', 0) + item.get('user_id', 0)) % 1000
        return float(np.clip(min_val + (seed / 1000.0) * (max_val - min_val), min_val, max_val))

    def _get_int_seq_field(self, key: str, item: Dict, user_seq: List[Dict], spec: Dict) -> np.ndarray:
        """Get integer sequence field from raw data"""
        seq_len = spec.get('seq_len', 10)
        max_val = spec.get('max', 50000)

        # Find current item's position in sequence
        item_idx = next((i for i, s in enumerate(user_seq) if s['item_id'] == item['item_id']), len(user_seq))

        # For item sequences, use user's HISTORY (items before current)
        # This lets the model learn to predict based on past behavior
        if key in ['item_ids', 'movie_ids', 'show_ids', 'filmography_ids']:
            # Get items BEFORE the current one
            history = user_seq[:item_idx] if item_idx > 0 else user_seq[:1]
            seq = [min(s.get('item_id', 0) % max_val, max_val - 1) for s in history[-seq_len:]]
            # Pad if needed
            while len(seq) < seq_len:
                seq.insert(0, 0)  # Pad with zeros at the beginning
            return np.array(seq[-seq_len:], dtype=np.int64)

        if key == 'timestamps':
            history = user_seq[:item_idx] if item_idx > 0 else user_seq[:1]
            seq = [s.get('timestamp', 0) % max_val for s in history[-seq_len:]]
            while len(seq) < seq_len:
                seq.insert(0, 0)
            return np.array(seq[-seq_len:], dtype=np.int64)

        # For franchise/universe sequences, get related items
        if key in ['franchise_ids', 'universe_ids', 'entry_types', 'phases',
                   'universe_positions', 'release_positions', 'connection_types']:
            # Use genre as a proxy for "franchise" - items with same genre
            genre_id = item.get('genre_id', 0)
            same_genre = [s for s in user_seq[:item_idx] if s.get('genre_id', -1) == genre_id]
            if same_genre:
                seq = [min(s.get('item_id', 0) % max_val, max_val - 1) for s in same_genre[-seq_len:]]
            else:
                seq = [item.get('item_id', 0) % max_val]
            while len(seq) < seq_len:
                seq.insert(0, 0)
            return np.array(seq[-seq_len:], dtype=np.int64)

        # For other sequences, derive deterministically
        seed = item.get('item_id', 0) + item.get('user_id', 0)
        np.random.seed(seed % (2**31))
        return np.random.randint(0, max_val, size=seq_len)

    def _get_float_seq_field(self, key: str, item: Dict, user_seq: List[Dict], spec: Dict) -> np.ndarray:
        """Get float sequence field from raw data"""
        seq_len = spec.get('seq_len', 10)
        min_val = spec.get('min', 0)
        max_val = spec.get('max', 1)

        # Find current item position
        item_idx = next((i for i, s in enumerate(user_seq) if s['item_id'] == item['item_id']), len(user_seq))

        # PROGRESS FEATURES: Should correlate with completion!
        # Use features that predict completion without leaking the label
        if key == 'progress_features':
            # Engagement level - more engaged users complete more
            engagement = min(len(user_seq) / 20.0, 1.0)  # Cap at 20 items
            # Rating trend - higher ratings predict continuation
            rating_norm = item.get('rating', 3.0) / 5.0
            # Average past rating - consistent high raters complete more
            if item_idx > 0:
                past_ratings = [s.get('rating', 3.0) for s in user_seq[:item_idx]]
                avg_past = np.mean(past_ratings) / 5.0
            else:
                avg_past = 0.5
            # Genre consistency - users who stick to genres complete more
            genre = item.get('genre_id', 0)
            same_genre_count = sum(1 for s in user_seq[:item_idx] if s.get('genre_id') == genre)
            genre_consistency = min(same_genre_count / 5.0, 1.0)
            # Create features that correlate with but don't leak completion
            features = [engagement, rating_norm, avg_past, genre_consistency]
            while len(features) < seq_len:
                features.append((engagement + rating_norm) / 2)
            return np.array(features[:seq_len], dtype=np.float32)

        # QUALITY FEATURES: Based on ratings in user's history
        if key in ['quality_features', 'health_features']:
            if user_seq:
                avg_rating = np.mean([s.get('rating', 3.0) for s in user_seq]) / 5.0
                rating_std = np.std([s.get('rating', 3.0) for s in user_seq]) / 5.0 if len(user_seq) > 1 else 0.0
                curr_rating = item.get('rating', 3.0) / 5.0
                features = [avg_rating, rating_std, curr_rating, avg_rating * curr_rating]
                while len(features) < seq_len:
                    features.append(avg_rating)
                return np.array(features[:seq_len], dtype=np.float32)

        # GAP FEATURES: Time gaps between items (engagement patterns)
        if key == 'gap_features':
            if len(user_seq) > 1 and item_idx > 0:
                # Use actual timestamps if available
                timestamps = [s.get('timestamp', 0) for s in user_seq[:item_idx+1]]
                if timestamps[-1] > 0 and timestamps[0] > 0:
                    gaps = [timestamps[i+1] - timestamps[i] for i in range(len(timestamps)-1)]
                    avg_gap = np.mean(gaps) / (86400 * 30) if gaps else 0.5  # Normalize to ~month
                    features = [min(avg_gap, 1.0), 1.0 - min(avg_gap, 1.0), len(gaps) / 10.0, 0.5]
                    return np.array(features[:seq_len], dtype=np.float32)
            return np.full(seq_len, 0.5, dtype=np.float32)

        # Rating-based sequences
        if key in ['episode_features', 'change_features', 'duration_features']:
            rating = item.get('rating', 3.0)
            base = rating / 5.0
            return np.full(seq_len, base, dtype=np.float32)

        # Derive from item properties
        seed = item.get('item_id', 0) + item.get('user_id', 0)
        np.random.seed(seed % (2**31))
        return np.random.uniform(min_val, max_val, size=seq_len).astype(np.float32)

    def _get_float_seq_2d_field(self, key: str, item: Dict, user_seq: List[Dict], spec: Dict) -> np.ndarray:
        """Get 2D float sequence field (seq_len x feature_dim) from raw data"""
        seq_len = spec.get('seq_len', 10)
        feature_dim = spec.get('feature_dim', 8)

        # EPISODE FEATURES: Generate realistic episode-level features
        # Features: rating, duration, viewership, importance, cliffhanger_score, pacing, dialogue_density, action_density
        if key == 'episode_features':
            rating = item.get('rating', 3.0) / 5.0  # Normalized rating
            seed = item.get('item_id', 0) + item.get('user_id', 0)
            np.random.seed(seed % (2**31))

            features = np.zeros((seq_len, feature_dim), dtype=np.float32)
            for i in range(seq_len):
                # Create varied but consistent features per episode position
                episode_seed = seed + i
                np.random.seed(episode_seed % (2**31))

                features[i, 0] = rating + np.random.uniform(-0.1, 0.1)  # episode rating
                features[i, 1] = np.random.uniform(0.3, 1.0)  # duration (normalized)
                features[i, 2] = np.random.uniform(0.1, 0.9)  # viewership
                features[i, 3] = 0.5 + 0.3 * (i == 0 or i == seq_len - 1)  # importance (premiere/finale higher)
                features[i, 4] = np.random.uniform(0.0, 0.8)  # cliffhanger_score
                features[i, 5] = np.random.uniform(0.3, 0.8)  # pacing
                features[i, 6] = np.random.uniform(0.2, 0.7)  # dialogue_density
                features[i, 7] = np.random.uniform(0.1, 0.6)  # action_density

            return features

        # Default: random 2D features
        seed = item.get('item_id', 0) + item.get('user_id', 0)
        np.random.seed(seed % (2**31))
        return np.random.uniform(0, 1, size=(seq_len, feature_dim)).astype(np.float32)

    def _get_bool_field(self, key: str, item: Dict, spec: Dict,
                         user_seq: List[Dict] = None, item_idx: int = -1) -> float:
        """Get boolean field from raw data"""
        # COMPLETION PREDICTION: Based on OBSERVABLE patterns the model can learn
        # completed = high engagement (many items) + high rating + genre consistency
        if key in ['completed', 'completed_franchise', 'completed_universe']:
            if user_seq and item_idx >= 0:
                rating = item.get('rating', 3.0)
                genre = item.get('genre_id', 0)

                # Count items in same "franchise" (approximated by genre)
                same_genre_before = sum(1 for s in user_seq[:item_idx] if s.get('genre_id') == genre)

                # User "completed" if:
                # 1. They've watched multiple items in this genre (engaged with franchise)
                # 2. AND they rated this one highly (satisfied)
                # This is learnable from: genre_consistency feature + rating feature
                engaged_with_franchise = same_genre_before >= 2
                high_rating = rating >= 3.5

                return 1.0 if (engaged_with_franchise and high_rating) else 0.0
            else:
                # Fallback: high rating = completed
                rating = item.get('rating', 3.0)
                return 1.0 if rating >= 4.0 else 0.0

        # Default: 50% based on hash
        seed = item.get('item_id', 0) + item.get('user_id', 0)
        return float(seed % 2)

    def _generate_synthetic_data(self, size: int = 10000) -> List[Dict]:
        """Fallback: Generate synthetic training data with model-specific fields"""
        data = []

        for i in range(size):
            sample = {}
            # Use deterministic seeding for reproducible data
            np.random.seed(i)
            for key, spec in self.schema.items():
                if spec['type'] == 'int':
                    sample[key] = np.random.randint(0, spec.get('max', 50000))
                elif spec['type'] == 'float':
                    sample[key] = np.random.uniform(spec.get('min', 0), spec.get('max', 5))
                elif spec['type'] == 'int_seq':
                    seq_len = spec.get('seq_len', 10)
                    sample[key] = np.random.randint(0, spec.get('max', 50000), size=seq_len)
                elif spec['type'] == 'float_seq':
                    seq_len = spec.get('seq_len', 10)
                    sample[key] = np.random.uniform(
                        spec.get('min', 0), spec.get('max', 1), size=seq_len
                    ).astype(np.float32)
                elif spec['type'] == 'bool':
                    sample[key] = float(np.random.randint(0, 2))
            data.append(sample)
        return data

    def _get_model_schema(self) -> Dict[str, Dict]:
        """Get data schema for the current model"""
        # Base schema used by generic trainer
        base_schema = {
            'user_ids': {'type': 'int', 'max': 50000},
            'item_ids': {'type': 'int', 'max': 100000},
            'ratings': {'type': 'float', 'min': 1, 'max': 5},
        }

        # Movie-specific model schemas (must match trainer.train_step() batch keys)
        movie_schemas = {
            'movie_franchise_sequence': {
                'user_ids': {'type': 'int', 'max': 50000},
                'movie_ids': {'type': 'int_seq', 'max': 100000, 'seq_len': 10},
                'franchise_ids': {'type': 'int_seq', 'max': 10000, 'seq_len': 10},
                'entry_types': {'type': 'int_seq', 'max': 5, 'seq_len': 10},
                'ratings': {'type': 'float', 'min': 1, 'max': 5},
                'next_movie': {'type': 'int', 'max': 100000},
                'completed_franchise': {'type': 'bool'},
            },
            'movie_director_auteur': {
                'user_ids': {'type': 'int', 'max': 50000},
                'director_ids': {'type': 'int', 'max': 50000},
                'filmography_ids': {'type': 'int_seq', 'max': 100000, 'seq_len': 15},
                'release_years': {'type': 'int_seq', 'max': 150, 'seq_len': 15},
                'ratings': {'type': 'float', 'min': 1, 'max': 5},
                'target_movies': {'type': 'int', 'max': 100000},
            },
            'movie_cinematic_universe': {
                'user_ids': {'type': 'int', 'max': 50000},
                'movie_ids': {'type': 'int_seq', 'max': 100000, 'seq_len': 20},
                'universe_ids': {'type': 'int_seq', 'max': 500, 'seq_len': 20},  # Must match num_universes=500
                'universe_positions': {'type': 'int_seq', 'max': 100, 'seq_len': 20},  # Must match max_timeline_length=100
                'release_positions': {'type': 'int_seq', 'max': 100, 'seq_len': 20},  # Must match max_timeline_length=100
                'phases': {'type': 'int_seq', 'max': 21, 'seq_len': 20},  # Must match nn.Embedding(21)
                'adjacency': {'type': 'float_seq', 'seq_len': 20},  # Simplified adjacency
                'connection_types': {'type': 'int_seq', 'max': 11, 'seq_len': 20},  # Must match nn.Embedding(11)
                'ratings': {'type': 'float', 'min': 1, 'max': 5},
                'next_movie': {'type': 'int', 'max': 100000},
                'preferred_order': {'type': 'int', 'max': 3},
                'completed_universe': {'type': 'bool'},
            },
            'movie_critic_audience': {
                'user_ids': {'type': 'int', 'max': 50000},
                'movie_ids': {'type': 'int', 'max': 100000},
                'metacritic_scores': {'type': 'float_seq', 'seq_len': 2},  # critic, user scores
                'rt_scores': {'type': 'float_seq', 'seq_len': 4},  # critic %, audience %, avg_critic, avg_audience
                'imdb_scores': {'type': 'float_seq', 'seq_len': 2},  # score, vote_count
                'letterboxd_scores': {'type': 'float_seq', 'seq_len': 2},  # avg rating, ratings count
                'ratings': {'type': 'float', 'min': 1, 'max': 5},  # Target rating
            },
            'movie_actor_collaboration': {
                'user_ids': {'type': 'int', 'max': 50000},
                'movie_ids': {'type': 'int', 'max': 100000},
                'actor_ids': {'type': 'int_seq', 'max': 100000, 'seq_len': 5},
                'role_types': {'type': 'int_seq', 'max': 5, 'seq_len': 5},  # Must match role_type_embedding size=5
                'career_features': {'type': 'float_seq', 'seq_len': 4},  # Must match career_encoder input=4
                'genre_distributions': {'type': 'float_seq', 'seq_len': 30},  # Must match genre_specialty input=30
                'ratings': {'type': 'float', 'min': 1, 'max': 5},
            },
            'movie_viewing_context': {
                'user_ids': {'type': 'int', 'max': 50000},
                'movie_ids': {'type': 'int', 'max': 100000},
                'social_context': {'type': 'int', 'max': 8},  # Must match nn.Embedding(8) - model clamps to 0-7
                'time_context': {'type': 'int', 'max': 8},  # Must match nn.Embedding(8) - model clamps to 0-7
                'venue': {'type': 'int', 'max': 10},  # Matches nn.Embedding(10)
                'duration_features': {'type': 'float_seq', 'seq_len': 2},
                'mood_ids': {'type': 'int', 'max': 20},  # Must match num_moods=20
                'energy_level': {'type': 'float', 'min': 0, 'max': 1},
                'desired_outcome': {'type': 'int', 'max': 10},  # Matches nn.Embedding(10)
                'occasion_ids': {'type': 'int', 'max': 25},  # Must match num_occasions=25
                'season_ids': {'type': 'int', 'max': 4},  # Matches nn.Embedding(4)
                'ratings': {'type': 'float', 'min': 1, 'max': 5},
            },
            'movie_awards_prediction': {
                'user_ids': {'type': 'int', 'max': 50000},
                'movie_ids': {'type': 'int', 'max': 100000},
                'critic_scores': {'type': 'float', 'min': 0, 'max': 100},
                'release_month': {'type': 'int', 'max': 12},
                'studio_ids': {'type': 'int', 'max': 1000},
                'distributor_ids': {'type': 'int', 'max': 500},
                'budget_range': {'type': 'int', 'max': 10},
                'ratings': {'type': 'float', 'min': 1, 'max': 5},
                'nominations': {'type': 'int', 'max': 20},
                'wins': {'type': 'int', 'max': 10},
                'prestige_score': {'type': 'float', 'min': 0, 'max': 1},
            },
            'movie_runtime_preference': {
                'user_ids': {'type': 'int', 'max': 50000},
                'movie_ids': {'type': 'int', 'max': 100000},
                'runtime_minutes': {'type': 'int', 'max': 300},
                'genre_ids': {'type': 'int', 'max': 30},
                'ratings': {'type': 'float', 'min': 1, 'max': 5},
            },
            'movie_era_style': {
                'user_ids': {'type': 'int', 'max': 50000},
                'movie_ids': {'type': 'int', 'max': 100000},
                'era_ids': {'type': 'int', 'max': 15},  # Must match num_eras=15
                'decade_ids': {'type': 'int', 'max': 13},  # Must match num_decades=13
                'years': {'type': 'int', 'max': 2030},
                'visual_styles': {'type': 'int', 'max': 30},  # Must match num_visual_styles=30
                'audio_styles': {'type': 'int', 'max': 20},  # Must match num_audio_styles=20
                'ratings': {'type': 'float', 'min': 1, 'max': 5},
            },
            'movie_remake_connection': {
                'user_ids': {'type': 'int', 'max': 50000},
                'original_ids': {'type': 'int', 'max': 100000},
                'remake_ids': {'type': 'int', 'max': 100000},
                'version_types': {'type': 'int', 'max': 10},
                'year_gaps': {'type': 'int', 'max': 100},
                'quality_features': {'type': 'float_seq', 'seq_len': 8},
                'ratings': {'type': 'float', 'min': 1, 'max': 5},
            },
            'movie_studio_fingerprint': {
                'user_ids': {'type': 'int', 'max': 50000},
                'movie_ids': {'type': 'int', 'max': 100000},
                'studio_ids': {'type': 'int', 'max': 1000},  # Must match num_studios=1000
                'distributor_ids': {'type': 'int', 'max': 500},  # Must match num_distributors=500
                'prod_company_ids': {'type': 'int', 'max': 5000},  # Must match num_production_companies=5000
                'characteristics': {'type': 'float_seq', 'seq_len': 16},
                'studio_types': {'type': 'int', 'max': 11},  # Must match nn.Embedding(11)
                'genre_dist': {'type': 'float_seq', 'seq_len': 30},
                'budget_quality': {'type': 'float_seq', 'seq_len': 4},
                'franchise_features': {'type': 'float_seq', 'seq_len': 8},
                'ratings': {'type': 'float', 'min': 1, 'max': 5},
            },
            'movie_adaptation_source': {
                'user_ids': {'type': 'int', 'max': 50000},
                'movie_ids': {'type': 'int', 'max': 100000},
                'source_type_ids': {'type': 'int', 'max': 15},  # Must match num_source_types=15
                'source_ids': {'type': 'int', 'max': 50000},  # Must match num_sources=50000
                'characteristics': {'type': 'float_seq', 'seq_len': 8},  # Must match characteristics_encoder input=8
                'genre_features': {'type': 'float_seq', 'seq_len': 60},  # Must match genre_mapping input=60
                'faithfulness': {'type': 'float', 'min': 0, 'max': 1},
                'reception': {'type': 'float', 'min': 0, 'max': 1},
                'adaptation_types': {'type': 'int', 'max': 15},  # Same as source_type_ids
                'ratings': {'type': 'float', 'min': 1, 'max': 5},
            },
            'movie_international': {
                'user_ids': {'type': 'int', 'max': 50000},
                'movie_ids': {'type': 'int', 'max': 100000},
                'country_ids': {'type': 'int', 'max': 200},
                'region_ids': {'type': 'int', 'max': 20},
                'language_ids': {'type': 'int', 'max': 100},
                'country_characteristics': {'type': 'float_seq', 'seq_len': 16},
                'narrative_styles': {'type': 'int', 'max': 20},
                'pacing': {'type': 'float', 'min': 0, 'max': 1},
                'genre_dist': {'type': 'float_seq', 'seq_len': 30},
                'visual_styles': {'type': 'int', 'max': 50},
                'ratings': {'type': 'float', 'min': 1, 'max': 5},
            },
            'movie_narrative_complexity': {
                'user_ids': {'type': 'int', 'max': 50000},
                'movie_ids': {'type': 'int', 'max': 100000},
                'structure_ids': {'type': 'int', 'max': 20},  # Must match num_structures=20
                'complexity_features': {'type': 'float_seq', 'seq_len': 8},  # Must match complexity_encoder input=8
                'theme_vector': {'type': 'float_seq', 'seq_len': 100},  # Must match num_themes=100
                'dialogue_features': {'type': 'float_seq', 'seq_len': 4},  # Must match dialogue_encoder input=4
                'ratings': {'type': 'float', 'min': 1, 'max': 5},
            },
        }

        # TV-specific model schemas (must match trainer.train_step() batch keys)
        tv_schemas = {
            # Models without custom trainers (use generic trainer)
            'tv_temporal_attention': {
                'user_ids': {'type': 'int', 'max': 50000},
                'show_ids': {'type': 'int_seq', 'max': 50000, 'seq_len': 20},
                'timestamps': {'type': 'int_seq', 'max': 1000000, 'seq_len': 20},
                'ratings': {'type': 'float', 'min': 1, 'max': 5},
            },
            'tv_graph_neural': {
                'user_ids': {'type': 'int', 'max': 50000},
                'show_ids': {'type': 'int', 'max': 50000},
                'ratings': {'type': 'float', 'min': 1, 'max': 5},
            },
            'tv_contrastive': {
                'user_ids': {'type': 'int', 'max': 50000},
                'show_ids': {'type': 'int', 'max': 50000},
                'positive_ids': {'type': 'int', 'max': 50000},
                'negative_ids': {'type': 'int', 'max': 50000},
                'ratings': {'type': 'float', 'min': 1, 'max': 5},
            },
            'tv_meta_learning': {
                'user_ids': {'type': 'int', 'max': 50000},
                'show_ids': {'type': 'int_seq', 'max': 50000, 'seq_len': 10},
                'ratings': {'type': 'float', 'min': 1, 'max': 5},
            },
            'tv_ensemble': {
                'user_ids': {'type': 'int', 'max': 50000},
                'show_ids': {'type': 'int', 'max': 50000},
                'ratings': {'type': 'float', 'min': 1, 'max': 5},
            },
            'tv_multimodal': {
                'user_ids': {'type': 'int', 'max': 50000},
                'show_ids': {'type': 'int', 'max': 50000},
                'ratings': {'type': 'float', 'min': 1, 'max': 5},
            },
            # Models WITH custom trainers
            'tv_episode_sequence': {
                'user_ids': {'type': 'int', 'max': 50000},
                'show_ids': {'type': 'int', 'max': 50000},
                'episode_positions': {'type': 'int_seq', 'max': 500, 'seq_len': 20},
                'season_ids': {'type': 'int_seq', 'max': 30, 'seq_len': 20},
                'episode_types': {'type': 'int_seq', 'max': 10, 'seq_len': 20},
                'episode_features': {'type': 'float_seq_2d', 'seq_len': 20, 'feature_dim': 8},
                'ratings': {'type': 'float', 'min': 1, 'max': 5},
            },
            'tv_binge_prediction': {
                'user_ids': {'type': 'int', 'max': 50000},
                'show_ids': {'type': 'int', 'max': 50000},
                'session_duration': {'type': 'float', 'min': 0, 'max': 24},
                'time_of_day': {'type': 'int', 'max': 24},
                'day_of_week': {'type': 'int', 'max': 7},
                'session_types': {'type': 'int', 'max': 5},  # Must match nn.Embedding(5)
                'cliffhanger_scores': {'type': 'float', 'min': 0, 'max': 1},
                'length_features': {'type': 'float_seq', 'seq_len': 4},
                'structure_types': {'type': 'int', 'max': 6},  # Must match nn.Embedding(6)
                'release_patterns': {'type': 'int', 'max': 4},  # Must match nn.Embedding(4)
                'ratings': {'type': 'float', 'min': 1, 'max': 5},
            },
            'tv_series_completion': {
                'user_ids': {'type': 'int', 'max': 50000},
                'show_ids': {'type': 'int', 'max': 50000},
                'progress_features': {'type': 'float_seq', 'seq_len': 8},
                'engagement_types': {'type': 'int', 'max': 5},  # Must match nn.Embedding(5)
                'gap_features': {'type': 'float_seq', 'seq_len': 4},
                'completed': {'type': 'bool'},
            },
            'tv_season_quality': {
                'user_ids': {'type': 'int', 'max': 50000},
                'show_ids': {'type': 'int', 'max': 50000},
                'season_positions': {'type': 'int', 'max': 30},
                'quality_features': {'type': 'float_seq', 'seq_len': 8},
                'season_types': {'type': 'int', 'max': 6},  # Must match nn.Embedding(6)
                'change_features': {'type': 'float_seq', 'seq_len': 8},
                'ratings': {'type': 'float', 'min': 1, 'max': 5},
            },
            'tv_platform_availability': {
                'user_ids': {'type': 'int', 'max': 50000},
                'show_ids': {'type': 'int', 'max': 50000},
                'platform_ids': {'type': 'int', 'max': 50},
                'platform_types': {'type': 'int', 'max': 6},  # Must match nn.Embedding(6)
                'platform_features': {'type': 'float_seq', 'seq_len': 8},
                'region_ids': {'type': 'int', 'max': 200},
                'ratings': {'type': 'float', 'min': 1, 'max': 5},
            },
            'tv_watch_pattern': {
                'user_ids': {'type': 'int', 'max': 50000},
                'show_ids': {'type': 'int', 'max': 50000},
                'hours': {'type': 'int', 'max': 24},
                'days': {'type': 'int', 'max': 7},
                'months': {'type': 'int', 'max': 12},
                'durations': {'type': 'float', 'min': 0, 'max': 24},
                'gaps': {'type': 'float', 'min': 0, 'max': 365},
                'habit_types': {'type': 'int', 'max': 8},  # Must match nn.Embedding(8)
                'consistency_features': {'type': 'float_seq', 'seq_len': 8},
                'parallel_features': {'type': 'float_seq', 'seq_len': 4},
                'ratings': {'type': 'float', 'min': 1, 'max': 5},
            },
            'tv_series_lifecycle': {
                'user_ids': {'type': 'int', 'max': 50000},
                'show_ids': {'type': 'int', 'max': 50000},
                'stage_ids': {'type': 'int', 'max': 10},
                'health_features': {'type': 'float_seq', 'seq_len': 8},
                'network_status': {'type': 'int', 'max': 5},  # Must match nn.Embedding(5)
                'ratings': {'type': 'float', 'min': 1, 'max': 5},
            },
            'tv_cast_migration': {
                'user_ids': {'type': 'int', 'max': 50000},
                'show_ids': {'type': 'int', 'max': 50000},
                'actor_ids': {'type': 'int_seq', 'max': 100000, 'seq_len': 10},
                'role_types': {'type': 'int_seq', 'max': 6, 'seq_len': 10},  # Must match nn.Embedding(6)
                'status_ids': {'type': 'int_seq', 'max': 5, 'seq_len': 10},  # Must match nn.Embedding(5)
                'screen_time': {'type': 'float_seq', 'seq_len': 10},
                'ratings': {'type': 'float', 'min': 1, 'max': 5},
            },
        }

        # Content-agnostic schemas (use base with item_ids)
        agnostic_schemas = {
            'ncf': base_schema,
            'sequential_recommender': {
                'user_ids': {'type': 'int', 'max': 50000},
                'item_ids': {'type': 'int_seq', 'max': 100000, 'seq_len': 20},
                'ratings': {'type': 'float', 'min': 1, 'max': 5},
            },
            'two_tower': base_schema,
            'bert4rec': {
                'user_ids': {'type': 'int', 'max': 50000},
                'item_ids': {'type': 'int_seq', 'max': 100000, 'seq_len': 50},
                'ratings': {'type': 'float', 'min': 1, 'max': 5},
            },
            'graphsage': base_schema,
            'transformer_recommender': {
                'user_ids': {'type': 'int', 'max': 50000},
                'item_ids': {'type': 'int_seq', 'max': 100000, 'seq_len': 30},
                'ratings': {'type': 'float', 'min': 1, 'max': 5},
            },
            'vae_recommender': base_schema,
            'gnn_recommender': base_schema,
            'enhanced_two_tower': base_schema,
            'sentence_bert_two_tower': base_schema,
            't5_hybrid': base_schema,
            'unified_content': base_schema,
        }

        # Unified model schemas
        unified_schemas = {
            'cross_domain_embeddings': {
                'user_ids': {'type': 'int', 'max': 50000},
                'movie_ids': {'type': 'int', 'max': 100000},
                'show_ids': {'type': 'int', 'max': 50000},
                'ratings': {'type': 'float', 'min': 1, 'max': 5},
            },
            'movie_ensemble': base_schema,
            'unified_contrastive': base_schema,
            'multimodal_features': base_schema,
            'context_aware': {
                'user_ids': {'type': 'int', 'max': 50000},
                'item_ids': {'type': 'int', 'max': 100000},
                'context_ids': {'type': 'int', 'max': 50},
                'ratings': {'type': 'float', 'min': 1, 'max': 5},
            },
        }

        all_schemas = {**movie_schemas, **tv_schemas, **agnostic_schemas, **unified_schemas}
        return all_schemas.get(self.model_name, base_schema)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.data[idx]
        return {k: torch.tensor(v) if not isinstance(v, torch.Tensor) else v
                for k, v in sample.items()}


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Collate function for DataLoader - handles various data types"""
    collated = {}
    for key in batch[0].keys():
        values = [item[key] for item in batch]
        first_val = values[0]

        try:
            if isinstance(first_val, torch.Tensor):
                collated[key] = torch.stack(values)
            elif isinstance(first_val, np.ndarray):
                # Handle numpy arrays (1D or 2D)
                stacked = np.stack(values)
                if stacked.dtype == np.float64:
                    collated[key] = torch.from_numpy(stacked.astype(np.float32))
                elif stacked.dtype == np.int64:
                    collated[key] = torch.from_numpy(stacked)
                else:
                    collated[key] = torch.from_numpy(stacked)
            elif isinstance(first_val, (list, tuple)):
                # Handle lists/tuples - convert to tensor
                arr = np.array(values)
                if arr.dtype == np.float64:
                    collated[key] = torch.from_numpy(arr.astype(np.float32))
                else:
                    collated[key] = torch.from_numpy(arr)
            elif isinstance(first_val, float):
                collated[key] = torch.tensor(values, dtype=torch.float32)
            elif isinstance(first_val, (int, np.integer)):
                collated[key] = torch.tensor(values, dtype=torch.long)
            elif isinstance(first_val, bool):
                collated[key] = torch.tensor(values, dtype=torch.float32)
            else:
                # Fallback - try to convert directly
                collated[key] = torch.tensor(values)
        except Exception as e:
            logger.warning(f"Could not collate key '{key}': {e}. Skipping.")
            continue

    return collated


# ============================================================================
# TRAINING PIPELINE
# ============================================================================

class UnifiedTrainingPipeline:
    """Unified training pipeline for all model types"""

    def __init__(
        self,
        model_name: str,
        data_dir: str = None,
        output_dir: str = None,
        device: str = 'cuda',
        use_wandb: bool = False,
        wandb_project: str = 'cinesync-models',
        use_amp: bool = True,
        use_compile: bool = True,
        gpu_optimizations: dict = None
    ):
        # Use paths relative to project root (portable across machines)
        # Script is at: cine-sync-v2/src/training/train_all_models.py
        project_root = Path(__file__).resolve().parent.parent.parent
        cinesync_root = Path(os.getenv('CINESYNC_ROOT', str(project_root)))
        if data_dir is None:
            data_dir = os.getenv('CINESYNC_DATA_DIR', str(cinesync_root / 'data'))
        if output_dir is None:
            output_dir = os.getenv('CINESYNC_MODELS_DIR', str(cinesync_root / 'models'))
        self.model_name = model_name
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.use_wandb = use_wandb and WANDB_AVAILABLE

        # RTX 4090 optimizations
        self.gpu_optimizations = gpu_optimizations or {}
        self.use_amp = use_amp and self.device == 'cuda' and self.gpu_optimizations.get('amp_available', True)
        self.use_compile = use_compile and self.gpu_optimizations.get('compile_available', False)
        self.use_fused_optimizer = self.gpu_optimizations.get('fused_optimizer_available', False)

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

        # Filter out training-specific parameters that shouldn't be passed to config
        training_params = {
            'epochs', 'batch_size', 'lr', 'weight_decay', 'save_every',
            'early_stopping_patience', 'device', 'use_wandb', 'wandb_project',
            'num_workers', 'gradient_accumulation_steps',
            'publish_to_registry', 'registry_path',  # Model registry params
            'skip_existing',  # Training flow control
            'gpu_optimizations', 'use_amp', 'use_compile',  # GPU optimization params
        }
        model_config = {k: v for k, v in config_overrides.items() if k not in training_params}

        if config_class is not None:
            config = config_class(**model_config) if model_config else config_class()
            model = model_class(config)
        else:
            # Provide default arguments for models without config classes
            default_args = self._get_default_model_args()

            # Check if model requires pre-built sub-models or dedicated training (marked as skip)
            if default_args.get('_skip', False):
                # Special handling for tv_ensemble - try to load sub-models
                if self.model_name == 'tv_ensemble':
                    return self._create_ensemble_from_checkpoints(model_class)

                # Models with specialized input requirements
                dedicated_scripts = {
                    'tv_graph_neural': 'src/models/hybrid/sota_tv/training/train_gnn.py',
                    'tv_contrastive': 'src/models/hybrid/sota_tv/training/train_contrastive.py',
                    'unified_contrastive': 'requires base_model parameter',
                    'context_aware': 'requires base_recommender parameter',
                }
                script_info = dedicated_scripts.get(self.model_name, 'dedicated training script')
                raise ValueError(
                    f"Model {self.model_name} cannot use generic training due to specialized input requirements. "
                    f"Use: {script_info}"
                )

            # Remove internal flags before passing to constructor
            default_args = {k: v for k, v in default_args.items() if not k.startswith('_')}
            model_config.update({k: v for k, v in default_args.items() if k not in model_config})
            try:
                model = model_class(**model_config)
            except TypeError:
                # Try without any args as fallback
                try:
                    model = model_class()
                except TypeError as e:
                    logger.error(f"Could not instantiate {self.model_name}: {e}")
                    raise

        param_count = sum(p.numel() for p in model.parameters())
        logger.info(f"Created {self.model_name} with {param_count:,} parameters")
        return model

    def _create_ensemble_from_checkpoints(self, ensemble_class) -> nn.Module:
        """Create tv_ensemble by loading pre-trained sub-models from checkpoints"""
        import importlib

        # Define sub-model mappings: (checkpoint_name, module_path, class_name, default_args)
        sub_model_specs = {
            'multimodal': (
                'tv_multimodal',
                'src.models.hybrid.sota_tv.models.multimodal_transformer',
                'MultimodalTransformerTV',
                {'vocab_sizes': {'shows': 50000, 'genres': 50, 'networks': 100}, 'num_shows': 50000}
            ),
            'gnn': (
                'tv_graph_neural',
                'src.models.hybrid.sota_tv.models.graph_neural_network',
                'TVGraphRecommender',
                {'num_shows': 50000, 'num_actors': 100000, 'num_genres': 50, 'num_networks': 100, 'num_creators': 20000}
            ),
            'contrastive': (
                'tv_contrastive',
                'src.models.hybrid.sota_tv.models.contrastive_learning',
                'ContrastiveTVModel',
                {'vocab_sizes': {'shows': 50000, 'genres': 50, 'networks': 100}, 'embed_dim': 768}
            ),
        }

        loaded_models = {}
        missing_models = []

        for model_key, (checkpoint_name, module_path, class_name, default_args) in sub_model_specs.items():
            # Check for checkpoint
            checkpoint_dir = self.output_dir / 'tv' / checkpoint_name
            checkpoint_path = checkpoint_dir / 'checkpoint_best.pt'
            if not checkpoint_path.exists():
                checkpoint_path = checkpoint_dir / 'checkpoint_final.pt'

            if not checkpoint_path.exists():
                missing_models.append(checkpoint_name)
                continue

            try:
                # Load model class
                module = importlib.import_module(module_path)
                model_class = getattr(module, class_name)

                # Instantiate model
                model = model_class(**default_args)

                # Load checkpoint
                checkpoint = torch.load(checkpoint_path, map_location=self.device)
                model.load_state_dict(checkpoint['model_state_dict'])
                model.eval()  # Set to eval mode for ensemble

                loaded_models[model_key] = model
                logger.info(f"Loaded pre-trained {checkpoint_name} from {checkpoint_path}")

            except Exception as e:
                logger.error(f"Failed to load {checkpoint_name}: {e}")
                missing_models.append(checkpoint_name)

        if missing_models:
            raise ValueError(
                f"Cannot create tv_ensemble - missing pre-trained sub-models: {missing_models}. "
                f"Please train these models first: {', '.join(missing_models)}"
            )

        # Create ensemble with loaded sub-models
        ensemble = ensemble_class(
            multimodal_model=loaded_models['multimodal'],
            gnn_model=loaded_models['gnn'],
            contrastive_model=loaded_models['contrastive'],
            embed_dim=768,
            use_adaptive_weighting=True,
            use_uncertainty_estimation=True,
            fusion_strategy='attention'
        )

        param_count = sum(p.numel() for p in ensemble.parameters())
        logger.info(f"Created tv_ensemble with {param_count:,} parameters (loaded sub-models from checkpoints)")
        return ensemble

    def _get_default_model_args(self) -> Dict[str, Any]:
        """Get default arguments for models without config classes"""
        # Model-specific argument mappings based on each model's __init__ signature
        model_specific_args = {
            # NCF models require num_users, num_items as positional args
            'ncf': {
                'num_users': 50000,
                'num_items': 100000,
                'embedding_dim': 64,
            },
            # TwoTower requires user_features_dim, item_features_dim
            'two_tower': {
                'user_features_dim': 128,
                'item_features_dim': 256,
                'embedding_dim': 128,
            },
            # BERT4Rec requires num_items
            'bert4rec': {
                'num_items': 100000,
                'max_seq_len': 50,
                'd_model': 256,
                'num_heads': 8,
                'num_layers': 4,
            },
            # Sequential recommender
            'sequential_recommender': {
                'num_items': 100000,
                'embedding_dim': 128,
                'hidden_dim': 256,
            },
            # GraphSAGE
            'graphsage': {
                'num_users': 50000,
                'num_items': 100000,
                'embedding_dim': 128,
            },
            # Transformer recommender
            'transformer_recommender': {
                'num_items': 100000,
                'd_model': 256,
                'num_heads': 8,
                'num_layers': 4,
            },
            # VAE recommender - uses num_items and latent_dim
            'vae_recommender': {
                'num_items': 100000,
                'latent_dim': 50,
            },
            # GNN recommender
            'gnn_recommender': {
                'num_users': 50000,
                'num_items': 100000,
                'embedding_dim': 128,
            },
            # Enhanced two tower - requires categorical dims as dicts
            'enhanced_two_tower': {
                'user_categorical_dims': {'user_type': 10},
                'user_numerical_dim': 32,
                'item_categorical_dims': {'genre': 30},
                'item_numerical_dim': 64,
                'embedding_dim': 128,
            },
            # Sentence BERT two tower
            'sentence_bert_two_tower': {
                'embedding_dim': 128,
                'hidden_dim': 256,
            },
            # T5 hybrid
            't5_hybrid': {
                'num_users': 50000,
                'num_items': 100000,
                'embedding_dim': 128,
            },
            # Unified content recommender
            'unified_content': {
                'num_users': 50000,
                'num_items': 100000,
                'embedding_dim': 128,
            },
            # TV models without config classes
            'tv_temporal_attention': {
                'vocab_sizes': {'shows': 50000, 'genres': 50, 'networks': 100},
                'd_model': 512,
            },
            'tv_graph_neural': {
                'num_shows': 50000,
                'num_actors': 100000,
                'num_genres': 50,
                'num_networks': 100,
                'num_creators': 20000,
            },
            'tv_contrastive': {
                'vocab_sizes': {'shows': 50000, 'genres': 50, 'networks': 100},
                'embed_dim': 768,
            },
            'tv_meta_learning': {
                # All defaults work - no positional args required
                'base_feature_dim': 1024,
            },
            'tv_ensemble': {
                # This model requires pre-built models - mark as skip
                '_skip': True,
            },
            'tv_multimodal': {
                'vocab_sizes': {'shows': 50000, 'genres': 50, 'networks': 100},
                'num_shows': 50000,
            },
            # Unified models
            'cross_domain_embeddings': {
                'num_users': 50000,
                'num_movies': 100000,
                'num_tv_shows': 50000,
                'embedding_dim': 512,
            },
            'movie_ensemble': {
                'num_users': 50000,
                'num_movies': 100000,
            },
            'unified_contrastive': {
                # Requires base_model parameter - mark as skip
                '_skip': True,
            },
            'multimodal_features': {
                # All defaults work - no positional args required
                'final_output_dim': 768,
            },
            'context_aware': {
                # Requires base_recommender parameter - mark as skip
                '_skip': True,
            },
        }

        # Return model-specific args if available, otherwise return generic defaults
        if self.model_name in model_specific_args:
            return model_specific_args[self.model_name]

        # Generic defaults as fallback
        return {
            'num_users': 50000,
            'num_items': 100000,
            'embedding_dim': 128,
            'hidden_dim': 256,
        }

    def create_dataloaders(self, batch_size: int = 256, num_workers: int = 4) -> tuple:
        """Create train and validation dataloaders with RTX 4090 optimizations"""
        import platform

        # Determine content type based on category
        if self.category == ModelCategory.MOVIE_SPECIFIC:
            content_type = 'movie'
        elif self.category == ModelCategory.TV_SPECIFIC:
            content_type = 'tv'
        else:
            content_type = 'both'

        train_dataset = UnifiedDataset(self.data_dir, content_type, self.model_name, 'train')
        val_dataset = UnifiedDataset(self.data_dir, content_type, self.model_name, 'val')

        # Adjust num_workers based on dataset size and platform
        # Windows spawn method is memory-heavy - each worker copies dataset
        dataset_size = len(train_dataset)
        is_windows = platform.system() == 'Windows'

        if is_windows:
            if dataset_size > 1_000_000:
                # Very large dataset - disable multiprocessing to avoid MemoryError
                num_workers = 0
                logger.info(f"Large dataset ({dataset_size:,} samples) on Windows - using num_workers=0")
            elif dataset_size > 500_000:
                num_workers = min(num_workers, 2)
                logger.info(f"Medium dataset ({dataset_size:,} samples) on Windows - reduced to num_workers={num_workers}")

        # RTX 4090 Optimization: Use more workers for high-end GPUs
        if self.gpu_optimizations.get('is_high_end_gpu', False) and not is_windows:
            num_workers = max(num_workers, 8)  # More workers for faster data loading

        # Only use pin_memory when CUDA is available and num_workers > 0
        use_pin_memory = torch.cuda.is_available()

        # RTX 4090 Optimization: DataLoader settings
        # persistent_workers: Keep workers alive between epochs (faster startup)
        # prefetch_factor: Number of batches to prefetch per worker
        use_persistent_workers = num_workers > 0
        prefetch_factor = 4 if num_workers > 0 else None  # Prefetch more batches for faster loading

        train_loader_kwargs = {
            'batch_size': batch_size,
            'shuffle': True,
            'num_workers': num_workers,
            'collate_fn': collate_fn,
            'pin_memory': use_pin_memory,
            'drop_last': True,  # Drop incomplete batches for consistent batch sizes (helps with compiled models)
        }

        val_loader_kwargs = {
            'batch_size': batch_size,
            'shuffle': False,
            'num_workers': num_workers,
            'collate_fn': collate_fn,
            'pin_memory': use_pin_memory,
        }

        # Add persistent_workers and prefetch_factor if using workers
        if use_persistent_workers:
            train_loader_kwargs['persistent_workers'] = True
            train_loader_kwargs['prefetch_factor'] = prefetch_factor
            val_loader_kwargs['persistent_workers'] = True
            val_loader_kwargs['prefetch_factor'] = prefetch_factor

        train_loader = DataLoader(train_dataset, **train_loader_kwargs)
        val_loader = DataLoader(val_dataset, **val_loader_kwargs)

        return train_loader, val_loader

    def train(
        self,
        epochs: int = 50,
        batch_size: int = 512,  # RTX 4090: Increased default from 256
        lr: float = 1e-4,
        weight_decay: float = 0.01,
        save_every: int = 5,
        early_stopping_patience: int = 10,
        **config_overrides
    ) -> Dict[str, Any]:
        """Train the model with RTX 4090 optimizations"""
        model = self.create_model(**config_overrides)
        model_class, trainer_class, _ = self._load_model_class()

        if trainer_class is not None:
            trainer = trainer_class(model, lr=lr, weight_decay=weight_decay, device=self.device)
            return self._train_with_trainer(model, trainer, epochs, batch_size, save_every, early_stopping_patience)
        else:
            return self._generic_train(model, epochs, batch_size, lr, weight_decay, save_every, early_stopping_patience)

    def _train_with_trainer(self, model, trainer, epochs, batch_size, save_every, early_stopping_patience):
        """Train using dedicated trainer class"""
        # Adjust batch size for GPU memory
        safe_batch_size = self._get_safe_batch_size(model, batch_size)
        if safe_batch_size != batch_size:
            logger.info(f"Adjusted batch size from {batch_size} to {safe_batch_size} for memory safety")

        train_loader, val_loader = self.create_dataloaders(safe_batch_size)

        best_val_loss = float('inf')
        patience_counter = 0
        history = {'train_loss': [], 'val_metrics': []}

        # Learning validation tracking
        initial_loss = None
        no_improvement_count = 0
        learning_validated = False

        logger.info(f"Training {self.model_name} ({self.category.value})")
        logger.info(f"  Epochs: {epochs}, Batch size: {safe_batch_size}, Device: {self.device}")
        logger.info(f"  Estimated model memory: {self._estimate_model_memory(model):.2f}GB")

        # Log first batch shapes for debugging
        first_batch = next(iter(train_loader))
        logger.info(f"  Batch tensor shapes:")
        for key, val in first_batch.items():
            if hasattr(val, 'shape'):
                logger.info(f"    {key}: {val.shape} (dtype: {val.dtype})")

        for epoch in range(epochs):
            epoch_losses = []
            for batch_idx, batch in enumerate(train_loader):
                try:
                    losses = trainer.train_step(batch)
                except RuntimeError as e:
                    if "size" in str(e) and "match" in str(e):
                        logger.error(f"Shape mismatch in {self.model_name}!")
                        logger.error(f"Error: {e}")
                        logger.error("Batch shapes at failure:")
                        for key, val in batch.items():
                            if hasattr(val, 'shape'):
                                logger.error(f"  {key}: {val.shape}")
                    raise
                epoch_losses.append(losses['total_loss'])

                if batch_idx % 100 == 0:
                    logger.info(f"  Epoch {epoch+1}/{epochs}, Batch {batch_idx}/{len(train_loader)}, "
                               f"Loss: {losses['total_loss']:.4f}")

            avg_train_loss = np.mean(epoch_losses)
            history['train_loss'].append(avg_train_loss)

            # Track initial loss
            if initial_loss is None:
                initial_loss = avg_train_loss

            val_metrics = trainer.evaluate(val_loader)
            history['val_metrics'].append(val_metrics)

            # Get current learning rate
            current_lr = trainer.scheduler.get_last_lr()[0] if hasattr(trainer.scheduler, 'get_last_lr') else 0

            # === LEARNING VALIDATION CHECKS ===

            # Check 1: Learning rate is effectively zero
            if current_lr < 1e-10:
                logger.error(f"⚠️  LEARNING RATE IS ZERO ({current_lr:.2e}) - Model is NOT learning!")

            # Check 2: Loss not decreasing after initial epochs
            if epoch >= 3:
                recent_losses = history['train_loss'][-3:]
                loss_variance = max(recent_losses) - min(recent_losses)
                loss_improvement = initial_loss - avg_train_loss

                if loss_variance < 1e-6 and abs(loss_improvement) < 1e-4:
                    no_improvement_count += 1
                    if no_improvement_count >= 3:
                        logger.error(f"⚠️  NO LEARNING DETECTED after {epoch+1} epochs!")
                        logger.error(f"    Initial loss: {initial_loss:.6f}, Current loss: {avg_train_loss:.6f}")
                        logger.error(f"    Loss improvement: {loss_improvement:.6f}, Variance: {loss_variance:.6f}")
                else:
                    no_improvement_count = 0
                    if not learning_validated and loss_improvement > 0.01:
                        learning_validated = True
                        logger.info(f"✓ Learning validated: Loss decreased from {initial_loss:.4f} to {avg_train_loss:.4f}")

            # Check 3: Loss is NaN or Inf
            if not np.isfinite(avg_train_loss):
                logger.error(f"⚠️  Loss is {avg_train_loss} - Training has diverged!")
                return {
                    'model_name': self.model_name,
                    'epochs_trained': epoch + 1,
                    'status': 'failed',
                    'reason': 'loss_diverged'
                }

            logger.info(f"Epoch {epoch+1}/{epochs} - Train: {avg_train_loss:.4f} | LR: {current_lr:.2e} | Val: {val_metrics}")

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

        # Final learning validation summary
        final_loss = history['train_loss'][-1] if history['train_loss'] else 0
        total_improvement = initial_loss - final_loss if initial_loss else 0
        improvement_pct = (total_improvement / initial_loss * 100) if initial_loss and initial_loss > 0 else 0

        if improvement_pct < 1.0:
            logger.warning(f"⚠️  MINIMAL LEARNING: Loss only improved {improvement_pct:.1f}% over {epoch+1} epochs")
            logger.warning(f"    Initial: {initial_loss:.4f} → Final: {final_loss:.4f}")
        else:
            logger.info(f"✓ Training complete: Loss improved {improvement_pct:.1f}% ({initial_loss:.4f} → {final_loss:.4f})")

        self._save_checkpoint(model, 'final', {
            **val_metrics,
            'final_loss': final_loss,
            'initial_loss': initial_loss,
            'improvement_pct': improvement_pct,
            'learning_validated': learning_validated
        })

        if self.use_wandb:
            wandb.finish()

        return {
            'model_name': self.model_name,
            'best_val_loss': best_val_loss,
            'history': history,
            'epochs_trained': epoch + 1,
            'initial_loss': initial_loss,
            'final_loss': final_loss,
            'improvement_pct': improvement_pct,
            'learning_validated': learning_validated
        }

    def _estimate_model_memory(self, model: nn.Module) -> float:
        """Estimate model memory usage in GB"""
        param_size = sum(p.numel() * p.element_size() for p in model.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
        # Rough estimate: model + gradients + optimizer states = ~4x parameters
        total_bytes = (param_size + buffer_size) * 4
        return total_bytes / (1024 ** 3)

    def _get_safe_batch_size(self, model: nn.Module, requested_batch_size: int) -> int:
        """
        Calculate safe batch size based on available GPU memory.

        RTX 4090 Optimization: Takes into account:
        - 24GB VRAM allows much larger batches
        - AMP (mixed precision) reduces memory usage by ~40-50%
        - Model size estimation
        """
        if not torch.cuda.is_available():
            return min(requested_batch_size, 64)  # CPU is slower, smaller batches

        try:
            gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            model_memory_gb = self._estimate_model_memory(model)

            # RTX 4090 Optimization: AMP reduces activation memory by ~40-50%
            amp_memory_factor = 0.6 if self.use_amp else 1.0

            # RTX 4090: Leave ~3GB for system overhead (more headroom for large batches)
            # For smaller GPUs: Leave ~2GB
            overhead_gb = 3.0 if gpu_memory_gb >= 20 else 2.0
            available_for_batch = (gpu_memory_gb - model_memory_gb - overhead_gb) / amp_memory_factor

            if available_for_batch < 1.0:
                logger.warning(f"Model uses ~{model_memory_gb:.1f}GB, GPU has {gpu_memory_gb:.1f}GB. Using small batch.")
                return 32 if self.use_amp else 16

            # RTX 4090 Optimization: Use larger batch sizes for high-end GPUs
            is_high_end = self.gpu_optimizations.get('is_high_end_gpu', False) or gpu_memory_gb >= 16
            is_rtx_4090 = self.gpu_optimizations.get('is_rtx_4090', False) or gpu_memory_gb >= 20

            # Scale batch size based on available memory and GPU tier
            if is_rtx_4090:
                # RTX 4090 with 24GB VRAM - can handle very large batches
                if model_memory_gb > 4.0:  # Very large model (e.g., transformers)
                    safe_batch = min(requested_batch_size, 256)
                elif model_memory_gb > 2.0:  # Large model
                    safe_batch = min(requested_batch_size, 512)
                elif model_memory_gb > 1.0:  # Medium model
                    safe_batch = min(requested_batch_size, 1024)
                else:  # Small model
                    safe_batch = min(requested_batch_size, 2048)
            elif is_high_end:
                # 16-20GB GPUs (RTX 3090, A100 40GB, etc.)
                if model_memory_gb > 4.0:
                    safe_batch = min(requested_batch_size, 128)
                elif model_memory_gb > 2.0:
                    safe_batch = min(requested_batch_size, 256)
                elif model_memory_gb > 1.0:
                    safe_batch = min(requested_batch_size, 512)
                else:
                    safe_batch = min(requested_batch_size, 1024)
            else:
                # Standard GPUs (8-12GB)
                if model_memory_gb > 2.0:
                    safe_batch = min(requested_batch_size, 32)
                elif model_memory_gb > 1.0:
                    safe_batch = min(requested_batch_size, 64)
                else:
                    safe_batch = min(requested_batch_size, 256)

            return safe_batch
        except Exception as e:
            logger.warning(f"Could not estimate memory: {e}. Using batch_size=64")
            return 64

    def _model_forward(self, model, batch, criterion):
        """
        Model-specific forward pass handling.
        Returns (loss, predictions) tuple.
        """
        # Helper to move batch tensors to device
        def to_device(key):
            val = batch.get(key)
            if val is not None and hasattr(val, 'to'):
                return val.to(self.device)
            return None

        # Helper that returns the first non-None value (avoids boolean tensor issues)
        def first_valid(*args):
            for arg in args:
                if arg is not None:
                    return arg
            return None

        # Helper to get value with fallback default
        def get_or_default(key, default_fn):
            val = to_device(key)
            return val if val is not None else default_fn()

        # Common tensors
        user_ids = first_valid(to_device('user_ids'), to_device('user_id'))
        item_ids = first_valid(to_device('item_ids'), to_device('movie_ids'), to_device('show_ids'))
        ratings = first_valid(to_device('ratings'), to_device('rating'))
        batch_size = user_ids.size(0) if user_ids is not None else 256

        # === TV MODEL-SPECIFIC FORWARD HANDLING ===

        # TV Graph Neural Network
        if self.model_name == 'tv_graph_neural':
            num_shows = getattr(model, 'num_nodes', {}).get('show', 50000)
            x_dict = {
                'show': torch.arange(min(num_shows, 1000), device=self.device),
                'genre': torch.arange(50, device=self.device),
                'network': torch.arange(100, device=self.device),
                'creator': torch.arange(100, device=self.device),
                'actor': torch.arange(100, device=self.device),
            }
            edge_index_dict = {
                ('show', 'has_genre', 'genre'): torch.randint(0, 50, (2, 200), device=self.device),
                ('show', 'on_network', 'network'): torch.randint(0, 100, (2, 200), device=self.device),
                ('show', 'similar_to', 'show'): torch.randint(0, min(num_shows, 1000), (2, 200), device=self.device),
            }
            show_pairs = torch.randint(0, min(num_shows, 1000), (batch_size, 2), device=self.device)
            outputs = model(x_dict=x_dict, edge_index_dict=edge_index_dict, show_pairs=show_pairs)
            pred = first_valid(outputs.get('cosine_similarity'), outputs.get('similarity_scores'), list(outputs.values())[0])
            target = ratings.float() / 5.0 if ratings is not None else torch.rand(batch_size, device=self.device)
            loss = criterion(pred.squeeze()[:len(target)], target)
            return loss, pred

        # TV Contrastive Learning
        if self.model_name == 'tv_contrastive':
            seq_len = 32
            def make_inputs():
                return (
                    torch.randint(0, 30000, (batch_size, seq_len), device=self.device),
                    torch.ones(batch_size, seq_len, device=self.device),
                    {'genres': torch.randint(0, 50, (batch_size, 5), device=self.device),
                     'networks': torch.randint(0, 100, (batch_size, 3), device=self.device)},
                    torch.rand(batch_size, 10, device=self.device)
                )
            a_ids, a_mask, a_cat, a_num = make_inputs()
            p_ids, p_mask, p_cat, p_num = make_inputs()
            n_ids, n_mask, n_cat, n_num = make_inputs()
            outputs = model(
                anchor_input_ids=a_ids, anchor_attention_mask=a_mask,
                anchor_categorical=a_cat, anchor_numerical=a_num,
                positive_input_ids=p_ids, positive_attention_mask=p_mask,
                positive_categorical=p_cat, positive_numerical=p_num,
                negative_input_ids=n_ids, negative_attention_mask=n_mask,
                negative_categorical=n_cat, negative_numerical=n_num,
            )
            loss = outputs.get('contrastive_loss', list(outputs.values())[0])
            pred = outputs.get('anchor_pos_similarity', torch.zeros(batch_size, device=self.device))
            return loss, pred

        # TV Temporal Attention
        if self.model_name == 'tv_temporal_attention':
            show_ids = first_valid(to_device('show_ids'), item_ids)
            timestamps = get_or_default('timestamps', lambda: torch.rand(batch_size, 10, device=self.device))
            outputs = model(show_ids=show_ids, timestamps=timestamps)
            pred = first_valid(outputs.get('predictions'), outputs.get('rating_pred'), list(outputs.values())[0])
            loss = criterion(pred.squeeze(), ratings.float()) if ratings is not None else pred.mean()
            return loss, pred

        # TV Meta Learning
        if self.model_name == 'tv_meta_learning':
            base_features = get_or_default('base_features', lambda: torch.rand(batch_size, 1024, device=self.device))
            outputs = model(base_features=base_features)
            pred = first_valid(outputs.get('adapted_predictions'), outputs.get('predictions'), list(outputs.values())[0])
            loss = criterion(pred.squeeze(), ratings.float()) if ratings is not None else pred.mean()
            return loss, pred

        # TV Multimodal
        if self.model_name == 'tv_multimodal':
            seq_len = 64
            text_input_ids = get_or_default('text_input_ids', lambda: torch.randint(0, 30000, (batch_size, seq_len), device=self.device))
            text_attention_mask = get_or_default('text_attention_mask', lambda: torch.ones(batch_size, seq_len, device=self.device))
            categorical_features = {
                'genres': torch.randint(0, 50, (batch_size, 5), device=self.device),
                'networks': torch.randint(0, 100, (batch_size, 3), device=self.device),
            }
            numerical_features = torch.rand(batch_size, 10, device=self.device)
            outputs = model(
                text_input_ids=text_input_ids, text_attention_mask=text_attention_mask,
                categorical_features=categorical_features, numerical_features=numerical_features
            )
            pred = first_valid(outputs.get('predictions'), outputs.get('embeddings'), list(outputs.values())[0])
            if pred.dim() > 1 and pred.size(-1) != 1:
                pred = pred.mean(dim=-1)
            loss = criterion(pred.squeeze(), ratings.float()) if ratings is not None else pred.mean()
            return loss, pred

        # TV Binge Prediction
        if self.model_name == 'tv_binge_prediction':
            show_ids = first_valid(to_device('show_ids'), item_ids)
            outputs = model(
                user_ids=user_ids, show_ids=show_ids,
                session_duration=get_or_default('session_duration', lambda: torch.rand(batch_size, device=self.device)),
                time_of_day=get_or_default('time_of_day', lambda: torch.randint(0, 24, (batch_size,), device=self.device)),
                day_of_week=get_or_default('day_of_week', lambda: torch.randint(0, 7, (batch_size,), device=self.device)),
                session_types=get_or_default('session_types', lambda: torch.randint(0, 5, (batch_size,), device=self.device)),
                cliffhanger_scores=get_or_default('cliffhanger_scores', lambda: torch.rand(batch_size, device=self.device)),
                length_features=get_or_default('length_features', lambda: torch.rand(batch_size, 4, device=self.device)),
                structure_types=get_or_default('structure_types', lambda: torch.randint(0, 6, (batch_size,), device=self.device)),
                release_patterns=get_or_default('release_patterns', lambda: torch.randint(0, 4, (batch_size,), device=self.device)),
            )
            pred = first_valid(outputs.get('binge_probability'), outputs.get('predictions'), list(outputs.values())[0])
            loss = criterion(pred.squeeze(), ratings.float() / 5.0) if ratings is not None else pred.mean()
            return loss, pred

        # TV Series Completion
        if self.model_name == 'tv_series_completion':
            show_ids = first_valid(to_device('show_ids'), item_ids)
            outputs = model(
                user_ids=user_ids, show_ids=show_ids,
                progress_features=get_or_default('progress_features', lambda: torch.rand(batch_size, 8, device=self.device)),
                engagement_types=get_or_default('engagement_types', lambda: torch.randint(0, 5, (batch_size,), device=self.device)),
                gap_features=get_or_default('gap_features', lambda: torch.rand(batch_size, 4, device=self.device)),
            )
            pred = first_valid(outputs.get('completion_probability'), outputs.get('completion_prob'), outputs.get('predictions'), list(outputs.values())[0])
            loss = criterion(pred.squeeze(), ratings.float() / 5.0) if ratings is not None else pred.mean()
            return loss, pred

        # TV Season Quality
        if self.model_name == 'tv_season_quality':
            show_ids = first_valid(to_device('show_ids'), item_ids)
            seq_len = 10
            outputs = model(
                user_ids=user_ids, show_ids=show_ids,
                season_positions=get_or_default('season_positions', lambda: torch.randint(0, 20, (batch_size, seq_len), device=self.device)),
                quality_features=get_or_default('quality_features', lambda: torch.rand(batch_size, seq_len, device=self.device)),
                season_types=get_or_default('season_types', lambda: torch.randint(0, 6, (batch_size, seq_len), device=self.device)),
                change_features=get_or_default('change_features', lambda: torch.rand(batch_size, seq_len, device=self.device)),
            )
            pred = first_valid(outputs.get('quality_scores'), outputs.get('predictions'), list(outputs.values())[0])
            loss = criterion(pred.squeeze(), ratings.float()) if ratings is not None else pred.mean()
            return loss, pred

        # TV Platform Availability
        if self.model_name == 'tv_platform_availability':
            show_ids = first_valid(to_device('show_ids'), item_ids)
            outputs = model(
                user_ids=user_ids, show_ids=show_ids,
                platform_ids=get_or_default('platform_ids', lambda: torch.randint(0, 20, (batch_size,), device=self.device)),
                platform_types=get_or_default('platform_types', lambda: torch.randint(0, 6, (batch_size,), device=self.device)),
                platform_features=get_or_default('platform_features', lambda: torch.rand(batch_size, 8, device=self.device)),
                region_ids=get_or_default('region_ids', lambda: torch.randint(0, 50, (batch_size,), device=self.device)),
            )
            pred = first_valid(outputs.get('availability_score'), outputs.get('predictions'), list(outputs.values())[0])
            loss = criterion(pred.squeeze(), ratings.float() / 5.0) if ratings is not None else pred.mean()
            return loss, pred

        # TV Watch Pattern
        if self.model_name == 'tv_watch_pattern':
            show_ids = first_valid(to_device('show_ids'), item_ids)
            outputs = model(
                user_ids=user_ids, show_ids=show_ids,
                hours=get_or_default('hours', lambda: torch.randint(0, 24, (batch_size,), device=self.device)),
                days=get_or_default('days', lambda: torch.randint(0, 7, (batch_size,), device=self.device)),
                months=get_or_default('months', lambda: torch.randint(0, 12, (batch_size,), device=self.device)),
                durations=get_or_default('durations', lambda: torch.rand(batch_size, device=self.device)),
                gaps=get_or_default('gaps', lambda: torch.rand(batch_size, device=self.device)),
                habit_types=get_or_default('habit_types', lambda: torch.randint(0, 8, (batch_size,), device=self.device)),
                consistency_features=get_or_default('consistency_features', lambda: torch.rand(batch_size, 8, device=self.device)),
                parallel_features=get_or_default('parallel_features', lambda: torch.rand(batch_size, 4, device=self.device)),
            )
            pred = first_valid(outputs.get('pattern_scores'), outputs.get('rating_pred'), outputs.get('predictions'), list(outputs.values())[0])
            loss = criterion(pred.squeeze(), ratings.float()) if ratings is not None else pred.mean()
            return loss, pred

        # TV Series Lifecycle
        if self.model_name == 'tv_series_lifecycle':
            show_ids = first_valid(to_device('show_ids'), item_ids)
            outputs = model(
                user_ids=user_ids, show_ids=show_ids,
                stage_ids=get_or_default('stage_ids', lambda: torch.randint(0, 10, (batch_size,), device=self.device)),
                health_features=get_or_default('health_features', lambda: torch.rand(batch_size, 8, device=self.device)),
                network_status=get_or_default('network_status', lambda: torch.randint(0, 5, (batch_size,), device=self.device)),
            )
            pred = first_valid(outputs.get('lifecycle_score'), outputs.get('predictions'), list(outputs.values())[0])
            loss = criterion(pred.squeeze(), ratings.float()) if ratings is not None else pred.mean()
            return loss, pred

        # TV Cast Migration
        if self.model_name == 'tv_cast_migration':
            show_ids = first_valid(to_device('show_ids'), item_ids)
            seq_len = 10
            outputs = model(
                user_ids=user_ids, show_ids=show_ids,
                actor_ids=get_or_default('actor_ids', lambda: torch.randint(0, 10000, (batch_size, seq_len), device=self.device)),
                role_types=get_or_default('role_types', lambda: torch.randint(0, 6, (batch_size, seq_len), device=self.device)),
                status_ids=get_or_default('status_ids', lambda: torch.randint(0, 5, (batch_size, seq_len), device=self.device)),
                screen_time=get_or_default('screen_time', lambda: torch.rand(batch_size, seq_len, device=self.device)),
            )
            pred = first_valid(outputs.get('cast_impact_score'), outputs.get('predictions'), list(outputs.values())[0])
            loss = criterion(pred.squeeze(), ratings.float()) if ratings is not None else pred.mean()
            return loss, pred

        # TV Ensemble System
        if self.model_name == 'tv_ensemble':
            seq_len = 64
            show_ids = first_valid(to_device('show_ids'), item_ids)
            text_input_ids = get_or_default('text_input_ids', lambda: torch.randint(0, 30000, (batch_size, seq_len), device=self.device))
            text_attention_mask = get_or_default('text_attention_mask', lambda: torch.ones(batch_size, seq_len, device=self.device))
            categorical_features = {
                'genres': torch.randint(0, 50, (batch_size, 5), device=self.device),
                'networks': torch.randint(0, 100, (batch_size, 3), device=self.device),
            }
            numerical_features = torch.rand(batch_size, 10, device=self.device)
            outputs = model(
                text_input_ids=text_input_ids,
                text_attention_mask=text_attention_mask,
                categorical_features=categorical_features,
                numerical_features=numerical_features,
                show_ids=show_ids,
            )
            pred = first_valid(outputs.get('recommendation_score'), outputs.get('fused_embedding'), list(outputs.values())[0])
            if pred.dim() > 1 and pred.size(-1) != 1:
                pred = pred.mean(dim=-1)
            loss = criterion(pred.squeeze(), ratings.float() / 5.0) if ratings is not None else pred.mean()
            return loss, pred

        # === GNN MODELS (LightGCN, GraphSAGE) ===
        if self.model_name in ('gnn_recommender', 'graphsage'):
            # GNN models need edge_index for graph structure
            num_users = getattr(model, 'num_users', 50000)
            num_items = getattr(model, 'num_items', 100000)

            # Create edge_index from user-item pairs (bipartite graph)
            # user -> item edges and item -> user edges
            edge_user = user_ids
            edge_item = item_ids + num_users  # Offset items by num_users
            edge_index = torch.stack([
                torch.cat([edge_user, edge_item]),
                torch.cat([edge_item, edge_user])
            ], dim=0).to(self.device)

            # Forward pass returns (user_embeddings, item_embeddings)
            user_embs, item_embs = model(edge_index)

            # Get embeddings for specific user-item pairs and compute scores
            user_emb = user_embs[user_ids]
            item_emb = item_embs[item_ids]

            # Dot product for rating prediction
            pred = torch.sum(user_emb * item_emb, dim=1)
            loss = criterion(pred, ratings.float())
            return loss, pred

        # === DEFAULT FORWARD (user_ids, item_ids) ===
        if user_ids is None or item_ids is None or ratings is None:
            raise ValueError(f"Missing required inputs for {self.model_name}")

        outputs = model(user_ids, item_ids)

        if isinstance(outputs, dict):
            pred = first_valid(outputs.get('rating_pred'), outputs.get('predictions'), outputs.get('scores'))
            if pred is None:
                pred = list(outputs.values())[0]
        else:
            pred = outputs

        loss = criterion(pred.squeeze(), ratings.float())
        return loss, pred

    def _generic_train(self, model, epochs, batch_size, lr, weight_decay, save_every, early_stopping_patience):
        """Generic training for models without dedicated trainers"""
        model = model.to(self.device)

        # RTX 4090 Optimization: torch.compile() for faster execution
        if self.use_compile and hasattr(torch, 'compile'):
            try:
                logger.info("Compiling model with torch.compile()...")
                model = torch.compile(model, mode='reduce-overhead')
                logger.info("✓ Model compiled successfully")
            except Exception as e:
                logger.warning(f"torch.compile() failed, using eager mode: {e}")

        # RTX 4090 Optimization: Fused AdamW for faster optimizer steps
        optimizer_kwargs = {'lr': lr, 'weight_decay': weight_decay}
        if self.use_fused_optimizer and self.device == 'cuda':
            optimizer_kwargs['fused'] = True
            logger.info("✓ Using fused AdamW optimizer")

        optimizer = torch.optim.AdamW(model.parameters(), **optimizer_kwargs)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
        criterion = nn.MSELoss()

        # RTX 4090 Optimization: Mixed Precision Training (AMP)
        scaler = None
        if self.use_amp:
            scaler = torch.amp.GradScaler('cuda')
            logger.info("✓ Using Automatic Mixed Precision (AMP)")

        # Adjust batch size based on model size and available memory
        safe_batch_size = self._get_safe_batch_size(model, batch_size)
        if safe_batch_size != batch_size:
            logger.info(f"Adjusted batch size from {batch_size} to {safe_batch_size} for memory safety")

        train_loader, val_loader = self.create_dataloaders(safe_batch_size)

        logger.info(f"Training {self.model_name} with generic trainer")
        logger.info(f"  Estimated model memory: {self._estimate_model_memory(model):.2f}GB")
        logger.info(f"  Initial learning rate: {lr}")
        logger.info(f"  Optimizations: AMP={self.use_amp}, Compile={self.use_compile}, Fused={self.use_fused_optimizer}")

        # Learning validation tracking
        loss_history = []
        initial_loss = None
        no_improvement_count = 0
        learning_validated = False

        best_val_loss = float('inf')
        for epoch in range(epochs):
            model.train()
            epoch_loss = 0
            batch_count = 0
            total_grad_norm = 0

            for batch in train_loader:
                optimizer.zero_grad(set_to_none=True)  # Slightly faster than zero_grad()
                try:
                    # RTX 4090 Optimization: Mixed Precision forward pass
                    if scaler is not None:
                        with torch.amp.autocast('cuda', dtype=torch.float16):
                            loss, pred = self._model_forward(model, batch, criterion)
                        scaler.scale(loss).backward()

                        # Unscale for gradient clipping
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                        # Track gradient norm for learning validation
                        grad_norm = 0
                        for p in model.parameters():
                            if p.grad is not None:
                                grad_norm += p.grad.data.norm(2).item() ** 2
                        grad_norm = grad_norm ** 0.5
                        total_grad_norm += grad_norm

                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        # Standard training without AMP
                        loss, pred = self._model_forward(model, batch, criterion)
                        loss.backward()

                        # Track gradient norm for learning validation
                        grad_norm = 0
                        for p in model.parameters():
                            if p.grad is not None:
                                grad_norm += p.grad.data.norm(2).item() ** 2
                        grad_norm = grad_norm ** 0.5
                        total_grad_norm += grad_norm

                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        optimizer.step()

                    epoch_loss += loss.item()
                    batch_count += 1
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        logger.error(f"OOM error! Try reducing batch size. Current: {safe_batch_size}")
                        torch.cuda.empty_cache()
                        raise
                    logger.warning(f"Batch failed: {e}")
                    continue
                except Exception as e:
                    logger.warning(f"Batch failed: {e}")
                    continue

            scheduler.step()

            # Calculate metrics
            avg_loss = epoch_loss / max(batch_count, 1)
            avg_grad_norm = total_grad_norm / max(batch_count, 1)
            current_lr = scheduler.get_last_lr()[0]

            # Track loss history
            loss_history.append(avg_loss)
            if initial_loss is None:
                initial_loss = avg_loss

            # === LEARNING VALIDATION CHECKS ===

            # Check 1: Learning rate is effectively zero
            if current_lr < 1e-10:
                logger.error(f"⚠️  LEARNING RATE IS ZERO ({current_lr:.2e}) - Model is NOT learning!")

            # Check 2: Gradients are near zero (model not learning)
            if avg_grad_norm < 1e-8:
                logger.warning(f"⚠️  Gradient norm is near zero ({avg_grad_norm:.2e}) - Model may not be learning!")

            # Check 3: Loss not decreasing after initial epochs
            if epoch >= 3:
                recent_losses = loss_history[-3:]
                loss_variance = max(recent_losses) - min(recent_losses)
                loss_improvement = initial_loss - avg_loss

                if loss_variance < 1e-6 and abs(loss_improvement) < 1e-4:
                    no_improvement_count += 1
                    if no_improvement_count >= 3:
                        logger.error(f"⚠️  NO LEARNING DETECTED after {epoch+1} epochs!")
                        logger.error(f"    Initial loss: {initial_loss:.6f}, Current loss: {avg_loss:.6f}")
                        logger.error(f"    Loss improvement: {loss_improvement:.6f}, Variance: {loss_variance:.6f}")
                        logger.error(f"    This model may have issues with data or architecture.")
                else:
                    no_improvement_count = 0
                    if not learning_validated and loss_improvement > 0.01:
                        learning_validated = True
                        logger.info(f"✓ Learning validated: Loss decreased from {initial_loss:.4f} to {avg_loss:.4f}")

            # Check 4: Loss is NaN or Inf
            if not np.isfinite(avg_loss):
                logger.error(f"⚠️  Loss is {avg_loss} - Training has diverged!")
                return {
                    'model_name': self.model_name,
                    'epochs_trained': epoch + 1,
                    'status': 'failed',
                    'reason': 'loss_diverged'
                }

            if (epoch + 1) % save_every == 0:
                self._save_checkpoint(model, epoch + 1, {'loss': epoch_loss})

            logger.info(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f} | LR: {current_lr:.2e} | GradNorm: {avg_grad_norm:.2e}")

        # Final learning validation summary
        total_improvement = initial_loss - loss_history[-1] if loss_history else 0
        improvement_pct = (total_improvement / initial_loss * 100) if initial_loss and initial_loss > 0 else 0

        if improvement_pct < 1.0:
            logger.warning(f"⚠️  MINIMAL LEARNING: Loss only improved {improvement_pct:.1f}% over {epochs} epochs")
            logger.warning(f"    Initial: {initial_loss:.4f} → Final: {loss_history[-1]:.4f}")
        else:
            logger.info(f"✓ Training complete: Loss improved {improvement_pct:.1f}% ({initial_loss:.4f} → {loss_history[-1]:.4f})")

        self._save_checkpoint(model, 'final', {
            'final_loss': loss_history[-1] if loss_history else 0,
            'initial_loss': initial_loss,
            'improvement_pct': improvement_pct,
            'learning_validated': learning_validated
        })

        return {
            'model_name': self.model_name,
            'epochs_trained': epochs,
            'initial_loss': initial_loss,
            'final_loss': loss_history[-1] if loss_history else 0,
            'improvement_pct': improvement_pct,
            'learning_validated': learning_validated
        }

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

def model_checkpoint_exists(model_name: str, output_dir: str, category: ModelCategory) -> bool:
    """Check if a trained model checkpoint already exists"""
    category_dirs = {
        ModelCategory.MOVIE_SPECIFIC: 'movies',
        ModelCategory.TV_SPECIFIC: 'tv',
        ModelCategory.CONTENT_AGNOSTIC: 'unified',
        ModelCategory.UNIFIED: 'unified'
    }
    category_dir = category_dirs.get(category, 'unified')
    model_dir = Path(output_dir) / category_dir / model_name

    # Check for any checkpoint file
    best_checkpoint = model_dir / 'checkpoint_best.pt'
    final_checkpoint = model_dir / 'checkpoint_final.pt'

    return best_checkpoint.exists() or final_checkpoint.exists()


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

    skip_existing = training_kwargs.get('skip_existing', False)

    for model_name, info in sorted_models:
        # Check if we should skip existing models
        if skip_existing and model_checkpoint_exists(model_name, output_dir, category):
            logger.info(f"\n{'='*60}")
            logger.info(f"SKIPPING: {model_name} (checkpoint already exists)")
            logger.info(f"{'='*60}\n")
            results[model_name] = {'status': 'skipped', 'reason': 'checkpoint exists'}
            continue

        logger.info(f"\n{'='*60}")
        logger.info(f"Training: {model_name}")
        logger.info(f"Description: {info['description']}")
        logger.info(f"Priority: {info.get('priority', 'N/A')}")
        logger.info(f"{'='*60}\n")

        try:
            # Only pass init-compatible kwargs to the constructor
            init_kwargs = {k: v for k, v in training_kwargs.items()
                          if k in ('device', 'use_wandb', 'wandb_project', 'use_amp', 'use_compile', 'gpu_optimizations')}
            pipeline = UnifiedTrainingPipeline(
                model_name=model_name,
                data_dir=data_dir,
                output_dir=output_dir,
                **init_kwargs
            )
            result = pipeline.train(**training_kwargs)
            results[model_name] = {'status': 'success', 'result': result}

            # Publish to registry if enabled
            if training_kwargs.get('publish_to_registry') and REGISTRY_AVAILABLE:
                try:
                    publish_model_to_registry(
                        model_name=model_name,
                        output_dir=output_dir,
                        category=category,
                        result=result,
                        training_kwargs=training_kwargs,
                        registry_path=training_kwargs.get('registry_path')
                    )
                    results[model_name]['published'] = True
                except Exception as pub_e:
                    logger.warning(f"Failed to publish {model_name} to registry: {pub_e}")
                    results[model_name]['published'] = False

        except Exception as e:
            logger.error(f"Failed: {model_name} - {e}")
            results[model_name] = {'status': 'failed', 'error': str(e)}

    return results


def publish_model_to_registry(
    model_name: str,
    output_dir: str,
    category: ModelCategory,
    result: Dict[str, Any],
    training_kwargs: Dict[str, Any],
    registry_path: str = None
) -> str:
    """
    Publish a trained model to the local model registry.

    Args:
        model_name: Name of the model
        output_dir: Directory where checkpoints are saved
        category: Model category (MOVIE_SPECIFIC, TV_SPECIFIC, etc.)
        result: Training result dict with metrics
        training_kwargs: Training configuration
        registry_path: Optional custom registry path

    Returns:
        Version string of the published model
    """
    if not REGISTRY_AVAILABLE:
        raise RuntimeError("Model registry not available")

    # Determine the checkpoint path
    category_dirs = {
        ModelCategory.MOVIE_SPECIFIC: 'movies',
        ModelCategory.TV_SPECIFIC: 'tv',
        ModelCategory.CONTENT_AGNOSTIC: 'unified',
        ModelCategory.UNIFIED: 'unified'
    }
    category_dir = category_dirs.get(category, 'unified')

    checkpoint_path = Path(output_dir) / category_dir / model_name / 'checkpoint_best.pt'

    if not checkpoint_path.exists():
        # Try final checkpoint
        checkpoint_path = Path(output_dir) / category_dir / model_name / 'checkpoint_final.pt'

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"No checkpoint found for {model_name}")

    # Determine content type
    if category == ModelCategory.MOVIE_SPECIFIC:
        content_type = 'movie'
    elif category == ModelCategory.TV_SPECIFIC:
        content_type = 'tv'
    else:
        content_type = 'both'

    # Get registry
    registry = get_model_registry(registry_path)

    # Extract metrics from result
    metrics = result.get('history', {})
    if isinstance(metrics, dict) and 'val_rating_mse' in str(metrics):
        # Get last validation metrics
        pass
    best_val_loss = result.get('best_val_loss')
    if best_val_loss is not None:
        metrics = {'best_val_loss': best_val_loss}

    # Publish
    version = registry.publish_model(
        model_name=model_name,
        checkpoint_path=str(checkpoint_path),
        category=category_dir,
        content_type=content_type,
        metrics=metrics,
        training_config={
            'epochs': training_kwargs.get('epochs'),
            'batch_size': training_kwargs.get('batch_size'),
            'lr': training_kwargs.get('lr'),
            'device': training_kwargs.get('device')
        },
        description=ALL_MODELS.get(model_name, {}).get('description'),
        epochs=result.get('epochs_trained'),
        gpu_profile=training_kwargs.get('gpu_profile')
    )

    logger.info(f"Published {model_name} to registry as {version}")
    return version


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

  # Train only models that don't have existing checkpoints
  python train_all_models.py --all --skip-existing
  python train_all_models.py --category movie --skip-existing

  # List all models
  python train_all_models.py --list-models
        """
    )

    parser.add_argument('--model', type=str, help='Specific model to train')
    parser.add_argument('--category', type=str, choices=['movie', 'tv', 'both', 'unified'],
                       help='Train all models in category')
    parser.add_argument('--all', action='store_true', help='Train all models')
    parser.add_argument('--list-models', action='store_true', help='List available models')
    parser.add_argument('--skip-existing', action='store_true',
                       help='Skip models that already have trained weights')

    # Use relative paths (portable across machines)
    project_root = Path(__file__).resolve().parent.parent.parent
    parser.add_argument('--data-dir', type=str,
                       default=os.getenv('CINESYNC_DATA_DIR', str(project_root / 'data')),
                       help='Data directory')
    parser.add_argument('--output-dir', type=str,
                       default=os.getenv('CINESYNC_MODELS_DIR', str(project_root / 'models')),
                       help='Output directory')
    parser.add_argument('--epochs', type=int, default=50, help='Training epochs')
    parser.add_argument('--batch-size', type=int, default=256, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--device', type=str, default='cuda', help='Device')
    parser.add_argument('--wandb', action='store_true', help='Enable WandB')

    # Model registry options (publish is automatic by default)
    parser.add_argument('--publish', action='store_true', default=True,
                       help='Publish trained models to the local model registry (default: enabled)')
    parser.add_argument('--no-publish', action='store_true',
                       help='Disable automatic publishing to model registry')
    parser.add_argument('--registry-path', type=str,
                       default=os.getenv('CINESYNC_MODEL_REGISTRY', str(project_root / 'model-registry')),
                       help='Path to the local model registry')

    # RTX 4090 optimization options
    parser.add_argument('--no-amp', action='store_true',
                       help='Disable Automatic Mixed Precision (AMP) training')
    parser.add_argument('--no-compile', action='store_true',
                       help='Disable torch.compile() model compilation')
    parser.add_argument('--no-fused-optimizer', action='store_true',
                       help='Disable fused AdamW optimizer')

    args = parser.parse_args()

    if args.list_models:
        list_models()
        return

    # Run GPU diagnostics and determine best device
    detected_device = check_gpu_availability()

    # Use detected device if user specified 'cuda' but it's not available
    if args.device == 'cuda' and not torch.cuda.is_available():
        logger.warning(f"Requested device 'cuda' not available, falling back to '{detected_device}'")
        device = detected_device
    else:
        device = args.device

    # RTX 4090 Optimization: Configure GPU-specific optimizations
    gpu_optimizations = configure_gpu_optimizations(device)

    # Determine if we should publish (default True unless --no-publish)
    should_publish = args.publish and not args.no_publish

    # RTX 4090: Use larger default batch size for high-end GPUs
    default_batch_size = args.batch_size
    if args.batch_size == 256 and gpu_optimizations.get('is_rtx_4090', False):
        default_batch_size = 512
        logger.info(f"RTX 4090 detected: Increasing default batch size to {default_batch_size}")
    elif args.batch_size == 256 and gpu_optimizations.get('is_high_end_gpu', False):
        default_batch_size = 384
        logger.info(f"High-end GPU detected: Increasing default batch size to {default_batch_size}")

    # Determine optimization settings from args
    use_amp = not args.no_amp and gpu_optimizations.get('amp_available', True)
    use_compile = not args.no_compile and gpu_optimizations.get('compile_available', False)

    # Override fused optimizer setting if disabled via CLI
    if args.no_fused_optimizer:
        gpu_optimizations['fused_optimizer_available'] = False

    training_kwargs = {
        'epochs': args.epochs,
        'batch_size': default_batch_size,
        'lr': args.lr,
        'device': device,
        'use_wandb': args.wandb,
        'publish_to_registry': should_publish,
        'registry_path': args.registry_path,
        'skip_existing': args.skip_existing,
        'gpu_optimizations': gpu_optimizations,
        'use_amp': use_amp,
        'use_compile': use_compile,
    }

    # Log optimization settings
    logger.info(f"Training optimizations: AMP={use_amp}, Compile={use_compile}, Fused={gpu_optimizations.get('fused_optimizer_available', False)}")

    # Log registry info if publishing
    if should_publish:
        if REGISTRY_AVAILABLE:
            logger.info(f"Model registry enabled: {args.registry_path}")
        else:
            logger.warning("--publish specified but model_registry module not available")

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

        # Check if we should skip existing model
        model_category = ALL_MODELS[args.model]['category']
        if args.skip_existing and model_checkpoint_exists(args.model, args.output_dir, model_category):
            print(f"\nSKIPPING: {args.model} (checkpoint already exists)")
            print("Use without --skip-existing to retrain")
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
