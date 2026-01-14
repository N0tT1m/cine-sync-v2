"""
CineSync v2 - Multi-Model Inference Server

Loads and serves all 46 recommendation models with GPU-optimized configurations.

Usage:
    # For RTX 4090
    python model_server.py --gpu rtx4090 --port 8000

    # For RTX 5090
    python model_server.py --gpu rtx5090 --port 8000
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import json

import torch
import torch.nn as nn

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.inference.gpu_configs import (
    get_config, GPUConfig, PrecisionMode, ModelLoadConfig,
    estimate_vram_usage, print_config_summary
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# Model Registry - Maps model names to their module paths and classes
# =============================================================================
MODEL_REGISTRY = {
    # Movie-specific models
    'movie_franchise_sequence': ('src.models.movie.franchise_sequence', 'FranchiseSequenceModel'),
    'movie_director_auteur': ('src.models.movie.director_auteur', 'DirectorAuteurModel'),
    'movie_cinematic_universe': ('src.models.movie.cinematic_universe', 'CinematicUniverseModel'),
    'movie_awards_prediction': ('src.models.movie.awards_prediction', 'AwardsPredictionModel'),
    'movie_runtime_preference': ('src.models.movie.runtime_preference', 'RuntimePreferenceModel'),
    'movie_era_style': ('src.models.movie.era_style', 'EraStyleModel'),
    'movie_critic_audience': ('src.models.movie.critic_audience', 'CriticAudienceModel'),
    'movie_remake_connection': ('src.models.movie.remake_connection', 'RemakeConnectionModel'),
    'movie_actor_collaboration': ('src.models.movie.actor_collaboration', 'ActorCollaborationModel'),
    'movie_studio_fingerprint': ('src.models.movie.studio_fingerprint', 'StudioFingerprintModel'),
    'movie_adaptation_source': ('src.models.movie.adaptation_source', 'AdaptationSourceModel'),
    'movie_international': ('src.models.movie.international_cinema', 'InternationalCinemaModel'),
    'movie_narrative_complexity': ('src.models.movie.narrative_complexity', 'NarrativeComplexityModel'),
    'movie_viewing_context': ('src.models.movie.viewing_context', 'ViewingContextModel'),

    # TV-specific models
    'tv_temporal_attention': ('src.models.hybrid.sota_tv.models.temporal_attention', 'TemporalAttentionTVModel'),
    'tv_graph_neural': ('src.models.hybrid.sota_tv.models.graph_neural_network', 'TVGraphNeuralNetwork'),
    'tv_contrastive': ('src.models.hybrid.sota_tv.models.contrastive_learning', 'ContrastiveTVLearning'),
    'tv_meta_learning': ('src.models.hybrid.sota_tv.models.meta_learning', 'MetaLearningTVModel'),
    'tv_ensemble': ('src.models.hybrid.sota_tv.models.ensemble_system', 'TVEnsembleSystem'),
    'tv_multimodal': ('src.models.hybrid.sota_tv.models.multimodal_transformer', 'MultimodalTVTransformer'),
    'tv_episode_sequence': ('src.models.hybrid.sota_tv.models.episode_sequence', 'EpisodeSequenceModel'),
    'tv_binge_prediction': ('src.models.hybrid.sota_tv.models.binge_prediction', 'BingePredictionModel'),
    'tv_series_completion': ('src.models.hybrid.sota_tv.models.series_completion', 'SeriesCompletionModel'),
    'tv_season_quality': ('src.models.hybrid.sota_tv.models.season_quality', 'SeasonQualityModel'),
    'tv_platform_availability': ('src.models.hybrid.sota_tv.models.platform_availability', 'PlatformAvailabilityModel'),
    'tv_watch_pattern': ('src.models.hybrid.sota_tv.models.watch_pattern', 'WatchPatternModel'),
    'tv_series_lifecycle': ('src.models.hybrid.sota_tv.models.series_lifecycle', 'SeriesLifecycleModel'),
    'tv_cast_migration': ('src.models.hybrid.sota_tv.models.cast_migration', 'CastMigrationModel'),

    # Content-agnostic models
    'ncf': ('src.models.collaborative.src.model', 'NeuralCollaborativeFiltering'),
    'sequential_recommender': ('src.models.sequential.src.model', 'SequentialRecommender'),
    'two_tower': ('src.models.two_tower.src.model', 'TwoTowerModel'),
    'bert4rec': ('src.models.advanced.bert4rec_recommender', 'BERT4Rec'),
    'graphsage': ('src.models.advanced.graphsage_recommender', 'GraphSAGERecommender'),
    'transformer_recommender': ('src.models.advanced.transformer_recommender', 'TransformerRecommender'),
    'vae_recommender': ('src.models.advanced.variational_autoencoder', 'VAERecommender'),
    'gnn_recommender': ('src.models.advanced.graph_neural_network', 'GNNRecommender'),
    'enhanced_two_tower': ('src.models.advanced.enhanced_two_tower', 'EnhancedTwoTower'),
    'sentence_bert_two_tower': ('src.models.advanced.sentence_bert_two_tower', 'SentenceBERTTwoTower'),
    't5_hybrid': ('src.models.advanced.t5_hybrid_recommender', 'T5HybridRecommender'),
    'unified_content': ('src.models.hybrid.content_recommender', 'UnifiedContentRecommender'),

    # Unified models
    'cross_domain_embeddings': ('src.models.unified.cross_domain_embeddings', 'CrossDomainEmbeddings'),
    'movie_ensemble': ('src.models.unified.movie_ensemble_system', 'MovieEnsembleSystem'),
    'unified_contrastive': ('src.models.unified.contrastive_learning', 'UnifiedContrastiveLearning'),
    'multimodal_features': ('src.models.unified.multimodal_features', 'MultimodalFeatures'),
    'context_aware': ('src.models.unified.context_aware', 'ContextAwareRecommender'),
}


@dataclass
class LoadedModel:
    """Container for a loaded model with metadata"""
    name: str
    model: nn.Module
    precision: PrecisionMode
    load_time_ms: float
    vram_mb: float


class SharedEmbeddings:
    """Shared embedding tables across all models"""

    def __init__(self, config: GPUConfig, device: torch.device):
        self.config = config
        self.device = device
        self.dtype = self._get_dtype(config.embedding_precision)

        # Standard embedding sizes
        self.num_users = 50000
        self.num_items = 100000
        self.embedding_dim = 512

        logger.info(f"Initializing shared embeddings ({config.embedding_precision.value})...")

        # Create shared embedding tables
        self.user_embedding = nn.Embedding(
            self.num_users, self.embedding_dim
        ).to(device=device, dtype=self.dtype)

        self.item_embedding = nn.Embedding(
            self.num_items, self.embedding_dim
        ).to(device=device, dtype=self.dtype)

        # Initialize with Xavier
        nn.init.xavier_normal_(self.user_embedding.weight)
        nn.init.xavier_normal_(self.item_embedding.weight)

        logger.info(f"  User embeddings: {self.num_users} x {self.embedding_dim}")
        logger.info(f"  Item embeddings: {self.num_items} x {self.embedding_dim}")

    def _get_dtype(self, precision: PrecisionMode) -> torch.dtype:
        return {
            PrecisionMode.FP32: torch.float32,
            PrecisionMode.FP16: torch.float16,
            PrecisionMode.BF16: torch.bfloat16,
            PrecisionMode.INT8: torch.float32,  # INT8 handled differently
        }[precision]

    def get_user_embeddings(self, user_ids: torch.Tensor) -> torch.Tensor:
        return self.user_embedding(user_ids.clamp(0, self.num_users - 1))

    def get_item_embeddings(self, item_ids: torch.Tensor) -> torch.Tensor:
        return self.item_embedding(item_ids.clamp(0, self.num_items - 1))

    def load_pretrained(self, checkpoint_dir: Path):
        """Load pretrained embeddings from checkpoint"""
        user_path = checkpoint_dir / "shared_user_embeddings.pt"
        item_path = checkpoint_dir / "shared_item_embeddings.pt"

        if user_path.exists():
            self.user_embedding.load_state_dict(torch.load(user_path, map_location=self.device))
            logger.info(f"  Loaded user embeddings from {user_path}")

        if item_path.exists():
            self.item_embedding.load_state_dict(torch.load(item_path, map_location=self.device))
            logger.info(f"  Loaded item embeddings from {item_path}")


class ModelServer:
    """
    Main inference server that manages all recommendation models.
    """

    def __init__(self, config: GPUConfig, checkpoint_dir: Optional[Path] = None):
        self.config = config
        self.checkpoint_dir = checkpoint_dir or PROJECT_ROOT / "checkpoints"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Model storage
        self.models: Dict[str, LoadedModel] = {}
        self.shared_embeddings: Optional[SharedEmbeddings] = None

        # Stats
        self.total_load_time_ms = 0
        self.total_vram_mb = 0

        logger.info(f"Initializing ModelServer with {config.name} configuration")
        logger.info(f"Device: {self.device}")

        if torch.cuda.is_available():
            logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
            logger.info(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    def initialize(self):
        """Initialize the server: load shared embeddings and all models"""
        start_time = time.time()

        # Print config summary
        print_config_summary(self.config)

        # Check VRAM estimate
        usage = estimate_vram_usage(self.config)
        if usage['headroom_gb'] < 0:
            logger.warning(f"WARNING: Estimated VRAM usage ({usage['total_gb']:.1f} GB) "
                          f"exceeds available ({self.config.vram_gb} GB)!")

        # Initialize shared embeddings
        if self.config.share_embeddings:
            self.shared_embeddings = SharedEmbeddings(self.config, self.device)
            if self.checkpoint_dir.exists():
                self.shared_embeddings.load_pretrained(self.checkpoint_dir)

        # Load models
        self._load_all_models()

        # Final stats
        total_time = time.time() - start_time
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Server initialized in {total_time:.2f}s")
        logger.info(f"Models loaded: {len(self.models)}")
        logger.info(f"Total load time: {self.total_load_time_ms / 1000:.2f}s")

        if torch.cuda.is_available():
            vram_used = torch.cuda.memory_allocated() / 1024**3
            vram_cached = torch.cuda.memory_reserved() / 1024**3
            logger.info(f"VRAM used: {vram_used:.2f} GB (cached: {vram_cached:.2f} GB)")

        logger.info(f"{'=' * 60}\n")

    def _load_all_models(self):
        """Load all enabled models based on configuration"""
        models_to_load = list(MODEL_REGISTRY.keys())

        # Filter based on config
        if self.config.enabled_models:
            models_to_load = [m for m in models_to_load if m in self.config.enabled_models]

        models_to_load = [m for m in models_to_load if m not in self.config.disabled_models]

        # Separate preload vs lazy load
        preload_models = []
        lazy_models = []

        for model_name in models_to_load:
            if model_name in self.config.model_overrides:
                if self.config.model_overrides[model_name].preload:
                    preload_models.append(model_name)
                else:
                    lazy_models.append(model_name)
            elif self.config.preload_all:
                preload_models.append(model_name)
            else:
                lazy_models.append(model_name)

        # Load preload models
        logger.info(f"\nPreloading {len(preload_models)} models...")
        for model_name in preload_models:
            self._load_model(model_name)

        # Register lazy models
        if lazy_models:
            logger.info(f"\n{len(lazy_models)} models marked for lazy loading:")
            for name in lazy_models:
                logger.info(f"  - {name}")

    def _load_model(self, model_name: str) -> Optional[LoadedModel]:
        """Load a single model"""
        if model_name in self.models:
            return self.models[model_name]

        if model_name not in MODEL_REGISTRY:
            logger.warning(f"Unknown model: {model_name}")
            return None

        module_path, class_name = MODEL_REGISTRY[model_name]

        # Determine precision
        if model_name in self.config.model_overrides:
            precision = self.config.model_overrides[model_name].precision
        elif any(t in model_name for t in ['bert', 't5', 'transformer']):
            precision = self.config.transformer_precision
        else:
            precision = self.config.default_precision

        dtype = {
            PrecisionMode.FP32: torch.float32,
            PrecisionMode.FP16: torch.float16,
            PrecisionMode.BF16: torch.bfloat16,
            PrecisionMode.INT8: torch.float32,
        }[precision]

        start_time = time.time()

        try:
            # Import and instantiate model
            module = __import__(module_path, fromlist=[class_name])
            model_class = getattr(module, class_name)
            model = model_class()

            # Move to device and precision
            model = model.to(device=self.device, dtype=dtype)
            model.eval()

            # Load checkpoint if exists
            checkpoint_path = self.checkpoint_dir / f"{model_name}_best.pt"
            if checkpoint_path.exists():
                state_dict = torch.load(checkpoint_path, map_location=self.device)
                if 'model_state_dict' in state_dict:
                    model.load_state_dict(state_dict['model_state_dict'])
                else:
                    model.load_state_dict(state_dict)
                logger.info(f"  Loaded checkpoint: {checkpoint_path.name}")

            # Optional: torch.compile for PyTorch 2.0+
            if self.config.use_torch_compile:
                if model_name in self.config.model_overrides:
                    if self.config.model_overrides[model_name].compile_model:
                        try:
                            model = torch.compile(model)
                            logger.info(f"  Compiled with torch.compile")
                        except Exception as e:
                            logger.warning(f"  torch.compile failed: {e}")

            load_time_ms = (time.time() - start_time) * 1000

            # Estimate VRAM (rough)
            param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
            vram_mb = param_bytes / 1024 / 1024

            loaded = LoadedModel(
                name=model_name,
                model=model,
                precision=precision,
                load_time_ms=load_time_ms,
                vram_mb=vram_mb
            )

            self.models[model_name] = loaded
            self.total_load_time_ms += load_time_ms
            self.total_vram_mb += vram_mb

            logger.info(f"  Loaded {model_name} ({precision.value}) - "
                       f"{load_time_ms:.0f}ms, {vram_mb:.1f}MB")

            return loaded

        except Exception as e:
            logger.error(f"  Failed to load {model_name}: {e}")
            return None

    def get_model(self, model_name: str) -> Optional[nn.Module]:
        """Get a model by name, lazy loading if necessary"""
        if model_name not in self.models:
            self._load_model(model_name)
        return self.models.get(model_name, LoadedModel).model if model_name in self.models else None

    def get_movie_models(self) -> List[str]:
        """Get list of movie-related model names"""
        return [name for name in self.models.keys()
                if name.startswith('movie_') or name in [
                    'ncf', 'sequential_recommender', 'two_tower', 'bert4rec',
                    'graphsage', 'transformer_recommender', 'enhanced_two_tower',
                    'sentence_bert_two_tower', 't5_hybrid', 'unified_content',
                    'movie_ensemble', 'context_aware'
                ]]

    def get_tv_models(self) -> List[str]:
        """Get list of TV-related model names"""
        return [name for name in self.models.keys()
                if name.startswith('tv_') or name in [
                    'ncf', 'sequential_recommender', 'two_tower', 'bert4rec',
                    'graphsage', 'transformer_recommender', 'enhanced_two_tower',
                    'sentence_bert_two_tower', 't5_hybrid', 'unified_content',
                    'context_aware'
                ]]

    @torch.no_grad()
    def recommend_movies(self, user_id: int, top_k: int = 20) -> Dict[str, Any]:
        """Get movie recommendations using all movie models"""
        # Implementation would aggregate predictions from all movie models
        # This is a placeholder for the actual recommendation logic
        pass

    @torch.no_grad()
    def recommend_tv(self, user_id: int, top_k: int = 20) -> Dict[str, Any]:
        """Get TV show recommendations using all TV models"""
        # Implementation would aggregate predictions from all TV models
        # This is a placeholder for the actual recommendation logic
        pass

    def health_check(self) -> Dict[str, Any]:
        """Return server health status"""
        vram_used = torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0
        vram_total = torch.cuda.get_device_properties(0).total_memory / 1024**3 if torch.cuda.is_available() else 0

        return {
            'status': 'healthy',
            'gpu': self.config.name,
            'models_loaded': len(self.models),
            'vram_used_gb': round(vram_used, 2),
            'vram_total_gb': round(vram_total, 2),
            'vram_percent': round(vram_used / vram_total * 100, 1) if vram_total > 0 else 0,
        }


def main():
    parser = argparse.ArgumentParser(description='CineSync v2 Model Inference Server')
    parser.add_argument('--gpu', type=str, default='rtx4090',
                       choices=['rtx4090', 'rtx5090', '4090', '5090'],
                       help='GPU configuration profile')
    parser.add_argument('--checkpoint-dir', type=str, default=None,
                       help='Directory containing model checkpoints')
    parser.add_argument('--port', type=int, default=8000,
                       help='Server port')
    parser.add_argument('--dry-run', action='store_true',
                       help='Print config without loading models')

    args = parser.parse_args()

    # Get GPU config
    config = get_config(args.gpu)

    if args.dry_run:
        print_config_summary(config)
        usage = estimate_vram_usage(config)
        print(f"\nDry run complete. Estimated VRAM: {usage['total_gb']:.2f} GB")
        return

    # Initialize server
    checkpoint_dir = Path(args.checkpoint_dir) if args.checkpoint_dir else None
    server = ModelServer(config, checkpoint_dir)
    server.initialize()

    # Print health check
    health = server.health_check()
    print(f"\nServer Health: {json.dumps(health, indent=2)}")

    # In production, you would start a FastAPI/Flask server here
    # For now, just keep the server running
    print(f"\nServer ready. Models loaded: {health['models_loaded']}")
    print(f"VRAM usage: {health['vram_used_gb']:.2f} / {health['vram_total_gb']:.2f} GB "
          f"({health['vram_percent']}%)")


if __name__ == "__main__":
    main()
