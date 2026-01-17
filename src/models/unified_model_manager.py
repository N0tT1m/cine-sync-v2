"""
CineSync v2 - Unified Model Management System
Handles all AI models with drop-in integration

Model Categories:
- 14 Movie-specific models
- 14 TV-specific models
- 25 Content-agnostic models
- 3 Unified (both) models
"""

import os
import logging
import json
import pickle
import asyncio
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import torch
import numpy as np

# Import paths updated for new structure
from .advanced.bert4rec_recommender import BERT4Rec
from .advanced.sentence_bert_two_tower import SentenceBERTTwoTowerModel
from .advanced.graphsage_recommender import GraphSAGERecommender
from .advanced.t5_hybrid_recommender import T5HybridRecommender
from .advanced.enhanced_two_tower import UltimateTwoTowerModel
from .advanced.variational_autoencoder import MultVAE

# Import unified hybrid system (replaces separate movie/tv)
from .hybrid.content_recommender import UnifiedContentRecommender
from .hybrid.config import load_config, ContentType

# Import SOTA TV models
from .hybrid.sota_tv.models.temporal_attention import TemporalAttentionTVModel
from .hybrid.sota_tv.models.graph_neural_network import TVGraphRecommender
from .hybrid.sota_tv.models.contrastive_learning import ContrastiveTVModel
from .hybrid.sota_tv.models.ensemble_system import TVEnsembleRecommender
from .hybrid.sota_tv.models.meta_learning import MetaLearningTVAdapter
from .hybrid.sota_tv.models.multimodal_transformer import MultimodalTransformerTV


class ModelCategory(Enum):
    """Model category enumeration"""
    MOVIE_SPECIFIC = "movie_specific"
    TV_SPECIFIC = "tv_specific"
    CONTENT_AGNOSTIC = "content_agnostic"
    UNIFIED = "unified"


@dataclass
class ModelConfig:
    """Configuration for each model type"""
    name: str
    model_class: type
    config_path: str
    weights_path: str
    enabled: bool = True
    content_type: str = "both"  # "movie", "tv", "both"
    category: ModelCategory = ModelCategory.CONTENT_AGNOSTIC
    priority: int = 1  # Higher = more priority
    description: str = ""


class ModelStatus:
    """Track model loading and health status"""
    def __init__(self):
        self.loaded_models: Dict[str, bool] = {}
        self.model_errors: Dict[str, str] = {}
        self.last_updated: Dict[str, datetime] = {}
        self.model_metrics: Dict[str, Dict] = {}


class UnifiedModelManager:
    """
    Central hub for managing all recommendation models.

    Supports:
    - 14 Movie-specific models
    - 14 TV-specific models
    - 25 Content-agnostic models
    - 3 Unified models

    Provides drop-in integration, training management, and admin controls.
    """

    def __init__(self, models_dir: str = "models", config_file: str = "model_config.json"):
        self.models_dir = Path(models_dir)
        self.config_file = config_file
        self.models: Dict[str, Any] = {}
        self.status = ModelStatus()
        self.logger = self._setup_logging()

        # Ensure models directory structure exists
        self._ensure_directory_structure()

        # Initialize model configurations
        self.model_configs = self._load_model_configs()

        # Training preferences
        self.training_preferences = {
            "auto_retrain": True,
            "min_feedback_threshold": 100,
            "excluded_genres": [],
            "excluded_users": [],
            "quality_filters": ["4K", "2K", "1080p"],
        }

    def _ensure_directory_structure(self):
        """Ensure proper directory structure exists"""
        dirs = [
            self.models_dir,
            self.models_dir / "movies",
            self.models_dir / "tv",
            self.models_dir / "weights",
            self.models_dir / "configs",
        ]
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)

    def _setup_logging(self) -> logging.Logger:
        """Setup logging for model management"""
        logger = logging.getLogger("UnifiedModelManager")
        logger.setLevel(logging.INFO)

        if not logger.handlers:
            handler = logging.FileHandler("unified_model_manager.log")
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        return logger

    def _load_model_configs(self) -> Dict[str, ModelConfig]:
        """Load or create default model configurations"""
        configs = {}

        # ============================================
        # CONTENT-AGNOSTIC MODELS (25 models)
        # ============================================
        configs["bert4rec"] = ModelConfig(
            name="BERT4Rec",
            model_class=BERT4Rec,
            config_path="configs/bert4rec_config.json",
            weights_path="weights/bert4rec.pt",
            content_type="both",
            category=ModelCategory.CONTENT_AGNOSTIC,
            priority=5,
            description="Bidirectional transformer for sequential recommendation"
        )
        configs["sentence_bert_two_tower"] = ModelConfig(
            name="Sentence-BERT Two-Tower",
            model_class=SentenceBERTTwoTowerModel,
            config_path="configs/sentence_bert_config.json",
            weights_path="weights/sentence_bert_two_tower.pt",
            content_type="both",
            category=ModelCategory.CONTENT_AGNOSTIC,
            priority=4,
            description="Semantic text embeddings for content matching"
        )
        configs["graphsage"] = ModelConfig(
            name="GraphSAGE",
            model_class=GraphSAGERecommender,
            config_path="configs/graphsage_config.json",
            weights_path="weights/graphsage.pt",
            content_type="both",
            category=ModelCategory.CONTENT_AGNOSTIC,
            priority=3,
            description="Graph neural network for user-item interactions"
        )
        configs["t5_hybrid"] = ModelConfig(
            name="T5 Hybrid",
            model_class=T5HybridRecommender,
            config_path="configs/t5_hybrid_config.json",
            weights_path="weights/t5_hybrid.pt",
            content_type="both",
            category=ModelCategory.CONTENT_AGNOSTIC,
            priority=4,
            description="T5-based content understanding with collaborative filtering"
        )
        configs["enhanced_two_tower"] = ModelConfig(
            name="Enhanced Two-Tower",
            model_class=UltimateTwoTowerModel,
            config_path="configs/enhanced_two_tower_config.json",
            weights_path="weights/enhanced_two_tower.pt",
            content_type="both",
            category=ModelCategory.CONTENT_AGNOSTIC,
            priority=3,
            description="Advanced two-tower architecture with cross-attention"
        )
        configs["variational_autoencoder"] = ModelConfig(
            name="Variational AutoEncoder",
            model_class=MultVAE,
            config_path="configs/vae_config.json",
            weights_path="weights/vae.pt",
            content_type="both",
            category=ModelCategory.CONTENT_AGNOSTIC,
            priority=2,
            description="VAE for collaborative filtering"
        )

        # ============================================
        # UNIFIED MODELS (handles both movie and TV)
        # ============================================
        configs["unified_recommender"] = ModelConfig(
            name="Unified Content Recommender",
            model_class=UnifiedContentRecommender,
            config_path="hybrid/config.py",
            weights_path="weights/unified_recommender.pt",
            content_type="both",
            category=ModelCategory.UNIFIED,
            priority=5,
            description="Unified NCF model for movies and TV with content-type parameter"
        )

        # ============================================
        # TV-SPECIFIC MODELS (14 models)
        # ============================================
        # Existing 6 SOTA TV models
        configs["tv_temporal_attention"] = ModelConfig(
            name="TV Temporal Attention",
            model_class=TemporalAttentionTVModel,
            config_path="configs/tv_temporal_attention.json",
            weights_path="tv/temporal_attention.pt",
            content_type="tv",
            category=ModelCategory.TV_SPECIFIC,
            priority=5,
            description="Temporal patterns and seasonal decomposition for TV"
        )
        configs["tv_graph_recommender"] = ModelConfig(
            name="TV Graph Recommender",
            model_class=TVGraphRecommender,
            config_path="configs/tv_graph.json",
            weights_path="tv/graph_recommender.pt",
            content_type="tv",
            category=ModelCategory.TV_SPECIFIC,
            priority=4,
            description="Heterogeneous graph with meta-paths for TV"
        )
        configs["tv_contrastive"] = ModelConfig(
            name="TV Contrastive Learning",
            model_class=ContrastiveTVModel,
            config_path="configs/tv_contrastive.json",
            weights_path="tv/contrastive.pt",
            content_type="tv",
            category=ModelCategory.TV_SPECIFIC,
            priority=4,
            description="Self-supervised contrastive learning for TV"
        )
        configs["tv_ensemble"] = ModelConfig(
            name="TV Ensemble Recommender",
            model_class=TVEnsembleRecommender,
            config_path="configs/tv_ensemble.json",
            weights_path="tv/ensemble.pt",
            content_type="tv",
            category=ModelCategory.TV_SPECIFIC,
            priority=5,
            description="Multi-model ensemble with uncertainty estimation"
        )
        configs["tv_meta_learning"] = ModelConfig(
            name="TV Meta Learning Adapter",
            model_class=MetaLearningTVAdapter,
            config_path="configs/tv_meta_learning.json",
            weights_path="tv/meta_learning.pt",
            content_type="tv",
            category=ModelCategory.TV_SPECIFIC,
            priority=3,
            description="Few-shot adaptation for new TV shows"
        )
        configs["tv_multimodal"] = ModelConfig(
            name="TV Multimodal Transformer",
            model_class=MultimodalTransformerTV,
            config_path="configs/tv_multimodal.json",
            weights_path="tv/multimodal.pt",
            content_type="tv",
            category=ModelCategory.TV_SPECIFIC,
            priority=4,
            description="Cross-modal attention for TV metadata"
        )

        # New TV models (8 more to reach 14) - placeholders for now
        tv_new_models = [
            ("tv_episode_sequence", "Episode Sequence Model", "Within-show episode recommendations"),
            ("tv_binge_prediction", "Binge Prediction Model", "Predict binge-watching behavior"),
            ("tv_series_completion", "Series Completion Model", "Predict show finish likelihood"),
            ("tv_season_quality", "Season Quality Model", "Per-season quality scoring"),
            ("tv_platform_availability", "Platform Availability Model", "Cross-platform recommendations"),
            ("tv_watch_pattern", "Watch Pattern Model", "Weekly vs binge pattern modeling"),
            ("tv_series_lifecycle", "Series Lifecycle Model", "New/returning/finale recommendations"),
            ("tv_cast_migration", "Cast Migration Model", "Actor/creator graph across shows"),
        ]

        for model_id, name, desc in tv_new_models:
            configs[model_id] = ModelConfig(
                name=name,
                model_class=None,  # To be implemented
                config_path=f"configs/{model_id}.json",
                weights_path=f"tv/{model_id}.pt",
                content_type="tv",
                category=ModelCategory.TV_SPECIFIC,
                priority=3,
                enabled=False,  # Disabled until implemented
                description=desc
            )

        # ============================================
        # MOVIE-SPECIFIC MODELS (14 models)
        # ============================================
        movie_models = [
            ("movie_franchise_sequence", "Franchise Sequence Model", "Sequel/prequel ordering", 5),
            ("movie_director_auteur", "Director Auteur Model", "Director filmography matching", 4),
            ("movie_cinematic_universe", "Cinematic Universe Model", "Connected universe navigation", 4),
            ("movie_awards_prediction", "Awards Prediction Model", "Oscar/prestige recommendations", 3),
            ("movie_runtime_preference", "Runtime Preference Model", "Time-aware movie selection", 3),
            ("movie_era_style", "Era Style Model", "Decade/era taste modeling", 3),
            ("movie_critic_audience", "Critic Audience Model", "Score alignment preferences", 4),
            ("movie_remake_connection", "Remake Connection Model", "Original/remake relationships", 3),
            ("movie_actor_collaboration", "Actor Collaboration Model", "Actor pairings and chemistry", 4),
            ("movie_studio_fingerprint", "Studio Fingerprint Model", "Studio style preferences", 3),
            ("movie_adaptation_source", "Adaptation Source Model", "Book/comic/game adaptations", 3),
            ("movie_international", "International Cinema Model", "Country/region preferences", 3),
            ("movie_narrative_complexity", "Narrative Complexity Model", "Storytelling structure prefs", 3),
            ("movie_viewing_context", "Viewing Context Model", "Context-aware recommendations", 4),
        ]

        for model_id, name, desc, priority in movie_models:
            configs[model_id] = ModelConfig(
                name=name,
                model_class=None,  # To be implemented
                config_path=f"configs/{model_id}.json",
                weights_path=f"movies/{model_id}.pt",
                content_type="movie",
                category=ModelCategory.MOVIE_SPECIFIC,
                priority=priority,
                enabled=False,  # Disabled until implemented
                description=desc
            )

        # Save default config if doesn't exist
        if not os.path.exists(self.config_file):
            self.save_model_configs(configs)

        return configs

    def save_model_configs(self, configs: Dict[str, ModelConfig]):
        """Save model configurations to JSON"""
        config_dict = {}
        for key, config in configs.items():
            config_dict[key] = {
                "name": config.name,
                "config_path": config.config_path,
                "weights_path": config.weights_path,
                "enabled": config.enabled,
                "content_type": config.content_type,
                "category": config.category.value,
                "priority": config.priority,
                "description": config.description
            }

        with open(self.config_file, 'w') as f:
            json.dump(config_dict, f, indent=2)

    def get_models_by_category(self, category: ModelCategory) -> Dict[str, ModelConfig]:
        """Get all models in a specific category"""
        return {
            name: config for name, config in self.model_configs.items()
            if config.category == category
        }

    def get_model_summary(self) -> Dict[str, int]:
        """Get summary count of models by category"""
        summary = {
            "movie_specific": 0,
            "tv_specific": 0,
            "content_agnostic": 0,
            "unified": 0,
            "total": 0,
            "enabled": 0,
            "loaded": 0
        }

        for name, config in self.model_configs.items():
            summary[config.category.value] += 1
            summary["total"] += 1
            if config.enabled:
                summary["enabled"] += 1
            if name in self.status.loaded_models and self.status.loaded_models[name]:
                summary["loaded"] += 1

        return summary

    async def load_all_models(self) -> Dict[str, bool]:
        """Load all enabled models asynchronously"""
        self.logger.info("Starting to load all models...")

        load_tasks = []
        enabled_models = [(name, config) for name, config in self.model_configs.items()
                         if config.enabled and config.model_class is not None]

        for model_name, config in enabled_models:
            task = asyncio.create_task(self._load_single_model(model_name, config))
            load_tasks.append((model_name, task))

        # Wait for all tasks
        for model_name, task in load_tasks:
            try:
                await task
                self.status.loaded_models[model_name] = True
                self.logger.info(f"Successfully loaded {model_name}")
            except Exception as e:
                self.status.model_errors[model_name] = str(e)
                self.status.loaded_models[model_name] = False
                self.logger.error(f"Failed to load {model_name}: {e}")

        summary = self.get_model_summary()
        self.logger.info(f"Loaded {summary['loaded']}/{summary['enabled']} enabled models")
        return self.status.loaded_models

    async def _load_single_model(self, model_name: str, config: ModelConfig):
        """Load a single model with error handling"""
        try:
            weights_path = self.models_dir / config.weights_path

            if weights_path.exists():
                # Load existing weights
                checkpoint = torch.load(weights_path, map_location='cpu', weights_only=False)
                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    model = self._initialize_new_model(model_name, config)
                    model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    model = checkpoint
            else:
                # Initialize new model if weights don't exist
                model = self._initialize_new_model(model_name, config)

            self.models[model_name] = model
            self.status.last_updated[model_name] = datetime.now()

        except Exception as e:
            raise Exception(f"Failed to load {model_name}: {str(e)}")

    def _initialize_new_model(self, model_name: str, config: ModelConfig):
        """Initialize a new model with default parameters"""
        default_params = {
            "bert4rec": {"num_items": 10000, "d_model": 512, "num_heads": 8, "num_layers": 6},
            "sentence_bert_two_tower": {"embedding_dim": 256},
            "graphsage": {"num_users": 100000, "num_items": 10000, "embedding_dim": 256},
            "t5_hybrid": {"num_users": 100000, "num_items": 10000, "embedding_dim": 256},
            "enhanced_two_tower": {"num_users": 100000, "num_items": 10000, "embedding_dim": 256},
            "variational_autoencoder": {"num_items": 10000, "hidden_dim": 256, "latent_dim": 64},
            "unified_recommender": {"num_users": 100000, "num_items": 50000, "embedding_dim": 64},
            # TV models
            "tv_temporal_attention": {"vocab_sizes": {"shows": 10000, "genres": 50, "networks": 100}, "d_model": 512},
            "tv_graph_recommender": {"num_users": 100000, "num_shows": 10000, "embedding_dim": 128},
            "tv_contrastive": {"vocab_sizes": {"genres": 50, "networks": 100}, "embed_dim": 768},
            "tv_ensemble": {"num_users": 100000, "num_shows": 10000},
            "tv_meta_learning": {"embedding_dim": 128, "num_genres": 20},
            "tv_multimodal": {"vocab_sizes": {"genres": 50, "networks": 100}, "num_shows": 10000},
        }

        params = default_params.get(model_name, {})
        if config.model_class is not None:
            return config.model_class(**params)
        return None

    def get_recommendations(
        self,
        user_id: int,
        content_type: str = "movie",
        top_k: int = 10,
        model_preference: Optional[str] = None
    ) -> List[Dict]:
        """
        Get recommendations using the best available model

        Args:
            user_id: User identifier
            content_type: "movie", "tv", or "both"
            top_k: Number of recommendations
            model_preference: Specific model to use (optional)

        Returns:
            List of recommendation dictionaries
        """

        # Select best model for content type
        available_models = self._get_available_models(content_type)

        if not available_models:
            self.logger.warning(f"No models available for content_type={content_type}")
            return []

        if model_preference and model_preference in available_models:
            selected_model = model_preference
        else:
            # Select highest priority available model
            selected_model = max(available_models.keys(),
                               key=lambda x: self.model_configs[x].priority)

        try:
            model = self.models[selected_model]
            config = self.model_configs[selected_model]

            # Route to appropriate recommendation method
            if config.category == ModelCategory.UNIFIED:
                ct = ContentType.MOVIE if content_type == "movie" else ContentType.TV
                return self._get_unified_recommendations(model, user_id, ct, top_k)
            elif config.category == ModelCategory.TV_SPECIFIC:
                return self._get_tv_recommendations(model, selected_model, user_id, top_k)
            elif config.category == ModelCategory.MOVIE_SPECIFIC:
                return self._get_movie_recommendations(model, selected_model, user_id, top_k)
            else:
                return self._get_agnostic_recommendations(model, selected_model, user_id, content_type, top_k)

        except Exception as e:
            self.logger.error(f"Recommendation failed for {selected_model}: {e}")
            # Fallback to next best model
            fallback_models = sorted(
                [m for m in available_models.keys() if m != selected_model],
                key=lambda x: self.model_configs[x].priority,
                reverse=True
            )

            for fallback in fallback_models:
                try:
                    return self.get_recommendations(user_id, content_type, top_k, fallback)
                except:
                    continue

            return []

    def _get_unified_recommendations(
        self, model: UnifiedContentRecommender, user_id: int,
        content_type: ContentType, top_k: int
    ) -> List[Dict]:
        """Get recommendations from unified model"""
        if hasattr(model, 'get_top_recommendations'):
            # Get all item candidates (would come from database in production)
            candidates = list(range(1, 10000))
            recs = model.get_top_recommendations(
                user_id, candidates, content_type, top_k=top_k
            )
            return [
                {"item_id": item_id, "score": score, "content_type": content_type.value}
                for item_id, score in recs
            ]
        return []

    def _get_tv_recommendations(
        self, model, model_name: str, user_id: int, top_k: int
    ) -> List[Dict]:
        """Get recommendations from TV-specific model"""
        if hasattr(model, 'get_recommendations'):
            return model.get_recommendations(user_id, k=top_k)
        elif hasattr(model, 'recommend'):
            return model.recommend(user_id, top_k)
        return self._get_agnostic_recommendations(model, model_name, user_id, "tv", top_k)

    def _get_movie_recommendations(
        self, model, model_name: str, user_id: int, top_k: int
    ) -> List[Dict]:
        """Get recommendations from movie-specific model"""
        if hasattr(model, 'get_recommendations'):
            return model.get_recommendations(user_id, k=top_k)
        elif hasattr(model, 'recommend'):
            return model.recommend(user_id, top_k)
        return self._get_agnostic_recommendations(model, model_name, user_id, "movie", top_k)

    def _get_available_models(self, content_type: str) -> Dict[str, ModelConfig]:
        """Get models that support the requested content type"""
        available = {}
        for name, config in self.model_configs.items():
            if (name in self.status.loaded_models and
                self.status.loaded_models[name] and
                config.enabled and
                (config.content_type == content_type or config.content_type == "both")):
                available[name] = config
        return available

    def _get_agnostic_recommendations(
        self, model, model_name: str, user_id: int, content_type: str, top_k: int
    ) -> List[Dict]:
        """Standardized interface for content-agnostic model recommendations"""

        if hasattr(model, 'get_recommendations'):
            return model.get_recommendations(user_id, k=top_k)
        elif hasattr(model, 'predict'):
            candidates = self._get_item_candidates(content_type, top_k * 2)
            scores = model.predict(user_id, candidates)

            recommendations = []
            for item_id, score in zip(candidates, scores):
                recommendations.append({
                    "item_id": item_id,
                    "score": float(score),
                    "content_type": content_type
                })

            return sorted(recommendations, key=lambda x: x['score'], reverse=True)[:top_k]
        else:
            self.logger.warning(f"Model {model_name} doesn't have a recommendation interface")
            return []

    def _get_item_candidates(self, content_type: str, num_candidates: int) -> List[int]:
        """Get candidate items for recommendation scoring"""
        # This would connect to database to get popular/relevant items
        return list(range(1, num_candidates + 1))

    def add_feedback(self, user_id: int, item_id: int, rating: float, content_type: str):
        """Add user feedback for model retraining"""
        feedback = {
            "user_id": user_id,
            "item_id": item_id,
            "rating": rating,
            "content_type": content_type,
            "timestamp": datetime.now().isoformat(),
        }

        if self._should_include_in_training(feedback):
            self._store_feedback(feedback)

            if self.training_preferences["auto_retrain"]:
                self._maybe_trigger_retraining()

    def _should_include_in_training(self, feedback: Dict) -> bool:
        """Check if feedback should be included in training"""
        if feedback["user_id"] in self.training_preferences["excluded_users"]:
            return False
        return True

    def _store_feedback(self, feedback: Dict):
        """Store feedback for later training"""
        feedback_file = self.models_dir / "training_feedback.jsonl"

        with open(feedback_file, 'a') as f:
            f.write(json.dumps(feedback) + '\n')

    def _maybe_trigger_retraining(self):
        """Check if we should trigger model retraining"""
        feedback_file = self.models_dir / "training_feedback.jsonl"

        if feedback_file.exists():
            with open(feedback_file, 'r') as f:
                feedback_count = sum(1 for _ in f)

            if feedback_count >= self.training_preferences["min_feedback_threshold"]:
                self.logger.info(f"Triggering retraining with {feedback_count} feedback samples")
                asyncio.create_task(self.retrain_models())

    async def retrain_models(self):
        """Retrain models with new feedback data"""
        self.logger.info("Starting model retraining...")

        feedback_data = self._load_feedback_data()

        for model_name, model in self.models.items():
            if hasattr(model, 'retrain') or hasattr(model, 'incremental_train'):
                try:
                    await self._retrain_single_model(model_name, model, feedback_data)
                except Exception as e:
                    self.logger.error(f"Retraining failed for {model_name}: {e}")

    def _load_feedback_data(self) -> List[Dict]:
        """Load feedback data for retraining"""
        feedback_file = self.models_dir / "training_feedback.jsonl"
        feedback_data = []

        if feedback_file.exists():
            with open(feedback_file, 'r') as f:
                for line in f:
                    feedback_data.append(json.loads(line))

        return feedback_data

    async def _retrain_single_model(self, model_name: str, model, feedback_data: List[Dict]):
        """Retrain a single model with feedback data"""
        self.logger.info(f"Retraining {model_name}...")

        training_data = self._convert_feedback_to_training_data(feedback_data, model_name)

        if hasattr(model, 'incremental_train'):
            model.incremental_train(training_data)
        elif hasattr(model, 'retrain'):
            model.retrain(training_data)

        self._save_model(model_name, model)
        self.logger.info(f"Completed retraining {model_name}")

    def _convert_feedback_to_training_data(self, feedback_data: List[Dict], model_name: str):
        """Convert feedback to model-specific training format"""
        return feedback_data

    def _save_model(self, model_name: str, model):
        """Save trained model to disk"""
        config = self.model_configs[model_name]
        weights_path = self.models_dir / config.weights_path

        weights_path.parent.mkdir(parents=True, exist_ok=True)

        if hasattr(model, 'state_dict'):
            torch.save({
                'model_state_dict': model.state_dict(),
                'model_name': model_name,
                'saved_at': datetime.now().isoformat()
            }, weights_path)
        else:
            torch.save(model, weights_path)

    def get_model_status(self) -> Dict:
        """Get status of all models"""
        summary = self.get_model_summary()
        return {
            "loaded_models": self.status.loaded_models,
            "model_errors": self.status.model_errors,
            "last_updated": {k: v.isoformat() for k, v in self.status.last_updated.items()},
            "summary": summary
        }

    def update_training_preferences(self, preferences: Dict):
        """Update training preferences from admin interface"""
        self.training_preferences.update(preferences)

        prefs_file = self.models_dir / "training_preferences.json"
        with open(prefs_file, 'w') as f:
            json.dump(self.training_preferences, f, indent=2)

        self.logger.info("Updated training preferences")

    def enable_model(self, model_name: str, enabled: bool = True):
        """Enable or disable a model"""
        if model_name in self.model_configs:
            self.model_configs[model_name].enabled = enabled
            self.save_model_configs(self.model_configs)
            self.logger.info(f"Model {model_name} {'enabled' if enabled else 'disabled'}")
            return True
        return False


# Global model manager instance
model_manager = UnifiedModelManager()

async def initialize_models():
    """Initialize all models - call this at startup"""
    return await model_manager.load_all_models()

def get_recommendations(user_id: int, content_type: str = "movie", top_k: int = 10) -> List[Dict]:
    """Simple interface for getting recommendations"""
    return model_manager.get_recommendations(user_id, content_type, top_k)

def add_user_feedback(user_id: int, item_id: int, rating: float, content_type: str = "movie"):
    """Simple interface for adding user feedback"""
    model_manager.add_feedback(user_id, item_id, rating, content_type)

def get_model_status() -> Dict:
    """Get status of all models"""
    return model_manager.get_model_status()
