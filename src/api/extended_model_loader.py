#!/usr/bin/env python3
"""
Extended Model Loader for CineSync v2
Integrates all 45 models with the unified inference system
"""

import sys
import os
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
import importlib
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class ExtendedModelType(Enum):
    """Extended model types covering all 45 models"""
    # Core models (from unified_inference_api)
    NCF = "ncf"
    SEQUENTIAL = "sequential_recommender"
    TWO_TOWER = "two_tower"

    # Movie-specific models
    MOVIE_FRANCHISE = "movie_franchise_sequence"
    MOVIE_DIRECTOR = "movie_director_auteur"
    MOVIE_UNIVERSE = "movie_cinematic_universe"
    MOVIE_AWARDS = "movie_awards_prediction"
    MOVIE_RUNTIME = "movie_runtime_preference"
    MOVIE_ERA = "movie_era_style"
    MOVIE_CRITIC = "movie_critic_audience"
    MOVIE_REMAKE = "movie_remake_connection"
    MOVIE_ACTOR = "movie_actor_collaboration"
    MOVIE_STUDIO = "movie_studio_fingerprint"
    MOVIE_ADAPTATION = "movie_adaptation_source"
    MOVIE_INTERNATIONAL = "movie_international"
    MOVIE_NARRATIVE = "movie_narrative_complexity"
    MOVIE_CONTEXT = "movie_viewing_context"

    # TV-specific models
    TV_TEMPORAL = "tv_temporal_attention"
    TV_GRAPH = "tv_graph_neural"
    TV_CONTRASTIVE = "tv_contrastive"
    TV_META = "tv_meta_learning"
    TV_ENSEMBLE = "tv_ensemble"
    TV_MULTIMODAL = "tv_multimodal"
    TV_EPISODE = "tv_episode_sequence"
    TV_BINGE = "tv_binge_prediction"
    TV_COMPLETION = "tv_series_completion"
    TV_SEASON = "tv_season_quality"
    TV_PLATFORM = "tv_platform_availability"
    TV_PATTERN = "tv_watch_pattern"
    TV_LIFECYCLE = "tv_series_lifecycle"
    TV_CAST = "tv_cast_migration"

    # Advanced content-agnostic models
    BERT4REC = "bert4rec"
    GRAPHSAGE = "graphsage"
    TRANSFORMER = "transformer_recommender"
    VAE = "vae_recommender"
    GNN = "gnn_recommender"
    ENHANCED_TWO_TOWER = "enhanced_two_tower"
    SENTENCE_BERT = "sentence_bert_two_tower"
    T5_HYBRID = "t5_hybrid"
    UNIFIED_CONTENT = "unified_content"

    # Unified models
    CROSS_DOMAIN = "cross_domain_embeddings"
    MOVIE_ENSEMBLE = "movie_ensemble"
    UNIFIED_CONTRASTIVE = "unified_contrastive"
    MULTIMODAL_FEATURES = "multimodal_features"
    CONTEXT_AWARE = "context_aware"


@dataclass
class ExtendedModelConfig:
    """Configuration for an extended model"""
    model_type: ExtendedModelType
    module_path: str
    model_class: str
    checkpoint_path: Optional[str] = None
    config_class: Optional[str] = None
    enabled: bool = True
    weight: float = 1.0
    content_type: str = "both"  # movie, tv, or both


# Model registry mapping to train_all_models.py
EXTENDED_MODEL_REGISTRY: Dict[ExtendedModelType, ExtendedModelConfig] = {
    # Movie-specific models
    ExtendedModelType.MOVIE_FRANCHISE: ExtendedModelConfig(
        model_type=ExtendedModelType.MOVIE_FRANCHISE,
        module_path="src.models.movie.franchise_sequence",
        model_class="FranchiseSequenceModel",
        config_class="FranchiseConfig",
        content_type="movie"
    ),
    ExtendedModelType.MOVIE_DIRECTOR: ExtendedModelConfig(
        model_type=ExtendedModelType.MOVIE_DIRECTOR,
        module_path="src.models.movie.director_auteur",
        model_class="DirectorAuteurModel",
        config_class="DirectorConfig",
        content_type="movie"
    ),
    ExtendedModelType.MOVIE_UNIVERSE: ExtendedModelConfig(
        model_type=ExtendedModelType.MOVIE_UNIVERSE,
        module_path="src.models.movie.cinematic_universe",
        model_class="CinematicUniverseModel",
        config_class="UniverseConfig",
        content_type="movie"
    ),
    ExtendedModelType.MOVIE_AWARDS: ExtendedModelConfig(
        model_type=ExtendedModelType.MOVIE_AWARDS,
        module_path="src.models.movie.awards_prediction",
        model_class="AwardsPredictionModel",
        config_class="AwardsConfig",
        content_type="movie"
    ),
    ExtendedModelType.MOVIE_RUNTIME: ExtendedModelConfig(
        model_type=ExtendedModelType.MOVIE_RUNTIME,
        module_path="src.models.movie.runtime_preference",
        model_class="RuntimePreferenceModel",
        config_class="RuntimeConfig",
        content_type="movie"
    ),
    ExtendedModelType.MOVIE_ERA: ExtendedModelConfig(
        model_type=ExtendedModelType.MOVIE_ERA,
        module_path="src.models.movie.era_style",
        model_class="EraStyleModel",
        config_class="EraConfig",
        content_type="movie"
    ),
    ExtendedModelType.MOVIE_CRITIC: ExtendedModelConfig(
        model_type=ExtendedModelType.MOVIE_CRITIC,
        module_path="src.models.movie.critic_audience",
        model_class="CriticAudienceModel",
        config_class="CriticConfig",
        content_type="movie"
    ),
    ExtendedModelType.MOVIE_REMAKE: ExtendedModelConfig(
        model_type=ExtendedModelType.MOVIE_REMAKE,
        module_path="src.models.movie.remake_connection",
        model_class="RemakeConnectionModel",
        config_class="RemakeConfig",
        content_type="movie"
    ),
    ExtendedModelType.MOVIE_ACTOR: ExtendedModelConfig(
        model_type=ExtendedModelType.MOVIE_ACTOR,
        module_path="src.models.movie.actor_collaboration",
        model_class="ActorCollaborationModel",
        config_class="ActorConfig",
        content_type="movie"
    ),
    ExtendedModelType.MOVIE_STUDIO: ExtendedModelConfig(
        model_type=ExtendedModelType.MOVIE_STUDIO,
        module_path="src.models.movie.studio_fingerprint",
        model_class="StudioFingerprintModel",
        config_class="StudioConfig",
        content_type="movie"
    ),
    ExtendedModelType.MOVIE_ADAPTATION: ExtendedModelConfig(
        model_type=ExtendedModelType.MOVIE_ADAPTATION,
        module_path="src.models.movie.adaptation_source",
        model_class="AdaptationSourceModel",
        config_class="AdaptationConfig",
        content_type="movie"
    ),
    ExtendedModelType.MOVIE_INTERNATIONAL: ExtendedModelConfig(
        model_type=ExtendedModelType.MOVIE_INTERNATIONAL,
        module_path="src.models.movie.international_cinema",
        model_class="InternationalCinemaModel",
        config_class="InternationalConfig",
        content_type="movie"
    ),
    ExtendedModelType.MOVIE_NARRATIVE: ExtendedModelConfig(
        model_type=ExtendedModelType.MOVIE_NARRATIVE,
        module_path="src.models.movie.narrative_complexity",
        model_class="NarrativeComplexityModel",
        config_class="NarrativeConfig",
        content_type="movie"
    ),
    ExtendedModelType.MOVIE_CONTEXT: ExtendedModelConfig(
        model_type=ExtendedModelType.MOVIE_CONTEXT,
        module_path="src.models.movie.viewing_context",
        model_class="ViewingContextModel",
        config_class="ViewingContextConfig",
        content_type="movie"
    ),

    # TV-specific models
    ExtendedModelType.TV_TEMPORAL: ExtendedModelConfig(
        model_type=ExtendedModelType.TV_TEMPORAL,
        module_path="src.models.hybrid.sota_tv.models.temporal_attention",
        model_class="TemporalAttentionTVModel",
        content_type="tv"
    ),
    ExtendedModelType.TV_GRAPH: ExtendedModelConfig(
        model_type=ExtendedModelType.TV_GRAPH,
        module_path="src.models.hybrid.sota_tv.models.graph_neural_network",
        model_class="TVGraphNeuralNetwork",
        content_type="tv"
    ),
    ExtendedModelType.TV_CONTRASTIVE: ExtendedModelConfig(
        model_type=ExtendedModelType.TV_CONTRASTIVE,
        module_path="src.models.hybrid.sota_tv.models.contrastive_learning",
        model_class="ContrastiveTVLearning",
        content_type="tv"
    ),
    ExtendedModelType.TV_META: ExtendedModelConfig(
        model_type=ExtendedModelType.TV_META,
        module_path="src.models.hybrid.sota_tv.models.meta_learning",
        model_class="MetaLearningTVModel",
        content_type="tv"
    ),
    ExtendedModelType.TV_ENSEMBLE: ExtendedModelConfig(
        model_type=ExtendedModelType.TV_ENSEMBLE,
        module_path="src.models.hybrid.sota_tv.models.ensemble_system",
        model_class="TVEnsembleSystem",
        content_type="tv"
    ),
    ExtendedModelType.TV_MULTIMODAL: ExtendedModelConfig(
        model_type=ExtendedModelType.TV_MULTIMODAL,
        module_path="src.models.hybrid.sota_tv.models.multimodal_transformer",
        model_class="MultimodalTVTransformer",
        content_type="tv"
    ),
    ExtendedModelType.TV_EPISODE: ExtendedModelConfig(
        model_type=ExtendedModelType.TV_EPISODE,
        module_path="src.models.hybrid.sota_tv.models.episode_sequence",
        model_class="EpisodeSequenceModel",
        config_class="EpisodeSequenceConfig",
        content_type="tv"
    ),
    ExtendedModelType.TV_BINGE: ExtendedModelConfig(
        model_type=ExtendedModelType.TV_BINGE,
        module_path="src.models.hybrid.sota_tv.models.binge_prediction",
        model_class="BingePredictionModel",
        config_class="BingePredictionConfig",
        content_type="tv"
    ),
    ExtendedModelType.TV_COMPLETION: ExtendedModelConfig(
        model_type=ExtendedModelType.TV_COMPLETION,
        module_path="src.models.hybrid.sota_tv.models.series_completion",
        model_class="SeriesCompletionModel",
        config_class="SeriesCompletionConfig",
        content_type="tv"
    ),
    ExtendedModelType.TV_SEASON: ExtendedModelConfig(
        model_type=ExtendedModelType.TV_SEASON,
        module_path="src.models.hybrid.sota_tv.models.season_quality",
        model_class="SeasonQualityModel",
        config_class="SeasonQualityConfig",
        content_type="tv"
    ),
    ExtendedModelType.TV_PLATFORM: ExtendedModelConfig(
        model_type=ExtendedModelType.TV_PLATFORM,
        module_path="src.models.hybrid.sota_tv.models.platform_availability",
        model_class="PlatformAvailabilityModel",
        config_class="PlatformConfig",
        content_type="tv"
    ),
    ExtendedModelType.TV_PATTERN: ExtendedModelConfig(
        model_type=ExtendedModelType.TV_PATTERN,
        module_path="src.models.hybrid.sota_tv.models.watch_pattern",
        model_class="WatchPatternModel",
        config_class="WatchPatternConfig",
        content_type="tv"
    ),
    ExtendedModelType.TV_LIFECYCLE: ExtendedModelConfig(
        model_type=ExtendedModelType.TV_LIFECYCLE,
        module_path="src.models.hybrid.sota_tv.models.series_lifecycle",
        model_class="SeriesLifecycleModel",
        config_class="LifecycleConfig",
        content_type="tv"
    ),
    ExtendedModelType.TV_CAST: ExtendedModelConfig(
        model_type=ExtendedModelType.TV_CAST,
        module_path="src.models.hybrid.sota_tv.models.cast_migration",
        model_class="CastMigrationModel",
        config_class="CastMigrationConfig",
        content_type="tv"
    ),

    # Advanced models
    ExtendedModelType.BERT4REC: ExtendedModelConfig(
        model_type=ExtendedModelType.BERT4REC,
        module_path="src.models.advanced.bert4rec_recommender",
        model_class="BERT4Rec",
        content_type="both"
    ),
    ExtendedModelType.GRAPHSAGE: ExtendedModelConfig(
        model_type=ExtendedModelType.GRAPHSAGE,
        module_path="src.models.advanced.graphsage_recommender",
        model_class="GraphSAGERecommender",
        content_type="both"
    ),
    ExtendedModelType.TRANSFORMER: ExtendedModelConfig(
        model_type=ExtendedModelType.TRANSFORMER,
        module_path="src.models.advanced.transformer_recommender",
        model_class="TransformerRecommender",
        content_type="both"
    ),
    ExtendedModelType.VAE: ExtendedModelConfig(
        model_type=ExtendedModelType.VAE,
        module_path="src.models.advanced.variational_autoencoder",
        model_class="VAERecommender",
        content_type="both"
    ),
    ExtendedModelType.GNN: ExtendedModelConfig(
        model_type=ExtendedModelType.GNN,
        module_path="src.models.advanced.graph_neural_network",
        model_class="GNNRecommender",
        content_type="both"
    ),
    ExtendedModelType.ENHANCED_TWO_TOWER: ExtendedModelConfig(
        model_type=ExtendedModelType.ENHANCED_TWO_TOWER,
        module_path="src.models.advanced.enhanced_two_tower",
        model_class="EnhancedTwoTower",
        content_type="both"
    ),
    ExtendedModelType.SENTENCE_BERT: ExtendedModelConfig(
        model_type=ExtendedModelType.SENTENCE_BERT,
        module_path="src.models.advanced.sentence_bert_two_tower",
        model_class="SentenceBERTTwoTower",
        content_type="both"
    ),
    ExtendedModelType.T5_HYBRID: ExtendedModelConfig(
        model_type=ExtendedModelType.T5_HYBRID,
        module_path="src.models.advanced.t5_hybrid_recommender",
        model_class="T5HybridRecommender",
        content_type="both"
    ),
    ExtendedModelType.UNIFIED_CONTENT: ExtendedModelConfig(
        model_type=ExtendedModelType.UNIFIED_CONTENT,
        module_path="src.models.hybrid.content_recommender",
        model_class="UnifiedContentRecommender",
        content_type="both"
    ),

    # Unified models
    ExtendedModelType.CROSS_DOMAIN: ExtendedModelConfig(
        model_type=ExtendedModelType.CROSS_DOMAIN,
        module_path="src.models.unified.cross_domain_embeddings",
        model_class="CrossDomainEmbeddings",
        content_type="both"
    ),
    ExtendedModelType.MOVIE_ENSEMBLE: ExtendedModelConfig(
        model_type=ExtendedModelType.MOVIE_ENSEMBLE,
        module_path="src.models.unified.movie_ensemble_system",
        model_class="MovieEnsembleSystem",
        content_type="movie"
    ),
    ExtendedModelType.UNIFIED_CONTRASTIVE: ExtendedModelConfig(
        model_type=ExtendedModelType.UNIFIED_CONTRASTIVE,
        module_path="src.models.unified.contrastive_learning",
        model_class="UnifiedContrastiveLearning",
        content_type="both"
    ),
    ExtendedModelType.MULTIMODAL_FEATURES: ExtendedModelConfig(
        model_type=ExtendedModelType.MULTIMODAL_FEATURES,
        module_path="src.models.unified.multimodal_features",
        model_class="MultimodalFeatures",
        content_type="both"
    ),
    ExtendedModelType.CONTEXT_AWARE: ExtendedModelConfig(
        model_type=ExtendedModelType.CONTEXT_AWARE,
        module_path="src.models.unified.context_aware",
        model_class="ContextAwareRecommender",
        content_type="both"
    ),
}


class ExtendedModelLoader:
    """
    Extended model loader that can load all 45 CineSync models.
    Provides a unified interface for model instantiation and verification.
    """

    def __init__(self, models_dir: str = None, device: str = None):
        self.models_dir = Path(models_dir) if models_dir else PROJECT_ROOT / "models"
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.loaded_models: Dict[ExtendedModelType, nn.Module] = {}
        self.model_errors: Dict[ExtendedModelType, str] = {}

        logger.info(f"ExtendedModelLoader initialized with device: {self.device}")

    def load_model(self, model_type: ExtendedModelType, checkpoint_path: str = None) -> Tuple[bool, Optional[str]]:
        """
        Load a specific model.

        Args:
            model_type: Type of model to load
            checkpoint_path: Optional path to model checkpoint

        Returns:
            Tuple of (success, error_message)
        """
        if model_type not in EXTENDED_MODEL_REGISTRY:
            return False, f"Unknown model type: {model_type}"

        config = EXTENDED_MODEL_REGISTRY[model_type]

        try:
            # Import module
            module = importlib.import_module(config.module_path)
            model_class = getattr(module, config.model_class)

            # Create model instance
            if config.config_class:
                config_class = getattr(module, config.config_class)
                model_config = config_class()
                model = model_class(model_config)
            else:
                model = model_class()

            # Load checkpoint if available
            if checkpoint_path:
                checkpoint = torch.load(checkpoint_path, map_location=self.device)
                if 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    model.load_state_dict(checkpoint)

            model = model.to(self.device)
            model.eval()

            self.loaded_models[model_type] = model
            logger.info(f"Successfully loaded model: {model_type.value}")
            return True, None

        except Exception as e:
            error_msg = f"Failed to load {model_type.value}: {str(e)}"
            self.model_errors[model_type] = error_msg
            logger.error(error_msg)
            return False, error_msg

    def verify_model(self, model_type: ExtendedModelType) -> Tuple[bool, Optional[str]]:
        """
        Verify a model can be instantiated (without loading weights).

        Args:
            model_type: Type of model to verify

        Returns:
            Tuple of (success, error_message)
        """
        if model_type not in EXTENDED_MODEL_REGISTRY:
            return False, f"Unknown model type: {model_type}"

        config = EXTENDED_MODEL_REGISTRY[model_type]

        try:
            # Try to import module and class
            module = importlib.import_module(config.module_path)
            model_class = getattr(module, config.model_class)

            # Verify config class if specified
            if config.config_class:
                config_class = getattr(module, config.config_class)
                # Try to instantiate config
                _ = config_class()

            return True, None

        except ImportError as e:
            return False, f"Import error: {str(e)}"
        except AttributeError as e:
            return False, f"Class not found: {str(e)}"
        except Exception as e:
            return False, f"Verification error: {str(e)}"

    def verify_all_models(self) -> Dict[str, Dict[str, Any]]:
        """
        Verify all 45 models can be instantiated.

        Returns:
            Dictionary with verification results for each model
        """
        results = {
            "movie_specific": {},
            "tv_specific": {},
            "content_agnostic": {},
            "unified": {},
            "summary": {
                "total": 0,
                "success": 0,
                "failed": 0
            }
        }

        for model_type in ExtendedModelType:
            config = EXTENDED_MODEL_REGISTRY.get(model_type)
            if not config:
                continue

            success, error = self.verify_model(model_type)
            results["summary"]["total"] += 1

            if success:
                results["summary"]["success"] += 1
            else:
                results["summary"]["failed"] += 1

            # Categorize result
            if model_type.value.startswith("movie_"):
                category = "movie_specific"
            elif model_type.value.startswith("tv_"):
                category = "tv_specific"
            elif model_type in [ExtendedModelType.CROSS_DOMAIN, ExtendedModelType.MOVIE_ENSEMBLE,
                               ExtendedModelType.UNIFIED_CONTRASTIVE, ExtendedModelType.MULTIMODAL_FEATURES,
                               ExtendedModelType.CONTEXT_AWARE]:
                category = "unified"
            else:
                category = "content_agnostic"

            results[category][model_type.value] = {
                "success": success,
                "error": error,
                "module": config.module_path,
                "class": config.model_class
            }

        return results

    def get_models_by_content_type(self, content_type: str) -> List[ExtendedModelType]:
        """Get models that support a specific content type"""
        matching = []
        for model_type, config in EXTENDED_MODEL_REGISTRY.items():
            if config.content_type == content_type or config.content_type == "both":
                matching.append(model_type)
        return matching

    def get_loaded_model(self, model_type: ExtendedModelType) -> Optional[nn.Module]:
        """Get a loaded model instance"""
        return self.loaded_models.get(model_type)

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about all models"""
        return {
            "total_models": len(EXTENDED_MODEL_REGISTRY),
            "loaded_models": [mt.value for mt in self.loaded_models.keys()],
            "failed_models": list(self.model_errors.keys()),
            "movie_specific": len([mt for mt in ExtendedModelType if mt.value.startswith("movie_")]),
            "tv_specific": len([mt for mt in ExtendedModelType if mt.value.startswith("tv_")]),
            "device": self.device
        }


def run_verification():
    """Run verification of all 45 models"""
    loader = ExtendedModelLoader()
    results = loader.verify_all_models()

    print("\n" + "="*70)
    print("CINESYNC V2 - MODEL VERIFICATION REPORT")
    print("="*70)

    print(f"\nSUMMARY:")
    print(f"  Total models: {results['summary']['total']}")
    print(f"  Verified: {results['summary']['success']}")
    print(f"  Failed: {results['summary']['failed']}")

    for category in ["movie_specific", "tv_specific", "content_agnostic", "unified"]:
        if results[category]:
            print(f"\n{category.upper().replace('_', ' ')} MODELS:")
            for model_name, info in results[category].items():
                status = "✓" if info['success'] else "✗"
                print(f"  {status} {model_name}")
                if not info['success']:
                    print(f"      Error: {info['error']}")

    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_verification()
