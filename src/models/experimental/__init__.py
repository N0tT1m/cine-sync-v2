"""Experimental models.

Models in this namespace are *not* on the shipped serving path. They have no
trained artifacts, no registry manifests, and are not wired into the inference
service's default ensemble. They exist for research and as candidates for
promotion once they prove out.

Why this is an aggregator and not a real package move:
  The original files (under `src/models/movie/`, `src/models/hybrid/sota_tv/`,
  `src/models/unified/movie_ensemble_system.py`, etc.) are still imported by
  `src/api/extended_model_loader.py`, `src/inference/model_server.py`, and
  multiple training scripts. Physically moving them would cascade into those
  consumers. Instead we pin the canonical import path here so new code has one
  obvious place to pull from, while the legacy paths keep working.

Canonical path for new code:
    from src.models.experimental import FranchiseSequenceModel

Experimental deps: some of these models require optional, heavy libraries
(`torch_geometric`, `sentence-transformers`, `open_clip_torch`, etc.). The
aggregator imports defensively — a missing dep skips that model instead of
breaking the whole namespace. Check `EXPERIMENTAL_LOAD_ERRORS` for what didn't
load in the current environment.

Promotion workflow when a model earns its keep:
  1. Add a manifest under `models/<name>/manifest.yaml`.
  2. Add the name to `settings.enabled_models`.
  3. Add a scorer adapter in `src/models/adapters/` if the model class
     doesn't already satisfy the Scorer protocol.
  4. Remove its re-export from this module and move training code out of here.
"""
from __future__ import annotations

import importlib
import logging
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)


# (module_path, class_name, exposed_name_override_or_None)
_SPECS: List[Tuple[str, str, str]] = [
    # Movie-domain (src/models/movie/) — 14 content-feature sub-models
    ("src.models.movie.actor_collaboration", "ActorCollaborationModel", ""),
    ("src.models.movie.adaptation_source", "AdaptationSourceModel", ""),
    ("src.models.movie.awards_prediction", "AwardsPredictionModel", ""),
    ("src.models.movie.cinematic_universe", "CinematicUniverseModel", ""),
    ("src.models.movie.critic_audience", "CriticAudienceModel", ""),
    ("src.models.movie.director_auteur", "DirectorAuteurModel", ""),
    ("src.models.movie.era_style", "EraStyleModel", ""),
    ("src.models.movie.franchise_sequence", "FranchiseSequenceModel", ""),
    ("src.models.movie.international_cinema", "InternationalCinemaModel", ""),
    ("src.models.movie.narrative_complexity", "NarrativeComplexityModel", ""),
    ("src.models.movie.remake_connection", "RemakeConnectionModel", ""),
    ("src.models.movie.runtime_preference", "RuntimePreferenceModel", ""),
    ("src.models.movie.studio_fingerprint", "StudioFingerprintModel", ""),
    ("src.models.movie.viewing_context", "ViewingContextModel", ""),
    # TV-domain (src/models/hybrid/sota_tv/models/) — needs torch_geometric etc.
    ("src.models.hybrid.sota_tv.models.binge_prediction", "BingePredictionModel", ""),
    ("src.models.hybrid.sota_tv.models.cast_migration", "CastMigrationModel", ""),
    ("src.models.hybrid.sota_tv.models.contrastive_learning", "ContrastiveTVLearning", ""),
    ("src.models.hybrid.sota_tv.models.ensemble_system", "TVEnsembleSystem", ""),
    ("src.models.hybrid.sota_tv.models.episode_sequence", "EpisodeSequenceModel", ""),
    ("src.models.hybrid.sota_tv.models.graph_neural_network", "TVGraphNeuralNetwork", ""),
    ("src.models.hybrid.sota_tv.models.meta_learning", "MetaLearningTVModel", ""),
    ("src.models.hybrid.sota_tv.models.multimodal_transformer", "MultimodalTVTransformer", ""),
    ("src.models.hybrid.sota_tv.models.platform_availability", "PlatformAvailabilityModel", ""),
    ("src.models.hybrid.sota_tv.models.season_quality", "SeasonQualityModel", ""),
    ("src.models.hybrid.sota_tv.models.series_completion", "SeriesCompletionModel", ""),
    ("src.models.hybrid.sota_tv.models.series_lifecycle", "SeriesLifecycleModel", ""),
    ("src.models.hybrid.sota_tv.models.temporal_attention", "TemporalAttentionTVModel", ""),
    ("src.models.hybrid.sota_tv.models.watch_pattern", "WatchPatternModel", ""),
    # Training-time scaffolds
    ("src.models.unified.movie_ensemble_system", "MovieEnsembleRecommender", ""),
    # Legacy two-tower variants (superseded by src.models.two_tower.TwoTowerModel)
    ("src.models.two_tower.src.model", "CollaborativeTwoTowerModel", ""),
    ("src.models.two_tower.src.model", "EnhancedTwoTowerModel", "LegacyEnhancedTwoTowerModel"),
    ("src.models.two_tower.src.model", "MultiTaskTwoTowerModel", ""),
    ("src.models.two_tower.src.model", "UltimateTwoTowerModel", ""),
    ("src.models.advanced.enhanced_two_tower", "EnhancedTwoTowerModel", "AdvancedEnhancedTwoTowerModel"),
    # Legacy sequential variants (superseded by src.models.sequential.SequentialRecommender)
    ("src.models.sequential.src.model", "AttentionalSequentialRecommender", ""),
    ("src.models.sequential.src.model", "HierarchicalSequentialRecommender", ""),
    ("src.models.sequential.src.model", "SessionBasedRecommender", ""),
    ("src.models.sequential.src.model", "TransformerSequentialRecommender", ""),
]


EXPERIMENTAL_LOAD_ERRORS: Dict[str, str] = {}
_loaded: List[str] = []


def _load() -> None:
    for module_path, class_name, alias in _SPECS:
        exposed = alias or class_name
        try:
            mod = importlib.import_module(module_path)
            globals()[exposed] = getattr(mod, class_name)
            _loaded.append(exposed)
        except Exception as e:
            EXPERIMENTAL_LOAD_ERRORS[exposed] = f"{type(e).__name__}: {e}"


_load()

if EXPERIMENTAL_LOAD_ERRORS:
    logger.info(
        "experimental namespace: %d loaded, %d unavailable (missing optional deps?)",
        len(_loaded), len(EXPERIMENTAL_LOAD_ERRORS),
    )


__all__ = _loaded + ["EXPERIMENTAL_LOAD_ERRORS"]
