"""
CineSync v2 - TV-Specific Recommendation Models

This module contains 14 specialized recommendation models designed for TV-specific
features like episode sequences, binge patterns, series lifecycles, etc.

Existing Models (6):
- TemporalAttentionTVModel: Temporal patterns in TV viewing
- TVGraphNeuralNetwork: Graph-based TV recommendations
- ContrastiveTVLearning: Self-supervised TV learning
- MetaLearningTVModel: Few-shot adaptation for TV
- TVEnsembleSystem: Ensemble of TV models
- MultimodalTVTransformer: Multimodal TV features

New Models (8):
- EpisodeSequenceModel: Episode-level sequence modeling
- BingePredictionModel: Binge-watching prediction
- SeriesCompletionModel: Series completion prediction
- SeasonQualityModel: Season quality variance
- PlatformAvailabilityModel: Streaming platform awareness
- WatchPatternModel: Viewing pattern modeling
- SeriesLifecycleModel: Series lifecycle stages
- CastMigrationModel: Cast changes across seasons
"""

# Existing models
from .temporal_attention import TemporalAttentionTVModel
from .graph_neural_network import TVGraphNeuralNetwork
from .contrastive_learning import ContrastiveTVLearning
from .meta_learning import MetaLearningTVModel
from .ensemble_system import TVEnsembleSystem
from .multimodal_transformer import MultimodalTVTransformer

# New TV-specific models
from .episode_sequence import EpisodeSequenceModel, EpisodeSequenceTrainer, EpisodeSequenceConfig
from .binge_prediction import BingePredictionModel, BingePredictionTrainer, BingePredictionConfig
from .series_completion import SeriesCompletionModel, SeriesCompletionTrainer, SeriesCompletionConfig
from .season_quality import SeasonQualityModel, SeasonQualityTrainer, SeasonQualityConfig
from .platform_availability import PlatformAvailabilityModel, PlatformAvailabilityTrainer, PlatformConfig
from .watch_pattern import WatchPatternModel, WatchPatternTrainer, WatchPatternConfig
from .series_lifecycle import SeriesLifecycleModel, SeriesLifecycleTrainer, LifecycleConfig
from .cast_migration import CastMigrationModel, CastMigrationTrainer, CastMigrationConfig

__all__ = [
    # Existing models
    'TemporalAttentionTVModel',
    'TVGraphNeuralNetwork',
    'ContrastiveTVLearning',
    'MetaLearningTVModel',
    'TVEnsembleSystem',
    'MultimodalTVTransformer',
    # New models
    'EpisodeSequenceModel',
    'EpisodeSequenceTrainer',
    'EpisodeSequenceConfig',
    'BingePredictionModel',
    'BingePredictionTrainer',
    'BingePredictionConfig',
    'SeriesCompletionModel',
    'SeriesCompletionTrainer',
    'SeriesCompletionConfig',
    'SeasonQualityModel',
    'SeasonQualityTrainer',
    'SeasonQualityConfig',
    'PlatformAvailabilityModel',
    'PlatformAvailabilityTrainer',
    'PlatformConfig',
    'WatchPatternModel',
    'WatchPatternTrainer',
    'WatchPatternConfig',
    'SeriesLifecycleModel',
    'SeriesLifecycleTrainer',
    'LifecycleConfig',
    'CastMigrationModel',
    'CastMigrationTrainer',
    'CastMigrationConfig',
]

__version__ = '2.0.0'
