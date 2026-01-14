"""
CineSync v2 - Unified Hybrid Recommendation Module

This module provides a unified recommendation system for both movies and TV shows.
The previous separate movie/ and tv/ directories have been consolidated into
a single architecture with content_type parameterization.

Usage:
    from src.models.hybrid import (
        UnifiedContentRecommender,
        ContentDataset,
        ContentType,
        load_config,
    )

    # Create model for both content types
    model = UnifiedContentRecommender(
        num_users=10000,
        num_items=50000,
        use_genre_features=True,
        use_tv_features=True
    )

    # Load config
    config = load_config("both")  # "movie", "tv", or "both"
"""

# Core model classes
from .content_recommender import (
    UnifiedContentRecommender,
    ContentDataset,
    ContentType,
    ContentFeatures,
    UnifiedRecommendationSystem,
    # Backward compatibility aliases
    MovieHybridRecommender,
    TVShowRecommenderModel,
)

# Configuration
from .config import (
    load_config,
    load_movie_config,
    load_tv_config,
    load_unified_config,
    AppConfig,
    DatabaseConfig,
    ModelConfig,
    TrainingConfig,
    DiscordConfig,
    ServerConfig,
    DataConfig,
    ContentType as ConfigContentType,
)

# Utilities
from .utils import (
    # Database
    DatabaseManager,
    load_ratings_data,
    load_movies_data,
    load_tv_data,
    load_content_data,
    save_feedback,
    # Error handling
    RecommendationError,
    DataValidationError,
    ModelError,
    handle_exceptions,
    retry_operation,
    validate_ids,
)

__all__ = [
    # Model classes
    'UnifiedContentRecommender',
    'ContentDataset',
    'ContentType',
    'ContentFeatures',
    'UnifiedRecommendationSystem',
    # Backward compatibility
    'MovieHybridRecommender',
    'TVShowRecommenderModel',
    # Configuration
    'load_config',
    'load_movie_config',
    'load_tv_config',
    'load_unified_config',
    'AppConfig',
    'DatabaseConfig',
    'ModelConfig',
    'TrainingConfig',
    'DiscordConfig',
    'ServerConfig',
    'DataConfig',
    'ConfigContentType',
    # Database utilities
    'DatabaseManager',
    'load_ratings_data',
    'load_movies_data',
    'load_tv_data',
    'load_content_data',
    'save_feedback',
    # Error handling
    'RecommendationError',
    'DataValidationError',
    'ModelError',
    'handle_exceptions',
    'retry_operation',
    'validate_ids',
]

__version__ = '2.0.0'
