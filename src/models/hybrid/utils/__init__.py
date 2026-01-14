"""
CineSync v2 - Hybrid Model Utilities

Shared utilities for the unified content recommendation system.
"""

from .database import (
    DatabaseManager,
    load_ratings_data,
    load_movies_data,
    load_tv_data,
    load_content_data,
    save_feedback
)

from .error_handling import (
    RecommendationError,
    DataValidationError,
    ModelError,
    handle_exceptions,
    retry_operation,
    validate_ids
)

__all__ = [
    # Database
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
    'validate_ids'
]
