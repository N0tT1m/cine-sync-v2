"""
CineSync v2 - Error Handling Utilities

Custom exceptions and error handling decorators for the recommendation system.
"""

import functools
import logging
import time
from typing import Callable, Optional, Type, Tuple, Any
import traceback

logger = logging.getLogger(__name__)


# Custom Exceptions

class RecommendationError(Exception):
    """Base exception for recommendation system errors"""
    pass


class DataValidationError(RecommendationError):
    """Raised when data validation fails"""
    pass


class ModelError(RecommendationError):
    """Raised when model operations fail"""
    pass


class DatabaseConnectionError(RecommendationError):
    """Raised when database connection fails"""
    pass


class ConfigurationError(RecommendationError):
    """Raised when configuration is invalid"""
    pass


class ContentTypeError(RecommendationError):
    """Raised when content type is invalid or unsupported"""
    pass


# Decorators

def handle_exceptions(
    default_return: Any = None,
    exceptions: Tuple[Type[Exception], ...] = (Exception,),
    log_level: int = logging.ERROR,
    reraise: bool = False
) -> Callable:
    """
    Decorator for handling exceptions with logging.

    Args:
        default_return: Value to return on exception
        exceptions: Tuple of exception types to catch
        log_level: Logging level for errors
        reraise: Whether to re-raise after logging

    Returns:
        Decorated function
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except exceptions as e:
                logger.log(
                    log_level,
                    f"Error in {func.__name__}: {e}\n{traceback.format_exc()}"
                )
                if reraise:
                    raise
                return default_return
        return wrapper
    return decorator


def retry_operation(
    max_retries: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: Tuple[Type[Exception], ...] = (Exception,)
) -> Callable:
    """
    Decorator for retrying operations with exponential backoff.

    Args:
        max_retries: Maximum number of retry attempts
        delay: Initial delay between retries (seconds)
        backoff: Multiplier for delay after each retry
        exceptions: Tuple of exception types to retry on

    Returns:
        Decorated function
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            current_delay = delay
            last_exception = None

            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_retries:
                        logger.warning(
                            f"Attempt {attempt + 1}/{max_retries + 1} failed for "
                            f"{func.__name__}: {e}. Retrying in {current_delay:.1f}s..."
                        )
                        time.sleep(current_delay)
                        current_delay *= backoff
                    else:
                        logger.error(
                            f"All {max_retries + 1} attempts failed for {func.__name__}: {e}"
                        )

            raise last_exception
        return wrapper
    return decorator


def validate_input(validator: Callable[[Any], bool], error_msg: str = "Invalid input") -> Callable:
    """
    Decorator for validating function inputs.

    Args:
        validator: Function that returns True if input is valid
        error_msg: Error message if validation fails

    Returns:
        Decorated function
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if not validator(*args, **kwargs):
                raise DataValidationError(error_msg)
            return func(*args, **kwargs)
        return wrapper
    return decorator


# Validation Functions

def validate_ids(
    user_id: int,
    item_id: int,
    num_users: int,
    num_items: int
) -> None:
    """
    Validate user and item IDs are within bounds.

    Args:
        user_id: User ID to validate
        item_id: Item ID to validate
        num_users: Total number of users
        num_items: Total number of items

    Raises:
        DataValidationError: If IDs are out of bounds
    """
    if user_id < 0 or user_id >= num_users:
        raise DataValidationError(
            f"User ID {user_id} out of bounds [0, {num_users})"
        )
    if item_id < 0 or item_id >= num_items:
        raise DataValidationError(
            f"Item ID {item_id} out of bounds [0, {num_items})"
        )


def validate_rating(rating: float, min_val: float = 0.0, max_val: float = 5.0) -> None:
    """
    Validate rating is within expected range.

    Args:
        rating: Rating value to validate
        min_val: Minimum allowed rating
        max_val: Maximum allowed rating

    Raises:
        DataValidationError: If rating is out of range
    """
    if rating < min_val or rating > max_val:
        raise DataValidationError(
            f"Rating {rating} out of range [{min_val}, {max_val}]"
        )


def validate_batch_size(batch_size: int, max_size: int = 10000) -> None:
    """
    Validate batch size is reasonable.

    Args:
        batch_size: Batch size to validate
        max_size: Maximum allowed batch size

    Raises:
        DataValidationError: If batch size is invalid
    """
    if batch_size <= 0:
        raise DataValidationError(f"Batch size must be positive, got {batch_size}")
    if batch_size > max_size:
        raise DataValidationError(
            f"Batch size {batch_size} exceeds maximum {max_size}"
        )


def validate_content_type(content_type: str) -> None:
    """
    Validate content type string.

    Args:
        content_type: Content type to validate

    Raises:
        ContentTypeError: If content type is invalid
    """
    valid_types = {'movie', 'tv', 'both'}
    if content_type.lower() not in valid_types:
        raise ContentTypeError(
            f"Invalid content type '{content_type}'. Must be one of: {valid_types}"
        )


# Context Managers

class ErrorContext:
    """Context manager for error handling with cleanup."""

    def __init__(
        self,
        operation_name: str,
        cleanup_func: Optional[Callable] = None,
        reraise: bool = True
    ):
        """
        Initialize error context.

        Args:
            operation_name: Name of operation for logging
            cleanup_func: Optional cleanup function to call on error
            reraise: Whether to re-raise exceptions
        """
        self.operation_name = operation_name
        self.cleanup_func = cleanup_func
        self.reraise = reraise

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            logger.error(
                f"Error during {self.operation_name}: {exc_val}\n"
                f"{traceback.format_exc()}"
            )
            if self.cleanup_func:
                try:
                    self.cleanup_func()
                except Exception as cleanup_error:
                    logger.error(f"Cleanup failed: {cleanup_error}")

            return not self.reraise

        return False
