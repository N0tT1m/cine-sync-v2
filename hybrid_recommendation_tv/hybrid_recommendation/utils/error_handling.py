import logging
import functools
import traceback
from typing import Any, Callable, Optional, Type, Union
import torch
import psycopg2

logger = logging.getLogger(__name__)


def handle_exceptions(
    default_return: Any = None,
    log_error: bool = True,
    reraise: bool = False,
    exception_types: tuple = (Exception,)
):
    """
    Decorator for handling exceptions in functions
    
    Args:
        default_return: Value to return if exception occurs
        log_error: Whether to log the error
        reraise: Whether to reraise the exception after handling
        exception_types: Tuple of exception types to catch
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except exception_types as e:
                if log_error:
                    logger.error(f"Error in {func.__name__}: {str(e)}")
                    logger.debug(f"Traceback: {traceback.format_exc()}")
                
                if reraise:
                    raise
                
                return default_return
        return wrapper
    return decorator


def handle_database_errors(default_return: Any = None):
    """Decorator specifically for database operation error handling"""
    return handle_exceptions(
        default_return=default_return,
        exception_types=(psycopg2.Error, psycopg2.DatabaseError, psycopg2.OperationalError)
    )


def handle_model_errors(default_return: Any = None):
    """Decorator specifically for model operation error handling"""
    return handle_exceptions(
        default_return=default_return,
        exception_types=(torch.cuda.OutOfMemoryError, RuntimeError, ValueError)
    )


class ModelValidationError(Exception):
    """Custom exception for model validation errors"""
    pass


class DatabaseConnectionError(Exception):
    """Custom exception for database connection errors"""
    pass


class ConfigurationError(Exception):
    """Custom exception for configuration errors"""
    pass


def validate_model_inputs(user_ids: torch.Tensor, movie_ids: torch.Tensor, 
                         num_users: int, num_movies: int) -> None:
    """
    Validate model inputs to prevent out-of-bounds errors
    
    Args:
        user_ids: User ID tensor
        movie_ids: Movie ID tensor
        num_users: Maximum number of users
        num_movies: Maximum number of movies
        
    Raises:
        ModelValidationError: If inputs are invalid
    """
    if torch.any(user_ids >= num_users) or torch.any(user_ids < 0):
        raise ModelValidationError(
            f"User IDs out of range! Max: {user_ids.max()}, Expected < {num_users}"
        )
    
    if torch.any(movie_ids >= num_movies) or torch.any(movie_ids < 0):
        raise ModelValidationError(
            f"Movie IDs out of range! Max: {movie_ids.max()}, Expected < {num_movies}"
        )


def safe_tensor_operation(operation: Callable, *args, **kwargs) -> Optional[torch.Tensor]:
    """
    Safely execute tensor operations with memory management
    
    Args:
        operation: The tensor operation to execute
        *args: Arguments for the operation
        **kwargs: Keyword arguments for the operation
        
    Returns:
        Result tensor or None if operation failed
    """
    try:
        with torch.no_grad():
            result = operation(*args, **kwargs)
            return result
    except torch.cuda.OutOfMemoryError:
        logger.error("CUDA out of memory. Clearing cache and retrying...")
        torch.cuda.empty_cache()
        try:
            with torch.no_grad():
                result = operation(*args, **kwargs)
                return result
        except torch.cuda.OutOfMemoryError:
            logger.error("CUDA out of memory even after clearing cache")
            return None
    except Exception as e:
        logger.error(f"Error in tensor operation: {e}")
        return None


def retry_operation(max_retries: int = 3, delay: float = 1.0):
    """
    Decorator for retrying operations with exponential backoff
    
    Args:
        max_retries: Maximum number of retry attempts
        delay: Initial delay between retries (seconds)
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            import time
            
            last_exception = None
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_retries:
                        wait_time = delay * (2 ** attempt)  # Exponential backoff
                        logger.warning(f"Attempt {attempt + 1} failed for {func.__name__}: {e}. Retrying in {wait_time}s...")
                        time.sleep(wait_time)
                    else:
                        logger.error(f"All {max_retries + 1} attempts failed for {func.__name__}")
            
            raise last_exception
        return wrapper
    return decorator


def validate_config(config) -> None:
    """
    Validate configuration object
    
    Args:
        config: Configuration object to validate
        
    Raises:
        ConfigurationError: If configuration is invalid
    """
    required_attrs = ['database', 'model', 'discord', 'server']
    
    for attr in required_attrs:
        if not hasattr(config, attr):
            raise ConfigurationError(f"Missing required configuration section: {attr}")
    
    # Validate database config
    db_config = config.database
    required_db_attrs = ['host', 'database', 'user', 'password', 'port']
    for attr in required_db_attrs:
        if not hasattr(db_config, attr) or getattr(db_config, attr) is None:
            raise ConfigurationError(f"Missing required database configuration: {attr}")
    
    # Validate model config
    model_config = config.model
    required_model_attrs = ['embedding_dim', 'hidden_dim', 'models_dir']
    for attr in required_model_attrs:
        if not hasattr(model_config, attr):
            raise ConfigurationError(f"Missing required model configuration: {attr}")
    
    # Validate Discord config
    if not config.discord.token:
        raise ConfigurationError("Discord token is required")


class ErrorHandler:
    """Centralized error handling utility"""
    
    def __init__(self, logger_name: str = None):
        self.logger = logging.getLogger(logger_name or __name__)
    
    def log_and_reraise(self, error: Exception, context: str = "") -> None:
        """Log an error with context and reraise it"""
        error_msg = f"{context}: {str(error)}" if context else str(error)
        self.logger.error(error_msg)
        self.logger.debug(f"Traceback: {traceback.format_exc()}")
        raise error
    
    def log_and_return_default(self, error: Exception, default: Any = None, context: str = "") -> Any:
        """Log an error with context and return a default value"""
        error_msg = f"{context}: {str(error)}" if context else str(error)
        self.logger.error(error_msg)
        self.logger.debug(f"Traceback: {traceback.format_exc()}")
        return default
    
    def handle_critical_error(self, error: Exception, context: str = "") -> None:
        """Handle critical errors that should terminate the application"""
        error_msg = f"CRITICAL ERROR - {context}: {str(error)}" if context else f"CRITICAL ERROR: {str(error)}"
        self.logger.critical(error_msg)
        self.logger.debug(f"Traceback: {traceback.format_exc()}")
        raise SystemExit(1)