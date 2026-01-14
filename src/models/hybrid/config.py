"""
CineSync v2 - Unified Content Recommendation Configuration

Centralized configuration for the hybrid recommendation system
supporting both movies and TV shows through content_type parameter.
"""

import os
from dataclasses import dataclass, field
from typing import Optional, List
from enum import Enum


class ContentType(Enum):
    """Content type enumeration for the recommendation system"""
    MOVIE = "movie"
    TV = "tv"
    BOTH = "both"


@dataclass
class DatabaseConfig:
    """Database connection configuration

    Stores PostgreSQL connection parameters for the CineSync database.
    All credentials should be provided via environment variables.
    """
    host: str
    database: str
    user: str
    password: str
    port: int

    def __post_init__(self):
        """Validate database configuration"""
        if not self.host:
            raise ValueError("DB_HOST is required")
        if not self.database:
            raise ValueError("DB_NAME is required")
        if not self.user:
            raise ValueError("DB_USER is required")
        # Password can be empty for local development with trust auth


@dataclass
class ModelConfig:
    """Machine learning model configuration

    Defines hyperparameters for the unified content recommender model.
    Supports both movies and TV shows with optional content-specific features.
    """
    # Content type
    content_type: ContentType = ContentType.BOTH

    # Architecture
    embedding_dim: int = 64
    hidden_dims: List[int] = field(default_factory=lambda: [128, 64, 32])
    dropout_rate: float = 0.2
    num_genres: int = 20

    # Feature flags
    use_genre_features: bool = True
    use_tv_features: bool = True

    # Training
    num_epochs: int = 30
    batch_size: int = 1024
    learning_rate: float = 0.001
    weight_decay: float = 1e-5
    patience: int = 5
    gradient_clip_norm: float = 1.0

    # Paths
    models_dir: str = "models"

    @property
    def model_filename(self) -> str:
        """Generate model filename based on content type"""
        return f"unified_{self.content_type.value}_recommender.pt"


@dataclass
class TrainingConfig:
    """Training-specific configuration"""
    use_wandb: bool = True
    wandb_project: str = "cine-sync-v2"
    wandb_entity: Optional[str] = None

    use_mixed_precision: bool = True
    num_workers: int = 4
    pin_memory: bool = True

    # Validation
    val_split: float = 0.1
    test_split: float = 0.1

    # Checkpointing
    save_every_n_epochs: int = 5
    keep_n_checkpoints: int = 3


@dataclass
class DiscordConfig:
    """Discord bot configuration"""
    token: str
    command_prefix: str = "!"


@dataclass
class ServerConfig:
    """Web server configuration for REST API"""
    host: str = "localhost"
    port: int = 3000
    api_base_url: str = "http://localhost:3000"


@dataclass
class DataConfig:
    """Data paths configuration"""
    # Movie data paths
    movie_ratings_path: Optional[str] = None
    movie_metadata_path: Optional[str] = None

    # TV data paths
    tv_ratings_path: Optional[str] = None
    tv_metadata_path: Optional[str] = None

    # Common
    data_dir: str = "data"


@dataclass
class AppConfig:
    """Main application configuration

    Combines all subsystem configurations into a single object.
    """
    database: DatabaseConfig
    model: ModelConfig
    training: TrainingConfig
    discord: DiscordConfig
    server: ServerConfig
    data: DataConfig
    debug: bool = False


def load_config(content_type: str = "both") -> AppConfig:
    """
    Load configuration from environment variables.

    Args:
        content_type: "movie", "tv", or "both"

    Returns:
        Complete application configuration
    """
    # Parse content type
    try:
        ct = ContentType(content_type.lower())
    except ValueError:
        ct = ContentType.BOTH

    # Parse hidden dims from comma-separated string
    hidden_dims_str = os.getenv('MODEL_HIDDEN_DIMS', '128,64,32')
    hidden_dims = [int(x.strip()) for x in hidden_dims_str.split(',')]

    return AppConfig(
        database=DatabaseConfig(
            host=os.getenv('DB_HOST', 'localhost'),
            database=os.getenv('DB_NAME', 'cinesync'),
            user=os.getenv('DB_USER', 'postgres'),
            password=os.getenv('DB_PASSWORD', ''),
            port=int(os.getenv('DB_PORT', '5432'))
        ),
        model=ModelConfig(
            content_type=ct,
            embedding_dim=int(os.getenv('MODEL_EMBEDDING_DIM', '64')),
            hidden_dims=hidden_dims,
            dropout_rate=float(os.getenv('MODEL_DROPOUT_RATE', '0.2')),
            num_genres=int(os.getenv('MODEL_NUM_GENRES', '20')),
            use_genre_features=os.getenv('USE_GENRE_FEATURES', 'true').lower() == 'true',
            use_tv_features=os.getenv('USE_TV_FEATURES', 'true').lower() == 'true',
            num_epochs=int(os.getenv('MODEL_EPOCHS', '30')),
            batch_size=int(os.getenv('MODEL_BATCH_SIZE', '1024')),
            learning_rate=float(os.getenv('MODEL_LEARNING_RATE', '0.001')),
            weight_decay=float(os.getenv('MODEL_WEIGHT_DECAY', '1e-5')),
            patience=int(os.getenv('MODEL_PATIENCE', '5')),
            models_dir=os.getenv('MODELS_DIR', 'models')
        ),
        training=TrainingConfig(
            use_wandb=os.getenv('USE_WANDB', 'true').lower() == 'true',
            wandb_project=os.getenv('WANDB_PROJECT', 'cine-sync-v2'),
            wandb_entity=os.getenv('WANDB_ENTITY'),
            use_mixed_precision=os.getenv('USE_MIXED_PRECISION', 'true').lower() == 'true',
            num_workers=int(os.getenv('NUM_WORKERS', '4')),
            val_split=float(os.getenv('VAL_SPLIT', '0.1')),
            test_split=float(os.getenv('TEST_SPLIT', '0.1'))
        ),
        discord=DiscordConfig(
            token=os.getenv('DISCORD_TOKEN', ''),
            command_prefix=os.getenv('DISCORD_PREFIX', '!')
        ),
        server=ServerConfig(
            host=os.getenv('SERVER_HOST', 'localhost'),
            port=int(os.getenv('SERVER_PORT', '3000')),
            api_base_url=os.getenv('API_BASE_URL', 'http://localhost:3000')
        ),
        data=DataConfig(
            movie_ratings_path=os.getenv('MOVIE_RATINGS_PATH'),
            movie_metadata_path=os.getenv('MOVIE_METADATA_PATH'),
            tv_ratings_path=os.getenv('TV_RATINGS_PATH'),
            tv_metadata_path=os.getenv('TV_METADATA_PATH'),
            data_dir=os.getenv('DATA_DIR', 'data')
        ),
        debug=os.getenv('DEBUG', 'false').lower() == 'true'
    )


# Convenience functions for specific content types
def load_movie_config() -> AppConfig:
    """Load configuration for movie recommendations"""
    return load_config("movie")


def load_tv_config() -> AppConfig:
    """Load configuration for TV recommendations"""
    return load_config("tv")


def load_unified_config() -> AppConfig:
    """Load configuration for unified (both) recommendations"""
    return load_config("both")
