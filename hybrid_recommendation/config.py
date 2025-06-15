# CineSync v2 - Configuration Management
# Centralized configuration for the hybrid recommendation system
# Uses dataclasses and environment variables for flexible deployment

import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class DatabaseConfig:
    """Database connection configuration
    
    Stores PostgreSQL connection parameters for the CineSync database
    that holds user ratings, movie metadata, and system data.
    """
    host: str        # Database server hostname/IP
    database: str    # Database name (usually 'cinesync')
    user: str        # Database username
    password: str    # Database password
    port: int        # Database port (usually 5432)


@dataclass
class ModelConfig:
    """Machine learning model configuration
    
    Defines hyperparameters and settings for training the hybrid
    recommendation model, including neural network architecture
    and training parameters.
    """
    embedding_dim: int    # Size of user/item embedding vectors
    hidden_dim: int       # Hidden layer dimensions
    num_epochs: int       # Number of training epochs
    batch_size: int       # Training batch size
    learning_rate: float  # Learning rate for optimizer
    models_dir: str       # Directory to save/load models


@dataclass
class DiscordConfig:
    """Discord bot configuration
    
    Settings for the Discord bot interface that allows users
    to get movie recommendations through Discord commands.
    """
    token: str           # Discord bot token
    command_prefix: str  # Command prefix (e.g., '!' for !recommend)


@dataclass
class ServerConfig:
    """Web server configuration
    
    Settings for the HTTP API server that provides REST endpoints
    for the recommendation system.
    """
    host: str         # Server bind address
    port: int         # Server port
    api_base_url: str # Base URL for API endpoints


@dataclass
class AppConfig:
    """Main application configuration
    
    Top-level configuration that combines all subsystem configurations
    into a single object for easy access throughout the application.
    """
    database: DatabaseConfig  # Database connection settings
    model: ModelConfig         # ML model hyperparameters
    discord: DiscordConfig     # Discord bot settings
    server: ServerConfig       # Web server settings
    debug: bool               # Enable debug mode


def load_config() -> AppConfig:
    """Load configuration from environment variables
    
    Creates application configuration by reading from environment variables
    with sensible defaults. This allows for flexible deployment across
    different environments (development, staging, production).
    
    Returns:
        AppConfig: Complete application configuration
    """
    return AppConfig(
        database=DatabaseConfig(
            host=os.getenv('DB_HOST', '192.168.1.78'),
            database=os.getenv('DB_NAME', 'cinesync'),
            user=os.getenv('DB_USER', 'postgres'),
            password=os.getenv('DB_PASSWORD', ''),
            port=int(os.getenv('DB_PORT', '5432'))
        ),
        model=ModelConfig(
            embedding_dim=int(os.getenv('MODEL_EMBEDDING_DIM', '64')),
            hidden_dim=int(os.getenv('MODEL_HIDDEN_DIM', '128')),
            num_epochs=int(os.getenv('MODEL_EPOCHS', '10')),
            batch_size=int(os.getenv('MODEL_BATCH_SIZE', '1024')),
            learning_rate=float(os.getenv('MODEL_LEARNING_RATE', '0.001')),
            models_dir=os.getenv('MODELS_DIR', 'models')
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
        debug=os.getenv('DEBUG', 'False').lower() == 'true'
    )