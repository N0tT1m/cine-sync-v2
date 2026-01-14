import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class DatabaseConfig:
    host: str = "localhost"
    port: int = 5432
    database: str = "cinesync"
    user: str = "postgres"
    password: str = ""


@dataclass
class ModelConfig:
    models_dir: str = "models"
    device: str = "auto"
    batch_size: int = 64
    learning_rate: float = 0.001
    epochs: int = 20


@dataclass
class DiscordConfig:
    token: str = ""


@dataclass
class Config:
    database: DatabaseConfig
    model: ModelConfig
    discord: DiscordConfig
    debug: bool = False
    recommendation_api_url: str = "http://192.168.1.64:5001"


def load_config() -> Config:
    """Load configuration from environment variables and defaults"""
    
    # Database config
    database = DatabaseConfig(
        host=os.getenv("DB_HOST", "localhost"),
        port=int(os.getenv("DB_PORT", "5432")),
        database=os.getenv("DB_NAME", "cinesync"),
        user=os.getenv("DB_USER", "postgres"),
        password=os.getenv("DB_PASSWORD", "")
    )
    
    # Model config  
    model = ModelConfig(
        models_dir=os.getenv("MODELS_DIR", "models"),
        device=os.getenv("DEVICE", "auto"),
        batch_size=int(os.getenv("BATCH_SIZE", "64")),
        learning_rate=float(os.getenv("LEARNING_RATE", "0.001")),
        epochs=int(os.getenv("EPOCHS", "20"))
    )
    
    # Discord config
    discord = DiscordConfig(
        token=os.getenv("DISCORD_TOKEN", "")
    )
    
    # General config
    debug = os.getenv("DEBUG", "false").lower() in ("true", "1", "yes")
    recommendation_api_url = os.getenv("RECOMMENDATION_API_URL", "http://192.168.1.64:5001")

    return Config(
        database=database,
        model=model,
        discord=discord,
        debug=debug,
        recommendation_api_url=recommendation_api_url
    )