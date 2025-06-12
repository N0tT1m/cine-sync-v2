"""
Simplified configuration system
"""
import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class DatabaseConfig:
    host: str = "localhost"
    port: int = 5432
    database: str = "cinesync"
    user: str = "postgres"
    password: str = "postgres"
    
    @classmethod
    def from_env(cls):
        return cls(
            host=os.getenv("DB_HOST", "localhost"),
            port=int(os.getenv("DB_PORT", "5432")),
            database=os.getenv("DB_NAME", "cinesync"),
            user=os.getenv("DB_USER", "postgres"),
            password=os.getenv("DB_PASSWORD", "postgres")
        )


@dataclass
class ModelConfig:
    models_dir: str = "models"
    embedding_dim: int = 64
    batch_size: int = 128
    num_epochs: int = 20
    learning_rate: float = 0.001


@dataclass
class SimpleConfig:
    database: DatabaseConfig
    model: ModelConfig
    debug: bool = False
    
    @classmethod
    def load(cls):
        """Load configuration from environment"""
        return cls(
            database=DatabaseConfig.from_env(),
            model=ModelConfig(),
            debug=os.getenv("DEBUG", "false").lower() == "true"
        )


def load_simple_config():
    """Simple config loader function"""
    return SimpleConfig.load()