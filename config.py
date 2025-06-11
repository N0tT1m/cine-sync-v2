import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class DatabaseConfig:
    host: str
    database: str
    user: str
    password: str
    port: int


@dataclass
class ModelConfig:
    embedding_dim: int
    hidden_dim: int
    num_epochs: int
    batch_size: int
    learning_rate: float
    models_dir: str


@dataclass
class DiscordConfig:
    token: str
    command_prefix: str


@dataclass
class ServerConfig:
    host: str
    port: int
    api_base_url: str


@dataclass
class AppConfig:
    database: DatabaseConfig
    model: ModelConfig
    discord: DiscordConfig
    server: ServerConfig
    debug: bool


def load_config() -> AppConfig:
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