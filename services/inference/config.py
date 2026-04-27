from pathlib import Path
from typing import List

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


REPO_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="CINEREC_",
        env_file=".env",
        extra="ignore",
    )

    host: str = "0.0.0.0"
    port: int = 8900

    models_dir: Path = REPO_ROOT / "models"
    feature_store_dir: Path = REPO_ROOT / "data" / "feature_store"

    enabled_models: List[str] = Field(
        default_factory=lambda: [
            "hybrid",
            "ncf",
            "sequential",
            "two_tower",
            "sbert_two_tower",
            "graphsage",
            "bert4rec",
            "contrastive",
            "multimodal",
            "vae",
        ]
    )

    ensemble_weights: dict = Field(
        default_factory=lambda: {
            "hybrid": 1.0,
            "ncf": 1.0,
            "sequential": 1.2,
            "two_tower": 1.2,
            "sbert_two_tower": 1.3,
            "graphsage": 1.1,
            "bert4rec": 1.2,
            "contrastive": 0.8,
            "multimodal": 1.1,
            "vae": 0.6,
        }
    )

    # blend strategy name; see services/inference/ensemble/stacker.py:STRATEGIES
    ensemble_strategy: str = "weighted_mean"

    # MMR diversity reranking (no-op until two_tower item_emb.npy exists)
    mmr_alpha: float = 0.7
    mmr_top_k: int = 50

    redis_url: str = "redis://localhost:6379/3"
    cache_ttl_seconds: int = 900

    device: str = "auto"
    max_candidates: int = 500
    default_limit: int = 20

    log_level: str = "INFO"


settings = Settings()
