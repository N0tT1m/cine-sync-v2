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

    # Consolidated to three complementary models, one per recommendation stage:
    #   two_tower       -> retrieval / collaborative candidate generation
    #   bert4rec        -> sequential re-ranking
    #   sbert_two_tower -> semantic content / cold-start
    enabled_models: List[str] = Field(
        default_factory=lambda: [
            "two_tower",
            "bert4rec",
            "sbert_two_tower",
        ]
    )

    # Weights reflect measured contribution, not model ambition. Leave-one-out
    # NDCG@10 on 1500 held-out users (popularity-matched negatives):
    #
    #   two_tower alone            0.288      popularity baseline  0.200
    #   sbert_two_tower alone      0.085      random baseline      0.052
    #   + sbert at weight 1.3      0.150      <- as originally configured
    #   + sbert at weight 0.1      0.290      <- neutral-to-positive
    #
    # sbert ranks movies barely above random because MovieLens ships no
    # overviews, so its movie text is title+genres only — it is a *content*
    # model with almost no content. Weighted 1.3 (above two_tower!) it dominated
    # the blend and halved ranking quality. It stays enabled at a low weight
    # because it is the only model that indexes TV at all: no per-user TV
    # interaction data exists, so collaborative filtering cannot cover TV, and
    # on a TV request two_tower contributes only low-confidence fallbacks that
    # WeightedMean discounts. Raise sbert's weight if movie overviews are ever
    # added to the feature store — re-measure with services.inference.evaluate
    # first.
    ensemble_weights: dict = Field(
        default_factory=lambda: {
            "two_tower": 1.2,
            "bert4rec": 1.2,
            "sbert_two_tower": 0.1,
        }
    )

    # blend strategy name; see services/inference/ensemble/stacker.py:STRATEGIES
    ensemble_strategy: str = "weighted_mean"

    # MMR diversity reranking (no-op until two_tower item_emb.npy exists).
    # This deliberately trades relevance for variety: measured at ~9% NDCG@10
    # (0.290 -> 0.257 on the 1500-user holdout). That cost buys a feed that
    # isn't five entries from one franchise, and NDCG cannot see that value —
    # so judge alpha by looking at real rails, not by the metric alone. Raise
    # alpha toward 1.0 to weight relevance more.
    mmr_alpha: float = 0.7
    mmr_top_k: int = 50

    redis_url: str = "redis://localhost:6379/3"
    cache_ttl_seconds: int = 900

    device: str = "auto"
    max_candidates: int = 500
    default_limit: int = 20

    log_level: str = "INFO"


settings = Settings()
