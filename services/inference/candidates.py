"""Candidate-pool resolution.

Phase 0: reads items.parquet from the feature store if it exists, otherwise
returns an empty list. Phase 2+ adds ANN retrieval (e.g. FAISS over SBERT
embeddings) before ranking.
"""
from __future__ import annotations

import logging
from functools import lru_cache
from typing import List, Optional

import pandas as pd

from .config import settings

logger = logging.getLogger(__name__)


@lru_cache(maxsize=8)
def _load_items(media_type: Optional[str], owned_only: bool) -> tuple[str, ...]:
    items_path = settings.feature_store_dir / "items.parquet"
    if not items_path.exists():
        logger.info("items.parquet not present; candidate pool is empty")
        return tuple()
    df = pd.read_parquet(items_path)
    if media_type and "media_type" in df.columns:
        df = df[df["media_type"] == media_type]
    if owned_only and "owned" in df.columns:
        df = df[df["owned"] == True]  # noqa: E712 — parquet bool col
    if "item_id" not in df.columns:
        logger.warning("items.parquet missing item_id column")
        return tuple()
    return tuple(df["item_id"].astype(str).tolist())


def candidate_pool(
    media_type: Optional[str] = None,
    owned_only: bool = False,
    limit: int = 500,
) -> List[str]:
    pool = list(_load_items(media_type, owned_only))
    return pool[:limit]


def invalidate() -> None:
    _load_items.cache_clear()
