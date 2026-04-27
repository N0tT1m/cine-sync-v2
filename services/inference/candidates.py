"""Candidate-pool resolution.

Two paths, picked at request time:

  1. ANN retrieval — if `models/two_tower/item_index.faiss` + `item_ids.json`
     are present AND a user_id is supplied, encode the user via the trained
     two-tower and return the top-K nearest items by inner product. This is
     the production retrieval stage; rankers only see ~1000 items.

  2. Parquet fallback — for unauthenticated/cold callers or when no index
     has been built. Reads `data/feature_store/items.parquet` if present;
     otherwise returns an empty list (the API stays up; rankers degrade to
     a popularity baseline upstream).
"""
from __future__ import annotations

import json
import logging
from functools import lru_cache
from typing import List, Optional

import numpy as np
import pandas as pd

from .config import settings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Parquet fallback — Phase 0 behaviour, kept as the safety net.
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# FAISS-backed ANN retrieval (lazy-loaded singleton).
# ---------------------------------------------------------------------------

class _ANNRetriever:
    """Wraps a FAISS HNSW index over two-tower item embeddings.

    Disabled silently when artifacts or the faiss library are missing — we
    never want a broken ANN index to take the API down.
    """

    def __init__(self) -> None:
        self.index = None
        self.item_ids: List[str] = []
        self._user_emb_cache: dict[str, np.ndarray] = {}
        self._init()

    def _init(self) -> None:
        models_dir = settings.models_dir / "two_tower"
        index_path = models_dir / "item_index.faiss"
        ids_path = models_dir / "item_ids.json"
        if not (index_path.exists() and ids_path.exists()):
            return
        try:
            import faiss
            self.index = faiss.read_index(str(index_path))
            with ids_path.open() as f:
                ids = json.load(f)
                self.item_ids = ids if isinstance(ids, list) else list(ids)
            logger.info("ANN retriever active (%d items)", len(self.item_ids))
        except Exception as e:
            logger.warning("ANN retriever disabled: %s", e)
            self.index = None
            self.item_ids = []

    @property
    def enabled(self) -> bool:
        return self.index is not None

    def _encode_user(self, user_id: str) -> Optional[np.ndarray]:
        if not self.enabled:
            return None
        if user_id in self._user_emb_cache:
            return self._user_emb_cache[user_id]
        try:
            from .registry import registry
            scorer = registry.get("two_tower")
            model = getattr(scorer, "model", None) if scorer is not None else None
            if model is None or not hasattr(model, "encode_user"):
                return None
            import torch
            user_idx = int(user_id) if user_id.isdigit() else hash(user_id) % 50000
            with torch.no_grad():
                emb = model.encode_user(
                    user_ids=torch.tensor([user_idx], dtype=torch.long)
                )
            arr = emb.cpu().numpy().astype(np.float32)[0]
            n = float(np.linalg.norm(arr))
            if n > 0:
                arr = arr / n
            self._user_emb_cache[user_id] = arr
            return arr
        except Exception as e:
            logger.debug("user encode failed for %s: %s", user_id, e)
            return None

    def retrieve(
        self, user_id: str, history: List[str], k: int
    ) -> Optional[List[str]]:
        if not self.enabled:
            return None
        emb = self._encode_user(user_id)
        if emb is None:
            return None
        _, idx = self.index.search(emb.reshape(1, -1), k + len(history))
        seen = set(history)
        out: List[str] = []
        for i in idx[0]:
            if 0 <= i < len(self.item_ids):
                iid = self.item_ids[i]
                if iid not in seen:
                    out.append(iid)
                    if len(out) >= k:
                        break
        return out


_retriever: Optional[_ANNRetriever] = None


def _get_retriever() -> _ANNRetriever:
    global _retriever
    if _retriever is None:
        _retriever = _ANNRetriever()
    return _retriever


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def candidate_pool(
    media_type: Optional[str] = None,
    owned_only: bool = False,
    limit: int = 500,
    user_id: Optional[str] = None,
    history: Optional[List[str]] = None,
) -> List[str]:
    """Return up to `limit` candidate item_ids for ranking.

    Tries ANN retrieval first when a user context is given; falls back to the
    parquet-based pool otherwise (or when no index has been built).
    """
    if user_id is not None:
        retriever = _get_retriever()
        if retriever.enabled:
            picks = retriever.retrieve(user_id, history or [], limit)
            if picks:
                return picks

    pool = list(_load_items(media_type, owned_only))
    return pool[:limit]


def invalidate() -> None:
    _load_items.cache_clear()
    global _retriever
    _retriever = None
