"""MMR (Maximal Marginal Relevance) diversity reranking.

Re-orders an already-blended ranked list so each pick trades relevance against
dissimilarity to already-picked items. Prevents the "10 Marvel movies in a
row" failure mode that pure pointwise rankers always exhibit.

    score_mmr(i) = alpha * relevance(i)
                 - (1 - alpha) * max_{j in picked} cos(emb_i, emb_j)

If `models/two_tower/item_emb.npy` (and `item_ids.json`) is missing the
reranker becomes a no-op pass-through, so the API stays up.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from ..config import settings
from ..schemas import ScoredItem

logger = logging.getLogger(__name__)


def _l2_normalize(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    n[n == 0] = 1.0
    return x / n


class MMRReranker:
    def __init__(
        self,
        alpha: float = 0.7,
        top_k: int = 50,
        embeddings_path: Optional[Path] = None,
        item_ids_path: Optional[Path] = None,
    ) -> None:
        self.alpha = alpha
        self.top_k = top_k
        self._embeddings: Optional[np.ndarray] = None
        self._index: Dict[str, int] = {}

        emb_path = embeddings_path or (settings.models_dir / "two_tower" / "item_emb.npy")
        ids_path = item_ids_path or (settings.models_dir / "two_tower" / "item_ids.json")

        if emb_path.exists() and ids_path.exists():
            try:
                emb = np.load(emb_path)
                with ids_path.open() as f:
                    ids = json.load(f)
                # Accept {"ids": [...]} OR {item_id: idx} OR a bare list.
                if isinstance(ids, dict):
                    ids = ids.get("ids") or sorted(ids, key=lambda k: ids[k])
                if len(ids) != emb.shape[0]:
                    raise ValueError(
                        f"embedding/id count mismatch: {emb.shape[0]} vs {len(ids)}"
                    )
                self._embeddings = _l2_normalize(emb.astype(np.float32))
                self._index = {str(i): n for n, i in enumerate(ids)}
                logger.info("MMR reranker active (%d items)", len(self._index))
            except Exception as e:
                logger.warning("MMR reranker disabled — failed to load embeddings: %s", e)
                self._embeddings = None
                self._index = {}

    @property
    def enabled(self) -> bool:
        return self._embeddings is not None

    def _emb(self, item_id: str) -> Optional[np.ndarray]:
        if self._embeddings is None:
            return None
        idx = self._index.get(str(item_id))
        return None if idx is None else self._embeddings[idx]

    def rerank(self, items: List[ScoredItem]) -> List[ScoredItem]:
        if not self.enabled or len(items) <= 1:
            return items

        # Cap MMR's working set so we don't pay O(K*N) over a 10K-item pool.
        pool_size = max(self.top_k * 4, self.top_k + 1)
        pool = items[:pool_size]
        tail = items[pool_size:]

        picked: List[ScoredItem] = [pool.pop(0)]
        picked_embs: List[np.ndarray] = []
        first_emb = self._emb(picked[0].item_id)
        if first_emb is not None:
            picked_embs.append(first_emb)

        while pool and len(picked) < self.top_k:
            best_i = 0
            best_score = -float("inf")
            for ci, c in enumerate(pool):
                emb = self._emb(c.item_id)
                if emb is None or not picked_embs:
                    sim = 0.0
                else:
                    sim = float(max(np.dot(emb, pe) for pe in picked_embs))
                mmr = self.alpha * c.score - (1.0 - self.alpha) * sim
                if mmr > best_score:
                    best_score = mmr
                    best_i = ci
            chosen = pool.pop(best_i)
            picked.append(chosen)
            emb = self._emb(chosen.item_id)
            if emb is not None:
                picked_embs.append(emb)

        return picked + pool + tail


__all__ = ["MMRReranker"]
