"""Serve a TwoTowerModel through the inference registry.

Protocol target: `score(item_ids, user_id=None, **ctx) -> List[ScoredItem]`.

Expected artifacts (all optional; the adapter degrades to uniform scores if missing):

    item_embeddings_path:  .npy or .pt tensor of shape (N_items, embedding_dim)
    item_id_map_path:      .json   {"item_id": row_index, ...}
    user_embeddings_path:  .npy or .pt tensor of shape (N_users, embedding_dim)
    user_id_map_path:      .json   {"user_id": row_index, ...}

Paths may be absolute or relative to the manifest directory — the loader
prepends `manifest_dir` before constructing the adapter.

Two-tower scoring strategy:
  1. Resolve a user embedding: lookup by user_id if provided and mapped.
  2. For each requested item_id, look up item embedding; unseen items fall
     back to a deterministic pseudo-score so the response is still usable
     (same contract as StubScorer).
  3. Score = dot(user_emb, item_emb). Embeddings are assumed L2-normalized
     (what `TwoTowerModel` produces by default).
"""
from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np

try:
    from services.inference.schemas import ScoredItem
except ImportError:
    from services.inference.schemas import ScoredItem  # type: ignore

logger = logging.getLogger(__name__)


def _load_tensor(path: Path) -> np.ndarray:
    if path.suffix in (".npy",):
        return np.load(path)
    if path.suffix in (".pt", ".pth", ".bin"):
        import torch
        t = torch.load(path, map_location="cpu", weights_only=False)
        if hasattr(t, "detach"):
            t = t.detach().cpu().numpy()
        return np.asarray(t)
    raise ValueError(f"unsupported embedding file: {path}")


def _load_id_map(path: Path) -> Dict[str, int]:
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    return {str(k): int(v) for k, v in raw.items()}


class TwoTowerScorer:
    """Dot-product scorer over cached two-tower embeddings."""

    def __init__(
        self,
        model,
        *,
        name: str,
        kind: str = "two_tower",
        item_embeddings_path: Optional[str] = None,
        item_id_map_path: Optional[str] = None,
        user_embeddings_path: Optional[str] = None,
        user_id_map_path: Optional[str] = None,
        manifest_dir: Optional[Path] = None,
    ) -> None:
        self.model = model
        self.name = name
        self.kind = kind
        base = Path(manifest_dir) if manifest_dir else Path.cwd()

        self._item_emb: Optional[np.ndarray] = None
        self._item_idx: Dict[str, int] = {}
        if item_embeddings_path and item_id_map_path:
            self._item_emb = _load_tensor(base / item_embeddings_path)
            self._item_idx = _load_id_map(base / item_id_map_path)
            logger.info(
                "%s: item index loaded (%d items, dim=%d)",
                name, len(self._item_idx), self._item_emb.shape[1],
            )

        self._user_emb: Optional[np.ndarray] = None
        self._user_idx: Dict[str, int] = {}
        if user_embeddings_path and user_id_map_path:
            self._user_emb = _load_tensor(base / user_embeddings_path)
            self._user_idx = _load_id_map(base / user_id_map_path)
            logger.info(
                "%s: user index loaded (%d users, dim=%d)",
                name, len(self._user_idx), self._user_emb.shape[1],
            )

    # ---- Scorer protocol ------------------------------------------

    def score(
        self,
        item_ids: Iterable[str],
        user_id: Optional[str] = None,
        **_ctx: object,
    ) -> List[ScoredItem]:
        items = list(item_ids)
        user_vec = self._resolve_user(user_id)

        out: List[ScoredItem] = []
        for item_id in items:
            row = self._item_idx.get(str(item_id))
            if user_vec is not None and row is not None and self._item_emb is not None:
                score = float(np.dot(user_vec, self._item_emb[row]))
                source = "embedding"
            else:
                score = _pseudo_score(self.name, user_id, item_id)
                source = "fallback"
            out.append(
                ScoredItem(
                    item_id=str(item_id),
                    score=_clip01(score),
                    model=self.name,
                    confidence=0.9 if source == "embedding" else 0.1,
                    features={"source": source},
                )
            )
        out.sort(key=lambda x: x.score, reverse=True)
        return out

    def _resolve_user(self, user_id: Optional[str]) -> Optional[np.ndarray]:
        if user_id is None or self._user_emb is None:
            return None
        row = self._user_idx.get(str(user_id))
        if row is None:
            return None
        return self._user_emb[row]


def _pseudo_score(name: str, user_id: Optional[str], item_id: str) -> float:
    digest = hashlib.sha256(f"{name}|{user_id or 'anon'}|{item_id}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def _clip01(x: float) -> float:
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x
