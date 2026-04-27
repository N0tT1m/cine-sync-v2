"""Serve a NeuralCollaborativeFiltering model through the inference registry.

Protocol target: `score(item_ids, user_id=None, **ctx) -> List[ScoredItem]`.

NCF takes (user_ids, item_ids) and returns a per-pair rating prediction. We
batch all candidates against a single user, then min-max-normalize to keep
scores roughly comparable to the other ensembled models.

Expected artifacts (both optional):

    user_id_map_path:  .json   {"user_id": row_index, ...}
    item_id_map_path:  .json   {"item_id": row_index, ...}

If the maps are absent the adapter hashes the supplied IDs into the
embedding range deterministically — useful for cold callers and lets the
endpoint stay up before the encoder export pipeline is finished.
"""
from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import torch

from services.inference.schemas import ScoredItem

logger = logging.getLogger(__name__)


def _load_id_map(path: Path) -> Dict[str, int]:
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    return {str(k): int(v) for k, v in raw.items()}


def _hash_to_range(value: str, modulus: int) -> int:
    if value.isdigit():
        return int(value) % modulus
    digest = hashlib.sha256(value.encode()).digest()
    return int.from_bytes(digest[:8], "big") % modulus


class NCFScorer:
    """Wraps NCF into the Scorer protocol with batched per-user scoring."""

    def __init__(
        self,
        model,
        *,
        name: str,
        kind: str = "neural_collaborative_filtering",
        user_id_map_path: Optional[str] = None,
        item_id_map_path: Optional[str] = None,
        manifest_dir: Optional[Path] = None,
    ) -> None:
        self.model = model
        self.name = name
        self.kind = kind
        base = Path(manifest_dir) if manifest_dir else Path.cwd()

        self._user_idx: Dict[str, int] = {}
        if user_id_map_path:
            self._user_idx = _load_id_map(base / user_id_map_path)

        self._item_idx: Dict[str, int] = {}
        if item_id_map_path:
            self._item_idx = _load_id_map(base / item_id_map_path)

        # Pull embedding sizes off the model so hashed fallbacks stay in range.
        user_emb = getattr(model, "user_embedding", None)
        item_emb = getattr(model, "item_embedding", None)
        self._num_users = user_emb.num_embeddings if user_emb is not None else 50000
        self._num_items = item_emb.num_embeddings if item_emb is not None else 100000

    # ---- Scorer protocol ---------------------------------------------

    @torch.inference_mode()
    def score(
        self,
        item_ids: Iterable[str],
        user_id: Optional[str] = None,
        **_ctx: object,
    ) -> List[ScoredItem]:
        items = list(item_ids)
        if not items:
            return []

        u_idx = self._resolve_user(user_id)
        item_indices = [self._resolve_item(i) for i in items]

        device = next(self.model.parameters()).device
        u_t = torch.full((len(items),), u_idx, dtype=torch.long, device=device)
        i_t = torch.tensor(item_indices, dtype=torch.long, device=device)

        try:
            outputs = self.model(u_t, i_t)
        except Exception as e:
            logger.debug("NCFScorer fallback for %s: %s", self.name, e)
            return self._fallback(items, user_id)

        if isinstance(outputs, dict):
            preds = outputs.get("rating_pred") or outputs.get("predictions") or next(iter(outputs.values()))
        else:
            preds = outputs
        scores = preds.detach().cpu().float().squeeze().numpy()
        scores = np.atleast_1d(scores)
        scores = _minmax_normalize(scores)

        out = [
            ScoredItem(
                item_id=str(iid),
                score=float(s),
                model=self.name,
                confidence=0.85,
                features={"source": "ncf"},
            )
            for iid, s in zip(items, scores)
        ]
        out.sort(key=lambda x: x.score, reverse=True)
        return out

    # ---- internals ---------------------------------------------------

    def _resolve_user(self, user_id: Optional[str]) -> int:
        if user_id is None:
            return 0
        if str(user_id) in self._user_idx:
            return self._user_idx[str(user_id)]
        return _hash_to_range(str(user_id), self._num_users)

    def _resolve_item(self, item_id: str) -> int:
        if str(item_id) in self._item_idx:
            return self._item_idx[str(item_id)]
        return _hash_to_range(str(item_id), self._num_items)

    def _fallback(self, items: List[str], user_id: Optional[str]) -> List[ScoredItem]:
        out: List[ScoredItem] = []
        for iid in items:
            digest = hashlib.sha256(f"{self.name}|{user_id or 'anon'}|{iid}".encode()).digest()
            score = int.from_bytes(digest[:8], "big") / 2**64
            out.append(
                ScoredItem(
                    item_id=str(iid),
                    score=score,
                    model=self.name,
                    confidence=0.1,
                    features={"source": "fallback"},
                )
            )
        out.sort(key=lambda x: x.score, reverse=True)
        return out


def _minmax_normalize(arr: np.ndarray) -> np.ndarray:
    lo, hi = float(arr.min()), float(arr.max())
    if hi - lo < 1e-9:
        return np.full_like(arr, 0.5, dtype=np.float32)
    return ((arr - lo) / (hi - lo)).astype(np.float32)


__all__ = ["NCFScorer"]
