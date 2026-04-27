"""Serve a BERT4Rec model through the inference registry.

Protocol target: `score(item_ids, user_id=None, **ctx) -> List[ScoredItem]`.

BERT4Rec scores by masked-LM completion: append a [MASK] token to the user's
watch history, run forward(apply_masking=False), then read the logit at the
mask position for each candidate. Mirrors how the model was trained.

Expected ctx kwarg: `watch_history` — list of item_id strings, most recent
last. Without it (cold caller) the adapter falls back to a uniform-ish
deterministic pseudo-score so the API stays up.

Expected artifacts (optional):
    item_id_map_path:  .json   {"item_id": row_index, ...}
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


class BERT4RecScorer:
    """Masked-LM scorer over a user's watch history."""

    def __init__(
        self,
        model,
        *,
        name: str,
        kind: str = "bert4rec",
        item_id_map_path: Optional[str] = None,
        manifest_dir: Optional[Path] = None,
    ) -> None:
        self.model = model
        self.name = name
        self.kind = kind
        base = Path(manifest_dir) if manifest_dir else Path.cwd()

        self._item_idx: Dict[str, int] = {}
        if item_id_map_path:
            self._item_idx = _load_id_map(base / item_id_map_path)

        # Pull tokens off the model — pad=0, mask=num_items+1 by default.
        self.pad_token = int(getattr(model, "pad_token", 0))
        self.mask_token = int(getattr(model, "mask_token", model.num_items + 1))
        self.num_items = int(model.num_items)
        self.max_seq_len = int(getattr(model, "max_seq_len", 200))

    # ---- Scorer protocol ---------------------------------------------

    @torch.inference_mode()
    def score(
        self,
        item_ids: Iterable[str],
        user_id: Optional[str] = None,
        **ctx: object,
    ) -> List[ScoredItem]:
        items = list(item_ids)
        if not items:
            return []

        history_ids = ctx.get("watch_history") or []
        if not history_ids:
            return self._fallback(items, user_id)

        device = next(self.model.parameters()).device

        # Build masked input: [pad_pad_pad_..._h1_h2_..._h_n_MASK]
        history_tokens = [self._resolve_item(h) for h in history_ids]
        # Reserve one slot for the mask; cap to max_seq_len.
        keep = self.max_seq_len - 1
        history_tokens = history_tokens[-keep:]
        seq = history_tokens + [self.mask_token]
        # Left-pad to max_seq_len so positional encoding aligns.
        pad_len = self.max_seq_len - len(seq)
        padded = [self.pad_token] * pad_len + seq
        mask_position = self.max_seq_len - 1

        seq_t = torch.tensor([padded], dtype=torch.long, device=device)

        try:
            out = self.model(seq_t, apply_masking=False)
        except Exception as e:
            logger.debug("BERT4RecScorer fallback for %s: %s", self.name, e)
            return self._fallback(items, user_id)

        logits = out["logits"] if isinstance(out, dict) else out
        # logits shape: (1, max_seq_len, num_items)
        mask_logits = logits[0, mask_position].detach().cpu().numpy()  # (num_items,)

        scores: List[float] = []
        for iid in items:
            idx = self._resolve_item(iid)
            if 0 <= idx < self.num_items:
                scores.append(float(mask_logits[idx]))
            else:
                scores.append(0.0)
        scores_arr = np.array(scores, dtype=np.float32)
        scores_arr = _minmax_normalize(scores_arr)

        result = [
            ScoredItem(
                item_id=str(iid),
                score=float(s),
                model=self.name,
                confidence=0.9,
                features={"source": "bert4rec"},
            )
            for iid, s in zip(items, scores_arr)
        ]
        result.sort(key=lambda x: x.score, reverse=True)
        return result

    # ---- internals ---------------------------------------------------

    def _resolve_item(self, item_id: str) -> int:
        if str(item_id) in self._item_idx:
            return self._item_idx[str(item_id)]
        return _hash_to_range(str(item_id), self.num_items)

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


__all__ = ["BERT4RecScorer"]
