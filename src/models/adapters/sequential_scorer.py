"""Serve a SequentialRecommender through the inference registry.

Protocol target: `score(item_ids, user_id=None, watch_history=..., **ctx) -> List[ScoredItem]`.

Expected artifacts:

    item_id_map_path:  .json   {"item_id": vocab_index, ...}   (index 0 reserved for padding)

Scoring strategy:
  1. Tokenize `watch_history` (list of item_id strings) → last-N vocab indices.
  2. Run `model.encode` to get hidden states; take the last position.
  3. Project onto item embeddings to obtain a logit per item_id.
  4. For each requested item_id, return its logit (softmax-normalized to [0, 1]);
     unseen item_ids fall back to a deterministic pseudo-score.

If the model or vocab is missing, every request falls back gracefully so the
service stays up.
"""
from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Dict, Iterable, List, Optional

try:
    from services.inference.schemas import ScoredItem
except ImportError:
    from services.inference.schemas import ScoredItem  # type: ignore

logger = logging.getLogger(__name__)


class SequentialScorer:
    """Next-item scorer over a trained SequentialRecommender."""

    def __init__(
        self,
        model,
        *,
        name: str,
        kind: str = "sasrec",
        item_id_map_path: Optional[str] = None,
        max_history: int = 200,
        manifest_dir: Optional[Path] = None,
    ) -> None:
        self.model = model
        self.name = name
        self.kind = kind
        self.max_history = max_history
        base = Path(manifest_dir) if manifest_dir else Path.cwd()

        self._item_idx: Dict[str, int] = {}
        if item_id_map_path:
            with (base / item_id_map_path).open("r", encoding="utf-8") as f:
                raw = json.load(f)
            self._item_idx = {str(k): int(v) for k, v in raw.items()}
            logger.info("%s: vocab loaded (%d items)", name, len(self._item_idx))

        if hasattr(model, "eval"):
            model.eval()

    def score(
        self,
        item_ids: Iterable[str],
        user_id: Optional[str] = None,
        watch_history: Optional[List[str]] = None,
        **_ctx: object,
    ) -> List[ScoredItem]:
        items = list(item_ids)
        logits = self._run(watch_history)

        out: List[ScoredItem] = []
        for item_id in items:
            vocab_row = self._item_idx.get(str(item_id))
            if logits is not None and vocab_row is not None and vocab_row < logits.shape[0]:
                score = float(_sigmoid(logits[vocab_row].item()))
                source = "model"
            else:
                score = _pseudo_score(self.name, user_id, item_id)
                source = "fallback"
            out.append(
                ScoredItem(
                    item_id=str(item_id),
                    score=score,
                    model=self.name,
                    confidence=0.9 if source == "model" else 0.1,
                    features={"source": source, "history_len": len(watch_history or [])},
                )
            )
        out.sort(key=lambda x: x.score, reverse=True)
        return out

    def _run(self, history: Optional[List[str]]):
        if not history or self.model is None or not self._item_idx:
            return None
        try:
            import torch
        except ImportError:
            return None

        # Map history to vocab indices; drop unknown tokens.
        tokens = [self._item_idx[h] for h in history if h in self._item_idx]
        if not tokens:
            return None
        tokens = tokens[-self.max_history :]
        seq = torch.tensor(tokens, dtype=torch.long).unsqueeze(0)

        with torch.no_grad():
            hidden = self.model.encode(seq)
            last = hidden[:, -1, :]
            # Reuse _project so tied-weights models just work.
            logits = self.model._project(last.unsqueeze(1)).squeeze(0).squeeze(0)
        return logits


def _pseudo_score(name: str, user_id: Optional[str], item_id: str) -> float:
    digest = hashlib.sha256(f"{name}|{user_id or 'anon'}|{item_id}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def _sigmoid(x: float) -> float:
    import math
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)
