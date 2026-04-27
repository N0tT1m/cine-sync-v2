"""Deterministic stub scorer used when a model has no trained artifacts yet.

Produces stable pseudo-scores from (model_name, user_id, item_id) so the API is
usable end-to-end during Phase 0. Every stub scorer is replaced by a real model
in Phase 2 without the serving contract changing.
"""
from __future__ import annotations

import hashlib
from typing import Iterable, List, Optional

from ..schemas import ScoredItem


class StubScorer:
    kind = "stub"

    def __init__(self, name: str, weight_bias: float = 0.0) -> None:
        self.name = name
        self.weight_bias = weight_bias

    def score(
        self,
        item_ids: Iterable[str],
        user_id: Optional[str] = None,
        **_: object,
    ) -> List[ScoredItem]:
        user_key = user_id or "anon"
        out: List[ScoredItem] = []
        for item_id in item_ids:
            digest = hashlib.sha256(
                f"{self.name}|{user_key}|{item_id}".encode()
            ).digest()
            # map to [0, 1] and shift by a small model-specific bias so the
            # ensemble has something to blend instead of identical scores
            base = int.from_bytes(digest[:8], "big") / 2**64
            score = min(max(base + self.weight_bias, 0.0), 1.0)
            out.append(
                ScoredItem(
                    item_id=item_id,
                    score=score,
                    model=self.name,
                    confidence=0.1,
                    features={"source": "stub"},
                )
            )
        out.sort(key=lambda x: x.score, reverse=True)
        return out
