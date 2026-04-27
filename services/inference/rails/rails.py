"""High-level rail builders — one function per home-page row.

Each rail picks the model that best captures that rail's signal, rather than
leaning on the blended ensemble for everything. The dispatch table is the
single source of truth for which model drives which rail.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Callable, Dict, List, Optional

from ..candidates import candidate_pool
from ..config import settings
from ..ensemble import ensemble
from ..registry import registry
from ..schemas import RailItem, RailResponse


@dataclass
class RailSpec:
    key: str
    model: str  # registry name, or "ensemble"
    reason_template: str


RAIL_REGISTRY: Dict[str, RailSpec] = {
    "recommended_for_you": RailSpec(
        key="recommended_for_you",
        model="ensemble",
        reason_template="blended picks across all models",
    ),
    "continue_watching_next": RailSpec(
        key="continue_watching_next",
        model="sequential",
        reason_template="what usually comes next",
    ),
    "because_you_watched": RailSpec(
        key="because_you_watched",
        model="contrastive",
        reason_template="similar to {seed}",
    ),
    "more_from_franchise": RailSpec(
        key="more_from_franchise",
        model="graphsage",
        reason_template="same franchise as {seed}",
    ),
    "semantic_discover": RailSpec(
        key="semantic_discover",
        model="sbert_two_tower",
        reason_template="thematically close to what you liked",
    ),
    "visual_vibe": RailSpec(
        key="visual_vibe",
        model="multimodal",
        reason_template="same visual style",
    ),
    "deep_cut": RailSpec(
        key="deep_cut",
        model="vae",
        reason_template="under-watched but might land",
    ),
}


def _reason(spec: RailSpec, seed: Optional[str]) -> str:
    if "{seed}" in spec.reason_template and seed:
        return spec.reason_template.format(seed=seed)
    if "{seed}" in spec.reason_template:
        return spec.reason_template.replace("{seed}", "this")
    return spec.reason_template


def build_rail(
    rail_key: str,
    user_id: str,
    media_type: Optional[str] = None,
    limit: int = 20,
    owned_only: bool = True,
    seed_item_id: Optional[str] = None,
    watch_history: Optional[List[str]] = None,
) -> RailResponse:
    spec = RAIL_REGISTRY.get(rail_key)
    if spec is None:
        raise ValueError(f"unknown rail: {rail_key}")

    pool = candidate_pool(
        media_type=media_type,
        owned_only=owned_only,
        limit=settings.max_candidates,
    )

    if spec.model == "ensemble":
        result = ensemble.score(
            pool,
            user_id=user_id,
            watch_history=watch_history,
        )
        scored = result.items
    else:
        scored = registry.score(
            spec.model,
            pool,
            user_id=user_id,
            watch_history=watch_history,
            seed_item_id=seed_item_id,
        )

    reason = _reason(spec, seed_item_id)
    items = [
        RailItem(item_id=s.item_id, score=s.score, reason=reason)
        for s in scored[:limit]
    ]
    return RailResponse(
        rail=rail_key,
        items=items,
        user_id=user_id,
        generated_at=datetime.utcnow(),
    )


def list_rails() -> List[str]:
    return list(RAIL_REGISTRY.keys())
