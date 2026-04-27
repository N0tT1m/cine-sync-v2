"""Ensemble blending over registry-scored model outputs.

The `Ensemble` class drives the pipeline (fan out to models, collect, blend,
rank); the actual blend is delegated to a `BlendStrategy` so we can swap
implementations without changing the API:

    WeightedMean      default; per-model weights from settings.ensemble_weights.
    LightGBMStacker   Phase 4 slot — raises until a trained ranker is wired in.

Add a new strategy: implement `BlendStrategy.blend(...)` and register it in
`STRATEGIES`. Configure via `settings.ensemble_strategy` (default: weighted_mean).
"""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from typing import Dict, Iterable, List, Optional, Protocol

from ..config import settings
from ..registry import registry
from ..schemas import EnsembleResponse, ScoredItem


class BlendStrategy(Protocol):
    """Contract for plugging a new ensemble strategy.

    Given per-model scored lists for the same candidate set, return one blended
    ranked list. All strategies receive the full weights dict — stateless ones
    (WeightedMean) use it directly; learned ones (LightGBMStacker) may ignore.
    """

    name: str

    def blend(
        self,
        per_model: Dict[str, List[ScoredItem]],
        weights: Dict[str, float],
    ) -> List[ScoredItem]: ...


class WeightedMean:
    """Score each item as the weight-normalized mean of per-model scores.

    Items missing from a given model simply don't contribute to the sum for
    that model — no zero-fill — which keeps partial-coverage models from
    dragging scores down against models that scored every candidate.
    """

    name = "weighted_mean"

    def blend(
        self,
        per_model: Dict[str, List[ScoredItem]],
        weights: Dict[str, float],
    ) -> List[ScoredItem]:
        acc: Dict[str, float] = defaultdict(float)
        weight_total: Dict[str, float] = defaultdict(float)
        features: Dict[str, Dict[str, float]] = defaultdict(dict)

        for model_name, scored in per_model.items():
            w = weights.get(model_name, 1.0)
            for s in scored:
                acc[s.item_id] += w * s.score
                weight_total[s.item_id] += w
                features[s.item_id][f"score_{model_name}"] = s.score

        total_weight = max(sum(weights.values()), 1e-6)

        out: List[ScoredItem] = []
        for item_id, total in acc.items():
            denom = weight_total[item_id] or 1.0
            out.append(
                ScoredItem(
                    item_id=item_id,
                    score=total / denom,
                    model="ensemble",
                    # confidence = fraction of ensemble weight that actually scored this item
                    confidence=min(1.0, denom / total_weight),
                    features=features[item_id],
                )
            )
        out.sort(key=lambda x: x.score, reverse=True)
        return out


class LightGBMStacker:
    """Phase 4 slot — a LightGBM ranker over [score_*, cold_start_flag, ...] features.

    Not implemented. Raises on use so misconfiguration is loud; falls back at
    the Ensemble level to WeightedMean if the stacker model isn't loaded.
    """

    name = "lightgbm_stacker"

    def __init__(self) -> None:
        self.model = None  # set by a future training+deploy step

    def blend(
        self,
        per_model: Dict[str, List[ScoredItem]],
        weights: Dict[str, float],
    ) -> List[ScoredItem]:
        if self.model is None:
            raise NotImplementedError("lightgbm stacker not trained yet")
        raise NotImplementedError("stacker inference not wired yet")


STRATEGIES: Dict[str, BlendStrategy] = {
    WeightedMean.name: WeightedMean(),
    LightGBMStacker.name: LightGBMStacker(),
}


def _resolve_strategy(name: Optional[str]) -> BlendStrategy:
    key = (name or WeightedMean.name).lower()
    strategy = STRATEGIES.get(key)
    if strategy is None:
        raise ValueError(
            f"unknown ensemble strategy {name!r}; known: {sorted(STRATEGIES)}"
        )
    return strategy


class Ensemble:
    """Fan out to registered models, blend with the configured strategy."""

    def __init__(self, strategy: Optional[BlendStrategy] = None) -> None:
        self.weights: Dict[str, float] = dict(settings.ensemble_weights)
        self.strategy: BlendStrategy = strategy or _resolve_strategy(
            getattr(settings, "ensemble_strategy", None)
        )

    def score(
        self,
        item_ids: Iterable[str],
        user_id: Optional[str] = None,
        models: Optional[List[str]] = None,
        **kwargs: object,
    ) -> EnsembleResponse:
        items = list(item_ids)
        model_names = models or settings.enabled_models
        per_model: Dict[str, List[ScoredItem]] = {
            name: registry.score(name, items, user_id=user_id, **kwargs)
            for name in model_names
        }

        try:
            blended = self.strategy.blend(per_model, self.weights)
        except NotImplementedError:
            # misconfigured/untrained stacker — fall back so the API stays up
            blended = WeightedMean().blend(per_model, self.weights)

        return EnsembleResponse(
            items=blended,
            models_used=list(per_model.keys()),
            weights={k: self.weights.get(k, 1.0) for k in per_model.keys()},
            generated_at=datetime.utcnow(),
            user_id=user_id,
        )


ensemble = Ensemble()
