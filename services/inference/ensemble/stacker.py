"""Ensemble blending over registry-scored model outputs.

The `Ensemble` class drives the pipeline (fan out to models, collect, blend,
rank, diversify); the actual blend is delegated to a `BlendStrategy` so we
can swap implementations without changing the API:

    WeightedMean      default; per-model weights from settings.ensemble_weights.
    LightGBMStacker   loads <models_dir>/_stacker/stacker.lgb when present;
                      raises NotImplementedError if no model file — Ensemble
                      catches that and falls back to WeightedMean so the API
                      stays up.

After blending, an MMRReranker re-orders the top-K to balance relevance
against diversity (no-op until item embeddings exist on disk).

Add a new strategy: implement `BlendStrategy.blend(...)` and register it in
`STRATEGIES`. Configure via `settings.ensemble_strategy` (default: weighted_mean).
"""
from __future__ import annotations

import json
import logging
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Protocol

from ..config import settings
from ..registry import registry
from ..schemas import EnsembleResponse, ScoredItem
from .diversify import MMRReranker

logger = logging.getLogger(__name__)


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
    """LightGBM ranker over per-model score features.

    Loads `<models_dir>/_stacker/stacker.lgb` and an optional `features.json`
    listing the feature names (defaults to `model.feature_name()`).

    Training the stacker is a separate offline step that consumes the eval
    harness's per-model score logs as input. If the model isn't on disk this
    raises NotImplementedError; the Ensemble catches and falls back to
    WeightedMean so the serving path stays up.
    """

    name = "lightgbm_stacker"

    def __init__(self, model_path: Optional[Path] = None) -> None:
        self.model = None
        self.feature_names: List[str] = []
        path = model_path or (settings.models_dir / "_stacker" / "stacker.lgb")
        if path.exists():
            try:
                import lightgbm as lgb
                self.model = lgb.Booster(model_file=str(path))
                feat_path = path.parent / "features.json"
                if feat_path.exists():
                    self.feature_names = json.loads(feat_path.read_text())
                else:
                    self.feature_names = list(self.model.feature_name())
                logger.info(
                    "LightGBM stacker loaded (%d features)", len(self.feature_names)
                )
            except Exception as e:
                logger.warning("LightGBM stacker load failed: %s", e)
                self.model = None

    def blend(
        self,
        per_model: Dict[str, List[ScoredItem]],
        weights: Dict[str, float],
    ) -> List[ScoredItem]:
        if self.model is None:
            raise NotImplementedError("LightGBM stacker model not present on disk")

        # Build per-item feature dict, then a stable feature matrix.
        item_feats: Dict[str, Dict[str, float]] = defaultdict(dict)
        for model_name, scored in per_model.items():
            for s in scored:
                item_feats[s.item_id][f"score_{model_name}"] = s.score

        item_order = list(item_feats.keys())
        if not item_order:
            return []

        import numpy as np
        X = np.array(
            [
                [item_feats[iid].get(name, 0.0) for name in self.feature_names]
                for iid in item_order
            ],
            dtype=np.float32,
        )
        scores = self.model.predict(X)

        out = [
            ScoredItem(
                item_id=iid,
                score=float(s),
                model="ensemble",
                confidence=1.0,
                features=item_feats[iid],
            )
            for iid, s in zip(item_order, scores)
        ]
        out.sort(key=lambda x: x.score, reverse=True)
        return out


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
    """Fan out to registered models, blend with the configured strategy, diversify."""

    def __init__(
        self,
        strategy: Optional[BlendStrategy] = None,
        reranker: Optional[MMRReranker] = None,
    ) -> None:
        self.weights: Dict[str, float] = dict(settings.ensemble_weights)
        self.strategy: BlendStrategy = strategy or _resolve_strategy(
            getattr(settings, "ensemble_strategy", None)
        )
        # MMRReranker is a no-op until item_emb.npy exists on disk, so it's
        # always safe to construct.
        self.reranker = reranker if reranker is not None else MMRReranker(
            alpha=getattr(settings, "mmr_alpha", 0.7),
            top_k=getattr(settings, "mmr_top_k", 50),
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

        if self.reranker.enabled:
            blended = self.reranker.rerank(blended)

        return EnsembleResponse(
            items=blended,
            models_used=list(per_model.keys()),
            weights={k: self.weights.get(k, 1.0) for k in per_model.keys()},
            generated_at=datetime.utcnow(),
            user_id=user_id,
        )


ensemble = Ensemble()
