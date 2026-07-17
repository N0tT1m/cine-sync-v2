from datetime import datetime

from fastapi import APIRouter, HTTPException

from ..candidates import candidate_pool
from ..config import settings
from ..ensemble import ensemble
from ..registry import registry
from ..schemas import EnsembleResponse, ScoreRequest, ScoreResponse

router = APIRouter(prefix="/score", tags=["score"])


def _resolve_candidates(req: ScoreRequest) -> list[str]:
    if req.item_ids:
        return req.item_ids
    if req.candidate_pool:
        return req.candidate_pool
    # user_id/history must be forwarded: candidate_pool only attempts ANN
    # retrieval when it has a user to encode, and falls back to the popularity
    # pool otherwise.
    pool = candidate_pool(
        media_type=req.media_type,
        owned_only=req.owned_only,
        limit=settings.max_candidates,
        user_id=req.user_id,
        history=req.watch_history,
    )
    if not pool:
        raise HTTPException(
            status_code=400,
            detail="no candidate items; pass item_ids or candidate_pool, "
            "or populate the feature store",
        )
    return pool


# Declared before `/{model}` so the path parameter doesn't shadow the literal.
@router.post("/ensemble", response_model=EnsembleResponse)
def score_ensemble(req: ScoreRequest) -> EnsembleResponse:
    # With every model a stub there is nothing to rank with, and returning 200s
    # full of hash noise would look like a working recommender to every caller.
    # 503 lets clients fall back to their own popularity path deliberately.
    if not ensemble.live_models():
        raise HTTPException(
            status_code=503,
            detail="no trained models loaded; all enabled models are stubs. "
            "Run: python -m services.inference.train --all",
        )
    candidates = _resolve_candidates(req)
    result = ensemble.score(
        candidates,
        user_id=req.user_id,
        watch_history=req.watch_history,
        context=req.context,
    )
    result.items = result.items[: req.limit]
    return result


@router.post("/{model}", response_model=ScoreResponse)
def score_model(model: str, req: ScoreRequest) -> ScoreResponse:
    name = model.split("@", 1)[0]
    if name not in settings.enabled_models:
        raise HTTPException(404, f"unknown model: {model}")

    candidates = _resolve_candidates(req)
    scored = registry.score(
        model,
        candidates,
        user_id=req.user_id,
        watch_history=req.watch_history,
        context=req.context,
    )[: req.limit]
    return ScoreResponse(
        model=model,
        items=scored,
        generated_at=datetime.utcnow(),
        user_id=req.user_id,
    )
