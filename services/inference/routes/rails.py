from fastapi import APIRouter, HTTPException

from ..rails import RAIL_REGISTRY, build_rail
from ..schemas import RailRequest, RailResponse

router = APIRouter(prefix="/rails", tags=["rails"])


@router.get("")
def list_rails() -> dict:
    return {
        "rails": [
            {
                "key": spec.key,
                "model": spec.model,
                "reason": spec.reason_template,
            }
            for spec in RAIL_REGISTRY.values()
        ]
    }


@router.post("/{rail_key}", response_model=RailResponse)
def build(rail_key: str, req: RailRequest) -> RailResponse:
    if rail_key not in RAIL_REGISTRY:
        raise HTTPException(404, f"unknown rail: {rail_key}")
    try:
        return build_rail(
            rail_key=rail_key,
            user_id=req.user_id,
            media_type=req.media_type,
            limit=req.limit,
            owned_only=req.owned_only,
            seed_item_id=req.seed_item_id,
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc))
