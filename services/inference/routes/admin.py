from fastapi import APIRouter, HTTPException

from ..candidates import invalidate as invalidate_candidates
from ..config import settings
from ..registry import registry

router = APIRouter(prefix="/admin", tags=["admin"])


@router.post("/reload/{model}")
def reload_model(model: str) -> dict:
    # allow bare name or name@version; only the name portion needs to be enabled
    name = model.split("@", 1)[0]
    if name not in settings.enabled_models:
        raise HTTPException(404, f"unknown model: {model}")
    loaded = registry.reload(model)
    return {
        "model": loaded.name,
        "version": loaded.version,
        "kind": loaded.kind,
        "is_stub": loaded.is_stub,
        "artifact_path": str(loaded.artifact_path) if loaded.artifact_path else None,
        "error": loaded.error,
    }


@router.post("/reload-all")
def reload_all() -> dict:
    registry.load_all()
    return {
        "reloaded": [
            {"name": s.name, "version": s.version, "is_stub": s.is_stub}
            for s in registry.status()
        ]
    }


@router.post("/reload-candidates")
def reload_candidates() -> dict:
    invalidate_candidates()
    return {"ok": True}


@router.get("/models")
def list_models() -> dict:
    cards = {c.name: c.model_dump(mode="json") for c in registry.cards()}
    return {
        "status": [s.model_dump(mode="json") for s in registry.status()],
        "cards": cards,
    }


@router.get("/config")
def show_config() -> dict:
    return {
        "enabled_models": settings.enabled_models,
        "ensemble_weights": settings.ensemble_weights,
        "models_dir": str(settings.models_dir),
        "feature_store_dir": str(settings.feature_store_dir),
        "device": settings.device,
        "max_candidates": settings.max_candidates,
    }
