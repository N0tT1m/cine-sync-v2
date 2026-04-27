from datetime import datetime

import torch
from fastapi import APIRouter

from ..config import settings
from ..registry import registry
from ..schemas import HealthResponse

router = APIRouter(tags=["health"])


def _device_label() -> str:
    if settings.device != "auto":
        return settings.device
    if torch.cuda.is_available():
        return f"cuda:{torch.cuda.get_device_name(0)}"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@router.get("/healthz", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        device=_device_label(),
        models=registry.status(),
        feature_store_present=settings.feature_store_dir.exists(),
        redis_ok=False,  # wired in Phase 3 cache integration
        uptime_seconds=registry.uptime(),
    )


@router.get("/readyz")
def ready() -> dict:
    ready_models = [m for m in registry.status() if m.loaded]
    return {
        "ready": len(ready_models) > 0,
        "model_count": len(ready_models),
        "generated_at": datetime.utcnow().isoformat(),
    }
