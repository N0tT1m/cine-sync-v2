from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field


class ScoreRequest(BaseModel):
    user_id: Optional[str] = None
    item_ids: Optional[List[str]] = None
    watch_history: Optional[List[str]] = None
    candidate_pool: Optional[List[str]] = None
    limit: int = 20
    owned_only: bool = False
    media_type: Optional[str] = None
    context: dict = Field(default_factory=dict)


class ScoredItem(BaseModel):
    item_id: str
    score: float
    model: str
    confidence: Optional[float] = None
    features: dict = Field(default_factory=dict)


class ScoreResponse(BaseModel):
    model: str
    items: List[ScoredItem]
    generated_at: datetime
    user_id: Optional[str] = None
    cache_hit: bool = False


class EnsembleResponse(BaseModel):
    items: List[ScoredItem]
    models_used: List[str]
    weights: dict
    generated_at: datetime
    user_id: Optional[str] = None
    cache_hit: bool = False


class RailRequest(BaseModel):
    user_id: str
    media_type: Optional[str] = None
    limit: int = 20
    owned_only: bool = True
    seed_item_id: Optional[str] = None


class RailItem(BaseModel):
    item_id: str
    score: float
    reason: Optional[str] = None


class RailResponse(BaseModel):
    rail: str
    items: List[RailItem]
    user_id: str
    generated_at: datetime


class ModelStatus(BaseModel):
    name: str
    loaded: bool
    kind: str
    version: Optional[str] = None
    artifact_path: Optional[str] = None
    loaded_at: Optional[datetime] = None
    error: Optional[str] = None
    is_stub: bool = True


class ModelCard(BaseModel):
    """Manifest metadata for one model version, parsed from manifest.yaml."""

    name: str
    version: str = "0.0.0"
    kind: str
    artifact: Optional[str] = None
    framework: str = "stub"
    loader: str = "stub"
    loader_config: dict = Field(default_factory=dict)
    dataset: Optional[str] = None
    trained_at: Optional[datetime] = None
    metrics: dict = Field(default_factory=dict)
    input_schema: List[str] = Field(default_factory=list)
    notes: Optional[str] = None
    deprecated: bool = False


class HealthResponse(BaseModel):
    status: str
    device: str
    models: List[ModelStatus]
    feature_store_present: bool
    redis_ok: bool
    uptime_seconds: float
