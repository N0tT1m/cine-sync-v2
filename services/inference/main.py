import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .config import settings
from .registry import registry
from .routes import admin, health, rails, score

logging.basicConfig(
    level=settings.log_level,
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    registry.load_all()
    yield


app = FastAPI(
    title="cine-sync inference",
    version="0.1.0",
    description="Unified recommendation service for cine-sync, mommy-milk-me-v2, nami-stream.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(score.router)
app.include_router(rails.router)
app.include_router(admin.router)


@app.get("/")
def root() -> dict:
    return {
        "service": "cine-sync inference",
        "models": settings.enabled_models,
        "endpoints": [
            "/healthz",
            "/readyz",
            "/score/{model}",
            "/score/ensemble",
            "/rails",
            "/rails/{key}",
            "/admin/*",
        ],
    }
