# cine-sync inference service

Unified FastAPI recommendation service. Serves nami-stream, mommy-milk-me-v2, and any future consumer over HTTP.

## Run locally

```bash
cd cine-sync-v2
pip install -r services/inference/requirements.txt
PYTHONPATH=. uvicorn services.inference.main:app --reload --port 8900
```

Browse: `http://localhost:8900/docs`

## Run in Docker

```bash
docker build -f services/inference/Dockerfile -t cinerec:dev .
docker run --rm -p 8900:8900 cinerec:dev
```

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | `/healthz` | Liveness + loaded-model status |
| GET | `/readyz` | Readiness — at least one model usable |
| POST | `/score/{model}` | Score a candidate list with one model |
| POST | `/score/ensemble` | Score with weighted-mean ensemble |
| GET | `/rails` | List available rails |
| POST | `/rails/{rail_key}` | Build a home-page rail for a user |
| POST | `/admin/reload/{model}` | Hot-reload a model artifact |
| POST | `/admin/reload-candidates` | Invalidate items.parquet cache |
| GET | `/admin/config` | Show current settings |

## Phase 0 behavior

No models are trained yet on this machine. Every slot is filled by a deterministic `StubScorer` that returns stable pseudo-scores keyed on `(model_name, user_id, item_id)`. This lets clients (nami-stream, mmm-v2) integrate and test the contract before training runs.

Once `cine-sync-v2/src/training/train_all_models.py` drops real artifacts under `models/<name>/manifest.yaml`, the registry swaps the stub for the real loader without any API change.

## Feature store

The service reads candidate pools from `cine-sync-v2/data/feature_store/items.parquet`. Until that file exists, callers must pass explicit `item_ids` / `candidate_pool` in requests.

## Configuration

All via env vars with the `CINEREC_` prefix. See `config.py` for the full list. Common overrides:

```bash
CINEREC_PORT=8900
CINEREC_MODELS_DIR=/app/models
CINEREC_FEATURE_STORE_DIR=/app/data/feature_store
CINEREC_ENABLED_MODELS='["hybrid","ncf","sequential"]'
CINEREC_LOG_LEVEL=DEBUG
```

## Client libraries

- Go (nami-stream): `nami-stream/src/services/recommendations/client.go`
- Go (mommy-milk-me-v2): `mommy-milk-me-v2/src/recommendations/recommendations.go` — flip `BASE_URL` env to point here.
