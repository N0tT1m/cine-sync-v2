# Model Refactor — Handoff

Goal: trim the sprawling model collection to a small set of canonical classes,
drive serving through a manifest-backed registry. **Training happens elsewhere**
— do not touch trainer scripts, do not try to run training.

## Task status

| # | Task | Status |
|---|------|--------|
| 1 | Wire real registry with manifests | done |
| 2 | Collapse two-tower variants | done |
| 3 | Collapse sequential variants | done |
| 4 | Unify ensembles | done |
| 5 | Move untrained models to `experimental/` | done (aggregator, not physical move) |
| 6 | Add `MODELS.md` catalog | done |
| 7 | Verify imports and service boot | done — full HTTP surface green |

## What was built

### Registry (`services/inference/registry/`)
- `manifest.py` — parses `<models_dir>/<name>/manifest.yaml` into a `ModelCard`.
- `loaders/__init__.py` — `Scorer` protocol + kind-dispatched `LOADERS` dict.
  Loader signature: `LoaderFn(card, artifact, manifest_dir) -> Scorer`.
- `loaders/torch_loader.py` — instantiates `loader_config.class_path`, loads
  state dict, optionally wraps in `scorer_adapter`. `manifest_dir` is threaded
  through so adapters can resolve relative artifact paths.
- `loader.py` — `ModelRegistry` discovers manifests, supports `name@version`
  addressing, falls back to `StubScorer` when no manifest / loader error.
- `routes/admin.py` — added `/admin/models` (status + cards), `/admin/reload-all`;
  `/admin/reload/{model}` now accepts bare name or `name@version`.
- `requirements.txt` — added `pyyaml==6.0.2`.

Manifest schema example (stored under `models/<name>/manifest.yaml`):
```yaml
name: sbert_two_tower
version: 0.2.0
kind: two_tower_sbert
framework: pytorch
artifact: weights.pt                       # relative to this manifest dir
loader_config:
  class_path: src.models.two_tower.TwoTowerModel
  init_kwargs: {user_text_dim: 384, item_text_dim: 384, embedding_dim: 128}
  scorer_adapter: src.models.adapters.TwoTowerScorer
  scorer_kwargs:
    item_embeddings_path: item_emb.npy
    item_id_map_path: item_ids.json
```

### Two-tower (`src/models/two_tower/unified.py`)
One `TwoTowerModel(**config)` replacing Standard / Ultimate / Enhanced /
MultiTask / Collaborative / SentenceBERT / Enhanced (advanced). Config knobs
turn features on/off:
- `num_users`, `num_items` → collaborative ID embeddings
- `user_categorical_dims`, `item_categorical_dims` → learned cat embeddings with sqrt-heuristic sizing
- `user_numerical_dim`, `item_numerical_dim` → dense features
- `user_text_dim`, `item_text_dim` → pre-computed SBERT/CLIP embeddings concatenated in
- `embedding_dim`, `hidden_layers`, `dropout`, `use_batch_norm`
- L2-normalized outputs + learnable temperature

Public API: `encode_user(...)`, `encode_item(...)`, `forward(user=..., item=...)`,
`similarity_matrix(user_emb, item_emb)`. Old classes still importable at their
legacy paths until task #5 moves them.

### Sequential (`src/models/sequential/unified.py`)
One `SequentialRecommender(num_items, architecture=...)` replacing the 5 classes
in `src/models/sequential/src/model.py`. `architecture='transformer'` (SASRec-style
causal, default) or `architecture='rnn'` (LSTM/GRU, emits `DeprecationWarning`).
Public API: `encode(seq)`, `forward(seq)` → logits `(B, T, num_items)`,
`predict_next(seq, top_k)`. Hierarchical + session variants will go to `experimental/`.
BERT4Rec stays separate (distinct MLM objective).

### Adapters (`src/models/adapters/`)
Wrap trained models into the Scorer protocol:
- `TwoTowerScorer` — dot-product over cached `(item_emb.npy, item_ids.json)`
  and optional user index; falls back to deterministic pseudo-scores if artifacts
  are missing.
- `SequentialScorer` — runs `model.encode(history)` → last-position projection →
  per-item logits. Tokenizes `watch_history` via `item_ids.json`.

Both receive `manifest_dir` from the loader, so `*_path` kwargs in the manifest
are resolved relative to the manifest directory.

## Task 4 — decided plan, not yet coded

`services/inference/ensemble/stacker.py` already has a working `Ensemble` class
that does weighted-mean blending over registry-scored outputs. Refactor it to
have an explicit plugin contract:

```python
class BlendStrategy(Protocol):
    def blend(self, per_model: Dict[str, List[ScoredItem]], weights: Dict[str, float]) -> List[ScoredItem]: ...

class WeightedMean(BlendStrategy): ...           # current behavior
class LightGBMStacker(BlendStrategy): ...        # Phase 4 slot, raises until trained
```

Ensemble picks the strategy by name (`settings.ensemble_strategy`, default
`weighted_mean`). That satisfies the "strategy plugins" requirement without
pulling the gigantic `MovieEnsembleRecommender` / `TVEnsembleRecommender`
training modules into the serving path — those two belong in `experimental/`
(task #5), because they're a training-time scaffold that instantiates all
sub-models in one mega `nn.Module` for joint training.

## Task 5 — plan

Move to `src/models/experimental/` (leave re-export shims at old paths so
trainer scripts out-of-repo don't break):

- `src/models/movie/*` (14 sub-models — actor_collaboration, critic_audience,
  director_auteur, international_cinema, viewing_context, cinematic_universe,
  awards_prediction, franchise_sequence, adaptation_source, etc.) — none of
  these have trained artifacts.
- `src/models/hybrid/sota_tv/*` (14 models — watch_pattern, episode_sequence,
  binge_prediction, meta_learning, temporal, etc., plus `ensemble_system.py`).
- The five legacy two-tower variants in `src/models/two_tower/src/model.py`
  (keep file; just add a deprecation note at the top pointing to `unified.py`).
- The four non-canonical sequential variants in `src/models/sequential/src/model.py`
  (HierarchicalSequentialRecommender, SessionBasedRecommender,
  AttentionalSequentialRecommender, TransformerSequentialRecommender — since
  `unified.SequentialRecommender` covers transformer + RNN).
- `src/models/unified/movie_ensemble_system.py` and
  `src/models/hybrid/sota_tv/models/ensemble_system.py`.

Consumers to keep working (grepped, must not break):
- `src/api/extended_model_loader.py` — imports from `src.models.hybrid.sota_tv.models.ensemble_system`
  and `src.models.unified.movie_ensemble_system`. Add re-exports from `experimental/`.
- `src/inference/model_server.py` — same modules, same pattern.
- `src/models/unified_model_manager.py`, `src/models/unified/__init__.py`.
- `src/training/train_all_models.py` — training-only, ignore (user said training is elsewhere).

Shim pattern — old path becomes:
```python
# src/models/unified/movie_ensemble_system.py
from src.models.experimental.unified.movie_ensemble_system import *  # noqa
```

## Task 6 — `MODELS.md` catalog

Single source of truth for operators:
- Shipped canonical models: `TwoTowerModel`, `SequentialRecommender`,
  BERT4Rec, NCF (whatever's canonical), GraphSAGE, VAE, multimodal, contrastive,
  hybrid CF+content.
- Experimental models (post task #5 move): list + one-line rationale why it's there.
- Registry manifest schema reference (point at the example above).
- Scorer adapter contract (`score(item_ids, user_id=None, **ctx) -> List[ScoredItem]`).
- How to register a new model: drop `models/<name>/manifest.yaml` + artifact,
  hit `POST /admin/reload/<name>`.

## Task 7 — verify

What's already passing (via `./venv/bin/python` smoke tests run during this session):
- `services.inference.registry.registry.load_all()` — boots all 9 enabled models
  as stubs (no manifests present yet).
- `services.inference.ensemble.ensemble.score(...)` — returns blended scores.
- `TwoTowerModel` — all 4 config shapes (collab, enhanced, SBERT, hybrid).
- `SequentialRecommender` — transformer + RNN paths emit correct shapes.
- `TwoTowerScorer` / `SequentialScorer` — fallback mode and with a fake vocab.

Still TODO for task 7:
- `./venv/bin/python -c "from src.models.movie import *; from src.models.hybrid.sota_tv.models import *"`
  after the experimental moves to confirm shims work.
- `./venv/bin/uvicorn services.inference.main:app --port 8900` and hit
  `/healthz`, `/admin/models`, `/score/hybrid`, `/score/ensemble`.

## Environment notes

- Primary venv for smoke-testing: `./venv/bin/python` (Python 3.13).
  Has: pydantic 2.10.3, pydantic-settings 2.7.0, fastapi 0.115.5, torch 2.11.0,
  pandas 2.2.3, pyarrow 18.1.0, pyyaml 6.0.2.
- Service deps declared in `services/inference/requirements.txt` (pins differ
  from what's installed in venv — service installs torch 2.5.1).
- `models/` has only `movies/` and `tv/` subdirectories, no manifests yet — so
  every model serves as a stub until real manifests drop.
- `data/feature_store/`, `services/inference/`, `src/enrichment/` are all
  untracked in git; the whole refactor is uncommitted.

## Non-obvious decisions

- Training is off-limits. Don't refactor anything imported only by
  `src/training/*` unless it simplifies serving.
- Registry does not raise on bad manifests — it logs a warning and serves a
  stub. Keeps the API up while artifacts churn.
- `TwoTowerScorer` / `SequentialScorer` fall back to deterministic pseudo-scores
  for unknown items instead of dropping them, so the ensemble always has data
  to blend.
- Kept `pyyaml` as a hard dep (manifest parsing). It's added to
  `services/inference/requirements.txt`.
- `loaders.LoaderFn` signature was widened to `(card, artifact, manifest_dir)`
  in this session — make sure any third-party loaders in-repo are updated if
  discovered later.

## Resume steps

1. Read this file.
2. Read `services/inference/registry/loader.py` to re-anchor on registry shape.
3. Pick up task #4: refactor `services/inference/ensemble/stacker.py` to the
   `BlendStrategy` protocol layout above.
4. Then task #5 (experimental moves + shims), then #6 (MODELS.md), then #7
   (boot + curl `/healthz`, `/admin/models`).
