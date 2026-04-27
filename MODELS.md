# Models — catalog

Single source of truth for what's shipped, what's experimental, and how the
inference service finds and serves a model. Training pipelines live elsewhere;
everything here is about **what the serving path sees**.

## Layout

```
src/models/
├── two_tower/unified.py        → TwoTowerModel            (canonical)
├── sequential/unified.py       → SequentialRecommender    (canonical)
├── advanced/bert4rec_recommender.py  → BERT4Rec            (canonical, distinct objective)
├── hybrid/                     → hybrid CF+content models  (canonical)
├── adapters/                   → Scorer adapters (TwoTowerScorer, SequentialScorer)
└── experimental/               → curated aggregator for research models
```

## Shipped models

The registry's `enabled_models` list is defined in `services/inference/config.py`.
Each entry serves as a `StubScorer` until a manifest lands. The inference
service boots and responds either way.

| Name              | Kind                          | Canonical class                                                 | Adapter                           |
|-------------------|-------------------------------|-----------------------------------------------------------------|-----------------------------------|
| `hybrid`          | hybrid_cf_content             | `src/models/hybrid` (existing recommender)                     | (none — class satisfies protocol) |
| `ncf`             | neural_collaborative_filtering | existing NCF class                                              | (none)                            |
| `sequential`      | sasrec                        | `src.models.sequential.SequentialRecommender`                    | `src.models.adapters.SequentialScorer` |
| `sbert_two_tower` | two_tower_sbert               | `src.models.two_tower.TwoTowerModel` (with `user_text_dim`)      | `src.models.adapters.TwoTowerScorer` |
| `graphsage`       | graph_neural_network          | `src.models.advanced.graphsage_recommender`                      | (TBD)                             |
| `bert4rec`        | bert4rec                      | `src.models.advanced.bert4rec_recommender`                       | (TBD)                             |
| `contrastive`     | contrastive_learning          | `src.models.advanced.contrastive_learning` (pending)             | (TBD)                             |
| `multimodal`      | multimodal_transformer        | `src.models.multimodal_content_understanding`                    | (TBD)                             |
| `vae`             | variational_autoencoder       | `src.models.advanced.variational_autoencoder`                    | (TBD)                             |

## Registry manifest

A trained model becomes addressable by dropping a directory under `models_dir`
(default: `<repo>/models/`):

```
models/
└── sbert_two_tower/
    ├── manifest.yaml
    ├── weights.pt
    ├── item_emb.npy
    └── item_ids.json
```

`manifest.yaml` schema — required fields in bold:

| Field             | Type              | Notes                                                          |
|-------------------|-------------------|----------------------------------------------------------------|
| **`name`**        | str               | Matches an entry in `settings.enabled_models` (bare name).     |
| `version`         | str               | Semver-ish. Default `"0.0.0"`. Address with `name@version`.    |
| **`kind`**        | str               | Free-form tag surfaced in `/healthz`. Drives ensemble intent.  |
| `framework`       | str               | `"pytorch"` / `"torch"` / `"stub"`. Picks the loader.          |
| `artifact`        | str               | Relative path to weights file inside this manifest dir.        |
| `loader_config`   | dict              | Framework-specific (see below).                                |
| `dataset`         | str               | Training dataset identifier.                                   |
| `trained_at`      | datetime          | When weights were produced.                                    |
| `metrics`         | dict              | E.g. `{ndcg@10: 0.32}`. Surfaced in `/admin/models`.           |
| `input_schema`    | list[str]         | Required context keys at serve time.                           |
| `notes`           | str               | Freeform.                                                      |
| `deprecated`      | bool              | For future rollback tooling.                                   |

`loader_config` for `framework: pytorch`:

```yaml
loader_config:
  class_path: src.models.two_tower.TwoTowerModel        # fully-qualified
  init_kwargs: {user_text_dim: 384, item_text_dim: 384, embedding_dim: 128}
  scorer_adapter: src.models.adapters.TwoTowerScorer    # optional
  scorer_kwargs:                                        # passed to adapter
    item_embeddings_path: item_emb.npy                  # resolved relative to manifest dir
    item_id_map_path: item_ids.json
```

If the class already exposes `.score(item_ids, user_id, **ctx)` that satisfies
the `Scorer` protocol, omit `scorer_adapter`. If neither is set, loading fails
and the registry falls back to a stub with a recorded error.

## Scorer protocol

Every served model reduces to:

```python
def score(item_ids: Iterable[str], user_id: Optional[str] = None, **ctx) -> List[ScoredItem]
```

`ScoredItem` fields: `item_id`, `score` (float in `[0, 1]` — not required but
ensembling assumes comparable ranges), `model`, `confidence`, `features`.

Adapters included:

- `TwoTowerScorer` — dot-product over a cached item embedding index (`.npy` or
  `.pt`) + `item_ids.json`. Optional user index for precomputed user
  embeddings. Falls back to deterministic pseudo-scores for unknown items.
- `SequentialScorer` — tokenizes `watch_history` via `item_ids.json`, runs
  `model.encode(seq)`, projects onto the item embedding to obtain per-item
  logits, returns sigmoid(logit) as score.

## Ensemble

`services/inference/ensemble/stacker.py` — fans out to every enabled model via
`registry.score(...)`, then delegates blending to a `BlendStrategy`:

- `WeightedMean` (default) — weighted mean with per-item normalization, so
  partial-coverage models don't drag scores down.
- `LightGBMStacker` — Phase 4 slot. Raises until trained; the `Ensemble`
  catches `NotImplementedError` and falls back to `WeightedMean` so the API
  stays up.

Add a strategy: implement `blend(per_model, weights) -> List[ScoredItem]` and
register in `STRATEGIES`. Configure via `settings.ensemble_strategy`.

## Experimental namespace

`src.models.experimental` re-exports research models into one namespace. Most
of these have optional heavy deps (`torch_geometric`, `sentence-transformers`,
`open_clip_torch`) — the aggregator imports defensively. Check
`EXPERIMENTAL_LOAD_ERRORS` for what failed to import in the current env.

Models currently aggregated as experimental:

- 14 movie-domain sub-models: `ActorCollaborationModel`, `AdaptationSourceModel`,
  `AwardsPredictionModel`, `CinematicUniverseModel`, `CriticAudienceModel`,
  `DirectorAuteurModel`, `EraStyleModel`, `FranchiseSequenceModel`,
  `InternationalCinemaModel`, `NarrativeComplexityModel`, `RemakeConnectionModel`,
  `RuntimePreferenceModel`, `StudioFingerprintModel`, `ViewingContextModel`.
- 14 TV-domain models under `sota_tv/models/`: `BingePredictionModel`,
  `CastMigrationModel`, `ContrastiveTVLearning`, `TVEnsembleSystem`,
  `EpisodeSequenceModel`, `TVGraphNeuralNetwork`, `MetaLearningTVModel`,
  `MultimodalTVTransformer`, `PlatformAvailabilityModel`, `SeasonQualityModel`,
  `SeriesCompletionModel`, `SeriesLifecycleModel`, `TemporalAttentionTVModel`,
  `WatchPatternModel`.
- Training-time ensemble scaffold: `MovieEnsembleRecommender`.
- Legacy two-tower variants: `CollaborativeTwoTowerModel`,
  `LegacyEnhancedTwoTowerModel`, `MultiTaskTwoTowerModel`,
  `UltimateTwoTowerModel`, `AdvancedEnhancedTwoTowerModel`.
- Legacy sequential variants: `AttentionalSequentialRecommender`,
  `HierarchicalSequentialRecommender`, `SessionBasedRecommender`,
  `TransformerSequentialRecommender`.

## Registering a new trained model

1. Train the model wherever you train (out of scope here).
2. Pick/confirm the `name` and add it to `settings.enabled_models` if it's new.
3. Drop the manifest under `<models_dir>/<name>/manifest.yaml` plus any
   artifacts referenced by `artifact`/`scorer_kwargs`.
4. Either restart the service or hit `POST /admin/reload/<name>`.
5. Verify via `GET /admin/models` — the entry should show
   `is_stub: false`, a `kind`, and the artifact path. `GET /healthz`
   reflects the same.

If the loader errors (shape mismatch, missing class, bad artifact), the
registry records the error and serves a stub — check `/admin/models` for the
`error` field.

## Admin endpoints cheat-sheet

| Method | Path                            | Purpose                                               |
|--------|---------------------------------|-------------------------------------------------------|
| GET    | `/healthz`                      | Device, per-model status, uptime.                     |
| GET    | `/readyz`                       | K8s-style readiness.                                  |
| GET    | `/admin/config`                 | Effective settings snapshot.                          |
| GET    | `/admin/models`                 | Full status + model cards (manifest contents).        |
| POST   | `/admin/reload/{name}`          | Re-read manifest and reload. Accepts `name@version`.  |
| POST   | `/admin/reload-all`             | Reload every enabled model.                           |
| POST   | `/admin/reload-candidates`      | Invalidate the cached candidate pool.                 |
| POST   | `/score/{name}`                 | Score one model.                                      |
| POST   | `/score/ensemble`               | Score the blended ensemble.                           |
