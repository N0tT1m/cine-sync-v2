# Feature store schema

One canonical layout that every model trains against and the inference service reads from. Producers are source-specific extractors under `sources/`; consumers are the trainers in `src/training/` and the inference service in `services/inference/`.

Everything is Parquet with snappy compression. Item IDs are strings throughout — the canonicalizer maps per-source ids to a stable cine-sync id before landing.

## Files

### `items.parquet`

One row per title across all media types. The `owned` flag is what drives nami-stream's "Available Now" rail.

| column | type | notes |
|---|---|---|
| item_id | string | cine-sync canonical id (prefix by source, e.g. `tmdb:603`, `anilist:21`) |
| media_type | string | `movie`, `tv`, `anime` |
| title | string | primary display title |
| title_alt | string | romaji / english / original, concatenated |
| tmdb_id | int | nullable |
| anilist_id | int | nullable |
| imdb_id | string | nullable |
| plex_guid | string | nullable |
| year | int | nullable |
| runtime_minutes | int | nullable |
| genres | list[string] | |
| tags | list[string] | AniList tags / TMDb keywords |
| overview | string | raw description, used by SBERT job |
| cast | list[string] | TMDb person ids; AniList character+VA ids |
| franchise | string | AniList relations root / TMDb collection; null if unknown |
| owned | bool | true if we have a file for it (nami-stream / mmm-v2 / plex) |
| source | string | primary source of this row |
| updated_at | timestamp | last canonicalize run |

### `interactions.parquet`

Every rating + progress event across every source. Trainers filter by `source` when they want a single-project model.

| column | type | notes |
|---|---|---|
| user_id | string | source-prefixed: `cinesync:1234`, `nami:uuid`, `mmm:uuid`, `discord:snowflake` |
| item_id | string | matches items.item_id |
| event_type | string | `rating`, `progress`, `complete`, `dismiss`, `rewatch` |
| value | float | rating 0-10, or progress fraction 0-1 |
| weight | float | derived: complete=1.0, dismiss=-0.5, progress scales linearly |
| timestamp | timestamp | event time |
| source | string | `cinesync_pg`, `nami_stream`, `mmm_v2`, `discord_bot`, `plex` |
| session_id | string | nullable, enables sequential batching |

### `item_features.parquet`

Dense feature columns populated by enrichment jobs. Keyed by `item_id`. Nullable per column so jobs can populate independently.

| column | type | populated by |
|---|---|---|
| item_id | string | canonicalizer |
| sbert_embedding | list[float32, 384] | `src/enrichment/sbert_embeddings.py` |
| clip_embedding | list[float32, 512] | `src/enrichment/clip_keyframes.py` |
| audio_embedding | list[float32, 512] | `src/enrichment/audio_fingerprint.py` (phase 2) |
| keyframe_count | int | clip job |
| text_length | int | sbert job |

### `graph_edges.parquet`

Edges used by GraphSAGE / LightGCN. Undirected; one row per edge.

| column | type | notes |
|---|---|---|
| src_id | string | node id (item or person or franchise) |
| dst_id | string | node id |
| edge_type | string | `sequel_of`, `cast_of`, `franchise_of`, `tag_shared`, `studio_of` |
| weight | float | default 1.0 |
| source | string | where this edge came from |

## Build flow

```
sources/cinesync_pg.py        ─┐
sources/nami_stream.py         ├─► build_feature_store.py ──► Parquet
sources/mommy_milk_me.py       │
sources/plex.py                ─┘
                                       │
                                       ▼
                 enrichment jobs (sbert, clip, audio) ──► item_features.parquet
```

Re-runs are idempotent: canonical ids are deterministic, and each job writes with `pa.write_table(..., existing_data_behavior='overwrite_or_ignore')`.
