# Feature store schema

One canonical layout that every model trains against and the inference service reads from. Producers are source-specific extractors under `sources/`; consumers are the trainers in `src/training/` and the inference service in `services/inference/`.

Everything is Parquet with snappy compression. Item IDs are strings throughout — the canonicalizer maps per-source ids to a stable cine-sync id before landing.

## Files

### `items.parquet`

One row per title across all media types. The `owned` flag is what drives nami-stream's "Available Now" rail.

| column | type | notes |
|---|---|---|
| item_id | string | cine-sync canonical id. TMDb ids are qualified by media type (`tmdb:movie:862`, `tmdb:tv:1399`) because TMDb numbers movies and TV in separate sequences — a bare `tmdb:1399` names both Game of Thrones and an unrelated film. Other spaces are globally unique and need no qualifier (`anilist:21`, `imdb:tt0111161`). Minted by `sources/base.py:canonical_item_id`, which rejects a TMDb id with no media type. |
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
| popularity | int | demand signal the serving candidate pool orders by. Observed interaction count where there is one; otherwise the source's own prior (the TV catalog has no per-user data, so it carries TMDb `vote_count`). Only comparable within a `media_type` — the scales differ. |
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
sources/nami_stream.py         │
sources/mommy_milk_me.py       ├─► build_feature_store.py ──► Parquet
sources/plex.py                │
sources/movielens.py           │   (movies + 32M ratings)
sources/tmdb_tv.py             ─┘   (TV catalog + overviews, no interactions)
                                       │
                                       ▼
                 enrichment jobs (sbert, clip, audio) ──► item_features.parquet
```

Which sources carry what matters for which model:

| | interactions | overviews |
|---|---|---|
| `movielens` | 32M ratings (movies only) | no |
| `tmdb_tv` | none — TMDb `vote_count` is an aggregate, not per-user | 90k |

That split is why TV cannot be served by collaborative filtering: no per-user TV
signal exists in this repo (the IMDb/Metacritic TV files are one aggregate row
per title). TV is covered by `sbert_two_tower` over overviews instead, which
needs no interaction data. Build both together:

```
python -m data.feature_store.build_feature_store --sources movielens,tmdb_tv
python -m src.enrichment.sbert_embeddings          # -> item_features.parquet
python -m services.inference.train --models two_tower,sbert_two_tower
```

Re-runs are idempotent: canonical ids are deterministic, and each job writes with `pa.write_table(..., existing_data_behavior='overwrite_or_ignore')`.
