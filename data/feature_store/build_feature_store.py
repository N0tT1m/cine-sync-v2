"""Union all sources into the canonical Parquet layout.

Usage:
    python -m data.feature_store.build_feature_store --out data/feature_store
    python -m data.feature_store.build_feature_store --sources nami_stream,cinesync_pg

Each source is independently retryable; failure of one does not block the others.
Items are merged by item_id — later rows update earlier ones, and owned=True wins.
"""
from __future__ import annotations

import argparse
import logging
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Iterable, List

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .sources import (
    cinesync_pg,
    mommy_milk_me,
    movielens,
    nami_stream,
    plex_library,
)
from .sources.base import EdgeRow, InteractionRow, ItemRow

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s - %(message)s")
logger = logging.getLogger("build_feature_store")

# Interactions are the one unbounded table — MovieLens alone is 32M rows, which
# will not fit in memory as dataclasses. They stream to Parquet in batches
# against this explicit schema; items/edges are small enough to buffer.
INTERACTION_SCHEMA = pa.schema(
    [
        ("user_id", pa.string()),
        ("item_id", pa.string()),
        ("event_type", pa.string()),
        ("value", pa.float64()),
        ("weight", pa.float64()),
        ("timestamp", pa.timestamp("us", tz="UTC")),
        ("source", pa.string()),
        ("session_id", pa.string()),
    ]
)

INTERACTION_BATCH_ROWS = 500_000


SOURCES = {
    "cinesync_pg": cinesync_pg.CineSyncPGSource,
    "nami_stream": nami_stream.NamiStreamSource,
    "mmm_v2": mommy_milk_me.MommyMilkMeSource,
    "plex": plex_library.PlexLibrarySource,
    "movielens": movielens.MovieLensSource,
}


def _merge_items(existing: Dict[str, ItemRow], incoming: Iterable[ItemRow]) -> None:
    for row in incoming:
        prev = existing.get(row.item_id)
        if prev is None:
            existing[row.item_id] = row
            continue
        # merge: prefer non-empty fields from incoming; owned is sticky True
        merged = ItemRow(
            item_id=prev.item_id,
            media_type=prev.media_type or row.media_type,
            title=row.title or prev.title,
            source=prev.source,  # keep original primary source
            title_alt=row.title_alt or prev.title_alt,
            tmdb_id=prev.tmdb_id or row.tmdb_id,
            anilist_id=prev.anilist_id or row.anilist_id,
            imdb_id=prev.imdb_id or row.imdb_id,
            plex_guid=prev.plex_guid or row.plex_guid,
            year=prev.year or row.year,
            runtime_minutes=prev.runtime_minutes or row.runtime_minutes,
            genres=sorted({*prev.genres, *row.genres}),
            tags=sorted({*prev.tags, *row.tags}),
            overview=row.overview or prev.overview,
            cast=sorted({*prev.cast, *row.cast}),
            franchise=prev.franchise or row.franchise,
            owned=prev.owned or row.owned,
            updated_at=row.updated_at,
        )
        existing[row.item_id] = merged


def _write_parquet(path: Path, rows: List[dict], schema_hint: str) -> None:
    if not rows:
        logger.warning("no rows for %s; writing empty parquet", schema_hint)
        df = pd.DataFrame(rows)
    else:
        df = pd.DataFrame(rows)
    table = pa.Table.from_pandas(df)
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path, compression="snappy")
    logger.info("wrote %s (%d rows)", path, len(df))


class InteractionWriter:
    """Batched Parquet writer for the interactions table.

    Rows arrive from a generator and are flushed every INTERACTION_BATCH_ROWS so
    peak memory stays flat regardless of source size. Closing without any rows
    still emits a valid empty file carrying INTERACTION_SCHEMA, so downstream
    readers get a schema rather than a headerless blank.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._writer = pq.ParquetWriter(path, INTERACTION_SCHEMA, compression="snappy")
        self._batch: List[dict] = []
        self.rows_written = 0
        # Interaction count per item, tallied on the single streaming pass.
        # Bounded by the item count (~87k), not the interaction count, so it
        # costs nothing to carry and saves a second 32M-row scan later.
        self.popularity: Counter = Counter()

    def extend(self, rows: Iterable[InteractionRow]) -> int:
        """Append rows; returns the count added by this call."""
        added = 0
        for row in rows:
            self._batch.append(
                {
                    "user_id": row.user_id,
                    "item_id": row.item_id,
                    "event_type": row.event_type,
                    "value": float(row.value),
                    "weight": float(row.weight),
                    "timestamp": row.timestamp,
                    "source": row.source,
                    "session_id": row.session_id,
                }
            )
            self.popularity[row.item_id] += 1
            added += 1
            if len(self._batch) >= INTERACTION_BATCH_ROWS:
                self._flush()
        self._flush()
        return added

    def _flush(self) -> None:
        if not self._batch:
            return
        table = pa.Table.from_pylist(self._batch, schema=INTERACTION_SCHEMA)
        self._writer.write_table(table)
        self.rows_written += len(self._batch)
        self._batch = []
        logger.info("interactions: %d rows written so far", self.rows_written)

    def close(self) -> None:
        self._flush()
        self._writer.close()


def run(out_dir: Path, selected: List[str]) -> None:
    items: Dict[str, ItemRow] = {}
    edges: List[EdgeRow] = []

    out_dir.mkdir(parents=True, exist_ok=True)
    interactions = InteractionWriter(out_dir / "interactions.parquet")

    try:
        for name in selected:
            cls = SOURCES.get(name)
            if cls is None:
                logger.error("unknown source: %s", name)
                continue
            src = cls()
            logger.info("reading source: %s", name)
            try:
                _merge_items(items, src.items())
                added = interactions.extend(src.interactions())
                edges.extend(src.edges())
                logger.info("source %s contributed %d interactions", name, added)
            except Exception:
                logger.exception("source %s failed; continuing", name)
    finally:
        interactions.close()

    logger.info(
        "wrote %s (%d rows)", out_dir / "interactions.parquet", interactions.rows_written
    )

    # `popularity` lets the serving-side candidate pool rank by observed demand.
    # Without it the pool falls back to Parquet row order, which is catalog id
    # order — an arbitrary slice, not the titles anyone actually watches.
    item_rows = []
    for row in items.values():
        d = asdict(row)
        d["popularity"] = interactions.popularity.get(row.item_id, 0)
        item_rows.append(d)
    item_rows.sort(key=lambda d: d["popularity"], reverse=True)

    _write_parquet(out_dir / "items.parquet", item_rows, "items")
    _write_parquet(
        out_dir / "graph_edges.parquet",
        [asdict(v) for v in edges],
        "graph_edges",
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("data/feature_store"))
    ap.add_argument(
        "--sources",
        default="cinesync_pg,nami_stream,mmm_v2,plex",
        help="comma-separated subset of sources",
    )
    args = ap.parse_args()
    run(args.out, [s.strip() for s in args.sources.split(",") if s.strip()])


if __name__ == "__main__":
    main()
