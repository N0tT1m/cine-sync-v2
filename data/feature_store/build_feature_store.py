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
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Iterable, List

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .sources import (
    cinesync_pg,
    mommy_milk_me,
    nami_stream,
    plex_library,
)
from .sources.base import EdgeRow, InteractionRow, ItemRow

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s - %(message)s")
logger = logging.getLogger("build_feature_store")


SOURCES = {
    "cinesync_pg": cinesync_pg.CineSyncPGSource,
    "nami_stream": nami_stream.NamiStreamSource,
    "mmm_v2": mommy_milk_me.MommyMilkMeSource,
    "plex": plex_library.PlexLibrarySource,
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


def run(out_dir: Path, selected: List[str]) -> None:
    items: Dict[str, ItemRow] = {}
    interactions: List[InteractionRow] = []
    edges: List[EdgeRow] = []

    for name in selected:
        cls = SOURCES.get(name)
        if cls is None:
            logger.error("unknown source: %s", name)
            continue
        src = cls()
        logger.info("reading source: %s", name)
        try:
            _merge_items(items, src.items())
            interactions.extend(src.interactions())
            edges.extend(src.edges())
        except Exception:
            logger.exception("source %s failed; continuing", name)

    _write_parquet(
        out_dir / "items.parquet",
        [asdict(v) for v in items.values()],
        "items",
    )
    _write_parquet(
        out_dir / "interactions.parquet",
        [asdict(v) for v in interactions],
        "interactions",
    )
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
