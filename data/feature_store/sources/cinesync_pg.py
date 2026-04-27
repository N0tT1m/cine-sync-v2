"""Extract items + interactions from the shared cine-sync PostgreSQL database.

The same DB that mommy-milk-me-v2 already reads from. Prefer psycopg3 over SQLAlchemy
to keep the dependency surface small for a batch job.
"""
from __future__ import annotations

import logging
import os
from datetime import datetime
from typing import Iterable, List

import psycopg

from .base import (
    EdgeRow,
    InteractionRow,
    ItemRow,
    canonical_item_id,
    canonical_user_id,
)

logger = logging.getLogger(__name__)


def _dsn() -> str:
    return (
        f"host={os.getenv('CINESYNC_PG_HOST', '192.168.1.74')} "
        f"port={os.getenv('CINESYNC_PG_PORT', '5432')} "
        f"dbname={os.getenv('CINESYNC_PG_DB', 'cinesync')} "
        f"user={os.getenv('CINESYNC_PG_USER', 'postgres')} "
        f"password={os.getenv('CINESYNC_PG_PASSWORD', '')}"
    )


class CineSyncPGSource:
    name = "cinesync_pg"

    def __init__(self, dsn: str | None = None) -> None:
        self.dsn = dsn or _dsn()

    def items(self) -> Iterable[ItemRow]:
        with psycopg.connect(self.dsn) as conn, conn.cursor() as cur:
            cur.execute(
                """
                SELECT tmdb_id, imdb_id, title, release_year, runtime,
                       overview, genres, 'movie' AS media_type
                FROM movies
                WHERE tmdb_id IS NOT NULL OR imdb_id IS NOT NULL
                """
            )
            for row in cur:
                tmdb_id, imdb_id, title, year, runtime, overview, genres, media_type = row
                try:
                    item_id = canonical_item_id(tmdb_id=tmdb_id, imdb_id=imdb_id)
                except ValueError:
                    continue
                yield ItemRow(
                    item_id=item_id,
                    media_type=media_type,
                    title=title or "",
                    source=self.name,
                    tmdb_id=tmdb_id,
                    imdb_id=imdb_id,
                    year=year,
                    runtime_minutes=runtime,
                    overview=overview or "",
                    genres=_parse_pg_array(genres),
                    owned=False,
                    updated_at=datetime.utcnow(),
                )

    def interactions(self) -> Iterable[InteractionRow]:
        with psycopg.connect(self.dsn) as conn, conn.cursor() as cur:
            cur.execute(
                """
                SELECT user_id, tmdb_id, rating, created_at
                FROM ratings
                WHERE rating IS NOT NULL AND tmdb_id IS NOT NULL
                """
            )
            for user_id, tmdb_id, rating, created_at in cur:
                yield InteractionRow(
                    user_id=canonical_user_id("cinesync", user_id),
                    item_id=f"tmdb:{tmdb_id}",
                    event_type="rating",
                    value=float(rating),
                    weight=1.0,
                    timestamp=created_at or datetime.utcnow(),
                    source=self.name,
                )

    def edges(self) -> Iterable[EdgeRow]:
        return []


def _parse_pg_array(value) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v) for v in value if v]
    if isinstance(value, str):
        return [v.strip() for v in value.strip("{}").split(",") if v.strip()]
    return []
