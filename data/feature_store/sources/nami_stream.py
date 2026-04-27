"""Extract items + interactions + graph edges from the nami-stream database.

nami-stream uses Postgres/SQLite via GORM. We read directly to avoid an HTTP round
trip for bulk extraction. Anime is first-class — AniList relations become graph
edges which GraphSAGE / LightGCN train against.
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
        f"host={os.getenv('NAMI_PG_HOST', 'localhost')} "
        f"port={os.getenv('NAMI_PG_PORT', '5435')} "
        f"dbname={os.getenv('NAMI_PG_DB', 'nami_stream')} "
        f"user={os.getenv('NAMI_PG_USER', 'postgres')} "
        f"password={os.getenv('NAMI_PG_PASSWORD', '')}"
    )


class NamiStreamSource:
    name = "nami_stream"

    def __init__(self, dsn: str | None = None) -> None:
        self.dsn = dsn or _dsn()

    def items(self) -> Iterable[ItemRow]:
        yield from self._movies()
        yield from self._tv()
        yield from self._anime()

    def _movies(self) -> Iterable[ItemRow]:
        with psycopg.connect(self.dsn) as conn, conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, tmdb_id, imdb_id, title, year, runtime,
                       overview, genres, rating
                FROM movies WHERE deleted_at IS NULL
                """
            )
            for row in cur:
                nami_id, tmdb_id, imdb_id, title, year, runtime, overview, genres, _rating = row
                try:
                    item_id = canonical_item_id(tmdb_id=tmdb_id, imdb_id=imdb_id, plex_guid=str(nami_id))
                except ValueError:
                    continue
                yield ItemRow(
                    item_id=item_id,
                    media_type="movie",
                    title=title or "",
                    source=self.name,
                    tmdb_id=tmdb_id,
                    imdb_id=imdb_id,
                    year=year,
                    runtime_minutes=runtime,
                    overview=overview or "",
                    genres=_arr(genres),
                    owned=True,  # if nami-stream has it, we own the file
                    updated_at=datetime.utcnow(),
                )

    def _tv(self) -> Iterable[ItemRow]:
        with psycopg.connect(self.dsn) as conn, conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, tmdb_id, title, year, overview, genres, rating
                FROM tv_shows WHERE deleted_at IS NULL
                """
            )
            for row in cur:
                nami_id, tmdb_id, title, year, overview, genres, _rating = row
                try:
                    item_id = canonical_item_id(tmdb_id=tmdb_id, plex_guid=str(nami_id))
                except ValueError:
                    continue
                yield ItemRow(
                    item_id=item_id,
                    media_type="tv",
                    title=title or "",
                    source=self.name,
                    tmdb_id=tmdb_id,
                    year=year,
                    overview=overview or "",
                    genres=_arr(genres),
                    owned=True,
                    updated_at=datetime.utcnow(),
                )

    def _anime(self) -> Iterable[ItemRow]:
        with psycopg.connect(self.dsn) as conn, conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, anilist_id, title, title_english, title_romaji,
                       year, overview, genres, rating
                FROM anime_shows WHERE deleted_at IS NULL
                """
            )
            for row in cur:
                nami_id, anilist_id, title, te, tr, year, overview, genres, _rating = row
                try:
                    item_id = canonical_item_id(anilist_id=anilist_id, plex_guid=str(nami_id))
                except ValueError:
                    continue
                yield ItemRow(
                    item_id=item_id,
                    media_type="anime",
                    title=te or tr or title or "",
                    title_alt=" | ".join(filter(None, [title, tr, te])),
                    source=self.name,
                    anilist_id=anilist_id,
                    year=year,
                    overview=overview or "",
                    genres=_arr(genres),
                    owned=True,
                    updated_at=datetime.utcnow(),
                )

    def interactions(self) -> Iterable[InteractionRow]:
        with psycopg.connect(self.dsn) as conn, conn.cursor() as cur:
            cur.execute(
                """
                SELECT user_id, media_id, media_type, position_seconds, duration_seconds,
                       updated_at
                FROM watch_progresses
                """
            )
            for user_id, media_id, media_type, pos, dur, updated_at in cur:
                dur_f = float(dur or 0)
                pos_f = float(pos or 0)
                frac = (pos_f / dur_f) if dur_f > 0 else 0.0
                # completion >= 85% counts as a completion event (positive signal)
                event_type = "complete" if frac >= 0.85 else "progress"
                weight = 1.0 if event_type == "complete" else frac
                yield InteractionRow(
                    user_id=canonical_user_id("nami", str(user_id)),
                    item_id=f"plex:{media_id}",  # resolved via items.plex_guid
                    event_type=event_type,
                    value=frac,
                    weight=weight,
                    timestamp=updated_at or datetime.utcnow(),
                    source=self.name,
                )

    def edges(self) -> Iterable[EdgeRow]:
        # anime franchise/sequel edges from stored AniList relations table.
        try:
            with psycopg.connect(self.dsn) as conn, conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT src_anilist_id, dst_anilist_id, relation_type
                    FROM anime_relations
                    """
                )
                for src, dst, rel in cur:
                    yield EdgeRow(
                        src_id=f"anilist:{src}",
                        dst_id=f"anilist:{dst}",
                        edge_type=(rel or "related").lower(),
                        source=self.name,
                    )
        except psycopg.Error as exc:
            logger.info("no anime_relations table yet: %s", exc)


def _arr(value) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v) for v in value if v]
    return [v.strip() for v in str(value).strip("{}").split(",") if v.strip()]
