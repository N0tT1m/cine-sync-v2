"""Extract interactions + items from mommy-milk-me-v2's Postgres.

mmm-v2 already shares the cinesync DB, so most items come in through that source.
This source adds mmm-v2-specific feedback: explicit ratings from its admin UI and
'not interested' dismissals captured by its feedback tables.
"""
from __future__ import annotations

import logging
import os
from datetime import datetime
from typing import Iterable

import psycopg

from .base import EdgeRow, InteractionRow, ItemRow, canonical_user_id

logger = logging.getLogger(__name__)


def _dsn() -> str:
    return (
        f"host={os.getenv('MMM_PG_HOST', '192.168.1.74')} "
        f"port={os.getenv('MMM_PG_PORT', '5432')} "
        f"dbname={os.getenv('MMM_PG_DB', 'cinesync')} "
        f"user={os.getenv('MMM_PG_USER', 'postgres')} "
        f"password={os.getenv('MMM_PG_PASSWORD', '')}"
    )


class MommyMilkMeSource:
    name = "mmm_v2"

    def __init__(self, dsn: str | None = None) -> None:
        self.dsn = dsn or _dsn()

    def items(self) -> Iterable[ItemRow]:
        return []

    def interactions(self) -> Iterable[InteractionRow]:
        yield from self._ratings()
        yield from self._feedback()

    def _ratings(self) -> Iterable[InteractionRow]:
        try:
            with psycopg.connect(self.dsn) as conn, conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT user_id, tmdb_id, rating, created_at
                    FROM mmm_user_ratings
                    WHERE rating IS NOT NULL AND tmdb_id IS NOT NULL
                    """
                )
                for user_id, tmdb_id, rating, ts in cur:
                    yield InteractionRow(
                        user_id=canonical_user_id("mmm", str(user_id)),
                        item_id=f"tmdb:{tmdb_id}",
                        event_type="rating",
                        value=float(rating),
                        weight=1.0,
                        timestamp=ts or datetime.utcnow(),
                        source=self.name,
                    )
        except psycopg.Error as exc:
            logger.info("no mmm_user_ratings table: %s", exc)

    def _feedback(self) -> Iterable[InteractionRow]:
        try:
            with psycopg.connect(self.dsn) as conn, conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT user_id, tmdb_id, feedback_type, created_at
                    FROM mmm_feedback
                    """
                )
                for user_id, tmdb_id, feedback_type, ts in cur:
                    if feedback_type == "dismiss":
                        yield InteractionRow(
                            user_id=canonical_user_id("mmm", str(user_id)),
                            item_id=f"tmdb:{tmdb_id}",
                            event_type="dismiss",
                            value=0.0,
                            weight=-0.5,
                            timestamp=ts or datetime.utcnow(),
                            source=self.name,
                        )
        except psycopg.Error as exc:
            logger.info("no mmm_feedback table: %s", exc)

    def edges(self) -> Iterable[EdgeRow]:
        return []
