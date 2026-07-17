"""Feature-store source for mommy-milk-me-v2.

mmm-v2 records every personalization signal in one unified table,
`interaction_events` — the substrate its own docs call "what the cross-content
scorer and discovery rails read from". Each row is a canonical ContentRef
(`source:kind:id`, e.g. `tmdb:movie:603`, `adult:content:1234`) with an event
type, weight, value, and a consent flag. We read that stream (consent only) and
its owned catalog (`adult_content`) and map both into the feature store.

Ref reconciliation: TMDb refs pass through unchanged — mmm-v2's `tmdb:movie:603`
is exactly the feature store's canonical id. The kind used to be dropped here to
match a 2-part convention, but TMDb numbers movies and TV in separate sequences,
so `tmdb:1399` names both Game of Thrones and an unrelated film; dropping the
kind merged them. Non-TMDb refs keep the 2-part form (`adult:content:1234` ->
`adult:1234`), whose ids are unique on their own.

Connects to mmm-v2's Postgres (default db `mommy_milk_me`). Env overrides:
    MMM_PG_HOST / MMM_PG_PORT / MMM_PG_DB / MMM_PG_USER / MMM_PG_PASSWORD
"""
from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from typing import Iterable

try:
    import psycopg
except ImportError:  # optional: only needed when this DB source is selected
    psycopg = None  # build_feature_store catches the failure per-source

from .base import EdgeRow, InteractionRow, ItemRow, canonical_user_id

logger = logging.getLogger(__name__)


def _dsn() -> str:
    return (
        f"host={os.getenv('MMM_PG_HOST', '192.168.1.74')} "
        f"port={os.getenv('MMM_PG_PORT', '5432')} "
        f"dbname={os.getenv('MMM_PG_DB', 'mommy_milk_me')} "
        f"user={os.getenv('MMM_PG_USER', 'postgres')} "
        f"password={os.getenv('MMM_PG_PASSWORD', '')}"
    )


def canonical_from_ref(ref: str) -> str | None:
    """mmm-v2 ContentRef -> feature-store canonical id. None if malformed.

    `tmdb:movie:603` -> `tmdb:movie:603`  (kind is part of the id: TMDb reuses
                                           numbers across movies and TV)
    `adult:content:1234` -> `adult:1234`  (id is unique without the kind)
    """
    parts = ref.split(":")
    if len(parts) == 3 and parts[0] and parts[2]:
        if parts[0] == "tmdb":
            return ref
        return f"{parts[0]}:{parts[2]}"
    if len(parts) == 2 and parts[0] and parts[1]:  # already 2-part, pass through
        return ref
    return None


# mmm-v2 event_type -> (feature-store event_type, synthetic rating on a 0..5
# scale). Explicit ratings keep their own score; implicit signals get a
# sensible positive/negative stand-in so the collaborative model has a target.
_EVENT_MAP = {
    "rate": ("rating", None),      # value carries the real score
    "complete": ("complete", 5.0),
    "favorite": ("favorite", 4.5),
    "watchlist": ("watchlist", 4.0),
    "view": ("view", 3.0),
    "click": ("click", 2.5),
    "skip": ("dismiss", 0.0),
}


class MommyMilkMeSource:
    name = "mmm_v2"

    def __init__(self, dsn: str | None = None) -> None:
        self.dsn = dsn or _dsn()

    # ---- interactions: the unified consented event stream -------------

    def interactions(self) -> Iterable[InteractionRow]:
        if psycopg is None:
            logger.info("psycopg not installed; mmm_v2 interactions skipped")
            return
        try:
            with psycopg.connect(self.dsn) as conn, conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT user_id, ref, event_type, weight, value, created_at
                    FROM interaction_events
                    WHERE consent = TRUE
                    """
                )
                for user_id, ref, event_type, weight, value, ts in cur:
                    item_id = canonical_from_ref(str(ref))
                    if item_id is None:
                        continue
                    mapped = _EVENT_MAP.get(str(event_type))
                    if mapped is None:
                        continue
                    fs_event, default_value = mapped
                    if default_value is None:            # explicit rating
                        row_value = float(value or 0.0)
                    elif str(event_type) == "complete":  # value is completion 0..1
                        frac = float(value or 0.0)
                        row_value = 5.0 * frac if 0.0 <= frac <= 1.0 else default_value
                    else:
                        row_value = default_value
                    yield InteractionRow(
                        user_id=canonical_user_id("mmm", str(user_id)),
                        item_id=item_id,
                        event_type=fs_event,
                        value=row_value,
                        weight=float(weight) if weight is not None else 1.0,
                        timestamp=ts or datetime.now(timezone.utc),
                        source=self.name,
                    )
        except psycopg.Error as exc:
            logger.info("mmm_v2 interaction_events unavailable: %s", exc)

    # ---- items: the owned adult catalog (rich text for the sbert tower) --

    def items(self) -> Iterable[ItemRow]:
        if psycopg is None:
            return
        enrichment = self._catalog_enrichment()
        try:
            with psycopg.connect(self.dsn) as conn, conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT id, title, studio, description, release_date, duration
                    FROM adult_content
                    WHERE deleted_at IS NULL
                    """
                )
                for cid, title, studio, description, release_date, duration in cur:
                    tags, performers = enrichment.get(int(cid), ([], []))
                    year = release_date.year if release_date else None
                    yield ItemRow(
                        item_id=f"adult:{cid}",
                        media_type="adult",
                        title=title or f"content_{cid}",
                        source=self.name,
                        year=year,
                        runtime_minutes=int(duration // 60) if duration else None,
                        genres=tags,          # tags act as genres for content sim
                        cast=performers,
                        overview=description or "",
                        franchise=studio or None,
                        owned=True,           # it's in the local library
                        updated_at=datetime.now(timezone.utc),
                    )
        except psycopg.Error as exc:
            logger.info("mmm_v2 adult_content unavailable: %s", exc)

    def _catalog_enrichment(self) -> dict[int, tuple[list[str], list[str]]]:
        """Best-effort tags + performers per content id. Degrades to empty on any
        schema mismatch so items() still yields title/studio/description text."""
        out: dict[int, tuple[list[str], list[str]]] = {}
        if psycopg is None:
            return out
        try:
            with psycopg.connect(self.dsn) as conn, conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT act.adult_content_id, array_agg(DISTINCT t.name)
                    FROM adult_content_tags act
                    JOIN adult_tags t ON t.id = act.adult_tag_id
                    GROUP BY act.adult_content_id
                    """
                )
                for cid, tags in cur:
                    out[int(cid)] = ([x for x in tags if x], [])
                cur.execute(
                    """
                    SELECT acp.adult_content_id, array_agg(DISTINCT p.name)
                    FROM adult_content_performers acp
                    JOIN adult_performers p ON p.id = acp.adult_performer_id
                    GROUP BY acp.adult_content_id
                    """
                )
                for cid, performers in cur:
                    tags, _ = out.get(int(cid), ([], []))
                    out[int(cid)] = (tags, [x for x in performers if x])
        except psycopg.Error as exc:
            logger.info("mmm_v2 tag/performer enrichment skipped: %s", exc)
        return out

    def edges(self) -> Iterable[EdgeRow]:
        return []
