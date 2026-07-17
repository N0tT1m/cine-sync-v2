"""MovieLens feature-store source.

Flows the bundled MovieLens CSVs (ml-32m / ml-latest-small) into the canonical
feature-store schema so `train.py --all` can run without any live database.

Items are keyed on the canonical ``tmdb:<id>`` id via ``links.csv`` — the same
id space the ``cinesync_pg`` source uses — so a feature store built from
MovieLens merges cleanly with real DB rows later.

Config (env vars, all optional):
    MOVIELENS_DIR       directory holding ratings.csv / movies.csv / links.csv
                        (default: data/movies/ml-32m, falling back to
                        data/movies/recommendation/ml-latest-small)
    MOVIELENS_MAX_ROWS  cap on interactions read (default: all)
"""
from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, Optional

import pandas as pd

from .base import EdgeRow, InteractionRow, ItemRow, canonical_item_id, canonical_user_id

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_DIRS = [
    _REPO_ROOT / "data" / "movies" / "ml-32m",
    _REPO_ROOT / "data" / "movies" / "recommendation" / "ml-latest-small",
]


def _resolve_dir() -> Optional[Path]:
    env = os.getenv("MOVIELENS_DIR")
    candidates = [Path(env)] if env else _DEFAULT_DIRS
    for d in candidates:
        if (d / "ratings.csv").exists() and (d / "movies.csv").exists():
            return d
    return None


def _parse_genres(raw: object) -> list[str]:
    if not isinstance(raw, str) or raw in ("", "(no genres listed)"):
        return []
    return [g.strip() for g in raw.split("|") if g.strip()]


class MovieLensSource:
    """Reads MovieLens CSVs and yields canonical feature-store rows."""

    name = "movielens"

    def __init__(self, data_dir: Optional[Path] = None) -> None:
        self.data_dir = Path(data_dir) if data_dir else _resolve_dir()
        self._movie_to_tmdb: Dict[int, int] = {}
        if self.data_dir is None:
            logger.warning(
                "movielens: no ratings.csv/movies.csv found (looked in %s); "
                "source will yield nothing",
                [str(d) for d in _DEFAULT_DIRS],
            )
            return
        self._load_links()

    def _load_links(self) -> None:
        links_path = self.data_dir / "links.csv"
        if not links_path.exists():
            logger.warning(
                "movielens: links.csv missing in %s; items without a tmdb id are "
                "dropped, so this yields nothing",
                self.data_dir,
            )
            return
        links = pd.read_csv(links_path, usecols=["movieId", "tmdbId"])
        links = links.dropna(subset=["tmdbId"])
        self._movie_to_tmdb = {
            int(m): int(t) for m, t in zip(links["movieId"], links["tmdbId"])
        }
        logger.info("movielens: mapped %d movieId->tmdbId", len(self._movie_to_tmdb))

    # ---- Source protocol ------------------------------------------

    def items(self) -> Iterable[ItemRow]:
        if self.data_dir is None or not self._movie_to_tmdb:
            return
        movies = pd.read_csv(self.data_dir / "movies.csv")
        now = datetime.now(timezone.utc)
        for row in movies.itertuples(index=False):
            tmdb_id = self._movie_to_tmdb.get(int(row.movieId))
            if tmdb_id is None:
                continue
            title, year = _split_title_year(row.title)
            yield ItemRow(
                item_id=canonical_item_id(tmdb_id=tmdb_id, media_type="movie"),
                media_type="movie",
                title=title,
                source=self.name,
                tmdb_id=tmdb_id,
                year=year,
                genres=_parse_genres(row.genres),
                owned=False,
                updated_at=now,
            )

    def interactions(self) -> Iterable[InteractionRow]:
        if self.data_dir is None or not self._movie_to_tmdb:
            return
        max_rows = os.getenv("MOVIELENS_MAX_ROWS")
        limit = int(max_rows) if max_rows else None
        seen = 0
        reader = pd.read_csv(
            self.data_dir / "ratings.csv",
            usecols=["userId", "movieId", "rating", "timestamp"],
            chunksize=500_000,
        )
        for chunk in reader:
            for row in chunk.itertuples(index=False):
                tmdb_id = self._movie_to_tmdb.get(int(row.movieId))
                if tmdb_id is None:
                    continue
                yield InteractionRow(
                    user_id=canonical_user_id(self.name, int(row.userId)),
                    item_id=canonical_item_id(tmdb_id=tmdb_id, media_type="movie"),
                    event_type="rating",
                    value=float(row.rating),
                    weight=1.0,
                    timestamp=datetime.fromtimestamp(int(row.timestamp), tz=timezone.utc),
                    source=self.name,
                )
                seen += 1
                if limit and seen >= limit:
                    logger.info("movielens: hit MOVIELENS_MAX_ROWS=%d", limit)
                    return

    def edges(self) -> Iterable[EdgeRow]:
        return []


def _split_title_year(raw: object) -> tuple[str, Optional[int]]:
    """MovieLens titles look like 'Toy Story (1995)'."""
    if not isinstance(raw, str):
        return "", None
    title = raw.strip()
    if title.endswith(")") and "(" in title:
        head, _, tail = title.rpartition("(")
        maybe_year = tail[:-1]
        if maybe_year.isdigit():
            return head.strip(), int(maybe_year)
    return title, None
