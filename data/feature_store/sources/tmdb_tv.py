"""TMDb TV catalog feature-store source.

Supplies the TV half of the catalog. MovieLens is movies-only, so without this
the feature store has no `media_type='tv'` rows at all and any TV request
resolves to an empty candidate pool.

This dataset carries `overview` text, which matters beyond browsing: the
semantic model (`sbert_two_tower`) embeds title+overview+genres, so overviews
are the only thing that makes content-based TV recommendations possible. There
is no per-user TV interaction data anywhere in the repo — the IMDb/Metacritic
TV files hold aggregate scores, one row per title — so collaborative filtering
cannot cover TV. Content similarity is the honest path for it.

Items are keyed `tmdb:tv:<id>` off the dataset's native TMDb id — the same form
mommy-milk-me's ContentRefs already use, so its refs map across without
translation. The media_type qualifier is load-bearing: TMDb numbers movies and
TV separately, so an unqualified `tmdb:1399` collides with a film of that id.

Config (env vars, all optional):
    TMDB_TV_CSV        path to TMDB_tv_dataset_v3.csv
                       (default: data/tv/kaggle/TMDB_tv_dataset_v3.csv)
    TMDB_TV_INCLUDE_ADULT  set to 1 to keep adult-flagged rows (default: drop)
    TMDB_TV_MIN_VOTES  drop rows below this vote_count (default: 0 = keep all)
"""
from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd

from .base import EdgeRow, InteractionRow, ItemRow, canonical_item_id

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_CSV = _REPO_ROOT / "data" / "tv" / "kaggle" / "TMDB_tv_dataset_v3.csv"

_USECOLS = [
    "id",
    "name",
    "original_name",
    "overview",
    "genres",
    "first_air_date",
    "adult",
    "vote_count",
    "vote_average",
    "episode_run_time",
    "networks",
]


def _resolve_csv() -> Optional[Path]:
    env = os.getenv("TMDB_TV_CSV")
    path = Path(env) if env else _DEFAULT_CSV
    return path if path.exists() else None


def _split_csv_field(raw: object) -> list[str]:
    """The dataset stores lists as comma-joined strings ('Drama, Crime')."""
    if not isinstance(raw, str) or not raw.strip():
        return []
    return [part.strip() for part in raw.split(",") if part.strip()]


def _year_of(raw: object) -> Optional[int]:
    if not isinstance(raw, str) or len(raw) < 4:
        return None
    head = raw[:4]
    return int(head) if head.isdigit() else None


def _truthy(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in ("true", "1", "yes")
    return bool(value) if pd.notna(value) else False


class TMDbTVSource:
    """Reads the TMDb TV dataset and yields canonical feature-store rows."""

    name = "tmdb_tv"

    def __init__(self, csv_path: Optional[Path] = None) -> None:
        self.csv_path = Path(csv_path) if csv_path else _resolve_csv()
        if self.csv_path is None:
            logger.warning(
                "tmdb_tv: %s not found; source yields nothing. Unzip "
                "data/tv/kaggle/tmdb-tv.zip or set TMDB_TV_CSV.",
                _DEFAULT_CSV,
            )

    def items(self) -> Iterable[ItemRow]:
        if self.csv_path is None:
            return
        include_adult = os.getenv("TMDB_TV_INCLUDE_ADULT", "") == "1"
        min_votes = int(os.getenv("TMDB_TV_MIN_VOTES", "0"))

        df = pd.read_csv(self.csv_path, usecols=_USECOLS, low_memory=False)
        total = len(df)

        df = df[df["id"].notna()]
        # The catalog ships ~2k adult-flagged titles. Dropping them here keeps
        # them out of the candidate pool entirely rather than relying on every
        # downstream caller to filter.
        if not include_adult:
            before = len(df)
            df = df[~df["adult"].map(_truthy)]
            logger.info("tmdb_tv: dropped %d adult-flagged rows", before - len(df))
        if min_votes > 0:
            df = df[df["vote_count"].fillna(0) >= min_votes]

        # Same tmdb id can repeat; keep the best-attested row per id.
        df = df.sort_values("vote_count", ascending=False, na_position="last")
        df = df.drop_duplicates(subset=["id"], keep="first")

        now = datetime.now(timezone.utc)
        emitted = 0
        with_overview = 0
        for row in df.itertuples(index=False):
            try:
                tmdb_id = int(row.id)
            except (TypeError, ValueError):
                continue
            title = row.name if isinstance(row.name, str) else ""
            if not title:
                continue
            overview = row.overview if isinstance(row.overview, str) else ""
            if overview:
                with_overview += 1
            original = row.original_name if isinstance(row.original_name, str) else ""
            runtime = None
            if pd.notna(row.episode_run_time):
                try:
                    runtime = int(float(row.episode_run_time))
                except (TypeError, ValueError):
                    runtime = None
            # No TV interactions exist, so vote_count is the only demand signal
            # available to order the TV candidate pool by.
            votes = 0
            if pd.notna(row.vote_count):
                try:
                    votes = int(float(row.vote_count))
                except (TypeError, ValueError):
                    votes = 0
            yield ItemRow(
                item_id=canonical_item_id(tmdb_id=tmdb_id, media_type="tv"),
                media_type="tv",
                title=title,
                source=self.name,
                title_alt=original if original and original != title else "",
                tmdb_id=tmdb_id,
                year=_year_of(row.first_air_date),
                runtime_minutes=runtime,
                genres=_split_csv_field(row.genres),
                tags=_split_csv_field(row.networks),
                overview=overview,
                owned=False,
                popularity=votes,
                updated_at=now,
            )
            emitted += 1
        logger.info(
            "tmdb_tv: emitted %d/%d tv items (%d with an overview)",
            emitted, total, with_overview,
        )

    def interactions(self) -> Iterable[InteractionRow]:
        # No per-user TV signal exists in this repo; vote_average is an aggregate
        # and cannot be attributed to a user, so emitting it would fabricate one.
        return []

    def edges(self) -> Iterable[EdgeRow]:
        return []
