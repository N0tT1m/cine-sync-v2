"""Mark TMDb / AniList items as owned based on the Plex library.

For items that exist in nami-stream, the owned flag is already set by that source.
This source is the fallback for content that's on the Plex server but hasn't been
ingested by nami-stream yet — so the recommender still knows we can serve it.
"""
from __future__ import annotations

import logging
import os
from typing import Iterable

from .base import EdgeRow, InteractionRow, ItemRow

logger = logging.getLogger(__name__)


class PlexLibrarySource:
    name = "plex"

    def __init__(self, base_url: str | None = None, token: str | None = None) -> None:
        self.base_url = base_url or os.getenv("PLEX_BASE_URL")
        self.token = token or os.getenv("PLEX_TOKEN")

    def items(self) -> Iterable[ItemRow]:
        if not self.base_url or not self.token:
            logger.info("plex not configured; skipping")
            return
        # Implementation placeholder: PlexAPI client → iterate libraries → yield
        # rows with owned=True. Phase 0 leaves this empty since nami-stream is the
        # primary owned-content source; wire up when the nami-stream scanner hasn't
        # picked up a title you have on the Plex server.
        return
        yield  # make this a generator

    def interactions(self) -> Iterable[InteractionRow]:
        return []

    def edges(self) -> Iterable[EdgeRow]:
        return []
