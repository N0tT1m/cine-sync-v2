from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Iterable, List, Optional, Protocol


@dataclass
class ItemRow:
    item_id: str
    media_type: str
    title: str
    source: str
    title_alt: str = ""
    tmdb_id: Optional[int] = None
    anilist_id: Optional[int] = None
    imdb_id: Optional[str] = None
    plex_guid: Optional[str] = None
    year: Optional[int] = None
    runtime_minutes: Optional[int] = None
    genres: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    overview: str = ""
    cast: List[str] = field(default_factory=list)
    franchise: Optional[str] = None
    owned: bool = False
    updated_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class InteractionRow:
    user_id: str
    item_id: str
    event_type: str
    value: float
    weight: float
    timestamp: datetime
    source: str
    session_id: Optional[str] = None


@dataclass
class EdgeRow:
    src_id: str
    dst_id: str
    edge_type: str
    source: str
    weight: float = 1.0


class Source(Protocol):
    name: str

    def items(self) -> Iterable[ItemRow]: ...
    def interactions(self) -> Iterable[InteractionRow]: ...
    def edges(self) -> Iterable[EdgeRow]: ...


def canonical_item_id(*, tmdb_id: Optional[int] = None, anilist_id: Optional[int] = None,
                      imdb_id: Optional[str] = None, plex_guid: Optional[str] = None) -> str:
    """Pick the most stable id we have. tmdb > anilist > imdb > plex."""
    if tmdb_id is not None:
        return f"tmdb:{tmdb_id}"
    if anilist_id is not None:
        return f"anilist:{anilist_id}"
    if imdb_id:
        return f"imdb:{imdb_id}"
    if plex_guid:
        return f"plex:{plex_guid}"
    raise ValueError("no stable id available for item")


def canonical_user_id(source: str, raw: str | int) -> str:
    return f"{source}:{raw}"
