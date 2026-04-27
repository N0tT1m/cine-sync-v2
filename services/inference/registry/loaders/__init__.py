"""Per-framework model loaders. Each loader turns a ModelCard + artifact path
into an object that satisfies the Scorer protocol:

    def score(
        item_ids: Iterable[str],
        user_id: Optional[str] = None,
        **ctx,
    ) -> List[ScoredItem]

Register a loader by framework name in LOADERS below. The registry looks up
card.framework to pick which loader to use. Unknown frameworks fall through
to the stub.
"""
from __future__ import annotations

from typing import Callable, Dict, Optional, Protocol, runtime_checkable
from pathlib import Path

from ...schemas import ModelCard, ScoredItem
from typing import Iterable, List


@runtime_checkable
class Scorer(Protocol):
    """Minimal contract every model must meet to be served."""

    name: str
    kind: str

    def score(
        self,
        item_ids: Iterable[str],
        user_id: Optional[str] = None,
        **ctx: object,
    ) -> List[ScoredItem]: ...


LoaderFn = Callable[[ModelCard, Optional[Path], Path], Scorer]


def _load_stub(card: ModelCard, artifact: Optional[Path], manifest_dir: Path) -> Scorer:
    from ..stub import StubScorer
    return StubScorer(name=card.name, weight_bias=0.0)


def _load_torch(card: ModelCard, artifact: Optional[Path], manifest_dir: Path) -> Scorer:
    from .torch_loader import load_torch_model
    return load_torch_model(card, artifact, manifest_dir)


LOADERS: Dict[str, LoaderFn] = {
    "stub": _load_stub,
    "pytorch": _load_torch,
    "torch": _load_torch,
}


def get_loader(framework: str) -> LoaderFn:
    return LOADERS.get(framework.lower(), _load_stub)
