"""Model registry: manifest-driven loading with graceful stub fallback.

Conventions:
  <models_dir>/<name>/manifest.yaml    -> one trained model version
  <models_dir>/<name>/<artifact_file>  -> weights, resolved from manifest.artifact

Addressing:
  "hybrid"         -> latest version registered under name "hybrid"
  "hybrid@0.2.0"   -> that exact version

If no manifest is found for an enabled model, the registry installs a
StubScorer so the serving contract still works end-to-end — the `/admin/config`
and `/healthz` endpoints will show `is_stub=true` for those entries.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from ..config import settings
from ..schemas import ModelCard, ModelStatus, ScoredItem
from .loaders import LOADERS, Scorer, get_loader
from .manifest import discover_manifests, load_manifest, resolve_artifact
from .stub import StubScorer

logger = logging.getLogger(__name__)


# Default kind for enabled-model names that don't (yet) have a manifest.
# Keeps stubs labeled sensibly in /healthz and drives the ensemble bias.
DEFAULT_KIND: Dict[str, str] = {
    "hybrid": "hybrid_cf_content",
    "ncf": "neural_collaborative_filtering",
    "sequential": "sasrec",
    "bert4rec": "bert4rec",
    "sbert_two_tower": "two_tower_sbert",
    "graphsage": "graph_neural_network",
    "contrastive": "contrastive_learning",
    "multimodal": "multimodal_transformer",
    "vae": "variational_autoencoder",
}

STUB_BIAS: Dict[str, float] = {
    "hybrid": 0.00,
    "ncf": 0.02,
    "sequential": 0.03,
    "sbert_two_tower": 0.04,
    "graphsage": 0.02,
    "bert4rec": 0.03,
    "contrastive": -0.02,
    "multimodal": 0.01,
    "vae": -0.04,
}


@dataclass
class LoadedModel:
    name: str
    kind: str
    scorer: Scorer
    version: str = "0.0.0"
    card: Optional[ModelCard] = None
    artifact_path: Optional[Path] = None
    loaded_at: datetime = field(default_factory=datetime.utcnow)
    error: Optional[str] = None
    is_stub: bool = True


def _parse_addr(addr: str) -> Tuple[str, Optional[str]]:
    """Split 'name@version' -> (name, version); bare 'name' -> (name, None)."""
    if "@" in addr:
        name, version = addr.split("@", 1)
        return name, version or None
    return addr, None


class ModelRegistry:
    def __init__(self) -> None:
        # keyed on "name@version" for precise addressing; also indexed by
        # name alone so callers can request the latest loaded version.
        self._by_addr: Dict[str, LoadedModel] = {}
        self._latest: Dict[str, str] = {}
        # manifests are discovered once at startup; reload() re-reads them.
        self._manifests: Dict[str, List[Tuple[Path, ModelCard]]] = {}
        self._started = time.monotonic()

    # ---- discovery ----------------------------------------------------

    def _refresh_manifests(self) -> None:
        self._manifests.clear()
        for manifest_dir, card in discover_manifests(settings.models_dir):
            self._manifests.setdefault(card.name, []).append((manifest_dir, card))
        for name, cards in self._manifests.items():
            # sort newest-version last, so _pick_manifest picks the last one
            cards.sort(key=lambda pair: pair[1].version)
        if self._manifests:
            logger.info(
                "registry discovered %d manifests across %d names",
                sum(len(v) for v in self._manifests.values()),
                len(self._manifests),
            )

    def _pick_manifest(
        self, name: str, version: Optional[str]
    ) -> Optional[Tuple[Path, ModelCard]]:
        cards = self._manifests.get(name) or []
        if not cards:
            return None
        if version is None:
            return cards[-1]
        for dir_, card in cards:
            if card.version == version:
                return dir_, card
        return None

    # ---- loading ------------------------------------------------------

    def load_all(self) -> None:
        self._refresh_manifests()
        for name in settings.enabled_models:
            self.load(name)

    def load(self, addr: str) -> LoadedModel:
        name, version = _parse_addr(addr)
        if not self._manifests:
            # first call before any startup refresh
            self._refresh_manifests()

        picked = self._pick_manifest(name, version)
        if picked is not None:
            manifest_dir, card = picked
            loaded = self._load_from_manifest(manifest_dir, card)
        else:
            loaded = self._load_stub(
                name=name,
                kind=DEFAULT_KIND.get(name, name),
                version=version or "0.0.0",
                reason=(
                    "no manifest.yaml found for this model"
                    if version is None
                    else f"no manifest with version={version} for {name!r}"
                ),
            )

        self._remember(loaded)
        self._log_loaded(loaded)
        return loaded

    def reload(self, addr: str) -> LoadedModel:
        self._refresh_manifests()
        return self.load(addr)

    def _load_from_manifest(
        self, manifest_dir: Path, card: ModelCard
    ) -> LoadedModel:
        artifact = resolve_artifact(manifest_dir, card)
        loader = get_loader(card.framework)
        try:
            scorer = loader(card, artifact, manifest_dir)
            return LoadedModel(
                name=card.name,
                kind=card.kind,
                version=card.version,
                scorer=scorer,
                card=card,
                artifact_path=artifact or manifest_dir,
                is_stub=(card.framework.lower() == "stub"),
            )
        except Exception as e:  # LoaderError + anything a loader might raise
            return self._load_stub(
                name=card.name,
                kind=card.kind,
                version=card.version,
                reason=f"loader '{card.framework}' failed: {e}",
                card=card,
                artifact_path=artifact,
            )

    def _load_stub(
        self,
        *,
        name: str,
        kind: str,
        version: str,
        reason: str,
        card: Optional[ModelCard] = None,
        artifact_path: Optional[Path] = None,
    ) -> LoadedModel:
        scorer = StubScorer(name=name, weight_bias=STUB_BIAS.get(name, 0.0))
        return LoadedModel(
            name=name,
            kind=kind,
            version=version,
            scorer=scorer,
            card=card,
            artifact_path=artifact_path,
            error=reason,
            is_stub=True,
        )

    def _remember(self, loaded: LoadedModel) -> None:
        addr = f"{loaded.name}@{loaded.version}"
        self._by_addr[addr] = loaded
        self._latest[loaded.name] = addr

    def _log_loaded(self, loaded: LoadedModel) -> None:
        if loaded.is_stub:
            logger.warning(
                "model %s@%s serving as stub: %s",
                loaded.name,
                loaded.version,
                loaded.error or "no manifest",
            )
        else:
            logger.info(
                "model %s@%s ready (artifact=%s)",
                loaded.name,
                loaded.version,
                loaded.artifact_path,
            )

    # ---- serving ------------------------------------------------------

    def score(
        self,
        model: str,
        item_ids: Iterable[str],
        user_id: Optional[str] = None,
        **kwargs: object,
    ) -> List[ScoredItem]:
        loaded = self._resolve(model)
        return loaded.scorer.score(item_ids, user_id=user_id, **kwargs)

    def _resolve(self, addr: str) -> LoadedModel:
        if addr in self._by_addr:
            return self._by_addr[addr]
        name, version = _parse_addr(addr)
        if version is None and name in self._latest:
            return self._by_addr[self._latest[name]]
        return self.load(addr)

    def status(self) -> List[ModelStatus]:
        return [
            ModelStatus(
                name=m.name,
                loaded=True,
                kind=m.kind,
                version=m.version,
                artifact_path=str(m.artifact_path) if m.artifact_path else None,
                loaded_at=m.loaded_at,
                error=m.error,
                is_stub=m.is_stub,
            )
            for m in self._by_addr.values()
        ]

    def cards(self) -> List[ModelCard]:
        return [m.card for m in self._by_addr.values() if m.card is not None]

    def uptime(self) -> float:
        return time.monotonic() - self._started


registry = ModelRegistry()


__all__ = [
    "DEFAULT_KIND",
    "LOADERS",
    "LoadedModel",
    "ModelRegistry",
    "load_manifest",
    "registry",
]
