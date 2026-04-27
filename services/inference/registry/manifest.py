"""Parse and validate manifest.yaml files that describe one trained model.

Convention: `<models_dir>/<name>/manifest.yaml` marks a trained model version.
One manifest = one (name, version). Put multiple versions of the same name
under separate directories (`hybrid-v1/`, `hybrid-v2/`) or distinguish via the
`version` field and pick per-name with `registry.load("hybrid@0.2.0")`.

If yaml parsing fails or required fields are missing, the caller should treat
the model as unavailable and fall back to a stub.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import yaml
from pydantic import ValidationError

from ..schemas import ModelCard


def load_manifest(path: Path) -> ModelCard:
    """Load one manifest.yaml into a ModelCard. Raises on parse/validation error."""
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    loader_cfg = data.pop("loader_config", {}) or {}
    # pydantic's default_factory will handle missing dicts; keep explicit.
    data["loader_config"] = loader_cfg
    try:
        card = ModelCard(**data)
    except ValidationError as e:
        raise ValueError(f"invalid manifest at {path}: {e}") from e
    return card


def discover_manifests(models_dir: Path) -> List[tuple[Path, ModelCard]]:
    """Walk `<models_dir>/<*>/manifest.yaml`. Returns (dir, card) for each valid one.

    Silently skips entries that fail to parse — the registry logs these as warnings.
    Callers that want strict behavior should iterate and call load_manifest directly.
    """
    out: List[tuple[Path, ModelCard]] = []
    if not models_dir.exists():
        return out
    for entry in sorted(models_dir.iterdir()):
        if not entry.is_dir():
            continue
        manifest = entry / "manifest.yaml"
        if not manifest.exists():
            continue
        try:
            card = load_manifest(manifest)
            out.append((entry, card))
        except ValueError:
            continue
    return out


def resolve_artifact(manifest_dir: Path, card: ModelCard) -> Optional[Path]:
    """Resolve card.artifact relative to the manifest dir. None if unset/missing."""
    if not card.artifact:
        return None
    artifact = (manifest_dir / card.artifact).resolve()
    return artifact if artifact.exists() else None
