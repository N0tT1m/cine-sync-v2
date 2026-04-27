"""PyTorch loader — instantiates the class named in the manifest and loads weights.

Manifest contract expected under `loader_config`:

    loader_config:
      class_path: src.models.hybrid.UnifiedContentRecommender
      init_kwargs: {num_users: 10000, num_items: 50000}
      scorer_adapter: src.models.adapters.TopKDotAdapter   # optional
      scorer_kwargs: {embedding_key: item_embedding}       # optional

If the instantiated object already exposes `.score(item_ids, user_id, **ctx)`
it is used directly. Otherwise `scorer_adapter` is loaded and wraps it.

Any failure (missing class, shape mismatch, no artifact) raises LoaderError;
the registry catches it and falls back to a stub so the API stays up.
"""
from __future__ import annotations

import importlib
from pathlib import Path
from typing import Optional

from ...schemas import ModelCard


class LoaderError(RuntimeError):
    pass


def _import(dotted: str):
    mod_path, _, attr = dotted.rpartition(".")
    if not mod_path:
        raise LoaderError(f"not a dotted path: {dotted!r}")
    try:
        mod = importlib.import_module(mod_path)
    except ImportError as e:
        raise LoaderError(f"cannot import {mod_path}: {e}") from e
    try:
        return getattr(mod, attr)
    except AttributeError as e:
        raise LoaderError(f"{mod_path} has no attribute {attr!r}") from e


def load_torch_model(card: ModelCard, artifact: Optional[Path], manifest_dir: Path):
    cfg = card.loader_config or {}
    class_path = cfg.get("class_path")
    if not class_path:
        raise LoaderError(f"manifest {card.name}: loader_config.class_path missing")

    cls = _import(class_path)
    init_kwargs = cfg.get("init_kwargs") or {}
    try:
        instance = cls(**init_kwargs)
    except Exception as e:
        raise LoaderError(f"failed to instantiate {class_path}: {e}") from e

    if artifact is not None:
        try:
            import torch  # deferred so pure-config mode doesn't require torch
        except ImportError as e:
            raise LoaderError("torch not installed") from e
        try:
            state = torch.load(artifact, map_location="cpu", weights_only=False)
        except Exception as e:
            raise LoaderError(f"cannot read artifact {artifact}: {e}") from e
        if hasattr(instance, "load_state_dict") and isinstance(state, dict):
            try:
                instance.load_state_dict(state, strict=False)
            except Exception as e:
                raise LoaderError(f"state_dict mismatch for {class_path}: {e}") from e
        if hasattr(instance, "eval"):
            instance.eval()

    adapter_path = cfg.get("scorer_adapter")
    if adapter_path:
        adapter_cls = _import(adapter_path)
        instance = adapter_cls(
            model=instance,
            name=card.name,
            kind=card.kind,
            manifest_dir=manifest_dir,
            **(cfg.get("scorer_kwargs") or {}),
        )
        return instance

    if hasattr(instance, "score"):
        if not hasattr(instance, "name"):
            instance.name = card.name
        if not hasattr(instance, "kind"):
            instance.kind = card.kind
        return instance

    raise LoaderError(
        f"{class_path} has no .score() and no scorer_adapter was configured"
    )
