"""Convert trainer checkpoints into registry-compatible manifest dirs.

The trainer in `train_all_models.py` saves wrapped checkpoints under
`models/{movies,tv,unified}/<name>/checkpoint_best.pt` of the form:

    {'model_state_dict': ..., 'metrics': ..., 'timestamp': ..., 'category': ...}

The inference registry (`services/inference/registry/loader.py`) wants:

    models/<servable_name>/
        manifest.yaml      # name, kind, class_path, init_kwargs, scorer_adapter, ...
        weights.pt         # flat state_dict
        metrics.json       # snapshot of training metrics

This module bridges the two: takes a wrapped checkpoint, writes the flat
artifact + manifest + metrics under the registry's expected layout. Trainer
model names that have no canonical serving slot (research/movie-specific
models) are skipped — `publish()` returns None.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import yaml

logger = logging.getLogger(__name__)


# Trainer model-name -> registry servable spec. Only models with an
# entry here become addressable by the inference registry. Add new
# entries when a new canonical class+adapter pair lands.
TRAINER_TO_SERVABLE: Dict[str, Dict[str, Any]] = {
    'two_tower': {
        'name': 'two_tower',
        'kind': 'two_tower_collaborative',
        'class_path': 'src.models.two_tower.TwoTowerModel',
        'scorer_adapter': 'src.models.adapters.TwoTowerScorer',
        # init kwargs to copy from the trainer-side dict into the manifest
        'init_kwargs_keys': (
            'num_users', 'num_items', 'embedding_dim',
            'user_categorical_dims', 'item_categorical_dims',
            'user_numerical_dim', 'item_numerical_dim',
            'user_text_dim', 'item_text_dim',
            'hidden_layers', 'dropout', 'use_batch_norm',
        ),
    },
    'sequential_recommender': {
        'name': 'sequential',
        'kind': 'sasrec',
        'class_path': 'src.models.sequential.SequentialRecommender',
        'scorer_adapter': 'src.models.adapters.SequentialScorer',
        'init_kwargs_keys': (
            'num_items', 'embedding_dim', 'hidden_dim',
            'num_layers', 'num_heads', 'max_seq_len', 'dropout',
            'architecture', 'rnn_type', 'tie_weights',
        ),
    },
}


def _flat_state_dict(loaded: Any) -> Dict[str, torch.Tensor]:
    """Unwrap the trainer's wrapped checkpoint into a flat state_dict."""
    if isinstance(loaded, dict) and 'model_state_dict' in loaded:
        return loaded['model_state_dict']
    if isinstance(loaded, dict) and all(isinstance(v, torch.Tensor) for v in loaded.values()):
        return loaded
    raise ValueError(
        f"unexpected checkpoint shape: keys={list(loaded.keys()) if isinstance(loaded, dict) else type(loaded)}"
    )


def _build_manifest(
    spec: Dict[str, Any],
    init_kwargs: Dict[str, Any],
    metrics: Dict[str, Any],
    dataset: Optional[str],
) -> Dict[str, Any]:
    filtered_init = {
        k: v for k, v in init_kwargs.items() if k in spec['init_kwargs_keys']
    }
    manifest: Dict[str, Any] = {
        'name': spec['name'],
        'version': '0.1.0',
        'kind': spec['kind'],
        'framework': 'pytorch',
        'artifact': 'weights.pt',
        'loader_config': {
            'class_path': spec['class_path'],
            'init_kwargs': filtered_init,
            'scorer_adapter': spec['scorer_adapter'],
            # scorer_kwargs left empty — adapter falls back to deterministic
            # pseudo-scores until item_emb.npy / item_ids.json are populated
            # by a downstream embedding-export step.
            'scorer_kwargs': {},
        },
        'trained_at': datetime.utcnow().isoformat() + 'Z',
        'metrics': metrics or {},
    }
    if dataset:
        manifest['dataset'] = dataset
    return manifest


def publish(
    trainer_model_name: str,
    checkpoint_path: Path,
    init_kwargs: Dict[str, Any],
    metrics: Dict[str, Any],
    models_dir: Path,
    dataset: Optional[str] = None,
) -> Optional[Path]:
    """Convert one trainer checkpoint into a registry-compatible manifest dir.

    Returns the path to the written manifest.yaml, or None if there is no
    servable mapping for this trainer model name (research/experimental).
    """
    spec = TRAINER_TO_SERVABLE.get(trainer_model_name)
    if spec is None:
        logger.info("no servable mapping for %s; skipping registry publish", trainer_model_name)
        return None

    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")

    loaded = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    state_dict = _flat_state_dict(loaded)

    out_dir = Path(models_dir) / spec['name']
    out_dir.mkdir(parents=True, exist_ok=True)

    weights_path = out_dir / 'weights.pt'
    torch.save(state_dict, weights_path)

    metrics_path = out_dir / 'metrics.json'
    metrics_path.write_text(json.dumps(metrics or {}, indent=2, default=str))

    manifest = _build_manifest(spec, init_kwargs, metrics or {}, dataset)
    manifest_path = out_dir / 'manifest.yaml'
    with manifest_path.open('w') as f:
        yaml.safe_dump(manifest, f, sort_keys=False)

    logger.info(
        "published %s -> %s (artifact=%s, manifest=%s)",
        trainer_model_name, spec['name'], weights_path, manifest_path,
    )
    return manifest_path


__all__ = ['publish', 'TRAINER_TO_SERVABLE']
