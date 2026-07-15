"""Shared helpers for feature-store-native trainers: IO, id maps, manifests."""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger("trainers")

REPO_ROOT = Path(__file__).resolve().parents[3]


def feature_store_dir() -> Path:
    return Path(os.getenv("CINESYNC_FEATURE_STORE", REPO_ROOT / "data" / "feature_store"))


def model_outdir(name: str) -> Path:
    out = Path(os.getenv("CINESYNC_MODEL_OUTDIR", REPO_ROOT / "models" / name))
    out.mkdir(parents=True, exist_ok=True)
    return out


# ---- feature store readers -------------------------------------------------

def load_items() -> pd.DataFrame:
    path = feature_store_dir() / "items.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing; run build_feature_store first")
    return pd.read_parquet(path)


def load_interactions() -> pd.DataFrame:
    path = feature_store_dir() / "interactions.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing; run build_feature_store first")
    df = pd.read_parquet(path)
    if df.empty:
        raise ValueError(f"{path} has no rows — nothing to train on")
    return df


def load_item_features() -> pd.DataFrame:
    path = feature_store_dir() / "item_features.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} missing; run `python -m src.enrichment.sbert_embeddings` first"
        )
    return pd.read_parquet(path)


# ---- math ------------------------------------------------------------------

def l2_normalize(mat: np.ndarray) -> np.ndarray:
    mat = np.asarray(mat, dtype=np.float32)
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return mat / norms


# ---- artifact writers ------------------------------------------------------

def save_embeddings(outdir: Path, prefix: str, emb: np.ndarray, ids: List[str]) -> Dict[str, str]:
    """Write <prefix>_emb.npy + <prefix>_ids.json ({id: row_index}).

    Returns the scorer_kwargs fragment pointing at the two files.
    """
    emb_file = f"{prefix}_emb.npy"
    ids_file = f"{prefix}_ids.json"
    np.save(outdir / emb_file, emb.astype(np.float32))
    id_map = {str(item_id): i for i, item_id in enumerate(ids)}
    (outdir / ids_file).write_text(json.dumps(id_map))
    logger.info("wrote %s (%s) + %s (%d ids)", emb_file, emb.shape, ids_file, len(id_map))
    return {f"{prefix}_embeddings_path": emb_file, f"{prefix}_id_map_path": ids_file}


def save_id_map(outdir: Path, filename: str, ids: List[str], start: int = 0) -> str:
    """Write {id: index} json (index starting at `start`). Returns filename."""
    id_map = {str(item_id): i + start for i, item_id in enumerate(ids)}
    (outdir / filename).write_text(json.dumps(id_map))
    logger.info("wrote %s (%d ids, base=%d)", filename, len(id_map), start)
    return filename


def write_manifest(
    outdir: Path,
    *,
    name: str,
    kind: str,
    loader_config: dict,
    framework: str = "pytorch",
    artifact: Optional[str] = None,
    metrics: Optional[dict] = None,
    dataset: str = "feature_store",
    notes: str = "",
) -> Path:
    manifest = {
        "name": name,
        "version": "0.1.0",
        "kind": kind,
        "framework": framework,
        "artifact": artifact,
        "loader_config": loader_config,
        "dataset": dataset,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "metrics": metrics or {},
        "notes": notes,
    }
    path = outdir / "manifest.yaml"
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(manifest, f, sort_keys=False)
    logger.info("wrote %s (framework=%s, adapter=%s)",
                path, framework, loader_config.get("scorer_adapter"))
    return path


def pick_device(prefer: str = "auto") -> str:
    if prefer != "auto":
        return prefer
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass
    return "cpu"
