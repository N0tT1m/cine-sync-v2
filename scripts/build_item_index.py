"""Build a FAISS HNSW index over trained two-tower item embeddings.

Output (sibling files under models/two_tower/):
    item_emb.npy      — (num_items, D) float32, L2-normalized
    item_ids.json     — list of item_id strings, index-aligned with item_emb
    item_index.faiss  — HNSW index for ANN retrieval (services/inference/candidates.py)

Run after `train_all_models.py --model two_tower` completes and the manifest
exists at models/two_tower/manifest.yaml.

Usage:
    ./venv/bin/python -m scripts.build_item_index
    ./venv/bin/python -m scripts.build_item_index --hnsw-m 32 --efConstruction 200
"""
from __future__ import annotations

import argparse
import importlib
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")
logger = logging.getLogger("build_index")


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_two_tower(manifest_path: Path):
    with manifest_path.open() as f:
        manifest = yaml.safe_load(f)
    cfg = manifest.get("loader_config") or {}
    class_path = cfg["class_path"]
    init_kwargs = cfg.get("init_kwargs") or {}
    mod_path, _, attr = class_path.rpartition(".")
    cls = getattr(importlib.import_module(mod_path), attr)
    model = cls(**init_kwargs)
    weights = torch.load(manifest_path.parent / manifest["artifact"], map_location="cpu", weights_only=False)
    if isinstance(weights, dict) and "model_state_dict" in weights:
        weights = weights["model_state_dict"]
    model.load_state_dict(weights, strict=False)
    model.eval()
    return model, init_kwargs


def _encode_all_items(model, num_items: int, batch_size: int, device: torch.device) -> np.ndarray:
    model.to(device)
    out: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, num_items, batch_size):
            ids = torch.arange(start, min(start + batch_size, num_items), device=device, dtype=torch.long)
            emb = model.encode_item(item_ids=ids).cpu().numpy()
            out.append(emb.astype(np.float32))
    embeddings = np.vstack(out)
    norms = np.linalg.norm(embeddings, axis=-1, keepdims=True)
    norms[norms == 0] = 1.0
    return embeddings / norms


def main() -> None:
    parser = argparse.ArgumentParser(description="Build FAISS HNSW item index")
    parser.add_argument(
        "--manifest",
        type=str,
        default=str(REPO_ROOT / "models" / "two_tower" / "manifest.yaml"),
    )
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--hnsw-m", type=int, default=32, help="HNSW M parameter")
    parser.add_argument(
        "--efConstruction", type=int, default=200, help="HNSW efConstruction parameter"
    )
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        logger.error("manifest not found: %s", manifest_path)
        sys.exit(1)

    model, init_kwargs = _load_two_tower(manifest_path)
    num_items = init_kwargs.get("num_items")
    if not num_items:
        logger.error("manifest init_kwargs.num_items missing — cannot enumerate items")
        sys.exit(1)

    logger.info("encoding %d items via two-tower item branch", num_items)
    embeddings = _encode_all_items(model, num_items, args.batch_size, torch.device(args.device))
    logger.info("encoded shape=%s", embeddings.shape)

    out_dir = manifest_path.parent
    np.save(out_dir / "item_emb.npy", embeddings)
    item_ids = [str(i) for i in range(num_items)]
    (out_dir / "item_ids.json").write_text(json.dumps(item_ids))

    try:
        import faiss
    except ImportError:
        logger.warning(
            "faiss not installed — wrote item_emb.npy + item_ids.json but skipping HNSW index. "
            "pip install faiss-cpu (or faiss-gpu) to enable ANN retrieval."
        )
        return

    dim = embeddings.shape[1]
    index = faiss.IndexHNSWFlat(dim, args.hnsw_m, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = args.efConstruction
    index.add(embeddings)
    faiss.write_index(index, str(out_dir / "item_index.faiss"))
    logger.info("wrote FAISS HNSW index to %s", out_dir / "item_index.faiss")


if __name__ == "__main__":
    main()
