"""Embed each item's text blob (title + overview + tags) with SBERT.

Output: item_features.parquet gains sbert_embedding (list[float32, 384]) and
text_length columns. Uses all-MiniLM-L6-v2 by default — 384-dim, fast on CPU
and GPU, a solid default for plot semantics.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd
import torch

from .feature_io import load_items, upsert_features

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s - %(message)s")
logger = logging.getLogger("sbert_embeddings")


def _build_text(row: pd.Series) -> str:
    title = row.get("title") or ""
    title_alt = row.get("title_alt") or ""
    overview = row.get("overview") or ""
    genres = row.get("genres") or []
    tags = row.get("tags") or []
    if not isinstance(genres, list):
        genres = []
    if not isinstance(tags, list):
        tags = []
    parts = [title]
    if title_alt and title_alt != title:
        parts.append(title_alt)
    if genres:
        parts.append("Genres: " + ", ".join(genres))
    if tags:
        parts.append("Tags: " + ", ".join(tags[:20]))
    if overview:
        parts.append(overview)
    return ". ".join(p for p in parts if p).strip()


def run(
    items_path: Path,
    features_path: Path,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    batch_size: int = 64,
    device: str = "auto",
) -> None:
    from sentence_transformers import SentenceTransformer  # local import for optional dep

    items = load_items(items_path)
    logger.info("loaded %d items", len(items))

    texts = items.apply(_build_text, axis=1).tolist()
    text_lengths = [len(t) for t in texts]

    resolved_device = device
    if device == "auto":
        if torch.cuda.is_available():
            resolved_device = "cuda"
        elif torch.backends.mps.is_available():
            resolved_device = "mps"
        else:
            resolved_device = "cpu"
    logger.info("encoding with %s on %s", model_name, resolved_device)

    model = SentenceTransformer(model_name, device=resolved_device)
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    upsert_features(
        features_path,
        column_updates=[
            ("sbert_embedding", [e.astype("float32").tolist() for e in embeddings]),
            ("text_length", text_lengths),
        ],
        keyed_by=items["item_id"],
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", type=Path, default=Path("data/feature_store/items.parquet"))
    ap.add_argument("--out", type=Path, default=Path("data/feature_store/item_features.parquet"))
    ap.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()
    run(args.items, args.out, args.model, args.batch_size, args.device)


if __name__ == "__main__":
    main()
