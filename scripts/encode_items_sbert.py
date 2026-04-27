"""One-shot encoder: SBERT-embed every item's text and dump to feature store.

Output: data/feature_store/items_text.parquet with columns:
    item_id (int), text (str), embedding (list[float])

Run once after training data lands; re-run when item catalogue changes
(adding new items only requires encoding the deltas — see --since flag).

Usage:
    ./venv/bin/python -m scripts.encode_items_sbert
    ./venv/bin/python -m scripts.encode_items_sbert --model all-MiniLM-L6-v2 --batch-size 256

Once this file exists, the trainer's two_tower default args can be flipped to
include `user_text_dim=384, item_text_dim=384` and the dataset will join
embeddings onto each row.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")
logger = logging.getLogger("encode_sbert")


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = REPO_ROOT / "data" / "feature_store" / "items_text.parquet"


def _build_item_text_corpus(data_dir: Path) -> pd.DataFrame:
    """Walk RealDataLoader's view of the catalogue and collect (item_id, text)."""
    sys.path.insert(0, str(REPO_ROOT))
    from src.training.train_all_models import RealDataLoader

    rdl = RealDataLoader(data_dir)
    seen: dict[int, str] = {}
    for record in (rdl.movie_data or []) + (rdl.tv_data or []):
        item_id = record.get("item_id")
        if item_id is None or item_id in seen:
            continue
        # Compose text from whatever metadata fields are present.
        parts: List[str] = []
        for k in ("title", "name", "description", "overview", "genres", "tagline"):
            v = record.get(k)
            if v:
                parts.append(str(v) if not isinstance(v, list) else " ".join(map(str, v)))
        text = " | ".join(parts).strip()
        if not text:
            text = f"item_{item_id}"
        seen[item_id] = text

    df = pd.DataFrame(
        sorted(seen.items()), columns=["item_id", "text"]
    )
    logger.info("collected %d unique items for SBERT encoding", len(df))
    return df


def _encode(corpus: pd.DataFrame, model_name: str, batch_size: int) -> np.ndarray:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        logger.error(
            "sentence-transformers not installed. pip install sentence-transformers"
        )
        sys.exit(1)

    model = SentenceTransformer(model_name)
    logger.info("loaded %s (dim=%d)", model_name, model.get_sentence_embedding_dimension())
    embeddings = model.encode(
        corpus["text"].tolist(),
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    return embeddings.astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description="SBERT-encode the item catalogue")
    parser.add_argument("--data-dir", type=str, default=str(REPO_ROOT / "data"))
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT))
    parser.add_argument(
        "--model", type=str, default="sentence-transformers/all-MiniLM-L6-v2"
    )
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    corpus = _build_item_text_corpus(data_dir)
    if corpus.empty:
        logger.error("no items found under %s", data_dir)
        sys.exit(1)

    embeddings = _encode(corpus, args.model, args.batch_size)
    corpus["embedding"] = list(embeddings)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    corpus.to_parquet(out, index=False)
    logger.info(
        "wrote %d items × %d dims -> %s",
        embeddings.shape[0],
        embeddings.shape[1],
        out,
    )


if __name__ == "__main__":
    main()
