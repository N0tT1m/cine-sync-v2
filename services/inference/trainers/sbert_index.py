"""sbert_two_tower 'trainer': build a semantic content index from the feature store.

No gradient training — the item tower is the frozen SBERT embedding produced by
`src.enrichment.sbert_embeddings`. The user tower is a rating-weighted mean of
the items a user liked (a content profile). Both are L2-normalized so the
served dot product is a cosine similarity in [-1, 1].

Emits (under models/sbert_two_tower/):
    item_emb.npy / item_ids.json   — content vectors keyed by canonical item_id
    user_emb.npy / user_ids.json   — per-user content profiles
    manifest.yaml                  — wires TwoTowerScorer
"""
from __future__ import annotations

import logging

import numpy as np

from . import _common as C

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")
logger = logging.getLogger("trainers.sbert")

NAME = "sbert_two_tower"
KIND = "two_tower_sbert"
POSITIVE_RATING = 3.5  # only profile from items the user actually liked


def _embedding_column(feats) -> str:
    for col in ("sbert_embedding", "embedding", "text_embedding"):
        if col in feats.columns:
            return col
    raise KeyError(
        f"no embedding column in item_features.parquet (have {list(feats.columns)})"
    )


def run() -> None:
    feats = C.load_item_features()
    emb_col = _embedding_column(feats)
    feats = feats[feats[emb_col].notna()].reset_index(drop=True)

    item_ids = [str(x) for x in feats["item_id"].tolist()]
    item_emb = C.l2_normalize(np.vstack([np.asarray(v, dtype=np.float32) for v in feats[emb_col]]))
    dim = item_emb.shape[1]
    row_of = {iid: i for i, iid in enumerate(item_ids)}
    logger.info("item tower: %d items, dim=%d", len(item_ids), dim)

    # user profiles: rating-weighted mean of liked items' vectors
    inter = C.load_interactions()
    inter = inter[inter["value"] >= POSITIVE_RATING]
    user_vectors: dict[str, np.ndarray] = {}
    weight_sums: dict[str, float] = {}
    for user_id, item_id, value in zip(inter["user_id"], inter["item_id"], inter["value"]):
        r = row_of.get(str(item_id))
        if r is None:
            continue
        w = float(value)
        uid = str(user_id)
        if uid not in user_vectors:
            user_vectors[uid] = np.zeros(dim, dtype=np.float32)
            weight_sums[uid] = 0.0
        user_vectors[uid] += w * item_emb[r]
        weight_sums[uid] += w

    user_ids = list(user_vectors.keys())
    if user_ids:
        user_emb = C.l2_normalize(
            np.vstack([user_vectors[u] / max(weight_sums[u], 1e-6) for u in user_ids])
        )
    else:
        user_emb = np.zeros((0, dim), dtype=np.float32)
    logger.info("user tower: %d profiles", len(user_ids))

    outdir = C.model_outdir(NAME)
    scorer_kwargs = {}
    scorer_kwargs.update(C.save_embeddings(outdir, "item", item_emb, item_ids))
    if user_ids:
        scorer_kwargs.update(C.save_embeddings(outdir, "user", user_emb, user_ids))

    C.write_manifest(
        outdir,
        name=NAME,
        kind=KIND,
        framework="pytorch",
        artifact=None,  # embedding-only; no torch module to load
        loader_config={
            "class_path": "src.models.adapters.null_model.NullModel",
            "init_kwargs": {},
            "scorer_adapter": "src.models.adapters.TwoTowerScorer",
            "scorer_kwargs": scorer_kwargs,
        },
        metrics={"num_items": len(item_ids), "num_users": len(user_ids), "dim": dim},
        notes="frozen SBERT item tower; rating-weighted user content profiles",
    )
    logger.info("sbert_two_tower index complete")


if __name__ == "__main__":
    run()
