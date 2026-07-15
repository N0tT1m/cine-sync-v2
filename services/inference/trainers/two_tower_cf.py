"""two_tower 'trainer': collaborative matrix-factorization over interactions.

Learns a user embedding table and an item embedding table from the feature
store's interactions.parquet, predicting the (scaled) rating as the dot product
of the two towers. After training, the L2-normalized item/user embeddings are
exported keyed by canonical feature-store ids so TwoTowerScorer serves them
directly.

Emits (under models/two_tower/):
    item_emb.npy / item_ids.json   — learned item vectors, keyed by item_id
    user_emb.npy / user_ids.json   — learned user vectors, keyed by user_id
    weights.pt                     — full state dict (kept for reproducibility)
    manifest.yaml                  — wires TwoTowerScorer

Env knobs: TT_DIM, TT_EPOCHS, TT_BATCH, TT_LR, TT_MIN_INTERACTIONS.
"""
from __future__ import annotations

import logging
import os

import numpy as np

from . import _common as C

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")
logger = logging.getLogger("trainers.two_tower")

NAME = "two_tower"
KIND = "two_tower"
RATING_SCALE = 5.0  # ratings are 0.5..5 -> scaled to (0,1] for a sigmoid target


def run() -> None:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    dim = int(os.getenv("TT_DIM", "64"))
    epochs = int(os.getenv("TT_EPOCHS", "8"))
    batch = int(os.getenv("TT_BATCH", "4096"))
    lr = float(os.getenv("TT_LR", "0.01"))
    min_inter = int(os.getenv("TT_MIN_INTERACTIONS", "5"))
    device = C.pick_device()

    inter = C.load_interactions()[["user_id", "item_id", "value"]].dropna()
    inter["user_id"] = inter["user_id"].astype(str)
    inter["item_id"] = inter["item_id"].astype(str)

    # prune sparse users/items so embeddings see enough signal
    for _ in range(2):
        uc = inter["user_id"].value_counts()
        ic = inter["item_id"].value_counts()
        inter = inter[
            inter["user_id"].isin(uc[uc >= min_inter].index)
            & inter["item_id"].isin(ic[ic >= min_inter].index)
        ]
    if inter.empty:
        raise ValueError(f"no interactions left after min_interactions={min_inter} pruning")

    user_ids = sorted(inter["user_id"].unique())
    item_ids = sorted(inter["item_id"].unique())
    u2i = {u: i for i, u in enumerate(user_ids)}
    i2i = {it: i for i, it in enumerate(item_ids)}
    logger.info("training on %d interactions | %d users | %d items | dim=%d",
                len(inter), len(user_ids), len(item_ids), dim)

    u_idx = torch.tensor(inter["user_id"].map(u2i).to_numpy(), dtype=torch.long)
    i_idx = torch.tensor(inter["item_id"].map(i2i).to_numpy(), dtype=torch.long)
    y = torch.tensor((inter["value"].to_numpy() / RATING_SCALE).clip(0, 1), dtype=torch.float32)

    loader = DataLoader(TensorDataset(u_idx, i_idx, y), batch_size=batch, shuffle=True)

    class TwoTowerMF(nn.Module):
        def __init__(self, n_users: int, n_items: int, d: int):
            super().__init__()
            self.user_emb = nn.Embedding(n_users, d)
            self.item_emb = nn.Embedding(n_items, d)
            self.user_bias = nn.Embedding(n_users, 1)
            self.item_bias = nn.Embedding(n_items, 1)
            nn.init.normal_(self.user_emb.weight, std=0.05)
            nn.init.normal_(self.item_emb.weight, std=0.05)
            nn.init.zeros_(self.user_bias.weight)
            nn.init.zeros_(self.item_bias.weight)

        def forward(self, u, it):
            dot = (self.user_emb(u) * self.item_emb(it)).sum(-1)
            return torch.sigmoid(dot + self.user_bias(u).squeeze(-1) + self.item_bias(it).squeeze(-1))

    model = TwoTowerMF(len(user_ids), len(item_ids), dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    loss_fn = nn.MSELoss()

    model.train()
    last = 0.0
    for ep in range(epochs):
        total, n = 0.0, 0
        for ub, ib, yb in loader:
            ub, ib, yb = ub.to(device), ib.to(device), yb.to(device)
            opt.zero_grad()
            pred = model(ub, ib)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
            total += loss.item() * len(yb)
            n += len(yb)
        last = total / max(n, 1)
        logger.info("epoch %d/%d  mse=%.5f", ep + 1, epochs, last)

    model.eval()
    with torch.no_grad():
        item_vecs = C.l2_normalize(model.item_emb.weight.detach().cpu().numpy())
        user_vecs = C.l2_normalize(model.user_emb.weight.detach().cpu().numpy())

    outdir = C.model_outdir(NAME)
    scorer_kwargs = {}
    scorer_kwargs.update(C.save_embeddings(outdir, "item", item_vecs, item_ids))
    scorer_kwargs.update(C.save_embeddings(outdir, "user", user_vecs, user_ids))
    torch.save(model.state_dict(), outdir / "weights.pt")

    C.write_manifest(
        outdir,
        name=NAME,
        kind=KIND,
        framework="pytorch",
        artifact=None,  # served from exported embeddings, not the raw state dict
        loader_config={
            "class_path": "src.models.adapters.null_model.NullModel",
            "init_kwargs": {},
            "scorer_adapter": "src.models.adapters.TwoTowerScorer",
            "scorer_kwargs": scorer_kwargs,
        },
        metrics={"train_mse": round(last, 5), "num_users": len(user_ids),
                 "num_items": len(item_ids), "dim": dim},
        notes="collaborative MF two-tower; embeddings L2-normalized for cosine scoring",
    )
    logger.info("two_tower training complete")


if __name__ == "__main__":
    run()
