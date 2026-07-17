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

Env knobs: TT_DIM, TT_EPOCHS, TT_BATCH, TT_LR, TT_WD, TT_MIN_INTERACTIONS,
TT_VAL_FRACTION.

On weight decay: TT_WD defaults to 0. Adam applies L2 as a gradient term, which
its per-parameter scaling then amplifies, so even 1e-5 shrinks the embeddings
faster than 32M ratings can push signal into them. The dot product collapses and
the bias terms fit the global mean — training looks converged while the model has
learned nothing. Measured on a 3.1M-rating holdout (val RMSE, rating units):

    wd=1e-5  0.949   (vs 1.047 for predicting the mean — near-useless)
    wd=1e-6  0.815
    wd=0     0.802

Raise TT_WD only alongside a val_rmse check.
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
    lr = float(os.getenv("TT_LR", "0.005"))
    wd = float(os.getenv("TT_WD", "0.0"))  # see module docstring before raising
    val_fraction = float(os.getenv("TT_VAL_FRACTION", "0.05"))
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

    # Hold out a validation slice. Train MSE alone cannot distinguish a model
    # that learned preferences from one that only learned the global mean, so
    # without this there is no signal for whether the artifact is worth serving.
    perm = torch.randperm(len(y), generator=torch.Generator().manual_seed(0))
    u_idx, i_idx, y = u_idx[perm], i_idx[perm], y[perm]
    n_val = int(len(y) * val_fraction)
    if n_val > 0:
        vu, vi, vy = u_idx[:n_val].to(device), i_idx[:n_val].to(device), y[:n_val].to(device)
        tu, ti, ty = u_idx[n_val:], i_idx[n_val:], y[n_val:]
    else:
        vu = vi = vy = None
        tu, ti, ty = u_idx, i_idx, y

    # What predicting the training mean would score — the bar any useful model
    # must clear, and the number a collapsed model lands on.
    baseline_rmse = None
    if vy is not None:
        baseline_rmse = (
            torch.sqrt(((vy - ty.mean().to(device)) ** 2).mean()).item() * RATING_SCALE
        )
        logger.info("baseline (predict-mean) val_rmse=%.4f rating units", baseline_rmse)

    loader = DataLoader(TensorDataset(tu, ti, ty), batch_size=batch, shuffle=True)

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
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    loss_fn = nn.MSELoss()
    logger.info("lr=%s weight_decay=%s epochs=%d batch=%d", lr, wd, epochs, batch)

    def val_rmse() -> float | None:
        if vy is None:
            return None
        model.eval()
        with torch.no_grad():
            total, n = 0.0, 0
            for s in range(0, len(vy), 65536):  # chunked: the val slice can be ~1.6M rows
                p = model(vu[s : s + 65536], vi[s : s + 65536])
                total += ((p - vy[s : s + 65536]) ** 2).sum().item()
                n += len(p)
        return (total / max(n, 1)) ** 0.5 * RATING_SCALE

    # Unregularized MF on dense ratings starts overfitting almost immediately
    # (val_rmse rose every epoch after the first on ML-32M), so stop once it
    # stops improving rather than burning epochs that make the model worse.
    patience = int(os.getenv("TT_PATIENCE", "2"))
    since_improved = 0

    last = 0.0
    best_val = None
    best_state = None
    for ep in range(epochs):
        model.train()
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
        v = val_rmse()
        if v is None:
            logger.info("epoch %d/%d  mse=%.5f", ep + 1, epochs, last)
            continue
        logger.info("epoch %d/%d  mse=%.5f  val_rmse=%.4f", ep + 1, epochs, last, v)
        # Keep the best epoch, not the last one.
        if best_val is None or v < best_val:
            best_val = v
            best_state = {k: t.detach().clone() for k, t in model.state_dict().items()}
            since_improved = 0
        else:
            since_improved += 1
            if since_improved >= patience:
                logger.info("early stop at epoch %d (no val gain in %d)", ep + 1, patience)
                break

    if best_state is not None:
        model.load_state_dict(best_state)
        logger.info("restored best epoch (val_rmse=%.4f)", best_val)

    if best_val is not None and baseline_rmse is not None and best_val >= baseline_rmse:
        # Serving this would rank by noise while looking healthy.
        raise ValueError(
            f"two_tower did not beat the predict-mean baseline "
            f"(val_rmse={best_val:.4f} >= baseline={baseline_rmse:.4f}); "
            f"refusing to write an artifact. Check TT_WD/TT_LR."
        )

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
        metrics={
            "train_mse": round(last, 5),
            # val_rmse and baseline_rmse are in rating units (0.5-5) and directly
            # comparable: val_rmse must be below baseline_rmse for the model to
            # be worth serving over a popularity list.
            "val_rmse": round(best_val, 4) if best_val is not None else None,
            "baseline_rmse": round(baseline_rmse, 4) if baseline_rmse is not None else None,
            "num_users": len(user_ids),
            "num_items": len(item_ids),
            "dim": dim,
            "lr": lr,
            "weight_decay": wd,
        },
        notes="collaborative MF two-tower; embeddings L2-normalized for cosine scoring",
    )
    logger.info("two_tower training complete")


if __name__ == "__main__":
    run()
