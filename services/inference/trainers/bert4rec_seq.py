"""bert4rec 'trainer': cloze sequential training over user watch histories.

Builds per-user item-token sequences (ordered by timestamp) from the feature
store, then trains the repo's BERT4Rec with a masked-last-item objective that
mirrors how BERT4RecScorer serves: history + [MASK] -> predict the next item.

Token scheme (must match BERT4RecScorer):
    0            = pad
    1..V         = items (V = vocab size)
    num_items+1  = mask       (num_items is passed as V+1)

Emits (under models/bert4rec/):
    weights.pt          — model state dict
    item_id_map.json    — {canonical item_id: token index in 1..V}
    manifest.yaml       — wires BERT4Rec + BERT4RecScorer

Env knobs: B4R_DIM, B4R_LAYERS, B4R_HEADS, B4R_MAXLEN, B4R_EPOCHS, B4R_BATCH,
           B4R_LR, B4R_MIN_ITEM_COUNT, B4R_MAX_EXAMPLES.
"""
from __future__ import annotations

import logging
import os

from . import _common as C

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")
logger = logging.getLogger("trainers.bert4rec")

NAME = "bert4rec"
KIND = "bert4rec"


def run() -> None:
    import numpy as np
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    from src.models.advanced.bert4rec_recommender import BERT4Rec

    d_model = int(os.getenv("B4R_DIM", "64"))
    n_layers = int(os.getenv("B4R_LAYERS", "2"))
    n_heads = int(os.getenv("B4R_HEADS", "2"))
    max_len = int(os.getenv("B4R_MAXLEN", "50"))
    epochs = int(os.getenv("B4R_EPOCHS", "5"))
    batch = int(os.getenv("B4R_BATCH", "256"))
    lr = float(os.getenv("B4R_LR", "0.001"))
    min_item_count = int(os.getenv("B4R_MIN_ITEM_COUNT", "5"))
    max_examples = int(os.getenv("B4R_MAX_EXAMPLES", "500000"))
    device = C.pick_device()

    inter = C.load_interactions()[["user_id", "item_id", "timestamp"]].dropna()
    inter["user_id"] = inter["user_id"].astype(str)
    inter["item_id"] = inter["item_id"].astype(str)

    # vocab: items seen enough to learn, mapped to 1..V (0 reserved for pad)
    ic = inter["item_id"].value_counts()
    vocab_items = list(ic[ic >= min_item_count].index)
    item2tok = {it: i + 1 for i, it in enumerate(vocab_items)}
    V = len(vocab_items)
    if V < 2:
        raise ValueError(f"vocab too small ({V}) — lower B4R_MIN_ITEM_COUNT")
    inter = inter[inter["item_id"].isin(item2tok)]

    inter = inter.sort_values(["user_id", "timestamp"])
    logger.info("building sequences: %d interactions | vocab V=%d | max_len=%d",
                len(inter), V, max_len)

    # cloze examples: for each user seq, hold out trailing positions one at a time
    inputs: list[list[int]] = []
    targets: list[int] = []
    mask_tok = V + 2  # == num_items+1 with num_items=V+1
    pad_tok = 0
    holdout_per_user = int(os.getenv("B4R_HOLDOUT_PER_USER", "3"))

    for _, grp in inter.groupby("user_id", sort=False):
        toks = [item2tok[it] for it in grp["item_id"]]
        if len(toks) < 2:
            continue
        # predict each of the last `holdout_per_user` items from its prefix
        for cut in range(max(1, len(toks) - holdout_per_user), len(toks)):
            hist = toks[:cut][-(max_len - 1):]
            seq = hist + [mask_tok]
            seq = [pad_tok] * (max_len - len(seq)) + seq
            inputs.append(seq)
            targets.append(toks[cut])
        if len(inputs) >= max_examples:
            logger.info("hit B4R_MAX_EXAMPLES=%d", max_examples)
            break

    if not inputs:
        raise ValueError("no training sequences built")
    logger.info("built %d cloze examples", len(inputs))

    X = torch.tensor(inputs, dtype=torch.long)
    Y = torch.tensor(targets, dtype=torch.long)
    loader = DataLoader(TensorDataset(X, Y), batch_size=batch, shuffle=True)

    num_items = V + 1  # head covers classes 0..V; item tokens live in 1..V
    model = BERT4Rec(
        num_items=num_items, max_seq_len=max_len, d_model=d_model,
        num_heads=n_heads, num_layers=n_layers, d_ff=d_model * 4, dropout=0.1,
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    mask_pos = max_len - 1

    model.train()
    last = 0.0
    for ep in range(epochs):
        total, n = 0.0, 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            out = model(xb, apply_masking=False)
            logits = out["logits"][:, mask_pos, :]  # (B, num_items)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()
            total += loss.item() * len(yb)
            n += len(yb)
        last = total / max(n, 1)
        logger.info("epoch %d/%d  ce=%.4f", ep + 1, epochs, last)

    outdir = C.model_outdir(NAME)
    torch.save(model.state_dict(), outdir / "weights.pt")
    C.save_id_map(outdir, "item_id_map.json", vocab_items, start=1)  # tokens 1..V

    C.write_manifest(
        outdir,
        name=NAME,
        kind=KIND,
        framework="pytorch",
        artifact="weights.pt",
        loader_config={
            "class_path": "src.models.advanced.bert4rec_recommender.BERT4Rec",
            "init_kwargs": {
                "num_items": num_items, "max_seq_len": max_len, "d_model": d_model,
                "num_heads": n_heads, "num_layers": n_layers, "d_ff": d_model * 4,
                "dropout": 0.1,
            },
            "scorer_adapter": "src.models.adapters.BERT4RecScorer",
            "scorer_kwargs": {"item_id_map_path": "item_id_map.json"},
        },
        metrics={"train_ce": round(last, 4), "vocab": V, "examples": len(inputs)},
        notes="cloze masked-last-item training; tokens 1..V, mask=V+2",
    )
    logger.info("bert4rec training complete")


if __name__ == "__main__":
    run()
