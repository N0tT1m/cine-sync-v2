"""Offline ranking evaluation for trained models.

Runs leave-last-N evaluation per user, sampling random negatives, and
computes NDCG@10, Recall@50, Hit@10, MRR. Writes results to
`models/<servable_name>/eval.json`.

Usage:
    ./venv/bin/python -m src.training.eval_harness --models two_tower,sequential
    ./venv/bin/python -m src.training.eval_harness --all

Currently supports the unified two_tower and sequential classes. Adding a new
model = adding an entry to SCORERS that maps a model kind/class to a callable
returning per-candidate scores.
"""
from __future__ import annotations

import argparse
import importlib
import json
import logging
import math
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import yaml

logger = logging.getLogger("eval")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")


# ----------------------------------------------------------------------------
# Per-model scoring functions
# ----------------------------------------------------------------------------

def _score_two_tower(
    model,
    user_id: int,
    history: List[int],
    candidates: List[int],
    device: torch.device,
) -> torch.Tensor:
    user_t = torch.tensor([user_id], dtype=torch.long, device=device)
    cand_t = torch.tensor(candidates, dtype=torch.long, device=device)
    with torch.no_grad():
        user_emb = model.encode_user(user_ids=user_t)             # (1, D)
        item_emb = model.encode_item(item_ids=cand_t)             # (C, D)
        return (user_emb @ item_emb.T).squeeze(0)                  # (C,)


def _score_sequential(
    model,
    user_id: int,
    history: List[int],
    candidates: List[int],
    device: torch.device,
) -> torch.Tensor:
    if not history:
        # Cold user — return uniform scores so they don't dominate metrics.
        return torch.zeros(len(candidates), device=device)
    max_len = getattr(model, "max_seq_len", 200)
    seq_ids = history[-max_len:]
    seq_t = torch.tensor([seq_ids], dtype=torch.long, device=device)
    cand_t = torch.tensor(candidates, dtype=torch.long, device=device)
    with torch.no_grad():
        hidden = model.encode(seq_t)[:, -1, :]                     # (1, D)
        cand_emb = model.item_embedding(cand_t)                    # (C, D)
        return (hidden @ cand_emb.T).squeeze(0)                    # (C,)


SCORERS: Dict[str, Callable] = {
    "two_tower_collaborative": _score_two_tower,
    "two_tower_sbert": _score_two_tower,
    "sasrec": _score_sequential,
    "sequential": _score_sequential,
}


# ----------------------------------------------------------------------------
# Eval harness core
# ----------------------------------------------------------------------------

def _load_model_from_manifest(manifest_path: Path) -> Tuple[object, dict]:
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
    return model, manifest


def _build_user_histories(
    data_dir: Path, max_users: int = 10000
) -> Tuple[Dict[int, List[Tuple[int, int]]], int]:
    """Group movie+TV interactions by user, sorted by timestamp.

    Returns (user_id -> [(item_id, timestamp), ...], num_items_seen).
    """
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.training.train_all_models import RealDataLoader

    rdl = RealDataLoader(data_dir)
    histories: Dict[int, List[Tuple[int, int]]] = defaultdict(list)
    for record in (rdl.movie_data or []) + (rdl.tv_data or []):
        u = record.get("user_id")
        i = record.get("item_id")
        ts = record.get("timestamp", 0)
        if u is None or i is None:
            continue
        histories[u].append((i, ts))

    for u in histories:
        histories[u].sort(key=lambda x: x[1])

    if len(histories) > max_users:
        sampled = random.Random(42).sample(list(histories.keys()), max_users)
        histories = {u: histories[u] for u in sampled}

    item_universe_size = len(rdl.item_id_map)
    return histories, item_universe_size


def _per_user_metrics(rank: int, k_hit: int = 10, k_recall: int = 50) -> Dict[str, float]:
    """Standard ranking metrics for a single positive at the given rank (1-indexed)."""
    return {
        f"hit@{k_hit}": 1.0 if rank <= k_hit else 0.0,
        f"ndcg@{k_hit}": (1.0 / math.log2(rank + 1)) if rank <= k_hit else 0.0,
        f"recall@{k_recall}": 1.0 if rank <= k_recall else 0.0,
        "mrr": 1.0 / rank,
    }


def evaluate_model(
    manifest_path: Path,
    histories: Dict[int, List[Tuple[int, int]]],
    item_universe: int,
    num_negatives: int = 100,
    max_users: int = 5000,
    device: Optional[torch.device] = None,
) -> Dict[str, float]:
    device = device or torch.device("cpu")
    model, manifest = _load_model_from_manifest(manifest_path)
    model.to(device)
    kind = manifest.get("kind") or ""
    scorer = SCORERS.get(kind)
    if scorer is None:
        raise ValueError(f"no eval scorer for kind={kind!r}; add to SCORERS")

    rng = random.Random(42)
    user_ids = list(histories.keys())
    if len(user_ids) > max_users:
        user_ids = rng.sample(user_ids, max_users)

    agg: Dict[str, List[float]] = defaultdict(list)
    skipped = 0
    for u in user_ids:
        seq = histories[u]
        if len(seq) < 2:
            skipped += 1
            continue
        history_items = [i for i, _ in seq[:-1]]
        target = seq[-1][0]
        seen = set(history_items) | {target}

        # Sample negatives from the item universe excluding seen items.
        negs: List[int] = []
        attempts = 0
        while len(negs) < num_negatives and attempts < num_negatives * 4:
            cand = rng.randint(0, max(0, item_universe - 1))
            if cand not in seen:
                negs.append(cand)
                seen.add(cand)
            attempts += 1
        if len(negs) < num_negatives:
            skipped += 1
            continue

        candidates = [target] + negs
        scores = scorer(model, u, history_items, candidates, device)
        # rank of target (1-indexed) = number of negatives strictly higher than it + 1
        target_score = scores[0].item()
        rank = int((scores[1:] > target_score).sum().item()) + 1

        per_user = _per_user_metrics(rank)
        for k, v in per_user.items():
            agg[k].append(v)

    final = {k: float(np.mean(v)) for k, v in agg.items()}
    final["users_evaluated"] = len(next(iter(agg.values()))) if agg else 0
    final["users_skipped"] = skipped
    final["num_negatives"] = num_negatives
    final["kind"] = kind
    return final


def discover_servable_models(models_dir: Path) -> Dict[str, Path]:
    out: Dict[str, Path] = {}
    if not models_dir.exists():
        return out
    for sub in models_dir.iterdir():
        manifest = sub / "manifest.yaml"
        if manifest.exists():
            out[sub.name] = manifest
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline ranking eval harness")
    parser.add_argument("--models", type=str, help="Comma-separated servable model names")
    parser.add_argument("--all", action="store_true", help="Evaluate every model with a manifest")
    parser.add_argument(
        "--data-dir",
        type=str,
        default=str(Path(__file__).resolve().parents[2] / "data"),
    )
    parser.add_argument(
        "--models-dir",
        type=str,
        default=str(Path(__file__).resolve().parents[2] / "models"),
    )
    parser.add_argument("--num-negatives", type=int, default=100)
    parser.add_argument("--max-users", type=int, default=5000)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    models_dir = Path(args.models_dir)
    available = discover_servable_models(models_dir)
    if not available:
        logger.error("no manifests found under %s", models_dir)
        sys.exit(1)

    if args.all:
        targets = list(available.keys())
    elif args.models:
        targets = [m.strip() for m in args.models.split(",")]
    else:
        targets = list(available.keys())

    logger.info("loading user histories from %s", args.data_dir)
    histories, item_universe = _build_user_histories(Path(args.data_dir))
    logger.info(
        "loaded %d users, item_universe=%d",
        len(histories),
        item_universe,
    )

    for name in targets:
        manifest_path = available.get(name)
        if manifest_path is None:
            logger.warning("no manifest for %s; skipping", name)
            continue
        logger.info("evaluating %s", name)
        try:
            metrics = evaluate_model(
                manifest_path=manifest_path,
                histories=histories,
                item_universe=item_universe,
                num_negatives=args.num_negatives,
                max_users=args.max_users,
                device=device,
            )
        except Exception as e:
            logger.exception("eval failed for %s: %s", name, e)
            continue
        eval_path = manifest_path.parent / "eval.json"
        eval_path.write_text(json.dumps(metrics, indent=2))
        logger.info(
            "%s: hit@10=%.3f ndcg@10=%.3f recall@50=%.3f mrr=%.3f -> %s",
            name,
            metrics.get("hit@10", 0),
            metrics.get("ndcg@10", 0),
            metrics.get("recall@50", 0),
            metrics.get("mrr", 0),
            eval_path,
        )


if __name__ == "__main__":
    main()
