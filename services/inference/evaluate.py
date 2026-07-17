"""Ranking evaluation for the serving scorers — the "is it any good?" harness.

Training reports RMSE, but nobody experiences RMSE: users see an ordered list.
A model can improve RMSE and still rank badly, so this measures ranking directly
via standard leave-one-out evaluation:

  1. Take each sampled user's most recent *liked* item (rating >= LIKE_THRESHOLD)
     as the held-out target.
  2. Build a candidate set of that target plus N_NEGATIVES items the user never
     rated.
  3. Score the candidates through the real serving adapter (not a reimplementation),
     and see where the target lands.

Reported metrics, over the sampled users:
    NDCG@10   position-discounted credit for ranking the target highly
    HR@10     fraction of users whose target made the top 10
    MRR       mean reciprocal rank of the target

Two user modes, because they answer different questions:
    lookup  — user_id exists in the trained user table (a MovieLens user)
    foldin  — user_id is unknown, so the vector is folded in from history. This
              is what every mommy-milk-me user gets, since cinerec never trained
              on them. Judge production quality by this number, not `lookup`.

Baselines are included because a ranking metric in isolation is unreadable:
`popularity` ranks by interaction count, `random` shuffles. A model that cannot
beat popularity is not earning its complexity.

Usage:
    python -m services.inference.evaluate --model two_tower --mode foldin
    python -m services.inference.evaluate --model two_tower --mode lookup --users 3000
    python -m services.inference.evaluate --ensemble
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import random
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from .config import settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s - %(message)s")
logger = logging.getLogger("evaluate")

LIKE_THRESHOLD = 4.0
N_NEGATIVES = 99
K = 10


def _load_interactions() -> pd.DataFrame:
    path = settings.feature_store_dir / "interactions.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing; build the feature store first")
    return pd.read_parquet(path, columns=["user_id", "item_id", "value", "timestamp"])


def _dcg_at_k(rank: Optional[int], k: int = K) -> float:
    """Binary-relevance DCG with a single relevant item at 1-based `rank`."""
    if rank is None or rank > k:
        return 0.0
    return 1.0 / math.log2(rank + 1)


def _rank_of(target: str, ordered: Sequence[str]) -> Optional[int]:
    for i, item in enumerate(ordered, start=1):
        if item == target:
            return i
    return None


def build_cases(
    df: pd.DataFrame,
    n_users: int,
    seed: int = 0,
    negatives: str = "popularity",
) -> List[dict]:
    """Leave-one-out cases: (user, history, target, negatives).

    `negatives` controls how the 99 distractors are drawn, and it decides what
    the metric actually means:

      uniform     — drawn evenly across the catalog. Looks neutral, but is not:
                    the held-out target is something the user chose to watch, so
                    it is popular, while uniform draws are mostly long tail. A
                    popularity ranker then wins by default (measured: ndcg@10
                    0.74 popularity vs 0.50 for the model) and the metric mostly
                    reports "was the target popular", not ranking skill.
      popularity  — drawn in proportion to interaction count, so distractors are
                    about as popular as the target. This neutralizes the
                    confound and forces the metric to measure preference
                    matching. Default, and the number worth trusting.
    """
    rng = random.Random(seed)
    item_counts = df["item_id"].value_counts()
    all_items = item_counts.index.tolist()

    if negatives == "popularity":
        weights = item_counts.to_numpy(dtype=float)
        # Sampling ∝ count would make blockbusters nearly every draw; the sqrt
        # keeps distractors in the target's popularity neighbourhood while
        # retaining catalog variety.
        weights = np.sqrt(weights)
        weights = weights / weights.sum()
        nprng = np.random.default_rng(seed)
        # One large pool up front — per-case weighted draws would dominate runtime.
        neg_pool = nprng.choice(
            len(all_items), size=max(n_users * (N_NEGATIVES + 40), 10_000), p=weights
        )
        neg_pool = [all_items[i] for i in neg_pool]
    else:
        neg_pool = None

    # Only users with enough history to both fold in and hold out.
    counts = df["user_id"].value_counts()
    eligible = counts[counts >= 10].index.tolist()
    rng.shuffle(eligible)

    cases: List[dict] = []
    by_user = df[df["user_id"].isin(set(eligible[: n_users * 3]))].sort_values("timestamp")
    for user_id, grp in by_user.groupby("user_id", sort=False):
        if len(cases) >= n_users:
            break
        liked = grp[grp["value"] >= LIKE_THRESHOLD]
        if len(liked) < 2:
            continue
        target = liked.iloc[-1]["item_id"]
        # History strictly precedes the target — no leakage of the answer.
        history = [i for i in grp["item_id"].tolist() if i != target]
        if not history:
            continue
        seen = set(grp["item_id"].tolist())
        negs: List[str] = []
        guard = 0
        while len(negs) < N_NEGATIVES and guard < N_NEGATIVES * 50:
            guard += 1
            if neg_pool is not None:
                cand = neg_pool[rng.randrange(len(neg_pool))]
            else:
                cand = all_items[rng.randrange(len(all_items))]
            if cand not in seen and cand not in negs:
                negs.append(cand)
        if len(negs) < N_NEGATIVES:
            continue
        negatives_for_case = negs
        cases.append(
            {
                "user_id": str(user_id),
                "history": history[-50:],
                "target": str(target),
                "candidates": [str(target)] + negatives_for_case,
            }
        )
    return cases


def _popularity_map(df: pd.DataFrame) -> Dict[str, int]:
    return df["item_id"].value_counts().to_dict()


def evaluate(
    score_fn,
    cases: List[dict],
    label: str,
) -> dict:
    ndcg_total = 0.0
    hits = 0
    rr_total = 0.0
    scored = 0

    for case in cases:
        ordered = score_fn(case)
        if not ordered:
            continue
        rank = _rank_of(case["target"], ordered)
        ndcg_total += _dcg_at_k(rank)  # IDCG is 1.0 for a single relevant item
        if rank is not None and rank <= K:
            hits += 1
        if rank is not None:
            rr_total += 1.0 / rank
        scored += 1

    n = max(scored, 1)
    result = {
        "label": label,
        "users": scored,
        f"ndcg@{K}": round(ndcg_total / n, 4),
        f"hr@{K}": round(hits / n, 4),
        "mrr": round(rr_total / n, 4),
    }
    logger.info(
        "%-22s users=%-5d ndcg@%d=%.4f  hr@%d=%.4f  mrr=%.4f",
        label, scored, K, result[f"ndcg@{K}"], K, result[f"hr@{K}"], result["mrr"],
    )
    return result


def _load_scorer(model_name: str):
    """Load through the real registry so we measure what actually serves."""
    from .registry import registry

    if registry.is_stub(model_name):
        raise SystemExit(
            f"{model_name} is a stub (untrained) — evaluating it would measure "
            f"hash noise. Train it first: python -m services.inference.train "
            f"--models {model_name}"
        )
    return registry


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="two_tower")
    ap.add_argument("--ensemble", action="store_true", help="evaluate the blended ensemble")
    ap.add_argument("--mode", choices=["foldin", "lookup"], default="foldin")
    ap.add_argument("--users", type=int, default=1000)
    ap.add_argument(
        "--negatives",
        choices=["popularity", "uniform"],
        default="popularity",
        help="how distractors are drawn; see build_cases (default corrects the "
        "popularity confound that flatters a popularity ranker)",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=None, help="write results JSON here")
    args = ap.parse_args()

    df = _load_interactions()
    logger.info("loaded %d interactions", len(df))
    cases = build_cases(df, args.users, args.seed, args.negatives)
    logger.info(
        "built %d evaluation cases (%d %s-sampled negatives each)",
        len(cases), N_NEGATIVES, args.negatives,
    )
    if not cases:
        raise SystemExit("no eligible users; is the feature store populated?")

    results = []

    # --- baselines ---------------------------------------------------
    pop = _popularity_map(df)
    results.append(
        evaluate(
            lambda c: sorted(c["candidates"], key=lambda i: pop.get(i, 0), reverse=True),
            cases,
            "baseline:popularity",
        )
    )
    rng = random.Random(args.seed)
    results.append(
        evaluate(
            lambda c: rng.sample(c["candidates"], len(c["candidates"])),
            cases,
            "baseline:random",
        )
    )

    # --- model -------------------------------------------------------
    if args.ensemble:
        from .ensemble import ensemble

        live = ensemble.live_models()
        if not live:
            raise SystemExit("no trained models; nothing to evaluate")
        label = f"ensemble({'+'.join(live)}):{args.mode}"

        def score_fn(c):
            uid = c["user_id"] if args.mode == "lookup" else f"eval-unknown:{c['user_id']}"
            resp = ensemble.score(
                c["candidates"], user_id=uid, watch_history=c["history"]
            )
            return [i.item_id for i in resp.items]
    else:
        registry = _load_scorer(args.model)
        label = f"{args.model}:{args.mode}"

        def score_fn(c):
            uid = c["user_id"] if args.mode == "lookup" else f"eval-unknown:{c['user_id']}"
            scored = registry.score(
                args.model, c["candidates"], user_id=uid, watch_history=c["history"]
            )
            return [i.item_id for i in scored]

    results.append(evaluate(score_fn, cases, label))

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(results, indent=2))
        logger.info("wrote %s", args.out)


if __name__ == "__main__":
    main()
