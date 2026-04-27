"""Training orchestrator — glue between the feature store and `models/<name>/`.

This is a thin wrapper:
  1. Validate that data/feature_store/*.parquet exists and has the columns the
     trainers expect.
  2. Invoke the specific trainer for each enabled model.
  3. Write models/<name>/manifest.yaml so the inference registry can load it.
  4. (Optional) POST /admin/reload/<name> to a running inference service so it
     picks up the new artifact without a restart.

The actual training loops live in src/training/train_all_models.py and the
per-model modules under src/models/. We don't reimplement them here.

Usage:
    python -m services.inference.train --models ncf,sequential,sbert_two_tower
    python -m services.inference.train --all
    python -m services.inference.train --all --hot-reload-url http://localhost:8900
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import httpx
import yaml

from .config import settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s - %(message)s")
logger = logging.getLogger("orchestrator")


@dataclass
class TrainSpec:
    name: str                      # matches registry key + models/<name>/ dir
    kind: str                      # descriptive; goes into manifest
    entry: List[str]               # command to run (module invocation)
    requires_embeddings: bool = False
    requires_graph: bool = False
    epochs: int = 10
    notes: str = ""


# Each TrainSpec maps a registry name → the trainer script to invoke. Entries
# point at real modules under src/ — the orchestrator's job is to run them in
# the right order, not to reimplement training.
TRAIN_SPECS: Dict[str, TrainSpec] = {
    "hybrid": TrainSpec(
        name="hybrid",
        kind="hybrid_cf_content",
        entry=["python", "-m", "src.models.hybrid.train"],
        epochs=10,
    ),
    "ncf": TrainSpec(
        name="ncf",
        kind="neural_collaborative_filtering",
        entry=["python", "-m", "src.models.collaborative.train"],
        epochs=15,
    ),
    "sequential": TrainSpec(
        name="sequential",
        kind="sasrec",
        entry=["python", "-m", "src.models.sequential.train"],
        epochs=20,
    ),
    "bert4rec": TrainSpec(
        name="bert4rec",
        kind="bert4rec",
        entry=["python", "-m", "src.models.advanced.bert4rec_recommender"],
        epochs=20,
    ),
    "sbert_two_tower": TrainSpec(
        name="sbert_two_tower",
        kind="two_tower_sbert",
        entry=["python", "-m", "src.models.advanced.sentence_bert_two_tower"],
        epochs=15,
        requires_embeddings=True,
    ),
    "graphsage": TrainSpec(
        name="graphsage",
        kind="graph_neural_network",
        entry=["python", "-m", "src.models.advanced.graphsage_recommender"],
        epochs=20,
        requires_graph=True,
    ),
    "contrastive": TrainSpec(
        name="contrastive",
        kind="contrastive_learning",
        entry=["python", "-m", "src.models.unified.contrastive_learning"],
        requires_embeddings=True,
    ),
    "multimodal": TrainSpec(
        name="multimodal",
        kind="multimodal_transformer",
        entry=["python", "-m", "src.models.multimodal_content_understanding"],
        requires_embeddings=True,
    ),
    "vae": TrainSpec(
        name="vae",
        kind="variational_autoencoder",
        entry=["python", "-m", "src.models.advanced.variational_autoencoder"],
        epochs=30,
    ),
}


def check_feature_store() -> bool:
    need = ["items.parquet", "interactions.parquet"]
    missing = [n for n in need if not (settings.feature_store_dir / n).exists()]
    if missing:
        logger.error(
            "feature store missing files: %s. "
            "Run: python -m data.feature_store.build_feature_store --out %s",
            missing, settings.feature_store_dir,
        )
        return False
    return True


def check_embeddings() -> bool:
    path = settings.feature_store_dir / "item_features.parquet"
    if not path.exists():
        logger.warning(
            "item_features.parquet missing. "
            "Run: python -m src.enrichment.sbert_embeddings"
        )
        return False
    return True


def check_graph() -> bool:
    path = settings.feature_store_dir / "graph_edges.parquet"
    if not path.exists():
        logger.warning("graph_edges.parquet missing; GraphSAGE will skip")
        return False
    return True


def write_manifest(spec: TrainSpec, metrics: Optional[dict]) -> Path:
    model_dir = settings.models_dir / spec.name
    model_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "name": spec.name,
        "kind": spec.kind,
        "trained_at": datetime.utcnow().isoformat(),
        "epochs": spec.epochs,
        "notes": spec.notes,
        "metrics": metrics or {},
        "artifacts": {
            "weights": "weights.pt",
            "encoders": "encoders.pkl",
        },
    }
    path = model_dir / "manifest.yaml"
    with open(path, "w") as f:
        yaml.safe_dump(manifest, f, sort_keys=False)
    logger.info("wrote %s", path)
    return path


def run_trainer(spec: TrainSpec, extra_env: Optional[Dict[str, str]] = None) -> bool:
    env = os.environ.copy()
    env.update(extra_env or {})
    env.setdefault(
        "CINESYNC_FEATURE_STORE", str(settings.feature_store_dir)
    )
    env.setdefault("CINESYNC_MODEL_OUTDIR", str(settings.models_dir / spec.name))

    logger.info("===== training %s =====", spec.name)
    start = time.monotonic()
    try:
        result = subprocess.run(
            spec.entry, env=env, check=False,
            stdout=sys.stdout, stderr=sys.stderr,
        )
    except FileNotFoundError as exc:
        logger.error("trainer entry missing for %s: %s", spec.name, exc)
        return False

    duration = time.monotonic() - start
    if result.returncode != 0:
        logger.error("%s failed (rc=%d) after %.1fs", spec.name, result.returncode, duration)
        return False

    logger.info("%s finished in %.1fs", spec.name, duration)
    # Prefer the trainer's own metrics file if it wrote one; otherwise empty.
    metrics_path = settings.models_dir / spec.name / "final_metrics.json"
    metrics = None
    if metrics_path.exists():
        try:
            metrics = json.loads(metrics_path.read_text())
        except json.JSONDecodeError:
            logger.warning("could not parse %s", metrics_path)
    write_manifest(spec, metrics)
    return True


def hot_reload(base_url: str, name: str) -> None:
    try:
        r = httpx.post(f"{base_url}/admin/reload/{name}", timeout=10)
        r.raise_for_status()
        logger.info("hot-reload ok: %s", r.json())
    except httpx.HTTPError as exc:
        logger.warning("hot-reload failed for %s: %s", name, exc)


def orchestrate(models: List[str], hot_reload_url: Optional[str]) -> int:
    if not check_feature_store():
        return 2

    have_embeddings = check_embeddings()
    have_graph = check_graph()

    failed: List[str] = []
    for name in models:
        spec = TRAIN_SPECS.get(name)
        if spec is None:
            logger.error("unknown model: %s (known: %s)", name, list(TRAIN_SPECS))
            failed.append(name)
            continue
        if spec.requires_embeddings and not have_embeddings:
            logger.warning("skipping %s: needs embeddings", name)
            failed.append(name)
            continue
        if spec.requires_graph and not have_graph:
            logger.warning("skipping %s: needs graph edges", name)
            failed.append(name)
            continue
        ok = run_trainer(spec)
        if not ok:
            failed.append(name)
            continue
        if hot_reload_url:
            hot_reload(hot_reload_url, name)

    if failed:
        logger.warning("failed / skipped: %s", failed)
        return 1
    logger.info("all requested models trained ok")
    return 0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="", help="comma-separated registry names")
    ap.add_argument("--all", action="store_true", help="train every spec in TRAIN_SPECS")
    ap.add_argument("--hot-reload-url", default=os.getenv("CINEREC_HOT_RELOAD_URL", ""))
    args = ap.parse_args()

    if args.all:
        models = list(TRAIN_SPECS.keys())
    else:
        models = [m.strip() for m in args.models.split(",") if m.strip()]
    if not models:
        ap.error("pass --all or --models a,b,c")

    rc = orchestrate(models, args.hot_reload_url or None)
    sys.exit(rc)


if __name__ == "__main__":
    main()
