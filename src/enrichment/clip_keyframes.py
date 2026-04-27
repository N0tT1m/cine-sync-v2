"""Extract keyframes from owned media files and embed them with OpenCLIP.

For each owned item, ffmpeg samples N frames evenly, OpenCLIP embeds them, and
the mean-pooled vector becomes the item's visual style vector (512-dim).

Expects items.parquet to have an `owned` column and a way to locate the file.
File lookup strategy:
    - plex_guid column → resolve via nami-stream DB media_files table
    - explicit --media-root override for mounted Plex shares
"""
from __future__ import annotations

import argparse
import logging
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np
import pandas as pd
import torch

from .feature_io import load_items, upsert_features

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s - %(message)s")
logger = logging.getLogger("clip_keyframes")


def _ffprobe_duration(path: Path) -> Optional[float]:
    try:
        out = subprocess.check_output(
            [
                "ffprobe", "-v", "error", "-show_entries",
                "format=duration", "-of", "default=noprint_wrappers=1:nokey=1",
                str(path),
            ],
            stderr=subprocess.DEVNULL,
        )
        return float(out.decode().strip())
    except (subprocess.CalledProcessError, ValueError, FileNotFoundError):
        return None


def _extract_keyframes(path: Path, n: int, out_dir: Path) -> List[Path]:
    duration = _ffprobe_duration(path)
    if not duration or duration <= 0:
        return []
    frames: List[Path] = []
    for i in range(n):
        t = duration * (i + 1) / (n + 1)
        out = out_dir / f"kf_{i:02d}.jpg"
        try:
            subprocess.check_call(
                [
                    "ffmpeg", "-y", "-ss", f"{t:.2f}", "-i", str(path),
                    "-frames:v", "1", "-q:v", "3", str(out),
                ],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            if out.exists():
                frames.append(out)
        except subprocess.CalledProcessError:
            continue
    return frames


def _resolve_file(item_id: str, row: pd.Series, media_root: Optional[Path]) -> Optional[Path]:
    # explicit path column wins
    if "file_path" in row and row["file_path"]:
        p = Path(str(row["file_path"]))
        if p.exists():
            return p
    # scan media_root for a filename that contains the title
    if media_root and media_root.exists():
        title = str(row.get("title") or "").strip()
        if not title:
            return None
        # cheap filename match; production should use the nami-stream media_files table
        for p in media_root.rglob("*"):
            if p.is_file() and title.lower()[:12] in p.name.lower():
                return p
    return None


def run(
    items_path: Path,
    features_path: Path,
    media_root: Optional[Path],
    keyframes_per_item: int,
    model_name: str,
    pretrained: str,
    batch_size: int,
    device: str,
    limit: Optional[int],
) -> None:
    import open_clip  # local import for optional dep
    from PIL import Image

    items = load_items(items_path)
    owned = items[items["owned"] == True].reset_index(drop=True)  # noqa: E712
    if limit:
        owned = owned.head(limit)
    logger.info("owned items to embed: %d", len(owned))

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name, pretrained=pretrained, device=device
    )
    model.eval()

    embeddings: List[Optional[np.ndarray]] = []
    keyframe_counts: List[int] = []

    for _idx, row in owned.iterrows():
        path = _resolve_file(row["item_id"], row, media_root)
        if path is None:
            embeddings.append(None)
            keyframe_counts.append(0)
            continue
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            frames = _extract_keyframes(path, keyframes_per_item, td_path)
            if not frames:
                embeddings.append(None)
                keyframe_counts.append(0)
                continue
            imgs = []
            for fp in frames:
                try:
                    imgs.append(preprocess(Image.open(fp).convert("RGB")))
                except Exception as exc:
                    logger.debug("bad frame %s: %s", fp, exc)
            if not imgs:
                embeddings.append(None)
                keyframe_counts.append(0)
                continue
            batch = torch.stack(imgs).to(device)
            with torch.no_grad():
                feats = model.encode_image(batch)
                feats = feats / feats.norm(dim=-1, keepdim=True)
                mean = feats.mean(dim=0)
                mean = mean / (mean.norm() + 1e-12)
            embeddings.append(mean.cpu().numpy().astype("float32"))
            keyframe_counts.append(len(frames))
        if (_idx + 1) % 25 == 0:
            logger.info("processed %d/%d", _idx + 1, len(owned))

    # pad missing with zeros (512-dim by default for ViT-B/32) so the parquet
    # has a uniform list length; downstream can filter on keyframe_count > 0.
    dim = next((e.shape[0] for e in embeddings if e is not None), 512)
    vec_rows = [
        (e.tolist() if e is not None else [0.0] * dim) for e in embeddings
    ]

    upsert_features(
        features_path,
        column_updates=[
            ("clip_embedding", vec_rows),
            ("keyframe_count", keyframe_counts),
        ],
        keyed_by=owned["item_id"],
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", type=Path, default=Path("data/feature_store/items.parquet"))
    ap.add_argument("--out", type=Path, default=Path("data/feature_store/item_features.parquet"))
    ap.add_argument("--media-root", type=Path, default=None,
                    help="root of your Plex / nami-stream media shares")
    ap.add_argument("--keyframes", type=int, default=8)
    ap.add_argument("--model", default="ViT-B-32")
    ap.add_argument("--pretrained", default="laion2b_s34b_b79k")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        raise RuntimeError("ffmpeg + ffprobe must be on PATH")

    run(
        args.items, args.out, args.media_root, args.keyframes,
        args.model, args.pretrained, args.batch_size, args.device, args.limit,
    )


if __name__ == "__main__":
    main()
