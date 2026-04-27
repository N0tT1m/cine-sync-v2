"""Shared IO for item_features.parquet — column-level upsert."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, Tuple

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)


def load_items(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"items.parquet missing at {path}. Run build_feature_store.py first."
        )
    return pd.read_parquet(path)


def upsert_features(
    features_path: Path,
    column_updates: Iterable[Tuple[str, object]],
    keyed_by: pd.Series,
) -> None:
    """Add/overwrite columns in item_features.parquet keyed on item_id.

    column_updates: iterable of (column_name, series_or_list_aligned_with_keyed_by).
    """
    features_path.parent.mkdir(parents=True, exist_ok=True)

    if features_path.exists():
        existing = pd.read_parquet(features_path)
    else:
        existing = pd.DataFrame({"item_id": []})

    incoming = pd.DataFrame({"item_id": keyed_by.values})
    for col, values in column_updates:
        incoming[col] = list(values)

    if existing.empty:
        merged = incoming
    else:
        cols_to_drop = [c for c in incoming.columns if c != "item_id" and c in existing.columns]
        if cols_to_drop:
            existing = existing.drop(columns=cols_to_drop)
        merged = existing.merge(incoming, on="item_id", how="outer")

    table = pa.Table.from_pandas(merged)
    pq.write_table(table, features_path, compression="snappy")
    logger.info("upserted %d rows into %s", len(merged), features_path)
