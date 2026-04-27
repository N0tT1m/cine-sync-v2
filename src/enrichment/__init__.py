"""Content enrichment jobs — read items.parquet, write item_features.parquet.

Jobs are idempotent per (item_id, column): rerunning overwrites just its column,
never touching the others. This keeps the SBERT, CLIP, and audio jobs independent.
"""
