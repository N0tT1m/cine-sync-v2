"""Feature-store-native trainers for the served model trio.

Each trainer reads the canonical feature store (data/feature_store/*.parquet),
trains/derives a model, and writes serving-ready artifacts + a complete
manifest.yaml so the inference registry loads a REAL scorer (not a stub).

Item ids stay in the feature store's canonical space (e.g. ``tmdb:862``), so
the embeddings/indexes a trainer emits line up with the candidate pool that
`candidates.py` builds from items.parquet. That alignment is the whole point:
without it, every lookup misses and scoring silently falls back to noise.

Run one via the orchestrator (services/inference/train.py) or directly:

    python -m services.inference.trainers.two_tower_cf
    python -m services.inference.trainers.sbert_index
    python -m services.inference.trainers.bert4rec_seq
"""
