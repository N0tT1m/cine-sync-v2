"""Placeholder 'model' for embedding-only scorers.

The two-tower and sbert models are served purely from cached embeddings
(see TwoTowerScorer) — there is no torch module to instantiate at serve time.
But the registry's torch loader requires a `class_path` it can construct, so
manifests for those models point at this no-op class with `artifact: null`.
Because the artifact is null, the loader never imports torch for these models.
"""
from __future__ import annotations


class NullModel:
    """Instantiable no-arg stand-in. Holds no weights and is never called."""

    def __init__(self, **_kwargs: object) -> None:  # tolerate any init_kwargs
        pass
