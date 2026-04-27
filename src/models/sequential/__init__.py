"""Sequential recommender.

`SequentialRecommender` is the canonical implementation — transformer (SASRec-style)
by default, with an RNN (LSTM/GRU) fallback for legacy artifacts. Hierarchical and
session variants were moved to `src/models/experimental/` as they weren't in
production use.

Use BERT4Rec (`src/models/advanced/bert4rec_recommender.py`) when you specifically
want bidirectional attention + masked-LM pretraining.
"""
from .unified import SequentialRecommender

__all__ = ["SequentialRecommender"]
