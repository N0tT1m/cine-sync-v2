"""Scorer adapters. Wrap trained model objects into the `Scorer` protocol
expected by the inference registry (`.score(item_ids, user_id, **ctx)`).

Selected per-model via manifest.yaml -> loader_config.scorer_adapter:

    loader_config:
      class_path: src.models.two_tower.TwoTowerModel
      init_kwargs: {num_users: 0, num_items: 0, user_text_dim: 384, item_text_dim: 384}
      scorer_adapter: src.models.adapters.TwoTowerScorer
      scorer_kwargs:
        item_embeddings_path: item_emb.npy
        item_id_map_path: item_ids.json
"""
from .bert4rec_scorer import BERT4RecScorer
from .ncf_scorer import NCFScorer
from .sequential_scorer import SequentialScorer
from .two_tower_scorer import TwoTowerScorer

__all__ = ["BERT4RecScorer", "NCFScorer", "SequentialScorer", "TwoTowerScorer"]
