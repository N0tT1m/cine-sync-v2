"""Two-tower recommender.

`TwoTowerModel` is the canonical implementation — config-driven, handles the
variants (collaborative IDs, categorical features, dense numerical, SBERT/CLIP
text) that used to be separate classes.

Legacy classes still live under `src/models/two_tower/src/model.py` and
`src/models/advanced/*_two_tower.py` for external trainers that haven't
migrated yet. Prefer `TwoTowerModel` for new work.
"""
from .unified import TowerInputs, TwoTowerModel

__all__ = ["TwoTowerModel", "TowerInputs"]
