"""Unified Two-Tower recommender.

Replaces the five variants that used to live in `src/models/two_tower/src/model.py`
(Standard, Ultimate, Enhanced, MultiTask, Collaborative) and the two in
`src/models/advanced/` (SentenceBERT, Enhanced). One class, config-driven.

The shape you want is turned on by which features you pass:

    tower = TwoTowerModel(
        num_users=N, num_items=M,                    # collaborative IDs
        user_categorical_dims={"age_bucket": 8},     # learned cat embeddings
        item_categorical_dims={"genre": 25},
        user_numerical_dim=4,
        item_numerical_dim=6,
        user_text_dim=384,                           # e.g. SBERT
        item_text_dim=384,
        embedding_dim=128,
        hidden_layers=[256, 128],
    )

Any of these inputs is optional; the tower only builds the branches it needs.
Unknown model variants (MoE, progressive-interaction, uncertainty heads, etc.)
from the old `UltimateTwoTowerModel` are intentionally out of scope — they
belong under `src/models/experimental/` if they're worth reviving.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def _adaptive_emb_dim(vocab_size: int, cap: int = 64) -> int:
    """sqrt-style heuristic bounded by `cap`; matches what the legacy variants did."""
    return min(cap, max(8, int(round(vocab_size ** 0.25)) * 4))


@dataclass
class TowerInputs:
    """Shape container for either the user or item branch."""

    num_ids: int = 0
    categorical_dims: Dict[str, int] = field(default_factory=dict)
    numerical_dim: int = 0
    text_dim: int = 0


class _Tower(nn.Module):
    """One branch of the two-tower model: ID + cat + numerical + text → embedding."""

    def __init__(
        self,
        inputs: TowerInputs,
        embedding_dim: int,
        hidden_layers: List[int],
        dropout: float,
        use_batch_norm: bool,
        id_emb_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.inputs = inputs
        id_emb_dim = id_emb_dim or embedding_dim

        total_in = 0
        if inputs.num_ids > 0:
            self.id_embedding: Optional[nn.Embedding] = nn.Embedding(inputs.num_ids, id_emb_dim)
            total_in += id_emb_dim
        else:
            self.id_embedding = None

        self.categorical_embeddings = nn.ModuleDict()
        for feature_name, vocab_size in inputs.categorical_dims.items():
            emb_dim = _adaptive_emb_dim(vocab_size)
            self.categorical_embeddings[feature_name] = nn.Embedding(vocab_size, emb_dim)
            total_in += emb_dim

        total_in += inputs.numerical_dim
        total_in += inputs.text_dim

        if total_in == 0:
            raise ValueError("tower has no inputs; supply ids, categorical, numerical, or text")

        self.mlp = self._build_mlp(total_in, hidden_layers, embedding_dim, dropout, use_batch_norm)
        self._init_weights()

    @staticmethod
    def _build_mlp(
        input_dim: int,
        hidden_layers: List[int],
        output_dim: int,
        dropout: float,
        use_batch_norm: bool,
    ) -> nn.Sequential:
        layers: List[nn.Module] = []
        prev = input_dim
        for h in hidden_layers:
            layers.append(nn.Linear(prev, h))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, output_dim))
        layers.append(nn.LayerNorm(output_dim))
        return nn.Sequential(*layers)

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Embedding):
                nn.init.xavier_normal_(m.weight)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        ids: Optional[torch.Tensor] = None,
        categorical: Optional[Dict[str, torch.Tensor]] = None,
        numerical: Optional[torch.Tensor] = None,
        text: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        parts: List[torch.Tensor] = []

        if self.id_embedding is not None:
            if ids is None:
                raise ValueError("this tower was built with num_ids > 0; pass `ids`")
            parts.append(self.id_embedding(ids))

        if self.categorical_embeddings:
            if categorical is None:
                raise ValueError("this tower expects categorical features; pass `categorical`")
            for name, embed in self.categorical_embeddings.items():
                if name not in categorical:
                    raise KeyError(f"missing categorical feature {name!r}")
                parts.append(embed(categorical[name]))

        if self.inputs.numerical_dim > 0:
            if numerical is None:
                raise ValueError("this tower expects numerical features; pass `numerical`")
            parts.append(numerical)

        if self.inputs.text_dim > 0:
            if text is None:
                raise ValueError("this tower expects a text embedding; pass `text`")
            parts.append(text)

        x = torch.cat(parts, dim=1) if len(parts) > 1 else parts[0]
        return self.mlp(x)


class TwoTowerModel(nn.Module):
    """Config-driven dual-encoder for retrieval and ranking.

    Output space: L2-normalized embeddings. Score is the dot-product (cosine
    similarity) scaled by a learnable temperature. This is the contract the
    inference registry adapter depends on — keep it stable across variants.
    """

    def __init__(
        self,
        *,
        num_users: int = 0,
        num_items: int = 0,
        user_categorical_dims: Optional[Dict[str, int]] = None,
        item_categorical_dims: Optional[Dict[str, int]] = None,
        user_numerical_dim: int = 0,
        item_numerical_dim: int = 0,
        user_text_dim: int = 0,
        item_text_dim: int = 0,
        embedding_dim: int = 128,
        hidden_layers: Optional[List[int]] = None,
        dropout: float = 0.2,
        use_batch_norm: bool = True,
        temperature_init: float = 0.1,
        normalize: bool = True,
    ) -> None:
        super().__init__()
        hidden_layers = hidden_layers or [256, 128]

        self.embedding_dim = embedding_dim
        self.normalize = normalize

        self.user_tower = _Tower(
            TowerInputs(
                num_ids=num_users,
                categorical_dims=user_categorical_dims or {},
                numerical_dim=user_numerical_dim,
                text_dim=user_text_dim,
            ),
            embedding_dim=embedding_dim,
            hidden_layers=hidden_layers,
            dropout=dropout,
            use_batch_norm=use_batch_norm,
        )
        self.item_tower = _Tower(
            TowerInputs(
                num_ids=num_items,
                categorical_dims=item_categorical_dims or {},
                numerical_dim=item_numerical_dim,
                text_dim=item_text_dim,
            ),
            embedding_dim=embedding_dim,
            hidden_layers=hidden_layers,
            dropout=dropout,
            use_batch_norm=use_batch_norm,
        )

        self.temperature = nn.Parameter(torch.tensor(float(temperature_init)))

    # ---- encoders -----------------------------------------------------

    def encode_user(
        self,
        user_ids: Optional[torch.Tensor] = None,
        user_categorical: Optional[Dict[str, torch.Tensor]] = None,
        user_numerical: Optional[torch.Tensor] = None,
        user_text: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        emb = self.user_tower(user_ids, user_categorical, user_numerical, user_text)
        return F.normalize(emb, p=2, dim=1) if self.normalize else emb

    def encode_item(
        self,
        item_ids: Optional[torch.Tensor] = None,
        item_categorical: Optional[Dict[str, torch.Tensor]] = None,
        item_numerical: Optional[torch.Tensor] = None,
        item_text: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        emb = self.item_tower(item_ids, item_categorical, item_numerical, item_text)
        return F.normalize(emb, p=2, dim=1) if self.normalize else emb

    # ---- scoring ------------------------------------------------------

    def forward(
        self,
        user: Optional[dict] = None,
        item: Optional[dict] = None,
        *,
        user_embedding: Optional[torch.Tensor] = None,
        item_embedding: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Score aligned (user, item) pairs. Pass either raw inputs or precomputed embeddings."""
        if user_embedding is None:
            if user is None:
                raise ValueError("pass `user` dict or `user_embedding`")
            user_embedding = self.encode_user(**user)
        if item_embedding is None:
            if item is None:
                raise ValueError("pass `item` dict or `item_embedding`")
            item_embedding = self.encode_item(**item)
        return (user_embedding * item_embedding).sum(dim=1) / self.temperature

    def similarity_matrix(
        self,
        user_embedding: torch.Tensor,
        item_embedding: torch.Tensor,
    ) -> torch.Tensor:
        """All-pairs cosine similarity scaled by temperature. Use for retrieval against a cached item index."""
        return user_embedding @ item_embedding.T / self.temperature


__all__ = ["TwoTowerModel", "TowerInputs"]
