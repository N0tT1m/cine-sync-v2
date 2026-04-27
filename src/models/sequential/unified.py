"""Unified sequential recommender.

Replaces the five variants in `src/models/sequential/src/model.py`
(SequentialRecommender/LSTM, TransformerSequentialRecommender, AttentionalSequentialRecommender,
HierarchicalSequentialRecommender, SessionBasedRecommender). One class, one config switch.

Two architectures:

    architecture='transformer'   # SASRec-style: causal self-attention + next-item logits. Default.
    architecture='rnn'           # LSTM/GRU — retained for legacy artifacts; deprecated for new work.

Both emit per-position logits over the item vocabulary, so downstream training code
can use standard next-item cross-entropy. The hierarchical (short+long) and session
variants belong under `src/models/experimental/` if they're worth reviving.

BERT4Rec is kept separate (`src/models/advanced/bert4rec_recommender.py`) because its
bidirectional attention + MLM pretraining objective diverges enough to merit its own class.
"""
from __future__ import annotations

import math
import warnings
from typing import Optional

import torch
import torch.nn as nn


class _SinusoidalPositionalEmbedding(nn.Module):
    def __init__(self, max_len: int, dim: int) -> None:
        super().__init__()
        pe = torch.zeros(max_len, dim)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, seq_len: int) -> torch.Tensor:
        return self.pe[:seq_len]


class SequentialRecommender(nn.Module):
    """Next-item predictor over a fixed item vocabulary.

    Args:
        num_items: size of the item vocabulary (index 0 reserved for padding).
        architecture: 'transformer' (default) or 'rnn'.
        embedding_dim: shared item embedding size.
        hidden_dim: only used by 'rnn'; RNN hidden size.
        num_layers: number of transformer blocks OR stacked RNN layers.
        num_heads: attention heads (transformer only).
        max_seq_len: max positions learned by the positional embedding.
        dropout: applied on embeddings + inside blocks.
        rnn_type: 'LSTM' or 'GRU' when architecture='rnn'.
        tie_weights: share item embedding with the output projection (standard for SASRec).
    """

    def __init__(
        self,
        num_items: int,
        *,
        architecture: str = "transformer",
        embedding_dim: int = 128,
        hidden_dim: int = 256,
        num_layers: int = 2,
        num_heads: int = 4,
        max_seq_len: int = 200,
        dropout: float = 0.2,
        rnn_type: str = "GRU",
        tie_weights: bool = True,
    ) -> None:
        super().__init__()
        if architecture not in ("transformer", "rnn"):
            raise ValueError(f"architecture must be 'transformer' or 'rnn', got {architecture!r}")

        self.num_items = num_items
        self.architecture = architecture
        self.embedding_dim = embedding_dim
        self.max_seq_len = max_seq_len
        self.pad_idx = 0

        # Item vocab uses index 0 for padding so the caller can batch ragged sequences.
        self.item_embedding = nn.Embedding(num_items + 1, embedding_dim, padding_idx=self.pad_idx)
        self.embed_dropout = nn.Dropout(dropout)

        if architecture == "transformer":
            self.pos_embedding = _SinusoidalPositionalEmbedding(max_seq_len, embedding_dim)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=embedding_dim,
                nhead=num_heads,
                dim_feedforward=embedding_dim * 4,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            self.output_norm = nn.LayerNorm(embedding_dim)
            out_in = embedding_dim
            self.rnn = None
        else:
            warnings.warn(
                "architecture='rnn' is retained for legacy artifacts; "
                "prefer 'transformer' for new training runs.",
                DeprecationWarning,
                stacklevel=2,
            )
            rnn_cls = {"LSTM": nn.LSTM, "GRU": nn.GRU}.get(rnn_type.upper())
            if rnn_cls is None:
                raise ValueError(f"rnn_type must be 'LSTM' or 'GRU', got {rnn_type!r}")
            self.rnn = rnn_cls(
                embedding_dim,
                hidden_dim,
                num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
            )
            self.output_norm = nn.Identity()
            out_in = hidden_dim
            self.transformer = None

        # Output projection — tied to item embedding when dims match (SASRec default).
        if tie_weights and architecture == "transformer" and out_in == embedding_dim:
            self.output_projection: Optional[nn.Linear] = None
        else:
            self.output_projection = nn.Linear(out_in, num_items)

        self._init_weights()

    # ---- internals -----------------------------------------------------

    def _init_weights(self) -> None:
        nn.init.xavier_normal_(self.item_embedding.weight)
        with torch.no_grad():
            self.item_embedding.weight[self.pad_idx].zero_()
        for name, p in self.named_parameters():
            if p.dim() < 2 or "item_embedding" in name:
                continue
            if "rnn" in name.lower() and "weight" in name:
                nn.init.orthogonal_(p)
            elif "weight" in name:
                nn.init.xavier_normal_(p)

    def _project(self, hidden: torch.Tensor) -> torch.Tensor:
        """Map hidden states to vocab logits, using tied embedding when configured."""
        if self.output_projection is not None:
            return self.output_projection(hidden)
        return hidden @ self.item_embedding.weight[: self.num_items].T

    # ---- public API ----------------------------------------------------

    def encode(self, sequences: torch.Tensor) -> torch.Tensor:
        """Return per-position hidden states of shape (B, T, D).

        `sequences` is (B, T) long ids. Padding uses index 0. For the transformer,
        a causal mask is applied so position t only attends to positions <= t.
        """
        x = self.item_embedding(sequences)
        x = self.embed_dropout(x)

        if self.transformer is not None:
            seq_len = sequences.size(1)
            if seq_len > self.max_seq_len:
                raise ValueError(
                    f"sequence length {seq_len} exceeds max_seq_len={self.max_seq_len}"
                )
            x = x + self.pos_embedding(seq_len).to(x.dtype).unsqueeze(0)
            # bool masks: True = "don't attend". Both masks share a dtype per torch's API.
            causal = torch.ones(seq_len, seq_len, dtype=torch.bool, device=sequences.device).triu(1)
            key_padding = sequences.eq(self.pad_idx)
            x = self.transformer(x, mask=causal, src_key_padding_mask=key_padding)
            x = self.output_norm(x)
            return x

        # RNN branch
        assert self.rnn is not None
        x, _ = self.rnn(x)
        return x

    def forward(self, sequences: torch.Tensor) -> torch.Tensor:
        """Per-position next-item logits over the item vocabulary. Shape (B, T, num_items)."""
        hidden = self.encode(sequences)
        return self._project(hidden)

    def predict_next(self, sequences: torch.Tensor, top_k: int = 10) -> torch.Tensor:
        """Top-K next-item indices for the last position. Shape (B, top_k)."""
        with torch.no_grad():
            hidden = self.encode(sequences)
            last = hidden[:, -1, :]
            logits = self._project(last.unsqueeze(1)).squeeze(1)
            return logits.topk(top_k, dim=-1).indices


__all__ = ["SequentialRecommender"]
