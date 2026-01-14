"""
Franchise Sequence Model (FSM)
Models movie franchise ordering, sequel/prequel relationships, and franchise viewing patterns.
Optimized for RTX 4090
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class FranchiseConfig:
    """Configuration for Franchise Sequence Model"""
    embedding_dim: int = 256
    hidden_dim: int = 512
    num_heads: int = 8
    num_layers: int = 4
    max_franchise_length: int = 50
    dropout: float = 0.1
    num_franchises: int = 10000
    num_movies: int = 100000
    num_users: int = 50000


class FranchisePositionalEncoding(nn.Module):
    """Positional encoding that captures franchise chronology and release order"""

    def __init__(self, d_model: int, max_len: int = 100):
        super().__init__()
        self.d_model = d_model

        # Chronological position encoding
        pe_chrono = torch.zeros(max_len, d_model // 2)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model // 2, 2).float() *
                           (-math.log(10000.0) / (d_model // 2)))
        pe_chrono[:, 0::2] = torch.sin(position * div_term)
        pe_chrono[:, 1::2] = torch.cos(position * div_term)

        # Release order encoding (different from chronological)
        pe_release = torch.zeros(max_len, d_model // 2)
        pe_release[:, 0::2] = torch.sin(position * div_term * 1.5)
        pe_release[:, 1::2] = torch.cos(position * div_term * 1.5)

        pe = torch.cat([pe_chrono, pe_release], dim=-1)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor, chronological_pos: Optional[torch.Tensor] = None,
                release_pos: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: Input tensor [batch, seq_len, d_model]
            chronological_pos: Story chronology positions
            release_pos: Release date positions
        """
        seq_len = x.size(1)
        if chronological_pos is not None:
            return x + self.pe[:, chronological_pos, :]
        return x + self.pe[:, :seq_len, :]


class FranchiseAttention(nn.Module):
    """Attention mechanism aware of franchise relationships"""

    def __init__(self, d_model: int, n_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

        # Franchise relationship type embeddings
        self.relation_embeddings = nn.Embedding(5, self.d_k)  # sequel, prequel, spinoff, reboot, same_universe

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, relation_matrix: Optional[torch.Tensor] = None,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len = x.size(0), x.size(1)
        residual = x

        Q = self.w_q(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)

        # Base attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        # Add franchise relationship bias if provided
        if relation_matrix is not None:
            rel_emb = self.relation_embeddings(relation_matrix)
            rel_scores = torch.einsum('bhid,ijd->bhij', Q, rel_emb)
            scores = scores + rel_scores * 0.5

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attention = F.softmax(scores, dim=-1)
        attention = self.dropout(attention)

        context = torch.matmul(attention, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = self.w_o(context)

        return self.layer_norm(output + residual)


class FranchiseSequenceEncoder(nn.Module):
    """Encoder for franchise movie sequences"""

    def __init__(self, config: FranchiseConfig):
        super().__init__()
        self.config = config

        # Movie and franchise embeddings
        self.movie_embedding = nn.Embedding(config.num_movies, config.embedding_dim)
        self.franchise_embedding = nn.Embedding(config.num_franchises, config.embedding_dim)

        # Positional encoding (must match hidden_dim since it's added after input_proj)
        self.pos_encoding = FranchisePositionalEncoding(config.hidden_dim, config.max_franchise_length)

        # Entry type embedding (original, sequel, prequel, spinoff, reboot)
        self.entry_type_embedding = nn.Embedding(5, config.embedding_dim)

        # Projection to hidden dimension
        self.input_proj = nn.Linear(config.embedding_dim * 3, config.hidden_dim)

        # Transformer layers
        self.attention_layers = nn.ModuleList([
            FranchiseAttention(config.hidden_dim, config.num_heads, config.dropout)
            for _ in range(config.num_layers)
        ])

        self.ffn_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(config.hidden_dim, config.hidden_dim * 4),
                nn.GELU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.hidden_dim * 4, config.hidden_dim),
                nn.Dropout(config.dropout),
            )
            for _ in range(config.num_layers)
        ])

        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(config.hidden_dim) for _ in range(config.num_layers)
        ])

    def forward(self, movie_ids: torch.Tensor, franchise_ids: torch.Tensor,
                entry_types: torch.Tensor, relation_matrix: Optional[torch.Tensor] = None,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            movie_ids: Movie IDs [batch, seq_len]
            franchise_ids: Franchise IDs [batch, seq_len]
            entry_types: Entry type IDs [batch, seq_len]
            relation_matrix: Pairwise relation types [seq_len, seq_len]
            mask: Attention mask
        """
        # Get embeddings
        movie_emb = self.movie_embedding(movie_ids)
        franchise_emb = self.franchise_embedding(franchise_ids)
        entry_emb = self.entry_type_embedding(entry_types)

        # Combine and project
        combined = torch.cat([movie_emb, franchise_emb, entry_emb], dim=-1)
        x = self.input_proj(combined)

        # Add positional encoding
        x = self.pos_encoding(x)

        # Apply transformer layers
        for attn, ffn, ln in zip(self.attention_layers, self.ffn_layers, self.layer_norms):
            x = attn(x, relation_matrix, mask)
            x = ln(x + ffn(x))

        return x


class FranchiseSequenceModel(nn.Module):
    """
    Complete Franchise Sequence Model for movie recommendations.
    Specializes in understanding and recommending movies within franchises.
    """

    def __init__(self, config: Optional[FranchiseConfig] = None):
        super().__init__()
        self.config = config or FranchiseConfig()

        # User embedding
        self.user_embedding = nn.Embedding(self.config.num_users, self.config.hidden_dim)

        # Franchise sequence encoder
        self.encoder = FranchiseSequenceEncoder(self.config)

        # User preference aggregation
        self.user_preference_attention = nn.MultiheadAttention(
            self.config.hidden_dim, self.config.num_heads,
            dropout=self.config.dropout, batch_first=True
        )

        # Franchise completion predictor
        self.completion_head = nn.Sequential(
            nn.Linear(self.config.hidden_dim * 2, self.config.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.hidden_dim, 1),
            nn.Sigmoid()
        )

        # Next movie predictor
        self.next_movie_head = nn.Sequential(
            nn.Linear(self.config.hidden_dim * 2, self.config.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.hidden_dim, self.config.num_movies)
        )

        # Rating predictor
        self.rating_head = nn.Sequential(
            nn.Linear(self.config.hidden_dim * 3, self.config.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.hidden_dim, 1)
        )

        # Order preference predictor (chronological vs release order)
        self.order_preference_head = nn.Sequential(
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim // 2),
            nn.GELU(),
            nn.Linear(self.config.hidden_dim // 2, 2),
            nn.Softmax(dim=-1)
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0, std=0.02)

    def forward(self, user_ids: torch.Tensor, movie_ids: torch.Tensor,
                franchise_ids: torch.Tensor, entry_types: torch.Tensor,
                relation_matrix: Optional[torch.Tensor] = None,
                mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass for franchise sequence modeling.

        Returns dict with:
            - sequence_encoding: Encoded franchise sequence
            - completion_prob: Probability user will complete franchise
            - next_movie_logits: Logits for next movie prediction
            - rating_pred: Predicted ratings
            - order_preference: Preferred viewing order [chrono, release]
        """
        batch_size = user_ids.size(0)

        # Encode franchise sequence
        sequence_encoding = self.encoder(movie_ids, franchise_ids, entry_types,
                                         relation_matrix, mask)

        # Get user embedding
        user_emb = self.user_embedding(user_ids)

        # Compute user preference via attention over sequence
        user_query = user_emb.unsqueeze(1)
        user_preference, _ = self.user_preference_attention(
            user_query, sequence_encoding, sequence_encoding
        )
        user_preference = user_preference.squeeze(1)

        # Get sequence representation (last position or mean pooling)
        if mask is not None:
            # Masked mean pooling
            mask_expanded = mask.unsqueeze(-1).float()
            seq_repr = (sequence_encoding * mask_expanded).sum(1) / mask_expanded.sum(1)
        else:
            seq_repr = sequence_encoding.mean(1)

        # Predict franchise completion probability
        completion_input = torch.cat([user_preference, seq_repr], dim=-1)
        completion_prob = self.completion_head(completion_input)

        # Predict next movie in franchise
        next_movie_logits = self.next_movie_head(completion_input)

        # Predict ratings
        rating_input = torch.cat([user_emb, user_preference, seq_repr], dim=-1)
        rating_pred = self.rating_head(rating_input)

        # Predict viewing order preference
        order_preference = self.order_preference_head(user_preference)

        return {
            'sequence_encoding': sequence_encoding,
            'completion_prob': completion_prob,
            'next_movie_logits': next_movie_logits,
            'rating_pred': rating_pred,
            'order_preference': order_preference,
            'user_preference': user_preference
        }

    def recommend_next(self, user_ids: torch.Tensor, watched_movies: torch.Tensor,
                       franchise_ids: torch.Tensor, entry_types: torch.Tensor,
                       top_k: int = 10) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Recommend next movie in franchise for users.

        Returns:
            movie_ids: Top-k recommended movie IDs
            scores: Recommendation scores
        """
        with torch.no_grad():
            outputs = self.forward(user_ids, watched_movies, franchise_ids, entry_types)
            logits = outputs['next_movie_logits']

            # Mask already watched movies
            for i, watched in enumerate(watched_movies):
                logits[i, watched] = -float('inf')

            scores, movie_ids = torch.topk(F.softmax(logits, dim=-1), top_k, dim=-1)

        return movie_ids, scores

    def get_franchise_embedding(self, franchise_id: int) -> torch.Tensor:
        """Get embedding for a franchise"""
        return self.encoder.franchise_embedding.weight[franchise_id]


class FranchiseSequenceTrainer:
    """Trainer for Franchise Sequence Model"""

    def __init__(self, model: FranchiseSequenceModel, lr: float = 1e-4,
                 weight_decay: float = 0.01, device: str = 'cuda'):
        self.model = model.to(device)
        self.device = device

        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=10, T_mult=2
        )

        # Loss functions
        self.rating_loss = nn.MSELoss()
        self.next_movie_loss = nn.CrossEntropyLoss()
        self.completion_loss = nn.BCELoss()
        self.order_loss = nn.CrossEntropyLoss()

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Single training step"""
        self.model.train()

        # Move batch to device
        user_ids = batch['user_ids'].to(self.device)
        movie_ids = batch['movie_ids'].to(self.device)
        franchise_ids = batch['franchise_ids'].to(self.device)
        entry_types = batch['entry_types'].to(self.device)
        target_ratings = batch['ratings'].to(self.device)
        target_next = batch['next_movie'].to(self.device)
        target_completion = batch['completed_franchise'].to(self.device).float()
        target_order = batch.get('order_preference')
        mask = batch.get('mask')

        if mask is not None:
            mask = mask.to(self.device)

        # Forward pass
        outputs = self.model(user_ids, movie_ids, franchise_ids, entry_types, mask=mask)

        # Compute losses
        rating_loss = self.rating_loss(outputs['rating_pred'].squeeze(), target_ratings)
        next_loss = self.next_movie_loss(outputs['next_movie_logits'], target_next)
        completion_loss = self.completion_loss(outputs['completion_prob'].squeeze(), target_completion)

        total_loss = rating_loss + next_loss + completion_loss * 0.5

        if target_order is not None:
            target_order = target_order.to(self.device)
            order_loss = self.order_loss(outputs['order_preference'], target_order)
            total_loss += order_loss * 0.3

        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()

        return {
            'total_loss': total_loss.item(),
            'rating_loss': rating_loss.item(),
            'next_movie_loss': next_loss.item(),
            'completion_loss': completion_loss.item()
        }

    def evaluate(self, dataloader) -> Dict[str, float]:
        """Evaluate model on validation set"""
        self.model.eval()
        total_metrics = {'rating_mse': 0, 'next_acc': 0, 'completion_acc': 0}
        num_batches = 0

        with torch.no_grad():
            for batch in dataloader:
                user_ids = batch['user_ids'].to(self.device)
                movie_ids = batch['movie_ids'].to(self.device)
                franchise_ids = batch['franchise_ids'].to(self.device)
                entry_types = batch['entry_types'].to(self.device)
                target_ratings = batch['ratings'].to(self.device)
                target_next = batch['next_movie'].to(self.device)
                target_completion = batch['completed_franchise'].to(self.device).float()

                outputs = self.model(user_ids, movie_ids, franchise_ids, entry_types)

                # Rating MSE
                total_metrics['rating_mse'] += F.mse_loss(
                    outputs['rating_pred'].squeeze(), target_ratings
                ).item()

                # Next movie accuracy
                pred_next = outputs['next_movie_logits'].argmax(dim=-1)
                total_metrics['next_acc'] += (pred_next == target_next).float().mean().item()

                # Completion accuracy
                pred_completion = (outputs['completion_prob'].squeeze() > 0.5).float()
                total_metrics['completion_acc'] += (pred_completion == target_completion).float().mean().item()

                num_batches += 1

        return {k: v / num_batches for k, v in total_metrics.items()}
