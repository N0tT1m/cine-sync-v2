"""
Episode Sequence Model (ESM-TV)
Models episode-level viewing sequences and within-season patterns.
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
class EpisodeSequenceConfig:
    """Configuration for Episode Sequence Model"""
    embedding_dim: int = 256
    hidden_dim: int = 512
    num_heads: int = 8
    num_layers: int = 4
    num_shows: int = 50000
    num_users: int = 50000
    max_episodes: int = 500
    max_seasons: int = 30
    dropout: float = 0.1


class EpisodeEncoder(nn.Module):
    """Encodes individual episode features"""

    def __init__(self, config: EpisodeSequenceConfig):
        super().__init__()

        # Episode position encoding
        self.episode_pos_embedding = nn.Embedding(config.max_episodes, config.embedding_dim // 2)
        self.season_embedding = nn.Embedding(config.max_seasons, config.embedding_dim // 4)

        # Episode type embedding
        self.episode_type_embedding = nn.Embedding(10, config.embedding_dim // 4)
        # Types: regular, premiere, finale, mid_season_finale, special, crossover,
        #        bottle, flashback, clip_show, musical

        # Episode features encoder
        self.features_encoder = nn.Sequential(
            nn.Linear(8, config.embedding_dim // 4),  # rating, duration, viewership, importance, cliffhanger_score
            nn.GELU(),
            nn.Linear(config.embedding_dim // 4, config.embedding_dim // 4)
        )

        # Combine
        self.episode_fusion = nn.Sequential(
            nn.Linear(config.embedding_dim + config.embedding_dim // 4, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.hidden_dim)
        )

    def forward(self, episode_positions: torch.Tensor, season_ids: torch.Tensor,
                episode_types: torch.Tensor, episode_features: torch.Tensor) -> torch.Tensor:
        """Encode episode"""
        pos_emb = self.episode_pos_embedding(episode_positions)
        season_emb = self.season_embedding(season_ids)
        type_emb = self.episode_type_embedding(episode_types)
        feat_emb = self.features_encoder(episode_features)

        combined = torch.cat([pos_emb, season_emb, type_emb, feat_emb], dim=-1)
        return self.episode_fusion(combined)


class EpisodeSequenceTransformer(nn.Module):
    """Transformer for episode sequences"""

    def __init__(self, config: EpisodeSequenceConfig):
        super().__init__()

        # Positional encoding
        self.pos_encoding = nn.Parameter(torch.randn(config.max_episodes, config.hidden_dim))

        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_dim,
            nhead=config.num_heads,
            dim_feedforward=config.hidden_dim * 4,
            dropout=config.dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=config.num_layers)

        # Episode skip prediction
        self.skip_head = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.GELU(),
            nn.Linear(config.hidden_dim // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, episode_repr: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Process episode sequence"""
        seq_len = episode_repr.size(1)
        x = episode_repr + self.pos_encoding[:seq_len].unsqueeze(0)

        if mask is not None:
            transformed = self.transformer(x, src_key_padding_mask=~mask)
        else:
            transformed = self.transformer(x)

        skip_probs = self.skip_head(transformed)
        return transformed, skip_probs


class EpisodeSequenceModel(nn.Module):
    """
    Complete Episode Sequence Model for TV show episode recommendations.
    """

    def __init__(self, config: Optional[EpisodeSequenceConfig] = None):
        super().__init__()
        self.config = config or EpisodeSequenceConfig()

        # Core embeddings
        self.show_embedding = nn.Embedding(self.config.num_shows, self.config.embedding_dim)
        self.user_embedding = nn.Embedding(self.config.num_users, self.config.hidden_dim)

        # Episode encoder
        self.episode_encoder = EpisodeEncoder(self.config)

        # Sequence transformer
        self.sequence_transformer = EpisodeSequenceTransformer(self.config)

        # User episode preference
        self.user_episode_preference = nn.Sequential(
            nn.Linear(self.config.hidden_dim * 2, self.config.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim)
        )

        # Next episode predictor
        self.next_episode_head = nn.Sequential(
            nn.Linear(self.config.hidden_dim * 2, self.config.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.hidden_dim, self.config.max_episodes)
        )

        # Episode rating predictor
        self.rating_head = nn.Sequential(
            nn.Linear(self.config.hidden_dim * 3, self.config.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.hidden_dim, 1)
        )

        # Continuation probability
        self.continue_head = nn.Sequential(
            nn.Linear(self.config.hidden_dim * 2, self.config.hidden_dim),
            nn.GELU(),
            nn.Linear(self.config.hidden_dim, 1),
            nn.Sigmoid()
        )

        # Watch order recommender (release vs optimal)
        self.order_head = nn.Sequential(
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim // 2),
            nn.GELU(),
            nn.Linear(self.config.hidden_dim // 2, 2),
            nn.Softmax(dim=-1)
        )

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0, std=0.02)

    def forward(self, user_ids: torch.Tensor, show_ids: torch.Tensor,
                episode_positions: torch.Tensor, season_ids: torch.Tensor,
                episode_types: torch.Tensor, episode_features: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Forward pass"""
        batch_size = user_ids.size(0)

        # Get embeddings
        user_emb = self.user_embedding(user_ids)
        show_emb = self.show_embedding(show_ids)

        # Encode episodes
        episode_repr = self.episode_encoder(episode_positions, season_ids,
                                           episode_types, episode_features)

        # Process sequence
        sequence_repr, skip_probs = self.sequence_transformer(episode_repr, mask)

        # Pool sequence
        if mask is not None:
            mask_expanded = mask.unsqueeze(-1).float()
            seq_pooled = (sequence_repr * mask_expanded).sum(1) / (mask_expanded.sum(1) + 1e-8)
        else:
            seq_pooled = sequence_repr.mean(dim=1)

        # User-sequence preference
        user_seq = torch.cat([user_emb, seq_pooled], dim=-1)
        episode_pref = self.user_episode_preference(user_seq)

        # Next episode prediction
        next_episode_logits = self.next_episode_head(user_seq)

        # Continuation probability
        continue_prob = self.continue_head(user_seq)

        # Watch order preference
        order_pref = self.order_head(user_emb)

        # Rating prediction
        show_repr = self.show_embedding(show_ids[:, 0] if show_ids.dim() > 1 else show_ids)
        rating_input = torch.cat([user_emb, show_repr.squeeze(), seq_pooled], dim=-1)
        rating_pred = self.rating_head(rating_input)

        return {
            'sequence_repr': sequence_repr,
            'episode_preference': episode_pref,
            'skip_probs': skip_probs,
            'next_episode_logits': next_episode_logits,
            'continue_prob': continue_prob,
            'order_preference': order_pref,
            'rating_pred': rating_pred
        }


class EpisodeSequenceTrainer:
    """Trainer for Episode Sequence Model"""

    def __init__(self, model: EpisodeSequenceModel, lr: float = 1e-4,
                 weight_decay: float = 0.01, device: str = 'cuda'):
        self.model = model.to(device)
        self.device = device

        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=10, T_mult=2
        )

        self.rating_loss = nn.MSELoss()
        self.next_loss = nn.CrossEntropyLoss()
        self.continue_loss = nn.BCELoss()
        self.skip_loss = nn.BCELoss()

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Single training step"""
        self.model.train()

        user_ids = batch['user_ids'].to(self.device)
        show_ids = batch['show_ids'].to(self.device)
        episode_positions = batch['episode_positions'].to(self.device)
        season_ids = batch['season_ids'].to(self.device)
        episode_types = batch['episode_types'].to(self.device)
        episode_features = batch['episode_features'].to(self.device)
        target_ratings = batch['ratings'].to(self.device)

        mask = batch.get('mask')
        if mask is not None:
            mask = mask.to(self.device)

        outputs = self.model(
            user_ids, show_ids, episode_positions, season_ids,
            episode_types, episode_features, mask
        )

        rating_loss = self.rating_loss(outputs['rating_pred'].squeeze(), target_ratings)
        total_loss = rating_loss

        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()

        return {
            'total_loss': total_loss.item(),
            'rating_loss': rating_loss.item()
        }

    def evaluate(self, dataloader) -> Dict[str, float]:
        """Evaluate model"""
        self.model.eval()
        total_metrics = {'rating_mse': 0}
        num_batches = 0

        with torch.no_grad():
            for batch in dataloader:
                user_ids = batch['user_ids'].to(self.device)
                show_ids = batch['show_ids'].to(self.device)
                episode_positions = batch['episode_positions'].to(self.device)
                season_ids = batch['season_ids'].to(self.device)
                episode_types = batch['episode_types'].to(self.device)
                episode_features = batch['episode_features'].to(self.device)
                target_ratings = batch['ratings'].to(self.device)

                outputs = self.model(
                    user_ids, show_ids, episode_positions, season_ids,
                    episode_types, episode_features
                )

                total_metrics['rating_mse'] += F.mse_loss(
                    outputs['rating_pred'].squeeze(), target_ratings
                ).item()

                num_batches += 1

        return {k: v / num_batches for k, v in total_metrics.items()}
