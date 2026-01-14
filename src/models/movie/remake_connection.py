"""
Remake Connection Model (RCM)
Models relationships between original films and their remakes/adaptations.
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
class RemakeConfig:
    """Configuration for Remake Connection Model"""
    embedding_dim: int = 256
    hidden_dim: int = 512
    num_heads: int = 8
    num_layers: int = 4
    num_movies: int = 100000
    num_users: int = 50000
    max_versions: int = 20  # Max versions of same story
    dropout: float = 0.1


class VersionEncoder(nn.Module):
    """Encodes different versions of the same story"""

    def __init__(self, config: RemakeConfig):
        super().__init__()

        # Version type embedding
        self.version_type_embedding = nn.Embedding(10, config.embedding_dim // 2)
        # Types: original, remake, reboot, reimagining, adaptation, sequel_remake,
        #        foreign_remake, shot_for_shot, modernization, period_update

        # Year gap encoding (time between original and remake)
        self.year_gap_encoder = nn.Sequential(
            nn.Linear(1, config.embedding_dim // 4),
            nn.GELU(),
            nn.Linear(config.embedding_dim // 4, config.embedding_dim // 4)
        )

        # Quality comparison features
        self.quality_encoder = nn.Sequential(
            nn.Linear(6, config.embedding_dim // 4),  # rating_original, rating_remake, delta, box_office_ratio, critic_delta, audience_delta
            nn.GELU(),
            nn.Linear(config.embedding_dim // 4, config.embedding_dim // 4)
        )

        # Combine
        self.version_fusion = nn.Sequential(
            nn.Linear(config.embedding_dim, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.hidden_dim)
        )

    def forward(self, version_types: torch.Tensor, year_gaps: torch.Tensor,
                quality_features: torch.Tensor) -> torch.Tensor:
        """Encode version relationships"""
        type_emb = self.version_type_embedding(version_types)
        gap_emb = self.year_gap_encoder(year_gaps.float().unsqueeze(-1) / 50.0)  # Normalize
        quality_emb = self.quality_encoder(quality_features)

        combined = torch.cat([type_emb, gap_emb, quality_emb], dim=-1)
        return self.version_fusion(combined)


class VersionComparator(nn.Module):
    """Compares different versions of the same story"""

    def __init__(self, config: RemakeConfig):
        super().__init__()

        # Pairwise comparison
        self.comparison_net = nn.Sequential(
            nn.Linear(config.hidden_dim * 2, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.hidden_dim)
        )

        # Cross-attention between versions
        self.cross_attention = nn.MultiheadAttention(
            config.hidden_dim, config.num_heads, dropout=config.dropout, batch_first=True
        )

        # Preference predictor
        self.preference_head = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.GELU(),
            nn.Linear(config.hidden_dim // 2, 2),  # original vs remake preference
            nn.Softmax(dim=-1)
        )

    def forward(self, original_repr: torch.Tensor,
                remake_repr: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compare original and remake"""
        # Direct comparison
        comparison = self.comparison_net(torch.cat([original_repr, remake_repr], dim=-1))

        # Cross-attention
        versions = torch.stack([original_repr, remake_repr], dim=1)
        attended, _ = self.cross_attention(versions, versions, versions)
        attended = attended.mean(dim=1)

        # Preference
        preference = self.preference_head(comparison)

        return attended, preference


class RemakeConnectionModel(nn.Module):
    """
    Complete Remake Connection Model for original/remake recommendations.
    """

    def __init__(self, config: Optional[RemakeConfig] = None):
        super().__init__()
        self.config = config or RemakeConfig()

        # Core embeddings
        self.movie_embedding = nn.Embedding(self.config.num_movies, self.config.embedding_dim)
        self.user_embedding = nn.Embedding(self.config.num_users, self.config.hidden_dim)

        # Version encoder
        self.version_encoder = VersionEncoder(self.config)

        # Version comparator
        self.version_comparator = VersionComparator(self.config)

        # Movie projection
        self.movie_proj = nn.Linear(self.config.embedding_dim, self.config.hidden_dim)

        # User version preference
        self.user_version_preference = nn.Sequential(
            nn.Linear(self.config.hidden_dim * 2, self.config.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim)
        )

        # Original preference head (does user prefer originals?)
        self.original_preference_head = nn.Sequential(
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim // 2),
            nn.GELU(),
            nn.Linear(self.config.hidden_dim // 2, 1),
            nn.Sigmoid()
        )

        # Remake quality predictor
        self.quality_predictor = nn.Sequential(
            nn.Linear(self.config.hidden_dim * 2, self.config.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.hidden_dim, 1)
        )

        # Watch order recommender (original first or remake first?)
        self.order_head = nn.Sequential(
            nn.Linear(self.config.hidden_dim * 2, self.config.hidden_dim),
            nn.GELU(),
            nn.Linear(self.config.hidden_dim, 3),  # original_first, remake_first, either
            nn.Softmax(dim=-1)
        )

        # Rating predictor
        self.rating_head = nn.Sequential(
            nn.Linear(self.config.hidden_dim * 3, self.config.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.hidden_dim, 1)
        )

        # Movie recommendation head
        self.movie_head = nn.Linear(self.config.hidden_dim, self.config.num_movies)

        # Discovery head (recommend unseen originals/remakes)
        self.discovery_head = nn.Sequential(
            nn.Linear(self.config.hidden_dim * 2, self.config.hidden_dim),
            nn.GELU(),
            nn.Linear(self.config.hidden_dim, 2)  # recommend_original, recommend_remake
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

    def forward(self, user_ids: torch.Tensor, original_ids: torch.Tensor,
                remake_ids: torch.Tensor, version_types: torch.Tensor,
                year_gaps: torch.Tensor, quality_features: torch.Tensor,
                watched_original: Optional[torch.Tensor] = None,
                watched_remake: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass for remake-aware recommendations.
        """
        batch_size = user_ids.size(0)

        # Get user embedding
        user_emb = self.user_embedding(user_ids)

        # Get movie embeddings
        original_emb = self.movie_embedding(original_ids)
        original_repr = self.movie_proj(original_emb)

        remake_emb = self.movie_embedding(remake_ids)
        remake_repr = self.movie_proj(remake_emb)

        # Encode version relationship
        version_repr = self.version_encoder(version_types, year_gaps, quality_features)

        # Compare versions
        compared_repr, version_preference = self.version_comparator(original_repr, remake_repr)

        # User version preference
        user_version = torch.cat([user_emb, version_repr], dim=-1)
        version_pref = self.user_version_preference(user_version)

        # Original preference
        original_pref = self.original_preference_head(user_emb)

        # Quality prediction for remake
        quality_input = torch.cat([original_repr, remake_repr], dim=-1)
        quality_pred = self.quality_predictor(quality_input)

        # Watch order
        order_input = torch.cat([user_emb, compared_repr], dim=-1)
        order_probs = self.order_head(order_input)

        # Rating prediction
        rating_input = torch.cat([user_emb, original_repr, remake_repr], dim=-1)
        rating_pred = self.rating_head(rating_input)

        # Movie recommendations
        movie_logits = self.movie_head(version_pref)

        # Discovery recommendations
        discovery_input = torch.cat([user_emb, compared_repr], dim=-1)
        discovery_logits = self.discovery_head(discovery_input)

        return {
            'version_preference': version_pref,
            'original_preference': original_pref,
            'version_comparison': version_preference,
            'quality_pred': quality_pred,
            'order_probs': order_probs,
            'rating_pred': rating_pred,
            'movie_logits': movie_logits,
            'discovery_logits': discovery_logits,
            'compared_repr': compared_repr
        }

    def recommend_counterpart(self, user_ids: torch.Tensor, watched_movie_id: torch.Tensor,
                              is_original: bool, counterpart_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Recommend the counterpart (original if watched remake, or vice versa).
        """
        with torch.no_grad():
            user_emb = self.user_embedding(user_ids)

            # Get watched movie repr
            watched_emb = self.movie_embedding(watched_movie_id)
            watched_repr = self.movie_proj(watched_emb)

            # Get counterpart reprs
            counterpart_emb = self.movie_embedding(counterpart_ids)
            counterpart_repr = self.movie_proj(counterpart_emb)

            # Compare
            if is_original:
                compared, preference = self.version_comparator(watched_repr, counterpart_repr)
            else:
                compared, preference = self.version_comparator(counterpart_repr, watched_repr)

            # Score counterparts
            user_version = torch.cat([user_emb.unsqueeze(1).expand(-1, counterpart_repr.size(1), -1),
                                     compared.unsqueeze(1).expand(-1, counterpart_repr.size(1), -1)], dim=-1)
            discovery = self.discovery_head(user_version)

            scores = discovery[:, :, 0] if is_original else discovery[:, :, 1]

        return counterpart_ids, scores

    def get_remake_preference_profile(self, user_ids: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Get user's remake preference profile"""
        with torch.no_grad():
            user_emb = self.user_embedding(user_ids)
            original_pref = self.original_preference_head(user_emb)

        return {
            'original_preference': original_pref,
            'user_embedding': user_emb
        }


class RemakeConnectionTrainer:
    """Trainer for Remake Connection Model"""

    def __init__(self, model: RemakeConnectionModel, lr: float = 1e-4,
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
        self.preference_loss = nn.CrossEntropyLoss()
        self.quality_loss = nn.MSELoss()

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Single training step"""
        self.model.train()

        user_ids = batch['user_ids'].to(self.device)
        original_ids = batch['original_ids'].to(self.device)
        remake_ids = batch['remake_ids'].to(self.device)
        version_types = batch['version_types'].to(self.device)
        year_gaps = batch['year_gaps'].to(self.device)
        quality_features = batch['quality_features'].to(self.device)
        target_ratings = batch['ratings'].to(self.device)

        outputs = self.model(
            user_ids, original_ids, remake_ids, version_types,
            year_gaps, quality_features
        )

        rating_loss = self.rating_loss(outputs['rating_pred'].squeeze(), target_ratings)

        if 'preferred_version' in batch:
            target_preference = batch['preferred_version'].to(self.device)
            pref_loss = self.preference_loss(outputs['version_comparison'], target_preference)
            total_loss = rating_loss + pref_loss
        else:
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
                original_ids = batch['original_ids'].to(self.device)
                remake_ids = batch['remake_ids'].to(self.device)
                version_types = batch['version_types'].to(self.device)
                year_gaps = batch['year_gaps'].to(self.device)
                quality_features = batch['quality_features'].to(self.device)
                target_ratings = batch['ratings'].to(self.device)

                outputs = self.model(
                    user_ids, original_ids, remake_ids, version_types,
                    year_gaps, quality_features
                )

                total_metrics['rating_mse'] += F.mse_loss(
                    outputs['rating_pred'].squeeze(), target_ratings
                ).item()

                num_batches += 1

        return {k: v / num_batches for k, v in total_metrics.items()}
