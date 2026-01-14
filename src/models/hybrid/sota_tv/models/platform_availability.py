"""
Platform Availability Model (PAM-TV)
Models streaming platform awareness and availability-based recommendations.
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
class PlatformConfig:
    """Configuration for Platform Availability Model"""
    embedding_dim: int = 256
    hidden_dim: int = 512
    num_heads: int = 8
    num_layers: int = 4
    num_shows: int = 50000
    num_users: int = 50000
    num_platforms: int = 50  # Netflix, Hulu, Disney+, etc.
    num_regions: int = 50
    dropout: float = 0.1


class PlatformEncoder(nn.Module):
    """Encodes streaming platform characteristics"""

    def __init__(self, config: PlatformConfig):
        super().__init__()

        # Platform embedding
        self.platform_embedding = nn.Embedding(config.num_platforms, config.embedding_dim)

        # Platform type embedding
        self.type_embedding = nn.Embedding(6, config.embedding_dim // 4)
        # Types: subscription, ad_supported, premium, free, rental, purchase

        # Platform features encoder
        self.features_encoder = nn.Sequential(
            nn.Linear(6, config.embedding_dim // 4),  # content_library_size, original_content_ratio, price, simultaneous_streams, download_support, 4k_available
            nn.GELU(),
            nn.Linear(config.embedding_dim // 4, config.embedding_dim // 4)
        )

        # Region availability
        self.region_embedding = nn.Embedding(config.num_regions, config.embedding_dim // 4)

        # Combine
        self.platform_fusion = nn.Sequential(
            nn.Linear(config.embedding_dim + config.embedding_dim // 4 * 2, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.hidden_dim)
        )

    def forward(self, platform_ids: torch.Tensor, platform_types: torch.Tensor,
                platform_features: torch.Tensor, region_ids: torch.Tensor) -> torch.Tensor:
        """Encode platform"""
        platform_emb = self.platform_embedding(platform_ids)
        type_emb = self.type_embedding(platform_types)
        feat_emb = self.features_encoder(platform_features)
        region_emb = self.region_embedding(region_ids)

        combined = torch.cat([platform_emb, type_emb, feat_emb, region_emb], dim=-1)
        return self.platform_fusion(combined)


class AvailabilityTracker(nn.Module):
    """Tracks show availability across platforms"""

    def __init__(self, config: PlatformConfig):
        super().__init__()

        # Availability status embedding
        self.status_embedding = nn.Embedding(5, config.embedding_dim // 4)
        # Status: full, partial, leaving_soon, coming_soon, not_available

        # Exclusivity encoder
        self.exclusivity_encoder = nn.Sequential(
            nn.Linear(3, config.embedding_dim // 4),  # is_exclusive, exclusive_window_days, co_production
            nn.GELU(),
            nn.Linear(config.embedding_dim // 4, config.embedding_dim // 4)
        )

        # Cross-platform attention
        self.platform_attention = nn.MultiheadAttention(
            config.hidden_dim, config.num_heads, dropout=config.dropout, batch_first=True
        )

    def forward(self, availability_status: torch.Tensor, exclusivity_features: torch.Tensor,
                platform_reprs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Track availability"""
        status_emb = self.status_embedding(availability_status)
        excl_emb = self.exclusivity_encoder(exclusivity_features)

        # Attention over platforms
        attended, attention_weights = self.platform_attention(
            platform_reprs, platform_reprs, platform_reprs
        )

        return attended, attention_weights


class PlatformAvailabilityModel(nn.Module):
    """
    Complete Platform Availability Model for platform-aware recommendations.
    """

    def __init__(self, config: Optional[PlatformConfig] = None):
        super().__init__()
        self.config = config or PlatformConfig()

        # Core embeddings
        self.show_embedding = nn.Embedding(self.config.num_shows, self.config.embedding_dim)
        self.user_embedding = nn.Embedding(self.config.num_users, self.config.hidden_dim)

        # Platform encoder
        self.platform_encoder = PlatformEncoder(self.config)

        # Availability tracker
        self.availability_tracker = AvailabilityTracker(self.config)

        # Show projection
        self.show_proj = nn.Linear(self.config.embedding_dim, self.config.hidden_dim)

        # User platform preference
        self.user_platform_preference = nn.Sequential(
            nn.Linear(self.config.hidden_dim * 2, self.config.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim)
        )

        # Platform affinity head
        self.platform_affinity_head = nn.Sequential(
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim // 2),
            nn.GELU(),
            nn.Linear(self.config.hidden_dim // 2, self.config.num_platforms)
        )

        # Subscription value predictor
        self.subscription_value_head = nn.Sequential(
            nn.Linear(self.config.hidden_dim * 2, self.config.hidden_dim),
            nn.GELU(),
            nn.Linear(self.config.hidden_dim, self.config.num_platforms)
        )

        # Cross-platform recommendation (which platform to watch on)
        self.platform_choice_head = nn.Sequential(
            nn.Linear(self.config.hidden_dim * 2, self.config.hidden_dim),
            nn.GELU(),
            nn.Linear(self.config.hidden_dim, self.config.num_platforms)
        )

        # Rating predictor
        self.rating_head = nn.Sequential(
            nn.Linear(self.config.hidden_dim * 3, self.config.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.hidden_dim, 1)
        )

        # Show recommendation head
        self.show_head = nn.Linear(self.config.hidden_dim, self.config.num_shows)

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
                platform_ids: torch.Tensor, platform_types: torch.Tensor,
                platform_features: torch.Tensor, region_ids: torch.Tensor,
                user_subscriptions: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Forward pass"""
        batch_size = user_ids.size(0)

        # Get embeddings
        user_emb = self.user_embedding(user_ids)
        show_emb = self.show_embedding(show_ids)
        show_repr = self.show_proj(show_emb)

        # Encode platform
        platform_repr = self.platform_encoder(platform_ids, platform_types,
                                             platform_features, region_ids)

        # User-platform preference
        user_platform = torch.cat([user_emb, platform_repr], dim=-1)
        platform_pref = self.user_platform_preference(user_platform)

        # Platform affinity
        platform_affinity = self.platform_affinity_head(user_emb)

        # Subscription value
        subscription_value = self.subscription_value_head(user_platform)

        # Platform choice for show
        show_platform = torch.cat([show_repr, platform_repr], dim=-1)
        platform_choice = self.platform_choice_head(show_platform)

        # Rating prediction
        rating_input = torch.cat([user_emb, show_repr, platform_repr], dim=-1)
        rating_pred = self.rating_head(rating_input)

        # Show recommendations
        show_logits = self.show_head(platform_pref)

        # Filter by user subscriptions if provided
        if user_subscriptions is not None:
            # Boost shows available on subscribed platforms
            pass

        return {
            'platform_preference': platform_pref,
            'platform_affinity': platform_affinity,
            'subscription_value': subscription_value,
            'platform_choice': platform_choice,
            'rating_pred': rating_pred,
            'show_logits': show_logits
        }

    def recommend_for_platform(self, user_ids: torch.Tensor, platform_id: int,
                              top_k: int = 10) -> Tuple[torch.Tensor, torch.Tensor]:
        """Recommend shows available on specific platform"""
        with torch.no_grad():
            user_emb = self.user_embedding(user_ids)
            platform_emb = self.platform_encoder.platform_embedding.weight[platform_id]

            user_platform = torch.cat([user_emb, platform_emb.expand(user_ids.size(0), -1)], dim=-1)
            pref = self.user_platform_preference(user_platform)

            show_logits = self.show_head(pref)
            top_scores, top_shows = torch.topk(F.softmax(show_logits, dim=-1), top_k, dim=-1)

        return top_shows, top_scores


class PlatformAvailabilityTrainer:
    """Trainer for Platform Availability Model"""

    def __init__(self, model: PlatformAvailabilityModel, lr: float = 1e-4,
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
        self.platform_loss = nn.CrossEntropyLoss()

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Single training step"""
        self.model.train()

        user_ids = batch['user_ids'].to(self.device)
        show_ids = batch['show_ids'].to(self.device)
        platform_ids = batch['platform_ids'].to(self.device)
        platform_types = batch['platform_types'].to(self.device)
        platform_features = batch['platform_features'].to(self.device)
        region_ids = batch['region_ids'].to(self.device)
        target_ratings = batch['ratings'].to(self.device)

        outputs = self.model(
            user_ids, show_ids, platform_ids, platform_types,
            platform_features, region_ids
        )

        rating_loss = self.rating_loss(outputs['rating_pred'].squeeze(), target_ratings)
        total_loss = rating_loss

        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()

        return {'total_loss': total_loss.item(), 'rating_loss': rating_loss.item()}

    def evaluate(self, dataloader) -> Dict[str, float]:
        """Evaluate model"""
        self.model.eval()
        total_metrics = {'rating_mse': 0}
        num_batches = 0

        with torch.no_grad():
            for batch in dataloader:
                user_ids = batch['user_ids'].to(self.device)
                show_ids = batch['show_ids'].to(self.device)
                platform_ids = batch['platform_ids'].to(self.device)
                platform_types = batch['platform_types'].to(self.device)
                platform_features = batch['platform_features'].to(self.device)
                region_ids = batch['region_ids'].to(self.device)
                target_ratings = batch['ratings'].to(self.device)

                outputs = self.model(
                    user_ids, show_ids, platform_ids, platform_types,
                    platform_features, region_ids
                )

                total_metrics['rating_mse'] += F.mse_loss(
                    outputs['rating_pred'].squeeze(), target_ratings
                ).item()
                num_batches += 1

        return {k: v / num_batches for k, v in total_metrics.items()}
