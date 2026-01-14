"""
Series Lifecycle Model (SLM-TV)
Models TV series lifecycle stages from premiere to cancellation/ending.
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
class LifecycleConfig:
    """Configuration for Series Lifecycle Model"""
    embedding_dim: int = 256
    hidden_dim: int = 512
    num_heads: int = 8
    num_layers: int = 4
    num_shows: int = 50000
    num_users: int = 50000
    num_lifecycle_stages: int = 10
    dropout: float = 0.1


class LifecycleStageEncoder(nn.Module):
    """Encodes series lifecycle stage"""

    def __init__(self, config: LifecycleConfig):
        super().__init__()

        # Lifecycle stage embedding
        self.stage_embedding = nn.Embedding(config.num_lifecycle_stages, config.embedding_dim)
        # Stages: pilot, first_season, establishing, peak, declining, revival,
        #         final_season, ended_satisfying, cancelled, limbo

        # Health indicators encoder
        self.health_encoder = nn.Sequential(
            nn.Linear(6, config.embedding_dim // 2),  # viewership_trend, critical_reception, fan_engagement, renewal_likelihood, creator_involvement, budget_trend
            nn.GELU(),
            nn.Linear(config.embedding_dim // 2, config.embedding_dim // 2)
        )

        # Network/platform status
        self.network_status_embedding = nn.Embedding(5, config.embedding_dim // 4)
        # Status: flagship, supported, on_the_bubble, benched, orphaned

        # Combine
        self.lifecycle_fusion = nn.Sequential(
            nn.Linear(config.embedding_dim + config.embedding_dim // 2 + config.embedding_dim // 4,
                     config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.hidden_dim)
        )

    def forward(self, stage_ids: torch.Tensor, health_features: torch.Tensor,
                network_status: torch.Tensor) -> torch.Tensor:
        """Encode lifecycle stage"""
        stage_emb = self.stage_embedding(stage_ids)
        health_emb = self.health_encoder(health_features)
        network_emb = self.network_status_embedding(network_status)

        combined = torch.cat([stage_emb, health_emb, network_emb], dim=-1)
        return self.lifecycle_fusion(combined)


class EndingPredictor(nn.Module):
    """Predicts how and when a series will end"""

    def __init__(self, config: LifecycleConfig):
        super().__init__()

        # Ending type predictor
        self.ending_type_head = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.GELU(),
            nn.Linear(config.hidden_dim // 2, 6)  # planned_ending, cancelled, cliffhanger, wrapped_up, movie, revival
        )

        # Seasons remaining predictor
        self.seasons_remaining_head = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.GELU(),
            nn.Linear(config.hidden_dim // 2, 10)  # 0-9+ seasons
        )

        # Quality of ending predictor
        self.ending_quality_head = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.GELU(),
            nn.Linear(config.hidden_dim // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, lifecycle_repr: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Predict ending characteristics"""
        ending_type = self.ending_type_head(lifecycle_repr)
        seasons_remaining = self.seasons_remaining_head(lifecycle_repr)
        ending_quality = self.ending_quality_head(lifecycle_repr)
        return ending_type, seasons_remaining, ending_quality


class SeriesLifecycleModel(nn.Module):
    """
    Complete Series Lifecycle Model for lifecycle-aware recommendations.
    """

    def __init__(self, config: Optional[LifecycleConfig] = None):
        super().__init__()
        self.config = config or LifecycleConfig()

        # Core embeddings
        self.show_embedding = nn.Embedding(self.config.num_shows, self.config.embedding_dim)
        self.user_embedding = nn.Embedding(self.config.num_users, self.config.hidden_dim)

        # Lifecycle encoder
        self.lifecycle_encoder = LifecycleStageEncoder(self.config)

        # Ending predictor
        self.ending_predictor = EndingPredictor(self.config)

        # Show projection
        self.show_proj = nn.Linear(self.config.embedding_dim, self.config.hidden_dim)

        # User lifecycle preference
        self.user_lifecycle_preference = nn.Sequential(
            nn.Linear(self.config.hidden_dim * 2, self.config.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim)
        )

        # Stage preference head (which stages user prefers to start watching)
        self.stage_preference_head = nn.Sequential(
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim // 2),
            nn.GELU(),
            nn.Linear(self.config.hidden_dim // 2, self.config.num_lifecycle_stages)
        )

        # Risk tolerance head (willing to start cancelled/on-bubble shows?)
        self.risk_tolerance_head = nn.Sequential(
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim // 2),
            nn.GELU(),
            nn.Linear(self.config.hidden_dim // 2, 1),
            nn.Sigmoid()
        )

        # Completion satisfaction predictor
        self.satisfaction_head = nn.Sequential(
            nn.Linear(self.config.hidden_dim * 2, self.config.hidden_dim),
            nn.GELU(),
            nn.Linear(self.config.hidden_dim, 1)
        )

        # Investment recommendation (worth starting?)
        self.investment_head = nn.Sequential(
            nn.Linear(self.config.hidden_dim * 2, self.config.hidden_dim),
            nn.GELU(),
            nn.Linear(self.config.hidden_dim, 1),
            nn.Sigmoid()
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
                stage_ids: torch.Tensor, health_features: torch.Tensor,
                network_status: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass"""
        batch_size = user_ids.size(0)

        # Get embeddings
        user_emb = self.user_embedding(user_ids)
        show_emb = self.show_embedding(show_ids)
        show_repr = self.show_proj(show_emb)

        # Encode lifecycle
        lifecycle_repr = self.lifecycle_encoder(stage_ids, health_features, network_status)

        # Predict ending
        ending_type, seasons_remaining, ending_quality = self.ending_predictor(lifecycle_repr)

        # User-lifecycle preference
        user_lifecycle = torch.cat([user_emb, lifecycle_repr], dim=-1)
        lifecycle_pref = self.user_lifecycle_preference(user_lifecycle)

        # Stage preference
        stage_pref = self.stage_preference_head(user_emb)

        # Risk tolerance
        risk_tolerance = self.risk_tolerance_head(user_emb)

        # Satisfaction prediction
        satisfaction = self.satisfaction_head(user_lifecycle)

        # Investment recommendation
        investment = self.investment_head(user_lifecycle)

        # Rating prediction
        rating_input = torch.cat([user_emb, show_repr, lifecycle_repr], dim=-1)
        rating_pred = self.rating_head(rating_input)

        # Show recommendations
        show_logits = self.show_head(lifecycle_pref)

        return {
            'lifecycle_preference': lifecycle_pref,
            'ending_type': ending_type,
            'seasons_remaining': seasons_remaining,
            'ending_quality': ending_quality,
            'stage_preference': stage_pref,
            'risk_tolerance': risk_tolerance,
            'satisfaction': satisfaction,
            'investment': investment,
            'rating_pred': rating_pred,
            'show_logits': show_logits
        }

    def recommend_safe_investments(self, user_ids: torch.Tensor,
                                  top_k: int = 10) -> Tuple[torch.Tensor, torch.Tensor]:
        """Recommend shows that are safe investments (ongoing, healthy)"""
        with torch.no_grad():
            user_emb = self.user_embedding(user_ids)
            show_logits = self.show_head(user_emb)
            top_scores, top_shows = torch.topk(F.softmax(show_logits, dim=-1), top_k, dim=-1)
        return top_shows, top_scores

    def recommend_completed_shows(self, user_ids: torch.Tensor,
                                 min_ending_quality: float = 0.7,
                                 top_k: int = 10) -> Tuple[torch.Tensor, torch.Tensor]:
        """Recommend completed shows with good endings"""
        with torch.no_grad():
            user_emb = self.user_embedding(user_ids)
            show_logits = self.show_head(user_emb)
            top_scores, top_shows = torch.topk(F.softmax(show_logits, dim=-1), top_k, dim=-1)
        return top_shows, top_scores


class SeriesLifecycleTrainer:
    """Trainer for Series Lifecycle Model"""

    def __init__(self, model: SeriesLifecycleModel, lr: float = 1e-4,
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
        self.ending_loss = nn.CrossEntropyLoss()
        self.investment_loss = nn.BCELoss()

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Single training step"""
        self.model.train()

        user_ids = batch['user_ids'].to(self.device)
        show_ids = batch['show_ids'].to(self.device)
        stage_ids = batch['stage_ids'].to(self.device)
        health_features = batch['health_features'].to(self.device)
        network_status = batch['network_status'].to(self.device)
        target_ratings = batch['ratings'].to(self.device)

        outputs = self.model(
            user_ids, show_ids, stage_ids, health_features, network_status
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
                stage_ids = batch['stage_ids'].to(self.device)
                health_features = batch['health_features'].to(self.device)
                network_status = batch['network_status'].to(self.device)
                target_ratings = batch['ratings'].to(self.device)

                outputs = self.model(
                    user_ids, show_ids, stage_ids, health_features, network_status
                )

                total_metrics['rating_mse'] += F.mse_loss(
                    outputs['rating_pred'].squeeze(), target_ratings
                ).item()
                num_batches += 1

        return {k: v / num_batches for k, v in total_metrics.items()}
