"""
Cast Migration Model (CMM-TV)
Models cast changes across seasons and their impact on show quality.
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
class CastMigrationConfig:
    """Configuration for Cast Migration Model"""
    embedding_dim: int = 256
    hidden_dim: int = 512
    num_heads: int = 8
    num_layers: int = 4
    num_shows: int = 50000
    num_users: int = 50000
    num_actors: int = 100000
    max_cast_size: int = 30
    max_seasons: int = 30
    dropout: float = 0.1


class CastEncoder(nn.Module):
    """Encodes cast composition"""

    def __init__(self, config: CastMigrationConfig):
        super().__init__()

        # Actor embedding
        self.actor_embedding = nn.Embedding(config.num_actors, config.embedding_dim)

        # Role importance embedding
        self.role_embedding = nn.Embedding(6, config.embedding_dim // 4)
        # Roles: lead, main, recurring, guest, cameo, narrator

        # Character status embedding
        self.status_embedding = nn.Embedding(5, config.embedding_dim // 4)
        # Status: active, departed, killed_off, recurring_return, recast

        # Screen time encoder
        self.screentime_encoder = nn.Sequential(
            nn.Linear(1, config.embedding_dim // 4),
            nn.GELU(),
            nn.Linear(config.embedding_dim // 4, config.embedding_dim // 4)
        )

        # Combine
        self.cast_fusion = nn.Sequential(
            nn.Linear(config.embedding_dim + config.embedding_dim // 4 * 3, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.hidden_dim)
        )

    def forward(self, actor_ids: torch.Tensor, role_types: torch.Tensor,
                status_ids: torch.Tensor, screen_time: torch.Tensor) -> torch.Tensor:
        """Encode cast member"""
        actor_emb = self.actor_embedding(actor_ids)
        role_emb = self.role_embedding(role_types)
        status_emb = self.status_embedding(status_ids)
        time_emb = self.screentime_encoder(screen_time.float().unsqueeze(-1))

        combined = torch.cat([actor_emb, role_emb, status_emb, time_emb], dim=-1)
        return self.cast_fusion(combined)


class MigrationAnalyzer(nn.Module):
    """Analyzes cast changes between seasons"""

    def __init__(self, config: CastMigrationConfig):
        super().__init__()

        # Change type embedding
        self.change_embedding = nn.Embedding(8, config.embedding_dim // 2)
        # Types: no_change, addition, departure, recast, promotion, demotion, death, return

        # Change impact encoder
        self.impact_encoder = nn.Sequential(
            nn.Linear(4, config.embedding_dim // 2),  # viewership_impact, critical_impact, fan_reaction, chemistry_change
            nn.GELU(),
            nn.Linear(config.embedding_dim // 2, config.embedding_dim // 2)
        )

        # Cross-season attention
        self.season_attention = nn.MultiheadAttention(
            config.hidden_dim, config.num_heads, dropout=config.dropout, batch_first=True
        )

        # Migration pattern classifier
        self.pattern_classifier = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.GELU(),
            nn.Linear(config.hidden_dim // 2, 5)  # stable, revolving_door, gradual_change, reboot, ensemble_shift
        )

    def forward(self, cast_reprs_by_season: torch.Tensor, change_types: torch.Tensor,
                impact_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Analyze cast migration"""
        change_emb = self.change_embedding(change_types)
        impact_emb = self.impact_encoder(impact_features)

        # Combine with cast representations
        migration_repr = cast_reprs_by_season + change_emb.unsqueeze(2).expand_as(cast_reprs_by_season[:, :, :1, :])

        # Cross-season attention
        batch_size, num_seasons, cast_size, hidden = cast_reprs_by_season.shape
        season_repr = cast_reprs_by_season.mean(dim=2)  # Pool over cast
        attended, _ = self.season_attention(season_repr, season_repr, season_repr)

        # Classify migration pattern
        pattern_logits = self.pattern_classifier(attended.mean(dim=1))

        return attended, pattern_logits


class CastMigrationModel(nn.Module):
    """
    Complete Cast Migration Model for cast-aware recommendations.
    """

    def __init__(self, config: Optional[CastMigrationConfig] = None):
        super().__init__()
        self.config = config or CastMigrationConfig()

        # Core embeddings
        self.show_embedding = nn.Embedding(self.config.num_shows, self.config.embedding_dim)
        self.user_embedding = nn.Embedding(self.config.num_users, self.config.hidden_dim)

        # Cast encoder
        self.cast_encoder = CastEncoder(self.config)

        # Migration analyzer
        self.migration_analyzer = MigrationAnalyzer(self.config)

        # Show projection
        self.show_proj = nn.Linear(self.config.embedding_dim, self.config.hidden_dim)

        # Cast ensemble encoder
        self.ensemble_attention = nn.MultiheadAttention(
            self.config.hidden_dim, self.config.num_heads,
            dropout=self.config.dropout, batch_first=True
        )

        # User cast preference
        self.user_cast_preference = nn.Sequential(
            nn.Linear(self.config.hidden_dim * 2, self.config.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim)
        )

        # Cast stability preference
        self.stability_preference_head = nn.Sequential(
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim // 2),
            nn.GELU(),
            nn.Linear(self.config.hidden_dim // 2, 1),
            nn.Sigmoid()  # 1 = prefers stable cast, 0 = tolerates changes
        )

        # Actor departure impact predictor
        self.departure_impact_head = nn.Sequential(
            nn.Linear(self.config.hidden_dim * 2, self.config.hidden_dim),
            nn.GELU(),
            nn.Linear(self.config.hidden_dim, 1)  # Impact on user enjoyment
        )

        # Chemistry score predictor
        self.chemistry_head = nn.Sequential(
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim // 2),
            nn.GELU(),
            nn.Linear(self.config.hidden_dim // 2, 1),
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

        # Actor recommendation head
        self.actor_head = nn.Linear(self.config.hidden_dim, self.config.num_actors)

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0, std=0.02)

    def encode_cast(self, actor_ids: torch.Tensor, role_types: torch.Tensor,
                   status_ids: torch.Tensor, screen_time: torch.Tensor) -> torch.Tensor:
        """Encode full cast"""
        batch_size, cast_size = actor_ids.shape

        # Encode each cast member
        cast_reprs = self.cast_encoder(
            actor_ids.view(-1), role_types.view(-1),
            status_ids.view(-1), screen_time.view(-1)
        )
        cast_reprs = cast_reprs.view(batch_size, cast_size, -1)

        # Ensemble attention
        ensemble_repr, _ = self.ensemble_attention(cast_reprs, cast_reprs, cast_reprs)

        return ensemble_repr

    def forward(self, user_ids: torch.Tensor, show_ids: torch.Tensor,
                actor_ids: torch.Tensor, role_types: torch.Tensor,
                status_ids: torch.Tensor, screen_time: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass"""
        batch_size = user_ids.size(0)

        # Get embeddings
        user_emb = self.user_embedding(user_ids)
        show_emb = self.show_embedding(show_ids)
        show_repr = self.show_proj(show_emb)

        # Encode cast
        cast_repr = self.encode_cast(actor_ids, role_types, status_ids, screen_time)
        cast_pooled = cast_repr.mean(dim=1)

        # User-cast preference
        user_cast = torch.cat([user_emb, cast_pooled], dim=-1)
        cast_pref = self.user_cast_preference(user_cast)

        # Stability preference
        stability_pref = self.stability_preference_head(user_emb)

        # Chemistry score
        chemistry = self.chemistry_head(cast_pooled)

        # Rating prediction
        rating_input = torch.cat([user_emb, show_repr, cast_pooled], dim=-1)
        rating_pred = self.rating_head(rating_input)

        # Show recommendations
        show_logits = self.show_head(cast_pref)

        # Actor recommendations
        actor_logits = self.actor_head(user_emb)

        return {
            'cast_preference': cast_pref,
            'cast_repr': cast_repr,
            'stability_preference': stability_pref,
            'chemistry': chemistry,
            'rating_pred': rating_pred,
            'show_logits': show_logits,
            'actor_logits': actor_logits
        }

    def recommend_by_cast_stability(self, user_ids: torch.Tensor, stable_only: bool = True,
                                    top_k: int = 10) -> Tuple[torch.Tensor, torch.Tensor]:
        """Recommend shows based on cast stability preference"""
        with torch.no_grad():
            user_emb = self.user_embedding(user_ids)
            show_logits = self.show_head(user_emb)
            top_scores, top_shows = torch.topk(F.softmax(show_logits, dim=-1), top_k, dim=-1)
        return top_shows, top_scores

    def find_shows_by_actor(self, actor_id: int, top_k: int = 10) -> Tuple[torch.Tensor, torch.Tensor]:
        """Find shows featuring a specific actor"""
        with torch.no_grad():
            actor_emb = self.cast_encoder.actor_embedding.weight[actor_id]
            # This would need show-actor mapping in production
            return torch.tensor([]), torch.tensor([])


class CastMigrationTrainer:
    """Trainer for Cast Migration Model"""

    def __init__(self, model: CastMigrationModel, lr: float = 1e-4,
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
        self.chemistry_loss = nn.MSELoss()

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Single training step"""
        self.model.train()

        user_ids = batch['user_ids'].to(self.device)
        show_ids = batch['show_ids'].to(self.device)
        actor_ids = batch['actor_ids'].to(self.device)
        role_types = batch['role_types'].to(self.device)
        status_ids = batch['status_ids'].to(self.device)
        screen_time = batch['screen_time'].to(self.device)
        target_ratings = batch['ratings'].to(self.device)

        outputs = self.model(
            user_ids, show_ids, actor_ids, role_types, status_ids, screen_time
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
                actor_ids = batch['actor_ids'].to(self.device)
                role_types = batch['role_types'].to(self.device)
                status_ids = batch['status_ids'].to(self.device)
                screen_time = batch['screen_time'].to(self.device)
                target_ratings = batch['ratings'].to(self.device)

                outputs = self.model(
                    user_ids, show_ids, actor_ids, role_types, status_ids, screen_time
                )

                total_metrics['rating_mse'] += F.mse_loss(
                    outputs['rating_pred'].squeeze(), target_ratings
                ).item()
                num_batches += 1

        return {k: v / num_batches for k, v in total_metrics.items()}
