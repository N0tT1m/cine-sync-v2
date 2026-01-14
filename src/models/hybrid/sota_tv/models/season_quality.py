"""
Season Quality Model (SQM-TV)
Models season-level quality variance and seasonal preferences.
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
class SeasonQualityConfig:
    """Configuration for Season Quality Model"""
    embedding_dim: int = 256
    hidden_dim: int = 512
    num_heads: int = 8
    num_layers: int = 4
    num_shows: int = 50000
    num_users: int = 50000
    max_seasons: int = 30
    dropout: float = 0.1


class SeasonEncoder(nn.Module):
    """Encodes individual season characteristics"""

    def __init__(self, config: SeasonQualityConfig):
        super().__init__()

        # Season position embedding
        self.position_embedding = nn.Embedding(config.max_seasons, config.embedding_dim // 2)

        # Quality features encoder
        self.quality_encoder = nn.Sequential(
            nn.Linear(6, config.embedding_dim // 2),  # rating, episode_count, avg_viewership, critic_score, audience_score, production_budget
            nn.GELU(),
            nn.Linear(config.embedding_dim // 2, config.embedding_dim // 2)
        )

        # Season type embedding
        self.type_embedding = nn.Embedding(6, config.embedding_dim // 4)
        # Types: premiere, standard, final, revival, spinoff_intro, crossover_heavy

        # Showrunner change indicator
        self.change_encoder = nn.Sequential(
            nn.Linear(3, config.embedding_dim // 4),  # showrunner_change, cast_change_ratio, tone_shift
            nn.GELU(),
            nn.Linear(config.embedding_dim // 4, config.embedding_dim // 4)
        )

        # Combine
        self.season_fusion = nn.Sequential(
            nn.Linear(config.embedding_dim + config.embedding_dim // 2, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.hidden_dim)
        )

    def forward(self, season_positions: torch.Tensor, quality_features: torch.Tensor,
                season_types: torch.Tensor, change_features: torch.Tensor) -> torch.Tensor:
        """Encode season"""
        pos_emb = self.position_embedding(season_positions)
        quality_emb = self.quality_encoder(quality_features)
        type_emb = self.type_embedding(season_types)
        change_emb = self.change_encoder(change_features)

        combined = torch.cat([pos_emb, quality_emb, type_emb, change_emb], dim=-1)
        return self.season_fusion(combined)


class SeasonTrajectoryAnalyzer(nn.Module):
    """Analyzes show quality trajectory across seasons"""

    def __init__(self, config: SeasonQualityConfig):
        super().__init__()

        # Bidirectional LSTM for trajectory
        self.trajectory_lstm = nn.LSTM(
            config.hidden_dim, config.hidden_dim // 2,
            num_layers=2, batch_first=True, bidirectional=True,
            dropout=config.dropout
        )

        # Trajectory type classifier
        self.trajectory_classifier = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.GELU(),
            nn.Linear(config.hidden_dim // 2, 6)  # consistent, improving, declining, peaked_early, peaked_late, volatile
        )

        # Peak season predictor
        self.peak_predictor = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.GELU(),
            nn.Linear(config.hidden_dim // 2, config.max_seasons)
        )

    def forward(self, season_reprs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Analyze trajectory"""
        trajectory_out, (h_n, _) = self.trajectory_lstm(season_reprs)
        final_state = h_n[-2:].transpose(0, 1).contiguous().view(season_reprs.size(0), -1)

        trajectory_type = self.trajectory_classifier(final_state)
        peak_logits = self.peak_predictor(final_state)

        return trajectory_out, trajectory_type, peak_logits


class SeasonQualityModel(nn.Module):
    """
    Complete Season Quality Model for season-aware recommendations.
    """

    def __init__(self, config: Optional[SeasonQualityConfig] = None):
        super().__init__()
        self.config = config or SeasonQualityConfig()

        # Core embeddings
        self.show_embedding = nn.Embedding(self.config.num_shows, self.config.embedding_dim)
        self.user_embedding = nn.Embedding(self.config.num_users, self.config.hidden_dim)

        # Season encoder
        self.season_encoder = SeasonEncoder(self.config)

        # Trajectory analyzer
        self.trajectory_analyzer = SeasonTrajectoryAnalyzer(self.config)

        # Show projection
        self.show_proj = nn.Linear(self.config.embedding_dim, self.config.hidden_dim)

        # User season preference
        self.user_season_preference = nn.Sequential(
            nn.Linear(self.config.hidden_dim * 2, self.config.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim)
        )

        # Season quality tolerance (how much quality drop user accepts)
        self.quality_tolerance_head = nn.Sequential(
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim // 2),
            nn.GELU(),
            nn.Linear(self.config.hidden_dim // 2, 1),
            nn.Sigmoid()
        )

        # Optimal stop season predictor
        self.stop_season_head = nn.Sequential(
            nn.Linear(self.config.hidden_dim * 2, self.config.hidden_dim),
            nn.GELU(),
            nn.Linear(self.config.hidden_dim, self.config.max_seasons)
        )

        # Season-specific rating predictor
        self.season_rating_head = nn.Sequential(
            nn.Linear(self.config.hidden_dim * 2, self.config.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.hidden_dim, 1)
        )

        # Show recommendation head
        self.show_head = nn.Linear(self.config.hidden_dim, self.config.num_shows)

        # Trajectory preference head
        self.trajectory_pref_head = nn.Sequential(
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim // 2),
            nn.GELU(),
            nn.Linear(self.config.hidden_dim // 2, 6)  # trajectory types
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
                season_positions: torch.Tensor, quality_features: torch.Tensor,
                season_types: torch.Tensor, change_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass"""
        batch_size = user_ids.size(0)

        # Get embeddings
        user_emb = self.user_embedding(user_ids)
        show_emb = self.show_embedding(show_ids)
        show_repr = self.show_proj(show_emb)

        # Encode seasons
        season_repr = self.season_encoder(season_positions, quality_features,
                                         season_types, change_features)

        # Analyze trajectory
        if season_repr.dim() == 3:
            trajectory_repr, trajectory_type, peak_logits = self.trajectory_analyzer(season_repr)
            season_pooled = season_repr.mean(dim=1)
        else:
            trajectory_repr = None
            trajectory_type = None
            peak_logits = None
            season_pooled = season_repr

        # User-season preference
        user_season = torch.cat([user_emb, season_pooled], dim=-1)
        season_pref = self.user_season_preference(user_season)

        # Quality tolerance
        quality_tolerance = self.quality_tolerance_head(user_emb)

        # Optimal stop season
        stop_season_logits = self.stop_season_head(user_season)

        # Season rating
        season_rating = self.season_rating_head(user_season)

        # Show recommendations
        show_logits = self.show_head(season_pref)

        # Trajectory preference
        trajectory_pref = self.trajectory_pref_head(user_emb)

        return {
            'season_preference': season_pref,
            'quality_tolerance': quality_tolerance,
            'stop_season_logits': stop_season_logits,
            'season_rating': season_rating,
            'trajectory_type': trajectory_type,
            'peak_logits': peak_logits,
            'trajectory_pref': trajectory_pref,
            'show_logits': show_logits
        }


class SeasonQualityTrainer:
    """Trainer for Season Quality Model"""

    def __init__(self, model: SeasonQualityModel, lr: float = 1e-4,
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
        self.trajectory_loss = nn.CrossEntropyLoss()

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Single training step"""
        self.model.train()

        user_ids = batch['user_ids'].to(self.device)
        show_ids = batch['show_ids'].to(self.device)
        season_positions = batch['season_positions'].to(self.device)
        quality_features = batch['quality_features'].to(self.device)
        season_types = batch['season_types'].to(self.device)
        change_features = batch['change_features'].to(self.device)
        target_ratings = batch['ratings'].to(self.device)

        outputs = self.model(
            user_ids, show_ids, season_positions, quality_features,
            season_types, change_features
        )

        rating_loss = self.rating_loss(outputs['season_rating'].squeeze(), target_ratings)
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
                season_positions = batch['season_positions'].to(self.device)
                quality_features = batch['quality_features'].to(self.device)
                season_types = batch['season_types'].to(self.device)
                change_features = batch['change_features'].to(self.device)
                target_ratings = batch['ratings'].to(self.device)

                outputs = self.model(
                    user_ids, show_ids, season_positions, quality_features,
                    season_types, change_features
                )

                total_metrics['rating_mse'] += F.mse_loss(
                    outputs['season_rating'].squeeze(), target_ratings
                ).item()
                num_batches += 1

        return {k: v / num_batches for k, v in total_metrics.items()}
