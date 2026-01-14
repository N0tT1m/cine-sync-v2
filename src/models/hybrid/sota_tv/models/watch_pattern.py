"""
Watch Pattern Model (WPM-TV)
Models user viewing patterns, habits, and temporal preferences.
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
class WatchPatternConfig:
    """Configuration for Watch Pattern Model"""
    embedding_dim: int = 256
    hidden_dim: int = 512
    num_heads: int = 8
    num_layers: int = 4
    num_shows: int = 50000
    num_users: int = 50000
    max_history_length: int = 100
    dropout: float = 0.1


class TemporalPatternEncoder(nn.Module):
    """Encodes temporal viewing patterns"""

    def __init__(self, config: WatchPatternConfig):
        super().__init__()

        # Time of day embedding (24 hours)
        self.hour_embedding = nn.Embedding(24, config.embedding_dim // 4)

        # Day of week embedding
        self.day_embedding = nn.Embedding(7, config.embedding_dim // 4)

        # Month embedding (seasonality)
        self.month_embedding = nn.Embedding(12, config.embedding_dim // 8)

        # Session duration encoder
        self.duration_encoder = nn.Sequential(
            nn.Linear(1, config.embedding_dim // 8),
            nn.GELU(),
            nn.Linear(config.embedding_dim // 8, config.embedding_dim // 8)
        )

        # Inter-session gap encoder
        self.gap_encoder = nn.Sequential(
            nn.Linear(1, config.embedding_dim // 8),
            nn.GELU(),
            nn.Linear(config.embedding_dim // 8, config.embedding_dim // 8)
        )

        # Combine
        # Input: hour(64) + day(64) + month(32) + duration(32) + gap(32) = 224
        temporal_input_dim = config.embedding_dim // 4 * 2 + config.embedding_dim // 8 * 3
        self.temporal_fusion = nn.Sequential(
            nn.Linear(temporal_input_dim, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.hidden_dim)
        )

    def forward(self, hours: torch.Tensor, days: torch.Tensor, months: torch.Tensor,
                durations: torch.Tensor, gaps: torch.Tensor) -> torch.Tensor:
        """Encode temporal pattern"""
        hour_emb = self.hour_embedding(hours)
        day_emb = self.day_embedding(days)
        month_emb = self.month_embedding(months)
        duration_emb = self.duration_encoder(durations.float().unsqueeze(-1) / 60.0)
        gap_emb = self.gap_encoder(gaps.float().unsqueeze(-1) / 24.0)

        combined = torch.cat([hour_emb, day_emb, month_emb, duration_emb, gap_emb], dim=-1)
        return self.temporal_fusion(combined)


class ViewingHabitAnalyzer(nn.Module):
    """Analyzes user viewing habits"""

    def __init__(self, config: WatchPatternConfig):
        super().__init__()

        # Habit type embedding
        self.habit_embedding = nn.Embedding(8, config.embedding_dim // 2)
        # Types: regular_scheduler, binge_watcher, casual_viewer, weekend_warrior,
        #        night_owl, morning_person, multitasker, focused_viewer

        # Consistency encoder
        self.consistency_encoder = nn.Sequential(
            nn.Linear(4, config.embedding_dim // 4),  # watch_regularity, time_consistency, genre_loyalty, completion_rate
            nn.GELU(),
            nn.Linear(config.embedding_dim // 4, config.embedding_dim // 4)
        )

        # Parallel watching encoder (multiple shows at once)
        self.parallel_encoder = nn.Sequential(
            nn.Linear(2, config.embedding_dim // 4),  # concurrent_shows, genre_mixing
            nn.GELU(),
            nn.Linear(config.embedding_dim // 4, config.embedding_dim // 4)
        )

        # Combine
        self.habit_fusion = nn.Sequential(
            nn.Linear(config.embedding_dim, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.hidden_dim)
        )

    def forward(self, habit_types: torch.Tensor, consistency_features: torch.Tensor,
                parallel_features: torch.Tensor) -> torch.Tensor:
        """Analyze habits"""
        habit_emb = self.habit_embedding(habit_types)
        consist_emb = self.consistency_encoder(consistency_features)
        parallel_emb = self.parallel_encoder(parallel_features)

        combined = torch.cat([habit_emb, consist_emb, parallel_emb], dim=-1)
        return self.habit_fusion(combined)


class WatchPatternModel(nn.Module):
    """
    Complete Watch Pattern Model for pattern-aware recommendations.
    """

    def __init__(self, config: Optional[WatchPatternConfig] = None):
        super().__init__()
        self.config = config or WatchPatternConfig()

        # Core embeddings
        self.show_embedding = nn.Embedding(self.config.num_shows, self.config.embedding_dim)
        self.user_embedding = nn.Embedding(self.config.num_users, self.config.hidden_dim)

        # Temporal pattern encoder
        self.temporal_encoder = TemporalPatternEncoder(self.config)

        # Habit analyzer
        self.habit_analyzer = ViewingHabitAnalyzer(self.config)

        # Show projection
        self.show_proj = nn.Linear(self.config.embedding_dim, self.config.hidden_dim)

        # Sequence encoder for watch history
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.config.hidden_dim,
            nhead=self.config.num_heads,
            dim_feedforward=self.config.hidden_dim * 4,
            dropout=self.config.dropout,
            activation='gelu',
            batch_first=True
        )
        self.history_encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.config.num_layers)

        # User pattern preference
        self.user_pattern_preference = nn.Sequential(
            nn.Linear(self.config.hidden_dim * 2, self.config.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim)
        )

        # Optimal watch time predictor
        self.optimal_time_head = nn.Sequential(
            nn.Linear(self.config.hidden_dim * 2, self.config.hidden_dim),
            nn.GELU(),
            nn.Linear(self.config.hidden_dim, 24)  # 24 hours
        )

        # Next watch predictor
        self.next_show_head = nn.Sequential(
            nn.Linear(self.config.hidden_dim * 2, self.config.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.hidden_dim, self.config.num_shows)
        )

        # Watch probability predictor (will user watch today?)
        self.watch_prob_head = nn.Sequential(
            nn.Linear(self.config.hidden_dim * 2, self.config.hidden_dim),
            nn.GELU(),
            nn.Linear(self.config.hidden_dim, 1),
            nn.Sigmoid()
        )

        # Habit classification head
        self.habit_classification_head = nn.Sequential(
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim // 2),
            nn.GELU(),
            nn.Linear(self.config.hidden_dim // 2, 8)  # habit types
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
                hours: torch.Tensor, days: torch.Tensor, months: torch.Tensor,
                durations: torch.Tensor, gaps: torch.Tensor,
                habit_types: torch.Tensor, consistency_features: torch.Tensor,
                parallel_features: torch.Tensor,
                history_shows: Optional[torch.Tensor] = None,
                history_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Forward pass"""
        batch_size = user_ids.size(0)

        # Get embeddings
        user_emb = self.user_embedding(user_ids)
        show_emb = self.show_embedding(show_ids)
        show_repr = self.show_proj(show_emb)

        # Encode temporal pattern
        temporal_repr = self.temporal_encoder(hours, days, months, durations, gaps)

        # Analyze habits
        habit_repr = self.habit_analyzer(habit_types, consistency_features, parallel_features)

        # Process watch history if provided
        if history_shows is not None:
            history_emb = self.show_embedding(history_shows)
            history_repr = self.show_proj(history_emb)
            if history_mask is not None:
                history_encoded = self.history_encoder(history_repr, src_key_padding_mask=~history_mask)
            else:
                history_encoded = self.history_encoder(history_repr)
            history_pooled = history_encoded.mean(dim=1)
        else:
            history_pooled = show_repr

        # User-pattern preference
        user_temporal = torch.cat([user_emb, temporal_repr], dim=-1)
        pattern_pref = self.user_pattern_preference(user_temporal)

        # Optimal watch time
        optimal_time = self.optimal_time_head(user_temporal)

        # Next show prediction
        user_history = torch.cat([user_emb, history_pooled], dim=-1)
        next_show_logits = self.next_show_head(user_history)

        # Watch probability
        watch_prob = self.watch_prob_head(user_temporal)

        # Habit classification
        habit_logits = self.habit_classification_head(habit_repr)

        # Rating prediction
        rating_input = torch.cat([user_emb, show_repr, pattern_pref], dim=-1)
        rating_pred = self.rating_head(rating_input)

        # Show recommendations
        show_logits = self.show_head(pattern_pref)

        return {
            'pattern_preference': pattern_pref,
            'optimal_time': optimal_time,
            'next_show_logits': next_show_logits,
            'watch_prob': watch_prob,
            'habit_logits': habit_logits,
            'rating_pred': rating_pred,
            'show_logits': show_logits
        }

    def recommend_for_time(self, user_ids: torch.Tensor, hour: int, day: int,
                          top_k: int = 10) -> Tuple[torch.Tensor, torch.Tensor]:
        """Recommend shows for specific time"""
        with torch.no_grad():
            user_emb = self.user_embedding(user_ids)
            batch_size = user_ids.size(0)
            device = user_ids.device

            hours = torch.tensor([hour], device=device).expand(batch_size)
            days = torch.tensor([day], device=device).expand(batch_size)
            months = torch.zeros(batch_size, dtype=torch.long, device=device)
            durations = torch.ones(batch_size, device=device) * 30
            gaps = torch.ones(batch_size, device=device) * 24

            temporal_repr = self.temporal_encoder(hours, days, months, durations, gaps)
            user_temporal = torch.cat([user_emb, temporal_repr], dim=-1)
            pref = self.user_pattern_preference(user_temporal)

            show_logits = self.show_head(pref)
            top_scores, top_shows = torch.topk(F.softmax(show_logits, dim=-1), top_k, dim=-1)

        return top_shows, top_scores


class WatchPatternTrainer:
    """Trainer for Watch Pattern Model"""

    def __init__(self, model: WatchPatternModel, lr: float = 1e-4,
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
        self.time_loss = nn.CrossEntropyLoss()
        self.habit_loss = nn.CrossEntropyLoss()

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Single training step"""
        self.model.train()

        user_ids = batch['user_ids'].to(self.device)
        show_ids = batch['show_ids'].to(self.device)
        hours = batch['hours'].to(self.device)
        days = batch['days'].to(self.device)
        months = batch['months'].to(self.device)
        durations = batch['durations'].to(self.device)
        gaps = batch['gaps'].to(self.device)
        habit_types = batch['habit_types'].to(self.device)
        consistency_features = batch['consistency_features'].to(self.device)
        parallel_features = batch['parallel_features'].to(self.device)
        target_ratings = batch['ratings'].to(self.device)

        outputs = self.model(
            user_ids, show_ids, hours, days, months, durations, gaps,
            habit_types, consistency_features, parallel_features
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
                hours = batch['hours'].to(self.device)
                days = batch['days'].to(self.device)
                months = batch['months'].to(self.device)
                durations = batch['durations'].to(self.device)
                gaps = batch['gaps'].to(self.device)
                habit_types = batch['habit_types'].to(self.device)
                consistency_features = batch['consistency_features'].to(self.device)
                parallel_features = batch['parallel_features'].to(self.device)
                target_ratings = batch['ratings'].to(self.device)

                outputs = self.model(
                    user_ids, show_ids, hours, days, months, durations, gaps,
                    habit_types, consistency_features, parallel_features
                )

                total_metrics['rating_mse'] += F.mse_loss(
                    outputs['rating_pred'].squeeze(), target_ratings
                ).item()
                num_batches += 1

        return {k: v / num_batches for k, v in total_metrics.items()}
