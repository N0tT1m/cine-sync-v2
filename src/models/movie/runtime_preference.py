"""
Runtime Preference Model (RPM)
Models user preferences for movie duration and time-aware recommendations.
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
class RuntimeConfig:
    """Configuration for Runtime Preference Model"""
    embedding_dim: int = 256
    hidden_dim: int = 512
    num_heads: int = 8
    num_layers: int = 4
    num_movies: int = 100000
    num_users: int = 50000
    max_runtime_minutes: int = 300
    num_time_slots: int = 24  # Hours in a day
    num_day_types: int = 3  # Weekday, weekend, holiday
    dropout: float = 0.1


class RuntimeEncoder(nn.Module):
    """Encodes movie runtime with context awareness"""

    def __init__(self, config: RuntimeConfig):
        super().__init__()

        # Runtime bucketing (short, medium, long, epic)
        self.runtime_bucket_embedding = nn.Embedding(5, config.embedding_dim)

        # Continuous runtime encoding
        self.runtime_encoder = nn.Sequential(
            nn.Linear(1, config.embedding_dim // 4),
            nn.GELU(),
            nn.Linear(config.embedding_dim // 4, config.embedding_dim // 2)
        )

        # Genre-runtime interaction (some genres have different expected lengths)
        self.genre_runtime_interaction = nn.Parameter(torch.randn(30, config.embedding_dim // 4))

        # Combine features
        self.feature_fusion = nn.Sequential(
            nn.Linear(config.embedding_dim + config.embedding_dim // 2 + config.embedding_dim // 4,
                     config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.hidden_dim)
        )

    def forward(self, runtime_minutes: torch.Tensor, genre_ids: torch.Tensor) -> torch.Tensor:
        """
        Encode runtime with context.

        Args:
            runtime_minutes: Movie runtime in minutes [batch]
            genre_ids: Primary genre IDs [batch]
        """
        # Bucket runtime
        buckets = torch.zeros_like(runtime_minutes, dtype=torch.long)
        buckets = torch.where(runtime_minutes < 90, torch.tensor(0, device=runtime_minutes.device), buckets)
        buckets = torch.where((runtime_minutes >= 90) & (runtime_minutes < 120),
                             torch.tensor(1, device=runtime_minutes.device), buckets)
        buckets = torch.where((runtime_minutes >= 120) & (runtime_minutes < 150),
                             torch.tensor(2, device=runtime_minutes.device), buckets)
        buckets = torch.where((runtime_minutes >= 150) & (runtime_minutes < 180),
                             torch.tensor(3, device=runtime_minutes.device), buckets)
        buckets = torch.where(runtime_minutes >= 180, torch.tensor(4, device=runtime_minutes.device), buckets)

        bucket_emb = self.runtime_bucket_embedding(buckets)

        # Continuous encoding
        runtime_norm = runtime_minutes.float().unsqueeze(-1) / 180.0  # Normalize by typical length
        continuous_emb = self.runtime_encoder(runtime_norm)

        # Genre-runtime interaction
        genre_runtime = self.genre_runtime_interaction[genre_ids.clamp(0, 29)]

        # Combine
        combined = torch.cat([bucket_emb, continuous_emb, genre_runtime], dim=-1)
        return self.feature_fusion(combined)


class TimeContextEncoder(nn.Module):
    """Encodes user's available time context"""

    def __init__(self, config: RuntimeConfig):
        super().__init__()

        # Time slot embedding (hour of day)
        self.time_slot_embedding = nn.Embedding(config.num_time_slots, config.embedding_dim // 4)

        # Day type embedding (weekday/weekend/holiday)
        self.day_type_embedding = nn.Embedding(config.num_day_types, config.embedding_dim // 4)

        # Available time encoder
        self.available_time_encoder = nn.Sequential(
            nn.Linear(1, config.embedding_dim // 4),
            nn.GELU(),
            nn.Linear(config.embedding_dim // 4, config.embedding_dim // 2)
        )

        # Combine
        self.context_fusion = nn.Sequential(
            nn.Linear(config.embedding_dim, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.hidden_dim)
        )

    def forward(self, time_slot: torch.Tensor, day_type: torch.Tensor,
                available_minutes: torch.Tensor) -> torch.Tensor:
        """Encode time context"""
        slot_emb = self.time_slot_embedding(time_slot)
        day_emb = self.day_type_embedding(day_type)
        available_emb = self.available_time_encoder(available_minutes.float().unsqueeze(-1) / 180.0)

        combined = torch.cat([slot_emb, day_emb, available_emb], dim=-1)
        return self.context_fusion(combined)


class RuntimePreferenceModel(nn.Module):
    """
    Complete Runtime Preference Model for time-aware movie recommendations.
    """

    def __init__(self, config: Optional[RuntimeConfig] = None):
        super().__init__()
        self.config = config or RuntimeConfig()

        # Core embeddings
        self.movie_embedding = nn.Embedding(self.config.num_movies, self.config.embedding_dim)
        self.user_embedding = nn.Embedding(self.config.num_users, self.config.hidden_dim)

        # Runtime encoder
        self.runtime_encoder = RuntimeEncoder(self.config)

        # Time context encoder
        self.time_context_encoder = TimeContextEncoder(self.config)

        # Movie projection
        self.movie_proj = nn.Linear(self.config.embedding_dim, self.config.hidden_dim)

        # User runtime preference learning
        self.user_runtime_preference = nn.Sequential(
            nn.Linear(self.config.hidden_dim * 2, self.config.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim)
        )

        # Context-aware preference
        self.context_preference = nn.Sequential(
            nn.Linear(self.config.hidden_dim * 3, self.config.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim)
        )

        # Runtime fit predictor (does movie fit available time?)
        self.fit_head = nn.Sequential(
            nn.Linear(self.config.hidden_dim * 2, self.config.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.hidden_dim, 1),
            nn.Sigmoid()
        )

        # Optimal runtime predictor (what runtime does user prefer?)
        self.optimal_runtime_head = nn.Sequential(
            nn.Linear(self.config.hidden_dim * 2, self.config.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.hidden_dim, 1),
            nn.ReLU()  # Runtime is positive
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

        # Attention satisfaction predictor (can user pay attention for full runtime?)
        self.attention_head = nn.Sequential(
            nn.Linear(self.config.hidden_dim * 2, self.config.hidden_dim),
            nn.GELU(),
            nn.Linear(self.config.hidden_dim, 1),
            nn.Sigmoid()
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

    def forward(self, user_ids: torch.Tensor, movie_ids: torch.Tensor,
                runtime_minutes: torch.Tensor, genre_ids: torch.Tensor,
                time_slot: Optional[torch.Tensor] = None,
                day_type: Optional[torch.Tensor] = None,
                available_minutes: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass for runtime-aware recommendations.
        """
        batch_size = user_ids.size(0)
        device = user_ids.device

        # Get user embedding
        user_emb = self.user_embedding(user_ids)

        # Get movie embedding
        movie_emb = self.movie_embedding(movie_ids)
        movie_repr = self.movie_proj(movie_emb)

        # Encode runtime
        runtime_repr = self.runtime_encoder(runtime_minutes, genre_ids)

        # User-runtime preference
        user_runtime = torch.cat([user_emb, runtime_repr], dim=-1)
        runtime_pref = self.user_runtime_preference(user_runtime)

        # Context-aware preference if context provided
        if time_slot is not None and day_type is not None and available_minutes is not None:
            context_repr = self.time_context_encoder(time_slot, day_type, available_minutes)
            context_input = torch.cat([user_emb, runtime_repr, context_repr], dim=-1)
            context_pref = self.context_preference(context_input)

            # Fit prediction
            fit_input = torch.cat([runtime_repr, context_repr], dim=-1)
            fit_prob = self.fit_head(fit_input)
        else:
            context_pref = runtime_pref
            fit_prob = torch.ones(batch_size, 1, device=device)

        # Optimal runtime for user
        optimal_runtime = self.optimal_runtime_head(user_runtime) * 180  # Scale back

        # Attention satisfaction
        attention_prob = self.attention_head(user_runtime)

        # Rating prediction
        rating_input = torch.cat([user_emb, movie_repr, runtime_pref], dim=-1)
        rating_pred = self.rating_head(rating_input)

        # Movie recommendations
        movie_logits = self.movie_head(context_pref)

        return {
            'runtime_preference': runtime_pref,
            'context_preference': context_pref,
            'fit_prob': fit_prob,
            'optimal_runtime': optimal_runtime,
            'attention_prob': attention_prob,
            'rating_pred': rating_pred,
            'movie_logits': movie_logits
        }

    def recommend_by_time(self, user_ids: torch.Tensor, available_minutes: torch.Tensor,
                          all_movie_runtimes: torch.Tensor, all_movie_genres: torch.Tensor,
                          buffer_minutes: int = 15, top_k: int = 10) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Recommend movies that fit within available time.

        Args:
            available_minutes: Available time in minutes
            buffer_minutes: Extra buffer time to account for
            top_k: Number of recommendations
        """
        with torch.no_grad():
            user_emb = self.user_embedding(user_ids)

            # Filter movies that fit in available time
            max_runtime = available_minutes - buffer_minutes
            valid_movies = all_movie_runtimes <= max_runtime.unsqueeze(-1)

            # Get scores for all movies
            all_movies = torch.arange(self.config.num_movies, device=user_ids.device)
            movie_emb = self.movie_embedding(all_movies)
            movie_repr = self.movie_proj(movie_emb)

            runtime_repr = self.runtime_encoder(all_movie_runtimes[0], all_movie_genres[0])

            # Score movies
            scores = []
            for i in range(user_ids.size(0)):
                user_expanded = user_emb[i:i+1].expand(movie_repr.size(0), -1)
                user_runtime = torch.cat([user_expanded, runtime_repr], dim=-1)
                pref = self.user_runtime_preference(user_runtime)
                movie_scores = self.movie_head(pref).diagonal()

                # Mask invalid movies
                movie_scores = movie_scores.masked_fill(~valid_movies[i], -float('inf'))
                scores.append(movie_scores)

            scores = torch.stack(scores)
            top_scores, top_movies = torch.topk(scores, top_k, dim=-1)

        return top_movies, top_scores

    def get_runtime_distribution(self, user_ids: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Get user's preferred runtime distribution"""
        with torch.no_grad():
            user_emb = self.user_embedding(user_ids)

            # Sample runtimes
            runtimes = torch.tensor([60, 90, 120, 150, 180, 210, 240], device=user_ids.device)
            genre_ids = torch.zeros_like(runtimes)  # Default genre

            preferences = []
            for runtime in runtimes:
                runtime_repr = self.runtime_encoder(
                    runtime.unsqueeze(0).expand(user_ids.size(0)),
                    genre_ids[0:1].expand(user_ids.size(0))
                )
                user_runtime = torch.cat([user_emb, runtime_repr], dim=-1)
                attention = self.attention_head(user_runtime)
                preferences.append(attention)

            preferences = torch.cat(preferences, dim=-1)

        return {
            'runtimes': runtimes,
            'preferences': preferences
        }


class RuntimePreferenceTrainer:
    """Trainer for Runtime Preference Model"""

    def __init__(self, model: RuntimePreferenceModel, lr: float = 1e-4,
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
        self.fit_loss = nn.BCELoss()
        self.runtime_loss = nn.L1Loss()
        self.attention_loss = nn.BCELoss()

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Single training step"""
        self.model.train()

        user_ids = batch['user_ids'].to(self.device)
        movie_ids = batch['movie_ids'].to(self.device)
        runtime_minutes = batch['runtime_minutes'].to(self.device)
        genre_ids = batch['genre_ids'].to(self.device)
        target_ratings = batch['ratings'].to(self.device)

        time_slot = batch.get('time_slot')
        day_type = batch.get('day_type')
        available_minutes = batch.get('available_minutes')

        if time_slot is not None:
            time_slot = time_slot.to(self.device)
            day_type = day_type.to(self.device)
            available_minutes = available_minutes.to(self.device)

        outputs = self.model(
            user_ids, movie_ids, runtime_minutes, genre_ids,
            time_slot, day_type, available_minutes
        )

        # Compute losses
        rating_loss = self.rating_loss(outputs['rating_pred'].squeeze(), target_ratings)

        total_loss = rating_loss

        if 'completed_movie' in batch:
            target_attention = batch['completed_movie'].to(self.device).float()
            attention_loss = self.attention_loss(outputs['attention_prob'].squeeze(), target_attention)
            total_loss += attention_loss * 0.5

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
                movie_ids = batch['movie_ids'].to(self.device)
                runtime_minutes = batch['runtime_minutes'].to(self.device)
                genre_ids = batch['genre_ids'].to(self.device)
                target_ratings = batch['ratings'].to(self.device)

                outputs = self.model(user_ids, movie_ids, runtime_minutes, genre_ids)

                total_metrics['rating_mse'] += F.mse_loss(
                    outputs['rating_pred'].squeeze(), target_ratings
                ).item()

                num_batches += 1

        return {k: v / num_batches for k, v in total_metrics.items()}
