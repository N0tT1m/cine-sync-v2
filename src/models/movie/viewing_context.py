"""
Viewing Context Model (VCM)
Models context-aware recommendations based on viewing situation, mood, and social context.
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
class ViewingContextConfig:
    """Configuration for Viewing Context Model"""
    embedding_dim: int = 256
    hidden_dim: int = 512
    num_heads: int = 8
    num_layers: int = 4
    num_movies: int = 100000
    num_users: int = 50000
    num_contexts: int = 30  # Different viewing contexts
    num_moods: int = 20
    num_occasions: int = 25
    dropout: float = 0.1


class ContextEncoder(nn.Module):
    """Encodes viewing context features"""

    def __init__(self, config: ViewingContextConfig):
        super().__init__()

        # Social context embedding
        self.social_embedding = nn.Embedding(8, config.embedding_dim // 4)
        # Contexts: alone, couple, family, friends, party, date_night, kids_present, mixed_ages

        # Time context embedding
        self.time_embedding = nn.Embedding(8, config.embedding_dim // 4)
        # Times: morning, afternoon, evening, late_night, weekend, weekday, holiday, special_occasion

        # Device/venue embedding
        self.venue_embedding = nn.Embedding(10, config.embedding_dim // 4)
        # Venues: home_tv, home_projector, laptop, phone, tablet, cinema, airplane, hotel, outdoor, vr

        # Duration constraint encoding
        self.duration_encoder = nn.Sequential(
            nn.Linear(2, config.embedding_dim // 8),  # available_time, commitment_level
            nn.GELU(),
            nn.Linear(config.embedding_dim // 8, config.embedding_dim // 8)
        )

        # Combine
        self.context_fusion = nn.Sequential(
            nn.Linear(config.embedding_dim // 4 * 3 + config.embedding_dim // 8, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.hidden_dim)
        )

    def forward(self, social_context: torch.Tensor, time_context: torch.Tensor,
                venue: torch.Tensor, duration_features: torch.Tensor) -> torch.Tensor:
        """Encode viewing context"""
        # Clamp indices to valid embedding range to prevent out-of-bounds errors
        social_context = social_context.clamp(0, 7)
        time_context = time_context.clamp(0, 7)
        venue = venue.clamp(0, 9)

        social_emb = self.social_embedding(social_context)
        time_emb = self.time_embedding(time_context)
        venue_emb = self.venue_embedding(venue)
        duration_emb = self.duration_encoder(duration_features)

        combined = torch.cat([social_emb, time_emb, venue_emb, duration_emb], dim=-1)
        return self.context_fusion(combined)


class MoodEncoder(nn.Module):
    """Encodes user mood and emotional state"""

    def __init__(self, config: ViewingContextConfig):
        super().__init__()

        # Mood embedding
        self.mood_embedding = nn.Embedding(config.num_moods, config.embedding_dim // 2)
        # Moods: happy, sad, stressed, relaxed, excited, bored, nostalgic, adventurous,
        #        romantic, thoughtful, scared, angry, tired, energetic, curious, melancholy, etc.

        # Energy level encoding
        self.energy_encoder = nn.Sequential(
            nn.Linear(1, config.embedding_dim // 8),
            nn.GELU(),
            nn.Linear(config.embedding_dim // 8, config.embedding_dim // 8)
        )

        # Desired outcome encoding
        self.outcome_embedding = nn.Embedding(10, config.embedding_dim // 4)
        # Outcomes: escape, learn, feel_emotions, laugh, think, cry, be_inspired,
        #           feel_scared, feel_nostalgic, bond_with_others

        # Combine
        self.mood_fusion = nn.Sequential(
            nn.Linear(config.embedding_dim // 2 + config.embedding_dim // 8 + config.embedding_dim // 4,
                     config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.hidden_dim)
        )

    def forward(self, mood_ids: torch.Tensor, energy_level: torch.Tensor,
                desired_outcome: torch.Tensor) -> torch.Tensor:
        """Encode mood"""
        # Clamp indices to valid embedding range to prevent out-of-bounds errors
        mood_ids = mood_ids.clamp(0, self.mood_embedding.num_embeddings - 1)
        desired_outcome = desired_outcome.clamp(0, 9)

        mood_emb = self.mood_embedding(mood_ids)
        energy_emb = self.energy_encoder(energy_level.float().unsqueeze(-1))
        outcome_emb = self.outcome_embedding(desired_outcome)

        combined = torch.cat([mood_emb, energy_emb, outcome_emb], dim=-1)
        return self.mood_fusion(combined)


class OccasionEncoder(nn.Module):
    """Encodes special occasions and events"""

    def __init__(self, config: ViewingContextConfig):
        super().__init__()

        # Occasion embedding
        self.occasion_embedding = nn.Embedding(config.num_occasions, config.embedding_dim)
        # Occasions: regular_evening, date_night, family_movie_night, halloween,
        #            christmas, new_years, birthday, anniversary, rainy_day,
        #            sick_day, lazy_sunday, movie_marathon, film_club, first_date,
        #            breakup, celebration, mourning, reunion, nostalgia_trip, etc.

        # Seasonal embedding
        self.season_embedding = nn.Embedding(4, config.embedding_dim // 4)

        # Combine
        self.occasion_fusion = nn.Sequential(
            nn.Linear(config.embedding_dim + config.embedding_dim // 4, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.hidden_dim)
        )

    def forward(self, occasion_ids: torch.Tensor, season_ids: torch.Tensor) -> torch.Tensor:
        """Encode occasion"""
        # Clamp indices to valid embedding range to prevent out-of-bounds errors
        occasion_ids = occasion_ids.clamp(0, self.occasion_embedding.num_embeddings - 1)
        season_ids = season_ids.clamp(0, 3)

        occasion_emb = self.occasion_embedding(occasion_ids)
        season_emb = self.season_embedding(season_ids)

        combined = torch.cat([occasion_emb, season_emb], dim=-1)
        return self.occasion_fusion(combined)


class ViewingContextModel(nn.Module):
    """
    Complete Viewing Context Model for context-aware recommendations.
    """

    def __init__(self, config: Optional[ViewingContextConfig] = None):
        super().__init__()
        self.config = config or ViewingContextConfig()

        # Core embeddings
        self.movie_embedding = nn.Embedding(self.config.num_movies, self.config.embedding_dim)
        self.user_embedding = nn.Embedding(self.config.num_users, self.config.hidden_dim)

        # Context encoders
        self.context_encoder = ContextEncoder(self.config)
        self.mood_encoder = MoodEncoder(self.config)
        self.occasion_encoder = OccasionEncoder(self.config)

        # Movie projection
        self.movie_proj = nn.Linear(self.config.embedding_dim, self.config.hidden_dim)

        # Context fusion
        self.full_context_fusion = nn.Sequential(
            nn.Linear(self.config.hidden_dim * 3, self.config.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim)
        )

        # User-context preference
        self.user_context_preference = nn.Sequential(
            nn.Linear(self.config.hidden_dim * 2, self.config.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim)
        )

        # Movie-context fit predictor
        self.movie_context_fit_head = nn.Sequential(
            nn.Linear(self.config.hidden_dim * 2, self.config.hidden_dim),
            nn.GELU(),
            nn.Linear(self.config.hidden_dim, 1),
            nn.Sigmoid()
        )

        # Mood-movie alignment predictor
        self.mood_alignment_head = nn.Sequential(
            nn.Linear(self.config.hidden_dim * 2, self.config.hidden_dim),
            nn.GELU(),
            nn.Linear(self.config.hidden_dim, 1),
            nn.Sigmoid()
        )

        # Social appropriateness predictor
        self.social_fit_head = nn.Sequential(
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

        # Movie recommendation head
        self.movie_head = nn.Linear(self.config.hidden_dim, self.config.num_movies)

        # Ensemble genre recommendation
        self.genre_recommendation_head = nn.Sequential(
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim // 2),
            nn.GELU(),
            nn.Linear(self.config.hidden_dim // 2, 30)  # 30 genres
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

    def encode_full_context(self, social_context: torch.Tensor, time_context: torch.Tensor,
                           venue: torch.Tensor, duration_features: torch.Tensor,
                           mood_ids: torch.Tensor, energy_level: torch.Tensor,
                           desired_outcome: torch.Tensor, occasion_ids: torch.Tensor,
                           season_ids: torch.Tensor) -> torch.Tensor:
        """Encode all context features"""
        context_repr = self.context_encoder(social_context, time_context, venue, duration_features)
        mood_repr = self.mood_encoder(mood_ids, energy_level, desired_outcome)
        occasion_repr = self.occasion_encoder(occasion_ids, season_ids)

        full_context = torch.cat([context_repr, mood_repr, occasion_repr], dim=-1)
        return self.full_context_fusion(full_context)

    def forward(self, user_ids: torch.Tensor, movie_ids: torch.Tensor,
                social_context: torch.Tensor, time_context: torch.Tensor,
                venue: torch.Tensor, duration_features: torch.Tensor,
                mood_ids: torch.Tensor, energy_level: torch.Tensor,
                desired_outcome: torch.Tensor, occasion_ids: torch.Tensor,
                season_ids: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass for context-aware recommendations.
        """
        batch_size = user_ids.size(0)

        # Clamp user and movie IDs to valid embedding range to prevent out-of-bounds errors
        user_ids = user_ids.clamp(0, self.config.num_users - 1)
        movie_ids = movie_ids.clamp(0, self.config.num_movies - 1)

        # Get user embedding
        user_emb = self.user_embedding(user_ids)

        # Get movie embedding
        movie_emb = self.movie_embedding(movie_ids)
        movie_repr = self.movie_proj(movie_emb)

        # Encode full context
        full_context = self.encode_full_context(
            social_context, time_context, venue, duration_features,
            mood_ids, energy_level, desired_outcome, occasion_ids, season_ids
        )

        # User-context preference
        user_context = torch.cat([user_emb, full_context], dim=-1)
        context_pref = self.user_context_preference(user_context)

        # Movie-context fit
        movie_context = torch.cat([movie_repr, full_context], dim=-1)
        movie_context_fit = self.movie_context_fit_head(movie_context)

        # Mood alignment
        mood_repr = self.mood_encoder(mood_ids, energy_level, desired_outcome)
        mood_movie = torch.cat([movie_repr, mood_repr], dim=-1)
        mood_alignment = self.mood_alignment_head(mood_movie)

        # Social appropriateness
        context_repr = self.context_encoder(social_context, time_context, venue, duration_features)
        social_movie = torch.cat([movie_repr, context_repr], dim=-1)
        social_fit = self.social_fit_head(social_movie)

        # Rating prediction
        rating_input = torch.cat([user_emb, movie_repr, full_context], dim=-1)
        rating_pred = self.rating_head(rating_input)

        # Movie recommendations
        movie_logits = self.movie_head(context_pref)

        # Genre recommendations for this context
        genre_recommendation = self.genre_recommendation_head(full_context)

        return {
            'context_preference': context_pref,
            'full_context': full_context,
            'movie_context_fit': movie_context_fit,
            'mood_alignment': mood_alignment,
            'social_fit': social_fit,
            'rating_pred': rating_pred,
            'movie_logits': movie_logits,
            'genre_recommendation': genre_recommendation
        }

    def recommend_for_context(self, user_ids: torch.Tensor,
                             social_context: int, time_context: int, venue: int,
                             mood: int, occasion: int,
                             top_k: int = 10) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get recommendations for a specific context"""
        with torch.no_grad():
            device = user_ids.device
            batch_size = user_ids.size(0)

            user_emb = self.user_embedding(user_ids)

            # Create context tensors
            social = torch.tensor([social_context], device=device).expand(batch_size)
            time = torch.tensor([time_context], device=device).expand(batch_size)
            venue_t = torch.tensor([venue], device=device).expand(batch_size)
            duration = torch.zeros(batch_size, 2, device=device)
            mood_t = torch.tensor([mood], device=device).expand(batch_size)
            energy = torch.ones(batch_size, device=device) * 0.5
            outcome = torch.zeros(batch_size, dtype=torch.long, device=device)
            occasion_t = torch.tensor([occasion], device=device).expand(batch_size)
            season = torch.zeros(batch_size, dtype=torch.long, device=device)

            # Encode context
            full_context = self.encode_full_context(
                social, time, venue_t, duration, mood_t, energy, outcome, occasion_t, season
            )

            # Get recommendations
            user_context = torch.cat([user_emb, full_context], dim=-1)
            context_pref = self.user_context_preference(user_context)
            movie_logits = self.movie_head(context_pref)

            top_scores, top_movies = torch.topk(F.softmax(movie_logits, dim=-1), top_k, dim=-1)

        return top_movies, top_scores

    def get_context_recommendations(self, context_type: str, user_ids: torch.Tensor,
                                    top_k: int = 10) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get recommendations for predefined context types"""
        # Predefined context mappings
        context_presets = {
            'date_night': (1, 2, 0, 8, 2),  # couple, evening, home_tv, romantic, date_night
            'family_movie': (2, 2, 0, 0, 3),  # family, evening, home_tv, happy, family_movie_night
            'late_night_alone': (0, 3, 0, 16, 0),  # alone, late_night, home_tv, tired, regular_evening
            'party': (4, 2, 0, 3, 0),  # party, evening, home_tv, excited, regular_evening
            'rainy_sunday': (0, 1, 0, 5, 8),  # alone, afternoon, home_tv, bored, rainy_day
        }

        if context_type in context_presets:
            social, time, venue, mood, occasion = context_presets[context_type]
            return self.recommend_for_context(
                user_ids, social, time, venue, mood, occasion, top_k
            )
        else:
            # Default to regular evening alone
            return self.recommend_for_context(user_ids, 0, 2, 0, 0, 0, top_k)


class ViewingContextTrainer:
    """Trainer for Viewing Context Model"""

    def __init__(self, model: ViewingContextModel, lr: float = 1e-4,
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

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Single training step"""
        self.model.train()

        user_ids = batch['user_ids'].to(self.device)
        movie_ids = batch['movie_ids'].to(self.device)
        social_context = batch['social_context'].to(self.device)
        time_context = batch['time_context'].to(self.device)
        venue = batch['venue'].to(self.device)
        duration_features = batch['duration_features'].to(self.device)
        mood_ids = batch['mood_ids'].to(self.device)
        energy_level = batch['energy_level'].to(self.device)
        desired_outcome = batch['desired_outcome'].to(self.device)
        occasion_ids = batch['occasion_ids'].to(self.device)
        season_ids = batch['season_ids'].to(self.device)
        target_ratings = batch['ratings'].to(self.device)

        outputs = self.model(
            user_ids, movie_ids, social_context, time_context, venue,
            duration_features, mood_ids, energy_level, desired_outcome,
            occasion_ids, season_ids
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
                movie_ids = batch['movie_ids'].to(self.device)
                social_context = batch['social_context'].to(self.device)
                time_context = batch['time_context'].to(self.device)
                venue = batch['venue'].to(self.device)
                duration_features = batch['duration_features'].to(self.device)
                mood_ids = batch['mood_ids'].to(self.device)
                energy_level = batch['energy_level'].to(self.device)
                desired_outcome = batch['desired_outcome'].to(self.device)
                occasion_ids = batch['occasion_ids'].to(self.device)
                season_ids = batch['season_ids'].to(self.device)
                target_ratings = batch['ratings'].to(self.device)

                outputs = self.model(
                    user_ids, movie_ids, social_context, time_context, venue,
                    duration_features, mood_ids, energy_level, desired_outcome,
                    occasion_ids, season_ids
                )

                total_metrics['rating_mse'] += F.mse_loss(
                    outputs['rating_pred'].squeeze(), target_ratings
                ).item()

                num_batches += 1

        return {k: v / num_batches for k, v in total_metrics.items()}
