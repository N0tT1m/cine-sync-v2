"""
International Cinema Model (ICM)
Models country/region preferences and cross-cultural film recommendations.
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
class InternationalConfig:
    """Configuration for International Cinema Model"""
    embedding_dim: int = 256
    hidden_dim: int = 512
    num_heads: int = 8
    num_layers: int = 4
    num_movies: int = 100000
    num_users: int = 50000
    num_countries: int = 200
    num_regions: int = 20  # Hollywood, Bollywood, European, East Asian, etc.
    num_languages: int = 100
    dropout: float = 0.1


class CountryEncoder(nn.Module):
    """Encodes country and regional film characteristics"""

    def __init__(self, config: InternationalConfig):
        super().__init__()

        # Country embedding
        self.country_embedding = nn.Embedding(config.num_countries, config.embedding_dim)

        # Region embedding
        self.region_embedding = nn.Embedding(config.num_regions, config.embedding_dim // 2)

        # Language embedding
        self.language_embedding = nn.Embedding(config.num_languages, config.embedding_dim // 2)

        # Country characteristics
        self.characteristics_encoder = nn.Sequential(
            nn.Linear(16, config.embedding_dim // 4),  # film_industry_size, avg_production_budget, oscar_wins, festival_wins, cultural_export_score, co_production_rate, subtitled_ratio, dubbing_preference, + 8 more features
            nn.GELU(),
            nn.Linear(config.embedding_dim // 4, config.embedding_dim // 4)
        )

        # Combine
        self.country_fusion = nn.Sequential(
            nn.Linear(config.embedding_dim * 2 + config.embedding_dim // 4, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.hidden_dim)
        )

    def forward(self, country_ids: torch.Tensor, region_ids: torch.Tensor,
                language_ids: torch.Tensor, characteristics: torch.Tensor) -> torch.Tensor:
        """Encode country features"""
        country_emb = self.country_embedding(country_ids)
        region_emb = self.region_embedding(region_ids)
        lang_emb = self.language_embedding(language_ids)
        char_emb = self.characteristics_encoder(characteristics)

        combined = torch.cat([country_emb, region_emb, lang_emb, char_emb], dim=-1)
        return self.country_fusion(combined)


class CulturalStyleEncoder(nn.Module):
    """Encodes cultural filmmaking styles"""

    def __init__(self, config: InternationalConfig):
        super().__init__()

        # Narrative style embedding
        self.narrative_style = nn.Embedding(20, config.embedding_dim // 4)
        # Styles: Hollywood_classic, European_arthouse, Asian_melodrama, Latin_magical_realism, etc.

        # Pacing preference (fast/slow)
        self.pacing_encoder = nn.Sequential(
            nn.Linear(1, config.embedding_dim // 8),
            nn.GELU(),
            nn.Linear(config.embedding_dim // 8, config.embedding_dim // 8)
        )

        # Genre distribution by region
        self.genre_dist_encoder = nn.Linear(30, config.embedding_dim // 4)

        # Visual style embedding
        self.visual_style = nn.Embedding(50, config.embedding_dim // 4)
        # Styles: Hollywood_polished, European_naturalistic, Asian_stylized, etc.

        # Combine
        self.style_fusion = nn.Sequential(
            nn.Linear(config.embedding_dim // 4 * 3 + config.embedding_dim // 8, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.hidden_dim)
        )

    def forward(self, narrative_styles: torch.Tensor, pacing: torch.Tensor,
                genre_dist: torch.Tensor, visual_styles: torch.Tensor) -> torch.Tensor:
        """Encode cultural style"""
        narrative_emb = self.narrative_style(narrative_styles)
        pacing_emb = self.pacing_encoder(pacing.float().unsqueeze(-1))
        genre_emb = self.genre_dist_encoder(genre_dist)
        visual_emb = self.visual_style(visual_styles)

        combined = torch.cat([narrative_emb, pacing_emb, genre_emb, visual_emb], dim=-1)
        return self.style_fusion(combined)


class InternationalCinemaModel(nn.Module):
    """
    Complete International Cinema Model for cross-cultural recommendations.
    """

    def __init__(self, config: Optional[InternationalConfig] = None):
        super().__init__()
        self.config = config or InternationalConfig()

        # Core embeddings
        self.movie_embedding = nn.Embedding(self.config.num_movies, self.config.embedding_dim)
        self.user_embedding = nn.Embedding(self.config.num_users, self.config.hidden_dim)

        # Country encoder
        self.country_encoder = CountryEncoder(self.config)

        # Cultural style encoder
        self.style_encoder = CulturalStyleEncoder(self.config)

        # Movie projection
        self.movie_proj = nn.Linear(self.config.embedding_dim, self.config.hidden_dim)

        # User international preference
        self.user_international_preference = nn.Sequential(
            nn.Linear(self.config.hidden_dim * 2, self.config.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim)
        )

        # Country affinity head
        self.country_affinity_head = nn.Sequential(
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim // 2),
            nn.GELU(),
            nn.Linear(self.config.hidden_dim // 2, self.config.num_countries)
        )

        # Region affinity head
        self.region_affinity_head = nn.Sequential(
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim // 2),
            nn.GELU(),
            nn.Linear(self.config.hidden_dim // 2, self.config.num_regions)
        )

        # Language barrier tolerance (subtitle/dub preference)
        self.language_tolerance_head = nn.Sequential(
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim // 2),
            nn.GELU(),
            nn.Linear(self.config.hidden_dim // 2, 3),  # subtitle_preferred, dub_preferred, native_only
            nn.Softmax(dim=-1)
        )

        # Cultural distance impact (how much unfamiliar cultures affect rating)
        self.cultural_distance_head = nn.Sequential(
            nn.Linear(self.config.hidden_dim * 2, self.config.hidden_dim),
            nn.GELU(),
            nn.Linear(self.config.hidden_dim, 1)
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

        # Exploration predictor (willingness to try new countries)
        self.exploration_head = nn.Sequential(
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim // 2),
            nn.GELU(),
            nn.Linear(self.config.hidden_dim // 2, 1),
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
                country_ids: torch.Tensor, region_ids: torch.Tensor,
                language_ids: torch.Tensor, country_characteristics: torch.Tensor,
                narrative_styles: torch.Tensor, pacing: torch.Tensor,
                genre_dist: torch.Tensor, visual_styles: torch.Tensor,
                user_country: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass for international cinema recommendations.
        """
        batch_size = user_ids.size(0)

        # Get user embedding
        user_emb = self.user_embedding(user_ids)

        # Get movie embedding
        movie_emb = self.movie_embedding(movie_ids)
        movie_repr = self.movie_proj(movie_emb)

        # Encode country
        country_repr = self.country_encoder(country_ids, region_ids, language_ids,
                                           country_characteristics)

        # Encode cultural style
        style_repr = self.style_encoder(narrative_styles, pacing, genre_dist, visual_styles)

        # Combined international representation
        combined_intl = country_repr + style_repr

        # User-international preference
        user_intl = torch.cat([user_emb, combined_intl], dim=-1)
        intl_pref = self.user_international_preference(user_intl)

        # Country affinity
        country_affinity = self.country_affinity_head(user_emb)

        # Region affinity
        region_affinity = self.region_affinity_head(user_emb)

        # Language tolerance
        lang_tolerance = self.language_tolerance_head(user_emb)

        # Cultural distance impact
        cultural_distance = self.cultural_distance_head(user_intl)

        # Exploration willingness
        exploration = self.exploration_head(user_emb)

        # Rating prediction
        rating_input = torch.cat([user_emb, movie_repr, combined_intl], dim=-1)
        rating_pred = self.rating_head(rating_input)

        # Movie recommendations
        movie_logits = self.movie_head(intl_pref)

        return {
            'international_preference': intl_pref,
            'country_repr': combined_intl,
            'country_affinity': country_affinity,
            'region_affinity': region_affinity,
            'language_tolerance': lang_tolerance,
            'cultural_distance': cultural_distance,
            'exploration': exploration,
            'rating_pred': rating_pred,
            'movie_logits': movie_logits
        }

    def recommend_from_country(self, user_ids: torch.Tensor, country_id: int,
                              top_k: int = 10) -> Tuple[torch.Tensor, torch.Tensor]:
        """Recommend movies from a specific country"""
        with torch.no_grad():
            user_emb = self.user_embedding(user_ids)
            country_emb = self.country_encoder.country_embedding.weight[country_id]

            user_country = torch.cat([user_emb, country_emb.expand(user_ids.size(0), -1)], dim=-1)
            pref = self.user_international_preference(user_country)

            movie_logits = self.movie_head(pref)
            top_scores, top_movies = torch.topk(F.softmax(movie_logits, dim=-1), top_k, dim=-1)

        return top_movies, top_scores

    def recommend_new_countries(self, user_ids: torch.Tensor,
                               watched_countries: torch.Tensor,
                               top_k: int = 5) -> Tuple[torch.Tensor, torch.Tensor]:
        """Recommend countries the user hasn't explored"""
        with torch.no_grad():
            user_emb = self.user_embedding(user_ids)
            country_affinity = self.country_affinity_head(user_emb)

            # Mask watched countries
            for i, watched in enumerate(watched_countries):
                country_affinity[i, watched] = -float('inf')

            top_scores, top_countries = torch.topk(F.softmax(country_affinity, dim=-1), top_k, dim=-1)

        return top_countries, top_scores

    def get_international_profile(self, user_ids: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Get user's international cinema profile"""
        with torch.no_grad():
            user_emb = self.user_embedding(user_ids)
            country_affinity = self.country_affinity_head(user_emb)
            region_affinity = self.region_affinity_head(user_emb)
            lang_tolerance = self.language_tolerance_head(user_emb)
            exploration = self.exploration_head(user_emb)

        return {
            'country_affinity': country_affinity,
            'region_affinity': region_affinity,
            'language_tolerance': lang_tolerance,
            'exploration': exploration
        }


class InternationalCinemaTrainer:
    """Trainer for International Cinema Model"""

    def __init__(self, model: InternationalCinemaModel, lr: float = 1e-4,
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
        self.affinity_loss = nn.CrossEntropyLoss()

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Single training step"""
        self.model.train()

        user_ids = batch['user_ids'].to(self.device)
        movie_ids = batch['movie_ids'].to(self.device)
        country_ids = batch['country_ids'].to(self.device)
        region_ids = batch['region_ids'].to(self.device)
        language_ids = batch['language_ids'].to(self.device)
        country_characteristics = batch['country_characteristics'].to(self.device)
        narrative_styles = batch['narrative_styles'].to(self.device)
        pacing = batch['pacing'].to(self.device)
        genre_dist = batch['genre_dist'].to(self.device)
        visual_styles = batch['visual_styles'].to(self.device)
        target_ratings = batch['ratings'].to(self.device)

        outputs = self.model(
            user_ids, movie_ids, country_ids, region_ids, language_ids,
            country_characteristics, narrative_styles, pacing, genre_dist, visual_styles
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
                country_ids = batch['country_ids'].to(self.device)
                region_ids = batch['region_ids'].to(self.device)
                language_ids = batch['language_ids'].to(self.device)
                country_characteristics = batch['country_characteristics'].to(self.device)
                narrative_styles = batch['narrative_styles'].to(self.device)
                pacing = batch['pacing'].to(self.device)
                genre_dist = batch['genre_dist'].to(self.device)
                visual_styles = batch['visual_styles'].to(self.device)
                target_ratings = batch['ratings'].to(self.device)

                outputs = self.model(
                    user_ids, movie_ids, country_ids, region_ids, language_ids,
                    country_characteristics, narrative_styles, pacing, genre_dist, visual_styles
                )

                total_metrics['rating_mse'] += F.mse_loss(
                    outputs['rating_pred'].squeeze(), target_ratings
                ).item()

                num_batches += 1

        return {k: v / num_batches for k, v in total_metrics.items()}
