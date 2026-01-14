"""
Awards Prediction Model (APM)
Predicts Oscar/prestige likelihood and recommends award-worthy films based on user preferences.
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
class AwardsConfig:
    """Configuration for Awards Prediction Model"""
    embedding_dim: int = 256
    hidden_dim: int = 512
    num_heads: int = 8
    num_layers: int = 4
    num_movies: int = 100000
    num_users: int = 50000
    num_awards: int = 100  # Oscar categories, Golden Globes, etc.
    num_festivals: int = 50  # Cannes, Venice, Sundance, etc.
    num_critics_groups: int = 30
    dropout: float = 0.1


class AwardsHistoryEncoder(nn.Module):
    """Encodes historical award patterns and trends"""

    def __init__(self, config: AwardsConfig):
        super().__init__()

        # Award category embeddings
        self.award_embedding = nn.Embedding(config.num_awards, config.embedding_dim)

        # Festival embeddings
        self.festival_embedding = nn.Embedding(config.num_festivals, config.embedding_dim)

        # Year trend encoding
        self.year_encoding = nn.Parameter(torch.randn(100, config.embedding_dim))  # 1950-2050

        # Award result embedding (nominated, won, snubbed)
        self.result_embedding = nn.Embedding(4, config.embedding_dim // 4)

        # Temporal attention for award trends
        self.temporal_attention = nn.MultiheadAttention(
            config.embedding_dim, config.num_heads, dropout=config.dropout, batch_first=True
        )

        self.output_proj = nn.Linear(config.embedding_dim, config.hidden_dim)

    def forward(self, award_ids: torch.Tensor, years: torch.Tensor,
                results: torch.Tensor) -> torch.Tensor:
        """
        Encode award history.

        Args:
            award_ids: Award category IDs [batch, seq_len]
            years: Award years [batch, seq_len]
            results: Award results [batch, seq_len]
        """
        award_emb = self.award_embedding(award_ids)
        year_emb = self.year_encoding[(years - 1950).clamp(0, 99)]
        result_emb = self.result_embedding(results)

        # Combine features
        combined = award_emb + year_emb
        combined = torch.cat([combined, result_emb.expand(-1, -1, combined.size(-1) // 4 * 4)], dim=-1)
        combined = combined[:, :, :self.award_embedding.embedding_dim]

        # Temporal attention
        attended, _ = self.temporal_attention(combined, combined, combined)

        return self.output_proj(attended)


class PrestigeFeatureExtractor(nn.Module):
    """Extracts prestige-related features from movies"""

    def __init__(self, config: AwardsConfig):
        super().__init__()

        # Critic score processor - accepts single score and expands to embedding
        self.critic_processor = nn.Sequential(
            nn.Linear(1, config.embedding_dim // 4),  # single critic score input
            nn.GELU(),
            nn.Linear(config.embedding_dim // 4, config.embedding_dim // 2)
        )

        # Release timing features - one-hot encoding done internally
        self.num_months = 12
        self.release_timing = nn.Sequential(
            nn.Linear(self.num_months, config.embedding_dim // 4),  # month of release one-hot
            nn.GELU(),
            nn.Linear(config.embedding_dim // 4, config.embedding_dim // 4)
        )

        # Studio prestige embedding
        self.studio_prestige = nn.Embedding(500, config.embedding_dim // 4)

        # Distributor embedding (A24, Focus Features, etc.)
        self.distributor_embedding = nn.Embedding(200, config.embedding_dim // 4)

        # Budget range embedding (indie vs blockbuster)
        self.budget_embedding = nn.Embedding(10, config.embedding_dim // 4)

        # Combine features
        # Input: critic(128) + timing(64) + studio(64) + dist(64) + budget(64) = 384
        feature_input_dim = config.embedding_dim // 2 + config.embedding_dim // 4 * 4
        self.feature_fusion = nn.Sequential(
            nn.Linear(feature_input_dim, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.hidden_dim)
        )

    def forward(self, critic_scores: torch.Tensor, release_month: torch.Tensor,
                studio_ids: torch.Tensor, distributor_ids: torch.Tensor,
                budget_range: torch.Tensor) -> torch.Tensor:
        """Extract prestige features"""
        # Handle critic_scores: ensure shape [batch, 1]
        if critic_scores.dim() == 1:
            critic_scores = critic_scores.unsqueeze(-1)
        critic_feat = self.critic_processor(critic_scores)

        # Handle release_month: one-hot encode from int to [batch, 12]
        if release_month.dim() == 1 and release_month.dtype in (torch.int64, torch.int32, torch.long):
            # Clamp to valid range [0, 11] for one-hot encoding
            release_month_clamped = release_month.clamp(0, self.num_months - 1)
            release_month_onehot = F.one_hot(release_month_clamped, num_classes=self.num_months).float()
        else:
            release_month_onehot = release_month
        timing_feat = self.release_timing(release_month_onehot)

        studio_feat = self.studio_prestige(studio_ids)
        dist_feat = self.distributor_embedding(distributor_ids)
        budget_feat = self.budget_embedding(budget_range)

        combined = torch.cat([critic_feat, timing_feat, studio_feat, dist_feat, budget_feat], dim=-1)
        return self.feature_fusion(combined)


class AwardsPredictionModel(nn.Module):
    """
    Complete Awards Prediction Model for Oscar/prestige recommendations.
    """

    def __init__(self, config: Optional[AwardsConfig] = None):
        super().__init__()
        self.config = config or AwardsConfig()

        # Core embeddings
        self.movie_embedding = nn.Embedding(self.config.num_movies, self.config.embedding_dim)
        self.user_embedding = nn.Embedding(self.config.num_users, self.config.hidden_dim)

        # Award history encoder
        self.history_encoder = AwardsHistoryEncoder(self.config)

        # Prestige feature extractor
        self.prestige_extractor = PrestigeFeatureExtractor(self.config)

        # Movie encoder
        self.movie_proj = nn.Linear(self.config.embedding_dim, self.config.hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.config.hidden_dim,
            nhead=self.config.num_heads,
            dim_feedforward=self.config.hidden_dim * 4,
            dropout=self.config.dropout,
            activation='gelu',
            batch_first=True
        )
        self.movie_transformer = nn.TransformerEncoder(encoder_layer, num_layers=self.config.num_layers)

        # User prestige preference modeling
        self.user_prestige_preference = nn.Sequential(
            nn.Linear(self.config.hidden_dim * 2, self.config.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim)
        )

        # Award category preference (which awards user cares about)
        self.award_preference = nn.Sequential(
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim // 2),
            nn.GELU(),
            nn.Linear(self.config.hidden_dim // 2, self.config.num_awards)
        )

        # Oscar nomination predictor
        self.nomination_head = nn.Sequential(
            nn.Linear(self.config.hidden_dim * 2, self.config.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.hidden_dim, self.config.num_awards)
        )

        # Oscar win predictor
        self.win_head = nn.Sequential(
            nn.Linear(self.config.hidden_dim * 2, self.config.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.hidden_dim, self.config.num_awards)
        )

        # Overall prestige score
        self.prestige_score_head = nn.Sequential(
            nn.Linear(self.config.hidden_dim * 2, self.config.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.hidden_dim, 1),
            nn.Sigmoid()
        )

        # User rating predictor
        self.rating_head = nn.Sequential(
            nn.Linear(self.config.hidden_dim * 3, self.config.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.hidden_dim, 1)
        )

        # Movie recommendation head
        self.movie_head = nn.Linear(self.config.hidden_dim, self.config.num_movies)

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0, std=0.02)

    def encode_movie_prestige(self, movie_ids: torch.Tensor,
                              critic_scores: torch.Tensor,
                              release_month: torch.Tensor,
                              studio_ids: torch.Tensor,
                              distributor_ids: torch.Tensor,
                              budget_range: torch.Tensor) -> torch.Tensor:
        """Encode movie with prestige features"""
        movie_emb = self.movie_embedding(movie_ids)
        movie_repr = self.movie_proj(movie_emb)

        prestige_feat = self.prestige_extractor(
            critic_scores, release_month, studio_ids, distributor_ids, budget_range
        )

        combined = movie_repr + prestige_feat.unsqueeze(1) if movie_repr.dim() == 3 else movie_repr + prestige_feat

        if combined.dim() == 3:
            combined = self.movie_transformer(combined)

        return combined

    def forward(self, user_ids: torch.Tensor, movie_ids: torch.Tensor,
                critic_scores: torch.Tensor, release_month: torch.Tensor,
                studio_ids: torch.Tensor, distributor_ids: torch.Tensor,
                budget_range: torch.Tensor,
                award_history_ids: Optional[torch.Tensor] = None,
                award_years: Optional[torch.Tensor] = None,
                award_results: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass for awards prediction.
        """
        batch_size = user_ids.size(0)

        # Get user embedding
        user_emb = self.user_embedding(user_ids)

        # Encode movie prestige
        movie_repr = self.encode_movie_prestige(
            movie_ids, critic_scores, release_month, studio_ids, distributor_ids, budget_range
        )

        # Pool movie representation if sequence
        if movie_repr.dim() == 3:
            movie_repr = movie_repr.mean(dim=1)

        # User-movie interaction
        user_movie = torch.cat([user_emb, movie_repr], dim=-1)

        # User prestige preference
        prestige_pref = self.user_prestige_preference(user_movie)

        # Award category preferences
        award_pref = self.award_preference(prestige_pref)

        # Predict nominations
        nomination_logits = self.nomination_head(user_movie)

        # Predict wins
        win_logits = self.win_head(user_movie)

        # Overall prestige score
        prestige_score = self.prestige_score_head(user_movie)

        # Rating prediction
        rating_input = torch.cat([user_emb, movie_repr, prestige_pref], dim=-1)
        rating_pred = self.rating_head(rating_input)

        # Movie recommendations
        movie_logits = self.movie_head(prestige_pref)

        return {
            'movie_repr': movie_repr,
            'nomination_logits': nomination_logits,
            'win_logits': win_logits,
            'prestige_score': prestige_score,
            'award_preference': award_pref,
            'rating_pred': rating_pred,
            'movie_logits': movie_logits,
            'prestige_preference': prestige_pref
        }

    def predict_oscar_nominations(self, movie_repr: torch.Tensor,
                                  top_k: int = 5) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict top-k most likely Oscar nominations for a movie"""
        with torch.no_grad():
            # Use movie representation directly
            dummy_user = torch.zeros(movie_repr.size(0), self.config.hidden_dim, device=movie_repr.device)
            user_movie = torch.cat([dummy_user, movie_repr], dim=-1)

            nomination_probs = torch.sigmoid(self.nomination_head(user_movie))
            top_probs, top_categories = torch.topk(nomination_probs, top_k, dim=-1)

        return top_categories, top_probs

    def recommend_prestige_films(self, user_ids: torch.Tensor,
                                 min_prestige: float = 0.7,
                                 top_k: int = 10) -> Tuple[torch.Tensor, torch.Tensor]:
        """Recommend high-prestige films for users"""
        with torch.no_grad():
            user_emb = self.user_embedding(user_ids)

            # Get all movie embeddings
            all_movies = torch.arange(self.config.num_movies, device=user_ids.device)
            movie_emb = self.movie_embedding(all_movies)
            movie_repr = self.movie_proj(movie_emb)

            # Score each movie for each user
            scores = []
            for i in range(user_ids.size(0)):
                user_expanded = user_emb[i:i+1].expand(movie_repr.size(0), -1)
                user_movie = torch.cat([user_expanded, movie_repr], dim=-1)
                prestige = self.prestige_score_head(user_movie).squeeze()
                scores.append(prestige)

            scores = torch.stack(scores)

            # Filter by minimum prestige and get top-k
            scores = scores.masked_fill(scores < min_prestige, -float('inf'))
            top_scores, top_movies = torch.topk(scores, top_k, dim=-1)

        return top_movies, top_scores


class AwardsPredictionTrainer:
    """Trainer for Awards Prediction Model"""

    def __init__(self, model: AwardsPredictionModel, lr: float = 1e-4,
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
        self.nomination_loss = nn.BCEWithLogitsLoss()
        self.win_loss = nn.BCEWithLogitsLoss()
        self.prestige_loss = nn.BCELoss()

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Single training step"""
        self.model.train()

        user_ids = batch['user_ids'].to(self.device)
        movie_ids = batch['movie_ids'].to(self.device)
        critic_scores = batch['critic_scores'].to(self.device)
        release_month = batch['release_month'].to(self.device)
        studio_ids = batch['studio_ids'].to(self.device)
        distributor_ids = batch['distributor_ids'].to(self.device)
        budget_range = batch['budget_range'].to(self.device)
        target_ratings = batch['ratings'].to(self.device)
        target_nominations = batch['nominations'].to(self.device)
        target_wins = batch['wins'].to(self.device)
        target_prestige = batch['prestige_score'].to(self.device)

        outputs = self.model(
            user_ids, movie_ids, critic_scores, release_month,
            studio_ids, distributor_ids, budget_range
        )

        # Compute losses
        rating_loss = self.rating_loss(outputs['rating_pred'].squeeze(), target_ratings)
        nomination_loss = self.nomination_loss(outputs['nomination_logits'], target_nominations.float())
        win_loss = self.win_loss(outputs['win_logits'], target_wins.float())
        prestige_loss = self.prestige_loss(outputs['prestige_score'].squeeze(), target_prestige)

        total_loss = rating_loss + nomination_loss + win_loss + prestige_loss

        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()

        return {
            'total_loss': total_loss.item(),
            'rating_loss': rating_loss.item(),
            'nomination_loss': nomination_loss.item(),
            'win_loss': win_loss.item(),
            'prestige_loss': prestige_loss.item()
        }

    def evaluate(self, dataloader) -> Dict[str, float]:
        """Evaluate model"""
        self.model.eval()
        total_metrics = {'rating_mse': 0, 'nomination_auc': 0, 'win_auc': 0}
        num_batches = 0

        with torch.no_grad():
            for batch in dataloader:
                user_ids = batch['user_ids'].to(self.device)
                movie_ids = batch['movie_ids'].to(self.device)
                critic_scores = batch['critic_scores'].to(self.device)
                release_month = batch['release_month'].to(self.device)
                studio_ids = batch['studio_ids'].to(self.device)
                distributor_ids = batch['distributor_ids'].to(self.device)
                budget_range = batch['budget_range'].to(self.device)
                target_ratings = batch['ratings'].to(self.device)

                outputs = self.model(
                    user_ids, movie_ids, critic_scores, release_month,
                    studio_ids, distributor_ids, budget_range
                )

                total_metrics['rating_mse'] += F.mse_loss(
                    outputs['rating_pred'].squeeze(), target_ratings
                ).item()

                num_batches += 1

        return {k: v / num_batches for k, v in total_metrics.items()}
