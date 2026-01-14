"""
Critic Audience Model (CAM)
Models alignment between user preferences and critic/audience scores.
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
class CriticConfig:
    """Configuration for Critic Audience Model"""
    embedding_dim: int = 256
    hidden_dim: int = 512
    num_heads: int = 8
    num_layers: int = 4
    num_movies: int = 100000
    num_users: int = 50000
    num_critics: int = 5000
    num_publications: int = 500
    dropout: float = 0.1


class ScoreEncoder(nn.Module):
    """Encodes critic and audience scores"""

    def __init__(self, config: CriticConfig):
        super().__init__()

        # Score encoders for different platforms
        self.metacritic_encoder = nn.Sequential(
            nn.Linear(2, config.embedding_dim // 4),  # critic, user
            nn.GELU(),
            nn.Linear(config.embedding_dim // 4, config.embedding_dim // 4)
        )

        self.rt_encoder = nn.Sequential(
            nn.Linear(4, config.embedding_dim // 4),  # critic %, audience %, avg_critic, avg_audience
            nn.GELU(),
            nn.Linear(config.embedding_dim // 4, config.embedding_dim // 4)
        )

        self.imdb_encoder = nn.Sequential(
            nn.Linear(2, config.embedding_dim // 4),  # score, vote_count (log-scaled)
            nn.GELU(),
            nn.Linear(config.embedding_dim // 4, config.embedding_dim // 4)
        )

        self.letterboxd_encoder = nn.Sequential(
            nn.Linear(2, config.embedding_dim // 4),  # avg rating, log(ratings count)
            nn.GELU(),
            nn.Linear(config.embedding_dim // 4, config.embedding_dim // 4)
        )

        # Score fusion
        self.score_fusion = nn.Sequential(
            nn.Linear(config.embedding_dim, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.hidden_dim)
        )

        # Critic-audience gap encoder
        self.gap_encoder = nn.Sequential(
            nn.Linear(4, config.embedding_dim // 4),  # gap values across platforms
            nn.GELU(),
            nn.Linear(config.embedding_dim // 4, config.embedding_dim // 4)
        )

    def forward(self, metacritic_scores: torch.Tensor, rt_scores: torch.Tensor,
                imdb_scores: torch.Tensor, letterboxd_scores: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode scores from multiple platforms.
        """
        mc_emb = self.metacritic_encoder(metacritic_scores)
        rt_emb = self.rt_encoder(rt_scores)
        imdb_emb = self.imdb_encoder(imdb_scores)
        lb_emb = self.letterboxd_encoder(letterboxd_scores)

        combined = torch.cat([mc_emb, rt_emb, imdb_emb, lb_emb], dim=-1)
        score_repr = self.score_fusion(combined)

        # Compute critic-audience gaps
        mc_gap = metacritic_scores[:, 0] - metacritic_scores[:, 1] * 10  # Scale user to 0-100
        rt_gap = rt_scores[:, 0] - rt_scores[:, 1]
        imdb_gap = imdb_scores[:, 0] - 7.0  # Deviation from "good" threshold
        lb_gap = letterboxd_scores[:, 0] - 3.5  # Deviation from middle

        gaps = torch.stack([mc_gap, rt_gap, imdb_gap, lb_gap], dim=-1)
        gap_emb = self.gap_encoder(gaps)

        return score_repr, gap_emb


class UserAlignmentPredictor(nn.Module):
    """Predicts user alignment with critics vs general audience"""

    def __init__(self, config: CriticConfig):
        super().__init__()

        # Historical alignment encoder
        self.alignment_encoder = nn.LSTM(
            config.hidden_dim, config.hidden_dim // 2,
            num_layers=2, batch_first=True, bidirectional=True,
            dropout=config.dropout
        )

        # Alignment predictor
        self.alignment_head = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.GELU(),
            nn.Linear(config.hidden_dim // 2, 3),  # critic, balanced, audience
            nn.Softmax(dim=-1)
        )

        # Per-genre alignment
        self.genre_alignment = nn.Linear(config.hidden_dim, 30 * 3)  # 30 genres x 3 alignment types

    def forward(self, history_repr: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict user alignment type"""
        lstm_out, (h_n, _) = self.alignment_encoder(history_repr)
        final_state = h_n[-2:].transpose(0, 1).contiguous().view(history_repr.size(0), -1)

        alignment = self.alignment_head(final_state)
        genre_alignment = self.genre_alignment(final_state).view(-1, 30, 3)

        return alignment, genre_alignment


class CriticAudienceModel(nn.Module):
    """
    Complete Critic Audience Model for understanding score preferences.
    """

    def __init__(self, config: Optional[CriticConfig] = None):
        super().__init__()
        self.config = config or CriticConfig()

        # Core embeddings
        self.movie_embedding = nn.Embedding(self.config.num_movies, self.config.embedding_dim)
        self.user_embedding = nn.Embedding(self.config.num_users, self.config.hidden_dim)
        self.critic_embedding = nn.Embedding(self.config.num_critics, self.config.embedding_dim // 2)
        self.publication_embedding = nn.Embedding(self.config.num_publications, self.config.embedding_dim // 2)

        # Score encoder
        self.score_encoder = ScoreEncoder(self.config)

        # Alignment predictor
        self.alignment_predictor = UserAlignmentPredictor(self.config)

        # Movie projection
        self.movie_proj = nn.Linear(self.config.embedding_dim, self.config.hidden_dim)

        # User score preference
        self.user_score_preference = nn.Sequential(
            nn.Linear(self.config.hidden_dim * 2, self.config.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim)
        )

        # Personal score predictor (what score would user give?)
        self.personal_score_head = nn.Sequential(
            nn.Linear(self.config.hidden_dim * 2, self.config.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.hidden_dim, 1)
        )

        # Score deviation predictor (how much user differs from consensus)
        self.deviation_head = nn.Sequential(
            nn.Linear(self.config.hidden_dim * 2, self.config.hidden_dim // 2),
            nn.GELU(),
            nn.Linear(self.config.hidden_dim // 2, 1)
        )

        # Contrarian detector (does user like underrated/hate overrated?)
        self.contrarian_head = nn.Sequential(
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim // 2),
            nn.GELU(),
            nn.Linear(self.config.hidden_dim // 2, 1),
            nn.Sigmoid()
        )

        # Movie recommendation head
        self.movie_head = nn.Linear(self.config.hidden_dim, self.config.num_movies)

        # Critic match predictor (which critics align with user?)
        self.critic_match_head = nn.Sequential(
            nn.Linear(self.config.hidden_dim * 2, self.config.hidden_dim),
            nn.GELU(),
            nn.Linear(self.config.hidden_dim, self.config.num_critics)
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
                metacritic_scores: torch.Tensor, rt_scores: torch.Tensor,
                imdb_scores: torch.Tensor, letterboxd_scores: torch.Tensor,
                user_history_scores: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass for critic-audience alignment.
        """
        batch_size = user_ids.size(0)

        # Get user embedding
        user_emb = self.user_embedding(user_ids)

        # Get movie embedding
        movie_emb = self.movie_embedding(movie_ids)
        movie_repr = self.movie_proj(movie_emb)

        # Encode scores
        score_repr, gap_emb = self.score_encoder(
            metacritic_scores, rt_scores, imdb_scores, letterboxd_scores
        )

        # User-score preference
        user_score = torch.cat([user_emb, score_repr], dim=-1)
        score_pref = self.user_score_preference(user_score)

        # Personal score prediction
        personal_score = self.personal_score_head(user_score)

        # Score deviation
        deviation = self.deviation_head(user_score)

        # Contrarian tendency
        contrarian = self.contrarian_head(user_emb)

        # Movie recommendations
        movie_logits = self.movie_head(score_pref)

        # Alignment prediction if history provided
        alignment = None
        genre_alignment = None
        if user_history_scores is not None:
            alignment, genre_alignment = self.alignment_predictor(user_history_scores)

        # Critic matching
        critic_emb = self.critic_embedding.weight.mean(dim=0, keepdim=True)
        critic_match_input = torch.cat([user_emb, critic_emb.expand(batch_size, -1)], dim=-1)
        critic_match_logits = self.critic_match_head(user_score)

        return {
            'score_preference': score_pref,
            'personal_score': personal_score,
            'deviation': deviation,
            'contrarian': contrarian,
            'alignment': alignment,
            'genre_alignment': genre_alignment,
            'movie_logits': movie_logits,
            'critic_match_logits': critic_match_logits,
            'gap_embedding': gap_emb
        }

    def find_matching_critics(self, user_ids: torch.Tensor, user_rating_history: torch.Tensor,
                              top_k: int = 10) -> Tuple[torch.Tensor, torch.Tensor]:
        """Find critics that align with user preferences"""
        with torch.no_grad():
            user_emb = self.user_embedding(user_ids)

            # Get all critic embeddings
            all_critics = torch.arange(self.config.num_critics, device=user_ids.device)
            critic_emb = self.critic_embedding(all_critics)

            # Score each critic for each user
            scores = []
            for i in range(user_ids.size(0)):
                user_expanded = user_emb[i:i+1].expand(critic_emb.size(0), -1)
                match_input = torch.cat([user_expanded, critic_emb], dim=-1)
                match_scores = self.critic_match_head(match_input).diagonal()
                scores.append(match_scores)

            scores = torch.stack(scores)
            top_scores, top_critics = torch.topk(scores, top_k, dim=-1)

        return top_critics, top_scores

    def recommend_by_score_profile(self, user_ids: torch.Tensor,
                                   min_critic_score: Optional[float] = None,
                                   max_critic_score: Optional[float] = None,
                                   top_k: int = 10) -> Tuple[torch.Tensor, torch.Tensor]:
        """Recommend movies filtered by score ranges"""
        with torch.no_grad():
            user_emb = self.user_embedding(user_ids)

            # Default: use user preference
            movie_logits = self.movie_head(user_emb)

            top_scores, top_movies = torch.topk(F.softmax(movie_logits, dim=-1), top_k, dim=-1)

        return top_movies, top_scores

    def get_score_alignment_profile(self, user_ids: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Get detailed score alignment profile for users"""
        with torch.no_grad():
            user_emb = self.user_embedding(user_ids)
            contrarian = self.contrarian_head(user_emb)

        return {
            'contrarian_tendency': contrarian,
            'user_embedding': user_emb
        }


class CriticAudienceTrainer:
    """Trainer for Critic Audience Model"""

    def __init__(self, model: CriticAudienceModel, lr: float = 1e-4,
                 weight_decay: float = 0.01, device: str = 'cuda'):
        self.model = model.to(device)
        self.device = device

        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=10, T_mult=2
        )

        self.score_loss = nn.MSELoss()
        self.alignment_loss = nn.CrossEntropyLoss()
        self.deviation_loss = nn.L1Loss()

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Single training step"""
        self.model.train()

        user_ids = batch['user_ids'].to(self.device)
        movie_ids = batch['movie_ids'].to(self.device)
        metacritic_scores = batch['metacritic_scores'].to(self.device)
        rt_scores = batch['rt_scores'].to(self.device)
        imdb_scores = batch['imdb_scores'].to(self.device)
        letterboxd_scores = batch['letterboxd_scores'].to(self.device)
        target_scores = batch['user_ratings'].to(self.device)

        outputs = self.model(
            user_ids, movie_ids, metacritic_scores, rt_scores,
            imdb_scores, letterboxd_scores
        )

        # Score prediction loss
        score_loss = self.score_loss(outputs['personal_score'].squeeze(), target_scores)

        total_loss = score_loss

        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()

        return {
            'total_loss': total_loss.item(),
            'score_loss': score_loss.item()
        }

    def evaluate(self, dataloader) -> Dict[str, float]:
        """Evaluate model"""
        self.model.eval()
        total_metrics = {'score_mse': 0}
        num_batches = 0

        with torch.no_grad():
            for batch in dataloader:
                user_ids = batch['user_ids'].to(self.device)
                movie_ids = batch['movie_ids'].to(self.device)
                metacritic_scores = batch['metacritic_scores'].to(self.device)
                rt_scores = batch['rt_scores'].to(self.device)
                imdb_scores = batch['imdb_scores'].to(self.device)
                letterboxd_scores = batch['letterboxd_scores'].to(self.device)
                target_scores = batch['user_ratings'].to(self.device)

                outputs = self.model(
                    user_ids, movie_ids, metacritic_scores, rt_scores,
                    imdb_scores, letterboxd_scores
                )

                total_metrics['score_mse'] += F.mse_loss(
                    outputs['personal_score'].squeeze(), target_scores
                ).item()

                num_batches += 1

        return {k: v / num_batches for k, v in total_metrics.items()}
