"""
Era Style Model (ESM)
Models user preferences for different filmmaking eras and period-specific styles.
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
class EraConfig:
    """Configuration for Era Style Model"""
    embedding_dim: int = 256
    hidden_dim: int = 512
    num_heads: int = 8
    num_layers: int = 4
    num_movies: int = 100000
    num_users: int = 50000
    num_eras: int = 15  # Silent, Golden Age, New Hollywood, Blockbuster, etc.
    num_decades: int = 13  # 1920s-2020s
    num_visual_styles: int = 30  # Noir, technicolor, CGI-heavy, etc.
    num_audio_styles: int = 20  # Silent, mono, stereo, Dolby Atmos, etc.
    dropout: float = 0.1


class EraEncoder(nn.Module):
    """Encodes film era characteristics"""

    def __init__(self, config: EraConfig):
        super().__init__()

        # Era embedding
        self.era_embedding = nn.Embedding(config.num_eras, config.embedding_dim)

        # Decade embedding
        self.decade_embedding = nn.Embedding(config.num_decades, config.embedding_dim // 2)

        # Year encoding (continuous)
        self.year_encoder = nn.Sequential(
            nn.Linear(1, config.embedding_dim // 4),
            nn.GELU(),
            nn.Linear(config.embedding_dim // 4, config.embedding_dim // 4)
        )

        # Visual style embedding
        self.visual_embedding = nn.Embedding(config.num_visual_styles, config.embedding_dim // 4)

        # Audio/technical style
        self.audio_embedding = nn.Embedding(config.num_audio_styles, config.embedding_dim // 4)

        # Combine era features
        # Input: era(256) + decade(128) + year(64) + visual(64) + audio(64) = 576
        era_input_dim = config.embedding_dim + config.embedding_dim // 2 + config.embedding_dim // 4 * 3
        self.era_fusion = nn.Sequential(
            nn.Linear(era_input_dim, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.hidden_dim)
        )

    def forward(self, era_ids: torch.Tensor, decade_ids: torch.Tensor,
                years: torch.Tensor, visual_styles: torch.Tensor,
                audio_styles: torch.Tensor) -> torch.Tensor:
        """
        Encode era features.

        Args:
            era_ids: Film era IDs [batch]
            decade_ids: Decade IDs [batch]
            years: Release years [batch]
            visual_styles: Visual style IDs [batch]
            audio_styles: Audio style IDs [batch]
        """
        era_emb = self.era_embedding(era_ids)
        decade_emb = self.decade_embedding(decade_ids)

        # Normalize years to 0-1 range (1920-2030)
        year_norm = (years.float() - 1920) / 110
        year_emb = self.year_encoder(year_norm.unsqueeze(-1))

        visual_emb = self.visual_embedding(visual_styles)
        audio_emb = self.audio_embedding(audio_styles)

        combined = torch.cat([era_emb, decade_emb, year_emb, visual_emb, audio_emb], dim=-1)
        return self.era_fusion(combined)


class EraStyleEvolution(nn.Module):
    """Models how user preferences for eras evolve over time"""

    def __init__(self, config: EraConfig):
        super().__init__()

        # LSTM for temporal modeling of era preferences
        self.preference_lstm = nn.LSTM(
            config.hidden_dim, config.hidden_dim // 2,
            num_layers=2, batch_first=True, bidirectional=True,
            dropout=config.dropout
        )

        # Attention over viewing history
        self.era_attention = nn.MultiheadAttention(
            config.hidden_dim, config.num_heads, dropout=config.dropout, batch_first=True
        )

        # Era transition modeling
        self.transition_net = nn.Sequential(
            nn.Linear(config.hidden_dim * 2, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, config.num_eras)
        )

    def forward(self, era_sequence: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Model era preference evolution.

        Args:
            era_sequence: Sequence of era representations [batch, seq_len, hidden]
            mask: Attention mask
        """
        # LSTM encoding
        lstm_out, (h_n, _) = self.preference_lstm(era_sequence)

        # Self-attention
        attended, attention_weights = self.era_attention(lstm_out, lstm_out, lstm_out,
                                                         key_padding_mask=mask)

        # Get current preference state
        current_state = h_n[-2:].transpose(0, 1).contiguous().view(era_sequence.size(0), -1)

        # Predict next era preference
        next_era_logits = self.transition_net(current_state)

        return attended, next_era_logits


class EraStyleModel(nn.Module):
    """
    Complete Era Style Model for understanding period preferences in films.
    """

    def __init__(self, config: Optional[EraConfig] = None):
        super().__init__()
        self.config = config or EraConfig()

        # Core embeddings
        self.movie_embedding = nn.Embedding(self.config.num_movies, self.config.embedding_dim)
        self.user_embedding = nn.Embedding(self.config.num_users, self.config.hidden_dim)

        # Era encoder
        self.era_encoder = EraEncoder(self.config)

        # Era evolution tracker
        self.era_evolution = EraStyleEvolution(self.config)

        # Movie projection
        self.movie_proj = nn.Linear(self.config.embedding_dim, self.config.hidden_dim)

        # User era preference
        self.user_era_preference = nn.Sequential(
            nn.Linear(self.config.hidden_dim * 2, self.config.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim)
        )

        # Era affinity predictor (how much user likes each era)
        self.era_affinity_head = nn.Sequential(
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim // 2),
            nn.GELU(),
            nn.Linear(self.config.hidden_dim // 2, self.config.num_eras)
        )

        # Nostalgia factor (preference for films from user's past)
        self.nostalgia_head = nn.Sequential(
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

        # Movie recommendation head
        self.movie_head = nn.Linear(self.config.hidden_dim, self.config.num_movies)

        # Style compatibility predictor
        self.style_compatibility_head = nn.Sequential(
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
                era_ids: torch.Tensor, decade_ids: torch.Tensor,
                years: torch.Tensor, visual_styles: torch.Tensor,
                audio_styles: torch.Tensor,
                user_birth_year: Optional[torch.Tensor] = None,
                history_eras: Optional[torch.Tensor] = None,
                history_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass for era-aware recommendations.
        """
        batch_size = user_ids.size(0)
        device = user_ids.device

        # Get user embedding
        user_emb = self.user_embedding(user_ids)

        # Get movie embedding
        movie_emb = self.movie_embedding(movie_ids)
        movie_repr = self.movie_proj(movie_emb)

        # Encode era
        era_repr = self.era_encoder(era_ids, decade_ids, years, visual_styles, audio_styles)

        # User-era preference
        user_era = torch.cat([user_emb, era_repr], dim=-1)
        era_pref = self.user_era_preference(user_era)

        # Era affinity
        era_affinity = self.era_affinity_head(era_pref)

        # Nostalgia factor
        nostalgia = self.nostalgia_head(user_emb)

        # If user birth year provided, adjust for nostalgia
        if user_birth_year is not None:
            age_at_release = years - user_birth_year
            nostalgia_bonus = torch.sigmoid((age_at_release.float() - 15) / 5)  # Peak at teens
            nostalgia = nostalgia * nostalgia_bonus.unsqueeze(-1)

        # Style compatibility
        style_compat = self.style_compatibility_head(user_era)

        # Rating prediction
        rating_input = torch.cat([user_emb, movie_repr, era_pref], dim=-1)
        rating_pred = self.rating_head(rating_input)

        # Movie recommendations
        movie_logits = self.movie_head(era_pref)

        # Era evolution if history provided
        next_era_logits = None
        if history_eras is not None:
            history_repr = self.era_encoder(
                history_eras[:, :, 0], history_eras[:, :, 1], history_eras[:, :, 2],
                history_eras[:, :, 3], history_eras[:, :, 4]
            )
            _, next_era_logits = self.era_evolution(history_repr, history_mask)

        return {
            'era_preference': era_pref,
            'era_affinity': era_affinity,
            'nostalgia_factor': nostalgia,
            'style_compatibility': style_compat,
            'rating_pred': rating_pred,
            'movie_logits': movie_logits,
            'next_era_logits': next_era_logits
        }

    def get_era_profile(self, user_ids: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Get user's era preference profile"""
        with torch.no_grad():
            user_emb = self.user_embedding(user_ids)

            # Get affinity for each era
            affinities = []
            for era_id in range(self.config.num_eras):
                era_tensor = torch.tensor([era_id], device=user_ids.device).expand(user_ids.size(0))
                decade_tensor = torch.zeros_like(era_tensor)
                year_tensor = torch.tensor([1970], device=user_ids.device).expand(user_ids.size(0))
                visual_tensor = torch.zeros_like(era_tensor)
                audio_tensor = torch.zeros_like(era_tensor)

                era_repr = self.era_encoder(era_tensor, decade_tensor, year_tensor,
                                           visual_tensor, audio_tensor)
                user_era = torch.cat([user_emb, era_repr], dim=-1)
                era_pref = self.user_era_preference(user_era)
                affinity = self.era_affinity_head(era_pref)
                affinities.append(affinity[:, era_id:era_id+1])

            affinities = torch.cat(affinities, dim=-1)

        return {
            'era_affinities': affinities,
            'nostalgia_factor': self.nostalgia_head(user_emb)
        }

    def recommend_by_era(self, user_ids: torch.Tensor, target_era: int,
                         top_k: int = 10) -> Tuple[torch.Tensor, torch.Tensor]:
        """Recommend movies from a specific era"""
        with torch.no_grad():
            user_emb = self.user_embedding(user_ids)

            # Create era representation
            era_tensor = torch.tensor([target_era], device=user_ids.device).expand(user_ids.size(0))
            decade_tensor = torch.zeros_like(era_tensor)
            year_tensor = torch.tensor([1970], device=user_ids.device).expand(user_ids.size(0))
            visual_tensor = torch.zeros_like(era_tensor)
            audio_tensor = torch.zeros_like(era_tensor)

            era_repr = self.era_encoder(era_tensor, decade_tensor, year_tensor,
                                       visual_tensor, audio_tensor)
            user_era = torch.cat([user_emb, era_repr], dim=-1)
            era_pref = self.user_era_preference(user_era)

            movie_logits = self.movie_head(era_pref)
            top_scores, top_movies = torch.topk(F.softmax(movie_logits, dim=-1), top_k, dim=-1)

        return top_movies, top_scores


class EraStyleTrainer:
    """Trainer for Era Style Model"""

    def __init__(self, model: EraStyleModel, lr: float = 1e-4,
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
        self.era_loss = nn.CrossEntropyLoss()
        self.compatibility_loss = nn.BCELoss()

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Single training step"""
        self.model.train()

        user_ids = batch['user_ids'].to(self.device)
        movie_ids = batch['movie_ids'].to(self.device)
        era_ids = batch['era_ids'].to(self.device)
        decade_ids = batch['decade_ids'].to(self.device)
        years = batch['years'].to(self.device)
        visual_styles = batch['visual_styles'].to(self.device)
        audio_styles = batch['audio_styles'].to(self.device)
        target_ratings = batch['ratings'].to(self.device)

        outputs = self.model(
            user_ids, movie_ids, era_ids, decade_ids, years, visual_styles, audio_styles
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
                era_ids = batch['era_ids'].to(self.device)
                decade_ids = batch['decade_ids'].to(self.device)
                years = batch['years'].to(self.device)
                visual_styles = batch['visual_styles'].to(self.device)
                audio_styles = batch['audio_styles'].to(self.device)
                target_ratings = batch['ratings'].to(self.device)

                outputs = self.model(
                    user_ids, movie_ids, era_ids, decade_ids, years, visual_styles, audio_styles
                )

                total_metrics['rating_mse'] += F.mse_loss(
                    outputs['rating_pred'].squeeze(), target_ratings
                ).item()

                num_batches += 1

        return {k: v / num_batches for k, v in total_metrics.items()}
