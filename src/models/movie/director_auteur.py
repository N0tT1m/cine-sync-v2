"""
Director Auteur Model (DAM)
Models director filmography patterns, visual styles, and thematic preferences.
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
class DirectorConfig:
    """Configuration for Director Auteur Model"""
    embedding_dim: int = 256
    hidden_dim: int = 512
    num_heads: int = 8
    num_layers: int = 4
    num_directors: int = 50000
    num_movies: int = 100000
    num_users: int = 50000
    num_genres: int = 30
    num_themes: int = 100
    num_visual_styles: int = 50
    dropout: float = 0.1


class DirectorStyleEncoder(nn.Module):
    """Encodes a director's signature style from their filmography"""

    def __init__(self, config: DirectorConfig):
        super().__init__()
        self.config = config

        # Director embedding
        self.director_embedding = nn.Embedding(config.num_directors, config.embedding_dim)

        # Style components
        self.genre_embedding = nn.Embedding(config.num_genres, config.embedding_dim // 4)
        self.theme_embedding = nn.Embedding(config.num_themes, config.embedding_dim // 4)
        self.visual_style_embedding = nn.Embedding(config.num_visual_styles, config.embedding_dim // 4)

        # Temporal evolution layer (director style changes over career)
        self.career_position_encoding = nn.Parameter(torch.randn(50, config.embedding_dim))

        # Style aggregation
        self.style_attention = nn.MultiheadAttention(
            config.embedding_dim, config.num_heads, dropout=config.dropout, batch_first=True
        )

        # Style projection
        self.style_proj = nn.Sequential(
            nn.Linear(config.embedding_dim, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.hidden_dim)
        )

    def forward(self, director_ids: torch.Tensor, movie_genres: torch.Tensor,
                movie_themes: torch.Tensor, visual_styles: torch.Tensor,
                career_positions: torch.Tensor) -> torch.Tensor:
        """
        Encode director's style from their filmography.

        Args:
            director_ids: Director IDs [batch]
            movie_genres: Genres of director's films [batch, num_films, num_genres]
            movie_themes: Themes [batch, num_films, num_themes]
            visual_styles: Visual styles [batch, num_films]
            career_positions: Position in filmography [batch, num_films]
        """
        batch_size = director_ids.size(0)

        # Base director embedding
        director_emb = self.director_embedding(director_ids)  # [batch, emb_dim]

        # Aggregate genre preferences
        genre_emb = self.genre_embedding(movie_genres)  # [batch, films, genres, emb/4]
        genre_style = genre_emb.mean(dim=(1, 2))  # [batch, emb/4]

        # Aggregate theme preferences
        theme_emb = self.theme_embedding(movie_themes)
        theme_style = theme_emb.mean(dim=(1, 2))

        # Visual style encoding
        visual_emb = self.visual_style_embedding(visual_styles)  # [batch, films, emb/4]
        visual_style = visual_emb.mean(dim=1)

        # Combine style components with learned weights
        combined_style = torch.cat([genre_style, theme_style, visual_style,
                                   genre_style * theme_style], dim=-1)

        # Add career evolution (director style evolves over time)
        career_emb = self.career_position_encoding[career_positions.clamp(0, 49)]

        # Attention over filmography with career context
        style_query = director_emb.unsqueeze(1)
        style_keys = career_emb + combined_style.unsqueeze(1).expand_as(career_emb)

        attended_style, _ = self.style_attention(style_query, style_keys, style_keys)

        # Final style representation
        style = self.style_proj(attended_style.squeeze(1) + director_emb)

        return style


class DirectorFilmographyTransformer(nn.Module):
    """Transformer for processing a director's filmography"""

    def __init__(self, config: DirectorConfig):
        super().__init__()
        self.config = config

        # Movie embedding
        self.movie_embedding = nn.Embedding(config.num_movies, config.embedding_dim)

        # Temporal embedding for release years
        self.year_embedding = nn.Embedding(150, config.embedding_dim // 4)  # 1900-2050

        # Project to hidden dim
        self.input_proj = nn.Linear(config.embedding_dim + config.embedding_dim // 4,
                                    config.hidden_dim)

        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_dim,
            nhead=config.num_heads,
            dim_feedforward=config.hidden_dim * 4,
            dropout=config.dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=config.num_layers)

    def forward(self, movie_ids: torch.Tensor, release_years: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Process a director's filmography"""
        # Embed movies
        movie_emb = self.movie_embedding(movie_ids)

        # Embed years (relative to 1900)
        year_indices = (release_years - 1900).clamp(0, 149)
        year_emb = self.year_embedding(year_indices)

        # Combine and project
        x = torch.cat([movie_emb, year_emb], dim=-1)
        x = self.input_proj(x)

        # Transform
        if mask is not None:
            x = self.transformer(x, src_key_padding_mask=~mask)
        else:
            x = self.transformer(x)

        return x


class DirectorAuteurModel(nn.Module):
    """
    Complete Director Auteur Model for understanding director styles
    and recommending movies based on director preferences.
    """

    def __init__(self, config: Optional[DirectorConfig] = None):
        super().__init__()
        self.config = config or DirectorConfig()

        # User embedding
        self.user_embedding = nn.Embedding(self.config.num_users, self.config.hidden_dim)

        # Director style encoder
        self.style_encoder = DirectorStyleEncoder(self.config)

        # Filmography transformer
        self.filmography_transformer = DirectorFilmographyTransformer(self.config)

        # User-director preference modeling
        self.user_director_interaction = nn.Sequential(
            nn.Linear(self.config.hidden_dim * 2, self.config.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim)
        )

        # Director similarity network
        self.director_similarity = nn.Sequential(
            nn.Linear(self.config.hidden_dim * 2, self.config.hidden_dim),
            nn.GELU(),
            nn.Linear(self.config.hidden_dim, 1),
            nn.Sigmoid()
        )

        # Style preference predictor
        self.style_preference = nn.Sequential(
            nn.Linear(self.config.hidden_dim * 2, self.config.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.hidden_dim, self.config.num_visual_styles)
        )

        # Rating predictor
        self.rating_head = nn.Sequential(
            nn.Linear(self.config.hidden_dim * 2, self.config.hidden_dim),
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

    def encode_director(self, director_id: torch.Tensor, filmography_ids: torch.Tensor,
                       release_years: torch.Tensor) -> torch.Tensor:
        """Get director representation from their filmography"""
        filmography_encoded = self.filmography_transformer(filmography_ids, release_years)
        director_repr = filmography_encoded.mean(dim=1)  # Pool over filmography
        return director_repr

    def forward(self, user_ids: torch.Tensor, director_ids: torch.Tensor,
                filmography_ids: torch.Tensor, release_years: torch.Tensor,
                movie_genres: Optional[torch.Tensor] = None,
                movie_themes: Optional[torch.Tensor] = None,
                visual_styles: Optional[torch.Tensor] = None,
                career_positions: Optional[torch.Tensor] = None,
                mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass for director-based recommendations.

        Returns:
            - director_repr: Director representation
            - user_director_score: User-director affinity
            - rating_pred: Predicted rating for director's movies
            - style_logits: Predicted style preferences
            - movie_logits: Movie recommendation scores
        """
        batch_size = user_ids.size(0)

        # Get user embedding
        user_emb = self.user_embedding(user_ids)

        # Encode director from filmography
        director_repr = self.encode_director(director_ids, filmography_ids, release_years)

        # User-director interaction
        user_director = torch.cat([user_emb, director_repr], dim=-1)
        interaction = self.user_director_interaction(user_director)

        # Predict rating
        rating_pred = self.rating_head(user_director)

        # Style preferences
        style_logits = self.style_preference(user_director)

        # Movie recommendations
        movie_logits = self.movie_head(interaction)

        return {
            'director_repr': director_repr,
            'user_director_score': torch.sigmoid(interaction.mean(dim=-1)),
            'rating_pred': rating_pred,
            'style_logits': style_logits,
            'movie_logits': movie_logits,
            'interaction': interaction
        }

    def find_similar_directors(self, director_id: torch.Tensor,
                               all_director_reprs: torch.Tensor,
                               top_k: int = 10) -> Tuple[torch.Tensor, torch.Tensor]:
        """Find directors with similar styles"""
        with torch.no_grad():
            director_repr = all_director_reprs[director_id]

            # Compute similarities
            similarities = []
            for i in range(all_director_reprs.size(0)):
                if i != director_id:
                    pair = torch.cat([director_repr, all_director_reprs[i]], dim=-1)
                    sim = self.director_similarity(pair.unsqueeze(0))
                    similarities.append((i, sim.item()))

            # Sort by similarity
            similarities.sort(key=lambda x: x[1], reverse=True)

            top_directors = torch.tensor([s[0] for s in similarities[:top_k]])
            top_scores = torch.tensor([s[1] for s in similarities[:top_k]])

        return top_directors, top_scores

    def recommend_by_director(self, user_ids: torch.Tensor, director_repr: torch.Tensor,
                              top_k: int = 10) -> Tuple[torch.Tensor, torch.Tensor]:
        """Recommend movies based on director preference"""
        with torch.no_grad():
            user_emb = self.user_embedding(user_ids)
            user_director = torch.cat([user_emb, director_repr], dim=-1)
            interaction = self.user_director_interaction(user_director)
            logits = self.movie_head(interaction)
            scores, movie_ids = torch.topk(F.softmax(logits, dim=-1), top_k, dim=-1)
        return movie_ids, scores


class DirectorAuteurTrainer:
    """Trainer for Director Auteur Model"""

    def __init__(self, model: DirectorAuteurModel, lr: float = 1e-4,
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
        self.style_loss = nn.CrossEntropyLoss()
        self.movie_loss = nn.CrossEntropyLoss()

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Single training step"""
        self.model.train()

        user_ids = batch['user_ids'].to(self.device)
        director_ids = batch['director_ids'].to(self.device)
        filmography_ids = batch['filmography_ids'].to(self.device)
        release_years = batch['release_years'].to(self.device)
        target_ratings = batch['ratings'].to(self.device)
        target_movies = batch['target_movies'].to(self.device)

        outputs = self.model(user_ids, director_ids, filmography_ids, release_years)

        # Compute losses
        rating_loss = self.rating_loss(outputs['rating_pred'].squeeze(), target_ratings)
        movie_loss = self.movie_loss(outputs['movie_logits'], target_movies)

        total_loss = rating_loss + movie_loss

        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()

        return {
            'total_loss': total_loss.item(),
            'rating_loss': rating_loss.item(),
            'movie_loss': movie_loss.item()
        }

    def evaluate(self, dataloader) -> Dict[str, float]:
        """Evaluate model"""
        self.model.eval()
        total_metrics = {'rating_mse': 0, 'movie_acc': 0}
        num_batches = 0

        with torch.no_grad():
            for batch in dataloader:
                user_ids = batch['user_ids'].to(self.device)
                director_ids = batch['director_ids'].to(self.device)
                filmography_ids = batch['filmography_ids'].to(self.device)
                release_years = batch['release_years'].to(self.device)
                target_ratings = batch['ratings'].to(self.device)
                target_movies = batch['target_movies'].to(self.device)

                outputs = self.model(user_ids, director_ids, filmography_ids, release_years)

                total_metrics['rating_mse'] += F.mse_loss(
                    outputs['rating_pred'].squeeze(), target_ratings
                ).item()

                pred_movies = outputs['movie_logits'].argmax(dim=-1)
                total_metrics['movie_acc'] += (pred_movies == target_movies).float().mean().item()

                num_batches += 1

        return {k: v / num_batches for k, v in total_metrics.items()}
