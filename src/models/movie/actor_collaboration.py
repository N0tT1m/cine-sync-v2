"""
Actor Collaboration Model (ACM)
Models actor pairings, ensemble chemistry, and casting-based recommendations.
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
class ActorConfig:
    """Configuration for Actor Collaboration Model"""
    embedding_dim: int = 256
    hidden_dim: int = 512
    num_heads: int = 8
    num_layers: int = 4
    num_movies: int = 100000
    num_users: int = 50000
    num_actors: int = 100000
    max_cast_size: int = 20
    dropout: float = 0.1


class ActorEncoder(nn.Module):
    """Encodes individual actor features and career patterns"""

    def __init__(self, config: ActorConfig):
        super().__init__()

        # Actor embedding
        self.actor_embedding = nn.Embedding(config.num_actors, config.embedding_dim)

        # Role type embedding
        self.role_type_embedding = nn.Embedding(5, config.embedding_dim // 4)
        # Types: lead, supporting, cameo, voice, ensemble

        # Career stage encoding
        self.career_encoder = nn.Sequential(
            nn.Linear(4, config.embedding_dim // 4),  # years_active, num_films, avg_rating, star_power
            nn.GELU(),
            nn.Linear(config.embedding_dim // 4, config.embedding_dim // 4)
        )

        # Genre specialty (actors often specialize)
        self.genre_specialty = nn.Linear(30, config.embedding_dim // 4)  # 30 genres

        # Combine features
        self.actor_fusion = nn.Sequential(
            nn.Linear(config.embedding_dim + config.embedding_dim // 4 * 3, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.hidden_dim)
        )

    def forward(self, actor_ids: torch.Tensor, role_types: torch.Tensor,
                career_features: torch.Tensor, genre_distribution: torch.Tensor) -> torch.Tensor:
        """Encode actor features"""
        actor_emb = self.actor_embedding(actor_ids)
        role_emb = self.role_type_embedding(role_types)
        career_emb = self.career_encoder(career_features)
        genre_emb = self.genre_specialty(genre_distribution)

        combined = torch.cat([actor_emb, role_emb, career_emb, genre_emb], dim=-1)
        return self.actor_fusion(combined)


class CollaborationEncoder(nn.Module):
    """Encodes actor collaboration patterns"""

    def __init__(self, config: ActorConfig):
        super().__init__()

        # Pairwise chemistry encoder
        self.chemistry_encoder = nn.Sequential(
            nn.Linear(config.hidden_dim * 2, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.hidden_dim)
        )

        # Collaboration history (outputs hidden_dim to match chemistry_encoder)
        self.collab_history = nn.Sequential(
            nn.Linear(5, config.embedding_dim // 4),  # num_films_together, avg_rating, years_span, genre_overlap, role_consistency
            nn.GELU(),
            nn.Linear(config.embedding_dim // 4, config.hidden_dim)
        )

        # Multi-head attention for ensemble dynamics
        self.ensemble_attention = nn.MultiheadAttention(
            config.hidden_dim, config.num_heads, dropout=config.dropout, batch_first=True
        )

    def forward(self, actor1_repr: torch.Tensor, actor2_repr: torch.Tensor,
                collab_features: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Encode collaboration between two actors"""
        # Chemistry encoding
        chemistry = self.chemistry_encoder(torch.cat([actor1_repr, actor2_repr], dim=-1))

        if collab_features is not None:
            history_emb = self.collab_history(collab_features)
            chemistry = chemistry + history_emb.expand_as(chemistry)

        return chemistry

    def encode_ensemble(self, cast_reprs: torch.Tensor,
                        mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Encode full cast ensemble"""
        attended, _ = self.ensemble_attention(cast_reprs, cast_reprs, cast_reprs,
                                              key_padding_mask=mask)
        return attended


class ActorCollaborationModel(nn.Module):
    """
    Complete Actor Collaboration Model for cast-based recommendations.
    """

    def __init__(self, config: Optional[ActorConfig] = None):
        super().__init__()
        self.config = config or ActorConfig()

        # Core embeddings
        self.movie_embedding = nn.Embedding(self.config.num_movies, self.config.embedding_dim)
        self.user_embedding = nn.Embedding(self.config.num_users, self.config.hidden_dim)

        # Actor encoder
        self.actor_encoder = ActorEncoder(self.config)

        # Collaboration encoder
        self.collab_encoder = CollaborationEncoder(self.config)

        # Movie projection
        self.movie_proj = nn.Linear(self.config.embedding_dim, self.config.hidden_dim)

        # User actor preference
        self.user_actor_preference = nn.Sequential(
            nn.Linear(self.config.hidden_dim * 2, self.config.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim)
        )

        # Actor affinity head (which actors user likes)
        self.actor_affinity_head = nn.Sequential(
            nn.Linear(self.config.hidden_dim * 2, self.config.hidden_dim),
            nn.GELU(),
            nn.Linear(self.config.hidden_dim, 1),
            nn.Sigmoid()
        )

        # Chemistry predictor (will user like this pairing?)
        self.chemistry_head = nn.Sequential(
            nn.Linear(self.config.hidden_dim * 2, self.config.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.hidden_dim, 1),
            nn.Sigmoid()
        )

        # Ensemble quality predictor
        self.ensemble_head = nn.Sequential(
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim // 2),
            nn.GELU(),
            nn.Linear(self.config.hidden_dim // 2, 1)
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

        # Actor recommendation head
        self.actor_recommendation_head = nn.Linear(self.config.hidden_dim, self.config.num_actors)

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
                   career_features: torch.Tensor, genre_distributions: torch.Tensor,
                   mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Encode a movie's full cast"""
        batch_size, cast_size = actor_ids.shape

        # Ensure career_features and genre_distributions have the expected shape
        # Expected: [batch, cast_size, feature_dim]
        # If career_features is [batch, 4], broadcast to [batch, cast_size, 4]
        if career_features.dim() == 2 and career_features.size(1) == 4:
            career_features = career_features.unsqueeze(1).expand(-1, cast_size, -1)

        # If genre_distributions is [batch, 30], broadcast to [batch, cast_size, 30]
        if genre_distributions.dim() == 2 and genre_distributions.size(1) == 30:
            genre_distributions = genre_distributions.unsqueeze(1).expand(-1, cast_size, -1)

        # Encode each actor
        actor_reprs = []
        for i in range(cast_size):
            actor_repr = self.actor_encoder(
                actor_ids[:, i], role_types[:, i],
                career_features[:, i], genre_distributions[:, i]
            )
            actor_reprs.append(actor_repr)

        actor_reprs = torch.stack(actor_reprs, dim=1)  # [batch, cast_size, hidden]

        # Encode ensemble
        ensemble_repr = self.collab_encoder.encode_ensemble(actor_reprs, mask)

        return ensemble_repr

    def forward(self, user_ids: torch.Tensor, movie_ids: torch.Tensor,
                actor_ids: torch.Tensor, role_types: torch.Tensor,
                career_features: torch.Tensor, genre_distributions: torch.Tensor,
                collab_features: Optional[torch.Tensor] = None,
                cast_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass for actor-based recommendations.
        """
        batch_size = user_ids.size(0)

        # Get user embedding
        user_emb = self.user_embedding(user_ids)

        # Get movie embedding
        movie_emb = self.movie_embedding(movie_ids)
        movie_repr = self.movie_proj(movie_emb)

        # Encode cast
        cast_repr = self.encode_cast(actor_ids, role_types, career_features,
                                    genre_distributions, cast_mask)
        ensemble_repr = cast_repr.mean(dim=1)  # Pool over cast

        # User-actor preference
        user_actor = torch.cat([user_emb, ensemble_repr], dim=-1)
        actor_pref = self.user_actor_preference(user_actor)

        # Actor affinity
        actor_affinity = self.actor_affinity_head(user_actor)

        # Ensemble quality
        ensemble_quality = self.ensemble_head(ensemble_repr)

        # Rating prediction
        rating_input = torch.cat([user_emb, movie_repr, ensemble_repr], dim=-1)
        rating_pred = self.rating_head(rating_input)

        # Movie recommendations
        movie_logits = self.movie_head(actor_pref)

        # Actor recommendations
        actor_logits = self.actor_recommendation_head(user_emb)

        return {
            'actor_preference': actor_pref,
            'cast_repr': cast_repr,
            'ensemble_repr': ensemble_repr,
            'actor_affinity': actor_affinity,
            'ensemble_quality': ensemble_quality,
            'rating_pred': rating_pred,
            'movie_logits': movie_logits,
            'actor_logits': actor_logits
        }

    def predict_chemistry(self, user_ids: torch.Tensor,
                         actor1_repr: torch.Tensor,
                         actor2_repr: torch.Tensor) -> torch.Tensor:
        """Predict if user will like a specific actor pairing"""
        with torch.no_grad():
            user_emb = self.user_embedding(user_ids)
            chemistry = self.collab_encoder(actor1_repr, actor2_repr)
            user_chemistry = torch.cat([user_emb, chemistry], dim=-1)
            return self.chemistry_head(user_chemistry)

    def recommend_by_actor(self, user_ids: torch.Tensor, actor_id: torch.Tensor,
                          top_k: int = 10) -> Tuple[torch.Tensor, torch.Tensor]:
        """Recommend movies featuring a specific actor"""
        with torch.no_grad():
            user_emb = self.user_embedding(user_ids)
            actor_emb = self.actor_encoder.actor_embedding(actor_id)

            # Get user-actor preference
            user_actor = torch.cat([user_emb, actor_emb.expand(user_ids.size(0), -1)], dim=-1)
            pref = self.user_actor_preference(user_actor)

            movie_logits = self.movie_head(pref)
            top_scores, top_movies = torch.topk(F.softmax(movie_logits, dim=-1), top_k, dim=-1)

        return top_movies, top_scores

    def find_favorite_actors(self, user_ids: torch.Tensor,
                            top_k: int = 10) -> Tuple[torch.Tensor, torch.Tensor]:
        """Find user's favorite actors"""
        with torch.no_grad():
            user_emb = self.user_embedding(user_ids)
            actor_logits = self.actor_recommendation_head(user_emb)
            top_scores, top_actors = torch.topk(F.softmax(actor_logits, dim=-1), top_k, dim=-1)

        return top_actors, top_scores


class ActorCollaborationTrainer:
    """Trainer for Actor Collaboration Model"""

    def __init__(self, model: ActorCollaborationModel, lr: float = 1e-4,
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
        self.chemistry_loss = nn.BCELoss()

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Single training step"""
        self.model.train()

        user_ids = batch['user_ids'].to(self.device)
        movie_ids = batch['movie_ids'].to(self.device)
        actor_ids = batch['actor_ids'].to(self.device)
        role_types = batch['role_types'].to(self.device)
        career_features = batch['career_features'].to(self.device)
        genre_distributions = batch['genre_distributions'].to(self.device)
        target_ratings = batch['ratings'].to(self.device)

        cast_mask = batch.get('cast_mask')
        if cast_mask is not None:
            cast_mask = cast_mask.to(self.device)

        outputs = self.model(
            user_ids, movie_ids, actor_ids, role_types,
            career_features, genre_distributions, cast_mask=cast_mask
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
                actor_ids = batch['actor_ids'].to(self.device)
                role_types = batch['role_types'].to(self.device)
                career_features = batch['career_features'].to(self.device)
                genre_distributions = batch['genre_distributions'].to(self.device)
                target_ratings = batch['ratings'].to(self.device)

                outputs = self.model(
                    user_ids, movie_ids, actor_ids, role_types,
                    career_features, genre_distributions
                )

                total_metrics['rating_mse'] += F.mse_loss(
                    outputs['rating_pred'].squeeze(), target_ratings
                ).item()

                num_batches += 1

        return {k: v / num_batches for k, v in total_metrics.items()}
