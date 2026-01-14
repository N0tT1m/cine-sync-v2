"""
Studio Fingerprint Model (SFM)
Models studio-specific styles, production patterns, and brand preferences.
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
class StudioConfig:
    """Configuration for Studio Fingerprint Model"""
    embedding_dim: int = 256
    hidden_dim: int = 512
    num_heads: int = 8
    num_layers: int = 4
    num_movies: int = 100000
    num_users: int = 50000
    num_studios: int = 1000
    num_distributors: int = 500
    num_production_companies: int = 5000
    dropout: float = 0.1


class StudioEncoder(nn.Module):
    """Encodes studio characteristics and style"""

    def __init__(self, config: StudioConfig):
        super().__init__()

        # Studio embedding
        self.studio_embedding = nn.Embedding(config.num_studios, config.embedding_dim)

        # Distributor embedding
        self.distributor_embedding = nn.Embedding(config.num_distributors, config.embedding_dim // 2)

        # Production company embedding
        self.prod_company_embedding = nn.Embedding(config.num_production_companies, config.embedding_dim // 2)

        # Studio characteristics (training data provides 16 features)
        self.characteristics_encoder = nn.Sequential(
            nn.Linear(16, config.embedding_dim // 4),  # avg_budget, avg_rating, genre_diversity, sequel_rate, oscar_rate, etc.
            nn.GELU(),
            nn.Linear(config.embedding_dim // 4, config.embedding_dim // 4)
        )

        # Studio type embedding (major, mini-major, indie, streaming, arthouse, international, etc.)
        # 11 types to support indices 0-10
        self.studio_type_embedding = nn.Embedding(11, config.embedding_dim // 4)

        # Combine features
        # Input: studio(256) + dist(128) + prod(128) + char(64) + type(64) = 640
        fusion_input_dim = config.embedding_dim + config.embedding_dim // 2 + config.embedding_dim // 2 + config.embedding_dim // 4 + config.embedding_dim // 4
        self.studio_fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.hidden_dim)
        )

    def forward(self, studio_ids: torch.Tensor, distributor_ids: torch.Tensor,
                prod_company_ids: torch.Tensor, characteristics: torch.Tensor,
                studio_types: torch.Tensor) -> torch.Tensor:
        """Encode studio features"""
        studio_emb = self.studio_embedding(studio_ids)
        dist_emb = self.distributor_embedding(distributor_ids)
        prod_emb = self.prod_company_embedding(prod_company_ids)
        char_emb = self.characteristics_encoder(characteristics)
        type_emb = self.studio_type_embedding(studio_types)

        combined = torch.cat([studio_emb, dist_emb, prod_emb, char_emb, type_emb], dim=-1)
        return self.studio_fusion(combined)


class StudioStyleAnalyzer(nn.Module):
    """Analyzes studio-specific filmmaking patterns"""

    def __init__(self, config: StudioConfig):
        super().__init__()

        # Genre distribution analyzer
        self.genre_analyzer = nn.Sequential(
            nn.Linear(30, config.embedding_dim // 2),
            nn.GELU(),
            nn.Linear(config.embedding_dim // 2, config.embedding_dim // 4)
        )

        # Budget-quality correlation
        self.budget_quality = nn.Sequential(
            nn.Linear(4, config.embedding_dim // 4),  # budget_range, quality_mean, quality_std, roi
            nn.GELU(),
            nn.Linear(config.embedding_dim // 4, config.embedding_dim // 4)
        )

        # Franchise tendency (expanded features)
        self.franchise_encoder = nn.Sequential(
            nn.Linear(8, config.embedding_dim // 4),  # franchise features: ratio, avg_length, reboot_rate, sequel_rate, spinoff_rate, crossover_rate, universe_rate, standalone_rate
            nn.GELU(),
            nn.Linear(config.embedding_dim // 4, config.embedding_dim // 4)
        )

        # Combine style features
        self.style_fusion = nn.Sequential(
            nn.Linear(config.embedding_dim // 4 * 3, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.hidden_dim)
        )

    def forward(self, genre_dist: torch.Tensor, budget_quality: torch.Tensor,
                franchise_features: torch.Tensor) -> torch.Tensor:
        """Analyze studio style"""
        genre_style = self.genre_analyzer(genre_dist)
        budget_style = self.budget_quality(budget_quality)
        franchise_style = self.franchise_encoder(franchise_features)

        combined = torch.cat([genre_style, budget_style, franchise_style], dim=-1)
        return self.style_fusion(combined)


class StudioFingerprintModel(nn.Module):
    """
    Complete Studio Fingerprint Model for studio-aware recommendations.
    """

    def __init__(self, config: Optional[StudioConfig] = None):
        super().__init__()
        self.config = config or StudioConfig()

        # Core embeddings
        self.movie_embedding = nn.Embedding(self.config.num_movies, self.config.embedding_dim)
        self.user_embedding = nn.Embedding(self.config.num_users, self.config.hidden_dim)

        # Studio encoder
        self.studio_encoder = StudioEncoder(self.config)

        # Style analyzer
        self.style_analyzer = StudioStyleAnalyzer(self.config)

        # Movie projection
        self.movie_proj = nn.Linear(self.config.embedding_dim, self.config.hidden_dim)

        # User studio preference
        self.user_studio_preference = nn.Sequential(
            nn.Linear(self.config.hidden_dim * 2, self.config.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim)
        )

        # Studio affinity head
        self.studio_affinity_head = nn.Sequential(
            nn.Linear(self.config.hidden_dim * 2, self.config.hidden_dim),
            nn.GELU(),
            nn.Linear(self.config.hidden_dim, self.config.num_studios)
        )

        # Brand loyalty predictor
        self.brand_loyalty_head = nn.Sequential(
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim // 2),
            nn.GELU(),
            nn.Linear(self.config.hidden_dim // 2, 1),
            nn.Sigmoid()
        )

        # Quality expectation predictor (expected rating for studio)
        self.quality_expectation_head = nn.Sequential(
            nn.Linear(self.config.hidden_dim * 2, self.config.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.config.dropout),
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
                studio_ids: torch.Tensor, distributor_ids: torch.Tensor,
                prod_company_ids: torch.Tensor, characteristics: torch.Tensor,
                studio_types: torch.Tensor, genre_dist: torch.Tensor,
                budget_quality: torch.Tensor, franchise_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass for studio-aware recommendations.
        """
        batch_size = user_ids.size(0)

        # Get user embedding
        user_emb = self.user_embedding(user_ids)

        # Get movie embedding
        movie_emb = self.movie_embedding(movie_ids)
        movie_repr = self.movie_proj(movie_emb)

        # Encode studio
        studio_repr = self.studio_encoder(
            studio_ids, distributor_ids, prod_company_ids, characteristics, studio_types
        )

        # Analyze style
        style_repr = self.style_analyzer(genre_dist, budget_quality, franchise_features)

        # Combined studio representation
        combined_studio = studio_repr + style_repr

        # User-studio preference
        user_studio = torch.cat([user_emb, combined_studio], dim=-1)
        studio_pref = self.user_studio_preference(user_studio)

        # Studio affinity
        studio_affinity = self.studio_affinity_head(user_studio)

        # Brand loyalty
        brand_loyalty = self.brand_loyalty_head(user_emb)

        # Quality expectation
        quality_expectation = self.quality_expectation_head(user_studio)

        # Rating prediction
        rating_input = torch.cat([user_emb, movie_repr, combined_studio], dim=-1)
        rating_pred = self.rating_head(rating_input)

        # Movie recommendations
        movie_logits = self.movie_head(studio_pref)

        return {
            'studio_preference': studio_pref,
            'studio_repr': combined_studio,
            'studio_affinity': studio_affinity,
            'brand_loyalty': brand_loyalty,
            'quality_expectation': quality_expectation,
            'rating_pred': rating_pred,
            'movie_logits': movie_logits
        }

    def get_favorite_studios(self, user_ids: torch.Tensor,
                            top_k: int = 10) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get user's favorite studios"""
        with torch.no_grad():
            user_emb = self.user_embedding(user_ids)

            # Create default studio representation
            dummy_studio = torch.zeros(user_ids.size(0), self.config.hidden_dim, device=user_ids.device)
            user_studio = torch.cat([user_emb, dummy_studio], dim=-1)

            affinity = self.studio_affinity_head(user_studio)
            top_scores, top_studios = torch.topk(F.softmax(affinity, dim=-1), top_k, dim=-1)

        return top_studios, top_scores

    def recommend_by_studio(self, user_ids: torch.Tensor, studio_id: int,
                           top_k: int = 10) -> Tuple[torch.Tensor, torch.Tensor]:
        """Recommend movies from a specific studio"""
        with torch.no_grad():
            user_emb = self.user_embedding(user_ids)
            studio_emb = self.studio_encoder.studio_embedding.weight[studio_id]

            user_studio = torch.cat([user_emb, studio_emb.expand(user_ids.size(0), -1)], dim=-1)
            pref = self.user_studio_preference(user_studio)

            movie_logits = self.movie_head(pref)
            top_scores, top_movies = torch.topk(F.softmax(movie_logits, dim=-1), top_k, dim=-1)

        return top_movies, top_scores


class StudioFingerprintTrainer:
    """Trainer for Studio Fingerprint Model"""

    def __init__(self, model: StudioFingerprintModel, lr: float = 1e-4,
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
        studio_ids = batch['studio_ids'].to(self.device)
        distributor_ids = batch['distributor_ids'].to(self.device)
        prod_company_ids = batch['prod_company_ids'].to(self.device)
        characteristics = batch['characteristics'].to(self.device)
        studio_types = batch['studio_types'].to(self.device)
        genre_dist = batch['genre_dist'].to(self.device)
        budget_quality = batch['budget_quality'].to(self.device)
        franchise_features = batch['franchise_features'].to(self.device)
        target_ratings = batch['ratings'].to(self.device)

        outputs = self.model(
            user_ids, movie_ids, studio_ids, distributor_ids,
            prod_company_ids, characteristics, studio_types,
            genre_dist, budget_quality, franchise_features
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
                studio_ids = batch['studio_ids'].to(self.device)
                distributor_ids = batch['distributor_ids'].to(self.device)
                prod_company_ids = batch['prod_company_ids'].to(self.device)
                characteristics = batch['characteristics'].to(self.device)
                studio_types = batch['studio_types'].to(self.device)
                genre_dist = batch['genre_dist'].to(self.device)
                budget_quality = batch['budget_quality'].to(self.device)
                franchise_features = batch['franchise_features'].to(self.device)
                target_ratings = batch['ratings'].to(self.device)

                outputs = self.model(
                    user_ids, movie_ids, studio_ids, distributor_ids,
                    prod_company_ids, characteristics, studio_types,
                    genre_dist, budget_quality, franchise_features
                )

                total_metrics['rating_mse'] += F.mse_loss(
                    outputs['rating_pred'].squeeze(), target_ratings
                ).item()

                num_batches += 1

        return {k: v / num_batches for k, v in total_metrics.items()}
