"""
Adaptation Source Model (ASM)
Models relationships between movies and their source materials (books, comics, games, etc.).
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
class AdaptationConfig:
    """Configuration for Adaptation Source Model"""
    embedding_dim: int = 256
    hidden_dim: int = 512
    num_heads: int = 8
    num_layers: int = 4
    num_movies: int = 100000
    num_users: int = 50000
    num_sources: int = 50000  # Books, comics, games, etc.
    num_source_types: int = 15  # novel, comic, manga, video_game, play, etc.
    dropout: float = 0.1


class SourceMaterialEncoder(nn.Module):
    """Encodes source material characteristics"""

    def __init__(self, config: AdaptationConfig):
        super().__init__()

        # Source type embedding
        self.source_type_embedding = nn.Embedding(config.num_source_types, config.embedding_dim)
        # Types: novel, comic, manga, video_game, play, short_story, poem,
        #        true_story, article, tv_series, anime, musical, opera, folk_tale, mythology

        # Source embedding
        self.source_embedding = nn.Embedding(config.num_sources, config.embedding_dim)

        # Source characteristics
        self.characteristics_encoder = nn.Sequential(
            nn.Linear(8, config.embedding_dim // 4),  # popularity, critical_acclaim, year_published, num_adaptations, is_series, length, cultural_significance, author_prestige
            nn.GELU(),
            nn.Linear(config.embedding_dim // 4, config.embedding_dim // 4)
        )

        # Genre mapping (source genre vs typical movie genre)
        self.genre_mapping = nn.Linear(30 * 2, config.embedding_dim // 4)  # source_genre + movie_genre

        # Combine
        self.source_fusion = nn.Sequential(
            nn.Linear(config.embedding_dim * 2 + config.embedding_dim // 2, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.hidden_dim)
        )

    def forward(self, source_type_ids: torch.Tensor, source_ids: torch.Tensor,
                characteristics: torch.Tensor, genre_features: torch.Tensor) -> torch.Tensor:
        """Encode source material"""
        type_emb = self.source_type_embedding(source_type_ids)
        source_emb = self.source_embedding(source_ids)
        char_emb = self.characteristics_encoder(characteristics)
        genre_emb = self.genre_mapping(genre_features)

        combined = torch.cat([type_emb, source_emb, char_emb, genre_emb], dim=-1)
        return self.source_fusion(combined)


class AdaptationQualityAnalyzer(nn.Module):
    """Analyzes adaptation quality and faithfulness"""

    def __init__(self, config: AdaptationConfig):
        super().__init__()

        # Faithfulness features (training data provides single scalar)
        self.faithfulness_encoder = nn.Sequential(
            nn.Linear(1, config.embedding_dim // 4),
            nn.GELU(),
            nn.Linear(config.embedding_dim // 4, config.embedding_dim // 4)
        )

        # Critical reception comparison (training data provides single scalar)
        self.reception_encoder = nn.Sequential(
            nn.Linear(1, config.embedding_dim // 4),
            nn.GELU(),
            nn.Linear(config.embedding_dim // 4, config.embedding_dim // 4)
        )

        # Adaptation type embedding
        self.adaptation_type_embedding = nn.Embedding(8, config.embedding_dim // 4)
        # Types: faithful, loose, reimagining, in_name_only, compressed, expanded, combined_sources, prequel/sequel

        # Combine
        self.quality_fusion = nn.Sequential(
            nn.Linear(config.embedding_dim // 4 * 3, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.hidden_dim)
        )

    def forward(self, faithfulness: torch.Tensor, reception: torch.Tensor,
                adaptation_types: torch.Tensor) -> torch.Tensor:
        """Analyze adaptation quality"""
        # Reshape scalar inputs to [batch, 1]
        if faithfulness.dim() == 1:
            faithfulness = faithfulness.unsqueeze(-1)
        if reception.dim() == 1:
            reception = reception.unsqueeze(-1)

        faith_emb = self.faithfulness_encoder(faithfulness)
        recep_emb = self.reception_encoder(reception)
        type_emb = self.adaptation_type_embedding(adaptation_types)

        combined = torch.cat([faith_emb, recep_emb, type_emb], dim=-1)
        return self.quality_fusion(combined)


class AdaptationSourceModel(nn.Module):
    """
    Complete Adaptation Source Model for source material-aware recommendations.
    """

    def __init__(self, config: Optional[AdaptationConfig] = None):
        super().__init__()
        self.config = config or AdaptationConfig()

        # Core embeddings
        self.movie_embedding = nn.Embedding(self.config.num_movies, self.config.embedding_dim)
        self.user_embedding = nn.Embedding(self.config.num_users, self.config.hidden_dim)

        # Source encoder
        self.source_encoder = SourceMaterialEncoder(self.config)

        # Quality analyzer
        self.quality_analyzer = AdaptationQualityAnalyzer(self.config)

        # Movie projection
        self.movie_proj = nn.Linear(self.config.embedding_dim, self.config.hidden_dim)

        # User source preference
        self.user_source_preference = nn.Sequential(
            nn.Linear(self.config.hidden_dim * 2, self.config.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim)
        )

        # Source type affinity
        self.source_type_affinity_head = nn.Sequential(
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim // 2),
            nn.GELU(),
            nn.Linear(self.config.hidden_dim // 2, self.config.num_source_types)
        )

        # Faithfulness preference (does user prefer faithful or loose adaptations?)
        self.faithfulness_preference_head = nn.Sequential(
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim // 2),
            nn.GELU(),
            nn.Linear(self.config.hidden_dim // 2, 1),
            nn.Sigmoid()  # 1 = prefers faithful, 0 = prefers loose
        )

        # Source familiarity impact (how much does knowing source affect enjoyment?)
        self.familiarity_impact_head = nn.Sequential(
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

        # Source recommendation head
        self.source_head = nn.Linear(self.config.hidden_dim, self.config.num_sources)

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
                source_type_ids: torch.Tensor, source_ids: torch.Tensor,
                characteristics: torch.Tensor, genre_features: torch.Tensor,
                faithfulness: torch.Tensor, reception: torch.Tensor,
                adaptation_types: torch.Tensor,
                user_read_source: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass for adaptation-aware recommendations.
        """
        batch_size = user_ids.size(0)

        # Get user embedding
        user_emb = self.user_embedding(user_ids)

        # Get movie embedding
        movie_emb = self.movie_embedding(movie_ids)
        movie_repr = self.movie_proj(movie_emb)

        # Encode source
        source_repr = self.source_encoder(source_type_ids, source_ids,
                                         characteristics, genre_features)

        # Analyze adaptation quality
        quality_repr = self.quality_analyzer(faithfulness, reception, adaptation_types)

        # Combined source representation
        combined_source = source_repr + quality_repr

        # User-source preference
        user_source = torch.cat([user_emb, combined_source], dim=-1)
        source_pref = self.user_source_preference(user_source)

        # Source type affinity
        source_type_affinity = self.source_type_affinity_head(user_emb)

        # Faithfulness preference
        faithfulness_pref = self.faithfulness_preference_head(user_emb)

        # Familiarity impact
        familiarity_impact = self.familiarity_impact_head(user_source)

        # Rating prediction
        rating_input = torch.cat([user_emb, movie_repr, combined_source], dim=-1)
        rating_pred = self.rating_head(rating_input)

        # Adjust rating if user knows source
        if user_read_source is not None:
            rating_adjustment = familiarity_impact * user_read_source.float().unsqueeze(-1)
            rating_pred = rating_pred + rating_adjustment * 0.5

        # Movie recommendations
        movie_logits = self.movie_head(source_pref)

        # Source recommendations (recommend original sources)
        source_logits = self.source_head(user_emb)

        return {
            'source_preference': source_pref,
            'source_repr': combined_source,
            'source_type_affinity': source_type_affinity,
            'faithfulness_preference': faithfulness_pref,
            'familiarity_impact': familiarity_impact,
            'rating_pred': rating_pred,
            'movie_logits': movie_logits,
            'source_logits': source_logits
        }

    def recommend_adaptations_of(self, source_type: int, top_k: int = 10,
                                user_ids: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Recommend best movie adaptations of a source type"""
        with torch.no_grad():
            if user_ids is not None:
                user_emb = self.user_embedding(user_ids)
            else:
                user_emb = torch.zeros(1, self.config.hidden_dim)

            source_type_emb = self.source_encoder.source_type_embedding.weight[source_type]

            user_source = torch.cat([user_emb, source_type_emb.expand(user_emb.size(0), -1)], dim=-1)
            pref = self.user_source_preference(user_source)

            movie_logits = self.movie_head(pref)
            top_scores, top_movies = torch.topk(F.softmax(movie_logits, dim=-1), top_k, dim=-1)

        return top_movies, top_scores

    def recommend_source_materials(self, user_ids: torch.Tensor,
                                  top_k: int = 10) -> Tuple[torch.Tensor, torch.Tensor]:
        """Recommend source materials user might enjoy"""
        with torch.no_grad():
            user_emb = self.user_embedding(user_ids)
            source_logits = self.source_head(user_emb)
            top_scores, top_sources = torch.topk(F.softmax(source_logits, dim=-1), top_k, dim=-1)

        return top_sources, top_scores


class AdaptationSourceTrainer:
    """Trainer for Adaptation Source Model"""

    def __init__(self, model: AdaptationSourceModel, lr: float = 1e-4,
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
        self.type_loss = nn.CrossEntropyLoss()

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Single training step"""
        self.model.train()

        user_ids = batch['user_ids'].to(self.device)
        movie_ids = batch['movie_ids'].to(self.device)
        source_type_ids = batch['source_type_ids'].to(self.device)
        source_ids = batch['source_ids'].to(self.device)
        characteristics = batch['characteristics'].to(self.device)
        genre_features = batch['genre_features'].to(self.device)
        faithfulness = batch['faithfulness'].to(self.device)
        reception = batch['reception'].to(self.device)
        adaptation_types = batch['adaptation_types'].to(self.device)
        target_ratings = batch['ratings'].to(self.device)

        outputs = self.model(
            user_ids, movie_ids, source_type_ids, source_ids,
            characteristics, genre_features, faithfulness, reception, adaptation_types
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
                source_type_ids = batch['source_type_ids'].to(self.device)
                source_ids = batch['source_ids'].to(self.device)
                characteristics = batch['characteristics'].to(self.device)
                genre_features = batch['genre_features'].to(self.device)
                faithfulness = batch['faithfulness'].to(self.device)
                reception = batch['reception'].to(self.device)
                adaptation_types = batch['adaptation_types'].to(self.device)
                target_ratings = batch['ratings'].to(self.device)

                outputs = self.model(
                    user_ids, movie_ids, source_type_ids, source_ids,
                    characteristics, genre_features, faithfulness, reception, adaptation_types
                )

                total_metrics['rating_mse'] += F.mse_loss(
                    outputs['rating_pred'].squeeze(), target_ratings
                ).item()

                num_batches += 1

        return {k: v / num_batches for k, v in total_metrics.items()}
