"""
Binge Prediction Model (BPM-TV)
Predicts and optimizes for binge-watching behavior patterns.
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
class BingePredictionConfig:
    """Configuration for Binge Prediction Model"""
    embedding_dim: int = 256
    hidden_dim: int = 512
    num_heads: int = 8
    num_layers: int = 4
    num_shows: int = 50000
    num_users: int = 50000
    max_session_length: int = 20
    dropout: float = 0.1


class BingePatternEncoder(nn.Module):
    """Encodes binge-watching patterns"""

    def __init__(self, config: BingePredictionConfig):
        super().__init__()

        # Session duration encoding
        self.duration_encoder = nn.Sequential(
            nn.Linear(1, config.embedding_dim // 4),
            nn.GELU(),
            nn.Linear(config.embedding_dim // 4, config.embedding_dim // 4)
        )

        # Time of day encoding
        self.time_embedding = nn.Embedding(24, config.embedding_dim // 4)

        # Day of week encoding
        self.day_embedding = nn.Embedding(7, config.embedding_dim // 4)

        # Session type embedding
        self.session_type_embedding = nn.Embedding(5, config.embedding_dim // 4)
        # Types: light (1-2 eps), moderate (3-5), heavy (6-10), marathon (10+), mixed

        # Combine
        self.pattern_fusion = nn.Sequential(
            nn.Linear(config.embedding_dim, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.hidden_dim)
        )

    def forward(self, session_duration: torch.Tensor, time_of_day: torch.Tensor,
                day_of_week: torch.Tensor, session_types: torch.Tensor) -> torch.Tensor:
        """Encode binge pattern"""
        duration_emb = self.duration_encoder(session_duration.float().unsqueeze(-1) / 10.0)
        time_emb = self.time_embedding(time_of_day)
        day_emb = self.day_embedding(day_of_week)
        session_emb = self.session_type_embedding(session_types)

        combined = torch.cat([duration_emb, time_emb, day_emb, session_emb], dim=-1)
        return self.pattern_fusion(combined)


class BingeabilityAnalyzer(nn.Module):
    """Analyzes show bingeability factors"""

    def __init__(self, config: BingePredictionConfig):
        super().__init__()

        # Cliffhanger intensity encoder
        self.cliffhanger_encoder = nn.Sequential(
            nn.Linear(1, config.embedding_dim // 4),
            nn.GELU(),
            nn.Linear(config.embedding_dim // 4, config.embedding_dim // 4)
        )

        # Episode length variance encoder
        self.length_encoder = nn.Sequential(
            nn.Linear(2, config.embedding_dim // 4),  # avg_length, variance
            nn.GELU(),
            nn.Linear(config.embedding_dim // 4, config.embedding_dim // 4)
        )

        # Narrative structure embedding
        self.structure_embedding = nn.Embedding(6, config.embedding_dim // 4)
        # Structures: serialized, procedural, anthology, limited, soap, hybrid

        # Release pattern embedding
        self.release_embedding = nn.Embedding(4, config.embedding_dim // 4)
        # Patterns: all_at_once, weekly, split_season, daily

        # Combine
        self.bingeability_fusion = nn.Sequential(
            nn.Linear(config.embedding_dim, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.hidden_dim)
        )

    def forward(self, cliffhanger_scores: torch.Tensor, length_features: torch.Tensor,
                structure_types: torch.Tensor, release_patterns: torch.Tensor) -> torch.Tensor:
        """Analyze bingeability"""
        cliff_emb = self.cliffhanger_encoder(cliffhanger_scores.unsqueeze(-1))
        length_emb = self.length_encoder(length_features)
        struct_emb = self.structure_embedding(structure_types)
        release_emb = self.release_embedding(release_patterns)

        combined = torch.cat([cliff_emb, length_emb, struct_emb, release_emb], dim=-1)
        return self.bingeability_fusion(combined)


class BingePredictionModel(nn.Module):
    """
    Complete Binge Prediction Model for optimizing binge recommendations.
    """

    def __init__(self, config: Optional[BingePredictionConfig] = None):
        super().__init__()
        self.config = config or BingePredictionConfig()

        # Core embeddings
        self.show_embedding = nn.Embedding(self.config.num_shows, self.config.embedding_dim)
        self.user_embedding = nn.Embedding(self.config.num_users, self.config.hidden_dim)

        # Pattern encoder
        self.pattern_encoder = BingePatternEncoder(self.config)

        # Bingeability analyzer
        self.bingeability_analyzer = BingeabilityAnalyzer(self.config)

        # Show projection
        self.show_proj = nn.Linear(self.config.embedding_dim, self.config.hidden_dim)

        # User binge preference
        self.user_binge_preference = nn.Sequential(
            nn.Linear(self.config.hidden_dim * 2, self.config.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim)
        )

        # Binge probability predictor
        self.binge_prob_head = nn.Sequential(
            nn.Linear(self.config.hidden_dim * 2, self.config.hidden_dim),
            nn.GELU(),
            nn.Linear(self.config.hidden_dim, 1),
            nn.Sigmoid()
        )

        # Session length predictor
        self.session_length_head = nn.Sequential(
            nn.Linear(self.config.hidden_dim * 2, self.config.hidden_dim),
            nn.GELU(),
            nn.Linear(self.config.hidden_dim, self.config.max_session_length)
        )

        # Optimal stopping point predictor
        self.stopping_point_head = nn.Sequential(
            nn.Linear(self.config.hidden_dim * 2, self.config.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.hidden_dim, self.config.max_session_length)
        )

        # Bingeability score predictor
        self.bingeability_head = nn.Sequential(
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

        # Show recommendation head
        self.show_head = nn.Linear(self.config.hidden_dim, self.config.num_shows)

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0, std=0.02)

    def forward(self, user_ids: torch.Tensor, show_ids: torch.Tensor,
                session_duration: torch.Tensor, time_of_day: torch.Tensor,
                day_of_week: torch.Tensor, session_types: torch.Tensor,
                cliffhanger_scores: torch.Tensor, length_features: torch.Tensor,
                structure_types: torch.Tensor, release_patterns: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass"""
        batch_size = user_ids.size(0)

        # Get embeddings
        user_emb = self.user_embedding(user_ids)
        show_emb = self.show_embedding(show_ids)
        show_repr = self.show_proj(show_emb)

        # Encode patterns
        pattern_repr = self.pattern_encoder(session_duration, time_of_day,
                                           day_of_week, session_types)

        # Analyze bingeability
        binge_repr = self.bingeability_analyzer(cliffhanger_scores, length_features,
                                               structure_types, release_patterns)

        # User-binge preference
        user_pattern = torch.cat([user_emb, pattern_repr], dim=-1)
        binge_pref = self.user_binge_preference(user_pattern)

        # Binge probability
        user_show = torch.cat([user_emb, binge_repr], dim=-1)
        binge_prob = self.binge_prob_head(user_show)

        # Session length prediction
        session_logits = self.session_length_head(user_show)

        # Stopping point prediction
        stopping_logits = self.stopping_point_head(user_show)

        # Bingeability score
        bingeability = self.bingeability_head(binge_repr)

        # Rating prediction
        rating_input = torch.cat([user_emb, show_repr, binge_pref], dim=-1)
        rating_pred = self.rating_head(rating_input)

        # Show recommendations
        show_logits = self.show_head(binge_pref)

        return {
            'binge_preference': binge_pref,
            'binge_prob': binge_prob,
            'session_length_logits': session_logits,
            'stopping_point_logits': stopping_logits,
            'bingeability': bingeability,
            'rating_pred': rating_pred,
            'show_logits': show_logits
        }

    def recommend_for_binge(self, user_ids: torch.Tensor, available_hours: float,
                           top_k: int = 10) -> Tuple[torch.Tensor, torch.Tensor]:
        """Recommend shows optimized for binge sessions"""
        with torch.no_grad():
            user_emb = self.user_embedding(user_ids)
            show_logits = self.show_head(user_emb)
            top_scores, top_shows = torch.topk(F.softmax(show_logits, dim=-1), top_k, dim=-1)
        return top_shows, top_scores


class BingePredictionTrainer:
    """Trainer for Binge Prediction Model"""

    def __init__(self, model: BingePredictionModel, lr: float = 1e-4,
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
        self.binge_loss = nn.BCELoss()
        self.session_loss = nn.CrossEntropyLoss()

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Single training step"""
        self.model.train()

        user_ids = batch['user_ids'].to(self.device)
        show_ids = batch['show_ids'].to(self.device)
        session_duration = batch['session_duration'].to(self.device)
        time_of_day = batch['time_of_day'].to(self.device)
        day_of_week = batch['day_of_week'].to(self.device)
        session_types = batch['session_types'].to(self.device)
        cliffhanger_scores = batch['cliffhanger_scores'].to(self.device)
        length_features = batch['length_features'].to(self.device)
        structure_types = batch['structure_types'].to(self.device)
        release_patterns = batch['release_patterns'].to(self.device)
        target_ratings = batch['ratings'].to(self.device)

        outputs = self.model(
            user_ids, show_ids, session_duration, time_of_day,
            day_of_week, session_types, cliffhanger_scores, length_features,
            structure_types, release_patterns
        )

        rating_loss = self.rating_loss(outputs['rating_pred'].squeeze(), target_ratings)
        total_loss = rating_loss

        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()

        return {'total_loss': total_loss.item(), 'rating_loss': rating_loss.item()}

    def evaluate(self, dataloader) -> Dict[str, float]:
        """Evaluate model"""
        self.model.eval()
        total_metrics = {'rating_mse': 0}
        num_batches = 0

        with torch.no_grad():
            for batch in dataloader:
                user_ids = batch['user_ids'].to(self.device)
                show_ids = batch['show_ids'].to(self.device)
                target_ratings = batch['ratings'].to(self.device)

                # Create dummy tensors for missing fields
                batch_size = user_ids.size(0)
                outputs = self.model(
                    user_ids, show_ids,
                    torch.zeros(batch_size, device=self.device),
                    torch.zeros(batch_size, dtype=torch.long, device=self.device),
                    torch.zeros(batch_size, dtype=torch.long, device=self.device),
                    torch.zeros(batch_size, dtype=torch.long, device=self.device),
                    torch.zeros(batch_size, device=self.device),
                    torch.zeros(batch_size, 2, device=self.device),
                    torch.zeros(batch_size, dtype=torch.long, device=self.device),
                    torch.zeros(batch_size, dtype=torch.long, device=self.device)
                )

                total_metrics['rating_mse'] += F.mse_loss(
                    outputs['rating_pred'].squeeze(), target_ratings
                ).item()
                num_batches += 1

        return {k: v / num_batches for k, v in total_metrics.items()}
