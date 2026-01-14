"""
Series Completion Model (SCM-TV)
Predicts series completion likelihood and models drop-off patterns.
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
class SeriesCompletionConfig:
    """Configuration for Series Completion Model"""
    embedding_dim: int = 256
    hidden_dim: int = 512
    num_heads: int = 8
    num_layers: int = 4
    num_shows: int = 50000
    num_users: int = 50000
    max_episodes: int = 500
    dropout: float = 0.1


class CompletionPatternEncoder(nn.Module):
    """Encodes user completion patterns"""

    def __init__(self, config: SeriesCompletionConfig):
        super().__init__()

        # Progress encoding
        self.progress_encoder = nn.Sequential(
            nn.Linear(4, config.embedding_dim // 2),  # episodes_watched, total_episodes, percentage, seasons_watched
            nn.GELU(),
            nn.Linear(config.embedding_dim // 2, config.embedding_dim // 2)
        )

        # Engagement pattern embedding
        self.engagement_embedding = nn.Embedding(5, config.embedding_dim // 4)
        # Patterns: steady, accelerating, decelerating, sporadic, abandoned

        # Gap analysis encoding
        self.gap_encoder = nn.Sequential(
            nn.Linear(3, config.embedding_dim // 4),  # avg_gap, max_gap, recent_gap
            nn.GELU(),
            nn.Linear(config.embedding_dim // 4, config.embedding_dim // 4)
        )

        # Combine
        self.pattern_fusion = nn.Sequential(
            nn.Linear(config.embedding_dim, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.hidden_dim)
        )

    def forward(self, progress_features: torch.Tensor, engagement_types: torch.Tensor,
                gap_features: torch.Tensor) -> torch.Tensor:
        """Encode completion pattern"""
        progress_emb = self.progress_encoder(progress_features)
        engage_emb = self.engagement_embedding(engagement_types)
        gap_emb = self.gap_encoder(gap_features)

        combined = torch.cat([progress_emb, engage_emb, gap_emb], dim=-1)
        return self.pattern_fusion(combined)


class DropOffPredictor(nn.Module):
    """Predicts where users might drop off"""

    def __init__(self, config: SeriesCompletionConfig):
        super().__init__()

        # LSTM for sequential prediction
        self.lstm = nn.LSTM(
            config.hidden_dim, config.hidden_dim // 2,
            num_layers=2, batch_first=True, bidirectional=True,
            dropout=config.dropout
        )

        # Episode-level drop-off predictor
        self.episode_dropoff = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.GELU(),
            nn.Linear(config.hidden_dim // 2, 1),
            nn.Sigmoid()
        )

        # Season-level drop-off predictor
        self.season_dropoff = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.GELU(),
            nn.Linear(config.hidden_dim // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, sequence_repr: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Predict drop-off points"""
        lstm_out, _ = self.lstm(sequence_repr)
        episode_probs = self.episode_dropoff(lstm_out)
        season_probs = self.season_dropoff(lstm_out.mean(dim=1, keepdim=True))
        return lstm_out, episode_probs, season_probs


class SeriesCompletionModel(nn.Module):
    """
    Complete Series Completion Model for predicting viewing completion.
    """

    def __init__(self, config: Optional[SeriesCompletionConfig] = None):
        super().__init__()
        self.config = config or SeriesCompletionConfig()

        # Core embeddings
        self.show_embedding = nn.Embedding(self.config.num_shows, self.config.embedding_dim)
        self.user_embedding = nn.Embedding(self.config.num_users, self.config.hidden_dim)

        # Pattern encoder
        self.pattern_encoder = CompletionPatternEncoder(self.config)

        # Drop-off predictor
        self.dropoff_predictor = DropOffPredictor(self.config)

        # Show projection
        self.show_proj = nn.Linear(self.config.embedding_dim, self.config.hidden_dim)

        # User completion preference
        self.user_completion_preference = nn.Sequential(
            nn.Linear(self.config.hidden_dim * 2, self.config.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim)
        )

        # Completion probability predictor
        self.completion_prob_head = nn.Sequential(
            nn.Linear(self.config.hidden_dim * 2, self.config.hidden_dim),
            nn.GELU(),
            nn.Linear(self.config.hidden_dim, 1),
            nn.Sigmoid()
        )

        # Time to completion predictor
        self.time_head = nn.Sequential(
            nn.Linear(self.config.hidden_dim * 2, self.config.hidden_dim),
            nn.GELU(),
            nn.Linear(self.config.hidden_dim, 10)  # Buckets: days to completion
        )

        # Re-engagement predictor (will user return after gap?)
        self.reengagement_head = nn.Sequential(
            nn.Linear(self.config.hidden_dim * 2, self.config.hidden_dim),
            nn.GELU(),
            nn.Linear(self.config.hidden_dim, 1),
            nn.Sigmoid()
        )

        # Satisfaction predictor (will user be satisfied if they complete?)
        self.satisfaction_head = nn.Sequential(
            nn.Linear(self.config.hidden_dim * 2, self.config.hidden_dim),
            nn.GELU(),
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
                progress_features: torch.Tensor, engagement_types: torch.Tensor,
                gap_features: torch.Tensor,
                episode_history: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Forward pass"""
        batch_size = user_ids.size(0)

        # Get embeddings
        user_emb = self.user_embedding(user_ids)
        show_emb = self.show_embedding(show_ids)
        show_repr = self.show_proj(show_emb)

        # Encode pattern
        pattern_repr = self.pattern_encoder(progress_features, engagement_types, gap_features)

        # User-pattern preference
        user_pattern = torch.cat([user_emb, pattern_repr], dim=-1)
        completion_pref = self.user_completion_preference(user_pattern)

        # Completion probability
        user_show = torch.cat([user_emb, show_repr], dim=-1)
        completion_prob = self.completion_prob_head(user_show)

        # Time to completion
        time_logits = self.time_head(user_show)

        # Re-engagement prediction
        reengagement = self.reengagement_head(user_pattern)

        # Satisfaction prediction
        satisfaction = self.satisfaction_head(user_show)

        # Drop-off prediction if history provided
        episode_dropoff = None
        season_dropoff = None
        if episode_history is not None:
            _, episode_dropoff, season_dropoff = self.dropoff_predictor(episode_history)

        # Show recommendations
        show_logits = self.show_head(completion_pref)

        return {
            'completion_preference': completion_pref,
            'completion_prob': completion_prob,
            'time_logits': time_logits,
            'reengagement': reengagement,
            'satisfaction': satisfaction,
            'episode_dropoff': episode_dropoff,
            'season_dropoff': season_dropoff,
            'show_logits': show_logits
        }

    def recommend_completable(self, user_ids: torch.Tensor, min_completion_prob: float = 0.7,
                             top_k: int = 10) -> Tuple[torch.Tensor, torch.Tensor]:
        """Recommend shows user is likely to complete"""
        with torch.no_grad():
            user_emb = self.user_embedding(user_ids)
            show_logits = self.show_head(user_emb)
            top_scores, top_shows = torch.topk(F.softmax(show_logits, dim=-1), top_k, dim=-1)
        return top_shows, top_scores


class SeriesCompletionTrainer:
    """Trainer for Series Completion Model"""

    def __init__(self, model: SeriesCompletionModel, lr: float = 1e-4,
                 weight_decay: float = 0.01, device: str = 'cuda'):
        self.model = model.to(device)
        self.device = device

        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=10, T_mult=2
        )

        self.completion_loss = nn.BCELoss()
        self.time_loss = nn.CrossEntropyLoss()
        self.satisfaction_loss = nn.MSELoss()

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Single training step"""
        self.model.train()

        user_ids = batch['user_ids'].to(self.device)
        show_ids = batch['show_ids'].to(self.device)
        progress_features = batch['progress_features'].to(self.device)
        engagement_types = batch['engagement_types'].to(self.device)
        gap_features = batch['gap_features'].to(self.device)
        target_completion = batch['completed'].to(self.device).float()

        outputs = self.model(
            user_ids, show_ids, progress_features, engagement_types, gap_features
        )

        completion_loss = self.completion_loss(outputs['completion_prob'].squeeze(), target_completion)
        total_loss = completion_loss

        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()

        return {'total_loss': total_loss.item(), 'completion_loss': completion_loss.item()}

    def evaluate(self, dataloader) -> Dict[str, float]:
        """Evaluate model"""
        self.model.eval()
        total_metrics = {'completion_acc': 0}
        num_batches = 0

        with torch.no_grad():
            for batch in dataloader:
                user_ids = batch['user_ids'].to(self.device)
                show_ids = batch['show_ids'].to(self.device)
                progress_features = batch['progress_features'].to(self.device)
                engagement_types = batch['engagement_types'].to(self.device)
                gap_features = batch['gap_features'].to(self.device)
                target_completion = batch['completed'].to(self.device).float()

                outputs = self.model(
                    user_ids, show_ids, progress_features, engagement_types, gap_features
                )

                pred_completion = (outputs['completion_prob'].squeeze() > 0.5).float()
                total_metrics['completion_acc'] += (
                    pred_completion == target_completion
                ).float().mean().item()
                num_batches += 1

        return {k: v / num_batches for k, v in total_metrics.items()}
