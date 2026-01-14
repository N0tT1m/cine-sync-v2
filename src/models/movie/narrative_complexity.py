"""
Narrative Complexity Model (NCM)
Models user preferences for different storytelling structures and complexity levels.
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
class NarrativeConfig:
    """Configuration for Narrative Complexity Model"""
    embedding_dim: int = 256
    hidden_dim: int = 512
    num_heads: int = 8
    num_layers: int = 4
    num_movies: int = 100000
    num_users: int = 50000
    num_structures: int = 20  # Linear, non-linear, multi-timeline, etc.
    num_themes: int = 100
    dropout: float = 0.1


class NarrativeStructureEncoder(nn.Module):
    """Encodes narrative structure and complexity"""

    def __init__(self, config: NarrativeConfig):
        super().__init__()

        # Structure type embedding
        self.structure_embedding = nn.Embedding(config.num_structures, config.embedding_dim)
        # Types: linear, flashback, non_linear, multiple_timeline, frame_story,
        #        parallel_plots, circular, reverse_chronology, stream_of_consciousness,
        #        episodic, anthology, mockumentary, unreliable_narrator, meta_narrative, etc.

        # Complexity features
        self.complexity_encoder = nn.Sequential(
            nn.Linear(8, config.embedding_dim // 4),  # num_plotlines, num_characters, time_jumps, twist_count, ambiguity_level, runtime, pacing_variability, exposition_density
            nn.GELU(),
            nn.Linear(config.embedding_dim // 4, config.embedding_dim // 4)
        )

        # Theme complexity
        self.theme_encoder = nn.Sequential(
            nn.Linear(config.num_themes, config.embedding_dim // 4),
            nn.GELU(),
            nn.Linear(config.embedding_dim // 4, config.embedding_dim // 4)
        )

        # Dialogue complexity
        self.dialogue_encoder = nn.Sequential(
            nn.Linear(4, config.embedding_dim // 8),  # dialogue_density, vocabulary_complexity, subtext_level, monologue_frequency
            nn.GELU(),
            nn.Linear(config.embedding_dim // 8, config.embedding_dim // 8)
        )

        # Combine
        self.structure_fusion = nn.Sequential(
            nn.Linear(config.embedding_dim + config.embedding_dim // 4 * 2 + config.embedding_dim // 8,
                     config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.hidden_dim)
        )

    def forward(self, structure_ids: torch.Tensor, complexity_features: torch.Tensor,
                theme_vector: torch.Tensor, dialogue_features: torch.Tensor) -> torch.Tensor:
        """Encode narrative structure"""
        struct_emb = self.structure_embedding(structure_ids)
        complex_emb = self.complexity_encoder(complexity_features)
        theme_emb = self.theme_encoder(theme_vector)
        dialogue_emb = self.dialogue_encoder(dialogue_features)

        combined = torch.cat([struct_emb, complex_emb, theme_emb, dialogue_emb], dim=-1)
        return self.structure_fusion(combined)


class CognitiveLoadPredictor(nn.Module):
    """Predicts cognitive load required for understanding"""

    def __init__(self, config: NarrativeConfig):
        super().__init__()

        # Load prediction network
        self.load_predictor = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim // 2, config.hidden_dim // 4),
            nn.GELU(),
            nn.Linear(config.hidden_dim // 4, 5)  # 5 cognitive load levels
        )

        # Attention requirement predictor
        self.attention_predictor = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.GELU(),
            nn.Linear(config.hidden_dim // 2, 1),
            nn.Sigmoid()  # 0 = casual viewing, 1 = requires full attention
        )

        # Rewatch value predictor
        self.rewatch_predictor = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.GELU(),
            nn.Linear(config.hidden_dim // 2, 1),
            nn.Sigmoid()  # Higher = more reward for rewatching
        )

    def forward(self, narrative_repr: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Predict cognitive requirements"""
        load = self.load_predictor(narrative_repr)
        attention = self.attention_predictor(narrative_repr)
        rewatch = self.rewatch_predictor(narrative_repr)
        return load, attention, rewatch


class NarrativeComplexityModel(nn.Module):
    """
    Complete Narrative Complexity Model for complexity-aware recommendations.
    """

    def __init__(self, config: Optional[NarrativeConfig] = None):
        super().__init__()
        self.config = config or NarrativeConfig()

        # Core embeddings
        self.movie_embedding = nn.Embedding(self.config.num_movies, self.config.embedding_dim)
        self.user_embedding = nn.Embedding(self.config.num_users, self.config.hidden_dim)

        # Narrative encoder
        self.narrative_encoder = NarrativeStructureEncoder(self.config)

        # Cognitive load predictor
        self.cognitive_predictor = CognitiveLoadPredictor(self.config)

        # Movie projection
        self.movie_proj = nn.Linear(self.config.embedding_dim, self.config.hidden_dim)

        # User complexity preference
        self.user_complexity_preference = nn.Sequential(
            nn.Linear(self.config.hidden_dim * 2, self.config.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim)
        )

        # Structure preference head
        self.structure_preference_head = nn.Sequential(
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim // 2),
            nn.GELU(),
            nn.Linear(self.config.hidden_dim // 2, self.config.num_structures)
        )

        # Optimal complexity predictor
        self.optimal_complexity_head = nn.Sequential(
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim // 2),
            nn.GELU(),
            nn.Linear(self.config.hidden_dim // 2, 5)  # 5 complexity levels
        )

        # Context-aware complexity (mood affects complexity tolerance)
        self.context_complexity_head = nn.Sequential(
            nn.Linear(self.config.hidden_dim + 4, self.config.hidden_dim // 2),  # +4 for mood/context
            nn.GELU(),
            nn.Linear(self.config.hidden_dim // 2, 5)
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

        # Complexity match predictor
        self.complexity_match_head = nn.Sequential(
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
                structure_ids: torch.Tensor, complexity_features: torch.Tensor,
                theme_vector: torch.Tensor, dialogue_features: torch.Tensor,
                viewing_context: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass for complexity-aware recommendations.
        """
        batch_size = user_ids.size(0)

        # Get user embedding
        user_emb = self.user_embedding(user_ids)

        # Get movie embedding
        movie_emb = self.movie_embedding(movie_ids)
        movie_repr = self.movie_proj(movie_emb)

        # Encode narrative
        narrative_repr = self.narrative_encoder(structure_ids, complexity_features,
                                               theme_vector, dialogue_features)

        # Predict cognitive load
        cognitive_load, attention_req, rewatch_value = self.cognitive_predictor(narrative_repr)

        # User-narrative preference
        user_narrative = torch.cat([user_emb, narrative_repr], dim=-1)
        complexity_pref = self.user_complexity_preference(user_narrative)

        # Structure preference
        structure_pref = self.structure_preference_head(user_emb)

        # Optimal complexity
        optimal_complexity = self.optimal_complexity_head(user_emb)

        # Context-aware complexity
        if viewing_context is not None:
            context_input = torch.cat([user_emb, viewing_context], dim=-1)
            context_complexity = self.context_complexity_head(context_input)
        else:
            context_complexity = optimal_complexity

        # Complexity match
        complexity_match = self.complexity_match_head(user_narrative)

        # Rating prediction
        rating_input = torch.cat([user_emb, movie_repr, narrative_repr], dim=-1)
        rating_pred = self.rating_head(rating_input)

        # Movie recommendations
        movie_logits = self.movie_head(complexity_pref)

        return {
            'complexity_preference': complexity_pref,
            'narrative_repr': narrative_repr,
            'cognitive_load': cognitive_load,
            'attention_required': attention_req,
            'rewatch_value': rewatch_value,
            'structure_preference': structure_pref,
            'optimal_complexity': optimal_complexity,
            'context_complexity': context_complexity,
            'complexity_match': complexity_match,
            'rating_pred': rating_pred,
            'movie_logits': movie_logits
        }

    def recommend_by_complexity(self, user_ids: torch.Tensor, target_complexity: int,
                               top_k: int = 10) -> Tuple[torch.Tensor, torch.Tensor]:
        """Recommend movies of a specific complexity level"""
        with torch.no_grad():
            user_emb = self.user_embedding(user_ids)

            # Create target complexity representation
            complexity_target = torch.zeros(user_ids.size(0), 5, device=user_ids.device)
            complexity_target[:, target_complexity] = 1.0

            movie_logits = self.movie_head(user_emb)
            top_scores, top_movies = torch.topk(F.softmax(movie_logits, dim=-1), top_k, dim=-1)

        return top_movies, top_scores

    def recommend_for_mood(self, user_ids: torch.Tensor, mood_context: torch.Tensor,
                          top_k: int = 10) -> Tuple[torch.Tensor, torch.Tensor]:
        """Recommend movies based on current mood/context"""
        with torch.no_grad():
            user_emb = self.user_embedding(user_ids)
            context_input = torch.cat([user_emb, mood_context], dim=-1)
            context_complexity = self.context_complexity_head(context_input)

            # Use context complexity to weight recommendations
            movie_logits = self.movie_head(user_emb)

            top_scores, top_movies = torch.topk(F.softmax(movie_logits, dim=-1), top_k, dim=-1)

        return top_movies, top_scores

    def get_complexity_profile(self, user_ids: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Get user's complexity preference profile"""
        with torch.no_grad():
            user_emb = self.user_embedding(user_ids)
            structure_pref = self.structure_preference_head(user_emb)
            optimal_complexity = self.optimal_complexity_head(user_emb)

        return {
            'structure_preference': F.softmax(structure_pref, dim=-1),
            'optimal_complexity': F.softmax(optimal_complexity, dim=-1)
        }


class NarrativeComplexityTrainer:
    """Trainer for Narrative Complexity Model"""

    def __init__(self, model: NarrativeComplexityModel, lr: float = 1e-4,
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
        self.complexity_loss = nn.CrossEntropyLoss()

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Single training step"""
        self.model.train()

        user_ids = batch['user_ids'].to(self.device)
        movie_ids = batch['movie_ids'].to(self.device)
        structure_ids = batch['structure_ids'].to(self.device)
        complexity_features = batch['complexity_features'].to(self.device)
        theme_vector = batch['theme_vector'].to(self.device)
        dialogue_features = batch['dialogue_features'].to(self.device)
        target_ratings = batch['ratings'].to(self.device)

        outputs = self.model(
            user_ids, movie_ids, structure_ids, complexity_features,
            theme_vector, dialogue_features
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
                structure_ids = batch['structure_ids'].to(self.device)
                complexity_features = batch['complexity_features'].to(self.device)
                theme_vector = batch['theme_vector'].to(self.device)
                dialogue_features = batch['dialogue_features'].to(self.device)
                target_ratings = batch['ratings'].to(self.device)

                outputs = self.model(
                    user_ids, movie_ids, structure_ids, complexity_features,
                    theme_vector, dialogue_features
                )

                total_metrics['rating_mse'] += F.mse_loss(
                    outputs['rating_pred'].squeeze(), target_ratings
                ).item()

                num_batches += 1

        return {k: v / num_batches for k, v in total_metrics.items()}
