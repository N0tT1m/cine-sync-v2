"""
Cinematic Universe Model (CUM)
Models interconnected movie universes like MCU, DC, Star Wars, etc.
Understands cross-movie references, character appearances, and timeline connections.
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
class UniverseConfig:
    """Configuration for Cinematic Universe Model"""
    embedding_dim: int = 256
    hidden_dim: int = 512
    num_heads: int = 8
    num_layers: int = 6
    num_universes: int = 500
    num_movies: int = 100000
    num_users: int = 50000
    num_characters: int = 20000
    num_story_arcs: int = 5000
    max_timeline_length: int = 100
    dropout: float = 0.1


class UniverseGraphAttention(nn.Module):
    """Graph attention for universe movie relationships"""

    def __init__(self, config: UniverseConfig):
        super().__init__()
        self.config = config

        # Connection type embeddings
        self.connection_types = nn.Embedding(10, config.embedding_dim)
        # Types: direct_sequel, prequel, spinoff, crossover, post_credits,
        #        shared_character, same_timeline, alternate_timeline, multiverse, cameo

        # Graph attention layers
        self.attention = nn.MultiheadAttention(
            config.embedding_dim, config.num_heads, dropout=config.dropout, batch_first=True
        )

        # Edge feature network
        self.edge_mlp = nn.Sequential(
            nn.Linear(config.embedding_dim * 2, config.embedding_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.embedding_dim, config.embedding_dim)
        )

    def forward(self, movie_features: torch.Tensor, adjacency: torch.Tensor,
                connection_types: torch.Tensor) -> torch.Tensor:
        """
        Args:
            movie_features: Movie node features [batch, num_movies, emb_dim]
            adjacency: Adjacency matrix [batch, num_movies, num_movies]
            connection_types: Connection type IDs [batch, num_movies, num_movies]
        """
        batch_size, num_nodes = movie_features.size(0), movie_features.size(1)

        # Get connection type embeddings
        conn_emb = self.connection_types(connection_types)  # [batch, n, n, emb]

        # Create attention mask from adjacency
        attention_mask = adjacency == 0

        # Apply attention with connection context
        attended, _ = self.attention(
            movie_features, movie_features, movie_features,
            key_padding_mask=None, attn_mask=attention_mask[0] if batch_size == 1 else None
        )

        # Aggregate edge features
        edge_features = self.edge_mlp(
            torch.cat([movie_features.unsqueeze(2).expand(-1, -1, num_nodes, -1),
                      conn_emb], dim=-1).mean(dim=2)
        )

        return attended + edge_features


class TimelineEncoder(nn.Module):
    """Encodes universe timeline with multiple time dimensions"""

    def __init__(self, config: UniverseConfig):
        super().__init__()

        # In-universe timeline encoding
        self.universe_time_encoding = nn.Parameter(
            torch.randn(config.max_timeline_length, config.embedding_dim)
        )

        # Release order encoding
        self.release_encoding = nn.Parameter(
            torch.randn(config.max_timeline_length, config.embedding_dim)
        )

        # Phase/saga encoding (e.g., MCU phases)
        self.phase_embedding = nn.Embedding(20, config.embedding_dim)

        # Timeline fusion (outputs hidden_dim to match movie_proj output)
        self.timeline_fusion = nn.Sequential(
            nn.Linear(config.embedding_dim * 3, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, config.hidden_dim)
        )

    def forward(self, universe_positions: torch.Tensor, release_positions: torch.Tensor,
                phases: torch.Tensor) -> torch.Tensor:
        """
        Encode timeline positions.

        Args:
            universe_positions: In-story chronological positions
            release_positions: Release order positions
            phases: Phase/saga IDs
        """
        universe_time = self.universe_time_encoding[universe_positions.clamp(0, 99)]
        release_time = self.release_encoding[release_positions.clamp(0, 99)]
        phase_emb = self.phase_embedding(phases)

        return self.timeline_fusion(torch.cat([universe_time, release_time, phase_emb], dim=-1))


class CharacterTracker(nn.Module):
    """Tracks character appearances and arcs across universe"""

    def __init__(self, config: UniverseConfig):
        super().__init__()

        self.character_embedding = nn.Embedding(config.num_characters, config.embedding_dim)

        # Character arc encoder
        self.arc_encoder = nn.LSTM(
            config.embedding_dim, config.hidden_dim // 2,
            num_layers=2, batch_first=True, bidirectional=True, dropout=config.dropout
        )

        # Character importance scorer
        self.importance_scorer = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.GELU(),
            nn.Linear(config.hidden_dim // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, character_ids: torch.Tensor,
                appearance_sequence: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Track characters across movies.

        Args:
            character_ids: Character IDs [batch, num_characters]
            appearance_sequence: Movie appearance sequence per character [batch, chars, seq_len]
        """
        char_emb = self.character_embedding(character_ids)

        # Encode character arcs
        batch_size, num_chars, seq_len = appearance_sequence.shape
        arc_input = char_emb.unsqueeze(2).expand(-1, -1, seq_len, -1)
        arc_input = arc_input.view(batch_size * num_chars, seq_len, -1)

        arc_encoded, _ = self.arc_encoder(arc_input)
        arc_encoded = arc_encoded.view(batch_size, num_chars, seq_len, -1)

        # Character importance
        importance = self.importance_scorer(arc_encoded.mean(dim=2))

        return arc_encoded, importance


class CinematicUniverseModel(nn.Module):
    """
    Complete Cinematic Universe Model for navigating interconnected movie universes.
    """

    def __init__(self, config: Optional[UniverseConfig] = None):
        super().__init__()
        self.config = config or UniverseConfig()

        # Core embeddings
        self.movie_embedding = nn.Embedding(self.config.num_movies, self.config.embedding_dim)
        self.universe_embedding = nn.Embedding(self.config.num_universes, self.config.embedding_dim)
        self.user_embedding = nn.Embedding(self.config.num_users, self.config.hidden_dim)
        self.story_arc_embedding = nn.Embedding(self.config.num_story_arcs, self.config.embedding_dim)

        # Timeline encoder
        self.timeline_encoder = TimelineEncoder(self.config)

        # Universe graph attention
        self.graph_attention = UniverseGraphAttention(self.config)

        # Character tracker
        self.character_tracker = CharacterTracker(self.config)

        # Movie encoder transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.config.hidden_dim,
            nhead=self.config.num_heads,
            dim_feedforward=self.config.hidden_dim * 4,
            dropout=self.config.dropout,
            activation='gelu',
            batch_first=True
        )
        self.movie_transformer = nn.TransformerEncoder(encoder_layer, num_layers=self.config.num_layers)

        # Feature projection
        self.movie_proj = nn.Linear(self.config.embedding_dim * 2, self.config.hidden_dim)

        # Viewing order predictor (chronological vs release vs recommended)
        self.order_predictor = nn.Sequential(
            nn.Linear(self.config.hidden_dim * 2, self.config.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.hidden_dim, 3)  # chronological, release, custom
        )

        # Next movie predictor
        self.next_movie_head = nn.Sequential(
            nn.Linear(self.config.hidden_dim * 2, self.config.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.hidden_dim, self.config.num_movies)
        )

        # Universe engagement predictor
        self.engagement_head = nn.Sequential(
            nn.Linear(self.config.hidden_dim * 2, self.config.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.hidden_dim, 1),
            nn.Sigmoid()
        )

        # Rating predictor
        self.rating_head = nn.Sequential(
            nn.Linear(self.config.hidden_dim * 3, self.config.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.hidden_dim, 1)
        )

        # Prerequisite predictor (which movies should be watched first)
        self.prerequisite_head = nn.Sequential(
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

    def encode_universe(self, movie_ids: torch.Tensor, universe_ids: torch.Tensor,
                       universe_positions: torch.Tensor, release_positions: torch.Tensor,
                       phases: torch.Tensor, adjacency: torch.Tensor,
                       connection_types: torch.Tensor) -> torch.Tensor:
        """Encode a cinematic universe's structure"""
        # Get base embeddings
        movie_emb = self.movie_embedding(movie_ids)
        universe_emb = self.universe_embedding(universe_ids)

        # Add timeline encoding
        timeline_emb = self.timeline_encoder(universe_positions, release_positions, phases)

        # Handle both per-batch and per-movie universe_ids
        # If universe_emb is 2D [batch, emb_dim], expand to match movie_emb [batch, num_movies, emb_dim]
        # If universe_emb is already 3D [batch, num_movies, emb_dim], use as-is
        if universe_emb.dim() == 2:
            universe_emb = universe_emb.unsqueeze(1).expand(-1, movie_emb.size(1), -1)

        # Combine features
        combined = torch.cat([movie_emb, universe_emb], dim=-1)
        combined = self.movie_proj(combined) + timeline_emb

        # Apply graph attention for universe structure
        universe_repr = self.graph_attention(combined, adjacency, connection_types)

        # Transform
        universe_repr = self.movie_transformer(universe_repr)

        return universe_repr

    def forward(self, user_ids: torch.Tensor, movie_ids: torch.Tensor,
                universe_ids: torch.Tensor, universe_positions: torch.Tensor,
                release_positions: torch.Tensor, phases: torch.Tensor,
                adjacency: torch.Tensor, connection_types: torch.Tensor,
                watched_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass for universe navigation recommendations.
        """
        batch_size = user_ids.size(0)

        # Get user embedding
        user_emb = self.user_embedding(user_ids)

        # Encode universe
        universe_repr = self.encode_universe(
            movie_ids, universe_ids, universe_positions, release_positions,
            phases, adjacency, connection_types
        )

        # Pool universe representation
        if watched_mask is not None:
            # Only consider watched movies
            mask_expanded = watched_mask.unsqueeze(-1).float()
            watched_repr = (universe_repr * mask_expanded).sum(1) / (mask_expanded.sum(1) + 1e-8)
            unwatched_repr = (universe_repr * (1 - mask_expanded)).sum(1) / ((1 - mask_expanded).sum(1) + 1e-8)
        else:
            watched_repr = universe_repr.mean(dim=1)
            unwatched_repr = watched_repr

        # User-universe interaction
        user_universe = torch.cat([user_emb, watched_repr], dim=-1)

        # Predict viewing order preference
        order_logits = self.order_predictor(user_universe)

        # Predict next movie
        next_movie_logits = self.next_movie_head(user_universe)

        # Predict engagement
        engagement = self.engagement_head(user_universe)

        # Predict ratings
        rating_input = torch.cat([user_emb, watched_repr, unwatched_repr], dim=-1)
        rating_pred = self.rating_head(rating_input)

        return {
            'universe_repr': universe_repr,
            'order_logits': order_logits,
            'next_movie_logits': next_movie_logits,
            'engagement': engagement,
            'rating_pred': rating_pred,
            'user_universe_repr': user_universe
        }

    def get_watch_order(self, universe_repr: torch.Tensor, user_emb: torch.Tensor,
                        watched_mask: torch.Tensor, order_type: str = 'recommended') -> torch.Tensor:
        """
        Get recommended watch order for unwatched movies.

        Args:
            order_type: 'chronological', 'release', or 'recommended'
        """
        with torch.no_grad():
            unwatched = ~watched_mask

            if order_type == 'recommended':
                # Use model to rank unwatched movies
                user_universe = torch.cat([user_emb, universe_repr.mean(dim=1)], dim=-1)
                scores = self.next_movie_head(user_universe)
                scores = scores.masked_fill(watched_mask, -float('inf'))
                order = torch.argsort(scores, dim=-1, descending=True)
            else:
                # Return chronological or release order (handled by data)
                order = torch.arange(universe_repr.size(1)).unsqueeze(0)

        return order

    def find_prerequisites(self, target_movie_id: int, universe_repr: torch.Tensor,
                          all_movie_ids: torch.Tensor) -> torch.Tensor:
        """Find which movies should be watched before a target movie"""
        with torch.no_grad():
            target_repr = universe_repr[:, target_movie_id, :]

            prerequisites = []
            for i in range(universe_repr.size(1)):
                if i != target_movie_id:
                    pair_repr = torch.cat([universe_repr[:, i, :], target_repr], dim=-1)
                    is_prereq = self.prerequisite_head(pair_repr)
                    prerequisites.append(is_prereq)

            prereq_scores = torch.cat(prerequisites, dim=-1)

        return prereq_scores


class CinematicUniverseTrainer:
    """Trainer for Cinematic Universe Model"""

    def __init__(self, model: CinematicUniverseModel, lr: float = 1e-4,
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
        self.order_loss = nn.CrossEntropyLoss()
        self.next_movie_loss = nn.CrossEntropyLoss()
        self.engagement_loss = nn.BCELoss()

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Single training step"""
        self.model.train()

        # Move to device
        user_ids = batch['user_ids'].to(self.device)
        movie_ids = batch['movie_ids'].to(self.device)
        universe_ids = batch['universe_ids'].to(self.device)
        universe_positions = batch['universe_positions'].to(self.device)
        release_positions = batch['release_positions'].to(self.device)
        phases = batch['phases'].to(self.device)
        adjacency = batch['adjacency'].to(self.device)
        connection_types = batch['connection_types'].to(self.device)
        watched_mask = batch.get('watched_mask')
        target_ratings = batch['ratings'].to(self.device)
        target_next = batch['next_movie'].to(self.device)
        target_order = batch['preferred_order'].to(self.device)
        target_engagement = batch['completed_universe'].to(self.device).float()

        if watched_mask is not None:
            watched_mask = watched_mask.to(self.device)

        # Forward pass
        outputs = self.model(
            user_ids, movie_ids, universe_ids, universe_positions,
            release_positions, phases, adjacency, connection_types, watched_mask
        )

        # Compute losses
        rating_loss = self.rating_loss(outputs['rating_pred'].squeeze(), target_ratings)
        order_loss = self.order_loss(outputs['order_logits'], target_order)
        next_loss = self.next_movie_loss(outputs['next_movie_logits'], target_next)
        engagement_loss = self.engagement_loss(outputs['engagement'].squeeze(), target_engagement)

        total_loss = rating_loss + order_loss + next_loss + engagement_loss * 0.5

        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()

        return {
            'total_loss': total_loss.item(),
            'rating_loss': rating_loss.item(),
            'order_loss': order_loss.item(),
            'next_movie_loss': next_loss.item(),
            'engagement_loss': engagement_loss.item()
        }

    def evaluate(self, dataloader) -> Dict[str, float]:
        """Evaluate model"""
        self.model.eval()
        total_metrics = {'rating_mse': 0, 'order_acc': 0, 'next_acc': 0, 'engagement_acc': 0}
        num_batches = 0

        with torch.no_grad():
            for batch in dataloader:
                user_ids = batch['user_ids'].to(self.device)
                movie_ids = batch['movie_ids'].to(self.device)
                universe_ids = batch['universe_ids'].to(self.device)
                universe_positions = batch['universe_positions'].to(self.device)
                release_positions = batch['release_positions'].to(self.device)
                phases = batch['phases'].to(self.device)
                adjacency = batch['adjacency'].to(self.device)
                connection_types = batch['connection_types'].to(self.device)
                target_ratings = batch['ratings'].to(self.device)
                target_next = batch['next_movie'].to(self.device)
                target_order = batch['preferred_order'].to(self.device)
                target_engagement = batch['completed_universe'].to(self.device).float()

                outputs = self.model(
                    user_ids, movie_ids, universe_ids, universe_positions,
                    release_positions, phases, adjacency, connection_types
                )

                total_metrics['rating_mse'] += F.mse_loss(
                    outputs['rating_pred'].squeeze(), target_ratings
                ).item()

                total_metrics['order_acc'] += (
                    outputs['order_logits'].argmax(dim=-1) == target_order
                ).float().mean().item()

                total_metrics['next_acc'] += (
                    outputs['next_movie_logits'].argmax(dim=-1) == target_next
                ).float().mean().item()

                pred_engagement = (outputs['engagement'].squeeze() > 0.5).float()
                total_metrics['engagement_acc'] += (
                    pred_engagement == target_engagement
                ).float().mean().item()

                num_batches += 1

        return {k: v / num_batches for k, v in total_metrics.items()}
