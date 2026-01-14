#!/usr/bin/env python3
"""
CineSync v2 - Unified Content Recommender
Single model architecture for both movies and TV shows

This replaces the separate movie_recommender.py and tv_recommender.py with
a unified model that handles both content types through a content_type parameter.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from enum import Enum
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


class ContentType(Enum):
    """Content type enumeration"""
    MOVIE = "movie"
    TV = "tv"


@dataclass
class ContentFeatures:
    """Container for content-specific features"""
    genre_features: Optional[torch.Tensor] = None
    # TV-specific features
    episode_count: Optional[torch.Tensor] = None
    season_count: Optional[torch.Tensor] = None
    duration: Optional[torch.Tensor] = None
    status: Optional[torch.Tensor] = None


class UnifiedContentRecommender(nn.Module):
    """
    Unified hybrid recommendation model for movies and TV shows.

    This model combines:
    - Neural Collaborative Filtering (NCF) for user-item interactions
    - Optional genre-based content features (for both content types)
    - TV-specific features (episode count, season count, duration, status)
    - Deep learning architecture for non-linear pattern learning

    Architecture:
    - Shared user embeddings across content types
    - Item embeddings (movies and TV shows share same embedding space)
    - Optional genre feature encoding
    - Conditional TV-specific feature encoding
    - MLP prediction layers with batch normalization

    Args:
        num_users: Total number of users
        num_items: Total number of items (movies + TV shows)
        num_genres: Number of unique genres (default: 20)
        embedding_dim: Dimension for user/item embeddings (default: 64)
        hidden_dims: List of hidden layer dimensions (default: [128, 64, 32])
        dropout_rate: Dropout rate for regularization (default: 0.2)
        use_genre_features: Whether to use genre features (default: True)
        use_tv_features: Whether to use TV-specific features (default: True)
    """

    def __init__(
        self,
        num_users: int,
        num_items: int,
        num_genres: int = 20,
        embedding_dim: int = 64,
        hidden_dims: List[int] = None,
        dropout_rate: float = 0.2,
        use_genre_features: bool = True,
        use_tv_features: bool = True
    ):
        super(UnifiedContentRecommender, self).__init__()

        if hidden_dims is None:
            hidden_dims = [128, 64, 32]

        self.num_users = num_users
        self.num_items = num_items
        self.num_genres = num_genres
        self.embedding_dim = embedding_dim
        self.hidden_dims = hidden_dims
        self.use_genre_features = use_genre_features
        self.use_tv_features = use_tv_features

        # User embeddings (shared across content types)
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.user_bias = nn.Embedding(num_users, 1)

        # Item embeddings (movies and TV shows in same space)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        self.item_bias = nn.Embedding(num_items, 1)

        # Global bias
        self.global_bias = nn.Parameter(torch.zeros(1))

        # Genre feature encoding (optional, works for both content types)
        if use_genre_features:
            self.genre_encoder = nn.Sequential(
                nn.Linear(num_genres, embedding_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate * 0.5)
            )

        # TV-specific feature encoding (conditional)
        if use_tv_features:
            tv_feature_dim = embedding_dim // 4
            self.episode_encoder = nn.Linear(1, tv_feature_dim)
            self.season_encoder = nn.Linear(1, tv_feature_dim)
            self.duration_encoder = nn.Linear(1, tv_feature_dim)
            self.status_embedding = nn.Embedding(5, tv_feature_dim)  # 5 status types

            # Projection to combine TV features
            self.tv_feature_projection = nn.Linear(tv_feature_dim * 4, embedding_dim)

        # Calculate MLP input dimension
        input_dim = embedding_dim * 2  # user + item
        if use_genre_features:
            input_dim += embedding_dim
        if use_tv_features:
            input_dim += embedding_dim  # TV features projected to embedding_dim

        # Deep neural network layers
        self.mlp_layers = nn.ModuleList()
        prev_dim = input_dim

        for i, hidden_dim in enumerate(hidden_dims):
            self.mlp_layers.append(nn.Linear(prev_dim, hidden_dim))
            self.mlp_layers.append(nn.BatchNorm1d(hidden_dim))
            self.mlp_layers.append(nn.ReLU())
            self.mlp_layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim

        # Output layer
        self.output_layer = nn.Linear(hidden_dims[-1], 1)

        # Initialize weights
        self._init_weights()

        logger.info(f"UnifiedContentRecommender initialized: {num_users} users, {num_items} items, "
                   f"genres={use_genre_features}, tv_features={use_tv_features}")

    def _init_weights(self):
        """Initialize model weights using Xavier initialization"""
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)
        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.item_bias.weight)

        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def _encode_tv_features(
        self,
        episode_count: torch.Tensor,
        season_count: torch.Tensor,
        duration: torch.Tensor,
        status: torch.Tensor
    ) -> torch.Tensor:
        """Encode TV-specific features into a single embedding"""
        episode_emb = F.relu(self.episode_encoder(episode_count.unsqueeze(-1) if episode_count.dim() == 1 else episode_count))
        season_emb = F.relu(self.season_encoder(season_count.unsqueeze(-1) if season_count.dim() == 1 else season_count))
        duration_emb = F.relu(self.duration_encoder(duration.unsqueeze(-1) if duration.dim() == 1 else duration))
        status_emb = self.status_embedding(status.long())

        # Concatenate and project
        tv_concat = torch.cat([episode_emb, season_emb, duration_emb, status_emb], dim=-1)
        tv_emb = F.relu(self.tv_feature_projection(tv_concat))

        return tv_emb

    def forward(
        self,
        user_ids: torch.Tensor,
        item_ids: torch.Tensor,
        content_type: ContentType = ContentType.MOVIE,
        genre_features: Optional[torch.Tensor] = None,
        tv_features: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass supporting both content types.

        Args:
            user_ids: Tensor of user IDs [batch_size]
            item_ids: Tensor of item IDs [batch_size]
            content_type: ContentType.MOVIE or ContentType.TV
            genre_features: Optional genre one-hot features [batch_size, num_genres]
            tv_features: Optional TV features [batch_size, 4]
                        (episode_count, season_count, duration, status)

        Returns:
            Predicted ratings [batch_size]
        """
        batch_size = user_ids.size(0)
        device = user_ids.device

        # Get base embeddings
        user_emb = self.user_embedding(user_ids)
        item_emb = self.item_embedding(item_ids)

        # Get bias terms
        user_bias = self.user_bias(user_ids).squeeze(-1)
        item_bias = self.item_bias(item_ids).squeeze(-1)

        # Build feature list
        features = [user_emb, item_emb]

        # Add genre features if enabled
        if self.use_genre_features:
            if genre_features is not None:
                genre_emb = self.genre_encoder(genre_features)
            else:
                # Zero padding if genres not provided
                genre_emb = torch.zeros(batch_size, self.embedding_dim, device=device)
            features.append(genre_emb)

        # Add TV features if enabled
        if self.use_tv_features:
            if content_type == ContentType.TV and tv_features is not None:
                tv_emb = self._encode_tv_features(
                    tv_features[:, 0],  # episode_count
                    tv_features[:, 1],  # season_count
                    tv_features[:, 2],  # duration
                    tv_features[:, 3]   # status
                )
            else:
                # Zero padding for movies or when TV features not provided
                tv_emb = torch.zeros(batch_size, self.embedding_dim, device=device)
            features.append(tv_emb)

        # Concatenate all features
        mlp_input = torch.cat(features, dim=-1)

        # Pass through MLP layers
        x = mlp_input
        for layer in self.mlp_layers:
            x = layer(x)

        # Final prediction
        mlp_output = self.output_layer(x).squeeze(-1)

        # Combine with bias terms
        prediction = mlp_output + user_bias + item_bias + self.global_bias

        return prediction

    def predict(
        self,
        user_ids: torch.Tensor,
        item_ids: torch.Tensor,
        content_type: ContentType = ContentType.MOVIE,
        genre_features: Optional[torch.Tensor] = None,
        tv_features: Optional[torch.Tensor] = None
    ) -> np.ndarray:
        """Make predictions and return numpy array"""
        self.eval()
        with torch.no_grad():
            predictions = self.forward(user_ids, item_ids, content_type, genre_features, tv_features)
            return predictions.cpu().numpy()

    def predict_for_user(
        self,
        user_id: int,
        item_ids: List[int],
        content_type: ContentType = ContentType.MOVIE,
        genre_features: Optional[torch.Tensor] = None,
        tv_features: Optional[torch.Tensor] = None,
        device: Optional[torch.device] = None
    ) -> np.ndarray:
        """Predict ratings for a specific user and list of items"""
        if device is None:
            device = next(self.parameters()).device

        self.eval()
        with torch.no_grad():
            user_tensor = torch.tensor([user_id] * len(item_ids), dtype=torch.long, device=device)
            item_tensor = torch.tensor(item_ids, dtype=torch.long, device=device)

            predictions = self.forward(
                user_tensor, item_tensor, content_type, genre_features, tv_features
            )
            return predictions.cpu().numpy()

    def get_top_recommendations(
        self,
        user_id: int,
        candidate_items: List[int],
        content_type: ContentType = ContentType.MOVIE,
        genre_features: Optional[torch.Tensor] = None,
        tv_features: Optional[torch.Tensor] = None,
        top_k: int = 10,
        device: Optional[torch.device] = None
    ) -> List[Tuple[int, float]]:
        """
        Get top-k recommendations for a user.

        Args:
            user_id: User ID
            candidate_items: List of candidate item IDs
            content_type: Content type (MOVIE or TV)
            genre_features: Optional genre features for candidates
            tv_features: Optional TV features for candidates
            top_k: Number of recommendations to return
            device: Device for computation

        Returns:
            List of (item_id, predicted_rating) tuples sorted by rating
        """
        predictions = self.predict_for_user(
            user_id, candidate_items, content_type, genre_features, tv_features, device
        )

        item_rating_pairs = list(zip(candidate_items, predictions.tolist()))
        item_rating_pairs.sort(key=lambda x: x[1], reverse=True)

        return item_rating_pairs[:top_k]

    def get_user_embedding(self, user_ids: torch.Tensor) -> torch.Tensor:
        """Get user embeddings for transfer learning or analysis"""
        return self.user_embedding(user_ids)

    def get_item_embedding(self, item_ids: torch.Tensor) -> torch.Tensor:
        """Get item embeddings for transfer learning or analysis"""
        return self.item_embedding(item_ids)


class ContentDataset(torch.utils.data.Dataset):
    """
    Unified dataset for both movies and TV shows.

    Args:
        user_ids: Array of user IDs
        item_ids: Array of item IDs
        ratings: Array of ratings
        content_type: Content type for all items in this dataset
        genre_features: Optional genre features [num_samples, num_genres]
        tv_features: Optional TV features [num_samples, 4]
    """

    def __init__(
        self,
        user_ids: np.ndarray,
        item_ids: np.ndarray,
        ratings: np.ndarray,
        content_type: ContentType = ContentType.MOVIE,
        genre_features: Optional[np.ndarray] = None,
        tv_features: Optional[np.ndarray] = None
    ):
        self.user_ids = torch.tensor(user_ids, dtype=torch.long)
        self.item_ids = torch.tensor(item_ids, dtype=torch.long)
        self.ratings = torch.tensor(ratings, dtype=torch.float32)
        self.content_type = content_type

        self.genre_features = None
        if genre_features is not None:
            self.genre_features = torch.tensor(genre_features, dtype=torch.float32)

        self.tv_features = None
        if tv_features is not None:
            self.tv_features = torch.tensor(tv_features, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.ratings)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = {
            'user_id': self.user_ids[idx],
            'item_id': self.item_ids[idx],
            'rating': self.ratings[idx],
            'content_type': self.content_type.value
        }

        if self.genre_features is not None:
            item['genre_features'] = self.genre_features[idx]

        if self.tv_features is not None:
            item['tv_features'] = self.tv_features[idx]

        return item


class UnifiedRecommendationSystem:
    """
    Complete recommendation system with data handling and model management.

    Supports both movies and TV shows through unified interface.
    """

    def __init__(self, model_dir: str = "models/"):
        """
        Initialize the recommendation system.

        Args:
            model_dir: Directory for saving/loading models
        """
        from pathlib import Path

        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)

        self.model: Optional[UnifiedContentRecommender] = None
        self.user_id_map: Dict[int, int] = {}
        self.item_id_map: Dict[int, int] = {}
        self.reverse_item_id_map: Dict[int, int] = {}
        self.item_metadata: Dict[int, Dict] = {}
        self.content_type_map: Dict[int, ContentType] = {}  # item_id -> content_type

        self.logger = logging.getLogger(__name__)

    def save_model(self, filename: str = "unified_recommender.pt"):
        """Save the trained model and all mappings"""
        if self.model is None:
            raise ValueError("No model to save. Train the model first.")

        save_path = self.model_dir / filename

        torch.save({
            'model_state_dict': self.model.state_dict(),
            'model_config': {
                'num_users': self.model.num_users,
                'num_items': self.model.num_items,
                'num_genres': self.model.num_genres,
                'embedding_dim': self.model.embedding_dim,
                'hidden_dims': self.model.hidden_dims,
                'use_genre_features': self.model.use_genre_features,
                'use_tv_features': self.model.use_tv_features
            },
            'user_id_map': self.user_id_map,
            'item_id_map': self.item_id_map,
            'reverse_item_id_map': self.reverse_item_id_map,
            'item_metadata': self.item_metadata,
            'content_type_map': {k: v.value for k, v in self.content_type_map.items()}
        }, save_path)

        self.logger.info(f"Model saved to {save_path}")

    def load_model(self, filename: str = "unified_recommender.pt", device: Optional[torch.device] = None):
        """Load a trained model and all mappings"""
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        load_path = self.model_dir / filename

        if not load_path.exists():
            raise FileNotFoundError(f"Model file not found: {load_path}")

        checkpoint = torch.load(load_path, map_location=device, weights_only=False)

        # Load mappings
        self.user_id_map = checkpoint['user_id_map']
        self.item_id_map = checkpoint['item_id_map']
        self.reverse_item_id_map = checkpoint['reverse_item_id_map']
        self.item_metadata = checkpoint['item_metadata']
        self.content_type_map = {
            k: ContentType(v) for k, v in checkpoint['content_type_map'].items()
        }

        # Initialize model
        config = checkpoint['model_config']
        self.model = UnifiedContentRecommender(
            num_users=config['num_users'],
            num_items=config['num_items'],
            num_genres=config['num_genres'],
            embedding_dim=config['embedding_dim'],
            hidden_dims=config['hidden_dims'],
            use_genre_features=config['use_genre_features'],
            use_tv_features=config['use_tv_features']
        ).to(device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        self.logger.info(f"Model loaded from {load_path}")

    def recommend(
        self,
        user_id: int,
        content_type: Optional[ContentType] = None,
        num_recommendations: int = 10,
        exclude_seen: bool = True,
        ratings_df: Optional[pd.DataFrame] = None
    ) -> List[Dict]:
        """
        Get recommendations for a user.

        Args:
            user_id: Original user ID
            content_type: Filter by content type (None = both)
            num_recommendations: Number of recommendations
            exclude_seen: Exclude items user has already rated
            ratings_df: Ratings dataframe to check seen items

        Returns:
            List of recommendation dictionaries
        """
        if self.model is None:
            raise ValueError("No model loaded")

        if user_id not in self.user_id_map:
            raise ValueError(f"User {user_id} not in training data")

        mapped_user_id = self.user_id_map[user_id]

        # Get candidate items
        candidate_items = list(self.item_id_map.keys())

        # Filter by content type if specified
        if content_type is not None:
            candidate_items = [
                item_id for item_id in candidate_items
                if self.content_type_map.get(self.item_id_map[item_id]) == content_type
            ]

        # Exclude seen items
        if exclude_seen and ratings_df is not None:
            seen_items = set(ratings_df[ratings_df['userId'] == user_id]['itemId'].values)
            candidate_items = [i for i in candidate_items if i not in seen_items]

        # Map to internal IDs
        mapped_item_ids = [self.item_id_map[i] for i in candidate_items]

        # Get predictions
        device = next(self.model.parameters()).device
        top_recs = self.model.get_top_recommendations(
            mapped_user_id, mapped_item_ids,
            content_type or ContentType.MOVIE,
            top_k=num_recommendations,
            device=device
        )

        # Build recommendation list
        recommendations = []
        for mapped_item_id, score in top_recs:
            original_id = self.reverse_item_id_map[mapped_item_id]
            item_content_type = self.content_type_map.get(mapped_item_id, ContentType.MOVIE)

            rec = {
                'item_id': original_id,
                'predicted_rating': float(score),
                'content_type': item_content_type.value
            }

            if original_id in self.item_metadata:
                rec.update(self.item_metadata[original_id])

            recommendations.append(rec)

        return recommendations


# Backward compatibility aliases
MovieHybridRecommender = UnifiedContentRecommender
TVShowRecommenderModel = UnifiedContentRecommender
