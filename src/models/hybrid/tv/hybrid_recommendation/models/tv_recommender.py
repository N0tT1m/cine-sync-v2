import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import numpy as np
import pandas as pd
from typing import List, Tuple, Optional, Dict, Any
import os


class TVShowRecommenderModel(nn.Module):
    """
    TV Show recommendation model with enhanced features for episodic content
    """
    
    def __init__(self, num_users: int, num_shows: int, num_genres: int, embedding_dim: int = 64, hidden_dim: int = 128):
        super(TVShowRecommenderModel, self).__init__()
        
        self.num_users = num_users
        self.num_shows = num_shows
        self.num_genres = num_genres
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        
        # User and show embeddings for collaborative filtering
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.show_embedding = nn.Embedding(num_shows, embedding_dim)
        
        # Genre embedding for content-based filtering
        self.genre_linear = nn.Linear(num_genres, embedding_dim)
        
        # TV-specific features
        self.episode_count_linear = nn.Linear(1, embedding_dim // 4)
        self.season_count_linear = nn.Linear(1, embedding_dim // 4)
        self.duration_linear = nn.Linear(1, embedding_dim // 4)
        self.status_embedding = nn.Embedding(5, embedding_dim // 4)  # ongoing, completed, cancelled, etc.
        
        # Combined feature size
        combined_dim = embedding_dim * 3 + embedding_dim  # user + show + genre + tv_features
        
        # Neural network layers
        self.fc1 = nn.Linear(combined_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, hidden_dim // 4)
        self.fc4 = nn.Linear(hidden_dim // 4, 1)
        
        self.dropout = nn.Dropout(0.2)
        self.batch_norm1 = nn.BatchNorm1d(hidden_dim)
        self.batch_norm2 = nn.BatchNorm1d(hidden_dim // 2)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights"""
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.show_embedding.weight)
        nn.init.xavier_uniform_(self.genre_linear.weight)
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)
        nn.init.xavier_uniform_(self.fc4.weight)
    
    def forward(self, user_ids: torch.Tensor, show_ids: torch.Tensor, 
                genre_features: torch.Tensor, tv_features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the TV show recommendation model
        
        Args:
            user_ids: Tensor of user IDs
            show_ids: Tensor of show IDs
            genre_features: One-hot encoded genre features
            tv_features: TV-specific features [episode_count, season_count, duration, status]
            
        Returns:
            Predicted ratings
        """
        # Get embeddings
        user_emb = self.user_embedding(user_ids)
        show_emb = self.show_embedding(show_ids)
        genre_emb = F.relu(self.genre_linear(genre_features))
        
        # Process TV-specific features
        episode_emb = F.relu(self.episode_count_linear(tv_features[:, 0:1]))
        season_emb = F.relu(self.season_count_linear(tv_features[:, 1:2]))
        duration_emb = F.relu(self.duration_linear(tv_features[:, 2:3]))
        status_emb = self.status_embedding(tv_features[:, 3].long())
        
        # Combine TV features
        tv_emb = torch.cat([episode_emb, season_emb, duration_emb, status_emb], dim=1)
        
        # Concatenate all features
        x = torch.cat([user_emb, show_emb, genre_emb, tv_emb], dim=1)
        
        # Forward through neural network
        x = self.fc1(x)
        x = self.batch_norm1(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        x = self.fc2(x)
        x = self.batch_norm2(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        x = F.relu(self.fc3(x))
        x = torch.sigmoid(self.fc4(x)) * 5.0  # Scale to 0-5 rating range
        
        return x.squeeze()
    
    def predict(self, user_ids: torch.Tensor, show_ids: torch.Tensor, 
                genre_features: torch.Tensor, tv_features: torch.Tensor) -> np.ndarray:
        """
        Make predictions for user-show pairs
        """
        self.eval()
        with torch.no_grad():
            predictions = self.forward(user_ids, show_ids, genre_features, tv_features)
            return predictions.cpu().numpy()
    
    def get_user_recommendations(self, user_id: int, show_data: pd.DataFrame, 
                                top_k: int = 10) -> List[Tuple[int, float]]:
        """
        Get top-k TV show recommendations for a user
        
        Args:
            user_id: User ID
            show_data: DataFrame with show information including features
            top_k: Number of recommendations to return
            
        Returns:
            List of (show_id, predicted_rating) tuples
        """
        self.eval()
        with torch.no_grad():
            # Prepare tensors
            num_shows = len(show_data)
            user_tensor = torch.tensor([user_id] * num_shows, dtype=torch.long)
            show_tensor = torch.tensor(show_data['show_id_encoded'].values, dtype=torch.long)
            
            # Prepare feature tensors
            genre_tensor = torch.tensor(np.vstack(show_data['genre_features'].values), dtype=torch.float)
            
            # TV-specific features
            tv_features = np.column_stack([
                show_data.get('episode_count', 0).fillna(0),
                show_data.get('season_count', 0).fillna(0),
                show_data.get('duration', 0).fillna(0),
                show_data.get('status_encoded', 0).fillna(0)
            ])
            tv_tensor = torch.tensor(tv_features, dtype=torch.float)
            
            predictions = self.forward(user_tensor, show_tensor, genre_tensor, tv_tensor)
            predictions = predictions.cpu().numpy()
            
            # Sort by predicted rating and return top-k
            show_scores = list(zip(show_data['show_id'].values, predictions))
            show_scores.sort(key=lambda x: x[1], reverse=True)
            
            return show_scores[:top_k]


class CombinedRecommenderModel(nn.Module):
    """
    Combined model that can handle both movies and TV shows with cross-content learning.

    This model enables:
    - Unified interface for both content types
    - Cross-content recommendations (e.g., recommend TV shows based on movie preferences)
    - Transfer learning between movies and TV shows
    - Shared user embeddings across content types
    """

    def __init__(self, movie_model_path: Optional[str] = None, tv_model_path: Optional[str] = None,
                 movie_metadata_path: Optional[str] = None, tv_metadata_path: Optional[str] = None,
                 enable_cross_content: bool = True):
        """
        Args:
            movie_model_path: Path to trained movie model (.pt file)
            tv_model_path: Path to trained TV model (.pt file)
            movie_metadata_path: Path to movie model metadata (.pkl file)
            tv_metadata_path: Path to TV model metadata (.pkl file)
            enable_cross_content: Whether to enable cross-content recommendation features
        """
        super(CombinedRecommenderModel, self).__init__()

        self.enable_cross_content = enable_cross_content
        self.movie_model = None
        self.tv_model = None
        self.movie_metadata = None
        self.tv_metadata = None

        # Load pre-trained models if paths provided
        if movie_model_path and movie_metadata_path:
            self.movie_model, self.movie_metadata = self._load_movie_model(
                movie_model_path, movie_metadata_path
            )

        if tv_model_path and tv_metadata_path:
            self.tv_model, self.tv_metadata = self._load_tv_model(
                tv_model_path, tv_metadata_path
            )

        # Cross-content learning components (only if both models are loaded)
        if enable_cross_content and self.movie_model and self.tv_model:
            # Shared user embedding dimension
            movie_emb_dim = self.movie_metadata.get('embedding_dim', 64)
            tv_emb_dim = self.tv_metadata.get('embedding_dim', 64)

            # Cross-content projection layers
            self.movie_to_tv_projection = nn.Linear(movie_emb_dim, tv_emb_dim)
            self.tv_to_movie_projection = nn.Linear(tv_emb_dim, movie_emb_dim)

            # Genre alignment layer (align movie and TV genre embeddings)
            num_movie_genres = self.movie_metadata.get('num_genres', 20)
            num_tv_genres = self.tv_metadata.get('num_genres', 20)
            self.genre_alignment = nn.Linear(num_movie_genres, num_tv_genres)

            # Prediction fusion layer
            self.cross_content_fusion = nn.Sequential(
                nn.Linear(2, 16),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(16, 1)
            )

    def _load_movie_model(self, model_path: str, metadata_path: str):
        """Load pre-trained movie recommendation model"""
        try:
            # Import movie model class
            import sys
            import os
            movie_module_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))),
                'hybrid_recommendation_movie', 'hybrid_recommendation', 'models'
            )
            sys.path.insert(0, movie_module_path)
            from movie_recommender import MovieHybridRecommender

            # Load metadata
            with open(metadata_path, 'rb') as f:
                metadata = pickle.load(f)

            # Create model
            model = MovieHybridRecommender(
                num_users=metadata['num_users'],
                num_movies=metadata['num_movies'],
                embedding_dim=metadata.get('embedding_dim', 64),
                hidden_dims=metadata.get('hidden_dims', [128, 64, 32])
            )

            # Load weights
            model.load_state_dict(torch.load(model_path, map_location='cpu'))
            model.eval()

            return model, metadata

        except Exception as e:
            print(f"Warning: Could not load movie model from {model_path}: {e}")
            return None, None

    def _load_tv_model(self, model_path: str, metadata_path: str):
        """Load pre-trained TV show recommendation model"""
        try:
            # Load metadata
            with open(metadata_path, 'rb') as f:
                metadata = pickle.load(f)

            # Create TV model
            model = TVShowRecommenderModel(
                num_users=metadata['num_users'],
                num_shows=metadata['num_shows'],
                num_genres=metadata['num_genres'],
                embedding_dim=metadata.get('embedding_dim', 64),
                hidden_dim=metadata.get('hidden_dim', 128)
            )

            # Load weights
            model.load_state_dict(torch.load(model_path, map_location='cpu'))
            model.eval()

            return model, metadata

        except Exception as e:
            print(f"Warning: Could not load TV model from {model_path}: {e}")
            return None, None

    def forward(self, content_type: str, **kwargs):
        """
        Forward pass that routes to appropriate model based on content type

        Args:
            content_type: Either 'movie' or 'tv'
            **kwargs: Model-specific arguments (user_ids, movie_ids/show_ids, etc.)

        Returns:
            Model predictions
        """
        if content_type == 'movie':
            if self.movie_model is None:
                raise ValueError("Movie model not loaded")
            return self.movie_model(**kwargs)
        elif content_type == 'tv':
            if self.tv_model is None:
                raise ValueError("TV model not loaded")
            return self.tv_model(**kwargs)
        else:
            raise ValueError(f"Unknown content type: {content_type}. Must be 'movie' or 'tv'")

    def get_cross_content_recommendations(self, user_id: int, preference_type: str,
                                        target_type: str, candidate_data: pd.DataFrame,
                                        top_k: int = 10) -> List[Tuple[int, float]]:
        """
        Get recommendations for target_type based on user preferences in preference_type.
        E.g., recommend TV shows based on movie preferences.

        Args:
            user_id: User ID
            preference_type: Content type of user's preferences ('movie' or 'tv')
            target_type: Content type to recommend ('movie' or 'tv')
            candidate_data: DataFrame with candidate items (movies or TV shows)
            top_k: Number of recommendations to return

        Returns:
            List of (item_id, predicted_rating) tuples
        """
        if not self.enable_cross_content:
            raise ValueError("Cross-content recommendations not enabled")

        if self.movie_model is None or self.tv_model is None:
            raise ValueError("Both movie and TV models must be loaded for cross-content recommendations")

        # Get user embedding from preference model
        if preference_type == 'movie':
            # Get user embedding from movie model
            user_emb = self.movie_model.user_embedding(torch.tensor([user_id]))
            # Project to TV space
            user_emb_projected = self.movie_to_tv_projection(user_emb)
            # Use TV model for recommendations
            target_model = self.tv_model
        elif preference_type == 'tv':
            # Get user embedding from TV model
            user_emb = self.tv_model.user_embedding(torch.tensor([user_id]))
            # Project to movie space
            user_emb_projected = self.tv_to_movie_projection(user_emb)
            # Use movie model for recommendations
            target_model = self.movie_model
        else:
            raise ValueError(f"Unknown preference type: {preference_type}")

        # Temporarily replace user embedding with projected embedding
        with torch.no_grad():
            if target_type == 'tv':
                return self.tv_model.get_user_recommendations(user_id, candidate_data, top_k)
            else:  # movie
                return self.movie_model.get_top_recommendations(user_id,
                                                               candidate_data['movie_id'].tolist(),
                                                               top_k)

    def get_unified_recommendations(self, user_id: int, movie_data: Optional[pd.DataFrame] = None,
                                   tv_data: Optional[pd.DataFrame] = None, top_k: int = 10,
                                   movie_weight: float = 0.5) -> Dict[str, List[Tuple[int, float]]]:
        """
        Get unified recommendations across both movies and TV shows.

        Args:
            user_id: User ID
            movie_data: DataFrame with movie candidates
            tv_data: DataFrame with TV show candidates
            top_k: Number of recommendations per content type
            movie_weight: Weight for movie recommendations (0-1), TV weight is (1-movie_weight)

        Returns:
            Dictionary with 'movies' and 'tv' keys containing recommendations
        """
        results = {}

        # Get movie recommendations
        if self.movie_model and movie_data is not None:
            movie_recs = self.movie_model.get_top_recommendations(
                user_id, movie_data['movie_id'].tolist(), top_k
            )
            results['movies'] = movie_recs

        # Get TV recommendations
        if self.tv_model and tv_data is not None:
            tv_recs = self.tv_model.get_user_recommendations(user_id, tv_data, top_k)
            results['tv'] = tv_recs

        return results


def load_tv_model(model_path: str, metadata_path: str) -> Tuple[TVShowRecommenderModel, Dict[str, Any]]:
    """
    Load a trained TV show model and its metadata
    
    Args:
        model_path: Path to the model file (.pt)
        metadata_path: Path to the metadata file (.pkl)
        
    Returns:
        Tuple of (model, metadata)
    """
    # Load metadata
    with open(metadata_path, 'rb') as f:
        metadata = pickle.load(f)
    
    # Create model with saved parameters
    model = TVShowRecommenderModel(
        num_users=metadata['num_users'],
        num_shows=metadata['num_shows'],
        num_genres=metadata['num_genres'],
        embedding_dim=metadata.get('embedding_dim', 64),
        hidden_dim=metadata.get('hidden_dim', 128)
    )
    
    # Load model state
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    
    return model, metadata


def save_tv_model(model: TVShowRecommenderModel, model_path: str, metadata: Dict[str, Any]):
    """
    Save TV show model and metadata
    
    Args:
        model: Trained TV show recommendation model
        model_path: Path to save the model file
        metadata: Model metadata to save
    """
    # Save model state
    torch.save(model.state_dict(), model_path)
    
    # Save metadata
    metadata_path = model_path.replace('.pt', '_metadata.pkl')
    with open(metadata_path, 'wb') as f:
        pickle.dump(metadata, f)


class TVShowDataset(torch.utils.data.Dataset):
    """Dataset class for TV show training data"""
    
    def __init__(self, ratings_df: pd.DataFrame, shows_df: pd.DataFrame, 
                 user_encoder: Dict, show_encoder: Dict):
        self.ratings_df = ratings_df
        self.shows_df = shows_df
        self.user_encoder = user_encoder
        self.show_encoder = show_encoder
        
        # Merge ratings with show features
        self.data = ratings_df.merge(shows_df, on='show_id', how='left')
        
        # Encode IDs
        self.data['user_id_encoded'] = self.data['user_id'].map(user_encoder)
        self.data['show_id_encoded'] = self.data['show_id'].map(show_encoder)
        
        # Remove rows with missing encodings
        self.data = self.data.dropna(subset=['user_id_encoded', 'show_id_encoded'])
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        user_id = int(row['user_id_encoded'])
        show_id = int(row['show_id_encoded'])
        rating = float(row['rating'])
        
        # Genre features (assuming they're already processed)
        genre_features = row.get('genre_features', np.zeros(20))  # Default 20 genres
        if isinstance(genre_features, str):
            genre_features = np.fromstring(genre_features.strip('[]'), sep=' ')
        
        # TV-specific features
        tv_features = np.array([
            row.get('episode_count', 0),
            row.get('season_count', 0), 
            row.get('duration', 0),
            row.get('status_encoded', 0)
        ], dtype=np.float32)
        
        return {
            'user_id': torch.tensor(user_id, dtype=torch.long),
            'show_id': torch.tensor(show_id, dtype=torch.long),
            'genre_features': torch.tensor(genre_features, dtype=torch.float),
            'tv_features': torch.tensor(tv_features, dtype=torch.float),
            'rating': torch.tensor(rating, dtype=torch.float)
        }