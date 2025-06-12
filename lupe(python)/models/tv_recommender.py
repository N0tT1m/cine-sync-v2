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
    
    def __init__(self, num_users: int, num_shows: int, num_genres: int = 20, 
                 embedding_dim: int = 64, hidden_dim: int = 128):
        super(TVShowRecommenderModel, self).__init__()
        
        self.num_users = num_users
        self.num_shows = num_shows
        self.num_genres = num_genres
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        
        # User and show embeddings for collaborative filtering
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.show_embedding = nn.Embedding(num_shows, embedding_dim)
        
        # TV-specific feature processing
        self.tv_features_dim = 4  # episode_count, season_count, duration, status
        self.tv_feature_network = nn.Linear(self.tv_features_dim, embedding_dim // 2)
        
        # Genre embedding
        self.genre_network = nn.Linear(num_genres, embedding_dim // 2)
        
        # Neural network layers
        input_dim = embedding_dim * 2 + embedding_dim  # user + show + (tv_features + genres)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, 1)
        
        self.dropout = nn.Dropout(0.2)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights"""
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.show_embedding.weight)
        nn.init.xavier_uniform_(self.tv_feature_network.weight)
        nn.init.xavier_uniform_(self.genre_network.weight)
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)
    
    def forward(self, user_ids: torch.Tensor, show_ids: torch.Tensor, 
                tv_features: torch.Tensor, genre_features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the TV show model
        
        Args:
            user_ids: Tensor of user IDs
            show_ids: Tensor of show IDs
            tv_features: Tensor of TV-specific features [episode_count, season_count, duration, status]
            genre_features: Tensor of genre features (one-hot encoded)
            
        Returns:
            Predicted ratings
        """
        # Get embeddings
        user_emb = self.user_embedding(user_ids)
        show_emb = self.show_embedding(show_ids)
        
        # Process TV-specific features
        tv_emb = F.relu(self.tv_feature_network(tv_features))
        
        # Process genre features
        genre_emb = F.relu(self.genre_network(genre_features))
        
        # Combine TV and genre features
        content_emb = torch.cat([tv_emb, genre_emb], dim=1)
        
        # Concatenate all embeddings
        x = torch.cat([user_emb, show_emb, content_emb], dim=1)
        
        # Forward through neural network
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = torch.sigmoid(self.fc3(x)) * 5.0  # Scale to 0-5 rating range
        
        return x.squeeze()
    
    def predict(self, user_ids: torch.Tensor, show_ids: torch.Tensor,
                tv_features: torch.Tensor, genre_features: torch.Tensor) -> np.ndarray:
        """
        Make predictions for user-show pairs
        
        Args:
            user_ids: User IDs
            show_ids: Show IDs
            tv_features: TV-specific features
            genre_features: Genre features
            
        Returns:
            Predicted ratings as numpy array
        """
        self.eval()
        with torch.no_grad():
            predictions = self.forward(user_ids, show_ids, tv_features, genre_features)
            return predictions.cpu().numpy()
    
    def get_user_recommendations(self, user_id: int, show_data: List[Dict], 
                               top_k: int = 10) -> List[Tuple[int, float]]:
        """
        Get top-k TV show recommendations for a user
        
        Args:
            user_id: User ID
            show_data: List of show dictionaries with features
            top_k: Number of recommendations to return
            
        Returns:
            List of (show_id, predicted_rating) tuples
        """
        self.eval()
        with torch.no_grad():
            show_ids = [show['id'] for show in show_data]
            user_tensor = torch.tensor([user_id] * len(show_ids), dtype=torch.long)
            show_tensor = torch.tensor(show_ids, dtype=torch.long)
            
            # Prepare TV features and genre features
            tv_features = []
            genre_features = []
            
            for show in show_data:
                # TV features: episode_count, season_count, duration, status
                tv_feat = [
                    show.get('episode_count', 0),
                    show.get('season_count', 0), 
                    show.get('duration', 0),
                    show.get('status', 0)  # 0=ended, 1=ongoing, 2=canceled
                ]
                tv_features.append(tv_feat)
                
                # Genre features (one-hot encoded)
                genre_features.append(show.get('genre_vector', [0] * self.num_genres))
            
            tv_tensor = torch.tensor(tv_features, dtype=torch.float32)
            genre_tensor = torch.tensor(genre_features, dtype=torch.float32)
            
            predictions = self.forward(user_tensor, show_tensor, tv_tensor, genre_tensor)
            predictions = predictions.cpu().numpy()
            
            # Sort by predicted rating and return top-k
            show_scores = list(zip(show_ids, predictions))
            show_scores.sort(key=lambda x: x[1], reverse=True)
            
            return show_scores[:top_k]


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
        num_genres=metadata.get('num_genres', 20),
        embedding_dim=metadata.get('embedding_dim', 64),
        hidden_dim=metadata.get('hidden_dim', 128)
    )
    
    # Load model state
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    
    return model, metadata


def load_tv_data(tv_shows_path: str) -> pd.DataFrame:
    """
    Load TV show data from CSV file
    
    Args:
        tv_shows_path: Path to TV shows CSV file
        
    Returns:
        DataFrame with TV show data
    """
    return pd.read_csv(tv_shows_path)


def load_tv_lookup(lookup_path: str) -> Dict[str, Any]:
    """
    Load TV show lookup dictionary
    
    Args:
        lookup_path: Path to TV show lookup pickle file
        
    Returns:
        TV show lookup dictionary
    """
    with open(lookup_path, 'rb') as f:
        return pickle.load(f)


def preprocess_tv_features(show_info: Dict) -> List[float]:
    """
    Preprocess TV show features for model input
    
    Args:
        show_info: Dictionary with show information
        
    Returns:
        List of processed features
    """
    # Extract and normalize TV-specific features
    episode_count = min(show_info.get('episode_count', 0), 1000) / 1000.0  # Normalize to 0-1
    season_count = min(show_info.get('season_count', 0), 50) / 50.0  # Normalize to 0-1
    duration = min(show_info.get('episode_run_time', 0), 180) / 180.0  # Normalize to 0-1
    
    # Status encoding: 0=ended, 1=ongoing, 2=canceled
    status_map = {'ended': 0.0, 'returning series': 1.0, 'canceled': 2.0}
    status = status_map.get(show_info.get('status', '').lower(), 0.0) / 2.0  # Normalize to 0-1
    
    return [episode_count, season_count, duration, status]


def create_genre_vector(genres: str, all_genres: List[str]) -> List[float]:
    """
    Create one-hot encoded genre vector
    
    Args:
        genres: Pipe-separated genre string
        all_genres: List of all possible genres
        
    Returns:
        One-hot encoded genre vector
    """
    if not isinstance(genres, str):
        return [0.0] * len(all_genres)
    
    show_genres = set(genre.strip().lower() for genre in genres.split('|'))
    genre_vector = []
    
    for genre in all_genres:
        genre_vector.append(1.0 if genre.lower() in show_genres else 0.0)
    
    return genre_vector