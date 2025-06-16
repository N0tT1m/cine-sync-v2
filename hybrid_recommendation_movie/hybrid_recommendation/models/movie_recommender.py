#!/usr/bin/env python3
# CineSync v2 - Movie Recommendation Model
# Hybrid neural collaborative filtering model for movie recommendations

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import pickle
import logging
from pathlib import Path

class MovieHybridRecommender(nn.Module):
    """Hybrid movie recommendation model combining collaborative and content-based filtering
    
    This model combines:
    - Neural Collaborative Filtering (NCF) for user-item interactions
    - Content-based features from movie metadata
    - Deep learning architecture for non-linear patterns
    """
    
    def __init__(self, num_users: int, num_movies: int, embedding_dim: int = 64, 
                 hidden_dims: List[int] = [128, 64, 32], dropout_rate: float = 0.2):
        """Initialize the hybrid movie recommender
        
        Args:
            num_users: Number of unique users in the dataset
            num_movies: Number of unique movies in the dataset
            embedding_dim: Dimension of user and movie embeddings
            hidden_dims: List of hidden layer dimensions
            dropout_rate: Dropout rate for regularization
        """
        super(MovieHybridRecommender, self).__init__()
        
        self.num_users = num_users
        self.num_movies = num_movies
        self.embedding_dim = embedding_dim
        
        # User and movie embeddings
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.movie_embedding = nn.Embedding(num_movies, embedding_dim)
        
        # Bias terms
        self.user_bias = nn.Embedding(num_users, 1)
        self.movie_bias = nn.Embedding(num_movies, 1)
        self.global_bias = nn.Parameter(torch.zeros(1))
        
        # Deep neural network layers
        input_dim = embedding_dim * 2  # Concatenated user and movie embeddings
        self.mlp_layers = nn.ModuleList()
        
        for i, hidden_dim in enumerate(hidden_dims):
            if i == 0:
                self.mlp_layers.append(nn.Linear(input_dim, hidden_dim))
            else:
                self.mlp_layers.append(nn.Linear(hidden_dims[i-1], hidden_dim))
            self.mlp_layers.append(nn.ReLU())
            self.mlp_layers.append(nn.Dropout(dropout_rate))
        
        # Output layer
        self.output_layer = nn.Linear(hidden_dims[-1], 1)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights using Xavier initialization"""
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.movie_embedding.weight)
        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.movie_bias.weight)
        
        for layer in self.mlp_layers:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
    
    def forward(self, user_ids: torch.Tensor, movie_ids: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model
        
        Args:
            user_ids: Tensor of user IDs
            movie_ids: Tensor of movie IDs
            
        Returns:
            Predicted ratings
        """
        # Get embeddings
        user_emb = self.user_embedding(user_ids)
        movie_emb = self.movie_embedding(movie_ids)
        
        # Get bias terms
        user_bias = self.user_bias(user_ids).squeeze()
        movie_bias = self.movie_bias(movie_ids).squeeze()
        
        # Concatenate embeddings for MLP
        mlp_input = torch.cat([user_emb, movie_emb], dim=-1)
        
        # Pass through MLP layers
        x = mlp_input
        for layer in self.mlp_layers:
            x = layer(x)
        
        # Final prediction
        mlp_output = self.output_layer(x).squeeze()
        
        # Combine MLP output with bias terms
        prediction = mlp_output + user_bias + movie_bias + self.global_bias
        
        return prediction
    
    def predict_for_user(self, user_id: int, movie_ids: List[int], 
                        device: torch.device = None) -> np.ndarray:
        """Predict ratings for a specific user and list of movies
        
        Args:
            user_id: ID of the user
            movie_ids: List of movie IDs to predict for
            device: Device to run predictions on
            
        Returns:
            Array of predicted ratings
        """
        if device is None:
            device = next(self.parameters()).device
        
        self.eval()
        with torch.no_grad():
            user_tensor = torch.tensor([user_id] * len(movie_ids), device=device)
            movie_tensor = torch.tensor(movie_ids, device=device)
            
            predictions = self.forward(user_tensor, movie_tensor)
            return predictions.cpu().numpy()
    
    def get_top_recommendations(self, user_id: int, movie_ids: List[int], 
                               top_k: int = 10, device: torch.device = None) -> List[Tuple[int, float]]:
        """Get top-k movie recommendations for a user
        
        Args:
            user_id: ID of the user
            movie_ids: List of candidate movie IDs
            top_k: Number of recommendations to return
            device: Device to run predictions on
            
        Returns:
            List of (movie_id, predicted_rating) tuples sorted by rating
        """
        predictions = self.predict_for_user(user_id, movie_ids, device)
        
        # Sort by predicted rating (descending)
        movie_rating_pairs = list(zip(movie_ids, predictions))
        movie_rating_pairs.sort(key=lambda x: x[1], reverse=True)
        
        return movie_rating_pairs[:top_k]

class MovieRecommendationSystem:
    """Complete movie recommendation system with data handling and model management"""
    
    def __init__(self, model_path: str = "models/"):
        """Initialize the recommendation system
        
        Args:
            model_path: Path to save/load model files
        """
        self.model_path = Path(model_path)
        self.model_path.mkdir(exist_ok=True)
        
        self.model = None
        self.user_id_map = {}
        self.movie_id_map = {}
        self.reverse_movie_id_map = {}
        self.movie_metadata = {}
        
        self.logger = logging.getLogger(__name__)
    
    def prepare_data(self, ratings_df: pd.DataFrame, 
                    movies_df: Optional[pd.DataFrame] = None) -> Tuple[torch.utils.data.Dataset, Dict]:
        """Prepare data for training
        
        Args:
            ratings_df: DataFrame with columns ['userId', 'movieId', 'rating']
            movies_df: Optional DataFrame with movie metadata
            
        Returns:
            Dataset and metadata dictionary
        """
        # Create ID mappings
        unique_users = ratings_df['userId'].unique()
        unique_movies = ratings_df['movieId'].unique()
        
        self.user_id_map = {user_id: idx for idx, user_id in enumerate(unique_users)}
        self.movie_id_map = {movie_id: idx for idx, movie_id in enumerate(unique_movies)}
        self.reverse_movie_id_map = {idx: movie_id for movie_id, idx in self.movie_id_map.items()}
        
        # Map IDs in the ratings DataFrame
        ratings_df = ratings_df.copy()
        ratings_df['userId'] = ratings_df['userId'].map(self.user_id_map)
        ratings_df['movieId'] = ratings_df['movieId'].map(self.movie_id_map)
        
        # Store movie metadata if provided
        if movies_df is not None:
            self.movie_metadata = movies_df.set_index('movieId').to_dict('index')
        
        # Create PyTorch dataset
        class MovieRatingDataset(torch.utils.data.Dataset):
            def __init__(self, ratings_df):
                self.user_ids = torch.tensor(ratings_df['userId'].values, dtype=torch.long)
                self.movie_ids = torch.tensor(ratings_df['movieId'].values, dtype=torch.long)
                self.ratings = torch.tensor(ratings_df['rating'].values, dtype=torch.float32)
            
            def __len__(self):
                return len(self.ratings)
            
            def __getitem__(self, idx):
                return self.user_ids[idx], self.movie_ids[idx], self.ratings[idx]
        
        dataset = MovieRatingDataset(ratings_df)
        
        metadata = {
            'num_users': len(unique_users),
            'num_movies': len(unique_movies),
            'num_ratings': len(ratings_df),
            'rating_range': (ratings_df['rating'].min(), ratings_df['rating'].max())
        }
        
        return dataset, metadata
    
    def train(self, dataset: torch.utils.data.Dataset, metadata: Dict,
              num_epochs: int = 50, batch_size: int = 1024, learning_rate: float = 0.001,
              device: torch.device = None) -> Dict:
        """Train the recommendation model
        
        Args:
            dataset: Training dataset
            metadata: Dataset metadata
            num_epochs: Number of training epochs
            batch_size: Training batch size  
            learning_rate: Learning rate for optimizer
            device: Device to train on
            
        Returns:
            Training history dictionary
        """
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize model
        self.model = MovieHybridRecommender(
            num_users=metadata['num_users'],
            num_movies=metadata['num_movies']
        ).to(device)
        
        # Setup training
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()
        
        # Training loop
        history = {'train_loss': []}
        
        self.model.train()
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            num_batches = 0
            
            for user_ids, movie_ids, ratings in dataloader:
                user_ids, movie_ids, ratings = user_ids.to(device), movie_ids.to(device), ratings.to(device)
                
                optimizer.zero_grad()
                predictions = self.model(user_ids, movie_ids)
                loss = criterion(predictions, ratings)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
            
            avg_loss = epoch_loss / num_batches
            history['train_loss'].append(avg_loss)
            
            if (epoch + 1) % 10 == 0:
                self.logger.info(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}")
        
        return history
    
    def save_model(self, filename: str = "movie_recommender.pth"):
        """Save the trained model and mappings"""
        if self.model is None:
            raise ValueError("No model to save. Train the model first.")
        
        save_path = self.model_path / filename
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'user_id_map': self.user_id_map,
            'movie_id_map': self.movie_id_map,
            'reverse_movie_id_map': self.reverse_movie_id_map,
            'movie_metadata': self.movie_metadata,
            'model_config': {
                'num_users': self.model.num_users,
                'num_movies': self.model.num_movies,
                'embedding_dim': self.model.embedding_dim
            }
        }, save_path)
        
        self.logger.info(f"Model saved to {save_path}")
    
    def load_model(self, filename: str = "movie_recommender.pth", device: torch.device = None):
        """Load a trained model and mappings"""
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        load_path = self.model_path / filename
        
        if not load_path.exists():
            raise FileNotFoundError(f"Model file not found: {load_path}")
        
        checkpoint = torch.load(load_path, map_location=device)
        
        # Load mappings
        self.user_id_map = checkpoint['user_id_map']
        self.movie_id_map = checkpoint['movie_id_map']
        self.reverse_movie_id_map = checkpoint['reverse_movie_id_map']
        self.movie_metadata = checkpoint['movie_metadata']
        
        # Initialize and load model
        config = checkpoint['model_config']
        self.model = MovieHybridRecommender(
            num_users=config['num_users'],
            num_movies=config['num_movies'],
            embedding_dim=config['embedding_dim']
        ).to(device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        self.logger.info(f"Model loaded from {load_path}")
    
    def recommend_movies(self, user_id: int, num_recommendations: int = 10,
                        exclude_seen: bool = True, ratings_df: Optional[pd.DataFrame] = None) -> List[Dict]:
        """Get movie recommendations for a user
        
        Args:
            user_id: Original user ID (before mapping)
            num_recommendations: Number of recommendations to return
            exclude_seen: Whether to exclude movies the user has already rated
            ratings_df: Optional ratings DataFrame to exclude seen movies
            
        Returns:
            List of recommendation dictionaries
        """
        if self.model is None:
            raise ValueError("No model loaded. Load or train a model first.")
        
        if user_id not in self.user_id_map:
            raise ValueError(f"User ID {user_id} not found in training data")
        
        mapped_user_id = self.user_id_map[user_id]
        
        # Get candidate movies
        candidate_movies = list(self.movie_id_map.keys())
        
        # Exclude movies the user has already seen if requested
        if exclude_seen and ratings_df is not None:
            user_ratings = ratings_df[ratings_df['userId'] == user_id]
            seen_movies = set(user_ratings['movieId'].values)
            candidate_movies = [m for m in candidate_movies if m not in seen_movies]
        
        # Map to internal IDs
        mapped_movie_ids = [self.movie_id_map[m] for m in candidate_movies]
        
        # Get predictions
        device = next(self.model.parameters()).device
        top_recs = self.model.get_top_recommendations(
            mapped_user_id, mapped_movie_ids, num_recommendations, device
        )
        
        # Convert back to original IDs and add metadata
        recommendations = []
        for mapped_movie_id, predicted_rating in top_recs:
            original_movie_id = self.reverse_movie_id_map[mapped_movie_id]
            
            rec = {
                'movie_id': original_movie_id,
                'predicted_rating': float(predicted_rating),
            }
            
            # Add movie metadata if available
            if original_movie_id in self.movie_metadata:
                rec.update(self.movie_metadata[original_movie_id])
            
            recommendations.append(rec)
        
        return recommendations