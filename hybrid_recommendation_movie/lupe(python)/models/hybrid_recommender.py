import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import numpy as np
import pandas as pd
from typing import List, Tuple, Optional, Dict, Any
import os


class HybridRecommenderModel(nn.Module):
    """
    Hybrid recommendation model combining collaborative filtering and content-based filtering
    """
    
    def __init__(self, num_users: int, num_items: int, embedding_dim: int = 64, hidden_dim: int = 128):
        super(HybridRecommenderModel, self).__init__()
        
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        
        # User and item embeddings for collaborative filtering
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        
        # Neural network layers
        self.fc1 = nn.Linear(embedding_dim * 2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, 1)
        
        self.dropout = nn.Dropout(0.2)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights"""
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)
    
    def forward(self, user_ids: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model
        
        Args:
            user_ids: Tensor of user IDs
            item_ids: Tensor of item IDs
            
        Returns:
            Predicted ratings
        """
        # Get embeddings
        user_emb = self.user_embedding(user_ids)
        item_emb = self.item_embedding(item_ids)
        
        # Concatenate embeddings
        x = torch.cat([user_emb, item_emb], dim=1)
        
        # Forward through neural network
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = torch.sigmoid(self.fc3(x)) * 5.0  # Scale to 0-5 rating range
        
        return x.squeeze()
    
    def predict(self, user_ids: torch.Tensor, item_ids: torch.Tensor) -> np.ndarray:
        """
        Make predictions for user-item pairs
        
        Args:
            user_ids: User IDs
            item_ids: Item IDs
            
        Returns:
            Predicted ratings as numpy array
        """
        self.eval()
        with torch.no_grad():
            predictions = self.forward(user_ids, item_ids)
            return predictions.cpu().numpy()
    
    def get_user_recommendations(self, user_id: int, item_ids: List[int], top_k: int = 10) -> List[Tuple[int, float]]:
        """
        Get top-k recommendations for a user
        
        Args:
            user_id: User ID
            item_ids: List of item IDs to score
            top_k: Number of recommendations to return
            
        Returns:
            List of (item_id, predicted_rating) tuples
        """
        self.eval()
        with torch.no_grad():
            user_tensor = torch.tensor([user_id] * len(item_ids), dtype=torch.long)
            item_tensor = torch.tensor(item_ids, dtype=torch.long)
            
            predictions = self.forward(user_tensor, item_tensor)
            predictions = predictions.cpu().numpy()
            
            # Sort by predicted rating and return top-k
            item_scores = list(zip(item_ids, predictions))
            item_scores.sort(key=lambda x: x[1], reverse=True)
            
            return item_scores[:top_k]


def load_model(model_path: str, metadata_path: str) -> Tuple[HybridRecommenderModel, Dict[str, Any]]:
    """
    Load a trained model and its metadata
    
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
    model = HybridRecommenderModel(
        num_users=metadata['num_users'],
        num_items=metadata['num_items'],
        embedding_dim=metadata.get('embedding_dim', 64),
        hidden_dim=metadata.get('hidden_dim', 128)
    )
    
    # Load model state
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    
    return model, metadata


def load_movie_data(movies_path: str) -> pd.DataFrame:
    """
    Load movie data from CSV file
    
    Args:
        movies_path: Path to movies CSV file
        
    Returns:
        DataFrame with movie data
    """
    return pd.read_csv(movies_path)


def load_mappings(mappings_path: str) -> Dict[str, Any]:
    """
    Load ID mappings
    
    Args:
        mappings_path: Path to mappings pickle file
        
    Returns:
        Dictionary with ID mappings
    """
    with open(mappings_path, 'rb') as f:
        return pickle.load(f)


def load_movie_lookup(lookup_path: str) -> Dict[str, Any]:
    """
    Load movie lookup dictionary
    
    Args:
        lookup_path: Path to movie lookup pickle file
        
    Returns:
        Movie lookup dictionary
    """
    with open(lookup_path, 'rb') as f:
        return pickle.load(f)