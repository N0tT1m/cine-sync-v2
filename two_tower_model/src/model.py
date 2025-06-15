# Two-Tower/Dual-Encoder Model Architectures for CineSync v2
# Implements various two-tower models for large-scale recommendation systems
# Based on the dual-encoder paradigm for efficient candidate retrieval and ranking

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import numpy as np


class TwoTowerModel(nn.Module):
    """
    Standard Two-Tower/Dual-Encoder model for large-scale recommendation systems.
    
    The two-tower architecture separates user and item encoding into independent towers,
    enabling efficient retrieval through dot-product similarity computation. This design
    allows for pre-computing item embeddings and using approximate nearest neighbor search
    for real-time candidate generation at scale.
    
    Key benefits:
    - Scalable to millions of users/items
    - Enables efficient approximate nearest neighbor search
    - Allows for separate optimization of user and item representations
    - Supports both content-based and collaborative signals
    """
    
    def __init__(self, user_features_dim: int, item_features_dim: int, 
                 embedding_dim: int = 128, hidden_layers: List[int] = [256, 128],
                 dropout: float = 0.2, use_batch_norm: bool = True):
        """
        Args:
            user_features_dim: Dimension of user feature vector
            item_features_dim: Dimension of item feature vector
            embedding_dim: Final embedding dimension for both towers
            hidden_layers: List of hidden layer dimensions
            dropout: Dropout rate
            use_batch_norm: Whether to use batch normalization
        """
        super(TwoTowerModel, self).__init__()
        
        self.embedding_dim = embedding_dim
        self.use_batch_norm = use_batch_norm
        
        # User tower: encodes user features into fixed-size embeddings
        # Independent from item tower to enable efficient candidate retrieval
        self.user_tower = self._build_tower(
            user_features_dim, hidden_layers, embedding_dim, dropout, "user"
        )
        
        # Item tower: encodes item features into fixed-size embeddings
        # Can be pre-computed for all items and cached for fast retrieval
        self.item_tower = self._build_tower(
            item_features_dim, hidden_layers, embedding_dim, dropout, "item"
        )
        
        # Temperature parameter for similarity scaling (learnable)
        # Controls the sharpness of the similarity distribution
        # Lower temperature = sharper predictions, higher temperature = smoother
        self.temperature = nn.Parameter(torch.ones(1) * 0.1)
        
        self._init_weights()
    
    def _build_tower(self, input_dim: int, hidden_layers: List[int], 
                    output_dim: int, dropout: float, tower_name: str) -> nn.Module:
        """Build a deep neural network tower (user or item encoder)
        
        Creates a multi-layer perceptron with batch normalization, ReLU activation,
        and dropout for regularization. The final layer is L2-normalized to enable
        cosine similarity computation through dot products.
        """
        layers = []
        
        # Start with input dimension
        prev_dim = input_dim
        
        # Build hidden layers with standard deep learning components
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(prev_dim, hidden_dim))  # Linear transformation
            
            if self.use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))   # Normalize activations
            
            layers.append(nn.ReLU())                        # Non-linear activation
            layers.append(nn.Dropout(dropout))             # Regularization
            prev_dim = hidden_dim
        
        # Final embedding layer with normalization
        layers.append(nn.Linear(prev_dim, output_dim))     # Project to embedding space
        layers.append(nn.LayerNorm(output_dim))            # Normalize final embeddings for stable training
        
        return nn.Sequential(*layers)
    
    def _init_weights(self):
        """Initialize model weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def encode_users(self, user_features: torch.Tensor) -> torch.Tensor:
        """Encode user features into normalized embeddings
        
        Args:
            user_features: Raw user feature vectors
            
        Returns:
            L2-normalized user embeddings for cosine similarity computation
        """
        user_embeddings = self.user_tower(user_features)
        # L2 normalize enables cosine similarity via dot product
        # ||u|| = ||v|| = 1, so u·v = cos(θ) where θ is the angle between vectors
        return F.normalize(user_embeddings, p=2, dim=1)
    
    def encode_items(self, item_features: torch.Tensor) -> torch.Tensor:
        """Encode item features into normalized embeddings
        
        Args:
            item_features: Raw item feature vectors
            
        Returns:
            L2-normalized item embeddings for cosine similarity computation
        """
        item_embeddings = self.item_tower(item_features)
        # L2 normalize enables cosine similarity via dot product
        # These embeddings can be pre-computed and cached for all items
        return F.normalize(item_embeddings, p=2, dim=1)
    
    def forward(self, user_features: torch.Tensor, item_features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass computing user-item similarity scores.
        
        Args:
            user_features: Tensor of shape (batch_size, user_features_dim)
            item_features: Tensor of shape (batch_size, item_features_dim)
        
        Returns:
            Similarity scores of shape (batch_size,)
        """
        user_embeddings = self.encode_users(user_features)
        item_embeddings = self.encode_items(item_features)
        
        # Compute dot product similarity (cosine similarity for normalized vectors)
        # For normalized vectors: u·v = ||u|| ||v|| cos(θ) = cos(θ)
        similarities = torch.sum(user_embeddings * item_embeddings, dim=1)
        
        # Scale by learnable temperature parameter
        # Lower temperature sharpens the distribution, higher temperature smooths it
        similarities = similarities / self.temperature
        
        return similarities
    
    def compute_similarity_matrix(self, user_features: torch.Tensor, 
                                 item_features: torch.Tensor) -> torch.Tensor:
        """
        Compute full similarity matrix between all users and items.
        
        Useful for batch evaluation and analysis, but not typically used in production
        due to memory constraints. For large-scale systems, use approximate nearest
        neighbor search with pre-computed item embeddings instead.
        
        Args:
            user_features: Tensor of shape (num_users, user_features_dim)
            item_features: Tensor of shape (num_items, item_features_dim)
        
        Returns:
            Similarity matrix of shape (num_users, num_items)
        """
        user_embeddings = self.encode_users(user_features)
        item_embeddings = self.encode_items(item_features)
        
        # Compute all pairwise similarities via matrix multiplication
        # For normalized embeddings, this gives cosine similarities
        similarities = torch.matmul(user_embeddings, item_embeddings.T)  # (num_users, num_items)
        similarities = similarities / self.temperature  # Apply temperature scaling
        
        return similarities


class EnhancedTwoTowerModel(nn.Module):
    """
    Enhanced Two-Tower model with sophisticated feature handling and cross-tower interactions.
    
    Extends the basic two-tower architecture with:
    - Separate handling of categorical and numerical features
    - Adaptive embedding sizes for categorical features
    - Optional cross-attention mechanism between towers
    - More sophisticated feature fusion strategies
    
    This model sacrifices some of the independence of the basic two-tower model
    but gains expressiveness through cross-tower interactions.
    """
    
    def __init__(self, user_categorical_dims: Dict[str, int], user_numerical_dim: int,
                 item_categorical_dims: Dict[str, int], item_numerical_dim: int,
                 embedding_dim: int = 128, hidden_layers: List[int] = [512, 256, 128],
                 dropout: float = 0.2, use_cross_attention: bool = True):
        """
        Args:
            user_categorical_dims: Dict of categorical feature names to vocab sizes
            user_numerical_dim: Number of numerical user features
            item_categorical_dims: Dict of categorical feature names to vocab sizes
            item_numerical_dim: Number of numerical item features
            embedding_dim: Final embedding dimension
            hidden_layers: Hidden layer dimensions
            dropout: Dropout rate
            use_cross_attention: Whether to use cross-tower attention
        """
        super(EnhancedTwoTowerModel, self).__init__()
        
        self.embedding_dim = embedding_dim
        self.use_cross_attention = use_cross_attention
        
        # User embeddings for categorical features with adaptive sizing
        self.user_categorical_embeddings = nn.ModuleDict()
        user_cat_total_dim = 0
        
        for feature_name, vocab_size in user_categorical_dims.items():
            # Adaptive embedding size based on vocabulary size
            # Heuristic: sqrt(vocab_size) but capped at 50 dimensions
            emb_dim = min(50, (vocab_size + 1) // 2)
            self.user_categorical_embeddings[feature_name] = nn.Embedding(vocab_size, emb_dim)
            user_cat_total_dim += emb_dim
        
        # Item embeddings for categorical features
        self.item_categorical_embeddings = nn.ModuleDict()
        item_cat_total_dim = 0
        
        for feature_name, vocab_size in item_categorical_dims.items():
            emb_dim = min(50, (vocab_size + 1) // 2)
            self.item_categorical_embeddings[feature_name] = nn.Embedding(vocab_size, emb_dim)
            item_cat_total_dim += emb_dim
        
        # Feature fusion layers
        user_total_dim = user_cat_total_dim + user_numerical_dim
        item_total_dim = item_cat_total_dim + item_numerical_dim
        
        # User tower
        self.user_tower = self._build_tower(user_total_dim, hidden_layers, embedding_dim, dropout)
        
        # Item tower
        self.item_tower = self._build_tower(item_total_dim, hidden_layers, embedding_dim, dropout)
        
        # Cross-attention mechanism for tower interactions
        # Allows user and item towers to attend to each other's representations
        # This breaks the independence but can improve accuracy
        if use_cross_attention:
            self.user_attention = nn.MultiheadAttention(embedding_dim, 8, batch_first=True)  # User attends to items
            self.item_attention = nn.MultiheadAttention(embedding_dim, 8, batch_first=True)  # Item attends to users
        
        # Temperature scaling
        self.temperature = nn.Parameter(torch.ones(1) * 0.1)
        
        self._init_weights()
    
    def _build_tower(self, input_dim: int, hidden_layers: List[int], 
                    output_dim: int, dropout: float) -> nn.Module:
        """Build tower with residual connections"""
        layers = []
        prev_dim = input_dim
        
        for i, hidden_dim in enumerate(hidden_layers):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        # Final projection
        layers.append(nn.Linear(prev_dim, output_dim))
        layers.append(nn.LayerNorm(output_dim))
        
        return nn.Sequential(*layers)
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Embedding):
                nn.init.xavier_normal_(module.weight)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def _process_user_features(self, user_categorical: Dict[str, torch.Tensor], 
                              user_numerical: torch.Tensor) -> torch.Tensor:
        """Process and combine user features"""
        categorical_embeddings = []
        
        for feature_name, feature_values in user_categorical.items():
            if feature_name in self.user_categorical_embeddings:
                emb = self.user_categorical_embeddings[feature_name](feature_values)
                categorical_embeddings.append(emb)
        
        # Combine categorical and numerical features
        if categorical_embeddings:
            cat_features = torch.cat(categorical_embeddings, dim=1)
            if user_numerical.size(1) > 0:
                user_features = torch.cat([cat_features, user_numerical], dim=1)
            else:
                user_features = cat_features
        else:
            user_features = user_numerical
        
        return user_features
    
    def _process_item_features(self, item_categorical: Dict[str, torch.Tensor],
                              item_numerical: torch.Tensor) -> torch.Tensor:
        """Process and combine item features"""
        categorical_embeddings = []
        
        for feature_name, feature_values in item_categorical.items():
            if feature_name in self.item_categorical_embeddings:
                emb = self.item_categorical_embeddings[feature_name](feature_values)
                categorical_embeddings.append(emb)
        
        # Combine categorical and numerical features
        if categorical_embeddings:
            cat_features = torch.cat(categorical_embeddings, dim=1)
            if item_numerical.size(1) > 0:
                item_features = torch.cat([cat_features, item_numerical], dim=1)
            else:
                item_features = cat_features
        else:
            item_features = item_numerical
        
        return item_features
    
    def encode_users(self, user_categorical: Dict[str, torch.Tensor], 
                    user_numerical: torch.Tensor) -> torch.Tensor:
        """Encode user features"""
        user_features = self._process_user_features(user_categorical, user_numerical)
        user_embeddings = self.user_tower(user_features)
        return F.normalize(user_embeddings, p=2, dim=1)
    
    def encode_items(self, item_categorical: Dict[str, torch.Tensor],
                    item_numerical: torch.Tensor) -> torch.Tensor:
        """Encode item features"""
        item_features = self._process_item_features(item_categorical, item_numerical)
        item_embeddings = self.item_tower(item_features)
        return F.normalize(item_embeddings, p=2, dim=1)
    
    def forward(self, user_categorical: Dict[str, torch.Tensor], user_numerical: torch.Tensor,
                item_categorical: Dict[str, torch.Tensor], item_numerical: torch.Tensor) -> torch.Tensor:
        """Forward pass with optional cross-attention"""
        user_embeddings = self.encode_users(user_categorical, user_numerical)
        item_embeddings = self.encode_items(item_categorical, item_numerical)
        
        # Apply cross-attention if enabled (breaks tower independence)
        if self.use_cross_attention:
            # User tower attends to item representations
            # Query: user, Key/Value: item - "what items should this user focus on?"
            user_emb_attended, _ = self.user_attention(
                user_embeddings.unsqueeze(1),    # Query: user embeddings
                item_embeddings.unsqueeze(1),    # Key: item embeddings  
                item_embeddings.unsqueeze(1)     # Value: item embeddings
            )
            user_embeddings = user_emb_attended.squeeze(1)
            
            # Item tower attends to user representations
            # Query: item, Key/Value: user - "what users should this item focus on?"
            item_emb_attended, _ = self.item_attention(
                item_embeddings.unsqueeze(1),    # Query: item embeddings
                user_embeddings.unsqueeze(1),    # Key: user embeddings
                user_embeddings.unsqueeze(1)     # Value: user embeddings
            )
            item_embeddings = item_emb_attended.squeeze(1)
        
        # Compute similarities
        similarities = torch.sum(user_embeddings * item_embeddings, dim=1)
        similarities = similarities / self.temperature
        
        return similarities


class MultiTaskTwoTowerModel(nn.Module):
    """
    Multi-task Two-Tower model for joint optimization of multiple objectives.
    
    Extends the two-tower architecture to predict multiple targets simultaneously:
    - Rating prediction (regression)
    - Click prediction (binary classification)
    - Custom objectives through task-specific heads
    
    This approach leverages shared representations while allowing task-specific
    optimization, often leading to better generalization than single-task models.
    """
    
    def __init__(self, user_features_dim: int, item_features_dim: int,
                 embedding_dim: int = 128, hidden_layers: List[int] = [256, 128],
                 task_heads: Dict[str, int] = None, dropout: float = 0.2):
        """
        Args:
            user_features_dim: User feature dimension
            item_features_dim: Item feature dimension  
            embedding_dim: Embedding dimension
            hidden_layers: Hidden layer sizes
            task_heads: Dict mapping task names to output dimensions
            dropout: Dropout rate
        """
        super(MultiTaskTwoTowerModel, self).__init__()
        
        self.embedding_dim = embedding_dim
        self.task_heads = task_heads or {'rating': 1, 'click': 1}
        
        # Shared towers
        self.user_tower = self._build_tower(user_features_dim, hidden_layers, embedding_dim, dropout)
        self.item_tower = self._build_tower(item_features_dim, hidden_layers, embedding_dim, dropout)
        
        # Task-specific prediction heads
        # Each task gets its own network for specialized predictions
        self.task_networks = nn.ModuleDict()
        
        for task_name, output_dim in self.task_heads.items():
            # Task head operates on concatenated user + item embeddings
            self.task_networks[task_name] = nn.Sequential(
                nn.Linear(embedding_dim * 2, embedding_dim),  # Fusion layer
                nn.ReLU(),                                    # Non-linearity
                nn.Dropout(dropout),                         # Regularization
                nn.Linear(embedding_dim, output_dim)         # Task-specific output
            )
        
        # Shared similarity computation
        self.temperature = nn.Parameter(torch.ones(1) * 0.1)
        
        self._init_weights()
    
    def _build_tower(self, input_dim: int, hidden_layers: List[int],
                    output_dim: int, dropout: float) -> nn.Module:
        """Build encoding tower"""
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        layers.append(nn.LayerNorm(output_dim))
        
        return nn.Sequential(*layers)
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def encode_users(self, user_features: torch.Tensor) -> torch.Tensor:
        """Encode user features"""
        return F.normalize(self.user_tower(user_features), p=2, dim=1)
    
    def encode_items(self, item_features: torch.Tensor) -> torch.Tensor:
        """Encode item features"""
        return F.normalize(self.item_tower(item_features), p=2, dim=1)
    
    def forward(self, user_features: torch.Tensor, item_features: torch.Tensor,
                tasks: Optional[List[str]] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass for multi-task prediction.
        
        Returns:
            Dict mapping task names to predictions
        """
        user_embeddings = self.encode_users(user_features)
        item_embeddings = self.encode_items(item_features)
        
        # Compute base similarity
        similarity = torch.sum(user_embeddings * item_embeddings, dim=1) / self.temperature
        
        # Multi-task predictions from shared representations
        predictions = {'similarity': similarity}  # Base similarity always computed
        
        # Combine user and item embeddings for task-specific heads
        combined_embedding = torch.cat([user_embeddings, item_embeddings], dim=1)
        
        # Compute predictions for requested tasks
        tasks_to_compute = tasks or list(self.task_heads.keys())
        
        for task_name in tasks_to_compute:
            if task_name in self.task_networks:
                task_output = self.task_networks[task_name](combined_embedding)
                
                # Apply task-specific output transformations
                if task_name == 'rating':
                    # Rating prediction: sigmoid scaled to 0-5 range
                    predictions[task_name] = torch.sigmoid(task_output) * 5.0
                elif task_name == 'click':
                    # Click prediction: sigmoid for probability
                    predictions[task_name] = torch.sigmoid(task_output)
                else:
                    # Custom tasks: raw output
                    predictions[task_name] = task_output
        
        return predictions


class CollaborativeTwoTowerModel(nn.Module):
    """
    Hybrid Two-Tower model combining collaborative filtering and content-based features.
    
    Integrates both collaborative signals (learned user/item embeddings) and content-based
    features (metadata, demographics) into unified user and item representations.
    This approach gets the best of both worlds:
    - Collaborative signals capture implicit user preferences and item similarities
    - Content features provide cold-start capability and interpretability
    """
    
    def __init__(self, num_users: int, num_items: int, user_features_dim: int, 
                 item_features_dim: int, embedding_dim: int = 64, 
                 hidden_layers: List[int] = [256, 128], dropout: float = 0.2):
        """
        Args:
            num_users: Number of unique users
            num_items: Number of unique items
            user_features_dim: Dimension of user content features
            item_features_dim: Dimension of item content features
            embedding_dim: Embedding dimension for collaborative embeddings
            hidden_layers: Hidden layer sizes
            dropout: Dropout rate
        """
        super(CollaborativeTwoTowerModel, self).__init__()
        
        self.embedding_dim = embedding_dim
        
        # Collaborative embeddings (learned from interaction data)
        # These capture latent user preferences and item characteristics
        self.user_collaborative_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_collaborative_embedding = nn.Embedding(num_items, embedding_dim)
        
        # Content-based towers (process metadata features)
        # These provide cold-start capability and feature interpretability
        self.user_content_tower = self._build_tower(
            user_features_dim, hidden_layers, embedding_dim, dropout
        )
        self.item_content_tower = self._build_tower(
            item_features_dim, hidden_layers, embedding_dim, dropout
        )
        
        # Fusion layers to combine collaborative and content signals
        # Learns optimal weighting between interaction patterns and content features
        self.user_fusion = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),  # Combine collab + content
            nn.ReLU(),                                    # Non-linear combination
            nn.Dropout(dropout),                         # Regularization
            nn.Linear(embedding_dim, embedding_dim),     # Final projection
            nn.LayerNorm(embedding_dim)                  # Stabilize training
        )
        
        self.item_fusion = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim, embedding_dim),
            nn.LayerNorm(embedding_dim)
        )
        
        self.temperature = nn.Parameter(torch.ones(1) * 0.1)
        
        self._init_weights()
    
    def _build_tower(self, input_dim: int, hidden_layers: List[int],
                    output_dim: int, dropout: float) -> nn.Module:
        """Build content feature tower"""
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        
        return nn.Sequential(*layers)
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Embedding):
                nn.init.xavier_normal_(module.weight)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def encode_users(self, user_ids: torch.Tensor, user_features: torch.Tensor) -> torch.Tensor:
        """Encode users using both collaborative and content signals
        
        Args:
            user_ids: User IDs for collaborative embeddings
            user_features: Content-based user features
            
        Returns:
            Fused user embeddings combining both signal types
        """
        # Collaborative embedding: learned from user interaction patterns
        user_collab_emb = self.user_collaborative_embedding(user_ids)
        
        # Content embedding: derived from user demographics/metadata
        user_content_emb = self.user_content_tower(user_features)
        
        # Fuse both signals through learned combination
        combined = torch.cat([user_collab_emb, user_content_emb], dim=1)
        user_embedding = self.user_fusion(combined)
        
        return F.normalize(user_embedding, p=2, dim=1)
    
    def encode_items(self, item_ids: torch.Tensor, item_features: torch.Tensor) -> torch.Tensor:
        """Encode items using both collaborative and content signals
        
        Args:
            item_ids: Item IDs for collaborative embeddings
            item_features: Content-based item features (genres, metadata)
            
        Returns:
            Fused item embeddings combining both signal types
        """
        # Collaborative embedding: learned from item interaction patterns
        item_collab_emb = self.item_collaborative_embedding(item_ids)
        
        # Content embedding: derived from item metadata (genres, descriptions, etc.)
        item_content_emb = self.item_content_tower(item_features)
        
        # Fuse both signals through learned combination
        combined = torch.cat([item_collab_emb, item_content_emb], dim=1)
        item_embedding = self.item_fusion(combined)
        
        return F.normalize(item_embedding, p=2, dim=1)
    
    def forward(self, user_ids: torch.Tensor, item_ids: torch.Tensor,
                user_features: torch.Tensor, item_features: torch.Tensor) -> torch.Tensor:
        """Forward pass combining collaborative and content signals"""
        user_embeddings = self.encode_users(user_ids, user_features)
        item_embeddings = self.encode_items(item_ids, item_features)
        
        similarities = torch.sum(user_embeddings * item_embeddings, dim=1)
        similarities = similarities / self.temperature
        
        return similarities