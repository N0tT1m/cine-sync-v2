import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import numpy as np


class TwoTowerModel(nn.Module):
    """
    Two-Tower/Dual-Encoder model for large-scale recommendation systems.
    Separate encoding towers for users and items with efficient dot-product similarity.
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
        
        # User tower
        self.user_tower = self._build_tower(
            user_features_dim, hidden_layers, embedding_dim, dropout, "user"
        )
        
        # Item tower
        self.item_tower = self._build_tower(
            item_features_dim, hidden_layers, embedding_dim, dropout, "item"
        )
        
        # Temperature parameter for similarity scaling
        self.temperature = nn.Parameter(torch.ones(1) * 0.1)
        
        self._init_weights()
    
    def _build_tower(self, input_dim: int, hidden_layers: List[int], 
                    output_dim: int, dropout: float, tower_name: str) -> nn.Module:
        """Build a tower (user or item encoder)"""
        layers = []
        
        # Input layer
        prev_dim = input_dim
        
        # Hidden layers
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            
            if self.use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        # Output layer (embedding)
        layers.append(nn.Linear(prev_dim, output_dim))
        layers.append(nn.LayerNorm(output_dim))  # Normalize final embeddings
        
        return nn.Sequential(*layers)
    
    def _init_weights(self):
        """Initialize model weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def encode_users(self, user_features: torch.Tensor) -> torch.Tensor:
        """Encode user features into embeddings"""
        user_embeddings = self.user_tower(user_features)
        # L2 normalize for cosine similarity
        return F.normalize(user_embeddings, p=2, dim=1)
    
    def encode_items(self, item_features: torch.Tensor) -> torch.Tensor:
        """Encode item features into embeddings"""
        item_embeddings = self.item_tower(item_features)
        # L2 normalize for cosine similarity
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
        
        # Compute dot product similarity
        similarities = torch.sum(user_embeddings * item_embeddings, dim=1)
        
        # Scale by temperature
        similarities = similarities / self.temperature
        
        return similarities
    
    def compute_similarity_matrix(self, user_features: torch.Tensor, 
                                 item_features: torch.Tensor) -> torch.Tensor:
        """
        Compute similarity matrix between all users and items.
        
        Args:
            user_features: Tensor of shape (num_users, user_features_dim)
            item_features: Tensor of shape (num_items, item_features_dim)
        
        Returns:
            Similarity matrix of shape (num_users, num_items)
        """
        user_embeddings = self.encode_users(user_features)
        item_embeddings = self.encode_items(item_features)
        
        # Compute all pairwise similarities
        similarities = torch.matmul(user_embeddings, item_embeddings.T)
        similarities = similarities / self.temperature
        
        return similarities


class EnhancedTwoTowerModel(nn.Module):
    """
    Enhanced Two-Tower model with additional features like attention and cross-tower interactions.
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
        
        # User embeddings for categorical features
        self.user_categorical_embeddings = nn.ModuleDict()
        user_cat_total_dim = 0
        
        for feature_name, vocab_size in user_categorical_dims.items():
            emb_dim = min(50, (vocab_size + 1) // 2)  # Adaptive embedding size
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
        
        # Cross-attention mechanism
        if use_cross_attention:
            self.user_attention = nn.MultiheadAttention(embedding_dim, 8, batch_first=True)
            self.item_attention = nn.MultiheadAttention(embedding_dim, 8, batch_first=True)
        
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
        
        # Apply cross-attention if enabled
        if self.use_cross_attention:
            # User attention over items
            user_emb_attended, _ = self.user_attention(
                user_embeddings.unsqueeze(1), 
                item_embeddings.unsqueeze(1),
                item_embeddings.unsqueeze(1)
            )
            user_embeddings = user_emb_attended.squeeze(1)
            
            # Item attention over users
            item_emb_attended, _ = self.item_attention(
                item_embeddings.unsqueeze(1),
                user_embeddings.unsqueeze(1), 
                user_embeddings.unsqueeze(1)
            )
            item_embeddings = item_emb_attended.squeeze(1)
        
        # Compute similarities
        similarities = torch.sum(user_embeddings * item_embeddings, dim=1)
        similarities = similarities / self.temperature
        
        return similarities


class MultiTaskTwoTowerModel(nn.Module):
    """
    Multi-task Two-Tower model that can predict ratings, clicks, and other objectives jointly.
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
        
        # Task-specific heads
        self.task_networks = nn.ModuleDict()
        
        for task_name, output_dim in self.task_heads.items():
            self.task_networks[task_name] = nn.Sequential(
                nn.Linear(embedding_dim * 2, embedding_dim),  # Combined user + item embedding
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(embedding_dim, output_dim)
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
        
        # Task-specific predictions
        predictions = {'similarity': similarity}
        
        # Combined embedding for task heads
        combined_embedding = torch.cat([user_embeddings, item_embeddings], dim=1)
        
        tasks_to_compute = tasks or list(self.task_heads.keys())
        
        for task_name in tasks_to_compute:
            if task_name in self.task_networks:
                task_output = self.task_networks[task_name](combined_embedding)
                
                if task_name == 'rating':
                    # Rating prediction (0-5 scale)
                    predictions[task_name] = torch.sigmoid(task_output) * 5.0
                elif task_name == 'click':
                    # Click prediction (0-1 probability)
                    predictions[task_name] = torch.sigmoid(task_output)
                else:
                    predictions[task_name] = task_output
        
        return predictions


class CollaborativeTwoTowerModel(nn.Module):
    """
    Two-Tower model that incorporates collaborative signals through user/item embeddings.
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
        
        # Collaborative embeddings
        self.user_collaborative_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_collaborative_embedding = nn.Embedding(num_items, embedding_dim)
        
        # Content towers
        self.user_content_tower = self._build_tower(
            user_features_dim, hidden_layers, embedding_dim, dropout
        )
        self.item_content_tower = self._build_tower(
            item_features_dim, hidden_layers, embedding_dim, dropout
        )
        
        # Fusion layers
        self.user_fusion = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim, embedding_dim),
            nn.LayerNorm(embedding_dim)
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
        """Encode users using both collaborative and content signals"""
        # Collaborative embedding
        user_collab_emb = self.user_collaborative_embedding(user_ids)
        
        # Content embedding
        user_content_emb = self.user_content_tower(user_features)
        
        # Fuse collaborative and content signals
        combined = torch.cat([user_collab_emb, user_content_emb], dim=1)
        user_embedding = self.user_fusion(combined)
        
        return F.normalize(user_embedding, p=2, dim=1)
    
    def encode_items(self, item_ids: torch.Tensor, item_features: torch.Tensor) -> torch.Tensor:
        """Encode items using both collaborative and content signals"""
        # Collaborative embedding
        item_collab_emb = self.item_collaborative_embedding(item_ids)
        
        # Content embedding  
        item_content_emb = self.item_content_tower(item_features)
        
        # Fuse collaborative and content signals
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