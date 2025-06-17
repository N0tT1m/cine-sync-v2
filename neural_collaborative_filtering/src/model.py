# Neural Collaborative Filtering Models for CineSync v2
# Implements various NCF architectures including standard NCF, simple variants, and enhanced versions
# Based on "Neural collaborative filtering" paper by He et al. (WWW 2017)

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Dict


class NeuralCollaborativeFiltering(nn.Module):
    """
    Neural Collaborative Filtering model combining Generalized Matrix Factorization (GMF)
    and Multi-Layer Perceptron (MLP) for enhanced recommendation performance.
    
    Paper: He, X., et al. "Neural collaborative filtering." WWW 2017.
    """
    
    def __init__(self, num_users, num_items, embedding_dim=64, hidden_layers=[128, 64], 
                 dropout=0.2, alpha=0.5):
        """Initialize Neural Collaborative Filtering model
        
        Args:
            num_users: Number of unique users in the dataset
            num_items: Number of unique items in the dataset
            embedding_dim: Dimension of embedding vectors
            hidden_layers: List of hidden layer sizes for MLP component
            dropout: Dropout rate for regularization
            alpha: Weight for combining GMF and MLP outputs (not used in this implementation)
        """
        super(NeuralCollaborativeFiltering, self).__init__()
        
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.alpha = alpha  # Weight for combining GMF and MLP
        
        # GMF (Generalized Matrix Factorization) embeddings
        # Separate embeddings for GMF path to learn different representations
        self.gmf_user_embedding = nn.Embedding(num_users, embedding_dim)
        self.gmf_item_embedding = nn.Embedding(num_items, embedding_dim)
        
        # MLP embeddings (separate from GMF to allow different representations)
        # This allows the model to learn different user/item representations for different purposes
        self.mlp_user_embedding = nn.Embedding(num_users, embedding_dim)
        self.mlp_item_embedding = nn.Embedding(num_items, embedding_dim)
        
        # MLP layers for learning complex user-item interactions
        mlp_layers = []
        input_size = embedding_dim * 2  # Concatenated user + item embeddings
        
        # Build MLP with multiple hidden layers, ReLU activation, and dropout
        for hidden_size in hidden_layers:
            mlp_layers.append(nn.Linear(input_size, hidden_size))  # Linear transformation
            mlp_layers.append(nn.ReLU())                           # Non-linear activation
            mlp_layers.append(nn.Dropout(dropout))                 # Regularization
            input_size = hidden_size
        
        self.mlp = nn.Sequential(*mlp_layers)
        
        # Final prediction layer combines GMF and MLP outputs
        # GMF output: embedding_dim, MLP output: last hidden layer size
        final_input_size = embedding_dim + hidden_layers[-1]
        self.prediction = nn.Linear(final_input_size, 1)  # Single output for rating prediction
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights using Xavier initialization for stable training"""
        for module in self.modules():
            if isinstance(module, nn.Embedding):
                nn.init.xavier_normal_(module.weight)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, user_ids, item_ids):
        """Forward pass through both GMF and MLP paths
        
        Args:
            user_ids: Tensor of user IDs [batch_size]
            item_ids: Tensor of item IDs [batch_size]
            
        Returns:
            Predicted ratings scaled to 0-5 range
        """
        # GMF path: learns linear user-item interactions through element-wise product
        gmf_user_emb = self.gmf_user_embedding(user_ids)
        gmf_item_emb = self.gmf_item_embedding(item_ids)
        gmf_output = gmf_user_emb * gmf_item_emb  # Element-wise product (like matrix factorization)
        
        # MLP path: learns non-linear user-item interactions through deep network
        mlp_user_emb = self.mlp_user_embedding(user_ids)
        mlp_item_emb = self.mlp_item_embedding(item_ids)
        mlp_input = torch.cat([mlp_user_emb, mlp_item_emb], dim=-1)  # Concatenate embeddings
        mlp_output = self.mlp(mlp_input)  # Pass through MLP layers
        
        # Combine GMF and MLP outputs for final prediction
        combined = torch.cat([gmf_output, mlp_output], dim=-1)  # Concatenate both paths
        prediction = self.prediction(combined)                   # Final linear layer
        
        # Apply sigmoid activation and scale to 0-5 rating range
        return torch.sigmoid(prediction).squeeze() * 5.0


class SimpleNCF(nn.Module):
    """
    Simplified Neural Collaborative Filtering model for faster training and inference.
    
    A streamlined version of NCF with single embeddings and a simpler MLP architecture.
    Suitable for rapid prototyping and when computational resources are limited.
    """
    
    def __init__(self, num_users, num_items, embedding_dim=32, hidden_dim=64, dropout=0.2):
        """Initialize Simple NCF model
        
        Args:
            num_users: Number of unique users
            num_items: Number of unique items
            embedding_dim: Dimension of embedding vectors (smaller for efficiency)
            hidden_dim: Hidden layer dimension
            dropout: Dropout rate for regularization
        """
        super(SimpleNCF, self).__init__()
        
        # Single set of embeddings (unlike full NCF which has separate GMF/MLP embeddings)
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        
        # Simple 3-layer MLP for user-item interaction modeling
        self.fc1 = nn.Linear(embedding_dim * 2, hidden_dim)      # Input layer
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)        # Hidden layer
        self.fc3 = nn.Linear(hidden_dim // 2, 1)                # Output layer
        
        self.dropout = nn.Dropout(dropout)
        
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Embedding):
                nn.init.xavier_normal_(module.weight)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, user_ids, item_ids):
        """Simple forward pass through concatenation and MLP
        
        Args:
            user_ids: Tensor of user IDs
            item_ids: Tensor of item IDs
            
        Returns:
            Predicted ratings scaled to 0-5 range
        """
        # Get embeddings and concatenate
        user_emb = self.user_embedding(user_ids)
        item_emb = self.item_embedding(item_ids)
        
        # Pass through simple MLP with ReLU activations and dropout
        x = torch.cat([user_emb, item_emb], dim=-1)  # Concatenate embeddings
        x = F.relu(self.fc1(x))                      # First hidden layer
        x = self.dropout(x)                          # Regularization
        x = F.relu(self.fc2(x))                      # Second hidden layer
        x = self.dropout(x)                          # Regularization
        x = self.fc3(x)                              # Output layer
        
        # Scale to 0-5 rating range
        return torch.sigmoid(x).squeeze() * 5.0


class CrossAttentionNCF(nn.Module):
    """
    Modern NCF with cross-attention and contrastive learning
    
    Advanced features:
    - Multi-head cross-attention between user/item embeddings
    - Feature-wise interaction networks
    - Contrastive learning with InfoNCE loss
    - Dynamic embedding dimensions
    - Mixture of experts for different user types
    """
    
    def __init__(self, num_users, num_items, num_genres=20, embedding_dim=256, 
                 hidden_layers=[512, 256, 128], dropout=0.2, num_heads=8, 
                 use_moe=True, num_experts=4, temperature=0.07):
        super(CrossAttentionNCF, self).__init__()
        
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.temperature = temperature
        self.use_moe = use_moe
        
        # Enhanced embeddings with learnable scaling
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        self.genre_embedding = nn.Embedding(num_genres, embedding_dim // 4)
        
        # Embedding scaling factors
        self.user_scale = nn.Parameter(torch.ones(1))
        self.item_scale = nn.Parameter(torch.ones(1))
        
        # Cross-attention mechanism
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Feature-wise interaction network
        self.feature_interaction = FeatureWiseInteraction(embedding_dim, dropout)
        
        # Mixture of Experts for user types
        if use_moe:
            self.moe_gate = nn.Linear(embedding_dim, num_experts)
            self.experts = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(embedding_dim * 2, hidden_layers[0]),
                    nn.LayerNorm(hidden_layers[0]),
                    nn.ReLU(),
                    nn.Dropout(dropout)
                ) for _ in range(num_experts)
            ])
        
        # Enhanced MLP with residual connections
        self.mlp_layers = nn.ModuleList()
        input_size = embedding_dim * 2
        
        for i, hidden_size in enumerate(hidden_layers):
            self.mlp_layers.append(nn.Sequential(
                nn.Linear(input_size, hidden_size),
                nn.LayerNorm(hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout)
            ))
            input_size = hidden_size
        
        # Multi-task prediction heads
        self.rating_head = nn.Linear(hidden_layers[-1], 1)
        self.genre_head = nn.Linear(hidden_layers[-1], num_genres)
        self.popularity_head = nn.Linear(hidden_layers[-1], 1)
        
        # Contrastive projection head
        self.projection_head = nn.Sequential(
            nn.Linear(hidden_layers[-1], embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim // 2)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Embedding):
                nn.init.xavier_normal_(module.weight)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, user_ids, item_ids, genre_ids=None, return_embeddings=False):
        batch_size = user_ids.size(0)
        
        # Get scaled embeddings
        user_emb = self.user_embedding(user_ids) * self.user_scale
        item_emb = self.item_embedding(item_ids) * self.item_scale
        
        # Cross-attention between user and item embeddings
        user_attended, _ = self.cross_attention(
            user_emb.unsqueeze(1), item_emb.unsqueeze(1), item_emb.unsqueeze(1)
        )
        item_attended, _ = self.cross_attention(
            item_emb.unsqueeze(1), user_emb.unsqueeze(1), user_emb.unsqueeze(1)
        )
        
        user_final = user_attended.squeeze(1) + user_emb
        item_final = item_attended.squeeze(1) + item_emb
        
        # Feature-wise interaction
        interaction_features = self.feature_interaction(user_final, item_final)
        
        # Concatenate features
        combined = torch.cat([user_final, item_final], dim=1)
        
        # Mixture of Experts (if enabled)
        if self.use_moe:
            gate_weights = F.softmax(self.moe_gate(user_final), dim=-1)
            expert_outputs = []
            
            for expert in self.experts:
                expert_out = expert(combined)
                expert_outputs.append(expert_out)
            
            expert_stack = torch.stack(expert_outputs, dim=2)  # [batch, hidden, num_experts]
            moe_output = torch.sum(expert_stack * gate_weights.unsqueeze(1), dim=2)
            x = moe_output
        else:
            x = combined
        
        # Enhanced MLP with residual connections
        residual = None
        for i, layer in enumerate(self.mlp_layers):
            x_new = layer(x)
            if residual is not None and x_new.size(-1) == residual.size(-1):
                x_new = x_new + residual
            residual = x_new
            x = x_new
        
        # Multi-task predictions
        rating_pred = torch.sigmoid(self.rating_head(x)).squeeze() * 5.0
        genre_pred = self.genre_head(x)
        popularity_pred = torch.sigmoid(self.popularity_head(x)).squeeze()
        
        outputs = {
            'rating': rating_pred,
            'genre_logits': genre_pred,
            'popularity': popularity_pred
        }
        
        if return_embeddings:
            projection = F.normalize(self.projection_head(x), dim=-1)
            outputs['embeddings'] = projection
            outputs['user_emb'] = user_final
            outputs['item_emb'] = item_final
        
        return outputs
    
    def compute_contrastive_loss(self, user_ids, pos_item_ids, neg_item_ids):
        """Compute InfoNCE contrastive loss"""
        # Get embeddings for positive pairs
        pos_outputs = self.forward(user_ids, pos_item_ids, return_embeddings=True)
        pos_embeddings = pos_outputs['embeddings']
        
        # Get embeddings for negative pairs
        neg_outputs = self.forward(user_ids, neg_item_ids, return_embeddings=True)
        neg_embeddings = neg_outputs['embeddings']
        
        # Compute similarity scores
        pos_sim = torch.sum(pos_embeddings * pos_embeddings, dim=1) / self.temperature
        neg_sim = torch.sum(pos_embeddings * neg_embeddings, dim=1) / self.temperature
        
        # InfoNCE loss
        logits = torch.cat([pos_sim.unsqueeze(1), neg_sim.unsqueeze(1)], dim=1)
        labels = torch.zeros(user_ids.size(0), dtype=torch.long, device=user_ids.device)
        
        return F.cross_entropy(logits, labels)


class FeatureWiseInteraction(nn.Module):
    """Feature-wise interaction network for learning cross-features"""
    
    def __init__(self, embedding_dim, dropout=0.1):
        super(FeatureWiseInteraction, self).__init__()
        
        self.embedding_dim = embedding_dim
        
        # Learnable interaction weights
        self.interaction_weights = nn.Parameter(torch.randn(embedding_dim, embedding_dim))
        self.bias = nn.Parameter(torch.zeros(embedding_dim))
        
        # Attention for feature importance
        self.feature_attention = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.Tanh(),
            nn.Linear(embedding_dim, embedding_dim),
            nn.Sigmoid()
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, user_emb, item_emb):
        # Element-wise product
        element_wise = user_emb * item_emb
        
        # Bilinear interaction
        bilinear = torch.matmul(user_emb.unsqueeze(1), self.interaction_weights)
        bilinear = torch.matmul(bilinear, item_emb.unsqueeze(-1)).squeeze()
        
        # Feature attention
        concat_features = torch.cat([user_emb, item_emb], dim=1)
        attention_weights = self.feature_attention(concat_features)
        
        # Combine interactions
        combined = element_wise + bilinear + self.bias
        attended = combined * attention_weights
        
        return self.dropout(attended)


class DeepNCF(nn.Module):
    """
    Enhanced Neural Collaborative Filtering with additional features and deeper architecture.
    
    Advanced NCF variant with:
    - Genre embeddings for content-based features
    - Deeper network architecture
    - Batch normalization for training stability
    - Attention mechanism for genre importance weighting
    """
    
    def __init__(self, num_users, num_items, num_genres=20, embedding_dim=128, 
                 hidden_layers=[256, 128, 64], dropout=0.3, use_batch_norm=True):
        """Initialize Enhanced Deep NCF model
        
        Args:
            num_users: Number of unique users
            num_items: Number of unique items
            num_genres: Number of unique genres for content features
            embedding_dim: Dimension of embedding vectors (larger for deep model)
            hidden_layers: List of hidden layer sizes (deeper architecture)
            dropout: Dropout rate for regularization
            use_batch_norm: Whether to use batch normalization
        """
        super(DeepNCF, self).__init__()
        
        self.use_batch_norm = use_batch_norm
        
        # User, item, and genre embeddings for hybrid collaborative + content-based filtering
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        self.genre_embedding = nn.Embedding(num_genres, embedding_dim // 4)  # Smaller genre embeddings
        
        # Deep network with configurable architecture
        layers = []
        input_size = embedding_dim * 2 + (embedding_dim // 4)  # User + Item + Genre features
        
        # Build deep network with optional batch normalization
        for hidden_size in hidden_layers:
            layers.append(nn.Linear(input_size, hidden_size))     # Linear transformation
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_size))        # Batch normalization for stability
            layers.append(nn.ReLU())                             # Non-linear activation
            layers.append(nn.Dropout(dropout))                   # Regularization
            input_size = hidden_size
        
        self.network = nn.Sequential(*layers)
        self.prediction = nn.Linear(hidden_layers[-1], 1)
        
        # Attention mechanism for weighting genre importance
        self.genre_attention = nn.Linear(embedding_dim // 4, 1)  # Learns genre importance weights
        
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Embedding):
                nn.init.xavier_normal_(module.weight)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, user_ids, item_ids, genre_ids=None):
        """Forward pass with optional genre features
        
        Args:
            user_ids: Tensor of user IDs
            item_ids: Tensor of item IDs
            genre_ids: Optional tensor of genre IDs [batch_size, max_genres]
            
        Returns:
            Predicted ratings with genre-aware features
        """
        # Get user and item embeddings
        user_emb = self.user_embedding(user_ids)
        item_emb = self.item_embedding(item_ids)
        
        # Process genre information with attention mechanism
        if genre_ids is not None:
            # Handle multiple genres per item with attention weighting
            genre_emb = self.genre_embedding(genre_ids)  # [batch, max_genres, emb_dim]
            
            # Apply attention to weight genre importance
            genre_attn = F.softmax(self.genre_attention(genre_emb), dim=1)  # Attention weights
            genre_emb = (genre_emb * genre_attn).sum(dim=1)                # Weighted genre embedding
        else:
            # Use zero genre embedding if no genre information available
            genre_emb = torch.zeros(user_emb.size(0), self.genre_embedding.embedding_dim, 
                                  device=user_emb.device)
        
        # Combine all features and pass through deep network
        x = torch.cat([user_emb, item_emb, genre_emb], dim=-1)  # Concatenate all features
        x = self.network(x)                                      # Deep network processing
        x = self.prediction(x)                                   # Final prediction layer
        
        # Scale to rating range
        return torch.sigmoid(x).squeeze() * 5.0