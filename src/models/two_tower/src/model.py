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


class UltimateTwoTowerModel(nn.Module):
    """
    State-of-the-art Two-Tower model with modern deep learning techniques:
    - Progressive layered interaction networks
    - Mixture of Experts (MoE) for user/item specialization  
    - Dynamic temperature scaling with uncertainty
    - Multi-task learning with uncertainty weighting
    - Knowledge distillation capabilities
    - Contrastive learning for better representations
    """
    
    def __init__(self, user_categorical_dims: Dict[str, int], user_numerical_dim: int,
                 item_categorical_dims: Dict[str, int], item_numerical_dim: int,
                 embedding_dim: int = 512, hidden_layers: List[int] = [1024, 512, 256],
                 dropout: float = 0.1, num_experts: int = 8, num_interaction_layers: int = 4,
                 use_progressive_interaction: bool = True, use_moe: bool = True,
                 temperature: float = 0.07, use_uncertainty: bool = True):
        super(UltimateTwoTowerModel, self).__init__()
        
        self.embedding_dim = embedding_dim
        self.num_experts = num_experts
        self.num_interaction_layers = num_interaction_layers
        self.use_progressive_interaction = use_progressive_interaction
        self.use_moe = use_moe
        self.use_uncertainty = use_uncertainty
        self.temperature = temperature
        
        # Advanced categorical embeddings with learnable sizes
        self.user_categorical_embeddings = nn.ModuleDict()
        user_cat_total_dim = 0
        
        for feature_name, vocab_size in user_categorical_dims.items():
            # Dynamic embedding size based on importance and vocabulary
            emb_dim = min(100, max(8, int(vocab_size ** 0.25) * 4))
            self.user_categorical_embeddings[feature_name] = nn.Sequential(
                nn.Embedding(vocab_size, emb_dim),
                nn.LayerNorm(emb_dim),
                nn.Dropout(dropout * 0.5)
            )
            user_cat_total_dim += emb_dim
        
        self.item_categorical_embeddings = nn.ModuleDict()
        item_cat_total_dim = 0
        
        for feature_name, vocab_size in item_categorical_dims.items():
            emb_dim = min(100, max(8, int(vocab_size ** 0.25) * 4))
            self.item_categorical_embeddings[feature_name] = nn.Sequential(
                nn.Embedding(vocab_size, emb_dim),
                nn.LayerNorm(emb_dim),
                nn.Dropout(dropout * 0.5)
            )
            item_cat_total_dim += emb_dim
        
        # Feature preprocessing
        user_total_dim = user_cat_total_dim + user_numerical_dim
        item_total_dim = item_cat_total_dim + item_numerical_dim
        
        self.user_feature_processor = FeatureProcessor(user_total_dim, embedding_dim, dropout)
        self.item_feature_processor = FeatureProcessor(item_total_dim, embedding_dim, dropout)
        
        # Enhanced towers with residual connections
        self.user_tower = EnhancedTower(embedding_dim, hidden_layers, embedding_dim, dropout)
        self.item_tower = EnhancedTower(embedding_dim, hidden_layers, embedding_dim, dropout)
        
        # Mixture of Experts for specialization
        if use_moe:
            self.user_moe = MixtureOfExperts(embedding_dim, num_experts, dropout)
            self.item_moe = MixtureOfExperts(embedding_dim, num_experts, dropout)
        
        # Progressive layered interaction network
        if use_progressive_interaction:
            self.interaction_layers = nn.ModuleList([
                ProgressiveInteractionLayer(embedding_dim, dropout)
                for _ in range(num_interaction_layers)
            ])
        
        # Dynamic temperature scaling with uncertainty
        if use_uncertainty:
            self.temperature_predictor = nn.Sequential(
                nn.Linear(embedding_dim * 2, embedding_dim),
                nn.ReLU(),
                nn.Linear(embedding_dim, 2)  # mean and log_var for temperature
            )
        else:
            self.temperature_param = nn.Parameter(torch.tensor(temperature))
        
        # Multi-task heads with uncertainty weighting
        self.task_heads = nn.ModuleDict({
            'similarity': UncertaintyWeightedHead(embedding_dim * 2, 1, dropout),
            'rating': UncertaintyWeightedHead(embedding_dim * 2, 1, dropout),
            'click': UncertaintyWeightedHead(embedding_dim * 2, 1, dropout),
            'genre': UncertaintyWeightedHead(embedding_dim * 2, 20, dropout)
        })
        
        # Contrastive learning components
        self.contrastive_projector = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim, embedding_dim // 2)
        )
        
        # Knowledge distillation adapter
        self.teacher_adapter = nn.Linear(embedding_dim, embedding_dim)
        
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Embedding):
                nn.init.xavier_normal_(module.weight)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def _process_categorical_features(self, categorical_dict, embeddings_dict):
        """Process categorical features with advanced embeddings"""
        embeddings = []
        for feature_name, feature_values in categorical_dict.items():
            if feature_name in embeddings_dict:
                emb = embeddings_dict[feature_name](feature_values)
                if len(emb.shape) == 3:  # Handle sequential output from embedding layers
                    emb = emb.squeeze(1)
                embeddings.append(emb)
        
        return torch.cat(embeddings, dim=1) if embeddings else None
    
    def encode_users(self, user_categorical: Dict[str, torch.Tensor], 
                    user_numerical: torch.Tensor) -> torch.Tensor:
        """Enhanced user encoding with MoE and advanced features"""
        # Process categorical features
        cat_features = self._process_categorical_features(
            user_categorical, self.user_categorical_embeddings
        )
        
        # Combine features
        if cat_features is not None and user_numerical.size(1) > 0:
            combined_features = torch.cat([cat_features, user_numerical], dim=1)
        elif cat_features is not None:
            combined_features = cat_features
        else:
            combined_features = user_numerical
        
        # Feature processing
        processed_features = self.user_feature_processor(combined_features)
        
        # Enhanced tower encoding
        user_embeddings = self.user_tower(processed_features)
        
        # Mixture of Experts
        if self.use_moe:
            user_embeddings = self.user_moe(user_embeddings)
        
        return F.normalize(user_embeddings, p=2, dim=1)
    
    def encode_items(self, item_categorical: Dict[str, torch.Tensor],
                    item_numerical: torch.Tensor) -> torch.Tensor:
        """Enhanced item encoding with MoE and advanced features"""
        # Process categorical features
        cat_features = self._process_categorical_features(
            item_categorical, self.item_categorical_embeddings
        )
        
        # Combine features
        if cat_features is not None and item_numerical.size(1) > 0:
            combined_features = torch.cat([cat_features, item_numerical], dim=1)
        elif cat_features is not None:
            combined_features = cat_features
        else:
            combined_features = item_numerical
        
        # Feature processing
        processed_features = self.item_feature_processor(combined_features)
        
        # Enhanced tower encoding
        item_embeddings = self.item_tower(processed_features)
        
        # Mixture of Experts
        if self.use_moe:
            item_embeddings = self.item_moe(item_embeddings)
        
        return F.normalize(item_embeddings, p=2, dim=1)
    
    def forward(self, user_categorical: Dict[str, torch.Tensor], user_numerical: torch.Tensor,
                item_categorical: Dict[str, torch.Tensor], item_numerical: torch.Tensor,
                return_embeddings: bool = False) -> Dict[str, torch.Tensor]:
        """Enhanced forward pass with progressive interaction and multi-task outputs"""
        
        user_embeddings = self.encode_users(user_categorical, user_numerical)
        item_embeddings = self.encode_items(item_categorical, item_numerical)
        
        # Progressive layered interaction
        if self.use_progressive_interaction:
            user_evolved = user_embeddings
            item_evolved = item_embeddings
            
            for interaction_layer in self.interaction_layers:
                user_evolved, item_evolved = interaction_layer(user_evolved, item_evolved)
            
            final_user_emb = user_evolved
            final_item_emb = item_evolved
        else:
            final_user_emb = user_embeddings
            final_item_emb = item_embeddings
        
        # Combined representation for multi-task learning
        combined_emb = torch.cat([final_user_emb, final_item_emb], dim=1)
        
        # Dynamic temperature scaling
        if self.use_uncertainty:
            temp_params = self.temperature_predictor(combined_emb)
            temp_mean = temp_params[:, 0:1]
            temp_logvar = temp_params[:, 1:2]
            
            # Reparameterization trick for temperature
            if self.training:
                temp_std = torch.exp(0.5 * temp_logvar)
                eps = torch.randn_like(temp_std)
                temperature = temp_mean + eps * temp_std
            else:
                temperature = temp_mean
            
            temperature = torch.sigmoid(temperature) * 0.5 + 0.01  # Scale to (0.01, 0.51)
        else:
            temperature = self.temperature_param
            temp_logvar = torch.zeros_like(temperature)
        
        # Multi-task predictions
        outputs = {}
        
        for task_name, task_head in self.task_heads.items():
            if task_name == 'similarity':
                # Base similarity computation
                similarity = torch.sum(final_user_emb * final_item_emb, dim=1, keepdim=True)
                if self.use_uncertainty:
                    similarity = similarity / temperature
                else:
                    similarity = similarity / temperature
                
                pred, uncertainty = task_head(combined_emb)
                outputs[task_name] = similarity.squeeze()
                outputs[f'{task_name}_uncertainty'] = uncertainty.squeeze()
            else:
                pred, uncertainty = task_head(combined_emb)
                
                if task_name == 'rating':
                    outputs[task_name] = torch.sigmoid(pred).squeeze() * 5.0
                elif task_name == 'click':
                    outputs[task_name] = torch.sigmoid(pred).squeeze()
                elif task_name == 'genre':
                    outputs[task_name] = F.softmax(pred, dim=-1)
                else:
                    outputs[task_name] = pred.squeeze()
                
                outputs[f'{task_name}_uncertainty'] = uncertainty.squeeze()
        
        if self.use_uncertainty:
            outputs['temperature'] = temperature.squeeze()
            outputs['temperature_uncertainty'] = temp_logvar.squeeze()
        
        if return_embeddings:
            outputs['user_embeddings'] = final_user_emb
            outputs['item_embeddings'] = final_item_emb
            outputs['contrastive_user'] = F.normalize(self.contrastive_projector(final_user_emb), dim=1)
            outputs['contrastive_item'] = F.normalize(self.contrastive_projector(final_item_emb), dim=1)
        
        return outputs
    
    def compute_contrastive_loss(self, anchor_user_cat, anchor_user_num, 
                               positive_item_cat, positive_item_num,
                               negative_item_cat, negative_item_num):
        """Contrastive learning loss for better representations"""
        
        # Get contrastive embeddings
        anchor_outputs = self.forward(anchor_user_cat, anchor_user_num, 
                                    positive_item_cat, positive_item_num, return_embeddings=True)
        negative_outputs = self.forward(anchor_user_cat, anchor_user_num,
                                      negative_item_cat, negative_item_num, return_embeddings=True)
        
        anchor_user = anchor_outputs['contrastive_user']
        positive_item = anchor_outputs['contrastive_item']
        negative_item = negative_outputs['contrastive_item']
        
        # Compute similarities
        pos_sim = torch.sum(anchor_user * positive_item, dim=1) / self.temperature
        neg_sim = torch.sum(anchor_user * negative_item, dim=1) / self.temperature
        
        # InfoNCE loss
        logits = torch.cat([pos_sim.unsqueeze(1), neg_sim.unsqueeze(1)], dim=1)
        labels = torch.zeros(anchor_user.size(0), dtype=torch.long, device=anchor_user.device)
        
        return F.cross_entropy(logits, labels)
    
    def knowledge_distillation_loss(self, student_outputs, teacher_outputs, temperature=4.0):
        """Knowledge distillation from teacher model"""
        student_logits = student_outputs['similarity']
        teacher_logits = self.teacher_adapter(teacher_outputs)
        
        # Soft targets
        soft_student = F.log_softmax(student_logits / temperature, dim=-1)
        soft_teacher = F.softmax(teacher_logits / temperature, dim=-1)
        
        return F.kl_div(soft_student, soft_teacher, reduction='batchmean') * (temperature ** 2)


# Helper classes for UltimateTwoTowerModel
class FeatureProcessor(nn.Module):
    """Advanced feature processing with attention"""
    
    def __init__(self, input_dim, output_dim, dropout=0.1):
        super(FeatureProcessor, self).__init__()
        
        self.feature_attention = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.Tanh(),
            nn.Linear(input_dim, input_dim),
            nn.Sigmoid()
        )
        
        self.processor = nn.Sequential(
            nn.Linear(input_dim, output_dim * 2),
            nn.LayerNorm(output_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim * 2, output_dim),
            nn.LayerNorm(output_dim)
        )
    
    def forward(self, x):
        # Feature-wise attention
        attention_weights = self.feature_attention(x)
        attended_features = x * attention_weights
        
        return self.processor(attended_features)


class EnhancedTower(nn.Module):
    """Enhanced tower with residual connections and attention"""
    
    def __init__(self, input_dim, hidden_layers, output_dim, dropout=0.1):
        super(EnhancedTower, self).__init__()
        
        self.layers = nn.ModuleList()
        prev_dim = input_dim
        
        for hidden_dim in hidden_layers:
            self.layers.append(EnhancedBlock(prev_dim, hidden_dim, dropout))
            prev_dim = hidden_dim
        
        # Final projection
        self.final_layer = nn.Sequential(
            nn.Linear(prev_dim, output_dim),
            nn.LayerNorm(output_dim)
        )
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.final_layer(x)


class EnhancedBlock(nn.Module):
    """Enhanced block with residual connection and attention"""
    
    def __init__(self, input_dim, output_dim, dropout=0.1):
        super(EnhancedBlock, self).__init__()
        
        self.linear1 = nn.Linear(input_dim, output_dim)
        self.linear2 = nn.Linear(output_dim, output_dim)
        self.norm1 = nn.LayerNorm(output_dim)
        self.norm2 = nn.LayerNorm(output_dim)
        self.dropout = nn.Dropout(dropout)
        
        # Residual connection adapter
        if input_dim != output_dim:
            self.residual_adapter = nn.Linear(input_dim, output_dim)
        else:
            self.residual_adapter = nn.Identity()
    
    def forward(self, x):
        # First sub-layer
        out = self.linear1(x)
        out = self.norm1(out)
        out = F.relu(out)
        out = self.dropout(out)
        
        # Second sub-layer
        out = self.linear2(out)
        out = self.norm2(out)
        
        # Residual connection
        residual = self.residual_adapter(x)
        return F.relu(out + residual)


class MixtureOfExperts(nn.Module):
    """Mixture of Experts for specialized processing"""
    
    def __init__(self, input_dim, num_experts, dropout=0.1):
        super(MixtureOfExperts, self).__init__()
        
        self.num_experts = num_experts
        
        # Gating network
        self.gate = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Linear(input_dim // 2, num_experts),
            nn.Softmax(dim=-1)
        )
        
        # Expert networks
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, input_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(input_dim, input_dim)
            ) for _ in range(num_experts)
        ])
    
    def forward(self, x):
        # Compute gate weights
        gate_weights = self.gate(x)  # [batch_size, num_experts]
        
        # Expert outputs
        expert_outputs = []
        for expert in self.experts:
            expert_outputs.append(expert(x))
        
        expert_outputs = torch.stack(expert_outputs, dim=2)  # [batch_size, input_dim, num_experts]
        
        # Weighted combination
        output = torch.sum(expert_outputs * gate_weights.unsqueeze(1), dim=2)
        
        return output


class ProgressiveInteractionLayer(nn.Module):
    """Progressive interaction between user and item embeddings"""
    
    def __init__(self, embedding_dim, dropout=0.1):
        super(ProgressiveInteractionLayer, self).__init__()
        
        # Cross-attention mechanisms
        self.user_to_item_attention = nn.MultiheadAttention(
            embedding_dim, 8, dropout=dropout, batch_first=True
        )
        self.item_to_user_attention = nn.MultiheadAttention(
            embedding_dim, 8, dropout=dropout, batch_first=True
        )
        
        # Layer normalization
        self.user_norm = nn.LayerNorm(embedding_dim)
        self.item_norm = nn.LayerNorm(embedding_dim)
        
        # Learnable combination weights
        self.alpha = nn.Parameter(torch.ones(1) * 0.1)
        self.beta = nn.Parameter(torch.ones(1) * 0.1)
    
    def forward(self, user_emb, item_emb):
        # Add sequence dimension for attention
        user_seq = user_emb.unsqueeze(1)
        item_seq = item_emb.unsqueeze(1)
        
        # Cross-attention
        user_attended, _ = self.user_to_item_attention(user_seq, item_seq, item_seq)
        item_attended, _ = self.item_to_user_attention(item_seq, user_seq, user_seq)
        
        # Remove sequence dimension
        user_attended = user_attended.squeeze(1)
        item_attended = item_attended.squeeze(1)
        
        # Residual connections with learnable weights
        user_evolved = self.user_norm(user_emb + self.alpha * user_attended)
        item_evolved = self.item_norm(item_emb + self.beta * item_attended)
        
        return user_evolved, item_evolved


class UncertaintyWeightedHead(nn.Module):
    """Prediction head with uncertainty estimation"""
    
    def __init__(self, input_dim, output_dim, dropout=0.1):
        super(UncertaintyWeightedHead, self).__init__()
        
        # Mean prediction
        self.mean_head = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(input_dim // 2, output_dim)
        )
        
        # Uncertainty prediction
        self.uncertainty_head = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(input_dim // 2, output_dim),
            nn.Softplus()  # Ensure positive uncertainty
        )
    
    def forward(self, x):
        mean = self.mean_head(x)
        uncertainty = self.uncertainty_head(x)
        return mean, uncertainty


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