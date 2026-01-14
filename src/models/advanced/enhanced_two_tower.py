# Enhanced Two-Tower Model for CineSync v2
# Advanced implementation with cross-attention, feature fusion, and multi-task learning
# Optimized for RTX 4090 with mixed precision training support

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import math


class MultiHeadCrossAttention(nn.Module):
    """Multi-head cross-attention mechanism for two-tower interaction
    
    Enables direct interaction between user and item towers through attention,
    allowing the model to learn complex user-item relationships beyond simple
    dot products. Essential for capturing nuanced preferences.
    """
    
    def __init__(self, d_model: int, num_heads: int = 8, dropout: float = 0.1):
        """Initialize multi-head cross-attention
        
        Args:
            d_model: Model dimension size
            num_heads: Number of attention heads for parallel processing
            dropout: Dropout rate for regularization
        """
        super(MultiHeadCrossAttention, self).__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # Dimension per attention head
        
        # Linear projections for query, key, and value transformations
        self.w_q = nn.Linear(d_model, d_model, bias=False)  # Query projection
        self.w_k = nn.Linear(d_model, d_model, bias=False)  # Key projection
        self.w_v = nn.Linear(d_model, d_model, bias=False)  # Value projection
        self.w_o = nn.Linear(d_model, d_model)              # Output projection
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Xavier uniform distribution for stable training"""
        for module in [self.w_q, self.w_k, self.w_v, self.w_o]:
            nn.init.xavier_uniform_(module.weight)
            
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            query: Query tensor [batch_size, seq_len_q, d_model]
            key: Key tensor [batch_size, seq_len_k, d_model]
            value: Value tensor [batch_size, seq_len_v, d_model]
            mask: Optional attention mask
        """
        batch_size = query.size(0)
        seq_len_q = query.size(1)
        seq_len_k = key.size(1)
        
        # Linear transformations and reshape for multi-head attention
        # Reshape: [batch, seq_len, d_model] -> [batch, num_heads, seq_len, d_k]
        Q = self.w_q(query).view(batch_size, seq_len_q, self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, seq_len_k, self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, seq_len_k, self.num_heads, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention: softmax(QK^T/√d_k)V
        # Scale by sqrt(d_k) to prevent softmax from saturating
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # Apply attention mask to prevent attending to padded positions
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)  # Large negative value
        
        # Convert attention scores to probabilities and apply dropout
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)  # Dropout on attention weights
        
        # Apply attention weights to values: weighted sum of value vectors
        attn_output = torch.matmul(attn_weights, V)
        
        # Concatenate attention heads back to original dimension
        # Reshape: [batch, num_heads, seq_len, d_k] -> [batch, seq_len, d_model]
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len_q, self.d_model
        )
        output = self.w_o(attn_output)  # Final linear projection
        
        # Residual connection and layer normalization for stable training
        return self.layer_norm(query + self.dropout(output))


class AdvancedFeatureFusion(nn.Module):
    """Advanced feature fusion with gating and attention mechanisms"""
    
    def __init__(self, input_dims: List[int], output_dim: int, dropout: float = 0.2):
        super(AdvancedFeatureFusion, self).__init__()
        
        self.input_dims = input_dims
        self.output_dim = output_dim
        total_input_dim = sum(input_dims)
        
        # Feature-specific projections
        self.feature_projections = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, output_dim),
                nn.LayerNorm(output_dim),
                nn.GELU()
            ) for dim in input_dims
        ])
        
        # Attention weights for feature importance
        self.feature_attention = nn.Sequential(
            nn.Linear(total_input_dim, len(input_dims)),
            nn.Softmax(dim=-1)
        )
        
        # Gating mechanism
        self.gate = nn.Sequential(
            nn.Linear(total_input_dim, output_dim),
            nn.Sigmoid()
        )
        
        # Final fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
    
    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            features: List of feature tensors with different dimensions
            
        Returns:
            Fused feature tensor
        """
        # Project features to same dimension
        projected_features = []
        for i, feature in enumerate(features):
            proj_feat = self.feature_projections[i](feature)
            projected_features.append(proj_feat)
        
        # Concatenate original features for attention
        concat_features = torch.cat(features, dim=-1)
        
        # Compute attention weights
        attn_weights = self.feature_attention(concat_features)  # [batch_size, num_features]
        
        # Weighted sum of projected features
        fused = torch.zeros_like(projected_features[0])
        for i, proj_feat in enumerate(projected_features):
            weight = attn_weights[:, i:i+1]  # [batch_size, 1]
            fused += weight * proj_feat
        
        # Apply gating
        gate_weights = self.gate(concat_features)
        fused = fused * gate_weights
        
        # Final fusion
        return self.fusion(fused)


class EnhancedTowerBlock(nn.Module):
    """Enhanced tower block with residual connections and advanced normalization"""
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, 
                 dropout: float = 0.2, use_residual: bool = True):
        super(EnhancedTowerBlock, self).__init__()
        
        self.use_residual = use_residual and (input_dim == output_dim)
        
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim)
        )
        
        # Residual projection if dimensions don't match
        if self.use_residual and input_dim != output_dim:
            self.residual_projection = nn.Linear(input_dim, output_dim)
        else:
            self.residual_projection = None
        
        self.final_activation = nn.GELU()
        self.final_dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.layers(x)
        
        if self.use_residual:
            if self.residual_projection is not None:
                residual = self.residual_projection(residual)
            out = out + residual
        
        return self.final_dropout(self.final_activation(out))


class UltimateTwoTowerModel(nn.Module):
    """
    Ultimate Two-Tower model with all advanced features for RTX 4090 optimization:
    - Cross-attention between towers
    - Advanced feature fusion
    - Multiple embedding strategies
    - Multi-task learning capabilities
    """
    
    def __init__(
        self,
        # User features (with defaults for standalone instantiation)
        user_categorical_dims: Dict[str, int] = None,
        user_numerical_dim: int = 10,
        num_users: int = 50000,

        # Item features (with defaults for standalone instantiation)
        item_categorical_dims: Dict[str, int] = None,
        item_numerical_dim: int = 10,
        num_items: int = 100000,

        # Model architecture
        embedding_dim: int = 512,
        tower_hidden_dims: List[int] = [1024, 512, 256],
        cross_attention_heads: int = 16,
        dropout: float = 0.2,
        
        # Advanced features
        use_cross_attention: bool = True,
        use_collaborative_embeddings: bool = True,
        use_content_attention: bool = True,
        temperature_scaling: bool = True,
        
        # Multi-task learning
        enable_rating_prediction: bool = True,
        enable_click_prediction: bool = True,
        enable_genre_prediction: bool = True
    ):
        super(UltimateTwoTowerModel, self).__init__()
        
        self.embedding_dim = embedding_dim
        self.use_cross_attention = use_cross_attention
        self.use_collaborative_embeddings = use_collaborative_embeddings
        self.use_content_attention = use_content_attention
        
        # Collaborative embeddings (if enabled)
        if use_collaborative_embeddings:
            self.user_collab_embedding = nn.Embedding(num_users, embedding_dim // 2)
            self.item_collab_embedding = nn.Embedding(num_items, embedding_dim // 2)
        
        # Handle None categorical dims with defaults
        if user_categorical_dims is None:
            user_categorical_dims = {'user_type': 10, 'age_group': 10}
        if item_categorical_dims is None:
            item_categorical_dims = {'genre': 30, 'content_type': 5}

        # User categorical embeddings
        self.user_categorical_embeddings = nn.ModuleDict()
        user_cat_dims = []
        for feature_name, vocab_size in user_categorical_dims.items():
            emb_dim = min(128, (vocab_size + 1) // 2)
            self.user_categorical_embeddings[feature_name] = nn.Embedding(vocab_size, emb_dim)
            user_cat_dims.append(emb_dim)

        # Item categorical embeddings
        self.item_categorical_embeddings = nn.ModuleDict()
        item_cat_dims = []
        for feature_name, vocab_size in item_categorical_dims.items():
            emb_dim = min(128, (vocab_size + 1) // 2)
            self.item_categorical_embeddings[feature_name] = nn.Embedding(vocab_size, emb_dim)
            item_cat_dims.append(emb_dim)
        
        # Feature fusion layers
        user_feature_dims = user_cat_dims + ([user_numerical_dim] if user_numerical_dim > 0 else [])
        item_feature_dims = item_cat_dims + ([item_numerical_dim] if item_numerical_dim > 0 else [])
        
        if use_collaborative_embeddings:
            user_feature_dims.append(embedding_dim // 2)
            item_feature_dims.append(embedding_dim // 2)
        
        self.user_feature_fusion = AdvancedFeatureFusion(user_feature_dims, embedding_dim, dropout)
        self.item_feature_fusion = AdvancedFeatureFusion(item_feature_dims, embedding_dim, dropout)
        
        # Tower architectures
        self.user_tower = self._build_enhanced_tower(embedding_dim, tower_hidden_dims, embedding_dim, dropout)
        self.item_tower = self._build_enhanced_tower(embedding_dim, tower_hidden_dims, embedding_dim, dropout)
        
        # Cross-attention mechanisms
        if use_cross_attention:
            self.user_cross_attention = MultiHeadCrossAttention(embedding_dim, cross_attention_heads, dropout)
            self.item_cross_attention = MultiHeadCrossAttention(embedding_dim, cross_attention_heads, dropout)
        
        # Content attention (for focusing on relevant features)
        if use_content_attention:
            self.user_content_attention = nn.MultiheadAttention(embedding_dim, 8, batch_first=True)
            self.item_content_attention = nn.MultiheadAttention(embedding_dim, 8, batch_first=True)
        
        # Temperature scaling
        if temperature_scaling:
            self.temperature = nn.Parameter(torch.ones(1) * 0.07)
        else:
            self.register_buffer('temperature', torch.ones(1))
        
        # Multi-task heads
        self.task_heads = nn.ModuleDict()
        
        if enable_rating_prediction:
            self.task_heads['rating'] = nn.Sequential(
                nn.Linear(embedding_dim * 2, embedding_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(embedding_dim, 1)
            )
        
        if enable_click_prediction:
            self.task_heads['click'] = nn.Sequential(
                nn.Linear(embedding_dim * 2, embedding_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(embedding_dim, 1)
            )
        
        if enable_genre_prediction:
            self.task_heads['genre'] = nn.Sequential(
                nn.Linear(embedding_dim * 2, embedding_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(embedding_dim, 20)  # Assume 20 genres
            )
        
        # Contrastive learning head
        self.contrastive_head = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim // 2),
            nn.GELU(),
            nn.Linear(embedding_dim // 2, embedding_dim)
        )
        
        self._init_weights()
    
    def _build_enhanced_tower(self, input_dim: int, hidden_dims: List[int], 
                             output_dim: int, dropout: float) -> nn.Module:
        """Build enhanced tower with residual connections"""
        layers = []
        
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(EnhancedTowerBlock(prev_dim, hidden_dim, hidden_dim, dropout))
            prev_dim = hidden_dim
        
        # Final projection
        layers.append(nn.Linear(prev_dim, output_dim))
        layers.append(nn.LayerNorm(output_dim))
        
        return nn.Sequential(*layers)
    
    def _init_weights(self):
        """Initialize model weights with advanced strategies"""
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
    
    def _process_user_features(self, user_categorical: Dict[str, torch.Tensor],
                              user_numerical: Optional[torch.Tensor],
                              user_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Process and fuse user features"""
        features = []
        
        # Categorical features
        for feature_name, feature_values in user_categorical.items():
            if feature_name in self.user_categorical_embeddings:
                emb = self.user_categorical_embeddings[feature_name](feature_values)
                features.append(emb)
        
        # Numerical features
        if user_numerical is not None and user_numerical.size(-1) > 0:
            features.append(user_numerical)
        
        # Collaborative embeddings
        if self.use_collaborative_embeddings and user_ids is not None:
            collab_emb = self.user_collab_embedding(user_ids)
            features.append(collab_emb)
        
        # Fuse features
        return self.user_feature_fusion(features)
    
    def _process_item_features(self, item_categorical: Dict[str, torch.Tensor],
                              item_numerical: Optional[torch.Tensor],
                              item_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Process and fuse item features"""
        features = []
        
        # Categorical features
        for feature_name, feature_values in item_categorical.items():
            if feature_name in self.item_categorical_embeddings:
                emb = self.item_categorical_embeddings[feature_name](feature_values)
                features.append(emb)
        
        # Numerical features
        if item_numerical is not None and item_numerical.size(-1) > 0:
            features.append(item_numerical)
        
        # Collaborative embeddings
        if self.use_collaborative_embeddings and item_ids is not None:
            collab_emb = self.item_collab_embedding(item_ids)
            features.append(collab_emb)
        
        # Fuse features
        return self.item_feature_fusion(features)
    
    def encode_users(self, user_categorical: Dict[str, torch.Tensor],
                    user_numerical: Optional[torch.Tensor] = None,
                    user_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Encode user features into embeddings"""
        # Process features
        user_features = self._process_user_features(user_categorical, user_numerical, user_ids)
        
        # Apply content attention if enabled
        if self.use_content_attention:
            user_features = user_features.unsqueeze(1)  # Add sequence dimension
            user_features, _ = self.user_content_attention(user_features, user_features, user_features)
            user_features = user_features.squeeze(1)  # Remove sequence dimension
        
        # Pass through tower
        user_embeddings = self.user_tower(user_features)
        
        # L2 normalize
        return F.normalize(user_embeddings, p=2, dim=-1)
    
    def encode_items(self, item_categorical: Dict[str, torch.Tensor],
                    item_numerical: Optional[torch.Tensor] = None,
                    item_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Encode item features into embeddings"""
        # Process features
        item_features = self._process_item_features(item_categorical, item_numerical, item_ids)
        
        # Apply content attention if enabled
        if self.use_content_attention:
            item_features = item_features.unsqueeze(1)  # Add sequence dimension
            item_features, _ = self.item_content_attention(item_features, item_features, item_features)
            item_features = item_features.squeeze(1)  # Remove sequence dimension
        
        # Pass through tower
        item_embeddings = self.item_tower(item_features)
        
        # L2 normalize
        return F.normalize(item_embeddings, p=2, dim=-1)
    
    def forward(self, user_categorical: Dict[str, torch.Tensor],
                item_categorical: Dict[str, torch.Tensor],
                user_numerical: Optional[torch.Tensor] = None,
                item_numerical: Optional[torch.Tensor] = None,
                user_ids: Optional[torch.Tensor] = None,
                item_ids: Optional[torch.Tensor] = None,
                return_embeddings: bool = False) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass with optional cross-attention and multi-task learning
        
        Returns:
            If return_embeddings=False: similarity scores
            If return_embeddings=True: dict with embeddings and all predictions
        """
        # Encode features
        user_embeddings = self.encode_users(user_categorical, user_numerical, user_ids)
        item_embeddings = self.encode_items(item_categorical, item_numerical, item_ids)
        
        # Apply cross-attention if enabled
        if self.use_cross_attention:
            # Add sequence dimension for attention
            user_emb_seq = user_embeddings.unsqueeze(1)
            item_emb_seq = item_embeddings.unsqueeze(1)
            
            # Cross-attention: users attend to items and vice versa
            user_attended = self.user_cross_attention(user_emb_seq, item_emb_seq, item_emb_seq)
            item_attended = self.item_cross_attention(item_emb_seq, user_emb_seq, user_emb_seq)
            
            # Remove sequence dimension
            user_embeddings = user_attended.squeeze(1)
            item_embeddings = item_attended.squeeze(1)
        
        # Compute similarities
        similarities = torch.sum(user_embeddings * item_embeddings, dim=-1) / self.temperature
        
        if not return_embeddings:
            return similarities
        
        # Multi-task predictions
        combined_embeddings = torch.cat([user_embeddings, item_embeddings], dim=-1)
        predictions = {'similarity': similarities}
        
        for task_name, task_head in self.task_heads.items():
            task_output = task_head(combined_embeddings)
            
            if task_name == 'rating':
                predictions[task_name] = torch.sigmoid(task_output.squeeze()) * 5.0
            elif task_name == 'click':
                predictions[task_name] = torch.sigmoid(task_output.squeeze())
            else:
                predictions[task_name] = task_output
        
        # Add embeddings for analysis
        predictions['user_embeddings'] = user_embeddings
        predictions['item_embeddings'] = item_embeddings
        
        return predictions
    
    def compute_similarity_matrix(self, user_features: Dict[str, torch.Tensor],
                                 item_features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute full similarity matrix for batch recommendation"""
        user_embeddings = self.encode_users(**user_features)
        item_embeddings = self.encode_items(**item_features)
        
        # Compute all pairwise similarities
        similarities = torch.matmul(user_embeddings, item_embeddings.T) / self.temperature
        
        return similarities


class TwoTowerTrainer:
    """Advanced trainer for Two-Tower models with mixed precision"""
    
    def __init__(
        self,
        model: UltimateTwoTowerModel,
        device: torch.device,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        use_amp: bool = True,
        gradient_clip_val: float = 1.0
    ):
        self.model = model.to(device)
        self.device = device
        self.use_amp = use_amp
        self.gradient_clip_val = gradient_clip_val
        
        # Advanced optimizer with different learning rates for different components
        param_groups = [
            {'params': [p for n, p in model.named_parameters() if 'embedding' in n], 'lr': learning_rate * 0.1},
            {'params': [p for n, p in model.named_parameters() if 'embedding' not in n], 'lr': learning_rate}
        ]
        
        self.optimizer = torch.optim.AdamW(
            param_groups,
            weight_decay=weight_decay,
            betas=(0.9, 0.999)
        )
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=learning_rate,
            steps_per_epoch=1000,  # Adjust based on your data
            epochs=100,
            pct_start=0.1
        )
        
        # Mixed precision scaler
        if use_amp:
            self.scaler = torch.cuda.amp.GradScaler()
        
        # Loss functions
        self.similarity_loss = nn.BCEWithLogitsLoss()
        self.rating_loss = nn.MSELoss()
        self.click_loss = nn.BCEWithLogitsLoss()
        self.genre_loss = nn.CrossEntropyLoss()
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Single training step with multi-task learning"""
        self.model.train()
        
        # Move batch to device
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                batch[key] = value.to(self.device)
            elif isinstance(value, dict):
                for k, v in value.items():
                    if isinstance(v, torch.Tensor):
                        value[k] = v.to(self.device)
        
        self.optimizer.zero_grad()
        
        if self.use_amp:
            with torch.cuda.amp.autocast():
                predictions = self.model(
                    user_categorical=batch['user_categorical'],
                    item_categorical=batch['item_categorical'],
                    user_numerical=batch.get('user_numerical'),
                    item_numerical=batch.get('item_numerical'),
                    user_ids=batch.get('user_ids'),
                    item_ids=batch.get('item_ids'),
                    return_embeddings=True
                )
                
                # Compute multi-task losses
                total_loss = 0
                losses = {}
                
                # Similarity loss
                if 'similarity_labels' in batch:
                    sim_loss = self.similarity_loss(predictions['similarity'], batch['similarity_labels'])
                    losses['similarity'] = sim_loss.item()
                    total_loss += sim_loss
                
                # Rating loss
                if 'rating' in predictions and 'rating_labels' in batch:
                    rating_loss = self.rating_loss(predictions['rating'], batch['rating_labels'])
                    losses['rating'] = rating_loss.item()
                    total_loss += rating_loss * 0.5
                
                # Click loss
                if 'click' in predictions and 'click_labels' in batch:
                    click_loss = self.click_loss(predictions['click'], batch['click_labels'])
                    losses['click'] = click_loss.item()
                    total_loss += click_loss * 0.3
                
                losses['total'] = total_loss.item()
            
            self.scaler.scale(total_loss).backward()
            
            if self.gradient_clip_val > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_val)
            
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            predictions = self.model(
                user_categorical=batch['user_categorical'],
                item_categorical=batch['item_categorical'],
                user_numerical=batch.get('user_numerical'),
                item_numerical=batch.get('item_numerical'),
                user_ids=batch.get('user_ids'),
                item_ids=batch.get('item_ids'),
                return_embeddings=True
            )
            
            # Compute losses (same as above but without autocast)
            total_loss = 0
            losses = {}
            
            if 'similarity_labels' in batch:
                sim_loss = self.similarity_loss(predictions['similarity'], batch['similarity_labels'])
                losses['similarity'] = sim_loss.item()
                total_loss += sim_loss
            
            if 'rating' in predictions and 'rating_labels' in batch:
                rating_loss = self.rating_loss(predictions['rating'], batch['rating_labels'])
                losses['rating'] = rating_loss.item()
                total_loss += rating_loss * 0.5
            
            if 'click' in predictions and 'click_labels' in batch:
                click_loss = self.click_loss(predictions['click'], batch['click_labels'])
                losses['click'] = click_loss.item()
                total_loss += click_loss * 0.3
            
            losses['total'] = total_loss.item()
            
            total_loss.backward()
            
            if self.gradient_clip_val > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_val)
            
            self.optimizer.step()
        
        self.scheduler.step()

        return losses


# Alias for compatibility with training script
EnhancedTwoTower = UltimateTwoTowerModel