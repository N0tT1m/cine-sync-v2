# Sentence-BERT Enhanced Two-Tower Model for CineSync v2
# Integrates Sentence-BERT for superior semantic understanding of content features
# Combines collaborative filtering with advanced content-based recommendations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import math
from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoTokenizer


class SentenceBERTEncoder(nn.Module):
    """Sentence-BERT encoder for content features with fine-tuning capability"""
    
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        output_dim: int = 384,
        freeze_base: bool = False,
        fine_tune_layers: int = 2
    ):
        super(SentenceBERTEncoder, self).__init__()
        
        self.model_name = model_name
        self.output_dim = output_dim
        
        # Load pre-trained Sentence-BERT model
        self.sentence_bert = SentenceTransformer(model_name)
        
        # Get the base dimension
        self.base_dim = self.sentence_bert.get_sentence_embedding_dimension()
        
        # Freeze or unfreeze layers
        if freeze_base:
            for param in self.sentence_bert.parameters():
                param.requires_grad = False
        else:
            # Only fine-tune last few layers
            total_layers = len(list(self.sentence_bert.parameters()))
            for i, param in enumerate(self.sentence_bert.parameters()):
                if i < total_layers - fine_tune_layers:
                    param.requires_grad = False
        
        # Projection layer to match output dimension
        if self.base_dim != output_dim:
            self.projection = nn.Sequential(
                nn.Linear(self.base_dim, output_dim),
                nn.LayerNorm(output_dim),
                nn.GELU(),
                nn.Dropout(0.1)
            )
        else:
            self.projection = nn.Identity()
    
    def encode_texts(self, texts: List[str]) -> torch.Tensor:
        """Encode list of texts using Sentence-BERT"""
        with torch.no_grad() if self.training else torch.enable_grad():
            embeddings = self.sentence_bert.encode(
                texts,
                convert_to_tensor=True,
                normalize_embeddings=True
            )
        
        return self.projection(embeddings)
    
    def forward(self, text_features: torch.Tensor) -> torch.Tensor:
        """Forward pass for pre-computed text embeddings"""
        return self.projection(text_features)


class ContentAwareFeatureFusion(nn.Module):
    """Advanced feature fusion with content-aware attention"""
    
    def __init__(
        self,
        content_dim: int,
        categorical_dims: List[int],
        numerical_dim: int,
        output_dim: int,
        dropout: float = 0.2
    ):
        super(ContentAwareFeatureFusion, self).__init__()
        
        self.content_dim = content_dim
        self.output_dim = output_dim
        
        # Content feature processing
        self.content_processor = nn.Sequential(
            nn.Linear(content_dim, output_dim // 2),
            nn.LayerNorm(output_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Categorical feature processing
        total_cat_dim = sum(categorical_dims) if categorical_dims else 0
        if total_cat_dim > 0:
            self.categorical_processor = nn.Sequential(
                nn.Linear(total_cat_dim, output_dim // 4),
                nn.LayerNorm(output_dim // 4),
                nn.GELU(),
                nn.Dropout(dropout)
            )
        else:
            self.categorical_processor = None
        
        # Numerical feature processing
        if numerical_dim > 0:
            self.numerical_processor = nn.Sequential(
                nn.Linear(numerical_dim, output_dim // 4),
                nn.LayerNorm(output_dim // 4),
                nn.GELU(),
                nn.Dropout(dropout)
            )
        else:
            self.numerical_processor = None
        
        # Content-aware attention
        fusion_input_dim = output_dim // 2  # Content
        if self.categorical_processor:
            fusion_input_dim += output_dim // 4
        if self.numerical_processor:
            fusion_input_dim += output_dim // 4
        
        self.content_attention = nn.MultiheadAttention(
            embed_dim=output_dim // 2,
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )
        
        # Final fusion
        self.fusion_layer = nn.Sequential(
            nn.Linear(fusion_input_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Gating mechanism for content importance
        self.content_gate = nn.Sequential(
            nn.Linear(fusion_input_dim, 1),
            nn.Sigmoid()
        )
    
    def forward(
        self,
        content_features: torch.Tensor,
        categorical_features: Optional[torch.Tensor] = None,
        numerical_features: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        
        # Process content features with self-attention
        content_processed = self.content_processor(content_features)
        content_attended, _ = self.content_attention(
            content_processed.unsqueeze(1),
            content_processed.unsqueeze(1),
            content_processed.unsqueeze(1)
        )
        content_attended = content_attended.squeeze(1)
        
        # Collect all features
        features = [content_attended]
        
        if categorical_features is not None and self.categorical_processor:
            cat_processed = self.categorical_processor(categorical_features)
            features.append(cat_processed)
        
        if numerical_features is not None and self.numerical_processor:
            num_processed = self.numerical_processor(numerical_features)
            features.append(num_processed)
        
        # Concatenate all features
        combined_features = torch.cat(features, dim=-1)
        
        # Apply content gating
        content_importance = self.content_gate(combined_features)
        
        # Final fusion
        fused_features = self.fusion_layer(combined_features)
        
        # Apply content gating to emphasize content when relevant
        return fused_features * (1 + content_importance)


class SemanticTowerBlock(nn.Module):
    """Enhanced tower block with semantic understanding"""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        dropout: float = 0.2,
        use_semantic_attention: bool = True
    ):
        super(SemanticTowerBlock, self).__init__()
        
        self.use_semantic_attention = use_semantic_attention
        
        # Main processing layers
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim)
        )
        
        # Semantic attention for feature importance
        if use_semantic_attention:
            self.semantic_attention = nn.Sequential(
                nn.Linear(input_dim, hidden_dim // 4),
                nn.GELU(),
                nn.Linear(hidden_dim // 4, input_dim),
                nn.Sigmoid()
            )
        
        # Residual connection
        if input_dim == output_dim:
            self.residual = nn.Identity()
        else:
            self.residual = nn.Linear(input_dim, output_dim)
        
        self.final_activation = nn.GELU()
        self.final_dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.residual(x)
        
        # Apply semantic attention
        if self.use_semantic_attention:
            attention_weights = self.semantic_attention(x)
            x = x * attention_weights
        
        # Main processing
        out = self.layers(x)
        
        # Residual connection
        out = out + residual
        
        return self.final_dropout(self.final_activation(out))


class SentenceBERTTwoTowerModel(nn.Module):
    """
    Two-Tower model enhanced with Sentence-BERT for semantic content understanding
    
    Key improvements:
    - Sentence-BERT for plot/description encoding
    - Content-aware feature fusion
    - Semantic attention mechanisms
    - Multi-modal understanding (content + collaborative)
    """
    
    def __init__(
        self,
        # Required parameters (no defaults)
        # User features
        user_categorical_dims: Dict[str, int],
        user_numerical_dim: int,
        num_users: int,
        
        # Item features
        item_categorical_dims: Dict[str, int],
        item_numerical_dim: int,
        num_items: int,
        
        # Optional parameters (with defaults)
        # Content features
        sentence_bert_model: str = "all-MiniLM-L6-v2",
        content_embedding_dim: int = 384,
        
        # Model architecture
        embedding_dim: int = 512,
        tower_hidden_dims: List[int] = [1024, 512, 256],
        dropout: float = 0.2,
        
        # Advanced features
        use_collaborative_embeddings: bool = True,
        use_semantic_attention: bool = True,
        freeze_sentence_bert: bool = False,
        fine_tune_bert_layers: int = 2,
        
        # Multi-task learning
        enable_rating_prediction: bool = True,
        enable_genre_prediction: bool = True,
        enable_content_similarity: bool = True
    ):
        super(SentenceBERTTwoTowerModel, self).__init__()
        
        self.embedding_dim = embedding_dim
        self.use_collaborative_embeddings = use_collaborative_embeddings
        self.use_semantic_attention = use_semantic_attention
        
        # Sentence-BERT encoder for content
        self.sentence_bert_encoder = SentenceBERTEncoder(
            model_name=sentence_bert_model,
            output_dim=content_embedding_dim,
            freeze_base=freeze_sentence_bert,
            fine_tune_layers=fine_tune_bert_layers
        )
        
        # Collaborative embeddings
        if use_collaborative_embeddings:
            self.user_collab_embedding = nn.Embedding(num_users, embedding_dim // 4)
            self.item_collab_embedding = nn.Embedding(num_items, embedding_dim // 4)
        
        # Categorical embeddings
        self.user_categorical_embeddings = nn.ModuleDict()
        user_cat_dims = []
        for feature_name, vocab_size in user_categorical_dims.items():
            emb_dim = min(64, (vocab_size + 1) // 2)
            self.user_categorical_embeddings[feature_name] = nn.Embedding(vocab_size, emb_dim)
            user_cat_dims.append(emb_dim)
        
        self.item_categorical_embeddings = nn.ModuleDict()
        item_cat_dims = []
        for feature_name, vocab_size in item_categorical_dims.items():
            emb_dim = min(64, (vocab_size + 1) // 2)
            self.item_categorical_embeddings[feature_name] = nn.Embedding(vocab_size, emb_dim)
            item_cat_dims.append(emb_dim)
        
        # Content-aware feature fusion
        self.user_feature_fusion = ContentAwareFeatureFusion(
            content_dim=content_embedding_dim,
            categorical_dims=user_cat_dims,
            numerical_dim=user_numerical_dim,
            output_dim=embedding_dim,
            dropout=dropout
        )
        
        self.item_feature_fusion = ContentAwareFeatureFusion(
            content_dim=content_embedding_dim,
            categorical_dims=item_cat_dims,
            numerical_dim=user_numerical_dim,
            output_dim=embedding_dim,
            dropout=dropout
        )
        
        # Enhanced towers with semantic blocks
        self.user_tower = self._build_semantic_tower(
            embedding_dim, tower_hidden_dims, embedding_dim, dropout
        )
        self.item_tower = self._build_semantic_tower(
            embedding_dim, tower_hidden_dims, embedding_dim, dropout
        )
        
        # Cross-modal attention between content and collaborative features
        self.cross_modal_attention = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )
        
        # Temperature scaling for similarity
        self.temperature = nn.Parameter(torch.ones(1) * 0.07)
        
        # Multi-task heads
        self.task_heads = nn.ModuleDict()
        
        if enable_rating_prediction:
            self.task_heads['rating'] = nn.Sequential(
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
        
        if enable_content_similarity:
            # Separate head for content-based similarity
            self.content_similarity_head = nn.Sequential(
                nn.Linear(content_embedding_dim * 2, content_embedding_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(content_embedding_dim, 1)
            )
        
        # Semantic matching head
        self.semantic_matching_head = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim, embedding_dim // 2),
            nn.GELU(),
            nn.Linear(embedding_dim // 2, 1)
        )
        
        self._init_weights()
    
    def _build_semantic_tower(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        dropout: float
    ) -> nn.Module:
        """Build tower with semantic understanding blocks"""
        layers = []
        
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(SemanticTowerBlock(
                prev_dim, hidden_dim, hidden_dim, dropout, self.use_semantic_attention
            ))
            prev_dim = hidden_dim
        
        # Final projection
        layers.append(nn.Linear(prev_dim, output_dim))
        layers.append(nn.LayerNorm(output_dim))
        
        return nn.Sequential(*layers)
    
    def _init_weights(self):
        """Initialize weights with advanced strategies"""
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
    
    def encode_users(
        self,
        user_content: Optional[torch.Tensor] = None,
        user_content_texts: Optional[List[str]] = None,
        user_categorical: Optional[Dict[str, torch.Tensor]] = None,
        user_numerical: Optional[torch.Tensor] = None,
        user_ids: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """Encode user features with content understanding"""
        
        # Process content features
        if user_content_texts is not None:
            content_features = self.sentence_bert_encoder.encode_texts(user_content_texts)
        elif user_content is not None:
            content_features = self.sentence_bert_encoder(user_content)
        else:
            # Create dummy content features
            batch_size = (user_ids.size(0) if user_ids is not None 
                         else list(user_categorical.values())[0].size(0))
            content_features = torch.zeros(
                batch_size, self.sentence_bert_encoder.output_dim,
                device=next(self.parameters()).device
            )
        
        # Process categorical features
        categorical_features = None
        if user_categorical:
            cat_embeddings = []
            for feature_name, feature_values in user_categorical.items():
                if feature_name in self.user_categorical_embeddings:
                    emb = self.user_categorical_embeddings[feature_name](feature_values)
                    cat_embeddings.append(emb)
            if cat_embeddings:
                categorical_features = torch.cat(cat_embeddings, dim=-1)
        
        # Feature fusion
        user_features = self.user_feature_fusion(
            content_features=content_features,
            categorical_features=categorical_features,
            numerical_features=user_numerical
        )
        
        # Add collaborative embeddings
        if self.use_collaborative_embeddings and user_ids is not None:
            collab_emb = self.user_collab_embedding(user_ids)
            # Combine with content-aware features using attention
            combined = torch.cat([user_features.unsqueeze(1), collab_emb.unsqueeze(1)], dim=1)
            attended, _ = self.cross_modal_attention(combined, combined, combined)
            user_features = attended.mean(dim=1)
        
        # Pass through tower
        user_embeddings = self.user_tower(user_features)
        
        return {
            'embeddings': F.normalize(user_embeddings, p=2, dim=-1),
            'content_features': content_features,
            'raw_features': user_features
        }
    
    def encode_items(
        self,
        item_content: Optional[torch.Tensor] = None,
        item_content_texts: Optional[List[str]] = None,
        item_categorical: Optional[Dict[str, torch.Tensor]] = None,
        item_numerical: Optional[torch.Tensor] = None,
        item_ids: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """Encode item features with content understanding"""
        
        # Process content features (plots, descriptions, etc.)
        if item_content_texts is not None:
            content_features = self.sentence_bert_encoder.encode_texts(item_content_texts)
        elif item_content is not None:
            content_features = self.sentence_bert_encoder(item_content)
        else:
            # Create dummy content features
            batch_size = (item_ids.size(0) if item_ids is not None 
                         else list(item_categorical.values())[0].size(0))
            content_features = torch.zeros(
                batch_size, self.sentence_bert_encoder.output_dim,
                device=next(self.parameters()).device
            )
        
        # Process categorical features
        categorical_features = None
        if item_categorical:
            cat_embeddings = []
            for feature_name, feature_values in item_categorical.items():
                if feature_name in self.item_categorical_embeddings:
                    emb = self.item_categorical_embeddings[feature_name](feature_values)
                    cat_embeddings.append(emb)
            if cat_embeddings:
                categorical_features = torch.cat(cat_embeddings, dim=-1)
        
        # Feature fusion
        item_features = self.item_feature_fusion(
            content_features=content_features,
            categorical_features=categorical_features,
            numerical_features=item_numerical
        )
        
        # Add collaborative embeddings
        if self.use_collaborative_embeddings and item_ids is not None:
            collab_emb = self.item_collab_embedding(item_ids)
            # Combine with content-aware features using attention
            combined = torch.cat([item_features.unsqueeze(1), collab_emb.unsqueeze(1)], dim=1)
            attended, _ = self.cross_modal_attention(combined, combined, combined)
            item_features = attended.mean(dim=1)
        
        # Pass through tower
        item_embeddings = self.item_tower(item_features)
        
        return {
            'embeddings': F.normalize(item_embeddings, p=2, dim=-1),
            'content_features': content_features,
            'raw_features': item_features
        }
    
    def forward(
        self,
        # User inputs
        user_content_texts: Optional[List[str]] = None,
        user_content: Optional[torch.Tensor] = None,
        user_categorical: Optional[Dict[str, torch.Tensor]] = None,
        user_numerical: Optional[torch.Tensor] = None,
        user_ids: Optional[torch.Tensor] = None,
        
        # Item inputs
        item_content_texts: Optional[List[str]] = None,
        item_content: Optional[torch.Tensor] = None,
        item_categorical: Optional[Dict[str, torch.Tensor]] = None,
        item_numerical: Optional[torch.Tensor] = None,
        item_ids: Optional[torch.Tensor] = None,
        
        return_all_outputs: bool = False
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass with semantic understanding
        
        Returns:
            If return_all_outputs=False: similarity scores
            If return_all_outputs=True: dict with all predictions and embeddings
        """
        
        # Encode features
        user_outputs = self.encode_users(
            user_content, user_content_texts, user_categorical, user_numerical, user_ids
        )
        item_outputs = self.encode_items(
            item_content, item_content_texts, item_categorical, item_numerical, item_ids
        )
        
        user_embeddings = user_outputs['embeddings']
        item_embeddings = item_outputs['embeddings']
        
        # Compute similarities
        collaborative_similarity = torch.sum(user_embeddings * item_embeddings, dim=-1)
        
        # Content-based similarity using Sentence-BERT features
        content_similarity = torch.sum(
            user_outputs['content_features'] * item_outputs['content_features'], dim=-1
        )
        
        # Semantic matching using learned representations
        combined_features = torch.cat([user_embeddings, item_embeddings], dim=-1)
        semantic_similarity = self.semantic_matching_head(combined_features).squeeze()
        
        # Combined similarity with temperature scaling
        total_similarity = (
            collaborative_similarity + 
            0.3 * content_similarity + 
            0.2 * semantic_similarity
        ) / self.temperature
        
        if not return_all_outputs:
            return total_similarity
        
        # Multi-task predictions
        predictions = {
            'similarity': total_similarity,
            'collaborative_similarity': collaborative_similarity,
            'content_similarity': content_similarity,
            'semantic_similarity': semantic_similarity
        }
        
        # Task-specific predictions
        for task_name, task_head in self.task_heads.items():
            if task_name == 'rating':
                predictions[task_name] = torch.sigmoid(task_head(combined_features).squeeze()) * 5.0
            elif task_name == 'genre':
                predictions[task_name] = task_head(combined_features)
        
        # Content similarity prediction
        if 'content_similarity_head' in dir(self):
            content_combined = torch.cat([
                user_outputs['content_features'], 
                item_outputs['content_features']
            ], dim=-1)
            predictions['content_match'] = torch.sigmoid(
                self.content_similarity_head(content_combined).squeeze()
            )
        
        # Add embeddings for analysis
        predictions.update({
            'user_embeddings': user_embeddings,
            'item_embeddings': item_embeddings,
            'user_content_features': user_outputs['content_features'],
            'item_content_features': item_outputs['content_features']
        })
        
        return predictions
    
    def get_recommendations(
        self,
        user_features: Dict,
        item_features: Dict,
        k: int = 10
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get top-k recommendations with semantic understanding"""
        
        self.eval()
        with torch.no_grad():
            predictions = self.forward(
                **user_features,
                **item_features,
                return_all_outputs=True
            )
            
            similarities = predictions['similarity']
            top_k_scores, top_k_indices = torch.topk(similarities, k, dim=-1)
            
            return top_k_indices, top_k_scores


class SentenceBERTTwoTowerTrainer:
    """Trainer for Sentence-BERT enhanced Two-Tower model"""
    
    def __init__(
        self,
        model: SentenceBERTTwoTowerModel,
        device: torch.device,
        learning_rate: float = 1e-4,
        sentence_bert_lr: float = 1e-5,  # Lower LR for pre-trained components
        weight_decay: float = 1e-4,
        use_amp: bool = True
    ):
        self.model = model.to(device)
        self.device = device
        self.use_amp = use_amp
        
        # Different learning rates for different components
        sentence_bert_params = []
        other_params = []
        
        for name, param in model.named_parameters():
            if 'sentence_bert_encoder' in name:
                sentence_bert_params.append(param)
            else:
                other_params.append(param)
        
        param_groups = [
            {'params': sentence_bert_params, 'lr': sentence_bert_lr},
            {'params': other_params, 'lr': learning_rate}
        ]
        
        self.optimizer = torch.optim.AdamW(
            param_groups,
            weight_decay=weight_decay,
            betas=(0.9, 0.999)
        )
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=100, T_mult=2, eta_min=1e-7
        )
        
        # Mixed precision
        if use_amp:
            self.scaler = torch.cuda.amp.GradScaler()
        
        # Loss functions
        self.similarity_loss = nn.BCEWithLogitsLoss()
        self.content_loss = nn.MSELoss()
        self.rating_loss = nn.MSELoss()
        self.semantic_loss = nn.MSELoss()
    
    def train_step(self, batch: Dict[str, any]) -> Dict[str, float]:
        """Training step with multi-objective learning"""
        self.model.train()
        
        # Move tensors to device
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
                    user_content_texts=batch.get('user_content_texts'),
                    user_categorical=batch.get('user_categorical'),
                    user_numerical=batch.get('user_numerical'),
                    user_ids=batch.get('user_ids'),
                    item_content_texts=batch.get('item_content_texts'),
                    item_categorical=batch.get('item_categorical'),
                    item_numerical=batch.get('item_numerical'),
                    item_ids=batch.get('item_ids'),
                    return_all_outputs=True
                )
                
                # Multi-objective loss
                total_loss = 0
                losses = {}
                
                # Main similarity loss
                if 'similarity_labels' in batch:
                    sim_loss = self.similarity_loss(
                        predictions['similarity'], 
                        batch['similarity_labels']
                    )
                    losses['similarity'] = sim_loss.item()
                    total_loss += sim_loss
                
                # Content similarity loss
                if 'content_similarity_labels' in batch:
                    content_loss = self.content_loss(
                        predictions['content_similarity'],
                        batch['content_similarity_labels']
                    )
                    losses['content'] = content_loss.item()
                    total_loss += content_loss * 0.3
                
                # Rating prediction loss
                if 'rating' in predictions and 'rating_labels' in batch:
                    rating_loss = self.rating_loss(
                        predictions['rating'],
                        batch['rating_labels']
                    )
                    losses['rating'] = rating_loss.item()
                    total_loss += rating_loss * 0.5
                
                # Semantic matching loss
                if 'semantic_similarity_labels' in batch:
                    semantic_loss = self.semantic_loss(
                        predictions['semantic_similarity'],
                        batch['semantic_similarity_labels']
                    )
                    losses['semantic'] = semantic_loss.item()
                    total_loss += semantic_loss * 0.2
                
                losses['total'] = total_loss.item()
            
            self.scaler.scale(total_loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            # Non-AMP training (similar structure)
            predictions = self.model(
                user_content_texts=batch.get('user_content_texts'),
                user_categorical=batch.get('user_categorical'),
                user_numerical=batch.get('user_numerical'),
                user_ids=batch.get('user_ids'),
                item_content_texts=batch.get('item_content_texts'),
                item_categorical=batch.get('item_categorical'),
                item_numerical=batch.get('item_numerical'),
                item_ids=batch.get('item_ids'),
                return_all_outputs=True
            )
            
            total_loss = 0
            losses = {}
            
            if 'similarity_labels' in batch:
                sim_loss = self.similarity_loss(
                    predictions['similarity'], 
                    batch['similarity_labels']
                )
                losses['similarity'] = sim_loss.item()
                total_loss += sim_loss
            
            if 'rating' in predictions and 'rating_labels' in batch:
                rating_loss = self.rating_loss(
                    predictions['rating'],
                    batch['rating_labels']
                )
                losses['rating'] = rating_loss.item()
                total_loss += rating_loss * 0.5
            
            losses['total'] = total_loss.item()
            
            total_loss.backward()
            self.optimizer.step()
        
        self.scheduler.step()
        
        return losses