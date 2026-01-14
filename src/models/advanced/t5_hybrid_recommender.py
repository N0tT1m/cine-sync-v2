# T5-Enhanced Hybrid Recommendation Model for CineSync v2
# Integrates T5 foundation model for superior content understanding
# Combines collaborative filtering with state-of-the-art NLP capabilities

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np
import math
from transformers import T5Model, T5Tokenizer, T5ForConditionalGeneration
from transformers import AutoModel, AutoTokenizer
import re


class T5ContentEncoder(nn.Module):
    """T5-based content encoder for movie/TV show descriptions and metadata"""
    
    def __init__(
        self,
        model_name: str = "t5-small",
        output_dim: int = 512,
        max_length: int = 512,
        freeze_encoder: bool = True,
        fine_tune_layers: int = 2,
        use_conditional_generation: bool = False
    ):
        super(T5ContentEncoder, self).__init__()
        
        self.model_name = model_name
        self.output_dim = output_dim
        self.max_length = max_length
        self.use_conditional_generation = use_conditional_generation
        
        # Load T5 model and tokenizer
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        
        if use_conditional_generation:
            self.t5_model = T5ForConditionalGeneration.from_pretrained(model_name)
            self.encoder = self.t5_model.encoder
        else:
            self.t5_model = T5Model.from_pretrained(model_name)
            self.encoder = self.t5_model.encoder
        
        # Get hidden dimension
        self.hidden_dim = self.encoder.config.d_model
        
        # Freeze encoder layers if specified
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
            
            # Only fine-tune last few layers
            if fine_tune_layers > 0:
                encoder_layers = list(self.encoder.block)
                for layer in encoder_layers[-fine_tune_layers:]:
                    for param in layer.parameters():
                        param.requires_grad = True
        
        # Content-aware projection layers
        self.content_projector = nn.Sequential(
            nn.Linear(self.hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        
        # Multi-aspect content understanding
        self.genre_projector = nn.Sequential(
            nn.Linear(self.hidden_dim, output_dim // 4),
            nn.GELU(),
            nn.Linear(output_dim // 4, 20)  # 20 genres
        )
        
        self.sentiment_projector = nn.Sequential(
            nn.Linear(self.hidden_dim, output_dim // 4),
            nn.GELU(),
            nn.Linear(output_dim // 4, 3)  # Positive, Neutral, Negative
        )
        
        self.theme_projector = nn.Sequential(
            nn.Linear(self.hidden_dim, output_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        
        # Attention pooling for variable length sequences
        self.attention_pooling = nn.MultiheadAttention(
            embed_dim=self.hidden_dim,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
        # Query vector for attention pooling
        self.query_vector = nn.Parameter(torch.randn(1, 1, self.hidden_dim))
    
    def preprocess_text(self, texts: List[str], prefix: str = "") -> Dict[str, torch.Tensor]:
        """Preprocess text inputs for T5"""
        processed_texts = []
        
        for text in texts:
            # Clean and format text
            cleaned_text = re.sub(r'\s+', ' ', text.strip())
            
            # Add task-specific prefix for T5
            if prefix:
                processed_text = f"{prefix}: {cleaned_text}"
            else:
                processed_text = cleaned_text
            
            processed_texts.append(processed_text)
        
        # Tokenize
        encoding = self.tokenizer(
            processed_texts,
            max_length=self.max_length,
            padding=True,
            truncation=True,
            return_tensors='pt'
        )
        
        return encoding
    
    def encode_content(
        self,
        texts: List[str],
        content_type: str = "movie description"
    ) -> Dict[str, torch.Tensor]:
        """
        Encode content using T5 with task-specific prompts
        
        Args:
            texts: List of text descriptions
            content_type: Type of content for task-specific encoding
            
        Returns:
            Dictionary with encoded features
        """
        # Create task-specific prefix
        if content_type == "movie description":
            prefix = "Analyze movie content"
        elif content_type == "user profile":
            prefix = "Understand user preferences"
        elif content_type == "genre classification":
            prefix = "Classify genre"
        else:
            prefix = "Encode content"
        
        # Preprocess and tokenize
        inputs = self.preprocess_text(texts, prefix)
        
        # Move to device
        device = next(self.parameters()).device
        input_ids = inputs['input_ids'].to(device)
        attention_mask = inputs['attention_mask'].to(device)
        
        # Encode with T5
        with torch.set_grad_enabled(self.training):
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            hidden_states = encoder_outputs.last_hidden_state  # [batch, seq_len, hidden_dim]
        
        # Attention pooling for variable length sequences
        batch_size = hidden_states.size(0)
        query = self.query_vector.expand(batch_size, -1, -1)
        
        pooled_output, attention_weights = self.attention_pooling(
            query, hidden_states, hidden_states,
            key_padding_mask=~attention_mask.bool()
        )
        pooled_output = pooled_output.squeeze(1)  # [batch, hidden_dim]
        
        # Multi-aspect projections
        content_features = self.content_projector(pooled_output)
        genre_logits = self.genre_projector(pooled_output)
        sentiment_logits = self.sentiment_projector(pooled_output)
        theme_features = self.theme_projector(pooled_output)
        
        return {
            'content_features': content_features,
            'genre_logits': genre_logits,
            'sentiment_logits': sentiment_logits,
            'theme_features': theme_features,
            'raw_hidden_states': hidden_states,
            'attention_weights': attention_weights,
            'pooled_features': pooled_output
        }
    
    def generate_content_summary(
        self,
        texts: List[str],
        max_new_tokens: int = 50
    ) -> List[str]:
        """Generate content summaries using T5's generation capabilities"""
        if not self.use_conditional_generation:
            raise ValueError("Content generation requires T5ForConditionalGeneration model")
        
        # Preprocess with summarization prefix
        inputs = self.preprocess_text(texts, "summarize")
        
        device = next(self.parameters()).device
        input_ids = inputs['input_ids'].to(device)
        attention_mask = inputs['attention_mask'].to(device)
        
        # Generate summaries
        with torch.no_grad():
            generated_ids = self.t5_model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id
            )
        
        # Decode generated text
        summaries = []
        for generated_id in generated_ids:
            summary = self.tokenizer.decode(generated_id, skip_special_tokens=True)
            summaries.append(summary)
        
        return summaries


class ContentAwareCollaborativeLayer(nn.Module):
    """Collaborative filtering layer enhanced with content understanding"""
    
    def __init__(
        self,
        num_users: int,
        num_items: int,
        embedding_dim: int,
        content_dim: int,
        dropout: float = 0.2
    ):
        super(ContentAwareCollaborativeLayer, self).__init__()
        
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.content_dim = content_dim
        
        # Traditional collaborative embeddings
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        
        # Content-aware transformations
        self.content_to_collab = nn.Sequential(
            nn.Linear(content_dim, embedding_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(embedding_dim)
        )
        
        # Fusion mechanisms
        self.user_content_fusion = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        self.item_content_fusion = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Cross-attention between collaborative and content features
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )
        
        # Gating mechanism for content importance
        self.content_gate = nn.Sequential(
            nn.Linear(embedding_dim + content_dim, 1),
            nn.Sigmoid()
        )
    
    def forward(
        self,
        user_ids: torch.Tensor,
        item_ids: torch.Tensor,
        user_content_features: Optional[torch.Tensor] = None,
        item_content_features: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with content-aware collaborative filtering
        
        Args:
            user_ids: User IDs [batch_size]
            item_ids: Item IDs [batch_size]
            user_content_features: User content features [batch_size, content_dim]
            item_content_features: Item content features [batch_size, content_dim]
            
        Returns:
            Dictionary with collaborative and content-aware embeddings
        """
        # Get base collaborative embeddings
        user_collab = self.user_embedding(user_ids)
        item_collab = self.item_embedding(item_ids)
        
        # Process content features if available
        if user_content_features is not None:
            user_content_emb = self.content_to_collab(user_content_features)
            
            # Cross-attention between collaborative and content
            user_combined = torch.stack([user_collab.unsqueeze(1), user_content_emb.unsqueeze(1)], dim=1)
            user_attended, _ = self.cross_attention(user_combined, user_combined, user_combined)
            user_attended = user_attended.mean(dim=1)
            
            # Gating mechanism
            gate_input = torch.cat([user_collab, user_content_features], dim=-1)
            content_importance = self.content_gate(gate_input)
            
            # Fuse collaborative and content embeddings
            user_fused = self.user_content_fusion(
                torch.cat([user_collab, user_content_emb], dim=-1)
            )
            user_final = user_collab + content_importance * user_fused
        else:
            user_final = user_collab
        
        # Similar processing for items
        if item_content_features is not None:
            item_content_emb = self.content_to_collab(item_content_features)
            
            item_combined = torch.stack([item_collab.unsqueeze(1), item_content_emb.unsqueeze(1)], dim=1)
            item_attended, _ = self.cross_attention(item_combined, item_combined, item_combined)
            item_attended = item_attended.mean(dim=1)
            
            gate_input = torch.cat([item_collab, item_content_features], dim=-1)
            content_importance = self.content_gate(gate_input)
            
            item_fused = self.item_content_fusion(
                torch.cat([item_collab, item_content_emb], dim=-1)
            )
            item_final = item_collab + content_importance * item_fused
        else:
            item_final = item_collab
        
        return {
            'user_embeddings': user_final,
            'item_embeddings': item_final,
            'user_collab': user_collab,
            'item_collab': item_collab
        }


class T5HybridRecommender(nn.Module):
    """
    T5-Enhanced Hybrid Recommendation Model
    
    Combines T5's language understanding with collaborative filtering
    for superior content-aware recommendations
    """
    
    def __init__(
        self,
        num_users: int,
        num_items: int,
        t5_model_name: str = "t5-small",
        embedding_dim: int = 512,
        content_dim: int = 512,
        hidden_dims: List[int] = [1024, 512, 256],
        dropout: float = 0.2,
        use_t5_generation: bool = False,
        freeze_t5_encoder: bool = True,
        fine_tune_t5_layers: int = 2
    ):
        super(T5HybridRecommender, self).__init__()
        
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.content_dim = content_dim
        
        # T5 content encoder
        self.t5_encoder = T5ContentEncoder(
            model_name=t5_model_name,
            output_dim=content_dim,
            freeze_encoder=freeze_t5_encoder,
            fine_tune_layers=fine_tune_t5_layers,
            use_conditional_generation=use_t5_generation
        )
        
        # Content-aware collaborative filtering
        self.collaborative_layer = ContentAwareCollaborativeLayer(
            num_users=num_users,
            num_items=num_items,
            embedding_dim=embedding_dim,
            content_dim=content_dim,
            dropout=dropout
        )
        
        # Deep neural layers for complex interactions
        self.interaction_layers = nn.ModuleList()
        input_dim = embedding_dim * 2  # User + Item embeddings
        
        for hidden_dim in hidden_dims:
            self.interaction_layers.append(nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            ))
            input_dim = hidden_dim
        
        # Multi-task prediction heads
        self.rating_head = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(input_dim // 2, 1)
        )
        
        self.genre_head = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(input_dim // 2, 20)  # 20 genres
        )
        
        self.sentiment_head = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(input_dim // 2, 3)  # Positive, Neutral, Negative
        )
        
        # Content similarity head
        self.content_similarity_head = nn.Sequential(
            nn.Linear(content_dim * 2, content_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(content_dim, 1)
        )
        
        # Preference matching head
        self.preference_matching_head = nn.Sequential(
            nn.Linear(embedding_dim * 2 + content_dim * 2, embedding_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim, 1)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.xavier_normal_(module.weight)
    
    def encode_content(
        self,
        user_texts: Optional[List[str]] = None,
        item_texts: Optional[List[str]] = None
    ) -> Dict[str, torch.Tensor]:
        """Encode textual content using T5"""
        results = {}
        
        if user_texts is not None:
            user_encoding = self.t5_encoder.encode_content(
                user_texts, content_type="user profile"
            )
            results['user_content'] = user_encoding
        
        if item_texts is not None:
            item_encoding = self.t5_encoder.encode_content(
                item_texts, content_type="movie description"
            )
            results['item_content'] = item_encoding
        
        return results
    
    def forward(
        self,
        user_ids: torch.Tensor,
        item_ids: torch.Tensor,
        user_texts: Optional[List[str]] = None,
        item_texts: Optional[List[str]] = None,
        user_content_features: Optional[torch.Tensor] = None,
        item_content_features: Optional[torch.Tensor] = None,
        return_all_outputs: bool = False
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass for T5-enhanced hybrid recommendation
        
        Args:
            user_ids: User IDs [batch_size]
            item_ids: Item IDs [batch_size]
            user_texts: List of user description texts
            item_texts: List of item description texts
            user_content_features: Pre-computed user content features
            item_content_features: Pre-computed item content features
            return_all_outputs: Whether to return all intermediate outputs
            
        Returns:
            Rating predictions or full output dictionary
        """
        batch_size = user_ids.size(0)
        
        # Encode textual content if provided
        if user_texts is not None or item_texts is not None:
            content_encoding = self.encode_content(user_texts, item_texts)
            
            if user_texts is not None:
                user_content_features = content_encoding['user_content']['content_features']
            if item_texts is not None:
                item_content_features = content_encoding['item_content']['content_features']
        
        # Content-aware collaborative filtering
        collab_outputs = self.collaborative_layer(
            user_ids=user_ids,
            item_ids=item_ids,
            user_content_features=user_content_features,
            item_content_features=item_content_features
        )
        
        user_embeddings = collab_outputs['user_embeddings']
        item_embeddings = collab_outputs['item_embeddings']
        
        # Combine user and item embeddings for interaction modeling
        combined_embeddings = torch.cat([user_embeddings, item_embeddings], dim=-1)
        
        # Deep interaction layers
        for layer in self.interaction_layers:
            combined_embeddings = layer(combined_embeddings)
        
        # Basic rating prediction
        rating_pred = self.rating_head(combined_embeddings).squeeze()
        rating_pred = torch.sigmoid(rating_pred) * 5.0  # Scale to 0-5
        
        if not return_all_outputs:
            return rating_pred
        
        # Multi-task predictions
        outputs = {
            'rating_pred': rating_pred,
            'user_embeddings': user_embeddings,
            'item_embeddings': item_embeddings,
            'combined_embeddings': combined_embeddings
        }
        
        # Genre prediction
        genre_logits = self.genre_head(combined_embeddings)
        outputs['genre_pred'] = F.softmax(genre_logits, dim=-1)
        
        # Sentiment prediction
        sentiment_logits = self.sentiment_head(combined_embeddings)
        outputs['sentiment_pred'] = F.softmax(sentiment_logits, dim=-1)
        
        # Content-based similarity (if content features available)
        if user_content_features is not None and item_content_features is not None:
            content_combined = torch.cat([user_content_features, item_content_features], dim=-1)
            content_similarity = self.content_similarity_head(content_combined).squeeze()
            outputs['content_similarity'] = torch.sigmoid(content_similarity)
            
            # Preference matching combining collaborative and content features
            preference_input = torch.cat([
                user_embeddings, item_embeddings,
                user_content_features, item_content_features
            ], dim=-1)
            preference_score = self.preference_matching_head(preference_input).squeeze()
            outputs['preference_score'] = torch.sigmoid(preference_score)
        
        return outputs
    
    def get_recommendations(
        self,
        user_id: int,
        item_texts: List[str],
        k: int = 10,
        exclude_items: Optional[List[int]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Get top-k recommendations with T5 content understanding
        
        Args:
            user_id: User ID
            item_texts: List of item descriptions
            k: Number of recommendations
            exclude_items: Items to exclude
            
        Returns:
            Tuple of (top_k_items, top_k_scores, additional_info)
        """
        self.eval()
        with torch.no_grad():
            num_items = len(item_texts)
            device = next(self.parameters()).device
            
            # Create batch inputs
            user_ids = torch.full((num_items,), user_id, device=device)
            item_ids = torch.arange(num_items, device=device)
            
            # Get predictions
            outputs = self.forward(
                user_ids=user_ids,
                item_ids=item_ids,
                item_texts=item_texts,
                return_all_outputs=True
            )
            
            scores = outputs['rating_pred']
            
            # Exclude items if specified
            if exclude_items:
                scores[exclude_items] = -float('inf')
            
            # Get top-k
            top_k_scores, top_k_indices = torch.topk(scores, k)
            
            # Additional information
            additional_info = {
                'genre_predictions': outputs['genre_pred'][top_k_indices],
                'sentiment_predictions': outputs['sentiment_pred'][top_k_indices]
            }
            
            if 'content_similarity' in outputs:
                additional_info['content_similarity'] = outputs['content_similarity'][top_k_indices]
            if 'preference_score' in outputs:
                additional_info['preference_score'] = outputs['preference_score'][top_k_indices]
            
            return top_k_indices, top_k_scores, additional_info
    
    def explain_recommendation(
        self,
        user_id: int,
        item_id: int,
        user_text: Optional[str] = None,
        item_text: Optional[str] = None
    ) -> Dict[str, Any]:
        """Generate explanation for a recommendation using T5's capabilities"""
        self.eval()
        with torch.no_grad():
            device = next(self.parameters()).device
            
            user_ids = torch.tensor([user_id], device=device)
            item_ids = torch.tensor([item_id], device=device)
            
            user_texts = [user_text] if user_text else None
            item_texts = [item_text] if item_text else None
            
            outputs = self.forward(
                user_ids=user_ids,
                item_ids=item_ids,
                user_texts=user_texts,
                item_texts=item_texts,
                return_all_outputs=True
            )
            
            explanation = {
                'predicted_rating': outputs['rating_pred'].item(),
                'genre_preferences': outputs['genre_pred'].cpu().numpy(),
                'sentiment_match': outputs['sentiment_pred'].cpu().numpy(),
                'collaborative_strength': torch.norm(outputs['user_embeddings']).item(),
                'content_relevance': outputs.get('content_similarity', torch.tensor(0.0)).item(),
                'overall_preference': outputs.get('preference_score', torch.tensor(0.0)).item()
            }
            
            return explanation


class T5HybridTrainer:
    """Trainer for T5-enhanced hybrid recommendation model"""
    
    def __init__(
        self,
        model: T5HybridRecommender,
        device: torch.device,
        learning_rate: float = 1e-4,
        t5_learning_rate: float = 1e-5,
        weight_decay: float = 1e-4,
        use_amp: bool = True,
        gradient_clip_val: float = 1.0
    ):
        self.model = model.to(device)
        self.device = device
        self.use_amp = use_amp
        self.gradient_clip_val = gradient_clip_val
        
        # Different learning rates for T5 and other components
        t5_params = []
        other_params = []
        
        for name, param in model.named_parameters():
            if 't5_encoder' in name:
                t5_params.append(param)
            else:
                other_params.append(param)
        
        param_groups = [
            {'params': t5_params, 'lr': t5_learning_rate},
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
        self.rating_loss = nn.MSELoss()
        self.genre_loss = nn.CrossEntropyLoss()
        self.sentiment_loss = nn.CrossEntropyLoss()
        self.similarity_loss = nn.BCELoss()
    
    def train_step(self, batch: Dict[str, Any]) -> Dict[str, float]:
        """Training step with multi-task learning"""
        self.model.train()
        
        # Move tensors to device
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                batch[key] = value.to(self.device)
        
        self.optimizer.zero_grad()
        
        if self.use_amp:
            with torch.cuda.amp.autocast():
                outputs = self.model(
                    user_ids=batch['user_ids'],
                    item_ids=batch['item_ids'],
                    user_texts=batch.get('user_texts'),
                    item_texts=batch.get('item_texts'),
                    return_all_outputs=True
                )
                
                # Multi-task loss computation
                total_loss = 0
                losses = {}
                
                # Rating prediction loss
                if 'ratings' in batch:
                    rating_loss = self.rating_loss(outputs['rating_pred'], batch['ratings'])
                    losses['rating'] = rating_loss.item()
                    total_loss += rating_loss
                
                # Genre prediction loss
                if 'genres' in batch:
                    genre_loss = self.genre_loss(
                        outputs['genre_pred'].view(-1, 20),
                        batch['genres'].view(-1)
                    )
                    losses['genre'] = genre_loss.item()
                    total_loss += 0.3 * genre_loss
                
                # Content similarity loss
                if 'content_similarity' in outputs and 'content_labels' in batch:
                    sim_loss = self.similarity_loss(
                        outputs['content_similarity'],
                        batch['content_labels']
                    )
                    losses['content_similarity'] = sim_loss.item()
                    total_loss += 0.2 * sim_loss
                
                losses['total'] = total_loss.item()
            
            self.scaler.scale(total_loss).backward()
            
            if self.gradient_clip_val > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_val)
            
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            outputs = self.model(
                user_ids=batch['user_ids'],
                item_ids=batch['item_ids'],
                user_texts=batch.get('user_texts'),
                item_texts=batch.get('item_texts'),
                return_all_outputs=True
            )
            
            total_loss = 0
            losses = {}
            
            if 'ratings' in batch:
                rating_loss = self.rating_loss(outputs['rating_pred'], batch['ratings'])
                losses['rating'] = rating_loss.item()
                total_loss += rating_loss
            
            losses['total'] = total_loss.item()
            
            total_loss.backward()
            
            if self.gradient_clip_val > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_val)
            
            self.optimizer.step()
        
        self.scheduler.step()
        
        return losses