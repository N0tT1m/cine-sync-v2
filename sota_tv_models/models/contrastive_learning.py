"""
Contrastive Learning TV Encoder (CL-TV)
Self-supervised learning for TV show representations using contrastive learning
Optimized for RTX 4090 (24GB VRAM)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import RobertaModel, RobertaTokenizer
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import random

class DataAugmentation:
    """Data augmentation strategies for TV show metadata"""
    
    def __init__(self, 
                 text_dropout_prob: float = 0.1,
                 genre_dropout_prob: float = 0.2,
                 actor_dropout_prob: float = 0.3,
                 temporal_noise_std: float = 0.1):
        self.text_dropout_prob = text_dropout_prob
        self.genre_dropout_prob = genre_dropout_prob
        self.actor_dropout_prob = actor_dropout_prob
        self.temporal_noise_std = temporal_noise_std
    
    def augment_text(self, text: str) -> str:
        """Apply text-level augmentations"""
        words = text.split()
        
        # Random word dropout
        if len(words) > 3:
            num_drop = int(len(words) * self.text_dropout_prob)
            if num_drop > 0:
                drop_indices = random.sample(range(len(words)), num_drop)
                words = [word for i, word in enumerate(words) if i not in drop_indices]
        
        # Random word order permutation (only for short segments)
        if len(words) > 5:
            # Randomly shuffle a small segment
            start_idx = random.randint(0, max(0, len(words) - 4))
            end_idx = min(start_idx + 3, len(words))
            segment = words[start_idx:end_idx]
            random.shuffle(segment)
            words[start_idx:end_idx] = segment
        
        return ' '.join(words)
    
    def augment_genres(self, genres: List[int], max_genres: int = 5) -> List[int]:
        """Apply genre-level augmentations"""
        if not genres:
            return genres
        
        # Random genre dropout
        if len(genres) > 1:
            num_keep = max(1, int(len(genres) * (1 - self.genre_dropout_prob)))
            if num_keep < len(genres):
                genres = random.sample(genres, num_keep)
        
        return genres[:max_genres]  # Limit to max_genres
    
    def augment_actors(self, actors: List[int], max_actors: int = 10) -> List[int]:
        """Apply actor-level augmentations"""
        if not actors:
            return actors
        
        # Random actor dropout
        if len(actors) > 2:
            num_keep = max(2, int(len(actors) * (1 - self.actor_dropout_prob)))
            if num_keep < len(actors):
                actors = random.sample(actors, num_keep)
        
        return actors[:max_actors]  # Limit to max_actors
    
    def augment_numerical(self, features: torch.Tensor) -> torch.Tensor:
        """Apply numerical feature augmentations"""
        # Add small random noise
        noise = torch.randn_like(features) * self.temporal_noise_std
        return features + noise
    
    def create_positive_pair(self, 
                           text: str,
                           genres: List[int],
                           actors: List[int],
                           numerical_features: torch.Tensor) -> Tuple[str, List[int], List[int], torch.Tensor]:
        """Create a positive augmented pair"""
        aug_text = self.augment_text(text)
        aug_genres = self.augment_genres(genres.copy() if isinstance(genres, list) else genres.tolist())
        aug_actors = self.augment_actors(actors.copy() if isinstance(actors, list) else actors.tolist())
        aug_numerical = self.augment_numerical(numerical_features.clone())
        
        return aug_text, aug_genres, aug_actors, aug_numerical

class TVEncoder(nn.Module):
    """TV show encoder for contrastive learning"""
    
    def __init__(self,
                 vocab_sizes: Dict[str, int],
                 embed_dim: int = 768,
                 hidden_dim: int = 1024,
                 num_layers: int = 4,
                 dropout: float = 0.1,
                 use_pretrained_text: bool = True):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        
        # Text encoder
        if use_pretrained_text:
            self.text_encoder = RobertaModel.from_pretrained('roberta-base')
            # Freeze some layers to reduce memory
            for param in self.text_encoder.embeddings.parameters():
                param.requires_grad = False
            for layer in self.text_encoder.encoder.layer[:6]:
                for param in layer.parameters():
                    param.requires_grad = False
        else:
            # Lightweight text encoder
            self.text_encoder = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=embed_dim,
                    nhead=8,
                    dim_feedforward=hidden_dim,
                    dropout=dropout,
                    activation='gelu',
                    batch_first=True
                ),
                num_layers=num_layers
            )
            self.text_embedding = nn.Embedding(vocab_sizes.get('text_vocab', 30000), embed_dim)
        
        # Categorical feature embeddings
        self.categorical_embeddings = nn.ModuleDict()
        for feature_name, vocab_size in vocab_sizes.items():
            if feature_name != 'text_vocab':
                self.categorical_embeddings[feature_name] = nn.Embedding(
                    vocab_size, embed_dim // 4, padding_idx=0
                )
        
        # Numerical feature projection
        self.numerical_projection = nn.Sequential(
            nn.Linear(10, embed_dim // 2),  # Adjust based on number of numerical features
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, embed_dim // 2)
        )
        
        # Feature fusion
        fusion_input_dim = embed_dim  # Text
        fusion_input_dim += len(self.categorical_embeddings) * (embed_dim // 4)  # Categorical
        fusion_input_dim += embed_dim // 2  # Numerical
        
        self.fusion_layers = nn.Sequential(
            nn.Linear(fusion_input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.LayerNorm(embed_dim)
        )
        
        # Projection head for contrastive learning
        self.projection_head = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim // 2),
            nn.LayerNorm(embed_dim // 2)
        )
        
        self.use_pretrained_text = use_pretrained_text
    
    def encode_text(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Encode text features"""
        if self.use_pretrained_text:
            outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
            return outputs.last_hidden_state[:, 0, :]  # [CLS] token
        else:
            embeddings = self.text_embedding(input_ids)
            # Simple mean pooling with attention mask
            mask_expanded = attention_mask.unsqueeze(-1).expand(embeddings.size())
            sum_embeddings = torch.sum(embeddings * mask_expanded, 1)
            sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
            return sum_embeddings / sum_mask
    
    def encode_categorical(self, categorical_features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Encode categorical features"""
        categorical_embeds = []
        
        for feature_name, values in categorical_features.items():
            if feature_name in self.categorical_embeddings:
                embed = self.categorical_embeddings[feature_name](values)
                
                # Handle multi-value features (e.g., genres, actors)
                if embed.dim() == 3:  # [batch, num_values, embed_dim]
                    embed = embed.mean(dim=1)  # Average pooling
                
                categorical_embeds.append(embed)
        
        if categorical_embeds:
            return torch.cat(categorical_embeds, dim=1)
        else:
            return torch.zeros(categorical_features[list(categorical_features.keys())[0]].size(0), 
                             0, device=next(self.parameters()).device)
    
    def forward(self,
                input_ids: torch.Tensor,
                attention_mask: torch.Tensor,
                categorical_features: Dict[str, torch.Tensor],
                numerical_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        
        # Encode different modalities
        text_features = self.encode_text(input_ids, attention_mask)
        categorical_features_encoded = self.encode_categorical(categorical_features)
        numerical_features_encoded = self.numerical_projection(numerical_features)
        
        # Concatenate all features
        all_features = [text_features]
        if categorical_features_encoded.size(1) > 0:
            all_features.append(categorical_features_encoded)
        all_features.append(numerical_features_encoded)
        
        fused_features = torch.cat(all_features, dim=1)
        
        # Apply fusion layers
        show_embedding = self.fusion_layers(fused_features)
        
        # Apply projection head for contrastive learning
        projected_embedding = self.projection_head(show_embedding)
        
        return {
            'show_embedding': show_embedding,
            'projected_embedding': projected_embedding
        }

class ContrastiveTVModel(nn.Module):
    """
    Complete contrastive learning model for TV shows
    
    Uses SimCLR-style contrastive learning with:
    - Data augmentation for positive pairs
    - Hard negative mining
    - Temperature-scaled cosine similarity
    - Multi-positive contrastive learning
    """
    
    def __init__(self,
                 vocab_sizes: Dict[str, int],
                 embed_dim: int = 768,
                 hidden_dim: int = 1024,
                 num_layers: int = 4,
                 dropout: float = 0.1,
                 temperature: float = 0.05,
                 use_hard_negatives: bool = True):
        super().__init__()
        
        self.temperature = temperature
        self.use_hard_negatives = use_hard_negatives
        
        # TV encoder
        self.encoder = TVEncoder(
            vocab_sizes=vocab_sizes,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout
        )
        
        # Data augmentation
        self.augmentation = DataAugmentation()
        
        # Hard negative mining parameters
        self.hard_negative_ratio = 0.3  # Ratio of hard negatives to use
        
    def compute_contrastive_loss(self,
                                query_embeddings: torch.Tensor,
                                positive_embeddings: torch.Tensor,
                                negative_embeddings: torch.Tensor,
                                use_hard_negatives: bool = True) -> torch.Tensor:
        """Compute contrastive loss with optional hard negative mining"""
        
        batch_size = query_embeddings.size(0)
        
        # Normalize embeddings
        query_norm = F.normalize(query_embeddings, dim=1)
        pos_norm = F.normalize(positive_embeddings, dim=1)
        neg_norm = F.normalize(negative_embeddings, dim=1)
        
        # Positive similarities
        pos_sim = torch.sum(query_norm * pos_norm, dim=1) / self.temperature
        
        # Negative similarities
        neg_sim = torch.matmul(query_norm, neg_norm.T) / self.temperature
        
        # Hard negative mining
        if use_hard_negatives and self.training:
            # Select hardest negatives (highest similarity)
            num_hard = int(neg_sim.size(1) * self.hard_negative_ratio)
            if num_hard > 0:
                hard_neg_sim, _ = torch.topk(neg_sim, num_hard, dim=1)
                # Combine with random negatives
                num_random = neg_sim.size(1) - num_hard
                if num_random > 0:
                    random_indices = torch.randperm(neg_sim.size(1), device=neg_sim.device)[:num_random]
                    random_neg_sim = neg_sim[:, random_indices]
                    neg_sim = torch.cat([hard_neg_sim, random_neg_sim], dim=1)
                else:
                    neg_sim = hard_neg_sim
        
        # InfoNCE loss
        logits = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1)
        labels = torch.zeros(batch_size, dtype=torch.long, device=query_embeddings.device)
        
        return F.cross_entropy(logits, labels)
    
    def forward(self,
                # Anchor samples
                anchor_input_ids: torch.Tensor,
                anchor_attention_mask: torch.Tensor,
                anchor_categorical: Dict[str, torch.Tensor],
                anchor_numerical: torch.Tensor,
                # Positive samples
                positive_input_ids: torch.Tensor,
                positive_attention_mask: torch.Tensor,
                positive_categorical: Dict[str, torch.Tensor],
                positive_numerical: torch.Tensor,
                # Negative samples
                negative_input_ids: torch.Tensor,
                negative_attention_mask: torch.Tensor,
                negative_categorical: Dict[str, torch.Tensor],
                negative_numerical: torch.Tensor) -> Dict[str, torch.Tensor]:
        
        # Encode anchor samples
        anchor_outputs = self.encoder(
            anchor_input_ids, anchor_attention_mask,
            anchor_categorical, anchor_numerical
        )
        
        # Encode positive samples
        positive_outputs = self.encoder(
            positive_input_ids, positive_attention_mask,
            positive_categorical, positive_numerical
        )
        
        # Encode negative samples
        negative_outputs = self.encoder(
            negative_input_ids, negative_attention_mask,
            negative_categorical, negative_numerical
        )
        
        # Compute contrastive loss
        contrastive_loss = self.compute_contrastive_loss(
            anchor_outputs['projected_embedding'],
            positive_outputs['projected_embedding'],
            negative_outputs['projected_embedding'],
            self.use_hard_negatives
        )
        
        # Compute additional metrics
        anchor_pos_sim = F.cosine_similarity(
            anchor_outputs['projected_embedding'],
            positive_outputs['projected_embedding'],
            dim=1
        )
        
        anchor_neg_sim = F.cosine_similarity(
            anchor_outputs['projected_embedding'].unsqueeze(1),
            negative_outputs['projected_embedding'].unsqueeze(0),
            dim=2
        ).mean(dim=1)
        
        return {
            'contrastive_loss': contrastive_loss,
            'anchor_embedding': anchor_outputs['show_embedding'],
            'positive_embedding': positive_outputs['show_embedding'],
            'negative_embedding': negative_outputs['show_embedding'],
            'anchor_pos_similarity': anchor_pos_sim,
            'anchor_neg_similarity': anchor_neg_sim,
            'projected_anchor': anchor_outputs['projected_embedding'],
            'projected_positive': positive_outputs['projected_embedding'],
            'projected_negative': negative_outputs['projected_embedding']
        }
    
    def get_show_embedding(self,
                          input_ids: torch.Tensor,
                          attention_mask: torch.Tensor,
                          categorical_features: Dict[str, torch.Tensor],
                          numerical_features: torch.Tensor) -> torch.Tensor:
        """Get embedding for a single show"""
        with torch.no_grad():
            outputs = self.encoder(input_ids, attention_mask, categorical_features, numerical_features)
            return outputs['show_embedding']
    
    def find_similar_shows(self,
                          query_embedding: torch.Tensor,
                          candidate_embeddings: torch.Tensor,
                          top_k: int = 10) -> Tuple[torch.Tensor, torch.Tensor]:
        """Find most similar shows to query"""
        with torch.no_grad():
            query_norm = F.normalize(query_embedding.unsqueeze(0), dim=1)
            candidate_norm = F.normalize(candidate_embeddings, dim=1)
            
            similarities = torch.matmul(query_norm, candidate_norm.T).squeeze(0)
            top_similarities, top_indices = torch.topk(similarities, min(top_k, len(similarities)))
            
            return top_indices, top_similarities

class MultiPositiveContrastiveLoss(nn.Module):
    """Advanced contrastive loss with multiple positives per anchor"""
    
    def __init__(self, temperature: float = 0.05, alpha: float = 0.1):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha  # Weight for multiple positives
    
    def forward(self,
                anchor_embeddings: torch.Tensor,
                positive_embeddings: torch.Tensor,
                negative_embeddings: torch.Tensor,
                num_positives_per_anchor: List[int]) -> torch.Tensor:
        """
        Compute contrastive loss with multiple positives per anchor
        
        Args:
            anchor_embeddings: [batch_size, embed_dim]
            positive_embeddings: [total_positives, embed_dim]
            negative_embeddings: [total_negatives, embed_dim]
            num_positives_per_anchor: List indicating number of positives for each anchor
        """
        
        # Normalize embeddings
        anchor_norm = F.normalize(anchor_embeddings, dim=1)
        pos_norm = F.normalize(positive_embeddings, dim=1)
        neg_norm = F.normalize(negative_embeddings, dim=1)
        
        total_loss = 0
        pos_start_idx = 0
        
        for i, num_pos in enumerate(num_positives_per_anchor):
            if num_pos == 0:
                continue
                
            # Get anchor embedding
            anchor = anchor_norm[i].unsqueeze(0)  # [1, embed_dim]
            
            # Get positive embeddings for this anchor
            pos_end_idx = pos_start_idx + num_pos
            positives = pos_norm[pos_start_idx:pos_end_idx]  # [num_pos, embed_dim]
            
            # Compute similarities
            pos_sim = torch.matmul(anchor, positives.T) / self.temperature  # [1, num_pos]
            neg_sim = torch.matmul(anchor, neg_norm.T) / self.temperature   # [1, num_neg]
            
            # For multiple positives, we can use different strategies:
            # Strategy 1: Treat each positive as a separate positive example
            for j in range(num_pos):
                pos_j = pos_sim[0, j].unsqueeze(0)  # [1]
                logits = torch.cat([pos_j.unsqueeze(0), neg_sim], dim=1)  # [1, 1 + num_neg]
                labels = torch.zeros(1, dtype=torch.long, device=anchor.device)
                loss_j = F.cross_entropy(logits, labels)
                total_loss += loss_j
            
            # Strategy 2: Use soft positive targets (weighted combination)
            if num_pos > 1:
                # Weight positives by their similarity to anchor
                pos_weights = F.softmax(pos_sim, dim=1)  # [1, num_pos]
                weighted_pos_sim = torch.sum(pos_weights * pos_sim, dim=1)  # [1]
                
                logits = torch.cat([weighted_pos_sim.unsqueeze(0), neg_sim], dim=1)
                labels = torch.zeros(1, dtype=torch.long, device=anchor.device)
                weighted_loss = F.cross_entropy(logits, labels)
                total_loss += self.alpha * weighted_loss
            
            pos_start_idx = pos_end_idx
        
        return total_loss / len(num_positives_per_anchor)

# Configuration for RTX 4090 optimization
def get_contrastive_config():
    """Optimized configuration for contrastive learning on RTX 4090"""
    return {
        'embed_dim': 768,
        'hidden_dim': 1536,  # Larger for 4090
        'num_layers': 6,
        'dropout': 0.1,
        'temperature': 0.05,
        'batch_size': 128,   # Large batch size important for contrastive learning
        'num_negatives': 63, # batch_size - 1 in-batch negatives
        'use_hard_negatives': True,
        'hard_negative_ratio': 0.3,
        'learning_rate': 1e-4,
        'weight_decay': 1e-5,
        'warmup_steps': 2000,
        'use_mixed_precision': True,
        'gradient_accumulation_steps': 2,
        'max_grad_norm': 1.0,
    }