"""
Multimodal Transformer TV Model (MTTV)
State-of-the-art TV recommendation using multimodal transformers
Optimized for RTX 4090 (24GB VRAM)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import RobertaModel, RobertaTokenizer
from torch.utils.checkpoint import checkpoint
import math
from typing import Dict, List, Optional, Tuple

class PositionalEncoding(nn.Module):
    """Positional encoding for temporal features"""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:x.size(0), :]

class MultiHeadCrossAttention(nn.Module):
    """Cross-attention between text and metadata features"""
    
    def __init__(self, d_model: int, n_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        
    def forward(self, query: torch.Tensor, key: torch.Tensor, 
                value: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size = query.size(0)
        residual = query
        
        # Linear transformations
        Q = self.w_q(query).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        
        # Attention
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            attention_scores.masked_fill_(mask == 0, -1e9)
            
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        context = torch.matmul(attention_weights, V)
        context = context.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model)
        
        output = self.w_o(context)
        return self.layer_norm(output + residual)

class MetadataEncoder(nn.Module):
    """Encoder for TV show metadata (genres, cast, network, etc.)"""
    
    def __init__(self, 
                 vocab_sizes: Dict[str, int],
                 embed_dim: int = 256,
                 hidden_dim: int = 512,
                 num_layers: int = 3,
                 dropout: float = 0.1):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        
        # Embedding layers for categorical features
        self.embeddings = nn.ModuleDict({
            name: nn.Embedding(vocab_size, embed_dim, padding_idx=0)
            for name, vocab_size in vocab_sizes.items()
        })
        
        # Numerical feature projection
        self.numerical_proj = nn.Linear(10, embed_dim)  # vote_count, vote_avg, seasons, episodes, etc.
        
        # Transformer encoder for metadata fusion
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=8,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(embed_dim)
        
        # Output projection
        self.output_proj = nn.Linear(embed_dim, 1024)  # Match RoBERTa-large dim
        
    def forward(self, 
                categorical_features: Dict[str, torch.Tensor],
                numerical_features: torch.Tensor) -> torch.Tensor:
        
        embeddings = []
        
        # Process categorical features
        for name, values in categorical_features.items():
            if name in self.embeddings:
                emb = self.embeddings[name](values)
                if emb.dim() == 3:  # Multi-value features (e.g., genres, cast)
                    emb = emb.mean(dim=1)  # Average pooling
                embeddings.append(emb)
        
        # Process numerical features
        num_emb = self.numerical_proj(numerical_features)
        embeddings.append(num_emb)
        
        # Stack and add positional encoding
        metadata_embeds = torch.stack(embeddings, dim=1)  # [batch, num_features, embed_dim]
        metadata_embeds = self.pos_encoding(metadata_embeds.transpose(0, 1)).transpose(0, 1)
        
        # Apply transformer
        encoded = self.transformer(metadata_embeds)
        
        # Global average pooling and projection
        pooled = encoded.mean(dim=1)
        return self.output_proj(pooled)

class MultimodalTransformerTV(nn.Module):
    """
    State-of-the-art Multimodal Transformer for TV Recommendation
    
    Combines:
    - RoBERTa for text understanding (plot, description)
    - Custom transformer for metadata fusion
    - Cross-attention between modalities
    - Contrastive learning objective
    """
    
    def __init__(self,
                 vocab_sizes: Dict[str, int],
                 num_shows: int,
                 embed_dim: int = 1024,
                 hidden_dim: int = 1024,
                 num_layers: int = 6,
                 num_heads: int = 16,
                 dropout: float = 0.1,
                 use_gradient_checkpointing: bool = True):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.use_gradient_checkpointing = use_gradient_checkpointing
        
        # Text encoder (RoBERTa)
        self.text_encoder = RobertaModel.from_pretrained('roberta-large')
        if use_gradient_checkpointing:
            self.text_encoder.gradient_checkpointing_enable()
        
        # Freeze early layers to save memory
        for param in self.text_encoder.embeddings.parameters():
            param.requires_grad = False
        for layer in self.text_encoder.encoder.layer[:12]:  # Freeze first 12 layers
            for param in layer.parameters():
                param.requires_grad = False
        
        # Metadata encoder
        self.metadata_encoder = MetadataEncoder(
            vocab_sizes=vocab_sizes,
            embed_dim=256,
            hidden_dim=512,
            num_layers=3,
            dropout=dropout
        )
        
        # Cross-attention layers
        self.text_to_meta_attention = MultiHeadCrossAttention(embed_dim, num_heads, dropout)
        self.meta_to_text_attention = MultiHeadCrossAttention(embed_dim, num_heads, dropout)
        
        # Fusion layers
        self.fusion_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim,
                dropout=dropout,
                activation='gelu',
                batch_first=True
            ) for _ in range(num_layers)
        ])
        
        # Output projections
        self.show_projection = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.LayerNorm(embed_dim)
        )
        
        # Recommendation head
        self.recommendation_head = nn.Sequential(
            nn.Linear(embed_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Temperature parameter for contrastive learning
        self.temperature = nn.Parameter(torch.tensor(0.07))
        
    def encode_text(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Encode text features using RoBERTa"""
        if self.use_gradient_checkpointing and self.training:
            outputs = checkpoint(
                self.text_encoder,
                input_ids,
                attention_mask,
                use_reentrant=False
            )
        else:
            outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        
        return outputs.last_hidden_state[:, 0, :]  # [CLS] token
    
    def encode_metadata(self, 
                       categorical_features: Dict[str, torch.Tensor],
                       numerical_features: torch.Tensor) -> torch.Tensor:
        """Encode metadata features"""
        return self.metadata_encoder(categorical_features, numerical_features)
    
    def fuse_modalities(self, 
                       text_features: torch.Tensor,
                       metadata_features: torch.Tensor) -> torch.Tensor:
        """Fuse text and metadata using cross-attention"""
        
        # Add sequence dimension for attention
        text_seq = text_features.unsqueeze(1)  # [batch, 1, embed_dim]
        meta_seq = metadata_features.unsqueeze(1)  # [batch, 1, embed_dim]
        
        # Cross-attention
        text_attended = self.text_to_meta_attention(text_seq, meta_seq, meta_seq)
        meta_attended = self.meta_to_text_attention(meta_seq, text_seq, text_seq)
        
        # Concatenate for fusion
        fused = torch.cat([text_attended, meta_attended], dim=1)  # [batch, 2, embed_dim]
        
        # Apply fusion layers
        for layer in self.fusion_layers:
            if self.use_gradient_checkpointing and self.training:
                fused = checkpoint(layer, fused, use_reentrant=False)
            else:
                fused = layer(fused)
        
        # Global average pooling
        return fused.mean(dim=1)
    
    def forward(self, 
                text_input_ids: torch.Tensor,
                text_attention_mask: torch.Tensor,
                categorical_features: Dict[str, torch.Tensor],
                numerical_features: torch.Tensor,
                target_input_ids: Optional[torch.Tensor] = None,
                target_attention_mask: Optional[torch.Tensor] = None,
                target_categorical: Optional[Dict[str, torch.Tensor]] = None,
                target_numerical: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        
        # Encode query show
        text_features = self.encode_text(text_input_ids, text_attention_mask)
        metadata_features = self.encode_metadata(categorical_features, numerical_features)
        query_embedding = self.fuse_modalities(text_features, metadata_features)
        query_embedding = self.show_projection(query_embedding)
        
        if target_input_ids is not None:
            # Encode target show for training
            target_text = self.encode_text(target_input_ids, target_attention_mask)
            target_meta = self.encode_metadata(target_categorical, target_numerical)
            target_embedding = self.fuse_modalities(target_text, target_meta)
            target_embedding = self.show_projection(target_embedding)
            
            # Recommendation score
            combined = torch.cat([query_embedding, target_embedding], dim=1)
            recommendation_score = self.recommendation_head(combined)
            
            # Contrastive similarity
            similarity = F.cosine_similarity(query_embedding, target_embedding, dim=1)
            contrastive_logits = similarity / self.temperature
            
            return {
                'query_embedding': query_embedding,
                'target_embedding': target_embedding,
                'recommendation_score': recommendation_score,
                'contrastive_logits': contrastive_logits,
                'similarity': similarity
            }
        else:
            return {
                'query_embedding': query_embedding
            }
    
    def get_show_embedding(self, 
                          text_input_ids: torch.Tensor,
                          text_attention_mask: torch.Tensor,
                          categorical_features: Dict[str, torch.Tensor],
                          numerical_features: torch.Tensor) -> torch.Tensor:
        """Get embedding for a single show"""
        with torch.no_grad():
            outputs = self.forward(
                text_input_ids=text_input_ids,
                text_attention_mask=text_attention_mask,
                categorical_features=categorical_features,
                numerical_features=numerical_features
            )
            return outputs['query_embedding']

class ContrastiveLoss(nn.Module):
    """Contrastive loss for learning show similarities"""
    
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
        
    def forward(self, 
                query_embeddings: torch.Tensor,
                positive_embeddings: torch.Tensor,
                negative_embeddings: torch.Tensor) -> torch.Tensor:
        
        batch_size = query_embeddings.size(0)
        
        # Normalize embeddings
        query_norm = F.normalize(query_embeddings, dim=1)
        pos_norm = F.normalize(positive_embeddings, dim=1)
        neg_norm = F.normalize(negative_embeddings, dim=1)
        
        # Positive similarities
        pos_sim = torch.sum(query_norm * pos_norm, dim=1) / self.temperature
        
        # Negative similarities
        neg_sim = torch.matmul(query_norm, neg_norm.T) / self.temperature
        
        # InfoNCE loss
        logits = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1)
        labels = torch.zeros(batch_size, dtype=torch.long, device=query_embeddings.device)
        
        return F.cross_entropy(logits, labels)

class TVRecommendationLoss(nn.Module):
    """Combined loss for TV recommendation"""
    
    def __init__(self, 
                 contrastive_weight: float = 1.0,
                 recommendation_weight: float = 1.0,
                 temperature: float = 0.07):
        super().__init__()
        self.contrastive_weight = contrastive_weight
        self.recommendation_weight = recommendation_weight
        self.contrastive_loss = ContrastiveLoss(temperature)
        self.recommendation_loss = nn.BCEWithLogitsLoss()
        
    def forward(self, 
                outputs: Dict[str, torch.Tensor],
                labels: torch.Tensor,
                negative_embeddings: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        
        losses = {}
        total_loss = 0
        
        # Recommendation loss
        if 'recommendation_score' in outputs:
            rec_loss = self.recommendation_loss(
                outputs['recommendation_score'].squeeze(), 
                labels.float()
            )
            losses['recommendation_loss'] = rec_loss
            total_loss += self.recommendation_weight * rec_loss
        
        # Contrastive loss
        if negative_embeddings is not None and 'query_embedding' in outputs and 'target_embedding' in outputs:
            cont_loss = self.contrastive_loss(
                outputs['query_embedding'],
                outputs['target_embedding'],
                negative_embeddings
            )
            losses['contrastive_loss'] = cont_loss
            total_loss += self.contrastive_weight * cont_loss
        
        losses['total_loss'] = total_loss
        return losses

# Model configuration for RTX 4090 optimization
def get_model_config():
    """Optimized configuration for RTX 4090 with RoBERTa-large"""
    return {
        'embed_dim': 1024,   # Match RoBERTa-large
        'hidden_dim': 2048,  # Larger hidden dim for 4090
        'num_layers': 8,     # More layers
        'num_heads': 16,     # Match RoBERTa-large heads (1024/64=16)
        'dropout': 0.1,
        'use_gradient_checkpointing': True,  # Essential for memory efficiency
        'batch_size': 32,    # Large batch size for 4090
        'accumulation_steps': 2,  # Effective batch size of 64
        'learning_rate': 1e-4,
        'warmup_steps': 1000,
        'max_grad_norm': 1.0,
    }