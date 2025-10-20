# Transformer-based Sequential Recommendation for CineSync v2
# Implements SASRec and enhanced variants for sequential recommendation
# Optimized for RTX 4090 with advanced attention mechanisms and multi-task learning

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from typing import Optional, Tuple, Dict, Any


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer-based sequential recommendation
    
    Adds position information to item embeddings so the model can understand
    the temporal order of user interactions, which is crucial for sequential
    recommendation tasks.
    """
    
    def __init__(self, d_model: int, max_len: int = 5000):
        """Initialize positional encoding with sinusoidal patterns
        
        Args:
            d_model: Model dimension size
            max_len: Maximum sequence length to support
        """
        super(PositionalEncoding, self).__init__()
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        # Create frequency terms for sinusoidal encoding
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        # Apply sine to even indices and cosine to odd indices
        pe[:, 0::2] = torch.sin(position * div_term)  # Even dimensions: sin
        pe[:, 1::2] = torch.cos(position * div_term)  # Odd dimensions: cos
        pe = pe.unsqueeze(0).transpose(0, 1)          # Shape: [max_len, 1, d_model]
        
        # Register as buffer (not a parameter, but part of model state)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input embeddings
        
        Args:
            x: Input embeddings [seq_len, batch_size, d_model]
            
        Returns:
            Position-encoded embeddings
        """
        return x + self.pe[:x.size(0), :]  # Add positional info to embeddings


class MultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention mechanism optimized for recommendations
    
    Core component of transformer that allows each position to attend to all
    positions in the sequence, enabling the model to capture complex dependencies
    between user interactions at different time steps.
    """
    
    def __init__(self, d_model: int, num_heads: int = 8, dropout: float = 0.1):
        """Initialize multi-head self-attention
        
        Args:
            d_model: Model dimension size
            num_heads: Number of parallel attention heads
            dropout: Dropout rate for regularization
        """
        super(MultiHeadSelfAttention, self).__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # Dimension per head
        
        # Linear projections for query, key, value, and output
        self.w_q = nn.Linear(d_model, d_model, bias=False)  # Query projection
        self.w_k = nn.Linear(d_model, d_model, bias=False)  # Key projection
        self.w_v = nn.Linear(d_model, d_model, bias=False)  # Value projection
        self.w_o = nn.Linear(d_model, d_model)              # Output projection
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with Xavier uniform for stable training"""
        for module in [self.w_q, self.w_k, self.w_v, self.w_o]:
            nn.init.xavier_uniform_(module.weight)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, d_model = x.size()
        
        # Transform input to queries, keys, values and split into heads
        # Shape: [batch, seq_len, d_model] -> [batch, num_heads, seq_len, d_k]
        Q = self.w_q(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention: softmax(QK^T/sqrt(d_k))V
        # Scale prevents softmax from saturating for large d_k
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # Apply causal mask to prevent attending to future positions
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)  # Large negative value
        
        # Convert scores to attention weights and apply dropout
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)  # Dropout on attention weights
        
        # Apply attention weights to values: weighted sum of value vectors
        attn_output = torch.matmul(attn_weights, V)
        
        # Concatenate attention heads back to original dimension
        # Shape: [batch, num_heads, seq_len, d_k] -> [batch, seq_len, d_model]
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, d_model
        )
        output = self.w_o(attn_output)  # Final linear projection
        
        # Residual connection and layer normalization for stable training
        return self.layer_norm(x + self.dropout(output))


class FeedForward(nn.Module):
    """Position-wise feed-forward network"""
    
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super(FeedForward, self).__init__()
        
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        
        self._init_weights()
    
    def _init_weights(self):
        nn.init.xavier_uniform_(self.linear1.weight)
        nn.init.xavier_uniform_(self.linear2.weight)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.linear2(self.dropout(F.gelu(self.linear1(x))))
        return self.layer_norm(residual + self.dropout(x))


class TransformerBlock(nn.Module):
    """Single transformer block for sequential recommendation"""
    
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super(TransformerBlock, self).__init__()
        
        self.attention = MultiHeadSelfAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.attention(x, mask)
        x = self.feed_forward(x)
        return x


class SASRec(nn.Module):
    """
    Self-Attentive Sequential Recommendation model.
    
    Paper: Kang, W. C., & McAuley, J. (2018). Self-attentive sequential recommendation.
    """
    
    def __init__(
        self,
        num_items: int,
        max_seq_len: int = 200,
        d_model: int = 512,
        num_heads: int = 8,
        num_layers: int = 6,
        d_ff: int = 2048,
        dropout: float = 0.1,
        pad_token: int = 0
    ):
        super(SASRec, self).__init__()
        
        self.num_items = num_items
        self.max_seq_len = max_seq_len
        self.d_model = d_model
        self.pad_token = pad_token
        
        # Item embedding (add 1 for padding token)
        self.item_embedding = nn.Embedding(num_items + 1, d_model, padding_idx=pad_token)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_len)
        
        # Transformer layers
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        # Output projection
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
        # For next item prediction
        self.output_projection = nn.Linear(d_model, num_items)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights"""
        for module in self.modules():
            if isinstance(module, nn.Embedding):
                nn.init.xavier_normal_(module.weight)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def create_attention_mask(self, seq: torch.Tensor) -> torch.Tensor:
        """Create causal attention mask to prevent looking at future items"""
        batch_size, seq_len = seq.size()
        
        # Padding mask
        pad_mask = (seq != self.pad_token).unsqueeze(1).unsqueeze(2)  # [B, 1, 1, L]
        
        # Causal mask (lower triangular)
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=seq.device)).unsqueeze(0).unsqueeze(0)
        
        # Combine masks
        mask = pad_mask & causal_mask.bool()
        
        return mask
    
    def forward(self, sequences: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for training.
        
        Args:
            sequences: Item sequences [batch_size, seq_len]
            
        Returns:
            Logits for next item prediction [batch_size, seq_len, num_items]
        """
        batch_size, seq_len = sequences.size()
        
        # Embedding
        x = self.item_embedding(sequences) * math.sqrt(self.d_model)
        x = self.positional_encoding(x.transpose(0, 1)).transpose(0, 1)
        x = self.dropout(x)
        
        # Create attention mask
        mask = self.create_attention_mask(sequences)
        
        # Transformer layers
        for transformer in self.transformer_blocks:
            x = transformer(x, mask)
        
        x = self.layer_norm(x)
        
        # Output projection
        logits = self.output_projection(x)
        
        return logits
    
    def predict_next(self, sequences: torch.Tensor, k: int = 10) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict next k items for given sequences.
        
        Args:
            sequences: Item sequences [batch_size, seq_len]
            k: Number of recommendations
            
        Returns:
            Tuple of (top_k_items, top_k_scores)
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(sequences)
            
            # Get predictions for the last position
            last_logits = logits[:, -1, :]  # [batch_size, num_items]
            
            # Get top-k predictions
            top_k_scores, top_k_items = torch.topk(last_logits, k, dim=-1)
            
            return top_k_items, F.softmax(top_k_scores, dim=-1)
    
    def get_item_embeddings(self) -> torch.Tensor:
        """Get item embeddings for similarity computation"""
        return self.item_embedding.weight[1:]  # Exclude padding token


class EnhancedSASRec(SASRec):
    """
    Enhanced SASRec with additional features for better performance on RTX 4090
    """
    
    def __init__(
        self,
        num_items: int,
        num_genres: int = 20,
        max_seq_len: int = 200,
        d_model: int = 768,  # Larger for RTX 4090
        num_heads: int = 12,
        num_layers: int = 12,
        d_ff: int = 3072,
        dropout: float = 0.1,
        pad_token: int = 0,
        use_genre_info: bool = True
    ):
        # Initialize parent with larger dimensions
        super(EnhancedSASRec, self).__init__(
            num_items=num_items,
            max_seq_len=max_seq_len,
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_layers,
            d_ff=d_ff,
            dropout=dropout,
            pad_token=pad_token
        )
        
        self.use_genre_info = use_genre_info
        
        if use_genre_info:
            # Genre embeddings
            self.genre_embedding = nn.Embedding(num_genres + 1, d_model // 4, padding_idx=0)
            
            # Genre fusion layer
            self.genre_fusion = nn.Sequential(
                nn.Linear(d_model + d_model // 4, d_model),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.LayerNorm(d_model)
            )
        
        # Content-based similarity learning
        self.content_projection = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, d_model)
        )
        
        # Multi-task heads
        self.rating_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1)
        )
        
        self.preference_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 2)  # Like/Dislike
        )
    
    def forward(self, sequences: torch.Tensor, genre_sequences: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Enhanced forward pass with genre information and multi-task learning.
        
        Args:
            sequences: Item sequences [batch_size, seq_len]
            genre_sequences: Genre sequences [batch_size, seq_len] (optional)
            
        Returns:
            Dictionary with various predictions
        """
        batch_size, seq_len = sequences.size()
        
        # Item embeddings
        x = self.item_embedding(sequences) * math.sqrt(self.d_model)
        
        # Add genre information if available
        if self.use_genre_info and genre_sequences is not None:
            genre_emb = self.genre_embedding(genre_sequences)
            combined = torch.cat([x, genre_emb], dim=-1)
            x = self.genre_fusion(combined)
        
        # Positional encoding
        x = self.positional_encoding(x.transpose(0, 1)).transpose(0, 1)
        x = self.dropout(x)
        
        # Create attention mask
        mask = self.create_attention_mask(sequences)
        
        # Transformer layers
        for transformer in self.transformer_blocks:
            x = transformer(x, mask)
        
        x = self.layer_norm(x)
        
        # Multiple predictions
        outputs = {}
        
        # Next item prediction
        outputs['next_item_logits'] = self.output_projection(x)
        
        # Rating prediction
        outputs['rating_pred'] = torch.sigmoid(self.rating_head(x)) * 5.0
        
        # Preference prediction
        outputs['preference_logits'] = self.preference_head(x)
        
        # Content similarity features
        outputs['content_features'] = self.content_projection(x)
        
        return outputs


class SASRecTrainer:
    """Advanced trainer for SASRec with mixed precision and optimization"""
    
    def __init__(
        self,
        model: SASRec,
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
        
        # Advanced optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(0.9, 0.999)
        )
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=10, T_mult=2, eta_min=1e-6
        )
        
        # Mixed precision scaler
        if use_amp:
            self.scaler = torch.cuda.amp.GradScaler()
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Single training step with mixed precision"""
        self.model.train()
        sequences = batch['sequences'].to(self.device)
        targets = batch['targets'].to(self.device)
        
        self.optimizer.zero_grad()
        
        if self.use_amp:
            with torch.cuda.amp.autocast():
                if isinstance(self.model, EnhancedSASRec):
                    outputs = self.model(sequences, batch.get('genre_sequences'))
                    logits = outputs['next_item_logits']
                else:
                    logits = self.model(sequences)
                
                # Reshape for loss computation
                logits = logits.view(-1, logits.size(-1))
                targets = targets.view(-1)
                
                loss = self.criterion(logits, targets)
            
            self.scaler.scale(loss).backward()
            
            if self.gradient_clip_val > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_val)
            
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            if isinstance(self.model, EnhancedSASRec):
                outputs = self.model(sequences, batch.get('genre_sequences'))
                logits = outputs['next_item_logits']
            else:
                logits = self.model(sequences)
            
            logits = logits.view(-1, logits.size(-1))
            targets = targets.view(-1)
            
            loss = self.criterion(logits, targets)
            loss.backward()
            
            if self.gradient_clip_val > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_val)
            
            self.optimizer.step()
        
        self.scheduler.step()
        
        return {
            'loss': loss.item(),
            'lr': self.optimizer.param_groups[0]['lr']
        }