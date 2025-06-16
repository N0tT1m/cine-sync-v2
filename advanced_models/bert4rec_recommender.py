# BERT4Rec Implementation for CineSync v2
# Bidirectional Encoder Representations from Transformers for Sequential Recommendation
# Improved over SASRec with bidirectional context and masked language modeling

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from typing import Optional, Tuple, Dict, Any, List
import random


class PositionalEncoding(nn.Module):
    """Learnable positional encoding for BERT4Rec"""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()
        self.position_embeddings = nn.Embedding(max_len, d_model)
        
    def forward(self, x: torch.Tensor, positions: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, _ = x.size()
        
        if positions is None:
            positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)
        
        pos_emb = self.position_embeddings(positions)
        return x + pos_emb


class MultiHeadBidirectionalAttention(nn.Module):
    """Bidirectional multi-head attention for BERT4Rec"""
    
    def __init__(self, d_model: int, num_heads: int = 8, dropout: float = 0.1):
        super(MultiHeadBidirectionalAttention, self).__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        
        self._init_weights()
    
    def _init_weights(self):
        for module in [self.w_q, self.w_k, self.w_v, self.w_o]:
            nn.init.xavier_uniform_(module.weight)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, d_model = x.size()
        residual = x
        
        # Transform to Q, K, V and reshape for multi-head attention
        Q = self.w_q(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention (bidirectional)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # Apply padding mask (no causal mask for bidirectional attention)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        attn_output = torch.matmul(attn_weights, V)
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, d_model
        )
        output = self.w_o(attn_output)
        
        return self.layer_norm(residual + self.dropout(output))


class BERTBlock(nn.Module):
    """BERT transformer block with bidirectional attention"""
    
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super(BERTBlock, self).__init__()
        
        self.attention = MultiHeadBidirectionalAttention(d_model, num_heads, dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        self.layer_norm = nn.LayerNorm(d_model)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.attention(x, mask)
        residual = x
        x = self.feed_forward(x)
        return self.layer_norm(residual + x)


class BERT4Rec(nn.Module):
    """
    BERT4Rec: Bidirectional Encoder Representations from Transformers for Sequential Recommendation
    
    Paper: Sun, F., Liu, J., Wu, J., Pei, C., Lin, X., Ou, W., & Jiang, P. (2019).
    BERT4Rec: Sequential recommendation with bidirectional encoder representations from transformer.
    """
    
    def __init__(
        self,
        num_items: int,
        max_seq_len: int = 200,
        d_model: int = 768,
        num_heads: int = 12,
        num_layers: int = 12,
        d_ff: int = 3072,
        dropout: float = 0.1,
        pad_token: int = 0,
        mask_token: int = None,
        mask_prob: float = 0.15
    ):
        super(BERT4Rec, self).__init__()
        
        self.num_items = num_items
        self.max_seq_len = max_seq_len
        self.d_model = d_model
        self.pad_token = pad_token
        self.mask_token = mask_token or (num_items + 1)  # Use next available token ID
        self.mask_prob = mask_prob
        
        # Embeddings (item + position)
        self.item_embedding = nn.Embedding(num_items + 2, d_model, padding_idx=pad_token)  # +2 for pad and mask
        self.positional_encoding = PositionalEncoding(d_model, max_seq_len)
        
        # BERT layers
        self.bert_layers = nn.ModuleList([
            BERTBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        # Output layers
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
        # Prediction head
        self.prediction_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
            nn.Linear(d_model, num_items)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.padding_idx is not None:
                    module.weight.data[module.padding_idx].fill_(0)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def create_padding_mask(self, seq: torch.Tensor) -> torch.Tensor:
        """Create padding mask for attention"""
        # Shape: [batch_size, 1, 1, seq_len]
        return (seq != self.pad_token).unsqueeze(1).unsqueeze(2)
    
    def mask_sequence(self, seq: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Apply masking strategy for BERT4Rec training
        
        Returns:
            masked_seq: Sequence with some items masked
            mask_positions: Boolean tensor indicating masked positions
            original_items: Original items at masked positions
        """
        batch_size, seq_len = seq.size()
        masked_seq = seq.clone()
        
        # Create mask for non-padding positions
        valid_positions = (seq != self.pad_token)
        
        # Randomly select positions to mask
        mask_positions = torch.zeros_like(seq, dtype=torch.bool)
        
        for i in range(batch_size):
            valid_indices = valid_positions[i].nonzero(as_tuple=True)[0]
            if len(valid_indices) > 0:
                num_to_mask = max(1, int(len(valid_indices) * self.mask_prob))
                mask_indices = torch.randperm(len(valid_indices))[:num_to_mask]
                positions_to_mask = valid_indices[mask_indices]
                mask_positions[i, positions_to_mask] = True
        
        # Store original items at masked positions
        original_items = seq[mask_positions]
        
        # Apply masking strategy
        for i in range(batch_size):
            for j in range(seq_len):
                if mask_positions[i, j]:
                    prob = random.random()
                    if prob < 0.8:  # 80% replace with [MASK]
                        masked_seq[i, j] = self.mask_token
                    elif prob < 0.9:  # 10% replace with random item
                        masked_seq[i, j] = random.randint(1, self.num_items)
                    # 10% keep original (no change needed)
        
        return masked_seq, mask_positions, original_items
    
    def forward(self, sequences: torch.Tensor, apply_masking: bool = True) -> Dict[str, torch.Tensor]:
        """
        Forward pass for BERT4Rec
        
        Args:
            sequences: Item sequences [batch_size, seq_len]
            apply_masking: Whether to apply masking (True for training, False for inference)
            
        Returns:
            Dictionary with predictions and metadata
        """
        if apply_masking and self.training:
            masked_seq, mask_positions, original_items = self.mask_sequence(sequences)
            input_seq = masked_seq
        else:
            input_seq = sequences
            mask_positions = None
            original_items = None
        
        # Embeddings
        x = self.item_embedding(input_seq)
        x = self.positional_encoding(x)
        x = self.dropout(x)
        
        # Create attention mask
        attention_mask = self.create_padding_mask(input_seq)
        
        # BERT layers
        for bert_layer in self.bert_layers:
            x = bert_layer(x, attention_mask)
        
        x = self.layer_norm(x)
        
        # Prediction
        logits = self.prediction_head(x)
        
        outputs = {
            'logits': logits,
            'mask_positions': mask_positions,
            'original_items': original_items,
            'hidden_states': x
        }
        
        return outputs
    
    def predict_next(self, sequences: torch.Tensor, k: int = 10) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict next k items using bidirectional context"""
        self.eval()
        with torch.no_grad():
            # Add mask token at the end for prediction
            batch_size, seq_len = sequences.size()
            extended_seq = torch.cat([
                sequences,
                torch.full((batch_size, 1), self.mask_token, device=sequences.device)
            ], dim=1)
            
            outputs = self.forward(extended_seq, apply_masking=False)
            
            # Get predictions for the last (masked) position
            last_logits = outputs['logits'][:, -1, :]
            
            # Get top-k predictions
            top_k_scores, top_k_items = torch.topk(last_logits, k, dim=-1)
            
            return top_k_items, F.softmax(top_k_scores, dim=-1)
    
    def get_item_embeddings(self) -> torch.Tensor:
        """Get item embeddings for similarity computation"""
        return self.item_embedding.weight[1:self.num_items+1]  # Exclude pad and mask tokens


class EnhancedBERT4Rec(BERT4Rec):
    """Enhanced BERT4Rec with genre information and multi-task learning"""
    
    def __init__(
        self,
        num_items: int,
        num_genres: int = 20,
        max_seq_len: int = 200,
        d_model: int = 768,
        num_heads: int = 12,
        num_layers: int = 12,
        d_ff: int = 3072,
        dropout: float = 0.1,
        pad_token: int = 0,
        mask_token: int = None,
        mask_prob: float = 0.15,
        use_genre_info: bool = True
    ):
        super(EnhancedBERT4Rec, self).__init__(
            num_items=num_items,
            max_seq_len=max_seq_len,
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_layers,
            d_ff=d_ff,
            dropout=dropout,
            pad_token=pad_token,
            mask_token=mask_token,
            mask_prob=mask_prob
        )
        
        self.use_genre_info = use_genre_info
        
        if use_genre_info:
            # Genre embeddings
            self.genre_embedding = nn.Embedding(num_genres + 1, d_model // 4, padding_idx=0)
            
            # Genre fusion
            self.genre_fusion = nn.Sequential(
                nn.Linear(d_model + d_model // 4, d_model),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.LayerNorm(d_model)
            )
        
        # Multi-task heads
        self.rating_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1)
        )
        
        self.next_item_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
            nn.Linear(d_model, num_items)
        )
        
        # Cold start handling
        self.user_embedding = nn.Embedding(100000, d_model // 2)  # Assuming max users
        self.cold_start_fusion = nn.Sequential(
            nn.Linear(d_model + d_model // 2, d_model),
            nn.GELU(),
            nn.Dropout(dropout)
        )
    
    def forward(
        self, 
        sequences: torch.Tensor, 
        genre_sequences: Optional[torch.Tensor] = None,
        user_ids: Optional[torch.Tensor] = None,
        apply_masking: bool = True
    ) -> Dict[str, torch.Tensor]:
        """Enhanced forward pass with genre and user information"""
        
        if apply_masking and self.training:
            masked_seq, mask_positions, original_items = self.mask_sequence(sequences)
            input_seq = masked_seq
        else:
            input_seq = sequences
            mask_positions = None
            original_items = None
        
        # Item embeddings
        x = self.item_embedding(input_seq)
        
        # Add genre information
        if self.use_genre_info and genre_sequences is not None:
            if apply_masking and self.training:
                genre_seq = genre_sequences.clone()
                genre_seq[mask_positions] = 0  # Mask genre info too
            else:
                genre_seq = genre_sequences
            
            genre_emb = self.genre_embedding(genre_seq)
            combined = torch.cat([x, genre_emb], dim=-1)
            x = self.genre_fusion(combined)
        
        # Add user information for cold start
        if user_ids is not None:
            user_emb = self.user_embedding(user_ids).unsqueeze(1).expand(-1, x.size(1), -1)
            combined = torch.cat([x, user_emb], dim=-1)
            x = self.cold_start_fusion(combined)
        
        # Positional encoding
        x = self.positional_encoding(x)
        x = self.dropout(x)
        
        # Attention mask
        attention_mask = self.create_padding_mask(input_seq)
        
        # BERT layers
        for bert_layer in self.bert_layers:
            x = bert_layer(x, attention_mask)
        
        x = self.layer_norm(x)
        
        # Multiple predictions
        outputs = {
            'next_item_logits': self.next_item_head(x),
            'rating_pred': torch.sigmoid(self.rating_head(x)) * 5.0,
            'mask_positions': mask_positions,
            'original_items': original_items,
            'hidden_states': x
        }
        
        return outputs


class BERT4RecTrainer:
    """Trainer for BERT4Rec with masked language modeling"""
    
    def __init__(
        self,
        model: BERT4Rec,
        device: torch.device,
        learning_rate: float = 5e-5,
        weight_decay: float = 0.01,
        warmup_steps: int = 1000,
        use_amp: bool = True
    ):
        self.model = model.to(device)
        self.device = device
        self.use_amp = use_amp
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        # Scheduler with warmup
        self.scheduler = torch.optim.lr_scheduler.LinearLR(
            self.optimizer,
            start_factor=0.1,
            total_iters=warmup_steps
        )
        
        # Mixed precision
        if use_amp:
            self.scaler = torch.cuda.amp.GradScaler()
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Training step with masked language modeling"""
        self.model.train()
        sequences = batch['sequences'].to(self.device)
        
        self.optimizer.zero_grad()
        
        if self.use_amp:
            with torch.cuda.amp.autocast():
                outputs = self.model(sequences, apply_masking=True)
                
                # Compute loss only on masked positions
                if outputs['mask_positions'] is not None:
                    masked_logits = outputs['next_item_logits'][outputs['mask_positions']]
                    masked_targets = outputs['original_items']
                    loss = self.criterion(masked_logits, masked_targets)
                else:
                    loss = torch.tensor(0.0, device=self.device)
            
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            outputs = self.model(sequences, apply_masking=True)
            
            if outputs['mask_positions'] is not None:
                masked_logits = outputs['next_item_logits'][outputs['mask_positions']]
                masked_targets = outputs['original_items']
                loss = self.criterion(masked_logits, masked_targets)
            else:
                loss = torch.tensor(0.0, device=self.device)
            
            loss.backward()
            self.optimizer.step()
        
        self.scheduler.step()
        
        return {
            'loss': loss.item(),
            'lr': self.optimizer.param_groups[0]['lr']
        }