import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Dict, List


class SequentialRecommender(nn.Module):
    """
    Base class for sequential recommendation models using RNN/LSTM architectures.
    Handles user interaction sequences to predict next items.
    """
    
    def __init__(self, num_items, embedding_dim=128, hidden_dim=256, num_layers=2, 
                 dropout=0.2, rnn_type='LSTM'):
        super(SequentialRecommender, self).__init__()
        
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.rnn_type = rnn_type
        
        # Item embedding layer
        self.item_embedding = nn.Embedding(num_items + 1, embedding_dim, padding_idx=0)
        
        # RNN layer
        if rnn_type == 'LSTM':
            self.rnn = nn.LSTM(
                embedding_dim, hidden_dim, num_layers, 
                batch_first=True, dropout=dropout if num_layers > 1 else 0
            )
        elif rnn_type == 'GRU':
            self.rnn = nn.GRU(
                embedding_dim, hidden_dim, num_layers,
                batch_first=True, dropout=dropout if num_layers > 1 else 0
            )
        else:
            raise ValueError(f"Unsupported RNN type: {rnn_type}")
        
        # Output layers
        self.dropout = nn.Dropout(dropout)
        self.output_layer = nn.Linear(hidden_dim, num_items)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights"""
        for name, param in self.named_parameters():
            if 'weight' in name:
                if 'embedding' in name:
                    nn.init.xavier_normal_(param)
                elif 'rnn' in name:
                    nn.init.orthogonal_(param)
                else:
                    nn.init.xavier_normal_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
    
    def forward(self, sequences, lengths=None):
        """
        Forward pass for sequential recommendation.
        
        Args:
            sequences: Tensor of shape (batch_size, seq_len) containing item IDs
            lengths: Optional tensor of actual sequence lengths
        
        Returns:
            Tensor of shape (batch_size, seq_len, num_items) with item probabilities
        """
        batch_size, seq_len = sequences.size()
        
        # Embed items
        embedded = self.item_embedding(sequences)  # (batch_size, seq_len, embedding_dim)
        
        # Pack sequences if lengths provided
        if lengths is not None:
            embedded = nn.utils.rnn.pack_padded_sequence(
                embedded, lengths, batch_first=True, enforce_sorted=False
            )
        
        # RNN forward pass
        if self.rnn_type == 'LSTM':
            rnn_output, (hidden, cell) = self.rnn(embedded)
        else:  # GRU
            rnn_output, hidden = self.rnn(embedded)
        
        # Unpack if packed
        if lengths is not None:
            rnn_output, _ = nn.utils.rnn.pad_packed_sequence(
                rnn_output, batch_first=True
            )
        
        # Apply dropout and get predictions
        rnn_output = self.dropout(rnn_output)
        logits = self.output_layer(rnn_output)  # (batch_size, seq_len, num_items)
        
        return logits


class TransformerSequentialRecommender(nn.Module):
    """
    Modern Transformer-based Sequential Recommender with advanced features:
    - Rotary Positional Encoding (RoPE)
    - Multi-scale temporal modeling
    - Contrastive learning
    - Knowledge distillation
    - Multi-task learning
    """
    
    def __init__(self, num_items, embedding_dim=256, num_heads=8, num_blocks=6, 
                 dropout=0.1, max_seq_len=512, use_rope=True, use_multiscale=True,
                 temperature=0.07, num_genres=20):
        super(TransformerSequentialRecommender, self).__init__()
        
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.num_blocks = num_blocks
        self.max_seq_len = max_seq_len
        self.use_rope = use_rope
        self.use_multiscale = use_multiscale
        self.temperature = temperature
        
        # Enhanced item embedding with learnable scaling
        self.item_embedding = nn.Embedding(num_items + 1, embedding_dim, padding_idx=0)
        self.embedding_scale = nn.Parameter(torch.sqrt(torch.tensor(embedding_dim, dtype=torch.float32)))
        
        # Rotary Positional Encoding or traditional positional encoding
        if use_rope:
            self.rope = RotaryPositionalEncoding(embedding_dim // num_heads)
            self.pos_embedding = None
        else:
            self.pos_embedding = nn.Embedding(max_seq_len, embedding_dim)
            self.rope = None
        
        # Multi-scale temporal convolutions
        if use_multiscale:
            self.temporal_convs = nn.ModuleList([
                nn.Conv1d(embedding_dim, embedding_dim, kernel_size=k, padding=k//2)
                for k in [3, 5, 7]
            ])
            self.temporal_fusion = nn.Linear(embedding_dim * 3, embedding_dim)
        
        # Enhanced Transformer blocks with pre-norm and better initialization
        self.transformer_blocks = nn.ModuleList([
            EnhancedTransformerBlock(
                embedding_dim, num_heads, dropout, use_rope=use_rope
            ) for _ in range(num_blocks)
        ])
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(embedding_dim)
        self.dropout = nn.Dropout(dropout)
        
        # Multi-task prediction heads
        self.next_item_head = nn.Linear(embedding_dim, num_items)
        self.genre_head = nn.Linear(embedding_dim, num_genres)
        self.session_length_head = nn.Linear(embedding_dim, 10)  # Predict session length category
        
        # Contrastive projection head
        self.projection_head = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim // 2)
        )
        
        # Knowledge distillation components
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
    
    def forward(self, sequences, mask=None, return_all_outputs=False):
        batch_size, seq_len = sequences.size()
        
        # Item embeddings with scaling
        item_emb = self.item_embedding(sequences) * self.embedding_scale
        
        # Multi-scale temporal modeling
        if self.use_multiscale:
            # Transpose for conv1d: [batch, embedding_dim, seq_len]
            conv_input = item_emb.transpose(1, 2)
            multiscale_features = []
            
            for conv in self.temporal_convs:
                conv_out = conv(conv_input)
                multiscale_features.append(conv_out.transpose(1, 2))
            
            # Combine multi-scale features
            multiscale_combined = torch.cat(multiscale_features, dim=-1)
            item_emb = self.temporal_fusion(multiscale_combined)
        
        # Positional encoding
        if self.use_rope:
            # RoPE is applied within attention mechanism
            x = item_emb
        else:
            positions = torch.arange(seq_len, device=sequences.device).unsqueeze(0).expand(batch_size, -1)
            pos_emb = self.pos_embedding(positions)
            x = item_emb + pos_emb
        
        x = self.dropout(x)
        
        # Create causal mask
        if mask is None:
            mask = self._create_causal_mask(seq_len, device=sequences.device)
        
        # Enhanced Transformer blocks
        all_layer_outputs = []
        for block in self.transformer_blocks:
            x = block(x, mask, rope=self.rope)
            if return_all_outputs:
                all_layer_outputs.append(x)
        
        # Final layer norm
        x = self.layer_norm(x)
        
        # Multi-task predictions
        next_item_logits = self.next_item_head(x)
        genre_logits = self.genre_head(x)
        session_length_logits = self.session_length_head(x[:, -1])  # Use last token
        
        outputs = {
            'next_item_logits': next_item_logits,
            'genre_logits': genre_logits,
            'session_length_logits': session_length_logits,
            'sequence_embeddings': x
        }
        
        if return_all_outputs:
            outputs['all_layer_outputs'] = all_layer_outputs
            # Contrastive embeddings
            projection = F.normalize(self.projection_head(x), dim=-1)
            outputs['contrastive_embeddings'] = projection
        
        return outputs
    
    def _create_causal_mask(self, seq_len, device):
        mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
        mask = mask.masked_fill(mask == 0, float('-inf'))
        mask = mask.masked_fill(mask == 1, 0.0)
        return mask
    
    def compute_contrastive_loss(self, anchor_seqs, positive_seqs, negative_seqs):
        """Compute contrastive loss for sequence representations"""
        anchor_outputs = self.forward(anchor_seqs, return_all_outputs=True)
        positive_outputs = self.forward(positive_seqs, return_all_outputs=True)
        negative_outputs = self.forward(negative_seqs, return_all_outputs=True)
        
        # Use last token embeddings for contrastive learning
        anchor_emb = anchor_outputs['contrastive_embeddings'][:, -1]
        positive_emb = positive_outputs['contrastive_embeddings'][:, -1]
        negative_emb = negative_outputs['contrastive_embeddings'][:, -1]
        
        # Compute similarities
        pos_sim = torch.sum(anchor_emb * positive_emb, dim=1) / self.temperature
        neg_sim = torch.sum(anchor_emb * negative_emb, dim=1) / self.temperature
        
        # InfoNCE loss
        logits = torch.cat([pos_sim.unsqueeze(1), neg_sim.unsqueeze(1)], dim=1)
        labels = torch.zeros(anchor_seqs.size(0), dtype=torch.long, device=anchor_seqs.device)
        
        return F.cross_entropy(logits, labels)
    
    def distillation_loss(self, student_outputs, teacher_outputs, temperature=4.0):
        """Knowledge distillation loss"""
        student_logits = student_outputs['next_item_logits']
        teacher_logits = self.teacher_adapter(teacher_outputs['sequence_embeddings'])
        teacher_logits = self.next_item_head(teacher_logits)
        
        # Soft targets
        soft_student = F.log_softmax(student_logits / temperature, dim=-1)
        soft_teacher = F.softmax(teacher_logits / temperature, dim=-1)
        
        distill_loss = F.kl_div(soft_student, soft_teacher, reduction='batchmean')
        return distill_loss * (temperature ** 2)


class RotaryPositionalEncoding(nn.Module):
    """Rotary Positional Encoding (RoPE) for better positional understanding"""
    
    def __init__(self, dim, max_seq_len=10000):
        super(RotaryPositionalEncoding, self).__init__()
        self.dim = dim
        
        # Precompute rotation matrix
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        
        # Cache for efficiency
        self.max_seq_len = max_seq_len
        self._cos_cached = None
        self._sin_cached = None
        self._seq_len_cached = None
    
    def _update_cos_sin_cache(self, seq_len, device):
        if (self._seq_len_cached != seq_len or 
            self._cos_cached is None or 
            self._cos_cached.device != device):
            
            self._seq_len_cached = seq_len
            positions = torch.arange(seq_len, device=device).float()
            
            freqs = torch.einsum('i,j->ij', positions, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1)
            
            self._cos_cached = emb.cos()
            self._sin_cached = emb.sin()
    
    def rotate_half(self, x):
        x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
        return torch.cat((-x2, x1), dim=-1)
    
    def apply_rotary_pos_emb(self, q, k, seq_len):
        self._update_cos_sin_cache(seq_len, q.device)
        
        cos = self._cos_cached[:seq_len].unsqueeze(0).unsqueeze(0)
        sin = self._sin_cached[:seq_len].unsqueeze(0).unsqueeze(0)
        
        q_rot = (q * cos) + (self.rotate_half(q) * sin)
        k_rot = (k * cos) + (self.rotate_half(k) * sin)
        
        return q_rot, k_rot


class EnhancedTransformerBlock(nn.Module):
    """Enhanced Transformer block with pre-norm and better attention"""
    
    def __init__(self, embedding_dim, num_heads, dropout=0.1, use_rope=False):
        super(EnhancedTransformerBlock, self).__init__()
        
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads
        self.use_rope = use_rope
        
        # Pre-norm design
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)
        
        # Multi-head attention with better initialization
        self.attention = nn.MultiheadAttention(
            embedding_dim, num_heads, dropout=dropout, batch_first=True
        )
        
        # Enhanced feed-forward with GLU activation
        self.feed_forward = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim * 4, embedding_dim),
            nn.Dropout(dropout)
        )
        
        # Learnable scaling factors
        self.attn_scale = nn.Parameter(torch.ones(1))
        self.ff_scale = nn.Parameter(torch.ones(1))
        
    def forward(self, x, mask=None, rope=None):
        # Pre-norm attention
        x_norm = self.norm1(x)
        
        # Apply RoPE if available
        if self.use_rope and rope is not None:
            # Split into Q, K, V
            qkv = torch.chunk(
                F.linear(x_norm, self.attention.in_proj_weight, self.attention.in_proj_bias), 
                3, dim=-1
            )
            q, k, v = qkv
            
            # Reshape for multi-head
            batch_size, seq_len = x.size(0), x.size(1)
            q = q.view(batch_size, seq_len, self.num_heads, self.head_dim)
            k = k.view(batch_size, seq_len, self.num_heads, self.head_dim)
            v = v.view(batch_size, seq_len, self.num_heads, self.head_dim)
            
            # Apply RoPE
            q, k = rope.apply_rotary_pos_emb(q, k, seq_len)
            
            # Reshape back
            q = q.reshape(batch_size, seq_len, -1)
            k = k.reshape(batch_size, seq_len, -1)
            v = v.reshape(batch_size, seq_len, -1)
            
            # Manual attention computation
            scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            if mask is not None:
                scores += mask
            attn_weights = F.softmax(scores, dim=-1)
            attn_output = torch.matmul(attn_weights, v)
        else:
            # Standard attention
            attn_output, _ = self.attention(x_norm, x_norm, x_norm, attn_mask=mask)
        
        # Residual connection with learnable scaling
        x = x + self.attn_scale * attn_output
        
        # Pre-norm feed-forward
        x_norm = self.norm2(x)
        ff_output = self.feed_forward(x_norm)
        
        # Residual connection with learnable scaling
        x = x + self.ff_scale * ff_output
        
        return x


class AttentionalSequentialRecommender(nn.Module):
    """
    Sequential recommender with self-attention mechanism for better long-term dependencies.
    Based on SASRec: Self-Attentive Sequential Recommendation.
    """
    
    def __init__(self, num_items, embedding_dim=128, num_heads=8, num_blocks=2, 
                 dropout=0.2, max_seq_len=200):
        super(AttentionalSequentialRecommender, self).__init__()
        
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.num_blocks = num_blocks
        self.max_seq_len = max_seq_len
        
        # Item and positional embeddings
        self.item_embedding = nn.Embedding(num_items + 1, embedding_dim, padding_idx=0)
        self.pos_embedding = nn.Embedding(max_seq_len, embedding_dim)
        
        # Self-attention blocks
        self.attention_blocks = nn.ModuleList([
            AttentionBlock(embedding_dim, num_heads, dropout)
            for _ in range(num_blocks)
        ])
        
        # Layer normalization and dropout
        self.layer_norm = nn.LayerNorm(embedding_dim)
        self.dropout = nn.Dropout(dropout)
        
        # Output layer
        self.output_layer = nn.Linear(embedding_dim, num_items)
        
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Embedding):
                nn.init.xavier_normal_(module.weight)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, sequences, mask=None):
        """
        Forward pass with self-attention.
        
        Args:
            sequences: Tensor of shape (batch_size, seq_len)
            mask: Optional attention mask
        
        Returns:
            Tensor of shape (batch_size, seq_len, num_items)
        """
        batch_size, seq_len = sequences.size()
        
        # Create positional indices
        positions = torch.arange(seq_len, device=sequences.device).unsqueeze(0).expand(batch_size, -1)
        
        # Embeddings
        item_emb = self.item_embedding(sequences)
        pos_emb = self.pos_embedding(positions)
        
        # Combine embeddings
        x = item_emb + pos_emb
        x = self.dropout(x)
        
        # Create attention mask if not provided
        if mask is None:
            # Causal mask for autoregressive prediction
            mask = torch.tril(torch.ones(seq_len, seq_len, device=sequences.device))
            mask = mask.unsqueeze(0).expand(batch_size, -1, -1)
        
        # Apply attention blocks
        for block in self.attention_blocks:
            x = block(x, mask)
        
        # Final layer norm and output
        x = self.layer_norm(x)
        logits = self.output_layer(x)
        
        return logits


class AttentionBlock(nn.Module):
    """Multi-head self-attention block for sequential modeling"""
    
    def __init__(self, embedding_dim, num_heads, dropout=0.2):
        super(AttentionBlock, self).__init__()
        
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads
        
        assert embedding_dim % num_heads == 0, "embedding_dim must be divisible by num_heads"
        
        # Multi-head attention
        self.attention = nn.MultiheadAttention(
            embedding_dim, num_heads, dropout=dropout, batch_first=True
        )
        
        # Feed-forward network
        self.feed_forward = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim * 4, embedding_dim),
            nn.Dropout(dropout)
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        # Self-attention with residual connection
        x_norm = self.norm1(x)
        attn_output, _ = self.attention(x_norm, x_norm, x_norm, attn_mask=mask)
        x = x + self.dropout(attn_output)
        
        # Feed-forward with residual connection
        x_norm = self.norm2(x)
        ff_output = self.feed_forward(x_norm)
        x = x + ff_output
        
        return x


class HierarchicalSequentialRecommender(nn.Module):
    """
    Hierarchical model that captures both short-term and long-term user preferences.
    Uses separate RNNs for different time scales.
    """
    
    def __init__(self, num_items, embedding_dim=128, short_hidden_dim=128, 
                 long_hidden_dim=256, num_layers=2, dropout=0.2):
        super(HierarchicalSequentialRecommender, self).__init__()
        
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        
        # Shared item embedding
        self.item_embedding = nn.Embedding(num_items + 1, embedding_dim, padding_idx=0)
        
        # Short-term preference modeling (recent interactions)
        self.short_term_rnn = nn.LSTM(
            embedding_dim, short_hidden_dim, num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0
        )
        
        # Long-term preference modeling (overall history)
        self.long_term_rnn = nn.LSTM(
            embedding_dim, long_hidden_dim, num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0
        )
        
        # Attention mechanism to combine short and long term
        self.attention = nn.MultiheadAttention(
            short_hidden_dim + long_hidden_dim, 4, batch_first=True
        )
        
        # Output layers
        self.dropout = nn.Dropout(dropout)
        self.output_layer = nn.Linear(short_hidden_dim + long_hidden_dim, num_items)
        
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Embedding):
                nn.init.xavier_normal_(module.weight)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, short_sequences, long_sequences, short_lengths=None, long_lengths=None):
        """
        Forward pass with hierarchical modeling.
        
        Args:
            short_sequences: Recent interactions (batch_size, short_seq_len)
            long_sequences: Full interaction history (batch_size, long_seq_len)
            short_lengths: Lengths for short sequences
            long_lengths: Lengths for long sequences
        
        Returns:
            Predictions for next items
        """
        # Embed sequences
        short_embedded = self.item_embedding(short_sequences)
        long_embedded = self.item_embedding(long_sequences)
        
        # Process short-term sequences
        if short_lengths is not None:
            short_embedded = nn.utils.rnn.pack_padded_sequence(
                short_embedded, short_lengths, batch_first=True, enforce_sorted=False
            )
        
        short_output, (short_hidden, _) = self.short_term_rnn(short_embedded)
        
        if short_lengths is not None:
            short_output, _ = nn.utils.rnn.pad_packed_sequence(
                short_output, batch_first=True
            )
        
        # Process long-term sequences
        if long_lengths is not None:
            long_embedded = nn.utils.rnn.pack_padded_sequence(
                long_embedded, long_lengths, batch_first=True, enforce_sorted=False
            )
        
        long_output, (long_hidden, _) = self.long_term_rnn(long_embedded)
        
        if long_lengths is not None:
            long_output, _ = nn.utils.rnn.pad_packed_sequence(
                long_output, batch_first=True
            )
        
        # Combine short and long term representations
        # Take the last output from each sequence
        batch_size = short_output.size(0)
        
        if short_lengths is not None:
            short_final = short_output[range(batch_size), short_lengths - 1]
        else:
            short_final = short_output[:, -1]
        
        if long_lengths is not None:
            long_final = long_output[range(batch_size), long_lengths - 1]
        else:
            long_final = long_output[:, -1]
        
        # Concatenate and apply attention
        combined = torch.cat([short_final, long_final], dim=-1).unsqueeze(1)
        attended, _ = self.attention(combined, combined, combined)
        attended = attended.squeeze(1)
        
        # Final predictions
        attended = self.dropout(attended)
        logits = self.output_layer(attended)
        
        return logits


class SessionBasedRecommender(nn.Module):
    """
    Session-based recommendation model using GRU for short-term session modeling.
    Optimized for scenarios with limited user history.
    """
    
    def __init__(self, num_items, embedding_dim=128, hidden_dim=256, num_layers=1, 
                 dropout=0.3, use_attention=True):
        super(SessionBasedRecommender, self).__init__()
        
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.use_attention = use_attention
        
        # Item embedding
        self.item_embedding = nn.Embedding(num_items + 1, embedding_dim, padding_idx=0)
        
        # GRU for session modeling
        self.gru = nn.GRU(
            embedding_dim, hidden_dim, num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0
        )
        
        # Attention mechanism for session representation
        if use_attention:
            self.attention_linear = nn.Linear(hidden_dim, 1)
        
        # Output layers
        self.dropout = nn.Dropout(dropout)
        self.output_layer = nn.Linear(hidden_dim, num_items)
        
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Embedding):
                nn.init.xavier_normal_(module.weight)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, sequences, lengths=None):
        """
        Forward pass for session-based recommendation.
        
        Args:
            sequences: Session item sequences (batch_size, seq_len)
            lengths: Actual sequence lengths
        
        Returns:
            Item predictions for next interaction
        """
        # Embed items
        embedded = self.item_embedding(sequences)
        
        # Pack sequences
        if lengths is not None:
            embedded = nn.utils.rnn.pack_padded_sequence(
                embedded, lengths, batch_first=True, enforce_sorted=False
            )
        
        # GRU forward pass
        gru_output, hidden = self.gru(embedded)
        
        # Unpack
        if lengths is not None:
            gru_output, _ = nn.utils.rnn.pad_packed_sequence(
                gru_output, batch_first=True
            )
        
        # Session representation: create unified session embedding from GRU sequence outputs
        if self.use_attention:
            # Attention-based session representation: learns which items are most important
            # Computes attention weights for each timestep and creates weighted sum
            attention_weights = F.softmax(self.attention_linear(gru_output), dim=1)
            session_repr = (gru_output * attention_weights).sum(dim=1)  # Weighted average of all timesteps
        else:
            # Use last hidden state as session representation (simpler approach)
            batch_size = gru_output.size(0)
            if lengths is not None:
                # Get last valid timestep for each sequence (handles variable lengths)
                session_repr = gru_output[range(batch_size), lengths - 1]
            else:
                # Use last timestep for fixed-length sequences
                session_repr = gru_output[:, -1]
        
        # Final predictions
        session_repr = self.dropout(session_repr)
        logits = self.output_layer(session_repr)
        
        return logits