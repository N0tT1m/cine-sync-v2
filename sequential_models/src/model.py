import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


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
        
        # Session representation
        if self.use_attention:
            # Attention-based session representation
            attention_weights = F.softmax(self.attention_linear(gru_output), dim=1)
            session_repr = (gru_output * attention_weights).sum(dim=1)
        else:
            # Use last hidden state
            batch_size = gru_output.size(0)
            if lengths is not None:
                session_repr = gru_output[range(batch_size), lengths - 1]
            else:
                session_repr = gru_output[:, -1]
        
        # Final predictions
        session_repr = self.dropout(session_repr)
        logits = self.output_layer(session_repr)
        
        return logits