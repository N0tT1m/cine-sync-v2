"""
Temporal Attention TV Model (TAT-TV)
Captures seasonal patterns and temporal dynamics in TV show popularity
Optimized for RTX 4090
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta

class SeasonalPositionalEncoding(nn.Module):
    """Positional encoding that captures seasonal patterns"""
    
    def __init__(self, d_model: int, max_len: int = 5000, seasonal_periods: List[int] = [7, 30, 365]):
        super().__init__()
        self.d_model = d_model
        self.seasonal_periods = seasonal_periods
        
        # Standard positional encoding
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        # Divide embedding dimensions among different encodings
        dims_per_encoding = d_model // (len(seasonal_periods) + 1)
        
        # Standard time encoding
        div_term = torch.exp(torch.arange(0, dims_per_encoding, 2).float() * 
                           (-math.log(10000.0) / dims_per_encoding))
        pe[:, 0:dims_per_encoding:2] = torch.sin(position * div_term)
        pe[:, 1:dims_per_encoding:2] = torch.cos(position * div_term)
        
        # Seasonal encodings
        start_dim = dims_per_encoding
        for i, period in enumerate(seasonal_periods):
            end_dim = start_dim + dims_per_encoding
            seasonal_pos = (position % period) / period * 2 * math.pi
            
            for j in range(0, min(dims_per_encoding, end_dim - start_dim), 2):
                pe[:, start_dim + j] = torch.sin(seasonal_pos * (j // 2 + 1)).squeeze()
                if start_dim + j + 1 < d_model:
                    pe[:, start_dim + j + 1] = torch.cos(seasonal_pos * (j // 2 + 1)).squeeze()
            
            start_dim = end_dim
        
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor, time_indices: Optional[torch.Tensor] = None) -> torch.Tensor:
        if time_indices is not None:
            # Handle batched time indices: (batch, seq_len) or (seq_len,)
            # pe shape: (max_len, 1, d_model)
            # Clamp indices to valid range
            time_indices = torch.clamp(time_indices, 0, self.pe.size(0) - 1)
            if time_indices.dim() == 2:
                # (batch, seq_len) -> lookup and reshape
                batch_size, seq_len = time_indices.shape
                flat_indices = time_indices.view(-1)
                time_pe = self.pe[flat_indices, 0, :]  # (batch*seq, d_model)
                time_pe = time_pe.view(batch_size, seq_len, -1)  # (batch, seq, d_model)
                # x is (seq, batch, d_model), so transpose time_pe
                return x + time_pe.transpose(0, 1)
            else:
                # (seq_len,) - original behavior
                time_pe = self.pe[time_indices, 0, :]  # (seq, d_model)
                return x + time_pe.unsqueeze(1)
        else:
            # Use sequential indices
            return x + self.pe[:x.size(0), :]

class TemporalAttentionLayer(nn.Module):
    """Multi-head attention with temporal awareness"""
    
    def __init__(self, d_model: int, n_heads: int = 8, dropout: float = 0.1, 
                 max_relative_position: int = 512):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.max_relative_position = max_relative_position
        
        # Standard attention weights
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model)
        
        # Relative positional encodings
        self.relative_position_k = nn.Parameter(torch.randn(max_relative_position * 2 + 1, self.d_k))
        self.relative_position_v = nn.Parameter(torch.randn(max_relative_position * 2 + 1, self.d_k))
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        
    def _get_relative_positions(self, seq_len: int) -> torch.Tensor:
        """Get relative position indices"""
        range_vec = torch.arange(seq_len)
        range_mat = range_vec.repeat(seq_len).view(seq_len, seq_len)
        distance_mat = range_mat - range_mat.T
        distance_mat_clipped = torch.clamp(distance_mat, 
                                         -self.max_relative_position, 
                                         self.max_relative_position)
        final_mat = distance_mat_clipped + self.max_relative_position
        return final_mat
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len = x.size(0), x.size(1)
        residual = x
        
        # Linear transformations
        Q = self.w_q(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        # Standard attention scores
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # Add relative positional encoding
        relative_positions = self._get_relative_positions(seq_len).to(x.device)
        rel_pos_k = self.relative_position_k[relative_positions]  # [seq_len, seq_len, d_k]
        
        # Compute relative attention scores
        rel_scores = torch.einsum('bhid,jid->bhij', Q, rel_pos_k.transpose(0, 1))
        attention_scores += rel_scores
        
        if mask is not None:
            attention_scores.masked_fill_(mask == 0, -1e9)
        
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        context = torch.matmul(attention_weights, V)
        
        # Add relative positional encoding to values
        rel_pos_v = self.relative_position_v[relative_positions]
        rel_values = torch.einsum('bhij,jid->bhid', attention_weights, rel_pos_v.transpose(0, 1))
        context += rel_values
        
        # Reshape and project
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = self.w_o(context)
        
        return self.layer_norm(output + residual)

class SeasonalDecomposition(nn.Module):
    """Decompose temporal patterns into trend and seasonal components"""
    
    def __init__(self, d_model: int, seasonal_periods: List[int] = [7, 30, 365]):
        super().__init__()
        self.d_model = d_model
        self.seasonal_periods = seasonal_periods
        
        # Learnable decomposition networks
        self.trend_net = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model, d_model)
        )
        
        self.seasonal_nets = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(d_model // 2, d_model)
            ) for _ in seasonal_periods
        ])
        
        self.residual_net = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model // 2, d_model)
        )
        
        # Combination weights
        self.combination_weights = nn.Parameter(torch.ones(len(seasonal_periods) + 2))
        
    def forward(self, x: torch.Tensor, time_indices: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, d_model = x.shape
        
        # Trend component
        trend = self.trend_net(x)
        
        # Seasonal components
        seasonal_components = []
        for i, (period, seasonal_net) in enumerate(zip(self.seasonal_periods, self.seasonal_nets)):
            seasonal = seasonal_net(x)
            seasonal_components.append(seasonal)
        
        # Residual component
        residual = self.residual_net(x)
        
        # Weighted combination
        weights = F.softmax(self.combination_weights, dim=0)
        result = weights[0] * trend + weights[-1] * residual
        
        for i, seasonal in enumerate(seasonal_components):
            result += weights[i + 1] * seasonal
        
        return result

class TemporalFusionTransformer(nn.Module):
    """Transformer block with temporal fusion capabilities"""
    
    def __init__(self, d_model: int, n_heads: int = 8, d_ff: int = 2048, 
                 dropout: float = 0.1, max_relative_position: int = 512):
        super().__init__()
        
        # Temporal attention
        self.temporal_attention = TemporalAttentionLayer(
            d_model, n_heads, dropout, max_relative_position
        )
        
        # Feed-forward network
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Temporal attention
        attn_output = self.temporal_attention(x, mask)
        x = self.norm1(x + attn_output)
        
        # Feed-forward
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x

class TemporalAttentionTVModel(nn.Module):
    """
    Complete Temporal Attention TV Model (TAT-TV)
    
    Captures:
    - Seasonal viewing patterns (weekly, monthly, yearly)
    - Temporal dynamics in show popularity
    - Long-range temporal dependencies
    - Trend analysis and forecasting
    """
    
    def __init__(self,
                 vocab_sizes: Dict[str, int],
                 d_model: int = 512,
                 n_heads: int = 8,
                 num_layers: int = 6,
                 d_ff: int = 2048,
                 max_seq_length: int = 1024,
                 dropout: float = 0.1,
                 seasonal_periods: List[int] = [7, 30, 365],
                 use_seasonal_decomposition: bool = True):
        super().__init__()
        
        self.d_model = d_model
        self.max_seq_length = max_seq_length
        self.use_seasonal_decomposition = use_seasonal_decomposition
        
        # Feature embeddings
        self.show_embedding = nn.Embedding(vocab_sizes.get('shows', 10000), d_model, padding_idx=0)
        self.genre_embedding = nn.Embedding(vocab_sizes.get('genres', 50), d_model // 4, padding_idx=0)
        self.network_embedding = nn.Embedding(vocab_sizes.get('networks', 100), d_model // 4, padding_idx=0)

        # Projection layer for combined embeddings (show + genre + network)
        # Max combined dim: d_model + d_model//4 + d_model//4 = d_model * 1.5
        combined_embed_dim = d_model + (d_model // 4) * 2
        self.combined_projection = nn.Linear(combined_embed_dim, d_model)

        # Temporal feature projection
        self.temporal_projection = nn.Sequential(
            nn.Linear(20, d_model // 2),  # Time-based features
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, d_model)
        )
        
        # Seasonal positional encoding
        self.seasonal_pos_encoding = SeasonalPositionalEncoding(
            d_model, max_seq_length, seasonal_periods
        )
        
        # Seasonal decomposition
        if use_seasonal_decomposition:
            self.seasonal_decomposition = SeasonalDecomposition(d_model, seasonal_periods)
        
        # Transformer layers
        self.transformer_layers = nn.ModuleList([
            TemporalFusionTransformer(
                d_model, n_heads, d_ff, dropout, max_relative_position=512
            ) for _ in range(num_layers)
        ])
        
        # Output heads
        self.popularity_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1)
        )
        
        self.trend_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1)
        )
        
        # Seasonal forecast heads
        self.seasonal_heads = nn.ModuleDict({
            f'seasonal_{period}': nn.Sequential(
                nn.Linear(d_model, d_model // 4),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model // 4, 1)
            ) for period in seasonal_periods
        })
        
        # Global attention pooling
        self.global_attention = nn.MultiheadAttention(
            d_model, num_heads=4, dropout=dropout, batch_first=True
        )
        
        # Final projection
        self.final_projection = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model)
        )
        
    def create_temporal_features(self, timestamps: torch.Tensor) -> torch.Tensor:
        """Create rich temporal features from timestamps (vectorized)"""
        # timestamps shape: (batch_size, seq_len)
        # Ensure float for calculations
        ts = timestamps.float()

        # Handle invalid timestamps (zeros or negatives)
        valid_mask = ts > 0
        # Use a default timestamp for invalid entries to avoid NaN
        ts = torch.where(valid_mask, ts, torch.ones_like(ts) * 1609459200.0)  # 2021-01-01

        # Extract time components using vectorized operations
        # Seconds since midnight
        seconds_in_day = ts % 86400
        hours = seconds_in_day / 3600

        # Days since epoch (approximate)
        days_since_epoch = ts / 86400

        # Day of week (0=Thursday for Unix epoch, so adjust)
        day_of_week = (days_since_epoch + 4) % 7  # 0=Monday

        # Approximate month and day of year
        # Days in a year approximately
        days_in_year = days_since_epoch % 365.25
        month_approx = (days_in_year / 30.44) % 12  # Approximate month (0-11)
        day_of_year_norm = days_in_year / 366

        # Year (approximate)
        year_approx = 1970 + days_since_epoch / 365.25
        year_norm = (year_approx - 2000) / 25

        # Cyclical time features
        two_pi = 2 * math.pi
        hour_sin = torch.sin(two_pi * hours / 24)
        hour_cos = torch.cos(two_pi * hours / 24)
        day_sin = torch.sin(two_pi * day_of_week / 7)
        day_cos = torch.cos(two_pi * day_of_week / 7)
        month_sin = torch.sin(two_pi * month_approx / 12)
        month_cos = torch.cos(two_pi * month_approx / 12)

        # Season indicators (based on approximate month)
        is_winter = ((month_approx >= 11) | (month_approx < 2)).float()
        is_spring = ((month_approx >= 2) & (month_approx < 5)).float()
        is_summer = ((month_approx >= 5) & (month_approx < 8)).float()
        is_fall = ((month_approx >= 8) & (month_approx < 11)).float()

        # Weekend indicator (Saturday=5, Sunday=6)
        is_weekend = (day_of_week >= 5).float()

        # Holiday proximity (placeholder)
        holiday_proximity = torch.zeros_like(ts)

        # TV season indicators
        is_tv_season = ((month_approx >= 8) | (month_approx < 4)).float()
        is_summer_break = ((month_approx >= 5) & (month_approx < 8)).float()

        # Prime time indicator (7pm-10pm = hours 19-22)
        is_prime_time = ((hours >= 19) & (hours <= 22)).float()

        # Padding features
        padding = torch.zeros_like(ts)

        # Stack all features: (batch, seq, 20)
        features = torch.stack([
            hour_sin, hour_cos, day_sin, day_cos, month_sin, month_cos,
            year_norm, day_of_year_norm, is_winter, is_spring, is_summer, is_fall,
            is_weekend, holiday_proximity, is_tv_season, is_summer_break,
            is_prime_time, padding, padding, padding
        ], dim=-1)

        # Zero out features for invalid timestamps
        features = features * valid_mask.unsqueeze(-1).float()

        return features
    
    def forward(self,
                show_ids: torch.Tensor,
                timestamps: torch.Tensor,
                genre_ids: Optional[torch.Tensor] = None,
                network_ids: Optional[torch.Tensor] = None,
                mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        
        batch_size, seq_len = show_ids.shape
        
        # Show embeddings
        show_embeds = self.show_embedding(show_ids)

        # Genre and network embeddings (use zeros if not provided)
        if genre_ids is not None:
            genre_embeds = self.genre_embedding(genre_ids).mean(dim=-2)  # Average over genres
        else:
            genre_embeds = torch.zeros(batch_size, seq_len, self.d_model // 4, device=show_embeds.device)

        if network_ids is not None:
            network_embeds = self.network_embedding(network_ids)
        else:
            network_embeds = torch.zeros(batch_size, seq_len, self.d_model // 4, device=show_embeds.device)

        # Combine and project embeddings
        combined_embeds = torch.cat([show_embeds, genre_embeds, network_embeds], dim=-1)
        x = self.combined_projection(combined_embeds)
        
        # Create temporal features
        temporal_features = self.create_temporal_features(timestamps)
        temporal_embeds = self.temporal_projection(temporal_features)
        
        # Combine show and temporal embeddings
        x = x + temporal_embeds
        
        # Add seasonal positional encoding
        time_indices = ((timestamps - timestamps.min()) / (24 * 3600)).long()  # Day indices
        x = self.seasonal_pos_encoding(x.transpose(0, 1), time_indices).transpose(0, 1)
        
        # Seasonal decomposition
        if self.use_seasonal_decomposition:
            x = self.seasonal_decomposition(x, time_indices)
        
        # Apply transformer layers
        for layer in self.transformer_layers:
            x = layer(x, mask)
        
        # Global attention pooling for sequence-level prediction
        global_context, _ = self.global_attention(x, x, x, key_padding_mask=mask == 0 if mask is not None else None)
        
        # Mean pooling over sequence
        if mask is not None:
            mask_expanded = mask.unsqueeze(-1).expand_as(global_context)
            global_context = global_context * mask_expanded
            seq_representation = global_context.sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp(min=1)
        else:
            seq_representation = global_context.mean(dim=1)
        
        # Final projection
        final_features = self.final_projection(seq_representation)
        
        # Output predictions
        popularity_pred = self.popularity_head(final_features)
        outputs = {
            'sequence_features': x,
            'global_features': final_features,
            'popularity_prediction': popularity_pred,
            'predictions': popularity_pred,  # Alias for training compatibility
            'trend_prediction': self.trend_head(final_features)
        }
        
        # Seasonal predictions
        for period_name, seasonal_head in self.seasonal_heads.items():
            outputs[f'{period_name}_prediction'] = seasonal_head(final_features)
        
        return outputs
    
    def forecast(self, 
                 show_ids: torch.Tensor,
                 timestamps: torch.Tensor,
                 forecast_steps: int = 30,
                 genre_ids: Optional[torch.Tensor] = None,
                 network_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forecast future popularity"""
        
        with torch.no_grad():
            # Get current features
            outputs = self.forward(show_ids, timestamps, genre_ids, network_ids)
            
            # Create future timestamps
            last_timestamp = timestamps[:, -1].unsqueeze(1)
            future_timestamps = last_timestamp + torch.arange(1, forecast_steps + 1, device=timestamps.device) * 86400  # Daily steps
            
            # Extend show_ids for forecast period
            last_show_id = show_ids[:, -1].unsqueeze(1)
            future_show_ids = last_show_id.expand(-1, forecast_steps)
            
            # Get forecast features
            forecast_outputs = self.forward(future_show_ids, future_timestamps, genre_ids, network_ids)
            
            return forecast_outputs['popularity_prediction']

class TemporalLoss(nn.Module):
    """Multi-task loss for temporal modeling"""
    
    def __init__(self, 
                 popularity_weight: float = 1.0,
                 trend_weight: float = 0.5,
                 seasonal_weight: float = 0.3,
                 forecast_weight: float = 0.8):
        super().__init__()
        self.popularity_weight = popularity_weight
        self.trend_weight = trend_weight
        self.seasonal_weight = seasonal_weight
        self.forecast_weight = forecast_weight
        
        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()
        
    def forward(self,
                outputs: Dict[str, torch.Tensor],
                targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        
        losses = {}
        total_loss = 0
        
        # Popularity prediction loss
        if 'popularity_prediction' in outputs and 'popularity' in targets:
            pop_loss = self.mse_loss(outputs['popularity_prediction'], targets['popularity'])
            losses['popularity_loss'] = pop_loss
            total_loss += self.popularity_weight * pop_loss
        
        # Trend prediction loss
        if 'trend_prediction' in outputs and 'trend' in targets:
            trend_loss = self.l1_loss(outputs['trend_prediction'], targets['trend'])
            losses['trend_loss'] = trend_loss
            total_loss += self.trend_weight * trend_loss
        
        # Seasonal losses
        seasonal_periods = [7, 30, 365]
        for period in seasonal_periods:
            seasonal_key = f'seasonal_{period}_prediction'
            target_key = f'seasonal_{period}'
            
            if seasonal_key in outputs and target_key in targets:
                seasonal_loss = self.mse_loss(outputs[seasonal_key], targets[target_key])
                losses[f'seasonal_{period}_loss'] = seasonal_loss
                total_loss += self.seasonal_weight * seasonal_loss
        
        # Forecast loss (if available)
        if 'forecast_prediction' in outputs and 'forecast_target' in targets:
            forecast_loss = self.mse_loss(outputs['forecast_prediction'], targets['forecast_target'])
            losses['forecast_loss'] = forecast_loss
            total_loss += self.forecast_weight * forecast_loss
        
        losses['total_loss'] = total_loss
        return losses

# Configuration for RTX 4090 optimization
def get_temporal_config():
    """Optimized configuration for temporal attention model on RTX 4090"""
    return {
        'd_model': 512,
        'n_heads': 8,
        'num_layers': 6,
        'd_ff': 2048,
        'max_seq_length': 1024,
        'dropout': 0.1,
        'seasonal_periods': [7, 30, 365],
        'use_seasonal_decomposition': True,
        'batch_size': 64,  # Can handle larger batches than transformers
        'learning_rate': 1e-4,
        'weight_decay': 1e-5,
        'warmup_steps': 1000,
        'max_grad_norm': 1.0,
        'use_mixed_precision': True,
        'gradient_accumulation_steps': 1,
        'patience': 15,
        'epochs': 40,
    }