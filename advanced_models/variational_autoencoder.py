# Variational Autoencoder Models for CineSync v2
# Implements MultVAE and enhanced variants for collaborative filtering
# Uses probabilistic latent representations for robust recommendations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Optional, List
import numpy as np
import math


class VariationalEncoder(nn.Module):
    """Variational encoder for collaborative filtering
    
    Encodes user interaction patterns into probabilistic latent representations.
    Uses reparameterization trick to enable backpropagation through stochastic
    sampling, allowing the model to learn meaningful latent user preferences.
    """
    
    def __init__(self, input_dim: int, hidden_dims: List[int], latent_dim: int, dropout: float = 0.5):
        """Initialize variational encoder
        
        Args:
            input_dim: Size of input layer (number of items)
            hidden_dims: List of hidden layer sizes
            latent_dim: Size of latent representation
            dropout: Dropout rate for regularization
        """
        super(VariationalEncoder, self).__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        # Build encoder layers with tanh activation and dropout
        encoder_layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),  # Linear transformation
                nn.Tanh(),                        # Non-linear activation
                nn.Dropout(dropout)               # Regularization
            ])
            prev_dim = hidden_dim
        
        # Combine all encoder layers into sequential model
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Separate projections for mean and log-variance of latent distribution
        self.mu_layer = nn.Linear(prev_dim, latent_dim)      # Mean of q(z|x)
        self.logvar_layer = nn.Linear(prev_dim, latent_dim)  # Log-variance of q(z|x)
        
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through encoder
        
        Args:
            x: Input tensor [batch_size, input_dim]
            
        Returns:
            Tuple of (z, mu, logvar) where z is sampled latent representation
        """
        # Encode
        h = self.encoder(x)
        
        # Compute mean and log variance
        mu = self.mu_layer(h)
        logvar = self.logvar_layer(h)
        
        # Reparameterization trick
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z = mu + eps * std
        else:
            z = mu
        
        return z, mu, logvar


class VariationalDecoder(nn.Module):
    """Variational decoder for collaborative filtering"""
    
    def __init__(self, latent_dim: int, hidden_dims: List[int], output_dim: int, dropout: float = 0.5):
        super(VariationalDecoder, self).__init__()
        
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        
        # Decoder layers
        decoder_layers = []
        prev_dim = latent_dim
        
        for hidden_dim in hidden_dims:
            decoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.Tanh(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        # Output layer
        decoder_layers.append(nn.Linear(prev_dim, output_dim))
        
        self.decoder = nn.Sequential(*decoder_layers)
        
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through decoder
        
        Args:
            z: Latent representation [batch_size, latent_dim]
            
        Returns:
            Reconstructed output [batch_size, output_dim]
        """
        return self.decoder(z)


class MultVAE(nn.Module):
    """
    Multinomial Variational Autoencoder for Collaborative Filtering
    
    Paper: Liang, D., et al. "Variational autoencoders for collaborative filtering." WWW 2018.
    """
    
    def __init__(
        self,
        num_items: int,
        hidden_dims: List[int] = [600, 200],
        latent_dim: int = 200,
        dropout: float = 0.5,
        beta: float = 1.0,
        gamma: float = 0.005
    ):
        super(MultVAE, self).__init__()
        
        self.num_items = num_items
        self.latent_dim = latent_dim
        self.beta = beta  # KL divergence weight
        self.gamma = gamma  # L2 regularization weight
        
        # Encoder and decoder
        self.encoder = VariationalEncoder(num_items, hidden_dims, latent_dim, dropout)
        
        # Decoder uses reverse hidden dimensions
        decoder_hidden_dims = hidden_dims[::-1]
        self.decoder = VariationalDecoder(latent_dim, decoder_hidden_dims, num_items, dropout)
        
        # Prior parameters (learnable)
        self.prior_mu = nn.Parameter(torch.zeros(latent_dim))
        self.prior_logvar = nn.Parameter(torch.zeros(latent_dim))
    
    def forward(self, user_ratings: torch.Tensor, 
                beta: Optional[float] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass
        
        Args:
            user_ratings: User rating matrix [batch_size, num_items]
            beta: Optional beta override for annealing
            
        Returns:
            Dictionary with reconstructions and latent variables
        """
        if beta is None:
            beta = self.beta
        
        # Normalize input (convert to probabilities)
        normalized_ratings = F.normalize(user_ratings, p=1, dim=1)
        
        # Encode
        z, mu, logvar = self.encoder(normalized_ratings)
        
        # Decode
        logits = self.decoder(z)
        
        # Compute losses
        # Reconstruction loss (multinomial likelihood)
        recon_loss = -torch.sum(F.log_softmax(logits, dim=1) * normalized_ratings, dim=1)
        
        # KL divergence with learned prior
        prior_mu = self.prior_mu.expand_as(mu)
        prior_logvar = self.prior_logvar.expand_as(logvar)
        
        kl_loss = 0.5 * torch.sum(
            prior_logvar - logvar + 
            (torch.exp(logvar) + (mu - prior_mu).pow(2)) / torch.exp(prior_logvar) - 1,
            dim=1
        )
        
        # Total loss
        total_loss = recon_loss + beta * kl_loss
        
        return {
            'logits': logits,
            'z': z,
            'mu': mu,
            'logvar': logvar,
            'recon_loss': recon_loss,
            'kl_loss': kl_loss,
            'total_loss': total_loss
        }
    
    def predict(self, user_ratings: torch.Tensor, k: int = 10) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Make predictions for users
        
        Args:
            user_ratings: User rating matrix [batch_size, num_items]
            k: Number of recommendations
            
        Returns:
            Tuple of (top_k_items, top_k_scores)
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(user_ratings)
            logits = outputs['logits']
            
            # Convert to probabilities
            probs = F.softmax(logits, dim=1)
            
            # Mask out items user has already rated
            mask = (user_ratings > 0).float()
            probs = probs * (1 - mask)
            
            # Get top-k
            top_k_probs, top_k_items = torch.topk(probs, k, dim=1)
            
            return top_k_items, top_k_probs


class HierarchicalVAE(nn.Module):
    """
    Modern Hierarchical VAE with advanced techniques:
    - Hierarchical latent spaces
    - Normalizing flows
    - Adversarial training
    - Disentangled representations
    - Contrastive learning
    """
    
    def __init__(
        self,
        num_items: int,
        num_users: int,
        num_genres: int = 20,
        hidden_dims: List[int] = [1024, 512, 256],
        latent_dim: int = 256,
        num_flows: int = 4,
        dropout: float = 0.2,
        beta: float = 1.0,
        use_flows: bool = True,
        use_adversarial: bool = True,
        temperature: float = 0.07
    ):
        super(HierarchicalVAE, self).__init__()
        
        self.num_items = num_items
        self.num_users = num_users
        self.latent_dim = latent_dim
        self.beta = beta
        self.use_flows = use_flows
        self.use_adversarial = use_adversarial
        self.temperature = temperature
        
        # Hierarchical encoder structure
        self.global_encoder = VariationalEncoder(num_items, hidden_dims, latent_dim, dropout)
        self.local_encoder = VariationalEncoder(num_items, hidden_dims[:2], latent_dim // 2, dropout)
        
        # Normalizing flows for posterior approximation
        if use_flows:
            self.flows = nn.ModuleList([
                PlanarFlow(latent_dim) for _ in range(num_flows)
            ])
        
        # Hierarchical decoder with attention
        self.global_decoder = HierarchicalDecoder(latent_dim, hidden_dims[::-1], num_items, dropout)
        self.local_decoder = HierarchicalDecoder(latent_dim // 2, hidden_dims[:2][::-1], num_items, dropout)
        
        # Adversarial discriminator
        if use_adversarial:
            self.discriminator = nn.Sequential(
                nn.Linear(latent_dim, hidden_dims[0]),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dims[0], hidden_dims[1]),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dims[1], 1)
            )
        
        # Disentanglement components
        self.content_encoder = VariationalEncoder(num_items, hidden_dims[:2], latent_dim // 4, dropout)
        self.style_encoder = VariationalEncoder(num_items, hidden_dims[:2], latent_dim // 4, dropout)
        
        # Cross-modal fusion
        self.fusion_layer = nn.Sequential(
            nn.Linear(latent_dim + latent_dim // 2, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Contrastive projection head
        self.projection_head = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim // 2)
        )
        
        # Multi-task heads
        self.reconstruction_head = nn.Linear(latent_dim, num_items)
        self.genre_head = nn.Linear(latent_dim, num_genres)
        self.popularity_head = nn.Linear(latent_dim, 10)  # Popularity buckets
        
        # Beta scheduling
        self.beta_scheduler = BetaScheduler(beta_max=beta, warmup_steps=5000)
        
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def reparameterize_with_flows(self, mu, logvar):
        """Reparameterization with normalizing flows"""
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z = mu + eps * std
            
            if self.use_flows:
                log_det_jacobian = 0
                for flow in self.flows:
                    z, ldj = flow(z)
                    log_det_jacobian += ldj
                return z, log_det_jacobian
            else:
                return z, torch.zeros_like(mu[:, 0])
        else:
            if self.use_flows:
                z = mu
                for flow in self.flows:
                    z, _ = flow(z)
                return z, torch.zeros_like(mu[:, 0])
            else:
                return mu, torch.zeros_like(mu[:, 0])
    
    def forward(self, user_ratings, step=None, return_all=False):
        batch_size = user_ratings.size(0)
        normalized_ratings = F.normalize(user_ratings, p=1, dim=1)
        
        # Hierarchical encoding
        global_z, global_mu, global_logvar = self.global_encoder(normalized_ratings)
        local_z, local_mu, local_logvar = self.local_encoder(normalized_ratings)
        
        # Apply flows to global latent space
        if self.use_flows:
            global_z, flow_ldj = self.reparameterize_with_flows(global_mu, global_logvar)
        else:
            flow_ldj = torch.zeros_like(global_mu[:, 0])
        
        # Disentangled representations
        content_z, content_mu, content_logvar = self.content_encoder(normalized_ratings)
        style_z, style_mu, style_logvar = self.style_encoder(normalized_ratings)
        
        # Fusion of hierarchical representations
        combined_z = torch.cat([global_z, local_z], dim=1)
        fused_z = self.fusion_layer(combined_z)
        
        # Hierarchical decoding
        global_recon = self.global_decoder(global_z)
        local_recon = self.local_decoder(local_z)
        
        # Final reconstruction
        reconstruction = self.reconstruction_head(fused_z)
        
        # Multi-task predictions
        genre_pred = self.genre_head(fused_z)
        popularity_pred = self.popularity_head(fused_z)
        
        # Compute losses
        current_beta = self.beta_scheduler.get_beta(step) if step is not None else self.beta
        
        # Reconstruction loss with hierarchical weighting
        recon_loss = (
            0.6 * F.mse_loss(reconstruction, normalized_ratings, reduction='none').sum(dim=1) +
            0.3 * F.mse_loss(global_recon, normalized_ratings, reduction='none').sum(dim=1) +
            0.1 * F.mse_loss(local_recon, normalized_ratings, reduction='none').sum(dim=1)
        )
        
        # KL divergences with improved calculation
        global_kl = self._kl_divergence(global_mu, global_logvar) - flow_ldj
        local_kl = self._kl_divergence(local_mu, local_logvar)
        content_kl = self._kl_divergence(content_mu, content_logvar)
        style_kl = self._kl_divergence(style_mu, style_logvar)
        
        total_kl = global_kl + 0.5 * local_kl + 0.3 * content_kl + 0.3 * style_kl
        
        # Disentanglement loss (mutual information)
        disentangle_loss = self._mutual_information_loss(content_z, style_z)
        
        # Adversarial loss
        adversarial_loss = torch.zeros_like(recon_loss)
        if self.use_adversarial:
            real_labels = torch.ones(batch_size, 1, device=user_ratings.device)
            fake_labels = torch.zeros(batch_size, 1, device=user_ratings.device)
            
            # Generator loss (fool discriminator)
            gen_loss = F.binary_cross_entropy_with_logits(
                self.discriminator(fused_z), real_labels, reduction='none'
            ).squeeze()
            adversarial_loss = 0.1 * gen_loss
        
        # Total loss
        total_loss = (
            recon_loss + 
            current_beta * total_kl + 
            0.1 * disentangle_loss + 
            adversarial_loss
        )
        
        outputs = {
            'reconstruction': reconstruction,
            'global_recon': global_recon,
            'local_recon': local_recon,
            'genre_pred': genre_pred,
            'popularity_pred': popularity_pred,
            'global_z': global_z,
            'local_z': local_z,
            'fused_z': fused_z,
            'recon_loss': recon_loss,
            'kl_loss': total_kl,
            'disentangle_loss': disentangle_loss,
            'adversarial_loss': adversarial_loss,
            'total_loss': total_loss,
            'beta': current_beta
        }
        
        if return_all:
            outputs.update({
                'content_z': content_z,
                'style_z': style_z,
                'flow_ldj': flow_ldj,
                'global_mu': global_mu,
                'global_logvar': global_logvar
            })
        
        return outputs
    
    def _kl_divergence(self, mu, logvar):
        """Improved KL divergence calculation"""
        return 0.5 * torch.sum(logvar.exp() + mu.pow(2) - logvar - 1, dim=1)
    
    def _mutual_information_loss(self, content_z, style_z):
        """Mutual information loss for disentanglement"""
        # Simple implementation using correlation
        content_norm = F.normalize(content_z, dim=1)
        style_norm = F.normalize(style_z, dim=1)
        
        correlation = torch.abs(torch.sum(content_norm * style_norm, dim=1))
        return correlation.mean()
    
    def compute_contrastive_loss(self, anchor_ratings, positive_ratings, negative_ratings):
        """Contrastive learning for better representations"""
        anchor_out = self.forward(anchor_ratings)
        positive_out = self.forward(positive_ratings)
        negative_out = self.forward(negative_ratings)
        
        # Project to contrastive space
        anchor_proj = F.normalize(self.projection_head(anchor_out['fused_z']), dim=1)
        positive_proj = F.normalize(self.projection_head(positive_out['fused_z']), dim=1)
        negative_proj = F.normalize(self.projection_head(negative_out['fused_z']), dim=1)
        
        # Compute similarities
        pos_sim = torch.sum(anchor_proj * positive_proj, dim=1) / self.temperature
        neg_sim = torch.sum(anchor_proj * negative_proj, dim=1) / self.temperature
        
        # InfoNCE loss
        logits = torch.cat([pos_sim.unsqueeze(1), neg_sim.unsqueeze(1)], dim=1)
        labels = torch.zeros(anchor_ratings.size(0), dtype=torch.long, device=anchor_ratings.device)
        
        return F.cross_entropy(logits, labels)


class PlanarFlow(nn.Module):
    """Planar normalizing flow for improved posterior approximation"""
    
    def __init__(self, dim):
        super(PlanarFlow, self).__init__()
        self.weight = nn.Parameter(torch.randn(1, dim))
        self.bias = nn.Parameter(torch.randn(1))
        self.scale = nn.Parameter(torch.randn(1, dim))
        
    def forward(self, z):
        # Planar transformation: f(z) = z + u * tanh(w^T z + b)
        linear_term = F.linear(z, self.weight, self.bias)
        transformed = z + self.scale * torch.tanh(linear_term)
        
        # Log determinant of Jacobian
        psi = (1 - torch.tanh(linear_term) ** 2) * self.weight
        log_det_jacobian = torch.log(torch.abs(1 + torch.sum(psi * self.scale, dim=1)) + 1e-8)
        
        return transformed, log_det_jacobian


class HierarchicalDecoder(nn.Module):
    """Hierarchical decoder with attention mechanisms"""
    
    def __init__(self, latent_dim, hidden_dims, output_dim, dropout=0.2):
        super(HierarchicalDecoder, self).__init__()
        
        # Multi-level decoding
        self.decoders = nn.ModuleList()
        prev_dim = latent_dim
        
        for hidden_dim in hidden_dims:
            self.decoders.append(nn.Sequential(
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ))
            prev_dim = hidden_dim
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(
            embed_dim=prev_dim,
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )
        
        # Output layer
        self.output_layer = nn.Linear(prev_dim, output_dim)
        
    def forward(self, z):
        x = z
        
        # Multi-level decoding
        for decoder in self.decoders:
            x = decoder(x)
        
        # Self-attention for better reconstruction
        x_expanded = x.unsqueeze(1)  # Add sequence dimension
        attended_x, _ = self.attention(x_expanded, x_expanded, x_expanded)
        x = attended_x.squeeze(1)
        
        return self.output_layer(x)


class BetaScheduler:
    """Beta scheduling for VAE training"""
    
    def __init__(self, beta_max=1.0, warmup_steps=5000, schedule='linear'):
        self.beta_max = beta_max
        self.warmup_steps = warmup_steps
        self.schedule = schedule
    
    def get_beta(self, step):
        if step is None:
            return self.beta_max
        
        if self.schedule == 'linear':
            return min(self.beta_max, step / self.warmup_steps * self.beta_max)
        elif self.schedule == 'cosine':
            if step < self.warmup_steps:
                return 0.5 * self.beta_max * (1 - math.cos(math.pi * step / self.warmup_steps))
            else:
                return self.beta_max
        else:
            return self.beta_max


class EnhancedMultVAE(MultVAE):
    """
    Enhanced MultVAE with additional features for better performance
    """
    
    def __init__(
        self,
        num_items: int,
        num_users: int,
        num_genres: int = 20,
        hidden_dims: List[int] = [1024, 512, 256],
        latent_dim: int = 256,
        dropout: float = 0.3,
        beta: float = 1.0,
        gamma: float = 0.005,
        use_user_features: bool = True,
        use_item_features: bool = True,
        use_attention: bool = True
    ):
        super(EnhancedMultVAE, self).__init__(
            num_items, hidden_dims, latent_dim, dropout, beta, gamma
        )
        
        self.num_users = num_users
        self.num_genres = num_genres
        self.use_user_features = use_user_features
        self.use_item_features = use_item_features
        self.use_attention = use_attention
        
        # User embeddings for cold start
        if use_user_features:
            self.user_embedding = nn.Embedding(num_users, latent_dim // 2)
            self.user_projection = nn.Linear(latent_dim // 2, latent_dim)
        
        # Item embeddings for content-based features
        if use_item_features:
            self.item_embedding = nn.Embedding(num_items, latent_dim // 4)
            self.genre_embedding = nn.Embedding(num_genres, latent_dim // 4)
            
            # Content fusion
            self.content_fusion = nn.Sequential(
                nn.Linear(latent_dim // 2, latent_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(latent_dim // 2, num_items)
            )
        
        # Attention mechanism for item importance
        if use_attention:
            self.attention = nn.MultiheadAttention(
                embed_dim=latent_dim,
                num_heads=8,
                dropout=dropout,
                batch_first=True
            )
            self.attention_norm = nn.LayerNorm(latent_dim)
        
        # Enhanced decoder with residual connections
        self.residual_decoder = nn.ModuleList([
            nn.Sequential(
                nn.Linear(latent_dim, latent_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ) for _ in range(3)
        ])
        
        # Multi-head decoder for different recommendation aspects
        self.rating_head = nn.Linear(latent_dim, num_items)
        self.popularity_head = nn.Linear(latent_dim, num_items)
        self.diversity_head = nn.Linear(latent_dim, num_items)
        
        # Gating mechanism to combine different heads
        self.gate = nn.Sequential(
            nn.Linear(latent_dim, 3),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, user_ratings: torch.Tensor, 
                user_ids: Optional[torch.Tensor] = None,
                item_genres: Optional[torch.Tensor] = None,
                beta: Optional[float] = None) -> Dict[str, torch.Tensor]:
        """
        Enhanced forward pass with additional features
        
        Args:
            user_ratings: User rating matrix [batch_size, num_items]
            user_ids: User IDs for cold start [batch_size]
            item_genres: Item genre matrix [num_items, num_genres]
            beta: Optional beta override
            
        Returns:
            Enhanced outputs with multiple prediction heads
        """
        if beta is None:
            beta = self.beta
        
        # Base VAE forward pass
        normalized_ratings = F.normalize(user_ratings, p=1, dim=1)
        z_base, mu, logvar = self.encoder(normalized_ratings)
        
        # Enhanced latent representation
        z = z_base
        
        # Add user features for cold start
        if self.use_user_features and user_ids is not None:
            user_emb = self.user_embedding(user_ids)
            user_features = self.user_projection(user_emb)
            z = z + 0.1 * user_features  # Small additive contribution
        
        # Apply attention if enabled
        if self.use_attention:
            z_expanded = z.unsqueeze(1)  # [batch_size, 1, latent_dim]
            z_attended, _ = self.attention(z_expanded, z_expanded, z_expanded)
            z = self.attention_norm(z + z_attended.squeeze(1))
        
        # Residual connections in decoder
        h = z
        for residual_layer in self.residual_decoder:
            h_residual = residual_layer(h)
            h = h + h_residual  # Residual connection
        
        # Multi-head predictions
        rating_logits = self.rating_head(h)
        popularity_logits = self.popularity_head(h)
        diversity_logits = self.diversity_head(h)
        
        # Gating mechanism
        gate_weights = self.gate(h)  # [batch_size, 3]
        
        # Combine predictions
        combined_logits = (
            gate_weights[:, 0:1] * rating_logits +
            gate_weights[:, 1:2] * popularity_logits +
            gate_weights[:, 2:3] * diversity_logits
        )
        
        # Content-based enhancement
        if self.use_item_features and item_genres is not None:
            # Simple content boosting
            content_features = self.content_fusion(h)
            combined_logits = combined_logits + 0.1 * content_features
        
        # Compute losses (similar to base MultVAE)
        recon_loss = -torch.sum(F.log_softmax(combined_logits, dim=1) * normalized_ratings, dim=1)
        
        prior_mu = self.prior_mu.expand_as(mu)
        prior_logvar = self.prior_logvar.expand_as(logvar)
        
        kl_loss = 0.5 * torch.sum(
            prior_logvar - logvar + 
            (torch.exp(logvar) + (mu - prior_mu).pow(2)) / torch.exp(prior_logvar) - 1,
            dim=1
        )
        
        # Additional regularization on gate weights
        gate_reg = torch.mean(torch.sum(gate_weights * torch.log(gate_weights + 1e-8), dim=1))
        
        total_loss = recon_loss + beta * kl_loss + 0.01 * gate_reg
        
        return {
            'logits': combined_logits,
            'rating_logits': rating_logits,
            'popularity_logits': popularity_logits,
            'diversity_logits': diversity_logits,
            'gate_weights': gate_weights,
            'z': z,
            'mu': mu,
            'logvar': logvar,
            'recon_loss': recon_loss,
            'kl_loss': kl_loss,
            'gate_reg': gate_reg,
            'total_loss': total_loss
        }


class DualVAE(nn.Module):
    """
    Dual VAE for both user and item representations
    """
    
    def __init__(
        self,
        num_users: int,
        num_items: int,
        latent_dim: int = 256,
        hidden_dims: List[int] = [512, 256],
        dropout: float = 0.3,
        beta: float = 1.0
    ):
        super(DualVAE, self).__init__()
        
        self.num_users = num_users
        self.num_items = num_items
        self.latent_dim = latent_dim
        self.beta = beta
        
        # User VAE (items as features)
        self.user_encoder = VariationalEncoder(num_items, hidden_dims, latent_dim, dropout)
        self.user_decoder = VariationalDecoder(latent_dim, hidden_dims[::-1], num_items, dropout)
        
        # Item VAE (users as features)
        self.item_encoder = VariationalEncoder(num_users, hidden_dims, latent_dim, dropout)
        self.item_decoder = VariationalDecoder(latent_dim, hidden_dims[::-1], num_users, dropout)
        
        # Cross-reconstruction layers
        self.user_to_item = nn.Linear(latent_dim, latent_dim)
        self.item_to_user = nn.Linear(latent_dim, latent_dim)
        
        # Shared prediction layer
        self.prediction_layer = nn.Bilinear(latent_dim, latent_dim, 1)
    
    def forward(self, user_item_matrix: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Dual VAE forward pass
        
        Args:
            user_item_matrix: User-item interaction matrix [num_users, num_items]
            
        Returns:
            Comprehensive outputs from both VAEs
        """
        # User VAE
        user_ratings = F.normalize(user_item_matrix, p=1, dim=1)
        user_z, user_mu, user_logvar = self.user_encoder(user_ratings)
        user_recon_logits = self.user_decoder(user_z)
        
        # Item VAE  
        item_ratings = F.normalize(user_item_matrix.T, p=1, dim=1)
        item_z, item_mu, item_logvar = self.item_encoder(item_ratings)
        item_recon_logits = self.item_decoder(item_z)
        
        # Cross-reconstruction
        user_from_item = self.item_to_user(item_z)
        item_from_user = self.user_to_item(user_z)
        
        # Prediction using bilinear layer
        # Expand for all user-item pairs
        user_z_expanded = user_z.unsqueeze(1).expand(-1, self.num_items, -1)
        item_z_expanded = item_z.unsqueeze(0).expand(self.num_users, -1, -1)
        
        predictions = self.prediction_layer(user_z_expanded, item_z_expanded).squeeze(-1)
        
        # Compute losses
        user_recon_loss = -torch.sum(F.log_softmax(user_recon_logits, dim=1) * user_ratings, dim=1)
        item_recon_loss = -torch.sum(F.log_softmax(item_recon_logits, dim=1) * item_ratings, dim=1)
        
        # KL losses
        user_kl_loss = 0.5 * torch.sum(user_logvar.exp() + user_mu.pow(2) - user_logvar - 1, dim=1)
        item_kl_loss = 0.5 * torch.sum(item_logvar.exp() + item_mu.pow(2) - item_logvar - 1, dim=1)
        
        # Cross-reconstruction losses
        cross_user_loss = F.mse_loss(user_from_item, user_z.detach())
        cross_item_loss = F.mse_loss(item_from_user, item_z.detach())
        
        total_loss = (
            user_recon_loss.mean() + item_recon_loss.mean() +
            self.beta * (user_kl_loss.mean() + item_kl_loss.mean()) +
            0.1 * (cross_user_loss + cross_item_loss)
        )
        
        return {
            'predictions': predictions,
            'user_logits': user_recon_logits,
            'item_logits': item_recon_logits,
            'user_z': user_z,
            'item_z': item_z,
            'user_recon_loss': user_recon_loss,
            'item_recon_loss': item_recon_loss,
            'user_kl_loss': user_kl_loss,
            'item_kl_loss': item_kl_loss,
            'cross_user_loss': cross_user_loss,
            'cross_item_loss': cross_item_loss,
            'total_loss': total_loss
        }


class VAETrainer:
    """Advanced trainer for VAE models with beta annealing and mixed precision"""
    
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        use_amp: bool = True,
        beta_annealing: bool = True,
        beta_max: float = 1.0,
        beta_steps: int = 10000
    ):
        self.model = model.to(device)
        self.device = device
        self.use_amp = use_amp
        self.beta_annealing = beta_annealing
        self.beta_max = beta_max
        self.beta_steps = beta_steps
        self.step_count = 0
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(0.9, 0.999)
        )
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=5000, T_mult=2, eta_min=1e-6
        )
        
        # Mixed precision scaler
        if use_amp:
            self.scaler = torch.cuda.amp.GradScaler()
    
    def get_beta(self) -> float:
        """Get current beta value for annealing"""
        if not self.beta_annealing:
            return self.beta_max
        
        # Linear annealing
        progress = min(self.step_count / self.beta_steps, 1.0)
        return progress * self.beta_max
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Single training step with beta annealing"""
        self.model.train()
        
        # Move batch to device
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                batch[key] = value.to(self.device)
        
        self.optimizer.zero_grad()
        
        current_beta = self.get_beta()
        
        if self.use_amp:
            with torch.cuda.amp.autocast():
                if isinstance(self.model, (EnhancedMultVAE, MultVAE)):
                    outputs = self.model(
                        user_ratings=batch['user_ratings'],
                        user_ids=batch.get('user_ids'),
                        item_genres=batch.get('item_genres'),
                        beta=current_beta
                    )
                elif isinstance(self.model, DualVAE):
                    outputs = self.model(batch['user_item_matrix'])
                else:
                    outputs = self.model(batch['user_ratings'], beta=current_beta)
                
                loss = outputs['total_loss'].mean() if outputs['total_loss'].dim() > 0 else outputs['total_loss']
            
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            if isinstance(self.model, (EnhancedMultVAE, MultVAE)):
                outputs = self.model(
                    user_ratings=batch['user_ratings'],
                    user_ids=batch.get('user_ids'),
                    item_genres=batch.get('item_genres'),
                    beta=current_beta
                )
            elif isinstance(self.model, DualVAE):
                outputs = self.model(batch['user_item_matrix'])
            else:
                outputs = self.model(batch['user_ratings'], beta=current_beta)
            
            loss = outputs['total_loss'].mean() if outputs['total_loss'].dim() > 0 else outputs['total_loss']
            loss.backward()
            self.optimizer.step()
        
        self.scheduler.step()
        self.step_count += 1
        
        # Prepare metrics
        metrics = {
            'total_loss': loss.item(),
            'beta': current_beta,
            'lr': self.optimizer.param_groups[0]['lr']
        }
        
        # Add component losses if available
        if 'recon_loss' in outputs:
            metrics['recon_loss'] = outputs['recon_loss'].mean().item()
        if 'kl_loss' in outputs:
            metrics['kl_loss'] = outputs['kl_loss'].mean().item()
        
        return metrics