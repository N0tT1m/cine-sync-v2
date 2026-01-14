"""
Unified Cross-Domain Embeddings for CineSync v2
Shares user knowledge between movies and TV shows for better recommendations
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union
import numpy as np


class UnifiedUserEmbedding(nn.Module):
    """
    Unified user embedding that works across movies and TV shows.

    Key benefits:
    - Transfers knowledge between domains (movies ↔ TV)
    - Handles cold-start better (learn from one domain, apply to another)
    - More efficient (single user representation)
    - Captures cross-domain preferences
    """

    def __init__(
        self,
        num_users: int,
        embedding_dim: int = 512,
        num_domains: int = 2,  # movies, tv
        use_domain_adaptation: bool = True,
        use_preference_disentanglement: bool = True,
        dropout: float = 0.1
    ):
        super(UnifiedUserEmbedding, self).__init__()

        self.num_users = num_users
        self.embedding_dim = embedding_dim
        self.num_domains = num_domains
        self.use_domain_adaptation = use_domain_adaptation
        self.use_preference_disentanglement = use_preference_disentanglement

        # Core user embedding (shared across all domains)
        self.user_embedding = nn.Embedding(num_users, embedding_dim)

        # Domain-specific adaptations
        if use_domain_adaptation:
            self.domain_adapters = nn.ModuleList([
                DomainAdapter(embedding_dim, dropout)
                for _ in range(num_domains)
            ])

        # Preference disentanglement: separate general vs domain-specific preferences
        if use_preference_disentanglement:
            # Shared preferences (e.g., likes action, dislikes slow pacing)
            self.shared_preference_encoder = nn.Sequential(
                nn.Linear(embedding_dim, embedding_dim),
                nn.LayerNorm(embedding_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            )

            # Domain-specific preferences (e.g., prefers episodic TV, marathon movies)
            self.domain_specific_encoders = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(embedding_dim, embedding_dim // 2),
                    nn.LayerNorm(embedding_dim // 2),
                    nn.GELU(),
                    nn.Dropout(dropout)
                ) for _ in range(num_domains)
            ])

            # Fusion layer
            self.preference_fusion = nn.Sequential(
                nn.Linear(embedding_dim + embedding_dim // 2, embedding_dim),
                nn.LayerNorm(embedding_dim),
                nn.GELU()
            )

        # Cross-domain attention (learn which domain informs the other)
        self.cross_domain_attention = nn.MultiheadAttention(
            embedding_dim, num_heads=8, dropout=dropout, batch_first=True
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize weights"""
        nn.init.xavier_normal_(self.user_embedding.weight)

        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        user_ids: torch.Tensor,
        domain: int,
        cross_domain_context: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Get user embeddings for a specific domain.

        Args:
            user_ids: User IDs [batch_size]
            domain: Domain index (0=movies, 1=tv)
            cross_domain_context: Optional context from other domain

        Returns:
            Dictionary with embeddings and attention weights
        """
        # Get base user embeddings
        user_emb = self.user_embedding(user_ids)  # [batch_size, embedding_dim]

        outputs = {'base_embedding': user_emb}

        # Disentangle preferences if enabled
        if self.use_preference_disentanglement:
            # Shared preferences (work across domains)
            shared_pref = self.shared_preference_encoder(user_emb)

            # Domain-specific preferences
            domain_pref = self.domain_specific_encoders[domain](user_emb)

            # Combine shared and domain-specific
            combined = torch.cat([shared_pref, domain_pref], dim=-1)
            user_emb = self.preference_fusion(combined)

            outputs['shared_preferences'] = shared_pref
            outputs['domain_preferences'] = domain_pref

        # Apply domain adaptation if enabled
        if self.use_domain_adaptation:
            user_emb = self.domain_adapters[domain](user_emb)

        # Cross-domain attention if context is provided
        if cross_domain_context is not None:
            user_emb_seq = user_emb.unsqueeze(1)  # [batch, 1, dim]
            context_seq = cross_domain_context.unsqueeze(1)  # [batch, 1, dim]

            attended, attn_weights = self.cross_domain_attention(
                user_emb_seq, context_seq, context_seq
            )

            # Residual connection
            user_emb = user_emb + attended.squeeze(1)
            outputs['cross_domain_attention'] = attn_weights

        outputs['final_embedding'] = user_emb

        return outputs

    def get_cross_domain_transfer(
        self,
        user_ids: torch.Tensor,
        source_domain: int,
        target_domain: int
    ) -> torch.Tensor:
        """
        Transfer user preferences from source domain to target domain.
        Useful for cold-start in one domain.

        Args:
            user_ids: User IDs
            source_domain: Domain with existing interactions
            target_domain: Domain with no/few interactions

        Returns:
            Transferred embeddings for target domain
        """
        # Get embeddings from source domain
        source_outputs = self.forward(user_ids, source_domain)

        # Use shared preferences for transfer
        if self.use_preference_disentanglement:
            shared_pref = source_outputs['shared_preferences']
            # Apply target domain-specific transformation
            domain_pref = self.domain_specific_encoders[target_domain](
                source_outputs['base_embedding']
            )
            combined = torch.cat([shared_pref, domain_pref], dim=-1)
            transferred = self.preference_fusion(combined)
        else:
            # Direct transfer with domain adaptation
            transferred = source_outputs['base_embedding']
            if self.use_domain_adaptation:
                transferred = self.domain_adapters[target_domain](transferred)

        return transferred


class DomainAdapter(nn.Module):
    """Adapts user embeddings to specific domains (movies vs TV)"""

    def __init__(self, embedding_dim: int, dropout: float = 0.1):
        super(DomainAdapter, self).__init__()

        # Adapter layers (keep it lightweight for efficiency)
        self.adapter = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim // 4, embedding_dim),
            nn.LayerNorm(embedding_dim)
        )

        # Gating mechanism to control adaptation strength
        self.gate = nn.Sequential(
            nn.Linear(embedding_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply domain adaptation with gating"""
        adapted = self.adapter(x)
        gate_weight = self.gate(x)

        # Weighted combination of original and adapted
        return x + gate_weight * adapted


class UnifiedItemEmbedding(nn.Module):
    """
    Unified item (movie/TV) embedding with multimodal features.

    Handles:
    - Different item types (movies have runtime, TV has seasons/episodes)
    - Shared features (genres, cast, plot)
    - Multimodal content (text, images, metadata)
    """

    def __init__(
        self,
        num_movies: int,
        num_tv_shows: int,
        embedding_dim: int = 512,
        num_genres: int = 20,
        num_cast: int = 10000,
        use_multimodal: bool = True,
        dropout: float = 0.1
    ):
        super(UnifiedItemEmbedding, self).__init__()

        self.embedding_dim = embedding_dim
        self.use_multimodal = use_multimodal

        # Separate embeddings for movies and TV (different ID spaces)
        self.movie_embedding = nn.Embedding(num_movies, embedding_dim)
        self.tv_embedding = nn.Embedding(num_tv_shows, embedding_dim)

        # Shared feature embeddings
        self.genre_embedding = nn.Embedding(num_genres, embedding_dim // 4)
        self.cast_embedding = nn.Embedding(num_cast, embedding_dim // 4)

        # Content-based encoders (shared across domains)
        if use_multimodal:
            # Text encoder for plot summaries
            self.text_encoder = nn.Sequential(
                nn.Linear(768, embedding_dim),  # Assume BERT embeddings
                nn.LayerNorm(embedding_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            )

            # Visual encoder for posters/thumbnails
            self.visual_encoder = nn.Sequential(
                nn.Linear(2048, embedding_dim),  # Assume ResNet features
                nn.LayerNorm(embedding_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            )

        # Metadata encoders
        self.metadata_encoder = nn.Sequential(
            nn.Linear(10, embedding_dim // 4),  # year, rating, popularity, etc.
            nn.LayerNorm(embedding_dim // 4),
            nn.GELU()
        )

        # Type-specific encoders
        self.movie_metadata_encoder = nn.Sequential(
            nn.Linear(3, embedding_dim // 8),  # runtime, budget, box_office
            nn.GELU()
        )

        self.tv_metadata_encoder = nn.Sequential(
            nn.Linear(3, embedding_dim // 8),  # num_seasons, num_episodes, status
            nn.GELU()
        )

        # Feature fusion
        self.feature_fusion = MultimodalFeatureFusion(
            embedding_dim, dropout
        )

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Embedding):
                nn.init.xavier_normal_(module.weight)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        item_ids: torch.Tensor,
        item_type: str,  # 'movie' or 'tv'
        genres: Optional[torch.Tensor] = None,
        cast: Optional[torch.Tensor] = None,
        text_features: Optional[torch.Tensor] = None,
        visual_features: Optional[torch.Tensor] = None,
        metadata: Optional[torch.Tensor] = None,
        type_specific_metadata: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Get item embeddings with multimodal features.

        Args:
            item_ids: Item IDs
            item_type: 'movie' or 'tv'
            genres: Genre IDs [batch, max_genres]
            cast: Cast member IDs [batch, max_cast]
            text_features: Pre-computed text embeddings [batch, 768]
            visual_features: Pre-computed visual features [batch, 2048]
            metadata: General metadata [batch, 10]
            type_specific_metadata: Movie or TV specific metadata [batch, 3]
        """
        features = []

        # Base collaborative embedding
        if item_type == 'movie':
            base_emb = self.movie_embedding(item_ids)
        else:  # tv
            base_emb = self.tv_embedding(item_ids)

        features.append(base_emb)

        # Genre features
        if genres is not None:
            genre_emb = self.genre_embedding(genres).mean(dim=1)  # Average over genres
            features.append(genre_emb)

        # Cast features
        if cast is not None:
            cast_emb = self.cast_embedding(cast).mean(dim=1)  # Average over cast
            features.append(cast_emb)

        # Multimodal features
        if self.use_multimodal:
            if text_features is not None:
                text_emb = self.text_encoder(text_features)
                features.append(text_emb)

            if visual_features is not None:
                visual_emb = self.visual_encoder(visual_features)
                features.append(visual_emb)

        # Metadata features
        if metadata is not None:
            meta_emb = self.metadata_encoder(metadata)
            features.append(meta_emb)

        # Type-specific metadata
        if type_specific_metadata is not None:
            if item_type == 'movie':
                type_emb = self.movie_metadata_encoder(type_specific_metadata)
            else:
                type_emb = self.tv_metadata_encoder(type_specific_metadata)
            features.append(type_emb)

        # Fuse all features
        fused_embedding = self.feature_fusion(features)

        return {
            'embedding': fused_embedding,
            'base_embedding': base_emb,
            'num_features': len(features)
        }


class MultimodalFeatureFusion(nn.Module):
    """Intelligently fuses features of different dimensions"""

    def __init__(self, target_dim: int, dropout: float = 0.1):
        super(MultimodalFeatureFusion, self).__init__()

        self.target_dim = target_dim

        # Attention-based fusion
        self.feature_attention = nn.Sequential(
            nn.Linear(target_dim, target_dim // 4),
            nn.Tanh(),
            nn.Linear(target_dim // 4, 1)
        )

        # Projection layers (created dynamically)
        self.projections = nn.ModuleDict()

        # Final fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(target_dim, target_dim),
            nn.LayerNorm(target_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        """
        Fuse features of different dimensions.

        Args:
            features: List of feature tensors (can have different dimensions)
        """
        batch_size = features[0].size(0)

        # Project all features to target dimension and pad if needed
        projected_features = []
        for i, feat in enumerate(features):
            feat_dim = feat.size(-1)

            # Project or pad to target dimension
            if feat_dim < self.target_dim:
                # Pad with zeros
                padding = torch.zeros(
                    batch_size, self.target_dim - feat_dim,
                    device=feat.device, dtype=feat.dtype
                )
                feat = torch.cat([feat, padding], dim=-1)
            elif feat_dim > self.target_dim:
                # Project down
                proj_key = f'proj_{feat_dim}_to_{self.target_dim}'
                if proj_key not in self.projections:
                    self.projections[proj_key] = nn.Linear(
                        feat_dim, self.target_dim
                    ).to(feat.device)
                feat = self.projections[proj_key](feat)

            projected_features.append(feat)

        # Stack features
        stacked = torch.stack(projected_features, dim=1)  # [batch, num_features, dim]

        # Compute attention weights
        attn_scores = self.feature_attention(stacked)  # [batch, num_features, 1]
        attn_weights = F.softmax(attn_scores, dim=1)

        # Weighted sum
        fused = (stacked * attn_weights).sum(dim=1)  # [batch, dim]

        # Final transformation
        fused = self.fusion(fused)

        return fused


class CrossDomainRecommender(nn.Module):
    """
    Complete cross-domain recommendation system.

    Uses unified embeddings for both movies and TV shows.
    """

    def __init__(
        self,
        num_users: int,
        num_movies: int,
        num_tv_shows: int,
        embedding_dim: int = 512,
        num_genres: int = 20,
        num_cast: int = 10000,
        dropout: float = 0.1
    ):
        super(CrossDomainRecommender, self).__init__()

        self.embedding_dim = embedding_dim

        # Unified user embeddings
        self.user_encoder = UnifiedUserEmbedding(
            num_users=num_users,
            embedding_dim=embedding_dim,
            num_domains=2,  # movies, tv
            dropout=dropout
        )

        # Unified item embeddings
        self.item_encoder = UnifiedItemEmbedding(
            num_movies=num_movies,
            num_tv_shows=num_tv_shows,
            embedding_dim=embedding_dim,
            num_genres=num_genres,
            num_cast=num_cast,
            dropout=dropout
        )

        # Interaction layers
        self.interaction = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim, embedding_dim // 2),
            nn.GELU(),
            nn.Linear(embedding_dim // 2, 1)
        )

        # Temperature for similarity scaling
        self.temperature = nn.Parameter(torch.ones(1) * 0.07)

    def forward(
        self,
        user_ids: torch.Tensor,
        item_ids: torch.Tensor,
        domain: str,  # 'movie' or 'tv'
        item_features: Optional[Dict] = None,
        cross_domain_context: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Predict user-item scores.

        Args:
            user_ids: User IDs [batch_size]
            item_ids: Item IDs [batch_size]
            domain: 'movie' or 'tv'
            item_features: Optional multimodal features
            cross_domain_context: Optional context from other domain
        """
        # Domain index (0=movie, 1=tv)
        domain_idx = 0 if domain == 'movie' else 1

        # Get user embeddings
        user_outputs = self.user_encoder(
            user_ids, domain_idx, cross_domain_context
        )
        user_emb = user_outputs['final_embedding']

        # Get item embeddings
        if item_features is None:
            item_features = {}

        item_outputs = self.item_encoder(
            item_ids, domain, **item_features
        )
        item_emb = item_outputs['embedding']

        # Normalize embeddings
        user_emb = F.normalize(user_emb, p=2, dim=-1)
        item_emb = F.normalize(item_emb, p=2, dim=-1)

        # Compute similarity
        similarity = torch.sum(user_emb * item_emb, dim=-1) / self.temperature

        return similarity

    def recommend(
        self,
        user_id: int,
        domain: str,
        candidate_items: torch.Tensor,
        item_features: Optional[Dict] = None,
        top_k: int = 10,
        use_cross_domain: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get top-k recommendations for a user.

        Args:
            user_id: User ID
            domain: 'movie' or 'tv'
            candidate_items: Candidate item IDs to rank
            item_features: Optional item features
            top_k: Number of recommendations
            use_cross_domain: Whether to use cross-domain context
        """
        self.eval()
        with torch.no_grad():
            batch_size = len(candidate_items)
            user_ids = torch.tensor([user_id] * batch_size)

            # Get cross-domain context if enabled
            cross_domain_context = None
            if use_cross_domain:
                other_domain = 'tv' if domain == 'movie' else 'movie'
                other_domain_idx = 1 if domain == 'movie' else 0
                # Use history from other domain as context
                user_outputs = self.user_encoder(
                    torch.tensor([user_id]), other_domain_idx
                )
                cross_domain_context = user_outputs['final_embedding']

            # Get scores
            scores = self.forward(
                user_ids, candidate_items, domain,
                item_features, cross_domain_context
            )

            # Get top-k
            top_scores, top_indices = torch.topk(scores, min(top_k, len(scores)))
            top_items = candidate_items[top_indices]

            return top_items, top_scores


# Example usage
if __name__ == "__main__":
    # Initialize
    model = CrossDomainRecommender(
        num_users=10000,
        num_movies=5000,
        num_tv_shows=3000,
        embedding_dim=512
    )

    # Example: Recommend movies
    user_id = 42
    candidate_movies = torch.randint(0, 5000, (100,))

    top_movies, scores = model.recommend(
        user_id=user_id,
        domain='movie',
        candidate_items=candidate_movies,
        top_k=10,
        use_cross_domain=True  # Use TV watching history to improve movie recs
    )

    print(f"Top 10 movie recommendations for user {user_id}:")
    print(f"Movie IDs: {top_movies}")
    print(f"Scores: {scores}")
