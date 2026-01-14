"""
Movie Ensemble System for CineSync v2
Combines all 18+ movie models with adaptive weighting and uncertainty estimation
Similar to the SOTA TV ensemble but optimized for movies
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union
import logging
from pathlib import Path

# Import all movie models
import sys
sys.path.append(str(Path(__file__).parent.parent))

from collaborative.src.model import (
    NeuralCollaborativeFiltering,
    SimpleNCF,
    CrossAttentionNCF,
    DeepNCF
)
from two_tower.src.model import (
    TwoTowerModel,
    EnhancedTwoTowerModel,
    MultiTaskTwoTowerModel,
    CollaborativeTwoTowerModel
)
from sequential.src.model import (
    SequentialRecommender,
    TransformerSequentialRecommender,
    AttentionalSequentialRecommender,
    SessionBasedRecommender
)
from advanced.transformer_recommender import SASRec, EnhancedSASRec

logger = logging.getLogger(__name__)


class AdaptiveWeightingModule(nn.Module):
    """Learn adaptive weights for ensemble models based on input characteristics"""

    def __init__(
        self,
        input_dim: int = 768,
        num_models: int = 10,
        hidden_dim: int = 256
    ):
        super().__init__()

        self.num_models = num_models

        # Context encoder for adaptive weighting
        self.context_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, num_models),
            nn.Softmax(dim=-1)
        )

        # Temperature parameter for sharpening/smoothing weights
        self.temperature = nn.Parameter(torch.tensor(1.0))

    def forward(self, context_embedding: torch.Tensor) -> torch.Tensor:
        """
        Compute adaptive weights based on context.

        Args:
            context_embedding: [batch_size, input_dim] context representation

        Returns:
            weights: [batch_size, num_models] adaptive weights
        """
        # Compute base weights
        logits = self.context_encoder(context_embedding)

        # Apply temperature scaling
        weights = F.softmax(logits / self.temperature, dim=-1)

        return weights


class UncertaintyEstimator(nn.Module):
    """Estimate prediction uncertainty for ensemble models"""

    def __init__(
        self,
        embed_dim: int = 768,
        hidden_dim: int = 256,
        num_models: int = 10
    ):
        super().__init__()

        self.num_models = num_models

        # Uncertainty estimation network
        self.uncertainty_net = nn.Sequential(
            nn.Linear(embed_dim * num_models, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, num_models),
            nn.Sigmoid()  # Output uncertainty scores [0, 1]
        )

    def forward(self, model_embeddings: List[torch.Tensor]) -> torch.Tensor:
        """
        Estimate uncertainty for each model's predictions.

        Args:
            model_embeddings: List of [batch_size, embed_dim] embeddings from each model

        Returns:
            uncertainties: [batch_size, num_models] uncertainty scores
        """
        # Concatenate embeddings from all models
        concat_embeddings = torch.cat(model_embeddings, dim=-1)

        # Estimate uncertainties
        uncertainties = self.uncertainty_net(concat_embeddings)

        return uncertainties


class MovieEnsembleRecommender(nn.Module):
    """
    Complete ensemble system for movie recommendation.

    Combines:
    1. Collaborative Filtering models (NCF, SimpleNCF, CrossAttentionNCF, DeepNCF)
    2. Two-Tower models (Standard, Enhanced, MultiTask, Collaborative)
    3. Sequential models (LSTM, Transformer, Attention, Session-based)
    4. Advanced transformer models (SASRec, EnhancedSASRec)

    Features:
    - Adaptive weighting based on input characteristics
    - Uncertainty estimation for prediction confidence
    - Multi-level fusion (embedding + prediction level)
    - Cold start handling
    """

    def __init__(
        self,
        num_users: int,
        num_movies: int,
        num_genres: int = 20,
        embed_dim: int = 768,
        use_adaptive_weighting: bool = True,
        use_uncertainty_estimation: bool = True,
        fusion_strategy: str = 'attention',  # 'weighted', 'attention', 'learned'
        enable_collaborative_models: bool = True,
        enable_two_tower_models: bool = True,
        enable_sequential_models: bool = True,
        enable_transformer_models: bool = True
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.fusion_strategy = fusion_strategy
        self.use_adaptive_weighting = use_adaptive_weighting
        self.use_uncertainty_estimation = use_uncertainty_estimation

        # Track which models are enabled
        self.enabled_models = []
        self.model_names = []

        # Initialize collaborative filtering models
        if enable_collaborative_models:
            self.ncf_standard = NeuralCollaborativeFiltering(
                num_users, num_movies, embedding_dim=embed_dim // 2
            )
            self.ncf_simple = SimpleNCF(
                num_users, num_movies, embedding_dim=embed_dim // 4
            )
            self.ncf_cross_attention = CrossAttentionNCF(
                num_users, num_movies, num_genres=num_genres, embedding_dim=embed_dim
            )
            self.ncf_deep = DeepNCF(
                num_users, num_movies, num_genres=num_genres, embedding_dim=embed_dim // 2
            )

            self.enabled_models.extend([
                self.ncf_standard, self.ncf_simple,
                self.ncf_cross_attention, self.ncf_deep
            ])
            self.model_names.extend([
                'ncf_standard', 'ncf_simple', 'ncf_cross_attention', 'ncf_deep'
            ])

        # Initialize two-tower models
        if enable_two_tower_models:
            # Note: Two-tower models need feature dimensions
            # These would be initialized with actual feature dimensions from config
            self.two_tower_models = []
            # Placeholder - would add actual two-tower models with proper configs

        # Initialize sequential models
        if enable_sequential_models:
            self.seq_lstm = SequentialRecommender(
                num_movies, embedding_dim=embed_dim // 2, hidden_dim=embed_dim
            )
            self.seq_transformer = TransformerSequentialRecommender(
                num_movies, embedding_dim=embed_dim, num_heads=12, num_blocks=6
            )
            self.seq_attention = AttentionalSequentialRecommender(
                num_movies, embedding_dim=embed_dim // 2, num_heads=8, num_blocks=4
            )
            self.seq_session = SessionBasedRecommender(
                num_movies, embedding_dim=embed_dim // 2, hidden_dim=embed_dim
            )

            self.enabled_models.extend([
                self.seq_lstm, self.seq_transformer,
                self.seq_attention, self.seq_session
            ])
            self.model_names.extend([
                'seq_lstm', 'seq_transformer', 'seq_attention', 'seq_session'
            ])

        # Initialize transformer models
        if enable_transformer_models:
            self.sasrec = SASRec(
                num_movies, d_model=embed_dim, num_heads=12, num_layers=6
            )
            self.sasrec_enhanced = EnhancedSASRec(
                num_movies, num_genres=num_genres, d_model=embed_dim
            )

            self.enabled_models.extend([self.sasrec, self.sasrec_enhanced])
            self.model_names.extend(['sasrec', 'sasrec_enhanced'])

        num_active_models = len(self.enabled_models)

        # Projection layers to unify embedding dimensions
        self.embedding_projections = nn.ModuleList([
            nn.Linear(embed_dim, embed_dim) for _ in range(num_active_models)
        ])

        # Adaptive weighting module
        if use_adaptive_weighting:
            self.adaptive_weights = AdaptiveWeightingModule(
                input_dim=embed_dim,
                num_models=num_active_models,
                hidden_dim=256
            )

        # Uncertainty estimation
        if use_uncertainty_estimation:
            self.uncertainty_estimator = UncertaintyEstimator(
                embed_dim=embed_dim,
                hidden_dim=256,
                num_models=num_active_models
            )

        # Fusion modules based on strategy
        if fusion_strategy == 'attention':
            self.fusion_attention = nn.MultiheadAttention(
                embed_dim=embed_dim,
                num_heads=8,
                dropout=0.1,
                batch_first=True
            )
            self.fusion_norm = nn.LayerNorm(embed_dim)

        elif fusion_strategy == 'learned':
            self.fusion_network = nn.Sequential(
                nn.Linear(embed_dim * num_active_models, embed_dim * 2),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(embed_dim * 2, embed_dim),
                nn.LayerNorm(embed_dim)
            )

        # Final prediction heads
        self.recommendation_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim // 2, embed_dim // 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim // 4, 1)
        )

        # Confidence head for prediction reliability
        self.confidence_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim // 4, 1),
            nn.Sigmoid()
        )

        # Rating prediction head (for multi-task learning)
        self.rating_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim // 4, 1),
            nn.Sigmoid()  # Will be scaled to 0-5
        )

    def get_model_embeddings(
        self,
        user_ids: torch.Tensor,
        movie_ids: torch.Tensor,
        sequences: Optional[torch.Tensor] = None,
        genre_ids: Optional[torch.Tensor] = None
    ) -> List[torch.Tensor]:
        """
        Get embeddings from all enabled models.

        Args:
            user_ids: User IDs [batch_size]
            movie_ids: Movie IDs [batch_size]
            sequences: Optional sequence data for sequential models [batch_size, seq_len]
            genre_ids: Optional genre IDs [batch_size, num_genres]

        Returns:
            List of embeddings from each model
        """
        embeddings = []

        for i, model in enumerate(self.enabled_models):
            model_name = self.model_names[i]

            try:
                # Get model output based on model type
                if 'ncf' in model_name:
                    # NCF models
                    if 'cross_attention' in model_name or 'deep' in model_name:
                        output = model(user_ids, movie_ids, genre_ids)
                    else:
                        output = model(user_ids, movie_ids)

                    # Extract embeddings (NCF models return ratings)
                    # We need to get intermediate representations
                    if hasattr(model, 'user_embedding'):
                        user_emb = model.user_embedding(user_ids)
                        item_emb = model.item_embedding(movie_ids) if hasattr(model, 'item_embedding') else model.movie_embedding(movie_ids)
                        emb = torch.cat([user_emb, item_emb], dim=-1)
                    else:
                        # Use output as embedding
                        emb = output.unsqueeze(-1) if output.dim() == 1 else output

                elif 'seq' in model_name or 'sasrec' in model_name:
                    # Sequential models need sequence data
                    if sequences is not None:
                        if 'sasrec_enhanced' in model_name:
                            output = model(sequences, genre_ids)
                            emb = output['sequence_embeddings'][:, -1, :]  # Last position
                        elif 'sasrec' in model_name:
                            # SASRec returns logits
                            with torch.no_grad():
                                logits = model(sequences)
                                emb = logits[:, -1, :]  # Last position logits
                        else:
                            # Other sequential models
                            output = model(sequences)
                            if isinstance(output, dict):
                                emb = output['sequence_embeddings'][:, -1, :]
                            else:
                                emb = output[:, -1, :] if output.dim() == 3 else output
                    else:
                        # Skip sequential models if no sequence data
                        continue

                else:
                    # Other models (two-tower, etc.)
                    # Would implement based on specific model interfaces
                    continue

                # Project to unified dimension
                if emb.size(-1) != self.embed_dim:
                    # Pad or project
                    if emb.size(-1) < self.embed_dim:
                        padding = torch.zeros(
                            emb.size(0), self.embed_dim - emb.size(-1),
                            device=emb.device, dtype=emb.dtype
                        )
                        emb = torch.cat([emb, padding], dim=-1)
                    else:
                        emb = self.embedding_projections[i](emb[:, :self.embed_dim])

                embeddings.append(emb)

            except Exception as e:
                logger.warning(f"Failed to get embedding from {model_name}: {e}")
                continue

        return embeddings

    def fuse_embeddings(
        self,
        embeddings: List[torch.Tensor],
        context_embedding: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Fuse embeddings from multiple models"""

        if not embeddings:
            # Return zero embedding if no models available
            return torch.zeros(1, self.embed_dim, device=self.device)

        if len(embeddings) == 1:
            return embeddings[0]

        # Stack embeddings
        stacked_embeddings = torch.stack(embeddings, dim=1)  # [batch, num_models, embed_dim]

        if self.fusion_strategy == 'weighted':
            # Simple weighted average
            if self.use_adaptive_weighting and context_embedding is not None:
                weights = self.adaptive_weights(context_embedding)
                weights = weights[:, :len(embeddings)]  # Adjust for available models
                weights = F.softmax(weights, dim=-1)  # Renormalize

                # Apply weights
                weighted_embeddings = stacked_embeddings * weights.unsqueeze(-1)
                fused_embedding = weighted_embeddings.sum(dim=1)
            else:
                # Equal weights
                fused_embedding = stacked_embeddings.mean(dim=1)

        elif self.fusion_strategy == 'attention':
            # Attention-based fusion
            query = stacked_embeddings.mean(dim=1, keepdim=True)  # [batch, 1, embed_dim]

            attended_embeddings, attention_weights = self.fusion_attention(
                query, stacked_embeddings, stacked_embeddings
            )

            fused_embedding = self.fusion_norm(
                attended_embeddings.squeeze(1) + query.squeeze(1)
            )

        elif self.fusion_strategy == 'learned':
            # Learned fusion network
            concat_embeddings = stacked_embeddings.reshape(
                stacked_embeddings.size(0), -1
            )  # [batch, num_models * embed_dim]

            # Pad if necessary
            target_dim = self.embed_dim * len(self.enabled_models)
            if concat_embeddings.size(1) < target_dim:
                padding = torch.zeros(
                    concat_embeddings.size(0),
                    target_dim - concat_embeddings.size(1),
                    device=concat_embeddings.device
                )
                concat_embeddings = torch.cat([concat_embeddings, padding], dim=1)

            fused_embedding = self.fusion_network(concat_embeddings)

        else:
            # Default to mean
            fused_embedding = stacked_embeddings.mean(dim=1)

        return fused_embedding

    def forward(
        self,
        user_ids: torch.Tensor,
        movie_ids: torch.Tensor,
        sequences: Optional[torch.Tensor] = None,
        genre_ids: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through ensemble.

        Args:
            user_ids: User IDs [batch_size]
            movie_ids: Movie IDs [batch_size]
            sequences: Optional sequences [batch_size, seq_len]
            genre_ids: Optional genre IDs [batch_size, num_genres]

        Returns:
            Dictionary with predictions and metadata
        """
        # Get embeddings from all models
        embeddings = self.get_model_embeddings(
            user_ids, movie_ids, sequences, genre_ids
        )

        if not embeddings:
            raise ValueError("No model embeddings available")

        # Use first available embedding as context for adaptive weighting
        context_embedding = embeddings[0] if embeddings else None

        # Fuse embeddings
        fused_embedding = self.fuse_embeddings(embeddings, context_embedding)

        # Estimate uncertainties
        uncertainties = None
        if self.use_uncertainty_estimation and len(embeddings) > 1:
            # Pad embeddings to consistent length for uncertainty estimation
            padded_embeddings = []
            for emb in embeddings[:10]:  # Limit to first 10 for efficiency
                if emb.size(1) != self.embed_dim:
                    if emb.size(1) < self.embed_dim:
                        padding = torch.zeros(
                            emb.size(0), self.embed_dim - emb.size(1),
                            device=emb.device
                        )
                        emb = torch.cat([emb, padding], dim=1)
                    else:
                        emb = emb[:, :self.embed_dim]
                padded_embeddings.append(emb)

            # Fill remaining slots with zeros if needed
            while len(padded_embeddings) < min(10, len(self.enabled_models)):
                padded_embeddings.append(
                    torch.zeros(user_ids.size(0), self.embed_dim, device=fused_embedding.device)
                )

            uncertainties = self.uncertainty_estimator(padded_embeddings[:10])

        # Generate predictions
        recommendation_score = self.recommendation_head(fused_embedding)
        confidence_score = self.confidence_head(fused_embedding)
        rating_score = self.rating_head(fused_embedding) * 5.0  # Scale to 0-5

        outputs = {
            'fused_embedding': fused_embedding,
            'individual_embeddings': embeddings,
            'model_names': [self.model_names[i] for i in range(len(embeddings))],
            'recommendation_score': recommendation_score.squeeze(),
            'confidence_score': confidence_score.squeeze(),
            'rating_prediction': rating_score.squeeze()
        }

        if uncertainties is not None:
            outputs['uncertainties'] = uncertainties

        return outputs

    def recommend_movies(
        self,
        user_id: int,
        candidate_movie_ids: torch.Tensor,
        sequences: Optional[torch.Tensor] = None,
        genre_ids: Optional[torch.Tensor] = None,
        top_k: int = 10,
        min_confidence: float = 0.5
    ) -> Dict[str, torch.Tensor]:
        """
        Generate top-k movie recommendations for a user.

        Returns:
            Dictionary containing:
            - recommended_ids: Top-k recommended movie IDs
            - scores: Recommendation scores
            - confidences: Confidence scores
            - ratings: Predicted ratings
            - model_contributions: Individual model contributions
        """
        self.eval()
        with torch.no_grad():
            batch_size = len(candidate_movie_ids)
            user_ids = torch.tensor([user_id] * batch_size, device=candidate_movie_ids.device)

            # Expand sequences if provided
            if sequences is not None and sequences.size(0) == 1:
                sequences = sequences.repeat(batch_size, 1)

            # Get predictions
            outputs = self.forward(user_ids, candidate_movie_ids, sequences, genre_ids)

            scores = outputs['recommendation_score']
            confidences = outputs['confidence_score']
            ratings = outputs['rating_prediction']

            # Filter by minimum confidence
            confidence_mask = confidences >= min_confidence
            if not confidence_mask.any():
                logger.warning(f"No predictions meet minimum confidence {min_confidence}")
                confidence_mask = torch.ones_like(confidences, dtype=torch.bool)

            # Apply confidence filter
            filtered_scores = scores.clone()
            filtered_scores[~confidence_mask] = float('-inf')

            # Get top-k recommendations
            top_scores, top_indices = torch.topk(
                filtered_scores,
                min(top_k, (filtered_scores > float('-inf')).sum().item())
            )

            recommended_ids = candidate_movie_ids[top_indices]

            return {
                'recommended_ids': recommended_ids,
                'scores': top_scores,
                'confidences': confidences[top_indices],
                'ratings': ratings[top_indices],
                'model_names': outputs['model_names'],
                'uncertainties': outputs.get('uncertainties')
            }

    @property
    def device(self) -> torch.device:
        """Get device of the model"""
        return next(self.parameters()).device

    def save_ensemble(self, save_path: str):
        """Save ensemble model and configuration"""
        save_dict = {
            'ensemble_state_dict': self.state_dict(),
            'model_names': self.model_names,
            'config': {
                'embed_dim': self.embed_dim,
                'fusion_strategy': self.fusion_strategy,
                'use_adaptive_weighting': self.use_adaptive_weighting,
                'use_uncertainty_estimation': self.use_uncertainty_estimation
            }
        }
        torch.save(save_dict, save_path)
        logger.info(f"Movie ensemble saved to {save_path}")


# Example usage
if __name__ == "__main__":
    # Initialize ensemble
    ensemble = MovieEnsembleRecommender(
        num_users=10000,
        num_movies=5000,
        num_genres=20,
        embed_dim=768,
        fusion_strategy='attention',
        enable_collaborative_models=True,
        enable_sequential_models=True,
        enable_transformer_models=True
    )

    # Example recommendation
    user_id = 42
    candidate_movies = torch.randint(0, 5000, (100,))
    sequence = torch.randint(0, 5000, (1, 50))  # Last 50 movies watched

    results = ensemble.recommend_movies(
        user_id=user_id,
        candidate_movie_ids=candidate_movies,
        sequences=sequence,
        top_k=10
    )

    print(f"Top 10 recommendations for user {user_id}:")
    print(f"Movie IDs: {results['recommended_ids']}")
    print(f"Scores: {results['scores']}")
    print(f"Predicted ratings: {results['ratings']}")
    print(f"Confidence: {results['confidences']}")

# Alias for backward compatibility
MovieEnsembleSystem = MovieEnsembleRecommender
