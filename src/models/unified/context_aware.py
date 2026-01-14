"""
Context-Aware Recommendation Layer for CineSync v2
Adapts recommendations based on temporal, device, social, and mood context
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import numpy as np


class TemporalContextEncoder(nn.Module):
    """
    Encode temporal context (time of day, day of week, season, etc.).

    Different times lead to different preferences:
    - Weeknight evening: shorter content, relaxing
    - Weekend afternoon: family-friendly, longer movies
    - Late night: thrillers, intense content
    """

    def __init__(self, output_dim: int = 128):
        super().__init__()

        self.output_dim = output_dim

        # Embeddings for different temporal features
        self.hour_embedding = nn.Embedding(24, 32)  # Hour of day (0-23)
        self.day_of_week_embedding = nn.Embedding(7, 16)  # Monday-Sunday
        self.month_embedding = nn.Embedding(12, 16)  # Month (1-12)

        # Cyclical encoding for continuous time features
        # sin/cos encoding for hour, day, etc. to capture cyclical nature
        self.cyclical_encoder = nn.Sequential(
            nn.Linear(6, 32),  # 3 features * 2 (sin + cos)
            nn.GELU(),
            nn.Linear(32, 32)
        )

        # Fusion layer
        total_dim = 32 + 16 + 16 + 32  # hour + day + month + cyclical
        self.fusion = nn.Sequential(
            nn.Linear(total_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
            nn.Dropout(0.1)
        )

    def forward(
        self,
        hour: Optional[torch.Tensor] = None,
        day_of_week: Optional[torch.Tensor] = None,
        month: Optional[torch.Tensor] = None,
        timestamp: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Encode temporal context.

        Args:
            hour: Hour of day (0-23) [batch_size]
            day_of_week: Day (0-6) [batch_size]
            month: Month (1-12) [batch_size]
            timestamp: Unix timestamp [batch_size] (alternative to individual features)

        Returns:
            Temporal context embedding [batch_size, output_dim]
        """
        features = []

        # If timestamp provided, extract temporal features
        if timestamp is not None:
            # Convert to datetime
            batch_size = timestamp.size(0)
            hour = torch.zeros(batch_size, dtype=torch.long)
            day_of_week = torch.zeros(batch_size, dtype=torch.long)
            month = torch.zeros(batch_size, dtype=torch.long)

            for i, ts in enumerate(timestamp):
                dt = datetime.fromtimestamp(ts.item())
                hour[i] = dt.hour
                day_of_week[i] = dt.weekday()
                month[i] = dt.month

        # Embeddings
        if hour is not None:
            hour_emb = self.hour_embedding(hour)
            features.append(hour_emb)

            # Cyclical encoding of hour (sin/cos to capture 23→0 continuity)
            hour_norm = hour.float() / 24.0
            hour_sin = torch.sin(2 * np.pi * hour_norm)
            hour_cos = torch.cos(2 * np.pi * hour_norm)

        if day_of_week is not None:
            day_emb = self.day_of_week_embedding(day_of_week)
            features.append(day_emb)

            # Cyclical encoding of day
            day_norm = day_of_week.float() / 7.0
            day_sin = torch.sin(2 * np.pi * day_norm)
            day_cos = torch.cos(2 * np.pi * day_norm)

        if month is not None:
            month_emb = self.month_embedding(month - 1)  # 0-indexed
            features.append(month_emb)

            # Cyclical encoding of month
            month_norm = month.float() / 12.0
            month_sin = torch.sin(2 * np.pi * month_norm)
            month_cos = torch.cos(2 * np.pi * month_norm)

        # Cyclical features
        if hour is not None and day_of_week is not None and month is not None:
            cyclical_features = torch.stack([
                hour_sin, hour_cos, day_sin, day_cos, month_sin, month_cos
            ], dim=-1)
            cyclical_emb = self.cyclical_encoder(cyclical_features)
            features.append(cyclical_emb)

        # Fuse all features
        if features:
            combined = torch.cat(features, dim=-1)
            output = self.fusion(combined)
        else:
            # Return zero embedding if no features
            output = torch.zeros(1, self.output_dim)

        return output


class DeviceContextEncoder(nn.Module):
    """
    Encode device context (mobile, TV, browser, etc.).

    Different devices suggest different preferences:
    - Mobile: shorter clips, commute content
    - TV: longer movies, shared viewing
    - Browser: varied content, exploratory
    """

    def __init__(
        self,
        num_device_types: int = 5,  # mobile, tablet, tv, desktop, browser
        output_dim: int = 64
    ):
        super().__init__()

        self.device_embedding = nn.Embedding(num_device_types, output_dim // 2)

        # Additional device features
        self.feature_encoder = nn.Sequential(
            nn.Linear(4, output_dim // 2),  # screen_size, is_mobile, is_shared, connection_speed
            nn.GELU()
        )

        self.fusion = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU()
        )

    def forward(
        self,
        device_type: torch.Tensor,
        device_features: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Encode device context.

        Args:
            device_type: Device type IDs [batch_size]
            device_features: Additional features [batch_size, 4]

        Returns:
            Device context embedding [batch_size, output_dim]
        """
        device_emb = self.device_embedding(device_type)

        if device_features is not None:
            feat_emb = self.feature_encoder(device_features)
            combined = torch.cat([device_emb, feat_emb], dim=-1)
        else:
            # Pad with zeros
            batch_size = device_type.size(0)
            padding = torch.zeros(batch_size, device_emb.size(-1), device=device_emb.device)
            combined = torch.cat([device_emb, padding], dim=-1)

        return self.fusion(combined)


class SocialContextEncoder(nn.Module):
    """
    Encode social context (watching alone, with partner, with family, etc.).

    Social context affects preferences:
    - Alone: niche interests, intense content
    - Partner: romance, drama
    - Family: family-friendly, crowd-pleasers
    - Friends: comedy, action
    """

    def __init__(self, output_dim: int = 64):
        super().__init__()

        # Viewing party size
        self.party_size_embedding = nn.Embedding(10, 32)  # 1-10+ people

        # Viewing mode
        self.viewing_mode_embedding = nn.Embedding(
            5, 32  # alone, partner, family, friends, public
        )

        self.fusion = nn.Sequential(
            nn.Linear(64, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU()
        )

    def forward(
        self,
        party_size: torch.Tensor,
        viewing_mode: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Encode social context.

        Args:
            party_size: Number of viewers [batch_size]
            viewing_mode: Viewing mode ID [batch_size]

        Returns:
            Social context embedding [batch_size, output_dim]
        """
        # Clip party size to max 9 (10+ = index 9)
        party_size = torch.clamp(party_size, 0, 9)
        party_emb = self.party_size_embedding(party_size)

        if viewing_mode is not None:
            mode_emb = self.viewing_mode_embedding(viewing_mode)
            combined = torch.cat([party_emb, mode_emb], dim=-1)
        else:
            # Infer viewing mode from party size
            combined = torch.cat([party_emb, party_emb], dim=-1)

        return self.fusion(combined)


class MoodContextEncoder(nn.Module):
    """
    Encode user mood/intent (inferred from recent activity).

    Mood affects preferences:
    - Relaxed: light comedies, feel-good
    - Stressed: escapism, fantasy
    - Energetic: action, thrillers
    - Contemplative: drama, documentaries
    """

    def __init__(self, output_dim: int = 64):
        super().__init__()

        # Mood dimensions (valence, arousal, dominance)
        self.mood_encoder = nn.Sequential(
            nn.Linear(3, 32),
            nn.GELU(),
            nn.Linear(32, output_dim // 2)
        )

        # Recent activity patterns (watch time, genres, skip rate)
        self.activity_encoder = nn.Sequential(
            nn.Linear(10, 32),
            nn.GELU(),
            nn.Linear(32, output_dim // 2)
        )

        self.fusion = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU()
        )

    def forward(
        self,
        mood_features: Optional[torch.Tensor] = None,
        activity_features: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Encode mood context.

        Args:
            mood_features: Mood dimensions [batch_size, 3] (valence, arousal, dominance)
            activity_features: Recent activity [batch_size, 10]

        Returns:
            Mood context embedding [batch_size, output_dim]
        """
        features = []

        if mood_features is not None:
            mood_emb = self.mood_encoder(mood_features)
            features.append(mood_emb)

        if activity_features is not None:
            activity_emb = self.activity_encoder(activity_features)
            features.append(activity_emb)

        if features:
            combined = torch.cat(features, dim=-1)
            return self.fusion(combined)
        else:
            # Return zero embedding
            return torch.zeros(1, self.output_dim)


class ContextAwareRecommender(nn.Module):
    """
    Complete context-aware recommendation system.

    Adapts recommendations based on multiple context dimensions:
    - Temporal: time of day, day of week, season
    - Device: mobile, TV, browser
    - Social: alone, with others
    - Mood: inferred from activity
    """

    def __init__(
        self,
        base_recommender: nn.Module,
        user_embed_dim: int = 512,
        item_embed_dim: int = 512,
        context_dim: int = 256,
        use_temporal: bool = True,
        use_device: bool = True,
        use_social: bool = True,
        use_mood: bool = True,
        context_fusion: str = 'attention'  # 'concat', 'attention', 'gated'
    ):
        super().__init__()

        self.base_recommender = base_recommender
        self.context_fusion = context_fusion

        # Context encoders
        self.context_encoders = nn.ModuleDict()

        if use_temporal:
            self.context_encoders['temporal'] = TemporalContextEncoder(
                output_dim=context_dim // 4
            )

        if use_device:
            self.context_encoders['device'] = DeviceContextEncoder(
                output_dim=context_dim // 4
            )

        if use_social:
            self.context_encoders['social'] = SocialContextEncoder(
                output_dim=context_dim // 4
            )

        if use_mood:
            self.context_encoders['mood'] = MoodContextEncoder(
                output_dim=context_dim // 4
            )

        # Calculate total context dimension
        num_contexts = len(self.context_encoders)
        total_context_dim = (context_dim // 4) * num_contexts

        # Context fusion
        if context_fusion == 'concat':
            # Concatenate contexts with user/item embeddings
            self.context_adapter = nn.Sequential(
                nn.Linear(total_context_dim, context_dim),
                nn.LayerNorm(context_dim),
                nn.GELU()
            )

            # Modulate user embeddings
            self.user_modulation = nn.Sequential(
                nn.Linear(user_embed_dim + context_dim, user_embed_dim),
                nn.LayerNorm(user_embed_dim),
                nn.GELU()
            )

            # Modulate item embeddings
            self.item_modulation = nn.Sequential(
                nn.Linear(item_embed_dim + context_dim, item_embed_dim),
                nn.LayerNorm(item_embed_dim),
                nn.GELU()
            )

        elif context_fusion == 'attention':
            # Attention-based fusion
            self.context_attention = nn.MultiheadAttention(
                context_dim, num_heads=8, batch_first=True
            )

            self.user_context_fusion = nn.MultiheadAttention(
                user_embed_dim, num_heads=8, batch_first=True
            )

        elif context_fusion == 'gated':
            # Gated fusion (context gates user/item)
            self.user_gate = nn.Sequential(
                nn.Linear(total_context_dim, user_embed_dim),
                nn.Sigmoid()
            )

            self.item_gate = nn.Sequential(
                nn.Linear(total_context_dim, item_embed_dim),
                nn.Sigmoid()
            )

    def encode_context(self, context_data: Dict[str, Dict]) -> torch.Tensor:
        """
        Encode all context dimensions.

        Args:
            context_data: Dict of {context_type: context_inputs}

        Returns:
            Combined context embedding
        """
        context_embeddings = []

        for context_name, encoder in self.context_encoders.items():
            if context_name in context_data:
                ctx_emb = encoder(**context_data[context_name])
                context_embeddings.append(ctx_emb)

        if context_embeddings:
            return torch.cat(context_embeddings, dim=-1)
        else:
            # Return zero embedding
            batch_size = list(context_data.values())[0].get('batch_size', 1)
            return torch.zeros(batch_size, 256)

    def forward(
        self,
        user_ids: torch.Tensor,
        item_ids: torch.Tensor,
        context_data: Optional[Dict[str, Dict]] = None,
        **base_model_kwargs
    ) -> torch.Tensor:
        """
        Context-aware forward pass.

        Args:
            user_ids: User IDs
            item_ids: Item IDs
            context_data: Context information
            **base_model_kwargs: Additional arguments for base model

        Returns:
            Context-aware predictions
        """
        # Get base model embeddings
        if hasattr(self.base_recommender, 'encode_users'):
            user_emb = self.base_recommender.encode_users(user_ids)
            item_emb = self.base_recommender.encode_items(item_ids)
        else:
            # Fallback: get from forward pass
            base_output = self.base_recommender(user_ids, item_ids, **base_model_kwargs)
            if isinstance(base_output, dict):
                user_emb = base_output.get('user_embedding')
                item_emb = base_output.get('item_embedding')
            else:
                # Can't separate embeddings, return base prediction
                return base_output

        # Encode context
        if context_data:
            context_emb = self.encode_context(context_data)

            # Apply context-aware modulation
            if self.context_fusion == 'concat':
                # Concatenate context with embeddings
                context_adapted = self.context_adapter(context_emb)

                user_with_context = torch.cat([user_emb, context_adapted], dim=-1)
                user_emb = self.user_modulation(user_with_context)

                item_with_context = torch.cat([item_emb, context_adapted], dim=-1)
                item_emb = self.item_modulation(item_with_context)

            elif self.context_fusion == 'attention':
                # Attention-based modulation
                # Context attends to user
                user_emb_seq = user_emb.unsqueeze(1)
                context_seq = context_emb.unsqueeze(1)

                user_modulated, _ = self.user_context_fusion(
                    user_emb_seq, context_seq, context_seq
                )
                user_emb = user_modulated.squeeze(1) + user_emb  # Residual

            elif self.context_fusion == 'gated':
                # Gate user and item embeddings
                user_gate = self.user_gate(context_emb)
                item_gate = self.item_gate(context_emb)

                user_emb = user_emb * user_gate
                item_emb = item_emb * item_gate

        # Compute final score
        score = torch.sum(user_emb * item_emb, dim=-1)

        return score

    def recommend(
        self,
        user_id: int,
        candidate_items: torch.Tensor,
        context_data: Optional[Dict] = None,
        top_k: int = 10
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get context-aware recommendations.

        Args:
            user_id: User ID
            candidate_items: Candidate item IDs
            context_data: Context information
            top_k: Number of recommendations

        Returns:
            (top_items, top_scores)
        """
        self.eval()
        with torch.no_grad():
            batch_size = len(candidate_items)
            user_ids = torch.tensor([user_id] * batch_size)

            # Broadcast context to batch
            if context_data:
                for ctx_name, ctx_inputs in context_data.items():
                    for key, value in ctx_inputs.items():
                        if isinstance(value, torch.Tensor) and value.size(0) == 1:
                            ctx_inputs[key] = value.repeat(batch_size, *([1] * (value.dim() - 1)))

            # Get scores
            scores = self.forward(user_ids, candidate_items, context_data)

            # Get top-k
            top_scores, top_indices = torch.topk(scores, min(top_k, len(scores)))
            top_items = candidate_items[top_indices]

            return top_items, top_scores


# Example usage
if __name__ == "__main__":
    from collaborative.src.model import NeuralCollaborativeFiltering

    # Base recommender
    base_model = NeuralCollaborativeFiltering(
        num_users=10000,
        num_items=5000,
        embedding_dim=512
    )

    # Wrap with context-aware layer
    context_model = ContextAwareRecommender(
        base_recommender=base_model,
        user_embed_dim=512,
        item_embed_dim=512,
        use_temporal=True,
        use_device=True,
        use_social=True,
        use_mood=True,
        context_fusion='attention'
    )

    # Example: Recommend on Friday evening, on TV, with family
    user_id = 42
    candidates = torch.randint(0, 5000, (100,))

    context = {
        'temporal': {
            'hour': torch.tensor([20]),  # 8 PM
            'day_of_week': torch.tensor([4]),  # Friday
            'month': torch.tensor([12])  # December
        },
        'device': {
            'device_type': torch.tensor([2]),  # TV
            'device_features': torch.tensor([[55.0, 0.0, 1.0, 1.0]])  # 55" screen, not mobile, shared, good connection
        },
        'social': {
            'party_size': torch.tensor([4]),  # Family of 4
            'viewing_mode': torch.tensor([2])  # Family mode
        }
    }

    top_items, scores = context_model.recommend(
        user_id, candidates, context, top_k=10
    )

    print(f"Context-aware recommendations for user {user_id}:")
    print(f"Items: {top_items}")
    print(f"Scores: {scores}")
