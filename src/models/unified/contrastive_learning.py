"""
Contrastive Learning Framework for CineSync v2
Adds powerful contrastive learning to all movie and TV models
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import numpy as np


class InfoNCELoss(nn.Module):
    """
    InfoNCE (Normalized Cross-Entropy) contrastive loss.

    Used in models like SimCLR, MoCo, and many recommendation systems.
    Pulls positive pairs together, pushes negative pairs apart.
    """

    def __init__(self, temperature: float = 0.07, reduction: str = 'mean'):
        super().__init__()
        self.temperature = temperature
        self.reduction = reduction

    def forward(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negatives: Optional[torch.Tensor] = None,
        in_batch_negatives: bool = True
    ) -> torch.Tensor:
        """
        Compute InfoNCE loss.

        Args:
            anchor: Anchor embeddings [batch_size, embed_dim]
            positive: Positive embeddings [batch_size, embed_dim]
            negatives: Optional negative embeddings [batch_size, num_neg, embed_dim]
            in_batch_negatives: Use other samples in batch as negatives

        Returns:
            Loss value
        """
        batch_size = anchor.size(0)

        # Normalize embeddings
        anchor = F.normalize(anchor, p=2, dim=-1)
        positive = F.normalize(positive, p=2, dim=-1)

        # Positive similarities
        pos_sim = torch.sum(anchor * positive, dim=-1) / self.temperature  # [batch]

        # Negative similarities
        if in_batch_negatives:
            # Use all other samples in batch as negatives (very efficient!)
            # Create matrix of all pairwise similarities
            sim_matrix = torch.matmul(anchor, positive.T) / self.temperature  # [batch, batch]

            # Mask out the positive pairs (diagonal)
            mask = torch.eye(batch_size, device=anchor.device, dtype=torch.bool)
            neg_sim = sim_matrix.masked_fill(mask, float('-inf'))

            # Combine positive and negative similarities
            logits = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1)  # [batch, batch+1]

            # Labels: first position is always positive
            labels = torch.zeros(batch_size, dtype=torch.long, device=anchor.device)

        else:
            # Use provided negatives
            if negatives is None:
                raise ValueError("Must provide negatives if in_batch_negatives=False")

            negatives = F.normalize(negatives, p=2, dim=-1)

            # Compute negative similarities
            neg_sim = torch.matmul(
                anchor.unsqueeze(1), negatives.transpose(1, 2)
            ).squeeze(1) / self.temperature  # [batch, num_neg]

            # Combine
            logits = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1)
            labels = torch.zeros(batch_size, dtype=torch.long, device=anchor.device)

        # Cross-entropy loss
        loss = F.cross_entropy(logits, labels, reduction=self.reduction)

        return loss


class ContrastiveLearningModule(nn.Module):
    """
    Contrastive learning wrapper that can be added to any recommendation model.

    Enhances model by:
    - Learning better user/item representations through contrastive loss
    - Handling data augmentation for creating positive/negative pairs
    - Supporting multiple contrastive strategies
    """

    def __init__(
        self,
        base_model: nn.Module,
        embed_dim: int = 512,
        projection_dim: int = 256,
        temperature: float = 0.07,
        use_momentum_encoder: bool = True,
        momentum: float = 0.999,
        queue_size: int = 65536
    ):
        super().__init__()

        self.base_model = base_model
        self.embed_dim = embed_dim
        self.projection_dim = projection_dim
        self.temperature = temperature
        self.use_momentum_encoder = use_momentum_encoder

        # Projection head (maps embeddings to contrastive space)
        self.projection_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, projection_dim),
            nn.BatchNorm1d(projection_dim)
        )

        # Momentum encoder (for MoCo-style contrastive learning)
        if use_momentum_encoder:
            self.momentum_encoder = self._create_momentum_encoder()
            self.momentum = momentum

            # Queue for storing negative samples
            self.register_buffer("queue", torch.randn(queue_size, projection_dim))
            self.queue = F.normalize(self.queue, dim=-1)
            self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
            self.queue_size = queue_size

        # Loss
        self.criterion = InfoNCELoss(temperature=temperature)

    def _create_momentum_encoder(self):
        """Create momentum encoder (copy of base model)"""
        import copy
        momentum_encoder = copy.deepcopy(self.base_model)

        # Freeze momentum encoder parameters
        for param in momentum_encoder.parameters():
            param.requires_grad = False

        return momentum_encoder

    @torch.no_grad()
    def _update_momentum_encoder(self):
        """Update momentum encoder using exponential moving average"""
        for param_base, param_momentum in zip(
            self.base_model.parameters(),
            self.momentum_encoder.parameters()
        ):
            param_momentum.data = (
                param_momentum.data * self.momentum +
                param_base.data * (1. - self.momentum)
            )

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys: torch.Tensor):
        """Update the queue with new keys"""
        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)

        # Replace oldest samples in queue
        if ptr + batch_size <= self.queue_size:
            self.queue[ptr:ptr + batch_size] = keys
        else:
            # Wrap around
            remaining = self.queue_size - ptr
            self.queue[ptr:] = keys[:remaining]
            self.queue[:batch_size - remaining] = keys[remaining:]

        # Update pointer
        ptr = (ptr + batch_size) % self.queue_size
        self.queue_ptr[0] = ptr

    def forward(
        self,
        anchor_inputs: Dict,
        positive_inputs: Dict,
        return_embeddings: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with contrastive learning.

        Args:
            anchor_inputs: Inputs for anchor samples
            positive_inputs: Inputs for positive samples (augmented anchors)
            return_embeddings: Whether to return embeddings

        Returns:
            Dictionary with loss and optional embeddings
        """
        # Get embeddings from base model
        anchor_emb = self.base_model(**anchor_inputs)

        # Handle different output types
        if isinstance(anchor_emb, dict):
            anchor_emb = anchor_emb.get('embedding', anchor_emb.get('final_embedding'))

        # Project to contrastive space
        anchor_proj = self.projection_head(anchor_emb)
        anchor_proj = F.normalize(anchor_proj, dim=-1)

        # Get positive embeddings
        if self.use_momentum_encoder and self.training:
            # Use momentum encoder for positives (MoCo style)
            with torch.no_grad():
                self._update_momentum_encoder()
                positive_emb = self.momentum_encoder(**positive_inputs)
                if isinstance(positive_emb, dict):
                    positive_emb = positive_emb.get('embedding', positive_emb.get('final_embedding'))
                positive_proj = self.projection_head(positive_emb)
                positive_proj = F.normalize(positive_proj, dim=-1)
        else:
            # Use same encoder for positives (SimCLR style)
            positive_emb = self.base_model(**positive_inputs)
            if isinstance(positive_emb, dict):
                positive_emb = positive_emb.get('embedding', positive_emb.get('final_embedding'))
            positive_proj = self.projection_head(positive_emb)
            positive_proj = F.normalize(positive_proj, dim=-1)

        # Compute contrastive loss
        if self.use_momentum_encoder and self.training:
            # Use queue as negatives (MoCo style)
            # Positive logits
            pos_logits = torch.sum(anchor_proj * positive_proj, dim=-1, keepdim=True)
            pos_logits /= self.temperature

            # Negative logits (from queue)
            neg_logits = torch.matmul(anchor_proj, self.queue.T)
            neg_logits /= self.temperature

            # Concatenate
            logits = torch.cat([pos_logits, neg_logits], dim=1)
            labels = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)

            loss = F.cross_entropy(logits, labels)

            # Update queue
            self._dequeue_and_enqueue(positive_proj)
        else:
            # Use in-batch negatives (SimCLR style)
            loss = self.criterion(
                anchor_proj, positive_proj, in_batch_negatives=True
            )

        outputs = {'contrastive_loss': loss}

        if return_embeddings:
            outputs.update({
                'anchor_embedding': anchor_emb,
                'positive_embedding': positive_emb,
                'anchor_projection': anchor_proj,
                'positive_projection': positive_proj
            })

        return outputs


class DataAugmentation:
    """
    Data augmentation strategies for creating positive pairs in recommendation.

    For collaborative filtering:
    - Sequence dropout: randomly drop items from sequence
    - Sequence reordering: slightly reorder items
    - Feature masking: mask some features

    For content-based:
    - Text augmentation: back-translation, synonym replacement
    - Image augmentation: crop, color jitter
    - Metadata augmentation: noise injection
    """

    @staticmethod
    def sequence_dropout(
        sequence: torch.Tensor,
        dropout_prob: float = 0.1,
        pad_token: int = 0
    ) -> torch.Tensor:
        """Randomly drop items from sequence"""
        mask = torch.rand(sequence.shape, device=sequence.device) > dropout_prob
        # Don't drop pad tokens
        mask |= (sequence == pad_token)
        return sequence * mask.long()

    @staticmethod
    def sequence_reorder(
        sequence: torch.Tensor,
        reorder_prob: float = 0.1,
        pad_token: int = 0
    ) -> torch.Tensor:
        """Slightly reorder sequence (swap adjacent items)"""
        augmented = sequence.clone()
        batch_size, seq_len = sequence.shape

        for i in range(batch_size):
            for j in range(seq_len - 1):
                if sequence[i, j] != pad_token and sequence[i, j+1] != pad_token:
                    if torch.rand(1).item() < reorder_prob:
                        # Swap
                        augmented[i, j], augmented[i, j+1] = augmented[i, j+1], augmented[i, j]

        return augmented

    @staticmethod
    def feature_masking(
        features: torch.Tensor,
        mask_prob: float = 0.15
    ) -> torch.Tensor:
        """Randomly mask features"""
        mask = torch.rand(features.shape, device=features.device) > mask_prob
        return features * mask

    @staticmethod
    def create_positive_pair(
        user_id: torch.Tensor,
        item_id: torch.Tensor,
        sequence: Optional[torch.Tensor] = None,
        features: Optional[torch.Tensor] = None,
        augmentation_type: str = 'mixed'
    ) -> Tuple[Dict, Dict]:
        """
        Create anchor and positive pair with augmentation.

        Args:
            user_id: User ID
            item_id: Item ID
            sequence: Optional sequence data
            features: Optional features
            augmentation_type: 'dropout', 'reorder', 'mask', or 'mixed'

        Returns:
            Tuple of (anchor_inputs, positive_inputs)
        """
        anchor_inputs = {
            'user_id': user_id,
            'item_id': item_id
        }

        positive_inputs = {
            'user_id': user_id,
            'item_id': item_id
        }

        # Augment sequence if provided
        if sequence is not None:
            anchor_inputs['sequence'] = sequence

            if augmentation_type in ['dropout', 'mixed']:
                augmented_seq = DataAugmentation.sequence_dropout(sequence)
            elif augmentation_type == 'reorder':
                augmented_seq = DataAugmentation.sequence_reorder(sequence)
            else:
                augmented_seq = sequence

            positive_inputs['sequence'] = augmented_seq

        # Augment features if provided
        if features is not None:
            anchor_inputs['features'] = features

            if augmentation_type in ['mask', 'mixed']:
                augmented_features = DataAugmentation.feature_masking(features)
            else:
                augmented_features = features

            positive_inputs['features'] = augmented_features

        return anchor_inputs, positive_inputs


class HardNegativeMiner:
    """
    Mine hard negatives for contrastive learning.

    Hard negatives are samples that are similar to anchor but not positive.
    These are more informative than random negatives.
    """

    def __init__(
        self,
        embed_dim: int,
        num_hard_negatives: int = 10,
        similarity_threshold: float = 0.7
    ):
        self.embed_dim = embed_dim
        self.num_hard_negatives = num_hard_negatives
        self.similarity_threshold = similarity_threshold

    def mine_hard_negatives(
        self,
        anchor_emb: torch.Tensor,
        candidate_emb: torch.Tensor,
        positive_indices: torch.Tensor
    ) -> torch.Tensor:
        """
        Mine hard negatives from candidates.

        Args:
            anchor_emb: Anchor embeddings [batch_size, embed_dim]
            candidate_emb: Candidate embeddings [num_candidates, embed_dim]
            positive_indices: Indices of positive samples [batch_size]

        Returns:
            Hard negative indices [batch_size, num_hard_negatives]
        """
        batch_size = anchor_emb.size(0)

        # Normalize
        anchor_emb = F.normalize(anchor_emb, p=2, dim=-1)
        candidate_emb = F.normalize(candidate_emb, p=2, dim=-1)

        # Compute similarities
        similarities = torch.matmul(anchor_emb, candidate_emb.T)  # [batch, num_candidates]

        # Mask out positive samples
        for i, pos_idx in enumerate(positive_indices):
            similarities[i, pos_idx] = -float('inf')

        # Get hard negatives (high similarity but not positive)
        # Filter by similarity threshold
        hard_mask = similarities > self.similarity_threshold

        # Get top-k hard negatives for each anchor
        hard_negative_indices = []
        for i in range(batch_size):
            hard_sims = similarities[i][hard_mask[i]]
            hard_idxs = torch.where(hard_mask[i])[0]

            if len(hard_idxs) >= self.num_hard_negatives:
                # Get top-k hardest
                _, top_k_indices = torch.topk(
                    hard_sims, self.num_hard_negatives
                )
                selected = hard_idxs[top_k_indices]
            else:
                # Not enough hard negatives, sample randomly
                num_needed = self.num_hard_negatives - len(hard_idxs)
                random_idxs = torch.randperm(candidate_emb.size(0))[:num_needed]
                selected = torch.cat([hard_idxs, random_idxs])

            hard_negative_indices.append(selected)

        return torch.stack(hard_negative_indices)


# Example usage
if __name__ == "__main__":
    # Example: Add contrastive learning to any model
    from collaborative.src.model import NeuralCollaborativeFiltering

    # Base model
    base_model = NeuralCollaborativeFiltering(
        num_users=10000,
        num_items=5000,
        embedding_dim=512
    )

    # Wrap with contrastive learning
    contrastive_model = ContrastiveLearningModule(
        base_model=base_model,
        embed_dim=512,
        projection_dim=256,
        temperature=0.07,
        use_momentum_encoder=True
    )

    # Training example
    batch_size = 32
    user_ids = torch.randint(0, 10000, (batch_size,))
    item_ids = torch.randint(0, 5000, (batch_size,))
    sequences = torch.randint(0, 5000, (batch_size, 50))

    # Create positive pairs with augmentation
    anchor_inputs = {
        'user_ids': user_ids,
        'item_ids': item_ids
    }

    # Augment for positive
    positive_inputs = {
        'user_ids': user_ids,
        'item_ids': item_ids  # Same items but could augment features
    }

    # Forward pass
    outputs = contrastive_model(
        anchor_inputs, positive_inputs, return_embeddings=True
    )

    print(f"Contrastive loss: {outputs['contrastive_loss'].item()}")
    print(f"Embedding shape: {outputs['anchor_embedding'].shape}")

# Alias for backward compatibility
UnifiedContrastiveLearning = ContrastiveLearningModule
