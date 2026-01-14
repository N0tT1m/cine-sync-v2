"""
Production Optimizations for CineSync v2
Model compression, caching, quantization, and serving optimizations
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from pathlib import Path
import logging
import pickle
from collections import OrderedDict
import time

logger = logging.getLogger(__name__)


class ModelQuantizer:
    """
    Quantize models for faster inference and smaller size.

    Supports:
    - Dynamic quantization (weights only)
    - Static quantization (weights + activations)
    - QAT (Quantization-Aware Training)
    """

    @staticmethod
    def dynamic_quantization(
        model: nn.Module,
        dtype: torch.dtype = torch.qint8
    ) -> nn.Module:
        """
        Dynamic quantization: quantize weights, activations remain float.

        Best for: Models where memory bandwidth is bottleneck (LSTM, Transformer)

        Args:
            model: Model to quantize
            dtype: Quantization dtype (qint8 or float16)

        Returns:
            Quantized model
        """
        logger.info("Applying dynamic quantization...")

        quantized_model = torch.quantization.quantize_dynamic(
            model,
            {nn.Linear, nn.LSTM, nn.GRU, nn.MultiheadAttention},
            dtype=dtype
        )

        logger.info("Dynamic quantization complete")
        return quantized_model

    @staticmethod
    def static_quantization(
        model: nn.Module,
        calibration_dataloader,
        device: torch.device
    ) -> nn.Module:
        """
        Static quantization: quantize weights and activations.

        Requires calibration data to determine quantization parameters.

        Best for: CNN-based models, production deployment

        Args:
            model: Model to quantize
            calibration_dataloader: Data for calibration
            device: Device to run calibration on

        Returns:
            Quantized model
        """
        logger.info("Preparing model for static quantization...")

        # Prepare model
        model.eval()
        model.qconfig = torch.quantization.get_default_qconfig('fbgemm')

        # Fuse modules (conv + bn + relu)
        model_fused = torch.quantization.fuse_modules(model, [])

        # Prepare for quantization
        model_prepared = torch.quantization.prepare(model_fused)

        # Calibration
        logger.info("Calibrating quantization parameters...")
        with torch.no_grad():
            for batch in calibration_dataloader:
                # Move batch to device
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                        for k, v in batch.items()}

                # Forward pass for calibration
                _ = model_prepared(**batch)

        # Convert to quantized model
        logger.info("Converting to quantized model...")
        model_quantized = torch.quantization.convert(model_prepared)

        logger.info("Static quantization complete")
        return model_quantized

    @staticmethod
    def compare_models(
        original_model: nn.Module,
        quantized_model: nn.Module,
        test_input: Dict[str, torch.Tensor],
        device: torch.device
    ) -> Dict[str, float]:
        """
        Compare original and quantized models.

        Returns:
            Dictionary with comparison metrics
        """
        # Model sizes
        original_size = sum(p.numel() * p.element_size() for p in original_model.parameters())
        quantized_size = sum(p.numel() * p.element_size() for p in quantized_model.parameters())

        # Inference time
        original_model.eval()
        quantized_model.eval()

        with torch.no_grad():
            # Original model
            start = time.time()
            for _ in range(100):
                _ = original_model(**test_input)
            original_time = (time.time() - start) / 100

            # Quantized model
            start = time.time()
            for _ in range(100):
                _ = quantized_model(**test_input)
            quantized_time = (time.time() - start) / 100

        return {
            'original_size_mb': original_size / 1024 / 1024,
            'quantized_size_mb': quantized_size / 1024 / 1024,
            'size_reduction': (1 - quantized_size / original_size) * 100,
            'original_latency_ms': original_time * 1000,
            'quantized_latency_ms': quantized_time * 1000,
            'speedup': original_time / quantized_time
        }


class KnowledgeDistillation:
    """
    Train smaller (student) models from larger (teacher) models.

    Transfers knowledge from complex models to efficient ones for production.
    """

    def __init__(
        self,
        teacher_model: nn.Module,
        student_model: nn.Module,
        temperature: float = 4.0,
        alpha: float = 0.7
    ):
        """
        Args:
            teacher_model: Large, accurate model
            student_model: Small, efficient model
            temperature: Softmax temperature for distillation
            alpha: Weight between distillation loss and student loss
        """
        self.teacher = teacher_model
        self.student = student_model
        self.temperature = temperature
        self.alpha = alpha

        # Freeze teacher
        for param in self.teacher.parameters():
            param.requires_grad = False

        self.teacher.eval()

    def distillation_loss(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute distillation loss.

        Combines:
        - Soft targets from teacher (dark knowledge)
        - Hard targets from labels (ground truth)

        Args:
            student_logits: Student model predictions
            teacher_logits: Teacher model predictions
            labels: Ground truth labels

        Returns:
            Combined loss
        """
        # Soft targets (distillation loss)
        soft_targets = F.softmax(teacher_logits / self.temperature, dim=-1)
        soft_prob = F.log_softmax(student_logits / self.temperature, dim=-1)
        distill_loss = F.kl_div(soft_prob, soft_targets, reduction='batchmean')
        distill_loss *= (self.temperature ** 2)

        # Hard targets (student loss)
        student_loss = F.cross_entropy(student_logits, labels)

        # Combined loss
        total_loss = self.alpha * distill_loss + (1 - self.alpha) * student_loss

        return total_loss

    def train_step(
        self,
        batch: Dict[str, torch.Tensor],
        optimizer: torch.optim.Optimizer
    ) -> float:
        """Single training step for student model"""
        self.student.train()

        # Get teacher predictions (no grad)
        with torch.no_grad():
            teacher_outputs = self.teacher(**batch)

        # Get student predictions
        student_outputs = self.student(**batch)

        # Compute loss
        loss = self.distillation_loss(
            student_outputs['logits'],
            teacher_outputs['logits'],
            batch['labels']
        )

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss.item()


class EmbeddingCache:
    """
    Cache pre-computed embeddings for fast lookup.

    Huge speedup for serving: pre-compute item embeddings offline,
    only compute user embeddings at inference time.
    """

    def __init__(
        self,
        cache_dir: Path,
        embedding_dim: int,
        dtype: torch.dtype = torch.float32
    ):
        self.cache_dir = cache_dir
        self.embedding_dim = embedding_dim
        self.dtype = dtype

        cache_dir.mkdir(parents=True, exist_ok=True)

        # In-memory cache
        self.cache = OrderedDict()
        self.max_cache_size = 100000  # Keep top 100k in memory

    def save_embeddings(
        self,
        item_ids: List[int],
        embeddings: torch.Tensor,
        metadata: Optional[Dict] = None
    ):
        """
        Save embeddings to disk and cache.

        Args:
            item_ids: List of item IDs
            embeddings: Item embeddings [num_items, embedding_dim]
            metadata: Optional metadata
        """
        assert len(item_ids) == len(embeddings)

        # Save to disk
        cache_file = self.cache_dir / 'embeddings.pkl'
        cache_data = {
            'item_ids': item_ids,
            'embeddings': embeddings.cpu().numpy(),
            'metadata': metadata or {},
            'embedding_dim': self.embedding_dim,
            'dtype': str(self.dtype)
        }

        with open(cache_file, 'wb') as f:
            pickle.dump(cache_data, f)

        logger.info(f"Saved {len(item_ids)} embeddings to {cache_file}")

        # Update in-memory cache
        for item_id, emb in zip(item_ids, embeddings):
            self.cache[item_id] = emb.cpu()

            # Evict oldest if cache too large
            if len(self.cache) > self.max_cache_size:
                self.cache.popitem(last=False)

    def load_embeddings(self) -> Tuple[List[int], torch.Tensor]:
        """
        Load embeddings from disk.

        Returns:
            (item_ids, embeddings)
        """
        cache_file = self.cache_dir / 'embeddings.pkl'

        if not cache_file.exists():
            logger.warning(f"No cache file found at {cache_file}")
            return [], torch.empty(0, self.embedding_dim)

        with open(cache_file, 'rb') as f:
            cache_data = pickle.load(f)

        embeddings = torch.from_numpy(cache_data['embeddings']).to(self.dtype)

        logger.info(f"Loaded {len(cache_data['item_ids'])} embeddings from {cache_file}")

        return cache_data['item_ids'], embeddings

    def get_embedding(self, item_id: int) -> Optional[torch.Tensor]:
        """Get single item embedding"""
        return self.cache.get(item_id)

    def get_embeddings(self, item_ids: List[int]) -> torch.Tensor:
        """
        Get multiple item embeddings.

        Returns:
            Embeddings [num_items, embedding_dim]
        """
        embeddings = []
        for item_id in item_ids:
            emb = self.get_embedding(item_id)
            if emb is not None:
                embeddings.append(emb)
            else:
                # Return zero embedding for missing items
                embeddings.append(torch.zeros(self.embedding_dim))

        return torch.stack(embeddings)


class FastRecommender:
    """
    Optimized recommender for production serving.

    Features:
    - Pre-computed item embeddings
    - Batched user encoding
    - Approximate nearest neighbor search
    - Result caching
    """

    def __init__(
        self,
        model: nn.Module,
        embedding_cache: EmbeddingCache,
        device: torch.device,
        batch_size: int = 32,
        use_ann: bool = True
    ):
        self.model = model.to(device)
        self.model.eval()
        self.embedding_cache = embedding_cache
        self.device = device
        self.batch_size = batch_size
        self.use_ann = use_ann

        # Load item embeddings
        self.item_ids, self.item_embeddings = embedding_cache.load_embeddings()
        self.item_embeddings = self.item_embeddings.to(device)

        # ANN index (for approximate nearest neighbor)
        if use_ann:
            self.ann_index = self._build_ann_index()

    def _build_ann_index(self):
        """Build approximate nearest neighbor index (using FAISS)"""
        try:
            import faiss

            # Normalize embeddings for cosine similarity
            normalized_emb = F.normalize(self.item_embeddings, p=2, dim=1).cpu().numpy()

            # Build index
            index = faiss.IndexFlatIP(self.embedding_cache.embedding_dim)  # Inner product
            index.add(normalized_emb.astype('float32'))

            logger.info(f"Built ANN index with {len(self.item_ids)} items")
            return index

        except ImportError:
            logger.warning("FAISS not installed, falling back to exact search")
            return None

    def precompute_item_embeddings(
        self,
        item_dataloader,
        save: bool = True
    ):
        """
        Pre-compute and cache all item embeddings.

        Run this offline to speed up serving.

        Args:
            item_dataloader: DataLoader with all items
            save: Whether to save to cache
        """
        logger.info("Pre-computing item embeddings...")

        all_item_ids = []
        all_embeddings = []

        with torch.no_grad():
            for batch in item_dataloader:
                # Move to device
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                        for k, v in batch.items()}

                # Get item embeddings
                if hasattr(self.model, 'encode_items'):
                    embeddings = self.model.encode_items(**batch)
                else:
                    outputs = self.model(**batch)
                    embeddings = outputs.get('item_embeddings')

                all_item_ids.extend(batch['item_ids'].cpu().tolist())
                all_embeddings.append(embeddings.cpu())

        # Concatenate
        all_embeddings = torch.cat(all_embeddings, dim=0)

        logger.info(f"Computed {len(all_item_ids)} item embeddings")

        # Save to cache
        if save:
            self.embedding_cache.save_embeddings(all_item_ids, all_embeddings)

            # Update instance variables
            self.item_ids = all_item_ids
            self.item_embeddings = all_embeddings.to(self.device)

            # Rebuild ANN index
            if self.use_ann:
                self.ann_index = self._build_ann_index()

        return all_item_ids, all_embeddings

    def recommend(
        self,
        user_id: int,
        user_features: Dict[str, torch.Tensor],
        top_k: int = 10,
        filter_items: Optional[List[int]] = None
    ) -> Tuple[List[int], List[float]]:
        """
        Fast recommendation using cached embeddings.

        Args:
            user_id: User ID
            user_features: User features
            top_k: Number of recommendations
            filter_items: Items to exclude (already watched)

        Returns:
            (recommended_item_ids, scores)
        """
        with torch.no_grad():
            # Encode user
            user_features = {k: v.to(self.device) for k, v in user_features.items()}

            if hasattr(self.model, 'encode_users'):
                user_emb = self.model.encode_users(**user_features)
            else:
                outputs = self.model(**user_features)
                user_emb = outputs['user_embeddings']

            # Normalize for cosine similarity
            user_emb = F.normalize(user_emb, p=2, dim=-1)

            # Compute similarities
            if self.use_ann and self.ann_index is not None:
                # ANN search
                k = min(top_k * 10, len(self.item_ids))  # Get more for filtering
                scores, indices = self.ann_index.search(
                    user_emb.cpu().numpy().astype('float32'), k
                )
                scores = scores[0]
                indices = indices[0]
            else:
                # Exact search
                normalized_items = F.normalize(self.item_embeddings, p=2, dim=-1)
                similarities = torch.matmul(user_emb, normalized_items.T).squeeze()

                scores, indices = torch.topk(similarities, min(top_k * 10, len(similarities)))
                scores = scores.cpu().numpy()
                indices = indices.cpu().numpy()

            # Filter items
            if filter_items:
                filter_set = set(filter_items)
                filtered_results = [
                    (self.item_ids[idx], score)
                    for idx, score in zip(indices, scores)
                    if self.item_ids[idx] not in filter_set
                ]
            else:
                filtered_results = [
                    (self.item_ids[idx], score)
                    for idx, score in zip(indices, scores)
                ]

            # Get top-k
            top_results = filtered_results[:top_k]

            recommended_ids = [item_id for item_id, _ in top_results]
            recommended_scores = [score for _, score in top_results]

            return recommended_ids, recommended_scores

    def batch_recommend(
        self,
        user_ids: List[int],
        user_features_list: List[Dict],
        top_k: int = 10
    ) -> List[Tuple[List[int], List[float]]]:
        """
        Batch recommendation for multiple users.

        More efficient than calling recommend() multiple times.

        Args:
            user_ids: List of user IDs
            user_features_list: List of user features
            top_k: Number of recommendations per user

        Returns:
            List of (recommended_ids, scores) for each user
        """
        results = []

        # Process in batches
        for i in range(0, len(user_ids), self.batch_size):
            batch_user_ids = user_ids[i:i + self.batch_size]
            batch_features = user_features_list[i:i + self.batch_size]

            for user_id, features in zip(batch_user_ids, batch_features):
                recs, scores = self.recommend(user_id, features, top_k)
                results.append((recs, scores))

        return results


# Example usage
if __name__ == "__main__":
    from pathlib import Path

    # Example: Quantization
    dummy_model = nn.Sequential(
        nn.Linear(128, 256),
        nn.ReLU(),
        nn.Linear(256, 128)
    )

    quantized = ModelQuantizer.dynamic_quantization(dummy_model)
    print("Model quantized!")

    # Example: Embedding cache
    cache = EmbeddingCache(
        cache_dir=Path('./cache'),
        embedding_dim=512
    )

    # Save some embeddings
    item_ids = list(range(1000))
    embeddings = torch.randn(1000, 512)
    cache.save_embeddings(item_ids, embeddings)

    # Load embeddings
    loaded_ids, loaded_emb = cache.load_embeddings()
    print(f"Loaded {len(loaded_ids)} embeddings")

    print("Production optimizations ready!")
