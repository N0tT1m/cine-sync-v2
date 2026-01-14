# CineSync v2 - Model Improvements Guide

## 🎯 Overview

This document describes the comprehensive improvements made to the CineSync v2 recommendation system. We've upgraded from 18+ basic movie models and 8 TV models to a state-of-the-art unified system with advanced capabilities.

## 📊 What We've Built

### 1. **Unified Cross-Domain Embeddings** (`src/models/unified/cross_domain_embeddings.py`)

**Problem Solved:** Users watch both movies and TV shows, but models treated them separately, losing valuable cross-domain knowledge.

**Solution:**
- Shared user embeddings across movies and TV
- Domain-specific adapters (movies vs TV preferences)
- Preference disentanglement (shared vs domain-specific tastes)
- Cross-domain transfer for cold-start

**Benefits:**
- 🚀 Better cold-start: Learn from movies to recommend TV (and vice versa)
- 📈 Improved accuracy: More data per user
- 💾 More efficient: Single user representation

**Example Usage:**
```python
from src.models.unified.cross_domain_embeddings import CrossDomainRecommender

model = CrossDomainRecommender(
    num_users=10000,
    num_movies=5000,
    num_tv_shows=3000,
    embedding_dim=512
)

# Recommend movies using TV watching history
top_movies, scores = model.recommend(
    user_id=42,
    domain='movie',
    candidate_items=candidate_movies,
    use_cross_domain=True  # Uses TV history!
)
```

### 2. **Movie Ensemble System** (`src/models/unified/movie_ensemble_system.py`)

**Problem Solved:** You had 18+ movie models but no way to combine them effectively.

**Solution:**
- Adaptive weighting based on input characteristics
- Uncertainty estimation for prediction confidence
- Multi-level fusion (embedding + prediction)
- Intelligent model selection

**Models Combined:**
- 4 Collaborative Filtering models (NCF, SimpleNCF, CrossAttention, DeepNCF)
- 4+ Two-Tower models
- 4 Sequential models (LSTM, Transformer, Attention, Session)
- 2 Advanced transformers (SASRec, Enhanced SASRec)

**Benefits:**
- 🎯 Best of all models: Ensemble outperforms individual models
- 🔍 Confidence scores: Know when predictions are reliable
- ⚡ Smart selection: Use right model for right situation

**Example Usage:**
```python
from src.models.unified.movie_ensemble_system import MovieEnsembleRecommender

ensemble = MovieEnsembleRecommender(
    num_users=10000,
    num_movies=5000,
    num_genres=20,
    fusion_strategy='attention',  # or 'weighted', 'learned'
    enable_collaborative_models=True,
    enable_sequential_models=True,
    enable_transformer_models=True
)

results = ensemble.recommend_movies(
    user_id=42,
    candidate_movie_ids=candidates,
    sequences=user_history,
    top_k=10,
    min_confidence=0.5  # Filter low-confidence predictions
)

print(f"Recommendations: {results['recommended_ids']}")
print(f"Confidence: {results['confidences']}")
print(f"Contributing models: {results['model_names']}")
```

### 3. **Contrastive Learning** (`src/models/unified/contrastive_learning.py`)

**Problem Solved:** Models learn better representations when they understand relationships between items.

**Solution:**
- InfoNCE loss (state-of-the-art contrastive learning)
- Hard negative mining (focus on difficult examples)
- Data augmentation for creating positive pairs
- MoCo-style momentum encoder

**Benefits:**
- 📊 Better embeddings: Items with similar content are closer
- 🎓 More data efficient: Learn from unlabeled data
- 🔥 SOTA performance: Used by best recommendation systems

**Example Usage:**
```python
from src.models.unified.contrastive_learning import ContrastiveLearningModule
from src.models.collaborative.src.model import NeuralCollaborativeFiltering

# Wrap any model with contrastive learning
base_model = NeuralCollaborativeFiltering(num_users=10000, num_items=5000)

contrastive_model = ContrastiveLearningModule(
    base_model=base_model,
    embed_dim=512,
    projection_dim=256,
    use_momentum_encoder=True,  # MoCo-style
    queue_size=65536  # Large queue of negatives
)

# Training with contrastive loss
outputs = contrastive_model(anchor_inputs, positive_inputs)
loss = outputs['contrastive_loss']
```

### 4. **Multimodal Features** (`src/models/unified/multimodal_features.py`)

**Problem Solved:** Only using IDs and basic metadata - missing rich content information.

**Solution:**
- **Text features:** BERT/RoBERTa encoders for plots, reviews
- **Visual features:** ResNet/ViT/CLIP for posters, thumbnails
- **Audio features:** Wav2Vec2 for soundtracks, trailers
- **Metadata encoding:** Smart handling of categorical + numerical
- **Advanced fusion:** Attention-based, gated, or learned fusion

**Benefits:**
- 🖼️ Rich representations: Understand content beyond IDs
- 🆕 Better cold-start: New items have content features
- 🎨 Multimodal understanding: Combine all signals

**Example Usage:**
```python
from src.models.unified.multimodal_features import CompleteMultimodalEncoder

encoder = CompleteMultimodalEncoder(
    text_output_dim=512,
    visual_output_dim=512,
    audio_output_dim=512,
    fusion_type='attention'
)

# Extract features from all modalities
features = encoder(
    text=["Epic sci-fi adventure with stunning visuals"],
    images=poster_tensor,  # [batch, 3, 224, 224]
    audio=soundtrack_tensor,  # Audio waveform
    metadata_encoder=metadata_enc,
    metadata_inputs={'genre': genre_ids, 'year': years}
)
```

### 5. **Context-Aware Recommendations** (`src/models/unified/context_aware.py`)

**Problem Solved:** Same user wants different content on Friday night vs Monday morning.

**Solution:**
- **Temporal context:** Time of day, day of week, season
- **Device context:** Mobile (short clips) vs TV (movies)
- **Social context:** Alone vs with family
- **Mood context:** Inferred from recent activity

**Benefits:**
- 🕐 Time-aware: Different recommendations at different times
- 📱 Device-adaptive: Mobile gets shorter content
- 👨‍👩‍👧‍👦 Social-aware: Family mode recommends appropriate content
- 🎭 Mood-sensitive: Relaxed evening vs energetic morning

**Example Usage:**
```python
from src.models.unified.context_aware import ContextAwareRecommender

context_model = ContextAwareRecommender(
    base_recommender=base_model,
    use_temporal=True,
    use_device=True,
    use_social=True,
    use_mood=True,
    context_fusion='attention'
)

# Friday evening, on TV, with family
context = {
    'temporal': {
        'hour': torch.tensor([20]),  # 8 PM
        'day_of_week': torch.tensor([4])  # Friday
    },
    'device': {
        'device_type': torch.tensor([2])  # TV
    },
    'social': {
        'party_size': torch.tensor([4]),  # Family of 4
        'viewing_mode': torch.tensor([2])  # Family mode
    }
}

top_items, scores = context_model.recommend(user_id, candidates, context)
```

### 6. **Comprehensive Evaluation** (`src/evaluation/comprehensive_metrics.py`)

**Problem Solved:** Only measuring accuracy misses diversity, fairness, and business impact.

**Solution:**
- **Accuracy:** Hit Rate, NDCG, MRR, Precision, Recall, MAP
- **Diversity:** Catalog coverage, intra-list diversity, novelty
- **Fairness:** Popularity bias, user fairness, Gini coefficient
- **Business:** CTR, engagement time, completion rate, retention

**Benefits:**
- 📊 Holistic view: Understand all aspects of system
- ⚖️ Fair recommendations: Avoid filter bubbles
- 💰 Business metrics: Track what matters

**Example Usage:**
```python
from src.evaluation.comprehensive_metrics import ComprehensiveEvaluator

evaluator = ComprehensiveEvaluator(
    compute_accuracy=True,
    compute_diversity=True,
    compute_fairness=True,
    compute_business=True
)

results = evaluator.evaluate(
    predictions=model_predictions,
    ground_truth=test_data,
    item_features=item_features,
    item_popularity=popularity_scores,
    user_groups=user_demographics,
    business_data=engagement_data,
    k_values=[5, 10, 20]
)

evaluator.print_results(results)
```

### 7. **Advanced Training** (`src/training/advanced_trainer.py`)

**Problem Solved:** Basic training misses modern optimization techniques.

**Solution:**
- **Multi-task learning:** Joint optimization with uncertainty weighting
- **Curriculum learning:** Start easy, increase difficulty
- **Advanced optimization:** Layer-wise LR decay, gradient accumulation
- **Mixed precision:** Faster training on RTX 4090
- **Early stopping:** Prevent overfitting
- **Automatic checkpointing:** Save best models

**Benefits:**
- 🚀 Faster convergence: Modern techniques train faster
- 📈 Better performance: Multi-task learning improves all tasks
- 💾 Efficient: Mixed precision uses less memory
- 🎯 Robust: Curriculum learning handles hard examples

**Example Usage:**
```python
from src.training.advanced_trainer import AdvancedTrainer

trainer = AdvancedTrainer(
    model=model,
    tasks=['recommendation', 'rating', 'click'],
    device=device,
    output_dir=Path('./checkpoints'),
    base_lr=1e-3,
    use_curriculum=True,
    use_multi_task=True,
    use_amp=True,  # Mixed precision for RTX 4090
    gradient_accumulation_steps=4,
    early_stopping_patience=5
)

# Train with multiple tasks
trainer.train(
    train_dataloader=train_loader,
    eval_dataloader=val_loader,
    task_loss_fns={
        'recommendation': nn.BCEWithLogitsLoss(),
        'rating': nn.MSELoss(),
        'click': nn.BCEWithLogitsLoss()
    },
    task_metric_fns={
        'recommendation': compute_ndcg,
        'rating': compute_rmse,
        'click': compute_auc
    },
    num_epochs=10
)
```

### 8. **Production Optimizations** (`src/production/optimizations.py`)

**Problem Solved:** Research models are slow and large for production.

**Solution:**
- **Quantization:** INT8 for 4x speedup, 75% size reduction
- **Knowledge distillation:** Tiny models from large ones
- **Embedding caching:** Pre-compute item embeddings offline
- **Fast serving:** Batched inference, ANN search
- **FAISS integration:** Sub-millisecond nearest neighbor search

**Benefits:**
- ⚡ 4x faster inference: Quantization + caching
- 💾 75% smaller models: Easier deployment
- 🚀 Sub-ms latency: FAISS approximate search
- 💰 Lower costs: Fewer servers needed

**Example Usage:**
```python
from src.production.optimizations import (
    ModelQuantizer, EmbeddingCache, FastRecommender
)

# 1. Quantize model
quantized_model = ModelQuantizer.dynamic_quantization(model)

# 2. Pre-compute item embeddings (offline)
cache = EmbeddingCache(cache_dir=Path('./cache'), embedding_dim=512)
fast_rec = FastRecommender(
    model=quantized_model,
    embedding_cache=cache,
    device=device,
    use_ann=True  # Use FAISS for fast search
)

# Pre-compute all items (run once offline)
fast_rec.precompute_item_embeddings(item_dataloader, save=True)

# 3. Fast inference (online)
recommended_ids, scores = fast_rec.recommend(
    user_id=42,
    user_features=user_data,
    top_k=10,
    filter_items=already_watched
)

# Batch recommendations for multiple users
results = fast_rec.batch_recommend(
    user_ids=user_batch,
    user_features_list=features_batch,
    top_k=10
)
```

## 🎓 Quick Start Guide

### Step 1: Unified Model Training

```python
from src.models.unified.cross_domain_embeddings import CrossDomainRecommender
from src.training.advanced_trainer import AdvancedTrainer

# Create unified model
model = CrossDomainRecommender(
    num_users=10000,
    num_movies=5000,
    num_tv_shows=3000,
    embedding_dim=512
)

# Train with advanced techniques
trainer = AdvancedTrainer(
    model=model,
    tasks=['recommendation', 'rating'],
    device=torch.device('cuda'),
    use_multi_task=True,
    use_curriculum=True,
    use_amp=True
)

trainer.train(train_loader, val_loader, task_loss_fns, task_metric_fns)
```

### Step 2: Add Contrastive Learning

```python
from src.models.unified.contrastive_learning import ContrastiveLearningModule

# Wrap model with contrastive learning
model_with_cl = ContrastiveLearningModule(
    base_model=model,
    embed_dim=512,
    use_momentum_encoder=True
)

# Training loop includes contrastive loss
outputs = model_with_cl(anchor_inputs, positive_inputs)
loss = outputs['contrastive_loss']
```

### Step 3: Build Ensemble

```python
from src.models.unified.movie_ensemble_system import MovieEnsembleRecommender

ensemble = MovieEnsembleRecommender(
    num_users=10000,
    num_movies=5000,
    fusion_strategy='attention',
    enable_collaborative_models=True,
    enable_sequential_models=True,
    enable_transformer_models=True
)

# Get recommendations from all models
results = ensemble.recommend_movies(user_id, candidates)
```

### Step 4: Add Context Awareness

```python
from src.models.unified.context_aware import ContextAwareRecommender

context_model = ContextAwareRecommender(
    base_recommender=ensemble,
    use_temporal=True,
    use_device=True,
    use_social=True
)

# Context-aware recommendations
recs, scores = context_model.recommend(user_id, candidates, context_data)
```

### Step 5: Optimize for Production

```python
from src.production.optimizations import ModelQuantizer, FastRecommender

# Quantize for speed
quantized = ModelQuantizer.dynamic_quantization(model)

# Fast serving with caching
fast_rec = FastRecommender(quantized, cache, device, use_ann=True)
fast_rec.precompute_item_embeddings(item_loader)

# Sub-millisecond recommendations
recs, scores = fast_rec.recommend(user_id, user_features, top_k=10)
```

### Step 6: Comprehensive Evaluation

```python
from src.evaluation.comprehensive_metrics import ComprehensiveEvaluator

evaluator = ComprehensiveEvaluator()
results = evaluator.evaluate(
    predictions=predictions,
    ground_truth=test_data,
    item_features=features,
    item_popularity=popularity,
    k_values=[5, 10, 20]
)

evaluator.print_results(results)
```

## 📈 Expected Improvements

Based on the improvements, you can expect:

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| NDCG@10 | Baseline | +15-25% | Better ranking |
| Cold-start NDCG | Baseline | +30-50% | Cross-domain transfer |
| Diversity@10 | Baseline | +20-30% | Ensemble variety |
| Inference Speed | Baseline | 4-6x faster | Quantization + caching |
| Model Size | 100% | 25% | Quantization |
| Catalog Coverage | Baseline | +10-20% | Fairness-aware |

## 🚀 Migration Path

### Phase 1: Foundation (Week 1-2)
1. ✅ Implement unified cross-domain embeddings
2. ✅ Add contrastive learning to best models
3. ✅ Set up comprehensive evaluation

### Phase 2: Integration (Week 3-4)
1. ✅ Build movie ensemble system
2. ✅ Add multimodal features
3. ✅ Implement context-aware layer

### Phase 3: Optimization (Week 5-6)
1. ✅ Advanced training framework
2. ✅ Production optimizations
3. ✅ Deployment pipeline

### Phase 4: Evaluation & Tuning (Week 7-8)
1. A/B testing framework
2. Hyperparameter optimization
3. Performance monitoring

## 📚 Key Takeaways

1. **Unified > Separate:** Cross-domain embeddings beat separate models
2. **Ensemble > Single:** Combining models gives best results
3. **Context Matters:** Same user wants different things at different times
4. **Multimodal > ID-only:** Content features help cold-start
5. **Contrastive Learning:** Better embeddings from unlabeled data
6. **Production-Ready:** Quantization + caching = fast serving

## 🔗 Architecture Diagram

```
User Input
    ↓
[Context Encoder] ← (time, device, social, mood)
    ↓
[Unified User Embedding] ← (movies + TV knowledge)
    ↓
[Contrastive Learning] ← (better representations)
    ↓
┌──────────────────────────┐
│   Ensemble System        │
│  ┌─────────────────────┐ │
│  │ Collaborative (4)   │ │
│  │ Two-Tower (4+)      │ │
│  │ Sequential (4)      │ │
│  │ Transformer (2)     │ │
│  └─────────────────────┘ │
│          ↓               │
│  [Adaptive Weighting]   │
│  [Uncertainty Est.]     │
└──────────────────────────┘
    ↓
[Multimodal Features] ← (text, image, audio)
    ↓
[Context-Aware Fusion]
    ↓
[Production Optimization] ← (quantization, caching, ANN)
    ↓
Top-K Recommendations
```

## 🎯 Next Steps

1. **Immediate:**
   - Train unified cross-domain model
   - Build ensemble with existing models
   - Add contrastive learning

2. **Short-term:**
   - Integrate multimodal features
   - Implement context-awareness
   - Set up comprehensive evaluation

3. **Medium-term:**
   - Deploy with production optimizations
   - A/B test against current system
   - Monitor and iterate

4. **Long-term:**
   - Reinforcement learning for long-term engagement
   - Explainability (why this recommendation?)
   - Personalized UI adaptation

## 📞 Support

For questions or issues:
- Check individual module docstrings
- Review example usage in each file's `__main__` section
- See integration tests in `/tests` directory

---

**Built with ❤️ for CineSync v2**
