# 🚀 Quick Wins - Start Here!

**3-Day Sprint for Immediate 15-20% Performance Boost**

---

## Day 1: Learning Rate Optimization

### What to do:
Add OneCycle learning rate scheduler to all models

### Where:
All `**/train_with_wandb.py` files

### Code:
```python
from torch.optim.lr_scheduler import OneCycleLR

scheduler = OneCycleLR(
    optimizer,
    max_lr=0.001,
    epochs=num_epochs,
    steps_per_epoch=len(train_loader),
    pct_start=0.1,  # 10% warmup
    anneal_strategy='cos'
)

# In training loop after optimizer.step():
scheduler.step()
```

### Expected Impact:
- 5-10% performance improvement
- Faster convergence
- Better final performance

---

## Day 2: Hard Negative Mining

### What to do:
Sample difficult negatives instead of random ones

### Where:
- `neural_collaborative_filtering/train_with_wandb.py`
- `two_tower_model/train_with_wandb.py`

### Code:
```python
def get_hard_negatives(model, user_emb, neg_items, k=5):
    """Sample negatives with high predicted scores"""
    with torch.no_grad():
        scores = model.score(user_emb, neg_items)
        hard_idx = torch.topk(scores, k).indices
    return neg_items[hard_idx]

# In training loop:
neg_items = get_hard_negatives(model, user_emb, all_items, k=5)
```

### Expected Impact:
- 10-15% improvement in ranking metrics
- Better discrimination between items
- Faster learning of subtle differences

---

## Day 3: Mixed Precision Training

### What to do:
Use FP16 for 2x faster training with same accuracy

### Where:
All training scripts

### Code:
```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for batch in dataloader:
    optimizer.zero_grad()

    with autocast():
        predictions = model(batch)
        loss = criterion(predictions, targets)

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

### Expected Impact:
- 30-50% faster training
- 40% less GPU memory
- Can use larger batch sizes

---

## Bonus: Better Metrics (30 minutes)

### What to do:
Track diversity and novelty alongside accuracy

### Code:
```python
def diversity_at_k(recommendations, k=10):
    """Average pairwise distance in recommendations"""
    pairs = 0
    diversity = 0
    for i in range(len(recommendations)):
        for j in range(i+1, len(recommendations)):
            diversity += 1 - cosine_similarity(
                item_features[recommendations[i]],
                item_features[recommendations[j]]
            )
            pairs += 1
    return diversity / pairs

def novelty_at_k(recommendations, item_popularity, k=10):
    """How unexpected are recommendations"""
    return -np.log2(item_popularity[recommendations[:k]]).mean()
```

### Track in WandB:
```python
wandb.log({
    'ndcg@10': ndcg,
    'diversity@10': diversity,
    'novelty@10': novelty,
    'coverage': catalog_coverage
})
```

---

## Files to Modify

### Priority 1 (All training files):
```
neural_collaborative_filtering/train_with_wandb.py
sequential_models/train_with_wandb.py
two_tower_model/train_with_wandb.py
hybrid_recommendation_movie/hybrid_recommendation/train_with_wandb.py
hybrid_recommendation_tv/hybrid_recommendation/train_with_wandb.py
```

### Priority 2 (Data loaders for hard negatives):
```
neural_collaborative_filtering/src/data_loader.py
two_tower_model/src/data_loader.py
```

---

## Expected Results After 3 Days

### Before:
- NDCG@10: 0.280
- Training time: 4 hours
- GPU memory: 16GB

### After:
- NDCG@10: **0.320-0.330** (15-18% improvement)
- Training time: **2 hours** (50% faster)
- GPU memory: **10GB** (38% reduction)
- More diverse recommendations
- Better long-tail coverage

---

## Next Steps After Quick Wins

Once you complete these, move to:
1. **Flash Attention** for transformer models (Week 2)
2. **FAISS Retrieval** for 100x faster inference (Week 3)
3. **Multi-task Learning** for 10-15% additional gains (Week 4)

See `MODEL_IMPROVEMENT_PLAN.md` for the complete roadmap.

---

## Testing Your Changes

```bash
# Before changes
python neural_collaborative_filtering/train_with_wandb.py --epochs 10 --tag baseline

# After changes
python neural_collaborative_filtering/train_with_wandb.py --epochs 10 --tag optimized

# Compare in WandB dashboard
# Look for: NDCG@10, training speed, memory usage
```

---

**Time Investment**: 3 days
**Performance Gain**: 15-20%
**Risk**: Low (well-tested techniques)
**Difficulty**: Easy

🎯 **Start with Day 1, see results immediately!**
