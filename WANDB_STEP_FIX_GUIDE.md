
# WandB Step Management Best Practices for CineSync v2

## Problem Summary
The step monotonicity warnings occurred because multiple components were managing wandb steps independently, causing conflicts where newer steps had lower values than previous ones.

## Root Causes Fixed:
1. **Multiple step counters** in WandbTrainingLogger
2. **Manual step assignment** in training scripts
3. **Step arithmetic** (step + 1) in multiple places
4. **Concurrent logging** without synchronization

## Solutions Implemented:

### 1. WandbTrainingLogger (wandb_training_integration.py)
✅ **FIXED:** Removed manual step increments
✅ **FIXED:** Use automatic wandb stepping
✅ **FIXED:** Added step monotonicity validation

```python
# OLD (PROBLEMATIC):
self.wandb_manager.log_metrics(log_dict, step=self.global_step, commit=False)
self.global_step += 1  # Multiple increments per epoch

# NEW (FIXED):
self.wandb_manager.log_metrics(log_dict, commit=False)  # Auto-increment
```

### 2. WandbManager (wandb_config.py)
✅ **FIXED:** Added step validation in log_metrics()
✅ **FIXED:** Fallback to automatic stepping when conflicts detected
✅ **FIXED:** Better error handling and logging

```python
# Added step monotonicity checking:
if step is not None and hasattr(wandb.run, 'step') and wandb.run.step is not None:
    current_wandb_step = wandb.run.step
    if step <= current_wandb_step:
        wandb.log(metrics, commit=commit)  # Use auto-increment
        return
```

### 3. Training Scripts
✅ **FIXED:** neural_collaborative_filtering/src/train.py
✅ **FIXED:** two_tower_model/src/train.py  
✅ **FIXED:** sequential_models/src/train.py

```python
# OLD (PROBLEMATIC):
final_step = wandb.run.step if wandb.run and hasattr(wandb.run, 'step') else 0
wandb.log(all_metrics, step=final_step)
wandb.log({"training_history": history}, step=final_step + 1)

# NEW (FIXED):
wandb.log(all_metrics)              # Auto-increment
wandb.log({"training_history": history})  # Auto-increment
```

## Best Practices Going Forward:

### ✅ DO:
- Let wandb handle step increments automatically
- Use `wandb.log(metrics)` without step parameter
- Use `commit=True` only for epoch-end logging
- Use `commit=False` for batch-level logging

### ❌ DON'T:
- Manually assign step values with `step=`
- Read and increment `wandb.run.step`
- Use step arithmetic (`step + 1`)
- Create multiple step counters

### 🔧 Example Correct Usage:

```python
# Batch logging (don't commit each batch)
for batch_idx, batch in enumerate(train_loader):
    loss = train_step(batch)
    wandb.log({'batch_loss': loss}, commit=False)

# Epoch logging (commit epoch summary)
epoch_metrics = {'epoch': epoch, 'train_loss': avg_loss}
wandb.log(epoch_metrics, commit=True)  # This increments the step
```

### 🎯 Key Principle:
**One logical step per epoch** - let wandb auto-increment, don't manage steps manually.

## Testing the Fix:
Run training with wandb enabled and verify no step warnings appear:
```bash
python neural_collaborative_filtering/src/train.py --use_wandb --epochs 5
```

## Monitoring:
- Check wandb dashboard for smooth step progression
- Look for absence of "step monotonicity" warnings
- Verify all metrics are logged correctly
