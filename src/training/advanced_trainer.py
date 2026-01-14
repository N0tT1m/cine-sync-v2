"""
Advanced Training Framework for CineSync v2
Multi-task learning, curriculum learning, and advanced optimization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from typing import Dict, List, Optional, Tuple, Callable
import logging
from pathlib import Path
from tqdm import tqdm
import numpy as np

logger = logging.getLogger(__name__)


class MultiTaskLoss(nn.Module):
    """
    Multi-task loss with uncertainty-based weighting.

    Based on "Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics"
    Learns task weights automatically based on task uncertainty.
    """

    def __init__(self, num_tasks: int, reduction: str = 'mean'):
        super().__init__()

        self.num_tasks = num_tasks
        self.reduction = reduction

        # Learnable uncertainty parameters (log variance)
        self.log_vars = nn.Parameter(torch.zeros(num_tasks))

    def forward(self, losses: List[torch.Tensor]) -> Tuple[torch.Tensor, Dict]:
        """
        Compute weighted multi-task loss.

        Args:
            losses: List of task-specific losses

        Returns:
            (total_loss, task_weights_dict)
        """
        total_loss = 0
        task_weights = {}

        for i, loss in enumerate(losses):
            # Weight = 1 / (2 * sigma^2), regularization = log(sigma)
            # where sigma = exp(log_var / 2)
            precision = torch.exp(-self.log_vars[i])
            weighted_loss = precision * loss + self.log_vars[i]

            total_loss += weighted_loss
            task_weights[f'task_{i}_weight'] = precision.item()

        if self.reduction == 'mean':
            total_loss = total_loss / self.num_tasks

        return total_loss, task_weights


class CurriculumScheduler:
    """
    Curriculum learning: start with easy examples, gradually increase difficulty.

    Strategies:
    - Difficulty based on loss (high loss = hard)
    - Difficulty based on confidence (low confidence = hard)
    - Difficulty based on sample characteristics
    """

    def __init__(
        self,
        strategy: str = 'loss_based',  # 'loss_based', 'confidence_based', 'random'
        start_percentile: float = 0.25,
        end_percentile: float = 1.0,
        num_epochs: int = 100
    ):
        self.strategy = strategy
        self.start_percentile = start_percentile
        self.end_percentile = end_percentile
        self.num_epochs = num_epochs

        self.current_epoch = 0
        self.sample_difficulties = None

    def update_difficulties(
        self,
        losses: torch.Tensor,
        confidences: Optional[torch.Tensor] = None
    ):
        """Update sample difficulties based on current performance"""
        if self.strategy == 'loss_based':
            self.sample_difficulties = losses.detach().cpu().numpy()
        elif self.strategy == 'confidence_based':
            if confidences is not None:
                # Lower confidence = harder
                self.sample_difficulties = (1 - confidences).detach().cpu().numpy()
            else:
                self.sample_difficulties = losses.detach().cpu().numpy()
        else:  # random
            self.sample_difficulties = np.random.rand(len(losses))

    def get_curriculum_mask(self, batch_size: int) -> torch.Tensor:
        """
        Get mask for which samples to include based on curriculum.

        Returns:
            Binary mask [batch_size]
        """
        if self.sample_difficulties is None:
            return torch.ones(batch_size, dtype=torch.bool)

        # Current percentile based on epoch
        progress = min(self.current_epoch / self.num_epochs, 1.0)
        current_percentile = (
            self.start_percentile +
            (self.end_percentile - self.start_percentile) * progress
        )

        # Threshold difficulty
        threshold = np.percentile(self.sample_difficulties, current_percentile * 100)

        # Include samples easier than threshold
        mask = self.sample_difficulties <= threshold

        return torch.from_numpy(mask)

    def step(self):
        """Move to next epoch"""
        self.current_epoch += 1


class AdvancedOptimizer:
    """
    Advanced optimization strategies.

    - Layer-wise learning rate decay (LLRD)
    - Gradient accumulation
    - Mixed precision training
    - Gradient clipping
    - Warmup + cosine decay scheduling
    """

    def __init__(
        self,
        model: nn.Module,
        base_lr: float = 1e-3,
        weight_decay: float = 0.01,
        llrd_factor: float = 0.95,
        use_amp: bool = True,
        gradient_accumulation_steps: int = 1,
        max_grad_norm: float = 1.0,
        warmup_steps: int = 1000,
        total_steps: int = 100000
    ):
        self.model = model
        self.use_amp = use_amp
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps

        # Create layer-wise learning rates
        param_groups = self._create_llrd_param_groups(model, base_lr, llrd_factor)

        # Optimizer (AdamW with weight decay)
        self.optimizer = torch.optim.AdamW(
            param_groups,
            lr=base_lr,
            weight_decay=weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8
        )

        # Scheduler (warmup + cosine decay)
        self.scheduler = self._create_scheduler()

        # Mixed precision scaler
        if use_amp:
            self.scaler = torch.cuda.amp.GradScaler()

        self.step_count = 0

    def _create_llrd_param_groups(
        self,
        model: nn.Module,
        base_lr: float,
        llrd_factor: float
    ) -> List[Dict]:
        """
        Create parameter groups with layer-wise learning rate decay.

        Later layers get higher learning rates, earlier layers get lower LRs.
        """
        # Get all layers
        layers = list(model.named_parameters())
        num_layers = len(layers)

        param_groups = []
        for i, (name, param) in enumerate(layers):
            # Compute layer-specific LR
            layer_lr = base_lr * (llrd_factor ** (num_layers - i - 1))

            param_groups.append({
                'params': [param],
                'lr': layer_lr,
                'name': name
            })

        return param_groups

    def _create_scheduler(self) -> _LRScheduler:
        """Create warmup + cosine decay scheduler"""
        def lr_lambda(step):
            if step < self.warmup_steps:
                # Warmup: linear increase
                return step / max(1, self.warmup_steps)
            else:
                # Cosine decay
                progress = (step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
                return 0.5 * (1 + np.cos(np.pi * progress))

        return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

    def step(self, loss: torch.Tensor):
        """Optimization step with gradient accumulation and clipping"""
        # Scale loss for gradient accumulation
        loss = loss / self.gradient_accumulation_steps

        if self.use_amp:
            # Mixed precision backward
            self.scaler.scale(loss).backward()

            if (self.step_count + 1) % self.gradient_accumulation_steps == 0:
                # Unscale gradients for clipping
                self.scaler.unscale_(self.optimizer)

                # Clip gradients
                if self.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.max_grad_norm
                    )

                # Optimizer step
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

                # Scheduler step
                self.scheduler.step()
        else:
            # Regular backward
            loss.backward()

            if (self.step_count + 1) % self.gradient_accumulation_steps == 0:
                # Clip gradients
                if self.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.max_grad_norm
                    )

                # Optimizer step
                self.optimizer.step()
                self.optimizer.zero_grad()

                # Scheduler step
                self.scheduler.step()

        self.step_count += 1

    def get_lr(self) -> float:
        """Get current learning rate"""
        return self.optimizer.param_groups[0]['lr']


class AdvancedTrainer:
    """
    Complete advanced training framework.

    Features:
    - Multi-task learning with uncertainty weighting
    - Curriculum learning
    - Advanced optimization (LLRD, gradient accumulation, mixed precision)
    - Early stopping
    - Model checkpointing
    - Logging and monitoring
    """

    def __init__(
        self,
        model: nn.Module,
        tasks: List[str],
        device: torch.device,
        output_dir: Path,
        base_lr: float = 1e-3,
        weight_decay: float = 0.01,
        use_curriculum: bool = True,
        use_multi_task: bool = True,
        use_amp: bool = True,
        gradient_accumulation_steps: int = 1,
        max_grad_norm: float = 1.0,
        warmup_steps: int = 1000,
        total_steps: int = 100000,
        eval_steps: int = 1000,
        save_steps: int = 5000,
        early_stopping_patience: int = 5,
        logging_steps: int = 100
    ):
        self.model = model.to(device)
        self.tasks = tasks
        self.device = device
        self.output_dir = output_dir
        self.eval_steps = eval_steps
        self.save_steps = save_steps
        self.early_stopping_patience = early_stopping_patience
        self.logging_steps = logging_steps

        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)

        # Multi-task loss
        if use_multi_task:
            self.multi_task_loss = MultiTaskLoss(len(tasks)).to(device)
        else:
            self.multi_task_loss = None

        # Curriculum scheduler
        if use_curriculum:
            self.curriculum = CurriculumScheduler(
                strategy='loss_based',
                num_epochs=total_steps // 1000  # Approximate epochs
            )
        else:
            self.curriculum = None

        # Optimizer
        self.optimizer_wrapper = AdvancedOptimizer(
            model=model,
            base_lr=base_lr,
            weight_decay=weight_decay,
            use_amp=use_amp,
            gradient_accumulation_steps=gradient_accumulation_steps,
            max_grad_norm=max_grad_norm,
            warmup_steps=warmup_steps,
            total_steps=total_steps
        )

        # Early stopping
        self.best_metric = float('-inf')
        self.patience_counter = 0

        # Training state
        self.global_step = 0
        self.epoch = 0
        self.training_history = []

    def train_step(
        self,
        batch: Dict[str, torch.Tensor],
        task_loss_fns: Dict[str, Callable]
    ) -> Dict[str, float]:
        """
        Single training step.

        Args:
            batch: Batch of training data
            task_loss_fns: Dict of {task_name: loss_function}

        Returns:
            Dictionary of losses
        """
        self.model.train()

        # Move batch to device
        batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()}

        # Forward pass
        outputs = self.model(**batch)

        # Compute task-specific losses
        task_losses = []
        loss_dict = {}

        for task_name in self.tasks:
            if task_name in task_loss_fns and task_name in outputs:
                loss = task_loss_fns[task_name](
                    outputs[task_name],
                    batch.get(f'{task_name}_labels')
                )
                task_losses.append(loss)
                loss_dict[f'{task_name}_loss'] = loss.item()

        # Multi-task loss
        if self.multi_task_loss and len(task_losses) > 1:
            total_loss, task_weights = self.multi_task_loss(task_losses)
            loss_dict.update(task_weights)
        else:
            total_loss = sum(task_losses) / len(task_losses)

        loss_dict['total_loss'] = total_loss.item()

        # Curriculum learning
        if self.curriculum:
            # Update difficulties based on sample losses
            sample_losses = task_losses[0].detach()  # Use first task
            self.curriculum.update_difficulties(sample_losses)

        # Optimization step
        self.optimizer_wrapper.step(total_loss)

        loss_dict['learning_rate'] = self.optimizer_wrapper.get_lr()

        return loss_dict

    def evaluate(
        self,
        eval_dataloader,
        task_metric_fns: Dict[str, Callable]
    ) -> Dict[str, float]:
        """
        Evaluation on validation set.

        Args:
            eval_dataloader: Validation data loader
            task_metric_fns: Dict of {task_name: metric_function}

        Returns:
            Dictionary of metrics
        """
        self.model.eval()

        all_outputs = {task: [] for task in self.tasks}
        all_labels = {task: [] for task in self.tasks}

        with torch.no_grad():
            for batch in tqdm(eval_dataloader, desc="Evaluating"):
                # Move batch to device
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                        for k, v in batch.items()}

                # Forward pass
                outputs = self.model(**batch)

                # Collect outputs and labels
                for task_name in self.tasks:
                    if task_name in outputs:
                        all_outputs[task_name].append(outputs[task_name].cpu())
                        if f'{task_name}_labels' in batch:
                            all_labels[task_name].append(batch[f'{task_name}_labels'].cpu())

        # Compute metrics
        metrics = {}
        for task_name in self.tasks:
            if task_name in task_metric_fns and all_outputs[task_name]:
                predictions = torch.cat(all_outputs[task_name], dim=0)
                labels = torch.cat(all_labels[task_name], dim=0) if all_labels[task_name] else None

                if labels is not None:
                    metric = task_metric_fns[task_name](predictions, labels)
                    metrics[f'{task_name}_metric'] = metric

        return metrics

    def train(
        self,
        train_dataloader,
        eval_dataloader,
        task_loss_fns: Dict[str, Callable],
        task_metric_fns: Dict[str, Callable],
        num_epochs: int = 10
    ):
        """
        Complete training loop.

        Args:
            train_dataloader: Training data loader
            eval_dataloader: Validation data loader
            task_loss_fns: Task-specific loss functions
            task_metric_fns: Task-specific metric functions
            num_epochs: Number of epochs to train
        """
        logger.info(f"Starting training for {num_epochs} epochs")

        for epoch in range(num_epochs):
            self.epoch = epoch

            # Training epoch
            epoch_losses = []
            pbar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")

            for batch in pbar:
                # Training step
                loss_dict = self.train_step(batch, task_loss_fns)
                epoch_losses.append(loss_dict['total_loss'])

                self.global_step += 1

                # Logging
                if self.global_step % self.logging_steps == 0:
                    pbar.set_postfix({
                        'loss': f"{loss_dict['total_loss']:.4f}",
                        'lr': f"{loss_dict['learning_rate']:.2e}"
                    })

                # Evaluation
                if self.global_step % self.eval_steps == 0:
                    metrics = self.evaluate(eval_dataloader, task_metric_fns)
                    logger.info(f"Step {self.global_step} - Metrics: {metrics}")

                    # Early stopping
                    avg_metric = np.mean(list(metrics.values()))
                    if avg_metric > self.best_metric:
                        self.best_metric = avg_metric
                        self.patience_counter = 0
                        self.save_checkpoint('best_model.pt')
                    else:
                        self.patience_counter += 1

                    if self.patience_counter >= self.early_stopping_patience:
                        logger.info("Early stopping triggered")
                        return

                # Checkpointing
                if self.global_step % self.save_steps == 0:
                    self.save_checkpoint(f'checkpoint_step_{self.global_step}.pt')

            # Epoch summary
            avg_epoch_loss = np.mean(epoch_losses)
            logger.info(f"Epoch {epoch+1} - Average Loss: {avg_epoch_loss:.4f}")

            # Curriculum update
            if self.curriculum:
                self.curriculum.step()

        logger.info("Training completed!")

    def save_checkpoint(self, filename: str):
        """Save model checkpoint"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer_wrapper.optimizer.state_dict(),
            'global_step': self.global_step,
            'epoch': self.epoch,
            'best_metric': self.best_metric
        }

        if self.multi_task_loss:
            checkpoint['multi_task_loss_state_dict'] = self.multi_task_loss.state_dict()

        save_path = self.output_dir / filename
        torch.save(checkpoint, save_path)
        logger.info(f"Checkpoint saved to {save_path}")

    def load_checkpoint(self, filename: str):
        """Load model checkpoint"""
        load_path = self.output_dir / filename
        checkpoint = torch.load(load_path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer_wrapper.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.global_step = checkpoint['global_step']
        self.epoch = checkpoint['epoch']
        self.best_metric = checkpoint['best_metric']

        if self.multi_task_loss and 'multi_task_loss_state_dict' in checkpoint:
            self.multi_task_loss.load_state_dict(checkpoint['multi_task_loss_state_dict'])

        logger.info(f"Checkpoint loaded from {load_path}")


# Example usage
if __name__ == "__main__":
    from pathlib import Path

    # Dummy model
    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(128, 64)
            self.task1_head = nn.Linear(64, 1)
            self.task2_head = nn.Linear(64, 10)

        def forward(self, features, **kwargs):
            x = self.fc(features)
            return {
                'task1': self.task1_head(x),
                'task2': self.task2_head(x)
            }

    model = DummyModel()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Trainer
    trainer = AdvancedTrainer(
        model=model,
        tasks=['task1', 'task2'],
        device=device,
        output_dir=Path('./checkpoints'),
        base_lr=1e-3,
        use_multi_task=True,
        use_curriculum=True,
        use_amp=True
    )

    print("Advanced trainer initialized successfully!")
