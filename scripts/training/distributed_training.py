#!/usr/bin/env python3
"""
Distributed Training Support for CineSync v2
Multi-GPU and distributed training optimizations for faster model training
"""

import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from typing import Optional, Callable, Any, Dict
import logging
from contextlib import contextmanager

logger = logging.getLogger(__name__)


class DistributedTrainingManager:
    """
    Manager for distributed training across multiple GPUs
    Supports both single-node multi-GPU and multi-node setups
    """
    
    def __init__(self, backend: str = 'nccl'):
        self.backend = backend
        self.world_size = None
        self.rank = None
        self.local_rank = None
        self.is_distributed = False
        self.device = None
        
    def setup_distributed(self, rank: int, world_size: int, 
                         master_addr: str = 'localhost', 
                         master_port: str = '12355'):
        """
        Setup distributed training environment
        
        Args:
            rank: Process rank
            world_size: Total number of processes
            master_addr: Master node address
            master_port: Master node port
        """
        os.environ['MASTER_ADDR'] = master_addr
        os.environ['MASTER_PORT'] = master_port
        
        # Initialize process group
        dist.init_process_group(
            backend=self.backend,
            rank=rank,
            world_size=world_size
        )
        
        self.rank = rank
        self.world_size = world_size
        self.local_rank = rank % torch.cuda.device_count()
        self.is_distributed = True
        
        # Set device for this process
        torch.cuda.set_device(self.local_rank)
        self.device = torch.device(f'cuda:{self.local_rank}')
        
        logger.info(f"Distributed training setup: rank={rank}, "
                   f"world_size={world_size}, local_rank={self.local_rank}")
    
    def wrap_model(self, model: torch.nn.Module, 
                   find_unused_parameters: bool = False) -> torch.nn.Module:
        """
        Wrap model for distributed training
        
        Args:
            model: PyTorch model
            find_unused_parameters: Whether to find unused parameters
            
        Returns:
            DDP-wrapped model
        """
        if not self.is_distributed:
            return model
        
        model = model.to(self.device)
        
        # Wrap with DistributedDataParallel
        ddp_model = DDP(
            model,
            device_ids=[self.local_rank],
            output_device=self.local_rank,
            find_unused_parameters=find_unused_parameters
        )
        
        logger.info("Model wrapped with DistributedDataParallel")
        return ddp_model
    
    def create_distributed_dataloader(self, dataset, batch_size: int,
                                    shuffle: bool = True, **kwargs) -> DataLoader:
        """
        Create distributed data loader with proper sampling
        
        Args:
            dataset: PyTorch dataset
            batch_size: Batch size per GPU
            shuffle: Whether to shuffle data
            **kwargs: Additional DataLoader arguments
            
        Returns:
            Distributed DataLoader
        """
        # Create distributed sampler
        sampler = DistributedSampler(
            dataset,
            num_replicas=self.world_size,
            rank=self.rank,
            shuffle=shuffle
        ) if self.is_distributed else None
        
        # Adjust batch size for distributed training
        # Each GPU gets batch_size samples, so effective batch size is batch_size * world_size
        
        # Create data loader
        dataloader_kwargs = {
            'batch_size': batch_size,
            'sampler': sampler,
            'shuffle': shuffle if sampler is None else False,
            'pin_memory': True,
            'num_workers': kwargs.get('num_workers', 4),
            'persistent_workers': kwargs.get('num_workers', 4) > 0,
            'prefetch_factor': 2,
            'drop_last': True
        }
        
        # Override with user kwargs
        dataloader_kwargs.update(kwargs)
        
        # Remove conflicting arguments
        if sampler is not None:
            dataloader_kwargs.pop('shuffle', None)
        
        return DataLoader(dataset, **dataloader_kwargs)
    
    def reduce_tensor(self, tensor: torch.Tensor, average: bool = True) -> torch.Tensor:
        """
        Reduce tensor across all processes
        
        Args:
            tensor: Tensor to reduce
            average: Whether to average or sum
            
        Returns:
            Reduced tensor
        """
        if not self.is_distributed:
            return tensor
        
        # Clone tensor to avoid modifying original
        reduced_tensor = tensor.clone()
        
        # All-reduce across processes
        dist.all_reduce(reduced_tensor, op=dist.ReduceOp.SUM)
        
        if average:
            reduced_tensor /= self.world_size
        
        return reduced_tensor
    
    def barrier(self):
        """Synchronization barrier"""
        if self.is_distributed:
            dist.barrier()
    
    def is_main_process(self) -> bool:
        """Check if this is the main process (rank 0)"""
        return not self.is_distributed or self.rank == 0
    
    def cleanup(self):
        """Cleanup distributed training"""
        if self.is_distributed:
            dist.destroy_process_group()
            logger.info("Distributed training cleanup completed")


class MultiGPUTrainingWrapper:
    """
    Wrapper for easy multi-GPU training without full distributed setup
    Uses DataParallel for simpler single-node multi-GPU training
    """
    
    def __init__(self, model: torch.nn.Module, device_ids: Optional[list] = None):
        self.model = model
        self.device_ids = device_ids or list(range(torch.cuda.device_count()))
        self.is_multi_gpu = len(self.device_ids) > 1
        
        if self.is_multi_gpu and torch.cuda.device_count() > 1:
            self.model = torch.nn.DataParallel(model, device_ids=self.device_ids)
            logger.info(f"Model wrapped with DataParallel on GPUs: {self.device_ids}")
        else:
            logger.info("Single GPU training")
    
    def get_model(self) -> torch.nn.Module:
        """Get the underlying model"""
        return self.model.module if self.is_multi_gpu else self.model
    
    def forward(self, *args, **kwargs):
        """Forward pass"""
        return self.model(*args, **kwargs)
    
    def train(self):
        """Set model to training mode"""
        self.model.train()
    
    def eval(self):
        """Set model to evaluation mode"""
        self.model.eval()


def launch_distributed_training(training_fn: Callable, 
                               world_size: int = None,
                               **kwargs) -> None:
    """
    Launch distributed training across multiple GPUs
    
    Args:
        training_fn: Training function to execute on each process
        world_size: Number of processes (defaults to number of GPUs)
        **kwargs: Additional arguments for training function
    """
    if world_size is None:
        world_size = torch.cuda.device_count()
    
    if world_size <= 1:
        logger.warning("Only 1 GPU available, running single-GPU training")
        training_fn(rank=0, world_size=1, **kwargs)
        return
    
    logger.info(f"Launching distributed training on {world_size} GPUs")
    
    # Use spawn method for better compatibility
    mp.spawn(
        training_fn,
        args=(world_size, kwargs),
        nprocs=world_size,
        join=True
    )


def distributed_training_wrapper(rank: int, world_size: int, 
                                training_args: Dict[str, Any],
                                model_fn: Callable,
                                train_fn: Callable) -> None:
    """
    Wrapper function for distributed training process
    
    Args:
        rank: Process rank
        world_size: Total number of processes
        training_args: Training arguments
        model_fn: Function that creates the model
        train_fn: Function that performs training
    """
    # Setup distributed training
    dist_manager = DistributedTrainingManager()
    dist_manager.setup_distributed(rank, world_size)
    
    try:
        # Create model
        model = model_fn(**training_args.get('model_args', {}))
        
        # Wrap model for distributed training
        model = dist_manager.wrap_model(model)
        
        # Run training
        train_fn(
            model=model,
            dist_manager=dist_manager,
            rank=rank,
            world_size=world_size,
            **training_args.get('train_args', {})
        )
        
    except Exception as e:
        logger.error(f"Training failed on rank {rank}: {e}")
        raise
    finally:
        # Cleanup
        dist_manager.cleanup()


@contextmanager
def automatic_mixed_precision():
    """Context manager for automatic mixed precision"""
    if torch.cuda.is_available():
        scaler = torch.cuda.amp.GradScaler()
        yield scaler
    else:
        yield None


class GradientSynchronization:
    """
    Manual gradient synchronization for custom distributed training patterns
    """
    
    def __init__(self, model: torch.nn.Module, world_size: int):
        self.model = model
        self.world_size = world_size
    
    def sync_gradients(self):
        """Manually synchronize gradients across processes"""
        if not dist.is_initialized():
            return
        
        for param in self.model.parameters():
            if param.grad is not None:
                # All-reduce gradients
                dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
                param.grad /= self.world_size


def get_optimal_world_size() -> int:
    """
    Determine optimal world size based on available hardware
    
    Returns:
        Optimal number of processes for distributed training
    """
    num_gpus = torch.cuda.device_count()
    
    if num_gpus == 0:
        return 1
    elif num_gpus <= 4:
        # Use all available GPUs for small counts
        return num_gpus
    else:
        # For larger GPU counts, consider memory and communication overhead
        # Typically 8 GPUs is a good balance
        return min(num_gpus, 8)


def create_distributed_optimizer(model: torch.nn.Module, 
                                optimizer_class: type,
                                lr: float,
                                world_size: int = 1,
                                **optimizer_kwargs) -> torch.optim.Optimizer:
    """
    Create optimizer with learning rate scaling for distributed training
    
    Args:
        model: PyTorch model
        optimizer_class: Optimizer class (e.g., torch.optim.Adam)
        lr: Base learning rate
        world_size: Number of distributed processes
        **optimizer_kwargs: Additional optimizer arguments
        
    Returns:
        Configured optimizer
    """
    # Scale learning rate by world size for distributed training
    # This is a common practice to maintain effective learning rate
    scaled_lr = lr * world_size if world_size > 1 else lr
    
    optimizer = optimizer_class(
        model.parameters(),
        lr=scaled_lr,
        **optimizer_kwargs
    )
    
    logger.info(f"Created optimizer with scaled LR: {scaled_lr} "
               f"(base: {lr}, world_size: {world_size})")
    
    return optimizer


# Example usage functions
def example_distributed_ncf_training():
    """Example of how to use distributed training with NCF model"""
    
    def create_ncf_model(**kwargs):
        from neural_collaborative_filtering.src.model import NCFModel
        return NCFModel(**kwargs)
    
    def train_ncf(model, dist_manager, rank, world_size, **kwargs):
        # Training logic here
        logger.info(f"Training NCF on rank {rank}")
        # ... training implementation ...
    
    # Launch distributed training
    training_args = {
        'model_args': {
            'num_users': 1000,
            'num_items': 1000,
            'embedding_dim': 64,
            'hidden_layers': [128, 64],
            'dropout': 0.2,
            'alpha': 0.5
        },
        'train_args': {
            'epochs': 10,
            'batch_size': 256
        }
    }
    
    launch_distributed_training(
        lambda rank, world_size, args: distributed_training_wrapper(
            rank, world_size, args, create_ncf_model, train_ncf
        ),
        **training_args
    )


if __name__ == "__main__":
    # Test distributed training setup
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    print(f"Optimal world size: {get_optimal_world_size()}")
    
    if torch.cuda.device_count() > 1:
        print("Multi-GPU training available")
        print("Use launch_distributed_training() to start distributed training")
    else:
        print("Single GPU or CPU training only")