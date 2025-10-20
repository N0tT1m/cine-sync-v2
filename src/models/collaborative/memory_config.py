#!/usr/bin/env python3
"""
Memory optimization configuration for Neural Collaborative Filtering
Optimized for low GPU memory (1.4GB) environments
"""

import torch
import gc
from typing import Dict, Any


class MemoryOptimizer:
    """Memory optimization utilities for NCF training"""
    
    @staticmethod
    def get_optimized_config() -> Dict[str, Any]:
        """Get performance-optimized configuration for RTX 4090 + Ryzen 9 3900X"""
        return {
            'batch_size': 16384,  # Large batch for RTX 4090 24GB VRAM
            'num_workers': 24,    # Full thread utilization (24 threads)
            'embedding_dim': 128,  # Larger embeddings for better quality
            'hidden_layers': [512, 256, 128],  # Deeper network
            'accumulation_steps': 1,  # No accumulation needed with large batches
            'pin_memory': True,
            'persistent_workers': True,  # Keep workers alive for speed
            'prefetch_factor': 4,  # Aggressive prefetching
        }
    
    @staticmethod
    def setup_performance_training(device: torch.device):
        """Configure PyTorch for maximum performance on RTX 4090"""
        if device.type == 'cuda':
            # Clear GPU cache
            torch.cuda.empty_cache()
            
            # Enable performance optimizations
            torch.backends.cudnn.benchmark = True  # Optimize for speed
            torch.backends.cudnn.enabled = True
            torch.backends.cuda.matmul.allow_tf32 = True  # Use Tensor Cores
            torch.backends.cudnn.allow_tf32 = True
            
            # Use almost all VRAM (RTX 4090 has 24GB)
            torch.cuda.set_per_process_memory_fraction(0.95)
            
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
            print(f"CUDA Cores: {torch.cuda.get_device_properties(0).multi_processor_count}")
            print(f"TensorCore optimizations: ENABLED")
    
    @staticmethod
    def cleanup_memory():
        """Clean up memory during training"""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    @staticmethod
    def get_dynamic_batch_size(model: torch.nn.Module, device: torch.device, 
                             max_memory_gb: float = 1.2) -> int:
        """Dynamically determine optimal batch size for available memory"""
        if device.type != 'cuda':
            return 512
            
        # Start with small batch and test
        batch_sizes = [64, 128, 256, 512, 1024]
        optimal_batch = 64
        
        for batch_size in batch_sizes:
            try:
                # Test forward pass
                test_user = torch.randint(0, 1000, (batch_size,)).to(device)
                test_item = torch.randint(0, 1000, (batch_size,)).to(device)
                
                with torch.no_grad():
                    _ = model(test_user, test_item)
                
                memory_used = torch.cuda.memory_allocated() / 1024**3
                if memory_used < max_memory_gb:
                    optimal_batch = batch_size
                else:
                    break
                    
            except RuntimeError as e:
                if "out of memory" in str(e):
                    break
                raise
            finally:
                torch.cuda.empty_cache()
        
        return optimal_batch


def apply_memory_optimizations():
    """Apply all memory optimizations"""
    # Reduce memory usage
    torch.set_default_dtype(torch.float32)  # Use FP32 instead of double
    
    # Enable garbage collection
    gc.enable()
    
    print("Memory optimizations applied")