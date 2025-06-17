#!/usr/bin/env python3
"""
High-performance configuration for Neural Collaborative Filtering
Optimized for RTX 4090 (24GB VRAM) + Ryzen 9 3900X (24 threads)
"""

import torch
import torch.nn as nn
from typing import Dict, Any
import multiprocessing as mp


class PerformanceOptimizer:
    """Maximum performance utilities for high-end hardware"""
    
    @staticmethod
    def get_rtx4090_config() -> Dict[str, Any]:
        """Get maximum performance configuration for RTX 4090 + Ryzen 9 3900X"""
        return {
            # Data loading optimizations
            'batch_size': 20480,  # Massive batch size for 24GB VRAM
            'num_workers': 24,    # Full CPU thread utilization 
            'pin_memory': True,
            'persistent_workers': True,
            'prefetch_factor': 6,  # Aggressive prefetching
            'drop_last': True,     # Consistent batch sizes for optimal GPU utilization
            
            # Model architecture for better quality
            'embedding_dim': 256,  # Large embeddings
            'hidden_layers': [1024, 512, 256, 128],  # Deep network
            'dropout': 0.1,        # Lower dropout for large dataset
            
            # Training optimizations
            'learning_rate': 0.003,  # Higher LR for large batches
            'weight_decay': 1e-6,    # Lower weight decay
            'accumulation_steps': 1,  # No accumulation needed
            
            # PyTorch optimizations
            'compile_model': True,    # torch.compile for 20-30% speedup
            'channels_last': True,    # Memory layout optimization
            'mixed_precision': True,  # FP16 for 2x throughput
        }
    
    @staticmethod
    def setup_maximum_performance():
        """Configure PyTorch for absolute maximum performance"""
        # Enable all CUDA optimizations
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        
        # Enable additional optimizations
        torch._C._set_backcompat_broadcast_warn(False)
        torch._C._set_backcompat_keepdim_warn(False)
        
        # Set number of threads for CPU operations
        torch.set_num_threads(24)  # Match CPU thread count
        torch.set_num_interop_threads(4)
        
        print("🚀 MAXIMUM PERFORMANCE MODE ACTIVATED")
        print(f"   Torch Threads: {torch.get_num_threads()}")
        print(f"   Interop Threads: {torch.get_num_interop_threads()}")
        print("   TensorCore: ENABLED")
        print("   CuDNN Benchmark: ENABLED")
        print("   TF32: ENABLED")
    
    @staticmethod
    def optimize_model_for_speed(model: nn.Module) -> nn.Module:
        """Apply model-level optimizations for maximum speed"""
        # Compile model for significant speedup (PyTorch 2.0+)
        try:
            model = torch.compile(model, mode='max-autotune')
            print("✅ Model compiled with torch.compile (max-autotune)")
        except Exception as e:
            print(f"⚠️  torch.compile failed: {e}")
        
        # Set to channels_last memory format if applicable
        try:
            model = model.to(memory_format=torch.channels_last)
            print("✅ Model set to channels_last memory format")
        except:
            pass  # Not all models support this
        
        return model
    
    @staticmethod
    def get_optimal_batch_size_rtx4090(model: nn.Module, device: torch.device) -> int:
        """Find maximum stable batch size for RTX 4090"""
        if device.type != 'cuda':
            return 1024
        
        # Test increasingly large batch sizes
        batch_sizes = [4096, 8192, 12288, 16384, 20480, 24576, 28672, 32768]
        optimal_batch = 4096
        
        model.eval()
        
        for batch_size in batch_sizes:
            try:
                # Test memory usage with dummy data
                test_user = torch.randint(0, 10000, (batch_size,)).to(device)
                test_item = torch.randint(0, 10000, (batch_size,)).to(device)
                
                torch.cuda.empty_cache()
                with torch.no_grad():
                    _ = model(test_user, test_item)
                
                # Check memory usage
                memory_used = torch.cuda.memory_allocated() / 1024**3
                memory_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
                memory_percent = memory_used / memory_total
                
                if memory_percent < 0.90:  # Keep under 90% VRAM usage
                    optimal_batch = batch_size
                    print(f"✅ Batch {batch_size}: {memory_used:.1f}GB/{memory_total:.1f}GB ({memory_percent:.1%})")
                else:
                    print(f"❌ Batch {batch_size}: Would exceed 90% VRAM")
                    break
                    
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"❌ Batch {batch_size}: OOM")
                    break
                raise
            finally:
                torch.cuda.empty_cache()
        
        print(f"🎯 Optimal batch size: {optimal_batch}")
        return optimal_batch
    
    @staticmethod
    def enable_aggressive_caching():
        """Enable aggressive memory and compute caching"""
        # Set memory pool to be more aggressive
        if torch.cuda.is_available():
            torch.cuda.set_per_process_memory_fraction(0.95)  # Use 95% of VRAM
            
            # Enable memory pooling for faster allocation/deallocation
            torch.cuda.empty_cache()
            
            print("🔥 Aggressive memory caching enabled (95% VRAM)")


def apply_rtx4090_optimizations():
    """Apply all RTX 4090 + Ryzen 9 3900X optimizations"""
    PerformanceOptimizer.setup_maximum_performance()
    PerformanceOptimizer.enable_aggressive_caching()
    
    print("🔥🔥🔥 RTX 4090 BEAST MODE ACTIVATED 🔥🔥🔥")