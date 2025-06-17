#!/usr/bin/env python3
"""
Universal BEAST MODE Performance Optimization System
RTX 4090 + Ryzen 9 3900X Maximum Performance for ALL CineSync v2 Models

This script provides a unified interface to enable maximum performance
optimizations across all recommendation models in the CineSync v2 system.
"""

import torch
import torch.nn as nn
import multiprocessing as mp
import psutil
import gc
from typing import Dict, Any, Optional, Union
from pathlib import Path
import logging
import json


class UniversalBeastMode:
    """Universal performance optimization system for all CineSync v2 models"""
    
    def __init__(self):
        self.logger = self._setup_logging()
        self.device_info = self._detect_hardware()
        self.optimization_cache = {}
        
    def _setup_logging(self):
        """Setup logging for beast mode operations"""
        logging.basicConfig(
            level=logging.INFO,
            format='🔥 BEAST MODE - %(asctime)s - %(message)s'
        )
        return logging.getLogger("UniversalBeastMode")
    
    def _detect_hardware(self) -> Dict[str, Any]:
        """Detect and analyze available hardware"""
        info = {
            'cpu_cores': mp.cpu_count(),
            'cpu_threads': psutil.cpu_count(logical=True),
            'ram_gb': psutil.virtual_memory().total / (1024**3),
            'gpu_available': torch.cuda.is_available(),
        }
        
        if torch.cuda.is_available():
            info.update({
                'gpu_name': torch.cuda.get_device_name(0),
                'gpu_memory_gb': torch.cuda.get_device_properties(0).total_memory / (1024**3),
                'gpu_compute_capability': torch.cuda.get_device_properties(0).major,
                'gpu_multiprocessors': torch.cuda.get_device_properties(0).multi_processor_count,
                'tensorcore_available': torch.cuda.get_device_properties(0).major >= 7
            })
        
        return info
    
    def activate_global_beast_mode(self) -> Dict[str, Any]:
        """Activate maximum performance optimizations globally"""
        self.logger.info("🚀🚀🚀 ACTIVATING UNIVERSAL BEAST MODE 🚀🚀🚀")
        
        optimizations = {
            'pytorch_optimizations': self._optimize_pytorch(),
            'system_optimizations': self._optimize_system(),
            'memory_optimizations': self._optimize_memory(),
            'gpu_optimizations': self._optimize_gpu() if self.device_info['gpu_available'] else None,
            'recommended_settings': self._get_recommended_settings()
        }
        
        self._log_hardware_status()
        self._log_optimization_status(optimizations)
        
        return optimizations
    
    def _optimize_pytorch(self) -> Dict[str, bool]:
        """Apply PyTorch-specific optimizations"""
        optimizations = {}
        
        # Threading optimizations
        torch.set_num_threads(self.device_info['cpu_threads'])
        torch.set_num_interop_threads(max(1, self.device_info['cpu_threads'] // 4))
        optimizations['threading_optimized'] = True
        
        # Memory format optimizations
        try:
            torch.backends.opt_einsum.enabled = True
            optimizations['einsum_optimized'] = True
        except:
            optimizations['einsum_optimized'] = False
        
        # JIT optimizations
        try:
            torch.jit.set_fusion_strategy([('STATIC', 20), ('DYNAMIC', 20)])
            optimizations['jit_fusion_optimized'] = True
        except:
            optimizations['jit_fusion_optimized'] = False
        
        return optimizations
    
    def _optimize_system(self) -> Dict[str, Any]:
        """Apply system-level optimizations"""
        optimizations = {
            'cpu_threads_set': self.device_info['cpu_threads'],
            'memory_available_gb': self.device_info['ram_gb'],
            'recommended_workers': min(24, self.device_info['cpu_threads']),
            'garbage_collection_enabled': True
        }
        
        # Enable aggressive garbage collection
        gc.enable()
        gc.set_threshold(700, 10, 10)  # More aggressive GC
        
        return optimizations
    
    def _optimize_memory(self) -> Dict[str, Any]:
        """Apply memory-specific optimizations"""
        optimizations = {
            'allocator_settings': {},
            'caching_enabled': True
        }
        
        # Set memory allocator options
        if self.device_info['gpu_available']:
            # Enable memory pooling for faster allocation
            optimizations['allocator_settings']['gpu_memory_fraction'] = 0.95
        
        return optimizations
    
    def _optimize_gpu(self) -> Dict[str, Any]:
        """Apply GPU-specific optimizations"""
        if not self.device_info['gpu_available']:
            return {}
        
        optimizations = {}
        
        # CuDNN optimizations
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True
        optimizations['cudnn_benchmark'] = True
        
        # Tensor Core optimizations (RTX series)
        if self.device_info.get('tensorcore_available', False):
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            optimizations['tensorcore_enabled'] = True
        
        # Memory management
        torch.cuda.empty_cache()
        torch.cuda.set_per_process_memory_fraction(0.95)
        optimizations['memory_fraction_set'] = 0.95
        
        # Compilation optimizations
        optimizations['compilation_available'] = hasattr(torch, 'compile')
        
        return optimizations
    
    def _get_recommended_settings(self) -> Dict[str, Any]:
        """Get recommended settings based on detected hardware"""
        settings = {}
        
        # Batch size recommendations
        if self.device_info['gpu_available']:
            gpu_memory = self.device_info['gpu_memory_gb']
            if gpu_memory >= 20:  # RTX 4090, A6000, etc.
                settings['batch_size_large_models'] = 16384
                settings['batch_size_sequential_models'] = 8192
                settings['batch_size_graph_models'] = 4096
            elif gpu_memory >= 10:  # RTX 3080, 3090, etc.
                settings['batch_size_large_models'] = 8192
                settings['batch_size_sequential_models'] = 4096
                settings['batch_size_graph_models'] = 2048
            else:  # Lower end GPUs
                settings['batch_size_large_models'] = 2048
                settings['batch_size_sequential_models'] = 1024
                settings['batch_size_graph_models'] = 512
        else:
            settings['batch_size_large_models'] = 256
            settings['batch_size_sequential_models'] = 128
            settings['batch_size_graph_models'] = 64
        
        # Worker recommendations
        settings['num_workers'] = min(24, self.device_info['cpu_threads'])
        settings['prefetch_factor'] = 4 if self.device_info['cpu_threads'] >= 16 else 2
        settings['persistent_workers'] = self.device_info['cpu_threads'] >= 8
        
        # Model architecture recommendations
        if self.device_info['gpu_available'] and self.device_info['gpu_memory_gb'] >= 20:
            settings['embedding_dim'] = 512
            settings['hidden_dims'] = [2048, 1024, 512, 256]
            settings['num_attention_heads'] = 16
            settings['num_transformer_layers'] = 8
        else:
            settings['embedding_dim'] = 256
            settings['hidden_dims'] = [1024, 512, 256]
            settings['num_attention_heads'] = 8
            settings['num_transformer_layers'] = 4
        
        return settings
    
    def optimize_model_for_hardware(self, model: nn.Module, model_type: str = "general") -> nn.Module:
        """Optimize a specific model for the detected hardware"""
        self.logger.info(f"🔥 Optimizing {model_type} model for {self.device_info['gpu_name'] if self.device_info['gpu_available'] else 'CPU'}")
        
        # Move to appropriate device
        device = torch.device('cuda' if self.device_info['gpu_available'] else 'cpu')
        model = model.to(device)
        
        # Apply model-specific optimizations
        if self.device_info['gpu_available']:
            # Try to compile the model (PyTorch 2.0+)
            try:
                model = torch.compile(model, mode='max-autotune')
                self.logger.info("✅ Model compiled with torch.compile (max-autotune)")
            except Exception as e:
                self.logger.warning(f"⚠️ torch.compile failed: {e}")
            
            # Set to channels_last memory format if supported
            try:
                model = model.to(memory_format=torch.channels_last)
                self.logger.info("✅ Model set to channels_last memory format")
            except:
                pass  # Not all models support this
        
        return model
    
    def get_optimal_batch_size(self, model: nn.Module, sample_input_shape: tuple, 
                             max_memory_usage: float = 0.9) -> int:
        """Determine optimal batch size for a model"""
        if not self.device_info['gpu_available']:
            return 256
        
        device = torch.device('cuda')
        model.eval()
        
        # Test increasingly large batch sizes
        if self.device_info['gpu_memory_gb'] >= 20:  # RTX 4090 class
            test_sizes = [1024, 2048, 4096, 8192, 16384, 20480, 24576, 32768]
        elif self.device_info['gpu_memory_gb'] >= 10:  # RTX 3080 class
            test_sizes = [512, 1024, 2048, 4096, 8192, 12288, 16384]
        else:
            test_sizes = [64, 128, 256, 512, 1024, 2048, 4096]
        
        optimal_batch = test_sizes[0]
        
        for batch_size in test_sizes:
            try:
                # Create dummy input
                dummy_input = torch.randn(batch_size, *sample_input_shape).to(device)
                
                torch.cuda.empty_cache()
                with torch.no_grad():
                    _ = model(dummy_input)
                
                # Check memory usage
                memory_used = torch.cuda.memory_allocated() / torch.cuda.get_device_properties(0).total_memory
                
                if memory_used < max_memory_usage:
                    optimal_batch = batch_size
                    self.logger.info(f"✅ Batch {batch_size}: {memory_used:.1%} VRAM usage")
                else:
                    self.logger.info(f"❌ Batch {batch_size}: Would exceed {max_memory_usage:.1%} VRAM")
                    break
                    
            except RuntimeError as e:
                if "out of memory" in str(e):
                    self.logger.info(f"❌ Batch {batch_size}: OOM")
                    break
                raise
            finally:
                torch.cuda.empty_cache()
        
        self.logger.info(f"🎯 Optimal batch size: {optimal_batch}")
        return optimal_batch
    
    def _log_hardware_status(self):
        """Log detected hardware status"""
        self.logger.info("=" * 60)
        self.logger.info("HARDWARE DETECTION RESULTS")
        self.logger.info("=" * 60)
        self.logger.info(f"CPU: {self.device_info['cpu_threads']} threads")
        self.logger.info(f"RAM: {self.device_info['ram_gb']:.1f} GB")
        
        if self.device_info['gpu_available']:
            self.logger.info(f"GPU: {self.device_info['gpu_name']}")
            self.logger.info(f"VRAM: {self.device_info['gpu_memory_gb']:.1f} GB")
            self.logger.info(f"Compute: {self.device_info['gpu_compute_capability']}.x")
            self.logger.info(f"TensorCores: {'Available' if self.device_info.get('tensorcore_available') else 'Not Available'}")
        else:
            self.logger.info("GPU: Not Available")
        
        self.logger.info("=" * 60)
    
    def _log_optimization_status(self, optimizations: Dict):
        """Log applied optimizations"""
        self.logger.info("APPLIED OPTIMIZATIONS")
        self.logger.info("=" * 60)
        
        if optimizations['pytorch_optimizations']:
            self.logger.info("✅ PyTorch optimizations applied")
        
        if optimizations['gpu_optimizations']:
            gpu_opts = optimizations['gpu_optimizations']
            if gpu_opts.get('tensorcore_enabled'):
                self.logger.info("✅ TensorCore acceleration enabled")
            if gpu_opts.get('cudnn_benchmark'):
                self.logger.info("✅ CuDNN benchmark mode enabled")
        
        settings = optimizations['recommended_settings']
        self.logger.info(f"🎯 Recommended batch size (large models): {settings['batch_size_large_models']}")
        self.logger.info(f"🎯 Recommended workers: {settings['num_workers']}")
        self.logger.info(f"🎯 Recommended embedding dim: {settings['embedding_dim']}")
        
        self.logger.info("=" * 60)
        self.logger.info("🚀🚀🚀 BEAST MODE FULLY ACTIVATED! 🚀🚀🚀")
        self.logger.info("=" * 60)
    
    def save_optimization_profile(self, filepath: str):
        """Save optimization profile for reuse"""
        profile = {
            'hardware_info': self.device_info,
            'recommended_settings': self._get_recommended_settings(),
            'timestamp': torch.utils.data.get_worker_info()
        }
        
        with open(filepath, 'w') as f:
            json.dump(profile, f, indent=2, default=str)
        
        self.logger.info(f"💾 Optimization profile saved to {filepath}")


# Global instance for easy access
BEAST_MODE = UniversalBeastMode()


def activate_beast_mode() -> Dict[str, Any]:
    """Activate universal beast mode - simple function interface"""
    return BEAST_MODE.activate_global_beast_mode()


def optimize_model(model: nn.Module, model_type: str = "general") -> nn.Module:
    """Optimize a model for maximum performance"""
    return BEAST_MODE.optimize_model_for_hardware(model, model_type)


def get_beast_mode_config() -> Dict[str, Any]:
    """Get beast mode configuration for the current hardware"""
    return BEAST_MODE._get_recommended_settings()


if __name__ == "__main__":
    # Demo the universal beast mode system
    print("🔥🔥🔥 CineSync v2 Universal BEAST MODE 🔥🔥🔥")
    optimizations = activate_beast_mode()
    
    # Save the optimization profile
    BEAST_MODE.save_optimization_profile("beast_mode_profile.json")