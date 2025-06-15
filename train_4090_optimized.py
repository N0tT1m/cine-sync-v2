#!/usr/bin/env python3
"""
RTX 4090 Optimized Training for CineSync v2
Maximizes performance on RTX 4090 while handling all datasets.
"""

import subprocess
import sys
import logging
import torch
import psutil
import gc

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_4090_compatibility():
    """Check if running on RTX 4090 and optimize accordingly
    
    Detects the GPU hardware and determines if high-end optimizations
    can be applied. Uses GPU memory as a proxy for performance tier.
    
    Returns:
        bool: True if high-end GPU detected (>=20GB VRAM), False otherwise
    """
    if not torch.cuda.is_available():
        logger.warning("CUDA not available - falling back to CPU training")
        return False
    
    # Query GPU specifications for optimization decisions
    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # Convert to GB
    
    logger.info(f"Detected GPU: {gpu_name}")
    logger.info(f"GPU Memory: {gpu_memory:.1f} GB")
    
    # Classify GPU tier based on memory capacity (RTX 4090 has 24GB)
    is_4090_class = gpu_memory >= 20  # High-end threshold
    
    if is_4090_class:
        logger.info("✅ High-end GPU detected - using optimized settings")
    else:
        logger.info("⚠️  Lower-end GPU detected - using conservative settings")
    
    return is_4090_class

def get_optimal_settings():
    """Get optimal training settings based on available hardware
    
    Analyzes system resources (GPU memory, RAM, CPU cores) to determine
    the best training configuration. Returns different parameter sets
    for different hardware tiers to maximize performance while avoiding
    out-of-memory errors.
    
    Returns:
        dict: Training configuration with hyperparameters and system settings
    """
    # Analyze available system resources for configuration decisions
    ram_gb = psutil.virtual_memory().total / (1024**3)  # System RAM in GB
    cpu_cores = psutil.cpu_count()                       # Available CPU cores
    is_high_end_gpu = check_4090_compatibility()         # GPU performance tier
    
    logger.info(f"System RAM: {ram_gb:.1f} GB")
    logger.info(f"CPU Cores: {cpu_cores}")
    
    if is_high_end_gpu and ram_gb >= 16:
        # RTX 4090 or similar high-end GPU with sufficient system RAM
        # Aggressive settings for maximum performance
        return {
            "profile": "4090_optimized",
            "embedding_dim": 128,           # Large embeddings for rich representations
            "hidden_dim": 256,             # Deep network for complex patterns
            "batch_size": 256,             # Large batches for stable gradients
            "learning_rate": 0.0008,       # Slightly lower LR for large batches
            "epochs": 35,                  # More epochs for thorough training
            "patience": 8,                 # Higher patience for convergence
            "gradient_accumulation": 2,    # Effective batch size = 512
            "mixed_precision": True        # FP16 for 2x speed on RTX 4090
        }
    elif is_high_end_gpu:
        # RTX 4090 but limited system RAM - reduce memory usage
        return {
            "profile": "4090_conservative", 
            "embedding_dim": 96,            # Smaller embeddings to save memory
            "hidden_dim": 192,             # Reduced network size
            "batch_size": 128,             # Smaller batches for memory efficiency
            "learning_rate": 0.001,        # Standard learning rate
            "epochs": 30,                  # Fewer epochs due to memory constraints
            "patience": 6,                 # Lower patience for faster completion
            "gradient_accumulation": 4,    # Accumulate to maintain effective batch size
            "mixed_precision": True        # Still use FP16 for speed
        }
    else:
        # Lower-end GPU (RTX 3070, GTX series) or CPU training
        return {
            "profile": "standard",
            "embedding_dim": 64,            # Minimal embeddings for memory efficiency
            "hidden_dim": 128,             # Small network for limited compute
            "batch_size": 64,              # Small batches for memory constraints
            "learning_rate": 0.002,        # Higher LR to compensate for small batches
            "epochs": 25,                  # Fewer epochs for faster completion
            "patience": 5,                 # Quick early stopping
            "gradient_accumulation": 8,    # High accumulation for stability
            "mixed_precision": False       # FP32 for compatibility
        }

def optimize_pytorch_for_4090():
    """Set PyTorch optimizations for RTX 4090
    
    Applies RTX 4090-specific optimizations:
    - Enables cuDNN benchmark mode for consistent input sizes
    - Enables Tensor Core usage with TF32 precision
    - Clears GPU memory to prevent fragmentation
    """
    if torch.cuda.is_available():
        # Enable cuDNN auto-tuner for optimal algorithm selection
        torch.backends.cudnn.benchmark = True       # Optimize for consistent input sizes
        torch.backends.cudnn.deterministic = False  # Allow non-deterministic ops for speed
        
        # Clear any existing GPU memory to prevent fragmentation
        torch.cuda.empty_cache()
        
        # Enable Tensor Core acceleration on RTX 4090 (TF32 precision)
        torch.backends.cuda.matmul.allow_tf32 = True   # Matrix multiplications
        torch.backends.cudnn.allow_tf32 = True         # Convolution operations
        
        logger.info("✅ PyTorch optimizations enabled for RTX 4090")

def create_4090_training_command(settings):
    """Create optimized training command based on hardware profile
    
    Builds the command line arguments for the enhanced training script
    using the hardware-optimized settings.
    
    Args:
        settings (dict): Hardware-optimized training configuration
        
    Returns:
        list: Command line arguments for subprocess execution
    """
    cmd = [
        sys.executable, "enhanced_training.py",
        "--embedding_dim", str(settings["embedding_dim"]),
        "--hidden_dim", str(settings["hidden_dim"]),
        "--batch_size", str(settings["batch_size"]),
        "--learning_rate", str(settings["learning_rate"]),
        "--epochs", str(settings["epochs"]),
        "--patience", str(settings["patience"]),
        "--use_cuda"
    ]
    
    if settings.get("mixed_precision", False):
        cmd.append("--use_mixed_precision")
    
    return cmd

def monitor_training_resources():
    """Monitor GPU and system resources during training
    
    Logs current GPU memory usage and system RAM consumption
    to help with resource optimization and debugging.
    """
    if torch.cuda.is_available():
        # Get current GPU memory usage statistics
        gpu_memory_used = torch.cuda.memory_allocated() / (1024**3)  # Currently allocated
        gpu_memory_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # Total available
        
        logger.info(f"GPU Memory: {gpu_memory_used:.1f}/{gpu_memory_total:.1f} GB ({gpu_memory_used/gpu_memory_total*100:.1f}%)")
    
    # Get system RAM usage statistics
    ram_used = psutil.virtual_memory().used / (1024**3)   # Currently used RAM
    ram_total = psutil.virtual_memory().total / (1024**3)  # Total system RAM
    
    logger.info(f"System RAM: {ram_used:.1f}/{ram_total:.1f} GB ({ram_used/ram_total*100:.1f}%)")

def main():
    """Main function for RTX 4090 optimized training
    
    Orchestrates the hardware-aware training process:
    1. Analyzes system resources to select optimal configuration
    2. Applies PyTorch optimizations for detected hardware
    3. Launches training with real-time output monitoring
    4. Handles errors and resource cleanup
    """
    logger.info("🚀 CineSync v2 - RTX 4090 Optimized Training")
    logger.info("=" * 60)
    
    # Get optimal settings
    settings = get_optimal_settings()
    
    logger.info(f"Selected Profile: {settings['profile']}")
    logger.info(f"Configuration:")
    for key, value in settings.items():
        if key != 'profile':
            logger.info(f"  {key}: {value}")
    
    # Apply PyTorch optimizations
    optimize_pytorch_for_4090()
    
    # Monitor initial resources
    monitor_training_resources()
    
    logger.info("=" * 60)
    logger.info("Starting training...")
    
    # Create and run training command
    cmd = create_4090_training_command(settings)
    
    try:
        # Perform pre-training cleanup to maximize available memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()  # Clear GPU memory cache
            gc.collect()              # Python garbage collection
        
        logger.info(f"Executing: {' '.join(cmd)}")
        
        # Launch training process with real-time output streaming
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, 
                                 universal_newlines=True, bufsize=1)
        
        # Stream training output in real-time for monitoring
        for line in process.stdout:
            print(line.strip())  # Display each line as it's produced
        
        process.wait()  # Wait for training to complete
        
        if process.returncode == 0:
            logger.info("✅ Training completed successfully!")
            monitor_training_resources()  # Log final resource usage
        else:
            logger.error(f"❌ Training failed with return code {process.returncode}")
            sys.exit(1)
            
    except subprocess.CalledProcessError as e:
        logger.error(f"Training failed: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        # Handle graceful shutdown on Ctrl+C
        logger.info("Training interrupted by user")
        sys.exit(0)
    except Exception as e:
        # Handle any unexpected errors during execution
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()