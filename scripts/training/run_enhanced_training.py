#!/usr/bin/env python3
"""
Quick start script for enhanced CineSync training
"""

import subprocess
import sys
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """Run enhanced training with optimal settings
    
    Quick-start script that launches the enhanced training pipeline with
    pre-configured parameters optimized for performance and stability.
    
    The script uses subprocess to launch the enhanced_training.py script
    with carefully chosen hyperparameters that work well across different
    hardware configurations.
    """
    logger.info("Starting CineSync v2 Enhanced Training...")
    
    # Training command with pre-tuned hyperparameters for optimal performance
    cmd = [
        sys.executable, "enhanced_training.py",
        "--embedding_dim", "128",     # Embedding dimension for user/item representations
        "--hidden_dim", "256",        # Hidden layer size for neural network
        "--batch_size", "512",        # Batch size optimized for memory usage
        "--learning_rate", "0.001",   # Learning rate for stable convergence
        "--epochs", "30",             # Maximum training epochs
        "--patience", "8",            # Early stopping patience
        "--use_cuda"                  # Enable GPU acceleration if available
    ]
    
    try:
        # Execute the training script with configured parameters
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        logger.info("Training completed successfully!")
        print(result.stdout)  # Display training output for monitoring
        
    except subprocess.CalledProcessError as e:
        # Handle training script failures with detailed error reporting
        logger.error(f"Training failed with return code {e.returncode}")
        print("STDOUT:", e.stdout)  # Show captured output for debugging
        print("STDERR:", e.stderr)  # Show error messages
        sys.exit(1)
    except Exception as e:
        # Handle unexpected errors (file not found, permission issues, etc.)
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()