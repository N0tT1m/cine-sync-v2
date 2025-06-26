#!/usr/bin/env python3
"""
Sequential Model Training Script - TV Shows/Anime Specialized
This script defaults to TV show datasets and is optimized for anime recommendations
"""

import argparse
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle
import os
import sys
from pathlib import Path
from datetime import datetime
import math

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

# Import wandb utilities
from wandb_config import init_wandb_for_training, WandbManager
from wandb_training_integration import WandbTrainingLogger

# Import the main training functions
from train_with_wandb import (
    SequentialDataset, PositionalEncoding, TransformerBlock, 
    AttentionalSequentialRecommender, setup_logging, WarmupScheduler,
    train_sequential_with_wandb
)

logger = logging.getLogger(__name__)

def parse_tv_args():
    """Parse command line arguments - TV Show specialized"""
    parser = argparse.ArgumentParser(description='Train Sequential Model for TV Shows/Anime with Wandb')
    
    # Data arguments - TV show datasets first
    possible_ratings_paths = [
        str(Path(__file__).parent.parent / 'tv' / 'misc' / 'reviews.csv'),
        '../tv/misc/reviews.csv',
        '../../tv/misc/reviews.csv',
        'tv/misc/reviews.csv',
        '/Users/timmy/workspace/ai-apps/cine-sync-v2/tv/misc/reviews.csv',
        # Fallback to movie data
        str(Path(__file__).parent.parent / 'movies' / 'cinesync' / 'ml-32m' / 'ratings.csv'),
        '../movies/cinesync/ml-32m/ratings.csv',
        '../../movies/cinesync/ml-32m/ratings.csv'
    ]
    
    possible_movies_paths = [
        str(Path(__file__).parent.parent / 'tv' / 'misc' / 'animes.csv'),
        '../tv/misc/animes.csv',
        '../../tv/misc/animes.csv',
        'tv/misc/animes.csv',
        '/Users/timmy/workspace/ai-apps/cine-sync-v2/tv/misc/animes.csv',
        # Fallback to movie data
        str(Path(__file__).parent.parent / 'movies' / 'cinesync' / 'ml-32m' / 'movies.csv'),
        '../movies/cinesync/ml-32m/movies.csv',
        '../../movies/cinesync/ml-32m/movies.csv'
    ]
    
    # Find first existing path
    default_ratings_path = None
    for path in possible_ratings_paths:
        if os.path.exists(path):
            default_ratings_path = path
            break
    
    default_movies_path = None
    for path in possible_movies_paths:
        if os.path.exists(path):
            default_movies_path = path
            break
    
    parser.add_argument('--ratings-path', type=str, default=default_ratings_path,
                       help='Path to ratings CSV file')
    parser.add_argument('--movies-path', type=str, default=default_movies_path,
                       help='Path to movies/shows CSV file')
    parser.add_argument('--min-interactions', type=int, default=10,
                       help='Minimum interactions per user (lower for TV shows)')
    parser.add_argument('--min-seq-length', type=int, default=3,
                       help='Minimum sequence length (lower for TV shows)')
    parser.add_argument('--max-seq-length', type=int, default=30,
                       help='Maximum sequence length (shorter for TV shows)')
    
    # Model arguments (optimized for TV shows)
    parser.add_argument('--embedding-dim', type=int, default=128,
                       help='Embedding dimension (smaller for TV shows)')
    parser.add_argument('--num-heads', type=int, default=4,
                       help='Number of attention heads (fewer for TV shows)')
    parser.add_argument('--num-layers', type=int, default=2,
                       help='Number of transformer layers (fewer for TV shows)')
    parser.add_argument('--d-ff', type=int, default=256,
                       help='Feed-forward dimension (smaller for TV shows)')
    parser.add_argument('--dropout', type=float, default=0.1,
                       help='Dropout rate')
    
    # Training arguments (optimized for smaller TV datasets)
    parser.add_argument('--learning-rate', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--batch-size', type=int, default=128,
                       help='Batch size')
    parser.add_argument('--epochs', type=int, default=30,
                       help='Number of epochs (fewer for TV shows)')
    parser.add_argument('--early-stopping-patience', type=int, default=8,
                       help='Early stopping patience')
    parser.add_argument('--warmup-steps', type=int, default=1000,
                       help='Warmup steps for learning rate schedule')
    
    # Wandb arguments
    parser.add_argument('--wandb-project', type=str, default='cinesync-v2-sequential-tv',
                       help='Wandb project name')
    parser.add_argument('--wandb-entity', type=str, default=None,
                       help='Wandb entity')
    parser.add_argument('--wandb-name', type=str, default=None,
                       help='Wandb run name')
    parser.add_argument('--wandb-tags', type=str, nargs='+', 
                       default=['sequential', 'transformer', 'tv-shows', 'anime'],
                       help='Wandb tags')
    parser.add_argument('--wandb-offline', action='store_true',
                       help='Run wandb in offline mode')
    
    # System arguments
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cpu', 'cuda'],
                       help='Device to use for training')
    parser.add_argument('--num-workers', type=int, default=4,
                       help='Number of data loader workers')
    
    return parser.parse_args()

def main():
    """Main function for TV show Sequential training"""
    setup_logging()
    args = parse_tv_args()
    
    print("🎬 Training Sequential Model for TV Shows/Anime")
    print("=" * 50)
    print(f"Ratings path: {args.ratings_path}")
    print(f"Content path: {args.movies_path}")
    print()
    
    # Check if datasets exist
    if not os.path.exists(args.ratings_path):
        logger.error(f"Ratings file not found: {args.ratings_path}")
        print("❌ Please ensure TV show datasets are available")
        return
    
    # Train model using the main training function
    model, metrics = train_sequential_with_wandb(args)
    
    print("✅ TV Show Sequential training completed successfully!")
    print(f"📊 Final metrics: {metrics}")

if __name__ == "__main__":
    main()