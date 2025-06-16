#!/usr/bin/env python3
# CineSync v2 - Movie Training Script
# Training script specifically for movie recommendations

import os
import sys
import pandas as pd
import numpy as np
import torch
import logging
from pathlib import Path
import argparse
from datetime import datetime

# Add the current directory to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.movie_recommender import MovieRecommendationSystem
from config import load_config

def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f'movie_training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def load_movie_data(data_path: str = "../../movies/") -> tuple:
    """Load movie datasets
    
    Args:
        data_path: Path to movie data directory
        
    Returns:
        Tuple of (ratings_df, movies_df)
    """
    logger = logging.getLogger(__name__)
    
    # Try to load MovieLens data first
    ml_path = Path(data_path) / "cinesync" / "ml-32m"
    if ml_path.exists():
        logger.info("Loading MovieLens 32M dataset...")
        ratings_df = pd.read_csv(ml_path / "ratings.csv")
        movies_df = pd.read_csv(ml_path / "movies.csv")
        return ratings_df, movies_df
    
    # Try alternative paths
    alt_paths = [
        Path("../../ml-32m"),
        Path("../../../ml-32m"),
        Path("../../kaggle_complete_dataset/ml-32m"),
    ]
    
    for path in alt_paths:
        if path.exists():
            logger.info(f"Loading MovieLens data from {path}...")
            ratings_df = pd.read_csv(path / "ratings.csv")
            movies_df = pd.read_csv(path / "movies.csv")
            return ratings_df, movies_df
    
    raise FileNotFoundError("Could not find MovieLens dataset in any expected location")

def preprocess_data(ratings_df: pd.DataFrame, movies_df: pd.DataFrame, 
                   sample_size: int = None) -> tuple:
    """Preprocess the movie data
    
    Args:
        ratings_df: Ratings DataFrame
        movies_df: Movies DataFrame
        sample_size: Optional sample size for testing
        
    Returns:
        Preprocessed (ratings_df, movies_df)
    """
    logger = logging.getLogger(__name__)
    
    logger.info(f"Original data: {len(ratings_df)} ratings, {len(movies_df)} movies")
    
    # Sample data if requested (for testing)
    if sample_size:
        logger.info(f"Sampling {sample_size} ratings for testing...")
        ratings_df = ratings_df.sample(n=sample_size, random_state=42)
    
    # Filter out users and movies with too few ratings
    min_user_ratings = 20
    min_movie_ratings = 5
    
    # Count ratings per user and movie
    user_counts = ratings_df['userId'].value_counts()
    movie_counts = ratings_df['movieId'].value_counts()
    
    # Filter users and movies
    valid_users = user_counts[user_counts >= min_user_ratings].index
    valid_movies = movie_counts[movie_counts >= min_movie_ratings].index
    
    ratings_df = ratings_df[
        (ratings_df['userId'].isin(valid_users)) & 
        (ratings_df['movieId'].isin(valid_movies))
    ]
    
    # Filter movies DataFrame to match
    movies_df = movies_df[movies_df['movieId'].isin(ratings_df['movieId'].unique())]
    
    logger.info(f"Filtered data: {len(ratings_df)} ratings, {len(movies_df)} movies")
    logger.info(f"Users: {ratings_df['userId'].nunique()}, Movies: {ratings_df['movieId'].nunique()}")
    
    return ratings_df, movies_df

def train_movie_model(ratings_df: pd.DataFrame, movies_df: pd.DataFrame,
                     config: dict) -> MovieRecommendationSystem:
    """Train the movie recommendation model
    
    Args:
        ratings_df: Ratings DataFrame
        movies_df: Movies DataFrame
        config: Configuration dictionary
        
    Returns:
        Trained MovieRecommendationSystem
    """
    logger = logging.getLogger(__name__)
    
    # Initialize recommendation system
    rec_system = MovieRecommendationSystem(model_path=config.model.models_dir)
    
    # Prepare data
    logger.info("Preparing training data...")
    dataset, metadata = rec_system.prepare_data(ratings_df, movies_df)
    
    logger.info(f"Dataset metadata: {metadata}")
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Training on device: {device}")
    
    # Train model
    logger.info("Starting model training...")
    history = rec_system.train(
        dataset=dataset,
        metadata=metadata,
        num_epochs=config.model.num_epochs,
        batch_size=config.model.batch_size,
        learning_rate=config.model.learning_rate,
        device=device
    )
    
    # Save model
    logger.info("Saving trained model...")
    rec_system.save_model("movie_recommender.pth")
    
    return rec_system

def test_recommendations(rec_system: MovieRecommendationSystem, 
                        ratings_df: pd.DataFrame, num_tests: int = 5):
    """Test the recommendation system with sample users
    
    Args:
        rec_system: Trained recommendation system
        ratings_df: Original ratings DataFrame
        num_tests: Number of test users to try
    """
    logger = logging.getLogger(__name__)
    
    logger.info("Testing recommendations...")
    
    # Get some random users who have ratings
    test_users = ratings_df['userId'].sample(n=num_tests, random_state=42).values
    
    for user_id in test_users:
        try:
            logger.info(f"\nTesting recommendations for user {user_id}:")
            
            # Get user's actual ratings for context
            user_ratings = ratings_df[ratings_df['userId'] == user_id].sort_values('rating', ascending=False)
            logger.info(f"User has {len(user_ratings)} ratings, top rated movies:")
            for _, row in user_ratings.head(3).iterrows():
                logger.info(f"  Movie {row['movieId']}: {row['rating']}")
            
            # Get recommendations
            recommendations = rec_system.recommend_movies(
                user_id=user_id,
                num_recommendations=5,
                exclude_seen=True,
                ratings_df=ratings_df
            )
            
            logger.info("Top recommendations:")
            for i, rec in enumerate(recommendations, 1):
                movie_title = rec.get('title', f"Movie {rec['movie_id']}")
                logger.info(f"  {i}. {movie_title} (predicted: {rec['predicted_rating']:.2f})")
                
        except Exception as e:
            logger.error(f"Error testing user {user_id}: {e}")

def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description='Train CineSync Movie Recommendation Model')
    parser.add_argument('--sample-size', type=int, help='Sample size for testing (optional)')
    parser.add_argument('--data-path', type=str, default="../../movies/", 
                       help='Path to movie data directory')
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging()
    logger.info("Starting CineSync Movie Model Training")
    
    try:
        # Load configuration
        config = load_config()
        logger.info(f"Loaded configuration: {config}")
        
        # Load data
        logger.info("Loading movie data...")
        ratings_df, movies_df = load_movie_data(args.data_path)
        
        # Preprocess data
        ratings_df, movies_df = preprocess_data(ratings_df, movies_df, args.sample_size)
        
        # Train model
        rec_system = train_movie_model(ratings_df, movies_df, config)
        
        # Test recommendations
        test_recommendations(rec_system, ratings_df)
        
        logger.info("Training completed successfully!")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise

if __name__ == "__main__":
    main()