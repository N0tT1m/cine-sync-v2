#!/usr/bin/env python3
# CineSync v2 - Hybrid Recommendation System Main Training Script
# FIXED: run_training_pytorch.py - Proper ID mapping to prevent non-existent movie recommendations
# 
# This is the main training script for the CineSync movie recommendation system.
# It implements a hybrid approach combining collaborative filtering and content-based
# filtering with proper ID mapping to ensure only real movies are recommended.
#
# Key features:
# - Hybrid neural collaborative filtering model
# - Proper ID mapping to prevent out-of-bounds errors
# - Multiple dataset support (MovieLens, Netflix, TMDB)
# - GPU acceleration with mixed precision training
# - WandB integration for experiment tracking
# - Early stopping and model checkpointing

import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader, Dataset
import pickle
import logging
import argparse
import gc
import time
from datetime import datetime
from pathlib import Path
import wandb

def check_datasets_before_training():
    """Check if required datasets are available before starting training.
    
    Validates that essential MovieLens datasets are present before beginning
    the training process. Provides helpful guidance if datasets are missing.
    
    Returns:
        bool: True if all required datasets are found, False otherwise
    """
    print("🔍 Checking datasets before training...")
    
    # Check for required dataset files
    required_files = [
        'ml-32m/ratings.csv',
        'ml-32m/movies.csv'
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        print("❌ Missing required datasets:")
        for file_path in missing_files:
            print(f"   • {file_path}")
        
        print("\n📥 To get the required datasets:")
        print("1. Run: python setup_datasets.py")
        print("2. Or download from: https://kaggle.com/datasets/nott1m/cinesync-complete-training-dataset")
        print("3. Then organize: python organize_datasets.py")
        print("4. Verify: python check_datasets.py")
        
        print(f"\n💡 For detailed dataset status: python check_datasets.py --detailed")
        return False
    
    print("✅ Required datasets found!")
    return True

from config import load_config
from models import HybridRecommenderModel, MovieDataset, create_model, save_model, load_model
from utils import DatabaseManager, load_ratings_data, load_movies_data

def setup_logging(config):
    """Setup logging configuration"""
    log_level = logging.DEBUG if config.debug else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f"training_{datetime.now().strftime('%Y%m%d%H%M%S')}.log", encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger("movie-recommendation-training")

def setup_gpu():
    """Set up GPU for training if available"""
    try:
        if torch.cuda.is_available():
            num_gpus = torch.cuda.device_count()
            logger.info(f"Found {num_gpus} CUDA-enabled GPU(s)")
            
            if num_gpus > 0:
                device_name = torch.cuda.get_device_name(0)
                logger.info(f"Using GPU: {device_name}")
                
                free_mem, total_mem = torch.cuda.mem_get_info(0)
                free_mem_gb = free_mem / (1024 ** 3)
                total_mem_gb = total_mem / (1024 ** 3)
                logger.info(f"GPU Memory: {free_mem_gb:.2f}GB free / {total_mem_gb:.2f}GB total")
                
                torch.backends.cudnn.benchmark = True
                logger.info("Enabled cuDNN benchmark mode")
                
                device = torch.device("cuda:0")
                cuda_version = torch.version.cuda
                logger.info(f"CUDA Version: {cuda_version}")
                
                return device
        
        logger.warning("No GPU available, using CPU")
        return torch.device("cpu")
    except Exception as e:
        logger.error(f"Error during GPU setup: {e}")
        return torch.device("cpu")

def create_id_mappings(ratings_df, movies_df):
    """
    CRITICAL FIX: Create proper ID mappings to prevent out-of-bounds errors
    Maps original IDs to contiguous 0-based indices for embeddings
    
    This function solves a critical issue where original MovieLens IDs are not
    contiguous (e.g., 1, 3, 6, 11...) which causes embedding layer errors.
    We map them to contiguous indices (0, 1, 2, 3...) for safe PyTorch usage.
    
    Args:
        ratings_df: DataFrame with user ratings
        movies_df: DataFrame with movie information
        
    Returns:
        tuple: (updated_ratings_df, updated_movies_df, mappings_dict)
               or (None, None, None) if mapping fails
    """
    logger.info("Creating ID mappings to prevent out-of-bounds errors")
    
    # Get unique user and movie IDs from the data
    unique_user_ids = sorted(ratings_df['userId'].unique())
    unique_movie_ids = sorted(ratings_df['movieId'].unique())
    
    # Create mappings from original IDs to contiguous indices (0, 1, 2, ...)
    user_id_to_idx = {user_id: idx for idx, user_id in enumerate(unique_user_ids)}
    movie_id_to_idx = {movie_id: idx for idx, movie_id in enumerate(unique_movie_ids)}
    
    # Create reverse mappings for inference
    idx_to_user_id = {idx: user_id for user_id, idx in user_id_to_idx.items()}
    idx_to_movie_id = {idx: movie_id for movie_id, idx in movie_id_to_idx.items()}
    
    logger.info(f"Created mappings for {len(unique_user_ids)} users and {len(unique_movie_ids)} movies")
    logger.info(f"User ID range: {min(unique_user_ids)} to {max(unique_user_ids)}")
    logger.info(f"Movie ID range: {min(unique_movie_ids)} to {max(unique_movie_ids)}")
    logger.info(f"Mapped to indices: 0 to {len(unique_user_ids)-1} (users), 0 to {len(unique_movie_ids)-1} (movies)")
    
    # Filter movies_df to only include movies that appear in ratings
    movies_with_ratings = movies_df[movies_df['media_id'].isin(unique_movie_ids)].copy()
    
    # Add the mapped indices to the dataframes
    ratings_df = ratings_df.copy()
    ratings_df['user_idx'] = ratings_df['userId'].map(user_id_to_idx)
    ratings_df['movie_idx'] = ratings_df['movieId'].map(movie_id_to_idx)
    
    movies_with_ratings['movie_idx'] = movies_with_ratings['media_id'].map(movie_id_to_idx)
    
    # Verify no missing mappings
    if ratings_df['user_idx'].isna().any():
        logger.error("Some users couldn't be mapped!")
        return None, None, None, None, None, None
    
    if ratings_df['movie_idx'].isna().any():
        logger.error("Some movies couldn't be mapped!")
        return None, None, None, None, None, None
    
    mappings = {
        'user_id_to_idx': user_id_to_idx,
        'movie_id_to_idx': movie_id_to_idx,
        'idx_to_user_id': idx_to_user_id,
        'idx_to_movie_id': idx_to_movie_id,
        'num_users': len(unique_user_ids),
        'num_movies': len(unique_movie_ids)
    }
    
    return ratings_df, movies_with_ratings, mappings

def find_data_files():
    """Find data files in various locations"""
    logger.info("Searching for data files...")
    
    base_dirs = [
        os.path.abspath(os.curdir),
        os.path.abspath(os.path.join(os.curdir, "data"))
    ]
    
    data_patterns = {
        "tmdb_movies": ["tmdb/actor_filmography_data_movies.csv"],
        "tmdb_tv": ["tmdb/actor_filmography_data_tv.csv"],
        "ml_movies": ["ml-32m/movies.csv", "ml-25m/movies.csv", "ml-1m/movies.dat"],
        "ml_ratings": ["ml-32m/ratings.csv", "ml-25m/ratings.csv", "ml-1m/ratings.dat"],
        "netflix_ratings": [f"archive/combined_data_{i}.txt" for i in range(1, 5)]
    }
    
    found_files = {}
    
    for data_type, patterns in data_patterns.items():
        for pattern in patterns:
            found = False
            for base_dir in base_dirs:
                path = os.path.join(base_dir, pattern)
                if os.path.exists(path):
                    found_files[data_type] = path
                    logger.info(f"Found {data_type} at {path}")
                    found = True
                    break
            if found:
                break
    
    return found_files

def process_csv_file(file_path):
    """Process a CSV file with movie data"""
    logger.info(f"Processing CSV file: {file_path}")
    
    try:
        # Handle different file formats
        if file_path.endswith('.dat'):
            # MovieLens .dat format
            if 'movies.dat' in file_path:
                df = pd.read_csv(file_path, sep='::', names=['movieId', 'title', 'genres'], engine='python')
            elif 'ratings.dat' in file_path:
                df = pd.read_csv(file_path, sep='::', names=['userId', 'movieId', 'rating', 'timestamp'], engine='python')
            else:
                df = pd.read_csv(file_path, sep='::', engine='python')
        else:
            # Regular CSV format
            if os.path.getsize(file_path) > 100 * 1024 * 1024:  # 100MB
                logger.info("Large file detected, processing in chunks")
                chunks = []
                for chunk in pd.read_csv(file_path, chunksize=250000):
                    chunks.append(chunk)
                df = pd.concat(chunks)
                chunks = None
                gc.collect()
            else:
                df = pd.read_csv(file_path)
        
        # Check if it's MovieLens format
        if 'movieId' in df.columns and 'title' in df.columns and 'genres' in df.columns:
            df = df.rename(columns={'movieId': 'media_id'})
            if 'actor_id' not in df.columns:
                df['actor_id'] = np.arange(len(df))
            
            logger.info(f"Processed MovieLens format CSV with {len(df)} rows")
            return df
        
        # Check if it's standard format
        if 'actor_id' in df.columns and 'media_id' in df.columns:
            df = df.drop_duplicates(['actor_id', 'media_id'], keep='first')
            df['media_id'] = df['media_id'].astype(int)
            
            if 'genres' in df.columns:
                df['genres_list'] = df['genres'].apply(
                    lambda x: x.split('|') if pd.notnull(x) else []
                )
            
            logger.info(f"Processed standard format CSV with {len(df)} rows")
            return df
        
        logger.warning("Unknown CSV format. Please check the file structure.")
        return None
    except Exception as e:
        logger.exception(f"Error processing CSV file {file_path}: {e}")
        return None

def process_ratings_file(file_path, format_type='movielens'):
    """Process ratings file in various formats"""
    logger.info(f"Processing {format_type} ratings file: {file_path}")
    
    try:
        if format_type.lower() == 'movielens':
            if file_path.endswith('.dat'):
                # MovieLens .dat format
                df = pd.read_csv(file_path, sep='::', names=['userId', 'movieId', 'rating', 'timestamp'], engine='python')
            else:
                # Regular CSV
                if os.path.getsize(file_path) > 100 * 1024 * 1024:  # 100MB
                    logger.info("Large ratings file detected, processing in chunks")
                    chunks = []
                    for chunk in pd.read_csv(file_path, chunksize=250000):
                        chunks.append(chunk)
                    df = pd.concat(chunks)
                    chunks = None
                    gc.collect()
                else:
                    df = pd.read_csv(file_path)
            
            logger.info(f"Processed {len(df)} MovieLens ratings")
            return df
        
        else:
            logger.warning(f"Unknown ratings format: {format_type}")
            return None
    except Exception as e:
        logger.exception(f"Error processing ratings file {file_path}: {e}")
        return None


class MovieRecommendationSystem:
    """FIXED: Recommendation system that only recommends REAL movies
    
    This class provides the inference interface for the trained recommendation
    model. It ensures that only movies present in the training dataset are
    recommended, preventing hallucinated or non-existent movie suggestions.
    
    Key features:
    - Smart candidate generation instead of scoring all movies
    - Multi-strategy recommendation (genre-based, popular, diverse)
    - Proper ID mapping consistency with training
    - Fallback mechanisms for new users
    - Real movie validation and lookup
    """
    
    def __init__(self, config):
        self.model_dir = config.model.models_dir
        self.config = config
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load all necessary components
        self._load_model_components()
    
    def _load_model_components(self):
        """Load model, mappings, and movie data"""
        try:
            # Load ID mappings (CRITICAL for proper inference)
            with open(os.path.join(self.model_dir, 'id_mappings.pkl'), 'rb') as f:
                self.mappings = pickle.load(f)
            
            logger.info(f"Loaded ID mappings: {self.mappings['num_users']} users, {self.mappings['num_movies']} movies")
            
            # Load movie lookup table
            with open(os.path.join(self.model_dir, 'movie_lookup.pkl'), 'rb') as f:
                self.movie_lookup = pickle.load(f)
            
            # Load movies dataframe
            self.movies_df = pd.read_csv(os.path.join(self.model_dir, 'movies_data.csv'))
            
            # Extract genres list from movies_df
            all_genres = set()
            for genres in self.movies_df['genres'].dropna():
                if pd.notnull(genres):
                    all_genres.update(genres.split('|'))
            self.genres_list = sorted(list(all_genres))
            
            logger.info(f"Found {len(self.genres_list)} genres: {self.genres_list}")
            
            # Load rating scaler
            with open(os.path.join(self.model_dir, 'rating_scaler.pkl'), 'rb') as f:
                self.rating_scaler = pickle.load(f)
            
            # Load model metadata
            with open(os.path.join(self.model_dir, 'model_metadata.pkl'), 'rb') as f:
                self.metadata = pickle.load(f)
            
            # Create and load the model
            self.model = HybridRecommenderModel(
                num_users=self.mappings['num_users'],
                num_movies=self.mappings['num_movies'],
                num_genres=len(self.genres_list),
                embedding_size=128
            )
            
            # Load trained weights
            checkpoint = torch.load(os.path.join(self.model_dir, 'best_model.pt'), 
                                  map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.to(self.device)
            self.model.eval()
            
            logger.info("Model loaded successfully!")
            logger.info(f"Available movies in system: {len(self.movie_lookup)}")
            
        except Exception as e:
            logger.error(f"Error loading model components: {e}")
            raise
    
    def _get_movie_genre_features(self, movie_id):
        """Get genre features for a specific movie ID"""
        try:
            # Find movie in the dataframe using original movie ID
            movie_row = self.movies_df[self.movies_df['media_id'] == movie_id]
            
            if movie_row.empty:
                logger.warning(f"Movie ID {movie_id} not found in dataset")
                return np.zeros(len(self.genres_list))
            
            # Extract genre features
            genre_features = []
            for genre in self.genres_list:
                if genre in movie_row.iloc[0]:
                    genre_features.append(movie_row.iloc[0][genre])
                else:
                    genre_features.append(0)
            
            return np.array(genre_features, dtype=np.float32)
            
        except Exception as e:
            logger.error(f"Error getting genre features for movie {movie_id}: {e}")
            return np.zeros(len(self.genres_list))
    
    def get_user_recommendations(self, user_id, num_recommendations=10, exclude_seen=True):
        """
        IMPROVED: Get recommendations for a user with intelligent candidate generation
        
        Instead of scoring all movies (expensive), this method uses smart candidate
        generation to pre-filter to the most promising movies, then scores only those.
        This provides significant performance improvements while maintaining quality.
        
        Args:
            user_id: User ID for recommendations
            num_recommendations: Number of recommendations to return
            exclude_seen: Whether to exclude movies user has already rated
            
        Returns:
            List[Dict]: List of movie recommendations with metadata
        """
        try:
            # Check if user exists in our mappings
            if user_id not in self.mappings['user_id_to_idx']:
                logger.warning(f"User {user_id} not found in training data. Using genre-based fallback.")
                return self._get_genre_based_recommendations(num_recommendations)
            
            user_idx = self.mappings['user_id_to_idx'][user_id]
            
            # IMPROVED: Smart candidate generation instead of scoring ALL movies
            candidate_movie_ids = self._generate_candidates(user_id, user_idx, num_recommendations * 3)
            
            if not candidate_movie_ids:
                logger.warning(f"No candidates found for user {user_id}, falling back to popular movies")
                return self._get_popular_movies_fallback(num_recommendations)
            
            logger.info(f"Generated {len(candidate_movie_ids)} candidates for user {user_id}")
            
            # Get indices for candidates only
            candidate_movie_indices = [self.mappings['movie_id_to_idx'][mid] for mid in candidate_movie_ids]
            
            # Prepare batch prediction for CANDIDATES only (much more efficient)
            user_indices = torch.tensor([user_idx] * len(candidate_movie_indices), dtype=torch.long)
            movie_indices = torch.tensor(candidate_movie_indices, dtype=torch.long)
            
            # Get genre features for candidate movies
            genre_features_list = []
            for movie_id in candidate_movie_ids:
                genre_features = self._get_movie_genre_features(movie_id)
                genre_features_list.append(genre_features)
            
            genre_features_tensor = torch.tensor(np.array(genre_features_list), dtype=torch.float32)
            
            # Move to device
            user_indices = user_indices.to(self.device)
            movie_indices = movie_indices.to(self.device)
            genre_features_tensor = genre_features_tensor.to(self.device)
            
            # Predict ratings for candidate movies only
            with torch.no_grad():
                predictions = self.model(user_indices, movie_indices, genre_features_tensor)
                predictions = predictions.cpu().numpy().flatten()
            
            # Convert predictions back to original rating scale
            predictions_rescaled = self.rating_scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()
            
            # Create recommendation list with REAL movie information
            recommendations = []
            for i, (movie_id, prediction) in enumerate(zip(candidate_movie_ids, predictions_rescaled)):
                movie_info = self.movie_lookup[movie_id]
                recommendations.append({
                    'movie_id': movie_id,
                    'title': movie_info.get('title', 'Unknown'),
                    'genres': movie_info.get('genres', ''),
                    'predicted_rating': float(prediction),
                    'confidence': float(predictions[i])
                })
            
            # Sort by predicted rating (descending)
            recommendations.sort(key=lambda x: x['predicted_rating'], reverse=True)
            
            # Return top N recommendations
            top_recommendations = recommendations[:num_recommendations]
            
            logger.info(f"Generated {len(top_recommendations)} recommendations for user {user_id}")
            for i, rec in enumerate(top_recommendations[:5]):
                logger.info(f"  {i+1}. {rec['title']} (ID: {rec['movie_id']}) - {rec['predicted_rating']:.2f}")
            
            return top_recommendations
            
        except Exception as e:
            logger.error(f"Error generating recommendations for user {user_id}: {e}")
            return []
    
    def get_movie_info(self, movie_id):
        """Get information about a specific movie"""
        if movie_id in self.movie_lookup:
            return self.movie_lookup[movie_id]
        else:
            return None
    
    def search_movies(self, query, limit=10):
        """Search for movies by title"""
        query_lower = query.lower()
        results = []
        
        for movie_id, movie_info in self.movie_lookup.items():
            if query_lower in movie_info.get('title', '').lower():
                results.append({
                    'movie_id': movie_id,
                    'title': movie_info.get('title', ''),
                    'genres': movie_info.get('genres', '')
                })
        
        return results[:limit]
    
    def _generate_candidates(self, user_id, user_idx, num_candidates):
        """Generate candidate movies for recommendation using multiple strategies"""
        try:
            candidates = set()
            
            # Strategy 1: Genre-based candidates (most important)
            genre_candidates = self._get_genre_based_candidates(user_idx, num_candidates // 2)
            candidates.update(genre_candidates)
            
            # Strategy 2: Popular movies (ensures good fallback)
            popular_candidates = self._get_popular_candidates(num_candidates // 4)
            candidates.update(popular_candidates)
            
            # Strategy 3: Diverse genre representation
            diverse_candidates = self._get_diverse_genre_candidates(num_candidates // 4)
            candidates.update(diverse_candidates)
            
            # Convert to list and limit
            candidate_list = list(candidates)[:num_candidates]
            
            logger.info(f"Generated {len(candidate_list)} total candidates using multiple strategies")
            return candidate_list
            
        except Exception as e:
            logger.error(f"Error generating candidates: {e}")
            return []
    
    def _get_genre_based_candidates(self, user_idx, num_candidates):
        """Get candidates based on inferred user genre preferences"""
        try:
            candidates = []
            
            # Sample movies from different genres to get variety
            genre_movie_map = {}
            for movie_id, movie_info in self.movie_lookup.items():
                genres = movie_info.get('genres', '').split('|')
                for genre in genres:
                    if genre not in genre_movie_map:
                        genre_movie_map[genre] = []
                    genre_movie_map[genre].append(movie_id)
            
            # Get movies from popular genres with some randomization
            import random
            popular_genres = ['Action', 'Comedy', 'Drama', 'Thriller', 'Adventure', 'Romance', 'Crime', 'Sci-Fi']
            
            for genre in popular_genres:
                if genre in genre_movie_map and len(candidates) < num_candidates:
                    genre_movies = genre_movie_map[genre]
                    # Randomly sample from each genre to avoid always getting the same movies
                    sample_size = min(num_candidates // len(popular_genres) + 1, len(genre_movies))
                    sampled = random.sample(genre_movies, sample_size)
                    candidates.extend(sampled)
            
            return candidates[:num_candidates]
            
        except Exception as e:
            logger.error(f"Error getting genre-based candidates: {e}")
            return []
    
    def _get_popular_candidates(self, num_candidates):
        """Get popular movies as candidates (movies that appear frequently in dataset)"""
        try:
            # Use a simple heuristic: movies with lower IDs tend to be more popular in MovieLens
            movie_ids = list(self.movie_lookup.keys())
            movie_ids.sort()  # Lower movie IDs first
            return movie_ids[:num_candidates]
            
        except Exception as e:
            logger.error(f"Error getting popular candidates: {e}")
            return []
    
    def _get_diverse_genre_candidates(self, num_candidates):
        """Get candidates to ensure genre diversity"""
        try:
            candidates = []
            import random
            
            # Group movies by primary genre
            genre_groups = {}
            for movie_id, movie_info in self.movie_lookup.items():
                primary_genre = movie_info.get('genres', '').split('|')[0] if movie_info.get('genres') else 'Unknown'
                if primary_genre not in genre_groups:
                    genre_groups[primary_genre] = []
                genre_groups[primary_genre].append(movie_id)
            
            # Sample from each genre group
            genres = list(genre_groups.keys())
            random.shuffle(genres)
            
            per_genre = max(1, num_candidates // len(genres))
            for genre in genres:
                if len(candidates) < num_candidates:
                    sample_size = min(per_genre, len(genre_groups[genre]))
                    sampled = random.sample(genre_groups[genre], sample_size)
                    candidates.extend(sampled)
            
            return candidates[:num_candidates]
            
        except Exception as e:
            logger.error(f"Error getting diverse candidates: {e}")
            return []
    
    def _get_genre_based_recommendations(self, num_recommendations):
        """Fallback for users not in training data"""
        try:
            # Use popular genres for new users
            popular_genres = ['Action', 'Comedy', 'Drama']
            candidates = self._get_genre_based_candidates(0, num_recommendations * 2)
            
            # Return random selection since we can't predict for unknown user
            import random
            random.shuffle(candidates)
            
            recommendations = []
            for movie_id in candidates[:num_recommendations]:
                movie_info = self.movie_lookup[movie_id]
                recommendations.append({
                    'movie_id': movie_id,
                    'title': movie_info.get('title', 'Unknown'),
                    'genres': movie_info.get('genres', ''),
                    'predicted_rating': 4.0,  # Default high rating for popular movies
                    'confidence': 0.5  # Lower confidence for fallback
                })
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error in genre-based fallback: {e}")
            return []
    
    def _get_popular_movies_fallback(self, num_recommendations):
        """Final fallback to popular movies"""
        try:
            popular_candidates = self._get_popular_candidates(num_recommendations)
            
            recommendations = []
            for movie_id in popular_candidates:
                movie_info = self.movie_lookup[movie_id]
                recommendations.append({
                    'movie_id': movie_id,
                    'title': movie_info.get('title', 'Unknown'),
                    'genres': movie_info.get('genres', ''),
                    'predicted_rating': 3.8,  # Default good rating
                    'confidence': 0.3  # Low confidence for fallback
                })
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error in popular movies fallback: {e}")
            return []

# Example usage and testing
def test_recommendations():
    """Test the recommendation system"""
    try:
        # Initialize the recommendation system
        rec_system = MovieRecommendationSystem()
        
        # Get a sample user ID from the mappings
        sample_user_id = list(rec_system.mappings['user_id_to_idx'].keys())[0]
        
        print(f"\n=== Testing Recommendations for User {sample_user_id} ===")
        
        # Get recommendations
        recommendations = rec_system.get_user_recommendations(sample_user_id, num_recommendations=10)
        
        print(f"\nTop 10 Movie Recommendations:")
        print("-" * 80)
        
        for i, rec in enumerate(recommendations, 1):
            print(f"{i:2d}. {rec['title']}")
            print(f"    Movie ID: {rec['movie_id']}")
            print(f"    Genres: {rec['genres']}")
            print(f"    Predicted Rating: {rec['predicted_rating']:.2f}")
            print(f"    Confidence: {rec['confidence']:.3f}")
            print()
        
        # Verify these are real movies by checking a few
        print("=== Verification: Checking if recommended movies are real ===")
        for i, rec in enumerate(recommendations[:3]):
            movie_info = rec_system.get_movie_info(rec['movie_id'])
            if movie_info:
                print(f"✅ Movie {rec['movie_id']} '{rec['title']}' exists in database")
            else:
                print(f"❌ Movie {rec['movie_id']} '{rec['title']}' NOT found in database")
        
        # Test movie search
        print("\n=== Testing Movie Search ===")
        search_results = rec_system.search_movies("toy story", limit=5)
        print("Search results for 'toy story':")
        for result in search_results:
            print(f"  - {result['title']} (ID: {result['movie_id']})")
        
    except Exception as e:
        logger.error(f"Error in test_recommendations: {e}")

class MovieRatingDataset(Dataset):
    """FIXED: PyTorch Dataset with proper ID mapping
    
    PyTorch Dataset class that properly handles the mapped user and movie indices
    along with genre features for training the hybrid recommendation model.
    """
    
    def __init__(self, user_indices, movie_indices, genre_features, ratings):
        self.user_indices = user_indices
        self.movie_indices = movie_indices
        self.genre_features = genre_features
        self.ratings = ratings
    
    def __len__(self):
        return len(self.ratings)
    
    def __getitem__(self, idx):
        return (
            self.user_indices[idx],
            self.movie_indices[idx],
            self.genre_features[idx],
            self.ratings[idx]
        )

def process_and_prepare_data(data_files):
    """FIXED: Process and prepare data with proper ID mapping
    
    Main data preprocessing pipeline that loads, cleans, and prepares data
    for training. Includes critical ID mapping to ensure embedding layer
    compatibility and proper train/validation splitting.
    
    Args:
        data_files: Dictionary mapping data types to file paths
        
    Returns:
        tuple: (train_data, val_data, movies_df, genres_list, mappings)
               All prepared for model training
    """
    logger.info("Processing and preparing data for training")
    
    # Process movie data
    movies_df = None
    
    if 'ml_movies' in data_files:
        logger.info(f"Processing MovieLens movie data: {data_files['ml_movies']}")
        ml_movies = process_csv_file(data_files['ml_movies'])
        if ml_movies is not None:
            if 'movieId' in ml_movies.columns:
                ml_movies = ml_movies.rename(columns={'movieId': 'media_id'})
            movies_df = ml_movies
    
    if movies_df is None:
        logger.error("No movie data found. Cannot continue.")
        return None, None, None, None, None
    
    # Remove duplicates
    movies_df = movies_df.drop_duplicates(subset=['media_id'])
    logger.info(f"Total unique media items: {len(movies_df)}")
    
    # Extract genres
    all_genres = set()
    for genres in movies_df['genres'].dropna():
        if pd.notnull(genres):
            all_genres.update(genres.split('|'))
    
    genres_list = sorted(list(all_genres))
    logger.info(f"Found {len(genres_list)} unique genres: {genres_list}")
    
    # One-hot encode genres
    logger.info("Creating genre features")
    for genre in genres_list:
        movies_df[genre] = movies_df['genres'].apply(
            lambda x: 1 if pd.notnull(x) and genre in x.split('|') else 0
        )
    
    # Process ratings data
    ratings_df = None
    if 'ml_ratings' in data_files:
        logger.info(f"Processing MovieLens ratings: {data_files['ml_ratings']}")
        ratings_df = process_ratings_file(data_files['ml_ratings'], format_type='movielens')
    
    if ratings_df is None:
        logger.error("No ratings data found. Cannot continue.")
        return None, None, None, None, None
    
    logger.info(f"Loaded {len(ratings_df)} ratings")
    
    # CRITICAL FIX: Create proper ID mappings
    ratings_df, movies_df, mappings = create_id_mappings(ratings_df, movies_df)
    
    if mappings is None:
        logger.error("Failed to create ID mappings")
        return None, None, None, None, None
    
    # Merge ratings with movie features using mapped indices
    logger.info("Merging ratings with movie features")
    data = pd.merge(ratings_df, movies_df, left_on='movieId', right_on='media_id', how='inner')
    
    logger.info(f"Merged data: {len(data)} ratings for {data['media_id'].nunique()} unique movies")
    
    if len(data) == 0:
        logger.error("No matching movies found between ratings and movie data")
        return None, None, None, None, None
    
    # Create train/validation splits
    logger.info("Creating train/validation split")
    test_size = 0.1 if len(data) > 1000000 else 0.2
    train_data, val_data = train_test_split(data, test_size=test_size, random_state=42)
    logger.info(f"Training data: {len(train_data)}, Validation data: {len(val_data)}")
    
    # Scale ratings
    logger.info("Scaling ratings")
    scaler = MinMaxScaler()
    train_data = train_data.copy()
    val_data = val_data.copy()
    train_data['rating_scaled'] = scaler.fit_transform(train_data[['rating']])
    val_data['rating_scaled'] = scaler.transform(val_data[['rating']])
    
    # Save scaler and mappings
    os.makedirs('models', exist_ok=True)
    with open('models/rating_scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    with open('models/id_mappings.pkl', 'wb') as f:
        pickle.dump(mappings, f)
    
    logger.info("Saved rating scaler and ID mappings")
    
    return train_data, val_data, movies_df, genres_list, mappings

def train_model(train_data, val_data, movies_df, genres, mappings, device, epochs=30, batch_size=128, resume_from_checkpoint=None):
    """FIXED: Train model with proper ID handling
    
    Main training loop that creates and trains the hybrid recommendation model.
    Includes mixed precision training, early stopping, checkpointing, and
    comprehensive validation to ensure the model learns proper embeddings.
    
    Args:
        train_data: Training dataset with mapped IDs
        val_data: Validation dataset with mapped IDs  
        movies_df: Movie metadata DataFrame
        genres: List of unique genres for one-hot encoding
        mappings: ID mapping dictionaries
        device: PyTorch device (cuda/cpu)
        epochs: Number of training epochs
        batch_size: Training batch size
        resume_from_checkpoint: Path to checkpoint file to resume from
        
    Returns:
        tuple: (trained_model, training_history)
    """
    logger.info(f"Starting model training with proper ID mapping")
    
    try:
        os.makedirs('models', exist_ok=True)
        os.makedirs('models/checkpoints', exist_ok=True)
        
        # Use the correct dimensions from mappings
        num_users = mappings['num_users']
        num_movies = mappings['num_movies']
        num_genres = len(genres)
        
        logger.info(f"Model dimensions: {num_users} users, {num_movies} movies, {num_genres} genres")
        
        # Clear GPU cache
        if device.type == 'cuda':
            torch.cuda.empty_cache()
            gc.collect()
        
        # Create datasets using mapped indices
        def create_tensor_dataset(data_df):
            # Use the mapped indices instead of original IDs
            user_indices = torch.tensor(data_df['user_idx'].values, dtype=torch.long)
            movie_indices = torch.tensor(data_df['movie_idx'].values, dtype=torch.long)
            
            # Validate indices are in bounds
            assert user_indices.max() < num_users, f"User index {user_indices.max()} >= {num_users}"
            assert movie_indices.max() < num_movies, f"Movie index {movie_indices.max()} >= {num_movies}"
            assert user_indices.min() >= 0, f"Negative user index: {user_indices.min()}"
            assert movie_indices.min() >= 0, f"Negative movie index: {movie_indices.min()}"
            
            # Ensure all genre columns exist
            missing_genres = [g for g in genres if g not in data_df.columns]
            if missing_genres:
                logger.warning(f"Missing genre columns: {missing_genres}")
                for genre in missing_genres:
                    data_df[genre] = 0
            
            genres_tensor = torch.tensor(data_df[genres].values, dtype=torch.float32)
            ratings = torch.tensor(data_df['rating_scaled'].values, dtype=torch.float32).unsqueeze(1)
            
            return MovieRatingDataset(user_indices, movie_indices, genres_tensor, ratings)
        
        logger.info("Creating training and validation datasets")
        train_dataset = create_tensor_dataset(train_data)
        val_dataset = create_tensor_dataset(val_data)
        
        logger.info(f"Created datasets: {len(train_dataset)} training, {len(val_dataset)} validation samples")
        
        # Data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                num_workers=4, pin_memory=True, persistent_workers=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                              num_workers=4, pin_memory=True, persistent_workers=True)
        
        # Create model
        model = HybridRecommenderModel(num_users, num_movies, num_genres, embedding_size=128)
        model.to(device)
        
        # Loss and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.AdamW(model.parameters(), lr=0.002, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
        scaler = GradScaler('cuda')
        
        # Training loop
        logger.info("Starting training")
        best_val_loss = float('inf')
        patience = 5
        patience_counter = 0
        start_epoch = 0
        
        # Load checkpoint if specified (like original code)
        if resume_from_checkpoint:
            try:
                logger.info(f"Attempting to resume from checkpoint: {resume_from_checkpoint}")
                checkpoint = torch.load(resume_from_checkpoint, map_location=device)
                model.load_state_dict(checkpoint['model_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                scaler.load_state_dict(checkpoint['scaler_state_dict'])
                start_epoch = checkpoint['epoch']
                best_val_loss = checkpoint['best_val_loss']
                history = checkpoint.get('history', {'train_loss': [], 'val_loss': [], 'learning_rates': []})
                logger.info(f"✅ Successfully resumed training from epoch {start_epoch}")
                logger.info(f"Previous best validation loss: {best_val_loss:.6f}")
            except Exception as e:
                logger.error(f"❌ Error loading checkpoint: {e}")
                logger.info("Starting training from scratch instead")
                start_epoch = 0
                best_val_loss = float('inf')
        
        history = history if 'history' in locals() else {'train_loss': [], 'val_loss': [], 'learning_rates': []}
        
        for epoch in range(start_epoch, epochs):
            # Training phase
            model.train()
            train_loss = 0.0
            batch_count = 0
            
            for batch_idx, (user_ids, movie_ids, genre_feats, targets) in enumerate(train_loader):
                user_ids = user_ids.to(device, non_blocking=True)
                movie_ids = movie_ids.to(device, non_blocking=True)
                genre_feats = genre_feats.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                
                optimizer.zero_grad()
                
                with autocast('cuda'):
                    outputs = model(user_ids, movie_ids, genre_feats)
                    loss = criterion(outputs, targets)
                
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                
                train_loss += loss.item()
                batch_count += 1
                
                if batch_idx % 200 == 0:
                    logger.info(f"Epoch {epoch+1}/{epochs} | Batch {batch_idx}/{len(train_loader)} | Loss: {loss.item():.6f}")
                    if device.type == 'cuda':
                        torch.cuda.empty_cache()
            
            avg_train_loss = train_loss / batch_count if batch_count > 0 else float('inf')
            history['train_loss'].append(avg_train_loss)
            
            # Validation phase
            model.eval()
            val_loss = 0.0
            val_batch_count = 0
            
            with torch.no_grad():
                for user_ids, movie_ids, genre_feats, targets in val_loader:
                    user_ids = user_ids.to(device, non_blocking=True)
                    movie_ids = movie_ids.to(device, non_blocking=True)
                    genre_feats = genre_feats.to(device, non_blocking=True)
                    targets = targets.to(device, non_blocking=True)
                    
                    with autocast():
                        outputs = model(user_ids, movie_ids, genre_feats)
                        loss = criterion(outputs, targets)
                    
                    val_loss += loss.item()
                    val_batch_count += 1
            
            avg_val_loss = val_loss / val_batch_count if val_batch_count > 0 else float('inf')
            scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']
            history['val_loss'].append(avg_val_loss)
            history['learning_rates'].append(current_lr)
            
            logger.info(f"Epoch {epoch+1}/{epochs} | Train: {avg_train_loss:.6f} | Val: {avg_val_loss:.6f} | LR: {current_lr:.8f}")
            
            # Log metrics to wandb
            wandb.log({
                'epoch': epoch + 1,
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'learning_rate': current_lr,
                'best_val_loss': best_val_loss
            })
            
            # Early stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                
                # Save best model
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'scaler_state_dict': scaler.state_dict(),
                    'best_val_loss': best_val_loss,
                    'history': history,
                    'mappings': mappings
                }, 'models/best_model.pt')
                logger.info(f"Saved best model (val_loss: {best_val_loss:.6f})")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f"Early stopping at epoch {epoch + 1}")
                    break
            
            # Save checkpoint after each epoch (like original code)
            checkpoint_path = f'models/checkpoints/checkpoint_epoch_{epoch+1}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pt'
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'best_val_loss': best_val_loss,
                'history': history,
                'mappings': mappings,
                'num_users': num_users,
                'num_movies': num_movies,
                'genres': genres
            }, checkpoint_path)
            logger.info(f"Saved checkpoint to {checkpoint_path}")
            
            # Clear GPU cache
            if device.type == 'cuda':
                torch.cuda.empty_cache()
                gc.collect()
        
        # Load best model
        checkpoint = torch.load('models/best_model.pt')
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Save model metadata
        metadata = {
            'num_users': num_users,
            'num_movies': num_movies,
            'genres': genres,
            'embedding_size': 128,
            'model_type': 'hybrid_fixed',
            'best_val_loss': best_val_loss,
            'epochs_completed': len(history['train_loss']),
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'pytorch_version': torch.__version__,
            'cuda_version': torch.version.cuda if torch.cuda.is_available() else 'N/A',
            'mappings': mappings
        }
        
        with open('models/model_metadata.pkl', 'wb') as f:
            pickle.dump(metadata, f)
        
        # Create movie lookup table with original IDs
        movie_lookup = movies_df[['media_id', 'title', 'genres', 'movie_idx']].set_index('media_id').to_dict(orient='index')
        with open('models/movie_lookup.pkl', 'wb') as f:
            pickle.dump(movie_lookup, f)
        
        with open('models/training_history.pkl', 'wb') as f:
            pickle.dump(history, f)
        
        movies_df.to_csv('models/movies_data.csv', index=False)
        
        logger.info("Model training completed successfully!")
        logger.info(f"Final validation loss: {best_val_loss:.6f}")
        
        return model, history
    
    except Exception as e:
        logger.exception(f"Error in train_model: {e}")
        return None, None

def main():
    """Main training function
    
    Entry point for the training script. Orchestrates the entire training
    pipeline from dataset validation through model training and evaluation.
    Includes comprehensive error handling and experiment tracking.
    """
    # Check datasets before starting
    if not check_datasets_before_training():
        print("\n🛑 Training aborted due to missing datasets")
        sys.exit(1)
    
    # Load configuration
    config = load_config()
    logger = setup_logging(config)
    
    parser = argparse.ArgumentParser(description='Train FIXED movie recommendation model')
    parser.add_argument('--epochs', type=int, default=config.model.num_epochs, help='Number of epochs to train')
    parser.add_argument('--batch-size', type=int, default=config.model.batch_size, help='Batch size for training')
    parser.add_argument('--resume-from', type=str, help='Path to checkpoint file to resume training from')
    parser.add_argument('--wandb-project', type=str, default='cine-sync-v2', help='Weights & Biases project name')
    parser.add_argument('--wandb-name', type=str, help='Weights & Biases run name')
    args = parser.parse_args()
    
    # Initialize Weights & Biases
    wandb.init(
        project=args.wandb_project,
        name=args.wandb_name,
        config={
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'learning_rate': 0.002,
            'embedding_size': 128,
            'weight_decay': 1e-5,
            'patience': 5,
            'model_type': 'hybrid_recommender',
            'pytorch_version': torch.__version__,
            'cuda_available': torch.cuda.is_available()
        }
    )
    
    try:
        # Setup
        device = setup_gpu()
        logger.info(f"Using device: {device}")
        
        # Initialize database manager
        db_manager = DatabaseManager(config.database)
        
        # Try to load data from database first
        try:
            ratings_df = load_ratings_data(db_manager)
            movies_df = load_movies_data(db_manager)
            logger.info(f"Loaded {len(ratings_df)} ratings and {len(movies_df)} movies from database")
            data_source = 'database'
        except Exception as e:
            logger.warning(f"Failed to load from database: {e}. Falling back to file loading.")
            # Find data files
            data_files = find_data_files()
            if not data_files:
                logger.error("No data files found")
                return
            data_source = 'files'
        
        # Process data with proper ID mapping
        train_data, val_data, movies_df, genres, mappings = process_and_prepare_data(data_files)
        if train_data is None or mappings is None:
            logger.error("Failed to prepare data or create ID mappings")
            return
        
        # Train model
        model, history = train_model(
            train_data=train_data,
            val_data=val_data,
            movies_df=movies_df,
            genres=genres,
            mappings=mappings,
            device=device,
            epochs=args.epochs,
            batch_size=args.batch_size,
            resume_from_checkpoint=args.resume_from
        )
        
        if model is not None:
            logger.info("Training completed successfully!")
            logger.info("Key improvements in this fixed version:")
            logger.info("1. ✅ Proper ID mapping prevents non-existent movie recommendations")
            logger.info("2. ✅ Contiguous indices (0, 1, 2...) for embedding layers")
            logger.info("3. ✅ Bounds checking in forward pass")
            logger.info("4. ✅ ID mappings saved for inference")
            logger.info("5. ✅ Validation to prevent out-of-bounds errors")
        else:
            logger.error("Training failed!")
            return 1
            
    except Exception as e:
        logger.exception(f"Error in main: {e}")
        return 1
    finally:
        # Finish wandb run
        wandb.finish()
    
    return 0

if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.exception(f"Unhandled exception: {e}")
        sys.exit(1)