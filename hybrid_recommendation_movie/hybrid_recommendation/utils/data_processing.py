"""
Simplified data processing utilities
"""
import pandas as pd
import numpy as np
import os
import logging
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

logger = logging.getLogger(__name__)


def load_csv_data(file_path, format_type='auto'):
    """Load and process CSV data with automatic format detection"""
    logger.info(f"Loading {file_path}")
    
    try:
        # Handle .dat files
        if file_path.endswith('.dat'):
            if 'movies.dat' in file_path:
                return pd.read_csv(file_path, sep='::', names=['movieId', 'title', 'genres'], engine='python')
            elif 'ratings.dat' in file_path:
                return pd.read_csv(file_path, sep='::', names=['userId', 'movieId', 'rating', 'timestamp'], engine='python')
        
        # Handle large files in chunks
        if os.path.getsize(file_path) > 100 * 1024 * 1024:  # 100MB
            chunks = pd.read_csv(file_path, chunksize=250000)
            return pd.concat(chunks)
        
        return pd.read_csv(file_path)
    
    except Exception as e:
        logger.error(f"Error loading {file_path}: {e}")
        return None


def prepare_genre_features(movies_df):
    """Extract and encode genre features"""
    logger.info("Processing genre features")
    
    # Extract all unique genres
    all_genres = set()
    for genres in movies_df['genres'].dropna():
        if pd.notnull(genres):
            all_genres.update(genres.split('|'))
    
    genres_list = sorted(list(all_genres))
    logger.info(f"Found {len(genres_list)} genres")
    
    # One-hot encode genres
    for genre in genres_list:
        movies_df[genre] = movies_df['genres'].apply(
            lambda x: 1 if pd.notnull(x) and genre in x.split('|') else 0
        )
    
    return movies_df, genres_list


def create_train_val_split(data, test_size=0.1):
    """Create training and validation splits with rating scaling"""
    logger.info("Creating train/validation split")
    
    # Adjust test size for large datasets
    if len(data) > 1000000:
        test_size = 0.05
    
    train_data, val_data = train_test_split(data, test_size=test_size, random_state=42)
    
    # Scale ratings
    scaler = MinMaxScaler()
    train_data = train_data.copy()
    val_data = val_data.copy()
    
    train_data['rating_scaled'] = scaler.fit_transform(train_data[['rating']])
    val_data['rating_scaled'] = scaler.transform(val_data[['rating']])
    
    logger.info(f"Split: {len(train_data)} train, {len(val_data)} validation")
    return train_data, val_data, scaler


def find_data_files(search_dirs=None):
    """Find data files in common locations"""
    if search_dirs is None:
        search_dirs = [os.getcwd(), os.path.join(os.getcwd(), "data")]
    
    patterns = {
        "ml_movies": ["ml-32m/movies.csv", "ml-25m/movies.csv", "ml-1m/movies.dat"],
        "ml_ratings": ["ml-32m/ratings.csv", "ml-25m/ratings.csv", "ml-1m/ratings.dat"],
    }
    
    found = {}
    for data_type, file_patterns in patterns.items():
        for pattern in file_patterns:
            for search_dir in search_dirs:
                path = os.path.join(search_dir, pattern)
                if os.path.exists(path):
                    found[data_type] = path
                    logger.info(f"Found {data_type} at {path}")
                    break
            if data_type in found:
                break
    
    return found