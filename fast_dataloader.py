#!/usr/bin/env python3
"""
Fast Data Loader for CineSync v2 - Rust Backend with Python Fallback
Optimized for movie and TV show recommendation datasets with Windows compatibility
"""

import os
import sys
import time
import threading
import queue
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any, Iterator
from concurrent.futures import ThreadPoolExecutor
import logging

# Try to import Rust dataloader
try:
    import cine_sync_dataloader
    RUST_AVAILABLE = True
    print("🚀 Rust dataloader available - using high-performance backend")
except ImportError:
    RUST_AVAILABLE = False
    print("⚠️  Rust dataloader not available - using optimized Python fallback")

logger = logging.getLogger(__name__)

@dataclass
class DataLoaderConfig:
    """Configuration for CineSync data loader."""
    batch_size: int = 32
    shuffle: bool = True
    buffer_size: int = 64
    num_prefetch_threads: int = 4
    use_rust: bool = True
    max_samples: Optional[int] = None
    include_movies: bool = True
    include_tv_shows: bool = True

class PythonDataLoader:
    """Optimized Python fallback data loader for CineSync."""
    
    def __init__(self, config: DataLoaderConfig):
        self.config = config
        self.movie_data = []
        self.tv_data = []
        self.combined_data = []
        self.current_epoch = 0
        self.performance_stats = {}
        
    def load_movies_csv(self, file_path: str) -> int:
        """Load movie ratings from CSV file."""
        try:
            df = pd.read_csv(file_path)
            
            # Flexible column mapping for different movie datasets
            column_mapping = self._detect_movie_columns(df.columns)
            
            if not column_mapping:
                raise ValueError(f"Could not detect required columns in {file_path}")
            
            movies = []
            for _, row in df.iterrows():
                try:
                    movie = {
                        'user_id': int(row[column_mapping['user_id']]),
                        'movie_id': int(row[column_mapping['movie_id']]),
                        'rating': float(row[column_mapping['rating']]),
                        'timestamp': row.get(column_mapping.get('timestamp'), None),
                        'title': str(row.get(column_mapping.get('title', ''), 'Unknown')),
                        'genres': str(row.get(column_mapping.get('genres', ''), '')).split('|'),
                        'year': int(row[column_mapping['year']]) if column_mapping.get('year') and pd.notna(row.get(column_mapping['year'])) else None,
                        'content_type': 0  # 0 for movies
                    }
                    movies.append(movie)
                except (ValueError, KeyError) as e:
                    continue  # Skip malformed rows
            
            self.movie_data = movies
            logger.info(f"Loaded {len(movies)} movie ratings from {file_path}")
            return len(movies)
            
        except Exception as e:
            logger.error(f"Failed to load movies from {file_path}: {e}")
            return 0
    
    def load_tv_shows_csv(self, file_path: str) -> int:
        """Load TV show ratings from CSV file."""
        try:
            df = pd.read_csv(file_path)
            
            # Flexible column mapping for different TV datasets
            column_mapping = self._detect_tv_columns(df.columns)
            
            if not column_mapping:
                raise ValueError(f"Could not detect required columns in {file_path}")
            
            shows = []
            for _, row in df.iterrows():
                try:
                    show = {
                        'user_id': int(row[column_mapping['user_id']]),
                        'show_id': int(row[column_mapping['show_id']]),
                        'rating': float(row[column_mapping['rating']]),
                        'timestamp': row.get(column_mapping.get('timestamp'), None),
                        'title': str(row.get(column_mapping.get('title', ''), 'Unknown')),
                        'genres': str(row.get(column_mapping.get('genres', ''), '')).split('|'),
                        'year': int(row[column_mapping['year']]) if column_mapping.get('year') and pd.notna(row.get(column_mapping['year'])) else None,
                        'seasons': int(row[column_mapping['seasons']]) if column_mapping.get('seasons') and pd.notna(row.get(column_mapping['seasons'])) else None,
                        'content_type': 1  # 1 for TV shows
                    }
                    shows.append(show)
                except (ValueError, KeyError) as e:
                    continue  # Skip malformed rows
            
            self.tv_data = shows
            logger.info(f"Loaded {len(shows)} TV show ratings from {file_path}")
            return len(shows)
            
        except Exception as e:
            logger.error(f"Failed to load TV shows from {file_path}: {e}")
            return 0
    
    def _detect_movie_columns(self, columns) -> Dict[str, str]:
        """Detect column names for movie datasets."""
        column_mapping = {}
        columns_lower = [col.lower() for col in columns]
        
        # User ID detection
        for user_col in ['userid', 'user_id', 'user', 'uid']:
            if user_col in columns_lower:
                column_mapping['user_id'] = columns[columns_lower.index(user_col)]
                break
        
        # Movie ID detection  
        for movie_col in ['movieid', 'movie_id', 'item_id', 'itemid', 'movie', 'mid']:
            if movie_col in columns_lower:
                column_mapping['movie_id'] = columns[columns_lower.index(movie_col)]
                break
        
        # Rating detection
        for rating_col in ['rating', 'score', 'rate']:
            if rating_col in columns_lower:
                column_mapping['rating'] = columns[columns_lower.index(rating_col)]
                break
        
        # Optional columns
        for timestamp_col in ['timestamp', 'time', 'date']:
            if timestamp_col in columns_lower:
                column_mapping['timestamp'] = columns[columns_lower.index(timestamp_col)]
                break
                
        for title_col in ['title', 'name', 'movie_title']:
            if title_col in columns_lower:
                column_mapping['title'] = columns[columns_lower.index(title_col)]
                break
                
        for genres_col in ['genres', 'genre', 'categories']:
            if genres_col in columns_lower:
                column_mapping['genres'] = columns[columns_lower.index(genres_col)]
                break
                
        for year_col in ['year', 'release_year', 'movie_year']:
            if year_col in columns_lower:
                column_mapping['year'] = columns[columns_lower.index(year_col)]
                break
        
        # Require minimum essential columns
        if all(key in column_mapping for key in ['user_id', 'movie_id', 'rating']):
            return column_mapping
        else:
            return {}
    
    def _detect_tv_columns(self, columns) -> Dict[str, str]:
        """Detect column names for TV show datasets."""
        column_mapping = {}
        columns_lower = [col.lower() for col in columns]
        
        # User ID detection
        for user_col in ['userid', 'user_id', 'user', 'uid']:
            if user_col in columns_lower:
                column_mapping['user_id'] = columns[columns_lower.index(user_col)]
                break
        
        # Show ID detection  
        for show_col in ['showid', 'show_id', 'series_id', 'tv_id', 'item_id', 'itemid']:
            if show_col in columns_lower:
                column_mapping['show_id'] = columns[columns_lower.index(show_col)]
                break
        
        # Rating detection
        for rating_col in ['rating', 'score', 'rate']:
            if rating_col in columns_lower:
                column_mapping['rating'] = columns[columns_lower.index(rating_col)]
                break
        
        # Optional columns
        for timestamp_col in ['timestamp', 'time', 'date']:
            if timestamp_col in columns_lower:
                column_mapping['timestamp'] = columns[columns_lower.index(timestamp_col)]
                break
                
        for title_col in ['title', 'name', 'show_title', 'series_title']:
            if title_col in columns_lower:
                column_mapping['title'] = columns[columns_lower.index(title_col)]
                break
                
        for genres_col in ['genres', 'genre', 'categories']:
            if genres_col in columns_lower:
                column_mapping['genres'] = columns[columns_lower.index(genres_col)]
                break
                
        for year_col in ['year', 'release_year', 'first_air_date']:
            if year_col in columns_lower:
                column_mapping['year'] = columns[columns_lower.index(year_col)]
                break
                
        for seasons_col in ['seasons', 'season_count', 'num_seasons']:
            if seasons_col in columns_lower:
                column_mapping['seasons'] = columns[columns_lower.index(seasons_col)]
                break
        
        # Require minimum essential columns
        if all(key in column_mapping for key in ['user_id', 'show_id', 'rating']):
            return column_mapping
        else:
            return {}
    
    def prepare_combined_data(self):
        """Combine and prepare all data for training."""
        self.combined_data = []
        
        if self.config.include_movies:
            self.combined_data.extend(self.movie_data)
            
        if self.config.include_tv_shows:
            self.combined_data.extend(self.tv_data)
        
        # Shuffle if requested
        if self.config.shuffle:
            np.random.shuffle(self.combined_data)
        
        # Limit samples if requested
        if self.config.max_samples:
            self.combined_data = self.combined_data[:self.config.max_samples]
    
    def create_batches(self) -> List[List[List[float]]]:
        """Create batches from the combined data."""
        if not self.combined_data:
            self.prepare_combined_data()
        
        batches = []
        for i in range(0, len(self.combined_data), self.config.batch_size):
            batch_data = self.combined_data[i:i + self.config.batch_size]
            
            # Convert to feature vectors
            batch = []
            for item in batch_data:
                features = [
                    float(item['user_id']),
                    float(item.get('movie_id', item.get('show_id', 0))),
                    float(item['rating']),
                    float(item['content_type']),
                    float(item.get('year', 0) or 0),
                    float(len(item.get('genres', []))),
                ]
                batch.append(features)
            
            batches.append(batch)
        
        return batches
    
    def get_batch_count(self) -> int:
        """Get the number of batches."""
        if not self.combined_data:
            self.prepare_combined_data()
        return (len(self.combined_data) + self.config.batch_size - 1) // self.config.batch_size
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        return {
            'total_samples': len(self.combined_data),
            'movie_samples': len(self.movie_data),
            'tv_samples': len(self.tv_data),
            'batch_size': self.config.batch_size,
            'current_epoch': self.current_epoch,
            'total_batches': self.get_batch_count(),
        }
    
    def reset_epoch(self):
        """Reset for next epoch."""
        self.current_epoch += 1
        if self.config.shuffle:
            np.random.shuffle(self.combined_data)

class CineSyncFastDataLoader:
    """High-performance data loader with Rust backend and Python fallback."""
    
    def __init__(self, config: DataLoaderConfig = None):
        self.config = config or DataLoaderConfig()
        self.rust_loader = None
        self.python_loader = None
        self.using_rust = False
        
        # Initialize the appropriate backend
        if self.config.use_rust and RUST_AVAILABLE:
            try:
                self.rust_loader = cine_sync_dataloader.CineSyncDataLoader(
                    self.config.batch_size,
                    self.config.shuffle,
                    self.config.buffer_size
                )
                self.using_rust = True
                logger.info("🚀 Using Rust backend for data loading")
            except Exception as e:
                logger.warning(f"Failed to initialize Rust loader: {e}")
                self.using_rust = False
        
        if not self.using_rust:
            self.python_loader = PythonDataLoader(self.config)
            logger.info("🐍 Using optimized Python backend for data loading")
    
    def load_datasets(self, data_paths: Dict[str, str]) -> Dict[str, int]:
        """Load multiple datasets from specified paths."""
        load_counts = {}
        
        # Load movie datasets
        if self.config.include_movies and 'movies' in data_paths:
            movie_paths = data_paths['movies']
            if isinstance(movie_paths, str):
                movie_paths = [movie_paths]
            
            total_movies = 0
            for path in movie_paths:
                if os.path.exists(path):
                    if self.using_rust:
                        count = self.rust_loader.load_movies_csv(path)
                    else:
                        count = self.python_loader.load_movies_csv(path)
                    total_movies += count
                    logger.info(f"Loaded {count} movie ratings from {path}")
            
            load_counts['movies'] = total_movies
        
        # Load TV show datasets
        if self.config.include_tv_shows and 'tv_shows' in data_paths:
            tv_paths = data_paths['tv_shows']
            if isinstance(tv_paths, str):
                tv_paths = [tv_paths]
            
            total_tv = 0
            for path in tv_paths:
                if os.path.exists(path):
                    if self.using_rust:
                        count = self.rust_loader.load_tv_shows_csv(path)
                    else:
                        count = self.python_loader.load_tv_shows_csv(path)
                    total_tv += count
                    logger.info(f"Loaded {count} TV show ratings from {path}")
            
            load_counts['tv_shows'] = total_tv
        
        return load_counts
    
    def create_batches(self) -> List[List[List[float]]]:
        """Create training batches."""
        if self.using_rust:
            return self.rust_loader.create_batches()
        else:
            return self.python_loader.create_batches()
    
    def get_batch_count(self) -> int:
        """Get the number of batches."""
        if self.using_rust:
            return self.rust_loader.get_batch_count()
        else:
            return self.python_loader.get_batch_count()
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        if self.using_rust:
            return self.rust_loader.get_performance_stats()
        else:
            return self.python_loader.get_performance_stats()
    
    def reset_epoch(self):
        """Reset for next epoch."""
        if self.using_rust:
            self.rust_loader.reset_epoch()
        else:
            self.python_loader.reset_epoch()
    
    def get_data_summary(self) -> Dict[str, Any]:
        """Get summary of loaded data."""
        if self.using_rust:
            return self.rust_loader.get_data_summary()
        else:
            # Python implementation
            movie_count = len(self.python_loader.movie_data)
            tv_count = len(self.python_loader.tv_data)
            
            return {
                'movies': {
                    'total_ratings': movie_count,
                    'unique_users': len(set(m['user_id'] for m in self.python_loader.movie_data)),
                    'unique_items': len(set(m['movie_id'] for m in self.python_loader.movie_data)),
                    'avg_rating': np.mean([m['rating'] for m in self.python_loader.movie_data]) if movie_count > 0 else 0,
                },
                'tv_shows': {
                    'total_ratings': tv_count,
                    'unique_users': len(set(t['user_id'] for t in self.python_loader.tv_data)),
                    'unique_items': len(set(t['show_id'] for t in self.python_loader.tv_data)),
                    'avg_rating': np.mean([t['rating'] for t in self.python_loader.tv_data]) if tv_count > 0 else 0,
                }
            }

def create_optimized_dataloader(
    data_paths: Dict[str, str],
    batch_size: int = 32,
    shuffle: bool = True,
    use_rust: bool = True,
    max_samples: Optional[int] = None
) -> CineSyncFastDataLoader:
    """
    Create an optimized data loader for CineSync training.
    
    Args:
        data_paths: Dictionary with 'movies' and/or 'tv_shows' keys containing file paths
        batch_size: Training batch size
        shuffle: Whether to shuffle data
        use_rust: Whether to use Rust backend (if available)
        max_samples: Maximum number of samples to use (for testing)
    
    Returns:
        Configured CineSyncFastDataLoader instance
    """
    config = DataLoaderConfig(
        batch_size=batch_size,
        shuffle=shuffle,
        use_rust=use_rust,
        max_samples=max_samples
    )
    
    loader = CineSyncFastDataLoader(config)
    load_counts = loader.load_datasets(data_paths)
    
    logger.info(f"Data loader initialized with {load_counts}")
    logger.info(f"Backend: {'Rust' if loader.using_rust else 'Python'}")
    
    return loader

if __name__ == "__main__":
    # Example usage
    data_paths = {
        'movies': 'movies/ml-32m/ratings.csv',
        'tv_shows': 'tv/tmdb/TMDB_tv_dataset_v3.csv'
    }
    
    loader = create_optimized_dataloader(
        data_paths=data_paths,
        batch_size=64,
        shuffle=True,
        use_rust=True
    )
    
    print("Data Summary:", loader.get_data_summary())
    print("Performance Stats:", loader.get_performance_stats())
    
    # Benchmark
    start_time = time.time()
    batches = loader.create_batches()
    end_time = time.time()
    
    print(f"Created {len(batches)} batches in {end_time - start_time:.3f}s")
    print(f"Performance: {len(batches) / (end_time - start_time):.1f} batches/sec")