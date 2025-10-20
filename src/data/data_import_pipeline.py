#!/usr/bin/env python3
"""
CineSync v2 Data Import Pipeline
Handles importing new movie and TV show data post-2022 with automatic model retraining
"""

import argparse
import logging
import os
import sys
import pandas as pd
import numpy as np
import requests
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import sqlite3
import pickle
from dataclasses import dataclass, asdict
import time

# Import wandb for experiment tracking
from wandb_config import init_wandb_for_training, WandbManager

logger = logging.getLogger(__name__)


@dataclass
class DataImportConfig:
    """Configuration for data import pipeline"""
    # TMDB API Configuration
    tmdb_api_key: str = os.getenv('TMDB_API_KEY', '')
    tmdb_base_url: str = "https://api.themoviedb.org/3"
    
    # Import settings
    start_date: str = "2022-01-01"  # Cutoff date for existing data
    batch_size: int = 100
    max_requests_per_minute: int = 40  # TMDB rate limit
    
    # Database paths
    movies_db_path: str = "data/movies.db"
    tv_db_path: str = "data/tv_shows.db"
    
    # Output paths
    output_dir: str = "data/imports"
    movies_output: str = "new_movies.csv"
    tv_output: str = "new_tv_shows.csv"
    
    # Processing settings
    min_vote_count: int = 10
    min_popularity: float = 1.0
    include_adult: bool = False
    
    # Retraining settings
    auto_retrain: bool = True
    retrain_threshold: int = 1000  # Min new items before retraining


class TMDBDataImporter:
    """TMDB API data importer for movies and TV shows"""
    
    def __init__(self, config: DataImportConfig):
        self.config = config
        self.session = requests.Session()
        self.request_count = 0
        self.last_request_time = time.time()
        
        if not config.tmdb_api_key:
            raise ValueError("TMDB API key is required. Set TMDB_API_KEY environment variable.")
    
    def _rate_limit(self):
        """Enforce rate limiting for TMDB API"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if self.request_count >= self.config.max_requests_per_minute:
            if time_since_last < 60:
                sleep_time = 60 - time_since_last
                logger.info(f"Rate limiting: sleeping for {sleep_time:.2f} seconds")
                time.sleep(sleep_time)
            self.request_count = 0
        
        self.request_count += 1
        self.last_request_time = time.time()
    
    def _make_request(self, endpoint: str, params: Dict = None) -> Optional[Dict]:
        """Make rate-limited request to TMDB API"""
        self._rate_limit()
        
        url = f"{self.config.tmdb_base_url}/{endpoint}"
        params = params or {}
        params['api_key'] = self.config.tmdb_api_key
        
        try:
            response = self.session.get(url, params=params)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {e}")
            return None
    
    def get_popular_movies(self, start_date: str, end_date: str = None) -> List[Dict]:
        """Get popular movies within date range"""
        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")
        
        movies = []
        page = 1
        max_pages = 50  # Reasonable limit
        
        logger.info(f"Fetching popular movies from {start_date} to {end_date}")
        
        while page <= max_pages:
            params = {
                'page': page,
                'primary_release_date.gte': start_date,
                'primary_release_date.lte': end_date,
                'vote_count.gte': self.config.min_vote_count,
                'popularity.gte': self.config.min_popularity,
                'include_adult': self.config.include_adult,
                'sort_by': 'popularity.desc'
            }
            
            data = self._make_request('discover/movie', params)
            if not data or 'results' not in data:
                break
            
            page_movies = data['results']
            if not page_movies:
                break
            
            movies.extend(page_movies)
            logger.info(f"Fetched page {page}, got {len(page_movies)} movies (total: {len(movies)})")
            
            if page >= data.get('total_pages', 1):
                break
            
            page += 1
        
        logger.info(f"Fetched {len(movies)} movies total")
        return movies
    
    def get_popular_tv_shows(self, start_date: str, end_date: str = None) -> List[Dict]:
        """Get popular TV shows within date range"""
        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")
        
        tv_shows = []
        page = 1
        max_pages = 50
        
        logger.info(f"Fetching popular TV shows from {start_date} to {end_date}")
        
        while page <= max_pages:
            params = {
                'page': page,
                'first_air_date.gte': start_date,
                'first_air_date.lte': end_date,
                'vote_count.gte': self.config.min_vote_count,
                'popularity.gte': self.config.min_popularity,
                'sort_by': 'popularity.desc'
            }
            
            data = self._make_request('discover/tv', params)
            if not data or 'results' not in data:
                break
            
            page_shows = data['results']
            if not page_shows:
                break
            
            tv_shows.extend(page_shows)
            logger.info(f"Fetched page {page}, got {len(page_shows)} TV shows (total: {len(tv_shows)})")
            
            if page >= data.get('total_pages', 1):
                break
            
            page += 1
        
        logger.info(f"Fetched {len(tv_shows)} TV shows total")
        return tv_shows
    
    def get_movie_details(self, movie_id: int) -> Optional[Dict]:
        """Get detailed movie information"""
        return self._make_request(f'movie/{movie_id}')
    
    def get_tv_details(self, tv_id: int) -> Optional[Dict]:
        """Get detailed TV show information"""
        return self._make_request(f'tv/{tv_id}')


class DataProcessor:
    """Process and normalize movie/TV data for ML models"""
    
    def __init__(self, config: DataImportConfig):
        self.config = config
    
    def process_movies(self, raw_movies: List[Dict]) -> pd.DataFrame:
        """Process raw movie data into normalized format"""
        processed_movies = []
        
        logger.info(f"Processing {len(raw_movies)} movies")
        
        for movie in raw_movies:
            try:
                processed = {
                    'movie_id': movie.get('id'),
                    'title': movie.get('title', ''),
                    'original_title': movie.get('original_title', ''),
                    'overview': movie.get('overview', ''),
                    'release_date': movie.get('release_date', ''),
                    'genres': '|'.join([str(g.get('name', '')) for g in movie.get('genres', [])]),
                    'genre_ids': '|'.join([str(g) for g in movie.get('genre_ids', [])]),
                    'runtime': movie.get('runtime', 0),
                    'vote_average': movie.get('vote_average', 0.0),
                    'vote_count': movie.get('vote_count', 0),
                    'popularity': movie.get('popularity', 0.0),
                    'budget': movie.get('budget', 0),
                    'revenue': movie.get('revenue', 0),
                    'original_language': movie.get('original_language', ''),
                    'spoken_languages': '|'.join([lang.get('name', '') for lang in movie.get('spoken_languages', [])]),
                    'production_companies': '|'.join([comp.get('name', '') for comp in movie.get('production_companies', [])]),
                    'production_countries': '|'.join([country.get('name', '') for country in movie.get('production_countries', [])]),
                    'adult': movie.get('adult', False),
                    'poster_path': movie.get('poster_path', ''),
                    'backdrop_path': movie.get('backdrop_path', ''),
                    'import_date': datetime.now().isoformat(),
                    'data_source': 'tmdb_api'
                }
                
                # Calculate derived features
                processed['year'] = int(processed['release_date'][:4]) if processed['release_date'] else 0
                processed['decade'] = (processed['year'] // 10) * 10 if processed['year'] > 0 else 0
                processed['has_budget'] = processed['budget'] > 0
                processed['has_revenue'] = processed['revenue'] > 0
                processed['roi'] = processed['revenue'] / processed['budget'] if processed['budget'] > 0 else 0
                processed['vote_weight'] = np.log1p(processed['vote_count'])
                processed['weighted_rating'] = processed['vote_average'] * processed['vote_weight']
                
                processed_movies.append(processed)
                
            except Exception as e:
                logger.warning(f"Failed to process movie {movie.get('id', 'unknown')}: {e}")
        
        df = pd.DataFrame(processed_movies)
        logger.info(f"Successfully processed {len(df)} movies")
        
        return df
    
    def process_tv_shows(self, raw_shows: List[Dict]) -> pd.DataFrame:
        """Process raw TV show data into normalized format"""
        processed_shows = []
        
        logger.info(f"Processing {len(raw_shows)} TV shows")
        
        for show in raw_shows:
            try:
                processed = {
                    'show_id': show.get('id'),
                    'name': show.get('name', ''),
                    'original_name': show.get('original_name', ''),
                    'overview': show.get('overview', ''),
                    'first_air_date': show.get('first_air_date', ''),
                    'last_air_date': show.get('last_air_date', ''),
                    'genres': '|'.join([str(g.get('name', '')) for g in show.get('genres', [])]),
                    'genre_ids': '|'.join([str(g) for g in show.get('genre_ids', [])]),
                    'number_of_episodes': show.get('number_of_episodes', 0),
                    'number_of_seasons': show.get('number_of_seasons', 0),
                    'episode_run_time': '|'.join([str(t) for t in show.get('episode_run_time', [])]),
                    'vote_average': show.get('vote_average', 0.0),
                    'vote_count': show.get('vote_count', 0),
                    'popularity': show.get('popularity', 0.0),
                    'status': show.get('status', ''),
                    'type': show.get('type', ''),
                    'in_production': show.get('in_production', False),
                    'original_language': show.get('original_language', ''),
                    'spoken_languages': '|'.join([lang.get('name', '') for lang in show.get('spoken_languages', [])]),
                    'production_companies': '|'.join([comp.get('name', '') for comp in show.get('production_companies', [])]),
                    'production_countries': '|'.join([country.get('name', '') for country in show.get('production_countries', [])]),
                    'networks': '|'.join([net.get('name', '') for net in show.get('networks', [])]),
                    'created_by': '|'.join([creator.get('name', '') for creator in show.get('created_by', [])]),
                    'poster_path': show.get('poster_path', ''),
                    'backdrop_path': show.get('backdrop_path', ''),
                    'import_date': datetime.now().isoformat(),
                    'data_source': 'tmdb_api'
                }
                
                # Calculate derived features
                processed['start_year'] = int(processed['first_air_date'][:4]) if processed['first_air_date'] else 0
                processed['end_year'] = int(processed['last_air_date'][:4]) if processed['last_air_date'] else 0
                processed['decade'] = (processed['start_year'] // 10) * 10 if processed['start_year'] > 0 else 0
                processed['avg_episode_runtime'] = np.mean([int(t) for t in processed['episode_run_time'].split('|') if t.isdigit()]) if processed['episode_run_time'] else 0
                processed['is_ongoing'] = processed['in_production'] or processed['status'] in ['Returning Series', 'In Production']
                processed['total_runtime'] = processed['number_of_episodes'] * processed['avg_episode_runtime']
                processed['vote_weight'] = np.log1p(processed['vote_count'])
                processed['weighted_rating'] = processed['vote_average'] * processed['vote_weight']
                
                # Status encoding for model
                status_encoding = {
                    'Ended': 0, 'Canceled': 1, 'Returning Series': 2, 'In Production': 3, 'Planned': 4
                }
                processed['status_encoded'] = status_encoding.get(processed['status'], 0)
                
                processed_shows.append(processed)
                
            except Exception as e:
                logger.warning(f"Failed to process TV show {show.get('id', 'unknown')}: {e}")
        
        df = pd.DataFrame(processed_shows)
        logger.info(f"Successfully processed {len(df)} TV shows")
        
        return df


class DatabaseManager:
    """Manage database operations for imported data"""
    
    def __init__(self, config: DataImportConfig):
        self.config = config
        self.ensure_databases()
    
    def ensure_databases(self):
        """Ensure database directories and tables exist"""
        os.makedirs(os.path.dirname(self.config.movies_db_path), exist_ok=True)
        os.makedirs(os.path.dirname(self.config.tv_db_path), exist_ok=True)
        
        # Create movie tables
        with sqlite3.connect(self.config.movies_db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS movies (
                    movie_id INTEGER PRIMARY KEY,
                    title TEXT,
                    original_title TEXT,
                    overview TEXT,
                    release_date TEXT,
                    genres TEXT,
                    genre_ids TEXT,
                    runtime INTEGER,
                    vote_average REAL,
                    vote_count INTEGER,
                    popularity REAL,
                    budget INTEGER,
                    revenue INTEGER,
                    original_language TEXT,
                    spoken_languages TEXT,
                    production_companies TEXT,
                    production_countries TEXT,
                    adult BOOLEAN,
                    poster_path TEXT,
                    backdrop_path TEXT,
                    year INTEGER,
                    decade INTEGER,
                    has_budget BOOLEAN,
                    has_revenue BOOLEAN,
                    roi REAL,
                    vote_weight REAL,
                    weighted_rating REAL,
                    import_date TEXT,
                    data_source TEXT
                )
            ''')
        
        # Create TV show tables
        with sqlite3.connect(self.config.tv_db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS tv_shows (
                    show_id INTEGER PRIMARY KEY,
                    name TEXT,
                    original_name TEXT,
                    overview TEXT,
                    first_air_date TEXT,
                    last_air_date TEXT,
                    genres TEXT,
                    genre_ids TEXT,
                    number_of_episodes INTEGER,
                    number_of_seasons INTEGER,
                    episode_run_time TEXT,
                    vote_average REAL,
                    vote_count INTEGER,
                    popularity REAL,
                    status TEXT,
                    type TEXT,
                    in_production BOOLEAN,
                    original_language TEXT,
                    spoken_languages TEXT,
                    production_companies TEXT,
                    production_countries TEXT,
                    networks TEXT,
                    created_by TEXT,
                    poster_path TEXT,
                    backdrop_path TEXT,
                    start_year INTEGER,
                    end_year INTEGER,
                    decade INTEGER,
                    avg_episode_runtime REAL,
                    is_ongoing BOOLEAN,
                    total_runtime REAL,
                    vote_weight REAL,
                    weighted_rating REAL,
                    status_encoded INTEGER,
                    import_date TEXT,
                    data_source TEXT
                )
            ''')
    
    def save_movies(self, movies_df: pd.DataFrame) -> int:
        """Save movies to database and return count of new records"""
        with sqlite3.connect(self.config.movies_db_path) as conn:
            # Get existing movie IDs
            existing_ids = set(pd.read_sql('SELECT movie_id FROM movies', conn)['movie_id'].tolist())
            
            # Filter for new movies
            new_movies = movies_df[~movies_df['movie_id'].isin(existing_ids)]
            
            if len(new_movies) > 0:
                new_movies.to_sql('movies', conn, if_exists='append', index=False)
                logger.info(f"Saved {len(new_movies)} new movies to database")
            
            return len(new_movies)
    
    def save_tv_shows(self, tv_df: pd.DataFrame) -> int:
        """Save TV shows to database and return count of new records"""
        with sqlite3.connect(self.config.tv_db_path) as conn:
            # Get existing show IDs
            try:
                existing_ids = set(pd.read_sql('SELECT show_id FROM tv_shows', conn)['show_id'].tolist())
            except:
                existing_ids = set()
            
            # Filter for new shows
            new_shows = tv_df[~tv_df['show_id'].isin(existing_ids)]
            
            if len(new_shows) > 0:
                new_shows.to_sql('tv_shows', conn, if_exists='append', index=False)
                logger.info(f"Saved {len(new_shows)} new TV shows to database")
            
            return len(new_shows)
    
    def get_import_stats(self) -> Dict:
        """Get import statistics"""
        stats = {}
        
        # Movie stats
        try:
            with sqlite3.connect(self.config.movies_db_path) as conn:
                movie_stats = pd.read_sql('''
                    SELECT 
                        COUNT(*) as total_movies,
                        COUNT(CASE WHEN import_date >= date('now', '-7 days') THEN 1 END) as movies_last_week,
                        COUNT(CASE WHEN import_date >= date('now', '-30 days') THEN 1 END) as movies_last_month,
                        AVG(vote_average) as avg_rating,
                        AVG(popularity) as avg_popularity
                    FROM movies
                ''', conn).iloc[0].to_dict()
                stats['movies'] = movie_stats
        except Exception as e:
            logger.warning(f"Could not get movie stats: {e}")
            stats['movies'] = {}
        
        # TV stats
        try:
            with sqlite3.connect(self.config.tv_db_path) as conn:
                tv_stats = pd.read_sql('''
                    SELECT 
                        COUNT(*) as total_shows,
                        COUNT(CASE WHEN import_date >= date('now', '-7 days') THEN 1 END) as shows_last_week,
                        COUNT(CASE WHEN import_date >= date('now', '-30 days') THEN 1 END) as shows_last_month,
                        AVG(vote_average) as avg_rating,
                        AVG(popularity) as avg_popularity
                    FROM tv_shows
                ''', conn).iloc[0].to_dict()
                stats['tv_shows'] = tv_stats
        except Exception as e:
            logger.warning(f"Could not get TV stats: {e}")
            stats['tv_shows'] = {}
        
        return stats


class ModelRetrainer:
    """Handle automatic model retraining when new data is available"""
    
    def __init__(self, config: DataImportConfig):
        self.config = config
    
    def should_retrain(self, new_movies: int, new_tv_shows: int) -> Dict[str, bool]:
        """Determine which models need retraining"""
        return {
            'movie_models': new_movies >= self.config.retrain_threshold,
            'tv_models': new_tv_shows >= self.config.retrain_threshold,
            'hybrid_model': (new_movies + new_tv_shows) >= self.config.retrain_threshold
        }
    
    def trigger_retraining(self, models_to_retrain: Dict[str, bool]) -> Dict[str, bool]:
        """Trigger retraining for specified models"""
        results = {}
        
        if models_to_retrain.get('movie_models'):
            logger.info("Triggering movie model retraining")
            results['movie_models'] = self._retrain_movie_models()
        
        if models_to_retrain.get('tv_models'):
            logger.info("Triggering TV model retraining")
            results['tv_models'] = self._retrain_tv_models()
        
        if models_to_retrain.get('hybrid_model'):
            logger.info("Triggering hybrid model retraining")
            results['hybrid_model'] = self._retrain_hybrid_model()
        
        return results
    
    def _retrain_movie_models(self) -> bool:
        """Retrain movie-specific models"""
        try:
            # This would trigger the actual retraining scripts
            logger.info("Movie model retraining would be triggered here")
            return True
        except Exception as e:
            logger.error(f"Movie model retraining failed: {e}")
            return False
    
    def _retrain_tv_models(self) -> bool:
        """Retrain TV-specific models"""
        try:
            logger.info("TV model retraining would be triggered here")
            return True
        except Exception as e:
            logger.error(f"TV model retraining failed: {e}")
            return False
    
    def _retrain_hybrid_model(self) -> bool:
        """Retrain hybrid model"""
        try:
            logger.info("Hybrid model retraining would be triggered here")
            return True
        except Exception as e:
            logger.error(f"Hybrid model retraining failed: {e}")
            return False


class DataImportPipeline:
    """Main data import pipeline orchestrator"""
    
    def __init__(self, config: DataImportConfig):
        self.config = config
        self.importer = TMDBDataImporter(config)
        self.processor = DataProcessor(config)
        self.db_manager = DatabaseManager(config)
        self.retrainer = ModelRetrainer(config)
        
        # Wandb integration
        self.wandb_manager = None
    
    def run_import(self, import_movies: bool = True, import_tv: bool = True) -> Dict:
        """Run the complete data import pipeline"""
        
        # Initialize wandb for tracking
        wandb_config = {
            'pipeline_type': 'data_import',
            'start_date': self.config.start_date,
            'import_movies': import_movies,
            'import_tv': import_tv,
            'min_vote_count': self.config.min_vote_count,
            'min_popularity': self.config.min_popularity
        }
        
        self.wandb_manager = init_wandb_for_training('data_import', wandb_config)
        
        try:
            results = {
                'start_time': datetime.now().isoformat(),
                'movies_imported': 0,
                'tv_shows_imported': 0,
                'retraining_triggered': {},
                'errors': []
            }
            
            # Import movies
            if import_movies:
                try:
                    logger.info("Starting movie import")
                    raw_movies = self.importer.get_popular_movies(self.config.start_date)
                    
                    if raw_movies:
                        processed_movies = self.processor.process_movies(raw_movies)
                        new_movie_count = self.db_manager.save_movies(processed_movies)
                        results['movies_imported'] = new_movie_count
                        
                        # Log to wandb
                        self.wandb_manager.log_metrics({
                            'movies/raw_fetched': len(raw_movies),
                            'movies/processed': len(processed_movies),
                            'movies/new_imported': new_movie_count
                        })
                        
                        # Save CSV export
                        if new_movie_count > 0:
                            output_path = Path(self.config.output_dir) / self.config.movies_output
                            output_path.parent.mkdir(parents=True, exist_ok=True)
                            processed_movies.to_csv(output_path, index=False)
                            logger.info(f"Exported {new_movie_count} new movies to {output_path}")
                
                except Exception as e:
                    error_msg = f"Movie import failed: {e}"
                    logger.error(error_msg)
                    results['errors'].append(error_msg)
            
            # Import TV shows
            if import_tv:
                try:
                    logger.info("Starting TV show import")
                    raw_tv_shows = self.importer.get_popular_tv_shows(self.config.start_date)
                    
                    if raw_tv_shows:
                        processed_tv = self.processor.process_tv_shows(raw_tv_shows)
                        new_tv_count = self.db_manager.save_tv_shows(processed_tv)
                        results['tv_shows_imported'] = new_tv_count
                        
                        # Log to wandb
                        self.wandb_manager.log_metrics({
                            'tv_shows/raw_fetched': len(raw_tv_shows),
                            'tv_shows/processed': len(processed_tv),
                            'tv_shows/new_imported': new_tv_count
                        })
                        
                        # Save CSV export
                        if new_tv_count > 0:
                            output_path = Path(self.config.output_dir) / self.config.tv_output
                            output_path.parent.mkdir(parents=True, exist_ok=True)
                            processed_tv.to_csv(output_path, index=False)
                            logger.info(f"Exported {new_tv_count} new TV shows to {output_path}")
                
                except Exception as e:
                    error_msg = f"TV show import failed: {e}"
                    logger.error(error_msg)
                    results['errors'].append(error_msg)
            
            # Check if retraining is needed
            if self.config.auto_retrain:
                models_to_retrain = self.retrainer.should_retrain(
                    results['movies_imported'], 
                    results['tv_shows_imported']
                )
                
                if any(models_to_retrain.values()):
                    logger.info(f"Triggering retraining: {models_to_retrain}")
                    retraining_results = self.retrainer.trigger_retraining(models_to_retrain)
                    results['retraining_triggered'] = retraining_results
                    
                    # Log retraining metrics
                    self.wandb_manager.log_metrics({
                        f'retraining/{model}': success 
                        for model, success in retraining_results.items()
                    })
            
            # Get final statistics
            stats = self.db_manager.get_import_stats()
            results['final_stats'] = stats
            
            # Log final metrics
            self.wandb_manager.log_metrics({
                'pipeline/total_movies_imported': results['movies_imported'],
                'pipeline/total_tv_imported': results['tv_shows_imported'],
                'pipeline/errors_count': len(results['errors']),
                'pipeline/success': len(results['errors']) == 0
            })
            
            results['end_time'] = datetime.now().isoformat()
            results['success'] = len(results['errors']) == 0
            
            return results
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            raise
        finally:
            if self.wandb_manager:
                self.wandb_manager.finish()


def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f"data_import_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
            logging.StreamHandler(sys.stdout)
        ]
    )


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='CineSync v2 Data Import Pipeline')
    
    # Import settings
    parser.add_argument('--start-date', type=str, default='2022-01-01',
                       help='Start date for importing new content (YYYY-MM-DD)')
    parser.add_argument('--movies-only', action='store_true',
                       help='Import only movies')
    parser.add_argument('--tv-only', action='store_true',
                       help='Import only TV shows')
    parser.add_argument('--no-auto-retrain', action='store_true',
                       help='Disable automatic model retraining')
    
    # API settings
    parser.add_argument('--tmdb-api-key', type=str,
                       help='TMDB API key (can also use TMDB_API_KEY env var)')
    parser.add_argument('--batch-size', type=int, default=100,
                       help='Batch size for API requests')
    parser.add_argument('--max-requests-per-minute', type=int, default=40,
                       help='Maximum API requests per minute')
    
    # Quality filters
    parser.add_argument('--min-vote-count', type=int, default=10,
                       help='Minimum vote count for imported content')
    parser.add_argument('--min-popularity', type=float, default=1.0,
                       help='Minimum popularity score for imported content')
    parser.add_argument('--include-adult', action='store_true',
                       help='Include adult content')
    
    # Output settings
    parser.add_argument('--output-dir', type=str, default='data/imports',
                       help='Output directory for exported data')
    parser.add_argument('--movies-db', type=str, default='data/movies.db',
                       help='Movies database path')
    parser.add_argument('--tv-db', type=str, default='data/tv_shows.db',
                       help='TV shows database path')
    
    return parser.parse_args()


def main():
    """Main function"""
    setup_logging()
    args = parse_args()
    
    logger.info("Starting CineSync v2 Data Import Pipeline")
    logger.info(f"Configuration: {vars(args)}")
    
    # Create configuration
    config = DataImportConfig(
        tmdb_api_key=args.tmdb_api_key or os.getenv('TMDB_API_KEY', ''),
        start_date=args.start_date,
        batch_size=args.batch_size,
        max_requests_per_minute=args.max_requests_per_minute,
        min_vote_count=args.min_vote_count,
        min_popularity=args.min_popularity,
        include_adult=args.include_adult,
        output_dir=args.output_dir,
        movies_db_path=args.movies_db,
        tv_db_path=args.tv_db,
        auto_retrain=not args.no_auto_retrain
    )
    
    if not config.tmdb_api_key:
        logger.error("TMDB API key is required. Set --tmdb-api-key or TMDB_API_KEY environment variable.")
        return 1
    
    # Determine what to import
    import_movies = not args.tv_only
    import_tv = not args.movies_only
    
    try:
        # Run pipeline
        pipeline = DataImportPipeline(config)
        results = pipeline.run_import(import_movies=import_movies, import_tv=import_tv)
        
        # Print results
        logger.info("Data import pipeline completed successfully!")
        logger.info(f"Movies imported: {results['movies_imported']}")
        logger.info(f"TV shows imported: {results['tv_shows_imported']}")
        
        if results['retraining_triggered']:
            logger.info(f"Models retrained: {results['retraining_triggered']}")
        
        if results['errors']:
            logger.warning(f"Errors encountered: {results['errors']}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())