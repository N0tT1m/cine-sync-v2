#!/usr/bin/env python3
"""
Enhanced Data Loader for CineSync v2
Handles loading and preprocessing of all available datasets including the new ones.
"""

import pandas as pd
import numpy as np
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import ast
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
import pickle

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedDatasetLoader:
    """Enhanced dataset loader for all CineSync datasets
    
    This loader handles the complex task of loading, preprocessing, and unifying
    multiple heterogeneous recommendation datasets including:
    - MovieLens 32M (collaborative filtering)
    - Netflix Prize Archive (historical movie ratings)
    - TMDB (The Movie Database) - movies and TV shows
    - MyAnimeList (anime ratings and reviews)
    - IMDb complete datasets (movies, TV, cast, crew)
    - Professional review data (Rotten Tomatoes, Metacritic)
    - Streaming platform data (Netflix, Amazon Prime, Disney+)
    
    Key Features:
    - Automatic data type detection and format standardization
    - Memory-efficient chunked loading for large datasets
    - Robust error handling for corrupted or missing files
    - Unified schema creation across different data sources
    - Label encoding and feature scaling for ML training
    """
    
    def __init__(self, base_path: str = "."):
        """Initialize the enhanced dataset loader
        
        Sets up directory paths and initializes data containers and preprocessors
        for handling multiple dataset formats.
        
        Args:
            base_path (str): Root directory containing all dataset folders
        """
        self.base_path = Path(base_path)
        # Define standard directory structure for different data types
        self.movies_path = self.base_path / "movies"  # Movie datasets (MovieLens, Netflix, etc.)
        self.tv_path = self.base_path / "tv"          # TV show and anime datasets
        self.imdb_path = self.base_path / "imdb"      # IMDb complete database dumps
        self.tmdb_path = self.base_path / "tmdb"      # TMDB API data exports
        
        # Initialize data containers for different content types
        self.movie_data = {}    # Movie-specific datasets
        self.tv_data = {}       # TV show and anime datasets
        self.rating_data = {}   # User rating matrices
        self.metadata = {}      # Dataset metadata and statistics
        
        # Initialize ML preprocessors (fit during data preparation)
        self.genre_encoder = None   # Encode genre strings to integers
        self.rating_scaler = None   # Scale ratings to standard range
        self.user_encoder = None    # Encode user IDs to consecutive integers
        self.item_encoder = None    # Encode item IDs to consecutive integers
    
    def load_all_datasets(self) -> Dict[str, pd.DataFrame]:
        """Load all available datasets from configured directories
        
        Scans the directory structure and loads all supported dataset formats.
        Each dataset type has specialized loading logic to handle format differences.
        
        Returns:
            Dict[str, pd.DataFrame]: Dictionary mapping dataset names to DataFrames
        """
        logger.info("Loading all datasets...")
        
        datasets = {}
        
        # Load movie-focused datasets (MovieLens, Netflix, TMDB movies, etc.)
        datasets.update(self.load_movie_datasets())
        
        # Load TV show and anime datasets (TMDB TV, MyAnimeList, etc.)
        datasets.update(self.load_tv_datasets())
        
        # Load comprehensive IMDb database dumps (all content types)
        datasets.update(self.load_imdb_datasets())
        
        # Load professional review and rating datasets
        datasets.update(self.load_review_datasets())
        
        # Load streaming platform availability data
        datasets.update(self.load_streaming_datasets())
        
        logger.info(f"Loaded {len(datasets)} datasets successfully")
        return datasets
    
    def load_movie_datasets(self) -> Dict[str, pd.DataFrame]:
        """Load all movie-related datasets"""
        datasets = {}
        
        # MovieLens 32M (Core dataset)
        ml32m_path = self.movies_path / "cinesync" / "ml-32m"
        if ml32m_path.exists():
            datasets['ml32m_ratings'] = self.load_movielens_ratings(ml32m_path)
            datasets['ml32m_movies'] = self.load_movielens_movies(ml32m_path)
            datasets['ml32m_tags'] = self.load_movielens_tags(ml32m_path)
            datasets['ml32m_links'] = self.load_movielens_links(ml32m_path)
        
        # Netflix Prize Archive
        netflix_path = self.movies_path / "cinesync" / "archive"
        if netflix_path.exists():
            datasets['netflix_ratings'] = self.load_netflix_ratings(netflix_path)
            datasets['netflix_movies'] = self.load_netflix_movies(netflix_path)
        
        # TMDB Movies
        tmdb_movies_path = self.movies_path / "tmdb-movies"
        if tmdb_movies_path.exists():
            datasets['tmdb_movies'] = self.load_tmdb_movies(tmdb_movies_path)
            datasets['tmdb_credits'] = self.load_tmdb_credits(tmdb_movies_path)
            datasets['tmdb_keywords'] = self.load_tmdb_keywords(tmdb_movies_path)
        
        # Rotten Tomatoes
        rotten_path = self.movies_path / "rotten"
        if rotten_path.exists():
            datasets['rotten_movies'] = self.load_rotten_tomatoes(rotten_path)
            datasets['rotten_reviews'] = self.load_rotten_reviews(rotten_path)
        
        # Metacritic Movies
        metacritic_path = self.movies_path / "metacritic"
        if metacritic_path.exists():
            datasets['metacritic_movies'] = self.load_metacritic_movies(metacritic_path)
        
        # Box Office Mojo
        boxoffice_path = self.movies_path / "boxoffice"
        if boxoffice_path.exists():
            datasets['boxoffice'] = self.load_boxoffice_data(boxoffice_path)
        
        # HetRec enhanced data
        hetrec_path = self.movies_path / "hetrec"
        if hetrec_path.exists():
            datasets['hetrec_movies'] = self.load_hetrec_data(hetrec_path)
        
        return datasets
    
    def load_tv_datasets(self) -> Dict[str, pd.DataFrame]:
        """Load all TV show and anime datasets"""
        datasets = {}
        
        # TMDB TV Shows
        tmdb_tv_path = self.tv_path / "tmdb"
        if tmdb_tv_path.exists():
            datasets['tmdb_tv'] = self.load_tmdb_tv_shows(tmdb_tv_path)
        
        # MyAnimeList
        anime_path = self.tv_path / "anime"
        if anime_path.exists():
            datasets['anime_shows'] = self.load_anime_shows(anime_path)
            datasets['anime_profiles'] = self.load_anime_profiles(anime_path)
            datasets['anime_reviews'] = self.load_anime_reviews(anime_path)
        
        # IMDb TV Series by Genre
        imdb_tv_path = self.tv_path / "imdb"
        if imdb_tv_path.exists():
            datasets['imdb_tv_genres'] = self.load_imdb_tv_genres(imdb_tv_path)
        
        # Netflix TV
        netflix_tv_path = self.tv_path / "netflix"
        if netflix_tv_path.exists():
            datasets['netflix_tv'] = self.load_netflix_tv(netflix_tv_path)
        
        # Misc TV datasets
        misc_tv_path = self.tv_path / "misc"
        if misc_tv_path.exists():
            datasets['misc_tv'] = self.load_misc_tv_data(misc_tv_path)
        
        return datasets
    
    def load_imdb_datasets(self) -> Dict[str, pd.DataFrame]:
        """Load IMDb complete datasets"""
        datasets = {}
        
        if not self.imdb_path.exists():
            return datasets
        
        # Load IMDb datasets with proper handling of large files
        imdb_files = {
            'title_basics': 'title.basics.tsv',
            'title_ratings': 'title.ratings.tsv',
            'title_crew': 'title.crew.tsv',
            'title_episode': 'title.episode.tsv',
            'title_principals': 'title.principals.tsv',
            'title_akas': 'title.akas.tsv',
            'name_basics': 'name.basics.tsv'
        }
        
        for dataset_name, filename in imdb_files.items():
            file_path = self.imdb_path / filename
            if file_path.exists():
                try:
                    # Use chunked reading for large files
                    logger.info(f"Loading {dataset_name}...")
                    df = pd.read_csv(file_path, sep='\t', na_values=['\\N'], 
                                   low_memory=False, chunksize=100000)
                    # Combine chunks
                    datasets[f'imdb_{dataset_name}'] = pd.concat(df, ignore_index=True)
                    logger.info(f"Loaded {dataset_name} with {len(datasets[f'imdb_{dataset_name}'])} rows")
                except Exception as e:
                    logger.error(f"Error loading {dataset_name}: {e}")
        
        return datasets
    
    def load_review_datasets(self) -> Dict[str, pd.DataFrame]:
        """Load professional review datasets"""
        datasets = {}
        
        # Metacritic TV reviews
        metacritic_tv_path = self.movies_path / "metacritic" / "feb_2023" / "tv.csv"
        if metacritic_tv_path.exists():
            datasets['metacritic_tv'] = pd.read_csv(metacritic_tv_path)
        
        return datasets
    
    def load_streaming_datasets(self) -> Dict[str, pd.DataFrame]:
        """Load streaming platform datasets"""
        datasets = {}
        
        # Amazon Prime
        amazon_path = self.movies_path / "amazon"
        if amazon_path.exists():
            datasets['amazon_prime'] = self.load_amazon_content(amazon_path)
        
        # Disney+
        disney_path = self.movies_path / "disney"
        if disney_path.exists():
            datasets['disney_plus'] = self.load_disney_content(disney_path)
        
        # Multi-platform streaming data
        streaming_path = self.movies_path / "streaming"
        if streaming_path.exists():
            datasets['multi_platform'] = self.load_multiplatform_content(streaming_path)
        
        return datasets
    
    # Specific loader methods for each dataset
    
    def load_movielens_ratings(self, path: Path) -> pd.DataFrame:
        """Load MovieLens ratings with timestamp conversion
        
        Loads the core MovieLens ratings file and converts Unix timestamps
        to proper datetime objects for temporal analysis.
        
        Args:
            path (Path): Path to MovieLens dataset directory
            
        Returns:
            pd.DataFrame: Ratings with columns [userId, movieId, rating, timestamp]
        """
        ratings_file = path / "ratings.csv"
        if ratings_file.exists():
            df = pd.read_csv(ratings_file)
            # Convert Unix timestamp to datetime for time-based analysis
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
            return df
        return pd.DataFrame()
    
    def load_movielens_movies(self, path: Path) -> pd.DataFrame:
        """Load MovieLens movies with genre processing"""
        movies_file = path / "movies.csv"
        if movies_file.exists():
            df = pd.read_csv(movies_file)
            # Process genres
            df['genres_list'] = df['genres'].str.split('|')
            return df
        return pd.DataFrame()
    
    def load_movielens_tags(self, path: Path) -> pd.DataFrame:
        """Load MovieLens tags with timestamp conversion
        
        Loads user-generated tags for movies, converting timestamps for analysis.
        
        Args:
            path (Path): Path to MovieLens dataset directory
            
        Returns:
            pd.DataFrame: Tags with columns [userId, movieId, tag, timestamp]
        """
        tags_file = path / "tags.csv"
        if tags_file.exists():
            df = pd.read_csv(tags_file)
            # Convert Unix timestamp to datetime for temporal analysis
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
            return df
        return pd.DataFrame()
    
    def load_movielens_links(self, path: Path) -> pd.DataFrame:
        """Load MovieLens links to external sources"""
        links_file = path / "links.csv"
        if links_file.exists():
            return pd.read_csv(links_file)
        return pd.DataFrame()
    
    def load_netflix_ratings(self, path: Path) -> pd.DataFrame:
        """Load Netflix Prize ratings from combined data files"""
        all_ratings = []
        
        for i in range(1, 5):
            file_path = path / f"combined_data_{i}.txt"
            if file_path.exists():
                try:
                    ratings = self._parse_netflix_file(file_path)
                    all_ratings.extend(ratings)
                except Exception as e:
                    logger.error(f"Error loading Netflix file {i}: {e}")
        
        if all_ratings:
            df = pd.DataFrame(all_ratings, columns=['movieId', 'userId', 'rating', 'date'])
            df['date'] = pd.to_datetime(df['date'])
            return df
        return pd.DataFrame()
    
    def _parse_netflix_file(self, file_path: Path) -> List[Tuple]:
        """Parse Netflix Prize data file format"""
        ratings = []
        current_movie_id = None
        
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line.endswith(':'):
                    current_movie_id = int(line[:-1])
                else:
                    parts = line.split(',')
                    if len(parts) == 3:
                        user_id, rating, date = parts
                        ratings.append((current_movie_id, int(user_id), float(rating), date))
        
        return ratings
    
    def load_netflix_movies(self, path: Path) -> pd.DataFrame:
        """Load Netflix movie titles"""
        movies_file = path / "movie_titles.csv"
        if movies_file.exists():
            # Netflix file has encoding issues, handle them
            try:
                df = pd.read_csv(movies_file, encoding='utf-8', header=None,
                               names=['movieId', 'year', 'title'])
            except UnicodeDecodeError:
                df = pd.read_csv(movies_file, encoding='latin-1', header=None,
                               names=['movieId', 'year', 'title'])
            return df
        return pd.DataFrame()
    
    def load_tmdb_movies(self, path: Path) -> pd.DataFrame:
        """Load TMDB movies metadata"""
        movies_file = path / "movies_metadata.csv"
        if movies_file.exists():
            df = pd.read_csv(movies_file, low_memory=False)
            # Process JSON columns
            if 'genres' in df.columns:
                df['genres_list'] = df['genres'].apply(self._parse_json_column)
            if 'production_companies' in df.columns:
                df['production_companies_list'] = df['production_companies'].apply(self._parse_json_column)
            return df
        return pd.DataFrame()
    
    def load_tmdb_credits(self, path: Path) -> pd.DataFrame:
        """Load TMDB credits data"""
        credits_file = path / "credits.csv"
        if credits_file.exists():
            df = pd.read_csv(credits_file)
            # Process JSON columns for cast and crew
            if 'cast' in df.columns:
                df['cast_list'] = df['cast'].apply(self._parse_json_column)
            if 'crew' in df.columns:
                df['crew_list'] = df['crew'].apply(self._parse_json_column)
            return df
        return pd.DataFrame()
    
    def load_tmdb_keywords(self, path: Path) -> pd.DataFrame:
        """Load TMDB keywords"""
        keywords_file = path / "keywords.csv"
        if keywords_file.exists():
            df = pd.read_csv(keywords_file)
            if 'keywords' in df.columns:
                df['keywords_list'] = df['keywords'].apply(self._parse_json_column)
            return df
        return pd.DataFrame()
    
    def load_tmdb_tv_shows(self, path: Path) -> pd.DataFrame:
        """Load TMDB TV shows"""
        tv_file = path / "TMDB_tv_dataset_v3.csv"
        if tv_file.exists():
            df = pd.read_csv(tv_file)
            # Process genre columns
            if 'genre_ids' in df.columns:
                df['genre_ids_list'] = df['genre_ids'].apply(self._parse_json_column)
            return df
        return pd.DataFrame()
    
    def load_anime_shows(self, path: Path) -> pd.DataFrame:
        """Load anime shows data"""
        animes_file = path / "animes.csv"
        if animes_file.exists():
            return pd.read_csv(animes_file)
        return pd.DataFrame()
    
    def load_anime_profiles(self, path: Path) -> pd.DataFrame:
        """Load anime user profiles"""
        profiles_file = path / "profiles.csv"
        if profiles_file.exists():
            return pd.read_csv(profiles_file)
        return pd.DataFrame()
    
    def load_anime_reviews(self, path: Path) -> pd.DataFrame:
        """Load anime reviews/ratings"""
        reviews_file = path / "reviews.csv"
        if reviews_file.exists():
            return pd.read_csv(reviews_file)
        return pd.DataFrame()
    
    def load_imdb_tv_genres(self, path: Path) -> pd.DataFrame:
        """Load IMDb TV series organized by genre"""
        all_series = []
        
        for genre_file in path.glob("*_series.csv"):
            genre = genre_file.stem.replace('_series', '')
            try:
                df = pd.read_csv(genre_file)
                df['primary_genre'] = genre
                all_series.append(df)
            except Exception as e:
                logger.error(f"Error loading {genre_file}: {e}")
        
        if all_series:
            return pd.concat(all_series, ignore_index=True)
        return pd.DataFrame()
    
    def load_rotten_tomatoes(self, path: Path) -> pd.DataFrame:
        """Load Rotten Tomatoes movies"""
        movies_file = path / "rotten_tomatoes_movies.csv"
        if movies_file.exists():
            return pd.read_csv(movies_file)
        return pd.DataFrame()
    
    def load_rotten_reviews(self, path: Path) -> pd.DataFrame:
        """Load Rotten Tomatoes reviews"""
        reviews_file = path / "rotten_tomatoes_critic_reviews.csv"
        if reviews_file.exists():
            return pd.read_csv(reviews_file)
        return pd.DataFrame()
    
    def load_metacritic_movies(self, path: Path) -> pd.DataFrame:
        """Load Metacritic movie reviews"""
        movies_file = path / "feb_2023" / "movies.csv"
        if movies_file.exists():
            return pd.read_csv(movies_file)
        return pd.DataFrame()
    
    def load_boxoffice_data(self, path: Path) -> pd.DataFrame:
        """Load Box Office Mojo data"""
        budget_file = path / "Mojo_budget_data.csv"
        if budget_file.exists():
            return pd.read_csv(budget_file)
        return pd.DataFrame()
    
    def load_hetrec_data(self, path: Path) -> pd.DataFrame:
        """Load HetRec enhanced MovieLens data"""
        movies_file = path / "movies.dat"
        if movies_file.exists():
            # HetRec uses :: separator
            df = pd.read_csv(movies_file, sep='::', engine='python',
                           names=['id', 'title', 'imdbID', 'spanishTitle', 'imdbPictureURL', 'year', 'rtID', 'rtAllCriticsRating', 'rtAllCriticsNumReviews', 'rtAllCriticsNumFresh', 'rtAllCriticsNumRotten', 'rtAllCriticsScore', 'rtTopCriticsRating', 'rtTopCriticsNumReviews', 'rtTopCriticsNumFresh', 'rtTopCriticsNumRotten', 'rtTopCriticsScore', 'rtAudienceRating', 'rtAudienceNumRatings', 'rtAudienceScore', 'rtPictureURL'])
            return df
        return pd.DataFrame()
    
    def load_netflix_tv(self, path: Path) -> pd.DataFrame:
        """Load Netflix TV shows"""
        netflix_file = path / "netflix_titles.csv"
        if netflix_file.exists():
            df = pd.read_csv(netflix_file)
            # Filter for TV shows only
            return df[df['type'] == 'TV Show'].copy()
        return pd.DataFrame()
    
    def load_amazon_content(self, path: Path) -> pd.DataFrame:
        """Load Amazon Prime content"""
        amazon_file = path / "amazon_prime_titles.csv"
        if amazon_file.exists():
            return pd.read_csv(amazon_file)
        return pd.DataFrame()
    
    def load_disney_content(self, path: Path) -> pd.DataFrame:
        """Load Disney+ content"""
        disney_file = path / "disney_plus_titles.csv"
        if disney_file.exists():
            return pd.read_csv(disney_file)
        return pd.DataFrame()
    
    def load_multiplatform_content(self, path: Path) -> pd.DataFrame:
        """Load multi-platform streaming availability"""
        streaming_file = path / "MoviesOnStreamingPlatforms.csv"
        if streaming_file.exists():
            return pd.read_csv(streaming_file)
        return pd.DataFrame()
    
    def load_misc_tv_data(self, path: Path) -> pd.DataFrame:
        """Load miscellaneous TV data"""
        all_misc = []
        
        for csv_file in path.glob("*.csv"):
            try:
                df = pd.read_csv(csv_file)
                df['source_file'] = csv_file.name
                all_misc.append(df)
            except Exception as e:
                logger.error(f"Error loading {csv_file}: {e}")
        
        if all_misc:
            return pd.concat(all_misc, ignore_index=True)
        return pd.DataFrame()
    
    def _parse_json_column(self, json_str: str) -> List[str]:
        """Parse JSON string column to extract names"""
        if pd.isna(json_str) or json_str == '':
            return []
        
        try:
            # Handle different JSON formats
            if isinstance(json_str, str):
                data = ast.literal_eval(json_str)
            else:
                return []
            
            if isinstance(data, list):
                # Extract 'name' field from each item
                return [item.get('name', '') for item in data if isinstance(item, dict)]
            return []
        except:
            return []
    
    def create_unified_dataset(self, datasets: Dict[str, pd.DataFrame]) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Create unified movie and TV datasets for training"""
        logger.info("Creating unified datasets...")
        
        # Combine movie datasets
        movie_datasets = []
        tv_datasets = []
        
        # Process MovieLens data (primary)
        if 'ml32m_ratings' in datasets and 'ml32m_movies' in datasets:
            ml_data = self._process_movielens_data(datasets['ml32m_ratings'], datasets['ml32m_movies'])
            movie_datasets.append(ml_data)
        
        # Process Netflix data
        if 'netflix_ratings' in datasets and 'netflix_movies' in datasets:
            netflix_data = self._process_netflix_data(datasets['netflix_ratings'], datasets['netflix_movies'])
            movie_datasets.append(netflix_data)
        
        # Process TMDB TV data
        if 'tmdb_tv' in datasets:
            tv_data = self._process_tmdb_tv_data(datasets['tmdb_tv'])
            tv_datasets.append(tv_data)
        
        # Process anime data
        if 'anime_reviews' in datasets and 'anime_shows' in datasets:
            anime_data = self._process_anime_data(datasets['anime_reviews'], datasets['anime_shows'])
            tv_datasets.append(anime_data)
        
        # Combine all datasets
        unified_movies = pd.concat(movie_datasets, ignore_index=True) if movie_datasets else pd.DataFrame()
        unified_tv = pd.concat(tv_datasets, ignore_index=True) if tv_datasets else pd.DataFrame()
        
        logger.info(f"Created unified movie dataset with {len(unified_movies)} rows")
        logger.info(f"Created unified TV dataset with {len(unified_tv)} rows")
        
        return unified_movies, unified_tv
    
    def _process_movielens_data(self, ratings_df: pd.DataFrame, movies_df: pd.DataFrame) -> pd.DataFrame:
        """Process MovieLens data for training"""
        # Merge ratings with movie info
        merged = pd.merge(ratings_df, movies_df, on='movieId', how='inner')
        
        # Add content type
        merged['content_type'] = 'movie'
        merged['source'] = 'movielens'
        
        # Standardize column names
        merged = merged.rename(columns={
            'movieId': 'content_id',
            'userId': 'user_id',
            'title': 'content_title',
            'genres': 'genres_raw'
        })
        
        return merged[['user_id', 'content_id', 'rating', 'content_title', 'genres_raw', 'content_type', 'source']]
    
    def _process_netflix_data(self, ratings_df: pd.DataFrame, movies_df: pd.DataFrame) -> pd.DataFrame:
        """Process Netflix Prize data for training"""
        # Merge ratings with movie info
        merged = pd.merge(ratings_df, movies_df, on='movieId', how='inner')
        
        # Add content type
        merged['content_type'] = 'movie'
        merged['source'] = 'netflix'
        
        # Standardize column names
        merged = merged.rename(columns={
            'movieId': 'content_id',
            'userId': 'user_id',
            'title': 'content_title'
        })
        
        # Netflix doesn't have genre info in basic dataset
        merged['genres_raw'] = ''
        
        return merged[['user_id', 'content_id', 'rating', 'content_title', 'genres_raw', 'content_type', 'source']]
    
    def _process_tmdb_tv_data(self, tv_df: pd.DataFrame) -> pd.DataFrame:
        """Process TMDB TV data for training"""
        # Create synthetic ratings based on vote_average and popularity
        processed_rows = []
        
        for _, row in tv_df.iterrows():
            if pd.notna(row.get('vote_average')) and row.get('vote_count', 0) > 10:
                # Create multiple synthetic user ratings around the average
                vote_avg = row['vote_average']
                vote_count = min(row.get('vote_count', 100), 1000)  # Cap at 1000 synthetic users
                
                # Generate ratings with normal distribution around vote_average
                ratings = np.random.normal(vote_avg, 1.5, min(vote_count // 10, 100))
                ratings = np.clip(ratings, 0.5, 10.0)  # TMDB uses 0-10 scale
                
                for i, rating in enumerate(ratings):
                    processed_rows.append({
                        'user_id': f"tmdb_user_{i}_{row.get('id', 0)}",
                        'content_id': row.get('id', 0),
                        'rating': rating / 2.0,  # Convert to 0-5 scale
                        'content_title': row.get('name', ''),
                        'genres_raw': ','.join(row.get('genre_ids_list', [])) if row.get('genre_ids_list') else '',
                        'content_type': 'tv',
                        'source': 'tmdb'
                    })
        
        return pd.DataFrame(processed_rows)
    
    def _process_anime_data(self, reviews_df: pd.DataFrame, anime_df: pd.DataFrame) -> pd.DataFrame:
        """Process anime/MyAnimeList data for training"""
        # Merge reviews with anime info
        merged = pd.merge(reviews_df, anime_df, left_on='anime_id', right_on='MAL_ID', how='inner')
        
        # Add content type
        merged['content_type'] = 'anime'
        merged['source'] = 'myanimelist'
        
        # Standardize column names
        merged = merged.rename(columns={
            'anime_id': 'content_id',
            'user_id': 'user_id',
            'score': 'rating',
            'Name': 'content_title',
            'Genres': 'genres_raw'
        })
        
        # Filter valid ratings
        merged = merged[merged['rating'] > 0]
        
        return merged[['user_id', 'content_id', 'rating', 'content_title', 'genres_raw', 'content_type', 'source']]
    
    def prepare_training_data(self, unified_movies: pd.DataFrame, unified_tv: pd.DataFrame) -> Dict:
        """Prepare data for model training"""
        logger.info("Preparing training data...")
        
        # Combine movies and TV
        all_data = pd.concat([unified_movies, unified_tv], ignore_index=True)
        
        # Remove invalid data
        all_data = all_data.dropna(subset=['user_id', 'content_id', 'rating'])
        all_data = all_data[all_data['rating'] > 0]
        
        # Encode users and items
        self.user_encoder = LabelEncoder()
        self.item_encoder = LabelEncoder()
        
        all_data['user_encoded'] = self.user_encoder.fit_transform(all_data['user_id'].astype(str))
        all_data['item_encoded'] = self.item_encoder.fit_transform(all_data['content_id'].astype(str))
        
        # Scale ratings to 0-1
        self.rating_scaler = StandardScaler()
        all_data['rating_scaled'] = self.rating_scaler.fit_transform(all_data[['rating']])
        
        # Split data
        train_data, test_data = train_test_split(all_data, test_size=0.2, random_state=42)
        
        # Create metadata
        metadata = {
            'num_users': len(self.user_encoder.classes_),
            'num_items': len(self.item_encoder.classes_),
            'user_encoder': self.user_encoder,
            'item_encoder': self.item_encoder,
            'rating_scaler': self.rating_scaler,
            'content_types': all_data['content_type'].unique().tolist(),
            'sources': all_data['source'].unique().tolist()
        }
        
        # Save encoders
        self.save_encoders(metadata)
        
        return {
            'train_data': train_data,
            'test_data': test_data,
            'metadata': metadata,
            'all_data': all_data
        }
    
    def save_encoders(self, metadata: Dict):
        """Save encoders and scalers for later use"""
        encoders_path = self.base_path / "models"
        encoders_path.mkdir(exist_ok=True)
        
        with open(encoders_path / "enhanced_encoders.pkl", 'wb') as f:
            pickle.dump(metadata, f)
        
        logger.info("Saved encoders and metadata")
    
    def load_encoders(self) -> Dict:
        """Load saved encoders and scalers"""
        encoders_path = self.base_path / "models" / "enhanced_encoders.pkl"
        
        if encoders_path.exists():
            with open(encoders_path, 'rb') as f:
                return pickle.load(f)
        else:
            logger.warning("No saved encoders found")
            return {}

class RecommendationDataset(Dataset):
    """PyTorch dataset for recommendation training"""
    
    def __init__(self, data_df: pd.DataFrame):
        self.users = torch.LongTensor(data_df['user_encoded'].values)
        self.items = torch.LongTensor(data_df['item_encoded'].values)
        self.ratings = torch.FloatTensor(data_df['rating_scaled'].values)
    
    def __len__(self):
        return len(self.users)
    
    def __getitem__(self, idx):
        return {
            'user': self.users[idx],
            'item': self.items[idx],
            'rating': self.ratings[idx]
        }

def create_data_loaders(train_data: pd.DataFrame, test_data: pd.DataFrame, 
                       batch_size: int = 1024) -> Tuple[DataLoader, DataLoader]:
    """Create PyTorch data loaders"""
    train_dataset = RecommendationDataset(train_data)
    test_dataset = RecommendationDataset(test_data)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    return train_loader, test_loader

if __name__ == "__main__":
    # Example usage
    loader = EnhancedDatasetLoader()
    
    # Load all datasets
    datasets = loader.load_all_datasets()
    
    # Create unified datasets
    unified_movies, unified_tv = loader.create_unified_dataset(datasets)
    
    # Prepare training data
    training_data = loader.prepare_training_data(unified_movies, unified_tv)
    
    # Create data loaders
    train_loader, test_loader = create_data_loaders(
        training_data['train_data'], 
        training_data['test_data']
    )
    
    logger.info("Enhanced data loading completed successfully!")
    logger.info(f"Training samples: {len(training_data['train_data'])}")
    logger.info(f"Test samples: {len(training_data['test_data'])}")
    logger.info(f"Number of users: {training_data['metadata']['num_users']}")
    logger.info(f"Number of items: {training_data['metadata']['num_items']}")