"""
Movie Data Preprocessor for Recommendation Models
Processes multiple movie datasets and creates unified training data
Mirrors the functionality of tv_preprocessor.py for consistency
"""

import os
import sys
import json
import csv
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import logging
from pathlib import Path
import pickle
from collections import defaultdict, Counter
import re
from datetime import datetime
from sklearn.preprocessing import LabelEncoder
import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MovieDataPreprocessor:
    """Comprehensive movie data preprocessor for recommendation models"""

    def __init__(self,
                 data_dir: str = "/Users/timmy/workspace/ai-apps/cine-sync-v2/data/movies",
                 output_dir: str = "./processed_data",
                 min_movies_per_actor: int = 3,
                 min_movies_per_genre: int = 10,
                 max_cast_size: int = 10,
                 max_genres_per_movie: int = 5,
                 min_ratings: int = 10):

        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)

        self.min_movies_per_actor = min_movies_per_actor
        self.min_movies_per_genre = min_movies_per_genre
        self.max_cast_size = max_cast_size
        self.max_genres_per_movie = max_genres_per_movie
        self.min_ratings = min_ratings

        # Encoders for categorical features
        self.encoders = {
            'genres': LabelEncoder(),
            'actors': LabelEncoder(),
            'directors': LabelEncoder(),
            'studios': LabelEncoder(),
            'languages': LabelEncoder(),
            'countries': LabelEncoder(),
            'ratings': LabelEncoder()  # MPAA ratings
        }

        # Vocabulary mappings
        self.vocab_mappings = {}
        self.reverse_mappings = {}

        # Statistics
        self.stats = {
            'total_movies': 0,
            'unique_actors': 0,
            'unique_genres': 0,
            'unique_directors': 0,
            'avg_runtime': 0,
            'avg_budget': 0,
            'avg_revenue': 0
        }

    def load_tmdb_movies(self) -> pd.DataFrame:
        """Load and process TMDB movie dataset"""
        # Try multiple possible file locations
        possible_files = [
            self.data_dir / "tmdb_5000_movies.csv",
            self.data_dir / "TMDB_movie_dataset_v11.csv",
            self.data_dir / "movies_metadata.csv",
            self.data_dir / "tmdb_movies.csv",
        ]

        tmdb_file = None
        for f in possible_files:
            if f.exists():
                tmdb_file = f
                break

        if not tmdb_file:
            logger.warning(f"TMDB movie file not found in {self.data_dir}")
            return pd.DataFrame()

        logger.info(f"Loading TMDB movie dataset from {tmdb_file}...")

        try:
            df = pd.read_csv(tmdb_file, low_memory=False)
            logger.info(f"Loaded {len(df)} movies from TMDB")

            # Clean and process TMDB data
            df = self._clean_tmdb_data(df)

            return df
        except Exception as e:
            logger.error(f"Error loading TMDB data: {e}")
            return pd.DataFrame()

    def _clean_tmdb_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and standardize TMDB movie data"""

        # Remove quotes from column names
        df.columns = [col.strip('\"') for col in df.columns]

        # Map common column name variations
        column_mapping = {
            'original_title': 'title',
            'movie_id': 'id',
            'imdb_id': 'imdb_id',
        }

        for old_name, new_name in column_mapping.items():
            if old_name in df.columns and new_name not in df.columns:
                df[new_name] = df[old_name]

        # Essential columns (flexible matching)
        if 'title' not in df.columns and 'original_title' in df.columns:
            df['title'] = df['original_title']

        required_cols = ['id', 'title']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            logger.warning(f"Missing required columns: {missing_cols}")
            return pd.DataFrame()

        # Remove rows with missing essential data
        df = df.dropna(subset=['id', 'title'])

        # Clean text fields
        df['title'] = df['title'].astype(str).str.strip()
        if 'overview' in df.columns:
            df['overview'] = df['overview'].fillna('').astype(str).str.strip()
        else:
            df['overview'] = ''

        # Parse JSON-like fields
        df = self._parse_json_fields(df)

        # Convert numeric fields
        numeric_fields = ['vote_average', 'vote_count', 'budget', 'revenue',
                         'runtime', 'popularity']
        for field in numeric_fields:
            if field in df.columns:
                df[field] = pd.to_numeric(df[field], errors='coerce').fillna(0)

        # Parse dates
        if 'release_date' in df.columns:
            df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')

        # Filter out movies with very low quality indicators
        if 'vote_count' in df.columns:
            df = df[df['vote_count'] >= self.min_ratings]

        logger.info(f"Cleaned TMDB data: {len(df)} movies remaining")
        return df

    def _parse_json_fields(self, df: pd.DataFrame) -> pd.DataFrame:
        """Parse JSON-like fields in TMDB data"""

        json_fields = {
            'genres': 'name',
            'production_companies': 'name',
            'production_countries': 'name',
            'spoken_languages': 'name',
            'keywords': 'name'
        }

        for field, key in json_fields.items():
            if field in df.columns:
                df[f'{field}_parsed'] = df[field].apply(
                    lambda x: self._extract_from_json_str(x, key)
                )

        return df

    def _extract_from_json_str(self, json_str: str, key: str) -> List[str]:
        """Extract values from JSON-like string"""
        if pd.isna(json_str) or json_str == '':
            return []

        try:
            # Try parsing as JSON first
            if isinstance(json_str, str) and json_str.startswith('['):
                import ast
                try:
                    parsed = ast.literal_eval(json_str)
                    if isinstance(parsed, list):
                        return [item.get(key, '') for item in parsed if isinstance(item, dict)]
                except:
                    pass

            # Fallback to regex
            pattern = rf'"{key}"\s*:\s*"([^"]*)"'
            matches = re.findall(pattern, str(json_str))
            return [match.strip() for match in matches if match.strip()]
        except:
            return []

    def load_movielens_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load MovieLens movies and ratings"""
        # Try multiple possible locations
        possible_dirs = [
            self.data_dir / "ml-25m",
            self.data_dir / "ml-latest",
            self.data_dir / "movielens",
            self.data_dir,
        ]

        movies_df = pd.DataFrame()
        ratings_df = pd.DataFrame()

        for dir_path in possible_dirs:
            movies_file = dir_path / "movies.csv"
            ratings_file = dir_path / "ratings.csv"

            if movies_file.exists():
                logger.info(f"Loading MovieLens from {dir_path}...")
                try:
                    movies_df = pd.read_csv(movies_file)
                    logger.info(f"Loaded {len(movies_df)} movies from MovieLens")

                    if ratings_file.exists():
                        # Load ratings in chunks for large files
                        ratings_chunks = []
                        for chunk in pd.read_csv(ratings_file, chunksize=1000000):
                            ratings_chunks.append(chunk)
                        ratings_df = pd.concat(ratings_chunks, ignore_index=True)
                        logger.info(f"Loaded {len(ratings_df)} ratings from MovieLens")
                    break
                except Exception as e:
                    logger.error(f"Error loading MovieLens data: {e}")

        return movies_df, ratings_df

    def load_imdb_data(self) -> pd.DataFrame:
        """Load IMDB movie data"""
        possible_files = [
            self.data_dir / "imdb_movies.csv",
            self.data_dir / "title.basics.tsv",
            self.data_dir / "imdb" / "title.basics.tsv",
        ]

        for imdb_file in possible_files:
            if imdb_file.exists():
                logger.info(f"Loading IMDB data from {imdb_file}...")
                try:
                    if imdb_file.suffix == '.tsv':
                        df = pd.read_csv(imdb_file, sep='\t', low_memory=False)
                        # Filter for movies only
                        df = df[df['titleType'] == 'movie']
                    else:
                        df = pd.read_csv(imdb_file, low_memory=False)

                    logger.info(f"Loaded {len(df)} movies from IMDB")
                    return df
                except Exception as e:
                    logger.error(f"Error loading IMDB data: {e}")

        return pd.DataFrame()

    def load_credits_data(self) -> pd.DataFrame:
        """Load cast and crew data"""
        possible_files = [
            self.data_dir / "tmdb_5000_credits.csv",
            self.data_dir / "credits.csv",
        ]

        for credits_file in possible_files:
            if credits_file.exists():
                logger.info(f"Loading credits from {credits_file}...")
                try:
                    df = pd.read_csv(credits_file, low_memory=False)
                    logger.info(f"Loaded credits for {len(df)} movies")
                    return df
                except Exception as e:
                    logger.error(f"Error loading credits: {e}")

        return pd.DataFrame()

    def unify_datasets(self) -> pd.DataFrame:
        """Combine and unify all movie datasets"""
        logger.info("Unifying movie datasets...")

        # Load all datasets
        tmdb_df = self.load_tmdb_movies()
        movielens_df, ratings_df = self.load_movielens_data()
        credits_df = self.load_credits_data()

        unified_movies = []
        movie_id = 0

        # Process TMDB data (primary source)
        if not tmdb_df.empty:
            # Merge with credits if available
            if not credits_df.empty and 'movie_id' in credits_df.columns:
                credits_df = credits_df.rename(columns={'movie_id': 'id'})
                tmdb_df = tmdb_df.merge(credits_df, on='id', how='left')

            for _, row in tmdb_df.iterrows():
                movie = self._process_tmdb_movie(row, movie_id)
                if movie:
                    unified_movies.append(movie)
                    movie_id += 1

        # Create lookup for MovieLens ratings
        rating_stats = {}
        if not ratings_df.empty:
            rating_agg = ratings_df.groupby('movieId').agg({
                'rating': ['mean', 'count', 'std']
            }).reset_index()
            rating_agg.columns = ['movieId', 'avg_rating', 'num_ratings', 'rating_std']
            rating_stats = {row['movieId']: row for _, row in rating_agg.iterrows()}

        # Create MovieLens title lookup
        movielens_lookup = {}
        if not movielens_df.empty:
            for _, row in movielens_df.iterrows():
                title_clean = self._clean_title(row.get('title', ''))
                movielens_lookup[title_clean] = {
                    'movieId': row.get('movieId'),
                    'genres': row.get('genres', '').split('|') if pd.notna(row.get('genres')) else []
                }

        # Enhance unified movies with MovieLens data
        for movie in unified_movies:
            title_clean = self._clean_title(movie['title'])

            if title_clean in movielens_lookup:
                ml_data = movielens_lookup[title_clean]
                movie_ml_id = ml_data['movieId']

                if movie_ml_id in rating_stats:
                    stats = rating_stats[movie_ml_id]
                    movie['movielens_avg_rating'] = float(stats['avg_rating'])
                    movie['movielens_num_ratings'] = int(stats['num_ratings'])
                    movie['movielens_rating_std'] = float(stats['rating_std']) if pd.notna(stats['rating_std']) else 0

                # Merge genres
                if not movie['genres'] and ml_data['genres']:
                    movie['genres'] = ml_data['genres']

        # Convert to DataFrame
        unified_df = pd.DataFrame(unified_movies)

        logger.info(f"Unified dataset created with {len(unified_df)} movies")
        return unified_df

    def _process_tmdb_movie(self, row: pd.Series, movie_id: int) -> Optional[Dict]:
        """Process a single TMDB movie"""
        try:
            # Extract genres
            genres = row.get('genres_parsed', [])
            if isinstance(genres, str):
                genres = [genres]
            elif not isinstance(genres, list):
                genres = []

            # Extract production companies (studios)
            studios = row.get('production_companies_parsed', [])
            if isinstance(studios, str):
                studios = [studios]
            elif not isinstance(studios, list):
                studios = []

            # Extract countries
            countries = row.get('production_countries_parsed', [])
            if isinstance(countries, str):
                countries = [countries]
            elif not isinstance(countries, list):
                countries = []

            # Extract languages
            languages = row.get('spoken_languages_parsed', [])
            if isinstance(languages, str):
                languages = [languages]
            elif not isinstance(languages, list):
                languages = []

            # Extract cast
            cast = []
            if 'cast' in row and pd.notna(row['cast']):
                cast = self._extract_from_json_str(row['cast'], 'name')[:self.max_cast_size]

            # Extract directors from crew
            directors = []
            if 'crew' in row and pd.notna(row['crew']):
                crew_str = str(row['crew'])
                # Look for director entries
                director_pattern = r'"job"\s*:\s*"Director"[^}]*"name"\s*:\s*"([^"]*)"'
                directors = re.findall(director_pattern, crew_str)
                if not directors:
                    # Alternative pattern
                    director_pattern = r'"name"\s*:\s*"([^"]*)"[^}]*"job"\s*:\s*"Director"'
                    directors = re.findall(director_pattern, crew_str)

            # Create movie dictionary
            movie = {
                'id': movie_id,
                'tmdb_id': row.get('id'),
                'imdb_id': row.get('imdb_id', ''),
                'title': row['title'],
                'overview': row.get('overview', ''),
                'tagline': row.get('tagline', ''),
                'genres': genres[:self.max_genres_per_movie],
                'studios': studios[:5],
                'countries': countries,
                'languages': languages,
                'cast': cast,
                'directors': directors[:3],
                'vote_average': float(row.get('vote_average', 0)),
                'vote_count': int(row.get('vote_count', 0)),
                'popularity': float(row.get('popularity', 0)),
                'budget': float(row.get('budget', 0)),
                'revenue': float(row.get('revenue', 0)),
                'runtime': float(row.get('runtime', 0)),
                'release_date': row.get('release_date'),
                'original_language': row.get('original_language', 'en'),
                'adult': bool(row.get('adult', False)),
                'movielens_avg_rating': 0,
                'movielens_num_ratings': 0,
                'movielens_rating_std': 0,
            }

            # Calculate derived features
            if movie['release_date'] and not pd.isna(movie['release_date']):
                try:
                    release_year = movie['release_date'].year
                    movie['release_year'] = release_year
                    movie['years_since_release'] = 2024 - release_year
                    movie['decade'] = release_year // 10 * 10
                except:
                    movie['release_year'] = 2020
                    movie['years_since_release'] = 0
                    movie['decade'] = 2020
            else:
                movie['release_year'] = 2020
                movie['years_since_release'] = 0
                movie['decade'] = 2020

            # Calculate ROI if budget > 0
            if movie['budget'] > 0:
                movie['roi'] = (movie['revenue'] - movie['budget']) / movie['budget']
            else:
                movie['roi'] = 0

            # Validate required fields
            if not movie['title']:
                return None

            return movie

        except Exception as e:
            logger.warning(f"Error processing movie: {e}")
            return None

    def _clean_title(self, title: str) -> str:
        """Clean title for matching across datasets"""
        if pd.isna(title):
            return ""

        # Convert to lowercase and remove special characters
        title = re.sub(r'[^\w\s]', '', str(title).lower())
        title = re.sub(r'\s+', ' ', title).strip()

        # Remove year from title (e.g., "Movie Name (2020)")
        title = re.sub(r'\s*\(\d{4}\)\s*$', '', title)

        # Remove common prefixes/suffixes
        title = re.sub(r'^(the|a|an)\s+', '', title)

        return title

    def build_vocabularies(self, df: pd.DataFrame) -> Dict[str, int]:
        """Build vocabulary mappings for categorical features"""
        logger.info("Building vocabularies...")

        vocab_sizes = {}

        # Build genre vocabulary
        all_genres = []
        for genres in df['genres']:
            if isinstance(genres, list):
                all_genres.extend(genres)

        genre_counts = Counter(all_genres)
        frequent_genres = [genre for genre, count in genre_counts.items()
                          if count >= self.min_movies_per_genre]

        self.encoders['genres'].fit(['<UNK>'] + frequent_genres)
        self.vocab_mappings['genres'] = {genre: idx for idx, genre in
                                       enumerate(self.encoders['genres'].classes_)}
        vocab_sizes['genres'] = len(self.encoders['genres'].classes_)

        # Build actor vocabulary
        all_actors = []
        for cast in df['cast']:
            if isinstance(cast, list):
                all_actors.extend(cast)

        actor_counts = Counter(all_actors)
        frequent_actors = [actor for actor, count in actor_counts.items()
                          if count >= self.min_movies_per_actor]

        self.encoders['actors'].fit(['<UNK>'] + frequent_actors)
        self.vocab_mappings['actors'] = {actor: idx for idx, actor in
                                        enumerate(self.encoders['actors'].classes_)}
        vocab_sizes['actors'] = len(self.encoders['actors'].classes_)

        # Build director vocabulary
        all_directors = []
        for directors in df['directors']:
            if isinstance(directors, list):
                all_directors.extend(directors)

        director_counts = Counter(all_directors)
        frequent_directors = [director for director, count in director_counts.items()
                             if count >= 2]

        self.encoders['directors'].fit(['<UNK>'] + frequent_directors)
        self.vocab_mappings['directors'] = {director: idx for idx, director in
                                           enumerate(self.encoders['directors'].classes_)}
        vocab_sizes['directors'] = len(self.encoders['directors'].classes_)

        # Build studio vocabulary
        all_studios = []
        for studios in df['studios']:
            if isinstance(studios, list):
                all_studios.extend(studios)

        studio_counts = Counter(all_studios)
        frequent_studios = [studio for studio, count in studio_counts.items()
                           if count >= 3]

        self.encoders['studios'].fit(['<UNK>'] + frequent_studios)
        self.vocab_mappings['studios'] = {studio: idx for idx, studio in
                                         enumerate(self.encoders['studios'].classes_)}
        vocab_sizes['studios'] = len(self.encoders['studios'].classes_)

        # Build language vocabulary
        languages = df['original_language'].fillna('en').unique().tolist()
        self.encoders['languages'].fit(languages)
        self.vocab_mappings['languages'] = {lang: idx for idx, lang in
                                           enumerate(self.encoders['languages'].classes_)}
        vocab_sizes['languages'] = len(self.encoders['languages'].classes_)

        # Save vocabulary mappings
        with open(self.output_dir / 'vocab_mappings.pkl', 'wb') as f:
            pickle.dump(self.vocab_mappings, f)

        with open(self.output_dir / 'encoders.pkl', 'wb') as f:
            pickle.dump(self.encoders, f)

        logger.info(f"Vocabulary sizes: {vocab_sizes}")
        return vocab_sizes

    def encode_categorical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical features using built vocabularies"""
        logger.info("Encoding categorical features...")

        df = df.copy()

        # Encode genres
        def encode_list(items, vocab_name):
            if not isinstance(items, list):
                return [0]  # UNK
            encoded = []
            vocab = self.vocab_mappings.get(vocab_name, {})
            for item in items:
                if item in vocab:
                    encoded.append(vocab[item])
                else:
                    encoded.append(0)  # UNK
            return encoded if encoded else [0]

        df['encoded_genres'] = df['genres'].apply(lambda x: encode_list(x, 'genres'))
        df['encoded_cast'] = df['cast'].apply(lambda x: encode_list(x, 'actors'))
        df['encoded_directors'] = df['directors'].apply(lambda x: encode_list(x, 'directors'))
        df['encoded_studios'] = df['studios'].apply(lambda x: encode_list(x, 'studios'))

        # Encode language
        df['encoded_language'] = df['original_language'].apply(
            lambda x: self.vocab_mappings['languages'].get(x, 0)
        )

        return df

    def create_training_data(self, df: pd.DataFrame) -> List[Dict]:
        """Create final training data format"""
        logger.info("Creating training data...")

        training_data = []

        for _, row in df.iterrows():
            # Prepare text
            text_parts = [row['title']]
            if row.get('tagline'):
                text_parts.append(row['tagline'])
            if row.get('overview'):
                text_parts.append(row['overview'])
            text = ' [SEP] '.join(text_parts)

            # Prepare categorical features
            categorical_features = {
                'genres': row['encoded_genres'],
                'cast': row['encoded_cast'],
                'directors': row['encoded_directors'],
                'studios': row['encoded_studios'],
                'language': [row['encoded_language']]
            }

            # Prepare numerical features (normalized)
            numerical_features = [
                float(row['vote_average']) / 10.0,
                float(row['vote_count']) / 10000.0,
                float(row['popularity']) / 100.0,
                float(row['budget']) / 200000000.0,  # Normalize to ~200M max
                float(row['revenue']) / 1000000000.0,  # Normalize to ~1B max
                float(row['runtime']) / 180.0,  # Normalize to ~3 hours
                float(row['years_since_release']) / 100.0,
                float(row['roi']) / 10.0,  # Cap ROI normalization
                float(row['movielens_avg_rating']) / 5.0,
                float(row['movielens_num_ratings']) / 100000.0,
                float(row['adult'])
            ]

            sample = {
                'id': row['id'],
                'title': row['title'],
                'text': text,
                'categorical_features': categorical_features,
                'numerical_features': numerical_features,
                'metadata': {
                    'tmdb_id': row.get('tmdb_id'),
                    'imdb_id': row.get('imdb_id'),
                    'genres': row['genres'],
                    'cast': row['cast'][:5],
                    'directors': row['directors'],
                    'vote_average': row['vote_average'],
                    'vote_count': row['vote_count'],
                    'runtime': row['runtime'],
                    'release_year': row['release_year']
                }
            }

            training_data.append(sample)

        return training_data

    def create_graph_data(self, df: pd.DataFrame) -> Dict:
        """Create graph data for GNN training"""
        logger.info("Creating graph data...")

        edge_index_dict = {}

        # Movie-Genre edges
        movie_genre_edges = [[], []]
        for idx, row in df.iterrows():
            for genre in row['encoded_genres'][:self.max_genres_per_movie]:
                movie_genre_edges[0].append(idx)
                movie_genre_edges[1].append(genre)

        edge_index_dict[('movie', 'has_genre', 'genre')] = torch.tensor(movie_genre_edges)

        # Movie-Actor edges
        movie_actor_edges = [[], []]
        for idx, row in df.iterrows():
            for actor in row['encoded_cast'][:self.max_cast_size]:
                movie_actor_edges[0].append(idx)
                movie_actor_edges[1].append(actor)

        edge_index_dict[('movie', 'features', 'actor')] = torch.tensor(movie_actor_edges)

        # Movie-Director edges
        movie_director_edges = [[], []]
        for idx, row in df.iterrows():
            for director in row['encoded_directors'][:3]:
                movie_director_edges[0].append(idx)
                movie_director_edges[1].append(director)

        edge_index_dict[('movie', 'directed_by', 'director')] = torch.tensor(movie_director_edges)

        # Movie-Studio edges
        movie_studio_edges = [[], []]
        for idx, row in df.iterrows():
            for studio in row['encoded_studios'][:3]:
                movie_studio_edges[0].append(idx)
                movie_studio_edges[1].append(studio)

        edge_index_dict[('movie', 'produced_by', 'studio')] = torch.tensor(movie_studio_edges)

        # Create similarity edges based on shared genres/cast
        similar_edges = self._create_similarity_edges(df)
        edge_index_dict[('movie', 'similar_to', 'movie')] = torch.tensor(similar_edges)

        return {
            'edge_index_dict': edge_index_dict,
            'num_movies': len(df),
            'num_genres': len(self.vocab_mappings['genres']),
            'num_actors': len(self.vocab_mappings['actors']),
            'num_directors': len(self.vocab_mappings['directors']),
            'num_studios': len(self.vocab_mappings['studios'])
        }

    def _create_similarity_edges(self, df: pd.DataFrame,
                                  genre_threshold: float = 0.3,
                                  cast_threshold: float = 0.2) -> List[List[int]]:
        """Create similarity edges between movies based on genre and cast overlap"""

        similar_edges = [[], []]

        # Limit comparisons for large datasets
        max_comparisons = min(len(df), 5000)
        sample_df = df.head(max_comparisons)

        for i, row1 in sample_df.iterrows():
            genres1 = set(row1['encoded_genres'])
            cast1 = set(row1['encoded_cast'][:5])

            for j, row2 in sample_df.iterrows():
                if i >= j:
                    continue

                genres2 = set(row2['encoded_genres'])
                cast2 = set(row2['encoded_cast'][:5])

                # Genre similarity (Jaccard)
                genre_intersection = len(genres1.intersection(genres2))
                genre_union = len(genres1.union(genres2))
                genre_sim = genre_intersection / genre_union if genre_union > 0 else 0

                # Cast similarity (Jaccard)
                cast_intersection = len(cast1.intersection(cast2))
                cast_union = len(cast1.union(cast2))
                cast_sim = cast_intersection / cast_union if cast_union > 0 else 0

                # Combined similarity
                if genre_sim >= genre_threshold or cast_sim >= cast_threshold:
                    similar_edges[0].extend([i, j])
                    similar_edges[1].extend([j, i])

        return similar_edges

    def split_data(self, training_data: List[Dict],
                   train_ratio: float = 0.8,
                   val_ratio: float = 0.1) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """Split data into train/val/test sets"""

        np.random.seed(42)
        np.random.shuffle(training_data)

        n_samples = len(training_data)
        n_train = int(n_samples * train_ratio)
        n_val = int(n_samples * val_ratio)

        train_data = training_data[:n_train]
        val_data = training_data[n_train:n_train + n_val]
        test_data = training_data[n_train + n_val:]

        logger.info(f"Data split: {len(train_data)} train, {len(val_data)} val, {len(test_data)} test")

        return train_data, val_data, test_data

    def save_processed_data(self, train_data: List[Dict], val_data: List[Dict],
                           test_data: List[Dict], graph_data: Dict, vocab_sizes: Dict):
        """Save all processed data"""

        # Save datasets
        with open(self.output_dir / 'train_data.json', 'w') as f:
            json.dump(train_data, f, indent=2, default=str)

        with open(self.output_dir / 'val_data.json', 'w') as f:
            json.dump(val_data, f, indent=2, default=str)

        with open(self.output_dir / 'test_data.json', 'w') as f:
            json.dump(test_data, f, indent=2, default=str)

        # Save graph data
        torch.save(graph_data, self.output_dir / 'graph_data.pt')

        # Save metadata
        metadata = {
            'vocab_sizes': vocab_sizes,
            'num_samples': {
                'train': len(train_data),
                'val': len(val_data),
                'test': len(test_data)
            },
            'preprocessing_config': {
                'min_movies_per_actor': self.min_movies_per_actor,
                'min_movies_per_genre': self.min_movies_per_genre,
                'max_cast_size': self.max_cast_size,
                'max_genres_per_movie': self.max_genres_per_movie
            },
            'stats': self.stats
        }

        with open(self.output_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"All processed data saved to {self.output_dir}")

    def process_all(self):
        """Main processing pipeline"""
        logger.info("Starting movie data preprocessing pipeline...")

        # Step 1: Unify datasets
        unified_df = self.unify_datasets()
        if unified_df.empty:
            logger.error("No data loaded. Exiting.")
            return

        # Step 2: Build vocabularies
        vocab_sizes = self.build_vocabularies(unified_df)

        # Step 3: Encode categorical features
        encoded_df = self.encode_categorical_features(unified_df)

        # Step 4: Create training data
        training_data = self.create_training_data(encoded_df)

        # Step 5: Create graph data
        graph_data = self.create_graph_data(encoded_df)

        # Step 6: Split data
        train_data, val_data, test_data = self.split_data(training_data)

        # Step 7: Update statistics
        self.stats.update({
            'total_movies': len(encoded_df),
            'unique_genres': len(self.vocab_mappings['genres']),
            'unique_actors': len(self.vocab_mappings['actors']),
            'unique_directors': len(self.vocab_mappings['directors']),
            'avg_runtime': encoded_df['runtime'].mean(),
            'avg_budget': encoded_df['budget'].mean(),
            'avg_revenue': encoded_df['revenue'].mean()
        })

        # Step 8: Save everything
        self.save_processed_data(train_data, val_data, test_data, graph_data, vocab_sizes)

        logger.info("Movie data preprocessing completed successfully!")
        logger.info(f"Final statistics: {self.stats}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Preprocess movie data for recommendation models')
    parser.add_argument('--data_dir', type=str,
                       default='/Users/timmy/workspace/ai-apps/cine-sync-v2/movies',
                       help='Directory containing movie datasets')
    parser.add_argument('--output_dir', type=str,
                       default='./processed_data',
                       help='Output directory for processed data')
    parser.add_argument('--min_movies_per_genre', type=int, default=10,
                       help='Minimum movies per genre to include')
    parser.add_argument('--max_genres_per_movie', type=int, default=5,
                       help='Maximum genres per movie')
    parser.add_argument('--min_ratings', type=int, default=10,
                       help='Minimum vote count to include movie')

    args = parser.parse_args()

    # Create preprocessor
    preprocessor = MovieDataPreprocessor(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        min_movies_per_genre=args.min_movies_per_genre,
        max_genres_per_movie=args.max_genres_per_movie,
        min_ratings=args.min_ratings
    )

    # Run preprocessing
    preprocessor.process_all()


if __name__ == "__main__":
    main()
