"""Real-dataset loader for building serving artifacts (SBERT text corpus,
two-tower item/user indexes).

Extracted verbatim from the former ``src/training/train_all_models.py`` during
the model consolidation so the embedding-index scripts keep working without the
legacy training monolith. IDs are remapped into fixed embedding-table ranges.
"""
from __future__ import annotations

import ast
import logging
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)


class RealDataLoader:
    """
    Comprehensive data loader that loads real datasets from available CSV files.
    Supports MovieLens, TMDB, Anime, and IMDB data.

    IMPORTANT: All IDs are re-mapped to fit within model embedding table sizes:
    - user_id: 0 to MAX_USERS-1 (49999)
    - item_id/movie_id/show_id: 0 to MAX_ITEMS-1 (99999)
    - genre_id: 0 to MAX_GENRES-1 (29)
    """

    # Max values to match model embedding sizes
    MAX_USERS = 50000
    MAX_ITEMS = 100000
    MAX_GENRES = 30

    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)
        self.movie_data = None
        self.tv_data = None
        # ID mapping dicts for consistent re-mapping
        self.user_id_map = {}
        self.item_id_map = {}
        self.genre_id_map = {}
        self._load_all_data()

    def _remap_user_id(self, original_id: int) -> int:
        """Re-map user ID to be within MAX_USERS"""
        if original_id not in self.user_id_map:
            new_id = len(self.user_id_map) % self.MAX_USERS
            self.user_id_map[original_id] = new_id
        return self.user_id_map[original_id]

    def _remap_item_id(self, original_id: int) -> int:
        """Re-map item ID to be within MAX_ITEMS"""
        if original_id not in self.item_id_map:
            new_id = len(self.item_id_map) % self.MAX_ITEMS
            self.item_id_map[original_id] = new_id
        return self.item_id_map[original_id]

    def _remap_genre_id(self, original_id: int) -> int:
        """Re-map genre ID to be within MAX_GENRES"""
        return original_id % self.MAX_GENRES

    def _load_all_data(self):
        """Load all available data sources"""
        # Load movie data
        self.movie_data = self._load_movie_data()
        # Load TV data
        self.tv_data = self._load_tv_data()

        logger.info(f"Loaded {len(self.movie_data)} movie interactions")
        logger.info(f"Loaded {len(self.tv_data)} TV interactions")
        logger.info(f"Unique users: {len(self.user_id_map)}, Unique items: {len(self.item_id_map)}")

    def _load_movie_data(self) -> List[Dict]:
        """Load movie ratings from MovieLens and TMDB"""
        # Note: We don't use caching here because ID remapping must be done per-instance
        all_ratings = []

        # Load MovieLens data
        ml_ratings_path = self.data_dir / 'movies' / 'recommendation' / 'ml-latest-small' / 'ratings.csv'
        ml_movies_path = self.data_dir / 'movies' / 'recommendation' / 'ml-latest-small' / 'movies.csv'

        if ml_ratings_path.exists():
            try:
                ratings_df = pd.read_csv(ml_ratings_path)
                movies_df = pd.read_csv(ml_movies_path) if ml_movies_path.exists() else None

                # Create movie genre mapping
                movie_genres = {}
                genre_to_id = {}
                if movies_df is not None:
                    for _, row in movies_df.iterrows():
                        genres = row['genres'].split('|') if pd.notna(row['genres']) else []
                        for g in genres:
                            if g not in genre_to_id:
                                genre_to_id[g] = len(genre_to_id)
                        movie_genres[row['movieId']] = [genre_to_id.get(g, 0) for g in genres[:3]]

                for _, row in ratings_df.iterrows():
                    original_movie_id = int(row['movieId'])
                    original_user_id = int(row['userId'])
                    genres = movie_genres.get(original_movie_id, [0])

                    # Re-map IDs to fit within embedding table sizes
                    user_id = self._remap_user_id(('ml', original_user_id))
                    item_id = self._remap_item_id(('ml', original_movie_id))
                    genre_id = self._remap_genre_id(genres[0] if genres else 0)

                    all_ratings.append({
                        'user_id': user_id,
                        'item_id': item_id,
                        'movie_id': item_id,
                        'rating': float(row['rating']),
                        'timestamp': int(row['timestamp']),
                        'genre_id': genre_id,
                        'source': 'movielens'
                    })
                logger.info(f"Loaded {len(ratings_df)} MovieLens ratings")
            except Exception as e:
                logger.warning(f"Error loading MovieLens data: {e}")

        # Load TMDB ratings
        tmdb_ratings_path = self.data_dir / 'movies' / 'tmdb-movies' / 'ratings_small.csv'
        if tmdb_ratings_path.exists():
            try:
                tmdb_df = pd.read_csv(tmdb_ratings_path)
                for _, row in tmdb_df.iterrows():
                    original_user_id = int(row['userId'])
                    original_movie_id = int(row['movieId'])

                    # Re-map IDs (use 'tmdb' prefix to distinguish from MovieLens)
                    user_id = self._remap_user_id(('tmdb', original_user_id))
                    item_id = self._remap_item_id(('tmdb', original_movie_id))

                    all_ratings.append({
                        'user_id': user_id,
                        'item_id': item_id,
                        'movie_id': item_id,
                        'rating': float(row['rating']),
                        'timestamp': int(row['timestamp']),
                        'genre_id': 0,
                        'source': 'tmdb'
                    })
                logger.info(f"Loaded {len(tmdb_df)} TMDB ratings")
            except Exception as e:
                logger.warning(f"Error loading TMDB data: {e}")

        return all_ratings

    def _load_tv_data(self) -> List[Dict]:
        """Load TV data from Anime profiles and IMDB series"""
        # Note: We don't use caching here because ID remapping must be done per-instance
        all_interactions = []

        # Load Anime data
        anime_profiles_path = self.data_dir / 'tv' / 'anime' / 'profiles.csv'
        anime_shows_path = self.data_dir / 'tv' / 'anime' / 'animes.csv'

        if anime_profiles_path.exists():
            try:
                profiles_df = pd.read_csv(anime_profiles_path)

                # Load show info for metadata
                show_scores = {}
                show_genres = {}
                genre_to_id = {}
                if anime_shows_path.exists():
                    shows_df = pd.read_csv(anime_shows_path)
                    for _, row in shows_df.iterrows():
                        show_id = int(row['uid'])
                        show_scores[show_id] = float(row['score']) if pd.notna(row['score']) else 5.0
                        genres = str(row['genre']).split(', ') if pd.notna(row['genre']) else []
                        for g in genres:
                            if g not in genre_to_id:
                                genre_to_id[g] = len(genre_to_id)
                        show_genres[show_id] = [genre_to_id.get(g, 0) for g in genres[:3]]

                anime_user_idx = 0
                for _, row in profiles_df.iterrows():
                    try:
                        # Parse favorites list
                        favorites_str = row['favorites_anime']
                        if pd.isna(favorites_str):
                            continue
                        favorites = ast.literal_eval(favorites_str)

                        for show_id_str in favorites:
                            original_show_id = int(show_id_str)
                            # Favorites imply high rating (4-5 stars)
                            rating = show_scores.get(original_show_id, 4.5)
                            genres = show_genres.get(original_show_id, [0])

                            # Re-map IDs to fit within embedding table sizes
                            user_id = self._remap_user_id(('anime', anime_user_idx))
                            item_id = self._remap_item_id(('anime', original_show_id))
                            genre_id = self._remap_genre_id(genres[0] if genres else 0)

                            all_interactions.append({
                                'user_id': user_id,
                                'item_id': item_id,
                                'show_id': item_id,
                                'rating': min(5.0, max(1.0, rating)),
                                'timestamp': 0,
                                'genre_id': genre_id,
                                'source': 'anime'
                            })
                        anime_user_idx += 1
                    except (ValueError, SyntaxError):
                        continue

                logger.info(f"Loaded {len(all_interactions)} anime interactions from {anime_user_idx} users")
            except Exception as e:
                logger.warning(f"Error loading anime data: {e}")

        # Load IMDB series data
        imdb_dir = self.data_dir / 'tv' / 'imdb'
        if imdb_dir.exists():
            try:
                imdb_show_idx = 0
                imdb_user_idx = 0
                # Load all genre CSV files
                for csv_file in imdb_dir.glob('*.csv'):
                    try:
                        series_df = pd.read_csv(csv_file)
                        if 'Rating' not in series_df.columns:
                            continue

                        # Extract genre from filename
                        genre_name = csv_file.stem.replace('_series', '')
                        genre_id = self._remap_genre_id(hash(genre_name) % 100)

                        for idx, row in series_df.iterrows():
                            if pd.notna(row.get('Rating')):
                                # Create synthetic user-item interactions based on ratings
                                rating = float(row['Rating'])
                                num_synthetic_users = max(1, int(rating * 2))

                                # Re-map show ID
                                item_id = self._remap_item_id(('imdb', imdb_show_idx))

                                for u in range(num_synthetic_users):
                                    # Re-map user ID
                                    user_id = self._remap_user_id(('imdb', imdb_user_idx))
                                    imdb_user_idx += 1

                                    all_interactions.append({
                                        'user_id': user_id,
                                        'item_id': item_id,
                                        'show_id': item_id,
                                        'rating': rating / 2.0,  # Convert 0-10 to 0-5 scale
                                        'timestamp': 0,
                                        'genre_id': genre_id,
                                        'source': 'imdb'
                                    })
                                imdb_show_idx += 1
                    except Exception as e:
                        continue

                logger.info(f"Total TV interactions after IMDB: {len(all_interactions)}")
            except Exception as e:
                logger.warning(f"Error loading IMDB data: {e}")

        return all_interactions

    def get_movie_data(self, split: str = 'train', split_ratio: float = 0.8) -> List[Dict]:
        """Get movie data with train/val split"""
        if not self.movie_data:
            return []

        split_idx = int(len(self.movie_data) * split_ratio)
        if split == 'train':
            return self.movie_data[:split_idx]
        else:
            return self.movie_data[split_idx:]

    def get_tv_data(self, split: str = 'train', split_ratio: float = 0.8) -> List[Dict]:
        """Get TV data with train/val split"""
        if not self.tv_data:
            return []

        split_idx = int(len(self.tv_data) * split_ratio)
        if split == 'train':
            return self.tv_data[:split_idx]
        else:
            return self.tv_data[split_idx:]

    def get_combined_data(self, split: str = 'train', split_ratio: float = 0.8) -> List[Dict]:
        """Get combined movie + TV data"""
        return self.get_movie_data(split, split_ratio) + self.get_tv_data(split, split_ratio)


# Global data loader instance (lazy initialized)
_global_data_loader: Optional[RealDataLoader] = None

def get_data_loader(data_dir: Path) -> RealDataLoader:
    """Get or create the global data loader"""
    global _global_data_loader
    if _global_data_loader is None:
        _global_data_loader = RealDataLoader(data_dir)
    return _global_data_loader
