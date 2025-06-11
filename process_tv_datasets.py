#!/usr/bin/env python3
"""
TV Show Dataset Processing Pipeline for CineSync v2
Processes the three high-quality TV show datasets for training
"""

import os
import pandas as pd
import numpy as np
import pickle
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import json
import ast

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TVDatasetProcessor:
    """Process and combine TV show datasets for training"""
    
    def __init__(self, data_dir: str = "tv"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Dataset paths - updated for actual file structure
        self.tmdb_path = self.data_dir / "tmdb-tv" / "TMDB_tv_dataset_v3.csv"
        self.mal_animes_path = self.data_dir / "anime-dataset" / "animes.csv"
        self.mal_profiles_path = self.data_dir / "anime-dataset" / "profiles.csv"
        self.mal_reviews_path = self.data_dir / "anime-dataset" / "reviews.csv"
        self.netflix_path = self.data_dir / "netflix-movies-and-tv" / "netflix_titles.csv"
        self.imdb_dir = self.data_dir / "imdb-tv"
        
        # Output paths
        self.processed_dir = self.data_dir / "processed"
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        
        self.encoders = {}
        self.scalers = {}
        
    def process_tmdb_tv_dataset(self) -> pd.DataFrame:
        """
        Process Full TMDb TV Shows Dataset (150K Shows)
        Expected columns: id, name, genres, first_air_date, vote_average, vote_count, etc.
        """
        logger.info("Processing TMDb TV Shows Dataset...")
        
        if not self.tmdb_path.exists():
            logger.warning(f"TMDb dataset not found at {self.tmdb_path}")
            return pd.DataFrame()
        
        try:
            df = pd.read_csv(self.tmdb_path)
            logger.info(f"Loaded {len(df)} TV shows from TMDb dataset")
            
            # Standardize column names
            column_mapping = {
                'id': 'show_id',
                'name': 'title',
                'original_name': 'original_title',
                'first_air_date': 'release_date',
                'vote_average': 'rating',
                'vote_count': 'rating_count',
                'popularity': 'popularity',
                'overview': 'description'
            }
            
            df = df.rename(columns=column_mapping)
            
            # Process genres (convert from JSON array to pipe-separated string)
            if 'genres' in df.columns:
                df['genres'] = df['genres'].apply(self._process_genres_json)
            
            # Add dataset source
            df['source'] = 'tmdb'
            df['content_type'] = 'tv_show'
            
            # Clean and validate data
            df = self._clean_tv_data(df)
            
            return df
            
        except Exception as e:
            logger.error(f"Error processing TMDb dataset: {e}")
            return pd.DataFrame()
    
    def process_myanimelist_dataset(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Process MyAnimeList Dataset (animes.csv, profiles.csv, reviews.csv)
        Returns: (shows_df, ratings_df)
        """
        logger.info("Processing MyAnimeList Dataset...")
        
        if not self.mal_animes_path.exists():
            logger.warning(f"MyAnimeList animes dataset not found at {self.mal_animes_path}")
            return pd.DataFrame(), pd.DataFrame()
        
        try:
            # Load animes data
            animes_df = pd.read_csv(self.mal_animes_path)
            logger.info(f"Loaded {len(animes_df)} animes from MyAnimeList dataset")
            
            # Load reviews for ratings if available
            ratings_df = pd.DataFrame()
            if self.mal_reviews_path.exists():
                reviews_df = pd.read_csv(self.mal_reviews_path)
                ratings_df = self._extract_mal_ratings_from_reviews(reviews_df)
                logger.info(f"Loaded {len(ratings_df)} ratings from reviews")
            
            # Process animes as shows
            shows_df = self._process_mal_animes(animes_df)
            
            return shows_df, ratings_df
            
        except Exception as e:
            logger.error(f"Error processing MyAnimeList dataset: {e}")
            return pd.DataFrame(), pd.DataFrame()
    
    def process_imdb_tv_dataset(self) -> pd.DataFrame:
        """
        Process IMDb TV Series Dataset from multiple genre files
        """
        logger.info("Processing IMDb TV Series Dataset...")
        
        if not self.imdb_dir.exists():
            logger.warning(f"IMDb directory not found at {self.imdb_dir}")
            return pd.DataFrame()
        
        try:
            all_imdb_data = []
            
            # Process each genre file
            for genre_file in self.imdb_dir.glob("*_series.csv"):
                try:
                    genre_name = genre_file.stem.replace('_series', '').replace('-', ' ').title()
                    df = pd.read_csv(genre_file)
                    df['primary_genre'] = genre_name
                    all_imdb_data.append(df)
                    logger.info(f"Loaded {len(df)} {genre_name} series")
                except Exception as e:
                    logger.warning(f"Error processing {genre_file}: {e}")
                    continue
            
            if not all_imdb_data:
                logger.warning("No IMDb data files processed successfully")
                return pd.DataFrame()
            
            # Combine all genre data
            combined_df = pd.concat(all_imdb_data, ignore_index=True)
            logger.info(f"Combined {len(combined_df)} TV series from IMDb dataset")
            
            # Standardize column names (adjust based on actual columns)
            column_mapping = {
                'Show_Id': 'show_id',
                'Name': 'title',
                'Year': 'release_year', 
                'Genre': 'genres',
                'IMDB_Rating': 'rating',
                'Votes': 'rating_count'
            }
            
            # Only rename columns that exist
            existing_columns = {k: v for k, v in column_mapping.items() if k in combined_df.columns}
            combined_df = combined_df.rename(columns=existing_columns)
            
            # Add dataset source
            combined_df['source'] = 'imdb'
            combined_df['content_type'] = 'tv_show'
            
            # Use primary_genre if genres column doesn't exist
            if 'genres' not in combined_df.columns and 'primary_genre' in combined_df.columns:
                combined_df['genres'] = combined_df['primary_genre']
            
            # Clean and validate data
            combined_df = self._clean_tv_data(combined_df)
            
            return combined_df
            
        except Exception as e:
            logger.error(f"Error processing IMDb dataset: {e}")
            return pd.DataFrame()
    
    def process_netflix_dataset(self) -> pd.DataFrame:
        """
        Process Netflix Movies and TV Shows Dataset
        """
        logger.info("Processing Netflix Dataset...")
        
        if not self.netflix_path.exists():
            logger.warning(f"Netflix dataset not found at {self.netflix_path}")
            return pd.DataFrame()
        
        try:
            df = pd.read_csv(self.netflix_path)
            logger.info(f"Loaded {len(df)} titles from Netflix dataset")
            
            # Filter for TV shows only
            tv_shows = df[df['type'] == 'TV Show'].copy()
            logger.info(f"Found {len(tv_shows)} TV shows in Netflix dataset")
            
            # Standardize column names
            column_mapping = {
                'show_id': 'show_id',
                'title': 'title',
                'director': 'director',
                'cast': 'cast',
                'country': 'country',
                'date_added': 'date_added',
                'release_year': 'release_year',
                'rating': 'content_rating',
                'duration': 'duration',
                'listed_in': 'genres',
                'description': 'description'
            }
            
            # Only rename columns that exist
            existing_columns = {k: v for k, v in column_mapping.items() if k in tv_shows.columns}
            tv_shows = tv_shows.rename(columns=existing_columns)
            
            # Add dataset source
            tv_shows['source'] = 'netflix'
            tv_shows['content_type'] = 'tv_show'
            
            # Clean and validate data
            tv_shows = self._clean_tv_data(tv_shows)
            
            return tv_shows
            
        except Exception as e:
            logger.error(f"Error processing Netflix dataset: {e}")
            return pd.DataFrame()
    
    def _extract_mal_shows(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract unique shows from MyAnimeList data"""
        # Group by anime/show and aggregate information
        shows = df.groupby('anime_id').agg({
            'Name': 'first',
            'Genres': 'first',
            'Type': 'first',
            'Episodes': 'first',
            'Rating': 'first',
            'Score': 'mean',  # Average user score
            'Scored By': 'first',
            'Popularity': 'first',
            'Members': 'first',
            'Favorites': 'first',
            'Aired': 'first',
            'Premiered': 'first',
            'Status': 'first',
            'Source': 'first',
            'Duration': 'first'
        }).reset_index()
        
        # Standardize column names
        shows = shows.rename(columns={
            'anime_id': 'show_id',
            'Name': 'title',
            'Genres': 'genres',
            'Score': 'rating',
            'Scored By': 'rating_count',
            'Aired': 'release_date'
        })
        
        # Add dataset source
        shows['source'] = 'myanimelist'
        shows['content_type'] = 'tv_show'
        
        # Filter for TV shows only
        tv_types = ['TV', 'TV Short', 'TV Special']
        shows = shows[shows['Type'].isin(tv_types)]
        
        return self._clean_tv_data(shows)
    
    def _process_mal_animes(self, animes_df: pd.DataFrame) -> pd.DataFrame:
        """Process MyAnimeList animes data"""
        # Standardize column names
        shows_df = animes_df.copy()
        
        column_mapping = {
            'anime_id': 'show_id',
            'Name': 'title',
            'Genres': 'genres',
            'Type': 'show_type',
            'Episodes': 'episode_count',
            'Rating': 'content_rating',
            'Score': 'rating',
            'Scored By': 'rating_count',
            'Popularity': 'popularity',
            'Members': 'members',
            'Favorites': 'favorites',
            'Aired': 'release_date',
            'Premiered': 'premiere_season',
            'Status': 'status',
            'Source': 'source_material',
            'Duration': 'duration'
        }
        
        # Only rename columns that exist
        existing_columns = {k: v for k, v in column_mapping.items() if k in shows_df.columns}
        shows_df = shows_df.rename(columns=existing_columns)
        
        # Add dataset source
        shows_df['source'] = 'myanimelist'
        shows_df['content_type'] = 'tv_show'
        
        # Filter for TV shows only (exclude movies, OVAs, etc.)
        if 'show_type' in shows_df.columns:
            tv_types = ['TV', 'TV Short', 'TV Special']
            shows_df = shows_df[shows_df['show_type'].isin(tv_types)]
        
        return shows_df
    
    def _extract_mal_ratings_from_reviews(self, reviews_df: pd.DataFrame) -> pd.DataFrame:
        """Extract ratings from MyAnimeList reviews"""
        if 'user_id' not in reviews_df.columns or 'anime_id' not in reviews_df.columns:
            logger.warning("Required columns not found in reviews data")
            return pd.DataFrame()
        
        # Extract ratings from reviews
        ratings_data = []
        for _, review in reviews_df.iterrows():
            # Try to extract numeric rating from review
            score = None
            if 'scores' in review and pd.notna(review['scores']):
                try:
                    # Parse score if it's in a parseable format
                    score_data = str(review['scores'])
                    # Extract numeric score (this may need adjustment based on actual format)
                    import re
                    scores = re.findall(r'\d+', score_data)
                    if scores:
                        score = float(scores[0]) / 2.0  # Convert to 0-5 scale if needed
                except:
                    pass
            
            if score is not None and 0 <= score <= 5:
                ratings_data.append({
                    'user_id': review['user_id'],
                    'show_id': review['anime_id'],
                    'rating': score
                })
        
        return pd.DataFrame(ratings_data)
    
    def _extract_mal_ratings(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract user ratings from MyAnimeList data"""
        # Extract user ratings (assuming there's user rating data)
        if 'user_id' in df.columns and 'rating' in df.columns:
            ratings = df[['user_id', 'anime_id', 'rating']].copy()
            ratings = ratings.rename(columns={
                'anime_id': 'show_id'
            })
            ratings = ratings.dropna()
            return ratings
        else:
            logger.warning("No user rating data found in MyAnimeList dataset")
            return pd.DataFrame()
    
    def _process_genres_json(self, genres_str: str) -> str:
        """Convert JSON genre array to pipe-separated string"""
        if pd.isna(genres_str) or genres_str == '':
            return ''
        
        try:
            if genres_str.startswith('['):
                # JSON array format
                genres_list = ast.literal_eval(genres_str)
                if isinstance(genres_list, list):
                    return '|'.join([g['name'] if isinstance(g, dict) else str(g) for g in genres_list])
            else:
                # Already processed or different format
                return str(genres_str)
        except:
            return str(genres_str)
        
        return ''
    
    def _clean_tv_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and standardize TV show data"""
        if df.empty:
            return df
        
        # Remove rows with missing essential data
        essential_cols = ['show_id', 'title']
        for col in essential_cols:
            if col in df.columns:
                df = df.dropna(subset=[col])
        
        # Standardize ratings to 0-5 scale
        if 'rating' in df.columns:
            df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
            # Convert different rating scales to 0-5
            max_rating = df['rating'].max()
            if max_rating > 5:
                df['rating'] = (df['rating'] / max_rating) * 5
        
        # Clean text fields
        text_cols = ['title', 'original_title', 'description']
        for col in text_cols:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip()
        
        # Process release dates/years
        if 'release_date' in df.columns:
            df['release_year'] = pd.to_datetime(df['release_date'], errors='coerce').dt.year
        elif 'release_year' in df.columns:
            df['release_year'] = pd.to_numeric(df['release_year'], errors='coerce')
        
        # Fill missing values
        df['genres'] = df.get('genres', '').fillna('')
        df['rating'] = df.get('rating', 0).fillna(0)
        df['rating_count'] = df.get('rating_count', 0).fillna(0)
        
        return df
    
    def combine_datasets(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Process and combine all TV show datasets
        Returns: (combined_shows_df, combined_ratings_df)
        """
        logger.info("Processing and combining all TV show datasets...")
        
        all_shows = []
        all_ratings = []
        
        # Process TMDb dataset
        tmdb_shows = self.process_tmdb_tv_dataset()
        if not tmdb_shows.empty:
            all_shows.append(tmdb_shows)
        
        # Process MyAnimeList dataset
        mal_shows, mal_ratings = self.process_myanimelist_dataset()
        if not mal_shows.empty:
            all_shows.append(mal_shows)
        if not mal_ratings.empty:
            all_ratings.append(mal_ratings)
        
        # Process IMDb dataset
        imdb_shows = self.process_imdb_tv_dataset()
        if not imdb_shows.empty:
            all_shows.append(imdb_shows)
        
        # Process Netflix dataset
        netflix_shows = self.process_netflix_dataset()
        if not netflix_shows.empty:
            all_shows.append(netflix_shows)
        
        # Combine shows
        if all_shows:
            combined_shows = pd.concat(all_shows, ignore_index=True, sort=False)
            combined_shows = self._deduplicate_shows(combined_shows)
        else:
            combined_shows = pd.DataFrame()
        
        # Combine ratings
        if all_ratings:
            combined_ratings = pd.concat(all_ratings, ignore_index=True, sort=False)
        else:
            combined_ratings = pd.DataFrame()
        
        logger.info(f"Combined dataset: {len(combined_shows)} shows, {len(combined_ratings)} ratings")
        
        return combined_shows, combined_ratings
    
    def _deduplicate_shows(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicate shows based on title similarity"""
        # Simple deduplication by title (you might want to use more sophisticated matching)
        df['title_lower'] = df['title'].str.lower().str.strip()
        
        # Keep the entry with the highest rating count for duplicates
        df = df.sort_values('rating_count', ascending=False)
        df = df.drop_duplicates(subset=['title_lower'], keep='first')
        df = df.drop('title_lower', axis=1)
        
        return df
    
    def create_training_data(self, min_ratings_per_show: int = 5) -> Dict:
        """
        Create training-ready data with proper ID mappings
        """
        logger.info("Creating training data for TV shows...")
        
        shows_df, ratings_df = self.combine_datasets()
        
        if shows_df.empty:
            logger.error("No TV show data available for training")
            return {}
        
        # Filter shows with minimum ratings
        if not ratings_df.empty:
            show_rating_counts = ratings_df['show_id'].value_counts()
            valid_shows = show_rating_counts[show_rating_counts >= min_ratings_per_show].index
            shows_df = shows_df[shows_df['show_id'].isin(valid_shows)]
            ratings_df = ratings_df[ratings_df['show_id'].isin(valid_shows)]
        
        # Create ID mappings
        show_ids = sorted(shows_df['show_id'].unique())
        self.encoders['show_id'] = {show_id: idx for idx, show_id in enumerate(show_ids)}
        self.encoders['show_id_reverse'] = {idx: show_id for show_id, idx in self.encoders['show_id'].items()}
        
        if not ratings_df.empty:
            user_ids = sorted(ratings_df['user_id'].unique())
            self.encoders['user_id'] = {user_id: idx for idx, user_id in enumerate(user_ids)}
            self.encoders['user_id_reverse'] = {idx: user_id for user_id, idx in self.encoders['user_id'].items()}
        
        # Process genres
        all_genres = set()
        for genres_str in shows_df['genres'].fillna(''):
            if genres_str:
                genres = genres_str.split('|')
                all_genres.update([g.strip() for g in genres if g.strip()])
        
        genre_list = sorted(list(all_genres))
        self.encoders['genres'] = {genre: idx for idx, genre in enumerate(genre_list)}
        
        # Create genre features for shows
        shows_df['genre_features'] = shows_df['genres'].apply(
            lambda x: self._create_genre_vector(x, self.encoders['genres'])
        )
        
        # Save processed data
        output_data = {
            'shows_df': shows_df,
            'ratings_df': ratings_df,
            'num_shows': len(show_ids),
            'num_users': len(ratings_df['user_id'].unique()) if not ratings_df.empty else 0,
            'num_genres': len(genre_list),
            'genre_list': genre_list,
            'encoders': self.encoders,
            'scalers': self.scalers
        }
        
        # Save to files
        self._save_processed_data(output_data)
        
        return output_data
    
    def _create_genre_vector(self, genres_str: str, genre_encoder: Dict) -> np.ndarray:
        """Create one-hot encoded genre vector"""
        vector = np.zeros(len(genre_encoder))
        
        if pd.isna(genres_str) or not genres_str:
            return vector
        
        genres = genres_str.split('|')
        for genre in genres:
            genre = genre.strip()
            if genre in genre_encoder:
                vector[genre_encoder[genre]] = 1
        
        return vector
    
    def _save_processed_data(self, data: Dict):
        """Save processed data to files"""
        logger.info("Saving processed TV show data...")
        
        # Save DataFrames
        data['shows_df'].to_csv(self.processed_dir / "tv_shows_data.csv", index=False)
        if not data['ratings_df'].empty:
            data['ratings_df'].to_csv(self.processed_dir / "tv_ratings_data.csv", index=False)
        
        # Save metadata
        metadata = {
            'num_shows': data['num_shows'],
            'num_users': data['num_users'],
            'num_genres': data['num_genres'],
            'genre_list': data['genre_list']
        }
        
        with open(self.processed_dir / "tv_metadata.pkl", 'wb') as f:
            pickle.dump(metadata, f)
        
        # Save encoders
        with open(self.processed_dir / "tv_encoders.pkl", 'wb') as f:
            pickle.dump(data['encoders'], f)
        
        # Save scalers
        with open(self.processed_dir / "tv_scalers.pkl", 'wb') as f:
            pickle.dump(data['scalers'], f)
        
        logger.info(f"Saved processed data to {self.processed_dir}")

def main():
    """Main function to process TV show datasets"""
    processor = TVDatasetProcessor()
    
    # Create training data
    training_data = processor.create_training_data()
    
    if training_data:
        logger.info("TV show dataset processing completed successfully!")
        logger.info(f"Processed {training_data['num_shows']} shows")
        logger.info(f"Processed {training_data['num_users']} users")
        logger.info(f"Found {training_data['num_genres']} unique genres")
    else:
        logger.error("Failed to process TV show datasets")

if __name__ == "__main__":
    main()