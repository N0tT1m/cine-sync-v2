import pandas as pd
import numpy as np
import pickle
import logging
import os
from typing import Dict, List, Tuple, Any, Optional
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

logger = logging.getLogger(__name__)


class TVDataLoader:
    """
    Utility class for loading and processing TV show data
    """
    
    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.tv_shows_df = None
        self.tv_lookup = {}
        self.tv_genres = []
        self.tv_id_mappings = {}
        self.tv_rating_scaler = None
        
    def load_tv_shows_data(self, csv_path: str) -> pd.DataFrame:
        """
        Load TV shows data from CSV file
        
        Args:
            csv_path: Path to TV shows CSV file
            
        Returns:
            DataFrame with TV show data
        """
        try:
            self.tv_shows_df = pd.read_csv(csv_path)
            logger.info(f"Loaded {len(self.tv_shows_df)} TV shows from {csv_path}")
            
            # Basic data cleaning
            self.tv_shows_df = self._clean_tv_data(self.tv_shows_df)
            
            return self.tv_shows_df
            
        except Exception as e:
            logger.error(f"Error loading TV shows data: {e}")
            raise
    
    def _clean_tv_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and preprocess TV show data"""
        try:
            # Remove duplicates
            initial_count = len(df)
            df = df.drop_duplicates(subset=['id'] if 'id' in df.columns else None)
            logger.info(f"Removed {initial_count - len(df)} duplicate TV shows")
            
            # Handle missing values
            if 'name' in df.columns:
                df = df.dropna(subset=['name'])
            elif 'title' in df.columns:
                df = df.dropna(subset=['title'])
            
            # Clean episode count and season count
            if 'number_of_episodes' in df.columns:
                df['number_of_episodes'] = pd.to_numeric(df['number_of_episodes'], errors='coerce')
                df['number_of_episodes'] = df['number_of_episodes'].fillna(0)
            
            if 'number_of_seasons' in df.columns:
                df['number_of_seasons'] = pd.to_numeric(df['number_of_seasons'], errors='coerce')
                df['number_of_seasons'] = df['number_of_seasons'].fillna(0)
            
            # Clean runtime
            if 'episode_run_time' in df.columns:
                # Handle list-like runtime values
                def clean_runtime(runtime):
                    if pd.isna(runtime):
                        return 0
                    if isinstance(runtime, str):
                        # Remove brackets and take first value
                        runtime = runtime.strip('[]').split(',')[0]
                        try:
                            return float(runtime)
                        except:
                            return 0
                    return float(runtime) if not pd.isna(runtime) else 0
                
                df['episode_run_time'] = df['episode_run_time'].apply(clean_runtime)
            
            # Clean genres
            if 'genres' in df.columns:
                df['genres'] = df['genres'].fillna('')
                # Convert list-like strings to pipe-separated
                def clean_genres(genres):
                    if pd.isna(genres) or genres == '':
                        return ''
                    if isinstance(genres, str):
                        # Handle different genre formats
                        if genres.startswith('[') and genres.endswith(']'):
                            # JSON-like format: [{"name": "Drama"}, {"name": "Crime"}]
                            import re
                            names = re.findall(r'"name":\s*"([^"]+)"', genres)
                            return '|'.join(names) if names else genres
                        elif ',' in genres and not '|' in genres:
                            # Comma-separated: Drama, Crime, Thriller
                            return '|'.join([g.strip().strip('"\'') for g in genres.split(',')])
                    return str(genres)
                
                df['genres'] = df['genres'].apply(clean_genres)
            
            # Standardize column names
            column_mapping = {
                'name': 'title',
                'first_air_date': 'release_date',
                'vote_average': 'rating',
                'number_of_episodes': 'episode_count',
                'number_of_seasons': 'season_count'
            }
            
            for old_col, new_col in column_mapping.items():
                if old_col in df.columns and new_col not in df.columns:
                    df[new_col] = df[old_col]
            
            logger.info(f"Cleaned TV data: {len(df)} shows remaining")
            return df
            
        except Exception as e:
            logger.error(f"Error cleaning TV data: {e}")
            return df
    
    def create_tv_lookup(self) -> Dict[int, Dict]:
        """Create TV show lookup dictionary"""
        if self.tv_shows_df is None:
            raise ValueError("TV shows data not loaded")
        
        try:
            self.tv_lookup = {}
            
            for _, row in self.tv_shows_df.iterrows():
                show_id = row.get('id', row.name)
                
                show_info = {
                    'title': row.get('title', row.get('name', 'Unknown')),
                    'genres': row.get('genres', ''),
                    'episode_count': int(row.get('episode_count', 0)),
                    'season_count': int(row.get('season_count', 0)),
                    'episode_run_time': float(row.get('episode_run_time', 0)),
                    'status': row.get('status', 'ended').lower(),
                    'rating': float(row.get('rating', 0)),
                    'release_date': str(row.get('release_date', '')),
                    'overview': row.get('overview', ''),
                }
                
                self.tv_lookup[int(show_id)] = show_info
            
            logger.info(f"Created TV lookup with {len(self.tv_lookup)} shows")
            return self.tv_lookup
            
        except Exception as e:
            logger.error(f"Error creating TV lookup: {e}")
            raise
    
    def extract_tv_genres(self) -> List[str]:
        """Extract unique genres from TV show data"""
        if not self.tv_lookup:
            self.create_tv_lookup()
        
        try:
            all_genres = set()
            
            for show_info in self.tv_lookup.values():
                genres = show_info.get('genres', '')
                if isinstance(genres, str) and genres.strip():
                    genre_list = [g.strip() for g in genres.split('|') if g.strip()]
                    all_genres.update(genre_list)
            
            self.tv_genres = sorted(list(all_genres))
            logger.info(f"Extracted {len(self.tv_genres)} TV genres")
            return self.tv_genres
            
        except Exception as e:
            logger.error(f"Error extracting TV genres: {e}")
            return []
    
    def create_tv_id_mappings(self) -> Dict[str, Any]:
        """Create ID mappings for TV shows and users"""
        if not self.tv_lookup:
            self.create_tv_lookup()
        
        try:
            show_ids = sorted(list(self.tv_lookup.keys()))
            
            # Create show ID mappings
            show_id_to_idx = {show_id: idx for idx, show_id in enumerate(show_ids)}
            idx_to_show_id = {idx: show_id for show_id, idx in show_id_to_idx.items()}
            
            # For now, create dummy user mappings (would be populated from actual user data)
            user_ids = list(range(1000))  # Placeholder for 1000 users
            user_id_to_idx = {user_id: idx for idx, user_id in enumerate(user_ids)}
            idx_to_user_id = {idx: user_id for user_id, idx in user_id_to_idx.items()}
            
            self.tv_id_mappings = {
                'show_id_to_idx': show_id_to_idx,
                'idx_to_show_id': idx_to_show_id,
                'user_id_to_idx': user_id_to_idx,
                'idx_to_user_id': idx_to_user_id,
                'num_shows': len(show_ids),
                'num_users': len(user_ids)
            }
            
            logger.info(f"Created TV ID mappings: {len(show_ids)} shows, {len(user_ids)} users")
            return self.tv_id_mappings
            
        except Exception as e:
            logger.error(f"Error creating TV ID mappings: {e}")
            raise
    
    def prepare_tv_features(self) -> pd.DataFrame:
        """Prepare TV show features for model training"""
        if self.tv_shows_df is None:
            raise ValueError("TV shows data not loaded")
        
        try:
            # Extract genre features
            if not self.tv_genres:
                self.extract_tv_genres()
            
            # Create genre one-hot encoding
            genre_features = []
            for _, row in self.tv_shows_df.iterrows():
                genres = row.get('genres', '')
                show_genres = set(g.strip().lower() for g in str(genres).split('|') if g.strip())
                
                genre_vector = []
                for genre in self.tv_genres:
                    genre_vector.append(1 if genre.lower() in show_genres else 0)
                
                genre_features.append(genre_vector)
            
            # Create genre columns
            genre_df = pd.DataFrame(genre_features, columns=self.tv_genres)
            
            # Combine with original features
            feature_columns = ['id', 'title', 'episode_count', 'season_count', 
                             'episode_run_time', 'status', 'rating']
            
            existing_columns = [col for col in feature_columns if col in self.tv_shows_df.columns]
            features_df = self.tv_shows_df[existing_columns].copy()
            
            # Add genre features
            features_df = pd.concat([features_df.reset_index(drop=True), 
                                   genre_df.reset_index(drop=True)], axis=1)
            
            logger.info(f"Prepared TV features: {len(features_df)} shows, {len(features_df.columns)} features")
            return features_df
            
        except Exception as e:
            logger.error(f"Error preparing TV features: {e}")
            raise
    
    def create_rating_scaler(self, ratings_data: Optional[pd.DataFrame] = None) -> MinMaxScaler:
        """Create rating scaler for TV shows"""
        try:
            if ratings_data is not None and 'rating' in ratings_data.columns:
                # Use actual ratings data
                ratings = ratings_data['rating'].dropna().values.reshape(-1, 1)
            else:
                # Use show ratings as fallback
                ratings = []
                for show_info in self.tv_lookup.values():
                    rating = show_info.get('rating', 0)
                    if rating > 0:
                        ratings.append(rating)
                
                if not ratings:
                    ratings = [1, 2, 3, 4, 5]  # Default range
                
                ratings = np.array(ratings).reshape(-1, 1)
            
            self.tv_rating_scaler = MinMaxScaler()
            self.tv_rating_scaler.fit(ratings)
            
            logger.info(f"Created TV rating scaler with range {ratings.min():.2f} to {ratings.max():.2f}")
            return self.tv_rating_scaler
            
        except Exception as e:
            logger.error(f"Error creating TV rating scaler: {e}")
            raise
    
    def save_tv_data(self, output_dir: str):
        """Save all processed TV data"""
        try:
            os.makedirs(output_dir, exist_ok=True)
            
            # Save TV lookup
            if self.tv_lookup:
                with open(os.path.join(output_dir, 'tv_lookup.pkl'), 'wb') as f:
                    pickle.dump(self.tv_lookup, f)
                logger.info("Saved TV lookup")
            
            # Save TV ID mappings
            if self.tv_id_mappings:
                with open(os.path.join(output_dir, 'tv_id_mappings.pkl'), 'wb') as f:
                    pickle.dump(self.tv_id_mappings, f)
                logger.info("Saved TV ID mappings")
            
            # Save TV rating scaler
            if self.tv_rating_scaler:
                with open(os.path.join(output_dir, 'tv_rating_scaler.pkl'), 'wb') as f:
                    pickle.dump(self.tv_rating_scaler, f)
                logger.info("Saved TV rating scaler")
            
            # Save processed TV data
            if self.tv_shows_df is not None:
                features_df = self.prepare_tv_features()
                features_df.to_csv(os.path.join(output_dir, 'tv_data.csv'), index=False)
                logger.info("Saved TV features data")
            
            # Save TV genres
            if self.tv_genres:
                with open(os.path.join(output_dir, 'tv_genres.pkl'), 'wb') as f:
                    pickle.dump(self.tv_genres, f)
                logger.info("Saved TV genres")
            
            logger.info(f"All TV data saved to {output_dir}")
            
        except Exception as e:
            logger.error(f"Error saving TV data: {e}")
            raise
    
    def load_processed_tv_data(self, data_dir: str):
        """Load already processed TV data"""
        try:
            # Load TV lookup
            tv_lookup_path = os.path.join(data_dir, 'tv_lookup.pkl')
            if os.path.exists(tv_lookup_path):
                with open(tv_lookup_path, 'rb') as f:
                    self.tv_lookup = pickle.load(f)
                logger.info(f"Loaded TV lookup with {len(self.tv_lookup)} shows")
            
            # Load TV ID mappings
            tv_mappings_path = os.path.join(data_dir, 'tv_id_mappings.pkl')
            if os.path.exists(tv_mappings_path):
                with open(tv_mappings_path, 'rb') as f:
                    self.tv_id_mappings = pickle.load(f)
                logger.info("Loaded TV ID mappings")
            
            # Load TV rating scaler
            tv_scaler_path = os.path.join(data_dir, 'tv_rating_scaler.pkl')
            if os.path.exists(tv_scaler_path):
                with open(tv_scaler_path, 'rb') as f:
                    self.tv_rating_scaler = pickle.load(f)
                logger.info("Loaded TV rating scaler")
            
            # Load TV data
            tv_data_path = os.path.join(data_dir, 'tv_data.csv')
            if os.path.exists(tv_data_path):
                self.tv_shows_df = pd.read_csv(tv_data_path)
                logger.info(f"Loaded TV data with {len(self.tv_shows_df)} shows")
            
            # Load TV genres
            tv_genres_path = os.path.join(data_dir, 'tv_genres.pkl')
            if os.path.exists(tv_genres_path):
                with open(tv_genres_path, 'rb') as f:
                    self.tv_genres = pickle.load(f)
                logger.info(f"Loaded {len(self.tv_genres)} TV genres")
            
        except Exception as e:
            logger.error(f"Error loading processed TV data: {e}")
            raise


def process_tv_dataset(csv_path: str, output_dir: str):
    """
    Main function to process TV show dataset
    
    Args:
        csv_path: Path to raw TV shows CSV file
        output_dir: Directory to save processed data
    """
    try:
        logger.info(f"Processing TV dataset from {csv_path}")
        
        # Initialize loader
        loader = TVDataLoader(output_dir)
        
        # Load and process data
        loader.load_tv_shows_data(csv_path)
        loader.create_tv_lookup()
        loader.extract_tv_genres()
        loader.create_tv_id_mappings()
        loader.create_rating_scaler()
        
        # Save all processed data
        loader.save_tv_data(output_dir)
        
        logger.info("TV dataset processing completed successfully")
        
        # Print summary
        print(f"\\nTV Dataset Processing Summary:")
        print(f"- Total TV Shows: {len(loader.tv_lookup):,}")
        print(f"- Unique Genres: {len(loader.tv_genres)}")
        print(f"- Sample Genres: {', '.join(loader.tv_genres[:10])}")
        print(f"- Output Directory: {output_dir}")
        
        return loader
        
    except Exception as e:
        logger.error(f"Error processing TV dataset: {e}")
        raise


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description="Process TV show dataset")
    parser.add_argument("--input", required=True, help="Path to input TV shows CSV file")
    parser.add_argument("--output", required=True, help="Output directory for processed data")
    
    args = parser.parse_args()
    
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Process dataset
    process_tv_dataset(args.input, args.output)