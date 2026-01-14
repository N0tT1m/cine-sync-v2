"""
TV Data Preprocessor for State-of-the-Art Models
Processes multiple TV datasets and creates unified training data
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

class TVDataPreprocessor:
    """Comprehensive TV data preprocessor for SOTA models"""
    
    def __init__(self,
                 data_dir: str = "/Users/timmy/workspace/ai-apps/cine-sync-v2/data/tv",
                 output_dir: str = "./processed_data",
                 min_shows_per_actor: int = 2,
                 min_shows_per_genre: int = 10,
                 max_cast_size: int = 15,
                 max_genres_per_show: int = 5):
        
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.min_shows_per_actor = min_shows_per_actor
        self.min_shows_per_genre = min_shows_per_genre
        self.max_cast_size = max_cast_size
        self.max_genres_per_show = max_genres_per_show
        
        # Encoders for categorical features
        self.encoders = {
            'genres': LabelEncoder(),
            'actors': LabelEncoder(),
            'networks': LabelEncoder(),
            'creators': LabelEncoder(),
            'languages': LabelEncoder(),
            'status': LabelEncoder(),
            'countries': LabelEncoder()
        }
        
        # Vocabulary mappings
        self.vocab_mappings = {}
        self.reverse_mappings = {}
        
        # Statistics
        self.stats = {
            'total_shows': 0,
            'unique_actors': 0,
            'unique_genres': 0,
            'unique_networks': 0,
            'avg_episodes_per_show': 0,
            'avg_seasons_per_show': 0
        }
    
    def load_tmdb_data(self) -> pd.DataFrame:
        """Load and process TMDB TV dataset"""
        tmdb_file = self.data_dir / "misc" / "TMDB_tv_dataset_v3.csv"
        
        if not tmdb_file.exists():
            logger.warning(f"TMDB file not found: {tmdb_file}")
            return pd.DataFrame()
        
        logger.info("Loading TMDB TV dataset...")
        
        try:
            df = pd.read_csv(tmdb_file, low_memory=False)
            logger.info(f"Loaded {len(df)} shows from TMDB")
            
            # Clean and process TMDB data
            df = self._clean_tmdb_data(df)
            
            return df
        except Exception as e:
            logger.error(f"Error loading TMDB data: {e}")
            return pd.DataFrame()
    
    def _clean_tmdb_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and standardize TMDB data"""
        
        # Remove quotes from column names and data
        df.columns = [col.strip('\"') for col in df.columns]
        
        # Essential columns
        required_cols = ['id', 'name', 'overview', 'genres', 'vote_average', 'vote_count']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            logger.warning(f"Missing required columns: {missing_cols}")
            return pd.DataFrame()
        
        # Remove rows with missing essential data
        df = df.dropna(subset=['id', 'name', 'overview'])
        
        # Clean text fields
        df['name'] = df['name'].astype(str).str.strip()
        df['overview'] = df['overview'].astype(str).str.strip()
        
        # Parse JSON-like fields
        df = self._parse_json_fields(df)
        
        # Convert numeric fields
        numeric_fields = ['vote_average', 'vote_count', 'number_of_seasons', 'number_of_episodes']
        for field in numeric_fields:
            if field in df.columns:
                df[field] = pd.to_numeric(df[field], errors='coerce').fillna(0)
        
        # Parse dates
        date_fields = ['first_air_date', 'last_air_date']
        for field in date_fields:
            if field in df.columns:
                df[field] = pd.to_datetime(df[field], errors='coerce')
        
        # Filter out shows with very low quality indicators
        df = df[
            (df['vote_count'] >= 10) &  # At least 10 votes
            (df['vote_average'] > 0) &  # Valid rating
            (df['overview'].str.len() > 20)  # Meaningful description
        ]
        
        logger.info(f"Cleaned TMDB data: {len(df)} shows remaining")
        return df
    
    def _parse_json_fields(self, df: pd.DataFrame) -> pd.DataFrame:
        """Parse JSON-like fields in TMDB data"""
        
        json_fields = {
            'genres': 'name',
            'networks': 'name', 
            'created_by': 'name',
            'production_companies': 'name',
            'production_countries': 'name',
            'spoken_languages': 'name'
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
            # Handle malformed JSON by using regex
            import re
            pattern = rf'"{key}"\s*:\s*"([^"]*)"'
            matches = re.findall(pattern, str(json_str))
            return [match.strip() for match in matches if match.strip()]
        except:
            return []
    
    def load_metacritic_data(self) -> pd.DataFrame:
        """Load and process Metacritic TV dataset"""
        metacritic_file = self.data_dir / "misc" / "metacritic_tv.csv"
        
        if not metacritic_file.exists():
            logger.warning(f"Metacritic file not found: {metacritic_file}")
            return pd.DataFrame()
        
        logger.info("Loading Metacritic TV dataset...")
        
        try:
            df = pd.read_csv(metacritic_file)
            logger.info(f"Loaded {len(df)} shows from Metacritic")
            
            # Clean metacritic data
            df = df.dropna(subset=['title', 'summary'])
            df['title'] = df['title'].astype(str).str.strip()
            df['summary'] = df['summary'].astype(str).str.strip()
            
            # Convert scores
            df['metascore'] = pd.to_numeric(df['metascore'], errors='coerce').fillna(0)
            df['user_score'] = pd.to_numeric(df['user_score'], errors='coerce').fillna(0)
            
            return df
        except Exception as e:
            logger.error(f"Error loading Metacritic data: {e}")
            return pd.DataFrame()
    
    def load_netflix_data(self) -> pd.DataFrame:
        """Load Netflix TV shows data"""
        netflix_file = self.data_dir / "netflix" / "netflix_titles.csv"
        
        if not netflix_file.exists():
            logger.warning(f"Netflix file not found: {netflix_file}")
            return pd.DataFrame()
        
        logger.info("Loading Netflix dataset...")
        
        try:
            df = pd.read_csv(netflix_file)
            # Filter for TV shows only
            df = df[df['type'] == 'TV Show'].copy()
            logger.info(f"Loaded {len(df)} TV shows from Netflix")
            
            # Clean data
            df = df.dropna(subset=['title', 'description'])
            df['title'] = df['title'].astype(str).str.strip()
            df['description'] = df['description'].astype(str).str.strip()
            
            # Parse duration (e.g., "3 Seasons" -> 3)
            df['num_seasons'] = df['duration'].str.extract('(\\d+)').astype(float).fillna(1)
            
            return df
        except Exception as e:
            logger.error(f"Error loading Netflix data: {e}")
            return pd.DataFrame()
    
    def unify_datasets(self) -> pd.DataFrame:
        """Combine and unify all TV datasets"""
        logger.info("Unifying TV datasets...")
        
        # Load all datasets
        tmdb_df = self.load_tmdb_data()
        metacritic_df = self.load_metacritic_data()
        netflix_df = self.load_netflix_data()
        
        unified_shows = []
        show_id = 0
        
        # Process TMDB data (primary source)
        if not tmdb_df.empty:
            for _, row in tmdb_df.iterrows():
                show = self._process_tmdb_show(row, show_id)
                if show:
                    unified_shows.append(show)
                    show_id += 1
        
        # Process Metacritic data (supplement scores)
        metacritic_lookup = {}
        if not metacritic_df.empty:
            for _, row in metacritic_df.iterrows():
                title_clean = self._clean_title(row['title'])
                metacritic_lookup[title_clean] = {
                    'metascore': row['metascore'],
                    'user_score': row['user_score']
                }
        
        # Process Netflix data (add platform info)
        netflix_lookup = set()
        if not netflix_df.empty:
            for _, row in netflix_df.iterrows():
                title_clean = self._clean_title(row['title'])
                netflix_lookup.add(title_clean)
        
        # Enhance unified shows with additional data
        for show in unified_shows:
            title_clean = self._clean_title(show['title'])
            
            # Add Metacritic scores
            if title_clean in metacritic_lookup:
                show['metascore'] = metacritic_lookup[title_clean]['metascore']
                show['user_score'] = metacritic_lookup[title_clean]['user_score']
            
            # Add platform availability
            show['on_netflix'] = title_clean in netflix_lookup
        
        # Convert to DataFrame
        unified_df = pd.DataFrame(unified_shows)
        
        logger.info(f"Unified dataset created with {len(unified_df)} shows")
        return unified_df
    
    def _process_tmdb_show(self, row: pd.Series, show_id: int) -> Optional[Dict]:
        """Process a single TMDB show"""
        try:
            # Extract genres
            genres = row.get('genres_parsed', [])
            if isinstance(genres, str):
                genres = [genres]
            elif not isinstance(genres, list):
                genres = []
            
            # Extract networks
            networks = row.get('networks_parsed', [])
            if isinstance(networks, str):
                networks = [networks]
            elif not isinstance(networks, list):
                networks = []
            
            # Extract creators
            creators = row.get('created_by_parsed', [])
            if isinstance(creators, str):
                creators = [creators]
            elif not isinstance(creators, list):
                creators = []
            
            # Create show dictionary
            show = {
                'id': show_id,
                'tmdb_id': row['id'],
                'title': row['name'],
                'overview': row['overview'],
                'genres': genres[:self.max_genres_per_show],
                'networks': networks,
                'creators': creators,
                'vote_average': float(row.get('vote_average', 0)),
                'vote_count': int(row.get('vote_count', 0)),
                'popularity': float(row.get('popularity', 0)),
                'num_seasons': int(row.get('number_of_seasons', 1)),
                'num_episodes': int(row.get('number_of_episodes', 1)),
                'first_air_date': row.get('first_air_date'),
                'last_air_date': row.get('last_air_date'),
                'status': row.get('status', 'Unknown'),
                'original_language': row.get('original_language', 'en'),
                'adult': bool(row.get('adult', False)),
                'metascore': 0,  # Will be filled from Metacritic
                'user_score': 0,  # Will be filled from Metacritic
                'on_netflix': False,  # Will be filled from Netflix data
            }
            
            # Calculate derived features
            if show['first_air_date'] and not pd.isna(show['first_air_date']):
                try:
                    air_year = show['first_air_date'].year
                    show['years_since_aired'] = 2024 - air_year
                    show['decade'] = air_year // 10 * 10
                except:
                    show['years_since_aired'] = 0
                    show['decade'] = 2020
            else:
                show['years_since_aired'] = 0
                show['decade'] = 2020
            
            # Validate required fields
            if not show['title'] or not show['overview'] or len(show['overview']) < 10:
                return None
            
            return show
            
        except Exception as e:
            logger.warning(f"Error processing show: {e}")
            return None
    
    def _clean_title(self, title: str) -> str:
        """Clean title for matching across datasets"""
        if pd.isna(title):
            return ""
        
        # Convert to lowercase and remove special characters
        title = re.sub(r'[^\w\s]', '', str(title).lower())
        title = re.sub(r'\s+', ' ', title).strip()
        
        # Remove common prefixes/suffixes
        title = re.sub(r'^(the|a|an)\s+', '', title)
        title = re.sub(r'\s+(tv|series|show)$', '', title)
        
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
        # Keep only genres that appear in at least min_shows_per_genre shows
        frequent_genres = [genre for genre, count in genre_counts.items() 
                          if count >= self.min_shows_per_genre]
        
        self.encoders['genres'].fit(['<UNK>'] + frequent_genres)
        self.vocab_mappings['genres'] = {genre: idx for idx, genre in 
                                       enumerate(self.encoders['genres'].classes_)}
        vocab_sizes['genres'] = len(self.encoders['genres'].classes_)
        
        # Build network vocabulary
        all_networks = []
        for networks in df['networks']:
            if isinstance(networks, list):
                all_networks.extend(networks)
        
        network_counts = Counter(all_networks)
        frequent_networks = [net for net, count in network_counts.items() if count >= 2]
        
        self.encoders['networks'].fit(['<UNK>'] + frequent_networks)
        self.vocab_mappings['networks'] = {net: idx for idx, net in 
                                         enumerate(self.encoders['networks'].classes_)}
        vocab_sizes['networks'] = len(self.encoders['networks'].classes_)
        
        # Build creator vocabulary
        all_creators = []
        for creators in df['creators']:
            if isinstance(creators, list):
                all_creators.extend(creators)
        
        creator_counts = Counter(all_creators)
        frequent_creators = [creator for creator, count in creator_counts.items() if count >= 2]
        
        self.encoders['creators'].fit(['<UNK>'] + frequent_creators)
        self.vocab_mappings['creators'] = {creator: idx for idx, creator in 
                                         enumerate(self.encoders['creators'].classes_)}
        vocab_sizes['creators'] = len(self.encoders['creators'].classes_)
        
        # Build status vocabulary
        statuses = df['status'].fillna('Unknown').unique().tolist()
        self.encoders['status'].fit(statuses)
        self.vocab_mappings['status'] = {status: idx for idx, status in 
                                       enumerate(self.encoders['status'].classes_)}
        vocab_sizes['status'] = len(self.encoders['status'].classes_)
        
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
        def encode_genre_list(genres):
            if not isinstance(genres, list):
                return [0]  # UNK
            encoded = []
            for genre in genres:
                if genre in self.vocab_mappings['genres']:
                    encoded.append(self.vocab_mappings['genres'][genre])
                else:
                    encoded.append(0)  # UNK
            return encoded if encoded else [0]
        
        df['encoded_genres'] = df['genres'].apply(encode_genre_list)
        
        # Encode networks
        def encode_network_list(networks):
            if not isinstance(networks, list):
                return [0]  # UNK
            encoded = []
            for network in networks:
                if network in self.vocab_mappings['networks']:
                    encoded.append(self.vocab_mappings['networks'][network])
                else:
                    encoded.append(0)  # UNK
            return encoded if encoded else [0]
        
        df['encoded_networks'] = df['networks'].apply(encode_network_list)
        
        # Encode creators
        def encode_creator_list(creators):
            if not isinstance(creators, list):
                return [0]  # UNK
            encoded = []
            for creator in creators:
                if creator in self.vocab_mappings['creators']:
                    encoded.append(self.vocab_mappings['creators'][creator])
                else:
                    encoded.append(0)  # UNK
            return encoded if encoded else [0]
        
        df['encoded_creators'] = df['creators'].apply(encode_creator_list)
        
        # Encode status
        df['encoded_status'] = df['status'].apply(
            lambda x: self.vocab_mappings['status'].get(x, 0)
        )
        
        # Encode language
        df['encoded_language'] = df['original_language'].apply(
            lambda x: self.vocab_mappings['languages'].get(x, 0)
        )
        
        return df
    
    def create_training_data(self, df: pd.DataFrame) -> Dict[str, List]:
        """Create final training data format"""
        logger.info("Creating training data...")
        
        training_data = []
        
        for _, row in df.iterrows():
            # Prepare text
            text = f"{row['title']} [SEP] {row['overview']}"
            
            # Prepare categorical features
            categorical_features = {
                'genres': row['encoded_genres'],
                'networks': row['encoded_networks'], 
                'creators': row['encoded_creators'],
                'status': [row['encoded_status']],
                'language': [row['encoded_language']]
            }
            
            # Prepare numerical features
            numerical_features = [
                float(row['vote_average']),
                float(row['vote_count']) / 10000.0,  # Normalize
                float(row['popularity']) / 100.0,    # Normalize
                float(row['num_seasons']),
                float(row['num_episodes']) / 100.0,  # Normalize
                float(row['years_since_aired']) / 50.0,  # Normalize
                float(row['metascore']) / 100.0,     # Normalize
                float(row['user_score']) / 10.0,     # Normalize
                float(row['on_netflix']),             # Binary
                float(row['adult'])                   # Binary
            ]
            
            sample = {
                'id': row['id'],
                'title': row['title'],
                'text': text,
                'categorical_features': categorical_features,
                'numerical_features': numerical_features,
                'metadata': {
                    'tmdb_id': row.get('tmdb_id'),
                    'genres': row['genres'],
                    'networks': row['networks'],
                    'creators': row['creators'],
                    'vote_average': row['vote_average'],
                    'vote_count': row['vote_count'],
                    'num_seasons': row['num_seasons'],
                    'num_episodes': row['num_episodes']
                }
            }
            
            training_data.append(sample)
        
        return training_data
    
    def create_graph_data(self, df: pd.DataFrame) -> Dict:
        """Create graph data for GNN training"""
        logger.info("Creating graph data...")
        
        # Node mappings
        show_to_idx = {show_id: idx for idx, show_id in enumerate(df['id'])}
        
        # Build actor vocabulary if we had actor data
        # For now, we'll use simplified relationships
        
        edge_index_dict = {}
        
        # Show-Genre edges
        show_genre_edges = [[], []]
        for idx, row in df.iterrows():
            show_idx = idx
            for genre in row['encoded_genres'][:self.max_genres_per_show]:
                show_genre_edges[0].append(show_idx)
                show_genre_edges[1].append(genre)
        
        edge_index_dict[('show', 'has_genre', 'genre')] = torch.tensor(show_genre_edges)
        
        # Show-Network edges
        show_network_edges = [[], []]
        for idx, row in df.iterrows():
            show_idx = idx
            for network in row['encoded_networks'][:3]:  # Limit to 3 networks
                show_network_edges[0].append(show_idx)
                show_network_edges[1].append(network)
        
        edge_index_dict[('show', 'on_network', 'network')] = torch.tensor(show_network_edges)
        
        # Show-Creator edges
        show_creator_edges = [[], []]
        for idx, row in df.iterrows():
            show_idx = idx
            for creator in row['encoded_creators'][:5]:  # Limit to 5 creators
                show_creator_edges[0].append(show_idx)
                show_creator_edges[1].append(creator)
        
        edge_index_dict[('show', 'created_by', 'creator')] = torch.tensor(show_creator_edges)
        
        # Create similarity edges based on shared genres
        similar_edges = self._create_similarity_edges(df)
        edge_index_dict[('show', 'similar_to', 'show')] = torch.tensor(similar_edges)
        
        return {
            'edge_index_dict': edge_index_dict,
            'num_shows': len(df),
            'num_genres': len(self.vocab_mappings['genres']),
            'num_networks': len(self.vocab_mappings['networks']),
            'num_creators': len(self.vocab_mappings['creators'])
        }
    
    def _create_similarity_edges(self, df: pd.DataFrame, threshold: float = 0.3) -> List[List[int]]:
        """Create similarity edges between shows based on genre overlap"""
        
        similar_edges = [[], []]
        
        for i, row1 in df.iterrows():
            genres1 = set(row1['encoded_genres'])
            
            for j, row2 in df.iterrows():
                if i >= j:  # Avoid duplicates and self-loops
                    continue
                
                genres2 = set(row2['encoded_genres'])
                
                # Calculate Jaccard similarity
                intersection = len(genres1.intersection(genres2))
                union = len(genres1.union(genres2))
                
                if union > 0:
                    similarity = intersection / union
                    if similarity >= threshold:
                        similar_edges[0].extend([i, j])
                        similar_edges[1].extend([j, i])  # Bidirectional
        
        return similar_edges
    
    def split_data(self, training_data: List[Dict], 
                   train_ratio: float = 0.8, 
                   val_ratio: float = 0.1) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """Split data into train/val/test sets"""
        
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
                'min_shows_per_actor': self.min_shows_per_actor,
                'min_shows_per_genre': self.min_shows_per_genre,
                'max_cast_size': self.max_cast_size,
                'max_genres_per_show': self.max_genres_per_show
            },
            'stats': self.stats
        }
        
        with open(self.output_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"All processed data saved to {self.output_dir}")
    
    def process_all(self):
        """Main processing pipeline"""
        logger.info("Starting TV data preprocessing pipeline...")
        
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
            'total_shows': len(encoded_df),
            'unique_genres': len(self.vocab_mappings['genres']),
            'unique_networks': len(self.vocab_mappings['networks']),
            'unique_creators': len(self.vocab_mappings['creators']),
            'avg_episodes_per_show': encoded_df['num_episodes'].mean(),
            'avg_seasons_per_show': encoded_df['num_seasons'].mean()
        })
        
        # Step 8: Save everything
        self.save_processed_data(train_data, val_data, test_data, graph_data, vocab_sizes)
        
        logger.info("TV data preprocessing completed successfully!")
        logger.info(f"Final statistics: {self.stats}")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Preprocess TV data for SOTA models')
    parser.add_argument('--data_dir', type=str, 
                       default='/Users/timmy/workspace/ai-apps/cine-sync-v2/tv',
                       help='Directory containing TV datasets')
    parser.add_argument('--output_dir', type=str, 
                       default='./processed_data',
                       help='Output directory for processed data')
    parser.add_argument('--min_shows_per_genre', type=int, default=10,
                       help='Minimum shows per genre to include')
    parser.add_argument('--max_genres_per_show', type=int, default=5,
                       help='Maximum genres per show')
    
    args = parser.parse_args()
    
    # Create preprocessor
    preprocessor = TVDataPreprocessor(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        min_shows_per_genre=args.min_shows_per_genre,
        max_genres_per_show=args.max_genres_per_show
    )
    
    # Run preprocessing
    preprocessor.process_all()

if __name__ == "__main__":
    main()