# Two-Tower Model Data Loading and Feature Engineering
# Comprehensive data pipeline for two-tower recommendation models
# Handles both categorical and numerical features with sophisticated engineering

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from typing import Dict, List, Tuple, Optional, Any
import logging
import pickle
from collections import defaultdict


class TwoTowerDataset(Dataset):
    """
    PyTorch Dataset for basic Two-Tower models with user and item features.
    
    Handles simple feature vectors for users and items, suitable for
    the basic TwoTowerModel architecture. Features are expected to be
    pre-processed into dense numerical vectors.
    """
    
    def __init__(self, user_features: torch.Tensor, item_features: torch.Tensor,
                 ratings: torch.Tensor, user_ids: Optional[torch.Tensor] = None,
                 item_ids: Optional[torch.Tensor] = None):
        """
        Args:
            user_features: Tensor of user features (batch_size, user_feature_dim)
            item_features: Tensor of item features (batch_size, item_feature_dim)
            ratings: Tensor of ratings/labels (batch_size,)
            user_ids: Optional user IDs for collaborative models
            item_ids: Optional item IDs for collaborative models
        """
        self.user_features = user_features
        self.item_features = item_features
        self.ratings = ratings
        self.user_ids = user_ids
        self.item_ids = item_ids
        
        # Ensure all feature tensors have consistent sample counts
        assert len(user_features) == len(item_features) == len(ratings), "Feature tensors must have same length"
    
    def __len__(self):
        return len(self.ratings)
    
    def __getitem__(self, idx):
        item = {
            'user_features': self.user_features[idx],
            'item_features': self.item_features[idx],
            'rating': self.ratings[idx]
        }
        
        if self.user_ids is not None:
            item['user_id'] = self.user_ids[idx]
        if self.item_ids is not None:
            item['item_id'] = self.item_ids[idx]
        
        return item


class EnhancedTwoTowerDataset(Dataset):
    """
    Advanced PyTorch Dataset for Enhanced Two-Tower models with mixed feature types.
    
    Supports both categorical and numerical features separately, enabling
    the model to apply different processing strategies (embeddings vs. linear layers).
    This design is more flexible and typically leads to better performance.
    """
    
    def __init__(self, user_categorical: Dict[str, torch.Tensor], 
                 user_numerical: torch.Tensor, item_categorical: Dict[str, torch.Tensor],
                 item_numerical: torch.Tensor, ratings: torch.Tensor):
        """
        Args:
            user_categorical: Dict of categorical user features
            user_numerical: Numerical user features
            item_categorical: Dict of categorical item features
            item_numerical: Numerical item features
            ratings: Target ratings/labels
        """
        self.user_categorical = user_categorical
        self.user_numerical = user_numerical
        self.item_categorical = item_categorical
        self.item_numerical = item_numerical
        self.ratings = ratings
        
        # Validate that all feature tensors have consistent sample counts
        # This is critical for proper batch formation during training
        first_user_cat = list(user_categorical.values())[0] if user_categorical else None
        first_item_cat = list(item_categorical.values())[0] if item_categorical else None
        
        lengths = [len(ratings)]  # Base length from target variable
        if first_user_cat is not None:
            lengths.append(len(first_user_cat))
        if user_numerical is not None:
            lengths.append(len(user_numerical))
        if first_item_cat is not None:
            lengths.append(len(first_item_cat))
        if item_numerical is not None:
            lengths.append(len(item_numerical))
        
        assert all(l == lengths[0] for l in lengths), "All feature tensors must have identical length"
    
    def __len__(self):
        return len(self.ratings)
    
    def __getitem__(self, idx):
        item = {
            'user_categorical': {k: v[idx] for k, v in self.user_categorical.items()},
            'user_numerical': self.user_numerical[idx] if self.user_numerical is not None else torch.tensor([]),
            'item_categorical': {k: v[idx] for k, v in self.item_categorical.items()},
            'item_numerical': self.item_numerical[idx] if self.item_numerical is not None else torch.tensor([]),
            'rating': self.ratings[idx]
        }
        return item


class TwoTowerDataLoader:
    """
    Comprehensive data preprocessing pipeline for Two-Tower recommendation models.
    
    Handles the complete data flow from raw CSV files to PyTorch DataLoaders:
    - Feature engineering from rating patterns and metadata
    - User preference profiling based on genre interactions
    - Item popularity and content features
    - Categorical/numerical feature separation and encoding
    - Negative sampling for implicit feedback scenarios
    - Train/validation/test splitting with proper data leakage prevention
    """
    
    def __init__(self, ratings_path: str, movies_path: Optional[str] = None,
                 min_interactions: int = 20):
        """
        Args:
            ratings_path: Path to ratings CSV file
            movies_path: Path to movies CSV file with metadata
            min_interactions: Minimum interactions per user/item
        """
        self.ratings_path = ratings_path
        self.movies_path = movies_path
        self.min_interactions = min_interactions
        
        self.logger = logging.getLogger(__name__)
        
        # Initialize preprocessing components
        self.user_encoder = LabelEncoder()      # Maps user IDs to contiguous indices
        self.item_encoder = LabelEncoder()      # Maps item IDs to contiguous indices  
        self.genre_encoders = {}                # Label encoders for categorical features
        self.scalers = {}                       # Standard scalers for numerical features
        
        # Load and process data
        self._load_data()
        self._engineer_features()
        self._fit_encoders()
    
    def _load_data(self):
        """Load ratings and movies data"""
        self.logger.info(f"Loading ratings from {self.ratings_path}")
        self.ratings_df = pd.read_csv(self.ratings_path)
        
        if self.movies_path:
            self.logger.info(f"Loading movies from {self.movies_path}")
            self.movies_df = pd.read_csv(self.movies_path)
        else:
            self.movies_df = None
        
        # Filter by minimum interactions
        self._filter_data()
    
    def _filter_data(self):
        """Filter sparse users and items to improve data quality
        
        Removes users and items with too few interactions, which:
        - Reduces noise from unreliable preference signals
        - Improves training stability and convergence
        - Focuses the model on users/items with sufficient data
        """
        original_size = len(self.ratings_df)
        
        # Filter out sparse users (cold start problem)
        user_counts = self.ratings_df['userId'].value_counts()
        valid_users = user_counts[user_counts >= self.min_interactions].index
        self.ratings_df = self.ratings_df[self.ratings_df['userId'].isin(valid_users)]
        
        # Filter out sparse items (long tail problem)
        item_counts = self.ratings_df['movieId'].value_counts()
        valid_items = item_counts[item_counts >= self.min_interactions].index
        self.ratings_df = self.ratings_df[self.ratings_df['movieId'].isin(valid_items)]
        
        filtered_size = len(self.ratings_df)
        self.logger.info(f"Filtered data: {original_size} -> {filtered_size} "
                        f"({filtered_size/original_size:.2%} retained)")
    
    def _engineer_features(self):
        """Engineer user and item features"""
        self.logger.info("Engineering features...")
        
        # User features
        self._create_user_features()
        
        # Item features
        self._create_item_features()
    
    def _create_user_features(self):
        """Engineer user features from historical rating patterns
        
        Creates behavioral features that capture user preferences and engagement:
        - Rating statistics (mean, std, count)
        - Temporal activity patterns
        - Engagement metrics (frequency, consistency)
        - Preference diversity indicators
        """
        # Aggregate user rating statistics to capture preference patterns
        user_stats = self.ratings_df.groupby('userId').agg({
            'rating': ['mean', 'std', 'count'],      # Rating behavior
            'timestamp': ['min', 'max']              # Temporal activity
        }).round(4)
        
        # Flatten column names
        user_stats.columns = ['_'.join(col) for col in user_stats.columns]
        user_stats = user_stats.reset_index()
        
        # Handle missing values (users with single rating have no std)
        user_stats['rating_std'] = user_stats['rating_std'].fillna(0)
        
        # Engineer behavioral features from basic statistics
        user_stats['rating_range'] = user_stats['rating_std'] > 1.0    # Binary: diverse vs. consistent raters
        user_stats['activity_days'] = (user_stats['timestamp_max'] - user_stats['timestamp_min']) / (24 * 3600)  # Days active
        user_stats['rating_frequency'] = user_stats['rating_count'] / (user_stats['activity_days'] + 1)          # Ratings per day
        
        # Create genre preferences if movies data available
        if self.movies_df is not None:
            self._create_user_genre_preferences(user_stats)
        
        self.user_features_df = user_stats
    
    def _create_user_genre_preferences(self, user_stats: pd.DataFrame):
        """Engineer user genre preference features from rating history
        
        Analyzes user's rating patterns across different genres to create
        preference profiles. This captures content-based user preferences
        that can improve recommendations, especially for cold-start scenarios.
        """
        # Merge ratings with movie genres
        ratings_with_genres = self.ratings_df.merge(
            self.movies_df[['movieId', 'genres']], on='movieId', how='left'
        )
        
        # Process genres
        all_genres = set()
        for genres_str in self.movies_df['genres'].dropna():
            genres = genres_str.split('|')
            all_genres.update(genres)
        
        all_genres = sorted(list(all_genres))
        
        # Calculate user preferences for each genre based on rating patterns
        # Higher ratings = stronger preference for that genre
        genre_preferences = defaultdict(lambda: defaultdict(float))
        
        for _, row in ratings_with_genres.iterrows():
            user_id = row['userId']
            rating = row['rating']      # User's rating for this movie
            genres_str = row['genres']  # Pipe-separated genre string
            
            if pd.notna(genres_str):
                genres = genres_str.split('|')  # Split 'Action|Adventure|Sci-Fi'
                for genre in genres:
                    if genre in all_genres:
                        # Accumulate rating for each genre
                        genre_preferences[user_id][genre] += rating
        
        # Normalize by number of ratings per genre to get average preference
        # This prevents bias toward genres with more rated movies
        for user_id in genre_preferences:
            user_genre_counts = defaultdict(int)
            # Count how many movies of each genre this user has rated
            for _, row in ratings_with_genres[ratings_with_genres['userId'] == user_id].iterrows():
                if pd.notna(row['genres']):
                    for genre in row['genres'].split('|'):
                        if genre in all_genres:
                            user_genre_counts[genre] += 1
            
            # Convert total rating to average rating per genre
            for genre in genre_preferences[user_id]:
                if user_genre_counts[genre] > 0:
                    genre_preferences[user_id][genre] /= user_genre_counts[genre]
        
        # Add genre preferences to user features
        for genre in all_genres:
            genre_col = f'genre_pref_{genre.lower().replace("-", "_")}'
            user_stats[genre_col] = user_stats['userId'].map(
                lambda x: genre_preferences[x].get(genre, 0)
            )
    
    def _create_item_features(self):
        """Engineer item features from rating statistics and metadata
        
        Creates comprehensive item profiles combining:
        - Popularity metrics from user ratings
        - Content features from metadata (year, genres)
        - Quality indicators (average rating, consistency)
        - Temporal patterns (when first/last rated)
        """
        # Basic item popularity and quality statistics from user ratings
        item_stats = self.ratings_df.groupby('movieId').agg({
            'rating': ['mean', 'std', 'count'],   # Quality and popularity metrics
            'timestamp': ['min', 'max']           # When first/last rated
        }).round(4)
        
        item_stats.columns = ['_'.join(col) for col in item_stats.columns]
        item_stats = item_stats.reset_index()
        item_stats['rating_std'] = item_stats['rating_std'].fillna(0)
        
        # Add movie metadata if available
        if self.movies_df is not None:
            # Merge with movie metadata
            item_stats = item_stats.merge(
                self.movies_df[['movieId', 'title', 'genres']], 
                on='movieId', how='left'
            )
            
            # Extract release year from movie titles (MovieLens format: "Title (Year)")
            item_stats['year'] = item_stats['title'].str.extract(r'\((\d{4})\)')  # Regex to extract 4-digit year
            item_stats['year'] = pd.to_numeric(item_stats['year'], errors='coerce')  # Convert to numeric
            item_stats['year'] = item_stats['year'].fillna(item_stats['year'].median())  # Fill missing with median
            
            # Process genres
            self._process_item_genres(item_stats)
        
        self.item_features_df = item_stats
    
    def _process_item_genres(self, item_stats: pd.DataFrame):
        """Process item genre information into model features
        
        Converts pipe-separated genre strings into binary feature vectors
        and additional genre-based statistics. This enables the model to
        learn content-based item similarities and preferences.
        """
        # Get all unique genres
        all_genres = set()
        for genres_str in item_stats['genres'].dropna():
            genres = genres_str.split('|')
            all_genres.update(genres)
        
        all_genres = sorted(list(all_genres))
        
        # Create binary genre features (one-hot encoding)
        # Each genre becomes a separate binary feature column
        for genre in all_genres:
            genre_col = f'genre_{genre.lower().replace("-", "_")}'
            item_stats[genre_col] = item_stats['genres'].apply(
                lambda x: 1 if pd.notna(x) and genre in x.split('|') else 0
            )
        
        # Count total genres per movie (genre diversity feature)
        # Movies with more genres might appeal to broader audiences
        item_stats['num_genres'] = item_stats['genres'].apply(
            lambda x: len(x.split('|')) if pd.notna(x) else 0
        )
    
    def _fit_encoders(self):
        """Fit all preprocessing components on the training data
        
        Creates mappings and scaling parameters that will be applied consistently
        across train/validation/test splits to prevent data leakage.
        """
        # Encode user and item IDs to contiguous indices for embedding layers
        self.user_encoder.fit(self.ratings_df['userId'].unique())
        self.item_encoder.fit(self.ratings_df['movieId'].unique())
        
        # Identify numerical and categorical columns
        self.user_numerical_cols = []
        self.user_categorical_cols = []
        self.item_numerical_cols = []
        self.item_categorical_cols = []
        
        # Classify user features as numerical or categorical based on data type and cardinality
        for col in self.user_features_df.columns:
            if col == 'userId':
                continue  # Skip ID column
            elif self.user_features_df[col].dtype in ['int64', 'float64']:
                # Numeric columns with low cardinality are treated as categorical
                if self.user_features_df[col].nunique() <= 10:
                    self.user_categorical_cols.append(col)
                else:
                    self.user_numerical_cols.append(col)
            else:
                # Non-numeric columns are categorical
                self.user_categorical_cols.append(col)
        
        # Item features
        for col in self.item_features_df.columns:
            if col in ['movieId', 'title', 'genres']:
                continue
            elif self.item_features_df[col].dtype in ['int64', 'float64']:
                if self.item_features_df[col].nunique() <= 10:  # Treat as categorical
                    self.item_categorical_cols.append(col)
                else:
                    self.item_numerical_cols.append(col)
            else:
                self.item_categorical_cols.append(col)
        
        # Fit scalers for numerical features (standardization to mean=0, std=1)
        # This ensures all numerical features have similar scales for stable training
        if self.user_numerical_cols:
            self.scalers['user_numerical'] = StandardScaler()
            self.scalers['user_numerical'].fit(self.user_features_df[self.user_numerical_cols])
        
        if self.item_numerical_cols:
            self.scalers['item_numerical'] = StandardScaler()
            self.scalers['item_numerical'].fit(self.item_features_df[self.item_numerical_cols])
        
        # Fit label encoders for categorical features
        # Each categorical feature gets mapped to contiguous integers for embedding layers
        for col in self.user_categorical_cols:
            encoder = LabelEncoder()
            encoder.fit(self.user_features_df[col].astype(str))  # Convert to string to handle mixed types
            self.genre_encoders[f'user_{col}'] = encoder
        
        for col in self.item_categorical_cols:
            encoder = LabelEncoder()
            encoder.fit(self.item_features_df[col].astype(str))
            self.genre_encoders[f'item_{col}'] = encoder
        
        self.logger.info(f"User features: {len(self.user_numerical_cols)} numerical, "
                        f"{len(self.user_categorical_cols)} categorical")
        self.logger.info(f"Item features: {len(self.item_numerical_cols)} numerical, "
                        f"{len(self.item_categorical_cols)} categorical")
    
    def create_training_data(self, test_size: float = 0.2, val_size: float = 0.1,
                           negative_sampling: bool = True, neg_ratio: float = 2.0) -> Tuple:
        """
        Create training, validation, and test datasets.
        
        Args:
            test_size: Proportion for test set
            val_size: Proportion of training data for validation
            negative_sampling: Whether to add negative samples
            neg_ratio: Ratio of negative to positive samples
        
        Returns:
            Tuple of processed data splits
        """
        # Merge ratings with user and item features
        data = self.ratings_df.copy()
        
        # Add encoded IDs
        data['user_encoded'] = self.user_encoder.transform(data['userId'])
        data['item_encoded'] = self.item_encoder.transform(data['movieId'])
        
        # Merge features
        data = data.merge(self.user_features_df, on='userId', how='left')
        data = data.merge(self.item_features_df, on='movieId', how='left')
        
        # Add negative samples if requested
        if negative_sampling:
            data = self._add_negative_samples(data, neg_ratio)
        
        # Split data
        train_val_data, test_data = train_test_split(
            data, test_size=test_size, random_state=42, stratify=data['userId']
        )
        
        train_data, val_data = train_test_split(
            train_val_data, test_size=val_size, random_state=42, stratify=train_val_data['userId']
        )
        
        # Process features for each split
        train_features = self._process_features(train_data)
        val_features = self._process_features(val_data)
        test_features = self._process_features(test_data)
        
        return train_features, val_features, test_features
    
    def _add_negative_samples(self, data: pd.DataFrame, neg_ratio: float) -> pd.DataFrame:
        """Add negative samples for implicit feedback training
        
        For recommendation systems, we often need negative examples to train
        the model to distinguish between relevant and irrelevant items.
        This function generates random user-item pairs that don't exist in
        the positive interaction data.
        """
        # Identify existing user-item interactions (positive samples)
        positive_pairs = set(zip(data['userId'], data['movieId']))
        
        # Get vocabulary of all users and items for negative sampling
        all_users = data['userId'].unique()
        all_items = data['movieId'].unique()
        
        # Generate negative samples by random sampling
        negative_samples = []
        num_negatives = int(len(data) * neg_ratio)  # Total negative samples to generate
        
        # Randomly sample user-item pairs that don't exist in positive data
        while len(negative_samples) < num_negatives:
            user = np.random.choice(all_users)
            item = np.random.choice(all_items)
            
            # Only use pairs that don't exist in positive interactions
            if (user, item) not in positive_pairs:
                # Retrieve pre-computed features for this user and item
                user_features = self.user_features_df[self.user_features_df['userId'] == user].iloc[0]
                item_features = self.item_features_df[self.item_features_df['movieId'] == item].iloc[0]
                
                # Create negative sample with rating=0 (negative label)
                negative_sample = {
                    'userId': user,
                    'movieId': item,
                    'rating': 0.0,  # Label as negative interaction
                    'timestamp': 0   # Dummy timestamp
                }
                
                # Add user features
                for col in self.user_features_df.columns:
                    if col != 'userId':
                        negative_sample[col] = user_features[col]
                
                # Add item features
                for col in self.item_features_df.columns:
                    if col != 'movieId':
                        negative_sample[col] = item_features[col]
                
                negative_samples.append(negative_sample)
        
        # Combine positive and negative samples
        negative_df = pd.DataFrame(negative_samples)
        combined_data = pd.concat([data, negative_df], ignore_index=True)
        
        return combined_data
    
    def _process_features(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Process features for model input"""
        # User features
        user_numerical = None
        if self.user_numerical_cols:
            user_numerical = self.scalers['user_numerical'].transform(
                data[self.user_numerical_cols]
            )
            user_numerical = torch.FloatTensor(user_numerical)
        
        user_categorical = {}
        for col in self.user_categorical_cols:
            encoded = self.genre_encoders[f'user_{col}'].transform(data[col].astype(str))
            user_categorical[col] = torch.LongTensor(encoded)
        
        # Item features
        item_numerical = None
        if self.item_numerical_cols:
            item_numerical = self.scalers['item_numerical'].transform(
                data[self.item_numerical_cols]
            )
            item_numerical = torch.FloatTensor(item_numerical)
        
        item_categorical = {}
        for col in self.item_categorical_cols:
            encoded = self.genre_encoders[f'item_{col}'].transform(data[col].astype(str))
            item_categorical[col] = torch.LongTensor(encoded)
        
        # Ratings and IDs
        ratings = torch.FloatTensor(data['rating'].values)
        user_ids = torch.LongTensor(data['user_encoded'].values)
        item_ids = torch.LongTensor(data['item_encoded'].values)
        
        return {
            'user_numerical': user_numerical,
            'user_categorical': user_categorical,
            'item_numerical': item_numerical,
            'item_categorical': item_categorical,
            'ratings': ratings,
            'user_ids': user_ids,
            'item_ids': item_ids
        }
    
    def create_data_loaders(self, batch_size: int = 1024, num_workers: int = 4,
                          model_type: str = 'enhanced') -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Create PyTorch data loaders.
        
        Args:
            batch_size: Batch size
            num_workers: Number of worker processes
            model_type: Type of model ('simple', 'enhanced', 'collaborative')
        
        Returns:
            Tuple of (train_loader, val_loader, test_loader)
        """
        train_features, val_features, test_features = self.create_training_data()
        
        if model_type == 'enhanced':
            # Enhanced Two-Tower dataset
            train_dataset = EnhancedTwoTowerDataset(
                train_features['user_categorical'],
                train_features['user_numerical'],
                train_features['item_categorical'],
                train_features['item_numerical'],
                train_features['ratings']
            )
            
            val_dataset = EnhancedTwoTowerDataset(
                val_features['user_categorical'],
                val_features['user_numerical'],
                val_features['item_categorical'],
                val_features['item_numerical'],
                val_features['ratings']
            )
            
            test_dataset = EnhancedTwoTowerDataset(
                test_features['user_categorical'],
                test_features['user_numerical'],
                test_features['item_categorical'],
                test_features['item_numerical'],
                test_features['ratings']
            )
        
        else:
            # Simple Two-Tower dataset (flatten all features)
            def flatten_features(features):
                all_features = []
                
                # Add numerical features
                if features['user_numerical'] is not None:
                    all_features.append(features['user_numerical'])
                if features['item_numerical'] is not None:
                    all_features.append(features['item_numerical'])
                
                # Add categorical features (as floats)
                for cat_features in features['user_categorical'].values():
                    all_features.append(cat_features.float().unsqueeze(1))
                for cat_features in features['item_categorical'].values():
                    all_features.append(cat_features.float().unsqueeze(1))
                
                return torch.cat(all_features, dim=1) if all_features else torch.zeros(len(features['ratings']), 1)
            
            # For simple model, we need to separate user and item features
            # This is a simplified approach - in practice, you'd design the feature split more carefully
            user_feature_size = (len(self.user_numerical_cols) + len(self.user_categorical_cols))
            
            train_all = flatten_features(train_features)
            val_all = flatten_features(val_features)
            test_all = flatten_features(test_features)
            
            train_dataset = TwoTowerDataset(
                train_all[:, :user_feature_size],
                train_all[:, user_feature_size:],
                train_features['ratings'],
                train_features['user_ids'],
                train_features['item_ids']
            )
            
            val_dataset = TwoTowerDataset(
                val_all[:, :user_feature_size],
                val_all[:, user_feature_size:],
                val_features['ratings'],
                val_features['user_ids'],
                val_features['item_ids']
            )
            
            test_dataset = TwoTowerDataset(
                test_all[:, :user_feature_size],
                test_all[:, user_feature_size:],
                test_features['ratings'],
                test_features['user_ids'],
                test_features['item_ids']
            )
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=True
        )
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True
        )
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True
        )
        
        return train_loader, val_loader, test_loader
    
    def get_model_config(self) -> Dict[str, Any]:
        """Get configuration for model initialization"""
        config = {
            'num_users': len(self.user_encoder.classes_),
            'num_items': len(self.item_encoder.classes_),
            'user_numerical_dim': len(self.user_numerical_cols),
            'item_numerical_dim': len(self.item_numerical_cols),
            'user_categorical_dims': {},
            'item_categorical_dims': {}
        }
        
        # Get categorical feature dimensions
        for col in self.user_categorical_cols:
            config['user_categorical_dims'][col] = len(self.genre_encoders[f'user_{col}'].classes_)
        
        for col in self.item_categorical_cols:
            config['item_categorical_dims'][col] = len(self.genre_encoders[f'item_{col}'].classes_)
        
        return config
    
    def save_preprocessors(self, save_path: str):
        """Save all preprocessors for inference"""
        preprocessors = {
            'user_encoder': self.user_encoder,
            'item_encoder': self.item_encoder,
            'genre_encoders': self.genre_encoders,
            'scalers': self.scalers,
            'user_numerical_cols': self.user_numerical_cols,
            'user_categorical_cols': self.user_categorical_cols,
            'item_numerical_cols': self.item_numerical_cols,
            'item_categorical_cols': self.item_categorical_cols,
            'user_features_df': self.user_features_df,
            'item_features_df': self.item_features_df
        }
        
        with open(save_path, 'wb') as f:
            pickle.dump(preprocessors, f)
        
        self.logger.info(f"Saved preprocessors to {save_path}")
    
    @classmethod
    def load_preprocessors(cls, load_path: str) -> Dict[str, Any]:
        """Load preprocessors for inference"""
        with open(load_path, 'rb') as f:
            preprocessors = pickle.load(f)
        
        return preprocessors