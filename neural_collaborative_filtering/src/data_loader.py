# Data Loading and Preprocessing for Neural Collaborative Filtering
# Handles MovieLens and similar datasets with proper encoding and negative sampling
# Optimized for PyTorch training with efficient data loading

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from typing import Tuple, Optional, Dict, Any
import logging


class MovieLensDataset(Dataset):
    """
    PyTorch Dataset for MovieLens ratings data optimized for Neural Collaborative Filtering.
    
    Efficiently handles user-item rating data with proper ID encoding and optional
    genre information. Designed for batch training with PyTorch DataLoader.
    """
    
    def __init__(self, ratings_df: pd.DataFrame, user_encoder: LabelEncoder, 
                 item_encoder: LabelEncoder, genre_encoder: Optional[LabelEncoder] = None,
                 movies_df: Optional[pd.DataFrame] = None):
        """
        Initialize MovieLens dataset for PyTorch training
        
        Args:
            ratings_df: DataFrame with columns ['userId', 'movieId', 'rating']
            user_encoder: Fitted LabelEncoder for user IDs (must be pre-fitted)
            item_encoder: Fitted LabelEncoder for item IDs (must be pre-fitted)
            genre_encoder: Optional LabelEncoder for genres
            movies_df: Optional DataFrame with movie metadata for genre features
        """
        self.ratings = ratings_df.copy()
        self.user_encoder = user_encoder
        self.item_encoder = item_encoder
        self.genre_encoder = genre_encoder
        self.movies_df = movies_df
        
        # Encode user and item IDs to contiguous indices for embedding layers
        # This maps arbitrary IDs (e.g., 1, 3, 17, 42...) to (0, 1, 2, 3...)
        self.ratings['user_id_encoded'] = self.user_encoder.transform(self.ratings['userId'])
        self.ratings['item_id_encoded'] = self.item_encoder.transform(self.ratings['movieId'])
        
        # Prepare genre information if available for content-based features
        if movies_df is not None and genre_encoder is not None:
            self._prepare_genres()
        
        # Convert to tensors for faster access during training
        # LongTensor for indices, FloatTensor for continuous ratings
        self.user_ids = torch.LongTensor(self.ratings['user_id_encoded'].values)
        self.item_ids = torch.LongTensor(self.ratings['item_id_encoded'].values)
        self.ratings_values = torch.FloatTensor(self.ratings['rating'].values)
        
        if hasattr(self, 'genre_ids'):
            self.genre_ids = torch.LongTensor(self.genre_ids)
    
    def _prepare_genres(self):
        """Prepare genre encoding for items
        
        Processes genre information by merging with movie metadata and
        encoding genres for use as additional features in the model.
        """
        # Merge ratings with movie metadata to get genre information
        merged = self.ratings.merge(self.movies_df[['movieId', 'genres']], on='movieId', how='left')
        
        # Handle multiple genres per movie (MovieLens format: 'Action|Adventure|Sci-Fi')
        genre_lists = merged['genres'].fillna('(no genres listed)').str.split('|')
        
        # Simplified approach: take the first genre for each movie
        # Note: More sophisticated implementations could handle multiple genres
        # using multi-hot encoding or attention mechanisms
        first_genres = [genres[0] if genres else '(no genres listed)' for genres in genre_lists]
        self.genre_ids = self.genre_encoder.transform(first_genres)
    
    def __len__(self):
        return len(self.ratings)
    
    def __getitem__(self, idx):
        """Get a single training sample
        
        Args:
            idx: Index of the sample
            
        Returns:
            Dict containing user_id, item_id, rating, and optionally genre_id
        """
        item = {
            'user_id': self.user_ids[idx],      # Encoded user ID
            'item_id': self.item_ids[idx],      # Encoded item ID
            'rating': self.ratings_values[idx]  # Rating value
        }
        
        # Add genre information if available
        if hasattr(self, 'genre_ids'):
            item['genre_id'] = self.genre_ids[idx]
        
        return item


class NCFDataLoader:
    """
    Data loader class for Neural Collaborative Filtering with preprocessing capabilities.
    
    Handles the complete data pipeline from raw CSV files to PyTorch DataLoaders:
    - Data loading and filtering
    - User/item ID encoding
    - Train/validation/test splitting
    - Genre processing for content features
    """
    
    def __init__(self, ratings_path: str, movies_path: Optional[str] = None, 
                 min_ratings_per_user: int = 10, min_ratings_per_item: int = 10):
        """
        Initialize NCF data loader with filtering parameters
        
        Args:
            ratings_path: Path to ratings CSV file (userId, movieId, rating, timestamp)
            movies_path: Optional path to movies CSV file (movieId, title, genres)
            min_ratings_per_user: Minimum ratings per user to include (filtering sparse users)
            min_ratings_per_item: Minimum ratings per item to include (filtering sparse items)
        """
        self.ratings_path = ratings_path
        self.movies_path = movies_path
        self.min_ratings_per_user = min_ratings_per_user
        self.min_ratings_per_item = min_ratings_per_item
        
        self.logger = logging.getLogger(__name__)
        
        # Initialize label encoders for converting IDs to contiguous indices
        self.user_encoder = LabelEncoder()                          # Maps user IDs to 0, 1, 2, ...
        self.item_encoder = LabelEncoder()                          # Maps item IDs to 0, 1, 2, ...
        self.genre_encoder = LabelEncoder() if movies_path else None # Maps genres to indices
        
        # Execute data pipeline: load → preprocess → encode
        self._load_data()       # Load CSV files
        self._preprocess_data() # Filter and normalize
        self._fit_encoders()    # Fit label encoders
    
    def _load_data(self):
        """Load ratings and movies data from CSV files"""
        self.logger.info(f"Loading ratings from {self.ratings_path}")
        self.ratings_df = pd.read_csv(self.ratings_path)
        
        if self.movies_path:
            self.logger.info(f"Loading movies from {self.movies_path}")
            self.movies_df = pd.read_csv(self.movies_path)
        else:
            self.movies_df = None
    
    def _preprocess_data(self):
        """Preprocess and filter data to remove sparse users/items
        
        Applies minimum rating thresholds to ensure sufficient data density
        for collaborative filtering and normalizes ratings for training stability.
        """
        original_size = len(self.ratings_df)
        
        # Filter out users with too few ratings (cold start problem)
        user_counts = self.ratings_df['userId'].value_counts()
        valid_users = user_counts[user_counts >= self.min_ratings_per_user].index
        self.ratings_df = self.ratings_df[self.ratings_df['userId'].isin(valid_users)]
        
        # Filter out items with too few ratings (long tail problem)
        item_counts = self.ratings_df['movieId'].value_counts()
        valid_items = item_counts[item_counts >= self.min_ratings_per_item].index
        self.ratings_df = self.ratings_df[self.ratings_df['movieId'].isin(valid_items)]
        
        filtered_size = len(self.ratings_df)
        self.logger.info(f"Filtered ratings: {original_size} -> {filtered_size} "
                        f"({filtered_size/original_size:.2%} retained)")
        
        # Normalize ratings to 0-1 scale for training stability
        # Converts 0.5-5.0 scale to 0.0-1.0 scale
        self.ratings_df['rating'] = (self.ratings_df['rating'] - 0.5) / 4.5
    
    def _fit_encoders(self):
        """Fit label encoders on filtered data
        
        Creates mappings from original IDs to contiguous indices required
        by PyTorch embedding layers. Also handles genre encoding if available.
        """
        unique_users = self.ratings_df['userId'].unique()
        unique_items = self.ratings_df['movieId'].unique()
        
        self.user_encoder.fit(unique_users)
        self.item_encoder.fit(unique_items)
        
        # Fit genre encoder if movie metadata is available
        if self.movies_df is not None and self.genre_encoder is not None:
            # Extract all unique genres from pipe-separated genre strings
            all_genres = set()
            for genres_str in self.movies_df['genres'].fillna('(no genres listed)'):
                genres = genres_str.split('|')  # Split 'Action|Adventure|Sci-Fi'
                all_genres.update(genres)
            self.genre_encoder.fit(list(all_genres))
        
        self.logger.info(f"Encoded {len(unique_users)} users and {len(unique_items)} items")
    
    def get_data_loaders(self, test_size: float = 0.2, val_size: float = 0.1, 
                        batch_size: int = 1024, num_workers: int = 4) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Create train, validation, and test data loaders with stratified splitting.
        
        Performs stratified splitting to ensure balanced user representation
        across splits, which is important for collaborative filtering.
        
        Args:
            test_size: Proportion of data for testing (0.2 = 20%)
            val_size: Proportion of training data for validation (0.1 = 10%)
            batch_size: Batch size for data loaders (larger = more efficient)
            num_workers: Number of worker processes for data loading
        
        Returns:
            Tuple of (train_loader, val_loader, test_loader)
        """
        # Stratified split to ensure balanced user representation
        # First split: separate test set
        train_val_df, test_df = train_test_split(
            self.ratings_df, test_size=test_size, random_state=42, stratify=self.ratings_df['userId']
        )
        
        # Second split: separate validation set from training
        train_df, val_df = train_test_split(
            train_val_df, test_size=val_size, random_state=42, stratify=train_val_df['userId']
        )
        
        # Create PyTorch datasets with shared encoders
        train_dataset = MovieLensDataset(
            train_df, self.user_encoder, self.item_encoder, self.genre_encoder, self.movies_df
        )
        val_dataset = MovieLensDataset(
            val_df, self.user_encoder, self.item_encoder, self.genre_encoder, self.movies_df
        )
        test_dataset = MovieLensDataset(
            test_df, self.user_encoder, self.item_encoder, self.genre_encoder, self.movies_df
        )
        
        # Create PyTorch data loaders with appropriate settings
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers  # Shuffle for training
        )
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers   # No shuffle for evaluation
        )
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers  # No shuffle for evaluation
        )
        
        self.logger.info(f"Created data loaders: train={len(train_dataset)}, "
                        f"val={len(val_dataset)}, test={len(test_dataset)}")
        
        return train_loader, val_loader, test_loader
    
    def get_model_config(self) -> Dict[str, Any]:
        """Get configuration for model initialization
        
        Returns dictionary with vocab sizes needed for embedding layers.
        
        Returns:
            Dict with num_users, num_items, and optionally num_genres
        """
        config = {
            'num_users': len(self.user_encoder.classes_),
            'num_items': len(self.item_encoder.classes_),
        }
        
        if self.genre_encoder is not None:
            config['num_genres'] = len(self.genre_encoder.classes_)
        
        return config
    
    def save_encoders(self, save_path: str):
        """Save encoders for inference
        
        Saves fitted label encoders so they can be reused during inference
        to maintain consistent ID mappings.
        
        Args:
            save_path: Path to save the encoders pickle file
        """
        import pickle
        
        encoders = {
            'user_encoder': self.user_encoder,
            'item_encoder': self.item_encoder,
        }
        
        if self.genre_encoder is not None:
            encoders['genre_encoder'] = self.genre_encoder
        
        with open(save_path, 'wb') as f:
            pickle.dump(encoders, f)
        
        self.logger.info(f"Saved encoders to {save_path}")
    
    @classmethod
    def load_encoders(cls, load_path: str) -> Dict[str, LabelEncoder]:
        """Load encoders for inference
        
        Loads previously saved encoders for consistent ID mapping during inference.
        
        Args:
            load_path: Path to the saved encoders pickle file
            
        Returns:
            Dictionary containing the fitted label encoders
        """
        import pickle
        
        with open(load_path, 'rb') as f:
            encoders = pickle.load(f)
        
        return encoders


def create_negative_samples(ratings_df: pd.DataFrame, num_negatives: int = 4) -> pd.DataFrame:
    """
    Create negative samples for implicit feedback training.
    
    For implicit feedback scenarios (like clicks, views), we need to generate
    negative samples (items the user hasn't interacted with) to train the model
    to distinguish between preferred and non-preferred items.
    
    Args:
        ratings_df: DataFrame with positive interactions (userId, movieId, rating)
        num_negatives: Number of negative samples per user (not per interaction)
    
    Returns:
        DataFrame with both positive and negative samples (rating=0 for negatives)
    """
    # Get vocabulary of all users and items in the dataset
    all_users = ratings_df['userId'].unique()
    all_items = ratings_df['movieId'].unique()
    
    # Create set of positive interactions for fast lookup
    positive_interactions = set(zip(ratings_df['userId'], ratings_df['movieId']))
    
    negative_samples = []
    
    # Generate negative samples for each user
    for user_id in all_users:
        # Get items this user has already interacted with
        user_items = set(ratings_df[ratings_df['userId'] == user_id]['movieId'])
        
        # Find items user hasn't interacted with (candidates for negative sampling)
        available_items = set(all_items) - user_items
        
        # Sample negative items without replacement
        if len(available_items) >= num_negatives:
            neg_items = np.random.choice(list(available_items), num_negatives, replace=False)
        else:
            neg_items = list(available_items)  # Use all available if not enough
        
        # Add negative samples with rating=0
        for item_id in neg_items:
            negative_samples.append({
                'userId': user_id,
                'movieId': item_id,
                'rating': 0.0,  # Mark as negative sample
                'timestamp': 0   # Dummy timestamp
            })
    
    # Combine original positive samples with generated negative samples
    negative_df = pd.DataFrame(negative_samples)
    combined_df = pd.concat([ratings_df, negative_df], ignore_index=True)
    
    return combined_df  # Returns DataFrame with both positive and negative samples