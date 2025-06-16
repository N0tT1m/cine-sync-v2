#!/usr/bin/env python3
"""
Simplified movie recommendation training script
Removes complexity while maintaining core functionality
"""
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pickle
import logging
import argparse
from datetime import datetime

# Import simplified utilities
from utils.id_mapping import create_id_mappings
from utils.data_processing import load_csv_data, prepare_genre_features, create_train_val_split, find_data_files
from config import load_config
from models import HybridRecommenderModel


def setup_logging():
    """Simple logging setup with file and console output
    
    Creates a logger that writes to both a timestamped log file
    and the console for real-time monitoring during training.
    
    Returns:
        logging.Logger: Configured logger instance
    """
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f"training_{datetime.now().strftime('%Y%m%d%H%M%S')}.log"),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger("training")


class SimpleDataset(Dataset):
    """Simplified PyTorch dataset for movie recommendations
    
    Wraps user-item-rating data for efficient batch loading during training.
    Includes genre features for content-based recommendations.
    """
    def __init__(self, user_indices, movie_indices, genre_features, ratings):
        """Initialize dataset with preprocessed tensors
        
        Args:
            user_indices: Encoded user IDs as tensor
            movie_indices: Encoded movie IDs as tensor
            genre_features: Binary genre features as tensor
            ratings: Target ratings as tensor
        """
        self.user_indices = user_indices
        self.movie_indices = movie_indices
        self.genre_features = genre_features
        self.ratings = ratings
    
    def __len__(self):
        return len(self.ratings)
    
    def __getitem__(self, idx):
        return (
            self.user_indices[idx],
            self.movie_indices[idx], 
            self.genre_features[idx],
            self.ratings[idx]
        )


def load_and_process_data(data_files):
    """Load and process all data in one streamlined function
    
    Handles the complete data pipeline from loading raw CSV files
    to creating training/validation splits with proper preprocessing.
    
    Args:
        data_files (dict): Dictionary containing paths to data files
        
    Returns:
        tuple: (train_data, val_data, movies_df, genres, mappings, scaler)
    """
    logger = logging.getLogger("training")
    logger.info("Loading and processing data")
    
    # Load movie metadata (titles, genres, etc.)
    movies_df = load_csv_data(data_files['ml_movies'])
    if movies_df is None:
        raise ValueError("Could not load movie data")
    
    # Standardize column names for compatibility across datasets
    if 'movieId' in movies_df.columns:
        movies_df = movies_df.rename(columns={'movieId': 'media_id'})
    
    # Load user-item ratings (the core collaborative filtering data)
    ratings_df = load_csv_data(data_files['ml_ratings'])
    if ratings_df is None:
        raise ValueError("Could not load ratings data")
    
    logger.info(f"Loaded {len(movies_df)} movies, {len(ratings_df)} ratings")
    
    # Extract and encode genre features for content-based recommendations
    movies_df, genres = prepare_genre_features(movies_df)
    
    # Create consecutive integer mappings for embedding layers
    ratings_df, movies_df, mappings = create_id_mappings(ratings_df, movies_df)
    
    # Combine ratings with movie features for hybrid recommendations
    data = ratings_df.merge(movies_df, left_on='movieId', right_on='media_id', how='inner')
    logger.info(f"Merged data: {len(data)} records")
    
    # Create splits
    train_data, val_data, scaler = create_train_val_split(data)
    
    return train_data, val_data, movies_df, genres, mappings, scaler


def create_data_loaders(train_data, val_data, genres, batch_size=128):
    """Create PyTorch data loaders for efficient batch processing
    
    Converts pandas DataFrames to PyTorch tensors and creates
    DataLoaders with appropriate batch sizes and worker processes.
    
    Args:
        train_data (pd.DataFrame): Training dataset
        val_data (pd.DataFrame): Validation dataset  
        genres (list): List of genre column names
        batch_size (int): Number of samples per batch
        
    Returns:
        tuple: (train_loader, val_loader) for model training
    """
    def make_dataset(data_df):
        """Convert DataFrame to PyTorch dataset with proper tensor types"""
        # Convert encoded user/movie IDs to LongTensor for embedding lookups
        user_indices = torch.tensor(data_df['user_idx'].values, dtype=torch.long)
        movie_indices = torch.tensor(data_df['content_idx'].values, dtype=torch.long)
        # Convert binary genre features to FloatTensor for neural network processing
        genre_features = torch.tensor(data_df[genres].values, dtype=torch.float32)
        # Convert scaled ratings to FloatTensor targets
        ratings = torch.tensor(data_df['rating_scaled'].values, dtype=torch.float32).unsqueeze(1)
        return SimpleDataset(user_indices, movie_indices, genre_features, ratings)
    
    train_dataset = make_dataset(train_data)
    val_dataset = make_dataset(val_data)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    return train_loader, val_loader


def train_model(model, train_loader, val_loader, device, epochs=20):
    """Simplified training loop with validation and early stopping
    
    Implements a basic training loop with:
    - MSE loss for rating prediction
    - Adam optimizer with learning rate scheduling
    - Validation monitoring and best model checkpointing
    
    Args:
        model: The hybrid recommendation model to train
        train_loader: DataLoader for training batches
        val_loader: DataLoader for validation batches
        device: torch.device for GPU/CPU computation
        epochs: Maximum number of training epochs
        
    Returns:
        torch.nn.Module: The trained model
    """
    logger = logging.getLogger("training")
    
    # Use MSE loss for rating prediction (regression task)
    criterion = nn.MSELoss()
    # Adam optimizer with modest learning rate for stable training
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    # Step learning rate decay every 7 epochs
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        # Training phase - model learns from training data
        model.train()  # Enable dropout and batch norm training mode
        train_loss = 0.0
        for user_ids, movie_ids, genre_feats, targets in train_loader:
            # Move all tensors to appropriate device (GPU/CPU)
            user_ids, movie_ids, genre_feats, targets = (
                user_ids.to(device), movie_ids.to(device), 
                genre_feats.to(device), targets.to(device)
            )
            
            # Standard PyTorch training step
            optimizer.zero_grad()  # Clear gradients from previous step
            outputs = model(user_ids, movie_ids, genre_feats)  # Forward pass
            loss = criterion(outputs, targets)  # Compute loss
            loss.backward()  # Backpropagation
            optimizer.step()  # Update parameters
            
            train_loss += loss.item()
        
        # Validation phase - evaluate model performance without training
        model.eval()  # Disable dropout and set batch norm to eval mode
        val_loss = 0.0
        with torch.no_grad():  # Disable gradient computation for efficiency
            for user_ids, movie_ids, genre_feats, targets in val_loader:
                # Move validation data to device
                user_ids, movie_ids, genre_feats, targets = (
                    user_ids.to(device), movie_ids.to(device),
                    genre_feats.to(device), targets.to(device)
                )
                outputs = model(user_ids, movie_ids, genre_feats)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        logger.info(f"Epoch {epoch+1}/{epochs} | Train: {avg_train_loss:.4f} | Val: {avg_val_loss:.4f}")
        
        # Save best model checkpoint based on validation performance
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            # Save complete training state for potential resumption
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_val_loss,
                'epoch': epoch
            }, 'models/best_model.pt')
        
        scheduler.step()
    
    return model


def save_artifacts(mappings, scaler, movies_df, genres):
    """Save all training artifacts for model deployment
    
    Saves all necessary components for making predictions:
    - ID mappings for encoding new users/movies
    - Rating scaler for preprocessing
    - Movie metadata for content-based features
    - Model configuration and metadata
    
    Args:
        mappings (dict): User and item ID mappings
        scaler: Fitted rating scaler
        movies_df (pd.DataFrame): Movie metadata
        genres (list): List of genre features
    """
    os.makedirs('models', exist_ok=True)
    
    # Save mappings and scaler
    with open('models/id_mappings.pkl', 'wb') as f:
        pickle.dump(mappings, f)
    
    with open('models/rating_scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    # Create movie lookup
    movie_lookup = movies_df[['media_id', 'title', 'genres']].set_index('media_id').to_dict(orient='index')
    with open('models/movie_lookup.pkl', 'wb') as f:
        pickle.dump(movie_lookup, f)
    
    # Save metadata
    metadata = {
        'num_users': mappings['num_users'],
        'num_content': mappings['num_content'],
        'genres': genres,
        'timestamp': datetime.now().isoformat()
    }
    with open('models/model_metadata.pkl', 'wb') as f:
        pickle.dump(metadata, f)
    
    # Save movies data
    movies_df.to_csv('models/movies_data.csv', index=False)


def main():
    """Main training function - simplified workflow
    
    Orchestrates the complete training pipeline:
    1. Argument parsing and logging setup
    2. Data loading and preprocessing
    3. Model creation and training
    4. Artifact saving for deployment
    
    Returns:
        int: Exit code (0 for success, 1 for failure)
    """
    logger = setup_logging()
    
    parser = argparse.ArgumentParser(description='Train movie recommendation model (simplified)')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=128, help='Batch size')
    args = parser.parse_args()
    
    try:
        # Setup compute device for training
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {device}")
        
        # Locate data files in standard directory structure
        data_files = find_data_files()
        if not data_files:
            raise ValueError("No data files found")
        
        # Process data
        train_data, val_data, movies_df, genres, mappings, scaler = load_and_process_data(data_files)
        
        # Create data loaders
        train_loader, val_loader = create_data_loaders(train_data, val_data, genres, args.batch_size)
        
        # Create hybrid recommendation model with collaborative and content-based components
        model = HybridRecommenderModel(
            num_users=mappings['num_users'],     # Number of unique users for embeddings
            num_items=mappings['num_content'],   # Number of unique movies for embeddings
            num_genres=len(genres),              # Number of genre features
            embedding_dim=64                     # Dimensionality of user/item embeddings
        ).to(device)
        
        logger.info(f"Model: {mappings['num_users']} users, {mappings['num_content']} items, {len(genres)} genres")
        
        # Execute training loop with validation monitoring
        model = train_model(model, train_loader, val_loader, device, args.epochs)
        
        # Save all artifacts needed for model deployment and inference
        save_artifacts(mappings, scaler, movies_df, genres)
        
        logger.info("Training completed successfully!")
        return 0
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())