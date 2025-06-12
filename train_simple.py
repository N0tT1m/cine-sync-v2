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
    """Simple logging setup"""
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
    """Simplified dataset class"""
    def __init__(self, user_indices, movie_indices, genre_features, ratings):
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
    """Load and process all data in one streamlined function"""
    logger = logging.getLogger("training")
    logger.info("Loading and processing data")
    
    # Load movies
    movies_df = load_csv_data(data_files['ml_movies'])
    if movies_df is None:
        raise ValueError("Could not load movie data")
    
    # Ensure correct column names
    if 'movieId' in movies_df.columns:
        movies_df = movies_df.rename(columns={'movieId': 'media_id'})
    
    # Load ratings
    ratings_df = load_csv_data(data_files['ml_ratings'])
    if ratings_df is None:
        raise ValueError("Could not load ratings data")
    
    logger.info(f"Loaded {len(movies_df)} movies, {len(ratings_df)} ratings")
    
    # Process genres
    movies_df, genres = prepare_genre_features(movies_df)
    
    # Create ID mappings
    ratings_df, movies_df, mappings = create_id_mappings(ratings_df, movies_df)
    
    # Merge data
    data = ratings_df.merge(movies_df, left_on='movieId', right_on='media_id', how='inner')
    logger.info(f"Merged data: {len(data)} records")
    
    # Create splits
    train_data, val_data, scaler = create_train_val_split(data)
    
    return train_data, val_data, movies_df, genres, mappings, scaler


def create_data_loaders(train_data, val_data, genres, batch_size=128):
    """Create PyTorch data loaders"""
    def make_dataset(data_df):
        user_indices = torch.tensor(data_df['user_idx'].values, dtype=torch.long)
        movie_indices = torch.tensor(data_df['content_idx'].values, dtype=torch.long)
        genre_features = torch.tensor(data_df[genres].values, dtype=torch.float32)
        ratings = torch.tensor(data_df['rating_scaled'].values, dtype=torch.float32).unsqueeze(1)
        return SimpleDataset(user_indices, movie_indices, genre_features, ratings)
    
    train_dataset = make_dataset(train_data)
    val_dataset = make_dataset(val_data)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    return train_loader, val_loader


def train_model(model, train_loader, val_loader, device, epochs=20):
    """Simplified training loop"""
    logger = logging.getLogger("training")
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        for user_ids, movie_ids, genre_feats, targets in train_loader:
            user_ids, movie_ids, genre_feats, targets = (
                user_ids.to(device), movie_ids.to(device), 
                genre_feats.to(device), targets.to(device)
            )
            
            optimizer.zero_grad()
            outputs = model(user_ids, movie_ids, genre_feats)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for user_ids, movie_ids, genre_feats, targets in val_loader:
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
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_val_loss,
                'epoch': epoch
            }, 'models/best_model.pt')
        
        scheduler.step()
    
    return model


def save_artifacts(mappings, scaler, movies_df, genres):
    """Save all training artifacts"""
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
    """Main training function - simplified"""
    logger = setup_logging()
    
    parser = argparse.ArgumentParser(description='Train movie recommendation model (simplified)')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=128, help='Batch size')
    args = parser.parse_args()
    
    try:
        # Setup
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {device}")
        
        # Find and load data
        data_files = find_data_files()
        if not data_files:
            raise ValueError("No data files found")
        
        # Process data
        train_data, val_data, movies_df, genres, mappings, scaler = load_and_process_data(data_files)
        
        # Create data loaders
        train_loader, val_loader = create_data_loaders(train_data, val_data, genres, args.batch_size)
        
        # Create model
        model = HybridRecommenderModel(
            num_users=mappings['num_users'],
            num_items=mappings['num_content'],
            num_genres=len(genres),
            embedding_dim=64
        ).to(device)
        
        logger.info(f"Model: {mappings['num_users']} users, {mappings['num_content']} items, {len(genres)} genres")
        
        # Train
        model = train_model(model, train_loader, val_loader, device, args.epochs)
        
        # Save artifacts
        save_artifacts(mappings, scaler, movies_df, genres)
        
        logger.info("Training completed successfully!")
        return 0
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())