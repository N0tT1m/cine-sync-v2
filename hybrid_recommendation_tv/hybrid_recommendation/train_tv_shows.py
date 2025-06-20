#!/usr/bin/env python3
"""
TV Show Model Training Script for CineSync v2
Trains a separate neural network model specifically for TV show recommendations
"""

import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader
import pickle
import logging
import argparse
import gc
import time
from datetime import datetime
from pathlib import Path

# Import RTX 4090 BEAST MODE optimizations
sys.path.append(str(Path(__file__).parent.parent.parent / 'neural_collaborative_filtering'))
from performance_config import PerformanceOptimizer, apply_rtx4090_optimizations

# Import custom modules
from config import load_config
from models.tv_recommender import TVShowRecommenderModel, TVShowDataset, save_tv_model
from process_tv_datasets import TVDatasetProcessor

def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f"tv_training_{datetime.now().strftime('%Y%m%d%H%M%S')}.log", encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger("tv-show-training")

def setup_gpu():
    """Set up GPU for training with RTX 4090 BEAST MODE"""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        num_gpus = torch.cuda.device_count()
        logger.info(f"Found {num_gpus} CUDA-enabled GPU(s)")
        logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
        
        # 🔥🔥🔥 ACTIVATE RTX 4090 BEAST MODE 🔥🔥🔥
        apply_rtx4090_optimizations()
        PerformanceOptimizer.setup_maximum_performance()
        
        # Clear GPU cache
        torch.cuda.empty_cache()
        gc.collect()
        
        logger.info("🚀 RTX 4090 TV SHOW BEAST MODE ACTIVATED!")
        return device
    else:
        logger.info("CUDA not available, using CPU")
        return torch.device('cpu')

def log_gpu_memory():
    """Log current GPU memory usage"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        cached = torch.cuda.memory_reserved() / 1024**3
        logger.info(f"GPU Memory - Allocated: {allocated:.2f}GB, Cached: {cached:.2f}GB")

class TVShowTrainer:
    """TV Show recommendation model trainer"""
    
    def __init__(self, config):
        self.config = config
        self.device = setup_gpu()
        self.scaler = GradScaler('cuda') if self.device.type == 'cuda' else None
        
        # Training parameters (RTX 4090 BEAST MODE)
        self.batch_size = getattr(config.model, 'batch_size', 2048)  # BEAST MODE batch size
        self.learning_rate = getattr(config.model, 'learning_rate', 0.003)  # Higher LR for large batches
        self.num_epochs = getattr(config.model, 'num_epochs', 20)
        self.embedding_dim = getattr(config.model, 'embedding_dim', 256)  # BEAST MODE embeddings
        self.hidden_dim = getattr(config.model, 'hidden_dim', 512)  # BEAST MODE hidden dims
        
        # Paths
        self.models_dir = Path(config.model.models_dir)
        self.models_dir.mkdir(exist_ok=True)
    
    def load_comprehensive_tv_datasets(self, tv_sources: list) -> tuple:
        """Load comprehensive TV datasets from multiple sources"""
        import glob
        import pandas as pd
        
        tv_datasets = {
            'tmdb': {
                'content': '../../tv/misc/TMDB_tv_dataset_v3.csv',
                'id_col': 'id',
                'title_col': 'name'
            },
            'netflix': {
                'content': '../../tv/netflix/netflix_titles.csv',
                'id_col': 'show_id',
                'title_col': 'title'
            },
            'amazon': {
                'content': '../../tv/amazon/amazon_prime_tv_shows.csv',
                'id_col': 'show_id',
                'title_col': 'title'
            },
            'disney': {
                'content': '../../tv/misc/disney_plus_tv_shows.csv',
                'id_col': 'show_id',
                'title_col': 'title'
            },
            'anime': {
                'content': '../../tv/anime/animes.csv',
                'id_col': 'anime_id',
                'title_col': 'Name'
            },
            'imdb': {
                'content': '../../tv/imdb/*.csv',
                'id_col': 'tconst',
                'title_col': 'primaryTitle'
            }
        }
        
        all_content = []
        
        for source in tv_sources:
            if source in tv_datasets:
                dataset_info = tv_datasets[source]
                logger.info(f"Loading TV dataset: {source}")
                
                content_path = dataset_info['content']
                if '*' in content_path:
                    content_files = glob.glob(content_path)
                else:
                    content_files = [content_path]
                
                for file_path in content_files:
                    if Path(file_path).exists():
                        try:
                            content_df = pd.read_csv(file_path)
                            
                            # Handle different CSV structures
                            if dataset_info['id_col'] in content_df.columns:
                                content_df = content_df.rename(columns={
                                    dataset_info['id_col']: 'show_id',
                                    dataset_info['title_col']: 'title'
                                })
                            elif 'id' in content_df.columns:
                                content_df = content_df.rename(columns={'id': 'show_id'})
                            
                            # Ensure required columns exist
                            if 'show_id' not in content_df.columns:
                                content_df['show_id'] = range(len(content_df))
                            if 'title' not in content_df.columns and 'Title' in content_df.columns:
                                content_df['title'] = content_df['Title']
                            
                            content_df['source'] = source
                            all_content.append(content_df)
                            
                        except Exception as e:
                            logger.warning(f"Failed to load {file_path}: {e}")
        
        # Combine all TV content
        if all_content:
            combined_content = pd.concat(all_content, ignore_index=True)
            combined_content = combined_content.drop_duplicates(subset=['title'], keep='first')
            combined_content['show_id'] = range(1, len(combined_content) + 1)
        else:
            raise ValueError("No TV datasets found")
        
        # Create synthetic ratings for TV content
        logger.info("Creating synthetic ratings for TV content training...")
        synthetic_ratings = self._create_synthetic_tv_ratings(combined_content)
        
        logger.info(f"Loaded {len(combined_content)} TV shows and {len(synthetic_ratings)} synthetic ratings")
        return synthetic_ratings, combined_content
    
    def _create_synthetic_tv_ratings(self, shows_df: pd.DataFrame) -> pd.DataFrame:
        """Create synthetic ratings specifically for TV shows"""
        import numpy as np
        
        # Create TV-specific viewing patterns
        num_users = 2000  # More users for TV content
        avg_shows_per_user = 30
        
        synthetic_ratings = []
        
        for user_id in range(1, num_users + 1):
            # Sample shows for this user
            num_shows = np.random.poisson(avg_shows_per_user)
            user_shows = shows_df.sample(n=min(num_shows, len(shows_df)))
            
            for _, show in user_shows.iterrows():
                # TV shows often have binge-watching patterns
                base_rating = 3.5
                
                # Adjust based on source (some platforms have higher-rated content)
                source = show.get('source', 'unknown')
                if source in ['netflix', 'disney']:
                    base_rating += 0.3
                elif source == 'anime':
                    base_rating += 0.2
                
                # Add variance
                rating = np.clip(
                    np.random.normal(base_rating, 0.8), 
                    1.0, 5.0
                )
                
                synthetic_ratings.append({
                    'user_id': user_id,
                    'show_id': show['show_id'],
                    'rating': rating
                })
        
        return pd.DataFrame(synthetic_ratings)
        
    def prepare_data(self) -> dict:
        """Prepare TV show training data from comprehensive TV datasets"""
        logger.info("Preparing TV show training data from all available TV datasets...")
        
        # Use comprehensive TV dataset loading
        tv_sources = ['tmdb', 'netflix', 'amazon', 'disney', 'anime', 'imdb']
        ratings_df, content_df = self.load_comprehensive_tv_datasets(tv_sources)
        
        # Process TV show datasets
        processor = TVDatasetProcessor()
        training_data = processor.create_training_data_from_dataframes(ratings_df, content_df)
        
        if not training_data or training_data['shows_df'].empty:
            raise ValueError("No TV show data available for training")
        
        shows_df = training_data['shows_df']
        ratings_df = training_data['ratings_df']
        
        logger.info(f"Loaded {len(shows_df)} TV shows and {len(ratings_df)} ratings")
        
        # Create additional TV-specific features
        shows_df = self._process_tv_features(shows_df)
        
        # Split ratings data for training/validation
        if not ratings_df.empty:
            train_ratings, val_ratings = train_test_split(
                ratings_df, test_size=0.2, random_state=42, stratify=ratings_df['user_id']
            )
        else:
            # If no ratings data, create synthetic ratings for content-based training
            train_ratings, val_ratings = self._create_synthetic_ratings(shows_df)
        
        # Create datasets
        train_dataset = TVShowDataset(
            train_ratings, shows_df, 
            training_data['encoders']['user_id'],
            training_data['encoders']['show_id']
        )
        
        val_dataset = TVShowDataset(
            val_ratings, shows_df,
            training_data['encoders']['user_id'], 
            training_data['encoders']['show_id']
        )
        
        # Create data loaders with RTX 4090 BEAST MODE settings
        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, 
            shuffle=True, num_workers=24, pin_memory=True,  # BEAST MODE workers
            persistent_workers=True, prefetch_factor=4
        )
        
        val_loader = DataLoader(
            val_dataset, batch_size=self.batch_size,
            shuffle=False, num_workers=24, pin_memory=True,  # BEAST MODE workers
            persistent_workers=True, prefetch_factor=4
        )
        
        return {
            'train_loader': train_loader,
            'val_loader': val_loader,
            'metadata': training_data,
            'shows_df': shows_df
        }
    
    def _process_tv_features(self, shows_df: pd.DataFrame) -> pd.DataFrame:
        """Process TV-specific features"""
        # Extract episode count (if available)
        shows_df['episode_count'] = pd.to_numeric(
            shows_df.get('Episodes', shows_df.get('episode_count', 0)), 
            errors='coerce'
        ).fillna(0)
        
        # Estimate season count (episodes / 12 as rough estimate)
        shows_df['season_count'] = np.ceil(shows_df['episode_count'] / 12).fillna(1)
        
        # Extract duration (convert to minutes)
        shows_df['duration'] = self._extract_duration(shows_df.get('Duration', ''))
        
        # Process status
        status_mapping = {
            'Finished Airing': 0, 'Completed': 0, 'Ended': 0,
            'Currently Airing': 1, 'Ongoing': 1, 'Running': 1,
            'Not yet aired': 2, 'Upcoming': 2,
            'Cancelled': 3, 'Dropped': 3,
            'Unknown': 4
        }
        
        shows_df['status_encoded'] = shows_df.get('Status', 'Unknown').map(
            lambda x: status_mapping.get(str(x), 4)
        )
        
        # Normalize numerical features
        scaler = MinMaxScaler()
        numerical_features = ['episode_count', 'season_count', 'duration']
        
        for feature in numerical_features:
            if feature in shows_df.columns:
                values = shows_df[feature].values.reshape(-1, 1)
                shows_df[feature] = scaler.fit_transform(values).flatten()
        
        return shows_df
    
    def _extract_duration(self, duration_series) -> pd.Series:
        """Extract duration in minutes from duration strings"""
        if isinstance(duration_series, str):
            duration_series = pd.Series([duration_series])
        
        def parse_duration(duration_str):
            if pd.isna(duration_str) or duration_str == '':
                return 24.0  # Default 24 minutes for TV episodes
            
            duration_str = str(duration_str).lower()
            minutes = 0
            
            # Extract hours and minutes
            if 'hr' in duration_str or 'hour' in duration_str:
                hours = float(''.join(filter(str.isdigit, duration_str.split('hr')[0])))
                minutes += hours * 60
            
            if 'min' in duration_str:
                mins = float(''.join(filter(str.isdigit, duration_str.split('min')[0].split()[-1])))
                minutes += mins
            
            # If just a number, assume minutes
            if not minutes and any(c.isdigit() for c in duration_str):
                minutes = float(''.join(filter(str.isdigit, duration_str)))
            
            return max(minutes, 1)  # Minimum 1 minute
        
        return duration_series.apply(parse_duration)
    
    def _create_synthetic_ratings(self, shows_df: pd.DataFrame) -> tuple:
        """Create synthetic ratings for content-based training when no user ratings available"""
        logger.info("Creating synthetic ratings for content-based training...")
        
        # Create synthetic users based on genre preferences
        genres = ['Action', 'Comedy', 'Drama', 'Horror', 'Romance', 'Sci-Fi', 'Thriller']
        synthetic_users = []
        
        for i, genre in enumerate(genres):
            for preference_strength in [0.7, 0.8, 0.9]:
                synthetic_users.append({
                    'user_id': f'synthetic_user_{i}_{int(preference_strength*10)}',
                    'preferred_genre': genre,
                    'preference_strength': preference_strength
                })
        
        # Generate ratings based on genre matching
        synthetic_ratings = []
        
        for user in synthetic_users:
            user_shows = shows_df.sample(min(100, len(shows_df)))  # Sample shows for each user
            
            for _, show in user_shows.iterrows():
                show_genres = str(show.get('genres', '')).lower()
                preferred_genre = user['preferred_genre'].lower()
                
                # Base rating based on genre match
                if preferred_genre in show_genres:
                    base_rating = 4.0 + np.random.normal(0, 0.5)
                else:
                    base_rating = 2.5 + np.random.normal(0, 1.0)
                
                # Add noise and clamp to valid range
                rating = np.clip(base_rating + np.random.normal(0, 0.3), 0.5, 5.0)
                
                synthetic_ratings.append({
                    'user_id': user['user_id'],
                    'show_id': show['show_id'],
                    'rating': rating
                })
        
        ratings_df = pd.DataFrame(synthetic_ratings)
        
        # Split synthetic ratings
        train_ratings, val_ratings = train_test_split(
            ratings_df, test_size=0.2, random_state=42
        )
        
        return train_ratings, val_ratings
    
    def create_model(self, metadata: dict) -> TVShowRecommenderModel:
        """Create TV show recommendation model"""
        model = TVShowRecommenderModel(
            num_users=metadata['num_users'],
            num_shows=metadata['num_shows'],
            num_genres=metadata['num_genres'],
            embedding_dim=self.embedding_dim,
            hidden_dim=self.hidden_dim
        )
        
        return model.to(self.device)
    
    def train_epoch(self, model, train_loader, optimizer, criterion, epoch):
        """Train model for one epoch"""
        model.train()
        total_loss = 0
        num_batches = 0
        
        for batch_idx, batch in enumerate(train_loader):
            # Move batch to device
            user_ids = batch['user_id'].to(self.device)
            show_ids = batch['show_id'].to(self.device)
            genre_features = batch['genre_features'].to(self.device)
            tv_features = batch['tv_features'].to(self.device)
            ratings = batch['rating'].to(self.device)
            
            optimizer.zero_grad()
            
            # Forward pass with mixed precision if using CUDA
            if self.scaler:
                with autocast('cuda'):
                    predictions = model(user_ids, show_ids, genre_features, tv_features)
                    loss = criterion(predictions, ratings)
                
                self.scaler.scale(loss).backward()
                self.scaler.step(optimizer)
                self.scaler.update()
            else:
                predictions = model(user_ids, show_ids, genre_features, tv_features)
                loss = criterion(predictions, ratings)
                loss.backward()
                optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            if batch_idx % 100 == 0:
                logger.info(f'Epoch {epoch}, Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}')
                if self.device.type == 'cuda':
                    log_gpu_memory()
        
        return total_loss / num_batches
    
    def validate(self, model, val_loader, criterion):
        """Validate model"""
        model.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                user_ids = batch['user_id'].to(self.device)
                show_ids = batch['show_id'].to(self.device)
                genre_features = batch['genre_features'].to(self.device)
                tv_features = batch['tv_features'].to(self.device)
                ratings = batch['rating'].to(self.device)
                
                if self.scaler:
                    with autocast():
                        predictions = model(user_ids, show_ids, genre_features, tv_features)
                        loss = criterion(predictions, ratings)
                else:
                    predictions = model(user_ids, show_ids, genre_features, tv_features)
                    loss = criterion(predictions, ratings)
                
                total_loss += loss.item()
                num_batches += 1
        
        return total_loss / num_batches
    
    def train(self):
        """Main training loop"""
        logger.info("Starting TV show model training...")
        
        # Prepare data
        data = self.prepare_data()
        train_loader = data['train_loader']
        val_loader = data['val_loader']
        metadata = data['metadata']
        
        # Create model
        model = self.create_model(metadata)
        logger.info(f"Created model with {sum(p.numel() for p in model.parameters())} parameters")
        
        # Setup training
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=self.learning_rate, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)
        
        best_val_loss = float('inf')
        patience_counter = 0
        patience = 5
        
        # Training loop
        for epoch in range(self.num_epochs):
            start_time = time.time()
            
            # Train
            train_loss = self.train_epoch(model, train_loader, optimizer, criterion, epoch)
            
            # Validate
            val_loss = self.validate(model, val_loader, criterion)
            
            # Update scheduler
            scheduler.step(val_loss)
            
            epoch_time = time.time() - start_time
            
            logger.info(f'Epoch {epoch+1}/{self.num_epochs} - '
                       f'Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, '
                       f'Time: {epoch_time:.2f}s')
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                
                # Save model
                model_path = self.models_dir / "best_tv_model.pt"
                save_tv_model(model, str(model_path), {
                    'num_users': metadata['num_users'],
                    'num_shows': metadata['num_shows'],
                    'num_genres': metadata['num_genres'],
                    'embedding_dim': self.embedding_dim,
                    'hidden_dim': self.hidden_dim,
                    'val_loss': best_val_loss,
                    'epoch': epoch + 1
                })
                
                logger.info(f"Saved best model with validation loss: {best_val_loss:.4f}")
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
            
            # Clear GPU cache
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
                gc.collect()
        
        # Save comprehensive final model artifacts
        final_model_path = self.models_dir / "final_tv_model.pt"
        save_tv_model(model, str(final_model_path), {
            'num_users': metadata['num_users'],
            'num_shows': metadata['num_shows'],
            'num_genres': metadata['num_genres'],
            'embedding_dim': self.embedding_dim,
            'hidden_dim': self.hidden_dim,
            'final_val_loss': val_loss,
            'epochs_trained': epoch + 1
        })
        
        # Save additional required files for compatibility
        torch.save(model.state_dict(), self.models_dir / "recommendation_model.pt")
        
        # Save comprehensive model metadata
        comprehensive_metadata = {
            'model_type': 'hybrid_tv',
            'num_users': metadata['num_users'],
            'num_shows': metadata['num_shows'], 
            'num_genres': metadata['num_genres'],
            'genre_list': metadata['genre_list'],
            'embedding_dim': self.embedding_dim,
            'hidden_dim': self.hidden_dim,
            'best_val_loss': best_val_loss,
            'final_val_loss': val_loss,
            'epochs_trained': epoch + 1,
            'total_parameters': sum(p.numel() for p in model.parameters()),
            'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
            'model_size_mb': sum(p.numel() * p.element_size() for p in model.parameters()) / (1024**2),
            'model_architecture': str(model)
        }
        
        with open(self.models_dir / "tv_metadata.pkl", 'wb') as f:
            pickle.dump(comprehensive_metadata, f)
        
        # Save encoders and metadata for inference
        with open(self.models_dir / "tv_encoders.pkl", 'wb') as f:
            pickle.dump(metadata['encoders'], f)
        
        # Save training history
        training_history = {
            'best_val_loss': best_val_loss,
            'final_val_loss': val_loss,
            'epochs_trained': epoch + 1
        }
        
        with open(self.models_dir / "training_history.pkl", 'wb') as f:
            pickle.dump(training_history, f)
        
        # Save final metrics JSON
        final_metrics = {
            'best_val_loss': best_val_loss,
            'final_val_loss': val_loss,
            'epochs_trained': epoch + 1,
            'model_type': 'hybrid_tv',
            'total_parameters': comprehensive_metadata['total_parameters']
        }
        
        import json
        with open(self.models_dir / "final_metrics.json", 'w') as f:
            json.dump(final_metrics, f, indent=2)
        
        # Create TV show lookup files
        if 'encoders' in metadata and 'show_encoder' in metadata['encoders']:
            show_encoder = metadata['encoders']['show_encoder']
            if hasattr(show_encoder, 'classes_'):
                show_lookup = {
                    'show_id_to_idx': {str(k): v for k, v in zip(show_encoder.classes_, range(len(show_encoder.classes_)))},
                    'idx_to_show_id': {v: str(k) for k, v in zip(show_encoder.classes_, range(len(show_encoder.classes_)))},
                    'num_shows': len(show_encoder.classes_)
                }
                
                # Save as movie_lookup for compatibility
                with open(self.models_dir / "movie_lookup.pkl", 'wb') as f:
                    pickle.dump({
                        'movie_id_to_idx': show_lookup['show_id_to_idx'],
                        'idx_to_movie_id': show_lookup['idx_to_show_id'],
                        'num_movies': show_lookup['num_shows']
                    }, f)
                
                with open(self.models_dir / "movie_lookup_backup.pkl", 'wb') as f:
                    pickle.dump({
                        'movie_id_to_idx': show_lookup['show_id_to_idx'],
                        'idx_to_movie_id': show_lookup['idx_to_show_id'],
                        'num_movies': show_lookup['num_shows']
                    }, f)
        
        # Save rating scaler
        rating_scaler = {
            'min_rating': 1.0,
            'max_rating': 5.0,
            'mean_rating': 3.5,
            'std_rating': 1.0
        }
        
        with open(self.models_dir / "rating_scaler.pkl", 'wb') as f:
            pickle.dump(rating_scaler, f)
        
        logger.info("TV show model training completed!")
        logger.info(f"Best validation loss: {best_val_loss:.4f}")
        
        return model, best_val_loss

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Train TV Show Recommendation Model')
    parser.add_argument('--config', type=str, default=None, help='Config file path')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size')
    parser.add_argument('--learning-rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--embedding-dim', type=int, default=64, help='Embedding dimension')
    parser.add_argument('--hidden-dim', type=int, default=128, help='Hidden dimension')
    
    args = parser.parse_args()
    
    global logger
    logger = setup_logging()
    
    # Load config
    config = load_config()
    
    # Override config with command line arguments
    if args.epochs:
        config.model.num_epochs = args.epochs
    if args.batch_size:
        config.model.batch_size = args.batch_size
    if args.learning_rate:
        config.model.learning_rate = args.learning_rate
    if args.embedding_dim:
        config.model.embedding_dim = args.embedding_dim
    if args.hidden_dim:
        config.model.hidden_dim = args.hidden_dim
    
    # Create trainer and train
    trainer = TVShowTrainer(config)
    model, best_loss = trainer.train()
    
    logger.info(f"Training completed with best validation loss: {best_loss:.4f}")

if __name__ == "__main__":
    main()