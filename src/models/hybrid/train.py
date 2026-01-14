#!/usr/bin/env python3
"""
CineSync v2 - Unified Content Recommender Training Script

Trains the unified hybrid recommendation model for movies, TV shows, or both.

Usage:
    python train.py --content-type movie
    python train.py --content-type tv
    python train.py --content-type both
"""

import argparse
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from config import load_config, ContentType, AppConfig
from content_recommender import UnifiedContentRecommender, ContentDataset, ContentType as ModelContentType
from utils.database import DatabaseManager, load_ratings_data, load_content_data
from utils.error_handling import handle_exceptions, validate_batch_size

# Optional: WandB for experiment tracking
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


def setup_logging(debug: bool = False) -> logging.Logger:
    """Configure logging for training."""
    log_level = logging.DEBUG if debug else logging.INFO
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

    logging.basicConfig(
        level=log_level,
        format=log_format,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(
                f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
            )
        ]
    )
    return logging.getLogger(__name__)


def setup_device() -> torch.device:
    """Setup compute device (GPU if available)."""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        logging.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
        logging.info("Using Apple Silicon MPS")
    else:
        device = torch.device('cpu')
        logging.info("Using CPU")
    return device


def prepare_data(
    ratings_df: pd.DataFrame,
    content_df: pd.DataFrame,
    content_type: ContentType,
    num_genres: int = 20
) -> Tuple[Dict, np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Prepare data for training.

    Returns:
        Tuple of (mappings, user_ids, item_ids, ratings, genre_features, tv_features)
    """
    logger = logging.getLogger(__name__)

    # Create ID encoders
    user_encoder = LabelEncoder()
    item_encoder = LabelEncoder()

    user_ids = user_encoder.fit_transform(ratings_df['userId'].values)
    item_ids = item_encoder.fit_transform(ratings_df['itemId'].values)
    ratings = ratings_df['rating'].values.astype(np.float32)

    # Scale ratings to [0, 1]
    rating_scaler = MinMaxScaler()
    ratings = rating_scaler.fit_transform(ratings.reshape(-1, 1)).flatten()

    num_users = len(user_encoder.classes_)
    num_items = len(item_encoder.classes_)

    logger.info(f"Data: {num_users} users, {num_items} items, {len(ratings)} ratings")

    # Process genre features
    genre_features = None
    if 'genres' in content_df.columns:
        all_genres = set()
        for genres_str in content_df['genres'].dropna():
            if isinstance(genres_str, str):
                all_genres.update(g.strip() for g in genres_str.split('|'))

        genre_list = sorted(list(all_genres))[:num_genres]
        genre_to_idx = {g: i for i, g in enumerate(genre_list)}

        # Create genre features for each item
        item_to_genres = {}
        for _, row in content_df.iterrows():
            item_id = row.get('item_id', row.get('media_id'))
            genres_str = row.get('genres', '')
            if pd.notna(genres_str) and isinstance(genres_str, str):
                genres = [g.strip() for g in genres_str.split('|')]
                genre_vec = np.zeros(num_genres, dtype=np.float32)
                for g in genres:
                    if g in genre_to_idx:
                        genre_vec[genre_to_idx[g]] = 1.0
                item_to_genres[item_id] = genre_vec

        # Map genre features to training data
        genre_features = np.zeros((len(item_ids), num_genres), dtype=np.float32)
        original_item_ids = item_encoder.inverse_transform(item_ids)
        for i, orig_id in enumerate(original_item_ids):
            if orig_id in item_to_genres:
                genre_features[i] = item_to_genres[orig_id]

        logger.info(f"Genre features: {len(genre_list)} genres encoded")

    # Process TV features (only for TV content)
    tv_features = None
    if content_type in [ContentType.TV, ContentType.BOTH]:
        tv_cols = ['episode_count', 'season_count', 'duration', 'status']
        if all(col in content_df.columns for col in tv_cols[:3]):
            # Create TV features mapping
            item_to_tv = {}
            for _, row in content_df.iterrows():
                item_id = row.get('item_id', row.get('show_id'))
                if row.get('content_type') == 'tv':
                    tv_vec = np.array([
                        float(row.get('episode_count', 0) or 0),
                        float(row.get('season_count', 0) or 0),
                        float(row.get('duration', 0) or 0),
                        float(row.get('status', 0) or 0)
                    ], dtype=np.float32)
                    item_to_tv[item_id] = tv_vec

            # Map TV features to training data
            if item_to_tv:
                tv_features = np.zeros((len(item_ids), 4), dtype=np.float32)
                original_item_ids = item_encoder.inverse_transform(item_ids)
                for i, orig_id in enumerate(original_item_ids):
                    if orig_id in item_to_tv:
                        tv_features[i] = item_to_tv[orig_id]

                logger.info(f"TV features: {len(item_to_tv)} items with TV metadata")

    # Store mappings
    mappings = {
        'user_encoder': user_encoder,
        'item_encoder': item_encoder,
        'rating_scaler': rating_scaler,
        'num_users': num_users,
        'num_items': num_items,
        'num_genres': num_genres
    }

    return mappings, user_ids, item_ids, ratings, genre_features, tv_features


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    content_type: ModelContentType,
    scaler: Optional[torch.cuda.amp.GradScaler] = None
) -> float:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    num_batches = 0

    for batch in train_loader:
        user_ids = batch['user_id'].to(device)
        item_ids = batch['item_id'].to(device)
        ratings = batch['rating'].to(device)

        genre_features = batch.get('genre_features')
        if genre_features is not None:
            genre_features = genre_features.to(device)

        tv_features = batch.get('tv_features')
        if tv_features is not None:
            tv_features = tv_features.to(device)

        optimizer.zero_grad()

        # Mixed precision training
        if scaler is not None:
            with torch.cuda.amp.autocast():
                predictions = model(
                    user_ids, item_ids, content_type,
                    genre_features, tv_features
                )
                loss = criterion(predictions, ratings)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            predictions = model(
                user_ids, item_ids, content_type,
                genre_features, tv_features
            )
            loss = criterion(predictions, ratings)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / max(num_batches, 1)


def evaluate(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    content_type: ModelContentType
) -> Dict[str, float]:
    """Evaluate model on validation set."""
    model.eval()
    total_loss = 0.0
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for batch in val_loader:
            user_ids = batch['user_id'].to(device)
            item_ids = batch['item_id'].to(device)
            ratings = batch['rating'].to(device)

            genre_features = batch.get('genre_features')
            if genre_features is not None:
                genre_features = genre_features.to(device)

            tv_features = batch.get('tv_features')
            if tv_features is not None:
                tv_features = tv_features.to(device)

            predictions = model(
                user_ids, item_ids, content_type,
                genre_features, tv_features
            )
            loss = criterion(predictions, ratings)

            total_loss += loss.item()
            all_predictions.extend(predictions.cpu().numpy())
            all_targets.extend(ratings.cpu().numpy())

    # Calculate metrics
    predictions = np.array(all_predictions)
    targets = np.array(all_targets)

    mse = np.mean((predictions - targets) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(predictions - targets))

    return {
        'loss': total_loss / len(val_loader),
        'mse': mse,
        'rmse': rmse,
        'mae': mae
    }


def train(args: argparse.Namespace, config: AppConfig):
    """Main training function."""
    logger = setup_logging(config.debug)
    device = setup_device()

    # Parse content type
    content_type = ContentType(args.content_type)
    model_content_type = ModelContentType(args.content_type)

    logger.info(f"Training unified model for content type: {content_type.value}")

    # Load data
    logger.info("Loading data...")

    if args.ratings_path and os.path.exists(args.ratings_path):
        ratings_df = pd.read_csv(args.ratings_path)
        # Standardize column names
        if 'movieId' in ratings_df.columns:
            ratings_df = ratings_df.rename(columns={'movieId': 'itemId'})
        if 'show_id' in ratings_df.columns:
            ratings_df = ratings_df.rename(columns={'show_id': 'itemId'})
    else:
        # Load from database
        db_manager = DatabaseManager(config.database)
        ratings_df = load_ratings_data(db_manager, content_type)

    if args.content_path and os.path.exists(args.content_path):
        content_df = pd.read_csv(args.content_path)
    else:
        db_manager = DatabaseManager(config.database)
        content_df = load_content_data(db_manager, content_type)

    logger.info(f"Loaded {len(ratings_df)} ratings, {len(content_df)} items")

    # Prepare data
    mappings, user_ids, item_ids, ratings, genre_features, tv_features = prepare_data(
        ratings_df, content_df, content_type, config.model.num_genres
    )

    # Create dataset
    dataset = ContentDataset(
        user_ids, item_ids, ratings,
        model_content_type,
        genre_features, tv_features
    )

    # Split data
    val_size = int(len(dataset) * config.training.val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.model.batch_size,
        shuffle=True,
        num_workers=config.training.num_workers,
        pin_memory=config.training.pin_memory
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.model.batch_size,
        shuffle=False,
        num_workers=config.training.num_workers,
        pin_memory=config.training.pin_memory
    )

    logger.info(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    # Create model
    model = UnifiedContentRecommender(
        num_users=mappings['num_users'],
        num_items=mappings['num_items'],
        num_genres=config.model.num_genres,
        embedding_dim=config.model.embedding_dim,
        hidden_dims=config.model.hidden_dims,
        dropout_rate=config.model.dropout_rate,
        use_genre_features=config.model.use_genre_features,
        use_tv_features=config.model.use_tv_features
    ).to(device)

    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Setup training
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.model.learning_rate,
        weight_decay=config.model.weight_decay
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )

    # Mixed precision scaler
    scaler = None
    if config.training.use_mixed_precision and device.type == 'cuda':
        scaler = torch.cuda.amp.GradScaler()

    # Initialize WandB
    if WANDB_AVAILABLE and config.training.use_wandb:
        wandb.init(
            project=config.training.wandb_project,
            name=f"unified_{content_type.value}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            config={
                'content_type': content_type.value,
                'num_users': mappings['num_users'],
                'num_items': mappings['num_items'],
                **vars(config.model)
            }
        )

    # Training loop
    best_val_loss = float('inf')
    patience_counter = 0

    logger.info("Starting training...")
    for epoch in range(config.model.num_epochs):
        train_loss = train_epoch(
            model, train_loader, optimizer, criterion,
            device, model_content_type, scaler
        )

        val_metrics = evaluate(model, val_loader, criterion, device, model_content_type)

        scheduler.step(val_metrics['loss'])

        # Logging
        logger.info(
            f"Epoch {epoch + 1}/{config.model.num_epochs} - "
            f"Train Loss: {train_loss:.4f}, Val Loss: {val_metrics['loss']:.4f}, "
            f"Val RMSE: {val_metrics['rmse']:.4f}"
        )

        if WANDB_AVAILABLE and config.training.use_wandb:
            wandb.log({
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'val_loss': val_metrics['loss'],
                'val_rmse': val_metrics['rmse'],
                'val_mae': val_metrics['mae'],
                'learning_rate': optimizer.param_groups[0]['lr']
            })

        # Save best model
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            patience_counter = 0

            save_path = Path(config.model.models_dir) / f"best_unified_{content_type.value}.pt"
            save_path.parent.mkdir(parents=True, exist_ok=True)

            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': best_val_loss,
                'mappings': mappings,
                'config': {
                    'num_users': mappings['num_users'],
                    'num_items': mappings['num_items'],
                    'num_genres': config.model.num_genres,
                    'embedding_dim': config.model.embedding_dim,
                    'hidden_dims': config.model.hidden_dims,
                    'use_genre_features': config.model.use_genre_features,
                    'use_tv_features': config.model.use_tv_features
                }
            }, save_path)

            logger.info(f"Saved best model to {save_path}")
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= config.model.patience:
            logger.info(f"Early stopping after {epoch + 1} epochs")
            break

    if WANDB_AVAILABLE and config.training.use_wandb:
        wandb.finish()

    logger.info(f"Training complete. Best val loss: {best_val_loss:.4f}")


def main():
    parser = argparse.ArgumentParser(
        description='Train unified content recommender'
    )
    parser.add_argument(
        '--content-type', type=str, default='both',
        choices=['movie', 'tv', 'both'],
        help='Content type to train on'
    )
    parser.add_argument(
        '--ratings-path', type=str,
        help='Path to ratings CSV file'
    )
    parser.add_argument(
        '--content-path', type=str,
        help='Path to content metadata CSV file'
    )
    parser.add_argument(
        '--epochs', type=int,
        help='Override number of training epochs'
    )
    parser.add_argument(
        '--batch-size', type=int,
        help='Override batch size'
    )
    parser.add_argument(
        '--learning-rate', type=float,
        help='Override learning rate'
    )
    parser.add_argument(
        '--no-wandb', action='store_true',
        help='Disable WandB logging'
    )
    parser.add_argument(
        '--debug', action='store_true',
        help='Enable debug mode'
    )

    args = parser.parse_args()

    # Load config
    config = load_config(args.content_type)

    # Override config with CLI args
    if args.epochs:
        config.model.num_epochs = args.epochs
    if args.batch_size:
        config.model.batch_size = args.batch_size
    if args.learning_rate:
        config.model.learning_rate = args.learning_rate
    if args.no_wandb:
        config.training.use_wandb = False
    if args.debug:
        config.debug = True

    train(args, config)


if __name__ == '__main__':
    main()
