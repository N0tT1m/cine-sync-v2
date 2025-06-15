#!/usr/bin/env python3
"""
Training script for Neural Collaborative Filtering models on MovieLens dataset.
"""

import argparse
import logging
import torch
import wandb
from pathlib import Path
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import NeuralCollaborativeFiltering, SimpleNCF, DeepNCF
from src.data_loader import NCFDataLoader
from src.trainer import NCFTrainer, NCFEvaluator


def setup_logging(log_level: str = "INFO"):
    """Setup logging configuration"""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('ncf_training.log')
        ]
    )


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train Neural Collaborative Filtering models')
    
    # Data arguments
    parser.add_argument('--ratings-path', type=str, default='../../ml-32m/ratings.csv',
                       help='Path to ratings CSV file')
    parser.add_argument('--movies-path', type=str, default='../../ml-32m/movies.csv',
                       help='Path to movies CSV file')
    parser.add_argument('--min-ratings-user', type=int, default=20,
                       help='Minimum ratings per user')
    parser.add_argument('--min-ratings-item', type=int, default=20,
                       help='Minimum ratings per item')
    
    # Model arguments
    parser.add_argument('--model-type', type=str, default='ncf', 
                       choices=['ncf', 'simple', 'deep'],
                       help='Type of NCF model to train')
    parser.add_argument('--embedding-dim', type=int, default=64,
                       help='Embedding dimension')
    parser.add_argument('--hidden-layers', type=int, nargs='+', default=[128, 64],
                       help='Hidden layer sizes for MLP')
    parser.add_argument('--dropout', type=float, default=0.2,
                       help='Dropout rate')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=1024,
                       help='Batch size')
    parser.add_argument('--learning-rate', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-5,
                       help='Weight decay for regularization')
    parser.add_argument('--patience', type=int, default=10,
                       help='Early stopping patience')
    
    # System arguments
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cuda', 'cpu'],
                       help='Device to use for training')
    parser.add_argument('--num-workers', type=int, default=4,
                       help='Number of data loader workers')
    parser.add_argument('--save-dir', type=str, default='./models',
                       help='Directory to save trained models')
    
    # Experiment tracking
    parser.add_argument('--use-wandb', action='store_true',
                       help='Use Weights & Biases for experiment tracking')
    parser.add_argument('--wandb-project', type=str, default='ncf-movielens',
                       help='Wandb project name')
    parser.add_argument('--experiment-name', type=str, default=None,
                       help='Experiment name for wandb')
    
    # Evaluation
    parser.add_argument('--evaluate-ranking', action='store_true',
                       help='Evaluate ranking metrics on test set')
    parser.add_argument('--top-k', type=int, nargs='+', default=[5, 10, 20],
                       help='Top-k values for ranking evaluation')
    
    return parser.parse_args()


def create_model(model_type: str, config: dict, args) -> torch.nn.Module:
    """Create NCF model based on type and configuration"""
    if model_type == 'ncf':
        model = NeuralCollaborativeFiltering(
            num_users=config['num_users'],
            num_items=config['num_items'],
            embedding_dim=args.embedding_dim,
            hidden_layers=args.hidden_layers,
            dropout=args.dropout
        )
    elif model_type == 'simple':
        model = SimpleNCF(
            num_users=config['num_users'],
            num_items=config['num_items'],
            embedding_dim=args.embedding_dim,
            hidden_dim=args.hidden_layers[0] if args.hidden_layers else 64,
            dropout=args.dropout
        )
    elif model_type == 'deep':
        model = DeepNCF(
            num_users=config['num_users'],
            num_items=config['num_items'],
            num_genres=config.get('num_genres', 20),
            embedding_dim=args.embedding_dim,
            hidden_layers=args.hidden_layers,
            dropout=args.dropout
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return model


def main():
    args = parse_args()
    setup_logging()
    logger = logging.getLogger(__name__)
    
    # Set device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    logger.info(f"Using device: {device}")
    
    # Initialize wandb if requested
    if args.use_wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.experiment_name,
            config=vars(args)
        )
    
    try:
        # Load data
        logger.info("Loading and preprocessing data...")
        data_loader = NCFDataLoader(
            ratings_path=args.ratings_path,
            movies_path=args.movies_path,
            min_ratings_per_user=args.min_ratings_user,
            min_ratings_per_item=args.min_ratings_item
        )
        
        # Get data loaders
        train_loader, val_loader, test_loader = data_loader.get_data_loaders(
            batch_size=args.batch_size,
            num_workers=args.num_workers
        )
        
        # Get model configuration
        model_config = data_loader.get_model_config()
        logger.info(f"Model config: {model_config}")
        
        # Create model
        model = create_model(args.model_type, model_config, args)
        logger.info(f"Created {args.model_type} model with {sum(p.numel() for p in model.parameters())} parameters")
        
        # Create trainer
        trainer = NCFTrainer(
            model=model,
            device=device,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay
        )
        
        # Train model
        logger.info("Starting training...")
        history = trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=args.epochs,
            patience=args.patience,
            save_dir=args.save_dir
        )
        
        # Log training results
        logger.info(f"Training completed. Best validation loss: {trainer.best_val_loss:.4f}")
        
        # Evaluate on test set
        logger.info("Evaluating on test set...")
        test_metrics = trainer.evaluate(test_loader)
        
        logger.info("Test Results:")
        for metric, value in test_metrics.items():
            logger.info(f"  {metric}: {value:.4f}")
        
        # Ranking evaluation if requested
        if args.evaluate_ranking:
            logger.info("Evaluating ranking metrics...")
            evaluator = NCFEvaluator(model, device)
            ranking_metrics = evaluator.evaluate_ranking(test_loader, args.top_k)
            
            logger.info("Ranking Results:")
            for metric, value in ranking_metrics.items():
                logger.info(f"  {metric}: {value:.4f}")
            
            # Combine all metrics
            all_metrics = {**test_metrics, **ranking_metrics}
        else:
            all_metrics = test_metrics
        
        # Log to wandb if enabled
        if args.use_wandb:
            wandb.log(all_metrics)
            wandb.log({"training_history": history})
        
        # Save encoders and final model info
        save_path = Path(args.save_dir)
        data_loader.save_encoders(save_path / 'encoders.pkl')
        
        # Save final metrics
        import json
        with open(save_path / 'final_metrics.json', 'w') as f:
            json.dump(all_metrics, f, indent=2)
        
        logger.info(f"Training completed successfully. Models saved to {args.save_dir}")
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        if args.use_wandb:
            wandb.finish(exit_code=1)
        raise
    
    finally:
        if args.use_wandb:
            wandb.finish()


if __name__ == "__main__":
    main()