#!/usr/bin/env python3
"""
Training script for Sequential Recommendation models.
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

from src.model import (
    SequentialRecommender, AttentionalSequentialRecommender,
    HierarchicalSequentialRecommender, SessionBasedRecommender
)
from src.data_loader import SequentialDataLoader
from src.trainer import SequentialTrainer, SequentialEvaluator


def setup_logging(log_level: str = "INFO"):
    """Setup logging configuration"""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('sequential_training.log')
        ]
    )


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train Sequential Recommendation models')
    
    # Data arguments
    parser.add_argument('--ratings-path', type=str, default='../../ml-32m/ratings.csv',
                       help='Path to ratings CSV file')
    parser.add_argument('--min-interactions', type=int, default=20,
                       help='Minimum interactions per user')
    parser.add_argument('--min-seq-length', type=int, default=5,
                       help='Minimum sequence length')
    parser.add_argument('--max-seq-length', type=int, default=50,
                       help='Maximum sequence length')
    
    # Model arguments
    parser.add_argument('--model-type', type=str, default='sequential',
                       choices=['sequential', 'attention', 'hierarchical', 'session'],
                       help='Type of sequential model')
    parser.add_argument('--embedding-dim', type=int, default=128,
                       help='Item embedding dimension')
    parser.add_argument('--hidden-dim', type=int, default=256,
                       help='Hidden dimension for RNN')
    parser.add_argument('--num-layers', type=int, default=2,
                       help='Number of RNN layers')
    parser.add_argument('--num-heads', type=int, default=8,
                       help='Number of attention heads (for attention model)')
    parser.add_argument('--num-blocks', type=int, default=2,
                       help='Number of attention blocks')
    parser.add_argument('--dropout', type=float, default=0.2,
                       help='Dropout rate')
    parser.add_argument('--rnn-type', type=str, default='LSTM',
                       choices=['LSTM', 'GRU'],
                       help='Type of RNN to use')
    
    # Training arguments
    parser.add_argument('--data-type', type=str, default='sequential',
                       choices=['sequential', 'session'],
                       help='Type of data processing')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=256,
                       help='Batch size')
    parser.add_argument('--learning-rate', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-5,
                       help='Weight decay')
    parser.add_argument('--patience', type=int, default=10,
                       help='Early stopping patience')
    
    # System arguments
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cuda', 'cpu'],
                       help='Device to use')
    parser.add_argument('--num-workers', type=int, default=4,
                       help='Number of data loader workers')
    parser.add_argument('--save-dir', type=str, default='./models',
                       help='Directory to save models')
    
    # Experiment tracking
    parser.add_argument('--use-wandb', action='store_true',
                       help='Use Weights & Biases')
    parser.add_argument('--wandb-project', type=str, default='sequential-rec',
                       help='Wandb project name')
    parser.add_argument('--experiment-name', type=str, default=None,
                       help='Experiment name')
    
    # Evaluation
    parser.add_argument('--evaluate-ranking', action='store_true',
                       help='Evaluate ranking metrics')
    parser.add_argument('--top-k', type=int, nargs='+', default=[5, 10, 20],
                       help='Top-k values for evaluation')
    
    return parser.parse_args()


def create_model(model_type: str, config: dict, args) -> torch.nn.Module:
    """Create sequential model based on type"""
    if model_type == 'sequential':
        model = SequentialRecommender(
            num_items=config['num_items'],
            embedding_dim=args.embedding_dim,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            dropout=args.dropout,
            rnn_type=args.rnn_type
        )
    elif model_type == 'attention':
        model = AttentionalSequentialRecommender(
            num_items=config['num_items'],
            embedding_dim=args.embedding_dim,
            num_heads=args.num_heads,
            num_blocks=args.num_blocks,
            dropout=args.dropout,
            max_seq_len=args.max_seq_length
        )
    elif model_type == 'hierarchical':
        model = HierarchicalSequentialRecommender(
            num_items=config['num_items'],
            embedding_dim=args.embedding_dim,
            short_hidden_dim=args.hidden_dim // 2,
            long_hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            dropout=args.dropout
        )
    elif model_type == 'session':
        model = SessionBasedRecommender(
            num_items=config['num_items'],
            embedding_dim=args.embedding_dim,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            dropout=args.dropout,
            use_attention=True
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
    
    # Initialize wandb
    if args.use_wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.experiment_name,
            config=vars(args)
        )
    
    try:
        # Load data
        logger.info("Loading and preprocessing data...")
        data_loader = SequentialDataLoader(
            ratings_path=args.ratings_path,
            min_interactions=args.min_interactions,
            min_seq_length=args.min_seq_length,
            max_seq_length=args.max_seq_length
        )
        
        # Create data loaders
        train_loader, val_loader, test_loader = data_loader.create_data_loaders(
            data_type=args.data_type,
            batch_size=args.batch_size,
            num_workers=args.num_workers
        )
        
        # Get model config
        model_config = data_loader.get_model_config()
        logger.info(f"Model config: {model_config}")
        
        # Create model
        model = create_model(args.model_type, model_config, args)
        logger.info(f"Created {args.model_type} model with {sum(p.numel() for p in model.parameters())} parameters")
        
        # Create trainer
        trainer = SequentialTrainer(
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
        
        logger.info(f"Training completed. Best validation loss: {trainer.best_val_loss:.4f}")
        
        # Evaluate on test set
        logger.info("Evaluating on test set...")
        test_metrics = trainer.evaluate(test_loader, args.top_k)
        
        logger.info("Test Results:")
        for metric, value in test_metrics.items():
            logger.info(f"  {metric}: {value:.4f}")
        
        # Additional evaluation if requested
        if args.evaluate_ranking:
            logger.info("Evaluating additional ranking metrics...")
            evaluator = SequentialEvaluator(model, device)
            
            # Next-item prediction metrics
            ranking_metrics = evaluator.evaluate_next_item_prediction(test_loader, args.top_k)
            logger.info("Ranking Results:")
            for metric, value in ranking_metrics.items():
                logger.info(f"  {metric}: {value:.4f}")
            
            # Sequence prediction metrics
            seq_metrics = evaluator.evaluate_sequence_prediction(test_loader, future_steps=5)
            logger.info("Sequence Prediction Results:")
            for metric, value in seq_metrics.items():
                logger.info(f"  {metric}: {value:.4f}")
            
            # Combine all metrics
            all_metrics = {**test_metrics, **ranking_metrics, **seq_metrics}
        else:
            all_metrics = test_metrics
        
        # Log to wandb
        if args.use_wandb:
            wandb.log(all_metrics)
            wandb.log({"training_history": history})
        
        # Save encoders and metrics
        save_path = Path(args.save_dir)
        data_loader.save_encoders(save_path / 'encoders.pkl')
        
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