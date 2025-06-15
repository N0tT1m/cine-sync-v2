#!/usr/bin/env python3
"""
Training script for Two-Tower/Dual-Encoder models.
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
    TwoTowerModel, EnhancedTwoTowerModel, 
    MultiTaskTwoTowerModel, CollaborativeTwoTowerModel
)
from src.data_loader import TwoTowerDataLoader
from src.trainer import TwoTowerTrainer, TwoTowerEvaluator


def setup_logging(log_level: str = "INFO"):
    """Setup logging configuration"""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('two_tower_training.log')
        ]
    )


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train Two-Tower models')
    
    # Data arguments
    parser.add_argument('--ratings-path', type=str, default='../../ml-32m/ratings.csv',
                       help='Path to ratings CSV file')
    parser.add_argument('--movies-path', type=str, default='../../ml-32m/movies.csv',
                       help='Path to movies CSV file')
    parser.add_argument('--min-interactions', type=int, default=20,
                       help='Minimum interactions per user/item')
    
    # Model arguments
    parser.add_argument('--model-type', type=str, default='enhanced',
                       choices=['simple', 'enhanced', 'multitask', 'collaborative'],
                       help='Type of Two-Tower model')
    parser.add_argument('--embedding-dim', type=int, default=128,
                       help='Embedding dimension')
    parser.add_argument('--hidden-layers', type=int, nargs='+', default=[512, 256, 128],
                       help='Hidden layer sizes')
    parser.add_argument('--dropout', type=float, default=0.2,
                       help='Dropout rate')
    parser.add_argument('--use-batch-norm', action='store_true', default=True,
                       help='Use batch normalization')
    parser.add_argument('--use-cross-attention', action='store_true',
                       help='Use cross-attention (enhanced model)')
    
    # Multi-task arguments
    parser.add_argument('--task-heads', type=str, nargs='+', default=['rating', 'click'],
                       help='Task heads for multi-task model')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=1024,
                       help='Batch size')
    parser.add_argument('--learning-rate', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-5,
                       help='Weight decay')
    parser.add_argument('--patience', type=int, default=10,
                       help='Early stopping patience')
    
    # Data processing
    parser.add_argument('--negative-sampling', action='store_true', default=True,
                       help='Use negative sampling')
    parser.add_argument('--neg-ratio', type=float, default=2.0,
                       help='Negative to positive sample ratio')
    
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
    parser.add_argument('--wandb-project', type=str, default='two-tower-rec',
                       help='Wandb project name')
    parser.add_argument('--experiment-name', type=str, default=None,
                       help='Experiment name')
    
    # Evaluation
    parser.add_argument('--evaluate-retrieval', action='store_true',
                       help='Evaluate retrieval metrics')
    parser.add_argument('--top-k', type=int, nargs='+', default=[5, 10, 20, 50],
                       help='Top-k values for retrieval evaluation')
    parser.add_argument('--build-index', action='store_true', default=True,
                       help='Build FAISS index for efficient retrieval')
    
    return parser.parse_args()


def create_model(model_type: str, config: dict, args) -> torch.nn.Module:
    """Create Two-Tower model based on type"""
    if model_type == 'simple':
        # Simple model with flattened features
        user_feature_size = config['user_numerical_dim'] + len(config['user_categorical_dims'])
        item_feature_size = config['item_numerical_dim'] + len(config['item_categorical_dims'])
        
        model = TwoTowerModel(
            user_features_dim=user_feature_size,
            item_features_dim=item_feature_size,
            embedding_dim=args.embedding_dim,
            hidden_layers=args.hidden_layers,
            dropout=args.dropout,
            use_batch_norm=args.use_batch_norm
        )
        
    elif model_type == 'enhanced':
        model = EnhancedTwoTowerModel(
            user_categorical_dims=config['user_categorical_dims'],
            user_numerical_dim=config['user_numerical_dim'],
            item_categorical_dims=config['item_categorical_dims'],
            item_numerical_dim=config['item_numerical_dim'],
            embedding_dim=args.embedding_dim,
            hidden_layers=args.hidden_layers,
            dropout=args.dropout,
            use_cross_attention=args.use_cross_attention
        )
        
    elif model_type == 'multitask':
        # Simple feature dimensions for multi-task
        user_feature_size = config['user_numerical_dim'] + len(config['user_categorical_dims'])
        item_feature_size = config['item_numerical_dim'] + len(config['item_categorical_dims'])
        
        # Create task heads dictionary
        task_heads = {task: 1 for task in args.task_heads}
        
        model = MultiTaskTwoTowerModel(
            user_features_dim=user_feature_size,
            item_features_dim=item_feature_size,
            embedding_dim=args.embedding_dim,
            hidden_layers=args.hidden_layers,
            task_heads=task_heads,
            dropout=args.dropout
        )
        
    elif model_type == 'collaborative':
        user_feature_size = config['user_numerical_dim'] + len(config['user_categorical_dims'])
        item_feature_size = config['item_numerical_dim'] + len(config['item_categorical_dims'])
        
        model = CollaborativeTwoTowerModel(
            num_users=config['num_users'],
            num_items=config['num_items'],
            user_features_dim=user_feature_size,
            item_features_dim=item_feature_size,
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
        data_loader = TwoTowerDataLoader(
            ratings_path=args.ratings_path,
            movies_path=args.movies_path,
            min_interactions=args.min_interactions
        )
        
        # Create data loaders
        train_loader, val_loader, test_loader = data_loader.create_data_loaders(
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            model_type=args.model_type
        )
        
        # Get model config
        model_config = data_loader.get_model_config()
        logger.info(f"Model config: {model_config}")
        
        # Create model
        model = create_model(args.model_type, model_config, args)
        logger.info(f"Created {args.model_type} model with {sum(p.numel() for p in model.parameters())} parameters")
        
        # Create trainer
        trainer = TwoTowerTrainer(
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
        test_metrics = trainer.evaluate(test_loader)
        
        logger.info("Test Results:")
        for metric, value in test_metrics.items():
            logger.info(f"  {metric}: {value:.4f}")
        
        # Retrieval evaluation if requested
        if args.evaluate_retrieval:
            logger.info("Evaluating retrieval metrics...")
            evaluator = TwoTowerEvaluator(model, device)
            
            # Retrieval metrics
            retrieval_metrics = evaluator.evaluate_retrieval(
                test_loader, 
                k_values=args.top_k,
                build_index=args.build_index
            )
            
            logger.info("Retrieval Results:")
            for metric, value in retrieval_metrics.items():
                logger.info(f"  {metric}: {value:.4f}")
            
            # Embedding analysis
            embedding_analysis = evaluator.analyze_embeddings(test_loader)
            logger.info("Embedding Analysis:")
            for category, stats in embedding_analysis.items():
                logger.info(f"  {category}: {stats}")
            
            # Combine all metrics
            all_metrics = {**test_metrics, **retrieval_metrics}
            all_metrics['embedding_analysis'] = embedding_analysis
        else:
            all_metrics = test_metrics
        
        # Log to wandb
        if args.use_wandb:
            wandb.log(all_metrics)
            wandb.log({"training_history": history})
        
        # Save preprocessors and metrics
        save_path = Path(args.save_dir)
        data_loader.save_preprocessors(save_path / 'preprocessors.pkl')
        
        import json
        with open(save_path / 'final_metrics.json', 'w') as f:
            json.dump(all_metrics, f, indent=2, default=str)
        
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