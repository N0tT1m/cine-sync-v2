#!/usr/bin/env python3
"""
Universal Training Script for Advanced Recommendation Models
RTX 4090 + Ryzen 9 3900X BEAST MODE optimized for maximum performance
"""

import argparse
import logging
import torch
import sys
import os
from pathlib import Path
import time
import json

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

# Import RTX 4090 BEAST MODE optimizations
sys.path.append(str(Path(__file__).parent.parent / 'neural_collaborative_filtering'))
from performance_config import PerformanceOptimizer, apply_rtx4090_optimizations

# Import all advanced models
from enhanced_two_tower import UltimateTwoTowerModel, TwoTowerTrainer
from transformer_recommender import SASRec, EnhancedSASRec, SASRecTrainer
from sentence_bert_two_tower import SentenceBERTTwoTowerModel, SentenceBERTTwoTowerTrainer
from graph_neural_network import UltraModernGraphTransformer, LightGCN, GCNTrainer
from variational_autoencoder import MultVAE, EnhancedMultVAE, VAETrainer
from t5_hybrid_recommender import T5HybridRecommender, T5HybridTrainer
from bert4rec_recommender import BERT4Rec, EnhancedBERT4Rec, BERT4RecTrainer
from graphsage_recommender import GraphSAGERecommender, GraphSAGETrainer


def setup_logging():
    """Setup comprehensive logging for advanced model training"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('advanced_training.log')
        ]
    )
    

def parse_args():
    """Parse command line arguments for advanced model training"""
    parser = argparse.ArgumentParser(description='Train Advanced Recommendation Models with RTX 4090 BEAST MODE')
    
    # Model selection
    parser.add_argument('--model-type', type=str, required=True,
                       choices=['enhanced_two_tower', 'sasrec', 'enhanced_sasrec', 'sentence_bert',
                               'graph_transformer', 'lightgcn', 'multvae', 'enhanced_multvae',
                               't5_hybrid', 'bert4rec', 'enhanced_bert4rec', 'graphsage'],
                       help='Advanced model type to train')
    
    # Data arguments
    parser.add_argument('--ratings-path', type=str, default='../movies/cinesync/ml-32m/ratings.csv',
                       help='Path to ratings data')
    parser.add_argument('--movies-path', type=str, default='../movies/cinesync/ml-32m/movies.csv',
                       help='Path to movies metadata')
    
    # RTX 4090 BEAST MODE arguments
    parser.add_argument('--batch-size', type=int, default=16384,
                       help='Batch size (RTX 4090 BEAST MODE - massive batches)')
    parser.add_argument('--num-workers', type=int, default=24,
                       help='Data loader workers (Ryzen 9 3900X - 24 threads)')
    parser.add_argument('--embedding-dim', type=int, default=256,
                       help='Embedding dimension (BEAST MODE - large embeddings)')
    parser.add_argument('--hidden-dims', type=int, nargs='+', default=[1024, 512, 256],
                       help='Hidden layer dimensions (BEAST MODE - deep networks)')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--learning-rate', type=float, default=0.003,
                       help='Learning rate (higher for large batches)')
    parser.add_argument('--weight-decay', type=float, default=1e-6,
                       help='Weight decay (lower for large models)')
    parser.add_argument('--patience', type=int, default=10,
                       help='Early stopping patience')
    
    # System arguments
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cuda', 'cpu'],
                       help='Device to use')
    parser.add_argument('--save-dir', type=str, default='./advanced_models',
                       help='Directory to save models')
    
    return parser.parse_args()


def create_model(model_type: str, config: dict, args):
    """Create advanced model based on type"""
    if model_type == 'enhanced_two_tower':
        model = UltimateTwoTowerModel(
            user_vocab_size=config['num_users'],
            item_vocab_size=config['num_items'],
            embedding_dim=args.embedding_dim,
            hidden_dims=args.hidden_dims,
            num_heads=8,
            num_experts=4
        )
        trainer_class = TwoTowerTrainer
        
    elif model_type == 'sasrec':
        model = SASRec(
            item_num=config['num_items'],
            embedding_dim=args.embedding_dim,
            hidden_size=args.hidden_dims[0],
            num_heads=8,
            num_blocks=4,
            dropout_rate=0.1
        )
        trainer_class = SASRecTrainer
        
    elif model_type == 'enhanced_sasrec':
        model = EnhancedSASRec(
            item_num=config['num_items'],
            embedding_dim=args.embedding_dim,
            hidden_size=args.hidden_dims[0],
            num_heads=16,  # More heads for BEAST MODE
            num_blocks=6,  # Deeper for better quality
            dropout_rate=0.1
        )
        trainer_class = SASRecTrainer
        
    elif model_type == 'sentence_bert':
        model = SentenceBERTTwoTowerModel(
            num_users=config['num_users'],
            num_items=config['num_items'],
            embedding_dim=args.embedding_dim,
            hidden_dims=args.hidden_dims
        )
        trainer_class = SentenceBERTTwoTowerTrainer
        
    elif model_type == 'graph_transformer':
        model = UltraModernGraphTransformer(
            num_users=config['num_users'],
            num_items=config['num_items'],
            embedding_dim=args.embedding_dim,
            num_heads=16,  # BEAST MODE attention
            num_layers=6   # Deep graph network
        )
        trainer_class = GCNTrainer
        
    elif model_type == 'lightgcn':
        model = LightGCN(
            num_users=config['num_users'],
            num_items=config['num_items'],
            embedding_dim=args.embedding_dim,
            num_layers=4
        )
        trainer_class = GCNTrainer
        
    elif model_type == 'multvae':
        model = MultVAE(
            num_items=config['num_items'],
            hidden_dims=args.hidden_dims,
            latent_dim=args.embedding_dim
        )
        trainer_class = VAETrainer
        
    elif model_type == 'enhanced_multvae':
        model = EnhancedMultVAE(
            num_items=config['num_items'],
            hidden_dims=args.hidden_dims + [args.embedding_dim * 2],  # Larger for BEAST MODE
            latent_dim=args.embedding_dim,
            num_components=8  # More mixture components
        )
        trainer_class = VAETrainer
        
    elif model_type == 't5_hybrid':
        model = T5HybridRecommender(
            num_users=config['num_users'],
            num_items=config['num_items'],
            embedding_dim=args.embedding_dim,
            hidden_dims=args.hidden_dims
        )
        trainer_class = T5HybridTrainer
        
    elif model_type == 'bert4rec':
        model = BERT4Rec(
            item_num=config['num_items'],
            hidden_size=args.hidden_dims[0],
            num_heads=8,
            num_layers=4,
            dropout_rate=0.1
        )
        trainer_class = BERT4RecTrainer
        
    elif model_type == 'enhanced_bert4rec':
        model = EnhancedBERT4Rec(
            item_num=config['num_items'],
            hidden_size=args.hidden_dims[0],
            num_heads=16,  # BEAST MODE heads
            num_layers=8,  # Deeper for better quality
            dropout_rate=0.1
        )
        trainer_class = BERT4RecTrainer
        
    elif model_type == 'graphsage':
        model = GraphSAGERecommender(
            num_users=config['num_users'],
            num_items=config['num_items'],
            embedding_dim=args.embedding_dim,
            hidden_dims=args.hidden_dims,
            num_layers=4
        )
        trainer_class = GraphSAGETrainer
        
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return model, trainer_class


def main():
    args = parse_args()
    setup_logging()
    logger = logging.getLogger(__name__)
    
    # Set device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    logger.info(f"Using device: {device}")
    
    # 🔥🔥🔥 ACTIVATE ADVANCED MODELS BEAST MODE 🔥🔥🔥
    apply_rtx4090_optimizations()
    
    if device.type == 'cuda':
        logger.info("🚀🚀🚀 RTX 4090 ADVANCED MODELS BEAST MODE ACTIVATED! 🚀🚀🚀")
        PerformanceOptimizer.setup_maximum_performance()
        
        # Get performance config
        perf_config = PerformanceOptimizer.get_rtx4090_config()
        
        # Override with BEAST MODE settings
        if args.batch_size == 16384:
            args.batch_size = perf_config['batch_size']
            logger.info(f"🔥 BEAST MODE batch size: {args.batch_size}")
        
        if args.num_workers == 24:
            args.num_workers = perf_config['num_workers']
            logger.info(f"💪 BEAST MODE workers: {args.num_workers}")
    
    try:
        # Load data (simplified for this example - you'll need proper data loading)
        logger.info("Loading data for advanced model training...")
        
        # Placeholder config - replace with actual data loading
        config = {
            'num_users': 10000,
            'num_items': 50000
        }
        
        # Create model
        model, trainer_class = create_model(args.model_type, config, args)
        logger.info(f"Created {args.model_type} model with {sum(p.numel() for p in model.parameters())} parameters")
        
        # Apply BEAST MODE model optimizations
        model = model.to(device)
        model = PerformanceOptimizer.optimize_model_for_speed(model)
        
        # Find optimal batch size for this model
        if device.type == 'cuda':
            optimal_batch = PerformanceOptimizer.get_optimal_batch_size_rtx4090(model, device)
            if optimal_batch > args.batch_size:
                logger.info(f"🚀 Upgrading {args.model_type} batch size from {args.batch_size} to {optimal_batch}")
                args.batch_size = optimal_batch
        
        # Create trainer
        trainer = trainer_class(
            model=model,
            device=device,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay
        )
        
        logger.info(f"🔥 {args.model_type.upper()} BEAST MODE TRAINING READY! 🔥")
        logger.info(f"   Model: {args.model_type}")
        logger.info(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
        logger.info(f"   Batch Size: {args.batch_size}")
        logger.info(f"   Workers: {args.num_workers}")
        logger.info(f"   Embedding Dim: {args.embedding_dim}")
        
        # Here you would add the actual training loop
        # For now, we just show the setup is complete
        
        logger.info("🚀 ADVANCED MODEL BEAST MODE SETUP COMPLETE! 🚀")
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()