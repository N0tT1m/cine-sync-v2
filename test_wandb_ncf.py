#!/usr/bin/env python3
"""
Quick test script to verify wandb logging works with NCF training
"""
import sys
import os
sys.path.append('neural_collaborative_filtering/src')

import wandb
from data_loader import NCFDataLoader
from model import SimpleNCF
from trainer import NCFTrainer
import torch

def test_wandb_logging():
    """Test NCF training with wandb logging for a few epochs"""
    
    # Initialize wandb
    wandb.init(
        project="ncf-test",
        name="test-epoch-logging",
        config={
            "epochs": 3,
            "batch_size": 32,
            "learning_rate": 0.001,
            "test": True
        }
    )
    
    print("✓ Wandb initialized")
    
    # Quick data loading with minimal processing
    data_loader = NCFDataLoader(
        ratings_path="data/ratings.csv",
        movies_path="data/movies.csv",
        min_ratings_per_user=5,
        min_ratings_per_item=5
    )
    
    print("✓ Data loader created")
    
    # Get small data loaders for quick test
    train_loader, val_loader, test_loader = data_loader.get_data_loaders(
        batch_size=32,
        num_workers=0  # No multiprocessing for test
    )
    
    print("✓ Data loaders ready")
    
    # Create simple model
    model_config = data_loader.get_model_config()
    model = SimpleNCF(
        num_users=model_config['num_users'],
        num_items=model_config['num_items'],
        embedding_dim=32,  # Small for fast test
        hidden_dim=64
    )
    
    print("✓ Model created")
    
    # Create trainer
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    trainer = NCFTrainer(
        model=model,
        device=device,
        learning_rate=0.001
    )
    
    print(f"✓ Trainer ready on {device}")
    
    # Train for just 3 epochs to test logging
    print("Starting training...")
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=3,
        patience=10
    )
    
    print("✓ Training completed")
    
    # Log final summary
    wandb.log({
        "test_completed": True,
        "epochs_trained": len(history['train_losses'])
    })
    
    wandb.finish()
    print("✓ Test completed successfully!")
    print("Check your W&B dashboard to see epoch-by-epoch metrics")

if __name__ == "__main__":
    test_wandb_logging()