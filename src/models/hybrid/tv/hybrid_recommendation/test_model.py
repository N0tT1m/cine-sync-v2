#!/usr/bin/env python3
"""
Test the trained Hybrid TV Model
Run this from inside the hybrid_recommendation directory
"""

import torch
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Import the model class
from train_with_wandb import HybridRecommenderModel

def load_model_and_data():
    """Load the trained model and data"""
    model_dir = Path('models')
    
    # Check for different possible model names
    possible_model_files = [
        'best_hybrid_tv_model.pt',
        'best_hybrid_model.pt', 
        'recommendation_model.pt',
        'tv_show_model.pt'
    ]
    
    model_path = None
    for filename in possible_model_files:
        path = model_dir / filename
        if path.exists():
            model_path = path
            break
    
    if model_path is None:
        print(f"❌ No model file found in {model_dir}")
        return None, None, None, None
    
    print(f"✅ Loading model from {model_path}")
    # Fix for PyTorch 2.6 weights_only security change
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    
    # Load metadata
    metadata_path = model_dir / 'hybrid_metadata.pkl'
    if metadata_path.exists():
        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)
        print(f"✅ Loaded metadata with keys: {list(metadata.keys())}")
    else:
        metadata = {}
        print("⚠️ No metadata file found")
    
    # Load ID mappings
    lookup_path = model_dir / 'movie_lookup.pkl'
    if lookup_path.exists():
        with open(lookup_path, 'rb') as f:
            id_mappings = pickle.load(f)
        print(f"✅ Loaded ID mappings")
    else:
        id_mappings = {}
        print("⚠️ No ID mappings found")
    
    # Load anime metadata
    anime_path = Path('../../tv/misc/animes.csv')
    if anime_path.exists():
        anime_df = pd.read_csv(anime_path)
        print(f"✅ Loaded anime metadata: {len(anime_df)} anime entries")
    else:
        anime_df = None
        print("⚠️ No anime metadata found")
    
    return checkpoint, metadata, id_mappings, anime_df

def create_model_from_checkpoint(checkpoint, metadata):
    """Recreate the model from checkpoint"""
    try:
        # Get model parameters from checkpoint or metadata
        config = checkpoint.get('config', {})
        model_metadata = checkpoint.get('metadata', metadata)
        
        # Extract model parameters
        num_users = model_metadata.get('num_users', config.get('num_users', 1000))
        num_items = model_metadata.get('num_movies', model_metadata.get('num_items', config.get('num_items', 1000)))
        embedding_dim = config.get('embedding_dim', 64)
        hidden_dim = config.get('hidden_dim', 128)
        dropout = config.get('dropout', 0.2)
        
        print(f"📊 Model config: users={num_users}, items={num_items}, embed_dim={embedding_dim}")
        
        # Create model
        model = HybridRecommenderModel(
            num_users=num_users,
            num_items=num_items,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            dropout=dropout
        )
        
        # Load state dict
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        print("✅ Model loaded successfully!")
        return model, num_users, num_items
        
    except Exception as e:
        print(f"❌ Error creating model: {e}")
        print(f"Available checkpoint keys: {list(checkpoint.keys())}")
        if 'config' in checkpoint:
            print(f"Config keys: {list(checkpoint['config'].keys())}")
        return None, None, None

def test_model_predictions(model, num_users, num_items, id_mappings, anime_df):
    """Test model predictions"""
    print(f"\n🧪 Testing Model Predictions:")
    print("-" * 50)
    
    # Test a few specific user-item pairs
    test_pairs = [
        (0, 0), (0, 1), (0, 2), (0, 10),
        (1, 0), (1, 5), (1, 10),
        (5, 1), (5, 20), (10, 15)
    ]
    
    print("Specific predictions:")
    with torch.no_grad():
        for user_id, item_id in test_pairs:
            if user_id < num_users and item_id < num_items:
                user_tensor = torch.LongTensor([user_id])
                item_tensor = torch.LongTensor([item_id])
                
                score = model(user_tensor, item_tensor).item()
                print(f"   User {user_id:2d} -> Anime {item_id:2d}: {score:.3f}")
    
    # Generate recommendations for a user
    print(f"\n🎯 Top 10 Recommendations for User 0:")
    print("-" * 50)
    
    user_id = 0
    with torch.no_grad():
        # Get predictions for all items for this user
        user_tensor = torch.LongTensor([user_id] * min(num_items, 1000))  # Limit to prevent memory issues
        item_tensor = torch.LongTensor(list(range(min(num_items, 1000))))
        
        scores = model(user_tensor, item_tensor)
        
        # Get top 10
        top_k = min(10, len(scores))
        top_indices = torch.topk(scores, top_k).indices.numpy()
        top_scores = torch.topk(scores, top_k).values.numpy()
        
        for i, (idx, score) in enumerate(zip(top_indices, top_scores)):
            anime_info = get_anime_info(idx, id_mappings, anime_df)
            print(f"{i+1:2d}. Score: {score:.3f} | {anime_info}")

def get_anime_info(item_idx, id_mappings, anime_df):
    """Get anime information from index"""
    try:
        # Try to get original anime UID from mappings
        if 'idx_to_movie_id' in id_mappings:
            anime_uid = id_mappings['idx_to_movie_id'].get(str(item_idx), None)
        else:
            anime_uid = None
        
        # Get anime details from metadata
        if anime_df is not None and anime_uid:
            try:
                anime_uid_int = int(anime_uid)
                anime_row = anime_df[anime_df['uid'] == anime_uid_int]
                
                if not anime_row.empty:
                    row = anime_row.iloc[0]
                    title = row['title'][:40]  # Truncate long titles
                    genre = str(row.get('genre', 'Unknown'))[:20]
                    score = row.get('score', 'N/A')
                    episodes = row.get('episodes', 'N/A')
                    
                    return f"{title:<40} | {genre:<20} | Score: {score} | Eps: {episodes}"
            except (ValueError, TypeError):
                pass
        
        return f"Anime item index: {item_idx} (UID: {anime_uid})"
        
    except Exception as e:
        return f"Anime idx: {item_idx} (error: {e})"

def show_model_info(model, checkpoint, metadata):
    """Show model information"""
    print(f"\n📈 Model Information:")
    print(f"   Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"   Model size: ~{sum(p.numel() * p.element_size() for p in model.parameters()) / (1024*1024):.1f} MB")
    
    if 'val_loss' in checkpoint:
        print(f"   Validation loss: {checkpoint['val_loss']:.4f}")
    if 'val_rmse' in checkpoint:
        print(f"   Validation RMSE: {checkpoint['val_rmse']:.4f}")
    if 'epoch' in checkpoint:
        print(f"   Training epochs: {checkpoint['epoch'] + 1}")
    
    print(f"   Training on: {metadata.get('num_ratings', 'Unknown')} ratings")
    print(f"   Users: {metadata.get('num_users', 'Unknown')}")
    print(f"   Items: {metadata.get('num_movies', metadata.get('num_items', 'Unknown'))}")

def main():
    """Main test function"""
    print("🎬 Testing Hybrid TV Model (Anime Recommendations)")
    print("=" * 60)
    
    # Load model and data
    checkpoint, metadata, id_mappings, anime_df = load_model_and_data()
    
    if checkpoint is None:
        print("❌ Could not load model. Make sure training completed successfully.")
        return
    
    # Create model
    model, num_users, num_items = create_model_from_checkpoint(checkpoint, metadata)
    
    if model is None:
        print("❌ Could not create model from checkpoint.")
        return
    
    # Show model info
    show_model_info(model, checkpoint, metadata)
    
    # Test predictions
    test_model_predictions(model, num_users, num_items, id_mappings, anime_df)
    
    print(f"\n✅ Model testing completed!")
    print(f"🚀 Your hybrid TV model is working and ready for anime recommendations!")

if __name__ == "__main__":
    main()