#!/usr/bin/env python3
"""
Test the trained Hybrid TV Model
Quick test script to load and test anime recommendations
"""

import torch
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add the hybrid TV model path
sys.path.append('hybrid_recommendation_tv/hybrid_recommendation')

def load_model_and_data():
    """Load the trained model and data"""
    model_dir = Path('hybrid_recommendation_tv/hybrid_recommendation/models')
    
    # Load model
    model_path = model_dir / 'best_hybrid_tv_model.pt'
    if not model_path.exists():
        print(f"❌ Model not found at {model_path}")
        return None, None, None, None
    
    print(f"✅ Loading model from {model_path}")
    checkpoint = torch.load(model_path, map_location='cpu')
    
    # Load metadata
    metadata_path = model_dir / 'hybrid_metadata.pkl'
    if metadata_path.exists():
        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)
        print(f"✅ Loaded metadata: {metadata.keys()}")
    else:
        metadata = {}
        print("⚠️ No metadata file found")
    
    # Load ID mappings
    lookup_path = model_dir / 'movie_lookup.pkl'
    if lookup_path.exists():
        with open(lookup_path, 'rb') as f:
            id_mappings = pickle.load(f)
        print(f"✅ Loaded ID mappings: {len(id_mappings.get('idx_to_movie_id', {}))} anime items")
    else:
        id_mappings = {}
        print("⚠️ No ID mappings found")
    
    # Load anime metadata
    anime_path = Path('tv/misc/animes.csv')
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
        # Try to import the model class
        from train_with_wandb import HybridRecommender
        
        # Get model parameters from checkpoint or metadata
        config = checkpoint.get('config', {})
        model_metadata = checkpoint.get('metadata', {})
        
        # Extract model parameters
        num_users = model_metadata.get('num_users', config.get('num_users', 1000))
        num_items = model_metadata.get('num_items', config.get('num_items', 1000))
        embedding_dim = config.get('embedding_dim', 64)
        hidden_dim = config.get('hidden_dim', 128)
        dropout = config.get('dropout', 0.3)
        
        print(f"📊 Model config: users={num_users}, items={num_items}, embed_dim={embedding_dim}")
        
        # Create model
        model = HybridRecommender(
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
        return None, None, None

def get_random_user_recommendations(model, num_users, num_items, id_mappings, anime_df, num_recs=10):
    """Get recommendations for a random user"""
    
    # Pick a random user
    user_id = np.random.randint(0, min(num_users, 100))  # Limit to first 100 users
    
    print(f"\n🎯 Getting recommendations for User {user_id}:")
    
    # Generate scores for all items for this user
    with torch.no_grad():
        user_tensor = torch.LongTensor([user_id] * num_items)
        item_tensor = torch.LongTensor(list(range(num_items)))
        
        # Get predictions
        scores = model(user_tensor, item_tensor)
        
        # Get top recommendations
        top_indices = torch.topk(scores, min(num_recs, len(scores))).indices.numpy()
        top_scores = torch.topk(scores, min(num_recs, len(scores))).values.numpy()
    
    print(f"\n🏆 Top {len(top_indices)} Anime Recommendations:")
    print("-" * 80)
    
    for i, (idx, score) in enumerate(zip(top_indices, top_scores)):
        # Get anime info
        anime_info = get_anime_info(idx, id_mappings, anime_df)
        
        print(f"{i+1:2d}. Score: {score:.3f} | {anime_info}")
    
    return top_indices, top_scores

def get_anime_info(item_idx, id_mappings, anime_df):
    """Get anime information from index"""
    try:
        # Get original anime UID from mappings
        if 'idx_to_movie_id' in id_mappings:
            anime_uid = id_mappings['idx_to_movie_id'].get(str(item_idx), f"anime_{item_idx}")
        else:
            anime_uid = f"anime_{item_idx}"
        
        # Get anime details from metadata
        if anime_df is not None:
            anime_row = anime_df[anime_df['uid'] == int(anime_uid)] if anime_uid.isdigit() else None
            
            if anime_row is not None and not anime_row.empty:
                title = anime_row.iloc[0]['title']
                genre = anime_row.iloc[0].get('genre', 'Unknown')
                score = anime_row.iloc[0].get('score', 'N/A')
                episodes = anime_row.iloc[0].get('episodes', 'N/A')
                
                return f"{title} | Genre: {genre} | Score: {score} | Episodes: {episodes}"
        
        return f"Anime ID: {anime_uid} (idx: {item_idx})"
        
    except Exception as e:
        return f"Anime idx: {item_idx} (error: {e})"

def test_specific_predictions(model, num_users, num_items):
    """Test specific user-item predictions"""
    print(f"\n🧪 Testing specific predictions:")
    print("-" * 50)
    
    # Test a few user-item pairs
    test_pairs = [
        (0, 0), (0, 1), (0, 2),
        (1, 0), (1, 1), (1, 2),
        (5, 10), (10, 20)
    ]
    
    with torch.no_grad():
        for user_id, item_id in test_pairs:
            if user_id < num_users and item_id < num_items:
                user_tensor = torch.LongTensor([user_id])
                item_tensor = torch.LongTensor([item_id])
                
                score = model(user_tensor, item_tensor).item()
                print(f"User {user_id:2d} -> Anime {item_id:2d}: {score:.3f}")

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
    
    print(f"\n📈 Model Info:")
    print(f"   Users: {num_users}")
    print(f"   Items: {num_items}")
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test specific predictions
    test_specific_predictions(model, num_users, num_items)
    
    # Get recommendations for random users
    try:
        for i in range(3):  # Test 3 random users
            get_random_user_recommendations(model, num_users, num_items, id_mappings, anime_df, num_recs=5)
            print()
    except Exception as e:
        print(f"❌ Error generating recommendations: {e}")
    
    print("✅ Testing completed!")

if __name__ == "__main__":
    main()