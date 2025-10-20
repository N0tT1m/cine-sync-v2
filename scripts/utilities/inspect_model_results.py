#!/usr/bin/env python3
"""
Simple inspection of the trained model results and data
"""

import json
import pickle
import pandas as pd
from pathlib import Path

def inspect_training_results():
    """Inspect what the model learned"""
    print("🎬 Hybrid TV Model Training Results")
    print("=" * 50)
    
    model_dir = Path('hybrid_recommendation_tv/hybrid_recommendation/models')
    
    # Check final metrics
    metrics_path = model_dir / 'final_metrics.json'
    if metrics_path.exists():
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)
        
        print("📊 Final Training Metrics:")
        for key, value in metrics.items():
            if isinstance(value, float):
                print(f"   {key}: {value:.4f}")
            else:
                print(f"   {key}: {value}")
        print()
    
    # Check training history
    history_path = model_dir / 'training_history.pkl'
    if history_path.exists():
        with open(history_path, 'rb') as f:
            history = pickle.load(f)
        
        print("📈 Training Progress:")
        if 'train_losses' in history:
            train_losses = history['train_losses']
            print(f"   Training epochs: {len(train_losses)}")
            print(f"   Initial loss: {train_losses[0]:.4f}")
            print(f"   Final loss: {train_losses[-1]:.4f}")
            print(f"   Loss reduction: {((train_losses[0] - train_losses[-1]) / train_losses[0] * 100):.1f}%")
        
        if 'val_losses' in history:
            val_losses = history['val_losses']
            print(f"   Best validation loss: {min(val_losses):.4f}")
        print()
    
    # Check anime data
    anime_path = Path('tv/misc/animes.csv')
    if anime_path.exists():
        anime_df = pd.read_csv(anime_path)
        print("📺 Anime Dataset Info:")
        print(f"   Total anime: {len(anime_df)}")
        print(f"   Columns: {list(anime_df.columns)}")
        
        # Show top rated anime
        if 'score' in anime_df.columns and 'title' in anime_df.columns:
            top_anime = anime_df.nlargest(10, 'score')[['title', 'score', 'genre']].head()
            print("\n🏆 Top Rated Anime in Dataset:")
            for idx, row in top_anime.iterrows():
                print(f"   {row['title']} - Score: {row['score']} - {row.get('genre', 'Unknown')}")
        print()
    
    # Check reviews data
    reviews_path = Path('tv/misc/reviews.csv')
    if reviews_path.exists():
        reviews_df = pd.read_csv(reviews_path)
        print("📝 Reviews Dataset Info:")
        print(f"   Total reviews: {len(reviews_df)}")
        print(f"   Unique users: {reviews_df['uid'].nunique()}")
        print(f"   Unique anime: {reviews_df['anime_uid'].nunique()}")
        print(f"   Average score: {reviews_df['score'].mean():.2f}")
        print(f"   Score range: {reviews_df['score'].min()} - {reviews_df['score'].max()}")
        print()
    
    # Check model mappings
    lookup_path = model_dir / 'movie_lookup.pkl'
    if lookup_path.exists():
        with open(lookup_path, 'rb') as f:
            mappings = pickle.load(f)
        
        print("🔗 Model ID Mappings:")
        print(f"   Mapped items: {len(mappings.get('idx_to_movie_id', {}))}")
        print(f"   Mapping keys: {list(mappings.keys())}")
        
        # Show a few mappings
        if 'idx_to_movie_id' in mappings:
            sample_mappings = dict(list(mappings['idx_to_movie_id'].items())[:5])
            print("   Sample mappings (idx -> anime_id):")
            for idx, anime_id in sample_mappings.items():
                print(f"     {idx} -> {anime_id}")
        print()

def show_model_architecture():
    """Show model architecture details"""
    model_dir = Path('hybrid_recommendation_tv/hybrid_recommendation/models')
    metadata_path = model_dir / 'hybrid_metadata.pkl'
    
    if metadata_path.exists():
        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)
        
        print("🏗️ Model Architecture:")
        for key, value in metadata.items():
            if isinstance(value, (int, float, str)):
                print(f"   {key}: {value}")
        print()

def main():
    """Main inspection function"""
    inspect_training_results()
    show_model_architecture()
    
    print("✅ Inspection completed!")
    print("\nTo test the model interactively, run:")
    print("   python test_hybrid_tv_model.py")

if __name__ == "__main__":
    main()