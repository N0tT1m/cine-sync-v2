#!/usr/bin/env python3
# fix_movie_lookup.py - Rebuild movie lookup with only trained movies

import os
import pickle
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("fix-movie-lookup")

def fix_movie_lookup_table():
    """Fix the movie lookup table to only include movies used in training"""
    
    logger.info("🔧 Fixing movie lookup table...")
    
    try:
        # 1. Load the ID mappings (these are correct - only movies used in training)
        with open('models/id_mappings.pkl', 'rb') as f:
            mappings = pickle.load(f)
        
        # Get the movie IDs that were actually used in training
        trained_movie_ids = set(mappings['movie_id_to_idx'].keys())
        logger.info(f"Found {len(trained_movie_ids)} movies used in training")
        
        # 2. Load the original MovieLens movies data (the REAL movies)
        logger.info("Loading original MovieLens movies...")
        
        # Try different possible locations
        movielens_files = [
            'ml-32m/movies.csv',
            'data/ml-32m/movies.csv',
            'ml-25m/movies.csv', 
            'data/ml-25m/movies.csv'
        ]
        
        movies_df = None
        for file_path in movielens_files:
            if os.path.exists(file_path):
                logger.info(f"Loading from {file_path}")
                movies_df = pd.read_csv(file_path)
                if 'movieId' in movies_df.columns:
                    movies_df = movies_df.rename(columns={'movieId': 'media_id'})
                break
        
        if movies_df is None:
            logger.error("Could not find original MovieLens movies.csv file")
            return False
        
        logger.info(f"Loaded {len(movies_df)} original MovieLens movies")
        
        # Show sample of original real movies
        logger.info("Sample original movies:")
        for idx, row in movies_df.head(10).iterrows():
            logger.info(f"  {row['media_id']}: {row['title']}")
        
        # 3. Filter to only movies that were used in training
        trained_movies_df = movies_df[movies_df['media_id'].isin(trained_movie_ids)].copy()
        logger.info(f"Filtered to {len(trained_movies_df)} movies that were actually trained on")
        
        # 4. Add the movie indices for easier lookup
        trained_movies_df['movie_idx'] = trained_movies_df['media_id'].map(mappings['movie_id_to_idx'])
        
        # 5. Rebuild the movie lookup table (ONLY real movies used in training)
        logger.info("Rebuilding movie lookup table...")
        
        new_movie_lookup = {}
        for _, row in trained_movies_df.iterrows():
            movie_id = row['media_id']
            new_movie_lookup[movie_id] = {
                'title': row['title'],
                'genres': row['genres'],
                'movie_idx': row['movie_idx']
            }
        
        logger.info(f"Created new lookup table with {len(new_movie_lookup)} REAL movies")
        
        # Show sample of new lookup
        logger.info("Sample movies in new lookup:")
        sample_ids = list(new_movie_lookup.keys())[:10]
        for movie_id in sample_ids:
            info = new_movie_lookup[movie_id]
            logger.info(f"  {movie_id}: {info['title']} ({info['genres']})")
        
        # 6. Save the corrected lookup table
        logger.info("Saving corrected movie lookup table...")
        
        # Backup the old one first
        if os.path.exists('models/movie_lookup.pkl'):
            os.rename('models/movie_lookup.pkl', 'models/movie_lookup_backup.pkl')
            logger.info("Backed up old lookup table to movie_lookup_backup.pkl")
        
        # Save the new one
        with open('models/movie_lookup.pkl', 'wb') as f:
            pickle.dump(new_movie_lookup, f)
        
        # 7. Also update the movies_data.csv to match
        logger.info("Updating movies_data.csv...")
        
        # Add genre columns (for compatibility with existing code)
        all_genres = set()
        for genres in trained_movies_df['genres'].dropna():
            if pd.notnull(genres):
                all_genres.update(genres.split('|'))
        
        genres_list = sorted(list(all_genres))
        logger.info(f"Found {len(genres_list)} genres: {genres_list}")
        
        # One-hot encode genres
        for genre in genres_list:
            trained_movies_df[genre] = trained_movies_df['genres'].apply(
                lambda x: 1 if pd.notnull(x) and genre in x.split('|') else 0
            )
        
        # Backup and save
        if os.path.exists('models/movies_data.csv'):
            os.rename('models/movies_data.csv', 'models/movies_data_backup.csv')
            logger.info("Backed up old movies_data.csv")
        
        trained_movies_df.to_csv('models/movies_data.csv', index=False)
        
        # 8. Verify the fix
        logger.info("🔍 Verifying the fix...")
        
        logger.info(f"✅ Movie lookup now has {len(new_movie_lookup)} movies")
        logger.info(f"✅ All movies in lookup were used in training")
        logger.info(f"✅ All movies are real MovieLens movies")
        
        # Check alignment
        lookup_ids = set(new_movie_lookup.keys())
        mapping_ids = set(mappings['movie_id_to_idx'].keys())
        
        if lookup_ids == mapping_ids:
            logger.info("✅ Perfect alignment between lookup and mappings")
        else:
            logger.error("❌ Still misaligned!")
            return False
        
        logger.info("🎉 Movie lookup table successfully fixed!")
        logger.info("Now your recommendations will only include REAL movies that the model was trained on.")
        
        return True
    
    except Exception as e:
        logger.error(f"Error fixing movie lookup: {e}")
        return False
        
    except Exception as e:
        logger.error(f"Error fixing movie lookup: {e}")
        return False

def test_fixed_lookup():
    """Test the fixed lookup table"""
    logger.info("🧪 Testing fixed lookup table...")
    
    try:
        # Load the fixed lookup
        with open('models/movie_lookup.pkl', 'rb') as f:
            movie_lookup = pickle.load(f)
        
        # Load mappings
        with open('models/id_mappings.pkl', 'rb') as f:
            mappings = pickle.load(f)
        
        logger.info(f"Fixed lookup has {len(movie_lookup)} movies")
        logger.info(f"Mappings have {len(mappings['movie_id_to_idx'])} movies")
        
        # Show some real movies
        logger.info("Sample real movies now available for recommendation:")
        sample_ids = list(movie_lookup.keys())[:15]
        for i, movie_id in enumerate(sample_ids, 1):
            info = movie_lookup[movie_id]
            logger.info(f"  {i:2d}. {info['title']} (ID: {movie_id})")
        
        return True
        
    except Exception as e:
        logger.error(f"Error testing fixed lookup: {e}")
        return False

if __name__ == "__main__":
    logger.info("Starting movie lookup fix...")
    
    success = fix_movie_lookup_table()
    
    if success:
        logger.info("Testing the fix...")
        test_fixed_lookup()
        logger.info("\n🎉 ALL DONE! Your movie recommendations should now be REAL movies!")
        logger.info("Run: python inference.py to test recommendations")
    else:
        logger.error("❌ Fix failed. Check the error messages above.")