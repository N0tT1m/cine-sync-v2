#!/usr/bin/env python3
# debug_mappings.py - Debug script to verify data mappings and movie validity

import os
import pickle
import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("debug-mappings")

def debug_model_mappings(model_dir='models'):
    """Debug the model's ID mappings and movie data"""
    
    print("="*80)
    print("DEBUGGING MOVIE RECOMMENDATION MAPPINGS")
    print("="*80)
    
    try:
        # 1. Check if all required files exist
        required_files = [
            'id_mappings.pkl',
            'movie_lookup.pkl', 
            'movies_data.csv',
            'rating_scaler.pkl',
            'best_model.pt'
        ]
        
        print("\n1. Checking required files:")
        for file in required_files:
            file_path = os.path.join(model_dir, file)
            exists = os.path.exists(file_path)
            print(f"   {file}: {'✅' if exists else '❌'}")
            if not exists:
                print(f"   ERROR: Missing {file}")
                return False
        
        # 2. Load and examine ID mappings
        print("\n2. Examining ID mappings:")
        with open(os.path.join(model_dir, 'id_mappings.pkl'), 'rb') as f:
            mappings = pickle.load(f)
        
        print(f"   Number of users: {mappings['num_users']}")
        print(f"   Number of movies: {mappings['num_movies']}")
        print(f"   User ID range: {min(mappings['user_id_to_idx'].keys())} to {max(mappings['user_id_to_idx'].keys())}")
        print(f"   Movie ID range: {min(mappings['movie_id_to_idx'].keys())} to {max(mappings['movie_id_to_idx'].keys())}")
        
        # Show sample mappings
        sample_users = list(mappings['user_id_to_idx'].items())[:5]
        sample_movies = list(mappings['movie_id_to_idx'].items())[:5]
        
        print(f"   Sample user mappings: {sample_users}")
        print(f"   Sample movie mappings: {sample_movies}")
        
        # 3. Load and examine movie lookup
        print("\n3. Examining movie lookup table:")
        with open(os.path.join(model_dir, 'movie_lookup.pkl'), 'rb') as f:
            movie_lookup = pickle.load(f)
        
        print(f"   Movies in lookup table: {len(movie_lookup)}")
        
        # Show sample movies
        sample_movie_ids = list(movie_lookup.keys())[:10]
        print(f"   Sample movies:")
        for movie_id in sample_movie_ids:
            movie_info = movie_lookup[movie_id]
            title = movie_info.get('title', 'NO TITLE')
            genres = movie_info.get('genres', 'NO GENRES')
            print(f"     ID {movie_id}: {title} ({genres})")
        
        # 4. Load and examine movies dataframe
        print("\n4. Examining movies dataframe:")
        movies_df = pd.read_csv(os.path.join(model_dir, 'movies_data.csv'))
        print(f"   Rows in movies_df: {len(movies_df)}")
        print(f"   Columns: {list(movies_df.columns)}")
        
        # Check for required columns
        required_cols = ['media_id', 'title', 'genres']
        for col in required_cols:
            if col in movies_df.columns:
                print(f"   ✅ Column '{col}' exists")
            else:
                print(f"   ❌ Column '{col}' missing")
        
        # Show sample rows
        print(f"   Sample movies from dataframe:")
        for idx, row in movies_df.head(5).iterrows():
            print(f"     {row.get('media_id', 'NO_ID')}: {row.get('title', 'NO_TITLE')}")
        
        # 5. Cross-validate mappings
        print("\n5. Cross-validating mappings:")
        
        # Check if all movies in lookup are in mappings
        lookup_movie_ids = set(movie_lookup.keys())
        mapping_movie_ids = set(mappings['movie_id_to_idx'].keys())
        
        print(f"   Movies in lookup: {len(lookup_movie_ids)}")
        print(f"   Movies in mappings: {len(mapping_movie_ids)}")
        
        missing_in_mappings = lookup_movie_ids - mapping_movie_ids
        missing_in_lookup = mapping_movie_ids - lookup_movie_ids
        
        if missing_in_mappings:
            print(f"   ❌ Movies in lookup but not in mappings: {len(missing_in_mappings)}")
            print(f"      Examples: {list(missing_in_mappings)[:10]}")
        else:
            print(f"   ✅ All lookup movies have mappings")
        
        if missing_in_lookup:
            print(f"   ❌ Movies in mappings but not in lookup: {len(missing_in_lookup)}")
            print(f"      Examples: {list(missing_in_lookup)[:10]}")
        else:
            print(f"   ✅ All mapped movies have lookup entries")
        
        # 6. Check movies dataframe alignment
        print("\n6. Checking dataframe alignment:")
        df_movie_ids = set(movies_df['media_id'].unique())
        
        print(f"   Unique movies in dataframe: {len(df_movie_ids)}")
        
        df_not_in_mappings = df_movie_ids - mapping_movie_ids
        mapping_not_in_df = mapping_movie_ids - df_movie_ids
        
        if df_not_in_mappings:
            print(f"   ❌ Movies in dataframe but not mapped: {len(df_not_in_mappings)}")
        else:
            print(f"   ✅ All dataframe movies are mapped")
        
        if mapping_not_in_df:
            print(f"   ❌ Mapped movies not in dataframe: {len(mapping_not_in_df)}")
        else:
            print(f"   ✅ All mapped movies in dataframe")
        
        # 7. Check for fake vs real movie titles
        print("\n7. Analyzing movie titles for validity:")
        
        real_movie_patterns = [
            'toy story', 'star wars', 'batman', 'superman', 'spider-man',
            'harry potter', 'lord of the rings', 'matrix', 'terminator',
            'jurassic park', 'indiana jones', 'back to the future'
        ]
        
        fake_patterns = [
            'movie_', 'film_', 'title_', 'unknown', 'test_', 'sample_'
        ]
        
        titles = [movie_lookup[mid].get('title', '').lower() for mid in list(movie_lookup.keys())[:100]]
        
        real_count = sum(1 for title in titles if any(pattern in title for pattern in real_movie_patterns))
        fake_count = sum(1 for title in titles if any(pattern in title for pattern in fake_patterns))
        
        print(f"   Sample of 100 movie titles:")
        print(f"   Likely real movies: {real_count}")
        print(f"   Likely fake/generated titles: {fake_count}")
        print(f"   Other titles: {100 - real_count - fake_count}")
        
        # Show some actual titles
        print(f"   Sample titles:")
        for i, title in enumerate(titles[:10]):
            print(f"     {i+1}. {title}")
        
        # 8. Final diagnosis
        print("\n" + "="*80)
        print("DIAGNOSIS:")
        
        if len(movie_lookup) == 0:
            print("❌ CRITICAL: No movies in lookup table!")
        elif missing_in_mappings or missing_in_lookup:
            print("❌ PROBLEM: Misaligned mappings between lookup and ID mappings")
        elif fake_count > real_count:
            print("❌ PROBLEM: Most movie titles appear to be fake/generated")
        else:
            print("✅ Mappings appear correct. Issue might be in inference logic.")
        
        print("="*80)
        
        return True
        
    except Exception as e:
        logger.error(f"Error during debugging: {e}")
        return False

def check_original_data_source():
    """Check the original data source to see if movies are real"""
    
    print("\n" + "="*80)
    print("CHECKING ORIGINAL DATA SOURCE")
    print("="*80)
    
    # Look for original data files
    data_dirs = ['.', 'data', '../data']
    
    for data_dir in data_dirs:
        movies_files = [
            'ml-32m/movies.csv',
            'ml-25m/movies.csv', 
            'ml-1m/movies.dat',
            'tmdb/actor_filmography_data_movies.csv'
        ]
        
        for movies_file in movies_files:
            file_path = os.path.join(data_dir, movies_file)
            if os.path.exists(file_path):
                print(f"\nFound original data: {file_path}")
                
                try:
                    if file_path.endswith('.dat'):
                        df = pd.read_csv(file_path, sep='::', names=['movieId', 'title', 'genres'], engine='python')
                    else:
                        df = pd.read_csv(file_path, nrows=20)  # Just sample
                    
                    print(f"Sample movies from original data:")
                    for idx, row in df.head(10).iterrows():
                        if 'title' in row:
                            print(f"  {row.get('movieId', row.get('media_id', 'NO_ID'))}: {row['title']}")
                    
                    return True
                    
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")
    
    print("No original movie data files found.")
    return False

if __name__ == "__main__":
    print("Starting debug analysis...")
    
    # Debug the trained model
    success = debug_model_mappings()
    
    if success:
        # Check original data source
        check_original_data_source()
    
    print("\nDebug analysis complete.")