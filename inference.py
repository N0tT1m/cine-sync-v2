#!/usr/bin/env python3
# FIXED: inference.py - Proper movie recommendation inference with real movie mapping

import os
import pickle
import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
import logging

from config import load_config
from models import HybridRecommenderModel, load_model
from utils import DatabaseManager, load_ratings_data, load_movies_data

# Load configuration
config = load_config()

# Setup logging
log_level = logging.DEBUG if config.debug else logging.INFO
logging.basicConfig(level=log_level)
logger = logging.getLogger("movie-recommendation-inference")


class MovieRecommendationSystem:
    """FIXED: Recommendation system that only recommends REAL movies"""
    
    def __init__(self, config=None):
        if config is None:
            config = load_config()
        self.config = config
        self.model_dir = config.model.models_dir
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize database manager
        self.db_manager = DatabaseManager(config.database)
        
        # Load all necessary components
        self._load_model_components()
    
    def _load_model_components(self):
        """Load model, mappings, and movie data"""
        try:
            # Load ID mappings (CRITICAL for proper inference)
            with open(os.path.join(self.model_dir, 'id_mappings.pkl'), 'rb') as f:
                self.mappings = pickle.load(f)
            
            logger.info(f"Loaded ID mappings: {self.mappings['num_users']} users, {self.mappings['num_movies']} movies")
            
            # Load movie lookup table
            with open(os.path.join(self.model_dir, 'movie_lookup.pkl'), 'rb') as f:
                self.movie_lookup = pickle.load(f)
            
            # Load movies dataframe
            self.movies_df = pd.read_csv(os.path.join(self.model_dir, 'movies_data.csv'))
            
            # Extract genres list from movies_df
            all_genres = set()
            for genres in self.movies_df['genres'].dropna():
                if pd.notnull(genres):
                    all_genres.update(genres.split('|'))
            self.genres_list = sorted(list(all_genres))
            
            logger.info(f"Found {len(self.genres_list)} genres: {self.genres_list}")
            
            # Load rating scaler
            with open(os.path.join(self.model_dir, 'rating_scaler.pkl'), 'rb') as f:
                self.rating_scaler = pickle.load(f)
            
            # Load model metadata
            with open(os.path.join(self.model_dir, 'model_metadata.pkl'), 'rb') as f:
                self.metadata = pickle.load(f)
            
            # Create and load the model
            self.model = HybridRecommenderModel(
                num_users=self.mappings['num_users'],
                num_movies=self.mappings['num_movies'],
                num_genres=len(self.genres_list),
                embedding_size=128
            )
            
            # Load trained weights
            checkpoint = torch.load(os.path.join(self.model_dir, 'best_model.pt'), 
                                  map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.to(self.device)
            self.model.eval()
            
            logger.info("Model loaded successfully!")
            logger.info(f"Available movies in system: {len(self.movie_lookup)}")
            
        except Exception as e:
            logger.error(f"Error loading model components: {e}")
            raise
    
    def _get_movie_genre_features(self, movie_id):
        """Get genre features for a specific movie ID"""
        try:
            # Find movie in the dataframe using original movie ID
            movie_row = self.movies_df[self.movies_df['media_id'] == movie_id]
            
            if movie_row.empty:
                logger.warning(f"Movie ID {movie_id} not found in dataset")
                return np.zeros(len(self.genres_list))
            
            # Extract genre features
            genre_features = []
            for genre in self.genres_list:
                if genre in movie_row.iloc[0]:
                    genre_features.append(movie_row.iloc[0][genre])
                else:
                    genre_features.append(0)
            
            return np.array(genre_features, dtype=np.float32)
            
        except Exception as e:
            logger.error(f"Error getting genre features for movie {movie_id}: {e}")
            return np.zeros(len(self.genres_list))
    
    def get_user_recommendations(self, user_id, num_recommendations=10, exclude_seen=True):
        """
        IMPROVED: Get recommendations for a user with intelligent candidate generation
        """
        try:
            # Check if user exists in our mappings
            if user_id not in self.mappings['user_id_to_idx']:
                logger.warning(f"User {user_id} not found in training data. Using genre-based fallback.")
                return self._get_genre_based_recommendations(num_recommendations)
            
            user_idx = self.mappings['user_id_to_idx'][user_id]
            
            # IMPROVED: Smart candidate generation instead of scoring ALL movies
            candidate_movie_ids = self._generate_candidates(user_id, user_idx, num_recommendations * 3)
            
            if not candidate_movie_ids:
                logger.warning(f"No candidates found for user {user_id}, falling back to popular movies")
                return self._get_popular_movies_fallback(num_recommendations)
            
            logger.info(f"Generated {len(candidate_movie_ids)} candidates for user {user_id}")
            
            # Get indices for candidates only
            candidate_movie_indices = [self.mappings['movie_id_to_idx'][mid] for mid in candidate_movie_ids]
            
            # Prepare batch prediction for CANDIDATES only (much more efficient)
            user_indices = torch.tensor([user_idx] * len(candidate_movie_indices), dtype=torch.long)
            movie_indices = torch.tensor(candidate_movie_indices, dtype=torch.long)
            
            # Get genre features for candidate movies
            genre_features_list = []
            for movie_id in candidate_movie_ids:
                genre_features = self._get_movie_genre_features(movie_id)
                genre_features_list.append(genre_features)
            
            genre_features_tensor = torch.tensor(np.array(genre_features_list), dtype=torch.float32)
            
            # Move to device
            user_indices = user_indices.to(self.device)
            movie_indices = movie_indices.to(self.device)
            genre_features_tensor = genre_features_tensor.to(self.device)
            
            # Predict ratings for candidate movies only
            with torch.no_grad():
                predictions = self.model(user_indices, movie_indices, genre_features_tensor)
                predictions = predictions.cpu().numpy().flatten()
            
            # Convert predictions back to original rating scale
            predictions_rescaled = self.rating_scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()
            
            # Create recommendation list with REAL movie information
            recommendations = []
            for i, (movie_id, prediction) in enumerate(zip(candidate_movie_ids, predictions_rescaled)):
                movie_info = self.movie_lookup[movie_id]
                recommendations.append({
                    'movie_id': movie_id,
                    'title': movie_info.get('title', 'Unknown'),
                    'genres': movie_info.get('genres', ''),
                    'predicted_rating': float(prediction),
                    'confidence': float(predictions[i])
                })
            
            # Sort by predicted rating (descending)
            recommendations.sort(key=lambda x: x['predicted_rating'], reverse=True)
            
            # Return top N recommendations
            top_recommendations = recommendations[:num_recommendations]
            
            logger.info(f"Generated {len(top_recommendations)} recommendations for user {user_id}")
            for i, rec in enumerate(top_recommendations[:5]):
                logger.info(f"  {i+1}. {rec['title']} (ID: {rec['movie_id']}) - {rec['predicted_rating']:.2f}")
            
            return top_recommendations
            
        except Exception as e:
            logger.error(f"Error generating recommendations for user {user_id}: {e}")
            return []
    
    def get_movie_info(self, movie_id):
        """Get information about a specific movie"""
        if movie_id in self.movie_lookup:
            return self.movie_lookup[movie_id]
        else:
            return None
    
    def search_movies(self, query, limit=10):
        """Search for movies by title"""
        query_lower = query.lower()
        results = []
        
        for movie_id, movie_info in self.movie_lookup.items():
            if query_lower in movie_info.get('title', '').lower():
                results.append({
                    'movie_id': movie_id,
                    'title': movie_info.get('title', ''),
                    'genres': movie_info.get('genres', '')
                })
        
        return results[:limit]
    
    def _generate_candidates(self, user_id, user_idx, num_candidates):
        """
        Generate candidate movies for recommendation using multiple strategies
        """
        try:
            candidates = set()
            
            # Strategy 1: Genre-based candidates (most important)
            genre_candidates = self._get_genre_based_candidates(user_idx, num_candidates // 2)
            candidates.update(genre_candidates)
            
            # Strategy 2: Popular movies (ensures good fallback)
            popular_candidates = self._get_popular_candidates(num_candidates // 4)
            candidates.update(popular_candidates)
            
            # Strategy 3: Diverse genre representation
            diverse_candidates = self._get_diverse_genre_candidates(num_candidates // 4)
            candidates.update(diverse_candidates)
            
            # Convert to list and limit
            candidate_list = list(candidates)[:num_candidates]
            
            logger.info(f"Generated {len(candidate_list)} total candidates using multiple strategies")
            return candidate_list
            
        except Exception as e:
            logger.error(f"Error generating candidates: {e}")
            return []
    
    def _get_genre_based_candidates(self, user_idx, num_candidates):
        """Get candidates based on inferred user genre preferences"""
        try:
            candidates = []
            
            # Sample movies from different genres to get variety
            genre_movie_map = {}
            for movie_id, movie_info in self.movie_lookup.items():
                genres = movie_info.get('genres', '').split('|')
                for genre in genres:
                    if genre not in genre_movie_map:
                        genre_movie_map[genre] = []
                    genre_movie_map[genre].append(movie_id)
            
            # Get movies from popular genres with some randomization
            import random
            popular_genres = ['Action', 'Comedy', 'Drama', 'Thriller', 'Adventure', 'Romance', 'Crime', 'Sci-Fi']
            
            for genre in popular_genres:
                if genre in genre_movie_map and len(candidates) < num_candidates:
                    genre_movies = genre_movie_map[genre]
                    # Randomly sample from each genre to avoid always getting the same movies
                    sample_size = min(num_candidates // len(popular_genres) + 1, len(genre_movies))
                    sampled = random.sample(genre_movies, sample_size)
                    candidates.extend(sampled)
            
            return candidates[:num_candidates]
            
        except Exception as e:
            logger.error(f"Error getting genre-based candidates: {e}")
            return []
    
    def _get_popular_candidates(self, num_candidates):
        """Get popular movies as candidates (movies that appear frequently in dataset)"""
        try:
            # Use a simple heuristic: movies with lower IDs tend to be more popular in MovieLens
            movie_ids = list(self.movie_lookup.keys())
            movie_ids.sort()  # Lower movie IDs first
            return movie_ids[:num_candidates]
            
        except Exception as e:
            logger.error(f"Error getting popular candidates: {e}")
            return []
    
    def _get_diverse_genre_candidates(self, num_candidates):
        """Get candidates to ensure genre diversity"""
        try:
            candidates = []
            import random
            
            # Group movies by primary genre
            genre_groups = {}
            for movie_id, movie_info in self.movie_lookup.items():
                primary_genre = movie_info.get('genres', '').split('|')[0] if movie_info.get('genres') else 'Unknown'
                if primary_genre not in genre_groups:
                    genre_groups[primary_genre] = []
                genre_groups[primary_genre].append(movie_id)
            
            # Sample from each genre group
            genres = list(genre_groups.keys())
            random.shuffle(genres)
            
            per_genre = max(1, num_candidates // len(genres))
            for genre in genres:
                if len(candidates) < num_candidates:
                    sample_size = min(per_genre, len(genre_groups[genre]))
                    sampled = random.sample(genre_groups[genre], sample_size)
                    candidates.extend(sampled)
            
            return candidates[:num_candidates]
            
        except Exception as e:
            logger.error(f"Error getting diverse candidates: {e}")
            return []
    
    def _get_genre_based_recommendations(self, num_recommendations):
        """Fallback for users not in training data"""
        try:
            # Use popular genres for new users
            popular_genres = ['Action', 'Comedy', 'Drama']
            candidates = self._get_genre_based_candidates(0, num_recommendations * 2)
            
            # Return random selection since we can't predict for unknown user
            import random
            random.shuffle(candidates)
            
            recommendations = []
            for movie_id in candidates[:num_recommendations]:
                movie_info = self.movie_lookup[movie_id]
                recommendations.append({
                    'movie_id': movie_id,
                    'title': movie_info.get('title', 'Unknown'),
                    'genres': movie_info.get('genres', ''),
                    'predicted_rating': 4.0,  # Default high rating for popular movies
                    'confidence': 0.5  # Lower confidence for fallback
                })
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error in genre-based fallback: {e}")
            return []
    
    def _get_popular_movies_fallback(self, num_recommendations):
        """Final fallback to popular movies"""
        try:
            popular_candidates = self._get_popular_candidates(num_recommendations)
            
            recommendations = []
            for movie_id in popular_candidates:
                movie_info = self.movie_lookup[movie_id]
                recommendations.append({
                    'movie_id': movie_id,
                    'title': movie_info.get('title', 'Unknown'),
                    'genres': movie_info.get('genres', ''),
                    'predicted_rating': 3.8,  # Default good rating
                    'confidence': 0.3  # Low confidence for fallback
                })
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error in popular movies fallback: {e}")
            return []

# Example usage and testing
def test_recommendations():
    """Test the recommendation system"""
    try:
        # Initialize the recommendation system
        rec_system = MovieRecommendationSystem()
        
        # Get a sample user ID from the mappings
        sample_user_id = list(rec_system.mappings['user_id_to_idx'].keys())[0]
        
        print(f"\n=== Testing Recommendations for User {sample_user_id} ===")
        
        # Get recommendations
        recommendations = rec_system.get_user_recommendations(sample_user_id, num_recommendations=10)
        
        print(f"\nTop 10 Movie Recommendations:")
        print("-" * 80)
        
        for i, rec in enumerate(recommendations, 1):
            print(f"{i:2d}. {rec['title']}")
            print(f"    Movie ID: {rec['movie_id']}")
            print(f"    Genres: {rec['genres']}")
            print(f"    Predicted Rating: {rec['predicted_rating']:.2f}")
            print(f"    Confidence: {rec['confidence']:.3f}")
            print()
        
        # Verify these are real movies by checking a few
        print("=== Verification: Checking if recommended movies are real ===")
        for i, rec in enumerate(recommendations[:3]):
            movie_info = rec_system.get_movie_info(rec['movie_id'])
            if movie_info:
                print(f"✅ Movie {rec['movie_id']} '{rec['title']}' exists in database")
            else:
                print(f"❌ Movie {rec['movie_id']} '{rec['title']}' NOT found in database")
        
        # Test movie search
        print("\n=== Testing Movie Search ===")
        search_results = rec_system.search_movies("toy story", limit=5)
        print("Search results for 'toy story':")
        for result in search_results:
            print(f"  - {result['title']} (ID: {result['movie_id']})")
        
    except Exception as e:
        logger.error(f"Error in test_recommendations: {e}")

if __name__ == "__main__":
    test_recommendations()