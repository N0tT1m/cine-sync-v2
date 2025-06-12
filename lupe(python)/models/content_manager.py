import torch
import numpy as np
import pandas as pd
import pickle
import os
import logging
from typing import List, Tuple, Optional, Dict, Any, Union
from sklearn.preprocessing import MinMaxScaler

from .hybrid_recommender import HybridRecommenderModel, load_model
from .tv_recommender import TVShowRecommenderModel, load_tv_model, preprocess_tv_features, create_genre_vector

logger = logging.getLogger(__name__)


class LupeContentManager:
    """
    Unified content manager for both movies and TV shows
    Handles loading models, data, and generating recommendations
    """
    
    def __init__(self, models_dir: str, device: str = None):
        self.models_dir = models_dir
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        
        # Models
        self.movie_model = None
        self.tv_model = None
        
        # Data and mappings
        self.movie_lookup = {}
        self.tv_lookup = {}
        self.movie_mappings = {}
        self.tv_mappings = {}
        self.movie_metadata = {}
        self.tv_metadata = {}
        
        # Scalers and features
        self.movie_rating_scaler = None
        self.tv_rating_scaler = None
        self.movie_genres = []
        self.tv_genres = []
        self.all_genres = []
        
        # Data frames
        self.movies_df = None
        self.tv_df = None
        
        logger.info(f"LupeContentManager initialized with device: {self.device}")
    
    def load_models(self):
        """Load both movie and TV show models and data"""
        try:
            self.load_movie_data()  # Load data first
            self.load_tv_data()     # Load TV data
            self.load_movie_model() # Try to load models (may fail)
            self.load_tv_model()
            self.combine_genres()
            logger.info("All models and data loaded successfully")
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            # Don't raise - allow fallback operation
            logger.info("Continuing with data-only operation (no AI models)")
    
    def load_movie_data(self):
        """Load movie data (lookup, dataframe, genres) - separate from model"""
        try:
            # Try multiple possible locations for movie data
            movie_data_locations = [
                os.path.join(self.models_dir, 'movie_lookup.pkl'),
                os.path.join(os.path.dirname(self.models_dir), 'movies', 'movie_lookup.pkl'),
                os.path.join(self.models_dir, '..', 'movies', 'movie_lookup.pkl')
            ]
            
            # Load movie lookup data
            for movie_lookup_path in movie_data_locations:
                if os.path.exists(movie_lookup_path):
                    with open(movie_lookup_path, 'rb') as f:
                        self.movie_lookup = pickle.load(f)
                    logger.info(f"Loaded {len(self.movie_lookup)} movies from {movie_lookup_path}")
                    break
            
            # If we didn't find movie_lookup.pkl, check if the movies data exists in the models dir
            if not self.movie_lookup:
                models_movie_lookup = os.path.join(self.models_dir, 'movie_lookup.pkl')
                if os.path.exists(models_movie_lookup):
                    with open(models_movie_lookup, 'rb') as f:
                        self.movie_lookup = pickle.load(f)
                    logger.info(f"Loaded {len(self.movie_lookup)} movies from models directory")
            
            # Try multiple possible locations for movie CSV data
            movie_csv_locations = [
                os.path.join(self.models_dir, 'movies_data.csv'),
                os.path.join(os.path.dirname(self.models_dir), 'movies', 'movies_data.csv'),
                os.path.join(self.models_dir, '..', 'movies', 'movies_data.csv')
            ]
            
            # Load movie dataframe
            for movies_data_path in movie_csv_locations:
                if os.path.exists(movies_data_path):
                    self.movies_df = pd.read_csv(movies_data_path)
                    logger.info(f"Loaded movie dataframe with {len(self.movies_df)} rows from {movies_data_path}")
                    break
            
            # Extract genres from loaded data
            self._extract_movie_genres()
                    
        except Exception as e:
            logger.error(f"Error loading movie data: {e}")
            # Don't raise - we want to continue
    
    def load_movie_model(self):
        """Load movie recommendation model"""
        try:
            # Load movie model
            movie_model_path = os.path.join(self.models_dir, 'best_model.pt')
            movie_metadata_path = os.path.join(self.models_dir, 'model_metadata.pkl')
            
            if os.path.exists(movie_model_path) and os.path.exists(movie_metadata_path):
                # Load movie mappings first
                mappings_path = os.path.join(self.models_dir, 'id_mappings.pkl')
                if os.path.exists(mappings_path):
                    with open(mappings_path, 'rb') as f:
                        self.movie_mappings = pickle.load(f)
                
                    with open(movie_metadata_path, 'rb') as f:
                        self.movie_metadata = pickle.load(f)
                    
                    # Create and load movie model
                    self.movie_model = HybridRecommenderModel(
                        num_users=self.movie_mappings['num_users'],
                        num_items=self.movie_mappings['num_movies'],
                        embedding_dim=128
                    )
                    
                    checkpoint = torch.load(movie_model_path, map_location=self.device, weights_only=False)
                    self.movie_model.load_state_dict(checkpoint['model_state_dict'])
                    self.movie_model.to(self.device)
                    self.movie_model.eval()
                    
                    logger.info("Movie model loaded successfully")
                else:
                    logger.warning("Movie mappings file not found")
            else:
                logger.warning("Movie model files not found")
            
            # Load movie rating scaler
            movie_scaler_path = os.path.join(self.models_dir, 'rating_scaler.pkl')
            if os.path.exists(movie_scaler_path):
                with open(movie_scaler_path, 'rb') as f:
                    self.movie_rating_scaler = pickle.load(f)
                    
        except Exception as e:
            logger.error(f"Error loading movie model: {e}")
            # Ensure model is None if loading failed
            self.movie_model = None
            self.movie_mappings = None
            logger.info("Movie model set to None due to loading failure - will use fallback recommendations")
    
    def load_tv_data(self):
        """Load TV show data (lookup, dataframe, genres) - separate from model"""
        try:
            # Try multiple possible locations for TV data
            tv_data_locations = [
                os.path.join(self.models_dir, 'tv_lookup.pkl'),
                os.path.join(os.path.dirname(self.models_dir), 'tv', 'tv_lookup.pkl'),
                os.path.join(self.models_dir, '..', 'tv', 'tv_lookup.pkl')
            ]
            
            # Load TV lookup data
            for tv_lookup_path in tv_data_locations:
                if os.path.exists(tv_lookup_path):
                    with open(tv_lookup_path, 'rb') as f:
                        self.tv_lookup = pickle.load(f)
                    logger.info(f"Loaded {len(self.tv_lookup)} TV shows from {tv_lookup_path}")
                    break
            
            # Try multiple possible locations for TV CSV data
            tv_csv_locations = [
                os.path.join(self.models_dir, 'tv_data.csv'),
                os.path.join(os.path.dirname(self.models_dir), 'tv', 'tv_data.csv'),
                os.path.join(self.models_dir, '..', 'tv', 'tv_data.csv')
            ]
            
            # Load TV dataframe
            for tv_data_path in tv_csv_locations:
                if os.path.exists(tv_data_path):
                    self.tv_df = pd.read_csv(tv_data_path)
                    logger.info(f"Loaded TV dataframe with {len(self.tv_df)} rows from {tv_data_path}")
                    break
            
            # Extract genres from loaded data
            self._extract_tv_genres()
                    
        except Exception as e:
            logger.error(f"Error loading TV data: {e}")
            # Don't raise - we want to continue
    
    def load_tv_model(self):
        """Load TV show recommendation model"""
        try:
            # Load TV model
            tv_model_path = os.path.join(self.models_dir, 'best_tv_model.pt')
            tv_metadata_path = os.path.join(self.models_dir, 'tv_metadata.pkl')
            
            if os.path.exists(tv_model_path) and os.path.exists(tv_metadata_path):
                self.tv_model, self.tv_metadata = load_tv_model(tv_model_path, tv_metadata_path)
                self.tv_model.to(self.device)
                logger.info("TV show model loaded successfully")
            else:
                logger.warning("TV show model files not found")
            
            # Load TV mappings
            tv_mappings_path = os.path.join(self.models_dir, 'tv_id_mappings.pkl')
            if os.path.exists(tv_mappings_path):
                with open(tv_mappings_path, 'rb') as f:
                    self.tv_mappings = pickle.load(f)
            
            # Load TV rating scaler
            tv_scaler_path = os.path.join(self.models_dir, 'tv_rating_scaler.pkl')
            if os.path.exists(tv_scaler_path):
                with open(tv_scaler_path, 'rb') as f:
                    self.tv_rating_scaler = pickle.load(f)
                    
        except Exception as e:
            logger.error(f"Error loading TV model: {e}")
            # Don't raise here since TV model might not exist yet
            logger.info("Continuing without TV show model")
    
    def _extract_movie_genres(self):
        """Extract genres from movie data"""
        all_genres = set()
        
        # From movies_df
        if self.movies_df is not None:
            if 'genres' in self.movies_df.columns:
                for genres in self.movies_df['genres'].dropna():
                    if pd.notnull(genres) and isinstance(genres, str):
                        all_genres.update(genre.strip() for genre in genres.split('|') if genre.strip())
                logger.info(f"Extracted genres from movies dataframe")
        
        # From movie_lookup - this is our primary source
        if self.movie_lookup:
            for movie_info in self.movie_lookup.values():
                genres = movie_info.get('genres', '')
                if isinstance(genres, str) and genres.strip():
                    all_genres.update(genre.strip() for genre in genres.split('|') if genre.strip())
            logger.info(f"Extracted genres from movie lookup data")
        
        self.movie_genres = sorted([genre for genre in all_genres if genre and len(genre) > 1])
        logger.info(f"Found {len(self.movie_genres)} movie genres: {self.movie_genres}")
        
        # Check for Western-related genres
        western_genres = [g for g in self.movie_genres if 'western' in g.lower()]
        if western_genres:
            logger.info(f"Western-related genres found: {western_genres}")
        else:
            logger.info("No Western-related genres found in movie data")
    
    def _extract_tv_genres(self):
        """Extract genres from TV show data"""
        all_genres = set()
        
        # From tv_df
        if self.tv_df is not None:
            if 'genres' in self.tv_df.columns:
                for genres in self.tv_df['genres'].dropna():
                    if pd.notnull(genres) and isinstance(genres, str):
                        all_genres.update(genre.strip() for genre in genres.split('|') if genre.strip())
                logger.info(f"Extracted genres from TV dataframe")
        
        # From tv_lookup
        if self.tv_lookup:
            for show_info in self.tv_lookup.values():
                genres = show_info.get('genres', '')
                if isinstance(genres, str) and genres.strip():
                    all_genres.update(genre.strip() for genre in genres.split('|') if genre.strip())
            logger.info(f"Extracted genres from TV lookup data")
        
        self.tv_genres = sorted([genre for genre in all_genres if genre and len(genre) > 1])
        logger.info(f"Found {len(self.tv_genres)} TV show genres: {self.tv_genres[:10] if self.tv_genres else 'none'}")
    
    def combine_genres(self):
        """Combine movie and TV genres into unified list"""
        all_genres = set(self.movie_genres + self.tv_genres)
        self.all_genres = sorted(list(all_genres))
        logger.info(f"Combined genres: {len(self.all_genres)} total genres")
    
    def get_available_genres(self, content_type: str = "mixed") -> List[str]:
        """
        Get available genres filtered by content type
        
        Args:
            content_type: "movie", "tv", or "mixed"
            
        Returns:
            List of genre names available for the specified content type
        """
        try:
            if content_type == "movie":
                return self.movie_genres if hasattr(self, 'movie_genres') and self.movie_genres else []
            elif content_type == "tv":
                return self.tv_genres if hasattr(self, 'tv_genres') and self.tv_genres else []
            else:  # mixed
                return self.all_genres if hasattr(self, 'all_genres') and self.all_genres else []
        except Exception as e:
            logger.error(f"Error getting available genres for {content_type}: {e}")
            return []
    
    def get_recommendations(self, user_id: int, content_type: str = "mixed", 
                          top_k: int = 10, genre_filter: str = None) -> List[Tuple[str, str, str, float]]:
        """
        Get recommendations for movies, TV shows, or mixed content
        
        Args:
            user_id: User ID for personalized recommendations
            content_type: "movie", "tv", or "mixed"
            top_k: Number of recommendations to return
            genre_filter: Optional genre filter
            
        Returns:
            List of (content_id, title, content_type, score) tuples
        """
        recommendations = []
        
        try:
            if content_type == "movie" or content_type == "mixed":
                movie_recs = self._get_movie_recommendations(user_id, top_k, genre_filter)
                recommendations.extend(movie_recs)
            
            if content_type == "tv" or content_type == "mixed":
                tv_recs = self._get_tv_recommendations(user_id, top_k, genre_filter)
                recommendations.extend(tv_recs)
            
            # If mixed, interleave and limit results
            if content_type == "mixed":
                recommendations = self._interleave_recommendations(recommendations, top_k)
            else:
                recommendations = recommendations[:top_k]
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error getting recommendations: {e}")
            return []
    
    def _get_movie_recommendations(self, user_id: int, limit: int, 
                                 genre_filter: str = None) -> List[Tuple[str, str, str, float]]:
        """Get movie recommendations"""
        if not self.movie_model or not self.movie_mappings:
            # Fallback to simple recommendations when model isn't available
            logger.info(f"Using fallback movie recommendations (model available: {self.movie_model is not None}, mappings available: {self.movie_mappings is not None})")
            return self._get_fallback_movie_recommendations(limit, genre_filter, user_id)
        
        try:
            # Use existing user or fallback
            if user_id not in self.movie_mappings['user_id_to_idx']:
                fallback_user_id = list(self.movie_mappings['user_id_to_idx'].keys())[0]
                user_idx = self.movie_mappings['user_id_to_idx'][fallback_user_id]
            else:
                user_idx = self.movie_mappings['user_id_to_idx'][user_id]
            
            recommendations = []
            available_movie_ids = list(self.movie_lookup.keys())
            
            # Sample for efficiency
            sample_size = min(500, len(available_movie_ids))
            sampled_ids = np.random.choice(available_movie_ids, sample_size, replace=False)
            
            with torch.no_grad():
                for movie_id in sampled_ids:
                    if movie_id not in self.movie_mappings['movie_id_to_idx']:
                        continue
                    
                    movie_info = self.movie_lookup[movie_id]
                    genres_value = movie_info.get('genres', '')
                    
                    # Apply genre filter
                    if genre_filter and isinstance(genres_value, str):
                        if genre_filter.lower() not in genres_value.lower():
                            continue
                    
                    movie_idx = self.movie_mappings['movie_id_to_idx'][movie_id]
                    
                    # Get prediction - ensure tensors are on same device as model
                    user_tensor = torch.tensor([user_idx], dtype=torch.long, device=self.device)
                    movie_tensor = torch.tensor([movie_idx], dtype=torch.long, device=self.device)
                    
                    score = self.movie_model(user_tensor, movie_tensor).item()
                    
                    # Inverse transform rating if scaler exists
                    if self.movie_rating_scaler:
                        score = self.movie_rating_scaler.inverse_transform([[score]])[0][0]
                    
                    recommendations.append((
                        str(movie_id),
                        movie_info.get('title', 'Unknown'),
                        'movie',
                        score + np.random.normal(0, 0.05)  # Add noise for diversity
                    ))
            
            # Sort by score and return
            recommendations.sort(key=lambda x: x[3], reverse=True)
            return recommendations[:limit]
            
        except Exception as e:
            logger.error(f"Error in movie recommendations: {e}")
            return []
    
    def _get_tv_recommendations(self, user_id: int, limit: int, 
                              genre_filter: str = None) -> List[Tuple[str, str, str, float]]:
        """Get TV show recommendations"""
        if not self.tv_model or not self.tv_mappings:
            # Fallback to simple TV recommendations
            return self._get_fallback_tv_recommendations(limit, genre_filter)
        
        try:
            # Use existing user or fallback
            if user_id not in self.tv_mappings['user_id_to_idx']:
                fallback_user_id = list(self.tv_mappings['user_id_to_idx'].keys())[0]
                user_idx = self.tv_mappings['user_id_to_idx'][fallback_user_id]
            else:
                user_idx = self.tv_mappings['user_id_to_idx'][user_id]
            
            recommendations = []
            available_show_ids = list(self.tv_lookup.keys())
            
            # Sample for efficiency
            sample_size = min(500, len(available_show_ids))
            sampled_ids = np.random.choice(available_show_ids, sample_size, replace=False)
            
            with torch.no_grad():
                for show_id in sampled_ids:
                    if show_id not in self.tv_mappings['show_id_to_idx']:
                        continue
                    
                    show_info = self.tv_lookup[show_id]
                    genres_value = show_info.get('genres', '')
                    
                    # Apply genre filter
                    if genre_filter and isinstance(genres_value, str):
                        if genre_filter.lower() not in genres_value.lower():
                            continue
                    
                    show_idx = self.tv_mappings['show_id_to_idx'][show_id]
                    
                    # Prepare features
                    tv_features = preprocess_tv_features(show_info)
                    genre_vector = create_genre_vector(genres_value, self.tv_genres)
                    
                    # Get prediction
                    user_tensor = torch.tensor([user_idx], dtype=torch.long).to(self.device)
                    show_tensor = torch.tensor([show_idx], dtype=torch.long).to(self.device)
                    tv_tensor = torch.tensor([tv_features], dtype=torch.float32).to(self.device)
                    genre_tensor = torch.tensor([genre_vector], dtype=torch.float32).to(self.device)
                    
                    score = self.tv_model(user_tensor, show_tensor, tv_tensor, genre_tensor).item()
                    
                    # Inverse transform rating if scaler exists
                    if self.tv_rating_scaler:
                        score = self.tv_rating_scaler.inverse_transform([[score]])[0][0]
                    
                    recommendations.append((
                        str(show_id),
                        show_info.get('title', 'Unknown'),
                        'tv',
                        score + np.random.normal(0, 0.05)  # Add noise for diversity
                    ))
            
            # Sort by score and return
            recommendations.sort(key=lambda x: x[3], reverse=True)
            return recommendations[:limit]
            
        except Exception as e:
            logger.error(f"Error in TV recommendations: {e}")
            return self._get_fallback_tv_recommendations(limit, genre_filter)
    
    def _get_fallback_movie_recommendations(self, limit: int, 
                                          genre_filter: str = None, user_id: int = None) -> List[Tuple[str, str, str, float]]:
        """Fallback movie recommendations with personalized scoring when model is not available"""
        if not self.movie_lookup:
            return []
        
        # Get user preferences if available
        user_genre_prefs = {}
        user_content_prefs = {}
        
        if user_id:
            try:
                user_genre_prefs, user_content_prefs = self._get_user_preferences(user_id, 'movie')
            except Exception as e:
                logger.error(f"Error loading user preferences: {e}")
        
        recommendations = []
        all_movies = list(self.movie_lookup.items())
        np.random.shuffle(all_movies)
        
        genre_matches = 0
        for movie_id, movie_info in all_movies:
            genres_value = movie_info.get('genres', '')
            
            # Apply genre filter
            if genre_filter and isinstance(genres_value, str):
                if genre_filter.lower() not in genres_value.lower():
                    continue
                else:
                    genre_matches += 1
            
            # Calculate personalized score
            score = self._calculate_personalized_score(
                movie_id, movie_info, user_genre_prefs, user_content_prefs
            )
            
            recommendations.append((
                str(movie_id),
                movie_info.get('title', 'Unknown'),
                'movie',
                score
            ))
            
            # Early exit if we have enough
            if len(recommendations) >= limit * 3:  # Get more for better selection
                break
        
        # Sort by personalized score
        recommendations.sort(key=lambda x: x[3], reverse=True)
        
        if genre_filter:
            logger.info(f"Movie genre filter '{genre_filter}': found {genre_matches} matches, returning {len(recommendations[:limit])} recommendations")
        
        if user_id and user_genre_prefs:
            logger.info(f"Used personalized scoring for user {user_id} with {len(user_genre_prefs)} genre preferences")
        
        return recommendations[:limit]
    
    def _get_user_preferences(self, user_id: int, content_type: str):
        """Get user's genre and content preferences from database"""
        user_genre_prefs = {}
        user_content_prefs = {}
        
        try:
            # Import here to avoid circular imports
            import psycopg2
            from config import load_config
            
            config = load_config()
            
            with psycopg2.connect(
                host=config.database.host,
                port=config.database.port,
                database=config.database.database,
                user=config.database.user,
                password=config.database.password
            ) as conn:
                cursor = conn.cursor()
                
                # Get genre preferences
                cursor.execute('''
                    SELECT genre, preference_score, interaction_count 
                    FROM user_genre_preferences 
                    WHERE user_id = %s AND content_type = %s
                ''', (user_id, content_type))
                
                for genre, score, count in cursor.fetchall():
                    # Weight by interaction count (more interactions = more reliable)
                    weighted_score = score * min(count / 5.0, 1.0)  # Cap at 5 interactions
                    user_genre_prefs[genre] = weighted_score
                
                # Get content preferences (liked/disliked specific content)
                cursor.execute('''
                    SELECT content_id, preference 
                    FROM user_preferences 
                    WHERE user_id = %s AND content_type = %s
                ''', (user_id, content_type))
                
                for content_id, preference in cursor.fetchall():
                    user_content_prefs[content_id] = preference
                
        except Exception as e:
            logger.error(f"Error fetching user preferences: {e}")
        
        return user_genre_prefs, user_content_prefs
    
    def _calculate_personalized_score(self, movie_id, movie_info, user_genre_prefs, user_content_prefs):
        """Calculate personalized score based on user preferences"""
        base_score = np.random.uniform(0.3, 0.7)  # Lower base randomness
        
        # Check if user has specific preference for this content
        if str(movie_id) in user_content_prefs:
            content_pref = user_content_prefs[str(movie_id)]
            if content_pref == 'love':
                return 0.95 + np.random.uniform(0, 0.05)
            elif content_pref == 'like':
                return 0.85 + np.random.uniform(0, 0.1)
            elif content_pref == 'dislike':
                return 0.1 + np.random.uniform(0, 0.15)
            elif content_pref == 'hate':
                return 0.05 + np.random.uniform(0, 0.1)
        
        # Calculate genre-based personalization
        if user_genre_prefs:
            genres_value = movie_info.get('genres', '')
            if isinstance(genres_value, str) and genres_value.strip():
                genre_list = [g.strip() for g in genres_value.split('|') if g.strip()]
                
                genre_scores = []
                for genre in genre_list:
                    if genre in user_genre_prefs:
                        genre_scores.append(user_genre_prefs[genre])
                
                if genre_scores:
                    avg_genre_score = sum(genre_scores) / len(genre_scores)
                    # Apply genre preference boost/penalty
                    if avg_genre_score > 0:
                        base_score += min(avg_genre_score * 0.3, 0.3)  # Max 30% boost
                    else:
                        base_score += max(avg_genre_score * 0.3, -0.3)  # Max 30% penalty
        
        # Ensure score stays in valid range
        return max(0.05, min(0.98, base_score))
    
    def _get_fallback_tv_recommendations(self, limit: int, 
                                       genre_filter: str = None) -> List[Tuple[str, str, str, float]]:
        """Fallback TV recommendations when model is not available"""
        if not self.tv_lookup:
            return []
        
        recommendations = []
        all_shows = list(self.tv_lookup.items())
        np.random.shuffle(all_shows)
        
        for show_id, show_info in all_shows:
            genres_value = show_info.get('genres', '')
            
            # Apply genre filter
            if genre_filter and isinstance(genres_value, str):
                if genre_filter.lower() not in genres_value.lower():
                    continue
            
            # Simple scoring based on popularity and randomness
            score = np.random.uniform(0.5, 1.0)
            
            recommendations.append((
                str(show_id),
                show_info.get('title', 'Unknown'),
                'tv',
                score
            ))
        
        recommendations.sort(key=lambda x: x[3], reverse=True)
        return recommendations[:limit]
    
    def _interleave_recommendations(self, recommendations: List[Tuple], 
                                  total_limit: int) -> List[Tuple[str, str, str, float]]:
        """Interleave movie and TV recommendations"""
        movies = [r for r in recommendations if r[2] == 'movie']
        tv_shows = [r for r in recommendations if r[2] == 'tv']
        
        # Sort both by score
        movies.sort(key=lambda x: x[3], reverse=True)
        tv_shows.sort(key=lambda x: x[3], reverse=True)
        
        # Interleave with slight preference for higher scores
        result = []
        i = j = 0
        
        while len(result) < total_limit and (i < len(movies) or j < len(tv_shows)):
            # Add movie if we have one and either no TV show or movie has higher score
            if i < len(movies) and (j >= len(tv_shows) or 
                                   (len(result) % 2 == 0) or 
                                   movies[i][3] > tv_shows[j][3]):
                result.append(movies[i])
                i += 1
            # Add TV show
            elif j < len(tv_shows):
                result.append(tv_shows[j])
                j += 1
            else:
                break
        
        return result
    
    def get_cross_content_recommendations(self, user_id: int, source_type: str, 
                                        target_type: str, top_k: int = 10) -> List[Tuple[str, str, str, float]]:
        """
        Get cross-content recommendations (e.g., TV shows based on movie preferences)
        
        Args:
            user_id: User ID
            source_type: "movie" or "tv" - content type to analyze preferences from
            target_type: "movie" or "tv" - content type to recommend
            top_k: Number of recommendations
            
        Returns:
            List of recommendations in target content type
        """
        # For now, use simple genre-based cross-recommendation
        # In the future, this could be enhanced with more sophisticated cross-domain learning
        
        try:
            # Get user's preferred genres from source content
            preferred_genres = self._analyze_user_genre_preferences(user_id, source_type)
            
            if not preferred_genres:
                # Fallback to general recommendations
                return self.get_recommendations(user_id, target_type, top_k)
            
            # Get recommendations in target type with preferred genres
            recommendations = []
            for genre in preferred_genres[:3]:  # Use top 3 preferred genres
                genre_recs = self.get_recommendations(user_id, target_type, 
                                                    top_k // len(preferred_genres[:3]), 
                                                    genre)
                recommendations.extend(genre_recs)
            
            # Remove duplicates and sort by score
            seen = set()
            unique_recs = []
            for rec in recommendations:
                if rec[0] not in seen:
                    unique_recs.append(rec)
                    seen.add(rec[0])
            
            unique_recs.sort(key=lambda x: x[3], reverse=True)
            return unique_recs[:top_k]
            
        except Exception as e:
            logger.error(f"Error in cross-content recommendations: {e}")
            return self.get_recommendations(user_id, target_type, top_k)
    
    def _analyze_user_genre_preferences(self, user_id: int, content_type: str) -> List[str]:
        """Analyze user's genre preferences from their ratings/interactions"""
        # This is a placeholder implementation
        # In a real system, this would analyze the user's rating history
        
        # For now, return some popular genres as a fallback
        if content_type == "movie":
            return ['Action', 'Drama', 'Comedy', 'Thriller', 'Sci-Fi']
        else:
            return ['Drama', 'Comedy', 'Crime', 'Action', 'Sci-Fi']
    
    def get_similar_content(self, content_id: str, content_type: str, 
                          top_k: int = 10) -> List[Tuple[str, str, str, float]]:
        """
        Get content similar to the specified content
        
        Args:
            content_id: ID of the reference content
            content_type: "movie" or "tv"
            top_k: Number of similar items to return
            
        Returns:
            List of similar content recommendations
        """
        try:
            if content_type == "movie":
                return self._get_similar_movies(content_id, top_k)
            elif content_type == "tv":
                return self._get_similar_tv_shows(content_id, top_k)
            else:
                return []
                
        except Exception as e:
            logger.error(f"Error getting similar content: {e}")
            return []
    
    def _get_similar_movies(self, movie_id: str, top_k: int) -> List[Tuple[str, str, str, float]]:
        """Get movies similar to the specified movie"""
        try:
            movie_id = int(movie_id)
            if movie_id not in self.movie_lookup:
                return []
            
            target_movie = self.movie_lookup[movie_id]
            target_genres = target_movie.get('genres', '')
            
            if not isinstance(target_genres, str):
                return []
            
            target_genre_set = set(target_genres.split('|')) if target_genres else set()
            similarities = []
            
            for mid, movie_info in self.movie_lookup.items():
                if mid == movie_id:
                    continue
                
                movie_genres = movie_info.get('genres', '')
                if not isinstance(movie_genres, str):
                    continue
                
                movie_genre_set = set(movie_genres.split('|')) if movie_genres else set()
                
                # Calculate genre similarity
                if target_genre_set and movie_genre_set:
                    similarity = len(target_genre_set & movie_genre_set) / len(target_genre_set | movie_genre_set)
                else:
                    similarity = 0
                
                similarities.append((
                    str(mid),
                    movie_info.get('title', 'Unknown'),
                    'movie',
                    similarity + np.random.uniform(0, 0.1)  # Add small random factor
                ))
            
            similarities.sort(key=lambda x: x[3], reverse=True)
            return similarities[:top_k]
            
        except Exception as e:
            logger.error(f"Error getting similar movies: {e}")
            return []
    
    def _get_similar_tv_shows(self, show_id: str, top_k: int) -> List[Tuple[str, str, str, float]]:
        """Get TV shows similar to the specified show"""
        try:
            show_id = int(show_id)
            if show_id not in self.tv_lookup:
                return []
            
            target_show = self.tv_lookup[show_id]
            target_genres = target_show.get('genres', '')
            
            if not isinstance(target_genres, str):
                return []
            
            target_genre_set = set(target_genres.split('|')) if target_genres else set()
            similarities = []
            
            for sid, show_info in self.tv_lookup.items():
                if sid == show_id:
                    continue
                
                show_genres = show_info.get('genres', '')
                if not isinstance(show_genres, str):
                    continue
                
                show_genre_set = set(show_genres.split('|')) if show_genres else set()
                
                # Calculate genre similarity
                if target_genre_set and show_genre_set:
                    similarity = len(target_genre_set & show_genre_set) / len(target_genre_set | show_genre_set)
                else:
                    similarity = 0
                
                similarities.append((
                    str(sid),
                    show_info.get('title', 'Unknown'),
                    'tv',
                    similarity + np.random.uniform(0, 0.1)  # Add small random factor
                ))
            
            similarities.sort(key=lambda x: x[3], reverse=True)
            return similarities[:top_k]
            
        except Exception as e:
            logger.error(f"Error getting similar TV shows: {e}")
            return []
    
    def get_model_status(self) -> Dict[str, Any]:
        """Get status information about loaded models and data"""
        return {
            'movie_model_loaded': self.movie_model is not None,
            'tv_model_loaded': self.tv_model is not None,
            'movie_count': len(self.movie_lookup),
            'tv_count': len(self.tv_lookup),
            'movie_genres': len(self.movie_genres),
            'tv_genres': len(self.tv_genres),
            'total_genres': len(self.all_genres),
            'device': str(self.device)
        }