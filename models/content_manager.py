"""
Content Manager for CineSync v2 - Manages both Movie and TV Show models
Handles model selection, cross-content recommendations, and unified inference
"""

import torch
import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import logging

from .hybrid_recommender import HybridRecommenderModel, load_model as load_movie_model
from .tv_recommender import TVShowRecommenderModel, load_tv_model

logger = logging.getLogger(__name__)

class LupeContentManager:
    """
    Lupe AI - Unified content recommendation manager for movies and TV shows
    """
    
    def __init__(self, models_dir: str = "models"):
        self.models_dir = Path(models_dir)
        self.movie_model = None
        self.tv_model = None
        self.movie_metadata = None
        self.tv_metadata = None
        self.movie_encoders = None
        self.tv_encoders = None
        
        # Cross-content learning weights
        self.cross_content_weights = {
            'movie_to_tv': 0.3,  # Weight for using movie preferences to recommend TV
            'tv_to_movie': 0.3   # Weight for using TV preferences to recommend movies
        }
    
    def load_models(self, movie_model_path: str = None, tv_model_path: str = None):
        """Load both movie and TV show models"""
        
        # Load movie model
        if movie_model_path is None:
            movie_model_path = self.models_dir / "best_model.pt"
            movie_metadata_path = self.models_dir / "model_metadata.pkl"
        else:
            movie_metadata_path = movie_model_path.replace('.pt', '_metadata.pkl')
        
        if Path(movie_model_path).exists():
            try:
                self.movie_model, self.movie_metadata = load_movie_model(
                    str(movie_model_path), str(movie_metadata_path)
                )
                logger.info("Loaded movie recommendation model")
                
                # Load movie encoders
                movie_encoders_path = self.models_dir / "id_mappings.pkl"
                if movie_encoders_path.exists():
                    with open(movie_encoders_path, 'rb') as f:
                        self.movie_encoders = pickle.load(f)
                
            except Exception as e:
                logger.error(f"Failed to load movie model: {e}")
        
        # Load TV show model  
        if tv_model_path is None:
            tv_model_path = self.models_dir / "best_tv_model.pt"
            tv_metadata_path = self.models_dir / "tv_metadata.pkl"
        else:
            tv_metadata_path = tv_model_path.replace('.pt', '_metadata.pkl')
        
        if Path(tv_model_path).exists():
            try:
                self.tv_model, self.tv_metadata = load_tv_model(
                    str(tv_model_path), str(tv_metadata_path)
                )
                logger.info("Loaded TV show recommendation model")
                
                # Load TV encoders
                tv_encoders_path = self.models_dir / "tv_encoders.pkl"
                if tv_encoders_path.exists():
                    with open(tv_encoders_path, 'rb') as f:
                        self.tv_encoders = pickle.load(f)
                
            except Exception as e:
                logger.error(f"Failed to load TV model: {e}")
    
    def get_recommendations(self, user_id: int, content_type: str = "mixed", 
                          top_k: int = 10, **kwargs) -> List[Dict]:
        """
        Get recommendations for a user
        
        Args:
            user_id: User ID
            content_type: "movie", "tv", or "mixed"
            top_k: Number of recommendations
            **kwargs: Additional parameters (genres, similar_to, etc.)
            
        Returns:
            List of recommendation dictionaries
        """
        
        if content_type == "movie":
            return self._get_movie_recommendations(user_id, top_k, **kwargs)
        elif content_type == "tv":
            return self._get_tv_recommendations(user_id, top_k, **kwargs)
        elif content_type == "mixed":
            return self._get_mixed_recommendations(user_id, top_k, **kwargs)
        else:
            raise ValueError(f"Unknown content type: {content_type}")
    
    def _get_movie_recommendations(self, user_id: int, top_k: int, **kwargs) -> List[Dict]:
        """Get movie recommendations"""
        if self.movie_model is None:
            logger.warning("Movie model not loaded")
            return []
        
        try:
            # Load movie data
            movie_data = self._load_movie_data()
            if movie_data.empty:
                return []
            
            # Get user encoding
            encoded_user_id = self._encode_user_id(user_id, 'movie')
            if encoded_user_id is None:
                # Handle new user with content-based recommendations
                return self._get_content_based_movie_recommendations(top_k, **kwargs)
            
            # Get collaborative filtering recommendations
            movie_ids = movie_data['movie_id_encoded'].values
            user_tensor = torch.tensor([encoded_user_id] * len(movie_ids), dtype=torch.long)
            movie_tensor = torch.tensor(movie_ids, dtype=torch.long)
            
            # Get predictions
            self.movie_model.eval()
            with torch.no_grad():
                predictions = self.movie_model(user_tensor, movie_tensor)
                predictions = predictions.cpu().numpy()
            
            # Sort and return top-k
            movie_scores = list(zip(movie_data['movie_id'].values, predictions))
            movie_scores.sort(key=lambda x: x[1], reverse=True)
            
            # Convert to recommendation format
            recommendations = []
            for movie_id, score in movie_scores[:top_k]:
                movie_info = movie_data[movie_data['movie_id'] == movie_id].iloc[0]
                recommendations.append({
                    'id': movie_id,
                    'title': movie_info.get('title', 'Unknown'),
                    'genres': movie_info.get('genres', ''),
                    'score': float(score),
                    'type': 'movie'
                })
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error getting movie recommendations: {e}")
            return []
    
    def _get_tv_recommendations(self, user_id: int, top_k: int, **kwargs) -> List[Dict]:
        """Get TV show recommendations"""
        if self.tv_model is None:
            logger.warning("TV model not loaded")
            return []
        
        try:
            # Load TV show data
            tv_data = self._load_tv_data()
            if tv_data.empty:
                return []
            
            # Get user encoding
            encoded_user_id = self._encode_user_id(user_id, 'tv')
            if encoded_user_id is None:
                # Handle new user with content-based recommendations
                return self._get_content_based_tv_recommendations(top_k, **kwargs)
            
            # Get recommendations using TV model
            recommendations = self.tv_model.get_user_recommendations(
                encoded_user_id, tv_data, top_k
            )
            
            # Convert to standard format
            result = []
            for show_id, score in recommendations:
                show_info = tv_data[tv_data['show_id'] == show_id].iloc[0]
                result.append({
                    'id': show_id,
                    'title': show_info.get('title', 'Unknown'),
                    'genres': show_info.get('genres', ''),
                    'score': float(score),
                    'type': 'tv_show'
                })
            
            return result
            
        except Exception as e:
            logger.error(f"Error getting TV recommendations: {e}")
            return []
    
    def _get_mixed_recommendations(self, user_id: int, top_k: int, **kwargs) -> List[Dict]:
        """Get mixed movie and TV show recommendations"""
        # Get both types of recommendations
        movie_count = top_k // 2
        tv_count = top_k - movie_count
        
        movie_recs = self._get_movie_recommendations(user_id, movie_count, **kwargs)
        tv_recs = self._get_tv_recommendations(user_id, tv_count, **kwargs)
        
        # Combine and sort by score
        all_recs = movie_recs + tv_recs
        all_recs.sort(key=lambda x: x['score'], reverse=True)
        
        return all_recs[:top_k]
    
    def get_similar_content(self, content_id: Union[str, int], content_type: str, 
                          top_k: int = 10) -> List[Dict]:
        """
        Get content similar to the given content
        
        Args:
            content_id: ID of the reference content
            content_type: "movie" or "tv"
            top_k: Number of similar items to return
        """
        
        if content_type == "movie":
            return self._get_similar_movies(content_id, top_k)
        elif content_type == "tv":
            return self._get_similar_tv_shows(content_id, top_k)
        else:
            raise ValueError(f"Unknown content type: {content_type}")
    
    def get_cross_content_recommendations(self, user_id: int, source_type: str, 
                                        target_type: str, top_k: int = 10) -> List[Dict]:
        """
        Get recommendations for target_type based on user preferences in source_type
        E.g., recommend TV shows based on movie preferences
        """
        
        if source_type == "movie" and target_type == "tv":
            return self._movie_to_tv_recommendations(user_id, top_k)
        elif source_type == "tv" and target_type == "movie":
            return self._tv_to_movie_recommendations(user_id, top_k)
        else:
            return self.get_recommendations(user_id, target_type, top_k)
    
    def _movie_to_tv_recommendations(self, user_id: int, top_k: int) -> List[Dict]:
        """Recommend TV shows based on movie preferences"""
        # Get user's movie preferences
        movie_preferences = self._analyze_user_movie_preferences(user_id)
        
        # Use preferences to filter/rank TV shows
        tv_recs = self._get_tv_recommendations(user_id, top_k * 2)
        
        # Re-rank based on movie preferences
        for rec in tv_recs:
            preference_boost = self._calculate_cross_content_boost(
                rec['genres'], movie_preferences, 'movie_to_tv'
            )
            rec['score'] = min(1.0, rec['score'] * (1 + preference_boost))
        
        # Sort and return top-k
        tv_recs.sort(key=lambda x: x['score'], reverse=True)
        return tv_recs[:top_k]
    
    def _tv_to_movie_recommendations(self, user_id: int, top_k: int) -> List[Dict]:
        """Recommend movies based on TV show preferences"""
        # Get user's TV preferences
        tv_preferences = self._analyze_user_tv_preferences(user_id)
        
        # Use preferences to filter/rank movies
        movie_recs = self._get_movie_recommendations(user_id, top_k * 2)
        
        # Re-rank based on TV preferences
        for rec in movie_recs:
            preference_boost = self._calculate_cross_content_boost(
                rec['genres'], tv_preferences, 'tv_to_movie'
            )
            rec['score'] = min(1.0, rec['score'] * (1 + preference_boost))
        
        # Sort and return top-k
        movie_recs.sort(key=lambda x: x['score'], reverse=True)
        return movie_recs[:top_k]
    
    def _encode_user_id(self, user_id: int, content_type: str) -> Optional[int]:
        """Encode user ID for the specified content type"""
        if content_type == 'movie' and self.movie_encoders:
            return self.movie_encoders.get('user_id', {}).get(user_id)
        elif content_type == 'tv' and self.tv_encoders:
            return self.tv_encoders.get('user_id', {}).get(user_id)
        return None
    
    def _load_movie_data(self) -> pd.DataFrame:
        """Load movie data for recommendations"""
        try:
            movie_data_path = self.models_dir / "movies_data.csv"
            if movie_data_path.exists():
                return pd.read_csv(movie_data_path)
        except Exception as e:
            logger.error(f"Error loading movie data: {e}")
        return pd.DataFrame()
    
    def _load_tv_data(self) -> pd.DataFrame:
        """Load TV show data for recommendations"""
        try:
            tv_data_path = Path("tv/processed/tv_shows_data.csv")
            if tv_data_path.exists():
                return pd.read_csv(tv_data_path)
        except Exception as e:
            logger.error(f"Error loading TV data: {e}")
        return pd.DataFrame()
    
    def _get_content_based_movie_recommendations(self, top_k: int, **kwargs) -> List[Dict]:
        """Get content-based movie recommendations for new users"""
        # Implement content-based filtering based on genres, popularity, etc.
        movie_data = self._load_movie_data()
        if movie_data.empty:
            return []
        
        # Simple popularity-based recommendations for new users
        if 'rating' in movie_data.columns and 'rating_count' in movie_data.columns:
            movie_data['weighted_rating'] = (
                movie_data['rating'] * np.log1p(movie_data['rating_count'])
            )
            top_movies = movie_data.nlargest(top_k, 'weighted_rating')
            
            recommendations = []
            for _, movie in top_movies.iterrows():
                recommendations.append({
                    'id': movie.get('movie_id', movie.get('id')),
                    'title': movie.get('title', 'Unknown'),
                    'genres': movie.get('genres', ''),
                    'score': float(movie.get('weighted_rating', 0)),
                    'type': 'movie'
                })
            
            return recommendations
        
        return []
    
    def _get_content_based_tv_recommendations(self, top_k: int, **kwargs) -> List[Dict]:
        """Get content-based TV show recommendations for new users"""
        # Similar to movies but for TV shows
        tv_data = self._load_tv_data()
        if tv_data.empty:
            return []
        
        # Simple popularity-based recommendations for new users
        if 'rating' in tv_data.columns and 'rating_count' in tv_data.columns:
            tv_data['weighted_rating'] = (
                tv_data['rating'] * np.log1p(tv_data['rating_count'])
            )
            top_shows = tv_data.nlargest(top_k, 'weighted_rating')
            
            recommendations = []
            for _, show in top_shows.iterrows():
                recommendations.append({
                    'id': show.get('show_id', show.get('id')),
                    'title': show.get('title', 'Unknown'),
                    'genres': show.get('genres', ''),
                    'score': float(show.get('weighted_rating', 0)),
                    'type': 'tv_show'
                })
            
            return recommendations
        
        return []
    
    def _analyze_user_movie_preferences(self, user_id: int) -> Dict:
        """Analyze user's movie preferences from their rating history"""
        # This would analyze the user's movie rating history
        # and return genre preferences, rating patterns, etc.
        return {
            'preferred_genres': ['Action', 'Drama'],  # Placeholder
            'avg_rating': 4.2,
            'rating_count': 150
        }
    
    def _analyze_user_tv_preferences(self, user_id: int) -> Dict:
        """Analyze user's TV show preferences from their rating history"""
        # This would analyze the user's TV rating history
        return {
            'preferred_genres': ['Drama', 'Comedy'],  # Placeholder
            'avg_rating': 4.0,
            'rating_count': 75
        }
    
    def _calculate_cross_content_boost(self, content_genres: str, user_preferences: Dict, 
                                     direction: str) -> float:
        """Calculate boost factor for cross-content recommendations"""
        if not content_genres or not user_preferences.get('preferred_genres'):
            return 0.0
        
        content_genre_list = [g.strip() for g in content_genres.split('|')]
        preferred_genres = user_preferences['preferred_genres']
        
        # Calculate genre overlap
        overlap = len(set(content_genre_list) & set(preferred_genres))
        total_genres = len(set(content_genre_list) | set(preferred_genres))
        
        if total_genres == 0:
            return 0.0
        
        # Jaccard similarity
        similarity = overlap / total_genres
        
        # Apply cross-content weight
        weight = self.cross_content_weights.get(direction, 0.3)
        
        return similarity * weight
    
    def _get_similar_movies(self, movie_id: Union[str, int], top_k: int) -> List[Dict]:
        """Get movies similar to the given movie"""
        # Implement movie similarity using embeddings or content features
        return []
    
    def _get_similar_tv_shows(self, show_id: Union[str, int], top_k: int) -> List[Dict]:
        """Get TV shows similar to the given show"""
        # Implement TV show similarity using embeddings or content features
        return []
    
    def is_ready(self) -> Dict[str, bool]:
        """Check if models are ready for inference"""
        return {
            'movie_model_loaded': self.movie_model is not None,
            'tv_model_loaded': self.tv_model is not None,
            'movie_encoders_loaded': self.movie_encoders is not None,
            'tv_encoders_loaded': self.tv_encoders is not None
        }
    
    def get_model_info(self) -> Dict:
        """Get information about loaded models"""
        info = {
            'models_loaded': [],
            'total_parameters': 0
        }
        
        if self.movie_model:
            info['models_loaded'].append('movie')
            info['movie_metadata'] = self.movie_metadata
            info['total_parameters'] += sum(p.numel() for p in self.movie_model.parameters())
        
        if self.tv_model:
            info['models_loaded'].append('tv')
            info['tv_metadata'] = self.tv_metadata
            info['total_parameters'] += sum(p.numel() for p in self.tv_model.parameters())
        
        return info