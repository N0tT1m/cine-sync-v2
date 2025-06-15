"""
Unified Content Manager for Lupe Discord Bot
Integrates with the new unified inference API for all CineSync models
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import pandas as pd
import pickle
import os
import logging
from typing import List, Tuple, Optional, Dict, Any, Union
from pathlib import Path

# Import the unified API
from unified_inference_api import UnifiedRecommendationAPI, ModelType, RecommendationResult

logger = logging.getLogger(__name__)


class UnifiedLupeContentManager:
    """
    Unified content manager that uses the new multi-model inference API
    Provides backward compatibility with existing bot code while adding new features
    """
    
    def __init__(self, models_dir: str = None, device: str = None, config_path: str = None):
        self.models_dir = models_dir or "."
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Use default config path if not provided
        if config_path is None:
            config_path = os.path.join(os.path.dirname(__file__), "unified_config.json")
        
        # Initialize unified API
        self.unified_api = UnifiedRecommendationAPI(config_path=config_path, device=self.device)
        
        # Legacy compatibility data structures
        self.movie_lookup = {}
        self.tv_lookup = {}
        self.movie_mappings = {}
        self.tv_mappings = {}
        self.movie_metadata = {}
        self.tv_metadata = {}
        self.movie_rating_scaler = None
        self.tv_rating_scaler = None
        self.movie_genres = []
        self.tv_genres = []
        self.all_genres = []
        
        # Data frames
        self.movies_df = None
        self.tv_df = None
        
        # Legacy model references (for compatibility)
        self.movie_model = None
        self.tv_model = None
        
        logger.info(f"UnifiedLupeContentManager initialized with device: {self.device}")
    
    def load_models(self):
        """Load all models and data"""
        try:
            # Load new unified models
            self.unified_api.load_all_models()
            
            # Load legacy data for compatibility
            self._load_legacy_data()
            
            logger.info("Unified models and legacy data loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading unified models: {e}")
            # Try to load legacy data only
            try:
                self._load_legacy_data()
                logger.info("Loaded legacy data only (no unified models)")
            except Exception as e2:
                logger.error(f"Error loading legacy data: {e2}")
    
    def _load_legacy_data(self):
        """Load legacy data structures for backward compatibility"""
        try:
            # Try to load movie lookup
            movie_lookup_path = os.path.join(self.models_dir, "movie_lookup.pkl")
            if os.path.exists(movie_lookup_path):
                with open(movie_lookup_path, 'rb') as f:
                    self.movie_lookup = pickle.load(f)
                logger.info(f"Loaded {len(self.movie_lookup)} movies from lookup")
            
            # Try to load TV lookup (might not exist)
            tv_lookup_path = os.path.join(self.models_dir, "tv_lookup.pkl")
            if os.path.exists(tv_lookup_path):
                with open(tv_lookup_path, 'rb') as f:
                    self.tv_lookup = pickle.load(f)
                logger.info(f"Loaded {len(self.tv_lookup)} TV shows from lookup")
            
            # Load movie genres from data
            if self.movie_lookup:
                movie_genres = set()
                for movie_info in self.movie_lookup.values():
                    if 'genres' in movie_info:
                        if isinstance(movie_info['genres'], list):
                            movie_genres.update(movie_info['genres'])
                        elif isinstance(movie_info['genres'], str):
                            movie_genres.update(movie_info['genres'].split('|'))
                self.movie_genres = sorted(list(movie_genres))
            
            # Load TV genres from data  
            if self.tv_lookup:
                tv_genres = set()
                for tv_info in self.tv_lookup.values():
                    if 'genres' in tv_info:
                        if isinstance(tv_info['genres'], list):
                            tv_genres.update(tv_info['genres'])
                        elif isinstance(tv_info['genres'], str):
                            tv_genres.update(tv_info['genres'].split('|'))
                self.tv_genres = sorted(list(tv_genres))
            
            # Combine all genres
            all_genres = set(self.movie_genres + self.tv_genres)
            self.all_genres = sorted(list(all_genres))
            
            logger.info(f"Loaded genres - Movies: {len(self.movie_genres)}, TV: {len(self.tv_genres)}, Total: {len(self.all_genres)}")
            
        except Exception as e:
            logger.error(f"Error loading legacy data: {e}")
            # Fallback genres
            self.all_genres = ['Action', 'Adventure', 'Animation', 'Children', 'Comedy', 'Crime', 
                              'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical', 
                              'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
    
    def get_recommendations(self, user_id: int = None, limit: int = 5, content_type: str = "mixed", 
                          genre: str = None, model_type: str = None) -> List[Tuple]:
        """
        Get recommendations using the unified API
        
        Args:
            user_id: User ID for personalized recommendations
            limit: Number of recommendations to return
            content_type: Type of content ("movie", "tv", "mixed")
            genre: Genre filter (optional)
            model_type: Specific model to use ("ncf", "sequential", "two_tower", "ensemble")
            
        Returns:
            List of tuples: (content_id, title, content_type, score)
        """
        try:
            # Map string model type to enum
            model_enum = None
            if model_type:
                model_map = {
                    "ncf": ModelType.NCF,
                    "neural_collaborative_filtering": ModelType.NCF,
                    "sequential": ModelType.SEQUENTIAL, 
                    "two_tower": ModelType.TWO_TOWER,
                    "ensemble": ModelType.ENSEMBLE
                }
                model_enum = model_map.get(model_type.lower())
            
            # Get recommendations from unified API
            if user_id is not None:
                recommendations = self.unified_api.get_recommendations(
                    user_id=user_id,
                    top_k=limit,
                    model_type=model_enum
                )
            else:
                # For anonymous users, use collaborative filtering approach
                recommendations = self.unified_api.get_recommendations(
                    user_id=1,  # Default user for general recommendations
                    top_k=limit * 2,  # Get more to filter
                    model_type=model_enum
                )
            
            # Convert to legacy format and apply filters
            results = []
            for rec in recommendations:
                content_id = rec.item_id
                
                # Determine content type and get metadata
                content_info = None
                detected_type = "movie"  # Default
                
                if content_id in self.movie_lookup:
                    content_info = self.movie_lookup[content_id]
                    detected_type = "movie"
                elif content_id in self.tv_lookup:
                    content_info = self.tv_lookup[content_id]
                    detected_type = "tv"
                else:
                    # Skip if we don't have metadata
                    continue
                
                # Apply content type filter
                if content_type != "mixed" and detected_type != content_type:
                    continue
                
                # Apply genre filter
                if genre:
                    item_genres = content_info.get('genres', [])
                    if isinstance(item_genres, str):
                        item_genres = item_genres.split('|')
                    if genre not in item_genres:
                        continue
                
                title = content_info.get('title', f"Unknown {detected_type.title()} {content_id}")
                
                results.append((content_id, title, detected_type, rec.score))
                
                if len(results) >= limit:
                    break
            
            return results
            
        except Exception as e:
            logger.error(f"Error getting recommendations: {e}")
            return self._fallback_recommendations(limit, content_type, genre)
    
    def get_similar_content(self, content_id: int, content_type: str, limit: int = 5) -> List[Tuple]:
        """Get similar content using unified API"""
        try:
            similar_items = self.unified_api.find_similar_items(
                item_id=content_id,
                top_k=limit
            )
            
            results = []
            for rec in similar_items:
                similar_id = rec.item_id
                
                # Get metadata
                content_info = None
                if content_type == "movie" and similar_id in self.movie_lookup:
                    content_info = self.movie_lookup[similar_id]
                elif content_type == "tv" and similar_id in self.tv_lookup:
                    content_info = self.tv_lookup[similar_id]
                
                if content_info:
                    title = content_info.get('title', f"Unknown {content_type.title()} {similar_id}")
                    results.append((similar_id, title, content_type, rec.score))
            
            return results
            
        except Exception as e:
            logger.error(f"Error getting similar content: {e}")
            return []
    
    def predict_next_items(self, sequence: List[int], limit: int = 5) -> List[Tuple]:
        """Predict next items in sequence (for sequential model)"""
        try:
            predictions = self.unified_api.predict_next_items(
                sequence=sequence,
                top_k=limit,
                model_type=ModelType.SEQUENTIAL
            )
            
            results = []
            for rec in predictions:
                content_id = rec.item_id
                
                # Get metadata
                content_info = None
                content_type = "movie"
                
                if content_id in self.movie_lookup:
                    content_info = self.movie_lookup[content_id]
                    content_type = "movie"
                elif content_id in self.tv_lookup:
                    content_info = self.tv_lookup[content_id]
                    content_type = "tv"
                
                if content_info:
                    title = content_info.get('title', f"Unknown {content_type.title()} {content_id}")
                    results.append((content_id, title, content_type, rec.score))
            
            return results
            
        except Exception as e:
            logger.error(f"Error predicting next items: {e}")
            return []
    
    def compare_models(self, user_id: int, limit: int = 5) -> Dict[str, List[Tuple]]:
        """Compare recommendations from different models"""
        try:
            comparison = self.unified_api.compare_models(user_id=user_id, top_k=limit)
            
            # Convert to legacy format
            results = {}
            for model_name, recommendations in comparison.items():
                results[model_name] = []
                for rec in recommendations:
                    content_id = rec.item_id
                    
                    # Get metadata
                    content_info = None
                    content_type = "movie"
                    
                    if content_id in self.movie_lookup:
                        content_info = self.movie_lookup[content_id]
                        content_type = "movie"
                    elif content_id in self.tv_lookup:
                        content_info = self.tv_lookup[content_id]
                        content_type = "tv"
                    
                    if content_info:
                        title = content_info.get('title', f"Unknown {content_type.title()} {content_id}")
                        results[model_name].append((content_id, title, content_type, rec.score))
            
            return results
            
        except Exception as e:
            logger.error(f"Error comparing models: {e}")
            return {}
    
    def get_model_health(self) -> Dict[str, bool]:
        """Get health status of all models"""
        try:
            return self.unified_api.health_check()
        except Exception as e:
            logger.error(f"Error checking model health: {e}")
            return {}
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about loaded models"""
        try:
            info = self.unified_api.get_model_info()
            
            # Add legacy data info
            info['legacy_data'] = {
                'movie_count': len(self.movie_lookup),
                'tv_count': len(self.tv_lookup),
                'movie_genres': len(self.movie_genres),
                'tv_genres': len(self.tv_genres),
                'total_genres': len(self.all_genres)
            }
            
            return info
            
        except Exception as e:
            logger.error(f"Error getting model info: {e}")
            return {'legacy_data': {
                'movie_count': len(self.movie_lookup),
                'tv_count': len(self.tv_lookup),
                'total_genres': len(self.all_genres)
            }}
    
    def _fallback_recommendations(self, limit: int, content_type: str, genre: str = None) -> List[Tuple]:
        """Fallback recommendations when unified API fails"""
        try:
            results = []
            
            # Use movie data if available
            if content_type in ["mixed", "movie"] and self.movie_lookup:
                for content_id, content_info in list(self.movie_lookup.items())[:limit * 2]:
                    if genre:
                        item_genres = content_info.get('genres', [])
                        if isinstance(item_genres, str):
                            item_genres = item_genres.split('|')
                        if genre not in item_genres:
                            continue
                    
                    title = content_info.get('title', f"Movie {content_id}")
                    results.append((content_id, title, "movie", 0.5))
                    
                    if len(results) >= limit:
                        break
            
            # Use TV data if needed
            if len(results) < limit and content_type in ["mixed", "tv"] and self.tv_lookup:
                for content_id, content_info in list(self.tv_lookup.items())[:limit * 2]:
                    if genre:
                        item_genres = content_info.get('genres', [])
                        if isinstance(item_genres, str):
                            item_genres = item_genres.split('|')
                        if genre not in item_genres:
                            continue
                    
                    title = content_info.get('title', f"TV Show {content_id}")
                    results.append((content_id, title, "tv", 0.5))
                    
                    if len(results) >= limit:
                        break
            
            return results[:limit]
            
        except Exception as e:
            logger.error(f"Error in fallback recommendations: {e}")
            return []
    
    # Legacy compatibility methods
    def get_available_genres(self, content_type: str = "mixed") -> List[str]:
        """Get available genres for content type"""
        if content_type == "movie":
            return self.movie_genres
        elif content_type == "tv":
            return self.tv_genres
        else:
            return self.all_genres
    
    def get_model_status(self) -> Dict[str, Any]:
        """Get model status (legacy compatibility)"""
        # Check if any models are loaded for legacy compatibility
        has_models = len(self.unified_api.models) > 0
        
        return {
            'movie_count': len(self.movie_lookup),
            'tv_count': len(self.tv_lookup),
            'movie_genres': len(self.movie_genres),
            'tv_genres': len(self.tv_genres),
            'total_genres': len(self.all_genres),
            'models_loaded': len(self.unified_api.models),
            'movie_model_loaded': has_models,  # For legacy compatibility
            'tv_model_loaded': has_models,     # For legacy compatibility  
            'device': str(self.device)
        }
    
    def get_cross_content_recommendations(self, user_id: int = None, limit: int = 5) -> List[Tuple]:
        """Get cross-content recommendations (movies + TV)"""
        return self.get_recommendations(user_id=user_id, limit=limit, content_type="mixed")