"""
Simplified content manager with reduced complexity
"""
import torch
import numpy as np
import pandas as pd
import pickle
import os
import logging
from typing import List, Tuple, Optional, Dict, Any

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.recommendation_base import MovieRecommender, TVRecommender, interleave_results

logger = logging.getLogger(__name__)


class SimpleContentManager:
    """
    Simplified unified content manager for movies and TV shows
    Reduced complexity while maintaining core functionality
    """
    
    def __init__(self, models_dir: str, device: str = None):
        self.models_dir = models_dir
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        
        # Recommenders
        self.movie_recommender = None
        self.tv_recommender = None
        
        # Genre data
        self.all_genres = []
        
        logger.info(f"SimpleContentManager initialized with device: {self.device}")
    
    def load_all(self):
        """Load all models and data"""
        try:
            self._load_movie_components()
            self._load_tv_components()
            self._combine_genres()
            logger.info("All components loaded successfully")
        except Exception as e:
            logger.error(f"Error loading components: {e}")
            logger.info("Continuing with available data")
    
    def _load_movie_components(self):
        """Load movie data and model"""
        try:
            # Load movie lookup
            movie_lookup_path = os.path.join(self.models_dir, 'movie_lookup.pkl')
            if os.path.exists(movie_lookup_path):
                with open(movie_lookup_path, 'rb') as f:
                    movie_lookup = pickle.load(f)
                logger.info(f"Loaded {len(movie_lookup)} movies")
            else:
                movie_lookup = {}
                logger.warning("No movie lookup found")
            
            # Load movie mappings
            mappings_path = os.path.join(self.models_dir, 'id_mappings.pkl')
            movie_mappings = {}
            if os.path.exists(mappings_path):
                with open(mappings_path, 'rb') as f:
                    movie_mappings = pickle.load(f)
            
            # Load movie model
            model = None
            scaler = None
            
            model_path = os.path.join(self.models_dir, 'best_model.pt')
            if os.path.exists(model_path) and movie_mappings:
                try:
                    from .hybrid_recommender import HybridRecommenderModel
                    
                    model = HybridRecommenderModel(
                        num_users=movie_mappings.get('num_users', 1),
                        num_items=movie_mappings.get('num_movies', movie_mappings.get('num_content', 1)),
                        embedding_dim=128
                    )
                    
                    checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
                    model.load_state_dict(checkpoint['model_state_dict'])
                    model.to(self.device)
                    model.eval()
                    logger.info("Movie model loaded")
                except Exception as e:
                    logger.warning(f"Could not load movie model: {e}")
            
            # Load scaler
            scaler_path = os.path.join(self.models_dir, 'rating_scaler.pkl')
            if os.path.exists(scaler_path):
                with open(scaler_path, 'rb') as f:
                    scaler = pickle.load(f)
            
            # Create movie recommender
            self.movie_recommender = MovieRecommender(movie_lookup, movie_mappings, model, scaler)
            
            # Extract movie genres
            self.movie_genres = self._extract_genres_from_lookup(movie_lookup)
            
        except Exception as e:
            logger.error(f"Error loading movie components: {e}")
            self.movie_recommender = MovieRecommender({}, {})
            self.movie_genres = []
    
    def _load_tv_components(self):
        """Load TV show data and model"""
        try:
            # Load TV lookup
            tv_lookup_path = os.path.join(self.models_dir, 'tv_lookup.pkl')
            if os.path.exists(tv_lookup_path):
                with open(tv_lookup_path, 'rb') as f:
                    tv_lookup = pickle.load(f)
                logger.info(f"Loaded {len(tv_lookup)} TV shows")
            else:
                tv_lookup = {}
                logger.warning("No TV lookup found")
            
            # Load TV mappings (if available)
            tv_mappings_path = os.path.join(self.models_dir, 'tv_id_mappings.pkl')
            tv_mappings = {}
            if os.path.exists(tv_mappings_path):
                with open(tv_mappings_path, 'rb') as f:
                    tv_mappings = pickle.load(f)
            
            # TV model loading would go here (similar pattern)
            # For now, use fallback only
            self.tv_recommender = TVRecommender(tv_lookup, tv_mappings)
            
            # Extract TV genres
            self.tv_genres = self._extract_genres_from_lookup(tv_lookup)
            
        except Exception as e:
            logger.error(f"Error loading TV components: {e}")
            self.tv_recommender = TVRecommender({}, {})
            self.tv_genres = []
    
    def _extract_genres_from_lookup(self, lookup_data: Dict) -> List[str]:
        """Extract unique genres from lookup data"""
        all_genres = set()
        
        for item_info in lookup_data.values():
            genres = item_info.get('genres', '')
            if isinstance(genres, str) and genres.strip():
                all_genres.update(genre.strip() for genre in genres.split('|') if genre.strip())
        
        return sorted(list(all_genres))
    
    def _combine_genres(self):
        """Combine all genres"""
        movie_genres = getattr(self, 'movie_genres', [])
        tv_genres = getattr(self, 'tv_genres', [])
        self.all_genres = sorted(list(set(movie_genres + tv_genres)))
        logger.info(f"Combined {len(self.all_genres)} total genres")
    
    def get_recommendations(self, user_id: int, content_type: str = "mixed", 
                          top_k: int = 10, genre_filter: str = None) -> List[Tuple[str, str, str, float]]:
        """
        Get recommendations with simplified logic
        
        Args:
            user_id: User ID for personalized recommendations
            content_type: "movie", "tv", or "mixed"
            top_k: Number of recommendations to return
            genre_filter: Optional genre filter
            
        Returns:
            List of (content_id, title, content_type, score) tuples
        """
        try:
            if content_type == "movie":
                return self.movie_recommender.get_recommendations(user_id, top_k, genre_filter)
            
            elif content_type == "tv":
                return self.tv_recommender.get_recommendations(user_id, top_k, genre_filter)
            
            elif content_type == "mixed":
                # Get recommendations from both
                movie_recs = self.movie_recommender.get_recommendations(user_id, top_k, genre_filter)
                tv_recs = self.tv_recommender.get_recommendations(user_id, top_k, genre_filter)
                
                # Interleave results
                return interleave_results(movie_recs, tv_recs, top_k)
            
            else:
                logger.warning(f"Unknown content type: {content_type}")
                return []
                
        except Exception as e:
            logger.error(f"Error getting recommendations: {e}")
            return []
    
    def get_similar_content(self, content_id: str, content_type: str, 
                          top_k: int = 10) -> List[Tuple[str, str, str, float]]:
        """Get similar content"""
        try:
            if content_type == "movie":
                return self.movie_recommender.get_similar_items(content_id, top_k)
            elif content_type == "tv":
                return self.tv_recommender.get_similar_items(content_id, top_k)
            else:
                return []
        except Exception as e:
            logger.error(f"Error getting similar content: {e}")
            return []
    
    def get_cross_content_recommendations(self, user_id: int, source_type: str, 
                                        target_type: str, top_k: int = 10) -> List[Tuple[str, str, str, float]]:
        """Cross-content recommendations using genre preferences"""
        # Simplified approach: just use general recommendations for target type
        return self.get_recommendations(user_id, target_type, top_k)
    
    def get_available_genres(self, content_type: str = "mixed") -> List[str]:
        """Get available genres"""
        if content_type == "movie":
            return getattr(self, 'movie_genres', [])
        elif content_type == "tv":
            return getattr(self, 'tv_genres', [])
        else:
            return self.all_genres
    
    def get_model_status(self) -> Dict[str, Any]:
        """Get status information"""
        movie_data_count = len(self.movie_recommender.lookup_data) if self.movie_recommender else 0
        tv_data_count = len(self.tv_recommender.lookup_data) if self.tv_recommender else 0
        movie_model_loaded = (self.movie_recommender and 
                            self.movie_recommender.model is not None) if self.movie_recommender else False
        tv_model_loaded = (self.tv_recommender and 
                         self.tv_recommender.model is not None) if self.tv_recommender else False
        
        return {
            'movie_model_loaded': movie_model_loaded,
            'tv_model_loaded': tv_model_loaded,
            'movie_count': movie_data_count,
            'tv_count': tv_data_count,
            'total_genres': len(self.all_genres),
            'device': str(self.device)
        }