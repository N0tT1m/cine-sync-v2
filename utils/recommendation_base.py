"""
Base recommendation utilities to reduce code duplication
"""
import numpy as np
import torch
import logging
from typing import List, Dict, Tuple, Any, Optional

logger = logging.getLogger(__name__)


class BaseRecommender:
    """Base class for recommendation logic to reduce duplication"""
    
    def __init__(self, lookup_data: Dict, mappings: Dict, model=None, scaler=None):
        self.lookup_data = lookup_data
        self.mappings = mappings 
        self.model = model
        self.scaler = scaler
    
    def get_recommendations(self, user_id: int, limit: int, genre_filter: str = None) -> List[Tuple[str, str, str, float]]:
        """Get recommendations with fallback if model not available"""
        if self.model and self.mappings:
            return self._get_model_recommendations(user_id, limit, genre_filter)
        else:
            return self._get_fallback_recommendations(user_id, limit, genre_filter)
    
    def _get_model_recommendations(self, user_id: int, limit: int, genre_filter: str = None) -> List[Tuple[str, str, str, float]]:
        """Get model-based recommendations (override in subclass)"""
        raise NotImplementedError
    
    def _get_fallback_recommendations(self, user_id: int, limit: int, genre_filter: str = None) -> List[Tuple[str, str, str, float]]:
        """Get fallback recommendations when model unavailable"""
        if not self.lookup_data:
            return []
        
        recommendations = []
        items = list(self.lookup_data.items())
        np.random.shuffle(items)
        
        for item_id, item_info in items:
            genres_value = item_info.get('genres', '')
            
            # Apply genre filter
            if genre_filter and isinstance(genres_value, str):
                if genre_filter.lower() not in genres_value.lower():
                    continue
            
            score = np.random.uniform(0.5, 1.0)
            content_type = getattr(self, 'content_type', 'unknown')
            
            recommendations.append((
                str(item_id),
                item_info.get('title', 'Unknown'),
                content_type,
                score
            ))
            
            if len(recommendations) >= limit:
                break
        
        recommendations.sort(key=lambda x: x[3], reverse=True)
        return recommendations[:limit]
    
    def get_similar_items(self, item_id: str, limit: int) -> List[Tuple[str, str, str, float]]:
        """Get similar items using genre-based similarity"""
        try:
            item_id = int(item_id)
            if item_id not in self.lookup_data:
                return []
            
            target_item = self.lookup_data[item_id]
            target_genres = target_item.get('genres', '')
            
            if not isinstance(target_genres, str):
                return []
            
            target_genre_set = set(target_genres.split('|')) if target_genres else set()
            similarities = []
            
            for iid, item_info in self.lookup_data.items():
                if iid == item_id:
                    continue
                
                item_genres = item_info.get('genres', '')
                if not isinstance(item_genres, str):
                    continue
                
                item_genre_set = set(item_genres.split('|')) if item_genres else set()
                
                # Calculate similarity
                if target_genre_set and item_genre_set:
                    similarity = len(target_genre_set & item_genre_set) / len(target_genre_set | item_genre_set)
                else:
                    similarity = 0
                
                content_type = getattr(self, 'content_type', 'unknown')
                similarities.append((
                    str(iid),
                    item_info.get('title', 'Unknown'),
                    content_type,
                    max(0, min(1, similarity + np.random.uniform(0, 0.05)))
                ))
            
            similarities.sort(key=lambda x: x[3], reverse=True)
            return similarities[:limit]
            
        except Exception as e:
            logger.error(f"Error getting similar items: {e}")
            return []


class MovieRecommender(BaseRecommender):
    """Movie-specific recommender"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.content_type = 'movie'
    
    def _get_model_recommendations(self, user_id: int, limit: int, genre_filter: str = None) -> List[Tuple[str, str, str, float]]:
        """Movie model recommendations"""
        if not self.model or not self.mappings:
            return []
        
        try:
            # Get user index
            if user_id not in self.mappings['user_id_to_idx']:
                user_id = list(self.mappings['user_id_to_idx'].keys())[0]  # Use fallback user
            
            user_idx = self.mappings['user_id_to_idx'][user_id]
            recommendations = []
            
            # Sample movies for efficiency
            available_ids = list(self.lookup_data.keys())
            sample_size = min(500, len(available_ids))
            sampled_ids = np.random.choice(available_ids, sample_size, replace=False)
            
            device = next(self.model.parameters()).device
            
            with torch.no_grad():
                for movie_id in sampled_ids:
                    if movie_id not in self.mappings.get('movie_id_to_idx', {}):
                        continue
                    
                    movie_info = self.lookup_data[movie_id]
                    genres_value = movie_info.get('genres', '')
                    
                    # Apply genre filter
                    if genre_filter and isinstance(genres_value, str):
                        if genre_filter.lower() not in genres_value.lower():
                            continue
                    
                    movie_idx = self.mappings['movie_id_to_idx'][movie_id]
                    
                    # Get prediction
                    user_tensor = torch.tensor([user_idx], dtype=torch.long, device=device)
                    movie_tensor = torch.tensor([movie_idx], dtype=torch.long, device=device)
                    
                    score = self.model(user_tensor, movie_tensor).item()
                    
                    # Inverse transform if scaler available
                    if self.scaler:
                        score = self.scaler.inverse_transform([[score]])[0][0]
                    
                    recommendations.append((
                        str(movie_id),
                        movie_info.get('title', 'Unknown'),
                        'movie',
                        max(0, min(1, score))
                    ))
            
            recommendations.sort(key=lambda x: x[3], reverse=True)
            return recommendations[:limit]
            
        except Exception as e:
            logger.error(f"Error in movie model recommendations: {e}")
            return []


class TVRecommender(BaseRecommender):
    """TV show-specific recommender"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.content_type = 'tv'
    
    def _get_model_recommendations(self, user_id: int, limit: int, genre_filter: str = None) -> List[Tuple[str, str, str, float]]:
        """TV model recommendations (placeholder - same pattern as movies)"""
        # Implementation would be similar to MovieRecommender but for TV shows
        return self._get_fallback_recommendations(user_id, limit, genre_filter)


def create_recommender(content_type: str, lookup_data: Dict, mappings: Dict, model=None, scaler=None):
    """Factory function to create appropriate recommender"""
    if content_type == 'movie':
        return MovieRecommender(lookup_data, mappings, model, scaler)
    elif content_type == 'tv':
        return TVRecommender(lookup_data, mappings, model, scaler)
    else:
        raise ValueError(f"Unknown content type: {content_type}")


def interleave_results(movie_results: List, tv_results: List, limit: int) -> List[Tuple[str, str, str, float]]:
    """Interleave movie and TV results for mixed recommendations"""
    result = []
    i = j = 0
    
    while len(result) < limit and (i < len(movie_results) or j < len(tv_results)):
        # Alternate between movies and TV shows, preferring higher scores
        if i < len(movie_results) and (j >= len(tv_results) or 
                                     (len(result) % 2 == 0) or 
                                     movie_results[i][3] > tv_results[j][3]):
            result.append(movie_results[i])
            i += 1
        elif j < len(tv_results):
            result.append(tv_results[j])
            j += 1
    
    return result