#!/usr/bin/env python3
"""
Preference Learner - Analyzes user patterns and preferences

Extracts meaningful patterns from user rating history:
- Favorite genres
- Favorite directors
- Favorite actors
- Preferred decades/eras
- Rating behavior (generous vs harsh)
- Diversity preferences
"""

import numpy as np
from typing import Dict, List, Any, Tuple
from collections import Counter
import logging
import json
from datetime import datetime

logger = logging.getLogger(__name__)


class PreferenceLearner:
    """
    Analyzes user preferences from their rating history
    Extracts patterns like favorite genres, directors, actors, decades
    """

    def __init__(self, db_manager, content_manager):
        """
        Initialize PreferenceLearner

        Args:
            db_manager: DatabaseManager instance
            content_manager: Content manager with movie metadata
        """
        self.db_manager = db_manager
        self.content_manager = content_manager
        self.user_preferences_cache = {}

    async def analyze_user_patterns(self, user_id: int) -> Dict[str, Any]:
        """
        Analyze user's viewing patterns

        Returns dictionary with:
        - favorite_genres: Top genres user rates highly
        - favorite_directors: Top directors
        - favorite_actors: Top actors
        - preferred_decades: Preferred time periods
        - rating_distribution: How user rates (harsh vs generous)
        - diversity_preference: Does user like variety or stick to favorites?

        Args:
            user_id: Discord user ID

        Returns:
            Dictionary of analyzed patterns
        """
        logger.info(f"Analyzing patterns for user {user_id}")

        # Get user's ratings
        with self.db_manager.get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute("""
                SELECT ur.movie_id, ur.rating, ur.timestamp
                FROM user_ratings ur
                WHERE ur.user_id = %s
                ORDER BY ur.timestamp DESC
            """, (user_id,))

            ratings = cursor.fetchall()

        if not ratings:
            return self._default_preferences()

        # Analyze patterns
        patterns = {
            'favorite_genres': await self._analyze_genres(ratings),
            'favorite_directors': await self._analyze_directors(ratings),
            'favorite_actors': await self._analyze_actors(ratings),
            'preferred_decades': await self._analyze_decades(ratings),
            'rating_distribution': self._analyze_rating_distribution(ratings),
            'diversity_score': self._calculate_diversity_score(ratings),
            'avg_rating': float(np.mean([r[1] for r in ratings])),
            'total_ratings': len(ratings)
        }

        # Cache it
        self.user_preferences_cache[user_id] = patterns

        # Store in database
        await self._store_preferences(user_id, patterns)

        return patterns

    async def _analyze_genres(self, ratings: List[Tuple]) -> List[Dict]:
        """
        Analyze favorite genres

        Args:
            ratings: List of (movie_id, rating, timestamp) tuples

        Returns:
            List of genre preference dictionaries sorted by score
        """
        genre_ratings = {}
        genre_counts = {}

        for movie_id, rating, _ in ratings:
            # Get movie metadata
            movie_info = {}
            if hasattr(self.content_manager, 'movie_lookup'):
                movie_info = self.content_manager.movie_lookup.get(movie_id, {})

            genres = movie_info.get('genres', [])

            if isinstance(genres, str):
                genres = [g.strip() for g in genres.split('|') if g.strip()]

            for genre in genres:
                if genre:
                    if genre not in genre_ratings:
                        genre_ratings[genre] = []
                    genre_ratings[genre].append(rating)
                    genre_counts[genre] = genre_counts.get(genre, 0) + 1

        # Calculate average rating per genre
        genre_scores = []
        for genre, ratings_list in genre_ratings.items():
            avg_rating = np.mean(ratings_list)
            count = genre_counts[genre]

            # Weight by both average rating and count
            score = avg_rating * (1 + np.log(count))

            genre_scores.append({
                'genre': genre,
                'avg_rating': float(avg_rating),
                'count': count,
                'score': float(score)
            })

        # Sort by score
        genre_scores.sort(key=lambda x: x['score'], reverse=True)

        return genre_scores[:10]  # Top 10 genres

    async def _analyze_directors(self, ratings: List[Tuple]) -> List[Dict]:
        """
        Analyze favorite directors

        Args:
            ratings: List of (movie_id, rating, timestamp) tuples

        Returns:
            List of director preference dictionaries
        """
        director_ratings = {}

        for movie_id, rating, _ in ratings:
            movie_info = {}
            if hasattr(self.content_manager, 'movie_lookup'):
                movie_info = self.content_manager.movie_lookup.get(movie_id, {})

            director = movie_info.get('director', 'Unknown')

            if director and director != 'Unknown':
                if director not in director_ratings:
                    director_ratings[director] = []
                director_ratings[director].append(rating)

        # Calculate scores
        director_scores = []
        for director, ratings_list in director_ratings.items():
            if len(ratings_list) >= 2:  # Need at least 2 movies
                avg_rating = np.mean(ratings_list)
                count = len(ratings_list)

                director_scores.append({
                    'director': director,
                    'avg_rating': float(avg_rating),
                    'count': count,
                    'score': float(avg_rating * (1 + np.log(count)))
                })

        director_scores.sort(key=lambda x: x['score'], reverse=True)
        return director_scores[:5]  # Top 5 directors

    async def _analyze_actors(self, ratings: List[Tuple]) -> List[Dict]:
        """
        Analyze favorite actors

        Args:
            ratings: List of (movie_id, rating, timestamp) tuples

        Returns:
            List of actor preference dictionaries
        """
        actor_ratings = {}

        for movie_id, rating, _ in ratings:
            movie_info = {}
            if hasattr(self.content_manager, 'movie_lookup'):
                movie_info = self.content_manager.movie_lookup.get(movie_id, {})

            cast = movie_info.get('cast', [])

            if isinstance(cast, str):
                cast = [c.strip() for c in cast.split('|')[:5]]  # Top 5 cast members

            for actor in cast:
                if actor:
                    if actor not in actor_ratings:
                        actor_ratings[actor] = []
                    actor_ratings[actor].append(rating)

        actor_scores = []
        for actor, ratings_list in actor_ratings.items():
            if len(ratings_list) >= 2:
                avg_rating = np.mean(ratings_list)
                count = len(ratings_list)

                actor_scores.append({
                    'actor': actor,
                    'avg_rating': float(avg_rating),
                    'count': count,
                    'score': float(avg_rating * (1 + np.log(count)))
                })

        actor_scores.sort(key=lambda x: x['score'], reverse=True)
        return actor_scores[:5]  # Top 5 actors

    async def _analyze_decades(self, ratings: List[Tuple]) -> List[Dict]:
        """
        Analyze preferred decades

        Args:
            ratings: List of (movie_id, rating, timestamp) tuples

        Returns:
            List of decade preference dictionaries
        """
        decade_ratings = {}

        for movie_id, rating, _ in ratings:
            movie_info = {}
            if hasattr(self.content_manager, 'movie_lookup'):
                movie_info = self.content_manager.movie_lookup.get(movie_id, {})

            year = movie_info.get('year')

            if year:
                try:
                    year = int(year)
                    decade = (year // 10) * 10
                    if decade not in decade_ratings:
                        decade_ratings[decade] = []
                    decade_ratings[decade].append(rating)
                except (ValueError, TypeError):
                    pass

        decade_scores = []
        for decade, ratings_list in decade_ratings.items():
            avg_rating = np.mean(ratings_list)
            count = len(ratings_list)

            decade_scores.append({
                'decade': f"{decade}s",
                'avg_rating': float(avg_rating),
                'count': count
            })

        decade_scores.sort(key=lambda x: x['avg_rating'], reverse=True)
        return decade_scores

    def _analyze_rating_distribution(self, ratings: List[Tuple]) -> Dict:
        """
        Analyze how user rates (harsh, generous, etc.)

        Args:
            ratings: List of (movie_id, rating, timestamp) tuples

        Returns:
            Dictionary with rating statistics
        """
        rating_values = [r[1] for r in ratings]

        return {
            'mean': float(np.mean(rating_values)),
            'median': float(np.median(rating_values)),
            'std': float(np.std(rating_values)),
            'min': float(np.min(rating_values)),
            'max': float(np.max(rating_values)),
            'rating_counts': {str(k): int(v) for k, v in dict(Counter(rating_values)).items()}
        }

    def _calculate_diversity_score(self, ratings: List[Tuple]) -> float:
        """
        Calculate diversity score (0-1)
        0 = watches same genres/types
        1 = very diverse watching habits

        Args:
            ratings: List of (movie_id, rating, timestamp) tuples

        Returns:
            Diversity score between 0 and 1
        """
        # Get genres for rated movies
        all_genres = []
        for movie_id, _, _ in ratings:
            movie_info = {}
            if hasattr(self.content_manager, 'movie_lookup'):
                movie_info = self.content_manager.movie_lookup.get(movie_id, {})

            genres = movie_info.get('genres', [])

            if isinstance(genres, str):
                genres = [g.strip() for g in genres.split('|') if g.strip()]

            all_genres.extend(genres)

        if not all_genres:
            return 0.5  # Default

        # Calculate entropy (diversity)
        genre_counts = Counter(all_genres)
        total = len(all_genres)

        entropy = 0
        for count in genre_counts.values():
            p = count / total
            entropy -= p * np.log(p + 1e-10)

        # Normalize to 0-1
        max_entropy = np.log(len(genre_counts)) if len(genre_counts) > 0 else 1
        diversity_score = entropy / max_entropy if max_entropy > 0 else 0.5

        return float(diversity_score)

    def _default_preferences(self) -> Dict:
        """Default preferences for new users"""
        return {
            'favorite_genres': [],
            'favorite_directors': [],
            'favorite_actors': [],
            'preferred_decades': [],
            'rating_distribution': {},
            'diversity_score': 0.5,
            'avg_rating': 0,
            'total_ratings': 0
        }

    async def _store_preferences(self, user_id: int, patterns: Dict):
        """
        Store preferences in database

        Args:
            user_id: Discord user ID
            patterns: Analyzed preference patterns
        """
        with self.db_manager.get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute("""
                INSERT INTO user_preferences
                    (user_id, favorite_genres, favorite_directors, favorite_actors,
                     preferred_decades, avg_rating, rating_distribution,
                     diversity_score, last_analyzed)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, NOW())
                ON CONFLICT (user_id)
                DO UPDATE SET
                    favorite_genres = EXCLUDED.favorite_genres,
                    favorite_directors = EXCLUDED.favorite_directors,
                    favorite_actors = EXCLUDED.favorite_actors,
                    preferred_decades = EXCLUDED.preferred_decades,
                    avg_rating = EXCLUDED.avg_rating,
                    rating_distribution = EXCLUDED.rating_distribution,
                    diversity_score = EXCLUDED.diversity_score,
                    last_analyzed = EXCLUDED.last_analyzed
            """, (
                user_id,
                json.dumps(patterns['favorite_genres']),
                json.dumps(patterns['favorite_directors']),
                json.dumps(patterns['favorite_actors']),
                json.dumps(patterns['preferred_decades']),
                patterns['avg_rating'],
                json.dumps(patterns['rating_distribution']),
                patterns['diversity_score']
            ))

            conn.commit()

    def get_cached_preferences(self, user_id: int) -> Optional[Dict]:
        """
        Get cached preferences if available

        Args:
            user_id: Discord user ID

        Returns:
            Cached preferences or None
        """
        return self.user_preferences_cache.get(user_id)

    def clear_cache(self, user_id: Optional[int] = None):
        """
        Clear preference cache

        Args:
            user_id: If provided, clear only this user's cache. Otherwise clear all.
        """
        if user_id:
            self.user_preferences_cache.pop(user_id, None)
            logger.info(f"Cleared preference cache for user {user_id}")
        else:
            self.user_preferences_cache.clear()
            logger.info("Cleared all preference caches")
