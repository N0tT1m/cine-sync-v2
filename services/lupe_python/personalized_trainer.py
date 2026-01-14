#!/usr/bin/env python3
"""
Personalized Trainer for Discord Bot
Manages per-user embeddings and learning

Features:
- Creates user embeddings from ratings/feedback
- Updates embeddings incrementally (no full retrain needed)
- Learns user preferences (genres, directors, actors)
- Re-ranks recommendations by user preferences
"""

import numpy as np
import torch
import pickle
import psycopg2
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
import logging
from contextlib import contextmanager

logger = logging.getLogger(__name__)


class PersonalizedTrainer:
    """
    Manages personalized recommendations for each Discord user

    Features:
    - Creates user embeddings from ratings/feedback
    - Updates embeddings incrementally (no full retrain needed)
    - Learns user preferences (genres, directors, actors)
    - Re-ranks recommendations by user preferences
    """

    def __init__(self, db_manager, content_manager, embedding_dim=256):
        """
        Initialize PersonalizedTrainer

        Args:
            db_manager: DatabaseManager instance
            content_manager: LupeContentManager or UnifiedLupeContentManager instance
            embedding_dim: Dimension of user embeddings
        """
        self.db_manager = db_manager
        self.content_manager = content_manager
        self.embedding_dim = embedding_dim

        # Cache for user embeddings (avoid DB queries every time)
        self.user_embeddings_cache = {}

        # Cache for user preferences
        self.user_preferences_cache = {}

        logger.info(f"PersonalizedTrainer initialized (embedding_dim={embedding_dim})")

    async def get_or_create_user_embedding(self, user_id: int) -> np.ndarray:
        """
        Get user embedding, create if doesn't exist

        Args:
            user_id: Discord user ID

        Returns:
            User embedding vector
        """
        # Check cache first
        if user_id in self.user_embeddings_cache:
            return self.user_embeddings_cache[user_id]

        # Try to load from database
        with self.db_manager.get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute(
                "SELECT embedding FROM user_embeddings WHERE user_id = %s",
                (user_id,)
            )

            row = cursor.fetchone()

            if row:
                # Deserialize embedding
                embedding = pickle.loads(row[0])
                self.user_embeddings_cache[user_id] = embedding
                return embedding

        # No embedding exists, create new one
        return await self.create_user_embedding(user_id)

    async def create_user_embedding(self, user_id: int) -> np.ndarray:
        """
        Create initial user embedding from their ratings and feedback

        Args:
            user_id: Discord user ID

        Returns:
            New user embedding
        """
        logger.info(f"Creating new embedding for user {user_id}")

        with self.db_manager.get_connection() as conn:
            cursor = conn.cursor()

            # Get user's ratings from user_ratings table
            cursor.execute("""
                SELECT movie_id, rating
                FROM user_ratings
                WHERE user_id = %s
                ORDER BY timestamp DESC
            """, (user_id,))

            ratings = cursor.fetchall()

            if not ratings:
                # New user with no ratings - return zero embedding
                embedding = np.zeros(self.embedding_dim)
                logger.info(f"User {user_id} has no ratings - using zero embedding")
            else:
                # Create embedding from rated items
                embedding = self._compute_embedding_from_ratings(ratings)
                logger.info(f"Created embedding from {len(ratings)} ratings")

            # Store in database
            embedding_bytes = pickle.dumps(embedding)

            cursor.execute("""
                INSERT INTO user_embeddings
                    (user_id, embedding, embedding_dim, rating_count, last_updated)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (user_id)
                DO UPDATE SET
                    embedding = EXCLUDED.embedding,
                    rating_count = EXCLUDED.rating_count,
                    last_updated = EXCLUDED.last_updated
            """, (user_id, embedding_bytes, self.embedding_dim, len(ratings), datetime.now()))

            conn.commit()

        # Cache it
        self.user_embeddings_cache[user_id] = embedding

        return embedding

    def _compute_embedding_from_ratings(self, ratings: List[Tuple]) -> np.ndarray:
        """
        Compute user embedding from their ratings
        Uses weighted average of item embeddings

        Args:
            ratings: List of (item_id, rating) tuples

        Returns:
            User embedding vector
        """
        embedding = np.zeros(self.embedding_dim)
        total_weight = 0

        for item_id, rating in ratings:
            # Get item embedding
            item_embedding = self._get_item_embedding(item_id)

            # Weight by rating (higher ratings = more influence)
            weight = (rating / 5.0) ** 2  # Quadratic weighting emphasizes high ratings

            embedding += weight * item_embedding
            total_weight += weight

        if total_weight > 0:
            embedding = embedding / total_weight

        # Normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        return embedding

    def _get_item_embedding(self, item_id: int) -> np.ndarray:
        """
        Get embedding for an item (movie/TV show)

        Uses content-based features to create a deterministic embedding
        from movie metadata (genres, year, etc.)

        Args:
            item_id: Item ID

        Returns:
            Item embedding vector
        """
        # Get movie metadata from content manager
        if hasattr(self.content_manager, 'movie_lookup'):
            movie_info = self.content_manager.movie_lookup.get(item_id, {})
        else:
            movie_info = {}

        # Create feature vector from metadata
        embedding = np.zeros(self.embedding_dim)

        # Use deterministic seed based on item_id for consistency
        np.random.seed(item_id % 2**31)
        base_embedding = np.random.randn(self.embedding_dim)

        # Normalize base embedding
        base_embedding = base_embedding / (np.linalg.norm(base_embedding) + 1e-10)

        # Add genre information if available
        genres = movie_info.get('genres', '')
        if genres:
            if isinstance(genres, str):
                genres_list = [g.strip() for g in genres.split('|')]
            else:
                genres_list = genres

            # Create genre hash and add to embedding
            for genre in genres_list:
                genre_hash = hash(genre) % self.embedding_dim
                embedding[genre_hash] += 1.0

        # Add year information if available
        year = movie_info.get('year')
        if year:
            # Map year to embedding dimension
            year_idx = int(year) % self.embedding_dim
            embedding[year_idx] += 0.5

        # Combine base and feature embeddings
        embedding = 0.7 * base_embedding + 0.3 * embedding

        # Normalize final embedding
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        return embedding

    async def update_user_embedding(self, user_id: int, item_id: int,
                                   feedback_type: str, rating: Optional[int] = None):
        """
        Incrementally update user embedding based on new feedback
        This is the "online learning" component - no full retrain needed!

        Args:
            user_id: Discord user ID
            item_id: Item that was rated/reviewed
            feedback_type: 'positive', 'negative', 'love', 'like', 'dislike', 'hate'
            rating: Optional numeric rating (1-5)
        """
        logger.info(f"Updating embedding for user {user_id}, item {item_id}, feedback: {feedback_type}")

        # Get current embedding
        current_embedding = await self.get_or_create_user_embedding(user_id)

        # Get item embedding
        item_embedding = self._get_item_embedding(item_id)

        # Determine learning rate and direction
        learning_rate = 0.1  # How much to adjust

        if feedback_type in ['love', 'positive'] or (rating and rating >= 4):
            # Positive feedback - move towards this item
            direction = 1.0
            if feedback_type == 'love' or rating == 5:
                learning_rate = 0.15  # Stronger update for "love"
        elif feedback_type in ['hate', 'negative'] or (rating and rating <= 2):
            # Negative feedback - move away from this item
            direction = -1.0
            if feedback_type == 'hate' or rating == 1:
                learning_rate = 0.15  # Stronger update for "hate"
        else:
            # Neutral - smaller update
            direction = 0.5 if (rating and rating >= 3) else -0.5
            learning_rate = 0.05

        # Update embedding
        updated_embedding = current_embedding + (direction * learning_rate * item_embedding)

        # Normalize to keep embedding on unit sphere
        norm = np.linalg.norm(updated_embedding)
        if norm > 0:
            updated_embedding = updated_embedding / norm

        # Store updated embedding
        with self.db_manager.get_connection() as conn:
            cursor = conn.cursor()

            embedding_bytes = pickle.dumps(updated_embedding)

            cursor.execute("""
                UPDATE user_embeddings
                SET embedding = %s,
                    last_updated = %s,
                    feedback_count = feedback_count + 1
                WHERE user_id = %s
            """, (embedding_bytes, datetime.now(), user_id))

            conn.commit()

        # Update cache
        self.user_embeddings_cache[user_id] = updated_embedding

        logger.info(f"✅ Updated embedding for user {user_id}")

    async def get_personalized_recommendations(self, user_id: int,
                                              top_k: int = 10,
                                              content_type: str = 'movie',
                                              base_recommendations: Optional[List[Tuple]] = None) -> List[Dict]:
        """
        Get personalized recommendations for a user by re-ranking with user embedding

        Args:
            user_id: Discord user ID
            top_k: Number of recommendations
            content_type: 'movie' or 'tv'
            base_recommendations: Optional pre-computed recommendations to re-rank

        Returns:
            List of recommended items with scores
        """
        logger.info(f"Getting personalized recommendations for user {user_id}")

        # Check if user has enough data for personalization
        rating_count = await self._get_user_rating_count(user_id)

        if rating_count < 5:
            # Not enough data - return base recommendations or empty
            logger.info(f"User {user_id} has only {rating_count} ratings - skipping personalization")
            if base_recommendations:
                return [
                    {
                        'item_id': rec[0],
                        'title': rec[1] if len(rec) > 1 else '',
                        'content_type': rec[2] if len(rec) > 2 else content_type,
                        'score': rec[3] if len(rec) > 3 else 0.5,
                        'base_score': rec[3] if len(rec) > 3 else 0.5,
                        'personalization_score': 0.0,
                        'metadata': {}
                    }
                    for rec in base_recommendations[:top_k]
                ]
            return []

        # Get user embedding
        user_embedding = await self.get_or_create_user_embedding(user_id)

        # Use base recommendations if provided
        if not base_recommendations:
            logger.warning("No base recommendations provided, personalization may be limited")
            return []

        # Re-rank by similarity to user embedding
        personalized_recs = []
        for rec in base_recommendations:
            item_id = rec[0]
            title = rec[1] if len(rec) > 1 else ''
            rec_content_type = rec[2] if len(rec) > 2 else content_type
            base_score = rec[3] if len(rec) > 3 else 0.5

            # Get item embedding
            item_embedding = self._get_item_embedding(item_id)

            # Calculate similarity (cosine similarity)
            similarity = np.dot(user_embedding, item_embedding)

            # Normalize similarity to [0, 1]
            similarity = (similarity + 1) / 2

            # Combine base score with personalization
            # 60% personalization, 40% model score
            final_score = 0.6 * similarity + 0.4 * base_score

            personalized_recs.append({
                'item_id': item_id,
                'title': title,
                'content_type': rec_content_type,
                'score': final_score,
                'base_score': base_score,
                'personalization_score': similarity,
                'metadata': {}
            })

        # Sort by final score
        personalized_recs.sort(key=lambda x: x['score'], reverse=True)

        return personalized_recs[:top_k]

    async def _get_user_rating_count(self, user_id: int) -> int:
        """Get count of user's ratings"""
        with self.db_manager.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT COUNT(*) FROM user_ratings WHERE user_id = %s",
                (user_id,)
            )
            result = cursor.fetchone()
            return result[0] if result else 0

    def clear_cache(self, user_id: Optional[int] = None):
        """
        Clear embedding cache

        Args:
            user_id: If provided, clear only this user's cache. Otherwise clear all.
        """
        if user_id:
            self.user_embeddings_cache.pop(user_id, None)
            self.user_preferences_cache.pop(user_id, None)
            logger.info(f"Cleared cache for user {user_id}")
        else:
            self.user_embeddings_cache.clear()
            self.user_preferences_cache.clear()
            logger.info("Cleared all caches")
