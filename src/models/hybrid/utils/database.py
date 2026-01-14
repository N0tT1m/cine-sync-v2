"""
CineSync v2 - Unified Database Utilities

Database operations for the content recommendation system.
Supports both movies and TV shows with content_type parameter.
"""

import psycopg2
import psycopg2.extras
import pandas as pd
import logging
from datetime import datetime
from typing import Optional, Dict, Any, List, Union
from contextlib import contextmanager
from enum import Enum

logger = logging.getLogger(__name__)


class ContentType(Enum):
    """Content type enumeration"""
    MOVIE = "movie"
    TV = "tv"
    BOTH = "both"


# Whitelist of allowed table names to prevent SQL injection
ALLOWED_TABLES = frozenset({
    'ratings', 'movies', 'tv_shows', 'users', 'user_feedback',
    'recommendations', 'content_metadata', 'genres'
})


class DatabaseManager:
    """
    Database manager for content recommendation system.

    Provides connection management, query execution, and data loading
    for both movies and TV shows.
    """

    def __init__(self, config):
        """
        Initialize database manager.

        Args:
            config: DatabaseConfig object with connection parameters
        """
        self.config = config
        self._connection = None

    @contextmanager
    def get_connection(self, timeout: int = 30):
        """
        Context manager for database connections with proper cleanup.

        Args:
            timeout: Connection timeout in seconds

        Yields:
            Database connection
        """
        conn = None
        try:
            conn = psycopg2.connect(
                host=self.config.host,
                database=self.config.database,
                user=self.config.user,
                password=self.config.password,
                port=self.config.port,
                connect_timeout=timeout
            )
            yield conn
        except psycopg2.Error as e:
            logger.error(f"Database connection error: {e}")
            if conn is not None:
                conn.rollback()
            raise
        finally:
            if conn is not None:
                conn.close()

    def execute_query(
        self,
        query: str,
        params: Optional[tuple] = None,
        commit: bool = True
    ) -> Optional[List[tuple]]:
        """
        Execute a query and return results.

        Args:
            query: SQL query string
            params: Query parameters (for parameterized queries)
            commit: Whether to commit after execution

        Returns:
            Query results or None
        """
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(query, params)
                    if cursor.description:
                        results = cursor.fetchall()
                        return results
                    if commit:
                        conn.commit()
                    return None
        except psycopg2.Error as e:
            logger.error(f"Query execution error: {e}")
            raise

    def fetch_dataframe(
        self,
        query: str,
        params: Optional[Union[tuple, dict]] = None
    ) -> pd.DataFrame:
        """
        Fetch query results as pandas DataFrame.

        Args:
            query: SQL query string
            params: Query parameters

        Returns:
            DataFrame with query results
        """
        try:
            with self.get_connection() as conn:
                return pd.read_sql_query(query, conn, params=params)
        except Exception as e:
            logger.error(f"Error fetching DataFrame: {e}")
            raise

    def insert_batch(
        self,
        table: str,
        data: List[Dict[str, Any]],
        on_conflict: str = "DO NOTHING"
    ) -> int:
        """
        Insert batch data with conflict resolution.

        Args:
            table: Table name (must be in ALLOWED_TABLES)
            data: List of dictionaries with column:value pairs
            on_conflict: Conflict resolution clause

        Returns:
            Number of rows affected
        """
        if not data:
            return 0

        # Validate table name to prevent SQL injection
        if table not in ALLOWED_TABLES:
            raise ValueError(f"Table '{table}' not in allowed tables: {ALLOWED_TABLES}")

        # Validate on_conflict clause
        allowed_conflict_clauses = {'DO NOTHING', 'DO UPDATE SET'}
        if not any(on_conflict.upper().startswith(clause) for clause in allowed_conflict_clauses):
            raise ValueError(f"Invalid ON CONFLICT clause: {on_conflict}")

        columns = list(data[0].keys())
        placeholders = ', '.join(['%s'] * len(columns))
        columns_str = ', '.join(f'"{col}"' for col in columns)  # Quote column names

        # Use identifier quoting for table name
        query = f'''
            INSERT INTO "{table}" ({columns_str})
            VALUES ({placeholders})
            ON CONFLICT {on_conflict}
        '''

        try:
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    values = [[row[col] for col in columns] for row in data]
                    cursor.executemany(query, values)
                    conn.commit()
                    return cursor.rowcount
        except psycopg2.Error as e:
            logger.error(f"Batch insert error: {e}")
            raise

    def table_exists(self, table_name: str) -> bool:
        """Check if table exists in database."""
        query = """
            SELECT EXISTS (
                SELECT FROM information_schema.tables
                WHERE table_name = %s
            )
        """
        try:
            result = self.execute_query(query, (table_name,), commit=False)
            return result[0][0] if result else False
        except Exception as e:
            logger.error(f"Error checking table existence: {e}")
            return False

    def get_table_schema(self, table_name: str) -> List[Dict[str, str]]:
        """Get table column information."""
        query = """
            SELECT column_name, data_type, is_nullable
            FROM information_schema.columns
            WHERE table_name = %s
            ORDER BY ordinal_position
        """
        try:
            result = self.execute_query(query, (table_name,), commit=False)
            return [
                {
                    'column_name': row[0],
                    'data_type': row[1],
                    'is_nullable': row[2]
                }
                for row in (result or [])
            ]
        except Exception as e:
            logger.error(f"Error getting table schema: {e}")
            return []


# Data loading functions

def load_ratings_data(
    db_manager: DatabaseManager,
    content_type: ContentType = ContentType.BOTH
) -> pd.DataFrame:
    """
    Load ratings data from database.

    Args:
        db_manager: Database manager instance
        content_type: Type of content to load ratings for

    Returns:
        DataFrame with ratings data
    """
    if content_type == ContentType.MOVIE:
        query = """
            SELECT user_id as "userId", media_id as "itemId", rating,
                   created_at as timestamp, 'movie' as content_type
            FROM ratings
            WHERE rating IS NOT NULL AND content_type = 'movie'
            ORDER BY created_at
        """
    elif content_type == ContentType.TV:
        query = """
            SELECT user_id as "userId", show_id as "itemId", rating,
                   created_at as timestamp, 'tv' as content_type
            FROM tv_ratings
            WHERE rating IS NOT NULL
            ORDER BY created_at
        """
    else:  # BOTH
        query = """
            SELECT user_id as "userId", media_id as "itemId", rating,
                   created_at as timestamp, 'movie' as content_type
            FROM ratings
            WHERE rating IS NOT NULL
            UNION ALL
            SELECT user_id as "userId", show_id as "itemId", rating,
                   created_at as timestamp, 'tv' as content_type
            FROM tv_ratings
            WHERE rating IS NOT NULL
            ORDER BY timestamp
        """

    return db_manager.fetch_dataframe(query)


def load_movies_data(db_manager: DatabaseManager) -> pd.DataFrame:
    """Load movies metadata from database."""
    query = """
        SELECT
            media_id as item_id,
            title,
            genres,
            year,
            tmdb_id,
            imdb_id,
            'movie' as content_type
        FROM movies
        WHERE title IS NOT NULL
        ORDER BY media_id
    """
    return db_manager.fetch_dataframe(query)


def load_tv_data(db_manager: DatabaseManager) -> pd.DataFrame:
    """Load TV shows metadata from database."""
    query = """
        SELECT
            show_id as item_id,
            title,
            genres,
            year,
            tmdb_id,
            episode_count,
            season_count,
            average_duration as duration,
            status,
            'tv' as content_type
        FROM tv_shows
        WHERE title IS NOT NULL
        ORDER BY show_id
    """
    return db_manager.fetch_dataframe(query)


def load_content_data(
    db_manager: DatabaseManager,
    content_type: ContentType = ContentType.BOTH
) -> pd.DataFrame:
    """
    Load content metadata based on type.

    Args:
        db_manager: Database manager instance
        content_type: Type of content to load

    Returns:
        DataFrame with content metadata
    """
    if content_type == ContentType.MOVIE:
        return load_movies_data(db_manager)
    elif content_type == ContentType.TV:
        return load_tv_data(db_manager)
    else:
        # Load both and concatenate
        movies_df = load_movies_data(db_manager)
        tv_df = load_tv_data(db_manager)

        # Align columns (TV has extra columns that movies don't)
        for col in ['episode_count', 'season_count', 'duration', 'status']:
            if col not in movies_df.columns:
                movies_df[col] = None

        return pd.concat([movies_df, tv_df], ignore_index=True)


def save_feedback(
    db_manager: DatabaseManager,
    user_id: int,
    item_id: int,
    feedback_type: str,
    content_type: ContentType = ContentType.MOVIE,
    rating: Optional[float] = None
) -> bool:
    """
    Save user feedback to database.

    Args:
        db_manager: Database manager instance
        user_id: User ID
        item_id: Item ID (movie or TV show)
        feedback_type: Type of feedback (e.g., 'like', 'dislike', 'watched')
        content_type: Content type
        rating: Optional rating value

    Returns:
        True if successful
    """
    try:
        data = [{
            'user_id': user_id,
            'item_id': item_id,
            'content_type': content_type.value,
            'feedback_type': feedback_type,
            'rating': rating,
            'created_at': datetime.now()
        }]

        rows_affected = db_manager.insert_batch('user_feedback', data)
        return rows_affected > 0
    except Exception as e:
        logger.error(f"Error saving feedback: {e}")
        return False
