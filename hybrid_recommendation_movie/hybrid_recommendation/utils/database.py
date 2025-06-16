import psycopg2
import pandas as pd
import logging
from typing import Optional, Dict, Any, List
from contextlib import contextmanager
from config import DatabaseConfig

logger = logging.getLogger(__name__)


class DatabaseManager:
    def __init__(self, config: DatabaseConfig):
        self.config = config
        self._connection = None
    
    @contextmanager
    def get_connection(self):
        """Context manager for database connections with proper cleanup"""
        conn = None
        try:
            conn = psycopg2.connect(
                host=self.config.host,
                database=self.config.database,
                user=self.config.user,
                password=self.config.password,
                port=self.config.port
            )
            yield conn
        except psycopg2.Error as e:
            logger.error(f"Database connection error: {e}")
            if conn:
                conn.rollback()
            raise
        finally:
            if conn:
                conn.close()
    
    def execute_query(self, query: str, params: Optional[tuple] = None) -> Optional[List[tuple]]:
        """Execute a query and return results"""
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(query, params)
                    if cursor.description:
                        return cursor.fetchall()
                    conn.commit()
                    return None
        except psycopg2.Error as e:
            logger.error(f"Query execution error: {e}")
            raise
    
    def fetch_dataframe(self, query: str, params: Optional[tuple] = None) -> pd.DataFrame:
        """Fetch query results as pandas DataFrame"""
        try:
            with self.get_connection() as conn:
                return pd.read_sql_query(query, conn, params=params)
        except Exception as e:
            logger.error(f"Error fetching DataFrame: {e}")
            raise
    
    def insert_batch(self, table: str, data: List[Dict[str, Any]], 
                    on_conflict: str = "DO NOTHING") -> int:
        """Insert batch data with conflict resolution"""
        if not data:
            return 0
        
        columns = list(data[0].keys())
        placeholders = ', '.join(['%s'] * len(columns))
        columns_str = ', '.join(columns)
        
        query = f"""
            INSERT INTO {table} ({columns_str}) 
            VALUES ({placeholders}) 
            ON CONFLICT {on_conflict}
        """
        
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
        """Check if table exists"""
        query = """
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_name = %s
            )
        """
        try:
            result = self.execute_query(query, (table_name,))
            return result[0][0] if result else False
        except Exception as e:
            logger.error(f"Error checking table existence: {e}")
            return False
    
    def get_table_schema(self, table_name: str) -> List[Dict[str, str]]:
        """Get table column information"""
        query = """
            SELECT column_name, data_type, is_nullable
            FROM information_schema.columns
            WHERE table_name = %s
            ORDER BY ordinal_position
        """
        try:
            result = self.execute_query(query, (table_name,))
            return [
                {
                    'column_name': row[0],
                    'data_type': row[1],
                    'is_nullable': row[2]
                }
                for row in result
            ] if result else []
        except Exception as e:
            logger.error(f"Error getting table schema: {e}")
            return []


def load_ratings_data(db_manager: DatabaseManager) -> pd.DataFrame:
    """Load ratings data from database"""
    query = """
        SELECT user_id as "userId", media_id as "movieId", rating, created_at as timestamp
        FROM ratings 
        WHERE rating IS NOT NULL
        ORDER BY created_at
    """
    return db_manager.fetch_dataframe(query)


def load_movies_data(db_manager: DatabaseManager) -> pd.DataFrame:
    """Load movies data from database"""
    query = """
        SELECT media_id, title, genres, year, tmdb_id, imdb_id
        FROM movies 
        WHERE title IS NOT NULL
        ORDER BY media_id
    """
    return db_manager.fetch_dataframe(query)


def save_feedback(db_manager: DatabaseManager, user_id: int, movie_id: int, 
                 feedback_type: str, rating: Optional[float] = None) -> bool:
    """Save user feedback to database"""
    try:
        data = [{
            'user_id': user_id,
            'movie_id': movie_id,
            'feedback_type': feedback_type,
            'rating': rating,
            'created_at': 'NOW()'
        }]
        
        rows_affected = db_manager.insert_batch('user_feedback', data)
        return rows_affected > 0
    except Exception as e:
        logger.error(f"Error saving feedback: {e}")
        return False