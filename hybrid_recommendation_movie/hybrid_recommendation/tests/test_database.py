import pytest
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
import psycopg2
from utils.database import DatabaseManager, load_ratings_data, load_movies_data, save_feedback
from config import DatabaseConfig


class TestDatabaseManager:
    """Test the DatabaseManager class"""
    
    @pytest.fixture
    def db_config(self):
        """Create a test database configuration"""
        return DatabaseConfig(
            host='localhost',
            database='test_db',
            user='test_user',
            password='test_password',
            port=5432
        )
    
    @pytest.fixture
    def db_manager(self, db_config):
        """Create a DatabaseManager instance"""
        return DatabaseManager(db_config)
    
    def test_init(self, db_manager, db_config):
        """Test DatabaseManager initialization"""
        assert db_manager.config == db_config
        assert db_manager._connection is None
    
    @patch('psycopg2.connect')
    def test_get_connection_success(self, mock_connect, db_manager):
        """Test successful database connection"""
        mock_conn = Mock()
        mock_connect.return_value = mock_conn
        
        with db_manager.get_connection() as conn:
            assert conn == mock_conn
        
        mock_connect.assert_called_once_with(
            host='localhost',
            database='test_db',
            user='test_user',
            password='test_password',
            port=5432
        )
        mock_conn.close.assert_called_once()
    
    @patch('psycopg2.connect')
    def test_get_connection_error(self, mock_connect, db_manager):
        """Test database connection error handling"""
        mock_connect.side_effect = psycopg2.Error('Connection failed')
        
        with pytest.raises(psycopg2.Error):
            with db_manager.get_connection():
                pass
    
    @patch('psycopg2.connect')
    def test_execute_query_select(self, mock_connect, db_manager):
        """Test executing a SELECT query"""
        mock_conn = Mock()
        mock_cursor = Mock()
        mock_cursor.description = [('id',), ('name',)]
        mock_cursor.fetchall.return_value = [(1, 'test')]
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
        mock_connect.return_value.__enter__.return_value = mock_conn
        
        result = db_manager.execute_query('SELECT * FROM test')
        
        assert result == [(1, 'test')]
        mock_cursor.execute.assert_called_once_with('SELECT * FROM test', None)
    
    @patch('psycopg2.connect')
    def test_execute_query_insert(self, mock_connect, db_manager):
        """Test executing an INSERT query"""
        mock_conn = Mock()
        mock_cursor = Mock()
        mock_cursor.description = None
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
        mock_connect.return_value.__enter__.return_value = mock_conn
        
        result = db_manager.execute_query('INSERT INTO test VALUES (1, \'test\')')
        
        assert result is None
        mock_conn.commit.assert_called_once()
    
    @patch('pandas.read_sql_query')
    @patch('psycopg2.connect')
    def test_fetch_dataframe(self, mock_connect, mock_read_sql, db_manager):
        """Test fetching data as DataFrame"""
        mock_conn = Mock()
        mock_connect.return_value.__enter__.return_value = mock_conn
        
        expected_df = pd.DataFrame({'id': [1, 2], 'name': ['test1', 'test2']})
        mock_read_sql.return_value = expected_df
        
        result = db_manager.fetch_dataframe('SELECT * FROM test')
        
        assert result.equals(expected_df)
        mock_read_sql.assert_called_once_with('SELECT * FROM test', mock_conn, params=None)
    
    @patch('psycopg2.connect')
    def test_insert_batch(self, mock_connect, db_manager):
        """Test batch insert operation"""
        mock_conn = Mock()
        mock_cursor = Mock()
        mock_cursor.rowcount = 2
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
        mock_connect.return_value.__enter__.return_value = mock_conn
        
        data = [
            {'id': 1, 'name': 'test1'},
            {'id': 2, 'name': 'test2'}
        ]
        
        result = db_manager.insert_batch('test_table', data)
        
        assert result == 2
        mock_cursor.executemany.assert_called_once()
        mock_conn.commit.assert_called_once()
    
    def test_insert_batch_empty_data(self, db_manager):
        """Test batch insert with empty data"""
        result = db_manager.insert_batch('test_table', [])
        assert result == 0
    
    @patch('psycopg2.connect')
    def test_table_exists_true(self, mock_connect, db_manager):
        """Test checking if table exists (returns True)"""
        mock_conn = Mock()
        mock_cursor = Mock()
        mock_cursor.fetchall.return_value = [(True,)]
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
        mock_connect.return_value.__enter__.return_value = mock_conn
        
        result = db_manager.table_exists('test_table')
        
        assert result is True
    
    @patch('psycopg2.connect')
    def test_table_exists_false(self, mock_connect, db_manager):
        """Test checking if table exists (returns False)"""
        mock_conn = Mock()
        mock_cursor = Mock()
        mock_cursor.fetchall.return_value = [(False,)]
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
        mock_connect.return_value.__enter__.return_value = mock_conn
        
        result = db_manager.table_exists('nonexistent_table')
        
        assert result is False
    
    @patch('psycopg2.connect')
    def test_get_table_schema(self, mock_connect, db_manager):
        """Test getting table schema"""
        mock_conn = Mock()
        mock_cursor = Mock()
        mock_cursor.fetchall.return_value = [
            ('id', 'integer', 'NO'),
            ('name', 'varchar', 'YES')
        ]
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
        mock_connect.return_value.__enter__.return_value = mock_conn
        
        result = db_manager.get_table_schema('test_table')
        
        expected = [
            {'column_name': 'id', 'data_type': 'integer', 'is_nullable': 'NO'},
            {'column_name': 'name', 'data_type': 'varchar', 'is_nullable': 'YES'}
        ]
        assert result == expected


class TestDatabaseUtilities:
    """Test database utility functions"""
    
    @patch.object(DatabaseManager, 'fetch_dataframe')
    def test_load_ratings_data(self, mock_fetch):
        """Test loading ratings data"""
        expected_df = pd.DataFrame({
            'userId': [1, 2, 3],
            'movieId': [10, 20, 30],
            'rating': [4.0, 3.5, 5.0],
            'timestamp': ['2023-01-01', '2023-01-02', '2023-01-03']
        })
        mock_fetch.return_value = expected_df
        
        db_manager = Mock()
        result = load_ratings_data(db_manager)
        
        assert result.equals(expected_df)
        mock_fetch.assert_called_once()
    
    @patch.object(DatabaseManager, 'fetch_dataframe')
    def test_load_movies_data(self, mock_fetch):
        """Test loading movies data"""
        expected_df = pd.DataFrame({
            'media_id': [1, 2, 3],
            'title': ['Movie 1', 'Movie 2', 'Movie 3'],
            'genres': ['Action', 'Comedy', 'Drama'],
            'year': [2020, 2021, 2022],
            'tmdb_id': [100, 200, 300],
            'imdb_id': ['tt1000000', 'tt2000000', 'tt3000000']
        })
        mock_fetch.return_value = expected_df
        
        db_manager = Mock()
        result = load_movies_data(db_manager)
        
        assert result.equals(expected_df)
        mock_fetch.assert_called_once()
    
    @patch.object(DatabaseManager, 'insert_batch')
    def test_save_feedback_success(self, mock_insert):
        """Test successful feedback saving"""
        mock_insert.return_value = 1
        
        db_manager = Mock()
        result = save_feedback(db_manager, user_id=123, movie_id=456, 
                              feedback_type='like', rating=4.5)
        
        assert result is True
        mock_insert.assert_called_once()
    
    @patch.object(DatabaseManager, 'insert_batch')
    def test_save_feedback_failure(self, mock_insert):
        """Test feedback saving failure"""
        mock_insert.return_value = 0
        
        db_manager = Mock()
        result = save_feedback(db_manager, user_id=123, movie_id=456, 
                              feedback_type='like', rating=4.5)
        
        assert result is False
    
    @patch.object(DatabaseManager, 'insert_batch')
    def test_save_feedback_exception(self, mock_insert):
        """Test feedback saving with exception"""
        mock_insert.side_effect = Exception('Database error')
        
        db_manager = Mock()
        result = save_feedback(db_manager, user_id=123, movie_id=456, 
                              feedback_type='like', rating=4.5)
        
        assert result is False