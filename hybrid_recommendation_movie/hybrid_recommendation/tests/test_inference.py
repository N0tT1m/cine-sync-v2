import pytest
import torch
import numpy as np
import pandas as pd
import tempfile
import os
import pickle
from unittest.mock import Mock, patch, MagicMock
from inference import MovieRecommendationSystem
from models.hybrid_recommender import HybridRecommenderModel
from config import AppConfig, ModelConfig, DatabaseConfig


class TestMovieRecommendationSystem:
    """Test the MovieRecommendationSystem class"""
    
    @pytest.fixture
    def mock_config(self):
        """Create a mock configuration"""
        config = Mock(spec=AppConfig)
        config.model = Mock(spec=ModelConfig)
        config.model.models_dir = 'test_models'
        config.database = Mock(spec=DatabaseConfig)
        config.debug = False
        return config
    
    @pytest.fixture
    def sample_mappings(self):
        """Create sample ID mappings"""
        return {
            'user_id_to_idx': {1: 0, 2: 1, 3: 2},
            'movie_id_to_idx': {10: 0, 20: 1, 30: 2},
            'idx_to_user_id': {0: 1, 1: 2, 2: 3},
            'idx_to_movie_id': {0: 10, 1: 20, 2: 30},
            'num_users': 3,
            'num_movies': 3
        }
    
    @pytest.fixture
    def sample_movie_lookup(self):
        """Create sample movie lookup table"""
        return {
            10: {'title': 'Movie A', 'genres': 'Action|Adventure', 'movie_idx': 0},
            20: {'title': 'Movie B', 'genres': 'Comedy|Romance', 'movie_idx': 1},
            30: {'title': 'Movie C', 'genres': 'Drama|Thriller', 'movie_idx': 2}
        }
    
    @pytest.fixture
    def sample_movies_df(self):
        """Create sample movies DataFrame"""
        return pd.DataFrame({
            'media_id': [10, 20, 30],
            'title': ['Movie A', 'Movie B', 'Movie C'],
            'genres': ['Action|Adventure', 'Comedy|Romance', 'Drama|Thriller'],
            'Action': [1, 0, 0],
            'Adventure': [1, 0, 0],
            'Comedy': [0, 1, 0],
            'Romance': [0, 1, 0],
            'Drama': [0, 0, 1],
            'Thriller': [0, 0, 1]
        })
    
    @pytest.fixture
    def sample_model(self):
        """Create a sample trained model"""
        model = HybridRecommenderModel(
            num_users=3,
            num_movies=3,
            embedding_dim=16,
            hidden_dim=32
        )
        model.eval()
        return model
    
    def create_temp_model_files(self, mappings, movie_lookup, movies_df, model):
        """Create temporary model files for testing"""
        temp_dir = tempfile.mkdtemp()
        
        # Save mappings
        with open(os.path.join(temp_dir, 'id_mappings.pkl'), 'wb') as f:
            pickle.dump(mappings, f)
        
        # Save movie lookup
        with open(os.path.join(temp_dir, 'movie_lookup.pkl'), 'wb') as f:
            pickle.dump(movie_lookup, f)
        
        # Save movies DataFrame
        movies_df.to_csv(os.path.join(temp_dir, 'movies_data.csv'), index=False)
        
        # Save rating scaler
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
        scaler.fit([[1], [5]])  # Fit to rating range 1-5
        with open(os.path.join(temp_dir, 'rating_scaler.pkl'), 'wb') as f:
            pickle.dump(scaler, f)
        
        # Save model metadata
        metadata = {
            'num_users': 3,
            'num_movies': 3,
            'genres': ['Action', 'Adventure', 'Comedy', 'Romance', 'Drama', 'Thriller'],
            'embedding_size': 16
        }
        with open(os.path.join(temp_dir, 'model_metadata.pkl'), 'wb') as f:
            pickle.dump(metadata, f)
        
        # Save model weights
        torch.save({
            'model_state_dict': model.state_dict(),
            'mappings': mappings
        }, os.path.join(temp_dir, 'best_model.pt'))
        
        return temp_dir
    
    @patch('utils.database.DatabaseManager')
    def test_init_success(self, mock_db_manager, mock_config, sample_mappings, 
                         sample_movie_lookup, sample_movies_df, sample_model):
        """Test successful initialization of MovieRecommendationSystem"""
        temp_dir = self.create_temp_model_files(
            sample_mappings, sample_movie_lookup, sample_movies_df, sample_model
        )
        
        try:
            mock_config.model.models_dir = temp_dir
            
            rec_system = MovieRecommendationSystem(mock_config)
            
            assert rec_system.mappings == sample_mappings
            assert rec_system.movie_lookup == sample_movie_lookup
            assert isinstance(rec_system.model, HybridRecommenderModel)
            assert len(rec_system.genres_list) == 6
            
        finally:
            # Cleanup
            import shutil
            shutil.rmtree(temp_dir)
    
    def test_get_movie_genre_features(self, mock_config, sample_mappings, 
                                    sample_movie_lookup, sample_movies_df, sample_model):
        """Test getting genre features for movies"""
        temp_dir = self.create_temp_model_files(
            sample_mappings, sample_movie_lookup, sample_movies_df, sample_model
        )
        
        try:
            mock_config.model.models_dir = temp_dir
            
            with patch('utils.database.DatabaseManager'):
                rec_system = MovieRecommendationSystem(mock_config)
                
                # Test valid movie
                features = rec_system._get_movie_genre_features(10)
                assert len(features) == 6  # Number of genres
                assert features[0] == 1  # Action
                assert features[1] == 1  # Adventure
                assert features[2] == 0  # Comedy
                
                # Test invalid movie
                features = rec_system._get_movie_genre_features(999)
                assert len(features) == 6
                assert all(f == 0 for f in features)
                
        finally:
            import shutil
            shutil.rmtree(temp_dir)
    
    def test_get_user_recommendations(self, mock_config, sample_mappings, 
                                    sample_movie_lookup, sample_movies_df, sample_model):
        """Test getting user recommendations"""
        temp_dir = self.create_temp_model_files(
            sample_mappings, sample_movie_lookup, sample_movies_df, sample_model
        )
        
        try:
            mock_config.model.models_dir = temp_dir
            
            with patch('utils.database.DatabaseManager'):
                rec_system = MovieRecommendationSystem(mock_config)
                
                # Test valid user
                recommendations = rec_system.get_user_recommendations(1, num_recommendations=2)
                
                assert len(recommendations) == 2
                for rec in recommendations:
                    assert 'movie_id' in rec
                    assert 'title' in rec
                    assert 'genres' in rec
                    assert 'predicted_rating' in rec
                    assert 'confidence' in rec
                    assert rec['movie_id'] in [10, 20, 30]
                
                # Check that recommendations are sorted by predicted rating
                assert recommendations[0]['predicted_rating'] >= recommendations[1]['predicted_rating']
                
        finally:
            import shutil
            shutil.rmtree(temp_dir)
    
    def test_get_user_recommendations_invalid_user(self, mock_config, sample_mappings, 
                                                  sample_movie_lookup, sample_movies_df, sample_model):
        """Test getting recommendations for invalid user (fallback behavior)"""
        temp_dir = self.create_temp_model_files(
            sample_mappings, sample_movie_lookup, sample_movies_df, sample_model
        )
        
        try:
            mock_config.model.models_dir = temp_dir
            
            with patch('utils.database.DatabaseManager'):
                rec_system = MovieRecommendationSystem(mock_config)
                
                # Test invalid user (should fall back to existing user)
                recommendations = rec_system.get_user_recommendations(999, num_recommendations=2)
                
                assert len(recommendations) == 2
                # Should still return valid recommendations
                for rec in recommendations:
                    assert rec['movie_id'] in [10, 20, 30]
                
        finally:
            import shutil
            shutil.rmtree(temp_dir)
    
    def test_get_movie_info(self, mock_config, sample_mappings, 
                           sample_movie_lookup, sample_movies_df, sample_model):
        """Test getting movie information"""
        temp_dir = self.create_temp_model_files(
            sample_mappings, sample_movie_lookup, sample_movies_df, sample_model
        )
        
        try:
            mock_config.model.models_dir = temp_dir
            
            with patch('utils.database.DatabaseManager'):
                rec_system = MovieRecommendationSystem(mock_config)
                
                # Test valid movie
                movie_info = rec_system.get_movie_info(10)
                assert movie_info is not None
                assert movie_info['title'] == 'Movie A'
                assert movie_info['genres'] == 'Action|Adventure'
                
                # Test invalid movie
                movie_info = rec_system.get_movie_info(999)
                assert movie_info is None
                
        finally:
            import shutil
            shutil.rmtree(temp_dir)
    
    def test_search_movies(self, mock_config, sample_mappings, 
                          sample_movie_lookup, sample_movies_df, sample_model):
        """Test movie search functionality"""
        temp_dir = self.create_temp_model_files(
            sample_mappings, sample_movie_lookup, sample_movies_df, sample_model
        )
        
        try:
            mock_config.model.models_dir = temp_dir
            
            with patch('utils.database.DatabaseManager'):
                rec_system = MovieRecommendationSystem(mock_config)
                
                # Test exact match
                results = rec_system.search_movies('Movie A')
                assert len(results) == 1
                assert results[0]['title'] == 'Movie A'
                assert results[0]['movie_id'] == 10
                
                # Test partial match
                results = rec_system.search_movies('movie')
                assert len(results) == 3  # Should match all movies
                
                # Test case insensitive
                results = rec_system.search_movies('MOVIE A')
                assert len(results) == 1
                assert results[0]['title'] == 'Movie A'
                
                # Test no match
                results = rec_system.search_movies('Nonexistent')
                assert len(results) == 0
                
                # Test limit
                results = rec_system.search_movies('movie', limit=2)
                assert len(results) == 2
                
        finally:
            import shutil
            shutil.rmtree(temp_dir)
    
    def test_model_loading_error(self, mock_config):
        """Test error handling when model files are missing"""
        mock_config.model.models_dir = '/nonexistent/path'
        
        with patch('utils.database.DatabaseManager'):
            with pytest.raises(Exception):  # Should raise an exception when files are missing
                MovieRecommendationSystem(mock_config)
    
    @patch('torch.cuda.is_available')
    def test_device_selection_cuda(self, mock_cuda_available, mock_config, 
                                  sample_mappings, sample_movie_lookup, 
                                  sample_movies_df, sample_model):
        """Test CUDA device selection when available"""
        mock_cuda_available.return_value = True
        temp_dir = self.create_temp_model_files(
            sample_mappings, sample_movie_lookup, sample_movies_df, sample_model
        )
        
        try:
            mock_config.model.models_dir = temp_dir
            
            with patch('utils.database.DatabaseManager'):
                rec_system = MovieRecommendationSystem(mock_config)
                assert rec_system.device.type == 'cuda'
                
        finally:
            import shutil
            shutil.rmtree(temp_dir)
    
    @patch('torch.cuda.is_available')
    def test_device_selection_cpu(self, mock_cuda_available, mock_config,
                                 sample_mappings, sample_movie_lookup, 
                                 sample_movies_df, sample_model):
        """Test CPU device selection when CUDA not available"""
        mock_cuda_available.return_value = False
        temp_dir = self.create_temp_model_files(
            sample_mappings, sample_movie_lookup, sample_movies_df, sample_model
        )
        
        try:
            mock_config.model.models_dir = temp_dir
            
            with patch('utils.database.DatabaseManager'):
                rec_system = MovieRecommendationSystem(mock_config)
                assert rec_system.device.type == 'cpu'
                
        finally:
            import shutil
            shutil.rmtree(temp_dir)