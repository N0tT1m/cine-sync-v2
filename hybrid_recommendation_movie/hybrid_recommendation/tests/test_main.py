import pytest
import torch
import pandas as pd
import numpy as np
import tempfile
import os
import shutil
from unittest.mock import Mock, patch, MagicMock
from main import (
    setup_logging,
    setup_gpu,
    create_id_mappings,
    find_data_files,
    process_csv_file,
    process_ratings_file,
    process_and_prepare_data,
    MovieRatingDataset
)
from config import AppConfig


class TestMainUtilities:
    """Test utility functions from main.py"""
    
    def test_setup_logging_debug(self):
        """Test logging setup in debug mode"""
        mock_config = Mock()
        mock_config.debug = True
        
        logger = setup_logging(mock_config)
        
        assert logger.name == "movie-recommendation-training"
        assert logger.level <= 10  # DEBUG level
    
    def test_setup_logging_info(self):
        """Test logging setup in info mode"""
        mock_config = Mock()
        mock_config.debug = False
        
        logger = setup_logging(mock_config)
        
        assert logger.name == "movie-recommendation-training"
        assert logger.level <= 20  # INFO level
    
    @patch('torch.cuda.is_available')
    @patch('torch.cuda.device_count')
    @patch('torch.cuda.get_device_name')
    @patch('torch.cuda.mem_get_info')
    def test_setup_gpu_available(self, mock_mem_info, mock_device_name, 
                                mock_device_count, mock_cuda_available):
        """Test GPU setup when CUDA is available"""
        mock_cuda_available.return_value = True
        mock_device_count.return_value = 1
        mock_device_name.return_value = "Test GPU"
        mock_mem_info.return_value = (8000000000, 12000000000)  # 8GB free, 12GB total
        
        device = setup_gpu()
        
        assert device.type == 'cuda'
        assert device.index == 0
    
    @patch('torch.cuda.is_available')
    def test_setup_gpu_not_available(self, mock_cuda_available):
        """Test GPU setup when CUDA is not available"""
        mock_cuda_available.return_value = False
        
        device = setup_gpu()
        
        assert device.type == 'cpu'
    
    @patch('torch.cuda.is_available')
    @patch('torch.cuda.device_count')
    def test_setup_gpu_no_devices(self, mock_device_count, mock_cuda_available):
        """Test GPU setup when CUDA is available but no devices"""
        mock_cuda_available.return_value = True
        mock_device_count.return_value = 0
        
        device = setup_gpu()
        
        assert device.type == 'cpu'


class TestIDMappings:
    """Test ID mapping functionality"""
    
    @pytest.fixture
    def sample_ratings_df(self):
        """Create sample ratings DataFrame"""
        return pd.DataFrame({
            'userId': [1, 2, 3, 1, 2],
            'movieId': [10, 20, 30, 20, 30],
            'rating': [4.0, 3.5, 5.0, 4.5, 3.0],
            'timestamp': [1000, 2000, 3000, 4000, 5000]
        })
    
    @pytest.fixture
    def sample_movies_df(self):
        """Create sample movies DataFrame"""
        return pd.DataFrame({
            'media_id': [10, 20, 30, 40],
            'title': ['Movie A', 'Movie B', 'Movie C', 'Movie D'],
            'genres': ['Action', 'Comedy', 'Drama', 'Horror']
        })
    
    def test_create_id_mappings_success(self, sample_ratings_df, sample_movies_df):
        """Test successful ID mapping creation"""
        result = create_id_mappings(sample_ratings_df, sample_movies_df)
        
        assert result is not None
        ratings_df, movies_df, mappings = result
        
        # Check mappings structure
        assert 'user_id_to_idx' in mappings
        assert 'movie_id_to_idx' in mappings
        assert 'idx_to_user_id' in mappings
        assert 'idx_to_movie_id' in mappings
        assert 'num_users' in mappings
        assert 'num_movies' in mappings
        
        # Check mapping correctness
        assert mappings['num_users'] == 3  # Users 1, 2, 3
        assert mappings['num_movies'] == 3  # Movies 10, 20, 30 (only ones with ratings)
        
        # Check that indices are contiguous starting from 0
        user_indices = list(mappings['user_id_to_idx'].values())
        movie_indices = list(mappings['movie_id_to_idx'].values())
        
        assert sorted(user_indices) == list(range(3))
        assert sorted(movie_indices) == list(range(3))
        
        # Check that mapped indices are added to DataFrames
        assert 'user_idx' in ratings_df.columns
        assert 'movie_idx' in ratings_df.columns
        assert 'movie_idx' in movies_df.columns
        
        # Check no missing mappings
        assert not ratings_df['user_idx'].isna().any()
        assert not ratings_df['movie_idx'].isna().any()
    
    def test_create_id_mappings_reverse_mappings(self, sample_ratings_df, sample_movies_df):
        """Test that reverse mappings are correct"""
        result = create_id_mappings(sample_ratings_df, sample_movies_df)
        ratings_df, movies_df, mappings = result
        
        # Test user mappings
        for user_id, idx in mappings['user_id_to_idx'].items():
            assert mappings['idx_to_user_id'][idx] == user_id
        
        # Test movie mappings
        for movie_id, idx in mappings['movie_id_to_idx'].items():
            assert mappings['idx_to_movie_id'][idx] == movie_id


class TestDataProcessing:
    """Test data processing functions"""
    
    def test_find_data_files_existing(self):
        """Test finding data files when they exist"""
        # Create temporary test files
        temp_dir = tempfile.mkdtemp()
        tmdb_dir = os.path.join(temp_dir, 'tmdb')
        os.makedirs(tmdb_dir)
        
        test_file = os.path.join(tmdb_dir, 'actor_filmography_data_movies.csv')
        with open(test_file, 'w') as f:
            f.write('test,data\n1,2\n')
        
        try:
            with patch('os.path.abspath') as mock_abspath:
                mock_abspath.return_value = temp_dir
                
                found_files = find_data_files()
                
                assert 'tmdb_movies' in found_files
                assert test_file in found_files['tmdb_movies']
                
        finally:
            shutil.rmtree(temp_dir)
    
    def test_find_data_files_none_exist(self):
        """Test finding data files when none exist"""
        with patch('os.path.exists') as mock_exists:
            mock_exists.return_value = False
            
            found_files = find_data_files()
            
            assert len(found_files) == 0
    
    def test_process_csv_file_movielens_format(self):
        """Test processing MovieLens format CSV file"""
        # Create temporary CSV file
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
        temp_file.write('movieId,title,genres\n')
        temp_file.write('1,Toy Story (1995),Adventure|Animation|Children|Comedy|Fantasy\n')
        temp_file.write('2,Jumanji (1995),Adventure|Children|Fantasy\n')
        temp_file.close()
        
        try:
            result = process_csv_file(temp_file.name)
            
            assert result is not None
            assert 'media_id' in result.columns
            assert 'title' in result.columns
            assert 'genres' in result.columns
            assert len(result) == 2
            
        finally:
            os.unlink(temp_file.name)
    
    def test_process_csv_file_standard_format(self):
        """Test processing standard format CSV file"""
        # Create temporary CSV file
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
        temp_file.write('actor_id,media_id,genres\n')
        temp_file.write('1,10,Action|Adventure\n')
        temp_file.write('2,20,Comedy|Romance\n')
        temp_file.close()
        
        try:
            result = process_csv_file(temp_file.name)
            
            assert result is not None
            assert 'actor_id' in result.columns
            assert 'media_id' in result.columns
            assert 'genres_list' in result.columns
            assert len(result) == 2
            
        finally:
            os.unlink(temp_file.name)
    
    def test_process_csv_file_invalid(self):
        """Test processing invalid CSV file"""
        # Create temporary file with invalid content
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
        temp_file.write('invalid,content\n')
        temp_file.write('no,proper,format\n')
        temp_file.close()
        
        try:
            result = process_csv_file(temp_file.name)
            
            assert result is None
            
        finally:
            os.unlink(temp_file.name)
    
    def test_process_ratings_file_csv(self):
        """Test processing ratings CSV file"""
        # Create temporary ratings file
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
        temp_file.write('userId,movieId,rating,timestamp\n')
        temp_file.write('1,10,4.0,1000\n')
        temp_file.write('2,20,3.5,2000\n')
        temp_file.close()
        
        try:
            result = process_ratings_file(temp_file.name, format_type='movielens')
            
            assert result is not None
            assert 'userId' in result.columns
            assert 'movieId' in result.columns
            assert 'rating' in result.columns
            assert 'timestamp' in result.columns
            assert len(result) == 2
            
        finally:
            os.unlink(temp_file.name)
    
    def test_process_ratings_file_dat(self):
        """Test processing ratings .dat file"""
        # Create temporary .dat file
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.dat', delete=False)
        temp_file.write('1::10::4.0::1000\n')
        temp_file.write('2::20::3.5::2000\n')
        temp_file.close()
        
        try:
            result = process_ratings_file(temp_file.name, format_type='movielens')
            
            assert result is not None
            assert 'userId' in result.columns
            assert 'movieId' in result.columns
            assert 'rating' in result.columns
            assert 'timestamp' in result.columns
            assert len(result) == 2
            
        finally:
            os.unlink(temp_file.name)


class TestMovieRatingDataset:
    """Test the MovieRatingDataset class"""
    
    @pytest.fixture
    def sample_dataset_data(self):
        """Create sample dataset data"""
        return {
            'user_indices': torch.tensor([0, 1, 2, 0, 1]),
            'movie_indices': torch.tensor([0, 1, 2, 1, 2]),
            'genre_features': torch.randn(5, 10),
            'ratings': torch.tensor([4.0, 3.5, 5.0, 4.5, 3.0])
        }
    
    def test_dataset_creation(self, sample_dataset_data):
        """Test dataset creation"""
        dataset = MovieRatingDataset(
            sample_dataset_data['user_indices'],
            sample_dataset_data['movie_indices'],
            sample_dataset_data['genre_features'],
            sample_dataset_data['ratings']
        )
        
        assert len(dataset) == 5
        assert hasattr(dataset, 'user_indices')
        assert hasattr(dataset, 'movie_indices')
        assert hasattr(dataset, 'genre_features')
        assert hasattr(dataset, 'ratings')
    
    def test_dataset_getitem(self, sample_dataset_data):
        """Test dataset item access"""
        dataset = MovieRatingDataset(
            sample_dataset_data['user_indices'],
            sample_dataset_data['movie_indices'],
            sample_dataset_data['genre_features'],
            sample_dataset_data['ratings']
        )
        
        user_idx, movie_idx, genre_feat, rating = dataset[0]
        
        assert user_idx == 0
        assert movie_idx == 0
        assert genre_feat.shape == (10,)
        assert rating == 4.0
        
        assert isinstance(user_idx, torch.Tensor)
        assert isinstance(movie_idx, torch.Tensor)
        assert isinstance(genre_feat, torch.Tensor)
        assert isinstance(rating, torch.Tensor)
    
    def test_dataset_length(self, sample_dataset_data):
        """Test dataset length"""
        dataset = MovieRatingDataset(
            sample_dataset_data['user_indices'],
            sample_dataset_data['movie_indices'],
            sample_dataset_data['genre_features'],
            sample_dataset_data['ratings']
        )
        
        assert len(dataset) == len(sample_dataset_data['ratings'])
    
    def test_dataset_empty(self):
        """Test empty dataset"""
        dataset = MovieRatingDataset(
            torch.tensor([]),
            torch.tensor([]),
            torch.empty(0, 10),
            torch.tensor([])
        )
        
        assert len(dataset) == 0


class TestProcessAndPrepareData:
    """Test the process_and_prepare_data function"""
    
    def test_process_and_prepare_data_no_files(self):
        """Test when no data files are provided"""
        result = process_and_prepare_data({})
        
        # Should return None for all outputs when no data is found
        assert all(r is None for r in result)
    
    @patch('main.process_csv_file')
    @patch('main.process_ratings_file')
    def test_process_and_prepare_data_success(self, mock_process_ratings, mock_process_csv):
        """Test successful data processing and preparation"""
        # Mock movie data
        mock_movies_df = pd.DataFrame({
            'media_id': [1, 2, 3],
            'title': ['Movie A', 'Movie B', 'Movie C'],
            'genres': ['Action|Adventure', 'Comedy', 'Drama|Thriller']
        })
        mock_process_csv.return_value = mock_movies_df
        
        # Mock ratings data
        mock_ratings_df = pd.DataFrame({
            'userId': [1, 2, 3, 1, 2],
            'movieId': [1, 2, 3, 2, 3],
            'rating': [4.0, 3.5, 5.0, 4.5, 3.0],
            'timestamp': [1000, 2000, 3000, 4000, 5000]
        })
        mock_process_ratings.return_value = mock_ratings_df
        
        data_files = {
            'ml_movies': 'test_movies.csv',
            'ml_ratings': 'test_ratings.csv'
        }
        
        with patch('main.create_id_mappings') as mock_create_mappings:
            # Mock successful ID mapping
            mock_mappings = {
                'user_id_to_idx': {1: 0, 2: 1, 3: 2},
                'movie_id_to_idx': {1: 0, 2: 1, 3: 2},
                'num_users': 3,
                'num_movies': 3
            }
            mock_create_mappings.return_value = (mock_ratings_df, mock_movies_df, mock_mappings)
            
            with patch('os.makedirs'), patch('pickle.dump'):
                result = process_and_prepare_data(data_files)
                
                train_data, val_data, movies_df, genres_list, mappings = result
                
                assert train_data is not None
                assert val_data is not None
                assert movies_df is not None
                assert genres_list is not None
                assert mappings is not None
                
                # Check that genres were extracted
                expected_genres = ['Action', 'Adventure', 'Comedy', 'Drama', 'Thriller']
                assert sorted(genres_list) == sorted(expected_genres)