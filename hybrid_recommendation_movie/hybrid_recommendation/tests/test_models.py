import pytest
import torch
import torch.nn as nn
import numpy as np
import tempfile
import os
from models.hybrid_recommender import (
    HybridRecommenderModel,
    MovieDataset,
    create_model,
    save_model,
    load_model
)


class TestHybridRecommenderModel:
    """Test the hybrid recommender model"""
    
    @pytest.fixture
    def sample_model(self):
        """Create a sample model for testing"""
        return HybridRecommenderModel(
            num_users=100,
            num_movies=200,
            embedding_dim=32,
            hidden_dim=64
        )
    
    def test_model_initialization(self, sample_model):
        """Test model initialization"""
        assert sample_model.num_users == 100
        assert sample_model.num_movies == 200
        assert sample_model.embedding_dim == 32
        assert sample_model.hidden_dim == 64
        
        # Check that all layers are properly initialized
        assert isinstance(sample_model.user_embedding, nn.Embedding)
        assert isinstance(sample_model.movie_embedding, nn.Embedding)
        assert isinstance(sample_model.user_bias, nn.Embedding)
        assert isinstance(sample_model.movie_bias, nn.Embedding)
        assert isinstance(sample_model.global_bias, nn.Parameter)
        
        # Check embedding dimensions
        assert sample_model.user_embedding.num_embeddings == 100
        assert sample_model.user_embedding.embedding_dim == 32
        assert sample_model.movie_embedding.num_embeddings == 200
        assert sample_model.movie_embedding.embedding_dim == 32
    
    def test_forward_pass(self, sample_model):
        """Test forward pass with valid inputs"""
        batch_size = 10
        user_ids = torch.randint(0, 100, (batch_size,))
        movie_ids = torch.randint(0, 200, (batch_size,))
        
        predictions = sample_model(user_ids, movie_ids)
        
        assert predictions.shape == (batch_size,)
        assert predictions.dtype == torch.float32
        assert not torch.isnan(predictions).any()
        assert not torch.isinf(predictions).any()
    
    def test_forward_pass_single_sample(self, sample_model):
        """Test forward pass with single sample"""
        user_id = torch.tensor([0])
        movie_id = torch.tensor([0])
        
        prediction = sample_model(user_id, movie_id)
        
        assert prediction.shape == (1,)
        assert not torch.isnan(prediction).any()
    
    def test_predict_for_user(self, sample_model):
        """Test batch prediction for a single user"""
        user_idx = 5
        movie_indices = torch.randint(0, 200, (20,))
        device = torch.device('cpu')
        
        predictions = sample_model.predict_for_user(user_idx, movie_indices, device)
        
        assert predictions.shape == (20,)
        assert not torch.isnan(predictions).any()
    
    def test_get_embeddings(self, sample_model):
        """Test getting user and movie embeddings"""
        user_embeddings = sample_model.get_user_embeddings()
        movie_embeddings = sample_model.get_movie_embeddings()
        
        assert user_embeddings.shape == (100, 32)
        assert movie_embeddings.shape == (200, 32)
        assert user_embeddings.dtype == torch.float32
        assert movie_embeddings.dtype == torch.float32
    
    def test_invalid_user_id(self, sample_model):
        """Test that invalid user IDs raise appropriate errors"""
        user_ids = torch.tensor([100])  # Out of bounds
        movie_ids = torch.tensor([0])
        
        with pytest.raises(IndexError):
            sample_model(user_ids, movie_ids)
    
    def test_invalid_movie_id(self, sample_model):
        """Test that invalid movie IDs raise appropriate errors"""
        user_ids = torch.tensor([0])
        movie_ids = torch.tensor([200])  # Out of bounds
        
        with pytest.raises(IndexError):
            sample_model(user_ids, movie_ids)
    
    def test_model_training_mode(self, sample_model):
        """Test model training mode functionality"""
        sample_model.train()
        assert sample_model.training
        
        sample_model.eval()
        assert not sample_model.training
    
    def test_gradient_computation(self, sample_model):
        """Test that gradients are computed correctly"""
        sample_model.train()
        
        user_ids = torch.tensor([0, 1, 2])
        movie_ids = torch.tensor([0, 1, 2])
        targets = torch.tensor([4.0, 3.5, 5.0])
        
        predictions = sample_model(user_ids, movie_ids)
        loss = nn.MSELoss()(predictions, targets)
        loss.backward()
        
        # Check that gradients are computed
        assert sample_model.user_embedding.weight.grad is not None
        assert sample_model.movie_embedding.weight.grad is not None


class TestMovieDataset:
    """Test the MovieDataset class"""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing"""
        return {
            'user_ids': np.array([1, 2, 3, 4, 5]),
            'movie_ids': np.array([10, 20, 30, 40, 50]),
            'ratings': np.array([4.0, 3.5, 5.0, 2.5, 4.5])
        }
    
    def test_dataset_initialization(self, sample_data):
        """Test dataset initialization"""
        dataset = MovieDataset(
            sample_data['user_ids'],
            sample_data['movie_ids'],
            sample_data['ratings']
        )
        
        assert len(dataset) == 5
        assert isinstance(dataset.user_ids, torch.LongTensor)
        assert isinstance(dataset.movie_ids, torch.LongTensor)
        assert isinstance(dataset.ratings, torch.FloatTensor)
    
    def test_dataset_getitem(self, sample_data):
        """Test dataset item access"""
        dataset = MovieDataset(
            sample_data['user_ids'],
            sample_data['movie_ids'],
            sample_data['ratings']
        )
        
        user_id, movie_id, rating = dataset[0]
        
        assert user_id == 1
        assert movie_id == 10
        assert rating == 4.0
        assert isinstance(user_id, torch.Tensor)
        assert isinstance(movie_id, torch.Tensor)
        assert isinstance(rating, torch.Tensor)
    
    def test_dataset_length(self, sample_data):
        """Test dataset length"""
        dataset = MovieDataset(
            sample_data['user_ids'],
            sample_data['movie_ids'],
            sample_data['ratings']
        )
        
        assert len(dataset) == len(sample_data['user_ids'])
    
    def test_empty_dataset(self):
        """Test empty dataset"""
        dataset = MovieDataset(
            np.array([]),
            np.array([]),
            np.array([])
        )
        
        assert len(dataset) == 0


class TestModelUtilities:
    """Test model utility functions"""
    
    def test_create_model(self):
        """Test model creation utility"""
        model = create_model(num_users=50, num_movies=100, embedding_dim=16, hidden_dim=32)
        
        assert isinstance(model, HybridRecommenderModel)
        assert model.num_users == 50
        assert model.num_movies == 100
        assert model.embedding_dim == 16
        assert model.hidden_dim == 32
    
    def test_save_and_load_model(self):
        """Test model saving and loading"""
        # Create a model and some dummy data
        model = create_model(num_users=10, num_movies=20, embedding_dim=8, hidden_dim=16)
        mappings = {
            'user_id_to_idx': {1: 0, 2: 1, 3: 2},
            'movie_id_to_idx': {10: 0, 20: 1, 30: 2},
            'num_users': 10,
            'num_movies': 20
        }
        
        # Train model briefly to set some weights
        model.train()
        user_ids = torch.tensor([0, 1, 2])
        movie_ids = torch.tensor([0, 1, 2])
        targets = torch.tensor([4.0, 3.5, 5.0])
        
        predictions = model(user_ids, movie_ids)
        loss = nn.MSELoss()(predictions, targets)
        loss.backward()
        
        # Save original weights
        original_user_weights = model.user_embedding.weight.data.clone()
        original_movie_weights = model.movie_embedding.weight.data.clone()
        
        # Save and load model
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
            temp_path = f.name
        
        try:
            save_model(model, temp_path, mappings)
            loaded_model, loaded_mappings, _ = load_model(temp_path, torch.device('cpu'))
            
            # Check that model architecture is preserved
            assert loaded_model.num_users == model.num_users
            assert loaded_model.num_movies == model.num_movies
            assert loaded_model.embedding_dim == model.embedding_dim
            assert loaded_model.hidden_dim == model.hidden_dim
            
            # Check that weights are preserved
            assert torch.allclose(loaded_model.user_embedding.weight.data, original_user_weights)
            assert torch.allclose(loaded_model.movie_embedding.weight.data, original_movie_weights)
            
            # Check that mappings are preserved
            assert loaded_mappings == mappings
            
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def test_load_nonexistent_model(self):
        """Test loading a non-existent model file"""
        with pytest.raises(FileNotFoundError):
            load_model('nonexistent_model.pt', torch.device('cpu'))