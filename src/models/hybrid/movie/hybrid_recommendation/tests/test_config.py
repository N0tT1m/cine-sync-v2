import pytest
import os
from unittest.mock import patch
from config import load_config, AppConfig, DatabaseConfig, ModelConfig, DiscordConfig, ServerConfig


class TestConfig:
    """Test configuration loading and validation"""
    
    def test_load_config_with_defaults(self):
        """Test that config loads with default values when env vars not set"""
        config = load_config()
        
        assert isinstance(config, AppConfig)
        assert isinstance(config.database, DatabaseConfig)
        assert isinstance(config.model, ModelConfig)
        assert isinstance(config.discord, DiscordConfig)
        assert isinstance(config.server, ServerConfig)
        
        # Test default values
        assert config.database.host == '192.168.1.78'
        assert config.database.database == 'cinesync'
        assert config.database.user == 'postgres'
        assert config.database.port == 5432
        
        assert config.model.embedding_dim == 64
        assert config.model.hidden_dim == 128
        assert config.model.num_epochs == 10
        assert config.model.batch_size == 1024
        assert config.model.learning_rate == 0.001
        assert config.model.models_dir == 'models'
        
        assert config.discord.command_prefix == '!'
        
        assert config.server.host == 'localhost'
        assert config.server.port == 3000
        assert config.server.api_base_url == 'http://localhost:3000'
        
        assert config.debug is False
    
    @patch.dict(os.environ, {
        'DB_HOST': 'test_host',
        'DB_NAME': 'test_db',
        'DB_USER': 'test_user',
        'DB_PASSWORD': 'test_password',
        'DB_PORT': '5433',
        'MODEL_EMBEDDING_DIM': '128',
        'MODEL_HIDDEN_DIM': '256',
        'MODEL_EPOCHS': '20',
        'MODEL_BATCH_SIZE': '512',
        'MODEL_LEARNING_RATE': '0.002',
        'MODELS_DIR': 'test_models',
        'DISCORD_TOKEN': 'test_token',
        'DISCORD_PREFIX': '?',
        'SERVER_HOST': '0.0.0.0',
        'SERVER_PORT': '8080',
        'API_BASE_URL': 'http://api.test.com',
        'DEBUG': 'true'
    })
    def test_load_config_with_env_vars(self):
        """Test that config loads with environment variables"""
        config = load_config()
        
        # Test database config
        assert config.database.host == 'test_host'
        assert config.database.database == 'test_db'
        assert config.database.user == 'test_user'
        assert config.database.password == 'test_password'
        assert config.database.port == 5433
        
        # Test model config
        assert config.model.embedding_dim == 128
        assert config.model.hidden_dim == 256
        assert config.model.num_epochs == 20
        assert config.model.batch_size == 512
        assert config.model.learning_rate == 0.002
        assert config.model.models_dir == 'test_models'
        
        # Test discord config
        assert config.discord.token == 'test_token'
        assert config.discord.command_prefix == '?'
        
        # Test server config
        assert config.server.host == '0.0.0.0'
        assert config.server.port == 8080
        assert config.server.api_base_url == 'http://api.test.com'
        
        assert config.debug is True
    
    @patch.dict(os.environ, {'DEBUG': 'false'})
    def test_debug_flag_false(self):
        """Test debug flag set to false"""
        config = load_config()
        assert config.debug is False
    
    @patch.dict(os.environ, {'DEBUG': 'True'})
    def test_debug_flag_true_mixed_case(self):
        """Test debug flag with mixed case"""
        config = load_config()
        assert config.debug is True
    
    @patch.dict(os.environ, {'MODEL_EPOCHS': 'invalid'})
    def test_invalid_int_env_var(self):
        """Test that invalid integer environment variable raises ValueError"""
        with pytest.raises(ValueError):
            load_config()
    
    @patch.dict(os.environ, {'MODEL_LEARNING_RATE': 'invalid'})
    def test_invalid_float_env_var(self):
        """Test that invalid float environment variable raises ValueError"""
        with pytest.raises(ValueError):
            load_config()