"""
CineSync v2 - Unified Model Management System
Handles all 6 AI models + 2 hybrid recommenders with drop-in integration
"""

import os
import logging
import json
import pickle
import asyncio
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass
from datetime import datetime
import torch
import numpy as np

# Import all model types
from advanced_models.bert4rec_recommender import BERT4Rec, BERT4RecTrainer
from advanced_models.sentence_bert_two_tower import SentenceBERTTwoTowerModel
from advanced_models.graphsage_recommender import GraphSAGERecommender
from advanced_models.t5_hybrid_recommender import T5HybridRecommender
from advanced_models.enhanced_two_tower import EnhancedTwoTowerModel
from advanced_models.variational_autoencoder import VariationalAutoEncoder

# Import existing hybrid systems
from hybrid_recommendation_movie.hybrid_recommendation.models.content_manager import LupeContentManager as MovieManager
from hybrid_recommendation_tv.hybrid_recommendation.models.tv_recommender import TVRecommender


@dataclass
class ModelConfig:
    """Configuration for each model type"""
    name: str
    model_class: type
    config_path: str
    weights_path: str
    enabled: bool = True
    content_type: str = "both"  # "movie", "tv", "both"
    priority: int = 1  # Higher = more priority
    
    
class ModelStatus:
    """Track model loading and health status"""
    def __init__(self):
        self.loaded_models: Dict[str, bool] = {}
        self.model_errors: Dict[str, str] = {}
        self.last_updated: Dict[str, datetime] = {}


class UnifiedModelManager:
    """
    Central hub for managing all 6 AI models + 2 hybrid recommenders
    Provides drop-in integration, training management, and admin controls
    """
    
    def __init__(self, models_dir: str = "models", config_file: str = "model_config.json"):
        self.models_dir = Path(models_dir)
        self.config_file = config_file
        self.models: Dict[str, Any] = {}
        self.status = ModelStatus()
        self.logger = self._setup_logging()
        
        # Ensure models directory exists
        self.models_dir.mkdir(exist_ok=True)
        
        # Initialize model configurations
        self.model_configs = self._load_model_configs()
        
        # Training preferences
        self.training_preferences = {
            "auto_retrain": True,
            "min_feedback_threshold": 100,
            "excluded_genres": [],
            "excluded_users": [],
            "quality_filters": ["4K", "2K", "1080p"],  # Only train on high quality downloads
        }
        
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for model management"""
        logger = logging.getLogger("UnifiedModelManager")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.FileHandler("unified_model_manager.log")
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            
        return logger
        
    def _load_model_configs(self) -> Dict[str, ModelConfig]:
        """Load or create default model configurations"""
        configs = {
            "bert4rec": ModelConfig(
                name="BERT4Rec",
                model_class=BERT4Rec,
                config_path="configs/bert4rec_config.json",
                weights_path="weights/bert4rec.pt",
                content_type="both",
                priority=5
            ),
            "sentence_bert_two_tower": ModelConfig(
                name="Sentence-BERT Two-Tower",
                model_class=SentenceBERTTwoTowerModel,
                config_path="configs/sentence_bert_config.json",
                weights_path="weights/sentence_bert_two_tower.pt",
                content_type="both",
                priority=4
            ),
            "graphsage": ModelConfig(
                name="GraphSAGE",
                model_class=GraphSAGERecommender,
                config_path="configs/graphsage_config.json",
                weights_path="weights/graphsage.pt",
                content_type="both",
                priority=3
            ),
            "t5_hybrid": ModelConfig(
                name="T5 Hybrid",
                model_class=T5HybridRecommender,
                config_path="configs/t5_hybrid_config.json",
                weights_path="weights/t5_hybrid.pt",
                content_type="both",
                priority=4
            ),
            "enhanced_two_tower": ModelConfig(
                name="Enhanced Two-Tower",
                model_class=EnhancedTwoTowerModel,
                config_path="configs/enhanced_two_tower_config.json",
                weights_path="weights/enhanced_two_tower.pt",
                content_type="both",
                priority=3
            ),
            "variational_autoencoder": ModelConfig(
                name="Variational AutoEncoder",
                model_class=VariationalAutoEncoder,
                config_path="configs/vae_config.json",
                weights_path="weights/vae.pt",
                content_type="both",
                priority=2
            ),
            "hybrid_movie": ModelConfig(
                name="Hybrid Movie Recommender",
                model_class=MovieManager,
                config_path="hybrid_recommendation_movie/hybrid_recommendation/config.py",
                weights_path="hybrid_recommendation_movie/hybrid_recommendation/models/",
                content_type="movie",
                priority=5
            ),
            "hybrid_tv": ModelConfig(
                name="Hybrid TV Recommender", 
                model_class=TVRecommender,
                config_path="hybrid_recommendation_tv/hybrid_recommendation/config.py",
                weights_path="hybrid_recommendation_tv/hybrid_recommendation/models/",
                content_type="tv",
                priority=5
            )
        }
        
        # Save default config if doesn't exist
        if not os.path.exists(self.config_file):
            self.save_model_configs(configs)
            
        return configs
        
    def save_model_configs(self, configs: Dict[str, ModelConfig]):
        """Save model configurations to JSON"""
        config_dict = {}
        for key, config in configs.items():
            config_dict[key] = {
                "name": config.name,
                "config_path": config.config_path,
                "weights_path": config.weights_path,
                "enabled": config.enabled,
                "content_type": config.content_type,
                "priority": config.priority
            }
            
        with open(self.config_file, 'w') as f:
            json.dump(config_dict, f, indent=2)
            
    async def load_all_models(self) -> Dict[str, bool]:
        """Load all enabled models asynchronously"""
        self.logger.info("Starting to load all models...")
        
        load_tasks = []
        for model_name, config in self.model_configs.items():
            if config.enabled:
                task = asyncio.create_task(self._load_single_model(model_name, config))
                load_tasks.append(task)
                
        results = await asyncio.gather(*load_tasks, return_exceptions=True)
        
        # Process results
        success_count = 0
        for i, (model_name, config) in enumerate([item for item in self.model_configs.items() if item[1].enabled]):
            if isinstance(results[i], Exception):
                self.status.model_errors[model_name] = str(results[i])
                self.status.loaded_models[model_name] = False
                self.logger.error(f"Failed to load {model_name}: {results[i]}")
            else:
                self.status.loaded_models[model_name] = True
                success_count += 1
                self.logger.info(f"Successfully loaded {model_name}")
                
        self.logger.info(f"Loaded {success_count}/{len(load_tasks)} models successfully")
        return self.status.loaded_models
        
    async def _load_single_model(self, model_name: str, config: ModelConfig):
        """Load a single model with error handling"""
        try:
            weights_path = self.models_dir / config.weights_path
            
            if model_name in ["hybrid_movie", "hybrid_tv"]:
                # Load existing hybrid models
                model = config.model_class(config.weights_path)
                if hasattr(model, 'load_models'):
                    model.load_models()
            else:
                # Load advanced models
                if weights_path.exists():
                    model = torch.load(weights_path, map_location='cpu')
                else:
                    # Initialize new model if weights don't exist
                    model = self._initialize_new_model(model_name, config)
                    
            self.models[model_name] = model
            self.status.last_updated[model_name] = datetime.now()
            
        except Exception as e:
            raise Exception(f"Failed to load {model_name}: {str(e)}")
            
    def _initialize_new_model(self, model_name: str, config: ModelConfig):
        """Initialize a new model with default parameters"""
        default_params = {
            "bert4rec": {"num_items": 10000, "d_model": 512, "num_heads": 8, "num_layers": 6},
            "sentence_bert_two_tower": {"embedding_dim": 256, "sentence_bert_model": "all-MiniLM-L6-v2"},
            "graphsage": {"num_users": 100000, "num_items": 10000, "embedding_dim": 256},
            "t5_hybrid": {"num_users": 100000, "num_items": 10000, "embedding_dim": 256},
            "enhanced_two_tower": {"num_users": 100000, "num_items": 10000, "embedding_dim": 256},
            "variational_autoencoder": {"input_dim": 10000, "latent_dim": 256}
        }
        
        params = default_params.get(model_name, {})
        return config.model_class(**params)
        
    def get_recommendations(
        self, 
        user_id: int, 
        content_type: str = "movie", 
        top_k: int = 10,
        model_preference: Optional[str] = None
    ) -> List[Dict]:
        """
        Get recommendations using the best available model
        
        Args:
            user_id: User identifier
            content_type: "movie", "tv", or "both"
            top_k: Number of recommendations
            model_preference: Specific model to use (optional)
            
        Returns:
            List of recommendation dictionaries
        """
        
        # Select best model for content type
        available_models = self._get_available_models(content_type)
        
        if model_preference and model_preference in available_models:
            selected_model = model_preference
        else:
            # Select highest priority available model
            selected_model = max(available_models.keys(), 
                               key=lambda x: self.model_configs[x].priority)
            
        try:
            model = self.models[selected_model]
            
            # Call appropriate recommendation method based on model type
            if selected_model in ["hybrid_movie", "hybrid_tv"]:
                return model.get_recommendations(user_id, content_type, top_k)
            else:
                # For advanced models, use a standardized interface
                return self._get_advanced_model_recommendations(
                    model, selected_model, user_id, content_type, top_k
                )
                
        except Exception as e:
            self.logger.error(f"Recommendation failed for {selected_model}: {e}")
            # Fallback to next best model
            fallback_models = sorted(available_models.keys(), 
                                   key=lambda x: self.model_configs[x].priority, 
                                   reverse=True)[1:]
            
            for fallback in fallback_models:
                try:
                    return self.get_recommendations(user_id, content_type, top_k, fallback)
                except:
                    continue
                    
            return []  # No models available
            
    def _get_available_models(self, content_type: str) -> Dict[str, ModelConfig]:
        """Get models that support the requested content type"""
        available = {}
        for name, config in self.model_configs.items():
            if (name in self.status.loaded_models and 
                self.status.loaded_models[name] and
                (config.content_type == content_type or config.content_type == "both")):
                available[name] = config
        return available
        
    def _get_advanced_model_recommendations(
        self, model, model_name: str, user_id: int, content_type: str, top_k: int
    ) -> List[Dict]:
        """Standardized interface for advanced model recommendations"""
        
        # Each advanced model needs a wrapper to convert to standard format
        if hasattr(model, 'get_recommendations'):
            return model.get_recommendations(user_id, k=top_k)
        elif hasattr(model, 'predict'):
            # For models that only predict, we need item candidates
            candidates = self._get_item_candidates(content_type, top_k * 2)
            scores = model.predict(user_id, candidates)
            
            # Convert to recommendation format
            recommendations = []
            for item_id, score in zip(candidates, scores):
                recommendations.append({
                    "item_id": item_id,
                    "score": float(score),
                    "content_type": content_type
                })
                
            return sorted(recommendations, key=lambda x: x['score'], reverse=True)[:top_k]
        else:
            self.logger.warning(f"Model {model_name} doesn't have a recommendation interface")
            return []
            
    def _get_item_candidates(self, content_type: str, num_candidates: int) -> List[int]:
        """Get candidate items for recommendation scoring"""
        # This would connect to your database to get popular/relevant items
        # For now, return dummy candidates
        return list(range(1, num_candidates + 1))
        
    def add_feedback(self, user_id: int, item_id: int, rating: float, content_type: str):
        """Add user feedback for model retraining"""
        feedback = {
            "user_id": user_id,
            "item_id": item_id,
            "rating": rating,
            "content_type": content_type,
            "timestamp": datetime.now(),
            "download_quality": None  # Can be set by download service
        }
        
        # Apply training filters
        if self._should_include_in_training(feedback):
            self._store_feedback(feedback)
            
            # Check if we should trigger retraining
            if self.training_preferences["auto_retrain"]:
                self._maybe_trigger_retraining()
                
    def _should_include_in_training(self, feedback: Dict) -> bool:
        """Check if feedback should be included in training based on preferences"""
        
        # Check user exclusions
        if feedback["user_id"] in self.training_preferences["excluded_users"]:
            return False
            
        # Check quality filters (if download_quality is set)
        if (feedback.get("download_quality") and 
            feedback["download_quality"] not in self.training_preferences["quality_filters"]):
            return False
            
        # Add more filters as needed
        return True
        
    def _store_feedback(self, feedback: Dict):
        """Store feedback for later training"""
        # This would connect to your database
        feedback_file = self.models_dir / "training_feedback.jsonl"
        
        with open(feedback_file, 'a') as f:
            f.write(json.dumps(feedback, default=str) + '\n')
            
    def _maybe_trigger_retraining(self):
        """Check if we should trigger model retraining"""
        feedback_file = self.models_dir / "training_feedback.jsonl"
        
        if feedback_file.exists():
            with open(feedback_file, 'r') as f:
                feedback_count = sum(1 for _ in f)
                
            if feedback_count >= self.training_preferences["min_feedback_threshold"]:
                self.logger.info(f"Triggering retraining with {feedback_count} feedback samples")
                asyncio.create_task(self.retrain_models())
                
    async def retrain_models(self):
        """Retrain models with new feedback data"""
        self.logger.info("Starting model retraining...")
        
        # Load feedback data
        feedback_data = self._load_feedback_data()
        
        # Retrain each model that supports it
        for model_name, model in self.models.items():
            if hasattr(model, 'retrain') or hasattr(model, 'incremental_train'):
                try:
                    await self._retrain_single_model(model_name, model, feedback_data)
                except Exception as e:
                    self.logger.error(f"Retraining failed for {model_name}: {e}")
                    
    def _load_feedback_data(self) -> List[Dict]:
        """Load feedback data for retraining"""
        feedback_file = self.models_dir / "training_feedback.jsonl"
        feedback_data = []
        
        if feedback_file.exists():
            with open(feedback_file, 'r') as f:
                for line in f:
                    feedback_data.append(json.loads(line))
                    
        return feedback_data
        
    async def _retrain_single_model(self, model_name: str, model, feedback_data: List[Dict]):
        """Retrain a single model with feedback data"""
        self.logger.info(f"Retraining {model_name}...")
        
        # Convert feedback to model-specific format
        training_data = self._convert_feedback_to_training_data(feedback_data, model_name)
        
        # Perform retraining
        if hasattr(model, 'incremental_train'):
            model.incremental_train(training_data)
        elif hasattr(model, 'retrain'):
            model.retrain(training_data)
            
        # Save updated model
        self._save_model(model_name, model)
        self.logger.info(f"Completed retraining {model_name}")
        
    def _convert_feedback_to_training_data(self, feedback_data: List[Dict], model_name: str):
        """Convert feedback to model-specific training format"""
        # This would need to be implemented for each model type
        # For now, return basic format
        return feedback_data
        
    def _save_model(self, model_name: str, model):
        """Save trained model to disk"""
        config = self.model_configs[model_name]
        weights_path = self.models_dir / config.weights_path
        
        # Ensure directory exists
        weights_path.parent.mkdir(parents=True, exist_ok=True)
        
        if model_name in ["hybrid_movie", "hybrid_tv"]:
            # Save hybrid model using its built-in method
            if hasattr(model, 'save_models'):
                model.save_models()
        else:
            # Save PyTorch model
            torch.save(model, weights_path)
            
    def get_model_status(self) -> Dict:
        """Get status of all models"""
        return {
            "loaded_models": self.status.loaded_models,
            "model_errors": self.status.model_errors,
            "last_updated": {k: v.isoformat() for k, v in self.status.last_updated.items()},
            "total_models": len(self.model_configs),
            "loaded_count": sum(self.status.loaded_models.values())
        }
        
    def update_training_preferences(self, preferences: Dict):
        """Update training preferences from admin interface"""
        self.training_preferences.update(preferences)
        
        # Save preferences
        prefs_file = self.models_dir / "training_preferences.json"
        with open(prefs_file, 'w') as f:
            json.dump(self.training_preferences, f, indent=2)
            
        self.logger.info("Updated training preferences")
        
    def drop_in_model(self, model_file_path: str, model_name: str, model_type: str):
        """
        Drop in a new model file for immediate use
        
        Args:
            model_file_path: Path to the model file
            model_name: Name for the model
            model_type: Type of model (bert4rec, graphsage, etc.)
        """
        try:
            # Copy model to models directory
            dest_path = self.models_dir / "weights" / f"{model_name}.pt"
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            
            if model_type in self.model_configs:
                # Copy file
                import shutil
                shutil.copy2(model_file_path, dest_path)
                
                # Update config
                new_config = self.model_configs[model_type].copy()
                new_config.weights_path = f"weights/{model_name}.pt"
                self.model_configs[model_name] = new_config
                
                # Load the model
                asyncio.create_task(self._load_single_model(model_name, new_config))
                
                self.logger.info(f"Successfully dropped in model {model_name}")
                return True
            else:
                raise ValueError(f"Unknown model type: {model_type}")
                
        except Exception as e:
            self.logger.error(f"Failed to drop in model {model_name}: {e}")
            return False


# Global model manager instance
model_manager = UnifiedModelManager()

async def initialize_models():
    """Initialize all models - call this at startup"""
    return await model_manager.load_all_models()

def get_recommendations(user_id: int, content_type: str = "movie", top_k: int = 10) -> List[Dict]:
    """Simple interface for getting recommendations"""
    return model_manager.get_recommendations(user_id, content_type, top_k)

def add_user_feedback(user_id: int, item_id: int, rating: float, content_type: str = "movie"):
    """Simple interface for adding user feedback"""
    model_manager.add_feedback(user_id, item_id, rating, content_type)