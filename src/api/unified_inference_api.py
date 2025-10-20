#!/usr/bin/env python3
"""
Unified Inference API for all CineSync recommendation models.
Provides a single interface to access NCF, Sequential, and Two-Tower models.
"""

import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any, Union
import logging
import json
from dataclasses import dataclass
from enum import Enum

# Import all model inference classes
from neural_collaborative_filtering.src.inference import NCFInference
from sequential_models.src.inference import SequentialInference
from two_tower_model.src.inference import TwoTowerInference


class ModelType(Enum):
    """Available model types"""
    NCF = "neural_collaborative_filtering"
    SEQUENTIAL = "sequential"
    TWO_TOWER = "two_tower"
    ENSEMBLE = "ensemble"


@dataclass
class RecommendationResult:
    """Standardized recommendation result"""
    item_id: int
    score: float
    model_type: str
    confidence: Optional[float] = None
    metadata: Optional[Dict] = None


@dataclass
class ModelConfig:
    """Configuration for a single model"""
    model_type: ModelType
    model_path: str
    encoders_path: str
    weight: float = 1.0
    enabled: bool = True


class UnifiedRecommendationAPI:
    """
    Unified API for all CineSync recommendation models.
    
    Features:
    - Single interface for multiple models
    - Ensemble recommendations
    - Model comparison
    - Performance monitoring
    - Fallback strategies
    """
    
    def __init__(self, config_path: Optional[str] = None, device: str = 'auto'):
        """
        Initialize the unified API.
        
        Args:
            config_path: Path to model configuration file
            device: Device to use for inference
        """
        self.device = device if device != 'auto' else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger = logging.getLogger(__name__)
        
        # Model instances
        self.models: Dict[ModelType, Any] = {}
        self.model_configs: Dict[ModelType, ModelConfig] = {}
        
        # Performance tracking
        self.model_performance: Dict[ModelType, Dict[str, float]] = {}
        
        # Load configuration
        if config_path:
            self.load_config(config_path)
        else:
            self._setup_default_config()
    
    def _setup_default_config(self):
        """Setup default model configuration"""
        default_configs = [
            ModelConfig(
                model_type=ModelType.NCF,
                model_path="neural_collaborative_filtering/models/best_model.pt",
                encoders_path="neural_collaborative_filtering/models/encoders.pkl",
                weight=1.0
            ),
            ModelConfig(
                model_type=ModelType.SEQUENTIAL,
                model_path="sequential_models/models/best_model.pt", 
                encoders_path="sequential_models/models/encoders.pkl",
                weight=1.0
            ),
            ModelConfig(
                model_type=ModelType.TWO_TOWER,
                model_path="two_tower_model/models/best_model.pt",
                encoders_path="two_tower_model/models/preprocessors.pkl",
                weight=1.0
            )
        ]
        
        for config in default_configs:
            self.model_configs[config.model_type] = config
    
    def load_config(self, config_path: str):
        """Load model configuration from JSON file"""
        with open(config_path, 'r') as f:
            config_data = json.load(f)
        
        for model_name, model_info in config_data.items():
            try:
                model_type = ModelType(model_name)
                self.model_configs[model_type] = ModelConfig(
                    model_type=model_type,
                    model_path=model_info['model_path'],
                    encoders_path=model_info['encoders_path'],
                    weight=model_info.get('weight', 1.0),
                    enabled=model_info.get('enabled', True)
                )
            except ValueError:
                self.logger.warning(f"Unknown model type: {model_name}")
    
    def load_model(self, model_type: ModelType) -> bool:
        """
        Load a specific model.
        
        Args:
            model_type: Type of model to load
            
        Returns:
            True if successful, False otherwise
        """
        if model_type not in self.model_configs:
            self.logger.error(f"No configuration found for {model_type}")
            return False
        
        config = self.model_configs[model_type]
        
        if not config.enabled:
            self.logger.info(f"Model {model_type} is disabled")
            return False
        
        try:
            if model_type == ModelType.NCF:
                self.models[model_type] = NCFInference(
                    model_path=config.model_path,
                    encoders_path=config.encoders_path,
                    device=self.device
                )
            elif model_type == ModelType.SEQUENTIAL:
                self.models[model_type] = SequentialInference(
                    model_path=config.model_path,
                    encoders_path=config.encoders_path,
                    device=self.device
                )
            elif model_type == ModelType.TWO_TOWER:
                self.models[model_type] = TwoTowerInference(
                    model_path=config.model_path,
                    preprocessors_path=config.encoders_path,
                    device=self.device
                )
            
            self.logger.info(f"Successfully loaded {model_type} model")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load {model_type} model: {e}")
            return False
    
    def load_all_models(self):
        """Load all configured models"""
        for model_type in self.model_configs:
            self.load_model(model_type)
        
        loaded_models = list(self.models.keys())
        self.logger.info(f"Loaded {len(loaded_models)} models: {loaded_models}")
    
    def get_recommendations(self, user_id: int, top_k: int = 10, 
                          model_type: Optional[ModelType] = None,
                          exclude_seen: bool = True,
                          seen_items: Optional[List[int]] = None) -> List[RecommendationResult]:
        """
        Get recommendations from specified model or ensemble.
        
        Args:
            user_id: User ID for recommendations
            top_k: Number of recommendations
            model_type: Specific model to use (None for ensemble)
            exclude_seen: Whether to exclude seen items
            seen_items: List of items user has seen
            
        Returns:
            List of recommendation results
        """
        if model_type and model_type != ModelType.ENSEMBLE:
            return self._get_single_model_recommendations(
                user_id, top_k, model_type, exclude_seen, seen_items
            )
        else:
            return self._get_ensemble_recommendations(
                user_id, top_k, exclude_seen, seen_items
            )
    
    def _get_single_model_recommendations(self, user_id: int, top_k: int,
                                        model_type: ModelType, exclude_seen: bool,
                                        seen_items: Optional[List[int]]) -> List[RecommendationResult]:
        """Get recommendations from a single model"""
        if model_type not in self.models:
            self.logger.warning(f"Model {model_type} not loaded")
            return []
        
        model = self.models[model_type]
        
        try:
            if model_type == ModelType.NCF:
                recommendations = model.get_user_recommendations(
                    user_id=user_id,
                    top_k=top_k,
                    exclude_seen=exclude_seen,
                    seen_items=seen_items
                )
            elif model_type == ModelType.SEQUENTIAL:
                recommendations = model.get_user_recommendations(
                    user_id=user_id,
                    top_k=top_k
                )
            elif model_type == ModelType.TWO_TOWER:
                recommendations = model.get_user_recommendations(
                    user_id=user_id,
                    top_k=top_k,
                    exclude_seen=exclude_seen,
                    seen_items=seen_items
                )
            
            # Convert to RecommendationResult objects
            results = []
            for item_id, score in recommendations:
                results.append(RecommendationResult(
                    item_id=item_id,
                    score=score,
                    model_type=model_type.value,
                    confidence=self._calculate_confidence(score, model_type)
                ))
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error getting recommendations from {model_type}: {e}")
            return []
    
    def _get_ensemble_recommendations(self, user_id: int, top_k: int,
                                    exclude_seen: bool, 
                                    seen_items: Optional[List[int]]) -> List[RecommendationResult]:
        """Get ensemble recommendations from all available models"""
        all_recommendations: Dict[int, Dict[str, float]] = {}
        model_weights = {}
        
        # Collect recommendations from all models
        for model_type in self.models:
            config = self.model_configs[model_type]
            if not config.enabled:
                continue
            
            model_recs = self._get_single_model_recommendations(
                user_id, top_k * 2, model_type, exclude_seen, seen_items
            )
            
            model_weights[model_type] = config.weight
            
            for rec in model_recs:
                if rec.item_id not in all_recommendations:
                    all_recommendations[rec.item_id] = {}
                
                all_recommendations[rec.item_id][model_type.value] = rec.score
        
        # Combine scores using weighted average
        ensemble_scores = []
        for item_id, model_scores in all_recommendations.items():
            weighted_score = 0.0
            total_weight = 0.0
            
            for model_type in self.models:
                if model_type.value in model_scores:
                    weight = model_weights.get(model_type, 1.0)
                    weighted_score += model_scores[model_type.value] * weight
                    total_weight += weight
            
            if total_weight > 0:
                final_score = weighted_score / total_weight
                ensemble_scores.append((item_id, final_score, model_scores))
        
        # Sort by ensemble score and take top-k
        ensemble_scores.sort(key=lambda x: x[1], reverse=True)
        
        results = []
        for item_id, score, model_scores in ensemble_scores[:top_k]:
            results.append(RecommendationResult(
                item_id=item_id,
                score=score,
                model_type="ensemble",
                confidence=self._calculate_ensemble_confidence(model_scores),
                metadata={"individual_scores": model_scores}
            ))
        
        return results
    
    def predict_next_items(self, sequence: List[int], top_k: int = 10,
                          model_type: ModelType = ModelType.SEQUENTIAL) -> List[RecommendationResult]:
        """
        Predict next items based on sequence (primarily for sequential models).
        
        Args:
            sequence: List of item IDs in chronological order
            top_k: Number of predictions
            model_type: Model to use for prediction
            
        Returns:
            List of next item predictions
        """
        if model_type not in self.models:
            self.logger.warning(f"Model {model_type} not loaded")
            return []
        
        if model_type != ModelType.SEQUENTIAL:
            self.logger.warning(f"Next item prediction is primarily for sequential models")
        
        try:
            model = self.models[model_type]
            predictions = model.predict_next_items(sequence, top_k=top_k)
            
            results = []
            for item_id, score in predictions:
                results.append(RecommendationResult(
                    item_id=item_id,
                    score=score,
                    model_type=model_type.value,
                    confidence=self._calculate_confidence(score, model_type)
                ))
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error predicting next items: {e}")
            return []
    
    def find_similar_items(self, item_id: int, top_k: int = 10,
                          model_type: Optional[ModelType] = None) -> List[RecommendationResult]:
        """
        Find items similar to the given item.
        
        Args:
            item_id: Item ID to find similarities for
            top_k: Number of similar items
            model_type: Specific model to use (None for best available)
            
        Returns:
            List of similar items
        """
        # Choose best model for similarity if not specified
        if model_type is None:
            if ModelType.TWO_TOWER in self.models:
                model_type = ModelType.TWO_TOWER  # Best for similarity
            elif ModelType.NCF in self.models:
                model_type = ModelType.NCF
            else:
                self.logger.error("No suitable model for similarity search")
                return []
        
        if model_type not in self.models:
            self.logger.warning(f"Model {model_type} not loaded")
            return []
        
        try:
            model = self.models[model_type]
            similar_items = model.find_similar_items(item_id, top_k=top_k)
            
            results = []
            for similar_id, similarity in similar_items:
                results.append(RecommendationResult(
                    item_id=similar_id,
                    score=similarity,
                    model_type=model_type.value,
                    confidence=similarity  # Similarity is inherently confidence
                ))
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error finding similar items: {e}")
            return []
    
    def predict_rating(self, user_id: int, item_id: int,
                      model_type: Optional[ModelType] = None) -> Dict[str, float]:
        """
        Predict rating for user-item pair.
        
        Args:
            user_id: User ID
            item_id: Item ID
            model_type: Specific model (None for all available)
            
        Returns:
            Dictionary of model predictions
        """
        predictions = {}
        
        models_to_use = [model_type] if model_type else list(self.models.keys())
        
        for mt in models_to_use:
            if mt in self.models:
                try:
                    model = self.models[mt]
                    
                    if mt == ModelType.NCF:
                        rating = model.predict_rating(user_id, item_id)
                        predictions[mt.value] = rating
                    elif mt == ModelType.TWO_TOWER:
                        score = model.predict_user_item_score(user_id, item_id)
                        predictions[mt.value] = score
                    # Sequential models don't directly predict ratings
                    
                except Exception as e:
                    self.logger.error(f"Error predicting rating with {mt}: {e}")
        
        return predictions
    
    def _calculate_confidence(self, score: float, model_type: ModelType) -> float:
        """Calculate confidence score based on model type and score"""
        if model_type == ModelType.NCF:
            # For NCF, higher scores are more confident
            return min(score / 5.0, 1.0)  # Normalize rating to [0,1]
        elif model_type == ModelType.SEQUENTIAL:
            # For sequential, probability is already confidence
            return score
        elif model_type == ModelType.TWO_TOWER:
            # For two-tower, similarity scores are confidence
            return score
        else:
            return score
    
    def _calculate_ensemble_confidence(self, model_scores: Dict[str, float]) -> float:
        """Calculate confidence for ensemble predictions"""
        if len(model_scores) == 1:
            return list(model_scores.values())[0]
        
        # Confidence increases with agreement between models
        scores = list(model_scores.values())
        mean_score = np.mean(scores)
        score_variance = np.var(scores)
        
        # High mean + low variance = high confidence
        confidence = mean_score * (1 - min(score_variance, 1.0))
        return max(0.0, min(confidence, 1.0))
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about loaded models"""
        info = {
            "loaded_models": list(self.models.keys()),
            "model_configs": {mt.value: {
                "enabled": config.enabled,
                "weight": config.weight,
                "model_path": config.model_path
            } for mt, config in self.model_configs.items()},
            "device": self.device,
            "performance": self.model_performance
        }
        return info
    
    def compare_models(self, user_id: int, top_k: int = 5) -> Dict[str, List[RecommendationResult]]:
        """
        Compare recommendations from all models for a user.
        
        Args:
            user_id: User ID
            top_k: Number of recommendations per model
            
        Returns:
            Dictionary mapping model names to their recommendations
        """
        comparison = {}
        
        for model_type in self.models:
            recommendations = self._get_single_model_recommendations(
                user_id, top_k, model_type, True, None
            )
            comparison[model_type.value] = recommendations
        
        # Add ensemble
        ensemble_recs = self._get_ensemble_recommendations(user_id, top_k, True, None)
        comparison["ensemble"] = ensemble_recs
        
        return comparison
    
    def health_check(self) -> Dict[str, bool]:
        """Check health status of all models"""
        health = {}
        
        for model_type, model in self.models.items():
            try:
                # Try a simple operation to verify model works
                if model_type == ModelType.NCF:
                    # Try to get embeddings for first few items
                    model.get_item_embeddings([1, 2, 3])
                elif model_type == ModelType.SEQUENTIAL:
                    # Try to predict next items for simple sequence
                    model.predict_next_items([1, 2, 3], top_k=1)
                elif model_type == ModelType.TWO_TOWER:
                    # Try to predict score for simple user-item pair
                    model.predict_user_item_score(1, 1)
                
                health[model_type.value] = True
                
            except Exception as e:
                self.logger.error(f"Health check failed for {model_type}: {e}")
                health[model_type.value] = False
        
        return health


def create_sample_config():
    """Create a sample configuration file"""
    config = {
        "neural_collaborative_filtering": {
            "model_path": "neural_collaborative_filtering/models/best_model.pt",
            "encoders_path": "neural_collaborative_filtering/models/encoders.pkl",
            "weight": 1.0,
            "enabled": True
        },
        "sequential": {
            "model_path": "sequential_models/models/best_model.pt",
            "encoders_path": "sequential_models/models/encoders.pkl", 
            "weight": 1.2,  # Slightly higher weight for time-aware models
            "enabled": True
        },
        "two_tower": {
            "model_path": "two_tower_model/models/best_model.pt",
            "encoders_path": "two_tower_model/models/preprocessors.pkl",
            "weight": 0.8,  # Lower weight due to different data requirements
            "enabled": True
        }
    }
    
    with open("unified_config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    print("Sample configuration saved to unified_config.json")


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description='Unified CineSync Recommendation API')
    parser.add_argument('--config', type=str, help='Configuration file path')
    parser.add_argument('--user-id', type=int, default=1, help='User ID for testing')
    parser.add_argument('--create-config', action='store_true', help='Create sample config')
    parser.add_argument('--device', type=str, default='auto', help='Device to use')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    if args.create_config:
        create_sample_config()
        exit(0)
    
    # Create unified API
    api = UnifiedRecommendationAPI(config_path=args.config, device=args.device)
    
    # Load models
    api.load_all_models()
    
    # Health check
    health = api.health_check()
    print(f"Model health: {health}")
    
    # Get model info
    info = api.get_model_info()
    print(f"Loaded models: {info['loaded_models']}")
    
    # Test recommendations
    if len(api.models) > 0:
        print(f"\nTesting recommendations for user {args.user_id}:")
        
        # Individual model recommendations
        for model_type in api.models:
            recs = api.get_recommendations(args.user_id, top_k=3, model_type=model_type)
            print(f"\n{model_type.value} recommendations:")
            for i, rec in enumerate(recs, 1):
                print(f"  {i}. Item {rec.item_id}: {rec.score:.3f} (conf: {rec.confidence:.3f})")
        
        # Ensemble recommendations
        ensemble_recs = api.get_recommendations(args.user_id, top_k=5)
        print(f"\nEnsemble recommendations:")
        for i, rec in enumerate(ensemble_recs, 1):
            print(f"  {i}. Item {rec.item_id}: {rec.score:.3f} (conf: {rec.confidence:.3f})")