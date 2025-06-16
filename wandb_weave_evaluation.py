#!/usr/bin/env python3
"""
W&B Weave Evaluation Framework for CineSync v2
Test and evaluate locally stored recommendation models without uploading artifacts
"""

import os
import json
import asyncio
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import logging

# W&B Weave imports
import weave
from weave import Model, Evaluation

logger = logging.getLogger(__name__)


@dataclass
class RecommendationExample:
    """Single recommendation test example"""
    user_id: int
    item_id: int
    expected_rating: float
    context: Dict[str, Any] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'user_id': self.user_id,
            'item_id': self.item_id,
            'expected_rating': self.expected_rating,
            'context': self.context or {}
        }


class LocalModelWrapper(Model):
    """Base wrapper for locally stored recommendation models"""
    
    def __init__(self, model_path: str, model_type: str, device: str = 'cpu'):
        super().__init__()
        self.model_path = model_path
        self.model_type = model_type
        self.device = device
        self.model = None
        self.metadata = None
        
        # Load model and metadata
        self._load_model()
    
    def _load_model(self):
        """Load the PyTorch model from local path"""
        try:
            if not Path(self.model_path).exists():
                raise FileNotFoundError(f"Model not found: {self.model_path}")
            
            # Load PyTorch model
            self.model = torch.load(self.model_path, map_location=self.device)
            self.model.eval()
            
            # Load metadata if available
            metadata_path = str(self.model_path).replace('.pt', '_metadata.json')
            if Path(metadata_path).exists():
                with open(metadata_path, 'r') as f:
                    self.metadata = json.load(f)
            
            logger.info(f"Loaded {self.model_type} model from {self.model_path}")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    @weave.op()
    def predict(self, user_id: int, item_id: int, **kwargs) -> Dict[str, Any]:
        """Base prediction method - to be overridden by specific model types"""
        raise NotImplementedError("Subclasses must implement predict method")


class NCFModelWrapper(LocalModelWrapper):
    """W&B Weave wrapper for Neural Collaborative Filtering models"""
    
    @weave.op()
    def predict(self, user_id: int, item_id: int, **kwargs) -> Dict[str, Any]:
        """Predict rating for user-item pair using NCF model"""
        try:
            with torch.no_grad():
                user_tensor = torch.tensor([user_id], dtype=torch.long, device=self.device)
                item_tensor = torch.tensor([item_id], dtype=torch.long, device=self.device)
                
                prediction = self.model(user_tensor, item_tensor)
                rating = prediction.item()
                
                return {
                    'predicted_rating': rating,
                    'user_id': user_id,
                    'item_id': item_id,
                    'model_type': 'ncf',
                    'confidence': abs(rating - 2.5) / 2.5  # Normalized confidence based on distance from neutral
                }
                
        except Exception as e:
            logger.error(f"NCF prediction failed: {e}")
            return {
                'predicted_rating': 2.5,  # Neutral fallback
                'user_id': user_id,
                'item_id': item_id,
                'model_type': 'ncf',
                'error': str(e)
            }


class TwoTowerModelWrapper(LocalModelWrapper):
    """W&B Weave wrapper for Two-Tower models"""
    
    @weave.op()
    def predict(self, user_id: int, item_id: int, **kwargs) -> Dict[str, Any]:
        """Predict rating for user-item pair using Two-Tower model"""
        try:
            with torch.no_grad():
                user_tensor = torch.tensor([user_id], dtype=torch.long, device=self.device)
                item_tensor = torch.tensor([item_id], dtype=torch.long, device=self.device)
                
                prediction = self.model(user_tensor, item_tensor)
                rating = prediction.item()
                
                # Get user and item embeddings for additional insights
                user_embedding = self.model.user_tower(user_tensor)
                item_embedding = self.model.item_tower(item_tensor)
                similarity = torch.cosine_similarity(user_embedding, item_embedding, dim=1).item()
                
                return {
                    'predicted_rating': rating,
                    'user_id': user_id,
                    'item_id': item_id,
                    'model_type': 'two_tower',
                    'embedding_similarity': similarity,
                    'confidence': (similarity + 1) / 2  # Normalized to [0,1]
                }
                
        except Exception as e:
            logger.error(f"Two-Tower prediction failed: {e}")
            return {
                'predicted_rating': 2.5,
                'user_id': user_id,
                'item_id': item_id,
                'model_type': 'two_tower',
                'error': str(e)
            }


class SequentialModelWrapper(LocalModelWrapper):
    """W&B Weave wrapper for Sequential models"""
    
    @weave.op()
    def predict(self, user_sequence: List[int], target_item: int, **kwargs) -> Dict[str, Any]:
        """Predict next item probability using Sequential model"""
        try:
            with torch.no_grad():
                # Pad or truncate sequence to model's expected length
                max_len = getattr(self.model, 'max_seq_len', 50)
                if len(user_sequence) > max_len:
                    sequence = user_sequence[-max_len:]
                else:
                    sequence = [0] * (max_len - len(user_sequence)) + user_sequence
                
                seq_tensor = torch.tensor([sequence], dtype=torch.long, device=self.device)
                logits = self.model(seq_tensor)
                
                # Get probability for target item
                probs = torch.softmax(logits, dim=-1)
                target_prob = probs[0, target_item].item() if target_item < probs.shape[-1] else 0.0
                
                # Get top-k predictions
                top_k_probs, top_k_items = torch.topk(probs[0], k=10)
                top_predictions = [
                    {'item_id': item.item(), 'probability': prob.item()}
                    for item, prob in zip(top_k_items, top_k_probs)
                ]
                
                return {
                    'target_probability': target_prob,
                    'target_item': target_item,
                    'sequence_length': len(user_sequence),
                    'top_predictions': top_predictions,
                    'model_type': 'sequential',
                    'confidence': target_prob
                }
                
        except Exception as e:
            logger.error(f"Sequential prediction failed: {e}")
            return {
                'target_probability': 0.1,
                'target_item': target_item,
                'model_type': 'sequential',
                'error': str(e)
            }


class HybridModelWrapper(LocalModelWrapper):
    """W&B Weave wrapper for Hybrid models"""
    
    @weave.op()
    def predict(self, user_id: int, item_id: int, **kwargs) -> Dict[str, Any]:
        """Predict rating using Hybrid model (collaborative + content-based)"""
        try:
            with torch.no_grad():
                user_tensor = torch.tensor([user_id], dtype=torch.long, device=self.device)
                item_tensor = torch.tensor([item_id], dtype=torch.long, device=self.device)
                
                prediction = self.model(user_tensor, item_tensor)
                rating = prediction.item()
                
                return {
                    'predicted_rating': rating,
                    'user_id': user_id,
                    'item_id': item_id,
                    'model_type': 'hybrid',
                    'confidence': min(abs(rating - 2.5) / 2.5, 1.0)
                }
                
        except Exception as e:
            logger.error(f"Hybrid prediction failed: {e}")
            return {
                'predicted_rating': 2.5,
                'user_id': user_id,
                'item_id': item_id,
                'model_type': 'hybrid',
                'error': str(e)
            }


# Scoring Functions
@weave.op()
def rmse_score(expected_rating: float, output: Dict[str, Any]) -> Dict[str, float]:
    """Calculate RMSE between expected and predicted ratings"""
    predicted = output.get('predicted_rating', 2.5)
    error = (expected_rating - predicted) ** 2
    return {
        'rmse': error,
        'absolute_error': abs(expected_rating - predicted),
        'relative_error': abs(expected_rating - predicted) / expected_rating if expected_rating > 0 else 0
    }


@weave.op()
def accuracy_score(expected_rating: float, output: Dict[str, Any], threshold: float = 0.5) -> Dict[str, float]:
    """Calculate accuracy within threshold"""
    predicted = output.get('predicted_rating', 2.5)
    accurate = abs(expected_rating - predicted) <= threshold
    return {
        'accurate_within_threshold': float(accurate),
        'prediction_error': abs(expected_rating - predicted)
    }


@weave.op()
def ranking_score(target_item: int, output: Dict[str, Any], k: int = 10) -> Dict[str, float]:
    """Calculate ranking metrics for sequential models"""
    top_predictions = output.get('top_predictions', [])
    target_prob = output.get('target_probability', 0.0)
    
    # Check if target is in top-k
    top_k_items = [pred['item_id'] for pred in top_predictions[:k]]
    hit_at_k = float(target_item in top_k_items)
    
    # Calculate rank if in top predictions
    rank = None
    for i, pred in enumerate(top_predictions):
        if pred['item_id'] == target_item:
            rank = i + 1
            break
    
    return {
        f'hit_at_{k}': hit_at_k,
        'target_probability': target_prob,
        'rank': rank if rank else 999,
        'reciprocal_rank': 1.0 / rank if rank else 0.0
    }


class WeaveEvaluationManager:
    """Manager for W&B Weave evaluations of recommendation models"""
    
    def __init__(self, project_name: str = "cinesync-v2-evaluation"):
        self.project_name = project_name
        weave.init(project_name)
    
    def create_rating_evaluation_dataset(self, test_data: pd.DataFrame, sample_size: int = 1000) -> List[Dict]:
        """Create evaluation dataset from test data for rating prediction"""
        if len(test_data) > sample_size:
            test_sample = test_data.sample(n=sample_size, random_state=42)
        else:
            test_sample = test_data
        
        examples = []
        for _, row in test_sample.iterrows():
            examples.append({
                'user_id': int(row['user_id']),
                'item_id': int(row['item_id']),
                'expected_rating': float(row['rating'])
            })
        
        return examples
    
    def create_sequential_evaluation_dataset(self, sequence_data: List[Tuple], sample_size: int = 500) -> List[Dict]:
        """Create evaluation dataset for sequential models"""
        if len(sequence_data) > sample_size:
            sequences = np.random.choice(len(sequence_data), size=sample_size, replace=False)
            test_sequences = [sequence_data[i] for i in sequences]
        else:
            test_sequences = sequence_data
        
        examples = []
        for user_sequence, target_item in test_sequences:
            examples.append({
                'user_sequence': user_sequence,
                'target_item': target_item
            })
        
        return examples
    
    async def evaluate_rating_model(self, model_wrapper: LocalModelWrapper, 
                                  test_examples: List[Dict], 
                                  model_name: str) -> Dict[str, Any]:
        """Evaluate a rating prediction model"""
        
        # Create evaluation
        evaluation = Evaluation(
            dataset=test_examples,
            scorers=[rmse_score, accuracy_score],
            name=f"{model_name}_rating_evaluation"
        )
        
        # Run evaluation
        results = await evaluation.evaluate(model_wrapper)
        
        logger.info(f"Completed evaluation for {model_name}")
        return results
    
    async def evaluate_sequential_model(self, model_wrapper: SequentialModelWrapper,
                                      test_examples: List[Dict],
                                      model_name: str) -> Dict[str, Any]:
        """Evaluate a sequential recommendation model"""
        
        # Modify predict function for sequential evaluation
        @weave.op()
        def sequential_predict(example: Dict) -> Dict[str, Any]:
            return model_wrapper.predict(
                user_sequence=example['user_sequence'],
                target_item=example['target_item']
            )
        
        evaluation = Evaluation(
            dataset=test_examples,
            scorers=[ranking_score],
            name=f"{model_name}_sequential_evaluation"
        )
        
        results = await evaluation.evaluate(sequential_predict)
        
        logger.info(f"Completed sequential evaluation for {model_name}")
        return results
    
    def find_latest_model(self, model_type: str) -> Optional[str]:
        """Find the latest model of given type in models/ directory"""
        models_dir = Path("models") / model_type
        if not models_dir.exists():
            return None
        
        model_files = list(models_dir.glob("*.pt"))
        if not model_files:
            return None
        
        # Sort by modification time, return latest
        latest_model = max(model_files, key=lambda x: x.stat().st_mtime)
        return str(latest_model)
    
    async def run_comprehensive_evaluation(self, test_data: pd.DataFrame) -> Dict[str, Any]:
        """Run evaluation on all available local models"""
        results = {}
        
        # Model types and their wrapper classes
        model_configs = {
            'ncf': NCFModelWrapper,
            'two_tower': TwoTowerModelWrapper,
            'hybrid_movie': HybridModelWrapper,
            'hybrid_tv': HybridModelWrapper
        }
        
        # Create rating evaluation dataset
        rating_examples = self.create_rating_evaluation_dataset(test_data)
        
        for model_type, wrapper_class in model_configs.items():
            model_path = self.find_latest_model(model_type)
            if model_path:
                try:
                    # Create model wrapper
                    model_wrapper = wrapper_class(model_path, model_type)
                    
                    # Run evaluation
                    eval_results = await self.evaluate_rating_model(
                        model_wrapper, rating_examples, model_type
                    )
                    
                    results[model_type] = eval_results
                    
                except Exception as e:
                    logger.error(f"Failed to evaluate {model_type}: {e}")
                    results[model_type] = {'error': str(e)}
            else:
                logger.warning(f"No model found for {model_type}")
        
        return results


# Convenience function for quick evaluation
async def evaluate_all_models(test_csv_path: str, project_name: str = "cinesync-v2-evaluation"):
    """Quick function to evaluate all available models"""
    
    # Load test data
    test_data = pd.read_csv(test_csv_path)
    
    # Initialize evaluation manager
    evaluator = WeaveEvaluationManager(project_name)
    
    # Run comprehensive evaluation
    results = await evaluator.run_comprehensive_evaluation(test_data)
    
    return results


if __name__ == "__main__":
    # Example usage
    asyncio.run(evaluate_all_models("test_data.csv"))