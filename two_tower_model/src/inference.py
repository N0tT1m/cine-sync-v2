#!/usr/bin/env python3
"""
Inference script for Two-Tower/Dual-Encoder models.
"""

import torch
import pandas as pd
import numpy as np
from pathlib import Path
import pickle
import argparse
import logging
from typing import List, Dict, Tuple, Optional, Any
import faiss

from model import (
    TwoTowerModel, EnhancedTwoTowerModel, 
    MultiTaskTwoTowerModel, CollaborativeTwoTowerModel
)


class TwoTowerInference:
    """
    Inference class for Two-Tower models with efficient similarity search.
    """
    
    def __init__(self, model_path: str, preprocessors_path: str, device: str = 'auto'):
        """
        Args:
            model_path: Path to trained model checkpoint
            preprocessors_path: Path to saved preprocessors
            device: Device to use for inference
        """
        self.device = device if device != 'auto' else ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load preprocessors
        with open(preprocessors_path, 'rb') as f:
            self.preprocessors = pickle.load(f)
        
        self.user_encoder = self.preprocessors['user_encoder']
        self.item_encoder = self.preprocessors['item_encoder']
        self.genre_encoders = self.preprocessors['genre_encoders']
        self.scalers = self.preprocessors['scalers']
        self.user_features_df = self.preprocessors['user_features_df']
        self.item_features_df = self.preprocessors['item_features_df']
        
        # Feature column information
        self.user_numerical_cols = self.preprocessors['user_numerical_cols']
        self.user_categorical_cols = self.preprocessors['user_categorical_cols']
        self.item_numerical_cols = self.preprocessors['item_numerical_cols']
        self.item_categorical_cols = self.preprocessors['item_categorical_cols']
        
        # Load model
        self._load_model(model_path)
        
        # Initialize item index for efficient retrieval
        self.item_index = None
        self.item_embeddings_cache = None
        
        self.logger = logging.getLogger(__name__)
    
    def _load_model(self, model_path: str):
        """Load trained model"""
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Get model configuration
        model_config = checkpoint.get('model_config', {})
        model_class = model_config.get('model_class', 'TwoTowerModel')
        
        # Create model based on class
        if model_class == 'EnhancedTwoTowerModel':
            # Get categorical dimensions
            user_cat_dims = {}
            item_cat_dims = {}
            
            for col in self.user_categorical_cols:
                user_cat_dims[col] = len(self.genre_encoders[f'user_{col}'].classes_)
            for col in self.item_categorical_cols:
                item_cat_dims[col] = len(self.genre_encoders[f'item_{col}'].classes_)
            
            self.model = EnhancedTwoTowerModel(
                user_categorical_dims=user_cat_dims,
                user_numerical_dim=len(self.user_numerical_cols),
                item_categorical_dims=item_cat_dims,
                item_numerical_dim=len(self.item_numerical_cols),
                embedding_dim=model_config.get('embedding_dim', 128)
            )
            
        elif model_class == 'MultiTaskTwoTowerModel':
            user_feature_size = len(self.user_numerical_cols) + len(self.user_categorical_cols)
            item_feature_size = len(self.item_numerical_cols) + len(self.item_categorical_cols)
            
            self.model = MultiTaskTwoTowerModel(
                user_features_dim=user_feature_size,
                item_features_dim=item_feature_size,
                embedding_dim=model_config.get('embedding_dim', 128)
            )
            
        elif model_class == 'CollaborativeTwoTowerModel':
            user_feature_size = len(self.user_numerical_cols) + len(self.user_categorical_cols)
            item_feature_size = len(self.item_numerical_cols) + len(self.item_categorical_cols)
            
            self.model = CollaborativeTwoTowerModel(
                num_users=len(self.user_encoder.classes_),
                num_items=len(self.item_encoder.classes_),
                user_features_dim=user_feature_size,
                item_features_dim=item_feature_size,
                embedding_dim=model_config.get('embedding_dim', 128)
            )
            
        else:  # Default to TwoTowerModel
            user_feature_size = len(self.user_numerical_cols) + len(self.user_categorical_cols)
            item_feature_size = len(self.item_numerical_cols) + len(self.item_categorical_cols)
            
            self.model = TwoTowerModel(
                user_features_dim=user_feature_size,
                item_features_dim=item_feature_size,
                embedding_dim=model_config.get('embedding_dim', 128)
            )
        
        # Load model weights
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        self.logger.info(f"Loaded {model_class} model on {self.device}")
    
    def _process_user_features(self, user_id: int) -> Dict[str, torch.Tensor]:
        """Process user features for model input"""
        # Get user features
        try:
            user_row = self.user_features_df[self.user_features_df['userId'] == user_id].iloc[0]
        except (IndexError, KeyError):
            self.logger.warning(f"User {user_id} not found in features")
            return None
        
        # Process numerical features
        user_numerical = None
        if self.user_numerical_cols:
            numerical_values = user_row[self.user_numerical_cols].values.reshape(1, -1)
            if 'user_numerical' in self.scalers:
                numerical_values = self.scalers['user_numerical'].transform(numerical_values)
            user_numerical = torch.FloatTensor(numerical_values)
        
        # Process categorical features
        user_categorical = {}
        for col in self.user_categorical_cols:
            encoded_value = self.genre_encoders[f'user_{col}'].transform([str(user_row[col])])[0]
            user_categorical[col] = torch.LongTensor([encoded_value])
        
        return {
            'numerical': user_numerical,
            'categorical': user_categorical
        }
    
    def _process_item_features(self, item_id: int) -> Dict[str, torch.Tensor]:
        """Process item features for model input"""
        # Get item features
        try:
            item_row = self.item_features_df[self.item_features_df['movieId'] == item_id].iloc[0]
        except (IndexError, KeyError):
            self.logger.warning(f"Item {item_id} not found in features")
            return None
        
        # Process numerical features
        item_numerical = None
        if self.item_numerical_cols:
            numerical_values = item_row[self.item_numerical_cols].values.reshape(1, -1)
            if 'item_numerical' in self.scalers:
                numerical_values = self.scalers['item_numerical'].transform(numerical_values)
            item_numerical = torch.FloatTensor(numerical_values)
        
        # Process categorical features
        item_categorical = {}
        for col in self.item_categorical_cols:
            encoded_value = self.genre_encoders[f'item_{col}'].transform([str(item_row[col])])[0]
            item_categorical[col] = torch.LongTensor([encoded_value])
        
        return {
            'numerical': item_numerical,
            'categorical': item_categorical
        }
    
    def predict_user_item_score(self, user_id: int, item_id: int) -> float:
        """
        Predict score for a user-item pair.
        
        Args:
            user_id: Original user ID
            item_id: Original item ID
        
        Returns:
            Predicted score
        """
        user_features = self._process_user_features(user_id)
        item_features = self._process_item_features(item_id)
        
        if user_features is None or item_features is None:
            return 0.0
        
        with torch.no_grad():
            # Move to device
            if user_features['numerical'] is not None:
                user_features['numerical'] = user_features['numerical'].to(self.device)
            for key in user_features['categorical']:
                user_features['categorical'][key] = user_features['categorical'][key].to(self.device)
            
            if item_features['numerical'] is not None:
                item_features['numerical'] = item_features['numerical'].to(self.device)
            for key in item_features['categorical']:
                item_features['categorical'][key] = item_features['categorical'][key].to(self.device)
            
            # Predict based on model type
            if isinstance(self.model, EnhancedTwoTowerModel):
                prediction = self.model(
                    user_features['categorical'],
                    user_features['numerical'] if user_features['numerical'] is not None else torch.zeros(1, 0).to(self.device),
                    item_features['categorical'],
                    item_features['numerical'] if item_features['numerical'] is not None else torch.zeros(1, 0).to(self.device)
                )
            elif isinstance(self.model, CollaborativeTwoTowerModel):
                # Need user and item encodings
                user_encoded = self.user_encoder.transform([user_id])[0]
                item_encoded = self.item_encoder.transform([item_id])[0]
                
                user_id_tensor = torch.LongTensor([user_encoded]).to(self.device)
                item_id_tensor = torch.LongTensor([item_encoded]).to(self.device)
                
                # Flatten features for collaborative model
                user_flat = self._flatten_features(user_features)
                item_flat = self._flatten_features(item_features)
                
                prediction = self.model(user_id_tensor, item_id_tensor, user_flat, item_flat)
            else:
                # Flatten features for simple models
                user_flat = self._flatten_features(user_features)
                item_flat = self._flatten_features(item_features)
                
                prediction = self.model(user_flat, item_flat)
            
            if isinstance(prediction, dict):
                # Multi-task model - use rating prediction if available
                if 'rating' in prediction:
                    return prediction['rating'].item()
                else:
                    return prediction['similarity'].item()
            else:
                return prediction.item()
    
    def _flatten_features(self, features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Flatten categorical and numerical features"""
        feature_tensors = []
        
        if features['numerical'] is not None:
            feature_tensors.append(features['numerical'])
        
        for cat_tensor in features['categorical'].values():
            feature_tensors.append(cat_tensor.float())
        
        if feature_tensors:
            return torch.cat(feature_tensors, dim=1).to(self.device)
        else:
            return torch.zeros(1, 1).to(self.device)
    
    def build_item_index(self):
        """Build FAISS index for all items for efficient retrieval"""
        self.logger.info("Building item index...")
        
        all_items = list(self.item_features_df['movieId'].unique())
        item_embeddings = []
        
        batch_size = 100
        for i in range(0, len(all_items), batch_size):
            batch_items = all_items[i:i + batch_size]
            batch_embeddings = []
            
            for item_id in batch_items:
                item_features = self._process_item_features(item_id)
                if item_features is not None:
                    with torch.no_grad():
                        # Move to device
                        if item_features['numerical'] is not None:
                            item_features['numerical'] = item_features['numerical'].to(self.device)
                        for key in item_features['categorical']:
                            item_features['categorical'][key] = item_features['categorical'][key].to(self.device)
                        
                        # Get item embedding
                        if isinstance(self.model, EnhancedTwoTowerModel):
                            embedding = self.model.encode_items(
                                item_features['categorical'],
                                item_features['numerical'] if item_features['numerical'] is not None else torch.zeros(1, 0).to(self.device)
                            )
                        else:
                            item_flat = self._flatten_features(item_features)
                            embedding = self.model.encode_items(item_flat)
                        
                        batch_embeddings.append(embedding.cpu().numpy())
                else:
                    # Use zero embedding for missing items
                    batch_embeddings.append(np.zeros((1, self.model.embedding_dim)))
            
            if batch_embeddings:
                item_embeddings.extend(batch_embeddings)
        
        # Build FAISS index
        if item_embeddings:
            self.item_embeddings_cache = np.vstack(item_embeddings)
            embedding_dim = self.item_embeddings_cache.shape[1]
            
            # Use inner product index (cosine similarity for normalized vectors)
            self.item_index = faiss.IndexFlatIP(embedding_dim)
            self.item_index.add(self.item_embeddings_cache.astype(np.float32))
            
            self.indexed_items = all_items
            self.logger.info(f"Built index for {len(all_items)} items")
    
    def get_user_recommendations(self, user_id: int, top_k: int = 10,
                                exclude_seen: bool = True,
                                seen_items: Optional[List[int]] = None) -> List[Tuple[int, float]]:
        """
        Get top-k recommendations for a user using efficient similarity search.
        
        Args:
            user_id: Original user ID
            top_k: Number of recommendations
            exclude_seen: Whether to exclude seen items
            seen_items: List of items user has seen
        
        Returns:
            List of (item_id, score) tuples
        """
        # Build index if not exists
        if self.item_index is None:
            self.build_item_index()
        
        user_features = self._process_user_features(user_id)
        if user_features is None:
            return []
        
        with torch.no_grad():
            # Move to device
            if user_features['numerical'] is not None:
                user_features['numerical'] = user_features['numerical'].to(self.device)
            for key in user_features['categorical']:
                user_features['categorical'][key] = user_features['categorical'][key].to(self.device)
            
            # Get user embedding
            if isinstance(self.model, EnhancedTwoTowerModel):
                user_embedding = self.model.encode_users(
                    user_features['categorical'],
                    user_features['numerical'] if user_features['numerical'] is not None else torch.zeros(1, 0).to(self.device)
                )
            elif isinstance(self.model, CollaborativeTwoTowerModel):
                user_encoded = self.user_encoder.transform([user_id])[0]
                user_id_tensor = torch.LongTensor([user_encoded]).to(self.device)
                user_flat = self._flatten_features(user_features)
                user_embedding = self.model.encode_users(user_id_tensor, user_flat)
            else:
                user_flat = self._flatten_features(user_features)
                user_embedding = self.model.encode_users(user_flat)
            
            user_embedding_np = user_embedding.cpu().numpy().astype(np.float32)
        
        # Search for similar items
        search_k = min(top_k * 3, len(self.indexed_items))  # Search more items to account for filtering
        similarities, indices = self.item_index.search(user_embedding_np, search_k)
        
        # Convert to recommendations
        recommendations = []
        seen_set = set(seen_items) if seen_items else set()
        
        for similarity, idx in zip(similarities[0], indices[0]):
            if idx < len(self.indexed_items):
                item_id = self.indexed_items[idx]
                
                if exclude_seen and item_id in seen_set:
                    continue
                
                recommendations.append((item_id, float(similarity)))
                
                if len(recommendations) >= top_k:
                    break
        
        return recommendations
    
    def find_similar_items(self, item_id: int, top_k: int = 10) -> List[Tuple[int, float]]:
        """
        Find items similar to the given item.
        
        Args:
            item_id: Original item ID
            top_k: Number of similar items
        
        Returns:
            List of (similar_item_id, similarity) tuples
        """
        if self.item_index is None:
            self.build_item_index()
        
        try:
            item_idx = self.indexed_items.index(item_id)
            item_embedding = self.item_embeddings_cache[item_idx:item_idx+1].astype(np.float32)
            
            # Search for similar items
            similarities, indices = self.item_index.search(item_embedding, top_k + 1)  # +1 to exclude self
            
            similar_items = []
            for similarity, idx in zip(similarities[0], indices[0]):
                if idx < len(self.indexed_items):
                    similar_item_id = self.indexed_items[idx]
                    if similar_item_id != item_id:  # Exclude the item itself
                        similar_items.append((similar_item_id, float(similarity)))
                        
                    if len(similar_items) >= top_k:
                        break
            
            return similar_items
            
        except ValueError:
            self.logger.warning(f"Item {item_id} not found in index")
            return []
    
    def batch_predict(self, user_item_pairs: List[Tuple[int, int]]) -> List[float]:
        """
        Predict scores for multiple user-item pairs efficiently.
        
        Args:
            user_item_pairs: List of (user_id, item_id) tuples
        
        Returns:
            List of predicted scores
        """
        predictions = []
        
        for user_id, item_id in user_item_pairs:
            score = self.predict_user_item_score(user_id, item_id)
            predictions.append(score)
        
        return predictions


def main():
    parser = argparse.ArgumentParser(description='Two-Tower Model Inference')
    parser.add_argument('--model-path', type=str, required=True,
                       help='Path to trained model')
    parser.add_argument('--preprocessors-path', type=str, required=True,
                       help='Path to preprocessors file')
    parser.add_argument('--user-id', type=int, help='User ID for recommendations')
    parser.add_argument('--item-id', type=int, help='Item ID for similarity search')
    parser.add_argument('--top-k', type=int, default=10,
                       help='Number of recommendations/similar items')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use')
    parser.add_argument('--build-index', action='store_true',
                       help='Build item index for efficient search')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Create inference object
    tt_inference = TwoTowerInference(
        model_path=args.model_path,
        preprocessors_path=args.preprocessors_path,
        device=args.device
    )
    
    # Build index if requested
    if args.build_index:
        tt_inference.build_item_index()
    
    # Example usage
    if args.user_id and args.item_id:
        # Predict score
        score = tt_inference.predict_user_item_score(args.user_id, args.item_id)
        print(f"Predicted score for user {args.user_id}, item {args.item_id}: {score:.4f}")
    
    if args.user_id:
        # Get recommendations
        recommendations = tt_inference.get_user_recommendations(
            args.user_id, top_k=args.top_k
        )
        print(f"\nTop {args.top_k} recommendations for user {args.user_id}:")
        for item_id, score in recommendations:
            print(f"  Item {item_id}: {score:.4f}")
    
    if args.item_id:
        # Find similar items
        similar_items = tt_inference.find_similar_items(args.item_id, top_k=args.top_k)
        print(f"\nItems similar to item {args.item_id}:")
        for item_id, similarity in similar_items:
            print(f"  Item {item_id}: {similarity:.4f}")


if __name__ == "__main__":
    main()