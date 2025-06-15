#!/usr/bin/env python3
"""
Production-Ready Inference Engine for Two-Tower/Dual-Encoder Models

Provides efficient inference capabilities for all two-tower architectures:
- Real-time user-item score prediction
- Scalable user recommendation generation using FAISS
- Item-to-item similarity search
- Batch prediction for bulk processing

Optimized for production deployment with pre-computed item embeddings
and efficient similarity search using Facebook AI Similarity Search (FAISS).
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
    Production-ready inference engine for trained Two-Tower recommendation models.
    
    Provides comprehensive inference capabilities:
    - Model loading and reconstruction from checkpoints
    - Feature preprocessing using saved encoders and scalers
    - Efficient similarity search with FAISS indexing
    - Support for all model variants (basic, enhanced, multi-task, collaborative)
    - Scalable recommendation generation for production workloads
    
    Design principles:
    - Pre-compute item embeddings for fast retrieval
    - Use FAISS for approximate nearest neighbor search
    - Maintain consistency with training-time preprocessing
    - Support batch operations for efficiency
    """
    
    def __init__(self, model_path: str, preprocessors_path: str, device: str = 'auto'):
        """
        Args:
            model_path: Path to trained model checkpoint
            preprocessors_path: Path to saved preprocessors
            device: Device to use for inference
        """
        self.device = device if device != 'auto' else ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load all preprocessing components saved from training
        # These maintain the same transformations used during training
        with open(preprocessors_path, 'rb') as f:
            self.preprocessors = pickle.load(f)
        
        self.user_encoder = self.preprocessors['user_encoder']        # User ID encoding
        self.item_encoder = self.preprocessors['item_encoder']        # Item ID encoding
        self.genre_encoders = self.preprocessors['genre_encoders']    # Categorical feature encoders
        self.scalers = self.preprocessors['scalers']                  # Numerical feature scalers
        self.user_features_df = self.preprocessors['user_features_df'] # Pre-computed user features
        self.item_features_df = self.preprocessors['item_features_df'] # Pre-computed item features
        
        # Feature column information
        self.user_numerical_cols = self.preprocessors['user_numerical_cols']
        self.user_categorical_cols = self.preprocessors['user_categorical_cols']
        self.item_numerical_cols = self.preprocessors['item_numerical_cols']
        self.item_categorical_cols = self.preprocessors['item_categorical_cols']
        
        # Load model
        self._load_model(model_path)
        
        # Initialize FAISS index for efficient similarity search
        self.item_index = None                # FAISS index for item embeddings
        self.item_embeddings_cache = None     # Cache of all item embeddings
        self.indexed_items = None             # Mapping from index position to item ID
        
        self.logger = logging.getLogger(__name__)
    
    def _load_model(self, model_path: str):
        """Load trained model and reconstruct architecture from checkpoint
        
        Automatically detects model type and recreates the exact architecture
        used during training, then loads the trained weights.
        """
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Extract model configuration from training checkpoint
        model_config = checkpoint.get('model_config', {})
        model_class = model_config.get('model_class', 'TwoTowerModel')
        
        # Reconstruct model architecture based on saved configuration
        if model_class == 'EnhancedTwoTowerModel':
            # Enhanced model needs categorical vocabulary sizes for embedding layers
            user_cat_dims = {}
            item_cat_dims = {}
            
            # Extract vocabulary sizes from saved encoders
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
        """Process user features for model input using training-time preprocessing
        
        Applies the same feature engineering and encoding used during training
        to ensure consistency between training and inference.
        """
        # Retrieve pre-computed user features
        try:
            user_row = self.user_features_df[self.user_features_df['userId'] == user_id].iloc[0]
        except (IndexError, KeyError):
            self.logger.warning(f"User {user_id} not found in feature database")
            return None
        
        # Process numerical features with training-time scaling
        user_numerical = None
        if self.user_numerical_cols:
            numerical_values = user_row[self.user_numerical_cols].values.reshape(1, -1)
            if 'user_numerical' in self.scalers:
                # Apply same standardization used during training
                numerical_values = self.scalers['user_numerical'].transform(numerical_values)
            user_numerical = torch.FloatTensor(numerical_values)
        
        # Process categorical features with training-time encoding
        user_categorical = {}
        for col in self.user_categorical_cols:
            # Apply same label encoding used during training
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
        Predict relevance score for a specific user-item pair.
        
        This method is useful for:
        - A/B testing specific recommendations
        - Explaining recommendation scores
        - Filtering candidates by minimum score threshold
        
        Args:
            user_id: Original user ID from the dataset
            item_id: Original item ID from the dataset
        
        Returns:
            Predicted relevance score (interpretation depends on model type)
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
            
            # Model-specific prediction with proper input formatting
            if isinstance(self.model, EnhancedTwoTowerModel):
                # Enhanced model expects separate categorical and numerical inputs
                prediction = self.model(
                    user_features['categorical'],
                    user_features['numerical'] if user_features['numerical'] is not None else torch.zeros(1, 0).to(self.device),
                    item_features['categorical'],
                    item_features['numerical'] if item_features['numerical'] is not None else torch.zeros(1, 0).to(self.device)
                )
            elif isinstance(self.model, CollaborativeTwoTowerModel):
                # Collaborative model needs both IDs for embeddings and features for content
                user_encoded = self.user_encoder.transform([user_id])[0]
                item_encoded = self.item_encoder.transform([item_id])[0]
                
                user_id_tensor = torch.LongTensor([user_encoded]).to(self.device)
                item_id_tensor = torch.LongTensor([item_encoded]).to(self.device)
                
                # Flatten content features for collaborative model
                user_flat = self._flatten_features(user_features)
                item_flat = self._flatten_features(item_features)
                
                prediction = self.model(user_id_tensor, item_id_tensor, user_flat, item_flat)
            else:
                # Basic model expects flattened feature vectors
                user_flat = self._flatten_features(user_features)
                item_flat = self._flatten_features(item_features)
                
                prediction = self.model(user_flat, item_flat)
            
            # Extract prediction from model output
            if isinstance(prediction, dict):
                # Multi-task model: prioritize rating prediction, fallback to similarity
                if 'rating' in prediction:
                    return prediction['rating'].item()
                else:
                    return prediction['similarity'].item()
            else:
                # Single-task model: direct prediction
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
        """Build FAISS index for all items to enable efficient recommendation generation
        
        Pre-computes embeddings for all items in the catalog and builds a FAISS index
        for fast similarity search. This is essential for production-scale recommendation
        systems where you need to quickly find top-k items for any user.
        """
        self.logger.info("Building item index...")
        
        # Get all items in the catalog
        all_items = list(self.item_features_df['movieId'].unique())
        item_embeddings = []
        
        # Process items in batches for memory efficiency
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
                        
                        # Extract item embedding using model's encoder
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
        
        # Build FAISS index for efficient similarity search
        if item_embeddings:
            self.item_embeddings_cache = np.vstack(item_embeddings)
            embedding_dim = self.item_embeddings_cache.shape[1]
            
            # IndexFlatIP: exact inner product search (cosine similarity for L2-normalized vectors)
            self.item_index = faiss.IndexFlatIP(embedding_dim)
            self.item_index.add(self.item_embeddings_cache.astype(np.float32))
            
            self.indexed_items = all_items  # Maps index positions to item IDs
            self.logger.info(f"Built FAISS index for {len(all_items)} items")
    
    def get_user_recommendations(self, user_id: int, top_k: int = 10,
                                exclude_seen: bool = True,
                                seen_items: Optional[List[int]] = None) -> List[Tuple[int, float]]:
        """
        Generate top-k recommendations for a user using efficient FAISS-based retrieval.
        
        This is the main production method for generating recommendations.
        Uses pre-computed item embeddings and FAISS for sub-linear search complexity.
        
        Args:
            user_id: Original user ID from the dataset
            top_k: Number of recommendations to return
            exclude_seen: Whether to filter out items the user has already interacted with
            seen_items: Optional list of items to exclude (if not provided, no filtering)
        
        Returns:
            List of (item_id, similarity_score) tuples sorted by relevance
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
            
            # Generate user embedding using appropriate model interface
            if isinstance(self.model, EnhancedTwoTowerModel):
                user_embedding = self.model.encode_users(
                    user_features['categorical'],
                    user_features['numerical'] if user_features['numerical'] is not None else torch.zeros(1, 0).to(self.device)
                )
            elif isinstance(self.model, CollaborativeTwoTowerModel):
                # Collaborative model needs both user ID and content features
                user_encoded = self.user_encoder.transform([user_id])[0]
                user_id_tensor = torch.LongTensor([user_encoded]).to(self.device)
                user_flat = self._flatten_features(user_features)
                user_embedding = self.model.encode_users(user_id_tensor, user_flat)
            else:
                # Basic model uses flattened features
                user_flat = self._flatten_features(user_features)
                user_embedding = self.model.encode_users(user_flat)
            
            user_embedding_np = user_embedding.cpu().numpy().astype(np.float32)
        
        # Efficient similarity search using FAISS
        search_k = min(top_k * 3, len(self.indexed_items))  # Over-retrieve to account for filtering
        similarities, indices = self.item_index.search(user_embedding_np, search_k)
        
        # Convert FAISS results to recommendations with optional filtering
        recommendations = []
        seen_set = set(seen_items) if seen_items else set()
        
        for similarity, idx in zip(similarities[0], indices[0]):
            if idx < len(self.indexed_items):
                item_id = self.indexed_items[idx]
                
                # Filter out seen items if requested
                if exclude_seen and item_id in seen_set:
                    continue
                
                recommendations.append((item_id, float(similarity)))
                
                # Stop when we have enough recommendations
                if len(recommendations) >= top_k:
                    break
        
        return recommendations
    
    def find_similar_items(self, item_id: int, top_k: int = 10) -> List[Tuple[int, float]]:
        """
        Find items similar to a given item using item-to-item similarity.
        
        Useful for:
        - "Customers who liked this also liked" recommendations
        - Building item similarity matrices
        - Content discovery and exploration
        
        Args:
            item_id: Original item ID to find similarities for
            top_k: Number of similar items to return
        
        Returns:
            List of (similar_item_id, similarity_score) tuples
        """
        if self.item_index is None:
            self.build_item_index()
        
        try:
            # Find the item in our index
            item_idx = self.indexed_items.index(item_id)
            item_embedding = self.item_embeddings_cache[item_idx:item_idx+1].astype(np.float32)
            
            # Search for similar items (+1 to account for the item itself)
            similarities, indices = self.item_index.search(item_embedding, top_k + 1)
            
            # Process results and exclude the query item itself
            similar_items = []
            for similarity, idx in zip(similarities[0], indices[0]):
                if idx < len(self.indexed_items):
                    similar_item_id = self.indexed_items[idx]
                    if similar_item_id != item_id:  # Exclude the query item itself
                        similar_items.append((similar_item_id, float(similarity)))
                        
                    if len(similar_items) >= top_k:
                        break
            
            return similar_items
            
        except ValueError:
            self.logger.warning(f"Item {item_id} not found in index")
            return []
    
    def batch_predict(self, user_item_pairs: List[Tuple[int, int]]) -> List[float]:
        """
        Predict scores for multiple user-item pairs in batch for efficiency.
        
        Useful for:
        - Offline evaluation and A/B testing
        - Bulk scoring for analytics
        - Pre-computing scores for candidate sets
        
        Args:
            user_item_pairs: List of (user_id, item_id) tuples to score
        
        Returns:
            List of predicted scores in the same order as input pairs
        """
        predictions = []
        
        for user_id, item_id in user_item_pairs:
            score = self.predict_user_item_score(user_id, item_id)
            predictions.append(score)
        
        return predictions


def main():
    """Command-line interface for two-tower model inference demonstration"""
    parser = argparse.ArgumentParser(description='Two-Tower Model Inference Engine')
    parser.add_argument('--model-path', type=str, required=True,
                       help='Path to trained model checkpoint (.pt file)')
    parser.add_argument('--preprocessors-path', type=str, required=True,
                       help='Path to saved preprocessors (.pkl file)')
    parser.add_argument('--user-id', type=int, 
                       help='User ID for personalized recommendations')
    parser.add_argument('--item-id', type=int, 
                       help='Item ID for similarity search')
    parser.add_argument('--top-k', type=int, default=10,
                       help='Number of recommendations/similar items to return')
    parser.add_argument('--device', type=str, default='auto',
                       help='Computing device (auto, cuda, cpu)')
    parser.add_argument('--build-index', action='store_true',
                       help='Build FAISS index for efficient similarity search')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Initialize inference engine with trained model and preprocessors
    tt_inference = TwoTowerInference(
        model_path=args.model_path,
        preprocessors_path=args.preprocessors_path,
        device=args.device
    )
    
    # Build FAISS index for efficient retrieval if requested
    if args.build_index:
        tt_inference.build_item_index()
    
    # Demonstration of inference capabilities
    if args.user_id and args.item_id:
        # Predict relevance score for specific user-item pair
        score = tt_inference.predict_user_item_score(args.user_id, args.item_id)
        print(f"Predicted relevance score for user {args.user_id}, item {args.item_id}: {score:.4f}")
    
    if args.user_id:
        # Generate personalized recommendations using FAISS-based retrieval
        recommendations = tt_inference.get_user_recommendations(
            args.user_id, top_k=args.top_k
        )
        print(f"\nTop {args.top_k} personalized recommendations for user {args.user_id}:")
        for item_id, score in recommendations:
            print(f"  Item {item_id}: {score:.4f}")
    
    if args.item_id:
        # Find items similar to the given item
        similar_items = tt_inference.find_similar_items(args.item_id, top_k=args.top_k)
        print(f"\nItems most similar to item {args.item_id}:")
        for item_id, similarity in similar_items:
            print(f"  Item {item_id}: {similarity:.4f}")


if __name__ == "__main__":
    main()