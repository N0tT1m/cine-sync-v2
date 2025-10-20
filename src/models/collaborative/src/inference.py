#!/usr/bin/env python3
"""
Inference script for Neural Collaborative Filtering models.
"""

import torch
import pandas as pd
import numpy as np
from pathlib import Path
import pickle
import argparse
import logging
from typing import List, Dict, Tuple, Optional

from model import NeuralCollaborativeFiltering, SimpleNCF, DeepNCF


class NCFInference:
    """
    Inference class for Neural Collaborative Filtering models.
    """
    
    def __init__(self, model_path: str, encoders_path: str, device: str = 'auto'):
        """
        Args:
            model_path: Path to trained model checkpoint
            encoders_path: Path to saved encoders
            device: Device to use for inference
        """
        self.device = device if device != 'auto' else ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load encoders
        with open(encoders_path, 'rb') as f:
            encoders = pickle.load(f)
        
        self.user_encoder = encoders['user_encoder']
        self.item_encoder = encoders['item_encoder']
        self.genre_encoder = encoders.get('genre_encoder')
        
        # Load model
        self._load_model(model_path)
        
        self.logger = logging.getLogger(__name__)
    
    def _load_model(self, model_path: str):
        """Load trained model"""
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Get model configuration
        model_config = checkpoint.get('model_config', {})
        model_class = model_config.get('model_class', 'NeuralCollaborativeFiltering')
        
        # Create model based on class
        if model_class == 'SimpleNCF':
            self.model = SimpleNCF(
                num_users=len(self.user_encoder.classes_),
                num_items=len(self.item_encoder.classes_),
                embedding_dim=model_config.get('embedding_dim', 64)
            )
        elif model_class == 'DeepNCF':
            self.model = DeepNCF(
                num_users=len(self.user_encoder.classes_),
                num_items=len(self.item_encoder.classes_),
                num_genres=len(self.genre_encoder.classes_) if self.genre_encoder else 20,
                embedding_dim=model_config.get('embedding_dim', 128)
            )
        else:  # Default to NeuralCollaborativeFiltering
            self.model = NeuralCollaborativeFiltering(
                num_users=len(self.user_encoder.classes_),
                num_items=len(self.item_encoder.classes_),
                embedding_dim=model_config.get('embedding_dim', 64)
            )
        
        # Load model weights
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        self.logger.info(f"Loaded {model_class} model on {self.device}")
    
    def predict_rating(self, user_id: int, item_id: int) -> float:
        """
        Predict rating for a single user-item pair.
        
        Args:
            user_id: Original user ID
            item_id: Original item ID
        
        Returns:
            Predicted rating
        """
        try:
            # Encode user and item IDs
            user_encoded = self.user_encoder.transform([user_id])[0]
            item_encoded = self.item_encoder.transform([item_id])[0]
            
            # Convert to tensors
            user_tensor = torch.LongTensor([user_encoded]).to(self.device)
            item_tensor = torch.LongTensor([item_encoded]).to(self.device)
            
            with torch.no_grad():
                prediction = self.model(user_tensor, item_tensor)
                return prediction.item()
                
        except ValueError as e:
            self.logger.warning(f"Unknown user or item: {e}")
            return 0.0
    
    def get_user_recommendations(self, user_id: int, top_k: int = 10,
                                exclude_seen: bool = True,
                                seen_items: Optional[List[int]] = None) -> List[Tuple[int, float]]:
        """
        Get top-k recommendations for a user.
        
        Args:
            user_id: Original user ID
            top_k: Number of recommendations
            exclude_seen: Whether to exclude seen items
            seen_items: List of items user has already seen
        
        Returns:
            List of (item_id, predicted_rating) tuples
        """
        try:
            user_encoded = self.user_encoder.transform([user_id])[0]
        except ValueError:
            self.logger.warning(f"Unknown user ID: {user_id}")
            return []
        
        # Get all items
        all_items = list(self.item_encoder.classes_)
        
        # Exclude seen items if requested
        if exclude_seen and seen_items:
            all_items = [item for item in all_items if item not in seen_items]
        
        # Predict ratings for all items
        item_scores = []
        batch_size = 1000  # Process in batches for efficiency
        
        with torch.no_grad():
            for i in range(0, len(all_items), batch_size):
                batch_items = all_items[i:i + batch_size]
                
                # Encode items
                items_encoded = self.item_encoder.transform(batch_items)
                
                # Create tensors
                user_batch = torch.LongTensor([user_encoded] * len(batch_items)).to(self.device)
                item_batch = torch.LongTensor(items_encoded).to(self.device)
                
                # Predict
                predictions = self.model(user_batch, item_batch)
                
                # Store results
                for item_id, score in zip(batch_items, predictions.cpu().numpy()):
                    item_scores.append((item_id, float(score)))
        
        # Sort by score and return top-k
        item_scores.sort(key=lambda x: x[1], reverse=True)
        return item_scores[:top_k]
    
    def get_item_embeddings(self, item_ids: List[int]) -> np.ndarray:
        """
        Get item embeddings for similarity analysis.
        
        Args:
            item_ids: List of original item IDs
        
        Returns:
            Item embeddings array
        """
        try:
            items_encoded = self.item_encoder.transform(item_ids)
            item_tensors = torch.LongTensor(items_encoded).to(self.device)
            
            with torch.no_grad():
                embeddings = self.model.item_embedding(item_tensors)
                return embeddings.cpu().numpy()
        except ValueError as e:
            self.logger.warning(f"Error getting embeddings: {e}")
            return np.array([])
    
    def find_similar_items(self, item_id: int, top_k: int = 10) -> List[Tuple[int, float]]:
        """
        Find items similar to the given item.
        
        Args:
            item_id: Original item ID
            top_k: Number of similar items to return
        
        Returns:
            List of (similar_item_id, similarity_score) tuples
        """
        # Get all item embeddings
        all_items = list(self.item_encoder.classes_)
        all_embeddings = self.get_item_embeddings(all_items)
        
        if len(all_embeddings) == 0:
            return []
        
        # Get target item embedding
        target_embedding = self.get_item_embeddings([item_id])
        if len(target_embedding) == 0:
            return []
        
        # Calculate similarities
        similarities = np.dot(all_embeddings, target_embedding.T).squeeze()
        
        # Get top-k similar items (excluding the item itself)
        similar_indices = np.argsort(similarities)[::-1]
        similar_items = []
        
        for idx in similar_indices:
            if all_items[idx] != item_id and len(similar_items) < top_k:
                similar_items.append((all_items[idx], float(similarities[idx])))
        
        return similar_items
    
    def batch_predict(self, user_item_pairs: List[Tuple[int, int]]) -> List[float]:
        """
        Predict ratings for multiple user-item pairs efficiently.
        
        Args:
            user_item_pairs: List of (user_id, item_id) tuples
        
        Returns:
            List of predicted ratings
        """
        predictions = []
        batch_size = 1000
        
        for i in range(0, len(user_item_pairs), batch_size):
            batch_pairs = user_item_pairs[i:i + batch_size]
            
            # Separate and encode users and items
            users, items = zip(*batch_pairs)
            
            try:
                users_encoded = self.user_encoder.transform(users)
                items_encoded = self.item_encoder.transform(items)
                
                # Convert to tensors
                user_tensors = torch.LongTensor(users_encoded).to(self.device)
                item_tensors = torch.LongTensor(items_encoded).to(self.device)
                
                with torch.no_grad():
                    batch_predictions = self.model(user_tensors, item_tensors)
                    predictions.extend(batch_predictions.cpu().numpy())
                    
            except ValueError as e:
                # Handle unknown users/items
                self.logger.warning(f"Error in batch prediction: {e}")
                predictions.extend([0.0] * len(batch_pairs))
        
        return predictions


def main():
    parser = argparse.ArgumentParser(description='NCF Model Inference')
    parser.add_argument('--model-path', type=str, required=True,
                       help='Path to trained model')
    parser.add_argument('--encoders-path', type=str, required=True,
                       help='Path to encoders file')
    parser.add_argument('--user-id', type=int, help='User ID for recommendations')
    parser.add_argument('--item-id', type=int, help='Item ID for rating prediction')
    parser.add_argument('--top-k', type=int, default=10,
                       help='Number of recommendations')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Create inference object
    ncf_inference = NCFInference(
        model_path=args.model_path,
        encoders_path=args.encoders_path,
        device=args.device
    )
    
    # Example usage
    if args.user_id and args.item_id:
        # Predict rating
        rating = ncf_inference.predict_rating(args.user_id, args.item_id)
        print(f"Predicted rating for user {args.user_id}, item {args.item_id}: {rating:.4f}")
    
    if args.user_id:
        # Get recommendations
        recommendations = ncf_inference.get_user_recommendations(
            args.user_id, top_k=args.top_k
        )
        print(f"\nTop {args.top_k} recommendations for user {args.user_id}:")
        for item_id, score in recommendations:
            print(f"  Item {item_id}: {score:.4f}")
    
    if args.item_id:
        # Find similar items
        similar_items = ncf_inference.find_similar_items(args.item_id, top_k=5)
        print(f"\nItems similar to item {args.item_id}:")
        for item_id, similarity in similar_items:
            print(f"  Item {item_id}: {similarity:.4f}")


if __name__ == "__main__":
    main()