#!/usr/bin/env python3
"""
Inference script for Sequential Recommendation models.
"""

import torch
import pandas as pd
import numpy as np
from pathlib import Path
import pickle
import argparse
import logging
from typing import List, Dict, Tuple, Optional
from collections import defaultdict

from model import (
    SequentialRecommender, AttentionalSequentialRecommender,
    HierarchicalSequentialRecommender, SessionBasedRecommender
)


class SequentialInference:
    """
    Inference class for Sequential Recommendation models.
    """
    
    def __init__(self, model_path: str, encoders_path: str, device: str = 'auto'):
        """
        Args:
            model_path: Path to trained model checkpoint
            encoders_path: Path to saved encoders
            device: Device to use for inference
        """
        self.device = device if device != 'auto' else ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load encoders and user sequences
        with open(encoders_path, 'rb') as f:
            data = pickle.load(f)
        
        self.user_encoder = data['user_encoder']
        self.item_encoder = data['item_encoder']
        self.user_sequences = data.get('user_sequences', {})
        
        # Load model
        self._load_model(model_path)
        
        self.logger = logging.getLogger(__name__)
    
    def _load_model(self, model_path: str):
        """Load trained model"""
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Get model configuration
        model_config = checkpoint.get('model_config', {})
        model_class = model_config.get('model_class', 'SequentialRecommender')
        
        num_items = len(self.item_encoder.classes_) + 1  # +1 for padding
        
        # Create model based on class
        if model_class == 'AttentionalSequentialRecommender':
            self.model = AttentionalSequentialRecommender(
                num_items=num_items,
                embedding_dim=model_config.get('embedding_dim', 128),
                num_heads=8,
                num_blocks=2,
                max_seq_len=model_config.get('max_seq_length', 50)
            )
        elif model_class == 'HierarchicalSequentialRecommender':
            self.model = HierarchicalSequentialRecommender(
                num_items=num_items,
                embedding_dim=model_config.get('embedding_dim', 128)
            )
        elif model_class == 'SessionBasedRecommender':
            self.model = SessionBasedRecommender(
                num_items=num_items,
                embedding_dim=model_config.get('embedding_dim', 128),
                hidden_dim=model_config.get('hidden_dim', 256)
            )
        else:  # Default to SequentialRecommender
            self.model = SequentialRecommender(
                num_items=num_items,
                embedding_dim=model_config.get('embedding_dim', 128),
                hidden_dim=model_config.get('hidden_dim', 256)
            )
        
        # Load model weights
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        self.logger.info(f"Loaded {model_class} model on {self.device}")
    
    def predict_next_items(self, sequence: List[int], top_k: int = 10,
                          exclude_seen: bool = True) -> List[Tuple[int, float]]:
        """
        Predict next items given a sequence.
        
        Args:
            sequence: List of item IDs in chronological order
            top_k: Number of predictions to return
            exclude_seen: Whether to exclude items already in sequence
        
        Returns:
            List of (item_id, probability) tuples
        """
        try:
            # Encode sequence
            encoded_sequence = []
            for item_id in sequence:
                try:
                    encoded_item = self.item_encoder.transform([item_id])[0] + 1  # +1 for padding offset
                    encoded_sequence.append(encoded_item)
                except ValueError:
                    self.logger.warning(f"Unknown item ID: {item_id}")
            
            if not encoded_sequence:
                return []
            
            # Pad or truncate sequence
            max_len = 50  # Default max sequence length
            if len(encoded_sequence) > max_len:
                encoded_sequence = encoded_sequence[-max_len:]
            else:
                # Pad with zeros at the beginning
                encoded_sequence = [0] * (max_len - len(encoded_sequence)) + encoded_sequence
            
            # Convert to tensor
            sequence_tensor = torch.LongTensor([encoded_sequence]).to(self.device)
            length_tensor = torch.LongTensor([min(len(sequence), max_len)]).to(self.device)
            
            with torch.no_grad():
                if isinstance(self.model, AttentionalSequentialRecommender):
                    # Create causal mask
                    seq_len = sequence_tensor.size(1)
                    mask = torch.tril(torch.ones(seq_len, seq_len, device=self.device))
                    logits = self.model(sequence_tensor, mask)
                    predictions = logits[0, -1]  # Last position predictions
                    
                elif isinstance(self.model, SessionBasedRecommender):
                    predictions = self.model(sequence_tensor, length_tensor)
                    predictions = predictions[0]
                    
                elif isinstance(self.model, HierarchicalSequentialRecommender):
                    # For hierarchical model, we need both short and long sequences
                    # Use the same sequence for both (simplified)
                    predictions = self.model(sequence_tensor, sequence_tensor, length_tensor, length_tensor)
                    predictions = predictions[0]
                    
                else:  # SequentialRecommender
                    logits = self.model(sequence_tensor, length_tensor)
                    predictions = logits[0, -1]  # Last position predictions
                
                # Apply softmax to get probabilities
                probabilities = torch.softmax(predictions, dim=0)
                
                # Get top-k predictions
                top_k_values, top_k_indices = torch.topk(probabilities, min(top_k * 2, len(probabilities)))
                
                # Convert back to original item IDs and filter
                results = []
                seen_items = set(sequence) if exclude_seen else set()
                
                for value, idx in zip(top_k_values, top_k_indices):
                    idx_val = idx.item()
                    if idx_val == 0:  # Skip padding token
                        continue
                    
                    try:
                        # Convert back to original item ID
                        original_item = self.item_encoder.inverse_transform([idx_val - 1])[0]
                        
                        if original_item not in seen_items:
                            results.append((original_item, float(value.item())))
                            
                        if len(results) >= top_k:
                            break
                    except (ValueError, IndexError):
                        continue
                
                return results
                
        except Exception as e:
            self.logger.error(f"Error in prediction: {e}")
            return []
    
    def get_user_recommendations(self, user_id: int, top_k: int = 10) -> List[Tuple[int, float]]:
        """
        Get recommendations for a user based on their interaction history.
        
        Args:
            user_id: Original user ID
            top_k: Number of recommendations
        
        Returns:
            List of (item_id, score) tuples
        """
        try:
            user_encoded = self.user_encoder.transform([user_id])[0]
        except ValueError:
            self.logger.warning(f"Unknown user ID: {user_id}")
            return []
        
        # Get user's interaction sequence
        if user_encoded not in self.user_sequences:
            self.logger.warning(f"No sequence found for user {user_id}")
            return []
        
        user_sequence = self.user_sequences[user_encoded]
        
        # Convert encoded sequence back to original item IDs
        original_sequence = []
        for encoded_item in user_sequence:
            try:
                original_item = self.item_encoder.inverse_transform([encoded_item - 1])[0]
                original_sequence.append(original_item)
            except (ValueError, IndexError):
                continue
        
        if not original_sequence:
            return []
        
        # Predict next items
        return self.predict_next_items(original_sequence, top_k=top_k)
    
    def predict_sequence_continuation(self, sequence: List[int], 
                                    num_steps: int = 5) -> List[List[Tuple[int, float]]]:
        """
        Predict multiple next items in sequence.
        
        Args:
            sequence: Initial sequence
            num_steps: Number of future steps to predict
        
        Returns:
            List of predictions for each step
        """
        results = []
        current_sequence = sequence.copy()
        
        for step in range(num_steps):
            # Predict next item
            predictions = self.predict_next_items(current_sequence, top_k=5)
            
            if not predictions:
                break
            
            results.append(predictions)
            
            # Add most likely item to sequence for next prediction
            next_item = predictions[0][0]
            current_sequence.append(next_item)
            
            # Limit sequence length
            if len(current_sequence) > 50:
                current_sequence = current_sequence[-50:]
        
        return results
    
    def analyze_user_patterns(self, user_id: int) -> Dict[str, any]:
        """
        Analyze user interaction patterns.
        
        Args:
            user_id: Original user ID
        
        Returns:
            Dictionary with pattern analysis
        """
        try:
            user_encoded = self.user_encoder.transform([user_id])[0]
        except ValueError:
            return {}
        
        if user_encoded not in self.user_sequences:
            return {}
        
        sequence = self.user_sequences[user_encoded]
        
        # Convert to original item IDs
        original_sequence = []
        for encoded_item in sequence:
            try:
                original_item = self.item_encoder.inverse_transform([encoded_item - 1])[0]
                original_sequence.append(original_item)
            except (ValueError, IndexError):
                continue
        
        if not original_sequence:
            return {}
        
        # Basic pattern analysis
        analysis = {
            'sequence_length': len(original_sequence),
            'unique_items': len(set(original_sequence)),
            'repeat_ratio': 1 - len(set(original_sequence)) / len(original_sequence),
            'most_frequent_items': self._get_most_frequent(original_sequence, top_k=5),
            'recent_items': original_sequence[-10:] if len(original_sequence) >= 10 else original_sequence
        }
        
        return analysis
    
    def _get_most_frequent(self, sequence: List[int], top_k: int = 5) -> List[Tuple[int, int]]:
        """Get most frequent items in sequence"""
        counts = defaultdict(int)
        for item in sequence:
            counts[item] += 1
        
        sorted_items = sorted(counts.items(), key=lambda x: x[1], reverse=True)
        return sorted_items[:top_k]
    
    def batch_predict_sessions(self, sessions: List[List[int]], 
                              top_k: int = 10) -> List[List[Tuple[int, float]]]:
        """
        Predict next items for multiple sessions efficiently.
        
        Args:
            sessions: List of item sequences
            top_k: Number of predictions per session
        
        Returns:
            List of predictions for each session
        """
        all_predictions = []
        
        for session in sessions:
            predictions = self.predict_next_items(session, top_k=top_k)
            all_predictions.append(predictions)
        
        return all_predictions


def main():
    parser = argparse.ArgumentParser(description='Sequential Model Inference')
    parser.add_argument('--model-path', type=str, required=True,
                       help='Path to trained model')
    parser.add_argument('--encoders-path', type=str, required=True,
                       help='Path to encoders file')
    parser.add_argument('--user-id', type=int, help='User ID for recommendations')
    parser.add_argument('--sequence', type=int, nargs='+', 
                       help='Item sequence for next-item prediction')
    parser.add_argument('--top-k', type=int, default=10,
                       help='Number of recommendations')
    parser.add_argument('--num-steps', type=int, default=5,
                       help='Number of future steps for sequence prediction')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Create inference object
    seq_inference = SequentialInference(
        model_path=args.model_path,
        encoders_path=args.encoders_path,
        device=args.device
    )
    
    # Example usage
    if args.user_id:
        # Get user recommendations
        recommendations = seq_inference.get_user_recommendations(
            args.user_id, top_k=args.top_k
        )
        print(f"Top {args.top_k} recommendations for user {args.user_id}:")
        for item_id, score in recommendations:
            print(f"  Item {item_id}: {score:.4f}")
        
        # Analyze user patterns
        patterns = seq_inference.analyze_user_patterns(args.user_id)
        if patterns:
            print(f"\nUser {args.user_id} patterns:")
            for key, value in patterns.items():
                print(f"  {key}: {value}")
    
    if args.sequence:
        # Predict next items
        predictions = seq_inference.predict_next_items(
            args.sequence, top_k=args.top_k
        )
        print(f"\nNext item predictions for sequence {args.sequence}:")
        for item_id, score in predictions:
            print(f"  Item {item_id}: {score:.4f}")
        
        # Predict sequence continuation
        continuation = seq_inference.predict_sequence_continuation(
            args.sequence, num_steps=args.num_steps
        )
        print(f"\nSequence continuation for {args.num_steps} steps:")
        for step, step_predictions in enumerate(continuation):
            print(f"  Step {step + 1}: {step_predictions[:3]}")


if __name__ == "__main__":
    main()