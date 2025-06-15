#!/usr/bin/env python3
"""
Comprehensive Inference Engine for Sequential Recommendation Models

Provides production-ready inference capabilities for all sequential architectures:
- Next-item prediction for real-time recommendations
- User-based recommendations using interaction history
- Multi-step sequence continuation for session planning
- Batch processing for efficient bulk predictions
- User behavior pattern analysis

Supports all model types: RNN-based, attention-based, hierarchical, and session-based.
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
    Production-ready inference engine for trained sequential recommendation models.
    
    Handles model loading, ID encoding/decoding, and provides multiple inference modes:
    - Single sequence next-item prediction
    - User-based recommendations from interaction history
    - Multi-step sequence continuation
    - Batch processing for efficiency
    - User behavior analysis
    
    Automatically handles different model architectures and their specific requirements.
    """
    
    def __init__(self, model_path: str, encoders_path: str, device: str = 'auto'):
        """
        Args:
            model_path: Path to trained model checkpoint
            encoders_path: Path to saved encoders
            device: Device to use for inference
        """
        self.device = device if device != 'auto' else ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load encoders and user sequences from training
        # These maintain the ID mappings used during training for consistency
        with open(encoders_path, 'rb') as f:
            data = pickle.load(f)
        
        self.user_encoder = data['user_encoder']      # Maps user IDs to indices
        self.item_encoder = data['item_encoder']      # Maps item IDs to indices
        self.user_sequences = data.get('user_sequences', {})  # Training sequences
        
        # Load model
        self._load_model(model_path)
        
        self.logger = logging.getLogger(__name__)
    
    def _load_model(self, model_path: str):
        """Load trained model and reconstruct architecture from checkpoint
        
        Automatically detects model type and recreates the exact architecture
        used during training, then loads the trained weights.
        """
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Extract model configuration from checkpoint
        model_config = checkpoint.get('model_config', {})
        model_class = model_config.get('model_class', 'SequentialRecommender')
        
        num_items = len(self.item_encoder.classes_) + 1  # +1 for padding
        
        # Reconstruct model architecture based on saved configuration
        if model_class == 'AttentionalSequentialRecommender':
            # Self-attention based model (SASRec-style)
            self.model = AttentionalSequentialRecommender(
                num_items=num_items,
                embedding_dim=model_config.get('embedding_dim', 128),
                num_heads=8,  # Default attention heads
                num_blocks=2, # Default transformer blocks
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
        Core next-item prediction function for real-time recommendations.
        
        Takes a sequence of item interactions and predicts the most likely next items.
        Handles ID encoding/decoding, sequence padding, and model-specific inference.
        
        Args:
            sequence: List of item IDs in chronological order (recent items last)
            top_k: Number of top predictions to return
            exclude_seen: Whether to filter out items already in the sequence
        
        Returns:
            List of (item_id, probability) tuples sorted by likelihood
        """
        try:
            # Encode sequence using training-time item encoder
            # Convert original item IDs to the indices used during training
            encoded_sequence = []
            for item_id in sequence:
                try:
                    # +1 offset because 0 is reserved for padding in training
                    encoded_item = self.item_encoder.transform([item_id])[0] + 1
                    encoded_sequence.append(encoded_item)
                except ValueError:
                    # Skip unknown items not seen during training
                    self.logger.warning(f"Unknown item ID: {item_id}")
            
            if not encoded_sequence:
                return []
            
            # Pad or truncate sequence to match training format
            max_len = 50  # Should match training max_seq_length
            if len(encoded_sequence) > max_len:
                # Keep most recent items (recency bias for recommendations)
                encoded_sequence = encoded_sequence[-max_len:]
            else:
                # Left-pad with zeros (item 0 = padding token)
                encoded_sequence = [0] * (max_len - len(encoded_sequence)) + encoded_sequence
            
            # Convert to tensor
            sequence_tensor = torch.LongTensor([encoded_sequence]).to(self.device)
            length_tensor = torch.LongTensor([min(len(sequence), max_len)]).to(self.device)
            
            # Model inference with architecture-specific handling
            with torch.no_grad():
                if isinstance(self.model, AttentionalSequentialRecommender):
                    # Self-attention model needs causal mask to prevent future leakage
                    seq_len = sequence_tensor.size(1)
                    mask = torch.tril(torch.ones(seq_len, seq_len, device=self.device))  # Lower triangular
                    logits = self.model(sequence_tensor, mask)
                    predictions = logits[0, -1]  # Extract predictions from last valid position
                    
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
                
                # Convert logits to probabilities for ranking
                probabilities = torch.softmax(predictions, dim=0)
                
                # Get top candidates (2x top_k to allow for filtering)
                top_k_values, top_k_indices = torch.topk(probabilities, min(top_k * 2, len(probabilities)))
                
                # Convert model indices back to original item IDs and apply filtering
                results = []
                seen_items = set(sequence) if exclude_seen else set()
                
                for value, idx in zip(top_k_values, top_k_indices):
                    idx_val = idx.item()
                    if idx_val == 0:  # Skip padding token (not a real item)
                        continue
                    
                    try:
                        # Decode back to original item ID (reverse the +1 offset)
                        original_item = self.item_encoder.inverse_transform([idx_val - 1])[0]
                        
                        # Filter out previously seen items if requested
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
        Generate personalized recommendations for a user using their interaction history.
        
        Retrieves the user's historical interaction sequence from training data
        and uses it to predict the next items they're likely to interact with.
        
        Args:
            user_id: Original user ID from the dataset
            top_k: Number of recommendation candidates to return
        
        Returns:
            List of (item_id, predicted_score) tuples ranked by likelihood
        """
        try:
            user_encoded = self.user_encoder.transform([user_id])[0]
        except ValueError:
            self.logger.warning(f"Unknown user ID: {user_id}")
            return []
        
        # Retrieve user's interaction sequence from training data
        if user_encoded not in self.user_sequences:
            self.logger.warning(f"No interaction history found for user {user_id}")
            return []
        
        user_sequence = self.user_sequences[user_encoded]  # Encoded sequence from training
        
        # Decode sequence back to original item IDs for processing
        original_sequence = []
        for encoded_item in user_sequence:
            try:
                # Reverse the encoding process (subtract 1 to undo padding offset)
                original_item = self.item_encoder.inverse_transform([encoded_item - 1])[0]
                original_sequence.append(original_item)
            except (ValueError, IndexError):
                # Skip any corrupted or invalid encoded items
                continue
        
        if not original_sequence:
            return []
        
        # Predict next items
        return self.predict_next_items(original_sequence, top_k=top_k)
    
    def predict_sequence_continuation(self, sequence: List[int], 
                                    num_steps: int = 5) -> List[List[Tuple[int, float]]]:
        """
        Predict multi-step sequence continuation using autoregressive generation.
        
        Useful for session planning and understanding longer-term user behavior.
        Each prediction step uses the previous predictions to generate the next,
        simulating how a user's session might evolve.
        
        Args:
            sequence: Initial sequence of item interactions
            num_steps: Number of future interaction steps to predict
        
        Returns:
            List of prediction lists, one for each future step
        """
        results = []
        current_sequence = sequence.copy()
        
        for step in range(num_steps):
            # Predict next items based on current sequence state
            predictions = self.predict_next_items(current_sequence, top_k=5)
            
            if not predictions:
                break  # No valid predictions available
            
            results.append(predictions)
            
            # Autoregressive generation: add most likely item for next step
            next_item = predictions[0][0]  # Top prediction
            current_sequence.append(next_item)
            
            # Limit sequence length
            if len(current_sequence) > 50:
                current_sequence = current_sequence[-50:]
        
        return results
    
    def analyze_user_patterns(self, user_id: int) -> Dict[str, any]:
        """
        Comprehensive analysis of user interaction patterns and behavior.
        
        Provides insights into user preferences, diversity, and engagement patterns
        that can inform recommendation strategies and user understanding.
        
        Args:
            user_id: Original user ID to analyze
        
        Returns:
            Dictionary containing behavioral pattern metrics and insights
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
        
        # Comprehensive behavioral pattern analysis
        analysis = {
            'sequence_length': len(original_sequence),                              # Total interactions
            'unique_items': len(set(original_sequence)),                           # Item diversity
            'repeat_ratio': 1 - len(set(original_sequence)) / len(original_sequence), # Repetition tendency
            'most_frequent_items': self._get_most_frequent(original_sequence, top_k=5), # Preferred items
            'recent_items': original_sequence[-10:] if len(original_sequence) >= 10 else original_sequence # Recent activity
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
        Efficient batch processing for multiple session predictions.
        
        Useful for bulk recommendation generation, A/B testing scenarios,
        or batch processing of recommendation requests. Currently processes
        sessions sequentially but can be extended for true batch inference.
        
        Args:
            sessions: List of item interaction sequences to process
            top_k: Number of prediction candidates per session
        
        Returns:
            List of prediction lists, one for each input session
        """
        all_predictions = []
        
        for session in sessions:
            predictions = self.predict_next_items(session, top_k=top_k)
            all_predictions.append(predictions)
        
        return all_predictions


def main():
    """Command-line interface for sequential model inference demonstration"""
    parser = argparse.ArgumentParser(description='Sequential Recommendation Model Inference')
    parser.add_argument('--model-path', type=str, required=True,
                       help='Path to trained model checkpoint (.pt file)')
    parser.add_argument('--encoders-path', type=str, required=True,
                       help='Path to saved encoders and sequences (.pkl file)')
    parser.add_argument('--user-id', type=int, 
                       help='User ID for personalized recommendations')
    parser.add_argument('--sequence', type=int, nargs='+', 
                       help='Item sequence for next-item prediction (space-separated IDs)')
    parser.add_argument('--top-k', type=int, default=10,
                       help='Number of top recommendations to return')
    parser.add_argument('--num-steps', type=int, default=5,
                       help='Number of future steps for sequence continuation')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device for inference (auto, cuda, cpu)')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Initialize inference engine with trained model and encoders
    seq_inference = SequentialInference(
        model_path=args.model_path,      # Trained model weights
        encoders_path=args.encoders_path, # ID mappings from training
        device=args.device               # Computing device
    )
    
    # Demonstration of inference capabilities
    if args.user_id:
        # Generate personalized recommendations based on user history
        recommendations = seq_inference.get_user_recommendations(
            args.user_id, top_k=args.top_k
        )
        print(f"Top {args.top_k} personalized recommendations for user {args.user_id}:")
        for item_id, score in recommendations:
            print(f"  Item {item_id}: {score:.4f}")
        
        # Analyze user behavioral patterns
        patterns = seq_inference.analyze_user_patterns(args.user_id)
        if patterns:
            print(f"\nBehavioral analysis for user {args.user_id}:")
            for key, value in patterns.items():
                print(f"  {key}: {value}")
    
    if args.sequence:
        # Predict next items for given sequence
        predictions = seq_inference.predict_next_items(
            args.sequence, top_k=args.top_k
        )
        print(f"\nNext item predictions for sequence {args.sequence}:")
        for item_id, score in predictions:
            print(f"  Item {item_id}: {score:.4f}")
        
        # Demonstrate multi-step sequence continuation
        continuation = seq_inference.predict_sequence_continuation(
            args.sequence, num_steps=args.num_steps
        )
        print(f"\nAutoregressive sequence continuation for {args.num_steps} steps:")
        for step, step_predictions in enumerate(continuation):
            print(f"  Step {step + 1}: {step_predictions[:3]}")


if __name__ == "__main__":
    main()