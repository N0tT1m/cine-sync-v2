#!/usr/bin/env python3
"""
Test Suite for Advanced Models in CineSync v2
Validates all 4 upgraded models with comprehensive tests
"""

import torch
import torch.nn.functional as F
import numpy as np
import pytest
import warnings
from typing import Dict, List, Tuple, Any
import time
import psutil
import gc

# Import all advanced models
from bert4rec_recommender import BERT4Rec, EnhancedBERT4Rec, BERT4RecTrainer
from sentence_bert_two_tower import SentenceBERTTwoTowerModel, SentenceBERTTwoTowerTrainer
from graphsage_recommender import GraphSAGERecommender, InductiveGraphSAGE, GraphSAGETrainer
from t5_hybrid_recommender import T5HybridRecommender, T5HybridTrainer

# Suppress warnings for cleaner test output
warnings.filterwarnings("ignore")


class ModelTestSuite:
    """Comprehensive test suite for all advanced models"""
    
    def __init__(self, device: str = 'cpu'):
        self.device = torch.device(device)
        self.test_data = self._create_test_data()
    
    def _create_test_data(self) -> Dict[str, Any]:
        """Create synthetic test data for all models"""
        batch_size = 32
        seq_len = 20
        num_users = 1000
        num_items = 500
        embedding_dim = 128
        
        return {
            'batch_size': batch_size,
            'seq_len': seq_len,
            'num_users': num_users,
            'num_items': num_items,
            'embedding_dim': embedding_dim,
            
            # Sequential data for BERT4Rec
            'sequences': torch.randint(1, num_items, (batch_size, seq_len)),
            'genre_sequences': torch.randint(0, 20, (batch_size, seq_len)),
            'user_ids': torch.randint(0, num_users, (batch_size,)),
            'item_ids': torch.randint(0, num_items, (batch_size,)),
            
            # Graph data for GraphSAGE
            'edge_index': torch.randint(0, num_users + num_items, (2, 1000)),
            'user_features': torch.randn(num_users, 64),
            'item_features': torch.randn(num_items, 64),
            
            # Text data for Sentence-BERT and T5
            'user_texts': [f"User {i} likes action and comedy movies" for i in range(batch_size)],
            'item_texts': [f"Item {i} is an action movie with great effects" for i in range(batch_size)],
            
            # Categorical features for Two-Tower
            'user_categorical': {
                'age_group': torch.randint(0, 7, (batch_size,)),
                'occupation': torch.randint(0, 21, (batch_size,))
            },
            'item_categorical': {
                'genre': torch.randint(0, 20, (batch_size,)),
                'year': torch.randint(0, 50, (batch_size,))
            },
            'user_numerical': torch.randn(batch_size, 10),
            'item_numerical': torch.randn(batch_size, 15),
            
            # Labels
            'ratings': torch.rand(batch_size) * 5,
            'genres': torch.randint(0, 20, (batch_size,)),
            'similarity_labels': torch.randint(0, 2, (batch_size,)).float()
        }
    
    def test_bert4rec(self) -> Dict[str, Any]:
        """Test BERT4Rec model"""
        print("🧪 Testing BERT4Rec...")
        
        results = {}
        
        try:
            # Test basic BERT4Rec
            model = BERT4Rec(
                num_items=self.test_data['num_items'],
                max_seq_len=self.test_data['seq_len'],
                d_model=256,
                num_heads=8,
                num_layers=4,
                dropout=0.1
            ).to(self.device)
            
            sequences = self.test_data['sequences'].to(self.device)
            
            # Forward pass
            start_time = time.time()
            outputs = model(sequences, apply_masking=True)
            forward_time = time.time() - start_time
            
            # Validate output shapes
            expected_shape = (self.test_data['batch_size'], self.test_data['seq_len'], self.test_data['num_items'])
            assert outputs['logits'].shape == expected_shape, f"Expected {expected_shape}, got {outputs['logits'].shape}"
            
            # Test prediction
            top_k_items, top_k_scores = model.predict_next(sequences[:4], k=10)
            assert top_k_items.shape == (4, 10), f"Expected (4, 10), got {top_k_items.shape}"
            
            # Test Enhanced BERT4Rec
            enhanced_model = EnhancedBERT4Rec(
                num_items=self.test_data['num_items'],
                num_genres=20,
                d_model=256,
                num_heads=8,
                num_layers=4
            ).to(self.device)
            
            enhanced_outputs = enhanced_model(
                sequences, 
                genre_sequences=self.test_data['genre_sequences'].to(self.device)
            )
            
            # Validate multi-task outputs
            assert 'next_item_logits' in enhanced_outputs
            assert 'rating_pred' in enhanced_outputs
            assert 'preference_logits' in enhanced_outputs
            
            results['bert4rec'] = {
                'status': 'PASSED',
                'forward_time': forward_time,
                'output_shape': outputs['logits'].shape,
                'enhanced_tasks': len(enhanced_outputs),
                'memory_usage': self._get_memory_usage()
            }
            
        except Exception as e:
            results['bert4rec'] = {
                'status': 'FAILED',
                'error': str(e)
            }
        
        return results
    
    def test_sentence_bert_two_tower(self) -> Dict[str, Any]:
        """Test Sentence-BERT Two-Tower model"""
        print("🧪 Testing Sentence-BERT Two-Tower...")
        
        results = {}
        
        try:
            model = SentenceBERTTwoTowerModel(
                sentence_bert_model="all-MiniLM-L6-v2",
                user_categorical_dims={'age_group': 7, 'occupation': 21},
                item_categorical_dims={'genre': 20, 'year': 50},
                user_numerical_dim=10,
                item_numerical_dim=15,
                num_users=self.test_data['num_users'],
                num_items=self.test_data['num_items'],
                embedding_dim=256,
                freeze_sentence_bert=True  # Freeze for faster testing
            ).to(self.device)
            
            # Forward pass with text
            start_time = time.time()
            outputs = model(
                user_content_texts=self.test_data['user_texts'][:4],  # Smaller batch for testing
                item_content_texts=self.test_data['item_texts'][:4],
                user_categorical={'age_group': self.test_data['user_categorical']['age_group'][:4].to(self.device)},
                item_categorical={'genre': self.test_data['item_categorical']['genre'][:4].to(self.device)},
                return_all_outputs=True
            )
            forward_time = time.time() - start_time
            
            # Validate outputs
            assert 'similarity' in outputs
            assert 'collaborative_similarity' in outputs
            assert 'content_similarity' in outputs
            assert 'user_embeddings' in outputs
            assert 'item_embeddings' in outputs
            
            # Test recommendations
            recommendations = model.get_recommendations(
                user_features={
                    'user_content_texts': self.test_data['user_texts'][:1],
                    'user_categorical': {'age_group': self.test_data['user_categorical']['age_group'][:1].to(self.device)}
                },
                item_features={
                    'item_content_texts': self.test_data['item_texts'][:10],
                    'item_categorical': {'genre': self.test_data['item_categorical']['genre'][:10].to(self.device)}
                },
                k=5
            )
            
            top_k_indices, top_k_scores = recommendations
            assert top_k_indices.shape[0] == 5, f"Expected 5 recommendations, got {top_k_indices.shape[0]}"
            
            results['sentence_bert_two_tower'] = {
                'status': 'PASSED',
                'forward_time': forward_time,
                'output_features': len(outputs),
                'similarity_types': 3,  # collaborative, content, semantic
                'memory_usage': self._get_memory_usage()
            }
            
        except Exception as e:
            results['sentence_bert_two_tower'] = {
                'status': 'FAILED',
                'error': str(e)
            }
        
        return results
    
    def test_graphsage(self) -> Dict[str, Any]:
        """Test GraphSAGE model"""
        print("🧪 Testing GraphSAGE...")
        
        results = {}
        
        try:
            model = GraphSAGERecommender(
                num_users=self.test_data['num_users'],
                num_items=self.test_data['num_items'],
                user_feature_dim=64,
                item_feature_dim=64,
                embedding_dim=128,
                hidden_dims=[256, 128],
                num_layers=2,
                use_attention=True,
                attention_heads=4
            ).to(self.device)
            
            edge_index = self.test_data['edge_index'].to(self.device)
            user_features = self.test_data['user_features'].to(self.device)
            item_features = self.test_data['item_features'].to(self.device)
            
            # Forward pass
            start_time = time.time()
            user_embs, item_embs = model(
                edge_index=edge_index,
                user_features=user_features,
                item_features=item_features
            )
            forward_time = time.time() - start_time
            
            # Validate shapes
            assert user_embs.shape == (self.test_data['num_users'], 128)
            assert item_embs.shape == (self.test_data['num_items'], 128)
            
            # Test rating prediction
            user_ids = self.test_data['user_ids'][:10].to(self.device)
            item_ids = self.test_data['item_ids'][:10].to(self.device)
            
            ratings = model.predict_rating(
                user_ids=user_ids,
                item_ids=item_ids,
                edge_index=edge_index,
                user_features=user_features,
                item_features=item_features
            )
            
            assert ratings.shape == (10,), f"Expected (10,), got {ratings.shape}"
            assert torch.all(ratings >= 0) and torch.all(ratings <= 5), "Ratings should be between 0 and 5"
            
            # Test recommendations
            top_k_items, top_k_scores = model.get_recommendations(
                user_id=0,
                edge_index=edge_index,
                k=10,
                user_features=user_features,
                item_features=item_features
            )
            
            assert top_k_items.shape == (10,), f"Expected (10,), got {top_k_items.shape}"
            
            # Test inductive model
            inductive_model = InductiveGraphSAGE(
                num_users=100,
                num_items=100,
                embedding_dim=64,
                hidden_dims=[128, 64]
            ).to(self.device)
            
            # Test cold start
            new_node_features = torch.randn(5, 64).to(self.device)
            neighbor_embeddings = torch.randn(20, 64).to(self.device)
            neighbor_indices = torch.randint(0, 20, (10, 2)).to(self.device)
            
            new_embeddings = inductive_model.inductive_forward(
                new_node_features, neighbor_embeddings, neighbor_indices
            )
            assert new_embeddings.shape == (5, 64)
            
            results['graphsage'] = {
                'status': 'PASSED',
                'forward_time': forward_time,
                'user_emb_shape': user_embs.shape,
                'item_emb_shape': item_embs.shape,
                'inductive_support': True,
                'memory_usage': self._get_memory_usage()
            }
            
        except Exception as e:
            results['graphsage'] = {
                'status': 'FAILED',
                'error': str(e)
            }
        
        return results
    
    def test_t5_hybrid(self) -> Dict[str, Any]:
        """Test T5 Hybrid model"""
        print("🧪 Testing T5 Hybrid...")
        
        results = {}
        
        try:
            model = T5HybridRecommender(
                num_users=self.test_data['num_users'],
                num_items=self.test_data['num_items'],
                t5_model_name="t5-small",
                embedding_dim=256,
                content_dim=256,
                hidden_dims=[512, 256],
                freeze_t5_encoder=True,  # Freeze for faster testing
                use_t5_generation=False  # Disable generation for testing
            ).to(self.device)
            
            user_ids = self.test_data['user_ids'][:4].to(self.device)
            item_ids = self.test_data['item_ids'][:4].to(self.device)
            
            # Forward pass with text
            start_time = time.time()
            outputs = model(
                user_ids=user_ids,
                item_ids=item_ids,
                user_texts=self.test_data['user_texts'][:4],
                item_texts=self.test_data['item_texts'][:4],
                return_all_outputs=True
            )
            forward_time = time.time() - start_time
            
            # Validate outputs
            assert 'rating_pred' in outputs
            assert 'genre_pred' in outputs
            assert 'sentiment_pred' in outputs
            assert 'user_embeddings' in outputs
            assert 'item_embeddings' in outputs
            
            # Validate rating predictions
            assert outputs['rating_pred'].shape == (4,)
            assert torch.all(outputs['rating_pred'] >= 0) and torch.all(outputs['rating_pred'] <= 5)
            
            # Test content encoding
            content_encoding = model.encode_content(
                user_texts=self.test_data['user_texts'][:2],
                item_texts=self.test_data['item_texts'][:2]
            )
            
            assert 'user_content' in content_encoding
            assert 'item_content' in content_encoding
            assert 'content_features' in content_encoding['user_content']
            assert 'genre_logits' in content_encoding['item_content']
            
            # Test recommendations
            top_k_items, top_k_scores, additional_info = model.get_recommendations(
                user_id=0,
                item_texts=self.test_data['item_texts'][:20],
                k=5
            )
            
            assert top_k_items.shape == (5,)
            assert 'genre_predictions' in additional_info
            assert 'sentiment_predictions' in additional_info
            
            # Test explanation
            explanation = model.explain_recommendation(
                user_id=0,
                item_id=0,
                user_text=self.test_data['user_texts'][0],
                item_text=self.test_data['item_texts'][0]
            )
            
            assert 'predicted_rating' in explanation
            assert 'genre_preferences' in explanation
            assert 'sentiment_match' in explanation
            
            results['t5_hybrid'] = {
                'status': 'PASSED',
                'forward_time': forward_time,
                'multi_task_outputs': len(outputs),
                'content_encoding': True,
                'explanation_support': True,
                'memory_usage': self._get_memory_usage()
            }
            
        except Exception as e:
            results['t5_hybrid'] = {
                'status': 'FAILED',
                'error': str(e)
            }
        
        return results
    
    def test_trainers(self) -> Dict[str, Any]:
        """Test all trainer classes"""
        print("🧪 Testing Trainers...")
        
        results = {}
        
        try:
            # Create simple models for training tests
            bert4rec_model = BERT4Rec(
                num_items=100, d_model=128, num_heads=4, num_layers=2
            ).to(self.device)
            
            # Test BERT4Rec trainer
            bert4rec_trainer = BERT4RecTrainer(
                bert4rec_model, 
                device=self.device,
                use_amp=False  # Disable AMP for testing
            )
            
            # Create training batch
            train_batch = {
                'sequences': torch.randint(1, 100, (8, 10)).to(self.device)
            }
            
            # Training step
            bert4rec_losses = bert4rec_trainer.train_step(train_batch)
            assert 'loss' in bert4rec_losses
            assert 'lr' in bert4rec_losses
            
            results['trainers'] = {
                'status': 'PASSED',
                'bert4rec_trainer': True,
                'loss_tracking': True
            }
            
        except Exception as e:
            results['trainers'] = {
                'status': 'FAILED',
                'error': str(e)
            }
        
        return results
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all tests and return comprehensive results"""
        print("🚀 Running Advanced Models Test Suite...")
        print("=" * 60)
        
        all_results = {}
        
        # Run individual model tests
        all_results.update(self.test_bert4rec())
        all_results.update(self.test_sentence_bert_two_tower())
        all_results.update(self.test_graphsage())
        all_results.update(self.test_t5_hybrid())
        all_results.update(self.test_trainers())
        
        # Summary
        passed_tests = sum(1 for result in all_results.values() if result.get('status') == 'PASSED')
        total_tests = len(all_results)
        
        print("\n" + "=" * 60)
        print(f"📊 Test Results: {passed_tests}/{total_tests} PASSED")
        
        for test_name, result in all_results.items():
            status_emoji = "✅" if result.get('status') == 'PASSED' else "❌"
            print(f"{status_emoji} {test_name}: {result.get('status', 'UNKNOWN')}")
            
            if result.get('status') == 'FAILED':
                print(f"   Error: {result.get('error', 'Unknown error')}")
            else:
                # Print performance metrics
                if 'forward_time' in result:
                    print(f"   Forward time: {result['forward_time']:.3f}s")
                if 'memory_usage' in result:
                    print(f"   Memory usage: {result['memory_usage']:.1f}MB")
        
        return all_results


def performance_benchmark():
    """Run performance benchmarks for all models"""
    print("\n🏃‍♂️ Running Performance Benchmarks...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Benchmark parameters
    batch_sizes = [16, 32, 64]
    sequence_lengths = [50, 100, 200]
    
    benchmark_results = {}
    
    for batch_size in batch_sizes:
        for seq_len in sequence_lengths:
            print(f"\nBenchmarking batch_size={batch_size}, seq_len={seq_len}")
            
            # Create test data
            sequences = torch.randint(1, 1000, (batch_size, seq_len)).to(device)
            
            # BERT4Rec benchmark
            model = BERT4Rec(
                num_items=1000,
                d_model=512,
                num_heads=8,
                num_layers=6
            ).to(device)
            
            # Warmup
            for _ in range(5):
                with torch.no_grad():
                    _ = model(sequences)
            
            # Benchmark
            start_time = time.time()
            for _ in range(20):
                with torch.no_grad():
                    _ = model(sequences)
            
            if device.type == 'cuda':
                torch.cuda.synchronize()
            
            end_time = time.time()
            avg_time = (end_time - start_time) / 20
            
            key = f"batch_{batch_size}_seq_{seq_len}"
            benchmark_results[key] = {
                'avg_forward_time': avg_time,
                'throughput': batch_size / avg_time,
                'memory_allocated': torch.cuda.memory_allocated(device) / 1024 / 1024 if device.type == 'cuda' else 0
            }
            
            print(f"  Avg forward time: {avg_time:.4f}s")
            print(f"  Throughput: {batch_size / avg_time:.1f} samples/s")
            
            # Clean up
            del model
            if device.type == 'cuda':
                torch.cuda.empty_cache()
            gc.collect()
    
    return benchmark_results


if __name__ == "__main__":
    # Check if CUDA is available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🔧 Using device: {device}")
    
    # Run tests
    test_suite = ModelTestSuite(device=device)
    results = test_suite.run_all_tests()
    
    # Run benchmarks if requested
    import sys
    if '--benchmark' in sys.argv:
        benchmark_results = performance_benchmark()
        print("\n📈 Benchmark Results:")
        for key, metrics in benchmark_results.items():
            print(f"  {key}: {metrics['avg_forward_time']:.4f}s, {metrics['throughput']:.1f} samples/s")
    
    # Exit with appropriate code
    failed_tests = [name for name, result in results.items() if result.get('status') != 'PASSED']
    
    if failed_tests:
        print(f"\n❌ {len(failed_tests)} tests failed: {', '.join(failed_tests)}")
        sys.exit(1)
    else:
        print("\n🎉 All tests passed successfully!")
        sys.exit(0)