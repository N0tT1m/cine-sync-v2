#!/usr/bin/env python3
"""
Test script to verify all 4 models work on RTX 4090
"""
import torch
import torch.nn as nn
import numpy as np
import sys
import os
from typing import Dict, List

# Add model directories to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'advanced_models'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'neural_collaborative_filtering', 'src'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'sequential_models', 'src'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'two_tower_model', 'src'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'hybrid_recommendation', 'models'))

try:
    from enhanced_two_tower import UltimateTwoTowerModel
    from model import NeuralCollaborativeFiltering, SimpleNCF, DeepNCF
    from model import SequentialRecommender, AttentionalSequentialRecommender
    from model import TwoTowerModel, EnhancedTwoTowerModel
    from tv_recommender import TVShowRecommenderModel
except ImportError as e:
    print(f"Import error: {e}")
    print("Please ensure all model files are in correct directories")
    sys.exit(1)

def test_memory_usage():
    """Test GPU memory usage for all models"""
    if not torch.cuda.is_available():
        print("CUDA not available - testing on CPU")
        device = torch.device('cpu')
    else:
        device = torch.device('cuda')
        print(f"Testing on {torch.cuda.get_device_name()}")
        print(f"Total GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    results = {}
    
    # Test 1: Enhanced Two-Tower Model
    print("\n=== Testing Enhanced Two-Tower Model ===")
    try:
        torch.cuda.empty_cache()
        initial_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        
        model1 = UltimateTwoTowerModel(
            user_categorical_dims={'age_group': 10, 'gender': 3, 'occupation': 20},
            user_numerical_dim=5,
            num_users=10000,
            item_categorical_dims={'genre': 20, 'year': 100, 'director': 1000},
            item_numerical_dim=10,
            num_items=50000,
            embedding_dim=512,
            tower_hidden_dims=[1024, 512, 256],
            cross_attention_heads=16,
            dropout=0.2
        ).to(device)
        
        # Test forward pass
        batch_size = 32
        user_cat = {
            'age_group': torch.randint(0, 10, (batch_size,)).to(device),
            'gender': torch.randint(0, 3, (batch_size,)).to(device),
            'occupation': torch.randint(0, 20, (batch_size,)).to(device)
        }
        item_cat = {
            'genre': torch.randint(0, 20, (batch_size,)).to(device),
            'year': torch.randint(0, 100, (batch_size,)).to(device),
            'director': torch.randint(0, 1000, (batch_size,)).to(device)
        }
        user_num = torch.randn(batch_size, 5).to(device)
        item_num = torch.randn(batch_size, 10).to(device)
        user_ids = torch.randint(0, 10000, (batch_size,)).to(device)
        item_ids = torch.randint(0, 50000, (batch_size,)).to(device)
        
        with torch.no_grad():
            output = model1(user_cat, item_cat, user_num, item_num, user_ids, item_ids)
        
        memory_used = (torch.cuda.memory_allocated() - initial_memory) / 1e9 if torch.cuda.is_available() else 0
        results['Enhanced Two-Tower'] = {
            'status': 'SUCCESS',
            'memory_gb': memory_used,
            'output_shape': output.shape,
            'params': sum(p.numel() for p in model1.parameters())
        }
        print(f"✓ Success - Memory: {memory_used:.2f} GB, Params: {results['Enhanced Two-Tower']['params']:,}")
        
        del model1
        torch.cuda.empty_cache()
        
    except Exception as e:
        results['Enhanced Two-Tower'] = {'status': 'FAILED', 'error': str(e)}
        print(f"✗ Failed: {e}")
    
    # Test 2: Neural Collaborative Filtering
    print("\n=== Testing Neural Collaborative Filtering ===")
    try:
        torch.cuda.empty_cache()
        initial_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        
        model2 = NeuralCollaborativeFiltering(
            num_users=10000,
            num_items=50000,
            embedding_dim=128,
            hidden_layers=[256, 128, 64],
            dropout=0.2
        ).to(device)
        
        # Test forward pass
        batch_size = 128
        user_ids = torch.randint(0, 10000, (batch_size,)).to(device)
        item_ids = torch.randint(0, 50000, (batch_size,)).to(device)
        
        with torch.no_grad():
            output = model2(user_ids, item_ids)
        
        memory_used = (torch.cuda.memory_allocated() - initial_memory) / 1e9 if torch.cuda.is_available() else 0
        results['Neural Collaborative Filtering'] = {
            'status': 'SUCCESS',
            'memory_gb': memory_used,
            'output_shape': output.shape,
            'params': sum(p.numel() for p in model2.parameters())
        }
        print(f"✓ Success - Memory: {memory_used:.2f} GB, Params: {results['Neural Collaborative Filtering']['params']:,}")
        
        del model2
        torch.cuda.empty_cache()
        
    except Exception as e:
        results['Neural Collaborative Filtering'] = {'status': 'FAILED', 'error': str(e)}
        print(f"✗ Failed: {e}")
    
    # Test 3: Sequential Recommender
    print("\n=== Testing Sequential Recommender ===")
    try:
        torch.cuda.empty_cache()
        initial_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        
        model3 = AttentionalSequentialRecommender(
            num_items=50000,
            embedding_dim=256,
            num_heads=8,
            num_blocks=4,
            dropout=0.2,
            max_seq_len=100
        ).to(device)
        
        # Test forward pass
        batch_size = 64
        seq_len = 50
        sequences = torch.randint(1, 50000, (batch_size, seq_len)).to(device)
        
        with torch.no_grad():
            output = model3(sequences)
        
        memory_used = (torch.cuda.memory_allocated() - initial_memory) / 1e9 if torch.cuda.is_available() else 0
        results['Sequential Recommender'] = {
            'status': 'SUCCESS',
            'memory_gb': memory_used,
            'output_shape': output.shape,
            'params': sum(p.numel() for p in model3.parameters())
        }
        print(f"✓ Success - Memory: {memory_used:.2f} GB, Params: {results['Sequential Recommender']['params']:,}")
        
        del model3
        torch.cuda.empty_cache()
        
    except Exception as e:
        results['Sequential Recommender'] = {'status': 'FAILED', 'error': str(e)}
        print(f"✗ Failed: {e}")
    
    # Test 4: Hybrid TV Recommender
    print("\n=== Testing Hybrid TV Recommender ===")
    try:
        torch.cuda.empty_cache()
        initial_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        
        model4 = TVShowRecommenderModel(
            num_users=10000,
            num_shows=25000,
            num_genres=20,
            embedding_dim=128,
            hidden_dim=256
        ).to(device)
        
        # Test forward pass
        batch_size = 128
        user_ids = torch.randint(0, 10000, (batch_size,)).to(device)
        show_ids = torch.randint(0, 25000, (batch_size,)).to(device)
        genre_features = torch.randn(batch_size, 20).to(device)
        tv_features = torch.randn(batch_size, 4).to(device)
        
        with torch.no_grad():
            output = model4(user_ids, show_ids, genre_features, tv_features)
        
        memory_used = (torch.cuda.memory_allocated() - initial_memory) / 1e9 if torch.cuda.is_available() else 0
        results['Hybrid TV Recommender'] = {
            'status': 'SUCCESS',
            'memory_gb': memory_used,
            'output_shape': output.shape,
            'params': sum(p.numel() for p in model4.parameters())
        }
        print(f"✓ Success - Memory: {memory_used:.2f} GB, Params: {results['Hybrid TV Recommender']['params']:,}")
        
        del model4
        torch.cuda.empty_cache()
        
    except Exception as e:
        results['Hybrid TV Recommender'] = {'status': 'FAILED', 'error': str(e)}
        print(f"✗ Failed: {e}")
    
    return results

def test_concurrent_inference():
    """Test running multiple models concurrently"""
    print("\n=== Testing Concurrent Inference ===")
    
    if not torch.cuda.is_available():
        print("Skipping concurrent test - CUDA not available")
        return
    
    device = torch.device('cuda')
    
    try:
        torch.cuda.empty_cache()
        initial_memory = torch.cuda.memory_allocated()
        
        # Load smaller models concurrently
        model_ncf = SimpleNCF(num_users=10000, num_items=50000, embedding_dim=64).to(device)
        model_tv = TVShowRecommenderModel(num_users=10000, num_shows=25000, num_genres=20, 
                                         embedding_dim=64, hidden_dim=128).to(device)
        
        # Test concurrent inference
        batch_size = 64
        
        # NCF inference
        user_ids = torch.randint(0, 10000, (batch_size,)).to(device)
        item_ids = torch.randint(0, 50000, (batch_size,)).to(device)
        
        # TV inference
        show_ids = torch.randint(0, 25000, (batch_size,)).to(device)
        genre_features = torch.randn(batch_size, 20).to(device)
        tv_features = torch.randn(batch_size, 4).to(device)
        
        with torch.no_grad():
            ncf_output = model_ncf(user_ids, item_ids)
            tv_output = model_tv(user_ids, show_ids, genre_features, tv_features)
        
        memory_used = (torch.cuda.memory_allocated() - initial_memory) / 1e9
        print(f"✓ Concurrent inference successful - Memory: {memory_used:.2f} GB")
        
        del model_ncf, model_tv
        torch.cuda.empty_cache()
        
        return True
        
    except Exception as e:
        print(f"✗ Concurrent inference failed: {e}")
        return False

def main():
    print("=== CineSync Model Compatibility Test ===")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name()}")
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU Memory: {total_memory:.1f} GB")
        
        if total_memory < 20:
            print("⚠️  Warning: Less than 20GB GPU memory detected. Some models may not fit.")
    
    # Test individual models
    results = test_memory_usage()
    
    # Test concurrent inference
    concurrent_success = test_concurrent_inference()
    
    # Summary
    print("\n=== SUMMARY ===")
    total_memory = 0
    successful_models = 0
    
    for model_name, result in results.items():
        if result['status'] == 'SUCCESS':
            successful_models += 1
            total_memory += result['memory_gb']
            print(f"✓ {model_name}: {result['memory_gb']:.2f} GB ({result['params']:,} params)")
        else:
            print(f"✗ {model_name}: FAILED - {result.get('error', 'Unknown error')}")
    
    print(f"\nTotal memory for all models: {total_memory:.2f} GB")
    print(f"Successful models: {successful_models}/4")
    
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        if total_memory < gpu_memory * 0.8:  # 80% threshold
            print(f"✓ All models should fit on your GPU ({gpu_memory:.1f} GB)")
        else:
            print(f"⚠️  Models may not all fit simultaneously on your GPU ({gpu_memory:.1f} GB)")
            print("Consider training sequentially or using smaller batch sizes")
    
    if concurrent_success:
        print("✓ Concurrent inference test passed")
    
    print(f"\nRecommendation: {'Sequential training recommended' if total_memory > 20 else 'Can train multiple models simultaneously'}")

if __name__ == "__main__":
    main()