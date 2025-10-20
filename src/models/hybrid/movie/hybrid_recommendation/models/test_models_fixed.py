#!/usr/bin/env python3
"""
Fixed test script to verify all 4 CineSync models work correctly
"""
import torch
import torch.nn as nn
import numpy as np
import sys
import os
import importlib.util
from typing import Dict, List

# Add model directories to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))

sys.path.append(os.path.join(project_root, 'advanced_models'))
sys.path.append(os.path.join(project_root, 'neural_collaborative_filtering', 'src'))
sys.path.append(os.path.join(project_root, 'sequential_models', 'src'))
sys.path.append(os.path.join(project_root, 'two_tower_model', 'src'))
sys.path.append(current_dir)

# Import models individually and handle failures gracefully
models_available = {}

try:
    from enhanced_two_tower import UltimateTwoTowerModel
    models_available['enhanced_two_tower'] = True
    print("✓ Enhanced Two-Tower model imported")
except ImportError as e:
    models_available['enhanced_two_tower'] = False
    print(f"✗ Enhanced Two-Tower import failed: {e}")

try:
    from model import NeuralCollaborativeFiltering, SimpleNCF, DeepNCF
    models_available['ncf'] = True
    print("✓ Neural Collaborative Filtering models imported")
except ImportError as e:
    models_available['ncf'] = False
    print(f"✗ NCF import failed: {e}")

# Sequential models - import directly
try:
    sequential_models_path = os.path.join(project_root, 'sequential_models', 'src')
    if sequential_models_path not in sys.path:
        sys.path.insert(0, sequential_models_path)
    
    # Clear any cached imports 
    if 'model' in sys.modules:
        del sys.modules['model']
    
    spec = importlib.util.spec_from_file_location("sequential_model", 
                                                 os.path.join(sequential_models_path, "model.py"))
    sequential_model = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sequential_model)
    
    SequentialRecommender = sequential_model.SequentialRecommender
    AttentionalSequentialRecommender = sequential_model.AttentionalSequentialRecommender
    models_available['sequential'] = True
    print("✓ Sequential Recommender models imported")
except Exception as e:
    models_available['sequential'] = False
    print(f"✗ Sequential models import failed: {e}")

# Two-Tower models - import directly
try:
    two_tower_models_path = os.path.join(project_root, 'two_tower_model', 'src')
    
    spec = importlib.util.spec_from_file_location("two_tower_model", 
                                                 os.path.join(two_tower_models_path, "model.py"))
    two_tower_model = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(two_tower_model)
    
    TwoTowerModel = two_tower_model.TwoTowerModel
    EnhancedTwoTowerModel = two_tower_model.EnhancedTwoTowerModel
    models_available['two_tower'] = True
    print("✓ Two-Tower models imported")
except Exception as e:
    models_available['two_tower'] = False
    print(f"✗ Two-Tower models import failed: {e}")

try:
    from tv_recommender import TVShowRecommenderModel
    models_available['tv_recommender'] = True
    print("✓ TV Recommender model imported")
except ImportError as e:
    models_available['tv_recommender'] = False
    print(f"✗ TV Recommender import failed: {e}")

# Original Hybrid Recommender Model
try:
    hybrid_model_path = os.path.join(project_root, 'lupe(python)', 'models')
    
    spec = importlib.util.spec_from_file_location("hybrid_recommender", 
                                                 os.path.join(hybrid_model_path, "hybrid_recommender.py"))
    hybrid_recommender = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(hybrid_recommender)
    
    HybridRecommenderModel = hybrid_recommender.HybridRecommenderModel
    models_available['hybrid_recommender'] = True
    print("✓ Original Hybrid Recommender model imported")
except Exception as e:
    models_available['hybrid_recommender'] = False
    print(f"✗ Hybrid Recommender import failed: {e}")

def test_model_1_enhanced_two_tower():
    """Test 1: Enhanced Two-Tower Model"""
    print("\n=== Testing Enhanced Two-Tower Model ===")
    
    if not models_available['enhanced_two_tower']:
        print("✗ Enhanced Two-Tower model not available")
        return False, 0, 0
    
    try:
        torch.cuda.empty_cache()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        initial_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        
        model = UltimateTwoTowerModel(
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
            output = model(user_cat, item_cat, user_num, item_num, user_ids, item_ids)
        
        memory_used = (torch.cuda.memory_allocated() - initial_memory) / 1e9 if torch.cuda.is_available() else 0
        params = sum(p.numel() for p in model.parameters())
        
        print(f"✓ Enhanced Two-Tower: Memory={memory_used:.2f}GB, Params={params:,}, Output={output.shape}")
        
        del model
        torch.cuda.empty_cache()
        return True, memory_used, params
        
    except Exception as e:
        print(f"✗ Enhanced Two-Tower failed: {e}")
        return False, 0, 0

def test_model_2_neural_collaborative_filtering():
    """Test 2: Neural Collaborative Filtering"""
    print("\n=== Testing Neural Collaborative Filtering ===")
    
    if not models_available['ncf']:
        print("✗ Neural Collaborative Filtering models not available")
        return False, 0, 0
    
    try:
        torch.cuda.empty_cache()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        initial_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        
        model = NeuralCollaborativeFiltering(
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
            output = model(user_ids, item_ids)
        
        memory_used = (torch.cuda.memory_allocated() - initial_memory) / 1e9 if torch.cuda.is_available() else 0
        params = sum(p.numel() for p in model.parameters())
        
        print(f"✓ Neural Collaborative Filtering: Memory={memory_used:.2f}GB, Params={params:,}, Output={output.shape}")
        
        del model
        torch.cuda.empty_cache()
        return True, memory_used, params
        
    except Exception as e:
        print(f"✗ Neural Collaborative Filtering failed: {e}")
        return False, 0, 0

def test_model_3_sequential_recommender():
    """Test 3: Sequential Recommender"""
    print("\n=== Testing Sequential Recommender ===")
    
    if not models_available['sequential']:
        print("✗ Sequential Recommender models not available")
        return False, 0, 0
    
    try:
        torch.cuda.empty_cache()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        initial_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        
        model = SequentialRecommender(
            num_items=50000,
            embedding_dim=256,
            hidden_dim=256,
            num_layers=2,
            dropout=0.2,
            rnn_type='LSTM'
        ).to(device)
        
        # Test forward pass
        batch_size = 64
        seq_len = 50
        sequences = torch.randint(1, 50000, (batch_size, seq_len)).to(device)
        
        with torch.no_grad():
            # AttentionalSequentialRecommender doesn't need explicit mask
            output = model(sequences)
        
        memory_used = (torch.cuda.memory_allocated() - initial_memory) / 1e9 if torch.cuda.is_available() else 0
        params = sum(p.numel() for p in model.parameters())
        
        print(f"✓ Sequential Recommender: Memory={memory_used:.2f}GB, Params={params:,}, Output={output.shape}")
        
        del model
        torch.cuda.empty_cache()
        return True, memory_used, params
        
    except Exception as e:
        print(f"✗ Sequential Recommender failed: {e}")
        return False, 0, 0

def test_model_4_tv_recommender():
    """Test 4: TV Show Recommender"""
    print("\n=== Testing TV Show Recommender ===")
    
    if not models_available['tv_recommender']:
        print("✗ TV Show Recommender model not available")
        return False, 0, 0
    
    try:
        torch.cuda.empty_cache()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        initial_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        
        model = TVShowRecommenderModel(
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
        # TV features: [episode_count, season_count, duration, status (0-4)]
        tv_features = torch.stack([
            torch.randint(1, 100, (batch_size,)).float(),  # episode_count
            torch.randint(1, 10, (batch_size,)).float(),   # season_count
            torch.randint(20, 90, (batch_size,)).float(),  # duration
            torch.randint(0, 5, (batch_size,))             # status (stays as int)
        ], dim=1).to(device)
        
        with torch.no_grad():
            output = model(user_ids, show_ids, genre_features, tv_features)
        
        memory_used = (torch.cuda.memory_allocated() - initial_memory) / 1e9 if torch.cuda.is_available() else 0
        params = sum(p.numel() for p in model.parameters())
        
        print(f"✓ TV Show Recommender: Memory={memory_used:.2f}GB, Params={params:,}, Output={output.shape}")
        
        del model
        torch.cuda.empty_cache()
        return True, memory_used, params
        
    except Exception as e:
        print(f"✗ TV Show Recommender failed: {e}")
        return False, 0, 0

def test_concurrent_inference():
    """Test running smaller models concurrently"""
    print("\n=== Testing Concurrent Inference ===")
    
    if not (models_available['ncf'] and models_available['tv_recommender']):
        print("✗ Both NCF and TV Recommender models needed for concurrent test")
        return False
    
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        torch.cuda.empty_cache()
        initial_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        
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
        tv_features = torch.stack([
            torch.randint(1, 100, (batch_size,)).float(),  # episode_count
            torch.randint(1, 10, (batch_size,)).float(),   # season_count
            torch.randint(20, 90, (batch_size,)).float(),  # duration
            torch.randint(0, 5, (batch_size,))             # status (stays as int)
        ], dim=1).to(device)
        
        with torch.no_grad():
            ncf_output = model_ncf(user_ids, item_ids)
            tv_output = model_tv(user_ids, show_ids, genre_features, tv_features)
        
        memory_used = (torch.cuda.memory_allocated() - initial_memory) / 1e9 if torch.cuda.is_available() else 0
        print(f"✓ Concurrent inference successful - Memory: {memory_used:.2f} GB")
        
        del model_ncf, model_tv
        torch.cuda.empty_cache()
        return True
        
    except Exception as e:
        print(f"✗ Concurrent inference failed: {e}")
        return False

def test_model_5_two_tower():
    """Test 5: Two-Tower Model"""
    print("\n=== Testing Two-Tower Model ===")
    
    if not models_available['two_tower']:
        print("✗ Two-Tower models not available")
        return False, 0, 0
    
    try:
        torch.cuda.empty_cache()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        initial_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        
        model = TwoTowerModel(
            user_features_dim=100,
            item_features_dim=200,
            embedding_dim=128,
            hidden_layers=[256, 128],
            dropout=0.2
        ).to(device)
        
        # Test forward pass
        batch_size = 64
        user_features = torch.randn(batch_size, 100).to(device)
        item_features = torch.randn(batch_size, 200).to(device)
        
        with torch.no_grad():
            output = model(user_features, item_features)
        
        memory_used = (torch.cuda.memory_allocated() - initial_memory) / 1e9 if torch.cuda.is_available() else 0
        params = sum(p.numel() for p in model.parameters())
        
        print(f"✓ Two-Tower Model: Memory={memory_used:.2f}GB, Params={params:,}, Output={output.shape}")
        
        del model
        torch.cuda.empty_cache()
        return True, memory_used, params
        
    except Exception as e:
        print(f"✗ Two-Tower Model failed: {e}")
        return False, 0, 0

def test_model_6_hybrid_recommender():
    """Test 6: Original Hybrid Recommender Model"""
    print("\n=== Testing Original Hybrid Recommender Model ===")
    
    if not models_available['hybrid_recommender']:
        print("✗ Hybrid Recommender model not available")
        return False, 0, 0
    
    try:
        torch.cuda.empty_cache()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        initial_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        
        model = HybridRecommenderModel(
            num_users=10000,
            num_items=50000,
            embedding_dim=64,
            hidden_dim=128
        ).to(device)
        
        # Test forward pass
        batch_size = 128
        user_ids = torch.randint(0, 10000, (batch_size,)).to(device)
        item_ids = torch.randint(0, 50000, (batch_size,)).to(device)
        
        with torch.no_grad():
            output = model(user_ids, item_ids)
        
        memory_used = (torch.cuda.memory_allocated() - initial_memory) / 1e9 if torch.cuda.is_available() else 0
        params = sum(p.numel() for p in model.parameters())
        
        print(f"✓ Original Hybrid Recommender: Memory={memory_used:.2f}GB, Params={params:,}, Output={output.shape}")
        
        del model
        torch.cuda.empty_cache()
        return True, memory_used, params
        
    except Exception as e:
        print(f"✗ Original Hybrid Recommender failed: {e}")
        return False, 0, 0

def main():
    print("=== CineSync v2 Model Compatibility Test ===")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name()}")
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU Memory: {total_memory:.1f} GB")
    else:
        print("Running on CPU")
        total_memory = 0
    
    # Test all models
    results = []
    
    # Test 1: Enhanced Two-Tower
    success, memory, params = test_model_1_enhanced_two_tower()
    results.append(('Enhanced Two-Tower', success, memory, params))
    
    # Test 2: Neural Collaborative Filtering
    success, memory, params = test_model_2_neural_collaborative_filtering()
    results.append(('Neural Collaborative Filtering', success, memory, params))
    
    # Test 3: Sequential Recommender
    success, memory, params = test_model_3_sequential_recommender()
    results.append(('Sequential Recommender', success, memory, params))
    
    # Test 4: TV Recommender
    success, memory, params = test_model_4_tv_recommender()
    results.append(('TV Show Recommender', success, memory, params))
    
    # Test 5: Two-Tower Model (if available)
    if models_available['two_tower']:
        success, memory, params = test_model_5_two_tower()
        results.append(('Two-Tower Model', success, memory, params))
    
    # Test 6: Original Hybrid Recommender Model (if available)
    if models_available['hybrid_recommender']:
        success, memory, params = test_model_6_hybrid_recommender()
        results.append(('Original Hybrid Recommender', success, memory, params))
    
    # Test concurrent inference
    concurrent_success = test_concurrent_inference()
    
    # Summary
    print("\n=== SUMMARY ===")
    total_memory_used = 0
    successful_models = 0
    
    for model_name, success, memory, params in results:
        if success:
            successful_models += 1
            total_memory_used += memory
            print(f"✓ {model_name}: {memory:.2f} GB ({params:,} params)")
        else:
            print(f"✗ {model_name}: FAILED")
    
    print(f"\nTotal memory for all models: {total_memory_used:.2f} GB")
    total_models = len(results)
    print(f"Successful models: {successful_models}/{total_models}")
    
    if torch.cuda.is_available() and total_memory_used > 0:
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        if total_memory_used < gpu_memory * 0.8:  # 80% threshold
            print(f"✓ All models should fit on your GPU ({gpu_memory:.1f} GB)")
        else:
            print(f"⚠️  Models may not all fit simultaneously on your GPU ({gpu_memory:.1f} GB)")
            print("Consider training sequentially or using smaller batch sizes")
    
    if concurrent_success:
        print("✓ Concurrent inference test passed")
    else:
        print("✗ Concurrent inference test failed")
    
    if successful_models == total_models:
        print(f"\n🎉 All {total_models} models are working correctly!")
    else:
        print(f"\n⚠️  {total_models - successful_models} model(s) failed - check error messages above")

if __name__ == "__main__":
    main()