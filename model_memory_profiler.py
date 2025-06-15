#!/usr/bin/env python3
"""
Memory profiler for CineSync models - calculates theoretical memory usage
"""
import sys
import os
from typing import Dict, List, Tuple
import math

def calculate_parameter_count(model_config: Dict) -> int:
    """Calculate total parameters for a model configuration"""
    return model_config.get('total_params', 0)

def calculate_memory_usage(params: int, dtype_bytes: int = 4, batch_size: int = 32, 
                         sequence_length: int = 100, embedding_dim: int = 128) -> Dict[str, float]:
    """
    Calculate memory usage for a model
    
    Args:
        params: Number of model parameters
        dtype_bytes: Bytes per parameter (4 for float32, 2 for float16)
        batch_size: Training batch size
        sequence_length: For sequential models
        embedding_dim: Embedding dimension
    
    Returns:
        Dictionary with memory usage breakdown in GB
    """
    # Model parameters memory
    params_memory = params * dtype_bytes
    
    # Gradients memory (same as parameters)
    gradients_memory = params_memory
    
    # Optimizer states (AdamW has 2 states per parameter)
    optimizer_memory = params * dtype_bytes * 2
    
    # Activations memory (rough estimate)
    # This varies greatly based on architecture
    activations_memory = batch_size * embedding_dim * sequence_length * dtype_bytes * 10
    
    # Total memory
    total_memory = params_memory + gradients_memory + optimizer_memory + activations_memory
    
    return {
        'parameters_gb': params_memory / 1e9,
        'gradients_gb': gradients_memory / 1e9,
        'optimizer_gb': optimizer_memory / 1e9,
        'activations_gb': activations_memory / 1e9,
        'total_gb': total_memory / 1e9
    }

def get_model_configurations() -> Dict[str, Dict]:
    """Get configurations for all 4 models"""
    
    configs = {
        'Enhanced Two-Tower': {
            'description': 'UltimateTwoTowerModel with cross-attention and multi-task learning',
            'embedding_dims': {
                'user_categorical': {'age_group': 10, 'gender': 3, 'occupation': 20},
                'item_categorical': {'genre': 20, 'year': 100, 'director': 1000},
                'collaborative': {'users': 10000, 'items': 50000}
            },
            'architecture': {
                'embedding_dim': 512,
                'tower_hidden_dims': [1024, 512, 256],
                'cross_attention_heads': 16,
                'num_tasks': 3
            },
            'estimated_params': 0,
            'batch_size': 32,
            'dtype_bytes': 4
        },
        
        'Neural Collaborative Filtering': {
            'description': 'NCF with GMF and MLP components',
            'embedding_dims': {
                'users': 10000,
                'items': 50000
            },
            'architecture': {
                'embedding_dim': 128,
                'hidden_layers': [256, 128, 64]
            },
            'estimated_params': 0,
            'batch_size': 128,
            'dtype_bytes': 4
        },
        
        'Sequential Recommender': {
            'description': 'AttentionalSequentialRecommender with transformer blocks',
            'embedding_dims': {
                'items': 50000,
                'positions': 100
            },
            'architecture': {
                'embedding_dim': 256,
                'num_heads': 8,
                'num_blocks': 4,
                'max_seq_len': 100
            },
            'estimated_params': 0,
            'batch_size': 64,
            'sequence_length': 50,
            'dtype_bytes': 4
        },
        
        'Hybrid TV Recommender': {
            'description': 'TV show recommender with content and collaborative features',
            'embedding_dims': {
                'users': 10000,
                'shows': 25000,
                'status': 5
            },
            'architecture': {
                'embedding_dim': 128,
                'hidden_dim': 256,
                'num_genres': 20
            },
            'estimated_params': 0,
            'batch_size': 128,
            'dtype_bytes': 4
        }
    }
    
    # Calculate parameter counts for each model
    configs = calculate_all_parameters(configs)
    
    return configs

def calculate_all_parameters(configs: Dict[str, Dict]) -> Dict[str, Dict]:
    """Calculate parameter counts for all model configurations"""
    
    # Enhanced Two-Tower Model
    config = configs['Enhanced Two-Tower']
    params = 0
    
    # User categorical embeddings
    for feature, vocab_size in config['embedding_dims']['user_categorical'].items():
        emb_dim = min(128, (vocab_size + 1) // 2)
        params += vocab_size * emb_dim
    
    # Item categorical embeddings  
    for feature, vocab_size in config['embedding_dims']['item_categorical'].items():
        emb_dim = min(128, (vocab_size + 1) // 2)
        params += vocab_size * emb_dim
    
    # Collaborative embeddings
    params += config['embedding_dims']['collaborative']['users'] * (config['architecture']['embedding_dim'] // 2)
    params += config['embedding_dims']['collaborative']['items'] * (config['architecture']['embedding_dim'] // 2)
    
    # Tower layers
    tower_dims = config['architecture']['tower_hidden_dims']
    prev_dim = config['architecture']['embedding_dim']
    for hidden_dim in tower_dims:
        params += prev_dim * hidden_dim + hidden_dim  # weights + bias
        prev_dim = hidden_dim
    params += prev_dim * config['architecture']['embedding_dim'] + config['architecture']['embedding_dim']
    params *= 2  # Two towers
    
    # Cross-attention layers
    emb_dim = config['architecture']['embedding_dim']
    heads = config['architecture']['cross_attention_heads']
    params += 4 * emb_dim * emb_dim * 2  # Q, K, V, O projections for both towers
    
    # Multi-task heads
    params += config['architecture']['num_tasks'] * (emb_dim * 2 * emb_dim + emb_dim + emb_dim * 1 + 1)
    
    config['estimated_params'] = params
    
    # Neural Collaborative Filtering
    config = configs['Neural Collaborative Filtering']
    params = 0
    
    # Embeddings (GMF + MLP)
    emb_dim = config['architecture']['embedding_dim']
    params += config['embedding_dims']['users'] * emb_dim * 2  # GMF + MLP user embeddings
    params += config['embedding_dims']['items'] * emb_dim * 2  # GMF + MLP item embeddings
    
    # MLP layers
    hidden_layers = config['architecture']['hidden_layers']
    prev_dim = emb_dim * 2  # Concatenated user + item
    for hidden_dim in hidden_layers:
        params += prev_dim * hidden_dim + hidden_dim
        prev_dim = hidden_dim
    
    # Final prediction layer
    params += (emb_dim + hidden_layers[-1]) * 1 + 1
    
    config['estimated_params'] = params
    
    # Sequential Recommender
    config = configs['Sequential Recommender']
    params = 0
    
    # Embeddings
    emb_dim = config['architecture']['embedding_dim']
    params += config['embedding_dims']['items'] * emb_dim
    params += config['embedding_dims']['positions'] * emb_dim
    
    # Attention blocks
    num_blocks = config['architecture']['num_blocks']
    heads = config['architecture']['num_heads']
    for _ in range(num_blocks):
        # Multi-head attention
        params += 4 * emb_dim * emb_dim  # Q, K, V, O
        # Feed-forward
        params += emb_dim * (emb_dim * 4) + (emb_dim * 4)  # First layer
        params += (emb_dim * 4) * emb_dim + emb_dim  # Second layer
        # Layer norms
        params += emb_dim * 2 * 2  # Two layer norms per block
    
    # Output layer
    params += emb_dim * config['embedding_dims']['items'] + config['embedding_dims']['items']
    
    config['estimated_params'] = params
    
    # Hybrid TV Recommender
    config = configs['Hybrid TV Recommender']
    params = 0
    
    # Embeddings
    emb_dim = config['architecture']['embedding_dim']
    hidden_dim = config['architecture']['hidden_dim']
    params += config['embedding_dims']['users'] * emb_dim
    params += config['embedding_dims']['shows'] * emb_dim
    params += config['embedding_dims']['status'] * (emb_dim // 4)
    
    # Genre linear layer
    params += config['architecture']['num_genres'] * emb_dim + emb_dim
    
    # TV feature linear layers
    params += 1 * (emb_dim // 4) + (emb_dim // 4)  # episode_count
    params += 1 * (emb_dim // 4) + (emb_dim // 4)  # season_count  
    params += 1 * (emb_dim // 4) + (emb_dim // 4)  # duration
    
    # Neural network layers
    combined_dim = emb_dim * 3 + emb_dim  # user + show + genre + tv_features
    params += combined_dim * hidden_dim + hidden_dim
    params += hidden_dim * (hidden_dim // 2) + (hidden_dim // 2)
    params += (hidden_dim // 2) * (hidden_dim // 4) + (hidden_dim // 4)
    params += (hidden_dim // 4) * 1 + 1
    
    # Batch norm parameters
    params += hidden_dim * 2  # BatchNorm1d
    params += (hidden_dim // 2) * 2  # BatchNorm1d
    
    config['estimated_params'] = params
    
    return configs

def analyze_rtx4090_compatibility(configs: Dict[str, Dict]) -> Dict[str, Dict]:
    """Analyze compatibility with RTX 4090 (24GB VRAM)"""
    
    rtx4090_memory = 24.0  # GB
    results = {}
    
    for model_name, config in configs.items():
        # Calculate memory usage
        memory_usage = calculate_memory_usage(
            params=config['estimated_params'],
            dtype_bytes=config['dtype_bytes'],
            batch_size=config['batch_size'],
            sequence_length=config.get('sequence_length', 100),
            embedding_dim=config['architecture'].get('embedding_dim', 128)
        )
        
        # Determine compatibility
        total_memory = memory_usage['total_gb']
        can_fit = total_memory < rtx4090_memory * 0.9  # 90% threshold for safety
        
        # Recommendations
        recommendations = []
        if not can_fit:
            recommendations.append("Reduce batch size")
            recommendations.append("Use mixed precision (FP16)")
            recommendations.append("Enable gradient checkpointing")
        
        if total_memory > rtx4090_memory * 0.7:  # 70% threshold
            recommendations.append("Consider smaller embedding dimensions")
            recommendations.append("Use gradient accumulation")
        
        results[model_name] = {
            'memory_usage': memory_usage,
            'can_fit': can_fit,
            'memory_efficiency': (total_memory / rtx4090_memory) * 100,
            'recommendations': recommendations,
            'config': config
        }
    
    return results

def generate_training_commands(configs: Dict[str, Dict]) -> Dict[str, List[str]]:
    """Generate training commands for each model"""
    
    commands = {
        'Enhanced Two-Tower': [
            "cd advanced_models/",
            "python train_enhanced_two_tower.py \\",
            "    --embedding_dim 512 \\",
            "    --batch_size 16 \\",  # Reduced for memory
            "    --learning_rate 1e-4 \\",
            "    --epochs 50 \\",
            "    --use_mixed_precision \\",
            "    --gradient_checkpointing \\",
            "    --data_path ../data/processed/ \\",
            "    --output_dir ./models/"
        ],
        
        'Neural Collaborative Filtering': [
            "cd neural_collaborative_filtering/",
            "python src/train.py \\",
            "    --embedding_dim 128 \\",
            "    --batch_size 128 \\",
            "    --learning_rate 1e-3 \\",
            "    --epochs 100 \\",
            "    --hidden_layers 256 128 64 \\",
            "    --data_path ../data/processed/ \\",
            "    --output_dir ./models/"
        ],
        
        'Sequential Recommender': [
            "cd sequential_models/",
            "python src/train.py \\",
            "    --model_type attentional \\",
            "    --embedding_dim 256 \\",
            "    --batch_size 64 \\",
            "    --learning_rate 1e-4 \\",
            "    --epochs 50 \\",
            "    --num_blocks 4 \\",
            "    --num_heads 8 \\",
            "    --max_seq_len 100 \\",
            "    --data_path ../data/processed/ \\",
            "    --output_dir ./models/"
        ],
        
        'Hybrid TV Recommender': [
            "cd hybrid_recommendation/",
            "python train_tv_shows.py \\",
            "    --embedding_dim 128 \\",
            "    --hidden_dim 256 \\",
            "    --batch_size 128 \\",
            "    --learning_rate 1e-3 \\",
            "    --epochs 100 \\",
            "    --data_path ./data/tv/ \\",
            "    --output_dir ./models/"
        ]
    }
    
    return commands

def main():
    print("=== CineSync Model Memory Analysis ===")
    print("Analyzing 4 recommendation models for RTX 4090 compatibility\n")
    
    # Get model configurations
    configs = get_model_configurations()
    
    # Analyze RTX 4090 compatibility
    analysis = analyze_rtx4090_compatibility(configs)
    
    # Generate training commands
    training_commands = generate_training_commands(configs)
    
    # Print results
    total_memory_all = 0
    fits_all = True
    
    for model_name, result in analysis.items():
        config = result['config']
        memory = result['memory_usage']
        
        print(f"## {model_name}")
        print(f"Description: {config['description']}")
        print(f"Parameters: {config['estimated_params']:,}")
        print(f"Memory Breakdown:")
        print(f"  - Parameters: {memory['parameters_gb']:.2f} GB")
        print(f"  - Gradients: {memory['gradients_gb']:.2f} GB")
        print(f"  - Optimizer: {memory['optimizer_gb']:.2f} GB")
        print(f"  - Activations: {memory['activations_gb']:.2f} GB")
        print(f"  - Total: {memory['total_gb']:.2f} GB")
        print(f"RTX 4090 Compatibility: {'✓ YES' if result['can_fit'] else '✗ NO'}")
        print(f"Memory Usage: {result['memory_efficiency']:.1f}% of 24GB")
        
        if result['recommendations']:
            print("Recommendations:")
            for rec in result['recommendations']:
                print(f"  - {rec}")
        
        print(f"Training Command:")
        for cmd in training_commands[model_name]:
            print(f"  {cmd}")
        print()
        
        total_memory_all += memory['total_gb']
        if not result['can_fit']:
            fits_all = False
    
    # Summary
    print("=== SUMMARY ===")
    print(f"Total memory if all models trained simultaneously: {total_memory_all:.2f} GB")
    print(f"RTX 4090 can fit all models individually: {'✓ YES' if fits_all else '✗ NO'}")
    print(f"RTX 4090 can fit all models simultaneously: {'✓ YES' if total_memory_all < 24 else '✗ NO'}")
    
    print("\n=== RECOMMENDATIONS ===")
    print("1. Train models sequentially for best results")
    print("2. Use mixed precision (FP16) to reduce memory by ~50%")
    print("3. Enable gradient checkpointing for large models")
    print("4. Start with smaller models (NCF, TV Recommender) to validate pipeline")
    print("5. Use gradient accumulation to simulate larger batch sizes")
    
    return analysis

if __name__ == "__main__":
    main()