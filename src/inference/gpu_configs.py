"""
GPU Configuration Profiles for CineSync v2 Inference Server

Two optimized profiles:
- RTX 4090 (24GB VRAM)
- RTX 5090 (32GB VRAM)

Usage:
    from gpu_configs import get_config
    config = get_config('rtx4090')  # or 'rtx5090'
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
from enum import Enum


class PrecisionMode(Enum):
    FP32 = "fp32"
    FP16 = "fp16"
    BF16 = "bf16"
    INT8 = "int8"


@dataclass
class ModelLoadConfig:
    """Configuration for how a model should be loaded"""
    precision: PrecisionMode = PrecisionMode.FP32
    compile_model: bool = False  # torch.compile() for PyTorch 2.0+
    pin_memory: bool = True
    preload: bool = True  # Load at startup vs lazy load


@dataclass
class GPUConfig:
    """Complete GPU configuration profile"""
    name: str
    vram_gb: int

    # Precision settings
    default_precision: PrecisionMode = PrecisionMode.FP32
    embedding_precision: PrecisionMode = PrecisionMode.FP32
    transformer_precision: PrecisionMode = PrecisionMode.FP16  # Transformers benefit from FP16

    # Memory management
    share_embeddings: bool = True
    max_batch_size: int = 64
    reserved_vram_gb: float = 1.0  # Reserve for system/overhead

    # Model loading
    preload_all: bool = True
    lazy_load_threshold_mb: int = 500  # Lazy load models larger than this

    # Optimization flags
    use_torch_compile: bool = False
    use_cuda_graphs: bool = False
    use_flash_attention: bool = True
    pin_memory: bool = True

    # Model-specific overrides
    model_overrides: Dict[str, ModelLoadConfig] = field(default_factory=dict)

    # Which models to load (None = all)
    enabled_models: Optional[List[str]] = None
    disabled_models: List[str] = field(default_factory=list)


# =============================================================================
# RTX 4090 Configuration (24GB VRAM)
# =============================================================================
RTX_4090_CONFIG = GPUConfig(
    name="RTX 4090",
    vram_gb=24,

    # Precision - FP32 for most, FP16 for large transformers to fit comfortably
    default_precision=PrecisionMode.FP32,
    embedding_precision=PrecisionMode.FP32,
    transformer_precision=PrecisionMode.FP16,

    # Memory management
    share_embeddings=True,
    max_batch_size=64,
    reserved_vram_gb=1.5,  # Conservative reserve

    # Model loading
    preload_all=True,
    lazy_load_threshold_mb=800,  # Lazy load very large models

    # Optimizations
    use_torch_compile=True,  # 4090 benefits from torch.compile
    use_cuda_graphs=True,    # Good for repeated inference
    use_flash_attention=True,
    pin_memory=True,

    # Model-specific overrides for memory optimization
    model_overrides={
        # Large transformer models use FP16
        't5_hybrid': ModelLoadConfig(
            precision=PrecisionMode.FP16,
            compile_model=True,
            preload=True
        ),
        'sentence_bert_two_tower': ModelLoadConfig(
            precision=PrecisionMode.FP16,
            compile_model=True,
            preload=True
        ),
        'bert4rec': ModelLoadConfig(
            precision=PrecisionMode.FP16,
            compile_model=True,
            preload=True
        ),
        'multimodal_features': ModelLoadConfig(
            precision=PrecisionMode.FP16,
            compile_model=False,  # Vision models can be finicky with compile
            preload=True
        ),
        'tv_multimodal': ModelLoadConfig(
            precision=PrecisionMode.FP16,
            compile_model=True,
            preload=True
        ),
        # Ensemble models lazy load since they reference others
        'tv_ensemble': ModelLoadConfig(
            precision=PrecisionMode.FP32,
            preload=False
        ),
        'movie_ensemble': ModelLoadConfig(
            precision=PrecisionMode.FP32,
            preload=False
        ),
        'context_aware': ModelLoadConfig(
            precision=PrecisionMode.FP32,
            preload=False
        ),
    },

    disabled_models=[]  # All models enabled
)


# =============================================================================
# RTX 5090 Configuration (32GB VRAM)
# =============================================================================
RTX_5090_CONFIG = GPUConfig(
    name="RTX 5090",
    vram_gb=32,

    # Precision - Full FP32 everywhere, we have the VRAM
    default_precision=PrecisionMode.FP32,
    embedding_precision=PrecisionMode.FP32,
    transformer_precision=PrecisionMode.FP32,  # Can afford FP32 transformers

    # Memory management
    share_embeddings=True,
    max_batch_size=128,  # Larger batches for better throughput
    reserved_vram_gb=2.0,  # More headroom

    # Model loading
    preload_all=True,
    lazy_load_threshold_mb=2000,  # Only lazy load huge models

    # Optimizations - enable everything
    use_torch_compile=True,
    use_cuda_graphs=True,
    use_flash_attention=True,
    pin_memory=True,

    # Model-specific overrides - mostly just enable compile for speed
    model_overrides={
        't5_hybrid': ModelLoadConfig(
            precision=PrecisionMode.FP32,  # Full precision
            compile_model=True,
            preload=True
        ),
        'sentence_bert_two_tower': ModelLoadConfig(
            precision=PrecisionMode.FP32,
            compile_model=True,
            preload=True
        ),
        'bert4rec': ModelLoadConfig(
            precision=PrecisionMode.FP32,
            compile_model=True,
            preload=True
        ),
        'multimodal_features': ModelLoadConfig(
            precision=PrecisionMode.FP32,
            compile_model=True,  # 5090 should handle this
            preload=True
        ),
        'tv_multimodal': ModelLoadConfig(
            precision=PrecisionMode.FP32,
            compile_model=True,
            preload=True
        ),
        # Even ensemble models can preload
        'tv_ensemble': ModelLoadConfig(
            precision=PrecisionMode.FP32,
            preload=True
        ),
        'movie_ensemble': ModelLoadConfig(
            precision=PrecisionMode.FP32,
            preload=True
        ),
        'context_aware': ModelLoadConfig(
            precision=PrecisionMode.FP32,
            preload=True
        ),
    },

    disabled_models=[]  # All models enabled
)


# =============================================================================
# Configuration Registry
# =============================================================================
GPU_CONFIGS = {
    'rtx4090': RTX_4090_CONFIG,
    'rtx5090': RTX_5090_CONFIG,
    '4090': RTX_4090_CONFIG,
    '5090': RTX_5090_CONFIG,
}


def get_config(gpu_name: str) -> GPUConfig:
    """
    Get configuration for a specific GPU.

    Args:
        gpu_name: One of 'rtx4090', 'rtx5090', '4090', '5090'

    Returns:
        GPUConfig for the specified GPU

    Raises:
        ValueError: If GPU name not recognized
    """
    key = gpu_name.lower().replace(' ', '').replace('-', '')
    if key not in GPU_CONFIGS:
        available = ', '.join(GPU_CONFIGS.keys())
        raise ValueError(f"Unknown GPU: {gpu_name}. Available: {available}")
    return GPU_CONFIGS[key]


def estimate_vram_usage(config: GPUConfig) -> Dict[str, float]:
    """
    Estimate VRAM usage for a configuration.

    Returns:
        Dictionary with estimated VRAM breakdown in GB
    """
    # Base estimates (MB)
    EMBEDDING_SIZE_MB = 293  # Shared user/item embeddings

    MODEL_SIZES_MB = {
        # Movie-specific
        'movie_franchise_sequence': 150,
        'movie_director_auteur': 120,
        'movie_cinematic_universe': 200,
        'movie_awards_prediction': 100,
        'movie_runtime_preference': 80,
        'movie_era_style': 100,
        'movie_critic_audience': 150,
        'movie_remake_connection': 100,
        'movie_actor_collaboration': 180,
        'movie_studio_fingerprint': 100,
        'movie_adaptation_source': 100,
        'movie_international': 120,
        'movie_narrative_complexity': 150,
        'movie_viewing_context': 200,
        # TV-specific
        'tv_temporal_attention': 300,
        'tv_graph_neural': 250,
        'tv_contrastive': 200,
        'tv_meta_learning': 250,
        'tv_ensemble': 100,
        'tv_multimodal': 400,
        'tv_episode_sequence': 200,
        'tv_binge_prediction': 150,
        'tv_series_completion': 150,
        'tv_season_quality': 120,
        'tv_platform_availability': 100,
        'tv_watch_pattern': 150,
        'tv_series_lifecycle': 120,
        'tv_cast_migration': 150,
        # Content-agnostic
        'ncf': 100,
        'sequential_recommender': 150,
        'two_tower': 200,
        'bert4rec': 500,
        'graphsage': 300,
        'transformer_recommender': 400,
        'vae_recommender': 200,
        'gnn_recommender': 250,
        'enhanced_two_tower': 350,
        'sentence_bert_two_tower': 600,
        't5_hybrid': 1500,
        'unified_content': 200,
        # Unified
        'cross_domain_embeddings': 200,
        'movie_ensemble': 100,
        'unified_contrastive': 250,
        'multimodal_features': 800,
        'context_aware': 150,
    }

    # Calculate precision multiplier
    def precision_multiplier(precision: PrecisionMode) -> float:
        return {
            PrecisionMode.FP32: 1.0,
            PrecisionMode.FP16: 0.5,
            PrecisionMode.BF16: 0.5,
            PrecisionMode.INT8: 0.25,
        }[precision]

    # Calculate total
    total_mb = 0.0

    # Embeddings
    emb_mult = precision_multiplier(config.embedding_precision)
    embedding_mb = EMBEDDING_SIZE_MB * emb_mult if config.share_embeddings else EMBEDDING_SIZE_MB * 46 * emb_mult
    total_mb += embedding_mb

    # Model weights
    model_mb = 0.0
    for model_name, size_mb in MODEL_SIZES_MB.items():
        if config.enabled_models and model_name not in config.enabled_models:
            continue
        if model_name in config.disabled_models:
            continue

        # Get precision for this model
        if model_name in config.model_overrides:
            precision = config.model_overrides[model_name].precision
        elif 'bert' in model_name or 't5' in model_name or 'transformer' in model_name:
            precision = config.transformer_precision
        else:
            precision = config.default_precision

        mult = precision_multiplier(precision)
        model_mb += size_mb * mult

    total_mb += model_mb

    # Overhead (15%)
    overhead_mb = total_mb * 0.15
    total_mb += overhead_mb

    # Reserved
    reserved_mb = config.reserved_vram_gb * 1024

    return {
        'embeddings_gb': embedding_mb / 1024,
        'models_gb': model_mb / 1024,
        'overhead_gb': overhead_mb / 1024,
        'reserved_gb': config.reserved_vram_gb,
        'total_gb': total_mb / 1024,
        'available_gb': config.vram_gb,
        'headroom_gb': config.vram_gb - (total_mb / 1024) - config.reserved_vram_gb,
    }


def print_config_summary(config: GPUConfig):
    """Print a summary of the GPU configuration"""
    usage = estimate_vram_usage(config)

    print(f"\n{'=' * 60}")
    print(f"  {config.name} Configuration Summary")
    print(f"{'=' * 60}")
    print(f"\n  VRAM: {config.vram_gb} GB")
    print(f"  Default Precision: {config.default_precision.value}")
    print(f"  Transformer Precision: {config.transformer_precision.value}")
    print(f"  Max Batch Size: {config.max_batch_size}")
    print(f"\n  Optimizations:")
    print(f"    - Shared Embeddings: {config.share_embeddings}")
    print(f"    - torch.compile: {config.use_torch_compile}")
    print(f"    - CUDA Graphs: {config.use_cuda_graphs}")
    print(f"    - Flash Attention: {config.use_flash_attention}")
    print(f"\n  Estimated VRAM Usage:")
    print(f"    - Embeddings: {usage['embeddings_gb']:.2f} GB")
    print(f"    - Models: {usage['models_gb']:.2f} GB")
    print(f"    - Overhead: {usage['overhead_gb']:.2f} GB")
    print(f"    - Reserved: {usage['reserved_gb']:.2f} GB")
    print(f"    - Total: {usage['total_gb']:.2f} GB")
    print(f"    - Headroom: {usage['headroom_gb']:.2f} GB")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    # Print summaries for both configs
    print_config_summary(RTX_4090_CONFIG)
    print_config_summary(RTX_5090_CONFIG)
