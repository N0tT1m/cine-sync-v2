#!/usr/bin/env python3
"""
Master training script for State-of-the-Art TV Models
Optimized for RTX 4090 + Ryzen 9 3900X

Usage:
    python train_sota_models.py --stage preprocessing
    python train_sota_models.py --stage multimodal
    python train_sota_models.py --stage gnn
    python train_sota_models.py --stage contrastive
    python train_sota_models.py --stage ensemble
    python train_sota_models.py --stage all
"""

import os
import sys
import argparse
import logging
import subprocess
import time
from pathlib import Path
import torch
import json

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def check_hardware():
    """Check hardware capabilities"""
    logger.info("Checking hardware...")
    
    # Check CUDA
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name()
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info(f"GPU: {gpu_name}")
        logger.info(f"GPU Memory: {gpu_memory:.1f} GB")
        
        if "4090" not in gpu_name:
            logger.warning("Optimal performance requires RTX 4090")
    else:
        logger.error("CUDA not available! GPU training required.")
        return False
    
    # Check CPU cores
    import multiprocessing
    cpu_count = multiprocessing.cpu_count()
    logger.info(f"CPU Cores: {cpu_count}")
    
    if cpu_count < 8:
        logger.warning("Recommended: 8+ CPU cores for optimal data loading")
    
    return True

def run_preprocessing(args):
    """Run data preprocessing"""
    logger.info("🔄 Starting data preprocessing...")
    
    script_path = Path(__file__).parent / "data" / "tv_preprocessor.py"
    
    cmd = [
        sys.executable, str(script_path),
        "--data_dir", args.data_dir,
        "--output_dir", args.processed_data_dir,
        "--min_shows_per_genre", str(args.min_shows_per_genre),
        "--max_genres_per_show", str(args.max_genres_per_show)
    ]
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        logger.info("✅ Data preprocessing completed successfully!")
        logger.info(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Data preprocessing failed: {e}")
        logger.error(e.stderr)
        return False

def run_multimodal_training(args):
    """Run multimodal transformer training"""
    logger.info("🚀 Starting Multimodal Transformer training...")
    
    script_path = Path(__file__).parent / "training" / "train_multimodal.py"
    
    cmd = [
        sys.executable, str(script_path),
        "--data_path", args.processed_data_dir,
        "--output_dir", args.models_dir,
        "--wandb_project", args.wandb_project,
        "--wandb_run_name", f"multimodal-transformer-{int(time.time())}"
    ]
    
    if args.config_file:
        cmd.extend(["--config_file", args.config_file])
    
    try:
        result = subprocess.run(cmd, check=True)
        logger.info("✅ Multimodal Transformer training completed!")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Multimodal training failed: {e}")
        return False

def run_gnn_training(args):
    """Run Graph Neural Network training"""
    logger.info("🕸️ Starting Graph Neural Network training...")
    
    script_path = Path(__file__).parent / "training" / "train_gnn.py"
    
    cmd = [
        sys.executable, str(script_path),
        "--data_path", args.processed_data_dir,
        "--output_dir", args.models_dir,
        "--wandb_project", args.wandb_project,
        "--wandb_run_name", f"gnn-tv-{int(time.time())}"
    ]
    
    if args.config_file:
        cmd.extend(["--config_file", args.config_file])
    
    try:
        result = subprocess.run(cmd, check=True)
        logger.info("✅ Graph Neural Network training completed!")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ GNN training failed: {e}")
        return False

def run_contrastive_training(args):
    """Run Contrastive Learning training"""
    logger.info("🔗 Starting Contrastive Learning training...")
    
    script_path = Path(__file__).parent / "training" / "train_contrastive.py"
    
    cmd = [
        sys.executable, str(script_path),
        "--data_path", args.processed_data_dir,
        "--output_dir", args.models_dir,
        "--wandb_project", args.wandb_project,
        "--wandb_run_name", f"contrastive-tv-{int(time.time())}"
    ]
    
    if args.config_file:
        cmd.extend(["--config_file", args.config_file])
    
    try:
        result = subprocess.run(cmd, check=True)
        logger.info("✅ Contrastive Learning training completed!")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Contrastive training failed: {e}")
        return False

def run_temporal_training(args):
    """Run Temporal Attention training"""
    logger.info("⏰ Starting Temporal Attention training...")
    
    script_path = Path(__file__).parent / "training" / "train_temporal.py"
    
    cmd = [
        sys.executable, str(script_path),
        "--data_path", args.processed_data_dir,
        "--output_dir", args.models_dir,
        "--wandb_project", args.wandb_project,
        "--wandb_run_name", f"temporal-tv-{int(time.time())}"
    ]
    
    if args.config_file:
        cmd.extend(["--config_file", args.config_file])
    
    try:
        result = subprocess.run(cmd, check=True)
        logger.info("✅ Temporal Attention training completed!")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Temporal training failed: {e}")
        return False

def run_meta_learning_training(args):
    """Run Meta-Learning training"""
    logger.info("🧠 Starting Meta-Learning training...")
    
    script_path = Path(__file__).parent / "training" / "train_meta.py"
    
    cmd = [
        sys.executable, str(script_path),
        "--data_path", args.processed_data_dir,
        "--output_dir", args.models_dir,
        "--wandb_project", args.wandb_project,
        "--wandb_run_name", f"meta-learning-tv-{int(time.time())}"
    ]
    
    if args.config_file:
        cmd.extend(["--config_file", args.config_file])
    
    try:
        result = subprocess.run(cmd, check=True)
        logger.info("✅ Meta-Learning training completed!")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Meta-learning training failed: {e}")
        return False

def run_ensemble_training(args):
    """Run Ensemble training"""
    logger.info("🎭 Starting Ensemble training...")
    
    script_path = Path(__file__).parent / "training" / "train_ensemble.py"
    
    cmd = [
        sys.executable, str(script_path),
        "--data_path", args.processed_data_dir,
        "--output_dir", args.models_dir,
        "--wandb_project", args.wandb_project,
        "--wandb_run_name", f"ensemble-tv-{int(time.time())}"
    ]
    
    if args.config_file:
        cmd.extend(["--config_file", args.config_file])
    
    try:
        result = subprocess.run(cmd, check=True)
        logger.info("✅ Ensemble training completed!")
        logger.info("🎉 All SOTA TV models are now ready for inference!")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Ensemble training failed: {e}")
        return False

def create_config_file(args):
    """Create optimized config for RTX 4090"""
    config = {
        "model_config": {
            "embed_dim": 768,
            "hidden_dim": 2048,  # Large for RTX 4090
            "num_layers": 8,
            "num_heads": 16,
            "dropout": 0.1,
            "use_gradient_checkpointing": True,
            "batch_size": 32,
            "accumulation_steps": 2,
            "learning_rate": 1e-4,
            "warmup_steps": 1000,
            "max_grad_norm": 1.0,
            "use_mixed_precision": True
        },
        "training_config": {
            "epochs": 50,
            "patience": 10,
            "weight_decay": 1e-5,
            "contrastive_weight": 1.0,
            "recommendation_weight": 1.0
        },
        "hardware_config": {
            "device": "cuda",
            "num_workers": min(12, os.cpu_count()),  # Use all cores up to 12
            "pin_memory": True,
            "persistent_workers": True
        }
    }
    
    config_path = Path(args.output_dir) / "rtx4090_config.json"
    config_path.parent.mkdir(exist_ok=True)
    
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    logger.info(f"💾 Config saved to {config_path}")
    return config_path

def estimate_training_time():
    """Estimate total training time"""
    estimates = {
        "preprocessing": "5-10 minutes",
        "multimodal": "2-3 hours",
        "gnn": "1-2 hours", 
        "contrastive": "1-2 hours",
        "temporal": "1-2 hours",
        "meta": "2-3 hours",
        "ensemble": "30-60 minutes"
    }
    
    logger.info("⏱️ Estimated Training Times (RTX 4090):")
    total_min = 0
    for stage, time_est in estimates.items():
        logger.info(f"  {stage.capitalize()}: {time_est}")
        # Extract minimum time for rough total
        if "minutes" in time_est:
            total_min += int(time_est.split("-")[0].split()[0])
        elif "hours" in time_est:
            total_min += int(time_est.split("-")[0].split()[0]) * 60
    
    logger.info(f"  Total (minimum): ~{total_min // 60}h {total_min % 60}m")

def main():
    parser = argparse.ArgumentParser(description='Train State-of-the-Art TV Models')
    
    # Stage selection
    parser.add_argument('--stage', type=str, required=True,
                       choices=['preprocessing', 'multimodal', 'gnn', 'contrastive', 'temporal', 'meta', 'ensemble', 'all'],
                       help='Training stage to run')
    
    # Paths
    parser.add_argument('--data_dir', type=str, 
                       default='/Users/timmy/workspace/ai-apps/cine-sync-v2/tv',
                       help='Directory containing raw TV datasets')
    parser.add_argument('--output_dir', type=str, 
                       default='./sota_tv_outputs',
                       help='Output directory for all results')
    parser.add_argument('--config_file', type=str,
                       help='Custom config file path')
    
    # Preprocessing options
    parser.add_argument('--min_shows_per_genre', type=int, default=10,
                       help='Minimum shows per genre to include')
    parser.add_argument('--max_genres_per_show', type=int, default=5,
                       help='Maximum genres per show')
    
    # Wandb options
    parser.add_argument('--wandb_project', type=str, default='sota-tv-models',
                       help='Weights & Biases project name')
    parser.add_argument('--no_wandb', action='store_true',
                       help='Disable Weights & Biases logging')
    
    # Hardware options
    parser.add_argument('--force_cpu', action='store_true',
                       help='Force CPU training (not recommended)')
    
    args = parser.parse_args()
    
    # Setup directories
    args.output_dir = Path(args.output_dir)
    args.output_dir.mkdir(exist_ok=True)
    
    args.processed_data_dir = args.output_dir / "processed_data"
    args.models_dir = args.output_dir / "models"
    args.logs_dir = args.output_dir / "logs"
    
    for dir_path in [args.processed_data_dir, args.models_dir, args.logs_dir]:
        dir_path.mkdir(exist_ok=True)
    
    # Check hardware
    if not args.force_cpu and not check_hardware():
        logger.error("❌ Hardware check failed. Use --force_cpu to proceed anyway.")
        return 1
    
    # Create optimized config
    if not args.config_file:
        args.config_file = create_config_file(args)
    
    # Show training time estimates
    if args.stage == 'all':
        estimate_training_time()
    
    # Setup wandb
    if not args.no_wandb:
        try:
            import wandb
            logger.info("📊 Weights & Biases available for logging")
        except ImportError:
            logger.warning("⚠️ Weights & Biases not installed. Install with: pip install wandb")
            args.no_wandb = True
    
    # Run training stages
    success = True
    start_time = time.time()
    
    if args.stage == 'all':
        stages = ['preprocessing', 'multimodal', 'gnn', 'contrastive', 'temporal', 'meta', 'ensemble']
    else:
        stages = [args.stage]
    
    for stage in stages:
        stage_start = time.time()
        logger.info(f"\n{'='*50}")
        logger.info(f"🎯 Running stage: {stage.upper()}")
        logger.info(f"{'='*50}")
        
        if stage == 'preprocessing':
            stage_success = run_preprocessing(args)
        elif stage == 'multimodal':
            stage_success = run_multimodal_training(args)
        elif stage == 'gnn':
            stage_success = run_gnn_training(args)
        elif stage == 'contrastive':
            stage_success = run_contrastive_training(args)
        elif stage == 'temporal':
            stage_success = run_temporal_training(args)
        elif stage == 'meta':
            stage_success = run_meta_learning_training(args)
        elif stage == 'ensemble':
            stage_success = run_ensemble_training(args)
        
        stage_time = time.time() - stage_start
        
        if stage_success:
            logger.info(f"✅ {stage.capitalize()} completed in {stage_time/60:.1f} minutes")
        else:
            logger.error(f"❌ {stage.capitalize()} failed after {stage_time/60:.1f} minutes")
            success = False
            break
    
    total_time = time.time() - start_time
    
    if success:
        logger.info(f"\n🎉 ALL STAGES COMPLETED SUCCESSFULLY!")
        logger.info(f"⏱️ Total time: {total_time/3600:.1f} hours")
        logger.info(f"📁 Results saved to: {args.output_dir}")
        logger.info(f"\n🚀 Your state-of-the-art TV models are ready!")
        logger.info(f"💡 Next steps:")
        logger.info(f"   1. Test inference with the ensemble model")
        logger.info(f"   2. Integrate into your hybrid TV system") 
        logger.info(f"   3. Monitor performance with real users")
    else:
        logger.error(f"\n💥 Training failed after {total_time/60:.1f} minutes")
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)