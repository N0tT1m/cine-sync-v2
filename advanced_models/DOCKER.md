# Docker Training Setup for CineSync v2

This Docker setup provides a consistent environment for training advanced recommendation models with all dependencies pre-installed, including the problematic `torch-sparse` package.

## Quick Start

1. **Build the Docker image:**
   ```bash
   ./run_training.sh build
   ```

2. **Run training:**
   ```bash
   # List available models and options
   ./run_training.sh train --help
   
   # Train a specific model
   ./run_training.sh train --model sentence-bert --epochs 50 --batch-size 64
   ```

3. **Start Jupyter notebook (optional):**
   ```bash
   ./run_training.sh jupyter
   # Access at http://localhost:8888
   ```

## Available Commands

### Training Commands
```bash
# Build Docker image
./run_training.sh build

# Run training with specific model
./run_training.sh train --model MODEL_NAME [OPTIONS]

# Examples:
./run_training.sh train --model bert4rec --epochs 100
./run_training.sh train --model sentence-bert --epochs 50 --lr 0.001
./run_training.sh train --model graphsage --batch-size 128
```

### Development Commands
```bash
# Open interactive shell
./run_training.sh shell

# Start Jupyter notebook
./run_training.sh jupyter

# Start TensorBoard monitoring
./run_training.sh tensorboard
```

### Maintenance Commands
```bash
# Clean up Docker resources
./run_training.sh clean

# Show help
./run_training.sh help
```

## GPU Support

The Docker setup automatically detects and uses NVIDIA GPUs if available. Requirements:
- NVIDIA Docker runtime installed
- NVIDIA drivers installed on host
- RTX 4090 optimizations are pre-configured

## Volume Mounts

The container mounts the following directories:
- `.` → `/app` (source code)
- `../data` → `/app/data` (training data, read-only)
- `./logs` → `/app/logs` (training logs)
- `./checkpoints` → `/app/checkpoints` (model checkpoints)
- `./outputs` → `/app/outputs` (training outputs)

## Environment Variables

Pre-configured for optimal performance:
- `CUDA_VISIBLE_DEVICES=0`
- `TORCH_CUDA_ARCH_LIST=8.6` (RTX 4090 optimized)
- `PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512`

## Troubleshooting

### Docker Issues
```bash
# Check Docker status
docker info

# Check NVIDIA Docker support
docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi
```

### Memory Issues
If you encounter CUDA out of memory errors:
1. Reduce batch size: `--batch-size 32`
2. Enable gradient checkpointing in model config
3. Use mixed precision training

### Package Issues
The Docker image includes all dependencies with compatible versions. If you need additional packages:
```bash
# Open shell and install
./run_training.sh shell
pip install your-package
```

## Manual Docker Commands

If you prefer using Docker directly:

```bash
# Build
docker-compose build training

# Run training
docker-compose run --rm training python train_advanced_models.py --model sentence-bert

# Start services
docker-compose up -d tensorboard
docker-compose --profile jupyter up jupyter
```

## Performance Tips

1. **Pre-download models:** Run once to cache HuggingFace models:
   ```bash
   ./run_training.sh shell
   python -c "from transformers import AutoModel; AutoModel.from_pretrained('bert-base-uncased')"
   ```

2. **Monitor resources:**
   ```bash
   # In another terminal
   docker stats cinesync-training
   ```

3. **Use TensorBoard:** Monitor training progress at http://localhost:6007

## Next Steps

1. Build the image: `./run_training.sh build`
2. Test with help: `./run_training.sh train --help`  
3. Start training: `./run_training.sh train --model sentence-bert`
4. Monitor with TensorBoard: `./run_training.sh tensorboard`