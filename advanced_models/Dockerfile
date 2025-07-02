# Use NVIDIA PyTorch base image with CUDA support
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    curl \
    wget \
    unzip \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --upgrade pip

# Install PyTorch geometric and sparse libraries first
# Using the correct PyTorch version (2.5.1) and ensuring compatibility
RUN pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-2.5.1+cu124.html

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install additional packages that might cause issues
RUN pip install --no-cache-dir \
    accelerate \
    bitsandbytes \
    transformers \
    sentence-transformers \
    datasets \
    wandb \
    tensorboard

# Copy the application code
COPY . .

# Set environment variables
ENV PYTHONPATH=/app:/app/..
ENV CUDA_VISIBLE_DEVICES=0
ENV TORCH_CUDA_ARCH_LIST="8.6"  # RTX 4090 architecture
ENV PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Create necessary directories
RUN mkdir -p /app/logs /app/checkpoints /app/outputs

# Expose ports for tensorboard and wandb
EXPOSE 6006 8080

# Default command
CMD ["python", "train_advanced_models.py", "--help"]