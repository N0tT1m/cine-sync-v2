#!/bin/bash

# CineSync v2 Training Script with Docker
set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}🎬 CineSync v2 Advanced Models Training${NC}"
echo "=================================================="

# Function to print usage
usage() {
    echo "Usage: $0 [COMMAND] [OPTIONS]"
    echo ""
    echo "Commands:"
    echo "  build       Build the Docker image"
    echo "  train       Run training with specified model"
    echo "  jupyter     Start Jupyter notebook server"
    echo "  tensorboard Start TensorBoard monitoring"
    echo "  shell       Open interactive shell in container"
    echo "  clean       Clean up Docker resources"
    echo ""
    echo "Training Options:"
    echo "  --model MODEL_NAME    Model to train (bert4rec, sentence-bert, graphsage, t5, etc.)"
    echo "  --epochs N           Number of training epochs"
    echo "  --batch-size N       Batch size"
    echo "  --lr FLOAT           Learning rate"
    echo ""
    echo "Examples:"
    echo "  $0 build"
    echo "  $0 train --model sentence-bert --epochs 50"
    echo "  $0 jupyter"
    echo "  $0 shell"
}

# Build Docker image
build() {
    echo -e "${YELLOW}Building Docker image...${NC}"
    docker-compose build training
    echo -e "${GREEN}✅ Build complete${NC}"
}

# Run training
train() {
    local args="$@"
    echo -e "${YELLOW}Starting training with args: $args${NC}"
    docker-compose run --rm training python train_advanced_models.py $args
}

# Start Jupyter
jupyter() {
    echo -e "${YELLOW}Starting Jupyter notebook server...${NC}"
    echo -e "${GREEN}Access at: http://localhost:8888${NC}"
    docker-compose --profile jupyter up jupyter
}

# Start TensorBoard
tensorboard() {
    echo -e "${YELLOW}Starting TensorBoard...${NC}"
    echo -e "${GREEN}Access at: http://localhost:6007${NC}"
    docker-compose --profile monitoring up tensorboard
}

# Open shell
shell() {
    echo -e "${YELLOW}Opening interactive shell...${NC}"
    docker-compose run --rm training bash
}

# Clean up
clean() {
    echo -e "${YELLOW}Cleaning up Docker resources...${NC}"
    docker-compose down --volumes --remove-orphans
    docker system prune -f
    echo -e "${GREEN}✅ Cleanup complete${NC}"
}

# Check if Docker is running
check_docker() {
    if ! docker info > /dev/null 2>&1; then
        echo -e "${RED}❌ Docker is not running. Please start Docker first.${NC}"
        exit 1
    fi
}

# Check for NVIDIA Docker runtime
check_nvidia() {
    if ! docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi > /dev/null 2>&1; then
        echo -e "${YELLOW}⚠️  NVIDIA Docker runtime not available. GPU training will be disabled.${NC}"
    else
        echo -e "${GREEN}✅ NVIDIA GPU support detected${NC}"
    fi
}

# Main script logic
main() {
    check_docker
    check_nvidia
    
    case "$1" in
        build)
            build
            ;;
        train)
            shift
            train "$@"
            ;;
        jupyter)
            jupyter
            ;;
        tensorboard)
            tensorboard
            ;;
        shell)
            shell
            ;;
        clean)
            clean
            ;;
        help|--help|-h)
            usage
            ;;
        "")
            usage
            ;;
        *)
            echo -e "${RED}❌ Unknown command: $1${NC}"
            usage
            exit 1
            ;;
    esac
}

# Run main function with all arguments
main "$@"