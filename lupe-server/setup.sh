#!/bin/bash

# Movie Recommendation Server Setup Script
# This script helps you set up and run the Rust-based recommendation server

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check prerequisites
check_prerequisites() {
    print_status "Checking prerequisites..."
    
    # Check Rust
    if ! command_exists cargo; then
        print_error "Rust is not installed. Please install Rust from https://rustup.rs/"
        exit 1
    fi
    print_success "Rust is installed"
    
    # Check Python (optional, for model export)
    if command_exists python3; then
        print_success "Python 3 is available"
    elif command_exists python; then
        print_success "Python is available"
    else
        print_warning "Python not found. You'll need Python to export model artifacts."
    fi
    
    # Check CUDA (optional)
    if command_exists nvidia-smi; then
        print_success "NVIDIA GPU detected"
        nvidia-smi --query-gpu=name --format=csv,noheader,nounits | head -1
    else
        print_warning "No NVIDIA GPU detected. Server will run on CPU."
    fi
    
    # Check Docker (optional)
    if command_exists docker; then
        print_success "Docker is available"
    else
        print_warning "Docker not found. Docker deployment will not be available."
    fi
}

# Build the server
build_server() {
    print_status "Building the recommendation server..."
    
    # Set environment variables for PyTorch
    export TORCH_CUDA_VERSION=cu118
    
    if cargo build --release; then
        print_success "Server built successfully"
    else
        print_error "Failed to build server"
        exit 1
    fi
}

# Export model artifacts
export_model_artifacts() {
    local models_path="${1:-models}"
    
    if [ ! -d "$models_path" ]; then
        print_error "Models directory not found: $models_path"
        print_status "Please train your model first using: python run_training_pytorch.py"
        return 1
    fi
    
    print_status "Exporting model artifacts for Rust compatibility..."
    
    if [ -f "export_metadata_for_rust.py" ]; then
        python3 export_metadata_for_rust.py --models-path "$models_path"
        print_success "Model artifacts exported"
    else
        print_warning "Export script not found. Make sure export_metadata_for_rust.py is in the current directory."
    fi
}

# Run the server
run_server() {
    local models_path="${1:-models}"
    local port="${2:-3000}"
    local host="${3:-127.0.0.1}"
    local cpu_only="${4:-false}"
    
    print_status "Starting recommendation server..."
    print_status "Models path: $models_path"
    print_status "Server URL: http://$host:$port"
    
    # Build command
    local cmd="./target/release/movie-recommendation-server --models-path $models_path --port $port --host $host"
    
    if [ "$cpu_only" = "true" ]; then
        cmd="$cmd --cpu-only"
        print_status "Running in CPU-only mode"
    fi
    
    print_status "Running: $cmd"
    
    # Set logging level
    export RUST_LOG=${RUST_LOG:-info}
    
    # Run the server
    $cmd
}

# Docker build and run
docker_setup() {
    local models_path="${1:-models}"
    
    if ! command_exists docker; then
        print_error "Docker is not installed"
        exit 1
    fi
    
    print_status "Building Docker image..."
    docker build -t movie-recommendation-server .
    
    print_status "Running Docker container..."
    docker run -d \
        --name movie-recommendation-server \
        --gpus all \
        -p 3000:3000 \
        -v "$(pwd)/$models_path:/app/models:ro" \
        -e RUST_LOG=info \
        movie-recommendation-server
    
    print_success "Docker container started"
    print_status "Check logs with: docker logs movie-recommendation-server"
    print_status "Server available at: http://localhost:3000"
}

# Test the server
test_server() {
    local port="${1:-3000}"
    local host="${2:-localhost}"
    
    print_status "Testing server at http://$host:$port..."
    
    # Wait for server to start
    sleep 2
    
    # Test health endpoint
    if curl -f "http://$host:$port/health" >/dev/null 2>&1; then
        print_success "Health check passed"
        
        # Get server info
        print_status "Server information:"
        curl -s "http://$host:$port/health" | python3 -m json.tool 2>/dev/null || echo "Server is running"
    else
        print_error "Health check failed"
        return 1
    fi
}

# Show usage
show_usage() {
    echo "Movie Recommendation Server Setup Script"
    echo ""
    echo "Usage: $0 [COMMAND] [OPTIONS]"
    echo ""
    echo "Commands:"
    echo "  check      Check prerequisites"
    echo "  build      Build the server"
    echo "  export     Export model artifacts from pickle to JSON"
    echo "  run        Run the server (default)"
    echo "  docker     Build and run with Docker"
    echo "  test       Test the running server"
    echo "  all        Run complete setup (check, build, export, run)"
    echo ""
    echo "Options:"
    echo "  --models-path PATH    Path to models directory (default: models)"
    echo "  --port PORT          Server port (default: 3000)"
    echo "  --host HOST          Server host (default: 127.0.0.1)"
    echo "  --cpu-only           Force CPU usage"
    echo "  --help               Show this help"
    echo ""
    echo "Examples:"
    echo "  $0 all                           # Complete setup and run"
    echo "  $0 run --port 8080               # Run on port 8080"
    echo "  $0 docker --models-path ./models # Run with Docker"
}

# Parse command line arguments
COMMAND="run"
MODELS_PATH="models"
PORT="3000"
HOST="127.0.0.1"
CPU_ONLY="false"

while [[ $# -gt 0 ]]; do
    case $1 in
        check|build|export|run|docker|test|all)
            COMMAND="$1"
            shift
            ;;
        --models-path)
            MODELS_PATH="$2"
            shift 2
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        --host)
            HOST="$2"
            shift 2
            ;;
        --cpu-only)
            CPU_ONLY="true"
            shift
            ;;
        --help)
            show_usage
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
done

# Main execution
print_status "Movie Recommendation Server Setup"
print_status "Command: $COMMAND"

case $COMMAND in
    check)
        check_prerequisites
        ;;
    build)
        check_prerequisites
        build_server
        ;;
    export)
        export_model_artifacts "$MODELS_PATH"
        ;;
    run)
        check_prerequisites
        if [ ! -f "./target/release/movie-recommendation-server" ]; then
            print_status "Server not built yet. Building..."
            build_server
        fi
        export_model_artifacts "$MODELS_PATH"
        run_server "$MODELS_PATH" "$PORT" "$HOST" "$CPU_ONLY"
        ;;
    docker)
        check_prerequisites
        export_model_artifacts "$MODELS_PATH"
        docker_setup "$MODELS_PATH"
        ;;
    test)
        test_server "$PORT" "$HOST"
        ;;
    all)
        check_prerequisites
        build_server
        export_model_artifacts "$MODELS_PATH"
        print_status "Setup complete! Starting server..."
        run_server "$MODELS_PATH" "$PORT" "$HOST" "$CPU_ONLY"
        ;;
    *)
        print_error "Unknown command: $COMMAND"
        show_usage
        exit 1
        ;;
esac