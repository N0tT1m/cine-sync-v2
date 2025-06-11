# Movie Recommendation Server (Rust)

A high-performance Rust-based web server for hosting PyTorch movie recommendation models. This server can load and serve predictions from the PyTorch recommendation model trained with the provided Python script.

## Features

- **PyTorch Model Loading**: Loads TorchScript models using `tch` (PyTorch Rust bindings)
- **High Performance**: Asynchronous Rust web server using Axum
- **GPU Support**: Automatic CUDA detection and GPU acceleration
- **Multiple Model Types**: Supports both hybrid (collaborative filtering) and content-based models
- **RESTful API**: Clean HTTP endpoints for recommendations and model information
- **Error Handling**: Comprehensive error handling and logging
- **Cross-Platform**: Works on Linux, macOS, and Windows

## Prerequisites

### System Requirements

1. **Rust** (1.70 or later)
   ```bash
   curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
   ```

2. **PyTorch C++ Libraries** (libtorch)
   - The `tch` crate will attempt to download this automatically
   - For manual installation: https://pytorch.org/cplusplus/

3. **CUDA** (optional, for GPU acceleration)
   - CUDA 11.8 or later
   - Compatible GPU drivers

### Python Dependencies (for model export)

```bash
pip install torch pandas scikit-learn numpy pickle5
```

## Quick Start

### 1. Train Your Model

First, train your recommendation model using the provided Python script:

```bash
python run_training_pytorch.py --epochs 20 --batch-size 64
```

### 2. Export Model Artifacts

Convert pickle files to JSON for Rust compatibility:

```bash
python export_metadata_for_rust.py --models-path models
```

### 3. Build and Run the Server

```bash
# Clone and build
cargo build --release

# Run the server
cargo run --release -- --models-path models --port 3000
```

The server will be available at `http://localhost:3000`

## API Endpoints

### Health Check

```bash
curl http://localhost:3000/health
```

Response:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_type": "hybrid",
  "device": "Cuda(0)",
  "movies_count": 15000,
  "genres_count": 18,
  "uptime_seconds": 3600,
  "version": "0.1.0"
}
```

### Get Recommendations

#### For Hybrid Models (User-based)

```bash
curl -X POST http://localhost:3000/recommend \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": 123,
    "top_k": 10
  }'
```

#### For Content-based Models (Movie similarity)

```bash
curl -X POST http://localhost:3000/recommend \
  -H "Content-Type: application/json" \
  -d '{
    "movie_ids": [1, 2, 3],
    "top_k": 10
  }'
```

#### For Content-based Models (Genre-based)

```bash
curl -X POST http://localhost:3000/recommend \
  -H "Content-Type: application/json" \
  -d '{
    "genres": ["Action", "Comedy"],
    "top_k": 10
  }'
```

Response:
```json
{
  "recommendations": [
    {
      "movie_id": 4567,
      "title": "The Matrix",
      "genres": "Action|Sci-Fi",
      "score": 0.95,
      "rank": 1
    }
  ],
  "request_id": "uuid-here",
  "model_type": "hybrid"
}
```

### List Movies

```bash
curl http://localhost:3000/movies
```

### List Genres

```bash
curl http://localhost:3000/genres
```

## Configuration

### Command Line Options

```bash
cargo run --release -- --help
```

- `--models-path`: Path to the models directory (default: "models")
- `--port`: Port to bind the server to (default: 3000)
- `--host`: Host to bind the server to (default: "127.0.0.1")
- `--cpu-only`: Force CPU usage (disable CUDA)

### Environment Variables

- `RUST_LOG`: Set logging level (e.g., `RUST_LOG=info`)
- `CUDA_VISIBLE_DEVICES`: Control GPU visibility

## Model Directory Structure

Your models directory should contain:

```
models/
├── recommendation_model.pt      # TorchScript model
├── model_metadata.json         # Model metadata (exported from pickle)
├── movies_data.csv             # Movie information
├── movie_lookup.json           # Movie lookup table (optional)
├── rating_scaler.json          # Rating scaler (for hybrid models)
├── similarity_matrix.json      # Similarity matrix (for content-based)
└── training_history.json       # Training history (optional)
```

## Performance Tuning

### GPU Optimization

```bash
# For high-end GPUs (RTX 4090, A100, etc.)
export CUDA_VISIBLE_DEVICES=0
cargo run --release -- --models-path models

# For multi-GPU setups
export CUDA_VISIBLE_DEVICES=0,1
```

### Memory Management

- The server automatically manages batch sizes based on available memory
- For large models, consider increasing system RAM
- Monitor GPU memory usage with `nvidia-smi`

### Scaling

For production deployments:

1. **Load Balancing**: Run multiple instances behind a load balancer
2. **Caching**: Add Redis for caching frequent recommendations
3. **Database**: Store movie metadata in a proper database
4. **Monitoring**: Add metrics and health checks

## Troubleshooting

### Common Issues

1. **Model Loading Fails**
   ```
   Error: Failed to load TorchScript model
   ```
   - Ensure the model was saved correctly as TorchScript
   - Check that libtorch is properly installed
   - Verify CUDA compatibility if using GPU

2. **Metadata Loading Issues**
   ```
   Warning: Could not load metadata from pickle file
   ```
   - Run the export script: `python export_metadata_for_rust.py`
   - Check that JSON files are properly formatted

3. **CUDA Errors**
   ```
   Error: CUDA error: out of memory
   ```
   - Reduce batch size or use `--cpu-only`
   - Check GPU memory with `nvidia-smi`

4. **Port Already in Use**
   ```
   Error: Address already in use
   ```
   - Use a different port: `--port 3001`
   - Kill existing processes: `lsof -ti:3000 | xargs kill`

### Debug Mode

```bash
RUST_LOG=debug cargo run -- --models-path models
```

## Development

### Building from Source

```bash
git clone <repository>
cd movie-recommendation-server
cargo build
```

### Running Tests

```bash
cargo test
```

### Code Structure

- `src/main.rs`: Server setup and routing
- `src/model.rs`: PyTorch model loading and inference
- `src/data.rs`: Movie data and CSV handling
- `src/inference.rs`: Recommendation algorithms
- `src/error.rs`: Error handling

## License

[MIT License](LICENSE)

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## Support

For issues and questions:
- Check the troubleshooting section
- Review the logs for error details
- Open an issue on GitHub