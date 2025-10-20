# Multi-stage build for movie recommendation server
FROM nvidia/cuda:11.8-devel-ubuntu22.04 as builder

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    build-essential \
    pkg-config \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Rust
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

# Set up working directory
WORKDIR /app

# Copy Cargo files
COPY Cargo.toml Cargo.lock ./

# Copy source code
COPY src/ ./src/

# Build the application
ENV TORCH_CUDA_VERSION=cu118
RUN cargo build --release

# Runtime stage
FROM nvidia/cuda:11.8-runtime-ubuntu22.04

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Create app user
RUN useradd -m -u 1000 appuser

# Set up working directory
WORKDIR /app

# Copy the binary from builder stage
COPY --from=builder /app/target/release/movie-recommendation-server /usr/local/bin/

# Create models directory
RUN mkdir -p /app/models && chown -R appuser:appuser /app

# Switch to app user
USER appuser

# Expose port
EXPOSE 3000

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:3000/health || exit 1

# Run the application
CMD ["movie-recommendation-server", "--models-path", "/app/models", "--host", "0.0.0.0", "--port", "3000"]