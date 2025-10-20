# Two-Tower/Dual-Encoder Model

Separate encoding towers for users and items with efficient similarity computation.

## Model Architecture
- User tower: Demographics, behavior, preferences
- Item tower: Content features, metadata, embeddings
- Dot-product similarity for scalable inference

## Dataset Requirements
- Rich user and item feature data
- Content metadata (genres, cast, descriptions)
- Current dataset: TMDB movies/TV + user interactions

## Directory Structure
```
src/           # Source code
models/        # Trained model files
data/          # Processed feature data
notebooks/     # Jupyter notebooks for experimentation
tests/         # Unit tests
```