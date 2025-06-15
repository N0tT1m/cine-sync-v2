# Sequential Models (RNN/LSTM)

Time-aware recommendation using recurrent neural networks to model user behavior sequences.

## Model Architecture
- LSTM/GRU layers for sequence modeling
- Attention mechanisms for temporal focus
- Multi-task learning for rating and ranking

## Dataset Requirements
- Timestamped user interaction sequences
- Session-based or long-term user history
- Current dataset: 32M+ timestamped ratings

## Directory Structure
```
src/           # Source code
models/        # Trained model files
data/          # Processed sequential data
notebooks/     # Jupyter notebooks for experimentation
tests/         # Unit tests
```