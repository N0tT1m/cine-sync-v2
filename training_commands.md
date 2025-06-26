# Complete Training Commands for All AI Models

## 🎬 Hybrid Recommendation Model - Movies
**Specialized for movie recommendations**
```bash
cd hybrid_recommendation_movie/hybrid_recommendation
python train_with_wandb.py \
  --ratings-path ../../movies/cinesync/ml-32m/ratings.csv \
  --movies-path ../../movies/cinesync/ml-32m/movies.csv \
  --wandb-project cinesync-v2-hybrid-movies \
  --wandb-tags hybrid movies collaborative-filtering content-based \
  --epochs 100 \
  --batch-size 512 \
  --learning-rate 0.001 \
  --embedding-dim 128 \
  --dropout 0.3
```

## 📺 Hybrid Recommendation Model - TV Shows
**Specialized for TV show recommendations (using anime ratings dataset)**
```bash
cd hybrid_recommendation_tv/hybrid_recommendation
python train_with_wandb.py \
  --ratings-path ../../tv/misc/reviews.csv \
  --movies-path ../../tv/misc/animes.csv \
  --wandb-project cinesync-v2-hybrid-tv \
  --wandb-tags hybrid tv-shows anime collaborative-filtering content-based \
  --epochs 100 \
  --batch-size 512 \
  --learning-rate 0.001 \
  --embedding-dim 128 \
  --dropout 0.3
```

**Note**: TV show training uses anime reviews dataset which has actual user ratings (`uid`, `anime_uid`, `score`) similar to movie ratings format.

## 🧠 Neural Collaborative Filtering (NCF) - Movies & Shows
**Works with both movies and TV shows**
```bash
# For Movies
cd neural_collaborative_filtering
python train_with_wandb.py \
  --ratings-path ../movies/cinesync/ml-32m/ratings.csv \
  --movies-path ../movies/cinesync/ml-32m/movies.csv \
  --wandb-project cinesync-v2-ncf-movies \
  --wandb-tags ncf collaborative-filtering movies production \
  --epochs 100 \
  --batch-size 256 \
  --learning-rate 0.001 \
  --embedding-dim 64 \
  --hidden-layers 128 64 \
  --dropout 0.2

# For TV Shows (using anime dataset with actual ratings)
cd neural_collaborative_filtering
python train_with_wandb.py \
  --ratings-path ../tv/misc/reviews.csv \
  --movies-path ../tv/misc/animes.csv \
  --wandb-project cinesync-v2-ncf-tv \
  --wandb-tags ncf collaborative-filtering tv-shows anime production \
  --epochs 100 \
  --batch-size 256 \
  --learning-rate 0.001 \
  --embedding-dim 64 \
  --hidden-layers 128 64 \
  --dropout 0.2

# For Combined Dataset
cd neural_collaborative_filtering
python train_with_wandb.py \
  --ratings-path ../movies/cinesync/ml-32m/ratings.csv \
  --movies-path ../movies/cinesync/ml-32m/movies.csv \
  --wandb-project cinesync-v2-ncf-combined \
  --wandb-tags ncf collaborative-filtering movies tv-shows production \
  --epochs 100 \
  --batch-size 256 \
  --learning-rate 0.001 \
  --embedding-dim 64 \
  --hidden-layers 128 64 \
  --dropout 0.2
```

## 🔄 Sequential Recommendation Model - Movies & Shows
**Sequential patterns for both content types**
```bash
# For Movies
cd sequential_models
python train_with_wandb.py \
  --ratings-path ../movies/cinesync/ml-32m/ratings.csv \
  --movies-path ../movies/cinesync/ml-32m/movies.csv \
  --wandb-project cinesync-v2-sequential-movies \
  --wandb-tags sequential transformer movies production \
  --epochs 50 \
  --batch-size 64 \
  --learning-rate 0.0001 \
  --embedding-dim 256 \
  --num-heads 8 \
  --num-layers 4 \
  --max-seq-length 50 \
  --dropout 0.1

# For TV Shows (using anime dataset with actual ratings)
cd sequential_models
python train_with_wandb.py \
  --ratings-path ../tv/misc/reviews.csv \
  --movies-path ../tv/misc/animes.csv \
  --wandb-project cinesync-v2-sequential-tv \
  --wandb-tags sequential transformer tv-shows anime production \
  --epochs 50 \
  --batch-size 64 \
  --learning-rate 0.0001 \
  --embedding-dim 256 \
  --num-heads 8 \
  --num-layers 4 \
  --max-seq-length 50 \
  --dropout 0.1

# For Combined Dataset
cd sequential_models
python train_with_wandb.py \
  --ratings-path ../movies/cinesync/ml-32m/ratings.csv \
  --movies-path ../movies/cinesync/ml-32m/movies.csv \
  --wandb-project cinesync-v2-sequential-combined \
  --wandb-tags sequential transformer movies tv-shows production \
  --epochs 50 \
  --batch-size 64 \
  --learning-rate 0.0001 \
  --embedding-dim 256 \
  --num-heads 8 \
  --num-layers 4 \
  --max-seq-length 50 \
  --dropout 0.1
```

## 🏗️ Two-Tower Model - Movies & Shows
**Dual-tower architecture for both content types**
```bash
# For Movies
cd two_tower_model
python train_with_wandb.py \
  --ratings-path ../movies/cinesync/ml-32m/ratings.csv \
  --movies-path ../movies/cinesync/ml-32m/movies.csv \
  --wandb-project cinesync-v2-two-tower-movies \
  --wandb-tags two-tower attention movies production \
  --epochs 50 \
  --batch-size 32 \
  --learning-rate 0.0001 \
  --embedding-dim 256 \
  --hidden-dim 512 \
  --num-heads 8 \
  --num-layers 4 \
  --dropout 0.1

# For TV Shows (using anime dataset with actual ratings)
cd two_tower_model
python train_with_wandb.py \
  --ratings-path ../tv/misc/reviews.csv \
  --movies-path ../tv/misc/animes.csv \
  --wandb-project cinesync-v2-two-tower-tv \
  --wandb-tags two-tower attention tv-shows anime production \
  --epochs 50 \
  --batch-size 32 \
  --learning-rate 0.0001 \
  --embedding-dim 256 \
  --hidden-dim 512 \
  --num-heads 8 \
  --num-layers 4 \
  --dropout 0.1

# For Combined Dataset
cd two_tower_model
python train_with_wandb.py \
  --ratings-path ../movies/cinesync/ml-32m/ratings.csv \
  --movies-path ../movies/cinesync/ml-32m/movies.csv \
  --wandb-project cinesync-v2-two-tower-combined \
  --wandb-tags two-tower attention movies tv-shows production \
  --epochs 50 \
  --batch-size 32 \
  --learning-rate 0.0001 \
  --embedding-dim 256 \
  --hidden-dim 512 \
  --num-heads 8 \
  --num-layers 4 \
  --dropout 0.1
```

## 📊 Dataset Information

### Movie Datasets
- **Primary**: `/movies/cinesync/ml-32m/` (MovieLens 32M)
  - `ratings.csv` - 32M user-movie ratings
  - `movies.csv` - Movie metadata with genres
- **Alternative**: `/movies/tmdb-movies/movies_metadata.csv` (TMDB Movies)
- **Streaming**: `/movies/netflix/netflix_movies.csv`, `/movies/disney/disney_plus_movies.csv`

### TV Show Datasets
- **Primary**: `/tv/misc/TMDB_tv_dataset_v3.csv` (TMDB TV Shows)
- **IMDB**: `/tv/imdb/` (Genre-specific TV series)
- **Streaming**: `/tv/netflix/netflix_titles.csv`, `/tv/misc/disney_plus_tv_shows.csv`
- **Anime**: `/tv/anime/animes.csv`

## 🎯 Model Specialization Summary

| Model | Movies | TV Shows | Combined | Specialization |
|-------|--------|----------|----------|----------------|
| **Hybrid Movies** | ✅ Primary | ❌ | ❌ | Movie-focused hybrid approach |
| **Hybrid TV** | ❌ | ✅ Primary | ❌ | TV show-focused hybrid approach |
| **NCF** | ✅ | ✅ | ✅ | General collaborative filtering |
| **Sequential** | ✅ | ✅ | ✅ | Temporal pattern modeling |
| **Two-Tower** | ✅ | ✅ | ✅ | Cross-attention architecture |

## 🚀 Quick Start Commands

### Train All Movie Models
```bash
# Run all movie-focused models
./scripts/train_all_movies.sh
```

### Train All TV Models
```bash
# Run all TV-focused models
./scripts/train_all_tv.sh
```

### Train Hybrid Models (Specialized)
```bash
# Movies only
cd hybrid_recommendation_movie/hybrid_recommendation && python train_with_wandb.py
# TV Shows only
cd hybrid_recommendation_tv/hybrid_recommendation && python train_with_wandb.py
```

## 📈 Performance Monitoring

All models integrate with Weights & Biases (wandb) for:
- Real-time training metrics
- Model architecture visualization
- Hyperparameter tracking
- Experiment comparison
- Model artifact storage

Access your experiments at: https://wandb.ai/your-entity/