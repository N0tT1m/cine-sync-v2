# CineSync v2 Dataset Structure

This document outlines the required directory structure and dataset placement for training Lupe AI models.

## 📁 Root Directory Structure

```
cine-sync-v2/
├── movies/                          # Movie datasets directory
│   ├── ml-32m/                     # MovieLens 32M dataset
│   │   ├── ratings.csv
│   │   ├── movies.csv
│   │   ├── tags.csv
│   │   └── links.csv
│   ├── tmdb_movies/                # TMDB movie data (optional)
│   │   └── tmdb_5000_movies.csv
│   └── processed/                  # Processed movie data (auto-generated)
│       ├── movies_data.csv
│       ├── ratings_data.csv
│       └── movie_metadata.pkl
├── tv/                             # TV show datasets directory
│   ├── tmdb-tv/                   # TMDb TV Shows Dataset
│   │   └── TMDB_tv_dataset_v3.csv # Full TMDb TV Shows Dataset (150K)
│   ├── anime-dataset/             # MyAnimeList Dataset
│   │   ├── animes.csv            # Anime/TV show metadata
│   │   ├── profiles.csv          # User profiles
│   │   └── reviews.csv           # User reviews and ratings
│   ├── imdb-tv/                  # IMDb TV Series Dataset (by genre)
│   │   ├── action_series.csv     # Action TV series
│   │   ├── comedy_series.csv     # Comedy TV series
│   │   ├── drama_series.csv      # Drama TV series
│   │   ├── sci-fi_series.csv     # Sci-Fi TV series
│   │   └── ... (other genres)    # Additional genre files
│   ├── netflix-movies-and-tv/    # Netflix Dataset
│   │   └── netflix_titles.csv    # Netflix movies and TV shows
│   └── processed/                  # Processed TV data (auto-generated)
│       ├── tv_shows_data.csv
│       ├── tv_ratings_data.csv
│       ├── tv_metadata.pkl
│       ├── tv_encoders.pkl
│       └── tv_scalers.pkl
├── models/                         # Trained models directory
│   ├── best_model.pt              # Best movie model
│   ├── model_metadata.pkl         # Movie model metadata
│   ├── id_mappings.pkl            # Movie ID mappings
│   ├── best_tv_model.pt           # Best TV show model
│   ├── tv_metadata.pkl            # TV model metadata
│   ├── tv_encoders.pkl            # TV ID mappings
│   └── movie_lookup.pkl           # Movie lookup data
└── lupe(python)/                   # Discord bot application
    └── models/                     # Model classes
        ├── __init__.py
        ├── hybrid_recommender.py  # Movie recommendation model
        ├── tv_recommender.py      # TV show recommendation model
        └── content_manager.py     # Lupe AI unified manager
```

## 📋 Required Datasets

### Movies Directory (`/movies`)

#### 1. MovieLens 32M Dataset
- **Source**: [MovieLens](https://grouplens.org/datasets/movielens/32m/)
- **Location**: `movies/ml-32m/`
- **Files**:
  - `ratings.csv` - 32M ratings from 280K users
  - `movies.csv` - Movie metadata with genres
  - `tags.csv` - User-generated tags
  - `links.csv` - Links to IMDB and TMDB

#### 2. TMDB Movies (Optional)
- **Source**: [Kaggle TMDB 5000 Movies](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata)
- **Location**: `movies/tmdb_movies/`
- **Files**:
  - `tmdb_5000_movies.csv` - Movie metadata from TMDB

### TV Shows Directory (`/tv`)

#### 1. TMDb TV Shows Dataset (150K Shows)
- **Source**: TMDb TV Dataset v3
- **Location**: `tv/tmdb-tv/TMDB_tv_dataset_v3.csv`
- **Features**: Comprehensive metadata including genres, ratings, cast, crew, episode data

#### 2. MyAnimeList Dataset (Anime/TV Shows)
- **Source**: MyAnimeList export
- **Location**: `tv/anime-dataset/`
  - `animes.csv` - Anime/TV show metadata
  - `profiles.csv` - User profiles (optional)
  - `reviews.csv` - User reviews and ratings
- **Features**: Anime and TV series data, user ratings, episode counts, status

#### 3. IMDb TV Series Dataset (Genre-based)
- **Source**: IMDb TV series data organized by genre
- **Location**: `tv/imdb-tv/`
- **Files**: Multiple CSV files, one per genre:
  - `action_series.csv`, `comedy_series.csv`, `drama_series.csv`
  - `sci-fi_series.csv`, `thriller_series.csv`, etc.
- **Features**: Series metadata, ratings, cast information by genre

#### 4. Netflix Movies and TV Shows Dataset
- **Source**: Netflix content catalog
- **Location**: `tv/netflix-movies-and-tv/netflix_titles.csv`
- **Features**: Netflix TV shows and movies, genres, cast, descriptions

## 🚀 Setup Instructions

### 1. Create Directory Structure
```bash
cd cine-sync-v2
mkdir -p movies/ml-32m
mkdir -p movies/tmdb_movies
mkdir -p movies/processed
mkdir -p tv/processed
mkdir -p models
```

### 2. Download and Place Datasets

#### Movies:
```bash
# Download MovieLens 32M dataset
wget http://files.grouplens.org/datasets/movielens/ml-32m.zip
unzip ml-32m.zip -d movies/
mv movies/ml-32m/* movies/ml-32m/

# Download TMDB movies (optional)
# Place tmdb_5000_movies.csv in movies/tmdb_movies/
```

#### TV Shows:
```bash
# Download from Kaggle (requires Kaggle API)
kaggle datasets download -d asaniczka/full-tmdb-tv-shows-dataset-2023-150k-shows
kaggle datasets download -d marlesson/myanimelist-dataset-animes-profiles-reviews  
kaggle datasets download -d suraj520/imdb-tv-series-data

# Extract and rename files
unzip full-tmdb-tv-shows-dataset-2023-150k-shows.zip
mv full_tmdb_tv_shows_dataset.csv tv/tmdb_tv_shows.csv

unzip myanimelist-dataset-animes-profiles-reviews.zip
mv animes.csv tv/myanimelist_dataset.csv

unzip imdb-tv-series-data.zip
mv tv_series.csv tv/imdb_tv_series.csv
```

### 3. Process Datasets
```bash
# Process TV show datasets
python process_tv_datasets.py

# Process movie datasets (if using existing training script)
python main.py  # Your existing movie training script
```

### 4. Train Models
```bash
# Train movie model (existing)
python main.py

# Train TV show model (new)
python train_tv_shows.py --epochs 20 --batch-size 64
```

## 📊 Expected File Sizes

### Movies:
- `ml-32m/ratings.csv`: ~750MB
- `ml-32m/movies.csv`: ~1MB
- `tmdb_5000_movies.csv`: ~5MB

### TV Shows:
- `tmdb_tv_shows.csv`: ~50MB
- `myanimelist_dataset.csv`: ~200MB
- `imdb_tv_series.csv`: ~100MB

### Models:
- `best_model.pt`: ~200MB (movie model)
- `best_tv_model.pt`: ~150MB (TV model)
- Various metadata files: ~50MB total

## ⚠️ Important Notes

1. **Dataset Availability**: Some datasets may require Kaggle account and API setup
2. **Storage Requirements**: Total storage needed: ~2-3GB for all datasets
3. **Processing Time**: TV dataset processing may take 30-60 minutes depending on dataset sizes
4. **GPU Memory**: TV model training recommended with 8GB+ GPU memory
5. **Backup**: Keep original datasets safe as processing creates new files

## 🔧 Troubleshooting

### Missing Datasets
- Check file paths match exactly as specified
- Ensure datasets are unzipped correctly
- Verify file permissions for read access

### Processing Errors
- Check available disk space (need ~1GB free for processing)
- Ensure Python dependencies are installed
- Check CSV file encoding (should be UTF-8)

### Memory Issues
- Reduce batch size in training scripts
- Use CPU training if GPU memory is insufficient
- Process datasets in smaller chunks if needed

## 📞 Support

If you encounter issues with dataset setup:
1. Check the file paths in error messages
2. Verify dataset file formats match expectations
3. Ensure sufficient disk space and memory
4. Check the troubleshooting section in README.md