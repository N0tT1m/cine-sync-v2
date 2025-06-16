# CineSync TV - TV Show Hybrid Recommendation System

This is the **TV show hybrid recommendation system** designed specifically for **television shows and series**. It builds upon the original movie system but adds specialized features for episodic content.

## System Overview

- **Type**: TV show-focused hybrid recommendation system
- **Architecture**: Enhanced neural network for episodic content
- **Training Data**: TMDb TV Shows (150K) + MyAnimeList (80M) + IMDb datasets
- **Features**: Episode count, season data, show status, duration analysis
- **TV-Specific**: Handles ongoing series, episode progression, binge-watching patterns

## Key Components

### Model Architecture
- **TVShowRecommenderModel**: Specialized neural network for TV shows
- **TV-Specific Features**: Episode count, season count, duration, status
- **Enhanced Embeddings**: User, show, and genre embeddings
- **Content Features**: TV-specific metadata processing

### Core Files
- `models/tv_recommender.py` - Main TV show recommendation model
- `models/content_manager.py` - TV content management
- `train_tv_shows.py` - Training script for TV shows
- `process_tv_datasets.py` - TV dataset processing

## TV-Specific Features

### Episode and Season Handling
- **Episode Count**: Considers total number of episodes
- **Season Structure**: Analyzes season patterns
- **Show Status**: Ongoing, completed, cancelled series
- **Duration Analysis**: Episode runtime patterns

### Specialized Training
```bash
# Process TV show datasets
python process_tv_datasets.py

# Train TV show model
python train_tv_shows.py --epochs 20 --batch-size 64 --gpu

# Cross-validate TV model
python validate_models.py --content-type tv
```

## Usage

```python
from models.tv_recommender import TVShowRecommenderModel

# Initialize TV recommender
tv_model = TVShowRecommenderModel(
    num_users=num_users,
    num_shows=num_shows,
    num_genres=num_genres
)

# Get TV show recommendations
recommendations = tv_model.get_user_recommendations(
    user_id=123,
    show_data=tv_shows_df,
    top_k=10
)
```

## Expected Performance

- **Expected RMSE**: <0.20 (episodic content complexity)
- **Cross-Content Accuracy**: 80%+ (movie-to-TV recommendations)
- **Cold Start Performance**: 70%+ (new users)
- **Training Time**: ~60 minutes (estimated)

## TV Show Datasets

### Primary Datasets
- **TMDb TV Shows**: 150K+ TV shows with comprehensive metadata
- **MyAnimeList**: 300K users, 80M+ anime/series ratings
- **IMDb TV Series**: Industry-standard TV series data

### Data Processing
- Genre classification for TV shows
- Episode and season metadata extraction
- Show status classification (ongoing/completed/cancelled)
- Duration and runtime analysis

## Discord Bot Integration

The TV system provides specialized TV show commands:

```
/recommend tv 8               # Get 8 TV show recommendations
/similar "Breaking Bad"       # Find similar TV shows
/cross_recommend movie tv     # TV shows based on movie preferences
```

This system extends CineSync's capabilities to handle the unique characteristics of episodic television content.