# CineSync v2 - API Usage Examples

Complete examples for all APIs, Discord commands, and integrations.

---

## Table of Contents

1. [Discord Bot Commands](#1-discord-bot-commands)
2. [Unified Inference API](#2-unified-inference-api)
3. [Admin Interface API](#3-admin-interface-api)
4. [Database Operations](#4-database-operations)
5. [Model Training](#5-model-training)
6. [Personalization System](#6-personalization-system)

---

## 1. Discord Bot Commands

### 1.1 Recommendation Commands

#### /recommend - Basic Recommendations
```
# Get 10 movie recommendations
/recommend content_type:movie limit:10

# Get TV recommendations in a specific genre
/recommend content_type:tv genre:Drama limit:5

# Get mixed content recommendations
/recommend content_type:mixed genre:Sci-Fi limit:15

# Get recommendations for a specific user
/recommend content_type:movie user_id:123456789 limit:10

# Get recommendations based on a title
/recommend content_type:movie content_title:Inception limit:10
```

**Response Example:**
```
🎬 Movie Recommendations

Based on: Genre: Sci-Fi

1. Interstellar (2014)
   Genres: Adventure, Drama, Sci-Fi
   Match Score: 0.92

2. The Matrix (1999)
   Genres: Action, Sci-Fi
   Match Score: 0.89

3. Blade Runner 2049 (2017)
   Genres: Action, Drama, Sci-Fi
   Match Score: 0.87

[👍 Good Overall] [👎 Poor Overall] [⭐ Rate Individual] [💖 Like/Dislike Each] [💬 Detailed Feedback]
```

---

#### /similar - Find Similar Content
```
# Find movies similar to a specific title
/similar content_type:movie title:The Dark Knight limit:5

# Find TV shows similar to a title
/similar content_type:tv title:Breaking Bad limit:10
```

**Response Example:**
```
🎯 Similar to "The Dark Knight"

1. The Dark Knight Rises (2012)
   Similarity: 0.95
   Genres: Action, Crime, Drama

2. Batman Begins (2005)
   Similarity: 0.91
   Genres: Action, Adventure

3. Joker (2019)
   Similarity: 0.85
   Genres: Crime, Drama, Thriller
```

---

#### /cross_recommend - Cross-Content Recommendations
```
# Get TV shows based on a movie you like
/cross_recommend source_type:movie target_type:tv source_title:Inception limit:5

# Get movies based on a TV show you like
/cross_recommend source_type:tv target_type:movie source_title:Stranger Things limit:5
```

---

#### /recommend_advanced - Advanced Recommendations
```
# Use specific model
/recommend_advanced content_type:movie model:bert4rec genre:Horror limit:10

# Use ensemble with confidence filter
/recommend_advanced content_type:tv model:ensemble min_confidence:0.8 limit:5

# Available models: ensemble, ncf, sequential, two_tower
```

---

### 1.2 Personalization Commands

#### /my_recommendations - Personalized Recommendations
```
# Get personalized movie recommendations
/my_recommendations content_type:movie count:10

# Get personalized TV recommendations
/my_recommendations content_type:tv count:15
```

**Requirements:** User must have rated at least 5 items.

**Response Example:**
```
🎯 Your Personalized Recommendations

Based on your 47 ratings

1. Dune (2021)
   Genres: Action, Adventure, Sci-Fi
   Match: 94%
   Base: 0.85 | Personal: 0.98

2. Arrival (2016)
   Genres: Drama, Sci-Fi
   Match: 91%
   Base: 0.82 | Personal: 0.95
```

---

#### /my_stats - View Your Profile
```
/my_stats
```

**Response Example:**
```
📊 YourUsername's Recommendation Profile

Based on 47 ratings

🎭 Favorite Genres
Action - ⭐4.5 (12 rated)
Sci-Fi - ⭐4.3 (9 rated)
Drama - ⭐4.1 (8 rated)

🎬 Favorite Directors
Christopher Nolan - ⭐4.8
Denis Villeneuve - ⭐4.6

⭐ Favorite Actors
Leonardo DiCaprio - ⭐4.5
Christian Bale - ⭐4.3

📈 Rating Style
Average: 3.8/5
Style: Balanced ⚖️
Diversity: 72%

📅 Preferred Eras
2010s, 2000s, 1990s
```

---

#### /rate - Rate Content
```
# Rate a movie
/rate content_type:movie title:Inception rating:5

# Rate a TV show
/rate content_type:tv title:Breaking Bad rating:5
```

---

#### /rate_movies - Quick Rating Flow
```
# Rate 5 movies quickly
/rate_movies count:5

# Rate 3 movies
/rate_movies count:3
```

Opens an interactive modal to rate multiple movies at once.

---

#### /my_ratings - View Your Ratings
```
# View movie ratings
/my_ratings content_type:movie limit:20

# View TV ratings
/my_ratings content_type:tv limit:10
```

---

#### /my_preferences - View Preferences
```
/my_preferences
```

Shows your love/like/dislike/hate preferences for individual content.

---

### 1.3 Analytics Commands

#### /lupe_status - Bot Status
```
/lupe_status
```

**Response:**
```
🤖 Lupe AI Status

Models Loaded: 3
Device: cuda
Movie Count: 45,000
TV Count: 12,000
Total Genres: 24

Model Status:
✅ NCF: Loaded
✅ Sequential: Loaded
✅ Two-Tower: Loaded
```

---

#### /model_compare - Compare Models
```
/model_compare user_id:123456789 limit:5
```

**Response:**
```
🔬 Model Comparison for User 123456789

NCF Recommendations:
1. Inception (0.92)
2. The Matrix (0.88)

Sequential Recommendations:
1. Interstellar (0.90)
2. Arrival (0.85)

Two-Tower Recommendations:
1. Blade Runner 2049 (0.89)
2. Ex Machina (0.84)

Ensemble (Combined):
1. Inception (0.91)
2. Interstellar (0.88)
```

---

#### /model_health - Check Model Health
```
/model_health
```

**Response:**
```
🏥 Model Health Status

✅ neural_collaborative_filtering: Healthy
✅ sequential: Healthy
✅ two_tower: Healthy
❌ bert4rec: Not loaded
```

---

### 1.4 Admin Commands

#### /admin_review - Review Feedback
```
/admin_review limit:20
```

Shows pending feedback for admin approval.

---

#### /admin_approve - Approve Feedback
```
/admin_approve feedback_id:123
```

---

#### /admin_reject - Reject Feedback
```
/admin_reject feedback_id:123
```

---

#### /export_training_data - Export Data
```
/export_training_data
```

Exports approved feedback to CSV files.

---

### 1.5 Utility Commands

#### /genres - List Genres
```
# List all genres
/genres content_type:mixed

# List movie genres only
/genres content_type:movie

# List TV genres only
/genres content_type:tv
```

---

#### /search - Search for Content
```
/search title:Inception
```

---

## 2. Unified Inference API

### 2.1 Python Usage

```python
from src.api.unified_inference_api import UnifiedRecommendationAPI, ModelType

# Initialize API
api = UnifiedRecommendationAPI(
    config_path="unified_config.json",
    device="cuda"
)

# Load all models
api.load_all_models()

# Check health
health = api.health_check()
print(f"Model health: {health}")
# Output: {'neural_collaborative_filtering': True, 'sequential': True, 'two_tower': True}
```

---

### 2.2 Getting Recommendations

```python
# Get recommendations from ensemble (all models)
recommendations = api.get_recommendations(
    user_id=12345,
    top_k=10,
    model_type=None  # None = ensemble
)

for rec in recommendations:
    print(f"Item {rec.item_id}: {rec.score:.3f} (conf: {rec.confidence:.3f})")

# Output:
# Item 1234: 0.923 (conf: 0.891)
# Item 5678: 0.887 (conf: 0.856)
```

---

### 2.3 Using Specific Models

```python
# NCF recommendations
ncf_recs = api.get_recommendations(
    user_id=12345,
    top_k=5,
    model_type=ModelType.NCF
)

# Sequential recommendations
seq_recs = api.get_recommendations(
    user_id=12345,
    top_k=5,
    model_type=ModelType.SEQUENTIAL
)

# Two-tower recommendations
tt_recs = api.get_recommendations(
    user_id=12345,
    top_k=5,
    model_type=ModelType.TWO_TOWER
)
```

---

### 2.4 Predicting Next Items (Sequential)

```python
# User's viewing history
viewing_history = [101, 205, 307, 412]  # Movie IDs

# Predict what they'll watch next
next_items = api.predict_next_items(
    sequence=viewing_history,
    top_k=5,
    model_type=ModelType.SEQUENTIAL
)

for item in next_items:
    print(f"Predicted next: {item.item_id} ({item.score:.3f})")
```

---

### 2.5 Finding Similar Items

```python
# Find movies similar to movie ID 1234
similar = api.find_similar_items(
    item_id=1234,
    top_k=10,
    model_type=ModelType.TWO_TOWER  # Best for similarity
)

for item in similar:
    print(f"Similar item {item.item_id}: {item.score:.3f}")
```

---

### 2.6 Predicting Ratings

```python
# Predict how user 12345 would rate movie 5678
predictions = api.predict_rating(
    user_id=12345,
    item_id=5678,
    model_type=None  # All models
)

print(predictions)
# Output: {'neural_collaborative_filtering': 4.2, 'two_tower': 0.85}
```

---

### 2.7 Comparing Models

```python
# Compare recommendations from all models
comparison = api.compare_models(user_id=12345, top_k=5)

for model_name, recs in comparison.items():
    print(f"\n{model_name}:")
    for rec in recs:
        print(f"  {rec.item_id}: {rec.score:.3f}")
```

---

## 3. Admin Interface API

### 3.1 Authentication

```python
import requests

# Login
response = requests.post('http://localhost:5001/login', data={
    'username': 'admin',
    'password': 'your_password'
})

# Store session cookie
session = requests.Session()
session.post('http://localhost:5001/login', data={
    'username': 'admin',
    'password': 'your_password'
})
```

---

### 3.2 Model Management

```python
# Toggle model on/off
response = session.post(
    'http://localhost:5001/api/models/bert4rec/toggle'
)
print(response.json())
# Output: {'success': True, 'enabled': False}

# Reload a model
response = session.post(
    'http://localhost:5001/api/models/ncf/reload'
)
print(response.json())
# Output: {'success': True, 'message': 'Model ncf reloaded successfully'}
```

---

### 3.3 Training Preferences

```python
# Update training preferences
response = session.post(
    'http://localhost:5001/api/training/preferences',
    json={
        'auto_retrain': True,
        'min_feedback_threshold': 100,
        'excluded_genres': ['Adult', 'Documentary'],
        'excluded_users': [123, 456],
        'quality_filters': {'min_rating': 3.0}
    }
)
print(response.json())
# Output: {'success': True, 'message': 'Training preferences updated successfully'}
```

---

### 3.4 User Exclusion

```python
# Exclude a user from training data
response = session.post(
    'http://localhost:5001/api/training/exclude_user',
    json={'user_id': 123456}
)

# Include a previously excluded user
response = session.post(
    'http://localhost:5001/api/training/include_user',
    json={'user_id': 123456}
)
```

---

### 3.5 Trigger Retraining

```python
# Manually trigger model retraining
response = session.post(
    'http://localhost:5001/api/training/trigger_retrain'
)
print(response.json())
# Output: {'success': True, 'message': 'Model retraining started'}
```

---

### 3.6 Model Upload

```python
# Upload a new model
with open('my_model.pt', 'rb') as f:
    response = session.post(
        'http://localhost:5001/api/upload_model',
        files={'model_file': f},
        data={
            'model_name': 'my_custom_model',
            'model_type': 'ncf'
        }
    )
print(response.json())
```

---

## 4. Database Operations

### 4.1 Connection Setup

```python
import psycopg2
from contextlib import contextmanager

@contextmanager
def get_db_connection():
    conn = psycopg2.connect(
        host='localhost',
        port=5432,
        database='cinesync',
        user='postgres',
        password='your_password'
    )
    try:
        yield conn
    finally:
        conn.close()
```

---

### 4.2 User Ratings

```python
# Insert a rating
with get_db_connection() as conn:
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO user_ratings (user_id, movie_id, rating, timestamp)
        VALUES (%s, %s, %s, NOW())
        ON CONFLICT (user_id, movie_id)
        DO UPDATE SET rating = EXCLUDED.rating, timestamp = EXCLUDED.timestamp
    """, (user_id, movie_id, rating))
    conn.commit()

# Get user's ratings
with get_db_connection() as conn:
    cursor = conn.cursor()
    cursor.execute("""
        SELECT movie_id, rating, timestamp
        FROM user_ratings
        WHERE user_id = %s
        ORDER BY timestamp DESC
        LIMIT 50
    """, (user_id,))
    ratings = cursor.fetchall()
```

---

### 4.3 Feedback Storage

```python
# Store recommendation feedback
with get_db_connection() as conn:
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO feedback (
            user_id, username, feedback_type, recommendation_method,
            recommendations, original_query, content_type, model_used
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
    """, (
        user_id, username, 'positive', 'ensemble',
        json.dumps(recommendations), 'Sci-Fi movies', 'movie', 'ensemble'
    ))
    conn.commit()
```

---

### 4.4 User Preferences

```python
# Store individual content preference
with get_db_connection() as conn:
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO user_preferences (
            user_id, content_id, content_type, preference, recommendation_method
        ) VALUES (%s, %s, %s, %s, %s)
        ON CONFLICT (user_id, content_id, content_type)
        DO UPDATE SET preference = EXCLUDED.preference
    """, (user_id, content_id, 'movie', 'love', 'ensemble'))
    conn.commit()
```

---

## 5. Model Training

### 5.1 Training a Single Model

```bash
# Train franchise sequence model
python src/training/train_all_models.py \
    --model movie_franchise_sequence \
    --epochs 50 \
    --batch-size 256 \
    --lr 0.0001

# Train with WandB logging
python src/training/train_all_models.py \
    --model bert4rec \
    --epochs 100 \
    --wandb
```

---

### 5.2 Training by Category

```bash
# Train all movie models
python src/training/train_all_models.py --category movie --epochs 50

# Train all TV models
python src/training/train_all_models.py --category tv --epochs 50

# Train all content-agnostic models
python src/training/train_all_models.py --category both --epochs 50

# Train all unified models
python src/training/train_all_models.py --category unified --epochs 50
```

---

### 5.3 Training Everything

```bash
# Train all 45 models
python src/training/train_all_models.py --all --epochs 50 --batch-size 256
```

---

### 5.4 Programmatic Training

```python
from src.training.train_all_models import UnifiedTrainingPipeline

# Create pipeline
pipeline = UnifiedTrainingPipeline(
    model_name='movie_franchise_sequence',
    data_dir='data',
    output_dir='models',
    device='cuda',
    use_wandb=True
)

# Train
result = pipeline.train(
    epochs=50,
    batch_size=256,
    lr=1e-4,
    weight_decay=0.01,
    save_every=5,
    early_stopping_patience=10
)

print(f"Best validation loss: {result['best_val_loss']}")
print(f"Epochs trained: {result['epochs_trained']}")
```

---

## 6. Personalization System

### 6.1 Preference Learner

```python
from services.lupe_python.preference_learner import PreferenceLearner

# Initialize
learner = PreferenceLearner(db_manager)

# Analyze user patterns
patterns = await learner.analyze_user_patterns(user_id=12345)

print(patterns)
# Output:
# {
#     'total_ratings': 47,
#     'avg_rating': 3.8,
#     'favorite_genres': [
#         {'genre': 'Action', 'avg_rating': 4.5, 'count': 12},
#         {'genre': 'Sci-Fi', 'avg_rating': 4.3, 'count': 9}
#     ],
#     'favorite_directors': [
#         {'director': 'Christopher Nolan', 'avg_rating': 4.8}
#     ],
#     'favorite_actors': [...],
#     'preferred_decades': [
#         {'decade': '2010s', 'count': 20}
#     ],
#     'rating_distribution': {1: 2, 2: 5, 3: 10, 4: 18, 5: 12},
#     'diversity_score': 0.72
# }
```

---

### 6.2 Personalized Trainer

```python
from services.lupe_python.personalized_trainer import PersonalizedTrainer

# Initialize
trainer = PersonalizedTrainer(db_manager, lupe_manager)

# Get personalized recommendations
recommendations = await trainer.get_personalized_recommendations(
    user_id=12345,
    top_k=10,
    content_type='movie',
    base_recommendations=base_recs
)

# Update user embedding based on feedback
await trainer.update_user_embedding(
    user_id=12345,
    item_id=5678,
    feedback_type='positive',  # positive, negative, neutral
    rating=5
)
```

---

### 6.3 Integration Example

```python
# Complete personalization flow
async def get_personalized_recs(user_id: int, count: int = 10):
    # Check if user has enough data
    rating_count = await trainer._get_user_rating_count(user_id)

    if rating_count < 5:
        return None, "Need at least 5 ratings"

    # Get base recommendations
    base_recs = lupe.get_recommendations(
        user_id=user_id,
        limit=count * 2,
        content_type='movie'
    )

    # Personalize
    personalized = await trainer.get_personalized_recommendations(
        user_id=user_id,
        top_k=count,
        content_type='movie',
        base_recommendations=base_recs
    )

    return personalized, None
```

---

## Complete Integration Example

```python
"""
Complete example showing all systems working together
"""

import asyncio
from src.api.unified_inference_api import UnifiedRecommendationAPI, ModelType
from services.lupe_python.preference_learner import PreferenceLearner
from services.lupe_python.personalized_trainer import PersonalizedTrainer

async def complete_recommendation_flow(user_id: int):
    # 1. Initialize systems
    api = UnifiedRecommendationAPI(device='cuda')
    api.load_all_models()

    db_manager = DatabaseManager(config.database)
    preference_learner = PreferenceLearner(db_manager)
    personalized_trainer = PersonalizedTrainer(db_manager, api)

    # 2. Analyze user preferences
    user_patterns = await preference_learner.analyze_user_patterns(user_id)
    print(f"User has {user_patterns['total_ratings']} ratings")
    print(f"Favorite genres: {user_patterns['favorite_genres'][:3]}")

    # 3. Get base recommendations from ensemble
    base_recs = api.get_recommendations(
        user_id=user_id,
        top_k=20,
        model_type=ModelType.ENSEMBLE
    )

    # 4. Personalize recommendations
    if user_patterns['total_ratings'] >= 5:
        personalized_recs = await personalized_trainer.get_personalized_recommendations(
            user_id=user_id,
            top_k=10,
            content_type='movie',
            base_recommendations=[(r.item_id, "Unknown", "movie", r.score) for r in base_recs]
        )
        print("\nPersonalized Recommendations:")
        for rec in personalized_recs:
            print(f"  {rec['item_id']}: {rec['score']:.3f}")
    else:
        print("\nBase Recommendations (not enough data for personalization):")
        for rec in base_recs[:10]:
            print(f"  {rec.item_id}: {rec.score:.3f}")

    # 5. Simulate user feedback
    await personalized_trainer.update_user_embedding(
        user_id=user_id,
        item_id=base_recs[0].item_id,
        feedback_type='positive',
        rating=5
    )
    print("\nUser embedding updated with positive feedback")

    # 6. Store feedback in database
    with db_manager.get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO feedback (user_id, feedback_type, model_used)
            VALUES (%s, %s, %s)
        """, (user_id, 'positive', 'ensemble'))
        conn.commit()

    return personalized_recs if user_patterns['total_ratings'] >= 5 else base_recs

# Run
if __name__ == "__main__":
    recs = asyncio.run(complete_recommendation_flow(user_id=12345))
```

---

**Document Version:** 1.0.0
**Last Updated:** 2026-01-14
