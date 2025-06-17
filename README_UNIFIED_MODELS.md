# CineSync v2 - Unified AI Model Integration

🚀 **Complete AI Recommendation System with 6 Advanced Models + 2 Hybrid Recommenders**

## 🎯 Overview

CineSync v2 now supports **drop-in integration** of all 6 AI models with unified management, automatic training based on download preferences, and a comprehensive admin interface for data filtering.

### 🤖 **Supported AI Models:**

1. **BERT4Rec** - Sequential recommendation with bidirectional transformers
2. **Sentence-BERT Two-Tower** - Content-aware recommendations with semantic understanding  
3. **GraphSAGE** - Graph neural network for cold-start problems
4. **T5 Hybrid** - Text-to-text transformer for content encoding
5. **Enhanced Two-Tower** - Improved collaborative filtering
6. **Variational AutoEncoder** - Latent space recommendations
7. **Hybrid Movie Recommender** - Existing movie recommendation system
8. **Hybrid TV Recommender** - Existing TV show recommendation system

## 🚀 Quick Start

### 1. **One-Command Setup**

```bash
# Start the complete CineSync v2 system
python start_cinesync.py
```

This will:
- ✅ Check dependencies
- ✅ Initialize all AI models
- ✅ Launch admin interface at http://localhost:5001
- ✅ Setup training system with download preferences
- ✅ Create unified model management

### 2. **Access Admin Dashboard**

```
URL: http://localhost:5001
Login: admin
Password: admin123
```

## 📁 **New File Structure**

```
cine-sync-v2/
├── unified_model_manager.py     # Central model management
├── admin_interface.py           # Web admin dashboard
├── start_cinesync.py           # One-command startup
├── templates/                   # Admin interface templates
│   ├── base.html
│   ├── dashboard.html
│   ├── training.html
│   ├── upload.html
│   └── login.html
├── advanced_models/             # 6 AI models
│   ├── bert4rec_recommender.py
│   ├── sentence_bert_two_tower.py
│   ├── graphsage_recommender.py
│   ├── t5_hybrid_recommender.py
│   ├── enhanced_two_tower.py
│   └── variational_autoencoder.py
├── hybrid_recommendation_movie/ # Existing movie system
└── hybrid_recommendation_tv/    # Existing TV system
```

## 🎪 **Key Features**

### ✨ **Drop-in Model Integration**

```python
# Simply drop your trained models into the system
from unified_model_manager import model_manager

# Upload via admin interface or programmatically
model_manager.drop_in_model(
    model_file_path="path/to/your/model.pt",
    model_name="my_bert4rec_v2", 
    model_type="bert4rec"
)
```

### 🎯 **Smart Recommendations**

```python
from unified_model_manager import get_recommendations

# Get recommendations using the best available model
recommendations = get_recommendations(
    user_id=123,
    content_type="movie",  # or "tv" or "both"
    top_k=10
)
```

### 🎛️ **Training Based on Download Preferences**

The system automatically:
- ✅ Tracks download quality preferences (4K, 2K, 1080p only)
- ✅ Excludes unwanted genres from training
- ✅ Filters out specific users
- ✅ Auto-retrains when enough feedback is collected
- ✅ Provides admin controls for all preferences

## 🔧 **Admin Interface Features**

### 📊 **Dashboard**
- Real-time model status
- Training statistics
- Recent feedback monitoring
- Quick actions for model management

### 🤖 **Model Management**
- Enable/disable models
- Reload models without restart
- View model performance metrics
- Upload new models via web interface

### ⚙️ **Training Configuration**
- Set minimum feedback thresholds
- Configure quality filters (4K, 2K, 1080p, etc.)
- Exclude genres from training
- Manage user exclusions
- Manual retrain triggers

### 📈 **Analytics**
- Model performance comparison
- User engagement trends
- Recommendation accuracy metrics
- Training data distribution

## 🛠️ **Integration with Your App**

### 1. **Add to Your High-Seas Web App**

In your existing Angular service:

```typescript
// recommendation.service.ts
import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';

@Injectable()
export class RecommendationService {
  constructor(private http: HttpClient) {}

  getRecommendations(userId: number, contentType: string = 'movie', topK: number = 10) {
    return this.http.get(`/api/recommendations`, {
      params: { user_id: userId, content_type: contentType, top_k: topK }
    });
  }

  addFeedback(userId: number, itemId: number, rating: number, contentType: string) {
    return this.http.post(`/api/feedback`, {
      user_id: userId,
      item_id: itemId, 
      rating: rating,
      content_type: contentType
    });
  }
}
```

### 2. **Connect Download Service to Training**

In your existing download service:

```typescript
// enhanced-download.service.ts - Add this method
addDownloadFeedback(movieId: number, quality: string, userId: number) {
  // Convert download to implicit rating (higher quality = higher rating)
  const qualityRatings = {
    '4K': 5.0,
    '2K': 4.5, 
    '1080p': 4.0,
    '720p': 3.5,
    '480p': 3.0
  };
  
  const rating = qualityRatings[quality] || 3.0;
  
  // Send to CineSync training system
  this.http.post('/api/training/feedback', {
    user_id: userId,
    item_id: movieId,
    rating: rating,
    content_type: 'movie',
    download_quality: quality
  }).subscribe();
}
```

## 🎨 **Customization**

### **Model Priorities**

Configure which models to use first in `model_config.json`:

```json
{
  "bert4rec": {"priority": 5, "enabled": true},
  "t5_hybrid": {"priority": 4, "enabled": true},
  "graphsage": {"priority": 3, "enabled": false}
}
```

### **Training Filters**

Set download quality preferences:

```python
model_manager.update_training_preferences({
    "quality_filters": ["4K", "2K", "1080p"],  # Only train on high quality
    "excluded_genres": ["Horror", "Documentary"],
    "min_feedback_threshold": 500
})
```

## 🔍 **Monitoring & Debugging**

### **View System Status**

```bash
# Interactive mode with status commands
python start_cinesync.py

CineSync> status    # Show system status
CineSync> models    # Show model details  
CineSync> logs      # Show recent logs
CineSync> help      # Show all commands
```

### **Logs Location**

```
cinesync_startup.log          # System startup logs
unified_model_manager.log     # Model management logs
training_feedback.jsonl       # All training feedback data
```

## 🚨 **Production Deployment**

### **Environment Variables**

```bash
export ADMIN_SECRET_KEY="your-secure-secret-key"
export ADMIN_USERS="admin,manager,operator"
export ADMIN_PASSWORD="secure-password"
```

### **Daemon Mode**

```bash
# Run as background service
python start_cinesync.py --daemon

# Or use systemd/supervisor for production
```

### **Performance Tuning**

```python
# In unified_model_manager.py
model_manager.training_preferences.update({
    "auto_retrain": True,
    "min_feedback_threshold": 1000,  # Higher for production
    "quality_filters": ["4K", "2K", "1080p"]  # Focus on quality
})
```

## 🎭 **API Endpoints**

The unified system provides these endpoints:

```http
GET  /                           # Admin dashboard
GET  /models                     # Model management
GET  /training                   # Training configuration
GET  /upload                     # Model upload interface
POST /api/models/{name}/toggle   # Enable/disable model
POST /api/models/{name}/reload   # Reload specific model
POST /api/training/preferences   # Update training preferences
POST /api/upload_model          # Upload new model file
```

## 🎪 **Integration Examples**

### **Movie Detail Page**

Add recommendations to your existing movie detail components:

```typescript
// popular-movies-details.component.ts
ngOnInit() {
  // Existing code...
  
  // Add personalized recommendations
  this.recommendationService.getRecommendations(
    this.userId, 
    'movie', 
    5
  ).subscribe(recommendations => {
    this.similarMovies = recommendations;
  });
}
```

### **Home Page Enhancement**

```typescript
// home.component.ts  
ngOnInit() {
  // Add recommendation carousels
  this.loadPersonalizedRecommendations();
  this.loadTrendingContent();
}

loadPersonalizedRecommendations() {
  this.recommendationService.getRecommendations(
    this.userId,
    'both',  // Movies and TV
    20
  ).subscribe(recommendations => {
    this.personalizedContent = recommendations;
  });
}
```

## 📋 **Migration Guide**

### **From Existing Setup**

1. **Backup your current models**:
   ```bash
   cp -r hybrid_recommendation_movie/hybrid_recommendation/models/ backup/
   ```

2. **Install new dependencies**:
   ```bash
   pip install flask plotly torch transformers
   ```

3. **Start the unified system**:
   ```bash
   python start_cinesync.py
   ```

4. **Upload your existing models** via the admin interface

5. **Configure training preferences** based on your download patterns

## 🎯 **Next Steps**

1. **Start the system**: `python start_cinesync.py`
2. **Access admin**: http://localhost:5001
3. **Upload your models** via the web interface
4. **Configure training** based on your download preferences
5. **Integrate with your app** using the provided examples

## 🎉 **Benefits**

✅ **Drop-in integration** - Add any of 6 AI models instantly  
✅ **Automatic training** - Learns from download preferences  
✅ **Admin controls** - Filter genres, users, quality levels  
✅ **Web interface** - Manage everything through browser  
✅ **High performance** - Optimized for production use  
✅ **Monitoring** - Built-in analytics and logging  
✅ **Flexible** - Use any combination of models  

---

🎬 **Welcome to the future of AI-powered recommendations with CineSync v2!**