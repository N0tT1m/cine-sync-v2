"""
CineSync v2 - Admin Interface for Model Management
Web-based dashboard for managing AI models, training preferences, and data filtering
"""

from flask import Flask, render_template, request, jsonify, redirect, url_for, flash
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
import os
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import asyncio
from unified_model_manager import model_manager
import pandas as pd
import plotly.graph_objs as go
import plotly.utils

app = Flask(__name__, 
           template_folder='templates',
           static_folder='static')
app.secret_key = os.environ.get('ADMIN_SECRET_KEY', 'dev-key-change-in-production')

# Setup login manager
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# Simple user class for admin authentication
class AdminUser(UserMixin):
    def __init__(self, id):
        self.id = id

@login_manager.user_loader
def load_user(user_id):
    # In production, validate against real admin users
    admin_users = os.environ.get('ADMIN_USERS', 'admin').split(',')
    if user_id in admin_users:
        return AdminUser(user_id)
    return None

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        # Simple authentication - in production use proper auth
        admin_password = os.environ.get('ADMIN_PASSWORD', 'admin123')
        admin_users = os.environ.get('ADMIN_USERS', 'admin').split(',')
        
        if username in admin_users and password == admin_password:
            user = AdminUser(username)
            login_user(user)
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid credentials')
    
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

@app.route('/')
@login_required
def dashboard():
    """Main admin dashboard"""
    
    # Get model status
    model_status = model_manager.get_model_status()
    
    # Get training statistics
    training_stats = get_training_statistics()
    
    # Get recent feedback
    recent_feedback = get_recent_feedback(limit=50)
    
    return render_template('dashboard.html',
                         model_status=model_status,
                         training_stats=training_stats,
                         recent_feedback=recent_feedback)

@app.route('/models')
@login_required
def models_page():
    """Model management page"""
    
    model_configs = model_manager.model_configs
    model_status = model_manager.get_model_status()
    
    return render_template('models.html',
                         model_configs=model_configs,
                         model_status=model_status)

@app.route('/api/models/<model_name>/toggle', methods=['POST'])
@login_required
def toggle_model(model_name):
    """Enable/disable a model"""
    
    if model_name in model_manager.model_configs:
        current_status = model_manager.model_configs[model_name].enabled
        model_manager.model_configs[model_name].enabled = not current_status
        
        # Save configuration
        model_manager.save_model_configs(model_manager.model_configs)
        
        return jsonify({
            'success': True,
            'enabled': not current_status
        })
    
    return jsonify({'success': False, 'error': 'Model not found'}), 404

@app.route('/api/models/<model_name>/reload', methods=['POST'])
@login_required
def reload_model(model_name):
    """Reload a specific model"""
    
    if model_name in model_manager.model_configs:
        try:
            config = model_manager.model_configs[model_name]
            asyncio.run(model_manager._load_single_model(model_name, config))
            
            return jsonify({
                'success': True,
                'message': f'Model {model_name} reloaded successfully'
            })
        except Exception as e:
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500
    
    return jsonify({'success': False, 'error': 'Model not found'}), 404

@app.route('/training')
@login_required
def training_page():
    """Training preferences and data filtering page"""
    
    preferences = model_manager.training_preferences
    
    # Get available genres for filtering
    available_genres = get_available_genres()
    
    # Get user statistics for exclusion management
    user_stats = get_user_statistics()
    
    # Get training data overview
    training_overview = get_training_data_overview()
    
    return render_template('training.html',
                         preferences=preferences,
                         available_genres=available_genres,
                         user_stats=user_stats,
                         training_overview=training_overview)

@app.route('/api/training/preferences', methods=['POST'])
@login_required
def update_training_preferences():
    """Update training preferences"""
    
    try:
        data = request.get_json()
        
        # Validate preferences
        valid_preferences = {
            'auto_retrain',
            'min_feedback_threshold',
            'excluded_genres',
            'excluded_users',
            'quality_filters'
        }
        
        filtered_data = {k: v for k, v in data.items() if k in valid_preferences}
        
        # Update preferences
        model_manager.update_training_preferences(filtered_data)
        
        return jsonify({
            'success': True,
            'message': 'Training preferences updated successfully'
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/training/exclude_user', methods=['POST'])
@login_required
def exclude_user():
    """Exclude a user from training data"""
    
    user_id = request.json.get('user_id')
    
    if user_id:
        excluded_users = set(model_manager.training_preferences.get('excluded_users', []))
        excluded_users.add(int(user_id))
        
        model_manager.update_training_preferences({
            'excluded_users': list(excluded_users)
        })
        
        return jsonify({'success': True})
    
    return jsonify({'success': False, 'error': 'User ID required'}), 400

@app.route('/api/training/include_user', methods=['POST'])
@login_required
def include_user():
    """Include a previously excluded user"""
    
    user_id = request.json.get('user_id')
    
    if user_id:
        excluded_users = set(model_manager.training_preferences.get('excluded_users', []))
        excluded_users.discard(int(user_id))
        
        model_manager.update_training_preferences({
            'excluded_users': list(excluded_users)
        })
        
        return jsonify({'success': True})
    
    return jsonify({'success': False, 'error': 'User ID required'}), 400

@app.route('/api/training/trigger_retrain', methods=['POST'])
@login_required
def trigger_retrain():
    """Manually trigger model retraining"""
    
    try:
        # Run retraining asynchronously
        asyncio.run(model_manager.retrain_models())
        
        return jsonify({
            'success': True,
            'message': 'Model retraining started'
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/upload')
@login_required
def upload_page():
    """Model upload page for drop-in integration"""
    
    return render_template('upload.html')

@app.route('/api/upload_model', methods=['POST'])
@login_required
def upload_model():
    """Handle model file upload"""
    
    if 'model_file' not in request.files:
        return jsonify({'success': False, 'error': 'No file provided'}), 400
    
    file = request.files['model_file']
    model_name = request.form.get('model_name')
    model_type = request.form.get('model_type')
    
    if file.filename == '':
        return jsonify({'success': False, 'error': 'No file selected'}), 400
    
    if not model_name or not model_type:
        return jsonify({'success': False, 'error': 'Model name and type required'}), 400
    
    try:
        # Save uploaded file temporarily
        temp_path = f"/tmp/{file.filename}"
        file.save(temp_path)
        
        # Use drop-in functionality
        success = model_manager.drop_in_model(temp_path, model_name, model_type)
        
        # Clean up temp file
        os.remove(temp_path)
        
        if success:
            return jsonify({
                'success': True,
                'message': f'Model {model_name} uploaded and loaded successfully'
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Failed to load uploaded model'
            }), 500
            
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/analytics')
@login_required
def analytics_page():
    """Analytics and performance monitoring page"""
    
    # Get model performance metrics
    performance_metrics = get_model_performance_metrics()
    
    # Get user engagement statistics
    engagement_stats = get_user_engagement_stats()
    
    # Create performance charts
    performance_chart = create_performance_chart(performance_metrics)
    engagement_chart = create_engagement_chart(engagement_stats)
    
    return render_template('analytics.html',
                         performance_metrics=performance_metrics,
                         engagement_stats=engagement_stats,
                         performance_chart=performance_chart,
                         engagement_chart=engagement_chart)

@app.route('/api/analytics/model_performance')
@login_required
def get_model_performance():
    """API endpoint for model performance data"""
    
    metrics = get_model_performance_metrics()
    return jsonify(metrics)

def get_training_statistics() -> Dict:
    """Get training data statistics"""
    
    try:
        feedback_file = model_manager.models_dir / "training_feedback.jsonl"
        
        if feedback_file.exists():
            feedback_data = []
            with open(feedback_file, 'r') as f:
                for line in f:
                    feedback_data.append(json.loads(line))
            
            df = pd.DataFrame(feedback_data)
            
            return {
                'total_feedback': len(feedback_data),
                'unique_users': df['user_id'].nunique() if len(feedback_data) > 0 else 0,
                'unique_items': df['item_id'].nunique() if len(feedback_data) > 0 else 0,
                'avg_rating': df['rating'].mean() if len(feedback_data) > 0 else 0,
                'last_feedback': max([f['timestamp'] for f in feedback_data]) if feedback_data else None
            }
    except:
        pass
    
    return {
        'total_feedback': 0,
        'unique_users': 0,
        'unique_items': 0,
        'avg_rating': 0,
        'last_feedback': None
    }

def get_recent_feedback(limit: int = 50) -> List[Dict]:
    """Get recent feedback entries"""
    
    try:
        feedback_file = model_manager.models_dir / "training_feedback.jsonl"
        
        if feedback_file.exists():
            feedback_data = []
            with open(feedback_file, 'r') as f:
                for line in f:
                    feedback_data.append(json.loads(line))
            
            # Sort by timestamp and return recent entries
            feedback_data.sort(key=lambda x: x['timestamp'], reverse=True)
            return feedback_data[:limit]
    except:
        pass
    
    return []

def get_available_genres() -> List[str]:
    """Get list of available genres for filtering"""
    
    # This would connect to your database to get actual genres
    # For now, return common movie/TV genres
    return [
        'Action', 'Adventure', 'Animation', 'Biography', 'Comedy', 'Crime',
        'Documentary', 'Drama', 'Family', 'Fantasy', 'History', 'Horror',
        'Music', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Sport',
        'Superhero', 'Thriller', 'War', 'Western'
    ]

def get_user_statistics() -> List[Dict]:
    """Get user statistics for admin management"""
    
    try:
        feedback_file = model_manager.models_dir / "training_feedback.jsonl"
        
        if feedback_file.exists():
            feedback_data = []
            with open(feedback_file, 'r') as f:
                for line in f:
                    feedback_data.append(json.loads(line))
            
            df = pd.DataFrame(feedback_data)
            
            if len(feedback_data) > 0:
                user_stats = df.groupby('user_id').agg({
                    'rating': ['count', 'mean'],
                    'timestamp': 'max'
                }).round(2)
                
                user_stats.columns = ['feedback_count', 'avg_rating', 'last_activity']
                user_stats = user_stats.reset_index()
                
                # Add exclusion status
                excluded_users = set(model_manager.training_preferences.get('excluded_users', []))
                user_stats['excluded'] = user_stats['user_id'].isin(excluded_users)
                
                return user_stats.to_dict('records')
    except:
        pass
    
    return []

def get_training_data_overview() -> Dict:
    """Get overview of training data for filtering"""
    
    try:
        feedback_file = model_manager.models_dir / "training_feedback.jsonl"
        
        if feedback_file.exists():
            feedback_data = []
            with open(feedback_file, 'r') as f:
                for line in f:
                    feedback_data.append(json.loads(line))
            
            df = pd.DataFrame(feedback_data)
            
            if len(feedback_data) > 0:
                return {
                    'content_type_distribution': df['content_type'].value_counts().to_dict(),
                    'rating_distribution': df['rating'].value_counts().sort_index().to_dict(),
                    'daily_feedback': df.groupby(pd.to_datetime(df['timestamp']).dt.date).size().to_dict()
                }
    except:
        pass
    
    return {
        'content_type_distribution': {},
        'rating_distribution': {},
        'daily_feedback': {}
    }

def get_model_performance_metrics() -> Dict:
    """Get model performance metrics"""
    
    # This would connect to your monitoring system
    # For now, return mock data
    return {
        'bert4rec': {'accuracy': 0.85, 'latency': 45, 'throughput': 1200},
        'sentence_bert_two_tower': {'accuracy': 0.82, 'latency': 78, 'throughput': 800},
        'graphsage': {'accuracy': 0.79, 'latency': 120, 'throughput': 600},
        't5_hybrid': {'accuracy': 0.88, 'latency': 200, 'throughput': 300},
        'enhanced_two_tower': {'accuracy': 0.81, 'latency': 35, 'throughput': 1500},
        'variational_autoencoder': {'accuracy': 0.76, 'latency': 25, 'throughput': 2000},
        'hybrid_movie': {'accuracy': 0.83, 'latency': 30, 'throughput': 1800},
        'hybrid_tv': {'accuracy': 0.80, 'latency': 32, 'throughput': 1700}
    }

def get_user_engagement_stats() -> Dict:
    """Get user engagement statistics"""
    
    # Mock engagement data
    return {
        'daily_active_users': [120, 135, 142, 128, 156, 149, 163],
        'recommendation_clicks': [450, 523, 487, 612, 578, 634, 691],
        'feedback_submissions': [23, 31, 28, 35, 42, 38, 44]
    }

def create_performance_chart(metrics: Dict) -> str:
    """Create performance comparison chart"""
    
    models = list(metrics.keys())
    accuracies = [metrics[model]['accuracy'] for model in models]
    latencies = [metrics[model]['latency'] for model in models]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        name='Accuracy',
        x=models,
        y=accuracies,
        yaxis='y',
        marker_color='lightblue'
    ))
    
    fig.add_trace(go.Scatter(
        name='Latency (ms)',
        x=models,
        y=latencies,
        yaxis='y2',
        mode='lines+markers',
        marker_color='red'
    ))
    
    fig.update_layout(
        title='Model Performance Comparison',
        xaxis=dict(title='Models'),
        yaxis=dict(title='Accuracy', side='left'),
        yaxis2=dict(title='Latency (ms)', side='right', overlaying='y'),
        legend=dict(x=0.1, y=0.9)
    )
    
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

def create_engagement_chart(stats: Dict) -> str:
    """Create user engagement chart"""
    
    days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=days,
        y=stats['daily_active_users'],
        mode='lines+markers',
        name='Daily Active Users',
        line=dict(color='blue')
    ))
    
    fig.add_trace(go.Scatter(
        x=days,
        y=stats['recommendation_clicks'],
        mode='lines+markers',
        name='Recommendation Clicks',
        line=dict(color='green')
    ))
    
    fig.add_trace(go.Scatter(
        x=days,
        y=stats['feedback_submissions'],
        mode='lines+markers',
        name='Feedback Submissions',
        line=dict(color='orange')
    ))
    
    fig.update_layout(
        title='User Engagement Trends',
        xaxis=dict(title='Day of Week'),
        yaxis=dict(title='Count'),
        legend=dict(x=0.1, y=0.9)
    )
    
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

if __name__ == '__main__':
    # Create templates directory and basic templates if they don't exist
    os.makedirs('templates', exist_ok=True)
    os.makedirs('static', exist_ok=True)
    
    # Initialize models on startup
    asyncio.run(model_manager.load_all_models())
    
    # Run the admin interface
    app.run(host='0.0.0.0', port=5001, debug=True)