#!/usr/bin/env python3
# retrain_with_feedback.py - Retrain model using Discord bot feedback

import os
import sys
import pandas as pd
import numpy as np
import psycopg2
import psycopg2.extras
import pickle
import logging
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# Import your existing training functions
from run_training_pytorch import (
    setup_gpu, train_model, find_data_files, 
    process_and_prepare_data, log_gpu_memory
)

# Database configuration (same as Discord bot)
DB_CONFIG = {
    'host': '192.168.1.78',
    'database': 'postgres',
    'user': 'postgres',
    'password': 'password',
    'port': 5432
}

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("feedback-retraining")


def load_feedback_data():
    """Load feedback data from PostgreSQL database
    
    Connects to the Discord bot's PostgreSQL database to retrieve:
    - Direct user ratings from the /rate command
    - Feedback ratings from movie recommendations
    - Sentiment analysis data (positive/negative feedback)
    
    Returns:
        tuple: (ratings_df, feedback_df, sentiment_df) or (None, None, None) on error
    """
    logger.info("Loading feedback data from PostgreSQL")
    
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        
        # Get all direct user ratings from Discord /rate command
        cursor.execute('''
            SELECT user_id, movie_id, rating, timestamp
            FROM user_ratings
            ORDER BY timestamp
        ''')
        ratings_data = cursor.fetchall()
        
        # Get feedback ratings from recommendation interactions
        cursor.execute('''
            SELECT user_id, movie_id, rating, feedback_type, timestamp
            FROM feedback
            WHERE rating IS NOT NULL AND movie_id IS NOT NULL
            ORDER BY timestamp
        ''')
        feedback_ratings = cursor.fetchall()
        
        # Get sentiment feedback for user preference weighting
        cursor.execute('''
            SELECT user_id, feedback_type, COUNT(*) as count
            FROM feedback
            WHERE feedback_type IN ('positive', 'negative')
            GROUP BY user_id, feedback_type
        ''')
        feedback_sentiment = cursor.fetchall()
        
        conn.close()
        
        # Convert to DataFrames
        ratings_df = pd.DataFrame(ratings_data) if ratings_data else pd.DataFrame()
        feedback_df = pd.DataFrame(feedback_ratings) if feedback_ratings else pd.DataFrame()
        sentiment_df = pd.DataFrame(feedback_sentiment) if feedback_sentiment else pd.DataFrame()
        
        logger.info(f"Loaded {len(ratings_df)} direct ratings, {len(feedback_df)} feedback ratings")
        
        return ratings_df, feedback_df, sentiment_df
        
    except Exception as e:
        logger.error(f"Error loading feedback data: {e}")
        return None, None, None


def process_feedback_ratings(ratings_df, feedback_df, sentiment_df):
    """Process and combine feedback data with original ratings
    
    Converts raw database feedback into standardized format for training:
    - Normalizes user/movie IDs to integers
    - Standardizes rating scales
    - Removes duplicates (keeping most recent rating per user-movie pair)
    - Adds data source tracking for analysis
    
    Args:
        ratings_df (pd.DataFrame): Direct user ratings from Discord
        feedback_df (pd.DataFrame): Feedback ratings from recommendations
        sentiment_df (pd.DataFrame): Positive/negative feedback counts
        
    Returns:
        pd.DataFrame: Processed and combined ratings data
    """
    logger.info("Processing feedback ratings")
    
    combined_ratings = []
    
    # Process direct ratings from Discord bot /rate command
    if not ratings_df.empty:
        for _, row in ratings_df.iterrows():
            combined_ratings.append({
                'userId': int(row['user_id']),      # Discord user ID
                'movieId': int(row['movie_id']),    # TMDB or IMDb movie ID
                'rating': float(row['rating']),     # User's explicit rating
                'timestamp': row['timestamp'],       # When rating was given
                'source': 'discord_rating'          # Track data source
            })
    
    # Process implicit feedback ratings from recommendation responses
    if not feedback_df.empty:
        for _, row in feedback_df.iterrows():
            combined_ratings.append({
                'userId': int(row['user_id']),      # Discord user ID
                'movieId': int(row['movie_id']),    # Recommended movie ID
                'rating': float(row['rating']),     # Inferred rating from feedback
                'timestamp': row['timestamp'],       # When feedback was given
                'source': 'discord_feedback'        # Track as implicit feedback
            })
    
    # Create DataFrame
    if combined_ratings:
        new_ratings_df = pd.DataFrame(combined_ratings)
        
        # Remove duplicates - keep most recent rating for each user-movie pair
        # This handles cases where users rate the same movie multiple times
        new_ratings_df = new_ratings_df.sort_values('timestamp').drop_duplicates(
            ['userId', 'movieId'], keep='last'
        )
        
        logger.info(f"Created {len(new_ratings_df)} processed feedback ratings")
        return new_ratings_df
    
    return pd.DataFrame()


def create_user_preference_weights(sentiment_df):
    """Create user preference weights based on positive/negative feedback
    
    Analyzes user feedback patterns to create training sample weights:
    - Users with more positive feedback get higher weights
    - Users with mixed feedback get moderate weights
    - Helps the model learn from users with consistent preferences
    
    Args:
        sentiment_df (pd.DataFrame): Positive/negative feedback counts per user
        
    Returns:
        dict: Mapping of user_id to preference weight (0.0 to 1.0)
    """
    user_weights = {}
    
    if not sentiment_df.empty:
        # Group by user_id
        for user_id in sentiment_df['user_id'].unique():
            user_feedback = sentiment_df[sentiment_df['user_id'] == user_id]
            
            positive_count = user_feedback[user_feedback['feedback_type'] == 'positive']['count'].sum()
            negative_count = user_feedback[user_feedback['feedback_type'] == 'negative']['count'].sum()
            
            total_feedback = positive_count + negative_count
            if total_feedback > 0:
                # Calculate preference consistency weight with Laplace smoothing
                # More positive feedback = higher training weight
                weight = (positive_count + 1) / (total_feedback + 2)  # Add smoothing
                user_weights[user_id] = weight
            else:
                user_weights[user_id] = 0.5  # Neutral weight for users without sentiment data
    
    logger.info(f"Created preference weights for {len(user_weights)} users")
    return user_weights


def load_original_data():
    """Load original training data if available
    
    Attempts to load the baseline training data (MovieLens, etc.) to combine
    with new feedback data. This preserves the model's general knowledge
    while incorporating user-specific preferences.
    
    Returns:
        pd.DataFrame: Original training data or empty DataFrame if not found
    """
    logger.info("Loading original training data")
    
    try:
        # Try to load previously processed and cached training data
        if os.path.exists('data/original_train_data.csv'):
            original_df = pd.read_csv('data/original_train_data.csv')
            logger.info(f"Loaded {len(original_df)} original training samples")
            return original_df
        
        # If no cached data, reprocess from original dataset files
        data_files = find_data_files()
        if data_files:
            train_data, val_data, movies_df, genres = process_and_prepare_data(data_files)
            
            if train_data is not None and val_data is not None:
                # Combine train and validation for complete baseline dataset
                original_df = pd.concat([train_data, val_data], ignore_index=True)
                
                # Save for future use
                os.makedirs('data', exist_ok=True)
                original_df.to_csv('data/original_train_data.csv', index=False)
                logger.info(f"Processed and saved {len(original_df)} original samples")
                
                return original_df
        
        logger.warning("No original training data found")
        return pd.DataFrame()
        
    except Exception as e:
        logger.error(f"Error loading original data: {e}")
        return pd.DataFrame()


def combine_datasets(original_df, feedback_df, user_weights=None):
    """Combine original training data with new feedback data
    
    Merges baseline training data with new user feedback to create
    an enhanced training dataset. Handles:
    - Data format standardization
    - Duplicate resolution (prioritizing recent feedback)
    - Sample weighting based on user preference consistency
    - Temporal weighting to emphasize recent interactions
    
    Args:
        original_df (pd.DataFrame): Baseline training data (MovieLens, etc.)
        feedback_df (pd.DataFrame): New feedback data from Discord bot
        user_weights (dict): Optional user preference weights
        
    Returns:
        pd.DataFrame: Combined training dataset with sample weights
    """
    logger.info("Combining original and feedback datasets")
    
    if original_df.empty and feedback_df.empty:
        logger.error("No data available for training")
        return None
    
    # Handle cases where only one dataset is available
    if original_df.empty:
        logger.info("Using only feedback data for training")
        combined_df = feedback_df.copy()
    
    elif feedback_df.empty:
        logger.info("No new feedback data, using original data")
        combined_df = original_df.copy()
    
    # Combine both datasets
    else:
        logger.info("Combining original and feedback data")
        
        # Standardize column names for compatibility between datasets
        if 'userId' not in original_df.columns and 'user_id' in original_df.columns:
            original_df = original_df.rename(columns={'user_id': 'userId'})
        if 'movieId' not in original_df.columns and 'movie_id' in original_df.columns:
            original_df = original_df.rename(columns={'movie_id': 'movieId'})
        
        # Add source column to original data if not present
        if 'source' not in original_df.columns:
            original_df['source'] = 'original'
        
        # Combine datasets
        combined_df = pd.concat([original_df, feedback_df], ignore_index=True)
        
        # Resolve conflicts by preferring more recent feedback over original data
        combined_df = combined_df.sort_values(['userId', 'movieId', 'timestamp']).drop_duplicates(
            ['userId', 'movieId'], keep='last'
        )
    
    # Apply user preference weights to emphasize consistent users
    if user_weights:
        combined_df['sample_weight'] = combined_df['userId'].map(user_weights).fillna(0.5)
    else:
        combined_df['sample_weight'] = 1.0  # Equal weight for all samples
    
    # Apply temporal weighting to emphasize recent user preferences
    if 'timestamp' in combined_df.columns:
        # Convert timestamps and calculate relative recency
        combined_df['timestamp'] = pd.to_datetime(combined_df['timestamp'])
        min_date = combined_df['timestamp'].min()
        combined_df['days_since_start'] = (combined_df['timestamp'] - min_date).dt.days
        max_days = combined_df['days_since_start'].max()
        
        if max_days > 0:
            # Apply time-based weighting: recent ratings get 0.5-1.5x multiplier
            time_weight = 0.5 + (combined_df['days_since_start'] / max_days)
            combined_df['sample_weight'] *= time_weight
    
    logger.info(f"Combined dataset: {len(combined_df)} total samples")
    logger.info(f"Feedback samples: {len(combined_df[combined_df['source'] != 'original'])}")
    
    return combined_df


def retrain_model_with_feedback():
    """Main function to retrain model with feedback data
    
    Orchestrates the complete retraining pipeline:
    1. Loads feedback data from PostgreSQL database
    2. Combines with original training data
    3. Applies user preference and temporal weighting
    4. Retrains the model with enhanced dataset
    5. Backs up original model and saves new version
    
    Returns:
        bool: True if retraining succeeded, False otherwise
    """
    logger.info("Starting model retraining with feedback data")
    
    try:
        # Set up GPU
        device = setup_gpu()
        
        # Load feedback data from PostgreSQL
        ratings_df, feedback_df, sentiment_df = load_feedback_data()
        if ratings_df is None:
            logger.error("Failed to load feedback data")
            return False
        
        # Verify sufficient feedback data for meaningful retraining
        total_feedback = len(ratings_df) + len(feedback_df)
        if total_feedback < 5:
            logger.warning(f"Insufficient feedback data ({total_feedback} samples). Need at least 5.")
            return False
        
        # Process feedback ratings
        processed_feedback = process_feedback_ratings(ratings_df, feedback_df, sentiment_df)
        if processed_feedback.empty:
            logger.error("No valid feedback ratings found")
            return False
        
        # Create user preference weights
        user_weights = create_user_preference_weights(sentiment_df)
        
        # Load original training data
        original_data = load_original_data()
        
        # Combine datasets
        combined_data = combine_datasets(original_data, processed_feedback, user_weights)
        if combined_data is None:
            logger.error("Failed to combine datasets")
            return False
        
        # Load movie data
        if os.path.exists('models/movie_lookup.pkl'):
            with open('models/movie_lookup.pkl', 'rb') as f:
                movie_lookup = pickle.load(f)
            
            # Convert movie lookup to DataFrame
            movies_df = pd.DataFrame.from_dict(movie_lookup, orient='index').reset_index()
            movies_df = movies_df.rename(columns={'index': 'media_id'})
        else:
            logger.error("Movie lookup file not found")
            return False
        
        # Load genres from metadata
        if os.path.exists('models/model_metadata.pkl'):
            with open('models/model_metadata.pkl', 'rb') as f:
                metadata = pickle.load(f)
            genres = metadata.get('genres', [])
        else:
            logger.error("Model metadata not found")
            return False
        
        # Merge with movie features
        logger.info("Merging feedback data with movie features")
        
        # Ensure movieId is in movies_df
        if 'movieId' not in movies_df.columns and 'media_id' in movies_df.columns:
            movies_df['movieId'] = movies_df['media_id']
        
        # Add genre features to movies_df if not present
        for genre in genres:
            if genre not in movies_df.columns:
                movies_df[genre] = movies_df['genres'].apply(
                    lambda x: 1 if pd.notnull(x) and genre in str(x).split('|') else 0
                )
        
        # Merge combined data with movie features
        training_data = pd.merge(combined_data, movies_df, on='movieId', how='inner')
        
        if training_data.empty:
            logger.error("No matching movies found after merge")
            return False
        
        logger.info(f"Final training dataset: {len(training_data)} samples")
        
        # Create train/validation split
        train_data, val_data = train_test_split(
            training_data, 
            test_size=0.2, 
            random_state=42,
            stratify=training_data['rating'].round()  # Stratify by rounded rating
        )
        
        # Scale ratings
        scaler = MinMaxScaler()
        train_data['rating_scaled'] = scaler.fit_transform(train_data[['rating']])
        val_data['rating_scaled'] = scaler.transform(val_data[['rating']])
        
        # Save updated scaler
        with open('models/rating_scaler.pkl', 'wb') as f:
            pickle.dump(scaler, f)
        
        # Create backup of current model before retraining
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        if os.path.exists('models/recommendation_model.pt'):
            os.rename('models/recommendation_model.pt', f'models/backup_model_{timestamp}.pt')
            logger.info(f"Backed up original model to backup_model_{timestamp}.pt")
        
        # Train the model with feedback data
        logger.info("Starting model training with feedback data")
        
        # Retrain with conservative parameters to avoid overfitting on small feedback data
        model, history = train_model(
            train_data=train_data,
            val_data=val_data,
            movies_df=movies_df,
            genres=genres,
            device=device,
            epochs=10,        # Fewer epochs to prevent overfitting
            batch_size=32,    # Smaller batches for fine-tuning
            embedding_padding=50
        )
        
        if model is not None:
            logger.info("Model retraining completed successfully!")
            
            # Save retraining info
            retrain_metadata = {
                'retrained_at': datetime.now().isoformat(),
                'feedback_samples': len(processed_feedback),
                'total_samples': len(training_data),
                'original_samples': len(original_data) if not original_data.empty else 0,
                'user_weights_used': len(user_weights) > 0,
                'backup_model': f'backup_model_{timestamp}.pt'
            }
            
            with open('models/retrain_metadata.pkl', 'wb') as f:
                pickle.dump(retrain_metadata, f)
            
            return True
        else:
            logger.error("Model retraining failed")
            return False
            
    except Exception as e:
        logger.error(f"Error during retraining: {e}")
        return False


def export_feedback_summary():
    """Export a summary of feedback data for analysis
    
    Creates comprehensive statistics about user feedback patterns:
    - Feedback type distribution (positive/negative/ratings)
    - Most active users and their rating patterns
    - Temporal feedback trends
    
    Returns:
        bool: True if export succeeded, False otherwise
    """
    logger.info("Exporting feedback summary")
    
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        
        # Analyze feedback patterns by type
        cursor.execute('''
            SELECT 
                feedback_type,
                COUNT(*) as count,
                AVG(rating) as avg_rating
            FROM feedback 
            WHERE rating IS NOT NULL
            GROUP BY feedback_type
        ''')
        
        feedback_stats = cursor.fetchall()
        
        # Identify most active users for quality analysis
        cursor.execute('''
            SELECT 
                user_id,
                COUNT(*) as total_feedback,
                AVG(rating) as avg_rating,
                COUNT(DISTINCT movie_id) as unique_movies
            FROM user_ratings
            GROUP BY user_id
            ORDER BY total_feedback DESC
            LIMIT 20
        ''')
        
        user_activity = cursor.fetchall()
        
        conn.close()
        
        # Save summary
        summary = {
            'feedback_stats': feedback_stats,
            'top_users': user_activity,
            'export_time': datetime.now().isoformat()
        }
        
        with open('data/feedback_summary.pkl', 'wb') as f:
            pickle.dump(summary, f)
        
        logger.info("Feedback summary exported successfully")
        
        # Print summary
        print("\n=== FEEDBACK SUMMARY ===")
        print("\nFeedback by Type:")
        for stat in feedback_stats:
            print(f"  {stat['feedback_type']}: {stat['count']} samples, avg rating: {stat['avg_rating']:.2f}")
        
        print(f"\nTop {len(user_activity)} Most Active Users:")
        for user in user_activity:
            print(f"  User {user['user_id']}: {user['total_feedback']} ratings, avg: {user['avg_rating']:.2f}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error exporting feedback summary: {e}")
        return False


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Retrain model with Discord bot feedback')
    parser.add_argument('--summary-only', action='store_true', 
                       help='Only export feedback summary, do not retrain')
    parser.add_argument('--min-feedback', type=int, default=5,
                       help='Minimum feedback samples required for retraining')
    
    args = parser.parse_args()
    
    try:
        if args.summary_only:
            export_feedback_summary()
        else:
            # Check if we have enough feedback
            ratings_df, feedback_df, sentiment_df = load_feedback_data()
            if ratings_df is not None:
                total_feedback = len(ratings_df) + len(feedback_df)
                
                if total_feedback >= args.min_feedback:
                    logger.info(f"Found {total_feedback} feedback samples, starting retraining...")
                    
                    # Export summary first
                    export_feedback_summary()
                    
                    # Retrain model
                    success = retrain_model_with_feedback()
                    
                    if success:
                        print("\n🎉 Model retrained successfully with feedback data!")
                        print("The Discord bot will now use the updated model.")
                    else:
                        print("\n❌ Model retraining failed. Check logs for details.")
                        sys.exit(1)
                else:
                    print(f"\n⚠️  Insufficient feedback data ({total_feedback} samples).")
                    print(f"Need at least {args.min_feedback} samples to retrain.")
                    print("Export summary instead...")
                    export_feedback_summary()
            else:
                print("\n❌ Could not load feedback data from database.")
                sys.exit(1)
                
    except KeyboardInterrupt:
        logger.info("Retraining interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Unhandled exception: {e}")
        sys.exit(1)