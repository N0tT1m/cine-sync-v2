#!/usr/bin/env python3
"""
Advanced Feature Engineering Pipeline for CineSync v2
Implements temporal viewing patterns, seasonal preferences, and enhanced user behavior analysis
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import pickle
import logging
from collections import defaultdict, Counter
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import torch
import torch.nn as nn
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class TemporalPatternAnalyzer:
    """Analyze temporal viewing patterns including binge behavior and viewing velocity"""
    
    def __init__(self):
        # Temporal pattern configuration
        self.binge_threshold_hours = 6  # Sessions within 6 hours considered binge
        self.velocity_window_days = 30  # Calculate velocity over 30 days
        self.pattern_cache = {}
        
    def extract_temporal_features(self, user_viewing_history: List[Dict]) -> Dict[str, float]:
        """Extract comprehensive temporal viewing features"""
        
        if not user_viewing_history:
            return self._get_default_temporal_features()
        
        # Convert to DataFrame for easier processing
        df = pd.DataFrame(user_viewing_history)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp')
        
        features = {}
        
        # 1. Binge Behavior Analysis
        features.update(self._analyze_binge_behavior(df))
        
        # 2. Viewing Velocity Analysis
        features.update(self._analyze_viewing_velocity(df))
        
        # 3. Time-of-Day Patterns
        features.update(self._analyze_time_of_day_patterns(df))
        
        # 4. Day-of-Week Patterns
        features.update(self._analyze_day_of_week_patterns(df))
        
        # 5. Session Duration Patterns
        features.update(self._analyze_session_patterns(df))
        
        # 6. Content Completion Patterns
        features.update(self._analyze_completion_patterns(df))
        
        return features
    
    def _analyze_binge_behavior(self, df: pd.DataFrame) -> Dict[str, float]:
        """Analyze binge watching behavior"""
        binge_features = {}
        
        # Group consecutive viewing sessions
        df['time_diff'] = df['timestamp'].diff().dt.total_seconds() / 3600  # Hours
        
        # Identify binge sessions (consecutive views within threshold)
        binge_sessions = []
        current_session = []
        
        for idx, row in df.iterrows():
            if len(current_session) == 0 or row['time_diff'] <= self.binge_threshold_hours:
                current_session.append(row)
            else:
                if len(current_session) > 1:
                    binge_sessions.append(current_session)
                current_session = [row]
        
        if len(current_session) > 1:
            binge_sessions.append(current_session)
        
        # Calculate binge metrics
        total_sessions = len(df)
        binge_session_count = len(binge_sessions)
        
        binge_features['binge_frequency'] = binge_session_count / max(total_sessions, 1)
        binge_features['avg_binge_length'] = np.mean([len(session) for session in binge_sessions]) if binge_sessions else 0
        binge_features['max_binge_length'] = max([len(session) for session in binge_sessions]) if binge_sessions else 0
        
        # Binge content type preferences
        if binge_sessions:
            binge_content_types = []
            for session in binge_sessions:
                content_types = [item.get('content_type', 'unknown') for item in session]
                binge_content_types.extend(content_types)
            
            content_type_counts = Counter(binge_content_types)
            total_binge_items = len(binge_content_types)
            
            binge_features['binge_movie_preference'] = content_type_counts.get('movie', 0) / max(total_binge_items, 1)
            binge_features['binge_tv_preference'] = content_type_counts.get('tv', 0) / max(total_binge_items, 1)
        else:
            binge_features['binge_movie_preference'] = 0.0
            binge_features['binge_tv_preference'] = 0.0
        
        return binge_features
    
    def _analyze_viewing_velocity(self, df: pd.DataFrame) -> Dict[str, float]:
        """Analyze viewing velocity (content consumption rate)"""
        velocity_features = {}
        
        # Calculate daily viewing counts
        df['date'] = df['timestamp'].dt.date
        daily_counts = df.groupby('date').size()
        
        # Viewing velocity metrics
        velocity_features['avg_daily_velocity'] = daily_counts.mean()
        velocity_features['max_daily_velocity'] = daily_counts.max()
        velocity_features['velocity_std'] = daily_counts.std()
        velocity_features['velocity_consistency'] = 1 / (1 + velocity_features['velocity_std'])  # Higher = more consistent
        
        # Weekly patterns
        weekly_velocity = df.groupby(df['timestamp'].dt.isocalendar().week).size()
        velocity_features['avg_weekly_velocity'] = weekly_velocity.mean()
        velocity_features['weekly_velocity_trend'] = self._calculate_trend(weekly_velocity.values)
        
        # Acceleration (change in velocity over time)
        if len(daily_counts) > 7:
            recent_velocity = daily_counts.tail(7).mean()
            older_velocity = daily_counts.head(7).mean()
            velocity_features['velocity_acceleration'] = (recent_velocity - older_velocity) / max(older_velocity, 1)
        else:
            velocity_features['velocity_acceleration'] = 0.0
        
        return velocity_features
    
    def _analyze_time_of_day_patterns(self, df: pd.DataFrame) -> Dict[str, float]:
        """Analyze time-of-day viewing preferences"""
        time_features = {}
        
        # Extract hour of day
        df['hour'] = df['timestamp'].dt.hour
        
        # Define time slots
        time_slots = {
            'morning': (6, 12),    # 6 AM - 12 PM
            'afternoon': (12, 18), # 12 PM - 6 PM
            'evening': (18, 22),   # 6 PM - 10 PM
            'night': (22, 6)       # 10 PM - 6 AM (next day)
        }
        
        # Calculate viewing distribution across time slots
        total_views = len(df)
        for slot_name, (start_hour, end_hour) in time_slots.items():
            if start_hour < end_hour:
                slot_views = len(df[(df['hour'] >= start_hour) & (df['hour'] < end_hour)])
            else:  # Night slot spans midnight
                slot_views = len(df[(df['hour'] >= start_hour) | (df['hour'] < end_hour)])
            
            time_features[f'{slot_name}_viewing_preference'] = slot_views / max(total_views, 1)
        
        # Peak viewing hour
        hour_counts = df['hour'].value_counts()
        time_features['peak_viewing_hour'] = hour_counts.index[0] if len(hour_counts) > 0 else 20
        time_features['peak_hour_concentration'] = hour_counts.iloc[0] / max(total_views, 1) if len(hour_counts) > 0 else 0
        
        # Late night viewing tendency
        late_night_views = len(df[df['hour'].isin([22, 23, 0, 1, 2, 3])])
        time_features['late_night_tendency'] = late_night_views / max(total_views, 1)
        
        return time_features
    
    def _analyze_day_of_week_patterns(self, df: pd.DataFrame) -> Dict[str, float]:
        """Analyze day-of-week viewing patterns"""
        day_features = {}
        
        # Extract day of week (0=Monday, 6=Sunday)
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        
        # Weekend vs weekday preferences
        weekday_views = len(df[df['day_of_week'] < 5])  # Monday-Friday
        weekend_views = len(df[df['day_of_week'] >= 5])  # Saturday-Sunday
        total_views = len(df)
        
        day_features['weekday_preference'] = weekday_views / max(total_views, 1)
        day_features['weekend_preference'] = weekend_views / max(total_views, 1)
        
        # Peak viewing day
        day_counts = df['day_of_week'].value_counts()
        if len(day_counts) > 0:
            day_features['peak_viewing_day'] = day_counts.index[0]
            day_features['viewing_day_consistency'] = day_counts.iloc[0] / max(total_views, 1)
        else:
            day_features['peak_viewing_day'] = 5  # Default to Saturday
            day_features['viewing_day_consistency'] = 0.0
        
        return day_features
    
    def _analyze_session_patterns(self, df: pd.DataFrame) -> Dict[str, float]:
        """Analyze viewing session patterns"""
        session_features = {}
        
        # Group into sessions based on time gaps
        session_gap_hours = 2  # 2 hours gap defines new session
        df['time_diff_hours'] = df['timestamp'].diff().dt.total_seconds() / 3600
        
        sessions = []
        current_session = []
        
        for idx, row in df.iterrows():
            if len(current_session) == 0 or row['time_diff_hours'] <= session_gap_hours:
                current_session.append(row)
            else:
                sessions.append(current_session)
                current_session = [row]
        
        if current_session:
            sessions.append(current_session)
        
        # Session metrics
        session_lengths = [len(session) for session in sessions]
        
        session_features['avg_session_length'] = np.mean(session_lengths)
        session_features['max_session_length'] = max(session_lengths) if session_lengths else 0
        session_features['session_length_std'] = np.std(session_lengths)
        session_features['total_sessions'] = len(sessions)
        
        # Session frequency
        if len(df) > 0:
            time_span_days = (df['timestamp'].max() - df['timestamp'].min()).days + 1
            session_features['sessions_per_day'] = len(sessions) / max(time_span_days, 1)
        else:
            session_features['sessions_per_day'] = 0.0
        
        return session_features
    
    def _analyze_completion_patterns(self, df: pd.DataFrame) -> Dict[str, float]:
        """Analyze content completion patterns"""
        completion_features = {}
        
        # Check if completion data is available
        if 'completion_rate' in df.columns:
            completion_rates = df['completion_rate'].dropna()
            
            completion_features['avg_completion_rate'] = completion_rates.mean()
            completion_features['completion_consistency'] = 1 - completion_rates.std()  # Higher = more consistent
            completion_features['high_completion_rate'] = (completion_rates > 0.8).mean()  # Fraction of highly completed content
        else:
            # Use watch_duration if available
            if 'watch_duration' in df.columns and 'content_duration' in df.columns:
                completion_rates = df['watch_duration'] / df['content_duration'].clip(lower=1)  # Avoid division by zero
                completion_features['avg_completion_rate'] = completion_rates.mean()
                completion_features['completion_consistency'] = 1 - completion_rates.std()
                completion_features['high_completion_rate'] = (completion_rates > 0.8).mean()
            else:
                # Default values when no completion data available
                completion_features['avg_completion_rate'] = 0.7  # Assume reasonable completion
                completion_features['completion_consistency'] = 0.5
                completion_features['high_completion_rate'] = 0.5
        
        return completion_features
    
    def _calculate_trend(self, values: np.ndarray) -> float:
        """Calculate trend in time series data"""
        if len(values) < 2:
            return 0.0
        
        x = np.arange(len(values))
        slope, _, _, _, _ = stats.linregress(x, values)
        return float(slope)
    
    def _get_default_temporal_features(self) -> Dict[str, float]:
        """Return default temporal features for users with no history"""
        return {
            'binge_frequency': 0.0,
            'avg_binge_length': 0.0,
            'max_binge_length': 0.0,
            'binge_movie_preference': 0.5,
            'binge_tv_preference': 0.5,
            'avg_daily_velocity': 1.0,
            'max_daily_velocity': 1.0,
            'velocity_std': 0.0,
            'velocity_consistency': 1.0,
            'avg_weekly_velocity': 7.0,
            'weekly_velocity_trend': 0.0,
            'velocity_acceleration': 0.0,
            'morning_viewing_preference': 0.1,
            'afternoon_viewing_preference': 0.2,
            'evening_viewing_preference': 0.5,
            'night_viewing_preference': 0.2,
            'peak_viewing_hour': 20.0,
            'peak_hour_concentration': 0.3,
            'late_night_tendency': 0.2,
            'weekday_preference': 0.7,
            'weekend_preference': 0.3,
            'peak_viewing_day': 5.0,
            'viewing_day_consistency': 0.3,
            'avg_session_length': 2.0,
            'max_session_length': 5.0,
            'session_length_std': 1.0,
            'total_sessions': 10.0,
            'sessions_per_day': 0.5,
            'avg_completion_rate': 0.7,
            'completion_consistency': 0.5,
            'high_completion_rate': 0.5
        }


class SeasonalPreferenceAnalyzer:
    """Analyze seasonal viewing preferences and trends"""
    
    def __init__(self):
        self.seasonal_cache = {}
        
        # Define seasons
        self.seasons = {
            'spring': [3, 4, 5],   # March, April, May
            'summer': [6, 7, 8],   # June, July, August
            'fall': [9, 10, 11],   # September, October, November
            'winter': [12, 1, 2]   # December, January, February
        }
        
        # Holiday periods that affect viewing
        self.holiday_periods = {
            'christmas': (12, 15, 31),  # Dec 15-31
            'new_year': (1, 1, 7),      # Jan 1-7
            'summer_break': (6, 15, 8, 15),  # June 15 - Aug 15
            'thanksgiving': (11, 20, 30)  # Nov 20-30
        }
    
    def extract_seasonal_features(self, user_viewing_history: List[Dict]) -> Dict[str, float]:
        """Extract seasonal viewing preference features"""
        
        if not user_viewing_history:
            return self._get_default_seasonal_features()
        
        df = pd.DataFrame(user_viewing_history)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        features = {}
        
        # 1. Basic seasonal preferences
        features.update(self._analyze_seasonal_preferences(df))
        
        # 2. Holiday viewing patterns
        features.update(self._analyze_holiday_patterns(df))
        
        # 3. Genre-season correlations
        features.update(self._analyze_seasonal_genre_preferences(df))
        
        # 4. Weather-influenced viewing (proxy through seasonal intensity)
        features.update(self._analyze_weather_proxy_patterns(df))
        
        return features
    
    def _analyze_seasonal_preferences(self, df: pd.DataFrame) -> Dict[str, float]:
        """Analyze basic seasonal viewing preferences"""
        seasonal_features = {}
        
        # Extract month from timestamp
        df['month'] = df['timestamp'].dt.month
        
        # Calculate viewing distribution by season
        total_views = len(df)
        
        for season_name, months in self.seasons.items():
            season_views = len(df[df['month'].isin(months)])
            seasonal_features[f'{season_name}_viewing_preference'] = season_views / max(total_views, 1)
        
        # Find peak season
        season_counts = {}
        for season_name, months in self.seasons.items():
            season_counts[season_name] = len(df[df['month'].isin(months)])
        
        if season_counts:
            peak_season = max(season_counts, key=season_counts.get)
            seasonal_features['peak_viewing_season'] = list(self.seasons.keys()).index(peak_season)
            seasonal_features['seasonal_concentration'] = season_counts[peak_season] / max(total_views, 1)
        else:
            seasonal_features['peak_viewing_season'] = 0
            seasonal_features['seasonal_concentration'] = 0.25
        
        # Seasonal consistency (how evenly distributed across seasons)
        season_values = list(season_counts.values())
        if len(season_values) > 1:
            seasonal_features['seasonal_diversity'] = 1 - np.std(season_values) / (np.mean(season_values) + 1)
        else:
            seasonal_features['seasonal_diversity'] = 0.5
        
        return seasonal_features
    
    def _analyze_holiday_patterns(self, df: pd.DataFrame) -> Dict[str, float]:
        """Analyze viewing patterns during holidays"""
        holiday_features = {}
        
        df['month'] = df['timestamp'].dt.month
        df['day'] = df['timestamp'].dt.day
        
        total_views = len(df)
        
        # Check viewing during each holiday period
        for holiday_name, period in self.holiday_periods.items():
            if len(period) == 3:  # Single month period
                month, start_day, end_day = period
                holiday_views = len(df[(df['month'] == month) & 
                                     (df['day'] >= start_day) & 
                                     (df['day'] <= end_day)])
            else:  # Multi-month period
                start_month, start_day, end_month, end_day = period
                holiday_views = len(df[((df['month'] == start_month) & (df['day'] >= start_day)) |
                                     ((df['month'] == end_month) & (df['day'] <= end_day)) |
                                     ((df['month'] > start_month) & (df['month'] < end_month))])
            
            holiday_features[f'{holiday_name}_viewing_boost'] = holiday_views / max(total_views, 1)
        
        # Overall holiday viewing tendency
        all_holiday_views = sum([
            holiday_features[f'{holiday}_viewing_boost'] for holiday in self.holiday_periods.keys()
        ])
        holiday_features['overall_holiday_tendency'] = all_holiday_views / len(self.holiday_periods)
        
        return holiday_features
    
    def _analyze_seasonal_genre_preferences(self, df: pd.DataFrame) -> Dict[str, float]:
        """Analyze how genre preferences change with seasons"""
        genre_seasonal_features = {}
        
        if 'genres' not in df.columns:
            return {'seasonal_genre_adaptation': 0.5}
        
        df['month'] = df['timestamp'].dt.month
        df['season'] = df['month'].apply(self._month_to_season)
        
        # Common seasonal genre associations
        seasonal_genre_preferences = {
            'winter': ['Drama', 'Thriller', 'Mystery', 'Horror'],
            'spring': ['Romance', 'Comedy', 'Family', 'Animation'],
            'summer': ['Action', 'Adventure', 'Comedy', 'Sci-Fi'],
            'fall': ['Drama', 'Crime', 'Thriller', 'Documentary']
        }
        
        # Calculate how well user's viewing aligns with seasonal expectations
        alignment_scores = []
        
        for season in self.seasons.keys():
            season_views = df[df['season'] == season]
            if len(season_views) == 0:
                continue
            
            expected_genres = seasonal_genre_preferences[season]
            
            # Count views of expected genres in this season
            seasonal_genre_matches = 0
            total_seasonal_views = len(season_views)
            
            for _, row in season_views.iterrows():
                user_genres = row['genres'].split('|') if pd.notna(row['genres']) else []
                if any(genre in expected_genres for genre in user_genres):
                    seasonal_genre_matches += 1
            
            if total_seasonal_views > 0:
                alignment_scores.append(seasonal_genre_matches / total_seasonal_views)
        
        genre_seasonal_features['seasonal_genre_alignment'] = np.mean(alignment_scores) if alignment_scores else 0.5
        genre_seasonal_features['seasonal_genre_adaptation'] = np.std(alignment_scores) if len(alignment_scores) > 1 else 0.5
        
        return genre_seasonal_features
    
    def _analyze_weather_proxy_patterns(self, df: pd.DataFrame) -> Dict[str, float]:
        """Analyze viewing patterns that might correlate with weather (using seasonal proxy)"""
        weather_features = {}
        
        df['month'] = df['timestamp'].dt.month
        
        # Define "indoor" vs "outdoor" months (proxy for weather influence)
        indoor_months = [11, 12, 1, 2, 3]  # Late fall through early spring
        outdoor_months = [4, 5, 6, 7, 8, 9, 10]  # Late spring through early fall
        
        indoor_views = len(df[df['month'].isin(indoor_months)])
        outdoor_views = len(df[df['month'].isin(outdoor_months)])
        total_views = len(df)
        
        weather_features['indoor_season_preference'] = indoor_views / max(total_views, 1)
        weather_features['outdoor_season_preference'] = outdoor_views / max(total_views, 1)
        
        # Weather sensitivity (how much viewing varies between indoor/outdoor seasons)
        if total_views > 0:
            expected_indoor = total_views * (5/12)  # 5 indoor months out of 12
            weather_features['weather_sensitivity'] = abs(indoor_views - expected_indoor) / total_views
        else:
            weather_features['weather_sensitivity'] = 0.0
        
        return weather_features
    
    def _month_to_season(self, month: int) -> str:
        """Convert month number to season name"""
        for season, months in self.seasons.items():
            if month in months:
                return season
        return 'spring'  # Default
    
    def _get_default_seasonal_features(self) -> Dict[str, float]:
        """Return default seasonal features for users with no history"""
        return {
            'spring_viewing_preference': 0.25,
            'summer_viewing_preference': 0.25,
            'fall_viewing_preference': 0.25,
            'winter_viewing_preference': 0.25,
            'peak_viewing_season': 2,  # Fall
            'seasonal_concentration': 0.3,
            'seasonal_diversity': 0.7,
            'christmas_viewing_boost': 0.1,
            'new_year_viewing_boost': 0.05,
            'summer_break_viewing_boost': 0.08,
            'thanksgiving_viewing_boost': 0.04,
            'overall_holiday_tendency': 0.07,
            'seasonal_genre_alignment': 0.5,
            'seasonal_genre_adaptation': 0.5,
            'indoor_season_preference': 0.42,
            'outdoor_season_preference': 0.58,
            'weather_sensitivity': 0.1
        }


class UserEngagementAnalyzer:
    """Analyze user engagement metrics and patterns"""
    
    def __init__(self):
        self.engagement_cache = {}
        self.rating_scaler = StandardScaler()
        
    def extract_engagement_features(self, user_data: Dict[str, Any]) -> Dict[str, float]:
        """Extract comprehensive user engagement features"""
        
        features = {}
        
        # 1. Rating behavior analysis
        features.update(self._analyze_rating_behavior(user_data.get('ratings', [])))
        
        # 2. Review writing patterns
        features.update(self._analyze_review_patterns(user_data.get('reviews', [])))
        
        # 3. Social interaction metrics
        features.update(self._analyze_social_engagement(user_data.get('social_interactions', [])))
        
        # 4. Platform usage patterns
        features.update(self._analyze_platform_usage(user_data.get('usage_history', [])))
        
        # 5. Content discovery patterns
        features.update(self._analyze_discovery_patterns(user_data.get('discovery_history', [])))
        
        return features
    
    def _analyze_rating_behavior(self, ratings: List[Dict]) -> Dict[str, float]:
        """Analyze user rating patterns and tendencies"""
        rating_features = {}
        
        if not ratings:
            return self._get_default_rating_features()
        
        rating_values = [r['rating'] for r in ratings if 'rating' in r]
        
        if not rating_values:
            return self._get_default_rating_features()
        
        # Basic rating statistics
        rating_features['avg_rating'] = np.mean(rating_values)
        rating_features['rating_std'] = np.std(rating_values)
        rating_features['rating_range'] = max(rating_values) - min(rating_values)
        rating_features['total_ratings'] = len(rating_values)
        
        # Rating distribution
        rating_counts = Counter(rating_values)
        total_ratings = len(rating_values)
        
        for rating in [1, 2, 3, 4, 5]:
            rating_features[f'rating_{rating}_frequency'] = rating_counts.get(rating, 0) / total_ratings
        
        # Rating tendencies
        rating_features['positive_rating_tendency'] = sum(1 for r in rating_values if r >= 4) / total_ratings
        rating_features['negative_rating_tendency'] = sum(1 for r in rating_values if r <= 2) / total_ratings
        rating_features['rating_generosity'] = rating_features['avg_rating'] / 5.0
        
        # Rating consistency over time
        if len(ratings) > 10:
            recent_ratings = sorted(ratings, key=lambda x: x.get('timestamp', ''))[-10:]
            older_ratings = sorted(ratings, key=lambda x: x.get('timestamp', ''))[:-10]
            
            recent_avg = np.mean([r['rating'] for r in recent_ratings])
            older_avg = np.mean([r['rating'] for r in older_ratings])
            
            rating_features['rating_trend'] = (recent_avg - older_avg) / 5.0
            rating_features['rating_consistency'] = 1 - abs(rating_features['rating_trend'])
        else:
            rating_features['rating_trend'] = 0.0
            rating_features['rating_consistency'] = 0.8
        
        return rating_features
    
    def _analyze_review_patterns(self, reviews: List[Dict]) -> Dict[str, float]:
        """Analyze review writing patterns"""
        review_features = {}
        
        if not reviews:
            return {
                'review_writing_frequency': 0.0,
                'avg_review_length': 0.0,
                'review_sentiment_avg': 0.0,
                'review_detail_level': 0.0
            }
        
        # Review frequency
        review_features['review_writing_frequency'] = len(reviews) / 100  # Normalize
        
        # Review length analysis
        review_lengths = [len(r.get('text', '')) for r in reviews]
        review_features['avg_review_length'] = np.mean(review_lengths)
        review_features['review_length_consistency'] = 1 - (np.std(review_lengths) / (np.mean(review_lengths) + 1))
        
        # Review detail level (proxy)
        review_features['review_detail_level'] = min(np.mean(review_lengths) / 500, 1.0)  # Normalize to 0-1
        
        # Review sentiment (simplified analysis)
        positive_words = ['great', 'excellent', 'amazing', 'love', 'perfect', 'wonderful', 'fantastic']
        negative_words = ['bad', 'terrible', 'awful', 'hate', 'boring', 'waste', 'disappointing']
        
        sentiment_scores = []
        for review in reviews:
            text = review.get('text', '').lower()
            positive_count = sum(1 for word in positive_words if word in text)
            negative_count = sum(1 for word in negative_words if word in text)
            
            if positive_count + negative_count > 0:
                sentiment = (positive_count - negative_count) / (positive_count + negative_count)
            else:
                sentiment = 0.0
            
            sentiment_scores.append(sentiment)
        
        review_features['review_sentiment_avg'] = np.mean(sentiment_scores) if sentiment_scores else 0.0
        review_features['review_sentiment_consistency'] = 1 - np.std(sentiment_scores) if len(sentiment_scores) > 1 else 0.5
        
        return review_features
    
    def _analyze_social_engagement(self, social_interactions: List[Dict]) -> Dict[str, float]:
        """Analyze social engagement patterns"""
        social_features = {}
        
        if not social_interactions:
            return {
                'social_engagement_level': 0.0,
                'recommendation_sharing_frequency': 0.0,
                'community_participation': 0.0,
                'influence_score': 0.0
            }
        
        # Basic social metrics
        interaction_counts = Counter([i.get('type', 'unknown') for i in social_interactions])
        total_interactions = len(social_interactions)
        
        social_features['social_engagement_level'] = min(total_interactions / 50, 1.0)  # Normalize
        social_features['recommendation_sharing_frequency'] = interaction_counts.get('share', 0) / max(total_interactions, 1)
        social_features['community_participation'] = interaction_counts.get('comment', 0) / max(total_interactions, 1)
        
        # Influence score (based on likes, shares received)
        influence_actions = ['like_received', 'share_received', 'follow_received']
        influence_count = sum(interaction_counts.get(action, 0) for action in influence_actions)
        social_features['influence_score'] = min(influence_count / 20, 1.0)  # Normalize
        
        return social_features
    
    def _analyze_platform_usage(self, usage_history: List[Dict]) -> Dict[str, float]:
        """Analyze platform usage patterns"""
        usage_features = {}
        
        if not usage_history:
            return {
                'platform_loyalty': 0.7,
                'feature_exploration': 0.5,
                'session_depth': 0.5,
                'return_frequency': 0.5
            }
        
        # Platform loyalty (how often they return)
        unique_days = len(set(h.get('date') for h in usage_history if h.get('date')))
        total_sessions = len(usage_history)
        
        usage_features['platform_loyalty'] = min(unique_days / 30, 1.0)  # Normalize to 30 days
        usage_features['session_frequency'] = total_sessions / max(unique_days, 1)
        
        # Feature exploration
        unique_features = len(set(h.get('feature_used') for h in usage_history if h.get('feature_used')))
        usage_features['feature_exploration'] = min(unique_features / 10, 1.0)  # Normalize
        
        # Session depth
        session_actions = [h.get('actions_per_session', 1) for h in usage_history]
        usage_features['session_depth'] = min(np.mean(session_actions) / 10, 1.0)  # Normalize
        
        # Return frequency
        if len(usage_history) > 1:
            timestamps = [datetime.fromisoformat(h['timestamp']) for h in usage_history if 'timestamp' in h]
            if len(timestamps) > 1:
                timestamps.sort()
                time_diffs = [(timestamps[i+1] - timestamps[i]).days for i in range(len(timestamps)-1)]
                avg_return_days = np.mean(time_diffs)
                usage_features['return_frequency'] = max(0, 1 - (avg_return_days / 7))  # Weekly return = 1.0
            else:
                usage_features['return_frequency'] = 0.5
        else:
            usage_features['return_frequency'] = 0.5
        
        return usage_features
    
    def _analyze_discovery_patterns(self, discovery_history: List[Dict]) -> Dict[str, float]:
        """Analyze content discovery patterns"""
        discovery_features = {}
        
        if not discovery_history:
            return {
                'discovery_openness': 0.5,
                'genre_exploration': 0.5,
                'recommendation_acceptance': 0.5,
                'serendipity_seeking': 0.5
            }
        
        # Discovery openness (trying new content)
        discovery_methods = Counter([d.get('discovery_method', 'unknown') for d in discovery_history])
        total_discoveries = len(discovery_history)
        
        # Methods that indicate openness to discovery
        open_methods = ['recommendation', 'trending', 'random', 'similar_users']
        closed_methods = ['search', 'direct', 'bookmark']
        
        open_discoveries = sum(discovery_methods.get(method, 0) for method in open_methods)
        discovery_features['discovery_openness'] = open_discoveries / max(total_discoveries, 1)
        
        # Genre exploration
        genres_discovered = set()
        for d in discovery_history:
            if d.get('genres'):
                genres_discovered.update(d['genres'])
        
        discovery_features['genre_exploration'] = min(len(genres_discovered) / 15, 1.0)  # Normalize
        
        # Recommendation acceptance rate
        recommended_items = [d for d in discovery_history if d.get('discovery_method') == 'recommendation']
        if recommended_items:
            accepted_recommendations = sum(1 for d in recommended_items if d.get('user_action') in ['watched', 'rated', 'liked'])
            discovery_features['recommendation_acceptance'] = accepted_recommendations / len(recommended_items)
        else:
            discovery_features['recommendation_acceptance'] = 0.5
        
        # Serendipity seeking (random/explore behavior)
        serendipity_methods = ['random', 'trending', 'explore']
        serendipity_discoveries = sum(discovery_methods.get(method, 0) for method in serendipity_methods)
        discovery_features['serendipity_seeking'] = serendipity_discoveries / max(total_discoveries, 1)
        
        return discovery_features
    
    def _get_default_rating_features(self) -> Dict[str, float]:
        """Default rating features for users with no ratings"""
        return {
            'avg_rating': 3.5,
            'rating_std': 1.0,
            'rating_range': 4.0,
            'total_ratings': 0.0,
            'rating_1_frequency': 0.05,
            'rating_2_frequency': 0.10,
            'rating_3_frequency': 0.30,
            'rating_4_frequency': 0.35,
            'rating_5_frequency': 0.20,
            'positive_rating_tendency': 0.55,
            'negative_rating_tendency': 0.15,
            'rating_generosity': 0.70,
            'rating_trend': 0.0,
            'rating_consistency': 0.8
        }


class AdvancedFeatureEngineeringPipeline:
    """Complete advanced feature engineering pipeline"""
    
    def __init__(self):
        self.temporal_analyzer = TemporalPatternAnalyzer()
        self.seasonal_analyzer = SeasonalPreferenceAnalyzer()
        self.engagement_analyzer = UserEngagementAnalyzer()
        
        # Feature processing components
        self.feature_scaler = StandardScaler()
        self.feature_selector = None
        self.is_fitted = False
        
    def extract_all_features(self, user_data: Dict[str, Any]) -> Dict[str, float]:
        """Extract all advanced features for a user"""
        
        all_features = {}
        
        # 1. Temporal features
        viewing_history = user_data.get('viewing_history', [])
        temporal_features = self.temporal_analyzer.extract_temporal_features(viewing_history)
        all_features.update({f'temporal_{k}': v for k, v in temporal_features.items()})
        
        # 2. Seasonal features
        seasonal_features = self.seasonal_analyzer.extract_seasonal_features(viewing_history)
        all_features.update({f'seasonal_{k}': v for k, v in seasonal_features.items()})
        
        # 3. Engagement features
        engagement_features = self.engagement_analyzer.extract_engagement_features(user_data)
        all_features.update({f'engagement_{k}': v for k, v in engagement_features.items()})
        
        # 4. Cross-feature interactions
        interaction_features = self._compute_feature_interactions(all_features)
        all_features.update(interaction_features)
        
        return all_features
    
    def _compute_feature_interactions(self, features: Dict[str, float]) -> Dict[str, float]:
        """Compute meaningful feature interactions"""
        interactions = {}
        
        # Temporal-Seasonal interactions
        if 'temporal_binge_frequency' in features and 'seasonal_winter_viewing_preference' in features:
            interactions['winter_binge_tendency'] = features['temporal_binge_frequency'] * features['seasonal_winter_viewing_preference']
        
        # Engagement-Temporal interactions
        if 'engagement_platform_loyalty' in features and 'temporal_velocity_consistency' in features:
            interactions['loyal_consistent_viewer'] = features['engagement_platform_loyalty'] * features['temporal_velocity_consistency']
        
        # Seasonal-Engagement interactions
        if 'seasonal_seasonal_genre_alignment' in features and 'engagement_discovery_openness' in features:
            interactions['seasonal_discovery_balance'] = features['seasonal_seasonal_genre_alignment'] * (1 - features['engagement_discovery_openness'])
        
        return interactions
    
    def fit_transform(self, user_data_list: List[Dict[str, Any]]) -> np.ndarray:
        """Fit the pipeline and transform user data"""
        
        # Extract features for all users
        all_user_features = []
        feature_names = None
        
        for user_data in user_data_list:
            user_features = self.extract_all_features(user_data)
            
            if feature_names is None:
                feature_names = sorted(user_features.keys())
            
            # Ensure consistent feature ordering
            feature_vector = [user_features.get(name, 0.0) for name in feature_names]
            all_user_features.append(feature_vector)
        
        self.feature_names = feature_names
        feature_matrix = np.array(all_user_features)
        
        # Fit scaler and transform
        scaled_features = self.feature_scaler.fit_transform(feature_matrix)
        
        # Feature selection (optional)
        if scaled_features.shape[1] > 50:  # If too many features
            from sklearn.feature_selection import SelectKBest, f_regression
            self.feature_selector = SelectKBest(score_func=f_regression, k=50)
            scaled_features = self.feature_selector.fit_transform(scaled_features, np.random.random(len(user_data_list)))  # Placeholder target
        
        self.is_fitted = True
        return scaled_features
    
    def transform(self, user_data: Dict[str, Any]) -> np.ndarray:
        """Transform single user data using fitted pipeline"""
        
        if not self.is_fitted:
            raise ValueError("Pipeline must be fitted before transform")
        
        user_features = self.extract_all_features(user_data)
        feature_vector = [user_features.get(name, 0.0) for name in self.feature_names]
        feature_vector = np.array(feature_vector).reshape(1, -1)
        
        # Scale features
        scaled_features = self.feature_scaler.transform(feature_vector)
        
        # Apply feature selection if fitted
        if self.feature_selector is not None:
            scaled_features = self.feature_selector.transform(scaled_features)
        
        return scaled_features
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores"""
        
        if not self.is_fitted:
            return {}
        
        feature_names = self.feature_names
        if self.feature_selector is not None:
            # Get selected features
            selected_features = self.feature_selector.get_support()
            feature_names = [name for name, selected in zip(self.feature_names, selected_features) if selected]
            importance_scores = self.feature_selector.scores_[selected_features]
        else:
            # Use variance as proxy for importance
            importance_scores = np.var(self.feature_scaler.transform(np.eye(len(feature_names))), axis=0)
        
        # Normalize importance scores
        importance_scores = importance_scores / np.sum(importance_scores)
        
        return dict(zip(feature_names, importance_scores))
    
    def save_pipeline(self, filepath: str):
        """Save the fitted pipeline"""
        pipeline_data = {
            'temporal_analyzer': self.temporal_analyzer,
            'seasonal_analyzer': self.seasonal_analyzer,
            'engagement_analyzer': self.engagement_analyzer,
            'feature_scaler': self.feature_scaler,
            'feature_selector': self.feature_selector,
            'feature_names': getattr(self, 'feature_names', None),
            'is_fitted': self.is_fitted
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(pipeline_data, f)
        
        logger.info(f"Advanced feature engineering pipeline saved to {filepath}")
    
    def load_pipeline(self, filepath: str):
        """Load a fitted pipeline"""
        with open(filepath, 'rb') as f:
            pipeline_data = pickle.load(f)
        
        self.temporal_analyzer = pipeline_data['temporal_analyzer']
        self.seasonal_analyzer = pipeline_data['seasonal_analyzer']
        self.engagement_analyzer = pipeline_data['engagement_analyzer']
        self.feature_scaler = pipeline_data['feature_scaler']
        self.feature_selector = pipeline_data['feature_selector']
        self.feature_names = pipeline_data['feature_names']
        self.is_fitted = pipeline_data['is_fitted']
        
        logger.info(f"Advanced feature engineering pipeline loaded from {filepath}")


# Example usage and testing
if __name__ == "__main__":
    # Initialize pipeline
    pipeline = AdvancedFeatureEngineeringPipeline()
    
    # Example user data
    example_user_data = {
        'user_id': 12345,
        'viewing_history': [
            {
                'content_id': 1,
                'timestamp': '2024-01-15T20:30:00',
                'content_type': 'movie',
                'genres': 'Action|Thriller',
                'watch_duration': 7200,  # 2 hours
                'content_duration': 7800,  # 2.17 hours
                'completion_rate': 0.92
            },
            {
                'content_id': 2,
                'timestamp': '2024-01-15T22:45:00',
                'content_type': 'tv',
                'genres': 'Drama|Crime',
                'watch_duration': 3600,  # 1 hour
                'content_duration': 3600,  # 1 hour
                'completion_rate': 1.0
            }
        ],
        'ratings': [
            {'content_id': 1, 'rating': 4.5, 'timestamp': '2024-01-15T22:35:00'},
            {'content_id': 2, 'rating': 5.0, 'timestamp': '2024-01-15T23:45:00'}
        ],
        'reviews': [
            {'content_id': 1, 'text': 'Great action movie with excellent cinematography!', 'timestamp': '2024-01-16T10:00:00'}
        ],
        'social_interactions': [
            {'type': 'share', 'content_id': 1, 'timestamp': '2024-01-16T10:05:00'},
            {'type': 'like_received', 'content_id': 1, 'timestamp': '2024-01-16T15:20:00'}
        ],
        'usage_history': [
            {'date': '2024-01-15', 'feature_used': 'recommendations', 'actions_per_session': 5, 'timestamp': '2024-01-15T20:00:00'},
            {'date': '2024-01-16', 'feature_used': 'search', 'actions_per_session': 3, 'timestamp': '2024-01-16T19:30:00'}
        ],
        'discovery_history': [
            {'content_id': 1, 'discovery_method': 'recommendation', 'genres': ['Action', 'Thriller'], 'user_action': 'watched'},
            {'content_id': 2, 'discovery_method': 'trending', 'genres': ['Drama', 'Crime'], 'user_action': 'rated'}
        ]
    }
    
    # Extract features for single user
    features = pipeline.extract_all_features(example_user_data)
    
    print("Extracted Advanced Features:")
    for category in ['temporal', 'seasonal', 'engagement']:
        print(f"\n{category.upper()} FEATURES:")
        category_features = {k: v for k, v in features.items() if k.startswith(category)}
        for k, v in sorted(category_features.items()):
            print(f"  {k}: {v:.4f}")
    
    print(f"\nTotal features extracted: {len(features)}")
    
    # Example of fitting pipeline on multiple users (for demonstration)
    multiple_user_data = [example_user_data for _ in range(10)]  # Simulate 10 users
    
    try:
        transformed_features = pipeline.fit_transform(multiple_user_data)
        print(f"\nTransformed feature matrix shape: {transformed_features.shape}")
        
        # Get feature importance
        importance = pipeline.get_feature_importance()
        print(f"\nTop 10 most important features:")
        sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)
        for name, score in sorted_importance[:10]:
            print(f"  {name}: {score:.4f}")
        
    except Exception as e:
        print(f"Error in pipeline fitting: {e}")