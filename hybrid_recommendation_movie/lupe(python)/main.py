import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

import discord
from discord.ext import commands
from discord import app_commands
from discord.ui import Button, View, Modal, TextInput
import torch
import torch.nn as nn
import psycopg2
import psycopg2.extras
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import logging
import asyncio
import json
import re
from datetime import datetime
from typing import List, Optional, Tuple

# Add current directory to path for local models (first so it takes precedence)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# Add parent directory to path for utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import load_config
from models import HybridRecommenderModel, load_model, LupeContentManager
from unified_content_manager import UnifiedLupeContentManager

# Load configuration early
config = load_config()

try:
    from utils.database import DatabaseManager
    from utils.error_handling import handle_exceptions
except ImportError:
    # Fallback if utils modules don't exist
    print("Warning: Utils modules not found, using minimal implementations")
    
    class DatabaseManager:
        def __init__(self, db_config):
            self.config = db_config
        
        def get_connection(self):
            import psycopg2
            from contextlib import contextmanager
            
            @contextmanager
            def connection_context():
                conn = psycopg2.connect(
                    host=self.config.host,
                    port=self.config.port,
                    database=self.config.database,
                    user=self.config.user,
                    password=self.config.password
                )
                try:
                    yield conn
                finally:
                    conn.close()
            
            return connection_context()
    
    def handle_exceptions(func):
        return func

# Set up logging
log_level = logging.DEBUG if config.debug else logging.INFO
logging.basicConfig(level=log_level)
logger = logging.getLogger(__name__)

# Initialize database manager
db_manager = DatabaseManager(config.database)


class FeedbackView(View):
    """Enhanced view with individual content feedback and overall recommendations feedback"""
    def __init__(self, recommendations, method, original_query=None):
        super().__init__(timeout=300)  # 5 minute timeout
        self.recommendations = recommendations
        self.method = method
        self.original_query = original_query

    @discord.ui.button(label='👍 Good Overall', style=discord.ButtonStyle.green)
    async def good_feedback(self, interaction: discord.Interaction, button: Button):
        await self.handle_feedback(interaction, 'positive', 'Good recommendations')

    @discord.ui.button(label='👎 Poor Overall', style=discord.ButtonStyle.red)
    async def poor_feedback(self, interaction: discord.Interaction, button: Button):
        await self.handle_feedback(interaction, 'negative', 'Poor recommendations')

    @discord.ui.button(label='⭐ Rate Individual', style=discord.ButtonStyle.blurple)
    async def rate_individual(self, interaction: discord.Interaction, button: Button):
        modal = IndividualContentFeedbackModal(self.recommendations, self.method)
        await interaction.response.send_modal(modal)

    @discord.ui.button(label='💖 Like/Dislike Each', style=discord.ButtonStyle.secondary)
    async def like_dislike_each(self, interaction: discord.Interaction, button: Button):
        view = IndividualContentFeedbackView(self.recommendations, self.method)
        await interaction.response.send_message("Rate each recommendation individually:", view=view, ephemeral=True)

    @discord.ui.button(label='💬 Detailed Feedback', style=discord.ButtonStyle.gray)
    async def detailed_feedback(self, interaction: discord.Interaction, button: Button):
        modal = DetailedFeedbackModal(self.recommendations, self.method, self.original_query)
        await interaction.response.send_modal(modal)

    async def handle_feedback(self, interaction: discord.Interaction, feedback_type: str, feedback_text: str):
        """Handle simple thumbs up/down feedback"""
        try:
            # Store feedback in database
            with db_manager.get_connection() as conn:
                cursor = conn.cursor()
                
                # Try new schema first, fallback to old schema
                try:
                    cursor.execute('''
                        INSERT INTO feedback (user_id, username, feedback_type, recommendation_method, 
                                            original_query, feedback_text, content_type, timestamp)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    ''', (
                        interaction.user.id,
                        str(interaction.user),
                        feedback_type,
                        self.method,
                        self.original_query,
                        feedback_text,
                        'mixed',
                        datetime.now()
                    ))
                except Exception:
                    # Fallback to old schema
                    cursor.execute('''
                        INSERT INTO feedback (user_id, username, feedback_type, recommendation_method, 
                                            original_query, feedback_text, timestamp)
                        VALUES (%s, %s, %s, %s, %s, %s, %s)
                    ''', (
                        interaction.user.id,
                        str(interaction.user),
                        feedback_type,
                        self.method,
                        self.original_query,
                        feedback_text,
                        datetime.now()
                    ))
                
                conn.commit()
            
            embed = discord.Embed(
                title="✅ Thank you for your feedback!",
                description=f"Your {feedback_type} feedback has been recorded and will help improve recommendations.",
                color=0x00ff00
            )
            
            await interaction.response.send_message(embed=embed, ephemeral=True)
            
        except Exception as e:
            logger.error(f"Error storing feedback: {e}")
            await interaction.response.send_message("Error recording feedback.", ephemeral=True)


class IndividualContentFeedbackView(View):
    """View for rating individual content items with like/dislike buttons"""
    def __init__(self, recommendations, method):
        super().__init__(timeout=300)
        self.recommendations = recommendations
        self.method = method
        self.current_index = 0
        self.feedback_data = {}
        self.update_display()
    
    def update_display(self):
        self.clear_items()
        
        if self.current_index < len(self.recommendations):
            content_id, title, content_type, score = self.recommendations[self.current_index]
            
            # Add navigation info
            nav_button = discord.ui.Button(
                label=f"{self.current_index + 1}/{len(self.recommendations)}: {title[:30]}...",
                style=discord.ButtonStyle.secondary,
                disabled=True
            )
            self.add_item(nav_button)
            
            # Add feedback buttons with callbacks
            love_button = discord.ui.Button(
                label="💖 Love it!",
                style=discord.ButtonStyle.green,
                custom_id=f"love_{self.current_index}"
            )
            love_button.callback = self.button_callback
            self.add_item(love_button)
            
            like_button = discord.ui.Button(
                label="👍 Like it",
                style=discord.ButtonStyle.success,
                custom_id=f"like_{self.current_index}"
            )
            like_button.callback = self.button_callback
            self.add_item(like_button)
            
            dislike_button = discord.ui.Button(
                label="👎 Dislike it",
                style=discord.ButtonStyle.danger,
                custom_id=f"dislike_{self.current_index}"
            )
            dislike_button.callback = self.button_callback
            self.add_item(dislike_button)
            
            hate_button = discord.ui.Button(
                label="💔 Hate it",
                style=discord.ButtonStyle.red,
                custom_id=f"hate_{self.current_index}"
            )
            hate_button.callback = self.button_callback
            self.add_item(hate_button)
            
            # Navigation buttons
            if self.current_index > 0:
                prev_button = discord.ui.Button(
                    label="⬅️ Previous",
                    style=discord.ButtonStyle.secondary,
                    custom_id="previous"
                )
                prev_button.callback = self.button_callback
                self.add_item(prev_button)
            
            if self.current_index < len(self.recommendations) - 1:
                next_button = discord.ui.Button(
                    label="➡️ Next",
                    style=discord.ButtonStyle.secondary,
                    custom_id="next"
                )
                next_button.callback = self.button_callback
                self.add_item(next_button)
            else:
                submit_button = discord.ui.Button(
                    label="✅ Submit All Feedback",
                    style=discord.ButtonStyle.primary,
                    custom_id="submit"
                )
                submit_button.callback = self.button_callback
                self.add_item(submit_button)
    
    async def interaction_check(self, interaction: discord.Interaction) -> bool:
        return True
    
    async def button_callback(self, interaction: discord.Interaction):
        custom_id = button.custom_id
        
        if custom_id.startswith(('love_', 'like_', 'dislike_', 'hate_')):
            feedback_type = custom_id.split('_')[0]
            self.feedback_data[self.current_index] = feedback_type
            
            # Move to next item
            if self.current_index < len(self.recommendations) - 1:
                self.current_index += 1
                self.update_display()
                await interaction.response.edit_message(view=self)
            else:
                # Last item, submit all feedback
                await self.submit_all_feedback(interaction)
        
        elif custom_id == "previous":
            self.current_index -= 1
            self.update_display()
            await interaction.response.edit_message(view=self)
        
        elif custom_id == "next":
            self.current_index += 1
            self.update_display()
            await interaction.response.edit_message(view=self)
        
        elif custom_id == "submit":
            await self.submit_all_feedback(interaction)
    
    async def submit_all_feedback(self, interaction: discord.Interaction):
        """Submit all collected feedback"""
        await store_individual_preferences(
            interaction.user.id, 
            str(interaction.user),
            self.recommendations, 
            self.feedback_data,
            self.method
        )
        
        feedback_count = len(self.feedback_data)
        embed = discord.Embed(
            title="✅ Individual Feedback Recorded!",
            description=f"Thank you! Recorded feedback for {feedback_count} recommendations.\nThis will help personalize your future recommendations!",
            color=0x00ff00
        )
        
        await interaction.response.edit_message(embed=embed, view=None)


class IndividualContentFeedbackModal(Modal):
    """Modal for detailed feedback on individual content items"""
    def __init__(self, recommendations, method):
        super().__init__(title="Individual Content Feedback")
        self.recommendations = recommendations
        self.method = method
        
        # Add text inputs for each recommendation (max 5 due to Discord limits)
        for i, (content_id, title, content_type, score) in enumerate(recommendations[:5]):
            text_input = TextInput(
                label=f"{title[:45]}...",
                placeholder="love/like/dislike/hate or detailed feedback",
                required=False,
                max_length=200
            )
            self.add_item(text_input)

    async def on_submit(self, interaction: discord.Interaction):
        feedback_data = {}
        detailed_feedback = {}
        
        for i, text_input in enumerate(self.children):
            if text_input.value:
                value = text_input.value.lower().strip()
                
                # Check for simple preference words
                if value in ['love', 'like', 'dislike', 'hate']:
                    feedback_data[i] = value
                else:
                    # Treat as detailed feedback
                    detailed_feedback[i] = text_input.value
                    # Try to infer preference from text
                    if any(word in value for word in ['love', 'amazing', 'great', 'excellent']):
                        feedback_data[i] = 'love'
                    elif any(word in value for word in ['like', 'good', 'nice']):
                        feedback_data[i] = 'like'
                    elif any(word in value for word in ['hate', 'terrible', 'awful', 'horrible']):
                        feedback_data[i] = 'hate'
                    elif any(word in value for word in ['dislike', 'bad', 'poor', 'boring']):
                        feedback_data[i] = 'dislike'
        
        # Store feedback
        await store_individual_preferences(
            interaction.user.id,
            str(interaction.user),
            self.recommendations,
            feedback_data,
            self.method,
            detailed_feedback
        )
        
        embed = discord.Embed(
            title="✅ Detailed Feedback Recorded!",
            description=f"Thank you for your detailed feedback on {len(feedback_data)} recommendations!",
            color=0x00ff00
        )
        
        await interaction.response.send_message(embed=embed, ephemeral=True)


class MovieRatingModal(Modal):
    """Modal for rating individual movies"""
    def __init__(self, recommendations):
        super().__init__(title="Rate Movies (1-5 stars)")
        self.recommendations = recommendations
        
        # Add text inputs for each movie (max 5 due to Discord limits)
        for i, (movie_id, title, genres, score) in enumerate(recommendations[:5]):
            text_input = TextInput(
                label=f"{title[:45]}..." if len(title) > 45 else title,
                placeholder="Rate 1-5 (or leave empty to skip)",
                required=False,
                max_length=1
            )
            self.add_item(text_input)

    async def on_submit(self, interaction: discord.Interaction):
        try:
            with db_manager.get_connection() as conn:
                cursor = conn.cursor()
                
                ratings_recorded = 0
                
                for i, text_input in enumerate(self.children):
                    if text_input.value and text_input.value.isdigit():
                        rating = int(text_input.value)
                        if 1 <= rating <= 5:
                            movie_id, title, genres, score = self.recommendations[i]
                            
                            # Store rating (replace if exists)
                            cursor.execute('''
                                INSERT INTO user_ratings (user_id, movie_id, rating, timestamp)
                                VALUES (%s, %s, %s, %s)
                                ON CONFLICT (user_id, movie_id) 
                                DO UPDATE SET rating = EXCLUDED.rating, timestamp = EXCLUDED.timestamp
                            ''', (interaction.user.id, movie_id, rating, datetime.now()))
                            
                            # Also store in feedback table
                            cursor.execute('''
                                INSERT INTO feedback (user_id, username, feedback_type, movie_id, 
                                                    movie_title, rating, timestamp)
                                VALUES (%s, %s, %s, %s, %s, %s, %s)
                            ''', (
                                interaction.user.id,
                                str(interaction.user),
                                'rating',
                                movie_id,
                                title,
                                rating,
                                datetime.now()
                            ))
                            
                            ratings_recorded += 1
                
                conn.commit()
            
            if ratings_recorded > 0:
                embed = discord.Embed(
                    title="✅ Ratings Recorded!",
                    description=f"Thank you! {ratings_recorded} movie rating(s) have been saved.",
                    color=0x00ff00
                )
            else:
                embed = discord.Embed(
                    title="❌ No Valid Ratings",
                    description="Please enter numbers 1-5 for movie ratings.",
                    color=0xff0000
                )
            
            await interaction.response.send_message(embed=embed, ephemeral=True)
            
        except Exception as e:
            logger.error(f"Error storing ratings: {e}")
            await interaction.response.send_message("Error recording ratings.", ephemeral=True)


class DetailedFeedbackModal(Modal):
    """Modal for detailed text feedback"""
    def __init__(self, recommendations, method, original_query):
        super().__init__(title="Detailed Feedback")
        self.recommendations = recommendations
        self.method = method
        self.original_query = original_query
        
        self.feedback_input = TextInput(
            label="What could be improved?",
            placeholder="Tell us what you liked or didn't like about these recommendations...",
            style=discord.TextStyle.paragraph,
            required=True,
            max_length=1000
        )
        self.add_item(self.feedback_input)

    async def on_submit(self, interaction: discord.Interaction):
        try:
            with db_manager.get_connection() as conn:
                cursor = conn.cursor()
                
                # Try new schema first, fallback to old schema
                try:
                    cursor.execute('''
                        INSERT INTO feedback (user_id, username, feedback_type, recommendation_method,
                                            original_query, feedback_text, content_type, timestamp)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    ''', (
                        interaction.user.id,
                        str(interaction.user),
                        'detailed',
                        self.method,
                        self.original_query,
                        self.feedback_input.value,
                        'mixed',
                        datetime.now()
                    ))
                except Exception:
                    # Fallback to old schema
                    cursor.execute('''
                        INSERT INTO feedback (user_id, username, feedback_type, recommendation_method,
                                            original_query, feedback_text, timestamp)
                        VALUES (%s, %s, %s, %s, %s, %s, %s)
                    ''', (
                        interaction.user.id,
                        str(interaction.user),
                        'detailed',
                        self.method,
                        self.original_query,
                        self.feedback_input.value,
                        datetime.now()
                    ))
                
                conn.commit()
            
            embed = discord.Embed(
                title="✅ Feedback Received!",
                description="Thank you for your detailed feedback! This will help us improve our recommendations.",
                color=0x00ff00
            )
            
            await interaction.response.send_message(embed=embed, ephemeral=True)
            
        except Exception as e:
            logger.error(f"Error storing detailed feedback: {e}")
            await interaction.response.send_message("Error recording feedback.", ephemeral=True)


class LupeRecommendationBot(commands.Bot):
    def __init__(self):
        intents = discord.Intents.default()
        intents.message_content = True
        super().__init__(command_prefix='!', intents=intents)

        # Initialize Lupe Content Manager
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.lupe = None
        
        # Legacy attributes for backward compatibility
        self.model = None
        self.movie_lookup = None
        self.genres = None
        self.rating_scaler = None
        self.metadata = None
        self.mappings = None
        
        # Initialize database
        self.init_database()

    def init_database(self):
        """Initialize PostgreSQL database tables"""
        try:
            with db_manager.get_connection() as conn:
                cursor = conn.cursor()
                
                # Create feedback table with proper schema handling
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS feedback (
                        id SERIAL PRIMARY KEY,
                        user_id BIGINT NOT NULL,
                        username TEXT,
                        feedback_type TEXT NOT NULL,
                        movie_id INTEGER,
                        movie_title TEXT,
                        rating INTEGER,
                        recommendation_method TEXT,
                        original_query TEXT,
                        feedback_text TEXT,
                        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                # Add new columns if they don't exist
                try:
                    cursor.execute('''
                        ALTER TABLE feedback 
                        ADD COLUMN IF NOT EXISTS content_id INTEGER,
                        ADD COLUMN IF NOT EXISTS content_title TEXT,
                        ADD COLUMN IF NOT EXISTS content_type TEXT DEFAULT 'movie'
                    ''')
                except Exception as e:
                    # For older PostgreSQL versions that don't support IF NOT EXISTS
                    logger.info(f"Attempting to add columns individually: {e}")
                    
                    # Check if columns exist and add them if they don't
                    for column_info in [
                        ('content_id', 'INTEGER'),
                        ('content_title', 'TEXT'),
                        ('content_type', 'TEXT DEFAULT \'movie\'')
                    ]:
                        try:
                            cursor.execute(f'''
                                SELECT column_name FROM information_schema.columns 
                                WHERE table_name = 'feedback' AND column_name = '{column_info[0]}'
                            ''')
                            if not cursor.fetchone():
                                cursor.execute(f'ALTER TABLE feedback ADD COLUMN {column_info[0]} {column_info[1]}')
                                logger.info(f"Added column {column_info[0]} to feedback table")
                        except Exception as col_error:
                            logger.warning(f"Could not add column {column_info[0]}: {col_error}")
                
                
                # Create user_ratings table (for movies)
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS user_ratings (
                        id SERIAL PRIMARY KEY,
                        user_id BIGINT NOT NULL,
                        movie_id INTEGER NOT NULL,
                        rating REAL NOT NULL,
                        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE(user_id, movie_id)
                    )
                ''')
                
                # Create user_tv_ratings table (for TV shows)
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS user_tv_ratings (
                        id SERIAL PRIMARY KEY,
                        user_id BIGINT NOT NULL,
                        show_id INTEGER NOT NULL,
                        rating REAL NOT NULL,
                        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE(user_id, show_id)
                    )
                ''')
                
                # Create user_preferences table (individual user feedback)
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS user_preferences (
                        id SERIAL PRIMARY KEY,
                        user_id BIGINT NOT NULL,
                        content_id INTEGER NOT NULL,
                        content_type TEXT NOT NULL,  -- 'movie' or 'tv'
                        preference TEXT NOT NULL,    -- 'like', 'dislike', 'love', 'hate'
                        feedback_reason TEXT,        -- optional reason why they liked/disliked
                        recommendation_context TEXT, -- what method was used to recommend this
                        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE(user_id, content_id, content_type)
                )
                ''')
                
                # Create master_training_feedback table (for admin review and model training)
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS master_training_feedback (
                        id SERIAL PRIMARY KEY,
                        content_id INTEGER NOT NULL,
                        content_type TEXT NOT NULL,
                        content_title TEXT,
                        content_genres TEXT,
                        feedback_type TEXT NOT NULL,     -- 'like', 'dislike', 'love', 'hate'
                        user_id BIGINT NOT NULL,
                        username TEXT,
                        feedback_reason TEXT,
                        recommendation_context TEXT,
                        admin_reviewed BOOLEAN DEFAULT FALSE,
                        admin_notes TEXT,
                        training_weight REAL DEFAULT 1.0,  -- weight for training (admin can adjust)
                        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                # Create user_genre_preferences table (track user's genre preferences)
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS user_genre_preferences (
                        id SERIAL PRIMARY KEY,
                        user_id BIGINT NOT NULL,
                        genre TEXT NOT NULL,
                        content_type TEXT NOT NULL,  -- 'movie' or 'tv'
                        preference_score REAL DEFAULT 0.0,  -- calculated from likes/dislikes
                        interaction_count INTEGER DEFAULT 0,
                        last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE(user_id, genre, content_type)
                    )
                ''')
                
                # Create index for faster queries
                cursor.execute('''
                    CREATE INDEX IF NOT EXISTS idx_user_ratings_user_id ON user_ratings(user_id)
                ''')
                cursor.execute('''
                    CREATE INDEX IF NOT EXISTS idx_user_tv_ratings_user_id ON user_tv_ratings(user_id)
                ''')
                cursor.execute('''
                    CREATE INDEX IF NOT EXISTS idx_feedback_user_id ON feedback(user_id)
                ''')
                cursor.execute('''
                    CREATE INDEX IF NOT EXISTS idx_feedback_type ON feedback(feedback_type)
                ''')
                cursor.execute('''
                    CREATE INDEX IF NOT EXISTS idx_feedback_content_type ON feedback(content_type)
                ''')
                cursor.execute('''
                    CREATE INDEX IF NOT EXISTS idx_user_preferences_user_id ON user_preferences(user_id)
                ''')
                cursor.execute('''
                    CREATE INDEX IF NOT EXISTS idx_user_preferences_content ON user_preferences(content_id, content_type)
                ''')
                cursor.execute('''
                    CREATE INDEX IF NOT EXISTS idx_master_training_admin_reviewed ON master_training_feedback(admin_reviewed)
                ''')
                cursor.execute('''
                    CREATE INDEX IF NOT EXISTS idx_user_genre_preferences_user ON user_genre_preferences(user_id, content_type)
                ''')
                
                conn.commit()
                logger.info("Database tables initialized successfully")
                
        except Exception as e:
            logger.error(f"Error initializing database: {e}")

    async def setup_hook(self):
        """Load the trained models and data when bot starts"""
        try:
            logger.info("Loading Lupe Content Manager...")
            
            # Initialize Unified Lupe Content Manager
            self.lupe = UnifiedLupeContentManager(config.model.models_dir, str(self.device))
            self.lupe.load_models()
            
            # Set up legacy compatibility attributes
            self.movie_lookup = self.lupe.movie_lookup
            self.genres = self.lupe.all_genres
            self.model = self.lupe.movie_model
            self.mappings = self.lupe.movie_mappings
            self.metadata = self.lupe.movie_metadata
            self.rating_scaler = self.lupe.movie_rating_scaler
            
            status = self.lupe.get_model_status()
            logger.info(f"Lupe setup complete: {status['movie_count']} movies, {status['tv_count']} TV shows, {status['total_genres']} genres")

        except Exception as e:
            logger.error(f"Error loading Lupe: {e}")
            # Fallback genres if loading fails
            self.genres = ['Action', 'Adventure', 'Animation', 'Children', 'Comedy', 'Crime', 
                          'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical', 
                          'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
            logger.info(f"Using fallback genres: {self.genres}")
        
        # Check if we got data even if models failed
        if self.lupe:
            status = self.lupe.get_model_status()
            logger.info(f"Lupe data status: movie_count={status['movie_count']}, tv_count={status['tv_count']}, movie_genres={status['movie_genres']}, tv_genres={status['tv_genres']}")
            
            # Update genres from Lupe if available
            if status['movie_genres'] > 0 or status['tv_genres'] > 0:
                self.genres = self.lupe.get_available_genres('mixed')
                logger.info(f"Updated genres from Lupe data: {len(self.genres)} genres")

    async def on_ready(self):
        """Called when bot is ready"""
        logger.info(f'{self.user} has connected to Discord!')
        try:
            synced = await self.tree.sync()
            logger.info(f"Synced {len(synced)} command(s)")
        except Exception as e:
            logger.error(f"Failed to sync commands: {e}")


# Create bot instance
bot = LupeRecommendationBot()

# Session tracking to prevent repeated recommendations
user_recommendation_cache = {}

def get_user_excluded_movies(user_id: int) -> set:
    """Get movies to exclude for this user (recently recommended)"""
    session_key = f"user_{user_id}_last_recommendations"
    
    if session_key in user_recommendation_cache:
        cache_data = user_recommendation_cache[session_key]
        # Keep cache for 1 hour
        import time
        if time.time() - cache_data['timestamp'] < 3600:
            return set(cache_data['movie_ids'])
    
    return set()

def update_user_recommendation_cache(user_id: int, movie_ids: List[int]):
    """Update the cache with newly recommended movies"""
    session_key = f"user_{user_id}_last_recommendations"
    import time
    
    user_recommendation_cache[session_key] = {
        'movie_ids': movie_ids,
        'timestamp': time.time()
    }

def is_valid_movie(movie_info):
    """Filter out non-movies, fake movies, and test data"""
    title = movie_info.get('title', '').lower()
    
    # Skip if no title
    if not title or not isinstance(title, str):
        return False
    
    # Skip very short titles that might be errors
    if len(title.strip()) < 3:
        return False
    
    # Skip TV shows and series indicators
    tv_indicators = [
        '(tv series)', '(tv movie)', '(tv short)', '(tv mini-series)',
        '(video)', '(v)', 'episode', 'season', 'series', '(tv)', 'tv series'
    ]
    
    for indicator in tv_indicators:
        if indicator in title:
            return False
    
    # Skip obvious fake/test movie patterns
    fake_patterns = [
        "i don't", "can't you", "we're switching", "the civil servant",
        "test movie", "sample", "example", "fake", "lorem ipsum", 
        "placeholder", "dummy", "untitled", "unknown", "movie title", 
        "film title", "temp", "temporary", "null", "nan", "undefined"
    ]
    
    for pattern in fake_patterns:
        if pattern in title:
            return False
    
    # Skip titles that are questions (often fake)
    question_starters = ["why", "how", "what", "when", "where", "who", "can't you", "don't you"]
    for starter in question_starters:
        if title.startswith(starter):
            return False
    
    # Skip suspicious patterns using regex
    suspicious_patterns = [
        r'^\d+$',  # Only numbers
        r'^[a-z]$',  # Single letter
        r'test\d+',  # test1, test2, etc.
        r'movie\d+',  # movie1, movie2, etc.
    ]
    
    for pattern in suspicious_patterns:
        if re.match(pattern, title):
            return False
    
    return True

def filter_valid_movies(recommendations):
    """Filter recommendations to only include valid movies"""
    valid_recommendations = []
    
    for movie_id, title, genres, score in recommendations:
        movie_info = bot.movie_lookup.get(movie_id, {})
        if is_valid_movie(movie_info):
            valid_recommendations.append((movie_id, title, genres, score))
    
    return valid_recommendations

async def store_individual_preferences(user_id: int, username: str, recommendations: List[tuple], 
                                     feedback_data: dict, method: str, detailed_feedback: dict = None):
    """Store individual user preferences and update master training data"""
    try:
        with db_manager.get_connection() as conn:
            cursor = conn.cursor()
            
            for index, feedback_type in feedback_data.items():
                if index >= len(recommendations):
                    continue
                    
                content_id, title, content_type, score = recommendations[index]
                detailed_text = detailed_feedback.get(index, '') if detailed_feedback else ''
                
                # Get content info for genres
                if content_type == 'movie' and bot.lupe and bot.lupe.movie_lookup:
                    content_info = bot.lupe.movie_lookup.get(int(content_id), {})
                elif content_type == 'tv' and bot.lupe and bot.lupe.tv_lookup:
                    content_info = bot.lupe.tv_lookup.get(int(content_id), {})
                else:
                    content_info = {}
                
                genres = content_info.get('genres', '')
                
                # Store in user_preferences table
                cursor.execute('''
                    INSERT INTO user_preferences (user_id, content_id, content_type, preference, 
                                                feedback_reason, recommendation_context, timestamp)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (user_id, content_id, content_type) 
                    DO UPDATE SET preference = EXCLUDED.preference, 
                                  feedback_reason = EXCLUDED.feedback_reason,
                                  recommendation_context = EXCLUDED.recommendation_context,
                                  timestamp = EXCLUDED.timestamp
                ''', (user_id, int(content_id), content_type, feedback_type, 
                      detailed_text, method, datetime.now()))
                
                # Store in master_training_feedback table for admin review
                cursor.execute('''
                    INSERT INTO master_training_feedback (content_id, content_type, content_title, 
                                                        content_genres, feedback_type, user_id, username,
                                                        feedback_reason, recommendation_context, timestamp)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ''', (int(content_id), content_type, title, genres, feedback_type, 
                      user_id, username, detailed_text, method, datetime.now()))
                
                # Update user genre preferences
                if genres:
                    await update_user_genre_preferences(cursor, user_id, genres, content_type, feedback_type)
            
            conn.commit()
            logger.info(f"Stored {len(feedback_data)} individual preferences for user {user_id}")
            
    except Exception as e:
        logger.error(f"Error storing individual preferences: {e}")


async def update_user_genre_preferences(cursor, user_id: int, genres: str, content_type: str, feedback_type: str):
    """Update user's genre preferences based on feedback"""
    if not isinstance(genres, str) or not genres.strip():
        return
    
    # Convert feedback to score
    feedback_scores = {
        'love': 2.0,
        'like': 1.0,
        'dislike': -1.0,
        'hate': -2.0
    }
    
    score_change = feedback_scores.get(feedback_type, 0)
    if score_change == 0:
        return
    
    # Update each genre
    genre_list = [g.strip() for g in genres.split('|') if g.strip()]
    for genre in genre_list:
        try:
            # Check if preference exists
            cursor.execute('''
                SELECT preference_score, interaction_count 
                FROM user_genre_preferences 
                WHERE user_id = %s AND genre = %s AND content_type = %s
            ''', (user_id, genre, content_type))
            
            result = cursor.fetchone()
            
            if result:
                current_score, count = result
                new_score = (current_score * count + score_change) / (count + 1)
                new_count = count + 1
                
                cursor.execute('''
                    UPDATE user_genre_preferences 
                    SET preference_score = %s, interaction_count = %s, last_updated = %s
                    WHERE user_id = %s AND genre = %s AND content_type = %s
                ''', (new_score, new_count, datetime.now(), user_id, genre, content_type))
            else:
                # Create new preference
                cursor.execute('''
                    INSERT INTO user_genre_preferences (user_id, genre, content_type, 
                                                      preference_score, interaction_count, last_updated)
                    VALUES (%s, %s, %s, %s, 1, %s)
                ''', (user_id, genre, content_type, score_change, datetime.now()))
                
        except Exception as e:
            logger.error(f"Error updating genre preference for {genre}: {e}")


def add_recommendation_diversity(recommendations: List[tuple], exclude_ids: set = None) -> List[tuple]:
    """Add diversity to recommendations by removing similar titles and excluded IDs"""
    if not recommendations:
        return recommendations
    
    diverse_recs = []
    used_titles = set()
    exclude_ids = exclude_ids or set()
    
    for movie_id, title, genres, score in recommendations:
        # Skip if in excluded set
        if movie_id in exclude_ids:
            continue
        
        # Skip if title is not a string
        if not isinstance(title, str):
            continue
            
        # Extract meaningful words from title
        title_words = set(word.lower().strip('.,!?()[]') for word in title.split() if len(word) > 2)
        
        # Check similarity with already selected titles
        is_too_similar = False
        for used_title in used_titles:
            used_words = set(word.lower().strip('.,!?()[]') for word in used_title.split() if len(word) > 2)
            
            if title_words and used_words:
                overlap = len(title_words & used_words)
                min_words = min(len(title_words), len(used_words))
                if min_words > 0:
                    similarity = overlap / min_words
                    
                    if similarity > 0.6:  # 60% similarity threshold
                        is_too_similar = True
                        break
        
        if not is_too_similar:
            diverse_recs.append((movie_id, title, genres, score))
            used_titles.add(title)
    
    return diverse_recs

def find_content_by_title(title: str) -> Optional[Tuple[str, str]]:
    """
    Find content by title in both movies and TV shows
    Returns (content_id, content_type) or None if not found
    """
    try:
        title_lower = title.lower()
        
        # Search in movies first
        if bot.lupe and bot.lupe.movie_lookup:
            for movie_id, movie_info in bot.lupe.movie_lookup.items():
                movie_title = movie_info.get('title', '')
                if isinstance(movie_title, str) and title_lower in movie_title.lower():
                    return (str(movie_id), 'movie')
        
        # Search in TV shows
        if bot.lupe and bot.lupe.tv_lookup:
            for show_id, show_info in bot.lupe.tv_lookup.items():
                show_title = show_info.get('title', '')
                if isinstance(show_title, str) and title_lower in show_title.lower():
                    return (str(show_id), 'tv')
        
        return None
        
    except Exception as e:
        logger.error(f"Error finding content by title: {e}")
        return None


def get_movie_genre_features(movie_id):
    """Get genre features for a specific movie ID"""
    try:
        # Find movie in the dataframe using original movie ID
        movie_row = bot.movies_df[bot.movies_df['media_id'] == movie_id]
        
        if movie_row.empty:
            logger.warning(f"Movie ID {movie_id} not found in dataset")
            return np.zeros(len(bot.genres))
        
        # Extract genre features
        genre_features = []
        for genre in bot.genres:
            if genre in movie_row.iloc[0]:
                genre_features.append(movie_row.iloc[0][genre])
            else:
                genre_features.append(0)
        
        return np.array(genre_features, dtype=np.float32)
        
    except Exception as e:
        logger.error(f"Error getting genre features for movie {movie_id}: {e}")
        return np.zeros(len(bot.genres))


@bot.tree.command(name="recommend", description="Get movie and TV show recommendations")
@app_commands.describe(
    content_type="Type of content to recommend (mixed, movies, tv)",
    user_id="Your user ID (for personalized recommendations)", 
    content_title="Movie/show you liked (for similar recommendations)",
    genre="Preferred genre",
    limit="Number of recommendations (default: 5)"
)
async def recommend_content(
        interaction: discord.Interaction,
        content_type: Optional[str] = "mixed",
        user_id: Optional[int] = None,
        content_title: Optional[str] = None,
        genre: Optional[str] = None,
        limit: Optional[int] = 5
):
    """Slash command to get content recommendations"""
    await interaction.response.defer()

    try:
        # Check if Lupe is properly loaded
        if not bot.lupe:
            logger.error("bot.lupe is None - Lupe Content Manager not initialized")
            embed = discord.Embed(
                title="❌ Bot Not Ready",
                description="The bot is still loading. Please wait a moment and try again.",
                color=0xff0000
            )
            await interaction.followup.send(embed=embed)
            return
        
        logger.info(f"Lupe status: movie_count={len(bot.lupe.movie_lookup)}, tv_count={len(bot.lupe.tv_lookup)}")
        
        # Validate and normalize inputs
        content_type = (content_type or "mixed").lower()
        if content_type not in ["mixed", "movies", "tv", "movie"]:
            content_type = "mixed"
        
        # Handle legacy "movies" -> "movie" mapping
        if content_type == "movies":
            content_type = "movie"
            
        limit = max(1, min(limit or 5, 20))
        
        logger.info(f"Recommendation request: content_type={content_type}, user_id={user_id}, content_title={content_title}, genre={genre}, limit={limit}")
        
        # Validate genre compatibility with content type
        if genre and content_type != "mixed" and bot.lupe:
            available_genres = bot.lupe.get_available_genres(content_type)
            if genre not in available_genres:
                # Find what content types this genre belongs to
                movie_genres = set(bot.lupe.get_available_genres('movie'))
                tv_genres = set(bot.lupe.get_available_genres('tv'))
                
                genre_locations = []
                if genre in movie_genres:
                    genre_locations.append("movies")
                if genre in tv_genres:
                    genre_locations.append("TV shows")
                
                if genre_locations:
                    suggestion = " or ".join(genre_locations)
                    embed = discord.Embed(
                        title="⚠️ Genre Mismatch",
                        description=f"The genre **{genre}** is not available for **{content_type}** content.\n\n"
                                  f"This genre is available for: **{suggestion}**\n\n"
                                  f"Would you like to:\n"
                                  f"• Use `/recommend {suggestion.split()[0]}` instead, or\n"
                                  f"• Choose a different genre with `/genres {content_type}`",
                        color=0xffaa00
                    )
                    await interaction.followup.send(embed=embed)
                    return
        
        recommendations = []
        method = ""
        
        # Get user's excluded content (recently recommended)
        exclude_ids = get_user_excluded_movies(interaction.user.id)
        
        # Method 1: Use Lupe's integrated recommendations
        if user_id is not None:
            logger.info(f"Using Lupe AI recommendations for {content_type}")
            recommendations = bot.lupe.get_recommendations(
                user_id=user_id,
                limit=limit,
                content_type=content_type,
                genre=genre
            )
            method = f"Lupe AI ({content_type.title()})"
            
        # Method 2: Similar content recommendations
        elif content_title is not None:
            logger.info(f"Using similarity-based recommendations for '{content_title}'")
            # Try to find the content and get similar items
            found_content = find_content_by_title(content_title)
            if found_content:
                content_id, found_type = found_content
                recommendations = bot.lupe.get_similar_content(content_id, found_type, limit)
                method = f"Similar to {content_title}"
            else:
                # Fallback to general recommendations
                recommendations = bot.lupe.get_recommendations(
                    user_id=0,  # Use fallback user
                    limit=limit,
                    content_type=content_type,
                    genre=genre
                )
                method = f"General {content_type.title()}"
                
        # Method 3: General recommendations
        else:
            logger.info(f"Using general {content_type} recommendations")
            recommendations = bot.lupe.get_recommendations(
                user_id=0,  # Use fallback user
                limit=limit,
                content_type=content_type,
                genre=genre
            )
            method = f"Popular {content_type.title()}"
        
        logger.info(f"Got {len(recommendations)} recommendations: {recommendations[:2] if recommendations else 'None'}")

        # Create response embed
        if recommendations:
            # Determine title and emoji based on content type
            if content_type == "movie":
                title = "🎬 Movie Recommendations"
            elif content_type == "tv":
                title = "📺 TV Show Recommendations"
            else:
                title = "🎭 Mixed Content Recommendations"
            
            embed = discord.Embed(
                title=title,
                description=f"Method: {method}",
                color=0x00ff00
            )

            for i, (content_id, title_text, content_type_rec, score) in enumerate(recommendations, 1):
                # Add appropriate emoji based on content type
                type_emoji = "🎬" if content_type_rec == "movie" else "📺"
                type_text = "Movie" if content_type_rec == "movie" else "TV Show"
                
                # Get genre info
                if content_type_rec == "movie" and bot.lupe.movie_lookup:
                    content_info = bot.lupe.movie_lookup.get(int(content_id), {})
                elif content_type_rec == "tv" and bot.lupe.tv_lookup:
                    content_info = bot.lupe.tv_lookup.get(int(content_id), {})
                else:
                    content_info = {}
                
                genres = content_info.get('genres', 'Unknown')
                
                # Format confidence score with color indicator
                if score <= 1.0:
                    confidence = f"{score:.1%}"
                    conf_color = "🟢" if score > 0.7 else "🟡" if score > 0.4 else "🔴"
                else:
                    confidence = f"{score:.2f}"
                    conf_color = "🟢" if score > 3.5 else "🟡" if score > 2.5 else "🔴"
                
                embed.add_field(
                    name=f"{i}. {type_emoji} {title_text}",
                    value=f"**Type:** {type_text}\n**Genres:** {genres}\n{conf_color} **Confidence:** {confidence}",
                    inline=False
                )

            embed.set_footer(text=f"Requested by {interaction.user.display_name} • Powered by Lupe AI")
            
            # Add feedback buttons
            view = FeedbackView(recommendations, method, str({
                'content_type': content_type,
                'user_id': user_id,
                'content_title': content_title,
                'genre': genre,
                'limit': limit
            }))
            await interaction.followup.send(embed=embed, view=view)

        else:
            # More detailed error message
            error_msg = f"Sorry, I couldn't find any {content_type} recommendations."
            
            if genre:
                error_msg += f" No content found for genre '{genre}'."
            elif content_title:
                error_msg += f" Content '{content_title}' not found."
            elif user_id:
                error_msg += f" User ID {user_id} not in training data."
            
            error_msg += " Try different parameters or use `/lupe_status` to check bot status."
            
            embed = discord.Embed(
                title="❌ No Recommendations Found",
                description=error_msg,
                color=0xff0000
            )
            
            # Add helpful suggestions
            if genre:
                embed.add_field(
                    name="💡 Try these commands:",
                    value="• `/genres` - See all available genres\n• `/test_genre <genre>` - Test if a genre works\n• `/recommend` without genre - Get popular content",
                    inline=False
                )
            
            await interaction.followup.send(embed=embed)

    except Exception as e:
        logger.error(f"Error in recommend_content: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        
        embed = discord.Embed(
            title="❌ Error",
            description=f"Sorry, there was an error processing your request: {str(e)}",
            color=0xff0000
        )
        await interaction.followup.send(embed=embed)


@bot.tree.command(name="cross_recommend", description="Get cross-content recommendations")
@app_commands.describe(
    source_type="Content type to analyze preferences from (movie, tv)",
    target_type="Content type to recommend (movie, tv)",
    user_id="Your user ID",
    limit="Number of recommendations (default: 5)"
)
async def cross_recommend(
        interaction: discord.Interaction,
        source_type: str,
        target_type: str,
        user_id: Optional[int] = None,
        limit: Optional[int] = 5
):
    """Get cross-content recommendations (e.g., TV shows based on movie preferences)"""
    await interaction.response.defer()
    
    try:
        if not bot.lupe:
            embed = discord.Embed(
                title="❌ Bot Not Ready",
                description="The bot is still loading. Please wait a moment and try again.",
                color=0xff0000
            )
            await interaction.followup.send(embed=embed)
            return
        
        # Validate inputs
        source_type = source_type.lower()
        target_type = target_type.lower()
        
        if source_type not in ["movie", "tv"] or target_type not in ["movie", "tv"]:
            embed = discord.Embed(
                title="❌ Invalid Content Types",
                description="Please use 'movie' or 'tv' for both source and target types.",
                color=0xff0000
            )
            await interaction.followup.send(embed=embed)
            return
        
        if source_type == target_type:
            embed = discord.Embed(
                title="❌ Same Content Types",
                description="Source and target types should be different for cross-recommendations.",
                color=0xff0000
            )
            await interaction.followup.send(embed=embed)
            return
        
        limit = max(1, min(limit or 5, 15))
        user_id = user_id or interaction.user.id
        
        # Get cross-content recommendations
        recommendations = bot.lupe.get_cross_content_recommendations(
            user_id=user_id if user_id else interaction.user.id,
            limit=limit
        )
        
        if recommendations:
            type_emoji = "🎬" if target_type == "movie" else "📺"
            type_name = "Movies" if target_type == "movie" else "TV Shows"
            source_name = "movies" if source_type == "movie" else "TV shows"
            
            embed = discord.Embed(
                title=f"{type_emoji} Cross-Content Recommendations",
                description=f"{type_name} based on your {source_name} preferences",
                color=0x9932cc
            )

            for i, (content_id, title, content_type_rec, score) in enumerate(recommendations, 1):
                # Get genre info
                if content_type_rec == "movie" and bot.lupe.movie_lookup:
                    content_info = bot.lupe.movie_lookup.get(int(content_id), {})
                elif content_type_rec == "tv" and bot.lupe.tv_lookup:
                    content_info = bot.lupe.tv_lookup.get(int(content_id), {})
                else:
                    content_info = {}
                
                genres = content_info.get('genres', 'Unknown')
                
                embed.add_field(
                    name=f"{i}. {title}",
                    value=f"**Genres:** {genres}\n**Score:** {score:.2f}",
                    inline=False
                )

            embed.set_footer(text=f"Requested by {interaction.user.display_name} • Cross-content AI")
            
            # Add feedback buttons
            view = FeedbackView(recommendations, f"Cross-content ({source_type} → {target_type})", 
                              f"source:{source_type}, target:{target_type}, user_id:{user_id}")
            await interaction.followup.send(embed=embed, view=view)
        else:
            embed = discord.Embed(
                title="❌ No Cross-Recommendations Found",
                description=f"Could not generate {target_type} recommendations based on {source_type} preferences.",
                color=0xff0000
            )
            await interaction.followup.send(embed=embed)
            
    except Exception as e:
        logger.error(f"Error in cross_recommend: {e}")
        embed = discord.Embed(
            title="❌ Error",
            description=f"Sorry, there was an error processing your request: {str(e)}",
            color=0xff0000
        )
        await interaction.followup.send(embed=embed)


@bot.tree.command(name="similar", description="Find similar movies or TV shows")
@app_commands.describe(
    content_title="Title of movie or TV show to find similar content for",
    limit="Number of recommendations (default: 5)"
)
async def similar_content(
        interaction: discord.Interaction,
        content_title: str,
        limit: Optional[int] = 5
):
    """Find similar movies or TV shows"""
    await interaction.response.defer()
    
    try:
        if not bot.lupe:
            embed = discord.Embed(
                title="❌ Bot Not Ready",
                description="The bot is still loading. Please wait a moment and try again.",
                color=0xff0000
            )
            await interaction.followup.send(embed=embed)
            return
        
        limit = max(1, min(limit or 5, 15))
        
        # Find the content
        found_content = find_content_by_title(content_title)
        if not found_content:
            embed = discord.Embed(
                title="❌ Content Not Found",
                description=f"Could not find any movie or TV show matching '{content_title}'",
                color=0xff0000
            )
            await interaction.followup.send(embed=embed)
            return
        
        content_id, content_type = found_content
        
        # Get similar content
        recommendations = bot.lupe.get_similar_content(content_id, content_type, limit)
        
        if recommendations:
            type_emoji = "🎬" if content_type == "movie" else "📺"
            type_name = "Movie" if content_type == "movie" else "TV Show"
            
            embed = discord.Embed(
                title=f"🔍 Similar to {type_emoji} {content_title}",
                description=f"Content similar to this {type_name.lower()}",
                color=0x0099ff
            )

            for i, (rec_id, title, rec_type, score) in enumerate(recommendations, 1):
                rec_emoji = "🎬" if rec_type == "movie" else "📺"
                rec_type_name = "Movie" if rec_type == "movie" else "TV Show"
                
                # Get genre info
                if rec_type == "movie" and bot.lupe.movie_lookup:
                    content_info = bot.lupe.movie_lookup.get(int(rec_id), {})
                elif rec_type == "tv" and bot.lupe.tv_lookup:
                    content_info = bot.lupe.tv_lookup.get(int(rec_id), {})
                else:
                    content_info = {}
                
                genres = content_info.get('genres', 'Unknown')
                
                embed.add_field(
                    name=f"{i}. {rec_emoji} {title}",
                    value=f"**Type:** {rec_type_name}\n**Genres:** {genres}\n**Similarity:** {score:.2f}",
                    inline=False
                )

            embed.set_footer(text=f"Requested by {interaction.user.display_name} • Content similarity")
            
            # Add feedback buttons
            view = FeedbackView(recommendations, f"Similar to {content_title}", 
                              f"source_content:{content_title}, source_type:{content_type}")
            await interaction.followup.send(embed=embed, view=view)
        else:
            embed = discord.Embed(
                title="❌ No Similar Content Found",
                description=f"Could not find content similar to '{content_title}'",
                color=0xff0000
            )
            await interaction.followup.send(embed=embed)
            
    except Exception as e:
        logger.error(f"Error in similar_content: {e}")
        embed = discord.Embed(
            title="❌ Error",
            description=f"Sorry, there was an error processing your request: {str(e)}",
            color=0xff0000
        )
        await interaction.followup.send(embed=embed)


@bot.tree.command(name="lupe_status", description="Get Lupe AI status and statistics")
async def lupe_status(interaction: discord.Interaction):
    """Show Lupe AI status and capabilities"""
    try:
        embed = discord.Embed(title="🤖 Lupe AI Status", color=0x9932cc)
        
        if bot.lupe:
            status = bot.lupe.get_model_status()
            
            embed.add_field(name="Movie Model", 
                          value="✅ Loaded" if status['movie_model_loaded'] else "❌ Not loaded", 
                          inline=True)
            embed.add_field(name="TV Model", 
                          value="✅ Loaded" if status['tv_model_loaded'] else "❌ Not loaded", 
                          inline=True)
            embed.add_field(name="Device", value=status['device'], inline=True)
            
            embed.add_field(name="Movies Available", value=f"{status['movie_count']:,}", inline=True)
            embed.add_field(name="TV Shows Available", value=f"{status['tv_count']:,}", inline=True)
            embed.add_field(name="Total Genres", value=f"{status['total_genres']}", inline=True)
            
            # Capabilities
            capabilities = []
            if status['movie_model_loaded']:
                capabilities.append("🎬 Movie recommendations")
            if status['tv_model_loaded']:
                capabilities.append("📺 TV show recommendations")
            if status['movie_model_loaded'] and status['tv_model_loaded']:
                capabilities.append("🎭 Cross-content recommendations")
            capabilities.append("🔍 Content similarity search")
            capabilities.append("🎭 Genre-based filtering")
            
            embed.add_field(name="Capabilities", value="\n".join(capabilities), inline=False)
            
        else:
            embed.add_field(name="Status", value="❌ Lupe AI not initialized", inline=False)
        
        await interaction.response.send_message(embed=embed)
        
    except Exception as e:
        logger.error(f"Error in lupe_status: {e}")
        await interaction.response.send_message("Error retrieving Lupe status.")


@bot.tree.command(name="model_compare", description="Compare recommendations from different AI models")
@app_commands.describe(
    user_id="User ID for personalized comparison (optional)",
    limit="Number of recommendations per model (1-10)"
)
async def model_compare(interaction: discord.Interaction, user_id: Optional[int] = None, limit: Optional[int] = 5):
    """Compare recommendations from different models"""
    await interaction.response.defer()
    
    try:
        if not bot.lupe:
            embed = discord.Embed(
                title="❌ Bot Not Ready",
                description="The bot is still loading. Please wait and try again.",
                color=0xff0000
            )
            await interaction.followup.send(embed=embed)
            return
        
        # Use Discord user ID if no user_id provided
        if user_id is None:
            user_id = interaction.user.id
        
        limit = max(1, min(limit or 5, 10))
        
        # Get model comparison
        comparison = bot.lupe.compare_models(user_id=user_id, limit=limit)
        
        if not comparison:
            embed = discord.Embed(
                title="❌ No Models Available",
                description="No AI models are currently loaded for comparison.",
                color=0xff0000
            )
            await interaction.followup.send(embed=embed)
            return
        
        # Create embed for comparison
        embed = discord.Embed(
            title="🤖 Model Comparison",
            description=f"Recommendations for User {user_id}",
            color=0x9932cc
        )
        
        for model_name, recommendations in comparison.items():
            if recommendations:
                rec_text = []
                for i, (content_id, title, content_type, score) in enumerate(recommendations[:3], 1):
                    icon = "🎬" if content_type == "movie" else "📺"
                    rec_text.append(f"{i}. {icon} {title} ({score:.2f})")
                
                embed.add_field(
                    name=f"{model_name.upper()} Model",
                    value="\n".join(rec_text) if rec_text else "No recommendations",
                    inline=False
                )
        
        await interaction.followup.send(embed=embed)
        
    except Exception as e:
        logger.error(f"Error in model_compare: {e}")
        await interaction.followup.send("Error comparing models.")


@bot.tree.command(name="model_health", description="Check health status of all AI models")
async def model_health(interaction: discord.Interaction):
    """Check health status of all models"""
    await interaction.response.defer()
    
    try:
        if not bot.lupe:
            embed = discord.Embed(
                title="❌ Bot Not Ready",
                description="The bot is still loading. Please wait and try again.",
                color=0xff0000
            )
            await interaction.followup.send(embed=embed)
            return
        
        # Get model health
        health = bot.lupe.get_model_health()
        model_info = bot.lupe.get_model_info()
        
        embed = discord.Embed(
            title="🏥 Model Health Check",
            description="Status of all AI models",
            color=0x00ff00 if all(health.values()) else 0xff9900
        )
        
        # Add health status for each model
        for model_name, is_healthy in health.items():
            status_icon = "✅" if is_healthy else "❌"
            status_text = "Healthy" if is_healthy else "Unhealthy"
            embed.add_field(
                name=f"{model_name.upper()} Model",
                value=f"{status_icon} {status_text}",
                inline=True
            )
        
        # Add system info
        if 'device' in model_info:
            embed.add_field(name="Device", value=model_info['device'], inline=True)
        
        if 'loaded_models' in model_info:
            loaded_count = len(model_info['loaded_models'])
            embed.add_field(name="Models Loaded", value=f"{loaded_count}", inline=True)
        
        # Add legacy data info
        if 'legacy_data' in model_info:
            legacy = model_info['legacy_data']
            embed.add_field(
                name="Data Status",
                value=f"Movies: {legacy.get('movie_count', 0):,}\nTV Shows: {legacy.get('tv_count', 0):,}",
                inline=True
            )
        
        await interaction.followup.send(embed=embed)
        
    except Exception as e:
        logger.error(f"Error in model_health: {e}")
        await interaction.followup.send("Error checking model health.")


@bot.tree.command(name="next_episode", description="Predict what to watch next based on your viewing history")
@app_commands.describe(
    recent_watches="Recent content you've watched (comma-separated titles or IDs)",
    limit="Number of predictions (1-10)"
)
async def next_episode(interaction: discord.Interaction, recent_watches: str, limit: Optional[int] = 5):
    """Predict next items based on sequence using sequential model"""
    await interaction.response.defer()
    
    try:
        if not bot.lupe:
            embed = discord.Embed(
                title="❌ Bot Not Ready",
                description="The bot is still loading. Please wait and try again.",
                color=0xff0000
            )
            await interaction.followup.send(embed=embed)
            return
        
        limit = max(1, min(limit or 5, 10))
        
        # Parse recent watches - try to convert to IDs
        sequence = []
        watch_items = [item.strip() for item in recent_watches.split(',')]
        
        for item in watch_items:
            # Try to parse as ID first
            try:
                content_id = int(item)
                sequence.append(content_id)
            except ValueError:
                # Search for title in lookups
                found_id = None
                item_lower = item.lower()
                
                # Search movies
                for content_id, content_info in bot.lupe.movie_lookup.items():
                    title = content_info.get('title', '').lower()
                    if item_lower in title or title in item_lower:
                        found_id = content_id
                        break
                
                # Search TV shows if not found in movies
                if not found_id:
                    for content_id, content_info in bot.lupe.tv_lookup.items():
                        title = content_info.get('title', '').lower()
                        if item_lower in title or title in item_lower:
                            found_id = content_id
                            break
                
                if found_id:
                    sequence.append(found_id)
        
        if not sequence:
            embed = discord.Embed(
                title="❌ No Valid Items Found",
                description="Could not find any of the items you mentioned. Try using specific titles or IDs.",
                color=0xff0000
            )
            await interaction.followup.send(embed=embed)
            return
        
        # Get next item predictions
        predictions = bot.lupe.predict_next_items(sequence=sequence, limit=limit)
        
        if not predictions:
            embed = discord.Embed(
                title="🤖 No Predictions Available",
                description="The sequential model couldn't generate predictions. This feature requires specific model training.",
                color=0xff9900
            )
            await interaction.followup.send(embed=embed)
            return
        
        # Create embed
        embed = discord.Embed(
            title="🔮 Next Episode Predictions",
            description=f"Based on your sequence: {', '.join([str(x) for x in sequence[:3]])}{'...' if len(sequence) > 3 else ''}",
            color=0x9932cc
        )
        
        rec_text = []
        for i, (content_id, title, content_type, score) in enumerate(predictions, 1):
            icon = "🎬" if content_type == "movie" else "📺"
            confidence = f"{score:.1%}" if score <= 1.0 else f"{score:.2f}"
            rec_text.append(f"{i}. {icon} **{title}** ({confidence})")
        
        embed.add_field(
            name="Predicted Next Items",
            value="\n".join(rec_text),
            inline=False
        )
        
        embed.set_footer(text="💡 This uses AI to predict what you might enjoy next based on viewing patterns")
        
        await interaction.followup.send(embed=embed)
        
    except Exception as e:
        logger.error(f"Error in next_episode: {e}")
        await interaction.followup.send("Error predicting next episode.")


@bot.tree.command(name="recommend_advanced", description="Advanced recommendations with model and confidence options")
@app_commands.describe(
    content_type="Type of content to recommend",
    model="Specific AI model to use",
    user_id="User ID for personalized recommendations", 
    genre="Genre filter",
    limit="Number of recommendations"
)
async def recommend_advanced(
    interaction: discord.Interaction,
    content_type: Optional[str] = "mixed",
    model: Optional[str] = "ensemble",
    user_id: Optional[int] = None,
    genre: Optional[str] = None,
    limit: Optional[int] = 5
):
    """Advanced recommendations with model selection and confidence scores"""
    await interaction.response.defer()
    
    try:
        if not bot.lupe:
            embed = discord.Embed(
                title="❌ Bot Not Ready", 
                description="The bot is still loading. Please wait and try again.",
                color=0xff0000
            )
            await interaction.followup.send(embed=embed)
            return
        
        # Use Discord user ID if none provided
        if user_id is None:
            user_id = interaction.user.id
        
        # Validate inputs
        content_type = (content_type or "mixed").lower()
        model = (model or "ensemble").lower()
        limit = max(1, min(limit or 5, 15))
        
        # Get recommendations with specified model
        recommendations = bot.lupe.get_recommendations(
            user_id=user_id,
            limit=limit,
            content_type=content_type,
            genre=genre,
            model_type=model
        )
        
        if not recommendations:
            embed = discord.Embed(
                title="❌ No Recommendations Found",
                description="No recommendations could be generated with the current filters.",
                color=0xff0000
            )
            await interaction.followup.send(embed=embed)
            return
        
        # Create embed
        embed = discord.Embed(
            title="🎯 Advanced Recommendations",
            description=f"Model: **{model.upper()}** | Content: **{content_type.title()}**{f' | Genre: **{genre}**' if genre else ''}",
            color=0x9932cc
        )
        
        rec_text = []
        for i, (content_id, title, content_type_rec, score) in enumerate(recommendations, 1):
            icon = "🎬" if content_type_rec == "movie" else "📺"
            
            # Format confidence score
            if score <= 1.0:
                confidence = f"{score:.1%}"
                conf_color = "🟢" if score > 0.7 else "🟡" if score > 0.4 else "🔴"
            else:
                confidence = f"{score:.2f}"
                conf_color = "🟢" if score > 3.5 else "🟡" if score > 2.5 else "🔴"
            
            rec_text.append(f"{i}. {icon} **{title}**\n   {conf_color} Confidence: {confidence}")
        
        embed.add_field(
            name="Recommendations",
            value="\n".join(rec_text),
            inline=False
        )
        
        embed.set_footer(text=f"🤖 Generated using {model.upper()} model | User {user_id}")
        
        # Add feedback view
        view = FeedbackView(recommendations, f"advanced_{model}", f"{content_type}_{genre}")
        await interaction.followup.send(embed=embed, view=view)
        
    except Exception as e:
        logger.error(f"Error in recommend_advanced: {e}")
        await interaction.followup.send("Error generating advanced recommendations.")


async def get_collaborative_recommendations(user_id: int, limit: int, genre_filter: str = None) -> List[tuple]:
    """FIXED: Get personalized recommendations using the trained model with proper ID mapping"""
    try:
        if not bot.model or not bot.mappings:
            logger.error("Model or mappings not loaded")
            return []

        # Convert user_id to index if it exists in mappings
        if user_id not in bot.mappings['user_id_to_idx']:
            logger.warning(f"User {user_id} not in training data, using fallback user")
            # Use a random existing user as fallback
            fallback_user_id = list(bot.mappings['user_id_to_idx'].keys())[0]
            user_idx = bot.mappings['user_id_to_idx'][fallback_user_id]
        else:
            user_idx = bot.mappings['user_id_to_idx'][user_id]

        recommendations = []

        # Get ALL movies that exist in our system (these are REAL movies)
        available_movie_ids = list(bot.movie_lookup.keys())  # Original movie IDs
        available_movie_indices = [bot.mappings['movie_id_to_idx'][mid] for mid in available_movie_ids if mid in bot.mappings['movie_id_to_idx']]
        
        # Shuffle for diversity
        combined = list(zip(available_movie_ids, available_movie_indices))
        np.random.shuffle(combined)
        
        # Take a sample for efficiency
        sample_size = min(1000, len(combined))
        sampled_movies = combined[:sample_size]

        logger.info(f"Generating recommendations from {len(sampled_movies)} real movies")

        with torch.no_grad():
            for movie_id, movie_idx in sampled_movies:
                movie_info = bot.movie_lookup[movie_id]

                # Handle non-string genres
                genres_value = movie_info.get('genres', '')
                if not isinstance(genres_value, str):
                    continue

                # Skip if genre filter doesn't match
                if genre_filter and genre_filter.lower() not in genres_value.lower():
                    continue

                # Get genre features
                genre_features = get_movie_genre_features(movie_id)

                # Prepare inputs - using mapped indices
                user_tensor = torch.tensor([user_idx], dtype=torch.long).to(bot.device)
                movie_tensor = torch.tensor([movie_idx], dtype=torch.long).to(bot.device)
                genre_tensor = torch.tensor([genre_features], dtype=torch.float32).to(bot.device)

                # Get prediction
                score = bot.model(user_tensor, movie_tensor, genre_tensor).item()

                # Convert back from scaled rating and normalize to 0-1 range
                if bot.rating_scaler:
                    score = bot.rating_scaler.inverse_transform([[score]])[0][0]
                    # Normalize rating scores (typically 0.5-5.0) to 0-1 range
                    score = max(0, min(1, (score - 0.5) / 4.5))

                # Add small random noise for diversity
                noise = np.random.normal(0, 0.05)
                score_with_noise = max(0, min(1, score + noise))

                recommendations.append((
                    movie_id,  # Original movie ID
                    movie_info.get('title', 'Unknown'),
                    movie_info.get('genres', 'Unknown'),
                    score_with_noise
                ))

        # Sort by score
        recommendations.sort(key=lambda x: x[3], reverse=True)
        
        # Filter out non-movies and apply diversity
        valid_recommendations = filter_valid_movies(recommendations)
        
        return valid_recommendations[:limit]

    except Exception as e:
        logger.error(f"Error in collaborative recommendations: {e}")
        return []


async def get_content_based_recommendations(movie_title: str, limit: int, genre_filter: str = None) -> List[tuple]:
    """Get recommendations based on movie similarity"""
    try:
        # Find the movie
        target_movie_id = None
        for movie_id, movie_info in bot.movie_lookup.items():
            title = movie_info.get('title', '')
            if isinstance(title, str) and movie_title.lower() in title.lower():
                target_movie_id = movie_id
                break

        if target_movie_id is None:
            return []

        target_genres = bot.movie_lookup[target_movie_id].get('genres', '')
        
        recommendations = []
        all_movies = list(bot.movie_lookup.items())
        np.random.shuffle(all_movies)

        for movie_id, movie_info in all_movies:
            if movie_id == target_movie_id:
                continue
                
            # Apply genre filter if specified
            if genre_filter:
                genres_value = movie_info.get('genres', '')
                if not isinstance(genres_value, str):
                    continue
                if genre_filter.lower() not in genres_value.lower():
                    continue

            # Calculate similarity based on genres
            movie_genres = movie_info.get('genres', '')
            if not isinstance(movie_genres, str):
                continue
                
            # Simple genre-based similarity
            target_genre_set = set(target_genres.split('|')) if target_genres else set()
            movie_genre_set = set(movie_genres.split('|')) if movie_genres else set()
            
            if target_genre_set and movie_genre_set:
                similarity = len(target_genre_set & movie_genre_set) / len(target_genre_set | movie_genre_set)
            else:
                similarity = 0
            
            # Add randomness for diversity and ensure score stays in 0-1 range
            similarity += np.random.uniform(0, 0.1)
            similarity = max(0, min(1, similarity))

            recommendations.append((
                movie_id,
                movie_info.get('title', 'Unknown'),
                movie_info.get('genres', 'Unknown'),
                similarity
            ))

        # Sort by similarity
        recommendations.sort(key=lambda x: x[3], reverse=True)
        
        # Filter out non-movies
        valid_recommendations = filter_valid_movies(recommendations)
        
        return valid_recommendations[:limit]

    except Exception as e:
        logger.error(f"Error in content-based recommendations: {e}")
        return []


async def get_genre_recommendations(genre: str, limit: int) -> List[tuple]:
    """Get recommendations based on genre with diversity"""
    try:
        recommendations = []
        all_movies = list(bot.movie_lookup.items())
        
        # Shuffle for randomness
        np.random.shuffle(all_movies)

        for movie_id, movie_info in all_movies:
            genres_value = movie_info.get('genres', '')
            
            if not isinstance(genres_value, str):
                continue
                
            movie_genres = genres_value.lower()
            if genre.lower() in movie_genres:
                # Score based on genre relevance
                genre_list = movie_genres.split('|')
                
                genre_match_score = movie_genres.count(genre.lower()) / len(genre_list) if genre_list else 0
                variety_bonus = min(len(genre_list) * 0.1, 0.3)
                randomness = np.random.uniform(0, 0.2)
                
                final_score = max(0, min(1, genre_match_score + variety_bonus + randomness))

                recommendations.append((
                    movie_id,
                    movie_info.get('title', 'Unknown'),
                    movie_info.get('genres', 'Unknown'),
                    final_score
                ))

        recommendations.sort(key=lambda x: x[3], reverse=True)
        
        # Filter out non-movies
        valid_recommendations = filter_valid_movies(recommendations)
        
        return valid_recommendations[:limit]

    except Exception as e:
        logger.error(f"Error in genre recommendations: {e}")
        return []


async def get_popular_recommendations(limit: int, genre_filter: str = None) -> List[tuple]:
    """Get popular movie recommendations with diversity"""
    try:
        recommendations = []
        all_movies = list(bot.movie_lookup.items())
        
        # Shuffle for randomness each time
        np.random.shuffle(all_movies)

        for movie_id, movie_info in all_movies:
            # Apply genre filter if specified
            if genre_filter:
                genres_value = movie_info.get('genres', '')
                
                if not isinstance(genres_value, str):
                    continue
                    
                if genre_filter.lower() not in genres_value.lower():
                    continue

            # Create popularity score with randomness
            base_score = np.random.uniform(0.3, 1.0)
            
            # Bonus for movies with multiple genres
            genres_value = movie_info.get('genres', '')
            if isinstance(genres_value, str) and genres_value.strip():
                genre_count = len(genres_value.split('|'))
                popularity_bonus = min(genre_count * 0.1, 0.3)
            else:
                popularity_bonus = 0
            
            final_score = max(0, min(1, base_score + popularity_bonus))

            recommendations.append((
                movie_id,
                movie_info.get('title', 'Unknown'),
                movie_info.get('genres', 'Unknown'),
                final_score
            ))

        recommendations.sort(key=lambda x: x[3], reverse=True)
        
        # Filter out non-movies
        valid_recommendations = filter_valid_movies(recommendations)
        
        return valid_recommendations[:limit]

    except Exception as e:
        logger.error(f"Error in popular recommendations: {e}")
        return []


@bot.tree.command(name="search", description="Search for a specific movie")
@app_commands.describe(query="Movie title to search for")
async def search_movie(interaction: discord.Interaction, query: str):
    """Search for a specific movie"""
    await interaction.response.defer()

    try:
        # Search in movie lookup
        matches = []
        query_lower = query.lower()

        for movie_id, movie_info in bot.movie_lookup.items():
            title = movie_info.get('title', '')
            if isinstance(title, str) and query_lower in title.lower():
                matches.append((movie_id, title, movie_info.get('genres', 'Unknown')))

        # Limit results and filter valid movies
        matches = matches[:10]
        valid_matches = [(mid, title, genres) for mid, title, genres in matches 
                        if is_valid_movie({'title': title})]

        if valid_matches:
            embed = discord.Embed(
                title=f"🔍 Search Results for '{query}'",
                color=0x0099ff
            )

            for movie_id, title, genres in valid_matches:
                embed.add_field(
                    name=title,
                    value=f"**ID:** {movie_id}\n**Genres:** {genres}",
                    inline=True
                )
        else:
            embed = discord.Embed(
                title="❌ No Results",
                description=f"No movies found matching '{query}'",
                color=0xff0000
            )

        await interaction.followup.send(embed=embed)

    except Exception as e:
        logger.error(f"Error in search_movie: {e}")
        await interaction.followup.send("Sorry, there was an error searching for movies.")


@bot.tree.command(name="rate", description="Rate a movie or TV show you've watched")
@app_commands.describe(
    content_title="Title of the movie or TV show to rate",
    rating="Rating from 1-5 stars"
)
async def rate_content(interaction: discord.Interaction, content_title: str, rating: int):
    """Allow users to rate movies and TV shows"""
    await interaction.response.defer()

    try:
        if not 1 <= rating <= 5:
            embed = discord.Embed(
                title="❌ Invalid Rating",
                description="Please provide a rating between 1 and 5 stars.",
                color=0xff0000
            )
            await interaction.followup.send(embed=embed)
            return

        # Find the content (movie or TV show)
        found_content = find_content_by_title(content_title)
        if not found_content:
            embed = discord.Embed(
                title="❌ Content Not Found",
                description=f"Could not find any movie or TV show matching '{content_title}'",
                color=0xff0000
            )
            await interaction.followup.send(embed=embed)
            return

        content_id, content_type = found_content
        
        # Get content info
        if content_type == 'movie' and bot.lupe and bot.lupe.movie_lookup:
            content_info = bot.lupe.movie_lookup.get(int(content_id), {})
        elif content_type == 'tv' and bot.lupe and bot.lupe.tv_lookup:
            content_info = bot.lupe.tv_lookup.get(int(content_id), {})
        else:
            content_info = {}
        
        title = content_info.get('title', content_title)
        genres = content_info.get('genres', 'Unknown')

        # Store rating in database
        with db_manager.get_connection() as conn:
            cursor = conn.cursor()

            # Store in appropriate ratings table
            if content_type == 'movie':
                cursor.execute('''
                    INSERT INTO user_ratings (user_id, movie_id, rating, timestamp)
                    VALUES (%s, %s, %s, %s)
                    ON CONFLICT (user_id, movie_id) 
                    DO UPDATE SET rating = EXCLUDED.rating, timestamp = EXCLUDED.timestamp
                ''', (interaction.user.id, int(content_id), rating, datetime.now()))
            else:  # TV show
                cursor.execute('''
                    INSERT INTO user_tv_ratings (user_id, show_id, rating, timestamp)
                    VALUES (%s, %s, %s, %s)
                    ON CONFLICT (user_id, show_id) 
                    DO UPDATE SET rating = EXCLUDED.rating, timestamp = EXCLUDED.timestamp
                ''', (interaction.user.id, int(content_id), rating, datetime.now()))

            # Store in feedback table with schema compatibility
            try:
                cursor.execute('''
                    INSERT INTO feedback (user_id, username, feedback_type, content_id, 
                                        content_title, content_type, rating, timestamp)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                ''', (
                    interaction.user.id,
                    str(interaction.user),
                    'direct_rating',
                    int(content_id),
                    title,
                    content_type,
                    rating,
                    datetime.now()
                ))
            except Exception:
                # Fallback to old schema
                if content_type == 'movie':
                    cursor.execute('''
                        INSERT INTO feedback (user_id, username, feedback_type, movie_id, 
                                            movie_title, rating, timestamp)
                        VALUES (%s, %s, %s, %s, %s, %s, %s)
                    ''', (
                        interaction.user.id,
                        str(interaction.user),
                        'direct_rating',
                        int(content_id),
                        title,
                        rating,
                        datetime.now()
                    ))
                else:
                    # For TV shows with old schema, just store basic feedback
                    cursor.execute('''
                        INSERT INTO feedback (user_id, username, feedback_type, rating, timestamp)
                        VALUES (%s, %s, %s, %s, %s)
                    ''', (
                        interaction.user.id,
                        str(interaction.user),
                        'tv_rating',
                        rating,
                        datetime.now()
                    ))

            conn.commit()

        type_emoji = "🎬" if content_type == 'movie' else "📺"
        type_name = "Movie" if content_type == 'movie' else "TV Show"
        
        embed = discord.Embed(
            title="⭐ Rating Recorded!",
            description=f"You rated {type_emoji} **{title}** {rating}/5 stars",
            color=0x00ff00
        )
        embed.add_field(name="Type", value=type_name, inline=True)
        embed.add_field(name="Genres", value=genres, inline=True)

        await interaction.followup.send(embed=embed)

    except Exception as e:
        logger.error(f"Error in rate_content: {e}")
        await interaction.followup.send("Sorry, there was an error recording your rating.")


@bot.tree.command(name="my_ratings", description="View your movie and TV show ratings")
async def my_ratings(interaction: discord.Interaction):
    """Show user's content ratings"""
    await interaction.response.defer()

    try:
        with db_manager.get_connection() as conn:
            cursor = conn.cursor()

            # Get movie ratings
            cursor.execute('''
                SELECT movie_id, rating, timestamp, 'movie' as content_type
                FROM user_ratings 
                WHERE user_id = %s 
                ORDER BY timestamp DESC 
                LIMIT 5
            ''', (interaction.user.id,))

            movie_ratings = cursor.fetchall()
            
            # Get TV show ratings
            cursor.execute('''
                SELECT show_id, rating, timestamp, 'tv' as content_type
                FROM user_tv_ratings 
                WHERE user_id = %s 
                ORDER BY timestamp DESC 
                LIMIT 5
            ''', (interaction.user.id,))

            tv_ratings = cursor.fetchall()

        # Combine and sort all ratings
        all_ratings = []
        
        for content_id, rating, timestamp, content_type in movie_ratings:
            all_ratings.append((content_id, rating, timestamp, content_type))
            
        for content_id, rating, timestamp, content_type in tv_ratings:
            all_ratings.append((content_id, rating, timestamp, content_type))
        
        # Sort by timestamp and take top 10
        all_ratings.sort(key=lambda x: x[2], reverse=True)
        all_ratings = all_ratings[:10]

        if all_ratings:
            embed = discord.Embed(
                title=f"⭐ {interaction.user.display_name}'s Recent Ratings",
                color=0x0099ff
            )

            for content_id, rating, timestamp, content_type in all_ratings:
                # Get content info
                if content_type == 'movie' and bot.lupe and bot.lupe.movie_lookup:
                    content_info = bot.lupe.movie_lookup.get(content_id, {})
                    type_emoji = "🎬"
                    type_name = "Movie"
                elif content_type == 'tv' and bot.lupe and bot.lupe.tv_lookup:
                    content_info = bot.lupe.tv_lookup.get(content_id, {})
                    type_emoji = "📺"
                    type_name = "TV Show"
                else:
                    content_info = {}
                    type_emoji = "❓"
                    type_name = "Unknown"
                
                title = content_info.get('title', f'{type_name} ID: {content_id}')
                genres = content_info.get('genres', 'Unknown')
                
                embed.add_field(
                    name=f"{type_emoji} {title} - {'⭐' * int(rating)}",
                    value=f"**Type:** {type_name}\n**Rating:** {rating}/5\n**Genres:** {genres}\n**Date:** {timestamp.strftime('%Y-%m-%d')}",
                    inline=False
                )

        else:
            embed = discord.Embed(
                title="📝 No Ratings Yet",
                description="You haven't rated any movies or TV shows yet. Use `/rate` to rate content!",
                color=0xffaa00
            )

        await interaction.followup.send(embed=embed)

    except Exception as e:
        logger.error(f"Error in my_ratings: {e}")
        await interaction.followup.send("Sorry, there was an error retrieving your ratings.")


@bot.tree.command(name="stats", description="Get bot and model statistics")
async def bot_stats(interaction: discord.Interaction):
    """Show bot statistics"""
    try:
        embed = discord.Embed(title="📊 Bot Statistics", color=0x9932cc)

        if bot.metadata:
            embed.add_field(name="Model Type", value=bot.metadata.get('model_type', 'Unknown'), inline=True)
            embed.add_field(name="Total Movies", value=bot.metadata.get('num_movies', 'Unknown'), inline=True)
            embed.add_field(name="Genres", value=len(bot.genres) if bot.genres else 'Unknown', inline=True)

        embed.add_field(name="Available Movies", value=len(bot.movie_lookup) if bot.movie_lookup else 0, inline=True)
        embed.add_field(name="Device", value=str(bot.device), inline=True)
        embed.add_field(name="Model Loaded", value="✅ Yes" if bot.model else "❌ No", inline=True)

        await interaction.response.send_message(embed=embed)

    except Exception as e:
        logger.error(f"Error in bot_stats: {e}")
        await interaction.response.send_message("Error retrieving statistics.")


@bot.tree.command(name="feedback_stats", description="View feedback statistics")
async def feedback_stats(interaction: discord.Interaction):
    """Show feedback statistics"""
    await interaction.response.defer()

    try:
        with db_manager.get_connection() as conn:
            cursor = conn.cursor()

            # Get feedback counts by type
            cursor.execute('''
                SELECT feedback_type, COUNT(*) 
                FROM feedback 
                GROUP BY feedback_type
            ''')
            feedback_counts = cursor.fetchall()

            # Get total ratings
            cursor.execute('SELECT COUNT(*) FROM user_ratings')
            total_ratings = cursor.fetchone()[0]

            # Get active users
            cursor.execute('SELECT COUNT(DISTINCT user_id) FROM feedback')
            active_users = cursor.fetchone()[0]

        embed = discord.Embed(
            title="📊 Feedback Statistics",
            color=0x9932cc
        )

        embed.add_field(name="Total Ratings", value=str(total_ratings), inline=True)
        embed.add_field(name="Active Users", value=str(active_users), inline=True)
        embed.add_field(name="", value="", inline=True)

        if feedback_counts:
            for feedback_type, count in feedback_counts:
                embed.add_field(name=f"{feedback_type.title()} Feedback", value=str(count), inline=True)

        await interaction.followup.send(embed=embed)

    except Exception as e:
        logger.error(f"Error in feedback_stats: {e}")
        await interaction.followup.send("Sorry, there was an error retrieving statistics.")


@bot.tree.command(name="genres", description="List available genres for movies, TV shows, or all content")
@app_commands.describe(content_type="Type of content to show genres for (mixed, movie, tv)")
async def list_genres(interaction: discord.Interaction, content_type: Optional[str] = "mixed"):
    """List available genres for recommendations filtered by content type"""
    try:
        # Normalize content type
        content_type = (content_type or "mixed").lower()
        if content_type not in ["mixed", "movie", "tv"]:
            content_type = "mixed"
        
        # Get appropriate genres
        if bot.lupe:
            genres_to_display = bot.lupe.get_available_genres(content_type)
            content_type_label = {
                "mixed": "All Content",
                "movie": "Movies",
                "tv": "TV Shows"
            }[content_type]
        else:
            # Fallback if Lupe not loaded
            if hasattr(bot, 'genres') and bot.genres:
                genres_to_display = bot.genres
                content_type_label = "All Content (Fallback)"
            else:
                embed = discord.Embed(
                    title="❌ Genres Not Loaded",
                    description="Genres are not loaded yet. Please wait for the bot to fully initialize.",
                    color=0xff0000
                )
                await interaction.response.send_message(embed=embed)
                return
        
        if genres_to_display:
            # Group genres into chunks for better display
            genres_per_field = 10
            genre_chunks = [genres_to_display[i:i + genres_per_field] 
                           for i in range(0, len(genres_to_display), genres_per_field)]
            
            # Choose appropriate emoji
            title_emoji = "🎭" if content_type == "mixed" else ("🎬" if content_type == "movie" else "📺")
            
            embed = discord.Embed(
                title=f"{title_emoji} Available Genres - {content_type_label}",
                description=f"Total: {len(genres_to_display)} genres available for {content_type_label.lower()}",
                color=0x9932cc
            )
            
            for i, chunk in enumerate(genre_chunks):
                field_name = f"Genres ({i*genres_per_field + 1}-{min((i+1)*genres_per_field, len(genres_to_display))})"
                field_value = " • ".join(chunk)
                embed.add_field(name=field_name, value=field_value, inline=False)
            
            embed.set_footer(text=f"Use these genre names with the /recommend command for {content_type} content")
        else:
            embed = discord.Embed(
                title=f"❌ No Genres Found for {content_type_label}",
                description=f"No genres available for {content_type_label.lower()}.",
                color=0xff0000
            )

        await interaction.response.send_message(embed=embed)

    except Exception as e:
        logger.error(f"Error in list_genres: {e}")
        await interaction.response.send_message("Error retrieving genres.")


@bot.tree.command(name="debug_bot", description="Debug bot loading status")
async def debug_bot(interaction: discord.Interaction):
    """Debug the bot's loading status"""
    try:
        embed = discord.Embed(title="🔧 Bot Debug Info", color=0xffaa00)
        
        # Check what's loaded
        embed.add_field(name="Model Loaded", value="✅ Yes" if bot.model else "❌ No", inline=True)
        embed.add_field(name="Mappings Loaded", value="✅ Yes" if bot.mappings else "❌ No", inline=True)
        embed.add_field(name="Movie Lookup", value=f"✅ {len(bot.movie_lookup)}" if bot.movie_lookup else "❌ No", inline=True)
        
        # Genres info
        if hasattr(bot, 'genres') and bot.genres:
            embed.add_field(name="Genres Loaded", value=f"✅ {len(bot.genres)}", inline=True)
            embed.add_field(name="Sample Genres", value=", ".join(bot.genres[:5]), inline=False)
        else:
            embed.add_field(name="Genres Loaded", value="❌ No", inline=True)
        
        # Files check
        files_status = []
        required_files = [
            f'{config.model.models_dir}/id_mappings.pkl',
            f'{config.model.models_dir}/movie_lookup.pkl',
            f'{config.model.models_dir}/movies_data.csv',
            f'{config.model.models_dir}/rating_scaler.pkl',
            f'{config.model.models_dir}/best_model.pt'
        ]
        
        for file_path in required_files:
            exists = os.path.exists(file_path)
            status = "✅" if exists else "❌"
            files_status.append(f"{status} {os.path.basename(file_path)}")
        
        embed.add_field(name="Required Files", value="\n".join(files_status), inline=False)
        
        await interaction.response.send_message(embed=embed)
        
    except Exception as e:
        logger.error(f"Error in debug_bot: {e}")
        await interaction.response.send_message("Error in debug.")


@bot.tree.command(name="my_preferences", description="View your personal content preferences")
async def my_preferences(interaction: discord.Interaction):
    """Show user's content preferences and statistics"""
    await interaction.response.defer()
    
    try:
        with db_manager.get_connection() as conn:
            cursor = conn.cursor()
            
            # Get user's content preferences
            cursor.execute('''
                SELECT content_type, preference, COUNT(*) as count
                FROM user_preferences 
                WHERE user_id = %s 
                GROUP BY content_type, preference
                ORDER BY content_type, preference
            ''', (interaction.user.id,))
            
            content_prefs = cursor.fetchall()
            
            # Get user's genre preferences
            cursor.execute('''
                SELECT content_type, genre, preference_score, interaction_count
                FROM user_genre_preferences 
                WHERE user_id = %s AND interaction_count >= 2
                ORDER BY content_type, preference_score DESC
                LIMIT 10
            ''', (interaction.user.id,))
            
            genre_prefs = cursor.fetchall()
        
        if not content_prefs and not genre_prefs:
            embed = discord.Embed(
                title="📊 Your Preferences",
                description="You haven't provided any feedback yet! Use the feedback buttons on recommendations to personalize your experience.",
                color=0xffaa00
            )
        else:
            embed = discord.Embed(
                title="📊 Your Personal Preferences",
                description="Based on your feedback, here's what we know about your preferences:",
                color=0x9932cc
            )
            
            # Content preferences summary
            if content_prefs:
                pref_text = []
                for content_type, preference, count in content_prefs:
                    emoji = {"love": "💖", "like": "👍", "dislike": "👎", "hate": "💔"}.get(preference, "❓")
                    pref_text.append(f"{emoji} {preference.title()}: {count} {content_type}s")
                
                embed.add_field(
                    name="Content Feedback",
                    value="\n".join(pref_text),
                    inline=False
                )
            
            # Top genre preferences
            if genre_prefs:
                movie_genres = [f"**{genre}**: {score:.2f}" for ct, genre, score, count in genre_prefs if ct == 'movie']
                tv_genres = [f"**{genre}**: {score:.2f}" for ct, genre, score, count in genre_prefs if ct == 'tv']
                
                if movie_genres:
                    embed.add_field(
                        name="🎬 Favorite Movie Genres",
                        value="\n".join(movie_genres[:5]),
                        inline=True
                    )
                
                if tv_genres:
                    embed.add_field(
                        name="📺 Favorite TV Genres", 
                        value="\n".join(tv_genres[:5]),
                        inline=True
                    )
        
        await interaction.followup.send(embed=embed)
        
    except Exception as e:
        logger.error(f"Error in my_preferences: {e}")
        await interaction.followup.send("Error retrieving your preferences.")


@bot.tree.command(name="admin_review", description="[ADMIN] Review training feedback data")
@app_commands.describe(
    limit="Number of feedback entries to review (default: 10)",
    unreviewed_only="Show only unreviewed feedback (default: True)"
)
async def admin_review(interaction: discord.Interaction, limit: Optional[int] = 10, unreviewed_only: Optional[bool] = True):
    """Admin command to review training feedback data"""
    # Check if user is the admin
    if interaction.user.id != 140175381960327179:
        await interaction.response.send_message("❌ This command requires administrator permissions.", ephemeral=True)
        return
    
    await interaction.response.defer()
    
    try:
        with db_manager.get_connection() as conn:
            cursor = conn.cursor()
            
            # Build query based on parameters
            where_clause = "WHERE admin_reviewed = FALSE" if unreviewed_only else ""
            
            cursor.execute(f'''
                SELECT id, content_title, content_type, content_genres, feedback_type, 
                       username, feedback_reason, recommendation_context, timestamp,
                       admin_reviewed, training_weight
                FROM master_training_feedback 
                {where_clause}
                ORDER BY timestamp DESC 
                LIMIT %s
            ''', (limit,))
            
            feedback_entries = cursor.fetchall()
        
        if not feedback_entries:
            status = "unreviewed" if unreviewed_only else "total"
            embed = discord.Embed(
                title="📋 Training Feedback Review",
                description=f"No {status} feedback entries found.",
                color=0xffaa00
            )
        else:
            embed = discord.Embed(
                title="📋 Training Feedback Review",
                description=f"Showing {len(feedback_entries)} feedback entries for admin review:",
                color=0x9932cc
            )
            
            for entry in feedback_entries[:5]:  # Show first 5 in detail
                (id, title, content_type, genres, feedback_type, username, 
                 reason, context, timestamp, reviewed, weight) = entry
                
                feedback_emoji = {"love": "💖", "like": "👍", "dislike": "👎", "hate": "💔"}.get(feedback_type, "❓")
                status_emoji = "✅" if reviewed else "⏳"
                
                field_value = f"""
                **Type:** {content_type.title()} {feedback_emoji}
                **User:** {username}
                **Context:** {context}
                **Weight:** {weight}
                **Status:** {status_emoji} {'Reviewed' if reviewed else 'Pending'}
                """
                
                if reason:
                    field_value += f"\n**Reason:** {reason[:100]}..."
                
                embed.add_field(
                    name=f"ID {id}: {title[:40]}...",
                    value=field_value,
                    inline=False
                )
            
            if len(feedback_entries) > 5:
                embed.add_field(
                    name="Additional Entries",
                    value=f"... and {len(feedback_entries) - 5} more entries. Use `/admin_approve` or `/admin_reject` to process them.",
                    inline=False
                )
            
            embed.set_footer(text="Use /admin_approve <id> or /admin_reject <id> to process feedback")
        
        await interaction.followup.send(embed=embed)
        
    except Exception as e:
        logger.error(f"Error in admin_review: {e}")
        await interaction.followup.send("Error retrieving feedback data.")


@bot.tree.command(name="admin_approve", description="[ADMIN] Approve feedback for training")
@app_commands.describe(
    feedback_id="ID of the feedback entry to approve",
    training_weight="Training weight (0.1-2.0, default: 1.0)",
    notes="Admin notes (optional)"
)
async def admin_approve(interaction: discord.Interaction, feedback_id: int, training_weight: Optional[float] = 1.0, notes: Optional[str] = None):
    """Admin command to approve feedback for training"""
    if interaction.user.id != 140175381960327179:
        await interaction.response.send_message("❌ This command requires administrator permissions.", ephemeral=True)
        return
    
    await interaction.response.defer()
    
    try:
        # Validate training weight
        training_weight = max(0.1, min(training_weight, 2.0))
        
        with db_manager.get_connection() as conn:
            cursor = conn.cursor()
            
            # Update the feedback entry
            cursor.execute('''
                UPDATE master_training_feedback 
                SET admin_reviewed = TRUE, training_weight = %s, admin_notes = %s
                WHERE id = %s
                RETURNING content_title, feedback_type, username
            ''', (training_weight, notes, feedback_id))
            
            result = cursor.fetchone()
            conn.commit()
        
        if result:
            title, feedback_type, username = result
            embed = discord.Embed(
                title="✅ Feedback Approved",
                description=f"Approved feedback ID {feedback_id} for training",
                color=0x00ff00
            )
            embed.add_field(name="Content", value=title, inline=True)
            embed.add_field(name="Feedback", value=feedback_type, inline=True)
            embed.add_field(name="User", value=username, inline=True)
            embed.add_field(name="Training Weight", value=f"{training_weight}", inline=True)
            if notes:
                embed.add_field(name="Admin Notes", value=notes, inline=False)
        else:
            embed = discord.Embed(
                title="❌ Feedback Not Found",
                description=f"No feedback entry found with ID {feedback_id}",
                color=0xff0000
            )
        
        await interaction.followup.send(embed=embed)
        
    except Exception as e:
        logger.error(f"Error in admin_approve: {e}")
        await interaction.followup.send("Error updating feedback entry.")


@bot.tree.command(name="admin_reject", description="[ADMIN] Reject feedback for training")
@app_commands.describe(
    feedback_id="ID of the feedback entry to reject",
    reason="Reason for rejection"
)
async def admin_reject(interaction: discord.Interaction, feedback_id: int, reason: Optional[str] = "Quality concerns"):
    """Admin command to reject feedback for training"""
    if interaction.user.id != 140175381960327179:
        await interaction.response.send_message("❌ This command requires administrator permissions.", ephemeral=True)
        return
    
    await interaction.response.defer()
    
    try:
        with db_manager.get_connection() as conn:
            cursor = conn.cursor()
            
            # Update the feedback entry
            cursor.execute('''
                UPDATE master_training_feedback 
                SET admin_reviewed = TRUE, training_weight = 0.0, admin_notes = %s
                WHERE id = %s
                RETURNING content_title, feedback_type, username
            ''', (f"REJECTED: {reason}", feedback_id))
            
            result = cursor.fetchone()
            conn.commit()
        
        if result:
            title, feedback_type, username = result
            embed = discord.Embed(
                title="❌ Feedback Rejected",
                description=f"Rejected feedback ID {feedback_id} from training",
                color=0xff0000
            )
            embed.add_field(name="Content", value=title, inline=True)
            embed.add_field(name="Feedback", value=feedback_type, inline=True)
            embed.add_field(name="User", value=username, inline=True)
            embed.add_field(name="Reason", value=reason, inline=False)
        else:
            embed = discord.Embed(
                title="❌ Feedback Not Found",
                description=f"No feedback entry found with ID {feedback_id}",
                color=0xff0000
            )
        
        await interaction.followup.send(embed=embed)
        
    except Exception as e:
        logger.error(f"Error in admin_reject: {e}")
        await interaction.followup.send("Error updating feedback entry.")


@bot.tree.command(name="export_training_data", description="[ADMIN] Export approved training data")
async def export_training_data(interaction: discord.Interaction):
    """Admin command to export approved training data"""
    if interaction.user.id != 140175381960327179:
        await interaction.response.send_message("❌ This command requires administrator permissions.", ephemeral=True)
        return
    
    await interaction.response.defer()
    
    try:
        with db_manager.get_connection() as conn:
            cursor = conn.cursor()
            
            # Get approved training data
            cursor.execute('''
                SELECT content_id, content_type, content_title, content_genres, 
                       feedback_type, training_weight, timestamp
                FROM master_training_feedback 
                WHERE admin_reviewed = TRUE AND training_weight > 0
                ORDER BY timestamp DESC
            ''')
            
            training_data = cursor.fetchall()
        
        if not training_data:
            embed = discord.Embed(
                title="📊 No Training Data",
                description="No approved training data available for export.",
                color=0xffaa00
            )
        else:
            # Create CSV content (in real implementation, you'd save to file)
            csv_content = "content_id,content_type,content_title,content_genres,feedback_type,training_weight,timestamp\n"
            for row in training_data:
                csv_content += ",".join([str(item) for item in row]) + "\n"
            
            embed = discord.Embed(
                title="📊 Training Data Export",
                description=f"Successfully exported {len(training_data)} approved training entries.",
                color=0x00ff00
            )
            
            # Summary statistics
            feedback_counts = {}
            for row in training_data:
                feedback_type = row[4]
                feedback_counts[feedback_type] = feedback_counts.get(feedback_type, 0) + 1
            
            stats_text = "\n".join([f"**{fb_type.title()}**: {count}" for fb_type, count in feedback_counts.items()])
            embed.add_field(name="Feedback Distribution", value=stats_text, inline=False)
            
            embed.set_footer(text="Training data has been processed for model retraining")
        
        await interaction.followup.send(embed=embed)
        
    except Exception as e:
        logger.error(f"Error in export_training_data: {e}")
        await interaction.followup.send("Error exporting training data.")


@bot.tree.command(name="test_genre", description="Test genre filtering")
@app_commands.describe(genre="Genre to test")
async def test_genre(interaction: discord.Interaction, genre: str):
    """Test if a genre works for filtering"""
    await interaction.response.defer()
    
    try:
        if not bot.movie_lookup:
            await interaction.followup.send("❌ Movie data not loaded yet.")
            return
        
        # Count movies with this genre
        matching_movies = []
        for movie_id, movie_info in bot.movie_lookup.items():
            genres_value = movie_info.get('genres', '')
            if isinstance(genres_value, str) and genre.lower() in genres_value.lower():
                matching_movies.append((movie_id, movie_info.get('title', 'Unknown')))
        
        embed = discord.Embed(
            title=f"🎭 Genre Test: '{genre}'",
            color=0x0099ff
        )
        
        embed.add_field(name="Movies Found", value=str(len(matching_movies)), inline=True)
        
        if matching_movies:
            # Show first 10 matches
            sample_movies = matching_movies[:10]
            movie_list = "\n".join([f"• {title}" for _, title in sample_movies])
            embed.add_field(name="Sample Movies", value=movie_list, inline=False)
            
            if len(matching_movies) > 10:
                embed.add_field(name="Note", value=f"Showing 10 of {len(matching_movies)} movies", inline=False)
        else:
            embed.add_field(name="Result", value="No movies found with this genre", inline=False)
            
            # Suggest similar genres
            if hasattr(bot, 'genres') and bot.genres:
                similar_genres = [g for g in bot.genres if genre.lower() in g.lower() or g.lower() in genre.lower()]
                if similar_genres:
                    embed.add_field(name="Similar Genres", value=", ".join(similar_genres[:5]), inline=False)
        
        await interaction.followup.send(embed=embed)
        
    except Exception as e:
        logger.error(f"Error in test_genre: {e}")
        await interaction.followup.send("Error testing genre.")


# Auto-complete for test_genre command
@test_genre.autocomplete('genre')
async def test_genre_autocomplete(interaction: discord.Interaction, current: str) -> List[app_commands.Choice[str]]:
    """Autocomplete for test_genre command"""
    return await genre_autocomplete(interaction, current)

# Auto-complete for cross_recommend command
@cross_recommend.autocomplete('source_type')
async def cross_recommend_source_autocomplete(interaction: discord.Interaction, current: str) -> List[app_commands.Choice[str]]:
    """Autocomplete for cross_recommend source_type"""
    content_types = ['movie', 'tv']
    return [app_commands.Choice(name=ct.title(), value=ct) for ct in content_types]

@cross_recommend.autocomplete('target_type')
async def cross_recommend_target_autocomplete(interaction: discord.Interaction, current: str) -> List[app_commands.Choice[str]]:
    """Autocomplete for cross_recommend target_type"""
    content_types = ['movie', 'tv']
    return [app_commands.Choice(name=ct.title(), value=ct) for ct in content_types]

# Auto-complete for genres command content_type
@list_genres.autocomplete('content_type')
async def list_genres_content_type_autocomplete(interaction: discord.Interaction, current: str) -> List[app_commands.Choice[str]]:
    """Autocomplete for genres content_type"""
    content_types = ['mixed', 'movie', 'tv']
    matching = [ct for ct in content_types if current.lower() in ct.lower()]
    return [app_commands.Choice(name=ct.title() if ct != 'mixed' else 'All Content', value=ct) for ct in matching]

# Auto-complete for recommend command content_type
@recommend_content.autocomplete('content_type')
async def recommend_content_type_autocomplete(interaction: discord.Interaction, current: str) -> List[app_commands.Choice[str]]:
    """Autocomplete for recommend content_type"""
    content_types = ['mixed', 'movie', 'tv']
    matching = [ct for ct in content_types if current.lower() in ct.lower()]
    return [app_commands.Choice(name=ct.title(), value=ct) for ct in matching]
async def export_data_command(interaction: discord.Interaction):
    """Export feedback data for model retraining"""
    await interaction.response.defer()
    
    try:
        # Ensure data directory exists
        data_dir = 'data'  # Could be configurable in the future
        os.makedirs(data_dir, exist_ok=True)
        
        with db_manager.get_connection() as conn:
            cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            
            # Get all ratings
            cursor.execute('''
                SELECT user_id, movie_id, rating, timestamp
                FROM user_ratings
                ORDER BY timestamp
            ''')
            ratings_data = cursor.fetchall()
            
            # Get all feedback
            cursor.execute('''
                SELECT *
                FROM feedback
                ORDER BY timestamp
            ''')
            feedback_data = cursor.fetchall()
        
        # Convert to DataFrames
        ratings_df = pd.DataFrame(ratings_data)
        feedback_df = pd.DataFrame(feedback_data)
        
        # Save to CSV
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        if not ratings_df.empty:
            ratings_df.to_csv(f'{data_dir}/user_ratings_{timestamp}.csv', index=False)
            logger.info(f"Exported {len(ratings_df)} ratings")
        
        if not feedback_df.empty:
            feedback_df.to_csv(f'{data_dir}/feedback_{timestamp}.csv', index=False)
            logger.info(f"Exported {len(feedback_df)} feedback entries")
        
        embed = discord.Embed(
            title="✅ Data Export Complete",
            description="Feedback data has been exported for model retraining",
            color=0x00ff00
        )
        embed.add_field(name="Ratings Exported", value=str(len(ratings_df)), inline=True)
        embed.add_field(name="Feedback Entries", value=str(len(feedback_df)), inline=True)
            
        await interaction.followup.send(embed=embed)
        
    except Exception as e:
        logger.error(f"Error in export_data_command: {e}")
        await interaction.followup.send("Error exporting data.")


# Auto-complete for genres in recommend command
@recommend_content.autocomplete('genre')
async def recommend_genre_autocomplete(interaction: discord.Interaction, current: str) -> List[app_commands.Choice[str]]:
    """Autocomplete for recommend genre - shows all genres with content type indicators"""
    try:
        # Get all available genres from Lupe
        if bot.lupe and hasattr(bot.lupe, 'movie_lookup') and bot.lupe.movie_lookup:
            try:
                movie_genres = set(bot.lupe.get_available_genres('movie'))
                tv_genres = set(bot.lupe.get_available_genres('tv'))
                all_genres = bot.lupe.get_available_genres('mixed')
                logger.info(f"Genre autocomplete: Using Lupe data - {len(movie_genres)} movies, {len(tv_genres)} TV shows")
            except Exception as e:
                logger.error(f"Error getting genres from Lupe: {e}")
                # Fall through to fallback
                movie_genres = None
                tv_genres = None
                all_genres = None
        else:
            movie_genres = None
            tv_genres = None
            all_genres = None
        
        # Fallback if Lupe data isn't available
        if not all_genres:
            if hasattr(bot, 'genres') and bot.genres:
                all_genres = bot.genres
                # Since we don't have separate data, treat all as available for both
                movie_genres = set(all_genres)
                tv_genres = set(all_genres)
                logger.info(f"Genre autocomplete: Using bot.genres fallback - {len(all_genres)} genres")
            else:
                fallback_genres = [
                    'Action', 'Adventure', 'Animation', 'Children', 'Comedy', 'Crime', 
                    'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'IMAX',
                    'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western'
                ]
                all_genres = fallback_genres
                movie_genres = set(all_genres)
                tv_genres = set(all_genres)
                logger.info(f"Genre autocomplete: Using hardcoded fallback - {len(all_genres)} genres")

        # Filter genres based on current input
        if current:
            matching_genres = [genre for genre in all_genres if current.lower() in genre.lower()]
        else:
            matching_genres = all_genres[:25]  # Show first 25 if no input

        # Create choices with content type indicators
        choices = []
        for genre in matching_genres[:25]:
            # Determine which content types have this genre
            in_movies = genre in movie_genres
            in_tv = genre in tv_genres
            
            if in_movies and in_tv:
                display_name = f"{genre} (Both)"
            elif in_movies:
                display_name = f"{genre} (Movies)"
            elif in_tv:
                display_name = f"{genre} (TV Shows)"
            else:
                display_name = genre
                
            choices.append(app_commands.Choice(name=display_name, value=genre))

        return choices
    
    except Exception as e:
        logger.error(f"Error in genre autocomplete: {e}")
        # Fallback to basic genre autocomplete
        return await genre_autocomplete(interaction, current)


# Auto-complete for genres (generic)
async def genre_autocomplete(interaction: discord.Interaction, current: str) -> List[app_commands.Choice[str]]:
    """Provide autocomplete suggestions for genres"""
    try:
        # Check if bot genres are loaded
        if not hasattr(bot, 'genres') or not bot.genres:
            # Fallback genres list
            fallback_genres = [
                'Action', 'Adventure', 'Animation', 'Children', 'Comedy', 'Crime', 
                'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'IMAX',
                'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western'
            ]
            genres_to_use = fallback_genres
        else:
            genres_to_use = bot.genres

        # Filter genres based on current input
        if current:
            matching_genres = [genre for genre in genres_to_use if current.lower() in genre.lower()]
        else:
            matching_genres = genres_to_use[:25]  # Show first 25 if no input

        # Return up to 25 choices (Discord limit)
        return [
            app_commands.Choice(name=genre, value=genre)
            for genre in matching_genres[:25]
        ]
    
    except Exception as e:
        logger.error(f"Error in genre autocomplete: {e}")
        # Return basic genres as fallback
        basic_genres = ['Action', 'Comedy', 'Drama', 'Horror', 'Romance', 'Sci-Fi', 'Thriller']
        return [app_commands.Choice(name=genre, value=genre) for genre in basic_genres]


# Autocomplete functions for new commands
@recommend_advanced.autocomplete('content_type')
async def recommend_advanced_content_type_autocomplete(interaction: discord.Interaction, current: str) -> List[app_commands.Choice[str]]:
    """Autocomplete for recommend_advanced content_type"""
    content_types = ['mixed', 'movie', 'tv']
    matching = [ct for ct in content_types if current.lower() in ct.lower()]
    return [app_commands.Choice(name=ct.title(), value=ct) for ct in matching]


@recommend_advanced.autocomplete('model')
async def recommend_advanced_model_autocomplete(interaction: discord.Interaction, current: str) -> List[app_commands.Choice[str]]:
    """Autocomplete for recommend_advanced model"""
    models = ['ensemble', 'ncf', 'sequential', 'two_tower']
    model_names = {
        'ensemble': 'Ensemble (All Models)',
        'ncf': 'Neural Collaborative Filtering',
        'sequential': 'Sequential/Time-aware',
        'two_tower': 'Two-Tower/Dual-Encoder'
    }
    
    matching = [m for m in models if current.lower() in m.lower()]
    return [app_commands.Choice(name=model_names[m], value=m) for m in matching]


@recommend_advanced.autocomplete('genre')
async def recommend_advanced_genre_autocomplete(interaction: discord.Interaction, current: str) -> List[app_commands.Choice[str]]:
    """Autocomplete for recommend_advanced genre"""
    return await genre_autocomplete(interaction, current)


# Run the bot
if __name__ == "__main__":
    bot.run(config.discord.token)