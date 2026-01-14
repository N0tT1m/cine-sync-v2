#!/usr/bin/env python3
"""
Personalized Discord Commands for CineSync v2

New commands to be integrated into main.py:
- /my_recommendations - Get personalized recommendations
- /my_stats - View preference profile
- /rate - Quick rating flow
"""

import discord
from discord import app_commands
from discord.ui import Button, View, Modal, TextInput
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Global variables to be set from main.py
personalized_trainer = None
preference_learner = None
db_manager = None
bot_lupe = None


def setup_personalization(trainer, learner, db_mgr, lupe):
    """Initialize personalization components"""
    global personalized_trainer, preference_learner, db_manager, bot_lupe
    personalized_trainer = trainer
    preference_learner = learner
    db_manager = db_mgr
    bot_lupe = lupe


# Command 1: Personalized Recommendations
async def my_recommendations_command(
    interaction: discord.Interaction,
    count: int = 10,
    content_type: str = 'movie'
):
    """
    Get personalized recommendations based on your viewing history

    Args:
        interaction: Discord interaction
        count: Number of recommendations (default: 10)
        content_type: 'movie' or 'tv' (default: 'movie')
    """
    await interaction.response.defer()

    user_id = interaction.user.id

    try:
        # Get rating count
        rating_count = await personalized_trainer._get_user_rating_count(user_id)

        if rating_count < 5:
            # Not enough data
            embed = discord.Embed(
                title="🔍 Need More Data",
                description=f"You have {rating_count} ratings. Rate at least 5 movies to get personalized recommendations!\n\n"
                           f"Use `/rate` to rate movies you've watched.",
                color=0xffa500
            )
            await interaction.followup.send(embed=embed)
            return

        # For now, we'll get recommendations from the existing system
        # and re-rank them with personalization
        # This requires integration with the existing recommendation system

        # Placeholder: Get some base recommendations
        # In production, this would call the unified API or existing rec system
        with db_manager.get_connection() as conn:
            cursor = conn.cursor()

            # Get popular movies user hasn't rated
            cursor.execute("""
                SELECT m.movie_id, m.title, m.genres, m.year
                FROM movies m
                WHERE m.movie_id NOT IN (
                    SELECT movie_id FROM user_ratings WHERE user_id = %s
                )
                ORDER BY m.popularity DESC NULLS LAST
                LIMIT %s
            """, (user_id, count * 2))

            base_recs = [(row[0], row[1], content_type, 0.5) for row in cursor.fetchall()]

        if not base_recs:
            embed = discord.Embed(
                title="❌ No Recommendations Available",
                description="No new content found to recommend.",
                color=0xff0000
            )
            await interaction.followup.send(embed=embed)
            return

        # Get personalized recommendations
        recommendations = await personalized_trainer.get_personalized_recommendations(
            user_id=user_id,
            top_k=count,
            content_type=content_type,
            base_recommendations=base_recs
        )

        if not recommendations:
            await interaction.followup.send("No personalized recommendations available.")
            return

        # Create embed
        embed = discord.Embed(
            title=f"🎯 Your Personalized Recommendations",
            description=f"Based on your {rating_count} ratings",
            color=0x00ff00
        )

        for i, rec in enumerate(recommendations, 1):
            item_id = rec['item_id']
            title = rec.get('title', f'Movie {item_id}')
            score = rec['score']

            # Get movie info
            movie_info = {}
            if hasattr(bot_lupe, 'movie_lookup'):
                movie_info = bot_lupe.movie_lookup.get(item_id, {})

            genres = movie_info.get('genres', 'Unknown')
            year = movie_info.get('year', '')

            embed.add_field(
                name=f"{i}. {title} ({year})",
                value=f"**Genres**: {genres}\n**Match**: {score:.2%}\n"
                      f"Base: {rec['base_score']:.2f} | Personal: {rec['personalization_score']:.2f}",
                inline=False
            )

        await interaction.followup.send(embed=embed)

    except Exception as e:
        logger.error(f"Error getting personalized recommendations: {e}", exc_info=True)
        await interaction.followup.send(
            f"❌ Error getting recommendations: {e}",
            ephemeral=True
        )


# Command 2: User Profile Stats
async def my_stats_command(interaction: discord.Interaction):
    """Show user's preference profile"""
    await interaction.response.defer()

    user_id = interaction.user.id

    try:
        # Get user patterns
        patterns = await preference_learner.analyze_user_patterns(user_id)

        if patterns['total_ratings'] == 0:
            embed = discord.Embed(
                title="📊 Your Profile",
                description="You haven't rated any content yet! Use `/rate` to get started.",
                color=0xffa500
            )
            await interaction.followup.send(embed=embed)
            return

        # Create embed
        embed = discord.Embed(
            title=f"📊 {interaction.user.name}'s Recommendation Profile",
            description=f"Based on **{patterns['total_ratings']} ratings**",
            color=0x3498db
        )

        # Favorite genres
        if patterns['favorite_genres']:
            genres_text = "\n".join([
                f"**{g['genre']}** - ⭐{g['avg_rating']:.1f} ({g['count']} rated)"
                for g in patterns['favorite_genres'][:5]
            ])
            embed.add_field(
                name="🎭 Favorite Genres",
                value=genres_text,
                inline=False
            )

        # Favorite directors
        if patterns['favorite_directors']:
            directors_text = "\n".join([
                f"**{d['director']}** - ⭐{d['avg_rating']:.1f}"
                for d in patterns['favorite_directors'][:3]
            ])
            embed.add_field(
                name="🎬 Favorite Directors",
                value=directors_text,
                inline=True
            )

        # Favorite actors
        if patterns['favorite_actors']:
            actors_text = "\n".join([
                f"**{a['actor']}** - ⭐{a['avg_rating']:.1f}"
                for a in patterns['favorite_actors'][:3]
            ])
            embed.add_field(
                name="⭐ Favorite Actors",
                value=actors_text,
                inline=True
            )

        # Rating style
        rating_dist = patterns['rating_distribution']
        avg_rating = patterns['avg_rating']

        rating_style = "Generous 😊" if avg_rating > 3.5 else "Harsh 😤" if avg_rating < 2.5 else "Balanced ⚖️"

        embed.add_field(
            name="📈 Rating Style",
            value=f"**Average**: {avg_rating:.1f}/5\n**Style**: {rating_style}\n"
                  f"**Diversity**: {patterns['diversity_score']:.1%}",
            inline=False
        )

        # Preferred decades
        if patterns['preferred_decades']:
            decades_text = ", ".join([d['decade'] for d in patterns['preferred_decades'][:3]])
            embed.add_field(
                name="📅 Preferred Eras",
                value=decades_text,
                inline=False
            )

        await interaction.followup.send(embed=embed)

    except Exception as e:
        logger.error(f"Error getting user stats: {e}", exc_info=True)
        await interaction.followup.send(
            f"❌ Error getting stats: {e}",
            ephemeral=True
        )


# Command 3: Quick Rating Flow
async def rate_movies_command(interaction: discord.Interaction, count: int = 5):
    """Interactive rating flow to quickly rate movies"""
    await interaction.response.defer()

    user_id = interaction.user.id

    try:
        # Get popular movies user hasn't rated yet
        with db_manager.get_connection() as conn:
            cursor = conn.cursor()

            # Get movies user hasn't rated
            cursor.execute("""
                SELECT m.movie_id, m.title, m.genres, m.year
                FROM movies m
                WHERE m.movie_id NOT IN (
                    SELECT movie_id FROM user_ratings WHERE user_id = %s
                )
                ORDER BY m.popularity DESC NULLS LAST
                LIMIT %s
            """, (user_id, count))

            movies = cursor.fetchall()

        if not movies:
            await interaction.followup.send(
                "You've rated all available movies! 🎉",
                ephemeral=True
            )
            return

        # Create embed with movies to rate
        embed = discord.Embed(
            title="⭐ Rate These Movies",
            description="Click the button below to rate these movies and improve your recommendations!",
            color=0xffd700
        )

        for movie_id, title, genres, year in movies:
            embed.add_field(
                name=f"{title} ({year})",
                value=f"Genres: {genres}",
                inline=False
            )

        # Create modal for rating
        view = QuickRatingView(movies, user_id)

        await interaction.followup.send(embed=embed, view=view)

    except Exception as e:
        logger.error(f"Error in rate command: {e}", exc_info=True)
        await interaction.followup.send(
            f"❌ Error: {e}",
            ephemeral=True
        )


# Quick Rating View
class QuickRatingView(View):
    """View for quick movie rating"""
    def __init__(self, movies, user_id):
        super().__init__(timeout=300)
        self.movies = movies
        self.user_id = user_id

    @discord.ui.button(label='Rate Now', style=discord.ButtonStyle.primary)
    async def rate_button(self, interaction: discord.Interaction, button: Button):
        modal = QuickRatingModal(self.movies, self.user_id)
        await interaction.response.send_modal(modal)


class QuickRatingModal(Modal):
    """Modal for quick rating entry"""
    def __init__(self, movies, user_id):
        super().__init__(title="Rate Movies (1-5 or skip)")
        self.movies = movies
        self.user_id = user_id

        # Add inputs for each movie (max 5 due to Discord limits)
        for i, (movie_id, title, _, _) in enumerate(movies[:5]):
            text_input = TextInput(
                label=f"{title[:45]}..." if len(title) > 45 else title,
                placeholder="1-5 (or leave empty)",
                required=False,
                max_length=1
            )
            self.add_item(text_input)

    async def on_submit(self, interaction: discord.Interaction):
        ratings_recorded = 0

        with db_manager.get_connection() as conn:
            cursor = conn.cursor()

            for i, text_input in enumerate(self.children):
                if text_input.value and text_input.value.isdigit():
                    rating = int(text_input.value)
                    if 1 <= rating <= 5:
                        movie_id = self.movies[i][0]

                        # Store rating
                        cursor.execute("""
                            INSERT INTO user_ratings (user_id, movie_id, rating, timestamp)
                            VALUES (%s, %s, %s, NOW())
                            ON CONFLICT (user_id, movie_id)
                            DO UPDATE SET rating = EXCLUDED.rating, timestamp = EXCLUDED.timestamp
                        """, (self.user_id, movie_id, rating))

                        # Update user embedding
                        if personalized_trainer:
                            await personalized_trainer.update_user_embedding(
                                self.user_id,
                                movie_id,
                                'positive' if rating >= 4 else 'negative' if rating <= 2 else 'neutral',
                                rating
                            )

                        ratings_recorded += 1

            conn.commit()

        embed = discord.Embed(
            title="✅ Ratings Recorded!",
            description=f"Recorded {ratings_recorded} ratings. Your recommendations are now more personalized!",
            color=0x00ff00
        )

        await interaction.response.send_message(embed=embed, ephemeral=True)


# Helper function to update existing feedback handler
async def update_user_embedding_from_feedback(user_id: int, recommendations: list, feedback_type: str):
    """
    Update user embeddings based on feedback
    Call this from existing feedback handlers

    Args:
        user_id: Discord user ID
        recommendations: List of (content_id, title, content_type, score) tuples
        feedback_type: 'positive', 'negative', etc.
    """
    if not personalized_trainer:
        return

    try:
        for content_id, title, content_type, score in recommendations:
            await personalized_trainer.update_user_embedding(
                user_id=user_id,
                item_id=content_id,
                feedback_type=feedback_type,
                rating=None
            )
        logger.info(f"Updated embeddings for user {user_id} with {feedback_type} feedback")
    except Exception as e:
        logger.error(f"Error updating embedding from feedback: {e}")
