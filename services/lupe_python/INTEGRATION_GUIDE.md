# Integration Guide for Personalization System

This guide explains how to integrate the new personalization components into the existing Discord bot.

## Files Created

1. `database_extensions.sql` - New database tables
2. `migrate_personalization.py` - Migration script
3. `personalized_trainer.py` - Core personalization engine
4. `preference_learner.py` - Pattern analysis
5. `personalized_commands.py` - New Discord commands
6. `test_personalization.py` - Test script

## Integration Steps

### Step 1: Run Database Migration

```bash
cd services/lupe_python
python migrate_personalization.py
```

This will create the new tables:
- `user_embeddings`
- `user_preferences`
- `user_model_weights`

### Step 2: Modify main.py

Add these imports at the top of `main.py` (around line 30):

```python
from personalized_trainer import PersonalizedTrainer
from preference_learner import PreferenceLearner
import personalized_commands
```

### Step 3: Initialize Personalization Components

In the `on_ready` event handler (around line 2800+), add:

```python
@bot.event
async def on_ready():
    global personalized_trainer, preference_learner

    print(f'Logged in as {bot.user}!')

    # ... existing initialization code ...

    # Initialize personalization components
    try:
        personalized_trainer = PersonalizedTrainer(
            db_manager=db_manager,
            content_manager=bot.lupe,  # or unified_lupe if using unified
            embedding_dim=256
        )

        preference_learner = PreferenceLearner(
            db_manager=db_manager,
            content_manager=bot.lupe  # or unified_lupe if using unified
        )

        # Set up personalized commands
        personalized_commands.setup_personalization(
            personalized_trainer,
            preference_learner,
            db_manager,
            bot.lupe
        )

        print("✅ Personalization system ready!")
    except Exception as e:
        print(f"⚠️  Warning: Personalization system failed to initialize: {e}")
        personalized_trainer = None
        preference_learner = None

    # ... rest of existing code ...
```

### Step 4: Add New Slash Commands

Add these command definitions (around line 2900+):

```python
# Personalized Recommendations Command
@bot.tree.command(
    name="my_recommendations",
    description="Get recommendations personalized just for you"
)
async def my_recommendations(
    interaction: discord.Interaction,
    count: int = 10
):
    """Get personalized recommendations"""
    await personalized_commands.my_recommendations_command(
        interaction,
        count=count,
        content_type='movie'
    )


# User Stats Command
@bot.tree.command(
    name="my_stats",
    description="View your recommendation profile and preferences"
)
async def my_stats(interaction: discord.Interaction):
    """Show user's preference profile"""
    await personalized_commands.my_stats_command(interaction)


# Quick Rating Command
@bot.tree.command(
    name="rate",
    description="Rate movies to improve your recommendations"
)
async def rate_movies(interaction: discord.Interaction, count: int = 5):
    """Interactive rating flow to quickly rate movies"""
    await personalized_commands.rate_movies_command(interaction, count)
```

### Step 5: Integrate with Existing Feedback

Find the `handle_feedback` method in the `FeedbackView` class (around line 110-160) and add this code at the end:

```python
async def handle_feedback(self, interaction: discord.Interaction, feedback_type: str, feedback_text: str):
    # ... existing feedback storage code ...

    # NEW: Update user embedding after storing feedback
    if personalized_trainer:
        try:
            await personalized_commands.update_user_embedding_from_feedback(
                user_id=interaction.user.id,
                recommendations=self.recommendations,
                feedback_type=feedback_type
            )
        except Exception as e:
            logger.warning(f"Failed to update embedding: {e}")

    # ... rest of existing code ...
```

### Step 6: Integrate with Individual Rating Feedback

Find the `IndividualContentFeedbackModal.on_submit` method (around line 200-300) and add:

```python
async def on_submit(self, interaction: discord.Interaction):
    # ... existing code to store ratings ...

    # NEW: Update embeddings for each rating
    if personalized_trainer:
        for i, movie_id in enumerate(self.movie_ids):
            rating_input = self.children[i]
            if rating_input.value:
                try:
                    rating = int(rating_input.value)
                    if 1 <= rating <= 5:
                        await personalized_trainer.update_user_embedding(
                            user_id=interaction.user.id,
                            item_id=movie_id,
                            feedback_type='positive' if rating >= 4 else 'negative' if rating <= 2 else 'neutral',
                            rating=rating
                        )
                except Exception as e:
                    logger.warning(f"Failed to update embedding for movie {movie_id}: {e}")

    # ... rest of existing code ...
```

## Testing

### 1. Test Database Migration

```bash
python migrate_personalization.py
```

Expected output:
```
🚀 CineSync v2 Personalization Migration
============================================================
📄 Using SQL file: .../database_extensions.sql
🔌 Connecting to database...
✅ Connected to PostgreSQL
✅ Migration completed successfully!
```

### 2. Test Personalization System

```bash
python test_personalization.py
```

### 3. Test Discord Commands

Start the bot and test in Discord:

```
/rate
→ Should show popular movies to rate

/my_recommendations
→ Should give personalized recommendations (after 5+ ratings)

/my_stats
→ Should show your preference profile
```

## Troubleshooting

### Issue: "No module named 'personalized_trainer'"

**Solution**: Make sure you're in the right directory:
```bash
cd services/lupe_python
python main.py
```

### Issue: Database connection error

**Solution**: Check your .env file has correct database credentials:
```
DB_HOST=localhost
DB_PORT=5432
DB_NAME=cinesync
DB_USER=postgres
DB_PASSWORD=your_password
```

### Issue: User embeddings not updating

**Solution**: Check if feedback is being stored:
```sql
SELECT * FROM user_ratings WHERE user_id = YOUR_USER_ID;
SELECT * FROM user_embeddings WHERE user_id = YOUR_USER_ID;
```

### Issue: Recommendations not personalized

**Possible causes**:
1. User has < 5 ratings (need minimum data)
2. User embedding not created yet
3. Cache not cleared

**Solution**:
- Rate at least 5 movies first
- Check logs for errors
- Restart the bot to clear caches

## Configuration Options

### Embedding Dimension

Default: 256. Can be changed in initialization:

```python
personalized_trainer = PersonalizedTrainer(
    db_manager=db_manager,
    content_manager=bot.lupe,
    embedding_dim=512  # Increase for more expressiveness
)
```

### Learning Rate

Default: 0.1 for neutral, 0.15 for strong feedback. Modify in `personalized_trainer.py`:

```python
# In update_user_embedding method
learning_rate = 0.1  # Adjust this value
```

### Personalization Weight

Default: 60% personalization, 40% base score. Modify in `personalized_trainer.py`:

```python
# In get_personalized_recommendations method
final_score = 0.6 * similarity + 0.4 * base_score  # Adjust weights
```

## Next Steps

1. **Test with real users** - Get feedback on recommendation quality
2. **Monitor performance** - Check response times and accuracy
3. **Tune parameters** - Adjust learning rates and weights based on results
4. **Add more features**:
   - Explainable recommendations ("You might like this because...")
   - Group recommendations (for watch parties)
   - Mood-based filtering
   - Time-of-day recommendations

## Support

For issues or questions:
1. Check logs in `lupe_bot.log`
2. Review this integration guide
3. Check the main DISCORD_AI_AGENT_GUIDE.md for detailed information
