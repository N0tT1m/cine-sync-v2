#!/usr/bin/env python3
"""
Test script for personalization system
Tests all components of the personalization pipeline
"""

import sys
import os
import asyncio
import psycopg2
from pathlib import Path
import logging

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
load_dotenv()

from personalized_trainer import PersonalizedTrainer
from preference_learner import PreferenceLearner
from config import load_config

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MockDatabaseManager:
    """Mock database manager for testing"""
    def __init__(self, db_config):
        self.config = db_config

    def get_connection(self):
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


class MockContentManager:
    """Mock content manager for testing"""
    def __init__(self):
        self.movie_lookup = {
            1: {'title': 'The Matrix', 'genres': 'Action|Sci-Fi', 'year': 1999},
            2: {'title': 'Inception', 'genres': 'Action|Thriller', 'year': 2010},
            3: {'title': 'The Shawshank Redemption', 'genres': 'Drama', 'year': 1994},
            4: {'title': 'Pulp Fiction', 'genres': 'Crime|Drama', 'year': 1994},
            5: {'title': 'The Dark Knight', 'genres': 'Action|Crime', 'year': 2008}
        }


async def test_database_connection(db_manager):
    """Test database connection"""
    print("\n🧪 Test 1: Database Connection")
    print("=" * 60)

    try:
        with db_manager.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT version();")
            version = cursor.fetchone()[0]
            print(f"✅ Connected to: {version.split(',')[0]}")

            # Check if personalization tables exist
            cursor.execute("""
                SELECT table_name
                FROM information_schema.tables
                WHERE table_schema = 'public'
                AND table_name IN ('user_embeddings', 'user_preferences', 'user_model_weights')
            """)

            tables = cursor.fetchall()
            print(f"✅ Found {len(tables)} personalization tables:")
            for table in tables:
                print(f"   - {table[0]}")

            if len(tables) < 3:
                print("\n⚠️  Warning: Some tables missing. Run migration first:")
                print("   python migrate_personalization.py")
                return False

        return True

    except Exception as e:
        print(f"❌ Database connection failed: {e}")
        return False


async def test_personalized_trainer(personalized_trainer, test_user_id):
    """Test PersonalizedTrainer functionality"""
    print("\n🧪 Test 2: PersonalizedTrainer")
    print("=" * 60)

    try:
        # Test 1: Create user embedding
        print("\n2.1 Creating user embedding...")
        embedding = await personalized_trainer.create_user_embedding(test_user_id)
        print(f"✅ Created embedding with shape: {embedding.shape}")
        print(f"   Embedding norm: {np.linalg.norm(embedding):.4f}")

        # Test 2: Simulate positive feedback
        print("\n2.2 Simulating positive feedback...")
        await personalized_trainer.update_user_embedding(
            test_user_id,
            item_id=1,  # The Matrix
            feedback_type='love',
            rating=5
        )
        print("✅ Updated embedding with positive feedback")

        # Test 3: Simulate negative feedback
        print("\n2.3 Simulating negative feedback...")
        await personalized_trainer.update_user_embedding(
            test_user_id,
            item_id=3,  # The Shawshank Redemption
            feedback_type='dislike',
            rating=2
        )
        print("✅ Updated embedding with negative feedback")

        # Test 4: Get personalized recommendations
        print("\n2.4 Getting personalized recommendations...")
        base_recs = [
            (2, 'Inception', 'movie', 0.8),
            (4, 'Pulp Fiction', 'movie', 0.75),
            (5, 'The Dark Knight', 'movie', 0.7)
        ]

        personalized_recs = await personalized_trainer.get_personalized_recommendations(
            test_user_id,
            top_k=3,
            base_recommendations=base_recs
        )

        print(f"✅ Got {len(personalized_recs)} personalized recommendations:")
        for i, rec in enumerate(personalized_recs, 1):
            print(f"   {i}. {rec['title']}: {rec['score']:.3f} "
                  f"(base: {rec['base_score']:.3f}, personal: {rec['personalization_score']:.3f})")

        # Test 5: Cache test
        print("\n2.5 Testing cache...")
        cached_embedding = await personalized_trainer.get_or_create_user_embedding(test_user_id)
        print(f"✅ Retrieved from cache: {cached_embedding.shape}")

        return True

    except Exception as e:
        print(f"❌ PersonalizedTrainer test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_preference_learner(preference_learner, test_user_id, db_manager):
    """Test PreferenceLearner functionality"""
    print("\n🧪 Test 3: PreferenceLearner")
    print("=" * 60)

    try:
        # Add some test ratings
        print("\n3.1 Adding test ratings...")
        with db_manager.get_connection() as conn:
            cursor = conn.cursor()

            test_ratings = [
                (test_user_id, 1, 5),  # The Matrix - 5 stars
                (test_user_id, 2, 4),  # Inception - 4 stars
                (test_user_id, 3, 2),  # Shawshank - 2 stars
                (test_user_id, 4, 4),  # Pulp Fiction - 4 stars
                (test_user_id, 5, 5),  # Dark Knight - 5 stars
            ]

            for user_id, movie_id, rating in test_ratings:
                cursor.execute("""
                    INSERT INTO user_ratings (user_id, movie_id, rating, timestamp)
                    VALUES (%s, %s, %s, NOW())
                    ON CONFLICT (user_id, movie_id)
                    DO UPDATE SET rating = EXCLUDED.rating
                """, (user_id, movie_id, rating))

            conn.commit()
            print(f"✅ Added {len(test_ratings)} test ratings")

        # Test pattern analysis
        print("\n3.2 Analyzing user patterns...")
        patterns = await preference_learner.analyze_user_patterns(test_user_id)

        print(f"✅ Analysis complete:")
        print(f"   Total ratings: {patterns['total_ratings']}")
        print(f"   Average rating: {patterns['avg_rating']:.2f}")
        print(f"   Diversity score: {patterns['diversity_score']:.2%}")

        if patterns['favorite_genres']:
            print(f"\n   Top genres:")
            for genre in patterns['favorite_genres'][:3]:
                print(f"      - {genre['genre']}: ⭐{genre['avg_rating']:.1f} ({genre['count']} rated)")

        if patterns['preferred_decades']:
            print(f"\n   Preferred decades:")
            for decade in patterns['preferred_decades'][:2]:
                print(f"      - {decade['decade']}: ⭐{decade['avg_rating']:.1f}")

        # Test cache
        print("\n3.3 Testing preference cache...")
        cached = preference_learner.get_cached_preferences(test_user_id)
        if cached:
            print(f"✅ Retrieved from cache: {cached['total_ratings']} ratings")
        else:
            print("⚠️  Cache miss (unexpected)")

        return True

    except Exception as e:
        print(f"❌ PreferenceLearner test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_integration(personalized_trainer, preference_learner, test_user_id):
    """Test integration between components"""
    print("\n🧪 Test 4: Integration Test")
    print("=" * 60)

    try:
        # Test the full workflow
        print("\n4.1 Full personalization workflow...")

        # Get patterns
        patterns = await preference_learner.analyze_user_patterns(test_user_id)

        # Get embedding
        embedding = await personalized_trainer.get_or_create_user_embedding(test_user_id)

        # Get recommendations
        base_recs = [(i, f'Movie {i}', 'movie', 0.5) for i in range(10, 20)]
        personalized_recs = await personalized_trainer.get_personalized_recommendations(
            test_user_id,
            top_k=5,
            base_recommendations=base_recs
        )

        print(f"✅ Full workflow completed:")
        print(f"   Patterns analyzed: {patterns['total_ratings']} ratings")
        print(f"   Embedding created: {embedding.shape}")
        print(f"   Recommendations generated: {len(personalized_recs)}")

        # Test cache clearing
        print("\n4.2 Testing cache management...")
        personalized_trainer.clear_cache(test_user_id)
        preference_learner.clear_cache(test_user_id)
        print("✅ Caches cleared successfully")

        return True

    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def cleanup_test_data(db_manager, test_user_id):
    """Clean up test data"""
    print("\n🧹 Cleaning up test data...")

    try:
        with db_manager.get_connection() as conn:
            cursor = conn.cursor()

            # Remove test ratings
            cursor.execute("DELETE FROM user_ratings WHERE user_id = %s", (test_user_id,))

            # Remove test embeddings
            cursor.execute("DELETE FROM user_embeddings WHERE user_id = %s", (test_user_id,))

            # Remove test preferences
            cursor.execute("DELETE FROM user_preferences WHERE user_id = %s", (test_user_id,))

            conn.commit()
            print("✅ Test data cleaned up")

    except Exception as e:
        print(f"⚠️  Cleanup warning: {e}")


async def main():
    """Main test runner"""
    print("=" * 60)
    print("🧪 CineSync v2 Personalization System Test Suite")
    print("=" * 60)

    # Load configuration
    config = load_config()

    # Initialize database manager
    db_manager = MockDatabaseManager(config.database)

    # Test database connection first
    if not await test_database_connection(db_manager):
        print("\n❌ Database tests failed. Please check your connection and run migration.")
        print("   Run: python migrate_personalization.py")
        return False

    # Initialize content manager
    content_manager = MockContentManager()

    # Initialize personalization components
    personalized_trainer = PersonalizedTrainer(
        db_manager=db_manager,
        content_manager=content_manager,
        embedding_dim=256
    )

    preference_learner = PreferenceLearner(
        db_manager=db_manager,
        content_manager=content_manager
    )

    # Use a test user ID that won't conflict with real users
    test_user_id = 999999999

    # Run tests
    results = []

    results.append(await test_personalized_trainer(personalized_trainer, test_user_id))
    results.append(await test_preference_learner(preference_learner, test_user_id, db_manager))
    results.append(await test_integration(personalized_trainer, preference_learner, test_user_id))

    # Cleanup
    await cleanup_test_data(db_manager, test_user_id)

    # Summary
    print("\n" + "=" * 60)
    print("📊 Test Summary")
    print("=" * 60)

    passed = sum(results)
    total = len(results)

    print(f"\nTests passed: {passed}/{total}")

    if passed == total:
        print("✅ All tests passed!")
        print("\n🎉 Personalization system is ready to use!")
        print("\nNext steps:")
        print("1. Integrate commands into main.py (see INTEGRATION_GUIDE.md)")
        print("2. Restart the Discord bot")
        print("3. Test with real users using /rate, /my_recommendations, /my_stats")
    else:
        print("❌ Some tests failed. Please review the errors above.")

    return passed == total


if __name__ == '__main__':
    try:
        import numpy as np  # Import here to avoid issues in async code
        success = asyncio.run(main())
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n❌ Tests cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Test suite failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
