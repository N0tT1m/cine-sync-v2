#!/usr/bin/env python3
"""
Simplified main script demonstrating reduced complexity
"""
import logging
import sys
import os

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.simple_content_manager import SimpleContentManager
from simple_config import load_simple_config


def setup_logging():
    """Simple logging setup"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger("simple_main")


def demo_recommendations():
    """Demonstrate simplified recommendation system"""
    logger = setup_logging()
    logger.info("Starting simplified recommendation demo")
    
    try:
        # Load configuration
        config = load_simple_config()
        
        # Initialize content manager
        manager = SimpleContentManager(config.model.models_dir)
        manager.load_all()
        
        # Get status
        status = manager.get_model_status()
        logger.info(f"System status: {status}")
        
        # Demo user ID
        demo_user_id = 1
        
        # Get mixed recommendations
        logger.info("Getting mixed recommendations...")
        mixed_recs = manager.get_recommendations(demo_user_id, "mixed", 5)
        
        print("\n=== Mixed Recommendations ===")
        for i, (content_id, title, content_type, score) in enumerate(mixed_recs, 1):
            print(f"{i}. {title} ({content_type}) - Score: {score:.3f}")
        
        # Get movie recommendations
        logger.info("Getting movie recommendations...")
        movie_recs = manager.get_recommendations(demo_user_id, "movie", 3)
        
        print("\n=== Movie Recommendations ===")
        for i, (content_id, title, content_type, score) in enumerate(movie_recs, 1):
            print(f"{i}. {title} - Score: {score:.3f}")
        
        # Get similar content (if we have data)
        if movie_recs:
            first_movie_id = movie_recs[0][0]
            logger.info(f"Getting movies similar to ID {first_movie_id}...")
            similar = manager.get_similar_content(first_movie_id, "movie", 3)
            
            print(f"\n=== Movies Similar to '{movie_recs[0][1]}' ===")
            for i, (content_id, title, content_type, score) in enumerate(similar, 1):
                print(f"{i}. {title} - Similarity: {score:.3f}")
        
        # Show available genres
        genres = manager.get_available_genres("movie")
        if genres:
            print(f"\n=== Available Movie Genres ({len(genres)}) ===")
            print(", ".join(genres[:10]) + ("..." if len(genres) > 10 else ""))
        
        logger.info("Demo completed successfully")
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(demo_recommendations())