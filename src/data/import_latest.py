#!/usr/bin/env python3
"""
CineSync v2 - Quick Import Module

Simple interface for importing the latest movies, TV shows, and anime data.

Usage:
    # From command line
    python -m src.data.import_latest

    # From Python code
    from src.data.import_latest import import_all, import_anime_only

    # Import everything
    results = import_all(tmdb_api_key="your_key")

    # Import just anime (no API key needed)
    results = import_anime_only()
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.data.data_import_pipeline import (
    DataImportConfig,
    DataImportPipeline,
    get_dynamic_start_date,
    setup_logging
)


def import_all(
    tmdb_api_key: str = None,
    from_year: int = 2017,
    quick: bool = False
) -> dict:
    """
    Import all content types: movies, TV shows, and anime.

    Args:
        tmdb_api_key: TMDB API key (or set TMDB_API_KEY env var)
        from_year: Start year for data import (default: 2017)
        quick: If True, use quick mode for faster import

    Returns:
        Dictionary with import results
    """
    setup_logging(log_to_file=True)

    api_key = tmdb_api_key or os.getenv('TMDB_API_KEY', '')

    config = DataImportConfig(
        tmdb_api_key=api_key,
        start_date=get_dynamic_start_date(from_year=from_year),
        from_year=from_year,
        fetch_by_genre=not quick,
        max_pages=50 if quick else 500,
    )

    pipeline = DataImportPipeline(config)
    return pipeline.run_import(
        import_movies=bool(api_key),
        import_tv=bool(api_key),
        import_anime=True
    )


def import_movies_only(tmdb_api_key: str = None, from_year: int = 2017) -> dict:
    """Import only movies from TMDB (2017 forward by default)."""
    setup_logging(log_to_file=True)

    api_key = tmdb_api_key or os.getenv('TMDB_API_KEY', '')
    if not api_key:
        raise ValueError("TMDB API key required for movie import")

    config = DataImportConfig(
        tmdb_api_key=api_key,
        start_date=get_dynamic_start_date(from_year=from_year),
        from_year=from_year,
    )

    pipeline = DataImportPipeline(config)
    return pipeline.run_import(import_movies=True, import_tv=False, import_anime=False)


def import_tv_only(tmdb_api_key: str = None, from_year: int = 2017) -> dict:
    """Import only TV shows from TMDB (2017 forward by default)."""
    setup_logging(log_to_file=True)

    api_key = tmdb_api_key or os.getenv('TMDB_API_KEY', '')
    if not api_key:
        raise ValueError("TMDB API key required for TV import")

    config = DataImportConfig(
        tmdb_api_key=api_key,
        start_date=get_dynamic_start_date(from_year=from_year),
        from_year=from_year,
    )

    pipeline = DataImportPipeline(config)
    return pipeline.run_import(import_movies=False, import_tv=True, import_anime=False)


def import_anime_only(from_year: int = 2017) -> dict:
    """
    Import only anime from MyAnimeList (via Jikan API).
    No API key required. Fetches 2017 forward by default.
    """
    setup_logging(log_to_file=True)

    config = DataImportConfig(
        start_date=get_dynamic_start_date(from_year=from_year),
        from_year=from_year,
    )

    pipeline = DataImportPipeline(config)
    return pipeline.run_import(import_movies=False, import_tv=False, import_anime=True)


def get_database_stats() -> dict:
    """Get current database statistics."""
    from src.data.data_import_pipeline import DatabaseManager

    config = DataImportConfig()
    db_manager = DatabaseManager(config)
    return db_manager.get_import_stats()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Quick data import")
    parser.add_argument('--anime-only', action='store_true', help='Import only anime')
    parser.add_argument('--quick', action='store_true', help='Quick mode')
    parser.add_argument('--stats', action='store_true', help='Show stats only')
    args = parser.parse_args()

    if args.stats:
        stats = get_database_stats()
        print("\nDatabase Statistics:")
        for content_type, data in stats.items():
            print(f"\n{content_type.upper()}:")
            for key, value in data.items():
                print(f"  {key}: {value}")
    elif args.anime_only:
        print("Importing anime only...")
        results = import_anime_only()
        print(f"Imported {results['anime_imported']} new anime")
    else:
        print("Importing all content...")
        results = import_all(quick=args.quick)
        print(f"Imported: {results['movies_imported']} movies, "
              f"{results['tv_shows_imported']} TV shows, "
              f"{results['anime_imported']} anime")
