#!/usr/bin/env python3
"""
CineSync v2 Data Import Pipeline
Comprehensive data import for movies, TV shows, and anime with automatic updates.

Supports:
- TMDB API for movies and TV shows
- Jikan API (MyAnimeList) for anime
- Dynamic date handling for always-fresh data
- Automatic model retraining triggers
"""

import argparse
import logging
import os
import sys
import pandas as pd
import numpy as np
import requests
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import sqlite3
import pickle
from dataclasses import dataclass, asdict, field
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    # Look for .env in project root
    project_root = Path(__file__).parent.parent.parent
    env_path = project_root / '.env'
    if env_path.exists():
        load_dotenv(env_path)
except ImportError:
    pass  # python-dotenv not installed, use system env vars

# Import wandb for experiment tracking
try:
    from wandb_config import init_wandb_for_training, WandbManager
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

    def init_wandb_for_training(*args, **kwargs):
        return None

logger = logging.getLogger(__name__)


def get_dynamic_start_date(lookback_days: int = None, from_year: int = 2017) -> str:
    """
    Calculate start date for data import.

    By default, starts from 2017-01-01 to ensure comprehensive coverage.
    Can also specify lookback_days for relative date calculation.
    """
    if lookback_days is not None:
        return (datetime.now() - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
    return f"{from_year}-01-01"


@dataclass
class DataImportConfig:
    """Configuration for data import pipeline"""
    # TMDB API Configuration
    tmdb_api_key: str = field(default_factory=lambda: os.getenv('TMDB_API_KEY', ''))
    tmdb_base_url: str = "https://api.themoviedb.org/3"

    # Jikan API (MyAnimeList) Configuration - No API key required
    jikan_base_url: str = "https://api.jikan.moe/v4"
    jikan_rate_limit: int = 3  # Jikan allows 3 requests per second

    # Import settings - Default from 2017 for comprehensive coverage
    start_date: str = field(default_factory=lambda: get_dynamic_start_date(from_year=2017))
    end_date: str = field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d"))
    from_year: int = 2017  # Default start year for comprehensive data
    batch_size: int = 100
    max_requests_per_minute: int = 40  # TMDB rate limit
    max_pages: int = 500  # Increased for comprehensive coverage from 2017

    # Database paths
    db_path: str = "data/cinesync.db"  # Unified database
    movies_db_path: str = "data/movies.db"  # Legacy support
    tv_db_path: str = "data/tv_shows.db"  # Legacy support
    anime_db_path: str = "data/anime.db"

    # Output paths
    output_dir: str = "data/imports"
    movies_output: str = "movies_latest.csv"
    tv_output: str = "tv_shows_latest.csv"
    anime_output: str = "anime_latest.csv"

    # Processing settings - Strict quality filters for clean data
    min_vote_count: int = 50  # Require meaningful vote count
    min_popularity: float = 2.0  # Filter out obscure content
    min_vote_average: float = 1.0  # Must have some rating
    require_overview: bool = True  # Must have description
    require_poster: bool = True  # Must have poster image
    include_adult: bool = False

    # Anime-specific settings - Quality filters
    anime_min_score: float = 5.0  # Minimum MAL score (out of 10)
    anime_min_members: int = 1000  # Require community engagement
    anime_min_scored_by: int = 100  # Minimum number of ratings
    include_anime_types: List[str] = field(default_factory=lambda: ['tv', 'movie', 'ova', 'ona', 'special'])

    # Content type toggles
    import_movies: bool = True
    import_tv: bool = True
    import_anime: bool = True

    # Comprehensive fetch settings
    fetch_trending: bool = True
    fetch_popular: bool = True
    fetch_top_rated: bool = True
    fetch_now_playing: bool = True  # Currently in theaters
    fetch_upcoming: bool = True
    fetch_airing_today: bool = True  # TV airing today
    fetch_on_the_air: bool = True  # TV currently airing

    # Genre-specific fetching for comprehensive coverage
    fetch_by_genre: bool = True

    # Retraining settings
    auto_retrain: bool = True
    retrain_threshold: int = 500  # Lowered threshold

    # Parallel processing
    use_parallel: bool = True
    max_workers: int = 4


class TMDBDataImporter:
    """TMDB API data importer for movies and TV shows - Comprehensive fetching"""

    # TMDB Genre IDs for comprehensive coverage
    MOVIE_GENRES = {
        28: 'Action', 12: 'Adventure', 16: 'Animation', 35: 'Comedy',
        80: 'Crime', 99: 'Documentary', 18: 'Drama', 10751: 'Family',
        14: 'Fantasy', 36: 'History', 27: 'Horror', 10402: 'Music',
        9648: 'Mystery', 10749: 'Romance', 878: 'Science Fiction',
        10770: 'TV Movie', 53: 'Thriller', 10752: 'War', 37: 'Western'
    }

    TV_GENRES = {
        10759: 'Action & Adventure', 16: 'Animation', 35: 'Comedy',
        80: 'Crime', 99: 'Documentary', 18: 'Drama', 10751: 'Family',
        10762: 'Kids', 9648: 'Mystery', 10763: 'News', 10764: 'Reality',
        10765: 'Sci-Fi & Fantasy', 10766: 'Soap', 10767: 'Talk',
        10768: 'War & Politics', 37: 'Western'
    }

    def __init__(self, config: DataImportConfig):
        self.config = config
        self.session = requests.Session()
        self.session.headers.update({'Accept': 'application/json'})
        self.request_count = 0
        self.last_request_time = time.time()
        self._seen_movie_ids = set()
        self._seen_tv_ids = set()

        if not config.tmdb_api_key:
            raise ValueError("TMDB API key is required. Set TMDB_API_KEY environment variable.")

    def _rate_limit(self):
        """Enforce rate limiting for TMDB API"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time

        if self.request_count >= self.config.max_requests_per_minute:
            if time_since_last < 60:
                sleep_time = 60 - time_since_last
                logger.info(f"Rate limiting: sleeping for {sleep_time:.2f} seconds")
                time.sleep(sleep_time)
            self.request_count = 0

        self.request_count += 1
        self.last_request_time = time.time()

    def _make_request(self, endpoint: str, params: Dict = None, retries: int = 3) -> Optional[Dict]:
        """Make rate-limited request to TMDB API with retry logic"""
        self._rate_limit()

        url = f"{self.config.tmdb_base_url}/{endpoint}"
        params = params or {}
        params['api_key'] = self.config.tmdb_api_key

        for attempt in range(retries):
            try:
                response = self.session.get(url, params=params, timeout=30)
                response.raise_for_status()
                return response.json()
            except requests.exceptions.RequestException as e:
                if attempt < retries - 1:
                    wait_time = 2 ** attempt
                    logger.warning(f"Request failed (attempt {attempt + 1}), retrying in {wait_time}s: {e}")
                    time.sleep(wait_time)
                else:
                    logger.error(f"API request failed after {retries} attempts: {e}")
                    return None

    def _passes_quality_filter(self, item: Dict, content_type: str) -> bool:
        """Check if item meets quality standards"""
        # Must have required vote count
        if item.get('vote_count', 0) < self.config.min_vote_count:
            return False

        # Must have minimum popularity
        if item.get('popularity', 0) < self.config.min_popularity:
            return False

        # Must have some rating if votes exist
        if item.get('vote_count', 0) > 0 and item.get('vote_average', 0) < self.config.min_vote_average:
            return False

        # Require overview/description
        if self.config.require_overview and not item.get('overview', '').strip():
            return False

        # Require poster
        if self.config.require_poster and not item.get('poster_path'):
            return False

        # Must have a title
        title_field = 'title' if content_type == 'movie' else 'name'
        if not item.get(title_field, '').strip():
            return False

        # Must have a release/air date
        date_field = 'release_date' if content_type == 'movie' else 'first_air_date'
        if not item.get(date_field):
            return False

        return True

    def _fetch_paginated(self, endpoint: str, params: Dict, content_type: str,
                         max_pages: int = None) -> List[Dict]:
        """Generic paginated fetch with deduplication and quality filtering"""
        max_pages = max_pages or self.config.max_pages
        results = []
        page = 1
        seen_ids = self._seen_movie_ids if content_type == 'movie' else self._seen_tv_ids
        filtered_count = 0

        while page <= max_pages:
            params['page'] = page
            data = self._make_request(endpoint, params.copy())

            if not data or 'results' not in data:
                break

            page_results = data['results']
            if not page_results:
                break

            # Deduplicate and quality filter
            for item in page_results:
                item_id = item.get('id')
                if item_id and item_id not in seen_ids:
                    seen_ids.add(item_id)
                    if self._passes_quality_filter(item, content_type):
                        results.append(item)
                    else:
                        filtered_count += 1

            if page >= data.get('total_pages', 1):
                break

            page += 1

        if filtered_count > 0:
            logger.debug(f"Filtered out {filtered_count} low-quality items")

        return results

    def get_all_movies_comprehensive(self) -> List[Dict]:
        """Fetch movies from all available sources for comprehensive coverage"""
        all_movies = []
        start_date = self.config.start_date
        end_date = self.config.end_date

        logger.info(f"Starting comprehensive movie fetch from {start_date} to {end_date}")

        # 1. Trending movies (daily and weekly)
        if self.config.fetch_trending:
            logger.info("Fetching trending movies...")
            for time_window in ['day', 'week']:
                data = self._make_request(f'trending/movie/{time_window}')
                if data and 'results' in data:
                    for movie in data['results']:
                        if movie['id'] not in self._seen_movie_ids:
                            self._seen_movie_ids.add(movie['id'])
                            all_movies.append(movie)
            logger.info(f"Trending: {len(all_movies)} movies")

        # 2. Now playing (in theaters)
        if self.config.fetch_now_playing:
            logger.info("Fetching now playing movies...")
            movies = self._fetch_paginated('movie/now_playing', {}, 'movie', max_pages=10)
            all_movies.extend(movies)
            logger.info(f"Now playing: +{len(movies)} movies")

        # 3. Upcoming movies
        if self.config.fetch_upcoming:
            logger.info("Fetching upcoming movies...")
            movies = self._fetch_paginated('movie/upcoming', {}, 'movie', max_pages=10)
            all_movies.extend(movies)
            logger.info(f"Upcoming: +{len(movies)} movies")

        # 4. Popular movies with date filter
        if self.config.fetch_popular:
            logger.info("Fetching popular movies...")
            params = {
                'primary_release_date.gte': start_date,
                'primary_release_date.lte': end_date,
                'vote_count.gte': self.config.min_vote_count,
                'sort_by': 'popularity.desc'
            }
            movies = self._fetch_paginated('discover/movie', params, 'movie')
            all_movies.extend(movies)
            logger.info(f"Popular: +{len(movies)} movies")

        # 5. Top rated movies
        if self.config.fetch_top_rated:
            logger.info("Fetching top rated movies...")
            params = {
                'primary_release_date.gte': start_date,
                'primary_release_date.lte': end_date,
                'vote_count.gte': self.config.min_vote_count * 2,
                'sort_by': 'vote_average.desc'
            }
            movies = self._fetch_paginated('discover/movie', params, 'movie', max_pages=20)
            all_movies.extend(movies)
            logger.info(f"Top rated: +{len(movies)} movies")

        # 6. By genre for comprehensive coverage
        if self.config.fetch_by_genre:
            logger.info("Fetching movies by genre...")
            for genre_id, genre_name in self.MOVIE_GENRES.items():
                params = {
                    'with_genres': genre_id,
                    'primary_release_date.gte': start_date,
                    'primary_release_date.lte': end_date,
                    'vote_count.gte': self.config.min_vote_count,
                    'sort_by': 'popularity.desc'
                }
                movies = self._fetch_paginated('discover/movie', params, 'movie', max_pages=10)
                logger.info(f"  {genre_name}: +{len(movies)} movies")
                all_movies.extend(movies)

        logger.info(f"Total unique movies fetched: {len(all_movies)}")
        return all_movies

    def get_all_tv_shows_comprehensive(self) -> List[Dict]:
        """Fetch TV shows from all available sources for comprehensive coverage"""
        all_shows = []
        start_date = self.config.start_date
        end_date = self.config.end_date

        logger.info(f"Starting comprehensive TV show fetch from {start_date} to {end_date}")

        # 1. Trending TV shows
        if self.config.fetch_trending:
            logger.info("Fetching trending TV shows...")
            for time_window in ['day', 'week']:
                data = self._make_request(f'trending/tv/{time_window}')
                if data and 'results' in data:
                    for show in data['results']:
                        if show['id'] not in self._seen_tv_ids:
                            self._seen_tv_ids.add(show['id'])
                            all_shows.append(show)
            logger.info(f"Trending: {len(all_shows)} TV shows")

        # 2. Airing today
        if self.config.fetch_airing_today:
            logger.info("Fetching TV shows airing today...")
            shows = self._fetch_paginated('tv/airing_today', {}, 'tv', max_pages=10)
            all_shows.extend(shows)
            logger.info(f"Airing today: +{len(shows)} TV shows")

        # 3. Currently on the air
        if self.config.fetch_on_the_air:
            logger.info("Fetching TV shows on the air...")
            shows = self._fetch_paginated('tv/on_the_air', {}, 'tv', max_pages=10)
            all_shows.extend(shows)
            logger.info(f"On the air: +{len(shows)} TV shows")

        # 4. Popular TV shows
        if self.config.fetch_popular:
            logger.info("Fetching popular TV shows...")
            params = {
                'first_air_date.gte': start_date,
                'first_air_date.lte': end_date,
                'vote_count.gte': self.config.min_vote_count,
                'sort_by': 'popularity.desc'
            }
            shows = self._fetch_paginated('discover/tv', params, 'tv')
            all_shows.extend(shows)
            logger.info(f"Popular: +{len(shows)} TV shows")

        # 5. Top rated TV shows
        if self.config.fetch_top_rated:
            logger.info("Fetching top rated TV shows...")
            params = {
                'first_air_date.gte': start_date,
                'first_air_date.lte': end_date,
                'vote_count.gte': self.config.min_vote_count * 2,
                'sort_by': 'vote_average.desc'
            }
            shows = self._fetch_paginated('discover/tv', params, 'tv', max_pages=20)
            all_shows.extend(shows)
            logger.info(f"Top rated: +{len(shows)} TV shows")

        # 6. By genre
        if self.config.fetch_by_genre:
            logger.info("Fetching TV shows by genre...")
            for genre_id, genre_name in self.TV_GENRES.items():
                params = {
                    'with_genres': genre_id,
                    'first_air_date.gte': start_date,
                    'first_air_date.lte': end_date,
                    'vote_count.gte': self.config.min_vote_count,
                    'sort_by': 'popularity.desc'
                }
                shows = self._fetch_paginated('discover/tv', params, 'tv', max_pages=10)
                logger.info(f"  {genre_name}: +{len(shows)} TV shows")
                all_shows.extend(shows)

        logger.info(f"Total unique TV shows fetched: {len(all_shows)}")
        return all_shows

    def get_movie_details(self, movie_id: int) -> Optional[Dict]:
        """Get detailed movie information with credits, keywords, and streaming providers"""
        return self._make_request(f'movie/{movie_id}', {
            'append_to_response': 'credits,keywords,videos,watch/providers'
        })

    def get_tv_details(self, tv_id: int) -> Optional[Dict]:
        """Get detailed TV show information with streaming providers"""
        return self._make_request(f'tv/{tv_id}', {
            'append_to_response': 'credits,keywords,videos,watch/providers'
        })

    def get_watch_providers(self, content_type: str, content_id: int, region: str = 'US') -> Optional[Dict]:
        """
        Get streaming availability for a movie or TV show.

        Returns providers for: flatrate (subscription), rent, buy
        Common providers: Netflix, Amazon Prime, Disney+, Hulu, HBO Max, Apple TV+, etc.
        """
        data = self._make_request(f'{content_type}/{content_id}/watch/providers')
        if data and 'results' in data:
            return data['results'].get(region, {})
        return None

    def get_available_watch_providers(self, region: str = 'US') -> List[Dict]:
        """Get list of all available streaming providers in a region"""
        movie_providers = self._make_request('watch/providers/movie', {'watch_region': region})
        tv_providers = self._make_request('watch/providers/tv', {'watch_region': region})

        all_providers = {}
        for data in [movie_providers, tv_providers]:
            if data and 'results' in data:
                for provider in data['results']:
                    pid = provider['provider_id']
                    if pid not in all_providers:
                        all_providers[pid] = provider

        return list(all_providers.values())

    def get_movie_details_batch(self, movie_ids: List[int], max_workers: int = 4) -> List[Dict]:
        """Fetch movie details in parallel"""
        results = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(self.get_movie_details, mid): mid for mid in movie_ids}
            for future in as_completed(futures):
                result = future.result()
                if result:
                    results.append(result)
        return results

    # Legacy compatibility methods
    def get_popular_movies(self, start_date: str, end_date: str = None) -> List[Dict]:
        """Legacy method - use get_all_movies_comprehensive for better coverage"""
        self.config.start_date = start_date
        self.config.end_date = end_date or datetime.now().strftime("%Y-%m-%d")
        return self.get_all_movies_comprehensive()

    def get_popular_tv_shows(self, start_date: str, end_date: str = None) -> List[Dict]:
        """Legacy method - use get_all_tv_shows_comprehensive for better coverage"""
        self.config.start_date = start_date
        self.config.end_date = end_date or datetime.now().strftime("%Y-%m-%d")
        return self.get_all_tv_shows_comprehensive()


class JikanAnimeImporter:
    """Jikan API (MyAnimeList) importer for anime content - No API key required"""

    ANIME_GENRES = {
        1: 'Action', 2: 'Adventure', 4: 'Comedy', 8: 'Drama',
        10: 'Fantasy', 14: 'Horror', 7: 'Mystery', 22: 'Romance',
        24: 'Sci-Fi', 36: 'Slice of Life', 30: 'Sports', 37: 'Supernatural',
        41: 'Suspense', 17: 'Martial Arts', 18: 'Mecha', 38: 'Military',
        19: 'Music', 6: 'Demons', 40: 'Psychological', 23: 'School'
    }

    def __init__(self, config: DataImportConfig):
        self.config = config
        self.session = requests.Session()
        self.session.headers.update({'Accept': 'application/json'})
        self.last_request_time = 0
        self._seen_anime_ids = set()

    def _rate_limit(self):
        """Jikan allows ~3 requests per second"""
        elapsed = time.time() - self.last_request_time
        if elapsed < 0.4:  # ~2.5 requests per second to be safe
            time.sleep(0.4 - elapsed)
        self.last_request_time = time.time()

    def _make_request(self, endpoint: str, params: Dict = None, retries: int = 3) -> Optional[Dict]:
        """Make rate-limited request to Jikan API"""
        self._rate_limit()

        url = f"{self.config.jikan_base_url}/{endpoint}"
        params = params or {}

        for attempt in range(retries):
            try:
                response = self.session.get(url, params=params, timeout=30)

                # Handle Jikan rate limiting (429)
                if response.status_code == 429:
                    wait_time = int(response.headers.get('Retry-After', 2))
                    logger.warning(f"Jikan rate limited, waiting {wait_time}s...")
                    time.sleep(wait_time)
                    continue

                response.raise_for_status()
                return response.json()
            except requests.exceptions.RequestException as e:
                if attempt < retries - 1:
                    wait_time = 2 ** attempt
                    logger.warning(f"Jikan request failed (attempt {attempt + 1}), retrying in {wait_time}s: {e}")
                    time.sleep(wait_time)
                else:
                    logger.error(f"Jikan API request failed after {retries} attempts: {e}")
                    return None

    def _passes_quality_filter(self, item: Dict) -> bool:
        """Check if anime meets quality standards"""
        # Must have minimum score
        score = item.get('score') or 0
        if score > 0 and score < self.config.anime_min_score:
            return False

        # Must have community engagement
        members = item.get('members') or 0
        if members < self.config.anime_min_members:
            return False

        # Must have enough ratings
        scored_by = item.get('scored_by') or 0
        if scored_by < self.config.anime_min_scored_by:
            return False

        # Must have a title
        if not item.get('title', '').strip():
            return False

        # Must have synopsis (for quality content)
        if not item.get('synopsis', '').strip():
            return False

        # Must have images
        images = item.get('images', {})
        if not images.get('jpg', {}).get('image_url'):
            return False

        # Filter by type
        anime_type = (item.get('type') or '').lower()
        if anime_type and anime_type not in self.config.include_anime_types:
            return False

        return True

    def _fetch_paginated(self, endpoint: str, params: Dict, max_pages: int = 25) -> List[Dict]:
        """Fetch paginated results from Jikan with quality filtering"""
        results = []
        page = 1
        filtered_count = 0

        while page <= max_pages:
            params['page'] = page
            data = self._make_request(endpoint, params.copy())

            if not data or 'data' not in data:
                break

            page_results = data['data']
            if not page_results:
                break

            for item in page_results:
                mal_id = item.get('mal_id')
                if mal_id and mal_id not in self._seen_anime_ids:
                    self._seen_anime_ids.add(mal_id)
                    if self._passes_quality_filter(item):
                        results.append(item)
                    else:
                        filtered_count += 1

            # Check pagination info
            pagination = data.get('pagination', {})
            if not pagination.get('has_next_page', False):
                break

            page += 1

        if filtered_count > 0:
            logger.debug(f"Filtered out {filtered_count} low-quality anime")

        return results

    def get_all_anime_comprehensive(self) -> List[Dict]:
        """Fetch anime from multiple sources for comprehensive coverage"""
        all_anime = []
        current_year = datetime.now().year
        start_year = current_year - 2  # Last 2 years

        logger.info(f"Starting comprehensive anime fetch from {start_year} to {current_year}")

        # 1. Currently airing anime
        logger.info("Fetching currently airing anime...")
        params = {
            'status': 'airing',
            'order_by': 'popularity',
            'sort': 'asc',
            'sfw': not self.config.include_adult
        }
        anime = self._fetch_paginated('anime', params)
        all_anime.extend(anime)
        logger.info(f"Currently airing: {len(anime)} anime")

        # 2. Upcoming anime
        logger.info("Fetching upcoming anime...")
        params = {
            'status': 'upcoming',
            'order_by': 'popularity',
            'sort': 'asc',
            'sfw': not self.config.include_adult
        }
        anime = self._fetch_paginated('anime', params, max_pages=10)
        all_anime.extend(anime)
        logger.info(f"Upcoming: +{len(anime)} anime")

        # 3. Top anime (all time, but useful for coverage)
        logger.info("Fetching top rated anime...")
        params = {
            'order_by': 'score',
            'sort': 'desc',
            'min_score': 7,
            'sfw': not self.config.include_adult
        }
        anime = self._fetch_paginated('anime', params, max_pages=20)
        all_anime.extend(anime)
        logger.info(f"Top rated: +{len(anime)} anime")

        # 4. Popular anime
        logger.info("Fetching popular anime...")
        params = {
            'order_by': 'popularity',
            'sort': 'asc',
            'sfw': not self.config.include_adult
        }
        anime = self._fetch_paginated('anime', params, max_pages=20)
        all_anime.extend(anime)
        logger.info(f"Popular: +{len(anime)} anime")

        # 5. Recent seasons
        logger.info("Fetching seasonal anime...")
        seasons = ['winter', 'spring', 'summer', 'fall']
        for year in range(start_year, current_year + 1):
            for season in seasons:
                # Skip future seasons
                if year == current_year:
                    current_month = datetime.now().month
                    current_season_idx = (current_month - 1) // 3
                    if seasons.index(season) > current_season_idx:
                        continue

                data = self._make_request(f'seasons/{year}/{season}')
                if data and 'data' in data:
                    for item in data['data']:
                        mal_id = item.get('mal_id')
                        if mal_id and mal_id not in self._seen_anime_ids:
                            self._seen_anime_ids.add(mal_id)
                            all_anime.append(item)
                    logger.info(f"  {season.capitalize()} {year}: +{len(data['data'])} anime")

        # 6. By genre for coverage
        logger.info("Fetching anime by genre...")
        for genre_id, genre_name in list(self.ANIME_GENRES.items())[:10]:  # Top 10 genres
            params = {
                'genres': str(genre_id),
                'order_by': 'popularity',
                'sort': 'asc',
                'sfw': not self.config.include_adult
            }
            anime = self._fetch_paginated('anime', params, max_pages=5)
            logger.info(f"  {genre_name}: +{len(anime)} anime")
            all_anime.extend(anime)

        # 7. By type (TV, Movie, OVA, etc.)
        logger.info("Fetching anime by type...")
        for anime_type in self.config.include_anime_types:
            params = {
                'type': anime_type,
                'order_by': 'start_date',
                'sort': 'desc',
                'sfw': not self.config.include_adult
            }
            anime = self._fetch_paginated('anime', params, max_pages=10)
            logger.info(f"  {anime_type.upper()}: +{len(anime)} anime")
            all_anime.extend(anime)

        logger.info(f"Total unique anime fetched: {len(all_anime)}")
        return all_anime

    def get_anime_details(self, mal_id: int) -> Optional[Dict]:
        """Get detailed anime information"""
        data = self._make_request(f'anime/{mal_id}/full')
        return data.get('data') if data else None

    def get_current_season(self) -> List[Dict]:
        """Get current season's anime"""
        data = self._make_request('seasons/now')
        return data.get('data', []) if data else []

    def get_upcoming_anime(self) -> List[Dict]:
        """Get upcoming anime"""
        data = self._make_request('seasons/upcoming')
        return data.get('data', []) if data else []


class DataProcessor:
    """Process and normalize movie/TV data for ML models"""

    # Major streaming providers to track
    STREAMING_PROVIDERS = {
        8: 'Netflix',
        9: 'Amazon Prime Video',
        337: 'Disney Plus',
        15: 'Hulu',
        384: 'HBO Max',
        350: 'Apple TV Plus',
        386: 'Peacock',
        531: 'Paramount Plus',
        283: 'Crunchyroll',
        2: 'Apple iTunes',
        3: 'Google Play Movies',
        192: 'YouTube',
        10: 'Amazon Video',
    }

    def __init__(self, config: DataImportConfig):
        self.config = config

    def _extract_streaming_providers(self, item: Dict, region: str = 'US') -> Dict[str, str]:
        """Extract streaming provider information from watch/providers data"""
        providers_data = item.get('watch/providers', {}).get('results', {}).get(region, {})

        result = {
            'streaming_providers': '',  # Subscription services
            'rent_providers': '',
            'buy_providers': '',
            'on_netflix': False,
            'on_amazon': False,
            'on_disney': False,
            'on_hulu': False,
            'on_hbo': False,
            'on_apple': False,
        }

        # Extract flatrate (subscription) providers
        flatrate = providers_data.get('flatrate', [])
        streaming_names = []
        for provider in flatrate:
            pid = provider.get('provider_id')
            name = provider.get('provider_name', self.STREAMING_PROVIDERS.get(pid, f'Provider_{pid}'))
            streaming_names.append(name)

            # Set boolean flags for major providers
            if pid == 8:
                result['on_netflix'] = True
            elif pid in [9, 10]:
                result['on_amazon'] = True
            elif pid == 337:
                result['on_disney'] = True
            elif pid == 15:
                result['on_hulu'] = True
            elif pid == 384:
                result['on_hbo'] = True
            elif pid == 350:
                result['on_apple'] = True

        result['streaming_providers'] = '|'.join(streaming_names)

        # Extract rent providers
        rent = providers_data.get('rent', [])
        result['rent_providers'] = '|'.join([p.get('provider_name', '') for p in rent])

        # Extract buy providers
        buy = providers_data.get('buy', [])
        result['buy_providers'] = '|'.join([p.get('provider_name', '') for p in buy])

        return result

    def process_movies(self, raw_movies: List[Dict]) -> pd.DataFrame:
        """Process raw movie data into normalized format"""
        processed_movies = []
        
        logger.info(f"Processing {len(raw_movies)} movies")
        
        for movie in raw_movies:
            try:
                processed = {
                    'movie_id': movie.get('id'),
                    'title': movie.get('title', ''),
                    'original_title': movie.get('original_title', ''),
                    'overview': movie.get('overview', ''),
                    'release_date': movie.get('release_date', ''),
                    'genres': '|'.join([str(g.get('name', '')) for g in movie.get('genres', [])]),
                    'genre_ids': '|'.join([str(g) for g in movie.get('genre_ids', [])]),
                    'runtime': movie.get('runtime', 0),
                    'vote_average': movie.get('vote_average', 0.0),
                    'vote_count': movie.get('vote_count', 0),
                    'popularity': movie.get('popularity', 0.0),
                    'budget': movie.get('budget', 0),
                    'revenue': movie.get('revenue', 0),
                    'original_language': movie.get('original_language', ''),
                    'spoken_languages': '|'.join([lang.get('name', '') for lang in movie.get('spoken_languages', [])]),
                    'production_companies': '|'.join([comp.get('name', '') for comp in movie.get('production_companies', [])]),
                    'production_countries': '|'.join([country.get('name', '') for country in movie.get('production_countries', [])]),
                    'adult': movie.get('adult', False),
                    'poster_path': movie.get('poster_path', ''),
                    'backdrop_path': movie.get('backdrop_path', ''),
                    'import_date': datetime.now().isoformat(),
                    'data_source': 'tmdb_api'
                }

                # Extract streaming provider info
                streaming_info = self._extract_streaming_providers(movie)
                processed.update(streaming_info)

                # Calculate derived features
                processed['year'] = int(processed['release_date'][:4]) if processed['release_date'] else 0
                processed['decade'] = (processed['year'] // 10) * 10 if processed['year'] > 0 else 0
                processed['has_budget'] = processed['budget'] > 0
                processed['has_revenue'] = processed['revenue'] > 0
                processed['roi'] = processed['revenue'] / processed['budget'] if processed['budget'] > 0 else 0
                processed['vote_weight'] = np.log1p(processed['vote_count'])
                processed['weighted_rating'] = processed['vote_average'] * processed['vote_weight']

                processed_movies.append(processed)
                
            except Exception as e:
                logger.warning(f"Failed to process movie {movie.get('id', 'unknown')}: {e}")
        
        df = pd.DataFrame(processed_movies)
        logger.info(f"Successfully processed {len(df)} movies")
        
        return df
    
    def process_tv_shows(self, raw_shows: List[Dict]) -> pd.DataFrame:
        """Process raw TV show data into normalized format"""
        processed_shows = []
        
        logger.info(f"Processing {len(raw_shows)} TV shows")
        
        for show in raw_shows:
            try:
                processed = {
                    'show_id': show.get('id'),
                    'name': show.get('name', ''),
                    'original_name': show.get('original_name', ''),
                    'overview': show.get('overview', ''),
                    'first_air_date': show.get('first_air_date', ''),
                    'last_air_date': show.get('last_air_date', ''),
                    'genres': '|'.join([str(g.get('name', '')) for g in show.get('genres', [])]),
                    'genre_ids': '|'.join([str(g) for g in show.get('genre_ids', [])]),
                    'number_of_episodes': show.get('number_of_episodes', 0),
                    'number_of_seasons': show.get('number_of_seasons', 0),
                    'episode_run_time': '|'.join([str(t) for t in show.get('episode_run_time', [])]),
                    'vote_average': show.get('vote_average', 0.0),
                    'vote_count': show.get('vote_count', 0),
                    'popularity': show.get('popularity', 0.0),
                    'status': show.get('status', ''),
                    'type': show.get('type', ''),
                    'in_production': show.get('in_production', False),
                    'original_language': show.get('original_language', ''),
                    'spoken_languages': '|'.join([lang.get('name', '') for lang in show.get('spoken_languages', [])]),
                    'production_companies': '|'.join([comp.get('name', '') for comp in show.get('production_companies', [])]),
                    'production_countries': '|'.join([country.get('name', '') for country in show.get('production_countries', [])]),
                    'networks': '|'.join([net.get('name', '') for net in show.get('networks', [])]),
                    'created_by': '|'.join([creator.get('name', '') for creator in show.get('created_by', [])]),
                    'poster_path': show.get('poster_path', ''),
                    'backdrop_path': show.get('backdrop_path', ''),
                    'import_date': datetime.now().isoformat(),
                    'data_source': 'tmdb_api'
                }

                # Extract streaming provider info
                streaming_info = self._extract_streaming_providers(show)
                processed.update(streaming_info)

                # Calculate derived features
                processed['start_year'] = int(processed['first_air_date'][:4]) if processed['first_air_date'] else 0
                processed['end_year'] = int(processed['last_air_date'][:4]) if processed['last_air_date'] else 0
                processed['decade'] = (processed['start_year'] // 10) * 10 if processed['start_year'] > 0 else 0
                processed['avg_episode_runtime'] = np.mean([int(t) for t in processed['episode_run_time'].split('|') if t.isdigit()]) if processed['episode_run_time'] else 0
                processed['is_ongoing'] = processed['in_production'] or processed['status'] in ['Returning Series', 'In Production']
                processed['total_runtime'] = processed['number_of_episodes'] * processed['avg_episode_runtime']
                processed['vote_weight'] = np.log1p(processed['vote_count'])
                processed['weighted_rating'] = processed['vote_average'] * processed['vote_weight']

                # Status encoding for model
                status_encoding = {
                    'Ended': 0, 'Canceled': 1, 'Returning Series': 2, 'In Production': 3, 'Planned': 4
                }
                processed['status_encoded'] = status_encoding.get(processed['status'], 0)
                
                processed_shows.append(processed)
                
            except Exception as e:
                logger.warning(f"Failed to process TV show {show.get('id', 'unknown')}: {e}")
        
        df = pd.DataFrame(processed_shows)
        logger.info(f"Successfully processed {len(df)} TV shows")

        return df

    def process_anime(self, raw_anime: List[Dict]) -> pd.DataFrame:
        """Process raw anime data from Jikan/MAL into normalized format"""
        processed_anime = []

        logger.info(f"Processing {len(raw_anime)} anime")

        for anime in raw_anime:
            try:
                # Extract nested data safely
                images = anime.get('images', {})
                jpg_images = images.get('jpg', {})

                aired = anime.get('aired', {})
                aired_from = aired.get('from', '')
                aired_to = aired.get('to', '')

                # Extract genres, themes, and demographics
                genres = [g.get('name', '') for g in anime.get('genres', [])]
                themes = [t.get('name', '') for t in anime.get('themes', [])]
                demographics = [d.get('name', '') for d in anime.get('demographics', [])]

                # Studios and producers
                studios = [s.get('name', '') for s in anime.get('studios', [])]
                producers = [p.get('name', '') for p in anime.get('producers', [])]

                processed = {
                    'mal_id': anime.get('mal_id'),
                    'title': anime.get('title', ''),
                    'title_english': anime.get('title_english', ''),
                    'title_japanese': anime.get('title_japanese', ''),
                    'type': anime.get('type', ''),  # TV, Movie, OVA, ONA, Special
                    'source': anime.get('source', ''),  # Manga, Light Novel, Original, etc.
                    'episodes': anime.get('episodes', 0),
                    'status': anime.get('status', ''),  # Airing, Finished, Upcoming
                    'airing': anime.get('airing', False),
                    'aired_from': aired_from[:10] if aired_from else '',
                    'aired_to': aired_to[:10] if aired_to else '',
                    'duration': anime.get('duration', ''),
                    'rating': anime.get('rating', ''),  # PG-13, R, etc.
                    'score': anime.get('score', 0.0),
                    'scored_by': anime.get('scored_by', 0),
                    'rank': anime.get('rank', 0),
                    'popularity': anime.get('popularity', 0),
                    'members': anime.get('members', 0),
                    'favorites': anime.get('favorites', 0),
                    'synopsis': anime.get('synopsis', ''),
                    'season': anime.get('season', ''),
                    'year': anime.get('year', 0),
                    'genres': '|'.join(genres),
                    'themes': '|'.join(themes),
                    'demographics': '|'.join(demographics),
                    'studios': '|'.join(studios),
                    'producers': '|'.join(producers),
                    'poster_url': jpg_images.get('large_image_url', ''),
                    'url': anime.get('url', ''),
                    'trailer_url': anime.get('trailer', {}).get('url', ''),
                    'import_date': datetime.now().isoformat(),
                    'data_source': 'jikan_mal'
                }

                # Calculate derived features
                if processed['aired_from']:
                    try:
                        processed['start_year'] = int(processed['aired_from'][:4])
                    except (ValueError, IndexError):
                        processed['start_year'] = processed.get('year', 0)
                else:
                    processed['start_year'] = processed.get('year', 0)

                processed['decade'] = (processed['start_year'] // 10) * 10 if processed['start_year'] > 0 else 0
                processed['is_ongoing'] = processed['airing'] or processed['status'] == 'Currently Airing'
                processed['score_normalized'] = processed['score'] / 2.0 if processed['score'] else 0  # Normalize to 5-point scale
                processed['log_members'] = np.log1p(processed['members'])
                processed['weighted_score'] = processed['score'] * np.log1p(processed['scored_by']) if processed['score'] else 0

                # Content type classification
                type_mapping = {'TV': 'series', 'Movie': 'movie', 'OVA': 'ova', 'ONA': 'ona', 'Special': 'special'}
                processed['content_type'] = type_mapping.get(processed['type'], 'other')

                processed_anime.append(processed)

            except Exception as e:
                logger.warning(f"Failed to process anime {anime.get('mal_id', 'unknown')}: {e}")

        df = pd.DataFrame(processed_anime)
        logger.info(f"Successfully processed {len(df)} anime")

        return df


class DatabaseManager:
    """Manage database operations for imported data - Movies, TV Shows, and Anime"""

    def __init__(self, config: DataImportConfig):
        self.config = config
        self.ensure_databases()

    def ensure_databases(self):
        """Ensure database directories and tables exist"""
        # Create directories
        for path in [self.config.db_path, self.config.movies_db_path,
                     self.config.tv_db_path, self.config.anime_db_path]:
            if path:
                os.makedirs(os.path.dirname(path) or '.', exist_ok=True)

        # Create unified database with all tables
        with sqlite3.connect(self.config.db_path) as conn:
            # Movies table with streaming info
            conn.execute('''
                CREATE TABLE IF NOT EXISTS movies (
                    movie_id INTEGER PRIMARY KEY,
                    title TEXT,
                    original_title TEXT,
                    overview TEXT,
                    release_date TEXT,
                    genres TEXT,
                    genre_ids TEXT,
                    runtime INTEGER,
                    vote_average REAL,
                    vote_count INTEGER,
                    popularity REAL,
                    budget INTEGER,
                    revenue INTEGER,
                    original_language TEXT,
                    spoken_languages TEXT,
                    production_companies TEXT,
                    production_countries TEXT,
                    adult BOOLEAN,
                    poster_path TEXT,
                    backdrop_path TEXT,
                    year INTEGER,
                    decade INTEGER,
                    has_budget BOOLEAN,
                    has_revenue BOOLEAN,
                    roi REAL,
                    vote_weight REAL,
                    weighted_rating REAL,
                    streaming_providers TEXT,
                    rent_providers TEXT,
                    buy_providers TEXT,
                    on_netflix BOOLEAN,
                    on_amazon BOOLEAN,
                    on_disney BOOLEAN,
                    on_hulu BOOLEAN,
                    on_hbo BOOLEAN,
                    on_apple BOOLEAN,
                    import_date TEXT,
                    data_source TEXT
                )
            ''')

            # TV shows table with streaming info
            conn.execute('''
                CREATE TABLE IF NOT EXISTS tv_shows (
                    show_id INTEGER PRIMARY KEY,
                    name TEXT,
                    original_name TEXT,
                    overview TEXT,
                    first_air_date TEXT,
                    last_air_date TEXT,
                    genres TEXT,
                    genre_ids TEXT,
                    number_of_episodes INTEGER,
                    number_of_seasons INTEGER,
                    episode_run_time TEXT,
                    vote_average REAL,
                    vote_count INTEGER,
                    popularity REAL,
                    status TEXT,
                    type TEXT,
                    in_production BOOLEAN,
                    original_language TEXT,
                    spoken_languages TEXT,
                    production_companies TEXT,
                    production_countries TEXT,
                    networks TEXT,
                    created_by TEXT,
                    poster_path TEXT,
                    backdrop_path TEXT,
                    start_year INTEGER,
                    end_year INTEGER,
                    decade INTEGER,
                    avg_episode_runtime REAL,
                    is_ongoing BOOLEAN,
                    total_runtime REAL,
                    vote_weight REAL,
                    weighted_rating REAL,
                    status_encoded INTEGER,
                    streaming_providers TEXT,
                    rent_providers TEXT,
                    buy_providers TEXT,
                    on_netflix BOOLEAN,
                    on_amazon BOOLEAN,
                    on_disney BOOLEAN,
                    on_hulu BOOLEAN,
                    on_hbo BOOLEAN,
                    on_apple BOOLEAN,
                    import_date TEXT,
                    data_source TEXT
                )
            ''')

            # Anime table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS anime (
                    mal_id INTEGER PRIMARY KEY,
                    title TEXT,
                    title_english TEXT,
                    title_japanese TEXT,
                    type TEXT,
                    source TEXT,
                    episodes INTEGER,
                    status TEXT,
                    airing BOOLEAN,
                    aired_from TEXT,
                    aired_to TEXT,
                    duration TEXT,
                    rating TEXT,
                    score REAL,
                    scored_by INTEGER,
                    rank INTEGER,
                    popularity INTEGER,
                    members INTEGER,
                    favorites INTEGER,
                    synopsis TEXT,
                    season TEXT,
                    year INTEGER,
                    genres TEXT,
                    themes TEXT,
                    demographics TEXT,
                    studios TEXT,
                    producers TEXT,
                    poster_url TEXT,
                    url TEXT,
                    trailer_url TEXT,
                    start_year INTEGER,
                    decade INTEGER,
                    is_ongoing BOOLEAN,
                    score_normalized REAL,
                    log_members REAL,
                    weighted_score REAL,
                    content_type TEXT,
                    import_date TEXT,
                    data_source TEXT
                )
            ''')

            # Create indexes for performance
            conn.execute('CREATE INDEX IF NOT EXISTS idx_movies_year ON movies(year)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_movies_popularity ON movies(popularity)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_tv_year ON tv_shows(start_year)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_tv_popularity ON tv_shows(popularity)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_anime_year ON anime(year)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_anime_score ON anime(score)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_anime_popularity ON anime(popularity)')

            # Streaming provider indexes for fast filtering
            conn.execute('CREATE INDEX IF NOT EXISTS idx_movies_netflix ON movies(on_netflix)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_movies_amazon ON movies(on_amazon)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_movies_disney ON movies(on_disney)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_movies_hulu ON movies(on_hulu)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_movies_hbo ON movies(on_hbo)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_movies_apple ON movies(on_apple)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_tv_netflix ON tv_shows(on_netflix)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_tv_amazon ON tv_shows(on_amazon)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_tv_disney ON tv_shows(on_disney)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_tv_hulu ON tv_shows(on_hulu)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_tv_hbo ON tv_shows(on_hbo)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_tv_apple ON tv_shows(on_apple)')

        # Also maintain legacy separate databases for backwards compatibility
        with sqlite3.connect(self.config.movies_db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS movies (
                    movie_id INTEGER PRIMARY KEY,
                    title TEXT,
                    original_title TEXT,
                    overview TEXT,
                    release_date TEXT,
                    genres TEXT,
                    genre_ids TEXT,
                    runtime INTEGER,
                    vote_average REAL,
                    vote_count INTEGER,
                    popularity REAL,
                    budget INTEGER,
                    revenue INTEGER,
                    original_language TEXT,
                    spoken_languages TEXT,
                    production_companies TEXT,
                    production_countries TEXT,
                    adult BOOLEAN,
                    poster_path TEXT,
                    backdrop_path TEXT,
                    year INTEGER,
                    decade INTEGER,
                    has_budget BOOLEAN,
                    has_revenue BOOLEAN,
                    roi REAL,
                    vote_weight REAL,
                    weighted_rating REAL,
                    import_date TEXT,
                    data_source TEXT
                )
            ''')

        with sqlite3.connect(self.config.tv_db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS tv_shows (
                    show_id INTEGER PRIMARY KEY,
                    name TEXT,
                    original_name TEXT,
                    overview TEXT,
                    first_air_date TEXT,
                    last_air_date TEXT,
                    genres TEXT,
                    genre_ids TEXT,
                    number_of_episodes INTEGER,
                    number_of_seasons INTEGER,
                    episode_run_time TEXT,
                    vote_average REAL,
                    vote_count INTEGER,
                    popularity REAL,
                    status TEXT,
                    type TEXT,
                    in_production BOOLEAN,
                    original_language TEXT,
                    spoken_languages TEXT,
                    production_companies TEXT,
                    production_countries TEXT,
                    networks TEXT,
                    created_by TEXT,
                    poster_path TEXT,
                    backdrop_path TEXT,
                    start_year INTEGER,
                    end_year INTEGER,
                    decade INTEGER,
                    avg_episode_runtime REAL,
                    is_ongoing BOOLEAN,
                    total_runtime REAL,
                    vote_weight REAL,
                    weighted_rating REAL,
                    status_encoded INTEGER,
                    import_date TEXT,
                    data_source TEXT
                )
            ''')
    
    def save_movies(self, movies_df: pd.DataFrame) -> int:
        """Save movies to database and return count of new records"""
        if movies_df.empty:
            return 0

        new_count = 0

        # Save to unified database
        with sqlite3.connect(self.config.db_path) as conn:
            try:
                existing_ids = set(pd.read_sql('SELECT movie_id FROM movies', conn)['movie_id'].tolist())
            except Exception:
                existing_ids = set()

            new_movies = movies_df[~movies_df['movie_id'].isin(existing_ids)]

            if len(new_movies) > 0:
                new_movies.to_sql('movies', conn, if_exists='append', index=False)
                new_count = len(new_movies)
                logger.info(f"Saved {new_count} new movies to unified database")

        # Also save to legacy database
        with sqlite3.connect(self.config.movies_db_path) as conn:
            try:
                existing_ids = set(pd.read_sql('SELECT movie_id FROM movies', conn)['movie_id'].tolist())
            except Exception:
                existing_ids = set()

            new_movies = movies_df[~movies_df['movie_id'].isin(existing_ids)]
            if len(new_movies) > 0:
                new_movies.to_sql('movies', conn, if_exists='append', index=False)

        return new_count

    def save_tv_shows(self, tv_df: pd.DataFrame) -> int:
        """Save TV shows to database and return count of new records"""
        if tv_df.empty:
            return 0

        new_count = 0

        # Save to unified database
        with sqlite3.connect(self.config.db_path) as conn:
            try:
                existing_ids = set(pd.read_sql('SELECT show_id FROM tv_shows', conn)['show_id'].tolist())
            except Exception:
                existing_ids = set()

            new_shows = tv_df[~tv_df['show_id'].isin(existing_ids)]

            if len(new_shows) > 0:
                new_shows.to_sql('tv_shows', conn, if_exists='append', index=False)
                new_count = len(new_shows)
                logger.info(f"Saved {new_count} new TV shows to unified database")

        # Also save to legacy database
        with sqlite3.connect(self.config.tv_db_path) as conn:
            try:
                existing_ids = set(pd.read_sql('SELECT show_id FROM tv_shows', conn)['show_id'].tolist())
            except Exception:
                existing_ids = set()

            new_shows = tv_df[~tv_df['show_id'].isin(existing_ids)]
            if len(new_shows) > 0:
                new_shows.to_sql('tv_shows', conn, if_exists='append', index=False)

        return new_count

    def save_anime(self, anime_df: pd.DataFrame) -> int:
        """Save anime to database and return count of new records"""
        if anime_df.empty:
            return 0

        new_count = 0

        # Save to unified database
        with sqlite3.connect(self.config.db_path) as conn:
            try:
                existing_ids = set(pd.read_sql('SELECT mal_id FROM anime', conn)['mal_id'].tolist())
            except Exception:
                existing_ids = set()

            new_anime = anime_df[~anime_df['mal_id'].isin(existing_ids)]

            if len(new_anime) > 0:
                new_anime.to_sql('anime', conn, if_exists='append', index=False)
                new_count = len(new_anime)
                logger.info(f"Saved {new_count} new anime to database")

        return new_count

    def get_import_stats(self) -> Dict:
        """Get import statistics from unified database"""
        stats = {}

        with sqlite3.connect(self.config.db_path) as conn:
            # Movie stats
            try:
                movie_stats = pd.read_sql('''
                    SELECT
                        COUNT(*) as total_movies,
                        COUNT(CASE WHEN import_date >= date('now', '-7 days') THEN 1 END) as movies_last_week,
                        COUNT(CASE WHEN import_date >= date('now', '-30 days') THEN 1 END) as movies_last_month,
                        AVG(vote_average) as avg_rating,
                        AVG(popularity) as avg_popularity,
                        MIN(year) as oldest_year,
                        MAX(year) as newest_year
                    FROM movies
                ''', conn).iloc[0].to_dict()
                stats['movies'] = movie_stats
            except Exception as e:
                logger.warning(f"Could not get movie stats: {e}")
                stats['movies'] = {}

            # TV stats
            try:
                tv_stats = pd.read_sql('''
                    SELECT
                        COUNT(*) as total_shows,
                        COUNT(CASE WHEN import_date >= date('now', '-7 days') THEN 1 END) as shows_last_week,
                        COUNT(CASE WHEN import_date >= date('now', '-30 days') THEN 1 END) as shows_last_month,
                        AVG(vote_average) as avg_rating,
                        AVG(popularity) as avg_popularity,
                        MIN(start_year) as oldest_year,
                        MAX(start_year) as newest_year
                    FROM tv_shows
                ''', conn).iloc[0].to_dict()
                stats['tv_shows'] = tv_stats
            except Exception as e:
                logger.warning(f"Could not get TV stats: {e}")
                stats['tv_shows'] = {}

            # Anime stats
            try:
                anime_stats = pd.read_sql('''
                    SELECT
                        COUNT(*) as total_anime,
                        COUNT(CASE WHEN import_date >= date('now', '-7 days') THEN 1 END) as anime_last_week,
                        COUNT(CASE WHEN import_date >= date('now', '-30 days') THEN 1 END) as anime_last_month,
                        AVG(score) as avg_score,
                        SUM(members) as total_members,
                        MIN(year) as oldest_year,
                        MAX(year) as newest_year,
                        COUNT(CASE WHEN airing = 1 THEN 1 END) as currently_airing
                    FROM anime
                ''', conn).iloc[0].to_dict()
                stats['anime'] = anime_stats
            except Exception as e:
                logger.warning(f"Could not get anime stats: {e}")
                stats['anime'] = {}

        return stats


class ModelRetrainer:
    """Handle automatic model retraining when new data is available"""

    def __init__(self, config: DataImportConfig):
        self.config = config

    def should_retrain(self, new_movies: int, new_tv_shows: int, new_anime: int = 0) -> Dict[str, bool]:
        """Determine which models need retraining"""
        threshold = self.config.retrain_threshold
        return {
            'movie_models': new_movies >= threshold,
            'tv_models': new_tv_shows >= threshold,
            'anime_models': new_anime >= threshold,
            'hybrid_model': (new_movies + new_tv_shows + new_anime) >= threshold
        }

    def trigger_retraining(self, models_to_retrain: Dict[str, bool]) -> Dict[str, bool]:
        """Trigger retraining for specified models"""
        results = {}

        if models_to_retrain.get('movie_models'):
            logger.info("Triggering movie model retraining")
            results['movie_models'] = self._retrain_movie_models()

        if models_to_retrain.get('tv_models'):
            logger.info("Triggering TV model retraining")
            results['tv_models'] = self._retrain_tv_models()

        if models_to_retrain.get('anime_models'):
            logger.info("Triggering anime model retraining")
            results['anime_models'] = self._retrain_anime_models()

        if models_to_retrain.get('hybrid_model'):
            logger.info("Triggering hybrid model retraining")
            results['hybrid_model'] = self._retrain_hybrid_model()

        return results

    def _retrain_movie_models(self) -> bool:
        """Retrain movie-specific models"""
        try:
            logger.info("Movie model retraining would be triggered here")
            return True
        except Exception as e:
            logger.error(f"Movie model retraining failed: {e}")
            return False

    def _retrain_tv_models(self) -> bool:
        """Retrain TV-specific models"""
        try:
            logger.info("TV model retraining would be triggered here")
            return True
        except Exception as e:
            logger.error(f"TV model retraining failed: {e}")
            return False

    def _retrain_anime_models(self) -> bool:
        """Retrain anime-specific models"""
        try:
            logger.info("Anime model retraining would be triggered here")
            return True
        except Exception as e:
            logger.error(f"Anime model retraining failed: {e}")
            return False

    def _retrain_hybrid_model(self) -> bool:
        """Retrain hybrid model"""
        try:
            logger.info("Hybrid model retraining would be triggered here")
            return True
        except Exception as e:
            logger.error(f"Hybrid model retraining failed: {e}")
            return False


class DataImportPipeline:
    """Main data import pipeline orchestrator - Movies, TV Shows, and Anime"""

    def __init__(self, config: DataImportConfig):
        self.config = config
        self.tmdb_importer = None
        self.anime_importer = None
        self.processor = DataProcessor(config)
        self.db_manager = DatabaseManager(config)
        self.retrainer = ModelRetrainer(config)
        self.wandb_manager = None

    def _init_importers(self):
        """Initialize importers lazily to handle missing API keys gracefully"""
        if self.config.tmdb_api_key and self.tmdb_importer is None:
            try:
                self.tmdb_importer = TMDBDataImporter(self.config)
            except ValueError as e:
                logger.warning(f"TMDB importer not available: {e}")

        if self.anime_importer is None:
            self.anime_importer = JikanAnimeImporter(self.config)

    def run_import(self, import_movies: bool = None, import_tv: bool = None,
                   import_anime: bool = None) -> Dict:
        """Run the complete data import pipeline for movies, TV, and anime"""

        # Use config defaults if not specified
        import_movies = import_movies if import_movies is not None else self.config.import_movies
        import_tv = import_tv if import_tv is not None else self.config.import_tv
        import_anime = import_anime if import_anime is not None else self.config.import_anime

        # Initialize importers
        self._init_importers()

        # Initialize wandb for tracking
        wandb_config = {
            'pipeline_type': 'data_import',
            'start_date': self.config.start_date,
            'end_date': self.config.end_date,
            'import_movies': import_movies,
            'import_tv': import_tv,
            'import_anime': import_anime,
            'min_vote_count': self.config.min_vote_count,
            'min_popularity': self.config.min_popularity
        }

        if WANDB_AVAILABLE:
            self.wandb_manager = init_wandb_for_training('data_import', wandb_config)

        try:
            results = {
                'start_time': datetime.now().isoformat(),
                'config': {
                    'start_date': self.config.start_date,
                    'end_date': self.config.end_date
                },
                'movies_imported': 0,
                'movies_fetched': 0,
                'tv_shows_imported': 0,
                'tv_shows_fetched': 0,
                'anime_imported': 0,
                'anime_fetched': 0,
                'retraining_triggered': {},
                'errors': []
            }

            # Import movies
            if import_movies:
                if self.tmdb_importer:
                    try:
                        logger.info("=" * 50)
                        logger.info("Starting comprehensive movie import")
                        logger.info("=" * 50)
                        raw_movies = self.tmdb_importer.get_all_movies_comprehensive()
                        results['movies_fetched'] = len(raw_movies)

                        if raw_movies:
                            processed_movies = self.processor.process_movies(raw_movies)
                            new_movie_count = self.db_manager.save_movies(processed_movies)
                            results['movies_imported'] = new_movie_count

                            if self.wandb_manager:
                                self.wandb_manager.log_metrics({
                                    'movies/raw_fetched': len(raw_movies),
                                    'movies/processed': len(processed_movies),
                                    'movies/new_imported': new_movie_count
                                })

                            # Save CSV export
                            output_path = Path(self.config.output_dir) / self.config.movies_output
                            output_path.parent.mkdir(parents=True, exist_ok=True)
                            processed_movies.to_csv(output_path, index=False)
                            logger.info(f"Exported {len(processed_movies)} movies to {output_path}")

                    except Exception as e:
                        error_msg = f"Movie import failed: {e}"
                        logger.error(error_msg)
                        results['errors'].append(error_msg)
                else:
                    logger.warning("TMDB API key not configured - skipping movie import")
                    results['errors'].append("TMDB API key not configured")

            # Import TV shows
            if import_tv:
                if self.tmdb_importer:
                    try:
                        logger.info("=" * 50)
                        logger.info("Starting comprehensive TV show import")
                        logger.info("=" * 50)
                        raw_tv_shows = self.tmdb_importer.get_all_tv_shows_comprehensive()
                        results['tv_shows_fetched'] = len(raw_tv_shows)

                        if raw_tv_shows:
                            processed_tv = self.processor.process_tv_shows(raw_tv_shows)
                            new_tv_count = self.db_manager.save_tv_shows(processed_tv)
                            results['tv_shows_imported'] = new_tv_count

                            if self.wandb_manager:
                                self.wandb_manager.log_metrics({
                                    'tv_shows/raw_fetched': len(raw_tv_shows),
                                    'tv_shows/processed': len(processed_tv),
                                    'tv_shows/new_imported': new_tv_count
                                })

                            # Save CSV export
                            output_path = Path(self.config.output_dir) / self.config.tv_output
                            output_path.parent.mkdir(parents=True, exist_ok=True)
                            processed_tv.to_csv(output_path, index=False)
                            logger.info(f"Exported {len(processed_tv)} TV shows to {output_path}")

                    except Exception as e:
                        error_msg = f"TV show import failed: {e}"
                        logger.error(error_msg)
                        results['errors'].append(error_msg)
                else:
                    logger.warning("TMDB API key not configured - skipping TV import")

            # Import anime
            if import_anime:
                try:
                    logger.info("=" * 50)
                    logger.info("Starting comprehensive anime import")
                    logger.info("=" * 50)
                    raw_anime = self.anime_importer.get_all_anime_comprehensive()
                    results['anime_fetched'] = len(raw_anime)

                    if raw_anime:
                        processed_anime = self.processor.process_anime(raw_anime)
                        new_anime_count = self.db_manager.save_anime(processed_anime)
                        results['anime_imported'] = new_anime_count

                        if self.wandb_manager:
                            self.wandb_manager.log_metrics({
                                'anime/raw_fetched': len(raw_anime),
                                'anime/processed': len(processed_anime),
                                'anime/new_imported': new_anime_count
                            })

                        # Save CSV export
                        output_path = Path(self.config.output_dir) / self.config.anime_output
                        output_path.parent.mkdir(parents=True, exist_ok=True)
                        processed_anime.to_csv(output_path, index=False)
                        logger.info(f"Exported {len(processed_anime)} anime to {output_path}")

                except Exception as e:
                    error_msg = f"Anime import failed: {e}"
                    logger.error(error_msg)
                    results['errors'].append(error_msg)

            # Check if retraining is needed
            if self.config.auto_retrain:
                models_to_retrain = self.retrainer.should_retrain(
                    results['movies_imported'],
                    results['tv_shows_imported'],
                    results['anime_imported']
                )

                if any(models_to_retrain.values()):
                    logger.info(f"Triggering retraining: {models_to_retrain}")
                    retraining_results = self.retrainer.trigger_retraining(models_to_retrain)
                    results['retraining_triggered'] = retraining_results

                    if self.wandb_manager:
                        self.wandb_manager.log_metrics({
                            f'retraining/{model}': success
                            for model, success in retraining_results.items()
                        })

            # Get final statistics
            stats = self.db_manager.get_import_stats()
            results['final_stats'] = stats

            # Log final metrics
            if self.wandb_manager:
                self.wandb_manager.log_metrics({
                    'pipeline/total_movies_imported': results['movies_imported'],
                    'pipeline/total_tv_imported': results['tv_shows_imported'],
                    'pipeline/total_anime_imported': results['anime_imported'],
                    'pipeline/errors_count': len(results['errors']),
                    'pipeline/success': len(results['errors']) == 0
                })

            results['end_time'] = datetime.now().isoformat()
            results['success'] = len(results['errors']) == 0

            # Print summary
            self._print_summary(results)

            return results

        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            raise
        finally:
            if self.wandb_manager:
                self.wandb_manager.finish()

    def _print_summary(self, results: Dict):
        """Print a nice summary of the import results"""
        print("\n" + "=" * 60)
        print("IMPORT SUMMARY")
        print("=" * 60)
        print(f"Start: {results['start_time']}")
        print(f"End:   {results.get('end_time', 'N/A')}")
        print("-" * 60)
        print(f"Movies:    {results['movies_fetched']:>6} fetched, {results['movies_imported']:>6} new")
        print(f"TV Shows:  {results['tv_shows_fetched']:>6} fetched, {results['tv_shows_imported']:>6} new")
        print(f"Anime:     {results['anime_fetched']:>6} fetched, {results['anime_imported']:>6} new")
        print("-" * 60)
        total_fetched = results['movies_fetched'] + results['tv_shows_fetched'] + results['anime_fetched']
        total_imported = results['movies_imported'] + results['tv_shows_imported'] + results['anime_imported']
        print(f"Total:     {total_fetched:>6} fetched, {total_imported:>6} new")
        print("-" * 60)

        if results.get('final_stats'):
            stats = results['final_stats']
            if stats.get('movies'):
                print(f"Database Movies: {stats['movies'].get('total_movies', 0)} total")
            if stats.get('tv_shows'):
                print(f"Database TV:     {stats['tv_shows'].get('total_shows', 0)} total")
            if stats.get('anime'):
                print(f"Database Anime:  {stats['anime'].get('total_anime', 0)} total")

        if results['errors']:
            print("-" * 60)
            print(f"Errors: {len(results['errors'])}")
            for err in results['errors']:
                print(f"  - {err}")

        print("=" * 60 + "\n")


def setup_logging(log_to_file: bool = True):
    """Setup logging configuration"""
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_to_file:
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        handlers.append(
            logging.FileHandler(log_dir / f"data_import_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        )

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='CineSync v2 Data Import Pipeline - Movies, TV Shows, and Anime',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Import everything (movies, TV, anime) with default settings
  python data_import_pipeline.py

  # Import only movies from the last year
  python data_import_pipeline.py --movies-only --lookback-days 365

  # Import only anime (no TMDB API key needed)
  python data_import_pipeline.py --anime-only

  # Full import with custom date range
  python data_import_pipeline.py --start-date 2024-01-01 --end-date 2026-01-16

  # Quick mode (less comprehensive, faster)
  python data_import_pipeline.py --quick
        """
    )

    # Content type selection
    content_group = parser.add_argument_group('Content Selection')
    content_group.add_argument('--movies-only', action='store_true',
                               help='Import only movies (requires TMDB API key)')
    content_group.add_argument('--tv-only', action='store_true',
                               help='Import only TV shows (requires TMDB API key)')
    content_group.add_argument('--anime-only', action='store_true',
                               help='Import only anime (no API key needed)')
    content_group.add_argument('--no-movies', action='store_true',
                               help='Skip movie import')
    content_group.add_argument('--no-tv', action='store_true',
                               help='Skip TV show import')
    content_group.add_argument('--no-anime', action='store_true',
                               help='Skip anime import')

    # Date settings
    date_group = parser.add_argument_group('Date Range')
    date_group.add_argument('--start-date', type=str, default='2017-01-01',
                            help='Start date for importing content (YYYY-MM-DD). Default: 2017-01-01')
    date_group.add_argument('--end-date', type=str,
                            help='End date for importing content (YYYY-MM-DD). Default: today')
    date_group.add_argument('--from-year', type=int, default=2017,
                            help='Start year for comprehensive import (default: 2017)')

    # API settings
    api_group = parser.add_argument_group('API Configuration')
    api_group.add_argument('--tmdb-api-key', type=str,
                           help='TMDB API key (or set TMDB_API_KEY env var)')
    api_group.add_argument('--max-pages', type=int, default=100,
                           help='Maximum pages to fetch per category (default: 100)')

    # Fetch mode
    mode_group = parser.add_argument_group('Fetch Mode')
    mode_group.add_argument('--quick', action='store_true',
                            help='Quick mode: skip genre-by-genre fetching for faster import')
    mode_group.add_argument('--no-trending', action='store_true',
                            help='Skip trending content')
    mode_group.add_argument('--no-genres', action='store_true',
                            help='Skip genre-by-genre fetching')

    # Quality filters
    filter_group = parser.add_argument_group('Quality Filters')
    filter_group.add_argument('--min-vote-count', type=int, default=50,
                              help='Minimum vote count for imported content (default: 50)')
    filter_group.add_argument('--min-popularity', type=float, default=2.0,
                              help='Minimum popularity score (default: 2.0)')
    filter_group.add_argument('--relaxed', action='store_true',
                              help='Use relaxed quality filters (more content, lower quality)')
    filter_group.add_argument('--include-adult', action='store_true',
                              help='Include adult content')

    # Output settings
    output_group = parser.add_argument_group('Output')
    output_group.add_argument('--output-dir', type=str, default='data/imports',
                              help='Output directory for exported CSVs')
    output_group.add_argument('--db-path', type=str, default='data/cinesync.db',
                              help='Unified database path')
    output_group.add_argument('--no-log-file', action='store_true',
                              help='Disable file logging')

    # Other
    other_group = parser.add_argument_group('Other Options')
    other_group.add_argument('--no-auto-retrain', action='store_true',
                             help='Disable automatic model retraining')
    other_group.add_argument('--stats-only', action='store_true',
                             help='Only show database statistics, no import')

    return parser.parse_args()


def show_stats_only(config: DataImportConfig):
    """Show database statistics without importing"""
    db_manager = DatabaseManager(config)
    stats = db_manager.get_import_stats()

    print("\n" + "=" * 60)
    print("DATABASE STATISTICS")
    print("=" * 60)

    if stats.get('movies'):
        m = stats['movies']
        print(f"\nMovies:")
        print(f"  Total: {m.get('total_movies', 0)}")
        print(f"  Last week: {m.get('movies_last_week', 0)}")
        print(f"  Last month: {m.get('movies_last_month', 0)}")
        print(f"  Avg rating: {m.get('avg_rating', 0):.2f}")
        print(f"  Year range: {m.get('oldest_year', 'N/A')} - {m.get('newest_year', 'N/A')}")

    if stats.get('tv_shows'):
        t = stats['tv_shows']
        print(f"\nTV Shows:")
        print(f"  Total: {t.get('total_shows', 0)}")
        print(f"  Last week: {t.get('shows_last_week', 0)}")
        print(f"  Last month: {t.get('shows_last_month', 0)}")
        print(f"  Avg rating: {t.get('avg_rating', 0):.2f}")
        print(f"  Year range: {t.get('oldest_year', 'N/A')} - {t.get('newest_year', 'N/A')}")

    if stats.get('anime'):
        a = stats['anime']
        print(f"\nAnime:")
        print(f"  Total: {a.get('total_anime', 0)}")
        print(f"  Last week: {a.get('anime_last_week', 0)}")
        print(f"  Last month: {a.get('anime_last_month', 0)}")
        print(f"  Avg score: {a.get('avg_score', 0):.2f}")
        print(f"  Currently airing: {a.get('currently_airing', 0)}")
        print(f"  Year range: {a.get('oldest_year', 'N/A')} - {a.get('newest_year', 'N/A')}")

    print("\n" + "=" * 60)


def main():
    """Main function"""
    args = parse_args()
    setup_logging(log_to_file=not args.no_log_file)

    # Calculate date range - default from 2017
    start_date = args.start_date
    end_date = args.end_date or datetime.now().strftime("%Y-%m-%d")

    logger.info("=" * 60)
    logger.info("CineSync v2 Data Import Pipeline")
    logger.info("Movies, TV Shows, and Anime")
    logger.info("=" * 60)
    logger.info(f"Date range: {start_date} to {end_date}")
    logger.info(f"Quality filters: min_votes={args.min_vote_count}, min_popularity={args.min_popularity}")

    # Create configuration
    config = DataImportConfig(
        tmdb_api_key=args.tmdb_api_key or os.getenv('TMDB_API_KEY', ''),
        start_date=start_date,
        end_date=end_date,
        from_year=args.from_year,
        max_pages=args.max_pages,
        min_vote_count=args.min_vote_count,
        min_popularity=args.min_popularity,
        include_adult=args.include_adult,
        output_dir=args.output_dir,
        db_path=args.db_path,
        auto_retrain=not args.no_auto_retrain,
        fetch_trending=not args.no_trending,
        fetch_by_genre=not args.no_genres and not args.quick,
    )

    # Relaxed mode - lower quality thresholds
    if args.relaxed:
        config.min_vote_count = 10
        config.min_popularity = 0.5
        config.anime_min_score = 3.0
        config.anime_min_members = 100
        config.anime_min_scored_by = 10
        logger.info("Relaxed mode enabled - lower quality thresholds")

    # Quick mode adjustments
    if args.quick:
        config.max_pages = 50
        config.fetch_by_genre = False
        logger.info("Quick mode enabled - reduced fetch scope")

    # Stats only mode
    if args.stats_only:
        show_stats_only(config)
        return 0

    # Determine what to import
    if args.movies_only:
        import_movies, import_tv, import_anime = True, False, False
    elif args.tv_only:
        import_movies, import_tv, import_anime = False, True, False
    elif args.anime_only:
        import_movies, import_tv, import_anime = False, False, True
    else:
        import_movies = not args.no_movies
        import_tv = not args.no_tv
        import_anime = not args.no_anime

    # Check API key for TMDB content
    if (import_movies or import_tv) and not config.tmdb_api_key:
        logger.warning("TMDB API key not set. Movies and TV import will be skipped.")
        logger.info("Set TMDB_API_KEY environment variable or use --tmdb-api-key")
        logger.info("Get a free API key at: https://www.themoviedb.org/settings/api")
        if not import_anime:
            logger.error("No content types can be imported. Exiting.")
            return 1
        import_movies = False
        import_tv = False

    logger.info(f"Import targets: Movies={import_movies}, TV={import_tv}, Anime={import_anime}")

    try:
        # Run pipeline
        pipeline = DataImportPipeline(config)
        results = pipeline.run_import(
            import_movies=import_movies,
            import_tv=import_tv,
            import_anime=import_anime
        )

        if results['success']:
            logger.info("Data import pipeline completed successfully!")
        else:
            logger.warning("Pipeline completed with errors")

        return 0 if results['success'] else 1

    except KeyboardInterrupt:
        logger.info("\nImport cancelled by user")
        return 130
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())