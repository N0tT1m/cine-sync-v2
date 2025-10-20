#!/usr/bin/env python3
"""
API Integration Module for CineSync v2
Handles external API calls with rate limiting for datasets that can't be downloaded directly.
"""

import asyncio
import aiohttp
import time
import json
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import pandas as pd
import os
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class RateLimitConfig:
    """Configuration for API rate limiting"""
    requests_per_second: float
    requests_per_minute: int
    requests_per_hour: int
    requests_per_day: int
    backoff_factor: float = 2.0
    max_retries: int = 3

class RateLimiter:
    """Advanced rate limiter with multiple time windows"""
    
    def __init__(self, config: RateLimitConfig):
        self.config = config
        self.request_times = []
        self.last_request_time = 0
        self.retry_count = 0
    
    async def wait_if_needed(self):
        """Wait if necessary to respect rate limits"""
        current_time = time.time()
        
        # Remove old request times (older than 1 day)
        cutoff_time = current_time - 86400  # 24 hours
        self.request_times = [t for t in self.request_times if t > cutoff_time]
        
        # Check various time windows
        await self._check_daily_limit(current_time)
        await self._check_hourly_limit(current_time)
        await self._check_minute_limit(current_time)
        await self._check_second_limit(current_time)
        
        # Record this request
        self.request_times.append(current_time)
        self.last_request_time = current_time
    
    async def _check_daily_limit(self, current_time: float):
        daily_requests = [t for t in self.request_times if t > current_time - 86400]
        if len(daily_requests) >= self.config.requests_per_day:
            wait_time = 86400 - (current_time - min(daily_requests))
            logger.warning(f"Daily rate limit reached. Waiting {wait_time:.1f} seconds")
            await asyncio.sleep(wait_time)
    
    async def _check_hourly_limit(self, current_time: float):
        hourly_requests = [t for t in self.request_times if t > current_time - 3600]
        if len(hourly_requests) >= self.config.requests_per_hour:
            wait_time = 3600 - (current_time - min(hourly_requests))
            logger.warning(f"Hourly rate limit reached. Waiting {wait_time:.1f} seconds")
            await asyncio.sleep(wait_time)
    
    async def _check_minute_limit(self, current_time: float):
        minute_requests = [t for t in self.request_times if t > current_time - 60]
        if len(minute_requests) >= self.config.requests_per_minute:
            wait_time = 60 - (current_time - min(minute_requests))
            logger.info(f"Minute rate limit reached. Waiting {wait_time:.1f} seconds")
            await asyncio.sleep(wait_time)
    
    async def _check_second_limit(self, current_time: float):
        if self.last_request_time > 0:
            time_since_last = current_time - self.last_request_time
            min_interval = 1.0 / self.config.requests_per_second
            if time_since_last < min_interval:
                wait_time = min_interval - time_since_last
                await asyncio.sleep(wait_time)

class TMDBIntegration:
    """TMDB API integration with rate limiting"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.themoviedb.org/3"
        self.rate_limiter = RateLimiter(RateLimitConfig(
            requests_per_second=2.5,  # Conservative limit
            requests_per_minute=40,
            requests_per_hour=1000,
            requests_per_day=20000
        ))
        self.session = None
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def fetch_movie_details(self, movie_id: int) -> Optional[Dict]:
        """Fetch detailed movie information"""
        await self.rate_limiter.wait_if_needed()
        
        url = f"{self.base_url}/movie/{movie_id}"
        params = {
            'api_key': self.api_key,
            'append_to_response': 'credits,keywords,reviews,similar'
        }
        
        try:
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    return await response.json()
                elif response.status == 429:
                    logger.warning("Rate limit hit, backing off")
                    await asyncio.sleep(5)
                    return await self.fetch_movie_details(movie_id)
                else:
                    logger.error(f"Error fetching movie {movie_id}: {response.status}")
                    return None
        except Exception as e:
            logger.error(f"Exception fetching movie {movie_id}: {e}")
            return None
    
    async def fetch_tv_details(self, tv_id: int) -> Optional[Dict]:
        """Fetch detailed TV show information"""
        await self.rate_limiter.wait_if_needed()
        
        url = f"{self.base_url}/tv/{tv_id}"
        params = {
            'api_key': self.api_key,
            'append_to_response': 'credits,keywords,reviews,similar,season/1'
        }
        
        try:
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    return await response.json()
                elif response.status == 429:
                    logger.warning("Rate limit hit, backing off")
                    await asyncio.sleep(5)
                    return await self.fetch_tv_details(tv_id)
                else:
                    logger.error(f"Error fetching TV show {tv_id}: {response.status}")
                    return None
        except Exception as e:
            logger.error(f"Exception fetching TV show {tv_id}: {e}")
            return None
    
    async def discover_content(self, content_type: str = "movie", **kwargs) -> List[Dict]:
        """Discover content with filters"""
        await self.rate_limiter.wait_if_needed()
        
        url = f"{self.base_url}/discover/{content_type}"
        params = {'api_key': self.api_key, **kwargs}
        
        try:
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get('results', [])
                else:
                    logger.error(f"Error discovering {content_type}: {response.status}")
                    return []
        except Exception as e:
            logger.error(f"Exception discovering {content_type}: {e}")
            return []

class TVMazeIntegration:
    """TV Maze API integration"""
    
    def __init__(self):
        self.base_url = "https://api.tvmaze.com"
        self.rate_limiter = RateLimiter(RateLimitConfig(
            requests_per_second=1.0,  # Very conservative
            requests_per_minute=20,
            requests_per_hour=1000,
            requests_per_day=10000
        ))
        self.session = None
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def fetch_show_details(self, show_id: int) -> Optional[Dict]:
        """Fetch detailed TV show information"""
        await self.rate_limiter.wait_if_needed()
        
        url = f"{self.base_url}/shows/{show_id}"
        params = {'embed': 'episodes'}
        
        try:
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    return await response.json()
                elif response.status == 429:
                    await asyncio.sleep(10)  # TV Maze suggests 10 second backoff
                    return await self.fetch_show_details(show_id)
                else:
                    logger.error(f"Error fetching show {show_id}: {response.status}")
                    return None
        except Exception as e:
            logger.error(f"Exception fetching show {show_id}: {e}")
            return None
    
    async def search_shows(self, query: str) -> List[Dict]:
        """Search for TV shows"""
        await self.rate_limiter.wait_if_needed()
        
        url = f"{self.base_url}/search/shows"
        params = {'q': query}
        
        try:
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    results = await response.json()
                    return [item['show'] for item in results]
                else:
                    logger.error(f"Error searching shows: {response.status}")
                    return []
        except Exception as e:
            logger.error(f"Exception searching shows: {e}")
            return []

class TraktIntegration:
    """Trakt.tv API integration"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.trakt.tv"
        self.rate_limiter = RateLimiter(RateLimitConfig(
            requests_per_second=0.5,  # 2 seconds between requests
            requests_per_minute=20,
            requests_per_hour=1000,
            requests_per_day=5000
        ))
        self.session = None
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession(headers={
            'Content-Type': 'application/json',
            'trakt-api-version': '2',
            'trakt-api-key': self.api_key
        })
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def get_popular_movies(self, limit: int = 100) -> List[Dict]:
        """Get popular movies from Trakt"""
        await self.rate_limiter.wait_if_needed()
        
        url = f"{self.base_url}/movies/popular"
        params = {'limit': limit, 'extended': 'full'}
        
        try:
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    logger.error(f"Error fetching popular movies: {response.status}")
                    return []
        except Exception as e:
            logger.error(f"Exception fetching popular movies: {e}")
            return []
    
    async def get_popular_shows(self, limit: int = 100) -> List[Dict]:
        """Get popular TV shows from Trakt"""
        await self.rate_limiter.wait_if_needed()
        
        url = f"{self.base_url}/shows/popular"
        params = {'limit': limit, 'extended': 'full'}
        
        try:
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    logger.error(f"Error fetching popular shows: {response.status}")
                    return []
        except Exception as e:
            logger.error(f"Exception fetching popular shows: {e}")
            return []

class DatasetEnhancer:
    """Enhance existing datasets with API data"""
    
    def __init__(self, tmdb_api_key: str, trakt_api_key: Optional[str] = None):
        self.tmdb_api_key = tmdb_api_key
        self.trakt_api_key = trakt_api_key
        self.output_dir = Path("enhanced_datasets")
        self.output_dir.mkdir(exist_ok=True)
    
    async def enhance_movie_dataset(self, movies_df: pd.DataFrame) -> pd.DataFrame:
        """Enhance movie dataset with TMDB data"""
        enhanced_movies = []
        
        async with TMDBIntegration(self.tmdb_api_key) as tmdb:
            for idx, row in movies_df.iterrows():
                if idx % 100 == 0:
                    logger.info(f"Processing movie {idx}/{len(movies_df)}")
                
                # Try to get TMDB ID from existing data
                tmdb_id = row.get('tmdbId') or row.get('tmdb_id')
                
                if tmdb_id:
                    details = await tmdb.fetch_movie_details(int(tmdb_id))
                    if details:
                        enhanced_row = row.to_dict()
                        enhanced_row.update({
                            'tmdb_popularity': details.get('popularity'),
                            'tmdb_vote_average': details.get('vote_average'),
                            'tmdb_vote_count': details.get('vote_count'),
                            'runtime': details.get('runtime'),
                            'budget': details.get('budget'),
                            'revenue': details.get('revenue'),
                            'spoken_languages': json.dumps([lang['name'] for lang in details.get('spoken_languages', [])]),
                            'production_companies': json.dumps([comp['name'] for comp in details.get('production_companies', [])]),
                            'keywords': json.dumps([kw['name'] for kw in details.get('keywords', {}).get('keywords', [])]),
                            'director': self._extract_director(details.get('credits', {})),
                            'main_actors': json.dumps(self._extract_main_actors(details.get('credits', {})))
                        })
                        enhanced_movies.append(enhanced_row)
                        
                        # Small delay to be respectful
                        await asyncio.sleep(0.1)
        
        return pd.DataFrame(enhanced_movies)
    
    async def enhance_tv_dataset(self, tv_df: pd.DataFrame) -> pd.DataFrame:
        """Enhance TV dataset with multiple API sources"""
        enhanced_shows = []
        
        async with TMDBIntegration(self.tmdb_api_key) as tmdb, \
                   TVMazeIntegration() as tvmaze:
            
            for idx, row in tv_df.iterrows():
                if idx % 50 == 0:
                    logger.info(f"Processing TV show {idx}/{len(tv_df)}")
                
                enhanced_row = row.to_dict()
                
                # Try TMDB first
                tmdb_id = row.get('tmdb_id') or row.get('id')
                if tmdb_id:
                    tmdb_details = await tmdb.fetch_tv_details(int(tmdb_id))
                    if tmdb_details:
                        enhanced_row.update({
                            'tmdb_popularity': tmdb_details.get('popularity'),
                            'tmdb_vote_average': tmdb_details.get('vote_average'),
                            'tmdb_vote_count': tmdb_details.get('vote_count'),
                            'number_of_episodes': tmdb_details.get('number_of_episodes'),
                            'number_of_seasons': tmdb_details.get('number_of_seasons'),
                            'status': tmdb_details.get('status'),
                            'networks': json.dumps([net['name'] for net in tmdb_details.get('networks', [])]),
                            'created_by': json.dumps([creator['name'] for creator in tmdb_details.get('created_by', [])])
                        })
                
                # Try TV Maze for additional episode data
                show_name = row.get('name') or row.get('title')
                if show_name:
                    tvmaze_shows = await tvmaze.search_shows(show_name)
                    if tvmaze_shows:
                        show_details = tvmaze_shows[0]  # Take first match
                        enhanced_row.update({
                            'tvmaze_rating': show_details.get('rating', {}).get('average'),
                            'tvmaze_runtime': show_details.get('runtime'),
                            'tvmaze_premiered': show_details.get('premiered'),
                            'tvmaze_ended': show_details.get('ended'),
                            'tvmaze_network': show_details.get('network', {}).get('name') if show_details.get('network') else None
                        })
                
                enhanced_shows.append(enhanced_row)
                
                # Delay between requests
                await asyncio.sleep(0.2)
        
        return pd.DataFrame(enhanced_shows)
    
    def _extract_director(self, credits: Dict) -> Optional[str]:
        """Extract director name from credits"""
        crew = credits.get('crew', [])
        for person in crew:
            if person.get('job') == 'Director':
                return person.get('name')
        return None
    
    def _extract_main_actors(self, credits: Dict, limit: int = 5) -> List[str]:
        """Extract main actor names from credits"""
        cast = credits.get('cast', [])
        return [actor.get('name') for actor in cast[:limit] if actor.get('name')]
    
    async def save_enhanced_datasets(self):
        """Save enhanced datasets to files"""
        # Load existing datasets and enhance them
        datasets_to_enhance = [
            ('movies/cinesync/ml-32m/movies.csv', 'enhanced_movies_ml32m.csv'),
            ('tv/tmdb/TMDB_tv_dataset_v3.csv', 'enhanced_tv_tmdb.csv'),
            ('movies/tmdb-movies/movies_metadata.csv', 'enhanced_movies_tmdb.csv')
        ]
        
        for input_path, output_filename in datasets_to_enhance:
            if os.path.exists(input_path):
                logger.info(f"Enhancing dataset: {input_path}")
                df = pd.read_csv(input_path)
                
                if 'movie' in input_path.lower():
                    enhanced_df = await self.enhance_movie_dataset(df)
                else:
                    enhanced_df = await self.enhance_tv_dataset(df)
                
                output_path = self.output_dir / output_filename
                enhanced_df.to_csv(output_path, index=False)
                logger.info(f"Saved enhanced dataset: {output_path}")

async def main():
    """Main function to demonstrate API integrations"""
    # Load API keys from environment
    tmdb_api_key = os.getenv('TMDB_API_KEY')
    trakt_api_key = os.getenv('TRAKT_API_KEY')
    
    if not tmdb_api_key:
        logger.error("TMDB_API_KEY environment variable required")
        return
    
    # Create dataset enhancer
    enhancer = DatasetEnhancer(tmdb_api_key, trakt_api_key)
    
    # Example: Enhance datasets
    try:
        await enhancer.save_enhanced_datasets()
        logger.info("Dataset enhancement completed successfully")
    except Exception as e:
        logger.error(f"Error during dataset enhancement: {e}")

if __name__ == "__main__":
    asyncio.run(main())