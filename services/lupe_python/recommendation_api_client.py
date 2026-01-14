"""
Recommendation API Client for Lupe Discord Bot

Connects to the mommy-milk-me-v2 recommendation service for:
- Getting AI-powered recommendations
- Recording user feedback for model training
- Fetching user preferences and history
"""

import aiohttp
import asyncio
import logging
from typing import Optional, List, Dict, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class RecommendationAPIConfig:
    """Configuration for the recommendation API"""
    base_url: str = "http://192.168.1.64:5001"
    timeout: int = 30
    enabled: bool = True


class RecommendationAPIClient:
    """
    Async client for the CineSync recommendation API.
    Provides methods for getting recommendations and recording feedback.
    """

    def __init__(self, config: Optional[RecommendationAPIConfig] = None):
        self.config = config or RecommendationAPIConfig()
        self._session: Optional[aiohttp.ClientSession] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session"""
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=self.config.timeout)
            self._session = aiohttp.ClientSession(timeout=timeout)
        return self._session

    async def close(self):
        """Close the session"""
        if self._session and not self._session.closed:
            await self._session.close()

    async def health_check(self) -> Dict[str, Any]:
        """Check if the recommendation service is healthy"""
        if not self.config.enabled:
            return {"success": False, "error": "API disabled"}

        try:
            session = await self._get_session()
            async with session.get(f"{self.config.base_url}/health") as resp:
                if resp.status == 200:
                    return await resp.json()
                return {"success": False, "error": f"HTTP {resp.status}"}
        except Exception as e:
            logger.warning(f"Health check failed: {e}")
            return {"success": False, "error": str(e)}

    async def get_recommendations(
        self,
        user_id: Optional[int] = None,
        content_type: str = "movie",
        limit: int = 10,
        genre: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get AI-powered recommendations.

        Args:
            user_id: Discord user ID for personalization
            content_type: 'movie', 'tv', or 'both'
            limit: Number of recommendations
            genre: Optional genre filter

        Returns:
            Dictionary with recommendations
        """
        if not self.config.enabled:
            return {"success": False, "error": "API disabled"}

        try:
            session = await self._get_session()
            params = {
                "content_type": content_type,
                "limit": limit
            }
            if user_id:
                params["user_id"] = user_id
            if genre:
                params["genre"] = genre

            async with session.get(
                f"{self.config.base_url}/recommend",
                params=params
            ) as resp:
                if resp.status == 200:
                    return await resp.json()
                return {"success": False, "error": f"HTTP {resp.status}"}
        except Exception as e:
            logger.error(f"Get recommendations failed: {e}")
            return {"success": False, "error": str(e)}

    async def get_personalized_recommendations(
        self,
        user_id: int,
        content_type: str = "movie",
        limit: int = 10
    ) -> Dict[str, Any]:
        """Get personalized recommendations for a user"""
        if not self.config.enabled:
            return {"success": False, "error": "API disabled"}

        try:
            session = await self._get_session()
            params = {
                "user_id": user_id,
                "content_type": content_type,
                "limit": limit
            }

            async with session.get(
                f"{self.config.base_url}/recommendations/personalized",
                params=params
            ) as resp:
                if resp.status == 200:
                    return await resp.json()
                return {"success": False, "error": f"HTTP {resp.status}"}
        except Exception as e:
            logger.error(f"Get personalized recommendations failed: {e}")
            return {"success": False, "error": str(e)}

    async def record_feedback(
        self,
        user_id: int,
        item_id: int,
        rating: float,
        content_type: str = "movie"
    ) -> Dict[str, Any]:
        """
        Record explicit user feedback (rating) for model training.

        Args:
            user_id: Discord user ID
            item_id: TMDB movie/show ID
            rating: Rating from 1-5
            content_type: 'movie' or 'tv'

        Returns:
            Success/failure response
        """
        if not self.config.enabled:
            return {"success": False, "error": "API disabled"}

        try:
            session = await self._get_session()
            payload = {
                "user_id": user_id,
                "item_id": item_id,
                "rating": rating,
                "content_type": content_type
            }

            async with session.post(
                f"{self.config.base_url}/feedback",
                json=payload
            ) as resp:
                if resp.status == 200:
                    return await resp.json()
                return {"success": False, "error": f"HTTP {resp.status}"}
        except Exception as e:
            logger.error(f"Record feedback failed: {e}")
            return {"success": False, "error": str(e)}

    async def record_interaction(
        self,
        user_id: int,
        item_id: int,
        interaction_type: str = "view",
        content_type: str = "movie"
    ) -> Dict[str, Any]:
        """
        Record implicit user interaction for model training.

        Args:
            user_id: Discord user ID
            item_id: TMDB movie/show ID
            interaction_type: 'view', 'click', 'watch', 'download', 'add_to_list', 'share'
            content_type: 'movie' or 'tv'

        Returns:
            Success/failure response
        """
        if not self.config.enabled:
            return {"success": False, "error": "API disabled"}

        try:
            session = await self._get_session()
            payload = {
                "user_id": user_id,
                "item_id": item_id,
                "interaction_type": interaction_type,
                "content_type": content_type
            }

            async with session.post(
                f"{self.config.base_url}/interaction",
                json=payload
            ) as resp:
                if resp.status == 200:
                    return await resp.json()
                return {"success": False, "error": f"HTTP {resp.status}"}
        except Exception as e:
            logger.error(f"Record interaction failed: {e}")
            return {"success": False, "error": str(e)}

    async def get_user_preferences(self, user_id: int) -> Dict[str, Any]:
        """Get user's genre preferences and taste profile"""
        if not self.config.enabled:
            return {"success": False, "error": "API disabled"}

        try:
            session = await self._get_session()
            async with session.get(
                f"{self.config.base_url}/user/{user_id}/preferences"
            ) as resp:
                if resp.status == 200:
                    return await resp.json()
                return {"success": False, "error": f"HTTP {resp.status}"}
        except Exception as e:
            logger.error(f"Get user preferences failed: {e}")
            return {"success": False, "error": str(e)}

    async def get_user_history(self, user_id: int, limit: int = 50) -> Dict[str, Any]:
        """Get user's watch history"""
        if not self.config.enabled:
            return {"success": False, "error": "API disabled"}

        try:
            session = await self._get_session()
            async with session.get(
                f"{self.config.base_url}/user/{user_id}/history",
                params={"limit": limit}
            ) as resp:
                if resp.status == 200:
                    return await resp.json()
                return {"success": False, "error": f"HTTP {resp.status}"}
        except Exception as e:
            logger.error(f"Get user history failed: {e}")
            return {"success": False, "error": str(e)}

    async def get_similar_items(
        self,
        item_id: int,
        content_type: str = "movie",
        limit: int = 10
    ) -> Dict[str, Any]:
        """Get similar items using AI embeddings"""
        if not self.config.enabled:
            return {"success": False, "error": "API disabled"}

        try:
            session = await self._get_session()
            async with session.get(
                f"{self.config.base_url}/similar/{item_id}",
                params={"content_type": content_type, "limit": limit}
            ) as resp:
                if resp.status == 200:
                    return await resp.json()
                return {"success": False, "error": f"HTTP {resp.status}"}
        except Exception as e:
            logger.error(f"Get similar items failed: {e}")
            return {"success": False, "error": str(e)}

    async def get_models_status(self) -> Dict[str, Any]:
        """Get detailed AI models status"""
        if not self.config.enabled:
            return {"success": False, "error": "API disabled"}

        try:
            session = await self._get_session()
            async with session.get(
                f"{self.config.base_url}/models/status"
            ) as resp:
                if resp.status == 200:
                    return await resp.json()
                return {"success": False, "error": f"HTTP {resp.status}"}
        except Exception as e:
            logger.error(f"Get models status failed: {e}")
            return {"success": False, "error": str(e)}


# Global client instance
_api_client: Optional[RecommendationAPIClient] = None


def get_recommendation_api_client(
    base_url: str = "http://192.168.1.64:5001"
) -> RecommendationAPIClient:
    """Get or create the global API client"""
    global _api_client
    if _api_client is None:
        config = RecommendationAPIConfig(base_url=base_url)
        _api_client = RecommendationAPIClient(config)
    return _api_client


async def init_recommendation_api(base_url: str = "http://192.168.1.64:5001") -> bool:
    """Initialize the recommendation API client and check health"""
    client = get_recommendation_api_client(base_url)
    health = await client.health_check()

    if health.get("success"):
        logger.info(f"Connected to recommendation API at {base_url}")
        return True
    else:
        logger.warning(f"Recommendation API not available: {health.get('error')}")
        return False
