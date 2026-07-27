import logging
import time
from typing import Any, Dict, List, Optional
from urllib.parse import urljoin

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from config.settings import settings
from .rate_limiter import TokenBucketRateLimiter
from .cache_manager import CacheManager

logger = logging.getLogger(__name__)


class SportradarClient:
    """
    Production-grade Sportradar Soccer API v4 client.
    Implements token-bucket rate limiting, exponential backoff via urllib3 Retry,
    and local filesystem caching.
    """

    def __init__(self):
        settings.validate()
        self.api_key = settings.SPORTRADAR_API_KEY
        self.access_level = settings.SPORTRADAR_ACCESS_LEVEL
        self.language = settings.SPORTRADAR_LANGUAGE
        self.base_url = f"{settings.SPORTRADAR_BASE_URL}/{self.access_level}/v4/{self.language}/"

        self.rate_limiter = TokenBucketRateLimiter(
            rate=settings.MAX_QPS,
            capacity=settings.RATE_LIMIT_BURST,
        )
        self.cache = CacheManager(settings.CACHE_DIR, settings.CACHE_TTL_SECONDS)

        self.session = requests.Session()
        retry_strategy = Retry(
            total=settings.MAX_RETRIES,
            backoff_factor=settings.BACKOFF_FACTOR,
            status_forcelist=settings.RETRY_STATUS_CODES,
            allowed_methods=["GET"],
            raise_on_status=False,
        )
        adapter = HTTPAdapter(max_retries=retry_strategy, pool_connections=10, pool_maxsize=20)
        self.session.mount("https://", adapter)
        self.session.headers.update(
            {
                "Accept": "application/json",
                "x-api-key": self.api_key,
                "User-Agent": "FootballPredictionEngine/1.0",
            }
        )

    def _build_url(self, endpoint: str) -> str:
        return urljoin(self.base_url, endpoint.lstrip("/"))

    def _request(self, endpoint: str, params: Optional[Dict] = None, use_cache: bool = True) -> Dict[str, Any]:
        url = self._build_url(endpoint)
        cache_key = f"{url}?{sorted((params or {}).items())}"

        if use_cache:
            cached = self.cache.get(cache_key)
            if cached is not None:
                logger.debug("Cache hit: %s", endpoint)
                return cached

        self.rate_limiter.acquire()
        logger.info("GET %s params=%s", endpoint, params)
        response = self.session.get(url, params=params or {}, timeout=30)

        if response.status_code == 429:
            retry_after = float(response.headers.get("Retry-After", 2.0))
            logger.warning("Rate limited. Sleeping %.1fs", retry_after)
            time.sleep(retry_after)
            self.rate_limiter.acquire()
            response = self.session.get(url, params=params or {}, timeout=30)

        response.raise_for_status()
        data = response.json()

        if use_cache:
            self.cache.set(cache_key, data)
        return data

    # ------------------------------------------------------------------
    # Core endpoints
    # ------------------------------------------------------------------

    def get_competitions(self) -> List[Dict]:
        data = self._request("competitions.json")
        return data.get("competitions", [])

    def get_competition_seasons(self, competition_id: str) -> List[Dict]:
        data = self._request(f"competitions/{competition_id}/seasons.json")
        return data.get("seasons", [])

    def get_season_schedule(self, season_id: str) -> List[Dict]:
        data = self._request(f"seasons/{season_id}/schedules.json")
        return data.get("schedules", [])

    def get_season_standings(self, season_id: str) -> Dict:
        return self._request(f"seasons/{season_id}/standings.json")

    def get_season_competitors(self, season_id: str) -> List[Dict]:
        data = self._request(f"seasons/{season_id}/competitors.json")
        return data.get("season_competitors", [])

    def get_seasonal_competitor_statistics(self, season_id: str, competitor_id: str) -> Dict:
        return self._request(f"seasons/{season_id}/competitors/{competitor_id}/statistics.json")

    def get_sport_event_summary(self, sport_event_id: str) -> Dict:
        return self._request(f"sport_events/{sport_event_id}/summary.json")

    def get_sport_event_timeline(self, sport_event_id: str) -> Dict:
        return self._request(f"sport_events/{sport_event_id}/timeline.json")

    def get_competitor_profile(self, competitor_id: str) -> Dict:
        return self._request(f"competitors/{competitor_id}/profile.json")

    def get_competitor_schedules(self, competitor_id: str) -> List[Dict]:
        data = self._request(f"competitors/{competitor_id}/schedules.json")
        return data.get("schedules", [])

    def get_daily_schedules(self, date_str: str) -> List[Dict]:
        """date_str format: YYYY-MM-DD"""
        data = self._request(f"schedules/{date_str}/schedules.json")
        return data.get("schedules", [])

    def get_probabilities(self, sport_event_id: str) -> Dict:
        """Requires Probabilities package. Gracefully returns empty dict on 404."""
        try:
            return self._request(f"sport_events/{sport_event_id}/sport_event_probabilities.json")
        except requests.HTTPError as exc:
            if exc.response is not None and exc.response.status_code == 404:
                logger.debug("Probabilities not available for %s", sport_event_id)
                return {}
            raise

    def get_missing_players(self, season_id: str) -> Dict:
        try:
            return self._request(f"seasons/{season_id}/missing_players.json")
        except requests.HTTPError as exc:
            if exc.response is not None and exc.response.status_code in (404, 403):
                return {}
            raise
