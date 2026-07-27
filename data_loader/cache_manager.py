import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Optional
import logging

logger = logging.getLogger(__name__)


class CacheManager:
    """Filesystem-based cache with TTL support for Sportradar responses."""

    def __init__(self, cache_dir: str, ttl_seconds: int = 86400):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.ttl_seconds = ttl_seconds

    def _key_to_path(self, key: str) -> Path:
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
        return self.cache_dir / f"{digest}.json"

    def get(self, key: str) -> Optional[Any]:
        path = self._key_to_path(key)
        if not path.exists():
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            if time.time() - payload["timestamp"] > self.ttl_seconds:
                path.unlink(missing_ok=True)
                return None
            return payload["data"]
        except (json.JSONDecodeError, KeyError, OSError) as exc:
            logger.warning("Cache read failed for %s: %s", key, exc)
            path.unlink(missing_ok=True)
            return None

    def set(self, key: str, data: Any) -> None:
        path = self._key_to_path(key)
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump({"timestamp": time.time(), "data": data}, f, ensure_ascii=False)
        except OSError as exc:
            logger.error("Cache write failed for %s: %s", key, exc)

    def clear(self) -> None:
        for p in self.cache_dir.glob("*.json"):
            p.unlink(missing_ok=True)
