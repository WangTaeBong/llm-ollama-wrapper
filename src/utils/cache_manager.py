import time
from functools import wraps
from typing import Any, Dict, Callable, Optional


class CacheManager:
    """Class responsible for cache management"""

    def __init__(self, max_size: int = 1000, default_ttl: int = 3600):
        self.cache: Dict[str, Any] = {}
        self.timestamps: Dict[str, float] = {}
        self.ttl: Dict[str, int] = {}
        self.max_size = max_size
        self.default_ttl = default_ttl  # Default TTL: 1 hour

    def get(self, key: str) -> Optional[Any]:
        """Retrieves the cached value for the given key."""
        if key not in self.cache:
            return None
        # Check TTL
        if key in self.ttl and self.timestamps[key] + self.ttl[key] < time.time():
            self._remove(key)
            return None
        return self.cache[key]

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Stores a value in the cache."""
        # Check cache size limit
        if len(self.cache) >= self.max_size and key not in self.cache:
            # Remove the oldest item
            oldest_key = min(self.timestamps.items(), key=lambda x: x[1])[0]
            self._remove(oldest_key)
        self.cache[key] = value
        self.timestamps[key] = time.time()
        if ttl is not None:
            self.ttl[key] = ttl
        elif key not in self.ttl:
            self.ttl[key] = self.default_ttl

    def _remove(self, key: str) -> None:
        """Removes an item from the cache."""
        if key in self.cache:
            del self.cache[key]
        if key in self.timestamps:
            del self.timestamps[key]
        if key in self.ttl:
            del self.ttl[key]

    def clear(self) -> None:
        """Clears all cache."""
        self.cache.clear()
        self.timestamps.clear()
        self.ttl.clear()

    def remove_expired(self) -> None:
        """Removes expired cache items."""
        current_time = time.time()
        expired_keys = [
            key for key in self.ttl
            if key in self.timestamps and self.timestamps[key] + self.ttl[key] < current_time
        ]
        for key in expired_keys:
            self._remove(key)


def cached(cache_manager: CacheManager, ttl: Optional[int] = None):
    """Decorator for caching function results"""

    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Create cache key (function name + arguments)
            key = f"{func.__name__}:{str(args)}"
