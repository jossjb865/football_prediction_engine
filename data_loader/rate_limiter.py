import time
import threading
from collections import deque
from typing import Optional


class TokenBucketRateLimiter:
    """Thread-safe token-bucket rate limiter that strictly respects QPS limits."""

    def __init__(self, rate: float, capacity: int = 3):
        """
        Args:
            rate: Tokens (requests) per second.
            capacity: Maximum burst size.
        """
        self.rate = float(rate)
        self.capacity = int(capacity)
        self.tokens = float(capacity)
        self.last_refill = time.monotonic()
        self.lock = threading.Lock()
        self._wait_times = deque(maxlen=100)

    def _refill(self) -> None:
        now = time.monotonic()
        elapsed = now - self.last_refill
        self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)
        self.last_refill = now

    def acquire(self, tokens: int = 1, timeout: Optional[float] = None) -> bool:
        """
        Block until the requested number of tokens is available.
        Returns True if acquired, False if timeout exceeded.
        """
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            with self.lock:
                self._refill()
                if self.tokens >= tokens:
                    self.tokens -= tokens
                    return True
                # Calculate sleep time needed
                needed = tokens - self.tokens
                sleep_for = needed / self.rate
            if deadline is not None and time.monotonic() + sleep_for > deadline:
                return False
            time.sleep(min(sleep_for, 0.05))  # fine-grained wake-ups
            self._wait_times.append(sleep_for)

    def get_stats(self) -> dict:
        with self.lock:
            return {
                "tokens_available": self.tokens,
                "rate": self.rate,
                "capacity": self.capacity,
                "avg_wait_last_100": sum(self._wait_times) / len(self._wait_times) if self._wait_times else 0.0,
            }
