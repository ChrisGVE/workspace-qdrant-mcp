"""
Rate Limiting and Abuse Detection for MCP Server.

Provides configurable rate limiting using token bucket and sliding window
algorithms to prevent abuse and ensure fair resource allocation.

Features:
- Per-client rate limiting based on IP or client ID
- Token bucket algorithm for burst handling
- Sliding window for precise rate control
- Configurable limits per endpoint
- Automatic cleanup of expired tracking data
- Thread-safe implementation

Usage:
    from common.security.rate_limiter import RateLimiter, RateLimitConfig

    # Configure rate limiter
    config = RateLimitConfig(
        requests_per_minute=60,
        burst_size=10,
        enabled=True
    )
    limiter = RateLimiter(config)

    # Check if request is allowed
    client_id = "client_123"
    if limiter.is_allowed(client_id):
        # Process request
        pass
    else:
        # Return 429 Too Many Requests
        pass
"""

import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple


@dataclass
class RateLimitConfig:
    """Rate limiting configuration."""

    # Basic limits
    requests_per_minute: int = 60  # Max requests per minute per client
    burst_size: int = 10  # Max burst size for token bucket
    enabled: bool = True  # Enable/disable rate limiting globally

    # Advanced configuration
    cleanup_interval_seconds: int = 300  # Cleanup old tracking data every 5 minutes
    track_by_ip: bool = True  # Track by IP address
    track_by_client_id: bool = True  # Track by client identifier

    # Per-endpoint limits (optional)
    endpoint_limits: Dict[str, Tuple[int, int]] = field(default_factory=dict)
    # Format: {"endpoint_name": (requests_per_minute, burst_size)}

    def get_endpoint_limit(self, endpoint: str) -> Tuple[int, int]:
        """Get rate limit for specific endpoint.

        Returns:
            Tuple of (requests_per_minute, burst_size)
        """
        return self.endpoint_limits.get(
            endpoint,
            (self.requests_per_minute, self.burst_size)
        )


class TokenBucket:
    """Token bucket rate limiter implementation.

    Allows burst traffic while maintaining average rate over time.
    """

    def __init__(self, rate_per_minute: int, burst_size: int):
        """Initialize token bucket.

        Args:
            rate_per_minute: Token refill rate (requests per minute)
            burst_size: Maximum bucket capacity (burst allowance)
        """
        self.rate_per_second = rate_per_minute / 60.0
        self.burst_size = burst_size
        self.tokens = float(burst_size)
        self.last_update = time.time()
        self.lock = threading.Lock()

    def consume(self, tokens: int = 1) -> bool:
        """Attempt to consume tokens from bucket.

        Args:
            tokens: Number of tokens to consume

        Returns:
            True if tokens were available and consumed, False otherwise
        """
        with self.lock:
            now = time.time()
            elapsed = now - self.last_update

            # Refill bucket based on elapsed time
            self.tokens = min(
                self.burst_size,
                self.tokens + (elapsed * self.rate_per_second)
            )
            self.last_update = now

            # Attempt to consume tokens
            if self.tokens >= tokens:
                self.tokens -= tokens
                return True

            return False

    def get_wait_time(self) -> float:
        """Get estimated wait time until next token is available.

        Returns:
            Wait time in seconds
        """
        with self.lock:
            if self.tokens >= 1.0:
                return 0.0

            tokens_needed = 1.0 - self.tokens
            return tokens_needed / self.rate_per_second


class SlidingWindow:
    """Sliding window rate limiter implementation.

    Provides precise rate limiting by tracking request timestamps.
    More accurate than token bucket but uses more memory.
    """

    def __init__(self, requests_per_minute: int, window_seconds: int = 60):
        """Initialize sliding window.

        Args:
            requests_per_minute: Maximum requests allowed in window
            window_seconds: Window duration in seconds
        """
        self.max_requests = requests_per_minute
        self.window_seconds = window_seconds
        self.requests = []  # List of (timestamp, ) tuples
        self.lock = threading.Lock()

    def is_allowed(self) -> bool:
        """Check if request is allowed within window.

        Returns:
            True if request is allowed, False if rate limit exceeded
        """
        with self.lock:
            now = time.time()
            cutoff = now - self.window_seconds

            # Remove requests outside the window
            self.requests = [req for req in self.requests if req > cutoff]

            # Check if under limit
            if len(self.requests) < self.max_requests:
                self.requests.append(now)
                return True

            return False

    def get_oldest_request(self) -> Optional[float]:
        """Get timestamp of oldest request in window.

        Returns:
            Timestamp of oldest request or None if window is empty
        """
        with self.lock:
            return self.requests[0] if self.requests else None


class RateLimiter:
    """Main rate limiter with per-client tracking.

    Supports both token bucket (for burst) and sliding window (for precision).
    """

    def __init__(self, config: RateLimitConfig):
        """Initialize rate limiter.

        Args:
            config: Rate limiting configuration
        """
        self.config = config
        self.buckets: Dict[str, TokenBucket] = {}
        self.windows: Dict[str, SlidingWindow] = {}
        self.last_cleanup = time.time()
        self.lock = threading.Lock()

        # Start cleanup thread if enabled
        if config.enabled:
            self._start_cleanup_thread()

    def is_allowed(
        self,
        client_id: str,
        endpoint: Optional[str] = None,
        use_sliding_window: bool = False
    ) -> bool:
        """Check if request from client is allowed.

        Args:
            client_id: Unique client identifier (IP, user ID, etc.)
            endpoint: Optional endpoint name for per-endpoint limits
            use_sliding_window: Use sliding window instead of token bucket

        Returns:
            True if request is allowed, False if rate limit exceeded
        """
        if not self.config.enabled:
            return True

        # Get rate limit for endpoint
        rate_per_minute, burst_size = self.config.get_endpoint_limit(endpoint or "default")

        # Build tracking key
        key = f"{client_id}:{endpoint or 'default'}"

        if use_sliding_window:
            return self._check_sliding_window(key, rate_per_minute)
        else:
            return self._check_token_bucket(key, rate_per_minute, burst_size)

    def _check_token_bucket(self, key: str, rate: int, burst: int) -> bool:
        """Check rate limit using token bucket algorithm."""
        with self.lock:
            if key not in self.buckets:
                self.buckets[key] = TokenBucket(rate, burst)

        return self.buckets[key].consume()

    def _check_sliding_window(self, key: str, rate: int) -> bool:
        """Check rate limit using sliding window algorithm."""
        with self.lock:
            if key not in self.windows:
                self.windows[key] = SlidingWindow(rate)

        return self.windows[key].is_allowed()

    def get_wait_time(self, client_id: str, endpoint: Optional[str] = None) -> float:
        """Get estimated wait time for client.

        Args:
            client_id: Unique client identifier
            endpoint: Optional endpoint name

        Returns:
            Estimated wait time in seconds until request would be allowed
        """
        key = f"{client_id}:{endpoint or 'default'}"

        with self.lock:
            if key in self.buckets:
                return self.buckets[key].get_wait_time()

        return 0.0

    def reset_client(self, client_id: str, endpoint: Optional[str] = None):
        """Reset rate limit tracking for a client.

        Args:
            client_id: Client identifier to reset
            endpoint: Optional specific endpoint to reset (None = reset all)
        """
        with self.lock:
            if endpoint:
                key = f"{client_id}:{endpoint}"
                self.buckets.pop(key, None)
                self.windows.pop(key, None)
            else:
                # Reset all endpoints for this client
                keys_to_remove = [
                    k for k in self.buckets.keys()
                    if k.startswith(f"{client_id}:")
                ]
                for key in keys_to_remove:
                    self.buckets.pop(key, None)
                    self.windows.pop(key, None)

    def _cleanup_old_data(self):
        """Remove old tracking data to prevent memory leaks."""
        with self.lock:
            # Remove buckets/windows for clients that haven't made requests recently
            now = time.time()
            cutoff = now - (self.config.cleanup_interval_seconds * 2)

            # For token buckets, check last_update time
            old_bucket_keys = [
                key for key, bucket in self.buckets.items()
                if bucket.last_update < cutoff
            ]
            for key in old_bucket_keys:
                del self.buckets[key]

            # For sliding windows, check if window is empty
            old_window_keys = [
                key for key, window in self.windows.items()
                if not window.requests or window.get_oldest_request() < cutoff
            ]
            for key in old_window_keys:
                del self.windows[key]

            self.last_cleanup = now

    def _start_cleanup_thread(self):
        """Start background thread for periodic cleanup."""
        def cleanup_worker():
            while self.config.enabled:
                time.sleep(self.config.cleanup_interval_seconds)
                self._cleanup_old_data()

        thread = threading.Thread(target=cleanup_worker, daemon=True)
        thread.start()

    def get_stats(self) -> Dict[str, int]:
        """Get rate limiter statistics.

        Returns:
            Dictionary with stats (active_clients, total_buckets, total_windows)
        """
        with self.lock:
            return {
                "active_buckets": len(self.buckets),
                "active_windows": len(self.windows),
                "total_tracked_clients": len(set(
                    k.split(":")[0] for k in list(self.buckets.keys()) + list(self.windows.keys())
                ))
            }


# Global rate limiter instance
_global_limiter: Optional[RateLimiter] = None


def get_rate_limiter(config: Optional[RateLimitConfig] = None) -> RateLimiter:
    """Get or create global rate limiter instance.

    Args:
        config: Optional configuration (only used on first call)

    Returns:
        Global rate limiter instance
    """
    global _global_limiter

    if _global_limiter is None:
        if config is None:
            config = RateLimitConfig()  # Use defaults
        _global_limiter = RateLimiter(config)

    return _global_limiter


def configure_rate_limiter(config: RateLimitConfig):
    """Configure global rate limiter.

    Args:
        config: Rate limiting configuration
    """
    global _global_limiter
    _global_limiter = RateLimiter(config)
