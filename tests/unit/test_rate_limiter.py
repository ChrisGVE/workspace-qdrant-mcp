"""
Unit tests for rate limiting and abuse detection.

Tests token bucket and sliding window algorithms, per-client tracking,
and thread safety.
"""

import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from common.security.rate_limiter import (
    RateLimitConfig,
    RateLimiter,
    SlidingWindow,
    TokenBucket,
    configure_rate_limiter,
    get_rate_limiter,
)


class TestTokenBucket:
    """Test token bucket algorithm."""

    def test_initial_burst_allowed(self):
        """Test that initial burst up to bucket size is allowed."""
        bucket = TokenBucket(rate_per_minute=60, burst_size=10)

        # Should allow burst_size requests immediately
        for _ in range(10):
            assert bucket.consume() is True

        # 11th request should be denied
        assert bucket.consume() is False

    def test_token_refill(self):
        """Test that tokens refill over time."""
        bucket = TokenBucket(rate_per_minute=60, burst_size=5)

        # Consume all tokens
        for _ in range(5):
            bucket.consume()

        # Wait for 1 second (should refill 1 token at 60/min = 1/sec)
        time.sleep(1.1)

        # Should allow one more request
        assert bucket.consume() is True

    def test_burst_refill_capped(self):
        """Test that tokens don't exceed burst size."""
        bucket = TokenBucket(rate_per_minute=60, burst_size=5)

        # Wait for long time
        time.sleep(10)

        # Should still only allow burst_size requests
        for _ in range(5):
            assert bucket.consume() is True

        assert bucket.consume() is False

    def test_get_wait_time(self):
        """Test wait time calculation."""
        bucket = TokenBucket(rate_per_minute=60, burst_size=1)

        # Consume token
        bucket.consume()

        # Wait time should be approximately 1 second
        wait_time = bucket.get_wait_time()
        assert 0.9 <= wait_time <= 1.1


class TestSlidingWindow:
    """Test sliding window algorithm."""

    def test_allows_requests_within_limit(self):
        """Test that requests within limit are allowed."""
        window = SlidingWindow(requests_per_minute=10, window_seconds=60)

        # Should allow 10 requests
        for _ in range(10):
            assert window.is_allowed() is True

        # 11th request should be denied
        assert window.is_allowed() is False

    def test_window_slides_over_time(self):
        """Test that window slides and old requests expire."""
        window = SlidingWindow(requests_per_minute=2, window_seconds=1)

        # Make 2 requests (fills window)
        assert window.is_allowed() is True
        assert window.is_allowed() is True

        # 3rd request denied
        assert window.is_allowed() is False

        # Wait for window to slide
        time.sleep(1.1)

        # Should allow more requests now
        assert window.is_allowed() is True

    def test_get_oldest_request(self):
        """Test retrieving oldest request timestamp."""
        window = SlidingWindow(requests_per_minute=10, window_seconds=60)

        # No requests yet
        assert window.get_oldest_request() is None

        # Make a request
        window.is_allowed()
        oldest = window.get_oldest_request()

        assert oldest is not None
        assert isinstance(oldest, float)


class TestRateLimiterBasic:
    """Test basic rate limiter functionality."""

    def test_rate_limiter_creation(self):
        """Test creating rate limiter with config."""
        config = RateLimitConfig(
            requests_per_minute=100,
            burst_size=20,
            enabled=True
        )
        limiter = RateLimiter(config)

        assert limiter.config.requests_per_minute == 100
        assert limiter.config.burst_size == 20
        assert limiter.config.enabled is True

    def test_disabled_limiter_allows_all(self):
        """Test that disabled limiter allows all requests."""
        config = RateLimitConfig(enabled=False)
        limiter = RateLimiter(config)

        # Should allow unlimited requests
        for _ in range(1000):
            assert limiter.is_allowed("client1") is True

    def test_per_client_tracking(self):
        """Test that limits are tracked per client."""
        config = RateLimitConfig(requests_per_minute=60, burst_size=2)
        limiter = RateLimiter(config)

        # Client 1 uses their burst
        assert limiter.is_allowed("client1") is True
        assert limiter.is_allowed("client1") is True
        assert limiter.is_allowed("client1") is False

        # Client 2 should still have their burst available
        assert limiter.is_allowed("client2") is True
        assert limiter.is_allowed("client2") is True

    def test_per_endpoint_limits(self):
        """Test different rate limits per endpoint."""
        config = RateLimitConfig(
            requests_per_minute=60,
            burst_size=10,
            endpoint_limits={
                "search": (30, 5),  # Stricter limit for search
                "store": (100, 20),  # More generous for store
            }
        )
        limiter = RateLimiter(config)

        # Search endpoint has stricter limit (burst=5)
        for _ in range(5):
            assert limiter.is_allowed("client1", endpoint="search") is True
        assert limiter.is_allowed("client1", endpoint="search") is False

        # Store endpoint has more generous limit (burst=20)
        for _ in range(20):
            assert limiter.is_allowed("client1", endpoint="store") is True
        assert limiter.is_allowed("client1", endpoint="store") is False

    def test_sliding_window_mode(self):
        """Test using sliding window algorithm."""
        config = RateLimitConfig(requests_per_minute=60, burst_size=10)
        limiter = RateLimiter(config)

        # Use sliding window instead of token bucket
        for _ in range(60):
            assert limiter.is_allowed("client1", use_sliding_window=True) is True

        # 61st request should be denied
        assert limiter.is_allowed("client1", use_sliding_window=True) is False


class TestRateLimiterAdvanced:
    """Test advanced rate limiter features."""

    def test_reset_client(self):
        """Test resetting client rate limit."""
        config = RateLimitConfig(requests_per_minute=60, burst_size=2)
        limiter = RateLimiter(config)

        # Use up client's burst
        limiter.is_allowed("client1")
        limiter.is_allowed("client1")
        assert limiter.is_allowed("client1") is False

        # Reset client
        limiter.reset_client("client1")

        # Should allow requests again
        assert limiter.is_allowed("client1") is True

    def test_reset_specific_endpoint(self):
        """Test resetting only specific endpoint for client."""
        config = RateLimitConfig(requests_per_minute=60, burst_size=2)
        limiter = RateLimiter(config)

        # Use up limits on search endpoint
        limiter.is_allowed("client1", endpoint="search")
        limiter.is_allowed("client1", endpoint="search")
        assert limiter.is_allowed("client1", endpoint="search") is False

        # Use up limits on store endpoint
        limiter.is_allowed("client1", endpoint="store")
        limiter.is_allowed("client1", endpoint="store")
        assert limiter.is_allowed("client1", endpoint="store") is False

        # Reset only search endpoint
        limiter.reset_client("client1", endpoint="search")

        # Search should be available, store still limited
        assert limiter.is_allowed("client1", endpoint="search") is True
        assert limiter.is_allowed("client1", endpoint="store") is False

    def test_get_wait_time(self):
        """Test getting estimated wait time."""
        config = RateLimitConfig(requests_per_minute=60, burst_size=1)
        limiter = RateLimiter(config)

        # Use up burst
        limiter.is_allowed("client1")

        # Should have non-zero wait time
        wait_time = limiter.get_wait_time("client1")
        assert wait_time > 0

    def test_get_stats(self):
        """Test getting rate limiter statistics."""
        config = RateLimitConfig(requests_per_minute=60, burst_size=10)
        limiter = RateLimiter(config)

        # Make some requests
        limiter.is_allowed("client1")
        limiter.is_allowed("client2")
        limiter.is_allowed("client3", endpoint="search")

        stats = limiter.get_stats()

        assert "active_buckets" in stats
        assert "active_windows" in stats
        assert "total_tracked_clients" in stats
        assert stats["total_tracked_clients"] == 3


class TestConcurrency:
    """Test thread safety and concurrent access."""

    def test_concurrent_requests_same_client(self):
        """Test concurrent requests from same client are handled correctly."""
        config = RateLimitConfig(requests_per_minute=60, burst_size=100)
        limiter = RateLimiter(config)

        allowed_count = 0
        denied_count = 0

        def make_request():
            nonlocal allowed_count, denied_count
            if limiter.is_allowed("client1"):
                allowed_count += 1
            else:
                denied_count += 1

        # Make 150 concurrent requests (burst is 100)
        with ThreadPoolExecutor(max_workers=20) as executor:
            futures = [executor.submit(make_request) for _ in range(150)]
            for future in futures:
                future.result()

        # Should allow burst_size (100) and deny the rest
        assert allowed_count == 100
        assert denied_count == 50

    def test_concurrent_requests_different_clients(self):
        """Test concurrent requests from different clients."""
        config = RateLimitConfig(requests_per_minute=60, burst_size=5)
        limiter = RateLimiter(config)

        results = {}

        def make_requests(client_id):
            results[client_id] = 0
            for _ in range(10):
                if limiter.is_allowed(client_id):
                    results[client_id] += 1

        # 10 clients making 10 requests each concurrently
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [
                executor.submit(make_requests, f"client{i}")
                for i in range(10)
            ]
            for future in futures:
                future.result()

        # Each client should be limited independently (5 allowed per client)
        for client_id, count in results.items():
            assert count == 5, f"{client_id} allowed {count} requests, expected 5"


class TestGlobalLimiter:
    """Test global rate limiter singleton."""

    def test_get_rate_limiter_default(self):
        """Test getting global limiter with defaults."""
        limiter = get_rate_limiter()
        assert isinstance(limiter, RateLimiter)
        assert limiter.config.enabled is True

    def test_configure_rate_limiter(self):
        """Test configuring global limiter."""
        config = RateLimitConfig(requests_per_minute=1000, burst_size=50)
        configure_rate_limiter(config)

        limiter = get_rate_limiter()
        assert limiter.config.requests_per_minute == 1000
        assert limiter.config.burst_size == 50


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_zero_rate_limit(self):
        """Test behavior with zero rate limit (should block all)."""
        config = RateLimitConfig(requests_per_minute=0, burst_size=0)
        limiter = RateLimiter(config)

        # Should deny all requests (no tokens ever)
        assert limiter.is_allowed("client1") is False

    def test_very_high_rate_limit(self):
        """Test with very high rate limits."""
        config = RateLimitConfig(requests_per_minute=1000000, burst_size=10000)
        limiter = RateLimiter(config)

        # Should allow many requests quickly
        for _ in range(10000):
            assert limiter.is_allowed("client1") is True

    def test_fractional_token_consumption(self):
        """Test that fractional tokens work correctly."""
        bucket = TokenBucket(rate_per_minute=60, burst_size=10)

        # Consume all tokens
        for _ in range(10):
            bucket.consume()

        # Wait for half a second (should refill 0.5 tokens)
        time.sleep(0.5)

        # Should not have enough for a full token yet
        assert bucket.consume() is False

        # Wait a bit more
        time.sleep(0.6)

        # Now should have enough
        assert bucket.consume() is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
