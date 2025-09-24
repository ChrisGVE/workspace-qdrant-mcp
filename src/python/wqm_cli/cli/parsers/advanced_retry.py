"""
Advanced retry mechanisms with exponential backoff and circuit breaker patterns.

This module provides sophisticated retry logic for web crawling operations,
including exponential backoff with jitter, circuit breakers for failing domains,
and intelligent retry condition detection.
"""

import asyncio
import random
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Type, Union
from urllib.parse import urlparse

import aiohttp
from loguru import logger


class RetryReason(Enum):
    """Reasons for retrying a request."""
    NETWORK_ERROR = "network_error"
    TIMEOUT = "timeout"
    SERVER_ERROR = "server_error"
    RATE_LIMITED = "rate_limited"
    SSL_ERROR = "ssl_error"
    CONNECTION_ERROR = "connection_error"
    TEMPORARY_FAILURE = "temporary_failure"


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failures detected, blocking requests
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class RetryConfig:
    """Configuration for retry behavior."""

    # Basic retry settings
    max_retries: int = 3
    base_delay: float = 1.0  # Base delay in seconds
    max_delay: float = 60.0  # Maximum delay in seconds
    exponential_base: float = 2.0  # Exponential backoff multiplier
    jitter: bool = True  # Add random jitter to prevent thundering herd

    # Retry conditions
    retry_status_codes: Set[int] = field(default_factory=lambda: {
        429,  # Too Many Requests
        500,  # Internal Server Error
        502,  # Bad Gateway
        503,  # Service Unavailable
        504,  # Gateway Timeout
        520,  # Cloudflare Unknown Error
        521,  # Cloudflare Web Server Is Down
        522,  # Cloudflare Connection Timed Out
        523,  # Cloudflare Origin Is Unreachable
        524,  # Cloudflare A Timeout Occurred
    })

    retry_exceptions: Set[Type[Exception]] = field(default_factory=lambda: {
        aiohttp.ClientConnectorError,
        aiohttp.ClientTimeoutError,
        aiohttp.ClientOSError,
        aiohttp.ServerTimeoutError,
        asyncio.TimeoutError,
    })

    # Circuit breaker settings
    circuit_failure_threshold: int = 5  # Failures before opening circuit
    circuit_recovery_timeout: float = 30.0  # Time before testing recovery
    circuit_success_threshold: int = 3  # Successes needed to close circuit


@dataclass
class RetryAttempt:
    """Information about a retry attempt."""
    attempt: int
    delay: float
    reason: RetryReason
    error: Optional[Exception] = None
    status_code: Optional[int] = None
    timestamp: float = field(default_factory=time.time)


@dataclass
class CircuitBreakerStats:
    """Statistics for circuit breaker."""
    state: CircuitState = CircuitState.CLOSED
    failure_count: int = 0
    success_count: int = 0
    last_failure_time: float = 0.0
    last_success_time: float = 0.0
    total_requests: int = 0
    blocked_requests: int = 0


class CircuitBreaker:
    """Circuit breaker for failing domains."""

    def __init__(self, config: RetryConfig):
        self.config = config
        self.domain_stats: Dict[str, CircuitBreakerStats] = {}

    def get_domain_stats(self, domain: str) -> CircuitBreakerStats:
        """Get circuit breaker stats for a domain."""
        if domain not in self.domain_stats:
            self.domain_stats[domain] = CircuitBreakerStats()
        return self.domain_stats[domain]

    def can_attempt_request(self, url: str) -> bool:
        """Check if request can be attempted."""
        domain = urlparse(url).netloc
        stats = self.get_domain_stats(domain)
        current_time = time.time()

        if stats.state == CircuitState.CLOSED:
            return True
        elif stats.state == CircuitState.OPEN:
            # Check if we should move to half-open
            if current_time - stats.last_failure_time >= self.config.circuit_recovery_timeout:
                stats.state = CircuitState.HALF_OPEN
                stats.success_count = 0
                logger.info(f"Circuit breaker half-open for domain: {domain}")
                return True
            else:
                stats.blocked_requests += 1
                return False
        elif stats.state == CircuitState.HALF_OPEN:
            return True

        return False

    def record_success(self, url: str) -> None:
        """Record successful request."""
        domain = urlparse(url).netloc
        stats = self.get_domain_stats(domain)
        stats.total_requests += 1
        stats.last_success_time = time.time()

        if stats.state == CircuitState.HALF_OPEN:
            stats.success_count += 1
            if stats.success_count >= self.config.circuit_success_threshold:
                stats.state = CircuitState.CLOSED
                stats.failure_count = 0
                logger.info(f"Circuit breaker closed for domain: {domain}")
        elif stats.state == CircuitState.CLOSED:
            # Reset failure count on success
            stats.failure_count = 0

    def record_failure(self, url: str, error: Exception, status_code: Optional[int] = None) -> None:
        """Record failed request."""
        domain = urlparse(url).netloc
        stats = self.get_domain_stats(domain)
        stats.total_requests += 1
        stats.failure_count += 1
        stats.last_failure_time = time.time()

        # Only open circuit for certain types of failures
        should_count_failure = (
            isinstance(error, (aiohttp.ClientConnectorError, aiohttp.ClientOSError)) or
            (status_code and status_code >= 500)
        )

        if should_count_failure and stats.state in [CircuitState.CLOSED, CircuitState.HALF_OPEN]:
            if stats.failure_count >= self.config.circuit_failure_threshold:
                stats.state = CircuitState.OPEN
                stats.success_count = 0
                logger.warning(f"Circuit breaker opened for domain: {domain} after {stats.failure_count} failures")
            elif stats.state == CircuitState.HALF_OPEN:
                # Go back to open if half-open test fails
                stats.state = CircuitState.OPEN
                logger.warning(f"Circuit breaker reopened for domain: {domain}")


class AdvancedRetryHandler:
    """Advanced retry handler with exponential backoff and circuit breaker."""

    def __init__(self, config: Optional[RetryConfig] = None):
        self.config = config or RetryConfig()
        self.circuit_breaker = CircuitBreaker(self.config)
        self.retry_history: Dict[str, List[RetryAttempt]] = {}

    def should_retry(self, error: Exception, status_code: Optional[int] = None, attempt: int = 0) -> Tuple[bool, RetryReason]:
        """Determine if request should be retried."""
        if attempt >= self.config.max_retries:
            return False, RetryReason.NETWORK_ERROR

        # Check status code
        if status_code:
            if status_code in self.config.retry_status_codes:
                if status_code == 429:
                    return True, RetryReason.RATE_LIMITED
                elif status_code >= 500:
                    return True, RetryReason.SERVER_ERROR
                else:
                    return True, RetryReason.TEMPORARY_FAILURE

        # Check exception type
        error_type = type(error)
        if error_type in self.config.retry_exceptions:
            if isinstance(error, (aiohttp.ClientTimeoutError, asyncio.TimeoutError)):
                return True, RetryReason.TIMEOUT
            elif isinstance(error, aiohttp.ClientConnectorError):
                return True, RetryReason.CONNECTION_ERROR
            elif isinstance(error, aiohttp.ClientSSLError):
                return True, RetryReason.SSL_ERROR
            else:
                return True, RetryReason.NETWORK_ERROR

        return False, RetryReason.NETWORK_ERROR

    def calculate_delay(self, attempt: int, reason: RetryReason) -> float:
        """Calculate delay for retry attempt."""
        # Base exponential backoff
        delay = self.config.base_delay * (self.config.exponential_base ** attempt)

        # Apply max delay limit
        delay = min(delay, self.config.max_delay)

        # Special handling for rate limiting
        if reason == RetryReason.RATE_LIMITED:
            delay = max(delay, 10.0)  # Minimum 10 seconds for rate limiting

        # Add jitter to prevent thundering herd
        if self.config.jitter:
            jitter_amount = delay * 0.1  # 10% jitter
            delay += random.uniform(-jitter_amount, jitter_amount)

        return max(0.1, delay)  # Minimum 0.1 second delay

    async def execute_with_retry(
        self,
        func: Callable,
        url: str,
        *args,
        **kwargs
    ) -> Any:
        """Execute function with retry logic."""
        # Check circuit breaker
        if not self.circuit_breaker.can_attempt_request(url):
            raise Exception(f"Circuit breaker open for domain: {urlparse(url).netloc}")

        last_error = None
        last_status_code = None

        for attempt in range(self.config.max_retries + 1):
            try:
                result = await func(*args, **kwargs)

                # Record success
                self.circuit_breaker.record_success(url)

                # Log retry success if this was a retry
                if attempt > 0:
                    logger.info(f"Request succeeded after {attempt} retries: {url}")

                return result

            except Exception as error:
                last_error = error
                last_status_code = None

                # Extract status code if available
                if hasattr(error, 'status') and error.status:
                    last_status_code = error.status
                elif hasattr(error, 'response') and error.response:
                    last_status_code = error.response.status

                # Determine if we should retry
                should_retry, reason = self.should_retry(error, last_status_code, attempt)

                if not should_retry:
                    # Record failure and break
                    self.circuit_breaker.record_failure(url, error, last_status_code)
                    break

                # Calculate delay
                delay = self.calculate_delay(attempt, reason)

                # Record retry attempt
                retry_attempt = RetryAttempt(
                    attempt=attempt + 1,
                    delay=delay,
                    reason=reason,
                    error=error,
                    status_code=last_status_code
                )

                if url not in self.retry_history:
                    self.retry_history[url] = []
                self.retry_history[url].append(retry_attempt)

                logger.warning(f"Request failed (attempt {attempt + 1}/{self.config.max_retries + 1}): {url} - {error}. Retrying in {delay:.2f}s")

                # Wait before retry
                await asyncio.sleep(delay)

        # All retries exhausted, record failure
        if last_error:
            self.circuit_breaker.record_failure(url, last_error, last_status_code)

        raise last_error or Exception("Max retries exceeded")

    def get_retry_stats(self, url: Optional[str] = None) -> Dict[str, Any]:
        """Get retry statistics."""
        stats = {
            'circuit_breaker_stats': {},
            'retry_history': {}
        }

        # Circuit breaker stats
        for domain, cb_stats in self.circuit_breaker.domain_stats.items():
            stats['circuit_breaker_stats'][domain] = {
                'state': cb_stats.state.value,
                'failure_count': cb_stats.failure_count,
                'success_count': cb_stats.success_count,
                'total_requests': cb_stats.total_requests,
                'blocked_requests': cb_stats.blocked_requests,
                'last_failure_time': cb_stats.last_failure_time,
                'last_success_time': cb_stats.last_success_time
            }

        # Retry history
        if url:
            if url in self.retry_history:
                stats['retry_history'][url] = [
                    {
                        'attempt': attempt.attempt,
                        'delay': attempt.delay,
                        'reason': attempt.reason.value,
                        'status_code': attempt.status_code,
                        'timestamp': attempt.timestamp,
                        'error': str(attempt.error) if attempt.error else None
                    }
                    for attempt in self.retry_history[url]
                ]
        else:
            for url, attempts in self.retry_history.items():
                stats['retry_history'][url] = len(attempts)

        return stats

    def reset_stats(self, url: Optional[str] = None) -> None:
        """Reset retry statistics."""
        if url:
            domain = urlparse(url).netloc
            if domain in self.circuit_breaker.domain_stats:
                del self.circuit_breaker.domain_stats[domain]
            if url in self.retry_history:
                del self.retry_history[url]
        else:
            self.circuit_breaker.domain_stats.clear()
            self.retry_history.clear()