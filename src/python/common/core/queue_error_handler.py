"""
Queue Error Handling and Retry System

Provides comprehensive error classification, retry strategies with exponential
backoff, dead letter queue management, and circuit breaker pattern for queue operations.

Features:
    - Error categorization (transient vs permanent)
    - Exponential backoff retry strategy
    - Per-error-type retry limits
    - Dead letter queue for permanently failed items
    - Circuit breaker pattern for repeated failures
    - Error metrics collection and monitoring

Example:
    ```python
    from workspace_qdrant_mcp.core.queue_error_handler import (
        ErrorHandler, ErrorCategory, ErrorType
    )

    handler = ErrorHandler(queue_client)

    # Handle error with automatic categorization
    should_retry = await handler.handle_error(
        file_path="/path/to/file.txt",
        error=exception,
        context={"operation": "ingest"}
    )
    ```
"""

import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

from loguru import logger


class ErrorCategory(Enum):
    """Error category classification."""
    TRANSIENT = "transient"  # Temporary errors, retry possible
    PERMANENT = "permanent"  # Permanent errors, no retry
    RATE_LIMIT = "rate_limit"  # Rate limiting, retry with backoff
    RESOURCE = "resource"  # Resource exhaustion, retry with delay


class ErrorType(Enum):
    """Specific error types with categorization."""

    # Transient errors (retry with backoff)
    NETWORK_TIMEOUT = ("network_timeout", ErrorCategory.TRANSIENT, 5)
    CONNECTION_REFUSED = ("connection_refused", ErrorCategory.TRANSIENT, 5)
    TEMPORARY_FAILURE = ("temporary_failure", ErrorCategory.TRANSIENT, 3)
    DATABASE_LOCKED = ("database_locked", ErrorCategory.TRANSIENT, 10)

    # Rate limit errors (retry with longer backoff)
    RATE_LIMIT_EXCEEDED = ("rate_limit_exceeded", ErrorCategory.RATE_LIMIT, 10)
    TOO_MANY_REQUESTS = ("too_many_requests", ErrorCategory.RATE_LIMIT, 8)

    # Resource errors (retry with delay)
    OUT_OF_MEMORY = ("out_of_memory", ErrorCategory.RESOURCE, 3)
    DISK_FULL = ("disk_full", ErrorCategory.RESOURCE, 3)
    QUOTA_EXCEEDED = ("quota_exceeded", ErrorCategory.RESOURCE, 5)

    # Permanent errors (no retry)
    FILE_NOT_FOUND = ("file_not_found", ErrorCategory.PERMANENT, 0)
    INVALID_FORMAT = ("invalid_format", ErrorCategory.PERMANENT, 0)
    PERMISSION_DENIED = ("permission_denied", ErrorCategory.PERMANENT, 0)
    INVALID_CONFIGURATION = ("invalid_configuration", ErrorCategory.PERMANENT, 0)
    VALIDATION_ERROR = ("validation_error", ErrorCategory.PERMANENT, 0)
    MALFORMED_DATA = ("malformed_data", ErrorCategory.PERMANENT, 0)

    def __init__(self, error_name: str, category: ErrorCategory, max_retries: int):
        self.error_name = error_name
        self.category = category
        self.max_retries = max_retries


@dataclass
class RetryConfig:
    """Retry strategy configuration."""

    # Base retry delay in seconds
    base_delay: float = 1.0

    # Maximum delay between retries
    max_delay: float = 300.0  # 5 minutes

    # Exponential backoff multiplier
    backoff_multiplier: float = 2.0

    # Jitter factor (0-1) to randomize delays
    jitter_factor: float = 0.1

    # Default max retries (when error type not specified)
    default_max_retries: int = 3


@dataclass
class CircuitBreakerConfig:
    """Circuit breaker pattern configuration."""

    # Number of failures before opening circuit
    failure_threshold: int = 5

    # Time window for counting failures (seconds)
    failure_window: int = 60

    # Time to wait before trying half-open state (seconds)
    recovery_timeout: int = 300

    # Success count needed in half-open state to close circuit
    success_threshold: int = 2


@dataclass
class CircuitBreakerState:
    """Circuit breaker state tracking."""

    state: str = "closed"  # closed, open, half-open
    failure_count: int = 0
    success_count: int = 0
    last_failure_time: float | None = None
    opened_at: float | None = None
    failures: list = field(default_factory=list)


@dataclass
class ErrorMetrics:
    """Error metrics for monitoring."""

    total_errors: int = 0
    transient_errors: int = 0
    permanent_errors: int = 0
    rate_limit_errors: int = 0
    resource_errors: int = 0
    retry_count: int = 0
    circuit_breaker_opens: int = 0
    dead_letter_items: int = 0
    successful_retries: int = 0
    failed_retries: int = 0


class ErrorHandler:
    """
    Comprehensive error handler for queue operations.

    Handles error classification, retry strategies, circuit breaker pattern,
    and dead letter queue management.
    """

    def __init__(
        self,
        queue_client,
        retry_config: RetryConfig | None = None,
        circuit_breaker_config: CircuitBreakerConfig | None = None
    ):
        """
        Initialize error handler.

        Args:
            queue_client: SQLiteQueueClient instance
            retry_config: Retry strategy configuration
            circuit_breaker_config: Circuit breaker configuration
        """
        self.queue_client = queue_client
        self.retry_config = retry_config or RetryConfig()
        self.circuit_breaker_config = circuit_breaker_config or CircuitBreakerConfig()

        # Circuit breaker state per collection
        self.circuit_breakers: dict[str, CircuitBreakerState] = {}

        # Error metrics
        self.metrics = ErrorMetrics()

        # Error type mapping from exception messages
        self.error_patterns = self._build_error_patterns()

    def _build_error_patterns(self) -> dict[str, ErrorType]:
        """Build mapping from error message patterns to error types."""
        return {
            "timeout": ErrorType.NETWORK_TIMEOUT,
            "timed out": ErrorType.NETWORK_TIMEOUT,
            "connection refused": ErrorType.CONNECTION_REFUSED,
            "database is locked": ErrorType.DATABASE_LOCKED,
            "rate limit": ErrorType.RATE_LIMIT_EXCEEDED,
            "too many requests": ErrorType.TOO_MANY_REQUESTS,
            "429": ErrorType.TOO_MANY_REQUESTS,
            "out of memory": ErrorType.OUT_OF_MEMORY,
            "disk full": ErrorType.DISK_FULL,
            "no space left": ErrorType.DISK_FULL,
            "quota exceeded": ErrorType.QUOTA_EXCEEDED,
            "file not found": ErrorType.FILE_NOT_FOUND,
            "no such file": ErrorType.FILE_NOT_FOUND,
            "invalid format": ErrorType.INVALID_FORMAT,
            "permission denied": ErrorType.PERMISSION_DENIED,
            "access denied": ErrorType.PERMISSION_DENIED,
            "validation error": ErrorType.VALIDATION_ERROR,
            "malformed": ErrorType.MALFORMED_DATA,
        }

    def classify_error(
        self,
        error: Exception,
        context: dict[str, Any] | None = None
    ) -> ErrorType:
        """
        Classify error by analyzing exception message and context.

        Args:
            error: Exception to classify
            context: Additional context for classification

        Returns:
            ErrorType classification
        """
        error_message = str(error).lower()

        # Check error patterns
        for pattern, error_type in self.error_patterns.items():
            if pattern in error_message:
                logger.debug(f"Classified error as {error_type.error_name}: {pattern}")
                return error_type

        # Default to temporary failure for unknown errors
        logger.warning(f"Unknown error type, classifying as temporary: {error}")
        return ErrorType.TEMPORARY_FAILURE

    def calculate_retry_delay(
        self,
        retry_count: int,
        error_type: ErrorType
    ) -> float:
        """
        Calculate retry delay with exponential backoff and jitter.

        Args:
            retry_count: Current retry attempt number (0-indexed)
            error_type: Type of error

        Returns:
            Delay in seconds before next retry
        """
        # Base delay with exponential backoff
        delay = self.retry_config.base_delay * (
            self.retry_config.backoff_multiplier ** retry_count
        )

        # Apply different multipliers based on error category
        if error_type.category == ErrorCategory.RATE_LIMIT:
            delay *= 3.0  # Longer delays for rate limits
        elif error_type.category == ErrorCategory.RESOURCE:
            delay *= 2.0  # Moderate delays for resource issues

        # Cap at max delay
        delay = min(delay, self.retry_config.max_delay)

        # Add jitter to prevent thundering herd
        import random
        jitter = delay * self.retry_config.jitter_factor * random.random()
        delay += jitter

        logger.debug(
            f"Calculated retry delay for {error_type.error_name}: "
            f"{delay:.2f}s (attempt {retry_count + 1})"
        )

        return delay

    def should_retry(
        self,
        error_type: ErrorType,
        retry_count: int
    ) -> bool:
        """
        Determine if error should be retried.

        Args:
            error_type: Type of error
            retry_count: Current retry count

        Returns:
            True if should retry, False otherwise
        """
        # Permanent errors never retry
        if error_type.category == ErrorCategory.PERMANENT:
            return False

        # Check if max retries exceeded
        max_retries = error_type.max_retries or self.retry_config.default_max_retries

        return retry_count < max_retries

    def get_circuit_breaker(self, collection: str) -> CircuitBreakerState:
        """Get or create circuit breaker state for collection."""
        if collection not in self.circuit_breakers:
            self.circuit_breakers[collection] = CircuitBreakerState()
        return self.circuit_breakers[collection]

    def check_circuit_breaker(self, collection: str) -> tuple[bool, str]:
        """
        Check circuit breaker state.

        Args:
            collection: Collection name

        Returns:
            Tuple of (can_proceed, reason)
        """
        breaker = self.get_circuit_breaker(collection)
        current_time = time.time()

        # If circuit is open, check if recovery timeout has elapsed
        if breaker.state == "open":
            if breaker.opened_at and (
                current_time - breaker.opened_at
            ) > self.circuit_breaker_config.recovery_timeout:
                # Try half-open state
                breaker.state = "half-open"
                breaker.success_count = 0
                logger.info(f"Circuit breaker {collection} entering half-open state")
                return True, "half-open"
            else:
                return False, "circuit_open"

        # Circuit closed or half-open, can proceed
        return True, breaker.state

    def record_circuit_breaker_failure(self, collection: str):
        """Record failure in circuit breaker."""
        breaker = self.get_circuit_breaker(collection)
        current_time = time.time()

        breaker.failure_count += 1
        breaker.last_failure_time = current_time
        breaker.failures.append(current_time)

        # Remove old failures outside time window
        window_start = current_time - self.circuit_breaker_config.failure_window
        breaker.failures = [f for f in breaker.failures if f > window_start]

        # Check if should open circuit
        if breaker.state == "closed":
            if len(breaker.failures) >= self.circuit_breaker_config.failure_threshold:
                breaker.state = "open"
                breaker.opened_at = current_time
                self.metrics.circuit_breaker_opens += 1
                logger.warning(
                    f"Circuit breaker opened for {collection} after "
                    f"{len(breaker.failures)} failures"
                )

        elif breaker.state == "half-open":
            # Failed in half-open, back to open
            breaker.state = "open"
            breaker.opened_at = current_time
            breaker.success_count = 0
            logger.warning(f"Circuit breaker {collection} reopened after half-open failure")

    def record_circuit_breaker_success(self, collection: str):
        """Record success in circuit breaker."""
        breaker = self.get_circuit_breaker(collection)

        if breaker.state == "half-open":
            breaker.success_count += 1

            if breaker.success_count >= self.circuit_breaker_config.success_threshold:
                breaker.state = "closed"
                breaker.failure_count = 0
                breaker.failures = []
                logger.info(f"Circuit breaker {collection} closed after successful recovery")

    async def handle_error(
        self,
        file_path: str,
        error: Exception,
        collection: str,
        context: dict[str, Any] | None = None
    ) -> tuple[bool, ErrorType]:
        """
        Handle error with classification, retry logic, and circuit breaker.

        Args:
            file_path: File path that caused error
            error: Exception that occurred
            collection: Collection name
            context: Additional error context

        Returns:
            Tuple of (should_retry, error_type)
        """
        # Classify error
        error_type = self.classify_error(error, context)

        # Update metrics
        self.metrics.total_errors += 1
        if error_type.category == ErrorCategory.TRANSIENT:
            self.metrics.transient_errors += 1
        elif error_type.category == ErrorCategory.PERMANENT:
            self.metrics.permanent_errors += 1
        elif error_type.category == ErrorCategory.RATE_LIMIT:
            self.metrics.rate_limit_errors += 1
        elif error_type.category == ErrorCategory.RESOURCE:
            self.metrics.resource_errors += 1

        # Check circuit breaker
        can_proceed, breaker_reason = self.check_circuit_breaker(collection)

        if not can_proceed:
            logger.warning(
                f"Circuit breaker open for {collection}, moving to dead letter queue"
            )
            await self.move_to_dead_letter_queue(
                file_path, error_type, str(error), {"circuit_breaker": "open"}
            )
            return False, error_type

        # Get current retry count
        # This would come from the queue item
        retry_count = 0  # Placeholder, should be fetched from queue

        # Check if should retry
        if not self.should_retry(error_type, retry_count):
            logger.info(
                f"Error {error_type.error_name} for {file_path} is not retryable, "
                f"moving to dead letter queue"
            )
            await self.move_to_dead_letter_queue(
                file_path, error_type, str(error), context or {}
            )
            self.record_circuit_breaker_failure(collection)
            return False, error_type

        # Record error and determine retry delay
        max_retries = error_type.max_retries or self.retry_config.default_max_retries

        should_retry_result, error_message_id = await self.queue_client.mark_error(
            file_path=file_path,
            error_type=error_type.error_name,
            error_message=str(error),
            error_details=context,
            max_retries=max_retries
        )

        if should_retry_result:
            self.metrics.retry_count += 1

            # Calculate and apply backoff delay
            delay = self.calculate_retry_delay(retry_count, error_type)
            logger.info(
                f"Retrying {file_path} after {delay:.2f}s "
                f"(attempt {retry_count + 1}/{max_retries})"
            )

            return True, error_type
        else:
            # Max retries exceeded, move to dead letter queue
            logger.warning(
                f"Max retries exceeded for {file_path}, moving to dead letter queue"
            )
            await self.move_to_dead_letter_queue(
                file_path, error_type, str(error), context or {}
            )
            self.record_circuit_breaker_failure(collection)
            return False, error_type

    async def move_to_dead_letter_queue(
        self,
        file_path: str,
        error_type: ErrorType,
        error_message: str,
        context: dict[str, Any]
    ):
        """
        Move permanently failed item to dead letter queue.

        Args:
            file_path: File path
            error_type: Error type
            error_message: Error message
            context: Error context
        """
        # In our design, dead letter queue is represented by removing from
        # ingestion_queue and keeping only the error message
        # For a full implementation, you might want a separate dead_letter_queue table

        error_details = {
            **context,
            "dead_letter_reason": error_type.error_name,
            "moved_to_dlq_at": datetime.utcnow().isoformat(),
        }

        # Record in messages table
        await self.queue_client.mark_error(
            file_path=file_path,
            error_type=f"DEAD_LETTER_{error_type.error_name}",
            error_message=error_message,
            error_details=error_details,
            max_retries=0  # Force removal from queue
        )

        self.metrics.dead_letter_items += 1

        logger.error(
            f"Moved {file_path} to dead letter queue: {error_type.error_name}"
        )

    def get_metrics(self) -> dict[str, Any]:
        """
        Get error handling metrics.

        Returns:
            Dictionary of metrics
        """
        return {
            "total_errors": self.metrics.total_errors,
            "transient_errors": self.metrics.transient_errors,
            "permanent_errors": self.metrics.permanent_errors,
            "rate_limit_errors": self.metrics.rate_limit_errors,
            "resource_errors": self.metrics.resource_errors,
            "retry_count": self.metrics.retry_count,
            "circuit_breaker_opens": self.metrics.circuit_breaker_opens,
            "dead_letter_items": self.metrics.dead_letter_items,
            "successful_retries": self.metrics.successful_retries,
            "failed_retries": self.metrics.failed_retries,
            "active_circuit_breakers": {
                name: state.state
                for name, state in self.circuit_breakers.items()
                if state.state != "closed"
            }
        }

    def reset_metrics(self):
        """Reset error metrics."""
        self.metrics = ErrorMetrics()
        logger.info("Error metrics reset")
