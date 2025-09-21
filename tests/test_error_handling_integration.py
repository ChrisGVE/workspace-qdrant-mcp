"""
Integration tests for comprehensive error handling system.

This module tests the integration between Python structured error handling
and logging systems, circuit breaker patterns, and recovery strategies.
"""

import asyncio
import json
import logging
import tempfile
import time
from pathlib import Path
from typing import Dict, Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import structlog

from common.core.error_handling import (
    WorkspaceError,
    ConfigurationError,
    NetworkError,
    DatabaseError,
    TimeoutError,
    CircuitBreakerOpenError,
    ErrorSeverity,
    ErrorCategory,
    ErrorRecoveryStrategy,
    ErrorRecovery,
    CircuitBreakerState,
    error_context,
    with_error_handling,
    safe_shutdown,
    get_error_stats,
    reset_error_stats,
)

# Configure logging for tests
logging.basicConfig(level=logging.DEBUG)


class TestWorkspaceError:
    """Test comprehensive WorkspaceError functionality."""

    def test_error_creation_with_context(self):
        """Test creating errors with structured context."""
        error = NetworkError(
            "Connection failed",
            url="https://example.com",
            attempt=2,
            max_attempts=3,
        )
        
        assert error.category == ErrorCategory.NETWORK
        assert error.severity == ErrorSeverity.MEDIUM
        assert error.retryable is True
        assert error.context["url"] == "https://example.com"
        assert error.context["attempt"] == 2
        assert error.context["max_attempts"] == 3

    def test_error_to_dict(self):
        """Test error serialization to dictionary."""
        error = DatabaseError(
            "Query timeout",
            operation="select_documents",
        )
        
        error_dict = error.to_dict()
        
        assert error_dict["message"] == "Query timeout"
        assert error_dict["category"] == "database"
        assert error_dict["severity"] == "medium"
        assert error_dict["retryable"] is True
        assert error_dict["context"]["operation"] == "select_documents"
        assert "timestamp" in error_dict

    def test_configuration_error(self):
        """Test configuration error with field context."""
        error = ConfigurationError(
            "Invalid URL format",
            field="qdrant_url",
        )
        
        assert error.category == ErrorCategory.CONFIGURATION
        assert error.severity == ErrorSeverity.HIGH
        assert error.retryable is False
        assert error.context["field"] == "qdrant_url"

    def test_timeout_error(self):
        """Test timeout error with operation context."""
        error = TimeoutError(
            "Operation timed out",
            operation="document_processing",
            duration_ms=30000,
        )
        
        assert error.category == ErrorCategory.TIMEOUT
        assert error.severity == ErrorSeverity.MEDIUM
        assert error.retryable is True
        assert error.context["operation"] == "document_processing"
        assert error.context["duration_ms"] == 30000


class TestErrorRecoveryStrategy:
    """Test error recovery strategy configurations."""

    def test_network_strategy(self):
        """Test network-specific recovery strategy."""
        strategy = ErrorRecoveryStrategy.network_strategy()
        
        assert strategy.max_retries == 5
        assert strategy.base_delay_ms == 500
        assert strategy.max_delay_ms == 10000
        assert strategy.exponential_backoff is True
        assert strategy.circuit_breaker_threshold == 3

    def test_database_strategy(self):
        """Test database-specific recovery strategy."""
        strategy = ErrorRecoveryStrategy.database_strategy()
        
        assert strategy.max_retries == 3
        assert strategy.base_delay_ms == 1000
        assert strategy.circuit_breaker_threshold == 5

    def test_file_strategy(self):
        """Test file operation recovery strategy."""
        strategy = ErrorRecoveryStrategy.file_strategy()
        
        assert strategy.max_retries == 2
        assert strategy.base_delay_ms == 50
        assert strategy.max_delay_ms == 1000
        assert strategy.exponential_backoff is False
        assert strategy.circuit_breaker_threshold is None

    def test_delay_calculation(self):
        """Test delay calculation with exponential backoff."""
        strategy = ErrorRecoveryStrategy.network_strategy()
        
        # Test exponential backoff
        assert strategy.calculate_delay_ms(1) == 500
        assert strategy.calculate_delay_ms(2) == 1000
        assert strategy.calculate_delay_ms(3) == 2000
        assert strategy.calculate_delay_ms(10) == 10000  # Max delay cap


class TestCircuitBreakerState:
    """Test circuit breaker state management."""

    def test_circuit_breaker_closed_state(self):
        """Test circuit breaker in closed state."""
        cb = CircuitBreakerState("test_service")
        
        assert cb.state == "closed"
        assert cb.should_allow_request() is True

    def test_circuit_breaker_failure_threshold(self):
        """Test circuit breaker opening after failures."""
        cb = CircuitBreakerState("test_service", failure_threshold=2)
        
        # First failure
        cb.record_failure()
        assert cb.state == "closed"
        assert cb.failure_count == 1
        
        # Second failure - should open circuit
        cb.record_failure()
        assert cb.state == "open"
        assert cb.failure_count == 2
        assert cb.should_allow_request() is False

    def test_circuit_breaker_success_reset(self):
        """Test circuit breaker reset on success."""
        cb = CircuitBreakerState("test_service")
        
        # Add failures
        cb.record_failure()
        cb.record_failure()
        
        # Success should reset
        cb.record_success()
        assert cb.state == "closed"
        assert cb.failure_count == 0

    def test_circuit_breaker_half_open_transition(self):
        """Test circuit breaker transitioning to half-open."""
        cb = CircuitBreakerState("test_service", failure_threshold=1, reset_timeout_ms=100)
        
        # Open the circuit
        cb.record_failure()
        assert cb.state == "open"
        
        # Wait for timeout (simulate)
        time.sleep(0.2)  # 200ms > 100ms timeout
        
        # Should allow request and transition to half-open
        assert cb.should_allow_request() is True
        assert cb.state == "half-open"

    def test_circuit_breaker_status(self):
        """Test circuit breaker status information."""
        cb = CircuitBreakerState("test_service", failure_threshold=3)
        
        status = cb.get_status()
        
        assert status["name"] == "test_service"
        assert status["state"] == "closed"
        assert status["failure_count"] == 0
        assert status["failure_threshold"] == 3
        assert status["last_failure_time"] is None
        assert status["last_success_time"] is None


class TestErrorRecovery:
    """Test error recovery with retry logic."""

    @pytest.fixture
    def error_recovery(self):
        """Create error recovery instance for testing."""
        reset_error_stats()  # Reset global stats
        return ErrorRecovery()

    @pytest.mark.asyncio
    async def test_successful_operation_no_retry(self, error_recovery):
        """Test successful operation without retry."""
        async def successful_operation():
            return "success"

        result = await error_recovery.execute_with_retry(
            successful_operation,
            "test_operation",
            ErrorRecoveryStrategy.file_strategy(),
        )
        
        assert result == "success"

    @pytest.mark.asyncio
    async def test_retryable_error_with_recovery(self, error_recovery):
        """Test retryable error with eventual recovery."""
        call_count = 0

        async def failing_then_succeeding_operation():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise NetworkError("Temporary failure", attempt=call_count, max_attempts=3)
            return "success"

        strategy = ErrorRecoveryStrategy(max_retries=5, base_delay_ms=1, exponential_backoff=False)
        
        result = await error_recovery.execute_with_retry(
            failing_then_succeeding_operation,
            "test_operation",
            strategy,
        )
        
        assert result == "success"
        assert call_count == 3  # Failed twice, succeeded on third attempt

    @pytest.mark.asyncio
    async def test_non_retryable_error_immediate_failure(self, error_recovery):
        """Test non-retryable error fails immediately."""
        async def non_retryable_failing_operation():
            raise ConfigurationError("Invalid configuration")

        with pytest.raises(ConfigurationError):
            await error_recovery.execute_with_retry(
                non_retryable_failing_operation,
                "test_operation",
                ErrorRecoveryStrategy.network_strategy(),
            )

    @pytest.mark.asyncio
    async def test_max_retries_exceeded(self, error_recovery):
        """Test maximum retries exceeded."""
        call_count = 0

        async def always_failing_operation():
            nonlocal call_count
            call_count += 1
            raise NetworkError("Persistent failure", attempt=call_count, max_attempts=2)

        strategy = ErrorRecoveryStrategy(max_retries=2, base_delay_ms=1, exponential_backoff=False)
        
        with pytest.raises(WorkspaceError) as exc_info:
            await error_recovery.execute_with_retry(
                always_failing_operation,
                "test_operation",
                strategy,
            )
        
        assert "All 2 retry attempts failed" in str(exc_info.value)
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_timeout_handling(self, error_recovery):
        """Test timeout handling in error recovery."""
        async def slow_operation():
            await asyncio.sleep(0.1)  # 100ms
            return "success"

        strategy = ErrorRecoveryStrategy(timeout_ms=50, max_retries=1)  # 50ms timeout
        
        with pytest.raises(TimeoutError):
            await error_recovery.execute_with_retry(
                slow_operation,
                "test_operation",
                strategy,
            )

    @pytest.mark.asyncio
    async def test_circuit_breaker_integration(self, error_recovery):
        """Test circuit breaker integration with error recovery."""
        call_count = 0

        async def failing_operation():
            nonlocal call_count
            call_count += 1
            raise NetworkError("Service unavailable")

        strategy = ErrorRecoveryStrategy(
            max_retries=10,
            circuit_breaker_threshold=2,
            base_delay_ms=1,
            exponential_backoff=False,
        )
        
        # First attempt should fail with NetworkError
        with pytest.raises(NetworkError):
            await error_recovery.execute_with_retry(
                failing_operation,
                "circuit_breaker_test",
                strategy,
            )
        
        # Second attempt should also fail with NetworkError
        with pytest.raises(NetworkError):
            await error_recovery.execute_with_retry(
                failing_operation,
                "circuit_breaker_test",
                strategy,
            )
        
        # Third attempt should fail with CircuitBreakerOpenError
        with pytest.raises(CircuitBreakerOpenError):
            await error_recovery.execute_with_retry(
                failing_operation,
                "circuit_breaker_test",
                strategy,
            )


class TestErrorDecorator:
    """Test error handling decorator."""

    @pytest.mark.asyncio
    async def test_with_error_handling_decorator(self):
        """Test with_error_handling decorator."""
        reset_error_stats()

        @with_error_handling(ErrorRecoveryStrategy.file_strategy(), "decorated_operation")
        async def decorated_function():
            return "success"

        result = await decorated_function()
        assert result == "success"

    @pytest.mark.asyncio
    async def test_with_error_handling_decorator_with_failure(self):
        """Test with_error_handling decorator with failure."""
        reset_error_stats()

        @with_error_handling(ErrorRecoveryStrategy.file_strategy(), "decorated_operation")
        async def failing_decorated_function():
            raise ConfigurationError("Configuration error")

        with pytest.raises(ConfigurationError):
            await failing_decorated_function()


class TestErrorContext:
    """Test error context manager."""

    @pytest.mark.asyncio
    async def test_error_context_success(self):
        """Test error context manager with successful operation."""
        async with error_context("test_operation", user_id="test_user"):
            # Operation succeeds
            pass

    @pytest.mark.asyncio
    async def test_error_context_with_error(self):
        """Test error context manager with error."""
        with pytest.raises(DatabaseError):
            async with error_context("test_operation", user_id="test_user"):
                raise DatabaseError("Database connection failed")


class TestSafeShutdown:
    """Test safe shutdown functionality."""

    @pytest.mark.asyncio
    async def test_safe_shutdown_success(self):
        """Test successful safe shutdown."""
        cleanup_called = False

        async def cleanup_function():
            nonlocal cleanup_called
            cleanup_called = True

        # Mock sys.exit to avoid actually exiting in test
        with patch('sys.exit') as mock_exit:
            await safe_shutdown([cleanup_function], timeout_seconds=1.0)
            
        assert cleanup_called
        mock_exit.assert_called_once_with(0)

    @pytest.mark.asyncio
    async def test_safe_shutdown_with_failing_cleanup(self):
        """Test safe shutdown with failing cleanup function."""
        cleanup_called = False

        async def failing_cleanup_function():
            nonlocal cleanup_called
            cleanup_called = True
            raise Exception("Cleanup failed")

        # Mock sys.exit to avoid actually exiting in test
        with patch('sys.exit') as mock_exit:
            await safe_shutdown([failing_cleanup_function], timeout_seconds=1.0)
            
        assert cleanup_called
        mock_exit.assert_called_once_with(0)  # Should still exit cleanly

    @pytest.mark.asyncio
    async def test_safe_shutdown_timeout(self):
        """Test safe shutdown with timeout."""
        async def slow_cleanup_function():
            await asyncio.sleep(2.0)  # Longer than timeout

        # Mock sys.exit to avoid actually exiting in test
        with patch('sys.exit') as mock_exit:
            await safe_shutdown([slow_cleanup_function], timeout_seconds=0.1)
            
        mock_exit.assert_called_once_with(0)


class TestErrorStatistics:
    """Test error statistics collection."""

    def test_error_stats_collection(self):
        """Test error statistics collection."""
        reset_error_stats()
        
        # Create some errors to generate stats
        from common.core.error_handling import error_monitor
        
        error1 = NetworkError("Network error 1")
        error2 = DatabaseError("Database error 1")
        error3 = NetworkError("Network error 2")
        
        error_monitor.report_error(error1, "test_context_1")
        error_monitor.report_error(error2, "test_context_2")
        error_monitor.report_error(error3, "test_context_3")
        
        error_monitor.report_recovery("network", 2)
        
        stats = get_error_stats()
        
        assert stats["total_errors"] == 3
        assert stats["errors_by_category"]["network"] == 2
        assert stats["errors_by_category"]["database"] == 1
        assert stats["retryable_errors"] == 3  # All are retryable
        assert stats["non_retryable_errors"] == 0
        assert stats["recovery_successes"] == 1

    def test_circuit_breaker_stats(self):
        """Test circuit breaker statistics in error monitor."""
        reset_error_stats()
        from common.core.error_handling import error_monitor
        
        # Get circuit breaker and trigger it
        cb = error_monitor.get_circuit_breaker("test_service", threshold=2)
        cb.record_failure()
        cb.record_failure()  # This should open the circuit
        
        stats = get_error_stats()
        
        assert "circuit_breakers" in stats
        assert "test_service" in stats["circuit_breakers"]
        assert stats["circuit_breakers"]["test_service"]["state"] == "open"
        assert stats["circuit_breakers"]["test_service"]["failure_count"] == 2


class TestStructuredLogging:
    """Test structured logging integration."""

    def test_structured_log_configuration(self):
        """Test structured logging configuration."""
        # Test that structlog is properly configured
        logger = structlog.get_logger(__name__)
        
        # This should not raise an exception
        logger.info("Test log message", test_field="test_value")

    def test_error_logging_with_context(self):
        """Test error logging with structured context."""
        from common.core.error_handling import error_monitor
        
        error = NetworkError(
            "Connection timeout",
            url="https://api.example.com",
            attempt=1,
            max_attempts=3,
        )
        
        # This should log with structured context
        error_monitor.report_error(error, "api_request")


@pytest.mark.integration
class TestErrorHandlingIntegration:
    """Integration tests for complete error handling system."""

    @pytest.mark.asyncio
    async def test_complete_error_flow(self):
        """Test complete error handling flow from detection to recovery."""
        reset_error_stats()
        
        error_recovery = ErrorRecovery()
        attempt_count = 0

        async def simulated_api_call():
            nonlocal attempt_count
            attempt_count += 1
            
            if attempt_count <= 2:
                # Simulate network errors on first two attempts
                raise NetworkError(
                    "API unavailable",
                    url="https://api.example.com",
                    attempt=attempt_count,
                    max_attempts=3,
                )
            else:
                # Succeed on third attempt
                return {"status": "success", "data": "api_response"}

        strategy = ErrorRecoveryStrategy.network_strategy()
        strategy.base_delay_ms = 1  # Speed up test
        
        # Execute with error recovery
        result = await error_recovery.execute_with_retry(
            simulated_api_call,
            "api_integration_test",
            strategy,
        )
        
        assert result["status"] == "success"
        assert attempt_count == 3
        
        # Check error statistics
        stats = get_error_stats()
        assert stats["total_errors"] == 2  # Two NetworkErrors reported
        assert stats["recovery_successes"] == 1
        assert stats["errors_by_category"]["network"] == 2

    @pytest.mark.asyncio
    async def test_circuit_breaker_integration_complete(self):
        """Test complete circuit breaker integration."""
        reset_error_stats()
        
        error_recovery = ErrorRecovery()

        async def unreliable_service():
            raise DatabaseError("Database connection failed")

        strategy = ErrorRecoveryStrategy.database_strategy()
        strategy.circuit_breaker_threshold = 2
        strategy.base_delay_ms = 1  # Speed up test

        # First two calls should fail with DatabaseError
        for i in range(2):
            with pytest.raises(DatabaseError):
                await error_recovery.execute_with_retry(
                    unreliable_service,
                    "database_service",
                    strategy,
                )

        # Third call should fail with CircuitBreakerOpenError
        with pytest.raises(CircuitBreakerOpenError):
            await error_recovery.execute_with_retry(
                unreliable_service,
                "database_service",
                strategy,
            )

        # Verify circuit breaker statistics
        stats = get_error_stats()
        assert "circuit_breakers" in stats
        assert "database_service" in stats["circuit_breakers"]
        assert stats["circuit_breakers"]["database_service"]["state"] == "open"

    def test_logging_configuration_production(self):
        """Test logging configuration for production use."""
        # This test would verify that logging is properly configured
        # for production environments with structured output
        
        from common.core.error_handling import error_monitor
        
        error = ConfigurationError(
            "Invalid API key format",
            field="api_key",
        )
        
        # Log error (in production this would go to structured logs)
        error_monitor.report_error(error, "configuration_validation")
        
        # In real implementation, you would verify log output format,
        # destination, and structured fields