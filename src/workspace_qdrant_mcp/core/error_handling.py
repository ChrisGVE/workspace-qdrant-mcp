"""
Comprehensive error handling for workspace-qdrant-mcp

This module provides structured error types, recovery strategies,
circuit breaker patterns, and comprehensive logging using structlog.
"""

import asyncio
import inspect
import logging
import time
import traceback
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from enum import Enum
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Type, Union

import structlog

# Configure structured logging
structlog.configure(
    processors=[
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.StackInfoRenderer(),
        structlog.dev.set_exc_info,
        structlog.processors.TimeStamper(fmt="ISO"),
        structlog.dev.ConsoleRenderer(),
    ],
    wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
    logger_factory=structlog.WriteLoggerFactory(),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger(__name__)


class ErrorSeverity(Enum):
    """Error severity levels for monitoring and alerting"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """Error categories for classification"""
    CONFIGURATION = "configuration"
    NETWORK = "network"
    DATABASE = "database"
    FILESYSTEM = "filesystem"
    PROCESSING = "processing"
    EMBEDDING = "embedding"
    IPC = "ipc"
    TASK = "task"
    VALIDATION = "validation"
    TIMEOUT = "timeout"
    RESOURCE = "resource"
    CIRCUIT_BREAKER = "circuit_breaker"
    AUTHENTICATION = "auth"
    INTERNAL = "internal"


class WorkspaceError(Exception):
    """Base error class for workspace-qdrant-mcp with structured context"""

    def __init__(
        self,
        message: str,
        *,
        category: ErrorCategory = ErrorCategory.INTERNAL,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        retryable: bool = False,
        context: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None,
    ):
        super().__init__(message)
        self.message = message
        self.category = category
        self.severity = severity
        self.retryable = retryable
        self.context = context or {}
        self.cause = cause
        self.timestamp = time.time()

    def to_dict(self) -> Dict[str, Any]:
        """Convert error to dictionary for logging and monitoring"""
        return {
            "message": self.message,
            "category": self.category.value,
            "severity": self.severity.value,
            "retryable": self.retryable,
            "context": self.context,
            "cause": str(self.cause) if self.cause else None,
            "timestamp": self.timestamp,
        }

    def __str__(self) -> str:
        context_str = f" (context: {self.context})" if self.context else ""
        cause_str = f" (caused by: {self.cause})" if self.cause else ""
        return f"[{self.category.value}] {self.message}{context_str}{cause_str}"


class ConfigurationError(WorkspaceError):
    """Configuration-related errors"""

    def __init__(self, message: str, field: Optional[str] = None, **kwargs):
        context = kwargs.pop("context", {})
        if field:
            context["field"] = field
        super().__init__(
            message,
            category=ErrorCategory.CONFIGURATION,
            severity=ErrorSeverity.HIGH,
            retryable=False,
            context=context,
            **kwargs,
        )


class NetworkError(WorkspaceError):
    """Network-related errors"""

    def __init__(
        self,
        message: str,
        url: Optional[str] = None,
        attempt: int = 1,
        max_attempts: int = 1,
        **kwargs,
    ):
        context = kwargs.pop("context", {})
        context.update({"url": url, "attempt": attempt, "max_attempts": max_attempts})
        super().__init__(
            message,
            category=ErrorCategory.NETWORK,
            severity=ErrorSeverity.MEDIUM,
            retryable=True,
            context=context,
            **kwargs,
        )


class DatabaseError(WorkspaceError):
    """Database-related errors"""

    def __init__(self, message: str, operation: Optional[str] = None, **kwargs):
        context = kwargs.pop("context", {})
        if operation:
            context["operation"] = operation
        super().__init__(
            message,
            category=ErrorCategory.DATABASE,
            severity=ErrorSeverity.MEDIUM,
            retryable=True,
            context=context,
            **kwargs,
        )


class FileSystemError(WorkspaceError):
    """File system-related errors"""

    def __init__(
        self,
        message: str,
        path: Optional[str] = None,
        operation: Optional[str] = None,
        **kwargs,
    ):
        context = kwargs.pop("context", {})
        context.update({"path": path, "operation": operation})
        retryable = operation not in ["delete", "write"] if operation else True
        super().__init__(
            message,
            category=ErrorCategory.FILESYSTEM,
            severity=ErrorSeverity.LOW,
            retryable=retryable,
            context=context,
            **kwargs,
        )


class ProcessingError(WorkspaceError):
    """Document processing errors"""

    def __init__(
        self,
        message: str,
        file_path: Optional[str] = None,
        document_type: Optional[str] = None,
        **kwargs,
    ):
        context = kwargs.pop("context", {})
        context.update({"file_path": file_path, "document_type": document_type})
        super().__init__(
            message,
            category=ErrorCategory.PROCESSING,
            severity=ErrorSeverity.LOW,
            retryable=False,
            context=context,
            **kwargs,
        )


class EmbeddingError(WorkspaceError):
    """Embedding generation errors"""

    def __init__(
        self, message: str, model: Optional[str] = None, retry_count: int = 0, **kwargs
    ):
        context = kwargs.pop("context", {})
        context.update({"model": model, "retry_count": retry_count})
        super().__init__(
            message,
            category=ErrorCategory.EMBEDDING,
            severity=ErrorSeverity.MEDIUM,
            retryable=True,
            context=context,
            **kwargs,
        )


class TimeoutError(WorkspaceError):
    """Timeout errors"""

    def __init__(
        self, message: str, operation: Optional[str] = None, duration_ms: Optional[int] = None, **kwargs
    ):
        context = kwargs.pop("context", {})
        context.update({"operation": operation, "duration_ms": duration_ms})
        super().__init__(
            message,
            category=ErrorCategory.TIMEOUT,
            severity=ErrorSeverity.MEDIUM,
            retryable=True,
            context=context,
            **kwargs,
        )


class CircuitBreakerOpenError(WorkspaceError):
    """Circuit breaker open errors"""

    def __init__(
        self,
        message: str,
        service: str,
        failure_count: int = 0,
        last_failure: Optional[str] = None,
        **kwargs,
    ):
        context = kwargs.pop("context", {})
        context.update({
            "service": service,
            "failure_count": failure_count,
            "last_failure": last_failure,
        })
        super().__init__(
            message,
            category=ErrorCategory.CIRCUIT_BREAKER,
            severity=ErrorSeverity.HIGH,
            retryable=False,
            context=context,
            **kwargs,
        )


@dataclass
class ErrorRecoveryStrategy:
    """Configuration for error recovery behavior"""
    max_retries: int = 3
    base_delay_ms: int = 100
    max_delay_ms: int = 30000
    exponential_backoff: bool = True
    circuit_breaker_threshold: Optional[int] = 5
    timeout_ms: Optional[int] = None

    @classmethod
    def network_strategy(cls) -> "ErrorRecoveryStrategy":
        """Recovery strategy for network operations"""
        return cls(
            max_retries=5,
            base_delay_ms=500,
            max_delay_ms=10000,
            exponential_backoff=True,
            circuit_breaker_threshold=3,
            timeout_ms=30000,
        )

    @classmethod
    def database_strategy(cls) -> "ErrorRecoveryStrategy":
        """Recovery strategy for database operations"""
        return cls(
            max_retries=3,
            base_delay_ms=1000,
            max_delay_ms=30000,
            exponential_backoff=True,
            circuit_breaker_threshold=5,
            timeout_ms=60000,
        )

    @classmethod
    def file_strategy(cls) -> "ErrorRecoveryStrategy":
        """Recovery strategy for file operations"""
        return cls(
            max_retries=2,
            base_delay_ms=50,
            max_delay_ms=1000,
            exponential_backoff=False,
            circuit_breaker_threshold=None,
            timeout_ms=5000,
        )

    def calculate_delay_ms(self, attempt: int) -> int:
        """Calculate delay for given attempt"""
        if not self.exponential_backoff:
            return self.base_delay_ms

        delay = self.base_delay_ms * (2 ** (attempt - 1))
        return min(delay, self.max_delay_ms)


@dataclass
class CircuitBreakerState:
    """Circuit breaker state tracking"""
    name: str
    failure_threshold: int = 5
    reset_timeout_ms: int = 60000
    failure_count: int = 0
    last_failure_time: Optional[float] = None
    last_success_time: Optional[float] = None
    state: str = "closed"  # closed, open, half-open

    def should_allow_request(self) -> bool:
        """Check if request should be allowed"""
        if self.state == "closed":
            return True
        
        if self.state == "open":
            if (
                self.last_failure_time
                and (time.time() - self.last_failure_time) * 1000 > self.reset_timeout_ms
            ):
                self.state = "half-open"
                return True
            return False
        
        # half-open: allow one test request
        return True

    def record_success(self):
        """Record successful operation"""
        self.failure_count = 0
        self.last_success_time = time.time()
        self.state = "closed"

    def record_failure(self):
        """Record failed operation"""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = "open"

    def get_status(self) -> Dict[str, Any]:
        """Get circuit breaker status"""
        return {
            "name": self.name,
            "state": self.state,
            "failure_count": self.failure_count,
            "failure_threshold": self.failure_threshold,
            "last_failure_time": self.last_failure_time,
            "last_success_time": self.last_success_time,
        }


class ErrorMonitor:
    """Error monitoring and metrics collection"""

    def __init__(self):
        self.stats = {
            "total_errors": 0,
            "errors_by_category": {},
            "retryable_errors": 0,
            "non_retryable_errors": 0,
            "recovery_successes": 0,
            "circuit_breaker_opens": 0,
        }
        self.circuit_breakers: Dict[str, CircuitBreakerState] = {}

    def report_error(self, error: WorkspaceError, context: Optional[str] = None):
        """Report error for monitoring"""
        self.stats["total_errors"] += 1
        category = error.category.value
        
        if category not in self.stats["errors_by_category"]:
            self.stats["errors_by_category"][category] = 0
        self.stats["errors_by_category"][category] += 1

        if error.retryable:
            self.stats["retryable_errors"] += 1
        else:
            self.stats["non_retryable_errors"] += 1

        # Log with structured context
        log_context = {
            "error_category": category,
            "error_severity": error.severity.value,
            "retryable": error.retryable,
            "context": context,
            "error_context": error.context,
        }

        if error.severity in [ErrorSeverity.LOW]:
            logger.info("Error reported", error=str(error), **log_context)
        elif error.severity == ErrorSeverity.MEDIUM:
            logger.warning("Error reported", error=str(error), **log_context)
        else:
            logger.error("Error reported", error=str(error), **log_context, exc_info=error.cause)

    def report_recovery(self, error_category: str, attempt: int):
        """Report successful error recovery"""
        self.stats["recovery_successes"] += 1
        logger.info(
            "Error recovery succeeded",
            error_category=error_category,
            attempt=attempt,
        )

    def get_circuit_breaker(self, name: str, threshold: int = 5) -> CircuitBreakerState:
        """Get or create circuit breaker"""
        if name not in self.circuit_breakers:
            self.circuit_breakers[name] = CircuitBreakerState(
                name=name, failure_threshold=threshold
            )
        return self.circuit_breakers[name]

    def get_stats(self) -> Dict[str, Any]:
        """Get error statistics"""
        return {
            **self.stats,
            "circuit_breakers": {
                name: cb.get_status() for name, cb in self.circuit_breakers.items()
            },
        }


# Global error monitor instance
error_monitor = ErrorMonitor()


class ErrorRecovery:
    """Error recovery with retry logic and circuit breaker protection"""

    def __init__(self, monitor: Optional[ErrorMonitor] = None):
        self.monitor = monitor or error_monitor

    async def execute_with_retry(
        self,
        operation: Callable[[], Any],
        operation_name: str,
        strategy: ErrorRecoveryStrategy,
        *args,
        **kwargs,
    ) -> Any:
        """Execute operation with automatic retry and circuit breaker protection"""
        circuit_breaker = None
        if strategy.circuit_breaker_threshold:
            circuit_breaker = self.monitor.get_circuit_breaker(
                operation_name, strategy.circuit_breaker_threshold
            )

        attempt = 1
        while attempt <= strategy.max_retries:
            try:
                # Check circuit breaker
                if circuit_breaker and not circuit_breaker.should_allow_request():
                    raise CircuitBreakerOpenError(
                        f"Circuit breaker open for {operation_name}",
                        service=operation_name,
                        failure_count=circuit_breaker.failure_count,
                        last_failure=str(circuit_breaker.last_failure_time),
                    )

                # Execute operation with timeout if specified
                if strategy.timeout_ms:
                    result = await asyncio.wait_for(
                        self._ensure_awaitable(operation(*args, **kwargs)),
                        timeout=strategy.timeout_ms / 1000.0,
                    )
                else:
                    result = await self._ensure_awaitable(operation(*args, **kwargs))

                # Record success
                if circuit_breaker:
                    circuit_breaker.record_success()

                if attempt > 1:
                    self.monitor.report_recovery(operation_name, attempt)

                return result

            except asyncio.TimeoutError as e:
                error = TimeoutError(
                    f"Operation {operation_name} timed out",
                    operation=operation_name,
                    duration_ms=strategy.timeout_ms,
                    cause=e,
                )
                if circuit_breaker:
                    circuit_breaker.record_failure()
                await self._handle_retry(error, attempt, strategy, operation_name)

            except WorkspaceError as e:
                if circuit_breaker:
                    circuit_breaker.record_failure()
                await self._handle_retry(e, attempt, strategy, operation_name)

            except Exception as e:
                # Convert generic exceptions to WorkspaceError
                error = WorkspaceError(
                    f"Unexpected error in {operation_name}: {str(e)}",
                    category=ErrorCategory.INTERNAL,
                    retryable=True,
                    cause=e,
                )
                if circuit_breaker:
                    circuit_breaker.record_failure()
                await self._handle_retry(error, attempt, strategy, operation_name)

            attempt += 1

        # All retries exhausted
        raise WorkspaceError(
            f"All {strategy.max_retries} retry attempts failed for {operation_name}",
            category=ErrorCategory.INTERNAL,
            severity=ErrorSeverity.HIGH,
            retryable=False,
        )

    async def _handle_retry(
        self,
        error: WorkspaceError,
        attempt: int,
        strategy: ErrorRecoveryStrategy,
        operation_name: str,
    ):
        """Handle retry logic"""
        self.monitor.report_error(error, operation_name)

        if not error.retryable or attempt >= strategy.max_retries:
            raise error

        delay_ms = strategy.calculate_delay_ms(attempt)
        logger.warning(
            "Retrying operation after error",
            operation=operation_name,
            attempt=attempt,
            max_attempts=strategy.max_retries,
            delay_ms=delay_ms,
            error=str(error),
        )

        await asyncio.sleep(delay_ms / 1000.0)

    async def _ensure_awaitable(self, result):
        """Ensure result is awaitable"""
        if inspect.iscoroutine(result) or hasattr(result, "__await__"):
            return await result
        return result


# Global error recovery instance
error_recovery = ErrorRecovery()


def with_error_handling(
    strategy: Optional[ErrorRecoveryStrategy] = None,
    operation_name: Optional[str] = None,
):
    """Decorator for automatic error handling and recovery"""

    def decorator(func: Callable) -> Callable:
        actual_strategy = strategy or ErrorRecoveryStrategy()
        actual_name = operation_name or f"{func.__module__}.{func.__qualname__}"

        @wraps(func)
        async def wrapper(*args, **kwargs):
            return await error_recovery.execute_with_retry(
                func, actual_name, actual_strategy, *args, **kwargs
            )

        return wrapper

    return decorator


@asynccontextmanager
async def error_context(context_name: str, **context_data):
    """Context manager for error tracking and structured logging"""
    # Add context to structlog
    structlog.contextvars.bind_contextvars(**context_data, operation=context_name)
    
    start_time = time.time()
    try:
        logger.debug("Operation started", operation=context_name, context=context_data)
        yield
        
        duration_ms = (time.time() - start_time) * 1000
        logger.debug(
            "Operation completed successfully",
            operation=context_name,
            duration_ms=duration_ms,
        )
        
    except Exception as e:
        duration_ms = (time.time() - start_time) * 1000
        
        # Convert to WorkspaceError if needed
        if not isinstance(e, WorkspaceError):
            error = WorkspaceError(
                f"Error in {context_name}: {str(e)}",
                category=ErrorCategory.INTERNAL,
                cause=e,
                context=context_data,
            )
        else:
            error = e
        
        error_monitor.report_error(error, context_name)
        
        logger.error(
            "Operation failed",
            operation=context_name,
            duration_ms=duration_ms,
            error=str(error),
            exc_info=e,
        )
        raise
    
    finally:
        # Clear context
        structlog.contextvars.clear_contextvars()


async def safe_shutdown(cleanup_functions: List[Callable], timeout_seconds: float = 30.0):
    """Safely shutdown with proper async cleanup instead of os._exit"""
    logger.info("Starting graceful shutdown", timeout_seconds=timeout_seconds)
    
    async def run_cleanup():
        for cleanup_func in cleanup_functions:
            try:
                if asyncio.iscoroutinefunction(cleanup_func):
                    await cleanup_func()
                else:
                    cleanup_func()
                logger.debug("Cleanup function completed", function=cleanup_func.__name__)
            except Exception as e:
                logger.error(
                    "Error during cleanup",
                    function=cleanup_func.__name__,
                    error=str(e),
                    exc_info=e,
                )
    
    try:
        await asyncio.wait_for(run_cleanup(), timeout=timeout_seconds)
        logger.info("Graceful shutdown completed successfully")
    except asyncio.TimeoutError:
        logger.warning(
            "Shutdown timeout exceeded, forcing exit",
            timeout_seconds=timeout_seconds,
        )
    except Exception as e:
        logger.error("Error during shutdown", error=str(e), exc_info=e)
    
    # Clean exit without os._exit
    import sys
    sys.exit(0)


def get_error_stats() -> Dict[str, Any]:
    """Get current error statistics"""
    return error_monitor.get_stats()


def reset_error_stats():
    """Reset error statistics (for testing)"""
    global error_monitor
    error_monitor = ErrorMonitor()