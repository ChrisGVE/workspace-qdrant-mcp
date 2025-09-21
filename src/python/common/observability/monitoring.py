"""
Operation monitoring and instrumentation for workspace-qdrant-mcp.

Provides decorators and context managers for automatic instrumentation of
operations with structured logging, metrics collection, and performance monitoring.
Designed for comprehensive observability of all system operations.

Features:
    - Automatic operation timing and metrics recording
    - Structured logging with operation context
    - Error tracking and alerting
    - Performance monitoring with thresholds
    - Integration with external monitoring systems

Decorators:
    - @monitor_async: Monitor async functions
    - @monitor_sync: Monitor synchronous functions
    - @monitor_performance: Monitor with performance thresholds

Example:
    ```python
    from workspace_qdrant_mcp.observability import monitor_async

    @monitor_async("search", critical=True, timeout_warning=2.0)
    async def perform_search(query: str, collection: str):
        # Function automatically monitored
        return search_results

    # Manual monitoring
    with OperationMonitor("custom_operation", collection="test"):
        # Perform operation
        pass
    ```
"""

import asyncio
import functools
import inspect
import time
import uuid
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Dict, Optional, TypeVar, Union

from loguru import logger
from python.common.logging import LogContext
from .metrics import metrics_instance, record_operation

# logger imported from loguru

# Type variables for decorators
F = TypeVar("F", bound=Callable[..., Any])
AF = TypeVar("AF", bound=Callable[..., Awaitable[Any]])


@dataclass
class OperationConfig:
    """Configuration for operation monitoring."""

    name: str
    critical: bool = False
    timeout_warning: Optional[float] = None
    slow_threshold: Optional[float] = None
    error_threshold: Optional[float] = None
    include_args: bool = False
    include_result: bool = False
    sample_rate: float = 1.0  # 0.0-1.0, for high-frequency operations


class OperationMonitor:
    """Context manager for monitoring operations with full observability."""

    def __init__(
        self,
        operation_name: str,
        critical: bool = False,
        timeout_warning: Optional[float] = None,
        slow_threshold: Optional[float] = None,
        **context,
    ):
        self.operation_name = operation_name
        self.critical = critical
        self.timeout_warning = timeout_warning
        self.slow_threshold = slow_threshold
        self.context = context
        self.operation_id = str(uuid.uuid4())
        self.start_time: Optional[float] = None

    def __enter__(self):
        """Start monitoring operation."""
        self.start_time = time.perf_counter()

        # Log operation start with context
        with LogContext(
            operation=self.operation_name,
            operation_id=self.operation_id,
            **self.context,
        ):
            logger.info(
                f"Operation started: {self.operation_name}",
                critical=self.critical,
                **self.context,
            )

        # Record metrics
        metrics_instance.start_operation(
            self.operation_id, self.operation_name, **self.context
        )

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Complete operation monitoring."""
        if self.start_time is None:
            logger.error(
                "Operation monitor not properly started", operation=self.operation_name
            )
            return

        duration = time.perf_counter() - self.start_time
        success = exc_type is None

        # Prepare log context
        log_context = {
            "operation": self.operation_name,
            "operation_id": self.operation_id,
            "duration_seconds": duration,
            "success": success,
            **self.context,
        }

        if not success:
            log_context.update(
                {
                    "error_type": exc_type.__name__ if exc_type else "Unknown",
                    "error_message": str(exc_val) if exc_val else "Unknown error",
                }
            )

        # Check for performance issues
        performance_issues = []
        if self.timeout_warning and duration > self.timeout_warning:
            performance_issues.append(
                f"exceeded timeout warning ({self.timeout_warning}s)"
            )

        if self.slow_threshold and duration > self.slow_threshold:
            performance_issues.append(
                f"exceeded slow threshold ({self.slow_threshold}s)"
            )

        # Log completion
        with LogContext(**log_context):
            if not success:
                if self.critical:
                    logger.critical(
                        f"Critical operation failed: {self.operation_name}",
                        exc_info=True,
                    )
                else:
                    logger.error(
                        f"Operation failed: {self.operation_name}", exc_info=True
                    )
            elif performance_issues:
                logger.warning(
                    f"Operation completed with issues: {self.operation_name}",
                    issues=performance_issues,
                )
            else:
                logger.info(f"Operation completed: {self.operation_name}")

        # Complete metrics tracking
        metrics_instance.complete_operation(
            self.operation_id, self.operation_name, success=success, **self.context
        )

        # Record additional performance metrics
        if performance_issues:
            metrics_instance.increment_counter(
                "slow_operations_total", operation=self.operation_name, **self.context
            )

        if self.critical and not success:
            metrics_instance.increment_counter(
                "critical_failures_total", operation=self.operation_name, **self.context
            )


@asynccontextmanager
async def async_operation_monitor(
    operation_name: str,
    critical: bool = False,
    timeout_warning: Optional[float] = None,
    slow_threshold: Optional[float] = None,
    **context,
):
    """Async context manager for operation monitoring."""
    monitor = OperationMonitor(
        operation_name, critical, timeout_warning, slow_threshold, **context
    )

    with monitor:
        yield monitor


def monitor_sync(
    operation_name: Optional[str] = None,
    critical: bool = False,
    timeout_warning: Optional[float] = None,
    slow_threshold: Optional[float] = None,
    include_args: bool = False,
    include_result: bool = False,
    **default_context,
) -> Callable[[F], F]:
    """Decorator for monitoring synchronous functions.

    Args:
        operation_name: Name for the operation (defaults to function name)
        critical: Whether this is a critical operation
        timeout_warning: Log warning if operation takes longer than this
        slow_threshold: Record as slow operation if longer than this
        include_args: Include function arguments in logs
        include_result: Include function result in logs
        **default_context: Default context labels for metrics

    Example:
        ```python
        @monitor_sync("document_parse", critical=True, timeout_warning=5.0)
        def parse_document(file_path: str) -> Dict:
            # Function implementation
            return parsed_data
        ```
    """

    def decorator(func: F) -> F:
        op_name = operation_name or func.__name__

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Build operation context
            context = dict(default_context)
            context["function"] = func.__name__
            context["module"] = func.__module__

            if include_args and args:
                context["args_count"] = len(args)
            if include_args and kwargs:
                # Include non-sensitive kwargs
                safe_kwargs = {
                    k: v
                    for k, v in kwargs.items()
                    if not any(
                        sensitive in k.lower()
                        for sensitive in ["password", "token", "key", "secret"]
                    )
                }
                if safe_kwargs:
                    context["kwargs"] = safe_kwargs

            # Monitor the operation
            with OperationMonitor(
                op_name, critical, timeout_warning, slow_threshold, **context
            ):
                try:
                    result = func(*args, **kwargs)

                    if include_result and result is not None:
                        # Log result info without sensitive data
                        if hasattr(result, "__len__"):
                            try:
                                context["result_length"] = len(result)
                            except (TypeError, AttributeError):
                                pass

                        context["result_type"] = type(result).__name__

                    return result

                except Exception as e:
                    # Error already logged by OperationMonitor
                    raise

        return wrapper

    return decorator


def monitor_async(
    operation_name: Optional[str] = None,
    critical: bool = False,
    timeout_warning: Optional[float] = None,
    slow_threshold: Optional[float] = None,
    include_args: bool = False,
    include_result: bool = False,
    **default_context,
) -> Callable[[AF], AF]:
    """Decorator for monitoring asynchronous functions.

    Args:
        operation_name: Name for the operation (defaults to function name)
        critical: Whether this is a critical operation
        timeout_warning: Log warning if operation takes longer than this
        slow_threshold: Record as slow operation if longer than this
        include_args: Include function arguments in logs
        include_result: Include function result in logs
        **default_context: Default context labels for metrics

    Example:
        ```python
        @monitor_async("vector_search", timeout_warning=2.0)
        async def search_vectors(query: str, limit: int = 10):
            # Async function implementation
            return search_results
        ```
    """

    def decorator(func: AF) -> AF:
        op_name = operation_name or func.__name__

        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            # Build operation context
            context = dict(default_context)
            context["function"] = func.__name__
            context["module"] = func.__module__

            if include_args and args:
                context["args_count"] = len(args)
            if include_args and kwargs:
                # Include non-sensitive kwargs
                safe_kwargs = {
                    k: v
                    for k, v in kwargs.items()
                    if not any(
                        sensitive in k.lower()
                        for sensitive in ["password", "token", "key", "secret"]
                    )
                }
                if safe_kwargs:
                    context["kwargs"] = safe_kwargs

            # Monitor the operation
            async with async_operation_monitor(
                op_name, critical, timeout_warning, slow_threshold, **context
            ):
                try:
                    result = await func(*args, **kwargs)

                    if include_result and result is not None:
                        # Log result info without sensitive data
                        if hasattr(result, "__len__"):
                            try:
                                context["result_length"] = len(result)
                            except (TypeError, AttributeError):
                                pass

                        context["result_type"] = type(result).__name__

                    return result

                except Exception as e:
                    # Error already logged by OperationMonitor
                    raise

        return wrapper

    return decorator


def monitor_performance(
    slow_threshold: float = 1.0,
    critical_threshold: float = 5.0,
    memory_threshold_mb: Optional[float] = None,
) -> Callable[[Union[F, AF]], Union[F, AF]]:
    """Decorator for performance monitoring with thresholds.

    Args:
        slow_threshold: Threshold for logging slow operations
        critical_threshold: Threshold for logging critical performance issues
        memory_threshold_mb: Memory usage threshold in MB

    Example:
        ```python
        @monitor_performance(slow_threshold=0.5, critical_threshold=2.0)
        async def expensive_operation():
            # Performance-critical operation
            pass
        ```
    """

    def decorator(func: Union[F, AF]) -> Union[F, AF]:
        is_async = inspect.iscoroutinefunction(func)

        if is_async:

            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                import psutil

                process = psutil.Process()

                # Measure initial memory
                initial_memory = process.memory_info().rss / 1024 / 1024  # MB
                start_time = time.perf_counter()

                try:
                    result = await func(*args, **kwargs)

                    # Measure performance
                    duration = time.perf_counter() - start_time
                    final_memory = process.memory_info().rss / 1024 / 1024  # MB
                    memory_delta = final_memory - initial_memory

                    # Check thresholds and log appropriately
                    context = {
                        "function": func.__name__,
                        "duration_seconds": duration,
                        "memory_delta_mb": memory_delta,
                    }

                    if duration > critical_threshold:
                        logger.critical(
                            "Performance critical: function exceeded critical threshold",
                            threshold_seconds=critical_threshold,
                            **context,
                        )
                        metrics_instance.increment_counter(
                            "performance_critical_total", function=func.__name__
                        )
                    elif duration > slow_threshold:
                        logger.warning(
                            "Performance warning: function is slow",
                            threshold_seconds=slow_threshold,
                            **context,
                        )
                        metrics_instance.increment_counter(
                            "performance_slow_total", function=func.__name__
                        )

                    if memory_threshold_mb and memory_delta > memory_threshold_mb:
                        logger.warning(
                            "Memory usage warning: function used significant memory",
                            threshold_mb=memory_threshold_mb,
                            **context,
                        )
                        metrics_instance.increment_counter(
                            "memory_heavy_operations_total", function=func.__name__
                        )

                    # Record performance metrics
                    metrics_instance.record_histogram(
                        "function_duration_seconds", duration, function=func.__name__
                    )
                    metrics_instance.record_histogram(
                        "function_memory_delta_mb", memory_delta, function=func.__name__
                    )

                    return result

                except Exception as e:
                    duration = time.perf_counter() - start_time
                    logger.error(
                        "Performance monitoring: function failed",
                        function=func.__name__,
                        duration_seconds=duration,
                        error=str(e),
                        exc_info=True,
                    )
                    raise

            return async_wrapper

        else:

            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                import psutil

                process = psutil.Process()

                # Measure initial memory
                initial_memory = process.memory_info().rss / 1024 / 1024  # MB
                start_time = time.perf_counter()

                try:
                    result = func(*args, **kwargs)

                    # Measure performance
                    duration = time.perf_counter() - start_time
                    final_memory = process.memory_info().rss / 1024 / 1024  # MB
                    memory_delta = final_memory - initial_memory

                    # Check thresholds and log appropriately
                    context = {
                        "function": func.__name__,
                        "duration_seconds": duration,
                        "memory_delta_mb": memory_delta,
                    }

                    if duration > critical_threshold:
                        logger.critical(
                            "Performance critical: function exceeded critical threshold",
                            threshold_seconds=critical_threshold,
                            **context,
                        )
                        metrics_instance.increment_counter(
                            "performance_critical_total", function=func.__name__
                        )
                    elif duration > slow_threshold:
                        logger.warning(
                            "Performance warning: function is slow",
                            threshold_seconds=slow_threshold,
                            **context,
                        )
                        metrics_instance.increment_counter(
                            "performance_slow_total", function=func.__name__
                        )

                    if memory_threshold_mb and memory_delta > memory_threshold_mb:
                        logger.warning(
                            "Memory usage warning: function used significant memory",
                            threshold_mb=memory_threshold_mb,
                            **context,
                        )
                        metrics_instance.increment_counter(
                            "memory_heavy_operations_total", function=func.__name__
                        )

                    # Record performance metrics
                    metrics_instance.record_histogram(
                        "function_duration_seconds", duration, function=func.__name__
                    )
                    metrics_instance.record_histogram(
                        "function_memory_delta_mb", memory_delta, function=func.__name__
                    )

                    return result

                except Exception as e:
                    duration = time.perf_counter() - start_time
                    logger.error(
                        "Performance monitoring: function failed",
                        function=func.__name__,
                        duration_seconds=duration,
                        error=str(e),
                        exc_info=True,
                    )
                    raise

            return sync_wrapper

    return decorator


class BatchOperationMonitor:
    """Monitor batch operations with progress tracking."""

    def __init__(self, operation_name: str, total_items: int, **context):
        self.operation_name = operation_name
        self.total_items = total_items
        self.context = context
        self.processed_items = 0
        self.failed_items = 0
        self.start_time: Optional[float] = None
        self.operation_id = str(uuid.uuid4())

    def __enter__(self):
        self.start_time = time.perf_counter()

        logger.info(
            f"Batch operation started: {self.operation_name}",
            operation_id=self.operation_id,
            total_items=self.total_items,
            **self.context,
        )

        metrics_instance.set_gauge(
            "batch_operation_total_items",
            self.total_items,
            operation=self.operation_name,
            **self.context,
        )

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time is None:
            return

        duration = time.perf_counter() - self.start_time
        success_rate = (self.processed_items - self.failed_items) / max(
            1, self.processed_items
        )

        logger.info(
            f"Batch operation completed: {self.operation_name}",
            operation_id=self.operation_id,
            duration_seconds=duration,
            processed_items=self.processed_items,
            failed_items=self.failed_items,
            success_rate=success_rate,
            **self.context,
        )

        # Record metrics
        metrics_instance.record_histogram(
            "batch_operation_duration_seconds",
            duration,
            operation=self.operation_name,
            **self.context,
        )
        metrics_instance.record_histogram(
            "batch_operation_success_rate",
            success_rate,
            operation=self.operation_name,
            **self.context,
        )
        metrics_instance.set_gauge(
            "batch_operation_total_items",
            0,
            operation=self.operation_name,
            **self.context,
        )

    def record_item_processed(self, success: bool = True):
        """Record processing of a single item."""
        self.processed_items += 1
        if not success:
            self.failed_items += 1

        metrics_instance.set_gauge(
            "batch_operation_processed_items",
            self.processed_items,
            operation=self.operation_name,
            **self.context,
        )

        # Log progress at intervals
        if self.processed_items % max(1, self.total_items // 10) == 0:
            progress = (self.processed_items / self.total_items) * 100
            logger.debug(
                f"Batch progress: {self.operation_name}",
                operation_id=self.operation_id,
                progress_percent=progress,
                processed_items=self.processed_items,
                total_items=self.total_items,
                **self.context,
            )


def monitor_batch_operation(
    operation_name: str, total_items: int, **context
) -> BatchOperationMonitor:
    """Create a batch operation monitor.

    Args:
        operation_name: Name of the batch operation
        total_items: Total number of items to process
        **context: Additional context for logging and metrics

    Example:
        ```python
        with monitor_batch_operation("document_ingestion", len(files)) as monitor:
            for file in files:
                try:
                    process_file(file)
                    monitor.record_item_processed(success=True)
                except Exception:
                    monitor.record_item_processed(success=False)
        ```
    """
    return BatchOperationMonitor(operation_name, total_items, **context)
