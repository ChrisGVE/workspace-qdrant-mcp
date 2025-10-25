"""
Real-time token usage tracking for context injection system.

This module provides comprehensive token usage tracking across tools, operations,
and sessions with persistence and real-time statistics.
"""

from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from threading import Lock
from typing import Any, Optional

from loguru import logger

from .token_budget import TokenCounter


class OperationType(Enum):
    """Types of operations that consume tokens."""

    CONTEXT_INJECTION = "context_injection"  # Memory rule injection
    USER_QUERY = "user_query"  # User input processing
    RULE_RETRIEVAL = "rule_retrieval"  # Rule fetching/filtering
    FORMATTING = "formatting"  # Rule formatting for LLM
    COMPRESSION = "compression"  # Rule compression
    SEARCH = "search"  # Vector search operations
    BATCH_PROCESSING = "batch_processing"  # Batch operations
    OTHER = "other"  # Other operations


@dataclass
class OperationUsage:
    """
    Token usage for a single operation.

    Attributes:
        operation_type: Type of operation performed
        operation_id: Unique operation identifier
        tokens_used: Total tokens consumed
        timestamp: When the operation occurred
        metadata: Additional operation-specific data
    """

    operation_type: OperationType
    operation_id: str
    tokens_used: int
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolUsageStats:
    """
    Token usage statistics for a specific tool.

    Tracks cumulative usage across all operations for a tool.

    Attributes:
        tool_name: Tool identifier (e.g., "claude", "codex", "gemini")
        total_tokens: Total tokens used by this tool
        operation_counts: Count of operations by type
        operation_tokens: Total tokens by operation type
        first_operation_at: Timestamp of first operation
        last_operation_at: Timestamp of most recent operation
        operations: List of all operations (optional, for detailed tracking)
    """

    tool_name: str
    total_tokens: int = 0
    operation_counts: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    operation_tokens: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    first_operation_at: datetime | None = None
    last_operation_at: datetime | None = None
    operations: list[OperationUsage] = field(default_factory=list)

    def add_operation(self, operation: OperationUsage) -> None:
        """
        Record an operation for this tool.

        Args:
            operation: Operation usage to record
        """
        self.total_tokens += operation.tokens_used
        op_type = operation.operation_type.value
        self.operation_counts[op_type] += 1
        self.operation_tokens[op_type] += operation.tokens_used

        if self.first_operation_at is None:
            self.first_operation_at = operation.timestamp
        self.last_operation_at = operation.timestamp

        self.operations.append(operation)

    def get_average_tokens_per_operation(self) -> float:
        """
        Calculate average tokens per operation.

        Returns:
            Average tokens (0.0 if no operations)
        """
        total_ops = sum(self.operation_counts.values())
        if total_ops == 0:
            return 0.0
        return self.total_tokens / total_ops

    def get_operation_percentage(self, operation_type: OperationType) -> float:
        """
        Get percentage of tokens used by operation type.

        Args:
            operation_type: Operation type to query

        Returns:
            Percentage of total tokens (0-100)
        """
        if self.total_tokens == 0:
            return 0.0
        op_tokens = self.operation_tokens.get(operation_type.value, 0)
        return (op_tokens / self.total_tokens) * 100.0


@dataclass
class SessionUsageSnapshot:
    """
    Snapshot of token usage at a point in time.

    Used for tracking session state and generating reports.

    Attributes:
        session_id: Session identifier
        timestamp: When snapshot was taken
        total_tokens: Total tokens used in session
        tool_stats: Statistics by tool
        recent_operations: Recent operations (last N)
    """

    session_id: str
    timestamp: datetime
    total_tokens: int
    tool_stats: dict[str, ToolUsageStats]
    recent_operations: list[OperationUsage] = field(default_factory=list)


class TokenUsageTracker:
    """
    Real-time token usage tracker with multi-level granularity.

    Provides tracking at multiple levels:
    - Global: All-time usage across all tools
    - Tool: Per-tool cumulative usage
    - Operation: Individual operation tracking
    - Session: Session-scoped usage

    Features:
    - Thread-safe operation tracking
    - Real-time statistics and reporting
    - Persistence support via snapshots
    - Context managers for automatic tracking
    - Integration with ClaudeBudgetManager
    """

    def __init__(
        self,
        session_id: str | None = None,
        track_detailed_operations: bool = True,
        max_operations_per_tool: int = 1000,
    ):
        """
        Initialize token usage tracker.

        Args:
            session_id: Optional session identifier (generated if not provided)
            track_detailed_operations: If True, store individual operations
            max_operations_per_tool: Maximum operations to track per tool (for memory)
        """
        self.session_id = session_id or self._generate_session_id()
        self.track_detailed_operations = track_detailed_operations
        self.max_operations_per_tool = max_operations_per_tool

        # Thread-safe data structures
        self._lock = Lock()
        self._tool_stats: dict[str, ToolUsageStats] = {}
        self._global_total_tokens = 0
        self._session_start = datetime.now(timezone.utc)

        logger.debug(
            f"Initialized TokenUsageTracker (session: {self.session_id}, "
            f"detailed: {track_detailed_operations})"
        )

    def track_operation(
        self,
        tool_name: str,
        operation_type: OperationType,
        tokens_used: int,
        operation_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> OperationUsage:
        """
        Track a token-consuming operation.

        Thread-safe operation recording with automatic statistics updates.

        Args:
            tool_name: Tool performing the operation
            operation_type: Type of operation
            tokens_used: Number of tokens consumed
            operation_id: Optional operation identifier (generated if not provided)
            metadata: Optional operation metadata

        Returns:
            OperationUsage record created
        """
        operation_id = operation_id or self._generate_operation_id(
            tool_name, operation_type
        )
        operation = OperationUsage(
            operation_type=operation_type,
            operation_id=operation_id,
            tokens_used=tokens_used,
            metadata=metadata or {},
        )

        with self._lock:
            # Ensure tool stats exist
            if tool_name not in self._tool_stats:
                self._tool_stats[tool_name] = ToolUsageStats(
                    tool_name=tool_name,
                    operations=[] if self.track_detailed_operations else [],
                )

            # Record operation
            if self.track_detailed_operations:
                self._tool_stats[tool_name].add_operation(operation)
                # Trim operations if exceeding limit
                self._trim_operations(tool_name)
            else:
                # Just update statistics without storing operation
                self._tool_stats[tool_name].total_tokens += tokens_used
                op_type = operation_type.value
                self._tool_stats[tool_name].operation_counts[op_type] += 1
                self._tool_stats[tool_name].operation_tokens[op_type] += tokens_used

            # Update global total
            self._global_total_tokens += tokens_used

        logger.debug(
            f"Tracked operation: {tool_name}.{operation_type.value} "
            f"({tokens_used} tokens, id: {operation_id})"
        )

        return operation

    def track_text(
        self,
        text: str,
        tool_name: str,
        operation_type: OperationType,
        operation_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        use_tokenizer: bool = True,
    ) -> OperationUsage:
        """
        Track text with automatic token counting.

        Convenience method that counts tokens and tracks the operation.

        Args:
            text: Text to count and track
            tool_name: Tool performing the operation
            operation_type: Type of operation
            operation_id: Optional operation identifier
            metadata: Optional operation metadata
            use_tokenizer: If True, use actual tokenizer for counting

        Returns:
            OperationUsage record created
        """
        tokens_used = TokenCounter.count_tokens(
            text, tool_name, use_tokenizer=use_tokenizer
        )
        return self.track_operation(
            tool_name=tool_name,
            operation_type=operation_type,
            tokens_used=tokens_used,
            operation_id=operation_id,
            metadata=metadata,
        )

    @contextmanager
    def track_context(
        self,
        tool_name: str,
        operation_type: OperationType,
        operation_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ):
        """
        Context manager for automatic operation tracking.

        Yields a token accumulator that can be called to add tokens.
        Automatically tracks the operation on context exit.

        Usage:
            with tracker.track_context("claude", OperationType.CONTEXT_INJECTION) as add_tokens:
                add_tokens(100)  # Add tokens as they're consumed
                add_tokens(50)

        Args:
            tool_name: Tool performing the operation
            operation_type: Type of operation
            operation_id: Optional operation identifier
            metadata: Optional operation metadata

        Yields:
            Function to add tokens: add_tokens(count: int) -> None
        """
        tokens_accumulator = [0]  # Mutable container for accumulation

        def add_tokens(count: int) -> None:
            """Add tokens to the current operation."""
            tokens_accumulator[0] += count

        try:
            yield add_tokens
        finally:
            # Track accumulated tokens on exit
            if tokens_accumulator[0] > 0:
                self.track_operation(
                    tool_name=tool_name,
                    operation_type=operation_type,
                    tokens_used=tokens_accumulator[0],
                    operation_id=operation_id,
                    metadata=metadata,
                )

    def get_tool_stats(self, tool_name: str) -> ToolUsageStats | None:
        """
        Get usage statistics for a specific tool.

        Args:
            tool_name: Tool to query

        Returns:
            ToolUsageStats or None if tool not tracked
        """
        with self._lock:
            return self._tool_stats.get(tool_name)

    def get_all_tool_stats(self) -> dict[str, ToolUsageStats]:
        """
        Get usage statistics for all tools.

        Returns:
            Dictionary of tool name to statistics
        """
        with self._lock:
            return dict(self._tool_stats)

    def get_total_tokens(self) -> int:
        """
        Get total tokens used across all tools.

        Returns:
            Total token count
        """
        with self._lock:
            return self._global_total_tokens

    def get_session_snapshot(self, include_recent: int = 10) -> SessionUsageSnapshot:
        """
        Get current session usage snapshot.

        Args:
            include_recent: Number of recent operations to include per tool

        Returns:
            SessionUsageSnapshot with current state
        """
        with self._lock:
            recent_ops = []
            if self.track_detailed_operations:
                # Collect recent operations from all tools
                all_ops = []
                for stats in self._tool_stats.values():
                    all_ops.extend(stats.operations)
                # Sort by timestamp (most recent first)
                all_ops.sort(key=lambda op: op.timestamp, reverse=True)
                recent_ops = all_ops[:include_recent]

            return SessionUsageSnapshot(
                session_id=self.session_id,
                timestamp=datetime.now(timezone.utc),
                total_tokens=self._global_total_tokens,
                tool_stats=dict(self._tool_stats),
                recent_operations=recent_ops,
            )

    def get_usage_report(self) -> dict[str, Any]:
        """
        Generate comprehensive usage report.

        Returns:
            Dictionary with detailed usage statistics
        """
        with self._lock:
            session_duration = (
                datetime.now(timezone.utc) - self._session_start
            ).total_seconds()

            tool_reports = {}
            for tool_name, stats in self._tool_stats.items():
                tool_reports[tool_name] = {
                    "total_tokens": stats.total_tokens,
                    "operation_counts": dict(stats.operation_counts),
                    "operation_tokens": dict(stats.operation_tokens),
                    "average_tokens_per_operation": stats.get_average_tokens_per_operation(),
                    "first_operation_at": (
                        stats.first_operation_at.isoformat()
                        if stats.first_operation_at
                        else None
                    ),
                    "last_operation_at": (
                        stats.last_operation_at.isoformat()
                        if stats.last_operation_at
                        else None
                    ),
                    "operation_count": sum(stats.operation_counts.values()),
                }

            return {
                "session_id": self.session_id,
                "session_start": self._session_start.isoformat(),
                "session_duration_seconds": session_duration,
                "total_tokens": self._global_total_tokens,
                "tokens_per_second": (
                    self._global_total_tokens / session_duration
                    if session_duration > 0
                    else 0
                ),
                "tool_count": len(self._tool_stats),
                "tools": tool_reports,
                "detailed_tracking_enabled": self.track_detailed_operations,
            }

    def reset_session(self, new_session_id: str | None = None) -> None:
        """
        Reset session tracking (clear all statistics).

        Args:
            new_session_id: Optional new session identifier
        """
        with self._lock:
            old_session_id = self.session_id
            self.session_id = new_session_id or self._generate_session_id()
            self._tool_stats.clear()
            self._global_total_tokens = 0
            self._session_start = datetime.now(timezone.utc)

        logger.info(f"Reset session tracking: {old_session_id} -> {self.session_id}")

    def export_snapshot(self) -> dict[str, Any]:
        """
        Export current state for persistence.

        Returns:
            JSON-serializable dictionary with full state
        """
        snapshot = self.get_session_snapshot(include_recent=100)

        # Convert dataclasses to dicts
        tool_stats_dict = {}
        for tool_name, stats in snapshot.tool_stats.items():
            tool_stats_dict[tool_name] = {
                "tool_name": stats.tool_name,
                "total_tokens": stats.total_tokens,
                "operation_counts": dict(stats.operation_counts),
                "operation_tokens": dict(stats.operation_tokens),
                "first_operation_at": (
                    stats.first_operation_at.isoformat()
                    if stats.first_operation_at
                    else None
                ),
                "last_operation_at": (
                    stats.last_operation_at.isoformat()
                    if stats.last_operation_at
                    else None
                ),
                "operations": [
                    {
                        "operation_type": op.operation_type.value,
                        "operation_id": op.operation_id,
                        "tokens_used": op.tokens_used,
                        "timestamp": op.timestamp.isoformat(),
                        "metadata": op.metadata,
                    }
                    for op in stats.operations
                ],
            }

        recent_ops_dict = [
            {
                "operation_type": op.operation_type.value,
                "operation_id": op.operation_id,
                "tokens_used": op.tokens_used,
                "timestamp": op.timestamp.isoformat(),
                "metadata": op.metadata,
            }
            for op in snapshot.recent_operations
        ]

        return {
            "session_id": snapshot.session_id,
            "timestamp": snapshot.timestamp.isoformat(),
            "total_tokens": snapshot.total_tokens,
            "tool_stats": tool_stats_dict,
            "recent_operations": recent_ops_dict,
            "session_start": self._session_start.isoformat(),
        }

    def _trim_operations(self, tool_name: str) -> None:
        """
        Trim operations list to max size (maintains most recent).

        Args:
            tool_name: Tool to trim operations for
        """
        stats = self._tool_stats.get(tool_name)
        if stats and len(stats.operations) > self.max_operations_per_tool:
            # Keep most recent operations
            stats.operations = stats.operations[-self.max_operations_per_tool :]

    @staticmethod
    def _generate_session_id() -> str:
        """
        Generate unique session identifier.

        Returns:
            Session ID string with timestamp
        """
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
        return f"usage_session_{timestamp}"

    @staticmethod
    def _generate_operation_id(tool_name: str, operation_type: OperationType) -> str:
        """
        Generate unique operation identifier.

        Args:
            tool_name: Tool name
            operation_type: Operation type

        Returns:
            Operation ID string
        """
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
        return f"{tool_name}_{operation_type.value}_{timestamp}"


class GlobalUsageTracker:
    """
    Singleton global usage tracker for cross-session tracking.

    Maintains global usage statistics across all sessions and provides
    aggregation and long-term tracking capabilities.
    """

    _instance: Optional["GlobalUsageTracker"] = None
    _lock = Lock()

    def __new__(cls):
        """Ensure singleton instance."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        """Initialize global tracker (only once)."""
        if self._initialized:
            return

        self._initialized = True
        self._sessions: dict[str, TokenUsageTracker] = {}
        self._global_total_tokens = 0
        self._creation_time = datetime.now(timezone.utc)

        logger.debug("Initialized GlobalUsageTracker singleton")

    def register_session(self, tracker: TokenUsageTracker) -> None:
        """
        Register a session tracker.

        Args:
            tracker: Session tracker to register
        """
        with self._lock:
            self._sessions[tracker.session_id] = tracker
            logger.debug(f"Registered session: {tracker.session_id}")

    def get_session(self, session_id: str) -> TokenUsageTracker | None:
        """
        Get session tracker by ID.

        Args:
            session_id: Session identifier

        Returns:
            TokenUsageTracker or None if not found
        """
        with self._lock:
            return self._sessions.get(session_id)

    def get_all_sessions(self) -> dict[str, TokenUsageTracker]:
        """
        Get all registered sessions.

        Returns:
            Dictionary of session ID to tracker
        """
        with self._lock:
            return dict(self._sessions)

    def get_global_report(self) -> dict[str, Any]:
        """
        Generate global usage report across all sessions.

        Returns:
            Dictionary with aggregated statistics
        """
        with self._lock:
            total_tokens = sum(t.get_total_tokens() for t in self._sessions.values())
            total_sessions = len(self._sessions)

            uptime = (
                datetime.now(timezone.utc) - self._creation_time
            ).total_seconds()

            return {
                "total_sessions": total_sessions,
                "total_tokens": total_tokens,
                "creation_time": self._creation_time.isoformat(),
                "uptime_seconds": uptime,
                "tokens_per_second": total_tokens / uptime if uptime > 0 else 0,
                "average_tokens_per_session": (
                    total_tokens / total_sessions if total_sessions > 0 else 0
                ),
                "sessions": {
                    session_id: tracker.get_total_tokens()
                    for session_id, tracker in self._sessions.items()
                },
            }
