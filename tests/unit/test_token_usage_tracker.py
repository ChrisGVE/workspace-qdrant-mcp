"""
Unit tests for token usage tracking system.

Tests TokenUsageTracker, OperationUsage, ToolUsageStats, and GlobalUsageTracker.
"""

import time
from datetime import datetime, timezone

import pytest

from src.python.common.core.context_injection.token_usage_tracker import (
    GlobalUsageTracker,
    OperationType,
    OperationUsage,
    SessionUsageSnapshot,
    TokenUsageTracker,
    ToolUsageStats,
)


# Test fixtures
@pytest.fixture
def tracker():
    """Create a fresh token usage tracker."""
    return TokenUsageTracker(
        session_id="test_session_001",
        track_detailed_operations=True,
        max_operations_per_tool=100,
    )


@pytest.fixture
def minimal_tracker():
    """Create a tracker without detailed operation tracking."""
    return TokenUsageTracker(
        session_id="test_session_minimal",
        track_detailed_operations=False,
    )


# OperationUsage Tests
def test_operation_usage_creation():
    """Test OperationUsage dataclass creation."""
    operation = OperationUsage(
        operation_type=OperationType.CONTEXT_INJECTION,
        operation_id="op_001",
        tokens_used=150,
        metadata={"rule_count": 5},
    )

    assert operation.operation_type == OperationType.CONTEXT_INJECTION
    assert operation.operation_id == "op_001"
    assert operation.tokens_used == 150
    assert operation.metadata["rule_count"] == 5
    assert isinstance(operation.timestamp, datetime)


# ToolUsageStats Tests
def test_tool_usage_stats_initialization():
    """Test ToolUsageStats initialization."""
    stats = ToolUsageStats(tool_name="claude")

    assert stats.tool_name == "claude"
    assert stats.total_tokens == 0
    assert stats.first_operation_at is None
    assert stats.last_operation_at is None
    assert len(stats.operations) == 0


def test_tool_usage_stats_add_operation():
    """Test adding operations to ToolUsageStats."""
    stats = ToolUsageStats(tool_name="claude")

    op1 = OperationUsage(
        operation_type=OperationType.CONTEXT_INJECTION,
        operation_id="op_001",
        tokens_used=100,
    )
    op2 = OperationUsage(
        operation_type=OperationType.USER_QUERY,
        operation_id="op_002",
        tokens_used=50,
    )

    stats.add_operation(op1)
    stats.add_operation(op2)

    assert stats.total_tokens == 150
    assert stats.operation_counts[OperationType.CONTEXT_INJECTION.value] == 1
    assert stats.operation_counts[OperationType.USER_QUERY.value] == 1
    assert stats.operation_tokens[OperationType.CONTEXT_INJECTION.value] == 100
    assert stats.operation_tokens[OperationType.USER_QUERY.value] == 50
    assert len(stats.operations) == 2
    assert stats.first_operation_at is not None
    assert stats.last_operation_at is not None


def test_tool_usage_stats_averages():
    """Test average calculations in ToolUsageStats."""
    stats = ToolUsageStats(tool_name="claude")

    # Add operations
    for i in range(5):
        stats.add_operation(
            OperationUsage(
                operation_type=OperationType.CONTEXT_INJECTION,
                operation_id=f"op_{i}",
                tokens_used=100,
            )
        )

    assert stats.get_average_tokens_per_operation() == 100.0

    # Add operation with different token count
    stats.add_operation(
        OperationUsage(
            operation_type=OperationType.USER_QUERY,
            operation_id="op_6",
            tokens_used=200,
        )
    )

    # 5 * 100 + 1 * 200 = 700 tokens / 6 operations = 116.67
    assert stats.get_average_tokens_per_operation() == pytest.approx(700 / 6)


def test_tool_usage_stats_percentages():
    """Test operation percentage calculations."""
    stats = ToolUsageStats(tool_name="claude")

    stats.add_operation(
        OperationUsage(
            operation_type=OperationType.CONTEXT_INJECTION,
            operation_id="op_1",
            tokens_used=150,
        )
    )
    stats.add_operation(
        OperationUsage(
            operation_type=OperationType.USER_QUERY,
            operation_id="op_2",
            tokens_used=50,
        )
    )

    # 150 out of 200 total = 75%
    assert stats.get_operation_percentage(
        OperationType.CONTEXT_INJECTION
    ) == pytest.approx(75.0)

    # 50 out of 200 total = 25%
    assert stats.get_operation_percentage(OperationType.USER_QUERY) == pytest.approx(
        25.0
    )

    # Operation type not tracked = 0%
    assert stats.get_operation_percentage(OperationType.COMPRESSION) == 0.0


# TokenUsageTracker Tests
def test_tracker_initialization(tracker):
    """Test TokenUsageTracker initialization."""
    assert tracker.session_id == "test_session_001"
    assert tracker.track_detailed_operations is True
    assert tracker.max_operations_per_tool == 100
    assert tracker.get_total_tokens() == 0


def test_tracker_track_operation(tracker):
    """Test tracking a single operation."""
    operation = tracker.track_operation(
        tool_name="claude",
        operation_type=OperationType.CONTEXT_INJECTION,
        tokens_used=150,
        operation_id="manual_op_001",
        metadata={"rule_count": 5},
    )

    assert operation.operation_id == "manual_op_001"
    assert operation.tokens_used == 150
    assert tracker.get_total_tokens() == 150

    # Check tool stats
    stats = tracker.get_tool_stats("claude")
    assert stats is not None
    assert stats.total_tokens == 150
    assert len(stats.operations) == 1


def test_tracker_multiple_tools(tracker):
    """Test tracking operations across multiple tools."""
    tracker.track_operation(
        tool_name="claude",
        operation_type=OperationType.CONTEXT_INJECTION,
        tokens_used=100,
    )
    tracker.track_operation(
        tool_name="codex",
        operation_type=OperationType.USER_QUERY,
        tokens_used=50,
    )
    tracker.track_operation(
        tool_name="gemini",
        operation_type=OperationType.FORMATTING,
        tokens_used=25,
    )

    assert tracker.get_total_tokens() == 175

    # Check individual tool stats
    assert tracker.get_tool_stats("claude").total_tokens == 100
    assert tracker.get_tool_stats("codex").total_tokens == 50
    assert tracker.get_tool_stats("gemini").total_tokens == 25

    # Check all tools
    all_stats = tracker.get_all_tool_stats()
    assert len(all_stats) == 3


def test_tracker_track_text(tracker):
    """Test tracking text with automatic token counting."""
    operation = tracker.track_text(
        text="This is a test message for token counting.",
        tool_name="claude",
        operation_type=OperationType.USER_QUERY,
        use_tokenizer=False,  # Use estimation for consistent testing
    )

    assert operation.tokens_used > 0
    assert tracker.get_total_tokens() > 0


def test_tracker_context_manager(tracker):
    """Test context manager for automatic tracking."""
    with tracker.track_context(
        tool_name="claude",
        operation_type=OperationType.CONTEXT_INJECTION,
        operation_id="ctx_op_001",
    ) as add_tokens:
        add_tokens(100)
        add_tokens(50)
        add_tokens(25)

    # Total should be 175
    assert tracker.get_total_tokens() == 175

    stats = tracker.get_tool_stats("claude")
    assert stats.total_tokens == 175


def test_tracker_context_manager_no_tokens(tracker):
    """Test context manager with no tokens added."""
    with tracker.track_context(
        tool_name="claude",
        operation_type=OperationType.CONTEXT_INJECTION,
    ):
        pass  # Don't add any tokens

    # Should not create an operation if no tokens added
    assert tracker.get_total_tokens() == 0


def test_tracker_minimal_mode(minimal_tracker):
    """Test tracker without detailed operation tracking."""
    minimal_tracker.track_operation(
        tool_name="claude",
        operation_type=OperationType.CONTEXT_INJECTION,
        tokens_used=100,
    )
    minimal_tracker.track_operation(
        tool_name="claude",
        operation_type=OperationType.USER_QUERY,
        tokens_used=50,
    )

    assert minimal_tracker.get_total_tokens() == 150

    stats = minimal_tracker.get_tool_stats("claude")
    assert stats.total_tokens == 150
    # Should not store operations in minimal mode
    assert len(stats.operations) == 0


def test_tracker_session_snapshot(tracker):
    """Test session snapshot generation."""
    # Add some operations
    tracker.track_operation(
        tool_name="claude",
        operation_type=OperationType.CONTEXT_INJECTION,
        tokens_used=100,
    )
    tracker.track_operation(
        tool_name="claude",
        operation_type=OperationType.USER_QUERY,
        tokens_used=50,
    )
    tracker.track_operation(
        tool_name="codex",
        operation_type=OperationType.FORMATTING,
        tokens_used=25,
    )

    snapshot = tracker.get_session_snapshot(include_recent=5)

    assert isinstance(snapshot, SessionUsageSnapshot)
    assert snapshot.session_id == "test_session_001"
    assert snapshot.total_tokens == 175
    assert len(snapshot.tool_stats) == 2
    assert len(snapshot.recent_operations) <= 5


def test_tracker_usage_report(tracker):
    """Test comprehensive usage report generation."""
    # Add operations
    tracker.track_operation(
        tool_name="claude",
        operation_type=OperationType.CONTEXT_INJECTION,
        tokens_used=150,
    )
    tracker.track_operation(
        tool_name="claude",
        operation_type=OperationType.USER_QUERY,
        tokens_used=50,
    )

    # Small delay to ensure measurable duration
    time.sleep(0.1)

    report = tracker.get_usage_report()

    assert report["session_id"] == "test_session_001"
    assert report["total_tokens"] == 200
    assert report["tool_count"] == 1
    assert report["detailed_tracking_enabled"] is True
    assert "claude" in report["tools"]
    assert report["tools"]["claude"]["total_tokens"] == 200
    assert report["tools"]["claude"]["operation_count"] == 2
    assert report["session_duration_seconds"] > 0


def test_tracker_reset_session(tracker):
    """Test session reset."""
    # Add operations
    tracker.track_operation(
        tool_name="claude",
        operation_type=OperationType.CONTEXT_INJECTION,
        tokens_used=100,
    )

    assert tracker.get_total_tokens() == 100

    # Reset session
    old_session_id = tracker.session_id
    tracker.reset_session(new_session_id="new_session_001")

    assert tracker.session_id == "new_session_001"
    assert tracker.session_id != old_session_id
    assert tracker.get_total_tokens() == 0
    assert len(tracker.get_all_tool_stats()) == 0


def test_tracker_export_snapshot(tracker):
    """Test snapshot export for persistence."""
    # Add operations
    tracker.track_operation(
        tool_name="claude",
        operation_type=OperationType.CONTEXT_INJECTION,
        tokens_used=100,
        metadata={"rule_count": 5},
    )

    snapshot = tracker.export_snapshot()

    assert snapshot["session_id"] == "test_session_001"
    assert snapshot["total_tokens"] == 100
    assert "tool_stats" in snapshot
    assert "claude" in snapshot["tool_stats"]
    assert "recent_operations" in snapshot
    assert "timestamp" in snapshot

    # Verify serializable (no datetime objects, all strings)
    import json

    json_str = json.dumps(snapshot)  # Should not raise
    assert len(json_str) > 0


def test_tracker_operation_trimming():
    """Test automatic operation list trimming."""
    tracker = TokenUsageTracker(
        session_id="trim_test",
        track_detailed_operations=True,
        max_operations_per_tool=10,  # Small limit for testing
    )

    # Add more operations than the limit
    for _i in range(20):
        tracker.track_operation(
            tool_name="claude",
            operation_type=OperationType.CONTEXT_INJECTION,
            tokens_used=10,
        )

    stats = tracker.get_tool_stats("claude")
    # Should only keep most recent 10
    assert len(stats.operations) == 10
    # Total tokens should still be correct
    assert stats.total_tokens == 200


def test_tracker_thread_safety():
    """Test basic thread safety of tracker operations."""
    import threading

    tracker = TokenUsageTracker(session_id="thread_test")
    num_threads = 5
    operations_per_thread = 10

    def add_operations():
        for _i in range(operations_per_thread):
            tracker.track_operation(
                tool_name="claude",
                operation_type=OperationType.CONTEXT_INJECTION,
                tokens_used=10,
            )

    threads = [threading.Thread(target=add_operations) for _ in range(num_threads)]

    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # Should have recorded all operations
    expected_total = num_threads * operations_per_thread * 10
    assert tracker.get_total_tokens() == expected_total


# GlobalUsageTracker Tests
def test_global_tracker_singleton():
    """Test GlobalUsageTracker singleton behavior."""
    tracker1 = GlobalUsageTracker()
    tracker2 = GlobalUsageTracker()

    assert tracker1 is tracker2


def test_global_tracker_register_session():
    """Test session registration in global tracker."""
    global_tracker = GlobalUsageTracker()

    session1 = TokenUsageTracker(session_id="global_session_001")
    session2 = TokenUsageTracker(session_id="global_session_002")

    global_tracker.register_session(session1)
    global_tracker.register_session(session2)

    assert global_tracker.get_session("global_session_001") is session1
    assert global_tracker.get_session("global_session_002") is session2


def test_global_tracker_get_all_sessions():
    """Test getting all sessions from global tracker."""
    global_tracker = GlobalUsageTracker()

    session1 = TokenUsageTracker(session_id="global_all_001")
    session2 = TokenUsageTracker(session_id="global_all_002")

    global_tracker.register_session(session1)
    global_tracker.register_session(session2)

    all_sessions = global_tracker.get_all_sessions()

    # Should include at least our two sessions (may include others from other tests)
    assert "global_all_001" in all_sessions
    assert "global_all_002" in all_sessions


def test_global_tracker_report():
    """Test global usage report generation."""
    global_tracker = GlobalUsageTracker()

    session1 = TokenUsageTracker(session_id="global_report_001")
    session1.track_operation(
        tool_name="claude",
        operation_type=OperationType.CONTEXT_INJECTION,
        tokens_used=100,
    )

    session2 = TokenUsageTracker(session_id="global_report_002")
    session2.track_operation(
        tool_name="codex",
        operation_type=OperationType.USER_QUERY,
        tokens_used=50,
    )

    global_tracker.register_session(session1)
    global_tracker.register_session(session2)

    report = global_tracker.get_global_report()

    assert "total_sessions" in report
    assert "total_tokens" in report
    assert "sessions" in report
    assert report["sessions"]["global_report_001"] == 100
    assert report["sessions"]["global_report_002"] == 50


# Integration Tests
def test_tracker_with_claude_budget_manager_integration():
    """Test integration pattern with ClaudeBudgetManager."""
    from src.python.common.core.context_injection.claude_budget_manager import (
        ClaudeBudgetManager,
        ClaudeModel,
    )

    # Create tracker and budget manager
    tracker = TokenUsageTracker(session_id="integration_test")
    ClaudeBudgetManager(
        model=ClaudeModel.SONNET_3_5,
        session_id="integration_test",
    )

    # Simulate usage pattern
    user_query = "What is the capital of France?"
    tracker.track_text(
        text=user_query,
        tool_name="claude",
        operation_type=OperationType.USER_QUERY,
        use_tokenizer=False,
    )

    # Context injection
    context_text = "Rule 1: Always be helpful. Rule 2: Provide accurate information."
    tracker.track_text(
        text=context_text,
        tool_name="claude",
        operation_type=OperationType.CONTEXT_INJECTION,
        use_tokenizer=False,
    )

    # Verify tracking
    assert tracker.get_total_tokens() > 0
    stats = tracker.get_tool_stats("claude")
    assert stats.operation_counts[OperationType.USER_QUERY.value] == 1
    assert stats.operation_counts[OperationType.CONTEXT_INJECTION.value] == 1


def test_tracker_export_import_pattern():
    """Test export/import pattern for persistence."""
    # Create tracker and add operations
    tracker = TokenUsageTracker(session_id="export_test")
    tracker.track_operation(
        tool_name="claude",
        operation_type=OperationType.CONTEXT_INJECTION,
        tokens_used=100,
        metadata={"test": "data"},
    )

    # Export snapshot
    snapshot = tracker.export_snapshot()

    # Verify snapshot is JSON-serializable
    import json

    json_data = json.dumps(snapshot)
    restored_snapshot = json.loads(json_data)

    # Verify data integrity
    assert restored_snapshot["session_id"] == "export_test"
    assert restored_snapshot["total_tokens"] == 100
    assert "claude" in restored_snapshot["tool_stats"]


def test_multiple_operation_types_tracking():
    """Test tracking multiple operation types comprehensively."""
    tracker = TokenUsageTracker(session_id="multi_op_test")

    # Track different operation types
    operation_types = [
        (OperationType.CONTEXT_INJECTION, 100),
        (OperationType.USER_QUERY, 50),
        (OperationType.RULE_RETRIEVAL, 25),
        (OperationType.FORMATTING, 30),
        (OperationType.COMPRESSION, 10),
        (OperationType.SEARCH, 40),
        (OperationType.BATCH_PROCESSING, 60),
        (OperationType.OTHER, 15),
    ]

    for op_type, tokens in operation_types:
        tracker.track_operation(
            tool_name="claude",
            operation_type=op_type,
            tokens_used=tokens,
        )

    # Verify all tracked
    stats = tracker.get_tool_stats("claude")
    expected_total = sum(tokens for _, tokens in operation_types)
    assert stats.total_tokens == expected_total
    assert len(stats.operation_counts) == len(operation_types)

    # Verify percentages
    for op_type, tokens in operation_types:
        percentage = stats.get_operation_percentage(op_type)
        expected_percentage = (tokens / expected_total) * 100
        assert percentage == pytest.approx(expected_percentage)
