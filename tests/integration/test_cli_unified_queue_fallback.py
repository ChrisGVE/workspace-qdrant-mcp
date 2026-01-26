"""
CLI Unified Queue Fallback Integration Tests (Task 37.33).

Tests that the Rust CLI falls back to unified_queue when daemon is unavailable.
Validates ADR-002 compliance: all writes route through daemon or unified_queue.

Test Strategy:
1. Ensure daemon is NOT running
2. Run CLI command that triggers write operation
3. Verify item is enqueued to unified_queue
4. Verify CLI output shows queued message with queue_id
"""

import json
import os
import sqlite3
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

import pytest


# Path to local CLI binary built with phase2 features
LOCAL_WQM_BINARY = Path(__file__).parent.parent.parent / "src" / "rust" / "cli" / "target" / "release" / "wqm"


def get_wqm_cmd() -> str:
    """Get the path to the wqm binary to use for tests."""
    # Prefer local binary if it exists (built with phase2 features)
    if LOCAL_WQM_BINARY.exists():
        return str(LOCAL_WQM_BINARY)
    # Fallback to system wqm
    return "wqm"


def get_state_db_path() -> Path:
    """Get the path to the state database (matches CLI logic)."""
    home = os.environ.get("HOME", "")
    if home:
        return Path(home) / ".workspace-qdrant" / "state.db"
    return Path("/tmp/.workspace-qdrant/state.db")


def ensure_unified_queue_table(db_path: Path) -> None:
    """Ensure the unified_queue table exists in the database."""
    db_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(db_path)
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS unified_queue (
            queue_id TEXT PRIMARY KEY,
            idempotency_key TEXT UNIQUE NOT NULL,
            item_type TEXT NOT NULL,
            op TEXT NOT NULL,
            tenant_id TEXT NOT NULL,
            collection TEXT NOT NULL,
            priority INTEGER DEFAULT 5,
            status TEXT DEFAULT 'pending',
            branch TEXT,
            payload_json TEXT,
            metadata TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            retry_count INTEGER DEFAULT 0,
            max_retries INTEGER DEFAULT 3,
            last_error TEXT,
            leased_by TEXT,
            lease_expires_at TEXT
        );

        CREATE INDEX IF NOT EXISTS idx_unified_queue_status ON unified_queue(status);
        CREATE INDEX IF NOT EXISTS idx_unified_queue_priority ON unified_queue(priority);

        -- Schema version tracking
        CREATE TABLE IF NOT EXISTS schema_version (version INTEGER);
        DELETE FROM schema_version;
        INSERT INTO schema_version VALUES (13);
    """)
    conn.close()


def query_unified_queue(db_path: Path, status: Optional[str] = None) -> List[Dict]:
    """Query items from unified_queue table."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    if status:
        cursor = conn.execute(
            "SELECT * FROM unified_queue WHERE status = ? ORDER BY created_at DESC",
            (status,)
        )
    else:
        cursor = conn.execute("SELECT * FROM unified_queue ORDER BY created_at DESC")

    results = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return results


def clear_unified_queue(db_path: Path) -> None:
    """Clear all items from unified_queue for testing."""
    if not db_path.exists():
        return
    conn = sqlite3.connect(db_path)
    conn.execute("DELETE FROM unified_queue")
    conn.commit()
    conn.close()


def is_daemon_running() -> bool:
    """Check if the daemon is running."""
    try:
        wqm_binary = get_wqm_cmd()
        result = subprocess.run(
            [wqm_binary, "service", "status"],
            capture_output=True,
            text=True,
            timeout=5
        )
        stdout_lower = result.stdout.lower()
        # Check for positive indicators but exclude "unhealthy"
        is_healthy = "healthy" in stdout_lower and "unhealthy" not in stdout_lower
        is_running = "running" in stdout_lower and "not running" not in stdout_lower
        return is_healthy or is_running
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


def run_wqm(args: List[str], timeout: int = 30) -> subprocess.CompletedProcess:
    """Run wqm CLI command and return result."""
    wqm_binary = get_wqm_cmd()
    cmd = [wqm_binary] + args
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout
        )
        return result
    except subprocess.TimeoutExpired:
        pytest.fail(f"Command timed out: {' '.join(cmd)}")
    except FileNotFoundError:
        pytest.skip("wqm command not found in PATH")


@pytest.fixture
def state_db_path():
    """Fixture providing the state database path with proper setup."""
    db_path = get_state_db_path()

    # Ensure the table exists
    ensure_unified_queue_table(db_path)

    # Clear any existing items before test
    clear_unified_queue(db_path)

    yield db_path

    # Cleanup after test
    clear_unified_queue(db_path)


@pytest.fixture
def ensure_daemon_stopped():
    """Fixture that ensures daemon is stopped for fallback tests."""
    if is_daemon_running():
        # Try to stop the daemon for testing
        result = subprocess.run(
            ["wqm", "service", "stop"],
            capture_output=True,
            timeout=10
        )
        # Wait a moment for daemon to stop
        import time
        time.sleep(1)

        if is_daemon_running():
            pytest.skip("Could not stop daemon for fallback test")

    yield
    # Note: We don't restart the daemon after tests - user can do that manually


def cli_binary_exists() -> bool:
    """Check if a CLI binary with ingest support exists."""
    wqm_binary = get_wqm_cmd()
    if not Path(wqm_binary).exists():
        return False
    # Check if it has ingest command
    try:
        result = subprocess.run(
            [wqm_binary, "ingest", "--help"],
            capture_output=True,
            timeout=5
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


@pytest.mark.integration
class TestCLIUnifiedQueueFallback:
    """Test CLI fallback to unified_queue when daemon unavailable (Task 37.33)."""

    @pytest.mark.skipif(
        not cli_binary_exists(),
        reason="CLI binary with ingest support not found - build with: cargo build --release --features phase2"
    )
    @pytest.mark.skipif(
        is_daemon_running(),
        reason="Daemon is running - skip to avoid interfering with real services"
    )
    def test_ingest_text_enqueues_when_daemon_unavailable(self, state_db_path):
        """
        Test that 'wqm ingest text' enqueues to unified_queue when daemon unavailable.

        Validates:
        - Content is enqueued to unified_queue (not written directly)
        - Output shows "queued" status
        - Output shows queue_id
        - Output shows fallback_mode='unified_queue'
        - item_type='content', op='ingest' in queue
        """
        # Test data
        test_content = "Test content for CLI fallback integration test"
        test_collection = "test-cli-fallback-collection"

        # Clear any existing items
        clear_unified_queue(state_db_path)

        # Run CLI command
        result = run_wqm(["ingest", "text", test_content, "-c", test_collection])

        # CLI should complete successfully (enqueue path)
        assert result.returncode == 0, f"CLI failed: {result.stderr}"

        # Output should indicate fallback to queue
        combined_output = result.stdout + result.stderr

        # Should show daemon not running warning
        assert "Daemon not running" in combined_output or "unified queue" in combined_output.lower(), \
            f"Expected fallback message, got: {combined_output}"

        # Should show queue_id
        assert "Queue ID" in combined_output or "queue_id" in combined_output.lower(), \
            f"Expected queue ID in output, got: {combined_output}"

        # Should show queued status
        assert "queued" in combined_output.lower() or "pending" in combined_output.lower(), \
            f"Expected queued status, got: {combined_output}"

        # Should show fallback mode
        assert "unified_queue" in combined_output or "Fallback Mode" in combined_output, \
            f"Expected fallback mode in output, got: {combined_output}"

        # Verify item in database
        items = query_unified_queue(state_db_path, status="pending")
        assert len(items) >= 1, "No items found in unified_queue"

        # Find our item
        found_item = None
        for item in items:
            if item["collection"] == test_collection:
                found_item = item
                break

        assert found_item is not None, \
            f"Could not find item for collection {test_collection} in queue"

        # Verify item properties
        assert found_item["item_type"] == "content", \
            f"Expected item_type='content', got: {found_item['item_type']}"
        assert found_item["op"] == "ingest", \
            f"Expected op='ingest', got: {found_item['op']}"
        assert found_item["status"] == "pending", \
            f"Expected status='pending', got: {found_item['status']}"

        # Verify payload contains our content
        payload = json.loads(found_item["payload_json"])
        assert payload["content"] == test_content, \
            f"Content mismatch: expected {test_content}, got {payload.get('content')}"
        assert payload["source_type"] == "cli", \
            f"Expected source_type='cli', got: {payload.get('source_type')}"

    @pytest.mark.skipif(
        not cli_binary_exists(),
        reason="CLI binary with ingest support not found - build with: cargo build --release --features phase2"
    )
    @pytest.mark.skipif(
        is_daemon_running(),
        reason="Daemon is running - skip to avoid interfering with real services"
    )
    def test_ingest_text_idempotency(self, state_db_path):
        """
        Test that duplicate CLI ingest commands don't create duplicate queue items.

        Validates idempotency key generation and UNIQUE constraint handling.
        """
        test_content = "Idempotency test content"
        test_collection = "test-idempotency-collection"

        clear_unified_queue(state_db_path)

        # First ingest
        result1 = run_wqm(["ingest", "text", test_content, "-c", test_collection])
        assert result1.returncode == 0

        # Second ingest with same content
        result2 = run_wqm(["ingest", "text", test_content, "-c", test_collection])
        assert result2.returncode == 0

        # Should indicate duplicate
        combined_output = result2.stdout + result2.stderr
        assert "duplicate" in combined_output.lower() or "already queued" in combined_output.lower(), \
            f"Expected duplicate indication, got: {combined_output}"

        # Verify only one item in queue
        items = query_unified_queue(state_db_path)
        matching_items = [i for i in items if i["collection"] == test_collection]

        assert len(matching_items) == 1, \
            f"Expected 1 item (idempotent), found {len(matching_items)}"

    @pytest.mark.skipif(
        not cli_binary_exists(),
        reason="CLI binary with ingest support not found - build with: cargo build --release --features phase2"
    )
    @pytest.mark.skipif(
        is_daemon_running(),
        reason="Daemon is running - skip to avoid interfering with real services"
    )
    def test_ingest_text_with_title(self, state_db_path):
        """Test ingest with optional title parameter."""
        test_content = "Content with title"
        test_collection = "test-title-collection"
        test_title = "My Custom Document Title"

        clear_unified_queue(state_db_path)

        result = run_wqm([
            "ingest", "text", test_content,
            "-c", test_collection,
            "-t", test_title
        ])

        assert result.returncode == 0

        # Verify enqueued
        items = query_unified_queue(state_db_path)
        assert len(items) >= 1

    @pytest.mark.skipif(
        not cli_binary_exists(),
        reason="CLI binary with ingest support not found - build with: cargo build --release --features phase2"
    )
    @pytest.mark.skipif(
        is_daemon_running(),
        reason="Daemon is running - skip to avoid interfering with real services"
    )
    def test_ingest_text_priority(self, state_db_path):
        """Test that CLI ingest uses appropriate priority (8 for content)."""
        test_content = "Priority test content"
        test_collection = "test-priority-collection"

        clear_unified_queue(state_db_path)

        result = run_wqm(["ingest", "text", test_content, "-c", test_collection])
        assert result.returncode == 0

        items = query_unified_queue(state_db_path)
        matching = [i for i in items if i["collection"] == test_collection]

        assert len(matching) == 1
        assert matching[0]["priority"] == 8, \
            f"Expected priority=8 for CLI content, got: {matching[0]['priority']}"


@pytest.mark.integration
class TestCLIQueueStatusWithUnifiedQueue:
    """Test CLI queue status commands include unified_queue info."""

    @pytest.mark.skipif(
        not cli_binary_exists(),
        reason="CLI binary with ingest support not found - build with: cargo build --release --features phase2"
    )
    @pytest.mark.skipif(
        is_daemon_running(),
        reason="Daemon is running - skip to avoid interfering with real services"
    )
    def test_admin_queue_shows_unified_queue_stats(self, state_db_path):
        """
        Test that 'wqm admin queue' shows unified_queue statistics.

        Task 37.37 added unified queue stats to the admin queue command.
        """
        # Add some items to unified_queue
        test_content = "Queue stats test content"
        test_collection = "test-stats-collection"

        clear_unified_queue(state_db_path)

        # Add item via CLI
        run_wqm(["ingest", "text", test_content, "-c", test_collection])

        # Check queue status
        result = run_wqm(["admin", "queue"])

        combined_output = result.stdout + result.stderr

        # Should show unified queue section
        assert "Unified Queue" in combined_output or "unified_queue" in combined_output.lower(), \
            f"Expected unified queue stats, got: {combined_output}"

        # Should show pending count
        assert "Pending" in combined_output or "pending" in combined_output.lower()

    @pytest.mark.skipif(
        not cli_binary_exists(),
        reason="CLI binary not found - build with: cargo build --release --features phase2"
    )
    @pytest.mark.skipif(
        is_daemon_running(),
        reason="Daemon is running - skip to avoid interfering with real services"
    )
    def test_status_shows_queue_info(self, state_db_path):
        """Test that 'wqm status' command shows queue information."""
        result = run_wqm(["status"])

        # Status should succeed
        assert result.returncode == 0

        combined_output = result.stdout + result.stderr

        # Should show some queue or status information
        assert "Queue" in combined_output or "Status" in combined_output or "Daemon" in combined_output


@pytest.mark.integration
class TestCLIFallbackErrorHandling:
    """Test CLI fallback error handling scenarios."""

    @pytest.mark.skipif(
        not cli_binary_exists(),
        reason="CLI binary with ingest support not found - build with: cargo build --release --features phase2"
    )
    @pytest.mark.skipif(
        is_daemon_running(),
        reason="Daemon is running - skip to avoid interfering with real services"
    )
    def test_ingest_text_requires_collection(self):
        """Test that ingest text requires collection argument."""
        result = run_wqm(["ingest", "text", "some content"])

        # Should fail due to missing required -c argument
        assert result.returncode != 0 or "collection" in (result.stdout + result.stderr).lower()

    @pytest.mark.skipif(
        not cli_binary_exists(),
        reason="CLI binary with ingest support not found - build with: cargo build --release --features phase2"
    )
    @pytest.mark.skipif(
        is_daemon_running(),
        reason="Daemon is running - skip to avoid interfering with real services"
    )
    def test_fallback_message_is_clear(self, state_db_path):
        """Test that fallback message is user-friendly."""
        clear_unified_queue(state_db_path)

        result = run_wqm(["ingest", "text", "test", "-c", "test-coll"])

        combined_output = result.stdout + result.stderr

        # Should have clear user guidance
        has_guidance = any([
            "daemon will process" in combined_output.lower(),
            "check status" in combined_output.lower(),
            "queued for processing" in combined_output.lower(),
            "wqm status" in combined_output or "wqm admin" in combined_output
        ])

        assert has_guidance, \
            f"Expected user guidance in fallback message, got: {combined_output}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
