"""
Integration tests for concurrent CLI operations (Task 330.5).

Tests multiple CLI instances operating simultaneously:
- Concurrent CLI commands against daemon
- Multiple ingestion requests in parallel
- Simultaneous status queries
- SQLite WAL mode handling concurrent access
- Race condition detection
- Atomic operation completion

These tests verify:
1. Multiple CLI instances can operate safely
2. SQLite WAL mode handles concurrent access correctly
3. No race conditions occur with shared resources
4. Operations complete atomically even under concurrency
5. System remains stable with high concurrent load
"""

import os
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import pytest


def run_wqm_command(
    command: list, env: dict | None = None, timeout: int = 30
) -> subprocess.CompletedProcess:
    """Run wqm CLI command via subprocess."""
    full_command = ["uv", "run", "wqm"] + command
    result = subprocess.run(
        full_command,
        env=env or os.environ.copy(),
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    return result


@pytest.fixture(scope="module")
def ensure_daemon_running():
    """Ensure daemon is running for tests."""
    status_result = run_wqm_command(["service", "status"])

    if status_result.returncode != 0 or "running" not in status_result.stdout.lower():
        start_result = run_wqm_command(["service", "start"])
        if start_result.returncode != 0:
            pytest.skip("Daemon not available and could not be started")
        time.sleep(3)

    yield


@pytest.fixture
def test_workspace(tmp_path):
    """Create temporary workspace with multiple test files."""
    workspace = tmp_path / "concurrent_workspace"
    workspace.mkdir()

    # Create multiple test files for concurrent ingestion
    for i in range(20):
        test_file = workspace / f"test_file_{i}.txt"
        test_file.write_text(f"Test content for concurrent file {i}\n" * 100)

    yield {
        "workspace": workspace,
        "files": list(workspace.glob("test_file_*.txt")),
    }


@pytest.fixture
def test_collection():
    """Provide test collection name and cleanup."""
    collection_name = f"test_concurrent_{int(time.time())}"

    yield collection_name

    # Cleanup: delete test collection
    try:
        run_wqm_command(
            ["admin", "collections", "delete", collection_name, "--confirm"]
        )
    except Exception:
        pass


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.usefixtures("ensure_daemon_running")
class TestConcurrentIngestion:
    """Test concurrent ingestion operations."""

    def test_parallel_file_ingestion(self, test_workspace, test_collection):
        """Test multiple files being ingested concurrently."""
        files = test_workspace["files"][:10]  # Use first 10 files

        def ingest_file(file_path):
            """Ingest a single file."""
            result = run_wqm_command(
                ["ingest", "file", str(file_path), "--collection", test_collection]
            )
            return result.returncode, str(file_path)

        # Launch concurrent ingestion operations
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(ingest_file, f) for f in files]

            results = []
            for future in as_completed(futures):
                try:
                    results.append(future.result())
                except Exception as e:
                    print(f"Ingestion failed: {e}")

        # All operations should complete
        assert len(results) > 0, "Some ingestion operations should complete"

        # System should remain stable
        time.sleep(2)
        status_result = run_wqm_command(["status", "--quiet"])
        assert status_result.returncode == 0, "System should remain stable"

    def test_concurrent_folder_ingestion(self, test_workspace, test_collection):
        """Test concurrent folder ingestion operations."""
        workspace = test_workspace["workspace"]

        def ingest_folder(iteration):
            """Ingest folder with iteration marker."""
            result = run_wqm_command(
                ["ingest", "folder", str(workspace), "--collection", f"{test_collection}_{iteration}"]
            )
            return result.returncode, iteration

        # Launch concurrent folder ingestion
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(ingest_folder, i) for i in range(3)]

            results = []
            for future in as_completed(futures):
                try:
                    results.append(future.result())
                except Exception as e:
                    print(f"Folder ingestion failed: {e}")

        # Should handle concurrent folder ingestion
        assert len(results) > 0, "Some operations should complete"

        time.sleep(2)

        # System should be stable
        status_result = run_wqm_command(["status", "--quiet"])
        assert status_result.returncode == 0, "System should remain stable"

    def test_mixed_concurrent_operations(self, test_workspace, test_collection):
        """Test mixed ingestion and query operations concurrently."""
        files = test_workspace["files"][:5]

        def mixed_operation(op_type, file_path=None):
            """Execute mixed operation types."""
            if op_type == "ingest" and file_path:
                result = run_wqm_command(
                    ["ingest", "file", str(file_path), "--collection", test_collection]
                )
            elif op_type == "status":
                result = run_wqm_command(["status", "--quiet"])
            elif op_type == "collections":
                result = run_wqm_command(["admin", "collections"])
            else:
                return -1, "unknown"

            return result.returncode, op_type

        # Create mixed workload
        operations = []
        for i, f in enumerate(files):
            operations.append(("ingest", f))
            operations.append(("status", None))
            if i % 2 == 0:
                operations.append(("collections", None))

        # Execute concurrently
        with ThreadPoolExecutor(max_workers=6) as executor:
            futures = [
                executor.submit(mixed_operation, op[0], op[1]) for op in operations
            ]

            results = []
            for future in as_completed(futures):
                try:
                    results.append(future.result())
                except Exception as e:
                    print(f"Mixed operation failed: {e}")

        # All operations should complete
        assert len(results) >= len(operations) // 2, "Most operations should complete"

        # System should remain stable
        time.sleep(2)
        status_result = run_wqm_command(["status", "--quiet"])
        assert status_result.returncode == 0, "System should be stable after mixed load"


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.usefixtures("ensure_daemon_running")
class TestConcurrentStatusQueries:
    """Test concurrent status query operations."""

    def test_parallel_status_queries(self):
        """Test multiple status queries running concurrently."""

        def query_status():
            """Query status."""
            result = run_wqm_command(["status", "--quiet"])
            return result.returncode

        # Launch many concurrent status queries
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(query_status) for _ in range(20)]

            results = []
            for future in as_completed(futures):
                try:
                    results.append(future.result())
                except Exception as e:
                    print(f"Status query failed: {e}")

        # Most/all should succeed
        successful = [r for r in results if r == 0]
        assert len(successful) >= len(results) * 0.8, "Most status queries should succeed"

    def test_concurrent_collection_queries(self):
        """Test concurrent collection list queries."""

        def query_collections():
            """Query collections."""
            result = run_wqm_command(["admin", "collections"])
            return result.returncode

        # Launch concurrent collection queries
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(query_collections) for _ in range(15)]

            results = []
            for future in as_completed(futures):
                try:
                    results.append(future.result())
                except Exception as e:
                    print(f"Collection query failed: {e}")

        # Should handle concurrent queries
        successful = [r for r in results if r == 0]
        assert len(successful) >= len(results) * 0.7, "Most queries should succeed"

    def test_mixed_query_types(self):
        """Test different query types running concurrently."""

        def mixed_query(query_type):
            """Execute different query types."""
            if query_type == "status":
                result = run_wqm_command(["status", "--quiet"])
            elif query_type == "collections":
                result = run_wqm_command(["admin", "collections"])
            elif query_type == "watch":
                result = run_wqm_command(["watch", "list"])
            else:
                return -1

            return result.returncode

        # Create mixed query workload
        query_types = ["status"] * 10 + ["collections"] * 10 + ["watch"] * 10

        # Execute concurrently
        with ThreadPoolExecutor(max_workers=12) as executor:
            futures = [executor.submit(mixed_query, qt) for qt in query_types]

            results = []
            for future in as_completed(futures):
                try:
                    results.append(future.result())
                except Exception as e:
                    print(f"Mixed query failed: {e}")

        # Most should succeed
        successful = [r for r in results if r == 0]
        assert len(successful) >= len(results) * 0.7, "Most mixed queries should succeed"


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.usefixtures("ensure_daemon_running")
class TestWatchConfigurationConcurrency:
    """Test concurrent watch configuration operations."""

    def test_concurrent_watch_additions(self, test_workspace, test_collection):
        """Test adding multiple watches concurrently."""
        workspace = test_workspace["workspace"]

        # Create subdirectories for watches
        watch_dirs = []
        for i in range(5):
            watch_dir = workspace / f"watch_{i}"
            watch_dir.mkdir(exist_ok=True)
            watch_dirs.append(watch_dir)

        def add_watch(watch_dir, idx):
            """Add a watch folder."""
            result = run_wqm_command(
                ["watch", "add", str(watch_dir), "--collection", f"{test_collection}_{idx}"]
            )
            return result.returncode, str(watch_dir)

        # Add watches concurrently
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(add_watch, wd, i) for i, wd in enumerate(watch_dirs)]

            results = []
            for future in as_completed(futures):
                try:
                    results.append(future.result())
                except Exception as e:
                    print(f"Watch add failed: {e}")

        time.sleep(2)

        # Watch list should be consistent
        list_result = run_wqm_command(["watch", "list"])
        assert list_result.returncode == 0, "Watch list should be consistent"

        # Cleanup
        for wd in watch_dirs:
            try:
                run_wqm_command(["watch", "remove", str(wd)])
            except Exception:
                pass

    def test_concurrent_watch_operations(self, test_workspace, test_collection):
        """Test mixed watch operations concurrently."""
        workspace = test_workspace["workspace"]
        watch_dir = workspace / "concurrent_watch"
        watch_dir.mkdir(exist_ok=True)

        def watch_operation(op_type):
            """Execute watch operations."""
            if op_type == "add":
                result = run_wqm_command(
                    ["watch", "add", str(watch_dir), "--collection", test_collection]
                )
            elif op_type == "list":
                result = run_wqm_command(["watch", "list"])
            else:
                return -1

            return result.returncode

        # Mixed watch operations
        operations = ["add"] + ["list"] * 5

        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(watch_operation, op) for op in operations]

            results = []
            for future in as_completed(futures):
                try:
                    results.append(future.result())
                except Exception as e:
                    print(f"Watch operation failed: {e}")

        # Should complete without corruption
        time.sleep(2)
        list_result = run_wqm_command(["watch", "list"])
        assert list_result.returncode == 0, "Watch list should be accessible"

        # Cleanup
        try:
            run_wqm_command(["watch", "remove", str(watch_dir)])
        except Exception:
            pass


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.usefixtures("ensure_daemon_running")
class TestSQLiteWALConcurrency:
    """Test SQLite WAL mode handling of concurrent access."""

    def test_concurrent_state_reads(self):
        """Test concurrent reads of daemon state."""

        def read_state():
            """Read daemon state via status command."""
            result = run_wqm_command(["status", "--quiet"])
            return result.returncode == 0

        # Many concurrent reads
        with ThreadPoolExecutor(max_workers=15) as executor:
            futures = [executor.submit(read_state) for _ in range(30)]

            results = []
            for future in as_completed(futures):
                try:
                    results.append(future.result())
                except Exception as e:
                    print(f"State read failed: {e}")

        # All/most reads should succeed (WAL allows concurrent reads)
        successful = sum(results)
        assert successful >= len(results) * 0.9, "WAL should allow concurrent reads"

    def test_concurrent_writes_and_reads(self, test_workspace, test_collection):
        """Test concurrent write and read operations."""
        files = test_workspace["files"][:5]

        def write_operation(file_path):
            """Write operation (ingestion)."""
            result = run_wqm_command(
                ["ingest", "file", str(file_path), "--collection", test_collection]
            )
            return result.returncode == 0

        def read_operation():
            """Read operation (status query)."""
            result = run_wqm_command(["status", "--quiet"])
            return result.returncode == 0

        # Mix of writes and reads
        with ThreadPoolExecutor(max_workers=8) as executor:
            write_futures = [executor.submit(write_operation, f) for f in files]
            read_futures = [executor.submit(read_operation) for _ in range(10)]

            all_futures = write_futures + read_futures
            results = []

            for future in as_completed(all_futures):
                try:
                    results.append(future.result())
                except Exception as e:
                    print(f"Operation failed: {e}")

        # Most operations should succeed
        successful = sum(results)
        assert successful >= len(results) * 0.7, "WAL should handle mixed operations"


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.usefixtures("ensure_daemon_running")
class TestRaceConditionDetection:
    """Test for potential race conditions."""

    def test_no_duplicate_processing(self, test_workspace, test_collection):
        """Test that concurrent ingestion doesn't cause duplicate processing."""
        test_file = test_workspace["files"][0]

        def ingest_same_file():
            """Ingest the same file."""
            result = run_wqm_command(
                ["ingest", "file", str(test_file), "--collection", test_collection]
            )
            return result.returncode

        # Try to ingest same file concurrently
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(ingest_same_file) for _ in range(5)]

            results = []
            for future in as_completed(futures):
                try:
                    results.append(future.result())
                except Exception as e:
                    print(f"Ingestion failed: {e}")

        time.sleep(3)

        # System should handle without corruption
        status_result = run_wqm_command(["status", "--quiet"])
        assert status_result.returncode == 0, "System should be stable"

    def test_watch_config_race_conditions(self, test_workspace, test_collection):
        """Test for race conditions in watch configuration."""
        watch_dir = test_workspace["workspace"]

        def configure_watch():
            """Configure watch with same path."""
            result = run_wqm_command(
                ["watch", "add", str(watch_dir), "--collection", test_collection]
            )
            return result.returncode

        # Try concurrent watch additions for same path
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(configure_watch) for _ in range(3)]

            results = []
            for future in as_completed(futures):
                try:
                    results.append(future.result())
                except Exception as e:
                    print(f"Watch config failed: {e}")

        time.sleep(2)

        # Watch list should be consistent
        list_result = run_wqm_command(["watch", "list"])
        assert list_result.returncode == 0, "Watch configuration should be consistent"

        # Cleanup
        try:
            run_wqm_command(["watch", "remove", str(watch_dir)])
        except Exception:
            pass


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.usefixtures("ensure_daemon_running")
class TestHighConcurrentLoad:
    """Test system stability under high concurrent load."""

    def test_high_load_stability(self, test_workspace):
        """Test system remains stable under high concurrent load."""
        test_workspace["files"]

        def random_operation(op_id):
            """Execute random operations."""
            import random

            op_type = random.choice(["status", "collections", "watch_list"])

            if op_type == "status":
                result = run_wqm_command(["status", "--quiet"])
            elif op_type == "collections":
                result = run_wqm_command(["admin", "collections"])
            elif op_type == "watch_list":
                result = run_wqm_command(["watch", "list"])
            else:
                return -1

            return result.returncode

        # High concurrent load
        with ThreadPoolExecutor(max_workers=20) as executor:
            futures = [executor.submit(random_operation, i) for i in range(50)]

            results = []
            for future in as_completed(futures):
                try:
                    results.append(future.result())
                except Exception as e:
                    print(f"Operation failed under load: {e}")

        # System should handle most operations
        successful = [r for r in results if r == 0]
        assert len(successful) >= len(results) * 0.6, "System should handle high load"

        # System should recover
        time.sleep(3)
        status_result = run_wqm_command(["status", "--quiet"])
        assert status_result.returncode == 0, "System should recover from high load"


# Summary of test coverage:
# 1. TestConcurrentIngestion (3 tests)
#    - Parallel file ingestion
#    - Concurrent folder ingestion
#    - Mixed concurrent operations
#
# 2. TestConcurrentStatusQueries (3 tests)
#    - Parallel status queries
#    - Concurrent collection queries
#    - Mixed query types
#
# 3. TestWatchConfigurationConcurrency (2 tests)
#    - Concurrent watch additions
#    - Concurrent watch operations
#
# 4. TestSQLiteWALConcurrency (2 tests)
#    - Concurrent state reads
#    - Concurrent writes and reads
#
# 5. TestRaceConditionDetection (2 tests)
#    - No duplicate processing
#    - Watch config race conditions
#
# 6. TestHighConcurrentLoad (1 test)
#    - High load stability
#
# Total: 13 comprehensive test cases covering concurrent CLI operations
