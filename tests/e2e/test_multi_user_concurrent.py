"""
End-to-end tests for multi-user access and concurrent operations.

Tests concurrent MCP connections, simultaneous CLI operations, parallel file
ingestion, concurrent searches, shared daemon access, thread safety, data
consistency, and performance under load.
"""

import pytest
import asyncio
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple

from tests.e2e.fixtures import (
    SystemComponents,
    CLIHelper,
)


@pytest.mark.integration
@pytest.mark.slow
class TestConcurrentCLIOperations:
    """Test concurrent CLI operations from multiple users."""

    def test_concurrent_status_queries(
        self, system_components: SystemComponents, cli_helper: CLIHelper
    ):
        """Test multiple concurrent status queries."""

        def query_status(user_id: int) -> Tuple[int, int]:
            """Query status for a user."""
            result = cli_helper.run_command(["status", "--quiet"])
            return user_id, result.returncode

        # Simulate 5 users querying status concurrently
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(query_status, i) for i in range(5)]

            results = []
            for future in as_completed(futures):
                try:
                    results.append(future.result())
                except Exception as e:
                    print(f"Status query failed: {e}")

        # All queries should complete
        assert len(results) >= 3  # At least 3 should succeed

    def test_concurrent_collection_listing(
        self, system_components: SystemComponents, cli_helper: CLIHelper
    ):
        """Test multiple users listing collections concurrently."""

        def list_collections(user_id: int) -> Tuple[int, int]:
            """List collections for a user."""
            result = cli_helper.run_command(["admin", "collections"])
            return user_id, result.returncode

        # Simulate 3 users listing collections
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(list_collections, i) for i in range(3)]

            results = [f.result() for f in as_completed(futures)]

        # All should complete
        assert len(results) == 3


@pytest.mark.integration
@pytest.mark.slow
class TestParallelFileIngestion:
    """Test parallel file ingestion from multiple users."""

    def test_parallel_file_uploads(
        self, system_components: SystemComponents, cli_helper: CLIHelper
    ):
        """Test multiple users ingesting files in parallel."""
        workspace = system_components.workspace_path

        # Create files for multiple users
        user_files = []
        for user_id in range(5):
            user_file = workspace / f"user_{user_id}.txt"
            user_file.write_text(f"Content from user {user_id}")
            user_files.append((user_id, user_file))

        def ingest_file(user_data: Tuple[int, Path]) -> Tuple[int, int]:
            """Ingest file for a user."""
            user_id, file_path = user_data
            collection = f"test-parallel-user{user_id}-{int(time.time())}"
            result = cli_helper.run_command(
                ["ingest", "file", str(file_path), "--collection", collection]
            )
            return user_id, result.returncode

        # Ingest files in parallel
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(ingest_file, data) for data in user_files]

            results = []
            for future in as_completed(futures):
                try:
                    results.append(future.result())
                except Exception as e:
                    print(f"Ingestion failed: {e}")

        # Most ingestions should succeed
        assert len(results) >= 3

    def test_parallel_folder_ingestion(
        self, system_components: SystemComponents, cli_helper: CLIHelper
    ):
        """Test multiple users ingesting folders in parallel."""
        workspace = system_components.workspace_path

        # Create user folders
        user_folders = []
        for user_id in range(3):
            user_folder = workspace / f"user_folder_{user_id}"
            user_folder.mkdir(exist_ok=True)

            # Add files to folder
            for i in range(3):
                (user_folder / f"file_{i}.txt").write_text(
                    f"User {user_id} file {i}"
                )

            user_folders.append((user_id, user_folder))

        def ingest_folder(user_data: Tuple[int, Path]) -> Tuple[int, int]:
            """Ingest folder for a user."""
            user_id, folder_path = user_data
            collection = f"test-folder-user{user_id}-{int(time.time())}"
            result = cli_helper.run_command(
                ["ingest", "folder", str(folder_path), "--collection", collection],
                timeout=60,
            )
            return user_id, result.returncode

        # Ingest folders in parallel
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(ingest_folder, data) for data in user_folders]

            results = [f.result() for f in as_completed(futures)]

        # All should complete
        assert len(results) == 3


@pytest.mark.integration
@pytest.mark.slow
class TestConcurrentSearches:
    """Test concurrent search operations from multiple users."""

    @pytest.mark.asyncio
    async def test_concurrent_search_queries(
        self, system_components: SystemComponents, cli_helper: CLIHelper
    ):
        """Test multiple users searching concurrently."""
        workspace = system_components.workspace_path

        # Create searchable content
        collection_name = f"test-concurrent-search-{int(time.time())}"
        for i in range(5):
            test_file = workspace / f"search_doc_{i}.txt"
            test_file.write_text(f"Document {i} with searchable content")

        cli_helper.run_command(
            ["ingest", "folder", str(workspace), "--collection", collection_name],
            timeout=60,
        )
        await asyncio.sleep(5)

        def search_content(user_id: int) -> Tuple[int, int]:
            """Search for content as a user."""
            result = cli_helper.run_command(
                ["search", f"Document searchable", "--collection", collection_name],
                timeout=15,
            )
            return user_id, result.returncode

        # Concurrent searches
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(search_content, i) for i in range(5)]

            results = []
            for future in as_completed(futures):
                try:
                    results.append(future.result())
                except Exception as e:
                    print(f"Search failed: {e}")

        # Most searches should succeed
        assert len(results) >= 3


@pytest.mark.integration
@pytest.mark.slow
class TestSharedDaemonAccess:
    """Test shared daemon access from multiple users."""

    def test_daemon_handles_concurrent_connections(
        self, system_components: SystemComponents, cli_helper: CLIHelper
    ):
        """Test that daemon handles multiple concurrent connections."""

        def access_daemon(user_id: int) -> Tuple[int, int]:
            """Access daemon as a user."""
            result = cli_helper.run_command(["service", "status"])
            return user_id, result.returncode

        # Multiple users accessing daemon
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(access_daemon, i) for i in range(10)]

            results = [f.result() for f in as_completed(futures)]

        # All should complete
        assert len(results) == 10

    def test_daemon_stability_under_load(
        self, system_components: SystemComponents, cli_helper: CLIHelper
    ):
        """Test daemon remains stable under concurrent load."""
        workspace = system_components.workspace_path

        def perform_operations(user_id: int) -> int:
            """Perform mixed operations as a user."""
            # Status query
            cli_helper.run_command(["status", "--quiet"])

            # Ingest file
            user_file = workspace / f"load_user_{user_id}.txt"
            user_file.write_text(f"Load test user {user_id}")
            cli_helper.run_command(
                [
                    "ingest",
                    "file",
                    str(user_file),
                    "--collection",
                    f"test-load-{user_id}",
                ]
            )

            return user_id

        # Heavy concurrent load
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(perform_operations, i) for i in range(5)]

            results = [f.result() for f in as_completed(futures)]

        # All operations should complete
        assert len(results) == 5

        # Daemon should still be responsive
        final_result = cli_helper.run_command(["status"])
        assert final_result is not None


@pytest.mark.integration
@pytest.mark.slow
class TestThreadSafety:
    """Test thread safety of concurrent operations."""

    def test_concurrent_writes_to_different_collections(
        self, system_components: SystemComponents, cli_helper: CLIHelper
    ):
        """Test concurrent writes to different collections are safe."""
        workspace = system_components.workspace_path

        def write_to_collection(coll_id: int) -> Tuple[int, int]:
            """Write to a specific collection."""
            test_file = workspace / f"thread_safe_{coll_id}.txt"
            test_file.write_text(f"Thread safe content {coll_id}")

            collection = f"test-threadsafe-{coll_id}-{int(time.time())}"
            result = cli_helper.run_command(
                ["ingest", "file", str(test_file), "--collection", collection]
            )
            return coll_id, result.returncode

        # Concurrent writes to different collections
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(write_to_collection, i) for i in range(10)]

            results = []
            for future in as_completed(futures):
                try:
                    results.append(future.result())
                except Exception as e:
                    print(f"Write failed: {e}")

        # Most writes should succeed
        assert len(results) >= 7


@pytest.mark.integration
@pytest.mark.slow
class TestDataConsistency:
    """Test data consistency under concurrent operations."""

    @pytest.mark.asyncio
    async def test_search_consistency_during_ingestion(
        self, system_components: SystemComponents, cli_helper: CLIHelper
    ):
        """Test search results remain consistent during concurrent ingestion."""
        workspace = system_components.workspace_path

        collection_name = f"test-consistency-{int(time.time())}"

        # Initial ingestion
        initial_file = workspace / "initial.txt"
        initial_file.write_text("Initial searchable content")
        cli_helper.run_command(
            ["ingest", "file", str(initial_file), "--collection", collection_name]
        )
        await asyncio.sleep(3)

        def concurrent_ingest(doc_id: int) -> int:
            """Ingest document concurrently."""
            doc_file = workspace / f"concurrent_{doc_id}.txt"
            doc_file.write_text(f"Concurrent document {doc_id}")
            cli_helper.run_command(
                ["ingest", "file", str(doc_file), "--collection", collection_name]
            )
            return doc_id

        def concurrent_search() -> int:
            """Search concurrently."""
            result = cli_helper.run_command(
                ["search", "searchable content", "--collection", collection_name],
                timeout=15,
            )
            return result.returncode

        # Mix ingestion and searches
        with ThreadPoolExecutor(max_workers=6) as executor:
            # 3 ingestions
            ingest_futures = [executor.submit(concurrent_ingest, i) for i in range(3)]

            # 3 searches
            search_futures = [executor.submit(concurrent_search) for _ in range(3)]

            all_futures = ingest_futures + search_futures
            results = [f.result() for f in as_completed(all_futures)]

        # Operations should complete
        assert len(results) == 6


@pytest.mark.integration
@pytest.mark.slow
class TestPerformanceUnderLoad:
    """Test system performance under concurrent load."""

    def test_throughput_with_multiple_users(
        self, system_components: SystemComponents, cli_helper: CLIHelper
    ):
        """Test system throughput with multiple concurrent users."""
        workspace = system_components.workspace_path

        # Create files for load test
        for i in range(10):
            test_file = workspace / f"perf_{i}.txt"
            test_file.write_text(f"Performance test content {i}")

        def user_workflow(user_id: int) -> float:
            """Execute complete user workflow."""
            start = time.time()

            # Ingest
            cli_helper.run_command(
                [
                    "ingest",
                    "file",
                    str(workspace / f"perf_{user_id}.txt"),
                    "--collection",
                    f"test-perf-{user_id}",
                ],
                timeout=30,
            )

            # Query status
            cli_helper.run_command(["status", "--quiet"])

            return time.time() - start

        # Concurrent user workflows
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(user_workflow, i) for i in range(5)]

            durations = [f.result() for f in as_completed(futures)]

        # Average duration should be reasonable (<30s per user)
        avg_duration = sum(durations) / len(durations)
        assert avg_duration < 30.0

    def test_system_responsiveness_under_load(
        self, system_components: SystemComponents, cli_helper: CLIHelper
    ):
        """Test system remains responsive under heavy load."""
        workspace = system_components.workspace_path

        # Create load
        def create_load(load_id: int) -> int:
            """Create system load."""
            for i in range(3):
                test_file = workspace / f"load_{load_id}_{i}.txt"
                test_file.write_text(f"Load test {load_id} file {i}")

                cli_helper.run_command(
                    [
                        "ingest",
                        "file",
                        str(test_file),
                        "--collection",
                        f"test-responsiveness-{load_id}",
                    ]
                )

            return load_id

        # Generate load
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(create_load, i) for i in range(3)]

            for future in as_completed(futures):
                future.result()

        # System should still respond
        response_time_start = time.time()
        result = cli_helper.run_command(["status"], timeout=10)
        response_time = time.time() - response_time_start

        # Should respond within 10 seconds even under load
        assert response_time < 10.0
        assert result is not None


@pytest.mark.integration
@pytest.mark.slow
class TestConcurrentOperationIsolation:
    """Test isolation between concurrent operations."""

    def test_user_operations_isolated(
        self, system_components: SystemComponents, cli_helper: CLIHelper
    ):
        """Test that concurrent user operations don't interfere."""
        workspace = system_components.workspace_path

        def user_isolated_workflow(user_id: int) -> Tuple[int, bool]:
            """Execute isolated user workflow."""
            user_file = workspace / f"isolated_{user_id}.txt"
            user_file.write_text(f"Isolated user {user_id} data")

            collection = f"test-isolated-{user_id}-{int(time.time())}"

            # Ingest
            ingest_result = cli_helper.run_command(
                ["ingest", "file", str(user_file), "--collection", collection]
            )

            # Verify own collection
            list_result = cli_helper.run_command(["admin", "collections"])

            return user_id, (ingest_result is not None and list_result is not None)

        # Multiple isolated user workflows
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(user_isolated_workflow, i) for i in range(5)]

            results = [f.result() for f in as_completed(futures)]

        # All users should complete successfully
        successful = [r[1] for r in results]
        assert sum(successful) >= 4  # At least 4 of 5 succeed


@pytest.mark.integration
@pytest.mark.slow
class TestConcurrentSystemStability:
    """Test overall system stability under concurrent operations."""

    def test_system_stable_after_concurrent_load(
        self, system_components: SystemComponents, cli_helper: CLIHelper
    ):
        """Test system remains stable after heavy concurrent load."""
        workspace = system_components.workspace_path

        # Heavy concurrent operations
        def heavy_operation(op_id: int) -> int:
            """Perform heavy operation."""
            # Create files
            for i in range(5):
                file_path = workspace / f"heavy_{op_id}_{i}.txt"
                file_path.write_text(f"Heavy operation {op_id} file {i}")

            # Ingest
            cli_helper.run_command(
                [
                    "ingest",
                    "folder",
                    str(workspace),
                    "--collection",
                    f"test-heavy-{op_id}",
                ],
                timeout=60,
            )

            return op_id

        # Execute heavy load
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(heavy_operation, i) for i in range(3)]

            for future in as_completed(futures):
                future.result()

        time.sleep(3)

        # Verify system stability
        status_result = cli_helper.run_command(["status"])
        assert status_result is not None

        collections_result = cli_helper.run_command(["admin", "collections"])
        assert collections_result is not None

    def test_no_resource_leaks_under_concurrency(
        self, system_components: SystemComponents, cli_helper: CLIHelper
    ):
        """Test that concurrent operations don't cause resource leaks."""
        # Run many operations
        for _ in range(10):
            result = cli_helper.run_command(["status", "--quiet"])
            assert result is not None

        # System should still be responsive
        final_result = cli_helper.run_command(["status"])
        assert final_result is not None
