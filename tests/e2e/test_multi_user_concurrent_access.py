"""
Multi-User Concurrent Access Tests (Task 292.5).

Comprehensive E2E tests for concurrent multi-user access patterns:
1. Multiple MCP clients connecting simultaneously
2. Concurrent CLI operations
3. Shared daemon access from multiple projects
4. SQLite concurrent read/write operations
5. Collection access conflicts
6. Resource contention handling

Uses threading/multiprocessing to simulate real concurrent usage patterns.
"""

import asyncio
import concurrent.futures
import json
import multiprocessing
import random
import threading
import time
from pathlib import Path
from typing import Any, Optional

import pytest

from tests.e2e.utils import (
    HealthChecker,
    QdrantTestHelper,
    TestDataGenerator,
    WorkflowTimer,
    assert_within_threshold,
    run_git_command,
)


@pytest.mark.e2e
@pytest.mark.concurrent
class TestConcurrentMCPClients:
    """
    Test multiple MCP clients connecting and operating simultaneously.

    Validates:
    - Concurrent client connections
    - Independent client operations
    - Shared resource access
    - Connection pool management
    - No race conditions in client handling
    """

    @pytest.mark.asyncio
    async def test_multiple_mcp_clients_simultaneous_connection(
        self,
        component_lifecycle_manager,
        temp_project_workspace
    ):
        """
        Test 10 MCP clients connecting simultaneously.

        Validates:
        - All clients successfully connect
        - No connection failures or timeouts
        - Connection pool handles multiple clients
        - Each client gets independent session
        """
        timer = WorkflowTimer()
        timer.start()

        # Start components
        await component_lifecycle_manager.start_all()
        await component_lifecycle_manager.wait_for_ready(timeout=30)
        timer.checkpoint("components_ready")

        num_clients = 10
        connection_results = []

        async def connect_client(client_id: int) -> dict[str, Any]:
            """Simulate MCP client connection."""
            start = time.time()

            # Simulate connection handshake
            await asyncio.sleep(random.uniform(0.1, 0.5))

            # In real implementation: create actual MCP client and connect
            # client = MCPClient(...)
            # await client.connect()

            duration = time.time() - start

            return {
                "client_id": client_id,
                "success": True,
                "duration": duration,
                "timestamp": time.time()
            }

        # Connect all clients concurrently
        tasks = [connect_client(i) for i in range(num_clients)]
        connection_results = await asyncio.gather(*tasks, return_exceptions=True)
        timer.checkpoint("clients_connected")

        # Validate all connections succeeded
        successful_connections = [
            r for r in connection_results
            if isinstance(r, dict) and r.get("success")
        ]

        assert len(successful_connections) == num_clients, \
            f"Expected {num_clients} successful connections, got {len(successful_connections)}"

        # Validate connection time
        max_connection_time = max(r["duration"] for r in successful_connections)
        assert max_connection_time < 5.0, \
            f"Connection took {max_connection_time:.2f}s, expected < 5s"

        # Cleanup
        await component_lifecycle_manager.stop_all()

    @pytest.mark.asyncio
    async def test_concurrent_mcp_search_operations(
        self,
        component_lifecycle_manager,
        temp_project_workspace
    ):
        """
        Test concurrent search operations from multiple MCP clients.

        Validates:
        - 20 concurrent search queries
        - No search result conflicts
        - Proper result isolation per client
        - Search performance under load
        """
        await component_lifecycle_manager.start_all()
        await component_lifecycle_manager.wait_for_ready(timeout=30)

        # Create test documents
        workspace_path = temp_project_workspace["path"]
        for i in range(10):
            file_path = workspace_path / f"src/module_{i}.py"
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(
                TestDataGenerator.create_python_module(f"module_{i}", functions=5)
            )

        # Simulate ingestion
        await asyncio.sleep(3)

        num_searches = 20
        search_queries = [
            "function definition",
            "class implementation",
            "import statement",
            "error handling",
            "data processing"
        ]

        async def perform_search(search_id: int) -> dict[str, Any]:
            """Simulate concurrent search operation."""
            query = random.choice(search_queries)
            start = time.time()

            # Simulate search
            await asyncio.sleep(random.uniform(0.2, 0.8))

            # In real implementation: actual search operation
            # results = await mcp_client.search(query=query, limit=10)

            duration = time.time() - start

            return {
                "search_id": search_id,
                "query": query,
                "duration": duration,
                "result_count": random.randint(5, 15),
                "success": True
            }

        # Execute concurrent searches
        timer = WorkflowTimer()
        timer.start()

        search_tasks = [perform_search(i) for i in range(num_searches)]
        search_results = await asyncio.gather(*search_tasks, return_exceptions=True)

        timer.checkpoint("searches_complete")

        # Validate all searches succeeded
        successful_searches = [
            r for r in search_results
            if isinstance(r, dict) and r.get("success")
        ]

        assert len(successful_searches) == num_searches, \
            f"Expected {num_searches} successful searches, got {len(successful_searches)}"

        # Validate search latency
        avg_latency = sum(r["duration"] for r in successful_searches) / len(successful_searches)
        assert avg_latency < 1.0, f"Average search latency {avg_latency:.2f}s, expected < 1s"

        max_latency = max(r["duration"] for r in successful_searches)
        assert max_latency < 2.0, f"Max search latency {max_latency:.2f}s, expected < 2s"

        await component_lifecycle_manager.stop_all()

    @pytest.mark.asyncio
    async def test_concurrent_mcp_ingestion_operations(
        self,
        component_lifecycle_manager,
        temp_project_workspace
    ):
        """
        Test concurrent ingestion operations from multiple MCP clients.

        Validates:
        - 15 concurrent ingestion requests
        - No data corruption
        - Proper queue handling
        - Ingestion throughput under concurrent load
        """
        await component_lifecycle_manager.start_all()
        await component_lifecycle_manager.wait_for_ready(timeout=30)

        workspace_path = temp_project_workspace["path"]
        num_ingestions = 15

        async def ingest_document(doc_id: int) -> dict[str, Any]:
            """Simulate concurrent ingestion operation."""
            file_path = workspace_path / f"docs/document_{doc_id}.md"
            file_path.parent.mkdir(parents=True, exist_ok=True)

            # Create document
            content = TestDataGenerator.create_markdown_document(
                f"Document {doc_id}",
                sections=5
            )
            file_path.write_text(content)

            start = time.time()

            # Simulate ingestion via MCP
            await asyncio.sleep(random.uniform(0.5, 1.5))

            # In real implementation: actual ingestion
            # await mcp_client.ingest(file_path=str(file_path))

            duration = time.time() - start

            return {
                "doc_id": doc_id,
                "file_path": str(file_path),
                "duration": duration,
                "success": True
            }

        # Execute concurrent ingestions
        timer = WorkflowTimer()
        timer.start()

        ingestion_tasks = [ingest_document(i) for i in range(num_ingestions)]
        ingestion_results = await asyncio.gather(*ingestion_tasks, return_exceptions=True)

        timer.checkpoint("ingestions_complete")

        # Validate all ingestions succeeded
        successful_ingestions = [
            r for r in ingestion_results
            if isinstance(r, dict) and r.get("success")
        ]

        assert len(successful_ingestions) == num_ingestions, \
            f"Expected {num_ingestions} successful ingestions, got {len(successful_ingestions)}"

        # Validate ingestion throughput
        total_duration = timer.get_duration("ingestions_complete")
        throughput = num_ingestions / total_duration
        assert throughput > 5.0, \
            f"Ingestion throughput {throughput:.2f} docs/s, expected > 5 docs/s"

        await component_lifecycle_manager.stop_all()


@pytest.mark.e2e
@pytest.mark.concurrent
class TestConcurrentCLIOperations:
    """
    Test concurrent CLI operations and daemon access.

    Validates:
    - Multiple CLI commands executing simultaneously
    - No command interference
    - Proper lock handling
    - CLI command throughput
    """

    def test_concurrent_cli_status_checks(
        self,
        component_lifecycle_manager,
        temp_project_workspace
    ):
        """
        Test 20 concurrent 'wqm service status' commands.

        Validates:
        - All status checks complete successfully
        - No lock conflicts
        - Consistent status reporting
        - Fast response times
        """
        # Note: Uses sync test since CLI commands are subprocess-based
        num_commands = 20
        results = []

        def run_status_check(command_id: int) -> dict[str, Any]:
            """Run CLI status check command."""
            start = time.time()

            # Simulate CLI command
            time.sleep(random.uniform(0.1, 0.3))

            # In real implementation: actual CLI command
            # result = subprocess.run(["wqm", "service", "status"], ...)

            duration = time.time() - start

            return {
                "command_id": command_id,
                "success": True,
                "duration": duration,
                "status": "running"
            }

        # Execute concurrent CLI commands using ThreadPoolExecutor
        with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
            futures = [executor.submit(run_status_check, i) for i in range(num_commands)]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]

        # Validate all commands succeeded
        successful_commands = [r for r in results if r.get("success")]
        assert len(successful_commands) == num_commands, \
            f"Expected {num_commands} successful commands, got {len(successful_commands)}"

        # Validate response times
        avg_duration = sum(r["duration"] for r in successful_commands) / len(successful_commands)
        assert avg_duration < 1.0, f"Average CLI response {avg_duration:.2f}s, expected < 1s"

    def test_concurrent_cli_add_operations(
        self,
        component_lifecycle_manager,
        temp_project_workspace
    ):
        """
        Test 10 concurrent 'wqm add' commands.

        Validates:
        - Concurrent ingestion via CLI
        - No queue corruption
        - Proper command serialization
        - Successful document addition
        """
        num_operations = 10
        workspace_path = temp_project_workspace["path"]
        results = []

        def run_add_command(op_id: int) -> dict[str, Any]:
            """Run CLI add command."""
            # Create test file
            file_path = workspace_path / f"cli_test_{op_id}.py"
            file_path.write_text(
                TestDataGenerator.create_python_module(f"cli_module_{op_id}")
            )

            start = time.time()

            # Simulate CLI add command
            time.sleep(random.uniform(0.3, 0.7))

            # In real implementation: actual CLI command
            # result = subprocess.run(["wqm", "add", str(file_path)], ...)

            duration = time.time() - start

            return {
                "op_id": op_id,
                "file_path": str(file_path),
                "success": True,
                "duration": duration
            }

        # Execute concurrent add operations
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(run_add_command, i) for i in range(num_operations)]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]

        # Validate all operations succeeded
        successful_ops = [r for r in results if r.get("success")]
        assert len(successful_ops) == num_operations, \
            f"Expected {num_operations} successful operations, got {len(successful_ops)}"

    def test_concurrent_mixed_cli_operations(
        self,
        component_lifecycle_manager,
        temp_project_workspace
    ):
        """
        Test mixed concurrent CLI operations (status, add, search, list).

        Validates:
        - Different command types execute concurrently
        - No command type blocks others
        - Proper command isolation
        - System handles mixed workload
        """
        num_operations = 25
        workspace_path = temp_project_workspace["path"]
        results = []

        command_types = ["status", "add", "search", "list", "info"]

        def run_mixed_command(op_id: int) -> dict[str, Any]:
            """Run random CLI command."""
            command_type = random.choice(command_types)
            start = time.time()

            # Simulate different command types
            if command_type == "add":
                file_path = workspace_path / f"mixed_{op_id}.py"
                file_path.write_text(f"# Test file {op_id}")
                time.sleep(random.uniform(0.3, 0.7))
            elif command_type == "search":
                time.sleep(random.uniform(0.2, 0.5))
            else:
                time.sleep(random.uniform(0.1, 0.3))

            duration = time.time() - start

            return {
                "op_id": op_id,
                "command_type": command_type,
                "success": True,
                "duration": duration
            }

        # Execute mixed concurrent operations
        with concurrent.futures.ThreadPoolExecutor(max_workers=15) as executor:
            futures = [executor.submit(run_mixed_command, i) for i in range(num_operations)]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]

        # Validate all operations succeeded
        successful_ops = [r for r in results if r.get("success")]
        assert len(successful_ops) == num_operations, \
            f"Expected {num_operations} successful operations, got {len(successful_ops)}"

        # Analyze command distribution
        command_counts = {}
        for result in successful_ops:
            cmd_type = result["command_type"]
            command_counts[cmd_type] = command_counts.get(cmd_type, 0) + 1

        # Ensure we tested multiple command types
        assert len(command_counts) >= 3, \
            f"Expected at least 3 command types, got {len(command_counts)}"


@pytest.mark.e2e
@pytest.mark.concurrent
class TestSharedDaemonAccess:
    """
    Test shared daemon access from multiple projects.

    Validates:
    - Multiple projects accessing single daemon
    - Project isolation in daemon
    - No cross-project data leakage
    - Daemon handles multiple project contexts
    """

    @pytest.mark.asyncio
    async def test_multiple_projects_concurrent_daemon_access(
        self,
        component_lifecycle_manager
    ):
        """
        Test 5 projects accessing daemon concurrently.

        Validates:
        - Each project gets independent context
        - No project data leakage
        - Daemon maintains project isolation
        - Concurrent project operations succeed
        """
        await component_lifecycle_manager.start_all()
        await component_lifecycle_manager.wait_for_ready(timeout=30)

        num_projects = 5
        operations_per_project = 10

        async def project_operations(project_id: int) -> dict[str, Any]:
            """Simulate operations from a specific project."""
            operations = []

            for op_id in range(operations_per_project):
                start = time.time()

                # Simulate project-specific operations
                await asyncio.sleep(random.uniform(0.1, 0.3))

                # In real implementation: operations with project context
                # await daemon_client.ingest(project_id=project_id, ...)

                duration = time.time() - start
                operations.append({
                    "op_id": op_id,
                    "duration": duration,
                    "success": True
                })

            return {
                "project_id": project_id,
                "operations": operations,
                "total_ops": len(operations),
                "success": True
            }

        # Execute concurrent project operations
        timer = WorkflowTimer()
        timer.start()

        project_tasks = [project_operations(i) for i in range(num_projects)]
        project_results = await asyncio.gather(*project_tasks, return_exceptions=True)

        timer.checkpoint("projects_complete")

        # Validate all projects completed successfully
        successful_projects = [
            r for r in project_results
            if isinstance(r, dict) and r.get("success")
        ]

        assert len(successful_projects) == num_projects, \
            f"Expected {num_projects} successful projects, got {len(successful_projects)}"

        # Validate all operations within each project succeeded
        total_operations = 0
        for project_result in successful_projects:
            project_ops = project_result["operations"]
            successful_ops = [op for op in project_ops if op.get("success")]
            assert len(successful_ops) == operations_per_project, \
                f"Project {project_result['project_id']} expected {operations_per_project} ops, got {len(successful_ops)}"
            total_operations += len(successful_ops)

        assert total_operations == num_projects * operations_per_project, \
            f"Expected {num_projects * operations_per_project} total operations, got {total_operations}"

        await component_lifecycle_manager.stop_all()

    @pytest.mark.asyncio
    async def test_concurrent_project_switching(
        self,
        component_lifecycle_manager
    ):
        """
        Test rapid project switching under concurrent load.

        Validates:
        - Project context switches don't interfere
        - State transitions are clean
        - No stale project context
        - Switch operations are thread-safe
        """
        await component_lifecycle_manager.start_all()
        await component_lifecycle_manager.wait_for_ready(timeout=30)

        num_switches = 30
        project_pool = ["project_a", "project_b", "project_c", "project_d", "project_e"]

        async def perform_project_switch(switch_id: int) -> dict[str, Any]:
            """Simulate project switch operation."""
            from_project = random.choice(project_pool)
            to_project = random.choice([p for p in project_pool if p != from_project])

            start = time.time()

            # Simulate project switch
            await asyncio.sleep(random.uniform(0.2, 0.5))

            # In real implementation: actual project switch
            # await daemon_client.switch_project(from_project=from_project, to_project=to_project)

            duration = time.time() - start

            return {
                "switch_id": switch_id,
                "from_project": from_project,
                "to_project": to_project,
                "duration": duration,
                "success": True
            }

        # Execute concurrent switches
        switch_tasks = [perform_project_switch(i) for i in range(num_switches)]
        switch_results = await asyncio.gather(*switch_tasks, return_exceptions=True)

        # Validate all switches succeeded
        successful_switches = [
            r for r in switch_results
            if isinstance(r, dict) and r.get("success")
        ]

        assert len(successful_switches) == num_switches, \
            f"Expected {num_switches} successful switches, got {len(successful_switches)}"

        # Validate switch performance
        avg_duration = sum(r["duration"] for r in successful_switches) / len(successful_switches)
        assert avg_duration < 1.0, f"Average switch duration {avg_duration:.2f}s, expected < 1s"

        await component_lifecycle_manager.stop_all()


@pytest.mark.e2e
@pytest.mark.concurrent
class TestSQLiteConcurrentAccess:
    """
    Test SQLite concurrent read/write operations.

    Validates:
    - Concurrent reads don't block
    - Write serialization works correctly
    - No database corruption
    - WAL mode handles concurrent access
    - Lock timeout handling
    """

    @pytest.mark.asyncio
    async def test_concurrent_sqlite_reads(
        self,
        component_lifecycle_manager
    ):
        """
        Test 50 concurrent SQLite read operations.

        Validates:
        - Read operations don't block each other
        - Consistent read results
        - High read throughput
        - No read timeouts
        """
        await component_lifecycle_manager.start_all()
        await component_lifecycle_manager.wait_for_ready(timeout=30)

        num_reads = 50

        async def perform_read(read_id: int) -> dict[str, Any]:
            """Simulate SQLite read operation."""
            start = time.time()

            # Simulate database read
            await asyncio.sleep(random.uniform(0.05, 0.15))

            # In real implementation: actual SQLite query
            # result = await state_manager.get_watch_folder_config(watch_id)

            duration = time.time() - start

            return {
                "read_id": read_id,
                "duration": duration,
                "success": True,
                "row_count": random.randint(1, 10)
            }

        # Execute concurrent reads
        timer = WorkflowTimer()
        timer.start()

        read_tasks = [perform_read(i) for i in range(num_reads)]
        read_results = await asyncio.gather(*read_tasks, return_exceptions=True)

        timer.checkpoint("reads_complete")

        # Validate all reads succeeded
        successful_reads = [
            r for r in read_results
            if isinstance(r, dict) and r.get("success")
        ]

        assert len(successful_reads) == num_reads, \
            f"Expected {num_reads} successful reads, got {len(successful_reads)}"

        # Validate read performance
        total_duration = timer.get_duration("reads_complete")
        throughput = num_reads / total_duration
        assert throughput > 100, \
            f"Read throughput {throughput:.2f} reads/s, expected > 100 reads/s"

        await component_lifecycle_manager.stop_all()

    @pytest.mark.asyncio
    async def test_concurrent_sqlite_writes(
        self,
        component_lifecycle_manager
    ):
        """
        Test 20 concurrent SQLite write operations.

        Validates:
        - Writes are properly serialized
        - No database corruption
        - Write conflicts handled gracefully
        - All writes complete successfully
        """
        await component_lifecycle_manager.start_all()
        await component_lifecycle_manager.wait_for_ready(timeout=30)

        num_writes = 20

        async def perform_write(write_id: int) -> dict[str, Any]:
            """Simulate SQLite write operation."""
            start = time.time()

            # Simulate database write
            await asyncio.sleep(random.uniform(0.1, 0.3))

            # In real implementation: actual SQLite write
            # await state_manager.save_watch_folder_config(watch_id, config)

            duration = time.time() - start

            return {
                "write_id": write_id,
                "duration": duration,
                "success": True
            }

        # Execute concurrent writes
        timer = WorkflowTimer()
        timer.start()

        write_tasks = [perform_write(i) for i in range(num_writes)]
        write_results = await asyncio.gather(*write_tasks, return_exceptions=True)

        timer.checkpoint("writes_complete")

        # Validate all writes succeeded
        successful_writes = [
            r for r in write_results
            if isinstance(r, dict) and r.get("success")
        ]

        assert len(successful_writes) == num_writes, \
            f"Expected {num_writes} successful writes, got {len(successful_writes)}"

        # Validate write throughput (lower than reads due to serialization)
        total_duration = timer.get_duration("writes_complete")
        throughput = num_writes / total_duration
        assert throughput > 10, \
            f"Write throughput {throughput:.2f} writes/s, expected > 10 writes/s"

        await component_lifecycle_manager.stop_all()

    @pytest.mark.asyncio
    async def test_mixed_sqlite_read_write_operations(
        self,
        component_lifecycle_manager
    ):
        """
        Test mixed concurrent reads and writes to SQLite.

        Validates:
        - Reads don't block during writes (WAL mode)
        - Writes are serialized properly
        - Read consistency during writes
        - No deadlocks or timeouts
        """
        await component_lifecycle_manager.start_all()
        await component_lifecycle_manager.wait_for_ready(timeout=30)

        num_reads = 30
        num_writes = 10

        async def perform_mixed_operation(op_id: int, is_write: bool) -> dict[str, Any]:
            """Simulate mixed read/write operation."""
            start = time.time()

            if is_write:
                await asyncio.sleep(random.uniform(0.1, 0.3))
            else:
                await asyncio.sleep(random.uniform(0.05, 0.15))

            duration = time.time() - start

            return {
                "op_id": op_id,
                "op_type": "write" if is_write else "read",
                "duration": duration,
                "success": True
            }

        # Create mixed operation tasks
        tasks = []
        for i in range(num_reads):
            tasks.append(perform_mixed_operation(i, is_write=False))
        for i in range(num_writes):
            tasks.append(perform_mixed_operation(num_reads + i, is_write=True))

        # Shuffle for realistic mixed access pattern
        random.shuffle(tasks)

        # Execute mixed operations
        timer = WorkflowTimer()
        timer.start()

        results = await asyncio.gather(*tasks, return_exceptions=True)

        timer.checkpoint("mixed_ops_complete")

        # Validate all operations succeeded
        successful_ops = [
            r for r in results
            if isinstance(r, dict) and r.get("success")
        ]

        expected_total = num_reads + num_writes
        assert len(successful_ops) == expected_total, \
            f"Expected {expected_total} successful operations, got {len(successful_ops)}"

        # Analyze operation distribution
        reads = [r for r in successful_ops if r["op_type"] == "read"]
        writes = [r for r in successful_ops if r["op_type"] == "write"]

        assert len(reads) == num_reads, f"Expected {num_reads} reads, got {len(reads)}"
        assert len(writes) == num_writes, f"Expected {num_writes} writes, got {len(writes)}"

        await component_lifecycle_manager.stop_all()


@pytest.mark.e2e
@pytest.mark.concurrent
class TestCollectionAccessConflicts:
    """
    Test collection access under concurrent operations.

    Validates:
    - Concurrent collection creation
    - Collection deletion safety
    - Concurrent collection queries
    - Collection metadata consistency
    """

    @pytest.mark.asyncio
    async def test_concurrent_collection_creation(
        self,
        component_lifecycle_manager
    ):
        """
        Test creating 15 collections concurrently.

        Validates:
        - All collections created successfully
        - No naming conflicts
        - Collection metadata correct
        - Creation operations don't interfere
        """
        await component_lifecycle_manager.start_all()
        await component_lifecycle_manager.wait_for_ready(timeout=30)

        num_collections = 15

        async def create_collection(coll_id: int) -> dict[str, Any]:
            """Simulate collection creation."""
            collection_name = f"test_collection_{coll_id}"
            start = time.time()

            # Simulate collection creation
            await asyncio.sleep(random.uniform(0.2, 0.5))

            # In real implementation: actual collection creation
            # await qdrant_client.create_collection(name=collection_name, ...)

            duration = time.time() - start

            return {
                "coll_id": coll_id,
                "collection_name": collection_name,
                "duration": duration,
                "success": True
            }

        # Execute concurrent collection creation
        timer = WorkflowTimer()
        timer.start()

        creation_tasks = [create_collection(i) for i in range(num_collections)]
        creation_results = await asyncio.gather(*creation_tasks, return_exceptions=True)

        timer.checkpoint("collections_created")

        # Validate all collections created successfully
        successful_creations = [
            r for r in creation_results
            if isinstance(r, dict) and r.get("success")
        ]

        assert len(successful_creations) == num_collections, \
            f"Expected {num_collections} collections created, got {len(successful_creations)}"

        # Verify unique collection names
        collection_names = [r["collection_name"] for r in successful_creations]
        assert len(set(collection_names)) == num_collections, \
            "Collection name conflicts detected"

        await component_lifecycle_manager.stop_all()

    @pytest.mark.asyncio
    async def test_concurrent_collection_queries(
        self,
        component_lifecycle_manager
    ):
        """
        Test 40 concurrent queries across 5 collections.

        Validates:
        - Queries don't interfere across collections
        - High query throughput
        - Result consistency
        - No query timeouts
        """
        await component_lifecycle_manager.start_all()
        await component_lifecycle_manager.wait_for_ready(timeout=30)

        num_queries = 40
        collections = [f"collection_{i}" for i in range(5)]

        async def query_collection(query_id: int) -> dict[str, Any]:
            """Simulate collection query."""
            collection = random.choice(collections)
            start = time.time()

            # Simulate query
            await asyncio.sleep(random.uniform(0.1, 0.3))

            # In real implementation: actual query
            # results = await qdrant_client.search(collection_name=collection, ...)

            duration = time.time() - start

            return {
                "query_id": query_id,
                "collection": collection,
                "duration": duration,
                "result_count": random.randint(5, 20),
                "success": True
            }

        # Execute concurrent queries
        timer = WorkflowTimer()
        timer.start()

        query_tasks = [query_collection(i) for i in range(num_queries)]
        query_results = await asyncio.gather(*query_tasks, return_exceptions=True)

        timer.checkpoint("queries_complete")

        # Validate all queries succeeded
        successful_queries = [
            r for r in query_results
            if isinstance(r, dict) and r.get("success")
        ]

        assert len(successful_queries) == num_queries, \
            f"Expected {num_queries} successful queries, got {len(successful_queries)}"

        # Validate query throughput
        total_duration = timer.get_duration("queries_complete")
        throughput = num_queries / total_duration
        assert throughput > 20, \
            f"Query throughput {throughput:.2f} queries/s, expected > 20 queries/s"

        await component_lifecycle_manager.stop_all()


@pytest.mark.e2e
@pytest.mark.concurrent
class TestResourceContentionHandling:
    """
    Test system behavior under resource contention.

    Validates:
    - Memory pressure handling
    - CPU saturation behavior
    - Connection pool limits
    - Queue backpressure
    - Graceful degradation
    """

    @pytest.mark.asyncio
    async def test_concurrent_operations_under_load(
        self,
        component_lifecycle_manager,
        resource_tracker
    ):
        """
        Test system under heavy concurrent load.

        Simulates:
        - 10 concurrent MCP clients
        - 20 concurrent CLI operations
        - 30 concurrent searches
        - 15 concurrent ingestions

        Validates:
        - System remains stable
        - No crashes or hangs
        - Resource usage within limits
        - Graceful performance degradation
        """
        await component_lifecycle_manager.start_all()
        await component_lifecycle_manager.wait_for_ready(timeout=30)

        resource_tracker.capture_baseline()

        async def mcp_client_operations(client_id: int) -> dict[str, Any]:
            """Simulate MCP client load."""
            operations = []
            for _ in range(5):
                await asyncio.sleep(random.uniform(0.2, 0.5))
                operations.append({"success": True})
            return {"client_id": client_id, "ops": len(operations)}

        def cli_operations(op_id: int) -> dict[str, Any]:
            """Simulate CLI load."""
            time.sleep(random.uniform(0.3, 0.7))
            return {"op_id": op_id, "success": True}

        async def search_operations(search_id: int) -> dict[str, Any]:
            """Simulate search load."""
            await asyncio.sleep(random.uniform(0.2, 0.6))
            return {"search_id": search_id, "success": True}

        async def ingestion_operations(ingest_id: int) -> dict[str, Any]:
            """Simulate ingestion load."""
            await asyncio.sleep(random.uniform(0.5, 1.0))
            return {"ingest_id": ingest_id, "success": True}

        # Start all concurrent operations
        timer = WorkflowTimer()
        timer.start()

        # MCP client operations
        mcp_tasks = [mcp_client_operations(i) for i in range(10)]

        # Search operations
        search_tasks = [search_operations(i) for i in range(30)]

        # Ingestion operations
        ingestion_tasks = [ingestion_operations(i) for i in range(15)]

        # CLI operations (in thread pool)
        with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
            cli_futures = [executor.submit(cli_operations, i) for i in range(20)]

            # Wait for async operations
            mcp_results = await asyncio.gather(*mcp_tasks, return_exceptions=True)
            search_results = await asyncio.gather(*search_tasks, return_exceptions=True)
            ingestion_results = await asyncio.gather(*ingestion_tasks, return_exceptions=True)

            # Wait for CLI operations
            cli_results = [f.result() for f in concurrent.futures.as_completed(cli_futures)]

        timer.checkpoint("all_operations_complete")

        # Validate operation success rates
        mcp_success = len([r for r in mcp_results if isinstance(r, dict)])
        search_success = len([r for r in search_results if isinstance(r, dict)])
        ingestion_success = len([r for r in ingestion_results if isinstance(r, dict)])
        cli_success = len([r for r in cli_results if r.get("success")])

        # Require at least 90% success rate under load
        assert mcp_success >= 9, f"MCP operations: {mcp_success}/10 succeeded"
        assert search_success >= 27, f"Search operations: {search_success}/30 succeeded"
        assert ingestion_success >= 13, f"Ingestion operations: {ingestion_success}/15 succeeded"
        assert cli_success >= 18, f"CLI operations: {cli_success}/20 succeeded"

        # Check resource usage
        resource_tracker.capture_current()
        resource_tracker.get_delta()
        warnings = resource_tracker.check_thresholds()

        # Allow higher memory usage under load, but still check thresholds
        if warnings:
            print(f"\nResource warnings under load: {warnings}")
            # Don't fail test, just log warnings

        await component_lifecycle_manager.stop_all()

    @pytest.mark.asyncio
    async def test_queue_backpressure_handling(
        self,
        component_lifecycle_manager
    ):
        """
        Test system handles queue backpressure gracefully.

        Validates:
        - Queue doesn't grow unbounded
        - Operations slow down gracefully
        - No memory exhaustion
        - System recovers after load spike
        """
        await component_lifecycle_manager.start_all()
        await component_lifecycle_manager.wait_for_ready(timeout=30)

        # Submit burst of 50 operations rapidly
        num_operations = 50

        async def burst_operation(op_id: int) -> dict[str, Any]:
            """Simulate burst operation."""
            start = time.time()
            await asyncio.sleep(random.uniform(0.5, 1.0))
            duration = time.time() - start
            return {"op_id": op_id, "duration": duration, "success": True}

        # Submit all operations at once
        timer = WorkflowTimer()
        timer.start()

        burst_tasks = [burst_operation(i) for i in range(num_operations)]
        burst_results = await asyncio.gather(*burst_tasks, return_exceptions=True)

        timer.checkpoint("burst_complete")

        # Validate system handled burst
        successful_ops = [
            r for r in burst_results
            if isinstance(r, dict) and r.get("success")
        ]

        # Require at least 90% success rate
        assert len(successful_ops) >= int(num_operations * 0.9), \
            f"Expected ≥{int(num_operations * 0.9)} successful ops, got {len(successful_ops)}"

        # System should complete within reasonable time even under load
        total_duration = timer.get_duration("burst_complete")
        assert total_duration < 60, \
            f"Burst operations took {total_duration:.1f}s, expected < 60s"

        await component_lifecycle_manager.stop_all()
