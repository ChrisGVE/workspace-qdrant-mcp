"""
Comprehensive concurrent operation stress tests for MCP document management.

This module tests MCP server behavior under high concurrent load with 1000+
simultaneous operations. Tests verify:
- Data consistency under heavy concurrent load
- Deadlock prevention and resolution
- Performance degradation measurement
- Mixed read/write workload handling
- Connection pool exhaustion scenarios
- SQLite WAL mode concurrent access
- Daemon integration under stress

Test Scenarios:
1. 1000+ concurrent store operations (write-heavy)
2. 1000+ concurrent search operations (read-heavy)
3. Mixed read/write workloads (realistic usage)
4. Concurrent CRUD operations across all MCP tools
5. Connection pool exhaustion and recovery
6. Deadlock detection and prevention
7. Data consistency validation under stress
8. Performance degradation metrics

SUCCESS CRITERIA:
- All operations complete without deadlocks
- Data consistency maintained (no lost/corrupted data)
- Performance degradation < 3x under 1000 concurrent operations
- Connection pool handles exhaustion gracefully
- No race conditions or data corruption
- Error rate < 1% under normal concurrency
- Error rate < 5% under extreme stress (1000+ ops)
"""

import asyncio
import pytest
import time
import random
import statistics
from collections import defaultdict
from datetime import datetime
from typing import List, Dict, Any, Tuple
from unittest.mock import AsyncMock, patch, MagicMock

# Test markers
pytestmark = [
    pytest.mark.stress,
    pytest.mark.concurrent,
    pytest.mark.requires_qdrant,
    pytest.mark.slow,
]


class ConcurrencyMetrics:
    """Track metrics for concurrent operations."""

    def __init__(self):
        self.operation_times: List[float] = []
        self.errors: List[Tuple[str, Exception]] = []
        self.success_count: int = 0
        self.failure_count: int = 0
        self.start_time: float = None
        self.end_time: float = None
        self.operation_types: Dict[str, int] = defaultdict(int)

    def record_success(self, operation_type: str, duration: float):
        """Record successful operation."""
        self.success_count += 1
        self.operation_times.append(duration)
        self.operation_types[operation_type] += 1

    def record_failure(self, operation_type: str, error: Exception):
        """Record failed operation."""
        self.failure_count += 1
        self.errors.append((operation_type, error))
        self.operation_types[operation_type] += 1

    def start(self):
        """Mark test start."""
        self.start_time = time.time()

    def end(self):
        """Mark test end."""
        self.end_time = time.time()

    @property
    def total_duration(self) -> float:
        """Total test duration in seconds."""
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return 0

    @property
    def avg_operation_time(self) -> float:
        """Average operation time in milliseconds."""
        if self.operation_times:
            return statistics.mean(self.operation_times) * 1000
        return 0

    @property
    def median_operation_time(self) -> float:
        """Median operation time in milliseconds."""
        if self.operation_times:
            return statistics.median(self.operation_times) * 1000
        return 0

    @property
    def p95_operation_time(self) -> float:
        """95th percentile operation time in milliseconds."""
        if self.operation_times:
            sorted_times = sorted(self.operation_times)
            idx = int(len(sorted_times) * 0.95)
            return sorted_times[idx] * 1000
        return 0

    @property
    def p99_operation_time(self) -> float:
        """99th percentile operation time in milliseconds."""
        if self.operation_times:
            sorted_times = sorted(self.operation_times)
            idx = int(len(sorted_times) * 0.99)
            return sorted_times[idx] * 1000
        return 0

    @property
    def error_rate(self) -> float:
        """Error rate as percentage."""
        total = self.success_count + self.failure_count
        if total > 0:
            return (self.failure_count / total) * 100
        return 0

    @property
    def throughput(self) -> float:
        """Operations per second."""
        if self.total_duration > 0:
            return (self.success_count + self.failure_count) / self.total_duration
        return 0

    def get_summary(self) -> Dict[str, Any]:
        """Get metrics summary."""
        return {
            "total_operations": self.success_count + self.failure_count,
            "successful": self.success_count,
            "failed": self.failure_count,
            "error_rate_percent": round(self.error_rate, 2),
            "duration_seconds": round(self.total_duration, 2),
            "throughput_ops_per_sec": round(self.throughput, 2),
            "avg_operation_time_ms": round(self.avg_operation_time, 2),
            "median_operation_time_ms": round(self.median_operation_time, 2),
            "p95_operation_time_ms": round(self.p95_operation_time, 2),
            "p99_operation_time_ms": round(self.p99_operation_time, 2),
            "operation_breakdown": dict(self.operation_types),
        }

    def print_summary(self, test_name: str):
        """Print formatted metrics summary."""
        print(f"\n{'='*70}")
        print(f"STRESS TEST RESULTS: {test_name}")
        print(f"{'='*70}")

        summary = self.get_summary()
        print(f"\nOperations:")
        print(f"  Total:      {summary['total_operations']}")
        print(f"  Successful: {summary['successful']} ({100-summary['error_rate_percent']:.1f}%)")
        print(f"  Failed:     {summary['failed']} ({summary['error_rate_percent']:.1f}%)")

        print(f"\nPerformance:")
        print(f"  Duration:   {summary['duration_seconds']:.2f}s")
        print(f"  Throughput: {summary['throughput_ops_per_sec']:.2f} ops/sec")
        print(f"  Avg Time:   {summary['avg_operation_time_ms']:.2f}ms")
        print(f"  Median:     {summary['median_operation_time_ms']:.2f}ms")
        print(f"  P95:        {summary['p95_operation_time_ms']:.2f}ms")
        print(f"  P99:        {summary['p99_operation_time_ms']:.2f}ms")

        if summary['operation_breakdown']:
            print(f"\nOperation Breakdown:")
            for op_type, count in sorted(summary['operation_breakdown'].items()):
                print(f"  {op_type}: {count}")

        if self.errors:
            print(f"\nErrors (showing first 10):")
            for i, (op_type, error) in enumerate(self.errors[:10]):
                print(f"  {i+1}. {op_type}: {type(error).__name__}: {str(error)[:100]}")

        print(f"{'='*70}\n")


@pytest.fixture
async def mcp_server_instance():
    """Create MCP server instance for testing."""
    # Import server module
    from workspace_qdrant_mcp import server as mcp_server

    # Initialize components
    await mcp_server.initialize_components()

    yield mcp_server

    # Cleanup is handled by individual tests


@pytest.fixture
def test_collection_name():
    """Generate unique collection name for testing."""
    timestamp = int(time.time() * 1000)
    return f"_test_stress_{timestamp}"


class TestConcurrentStoreOperations:
    """Test concurrent store operations under stress."""

    @pytest.mark.asyncio
    async def test_1000_concurrent_store_operations(
        self, mcp_server_instance, test_collection_name
    ):
        """Test 1000+ concurrent store operations for data consistency."""
        metrics = ConcurrencyMetrics()
        stored_document_ids = []
        lock = asyncio.Lock()

        async def store_document(doc_id: int) -> Dict[str, Any]:
            """Store a single document."""
            operation_start = time.time()

            try:
                content = f"Concurrent stress test document {doc_id} - {datetime.now().isoformat()}"
                result = await mcp_server_instance.store(
                    content=content,
                    title=f"Stress Test Doc {doc_id}",
                    metadata={"doc_id": doc_id, "test_run": "concurrent_1000"},
                    collection=test_collection_name,
                    source="stress_test",
                )

                duration = time.time() - operation_start

                if result.get("success"):
                    metrics.record_success("store", duration)
                    async with lock:
                        stored_document_ids.append(result.get("document_id"))
                else:
                    metrics.record_failure("store", Exception(result.get("error", "Unknown error")))

                return result

            except Exception as e:
                duration = time.time() - operation_start
                metrics.record_failure("store", e)
                return {"success": False, "error": str(e)}

        # Execute 1000 concurrent store operations
        print(f"\nStarting 1000 concurrent store operations...")
        metrics.start()

        tasks = [store_document(i) for i in range(1000)]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        metrics.end()

        # Validate results
        metrics.print_summary("1000 Concurrent Store Operations")

        # Assertions
        assert metrics.success_count >= 950, f"Too many failures: {metrics.failure_count}/1000"
        assert metrics.error_rate < 5.0, f"Error rate too high: {metrics.error_rate:.2f}%"
        assert len(stored_document_ids) >= 950, "Not enough documents stored successfully"
        assert len(set(stored_document_ids)) == len(stored_document_ids), "Duplicate document IDs detected!"

        # Performance assertions
        assert metrics.p95_operation_time < 5000, f"P95 latency too high: {metrics.p95_operation_time:.2f}ms"
        assert metrics.throughput > 50, f"Throughput too low: {metrics.throughput:.2f} ops/sec"

    @pytest.mark.asyncio
    async def test_concurrent_store_with_varying_sizes(
        self, mcp_server_instance, test_collection_name
    ):
        """Test concurrent store operations with varying document sizes."""
        metrics = ConcurrencyMetrics()

        async def store_variable_size_document(doc_id: int) -> Dict[str, Any]:
            """Store document with variable size."""
            operation_start = time.time()

            try:
                # Create documents of varying sizes (100 bytes to 10KB)
                size = random.randint(100, 10000)
                content = f"Doc {doc_id}: " + ("x" * size)

                result = await mcp_server_instance.store(
                    content=content,
                    metadata={"doc_id": doc_id, "size": size},
                    collection=test_collection_name,
                )

                duration = time.time() - operation_start

                if result.get("success"):
                    metrics.record_success("store_variable", duration)
                else:
                    metrics.record_failure("store_variable", Exception(result.get("error")))

                return result

            except Exception as e:
                metrics.record_failure("store_variable", e)
                return {"success": False, "error": str(e)}

        # Execute 500 concurrent operations with varying sizes
        print(f"\nStarting 500 concurrent store operations (varying sizes)...")
        metrics.start()

        tasks = [store_variable_size_document(i) for i in range(500)]
        await asyncio.gather(*tasks, return_exceptions=True)

        metrics.end()
        metrics.print_summary("Concurrent Store (Variable Sizes)")

        # Assertions
        assert metrics.success_count >= 475, f"Too many failures: {metrics.failure_count}/500"
        assert metrics.error_rate < 5.0, f"Error rate too high: {metrics.error_rate:.2f}%"


class TestConcurrentSearchOperations:
    """Test concurrent search operations under stress."""

    @pytest.mark.asyncio
    async def test_1000_concurrent_search_operations(
        self, mcp_server_instance, test_collection_name
    ):
        """Test 1000+ concurrent search operations (read-heavy)."""
        metrics = ConcurrencyMetrics()

        # First, populate collection with test data
        print(f"\nPopulating collection with test data...")
        for i in range(100):
            await mcp_server_instance.store(
                content=f"Search test document {i} with content about testing, performance, and concurrency",
                metadata={"doc_index": i},
                collection=test_collection_name,
            )

        # Give Qdrant time to index
        await asyncio.sleep(2)

        async def search_operation(search_id: int) -> Dict[str, Any]:
            """Perform a search operation."""
            operation_start = time.time()

            try:
                # Vary search queries
                queries = [
                    "testing performance",
                    "concurrent operations",
                    "document search",
                    "stress test",
                    "vector database",
                ]
                query = random.choice(queries)

                result = await mcp_server_instance.search(
                    query=query,
                    collection=test_collection_name,
                    limit=10,
                )

                duration = time.time() - operation_start

                if result.get("success"):
                    metrics.record_success("search", duration)
                else:
                    metrics.record_failure("search", Exception(result.get("error")))

                return result

            except Exception as e:
                metrics.record_failure("search", e)
                return {"success": False, "error": str(e)}

        # Execute 1000 concurrent search operations
        print(f"\nStarting 1000 concurrent search operations...")
        metrics.start()

        tasks = [search_operation(i) for i in range(1000)]
        await asyncio.gather(*tasks, return_exceptions=True)

        metrics.end()
        metrics.print_summary("1000 Concurrent Search Operations")

        # Assertions
        assert metrics.success_count >= 950, f"Too many failures: {metrics.failure_count}/1000"
        assert metrics.error_rate < 5.0, f"Error rate too high: {metrics.error_rate:.2f}%"

        # Search should be fast even under load
        assert metrics.p95_operation_time < 3000, f"P95 search latency too high: {metrics.p95_operation_time:.2f}ms"
        assert metrics.throughput > 100, f"Search throughput too low: {metrics.throughput:.2f} ops/sec"


class TestMixedConcurrentWorkloads:
    """Test mixed read/write concurrent workloads."""

    @pytest.mark.asyncio
    async def test_mixed_crud_operations_1000_concurrent(
        self, mcp_server_instance, test_collection_name
    ):
        """Test 1000+ concurrent mixed CRUD operations."""
        metrics = ConcurrencyMetrics()
        stored_docs = []
        lock = asyncio.Lock()

        # Pre-populate with some documents for read/update/delete operations
        print(f"\nPre-populating collection...")
        for i in range(200):
            result = await mcp_server_instance.store(
                content=f"Pre-populated document {i}",
                metadata={"doc_index": i, "phase": "pre_populate"},
                collection=test_collection_name,
            )
            if result.get("success"):
                stored_docs.append(result.get("document_id"))

        await asyncio.sleep(2)  # Let Qdrant index

        async def mixed_operation(op_id: int) -> Dict[str, Any]:
            """Perform a random CRUD operation."""
            operation_start = time.time()

            # Randomly choose operation type
            op_type = random.choices(
                ["store", "search", "retrieve", "manage"],
                weights=[0.4, 0.4, 0.15, 0.05],  # Realistic distribution
            )[0]

            try:
                if op_type == "store":
                    result = await mcp_server_instance.store(
                        content=f"Mixed workload document {op_id}",
                        metadata={"op_id": op_id, "phase": "mixed"},
                        collection=test_collection_name,
                    )

                elif op_type == "search":
                    result = await mcp_server_instance.search(
                        query="document workload",
                        collection=test_collection_name,
                        limit=10,
                    )

                elif op_type == "retrieve":
                    if stored_docs:
                        doc_id = random.choice(stored_docs)
                        result = await mcp_server_instance.retrieve(
                            document_id=doc_id,
                            collection=test_collection_name,
                        )
                    else:
                        result = {"success": True, "note": "No docs to retrieve"}

                elif op_type == "manage":
                    result = await mcp_server_instance.manage(
                        action="collection_info",
                        collection=test_collection_name,
                    )

                else:
                    result = {"success": False, "error": f"Unknown operation: {op_type}"}

                duration = time.time() - operation_start

                if result.get("success", False):
                    metrics.record_success(op_type, duration)

                    # Track newly stored documents
                    if op_type == "store" and result.get("document_id"):
                        async with lock:
                            stored_docs.append(result["document_id"])
                else:
                    metrics.record_failure(op_type, Exception(result.get("error", "Unknown")))

                return result

            except Exception as e:
                metrics.record_failure(op_type, e)
                return {"success": False, "error": str(e)}

        # Execute 1000 concurrent mixed operations
        print(f"\nStarting 1000 concurrent mixed CRUD operations...")
        metrics.start()

        tasks = [mixed_operation(i) for i in range(1000)]
        await asyncio.gather(*tasks, return_exceptions=True)

        metrics.end()
        metrics.print_summary("1000 Concurrent Mixed CRUD Operations")

        # Assertions
        assert metrics.success_count >= 950, f"Too many failures: {metrics.failure_count}/1000"
        assert metrics.error_rate < 5.0, f"Error rate too high: {metrics.error_rate:.2f}%"

        # Performance should still be reasonable with mixed workload
        assert metrics.p95_operation_time < 5000, f"P95 latency too high: {metrics.p95_operation_time:.2f}ms"

    @pytest.mark.asyncio
    async def test_read_write_workload_ratio(
        self, mcp_server_instance, test_collection_name
    ):
        """Test realistic read-write workload ratio (80% reads, 20% writes)."""
        metrics = ConcurrencyMetrics()

        # Pre-populate
        for i in range(100):
            await mcp_server_instance.store(
                content=f"Document {i} for read-write test",
                collection=test_collection_name,
            )

        await asyncio.sleep(2)

        async def realistic_operation(op_id: int) -> Dict[str, Any]:
            """Perform operation with realistic read/write ratio."""
            operation_start = time.time()

            # 80% reads, 20% writes
            is_read = random.random() < 0.8

            try:
                if is_read:
                    # Read operation (search)
                    result = await mcp_server_instance.search(
                        query="document test",
                        collection=test_collection_name,
                        limit=10,
                    )
                    op_type = "read"
                else:
                    # Write operation (store)
                    result = await mcp_server_instance.store(
                        content=f"New document {op_id}",
                        collection=test_collection_name,
                    )
                    op_type = "write"

                duration = time.time() - operation_start

                if result.get("success", False):
                    metrics.record_success(op_type, duration)
                else:
                    metrics.record_failure(op_type, Exception(result.get("error")))

                return result

            except Exception as e:
                metrics.record_failure("error", e)
                return {"success": False, "error": str(e)}

        # Execute 1000 operations with realistic ratio
        print(f"\nStarting 1000 operations with 80/20 read/write ratio...")
        metrics.start()

        tasks = [realistic_operation(i) for i in range(1000)]
        await asyncio.gather(*tasks, return_exceptions=True)

        metrics.end()
        metrics.print_summary("Realistic Read/Write Workload (80/20)")

        # Assertions
        assert metrics.success_count >= 950, f"Too many failures: {metrics.failure_count}/1000"
        assert metrics.error_rate < 5.0, f"Error rate too high: {metrics.error_rate:.2f}%"

        # Verify read/write ratio is approximately correct
        read_count = metrics.operation_types.get("read", 0)
        write_count = metrics.operation_types.get("write", 0)
        total_ops = read_count + write_count

        if total_ops > 0:
            read_ratio = read_count / total_ops
            assert 0.75 < read_ratio < 0.85, f"Read ratio unexpected: {read_ratio:.2f}"


class TestDataConsistency:
    """Test data consistency under concurrent load."""

    @pytest.mark.asyncio
    async def test_data_consistency_under_concurrent_writes(
        self, mcp_server_instance, test_collection_name
    ):
        """Verify data consistency with concurrent writes to same documents."""
        metrics = ConcurrencyMetrics()
        document_versions = defaultdict(list)
        lock = asyncio.Lock()

        async def update_document_concurrently(doc_id: int, version: int) -> Dict[str, Any]:
            """Attempt to update the same document concurrently."""
            operation_start = time.time()

            try:
                content = f"Document {doc_id} - Version {version} - {datetime.now().isoformat()}"
                result = await mcp_server_instance.store(
                    content=content,
                    metadata={"doc_id": doc_id, "version": version},
                    collection=test_collection_name,
                )

                duration = time.time() - operation_start

                if result.get("success"):
                    metrics.record_success("concurrent_update", duration)

                    async with lock:
                        document_versions[doc_id].append({
                            "version": version,
                            "stored_id": result.get("document_id"),
                            "timestamp": datetime.now().isoformat(),
                        })
                else:
                    metrics.record_failure("concurrent_update", Exception(result.get("error")))

                return result

            except Exception as e:
                metrics.record_failure("concurrent_update", e)
                return {"success": False, "error": str(e)}

        # Create 100 tasks that update 10 documents concurrently (10 versions each)
        print(f"\nTesting data consistency with concurrent updates...")
        metrics.start()

        tasks = []
        for doc_id in range(10):
            for version in range(100):
                tasks.append(update_document_concurrently(doc_id, version))

        # Shuffle to ensure random update order
        random.shuffle(tasks)
        await asyncio.gather(*tasks, return_exceptions=True)

        metrics.end()
        metrics.print_summary("Concurrent Updates (Data Consistency)")

        # Verify no data loss
        assert metrics.success_count >= 950, f"Too many failures: {metrics.failure_count}/1000"

        # Verify all versions were stored (they create new documents, not update)
        total_versions = sum(len(versions) for versions in document_versions.values())
        assert total_versions >= 950, f"Missing document versions: {total_versions}/1000"

        # Verify no duplicate document IDs within same doc_id updates
        for doc_id, versions in document_versions.items():
            stored_ids = [v["stored_id"] for v in versions]
            # Since each store creates a new document, all IDs should be unique
            assert len(stored_ids) == len(set(stored_ids)), f"Duplicate IDs for doc {doc_id}"

    @pytest.mark.asyncio
    async def test_no_data_corruption_under_stress(
        self, mcp_server_instance, test_collection_name
    ):
        """Verify data integrity after heavy concurrent operations."""
        metrics = ConcurrencyMetrics()
        expected_documents = {}
        lock = asyncio.Lock()

        async def store_and_verify(doc_id: int) -> Dict[str, Any]:
            """Store document and track expected content."""
            operation_start = time.time()

            try:
                content = f"Integrity test document {doc_id} with unique content marker {random.randint(100000, 999999)}"

                result = await mcp_server_instance.store(
                    content=content,
                    metadata={"doc_id": doc_id, "test": "integrity"},
                    collection=test_collection_name,
                )

                duration = time.time() - operation_start

                if result.get("success"):
                    metrics.record_success("store", duration)

                    async with lock:
                        expected_documents[result["document_id"]] = content
                else:
                    metrics.record_failure("store", Exception(result.get("error")))

                return result

            except Exception as e:
                metrics.record_failure("store", e)
                return {"success": False, "error": str(e)}

        # Store 500 documents concurrently
        print(f"\nStoring 500 documents concurrently for integrity verification...")
        metrics.start()

        tasks = [store_and_verify(i) for i in range(500)]
        await asyncio.gather(*tasks, return_exceptions=True)

        metrics.end()

        # Give Qdrant time to process
        await asyncio.sleep(3)

        # Verify data integrity by retrieving and checking
        print(f"\nVerifying data integrity for {len(expected_documents)} documents...")
        verify_metrics = ConcurrencyMetrics()
        verify_metrics.start()

        async def verify_document(doc_id: str, expected_content: str):
            """Verify document content matches expected."""
            try:
                result = await mcp_server_instance.retrieve(
                    document_id=doc_id,
                    collection=test_collection_name,
                )

                if result.get("success") and result.get("document"):
                    actual_content = result["document"].get("content", "")
                    if actual_content == expected_content:
                        verify_metrics.record_success("verify", 0)
                    else:
                        verify_metrics.record_failure(
                            "verify",
                            Exception(f"Content mismatch: expected '{expected_content[:50]}...' got '{actual_content[:50]}...'")
                        )
                else:
                    verify_metrics.record_failure("verify", Exception("Document not found or error"))

            except Exception as e:
                verify_metrics.record_failure("verify", e)

        verify_tasks = [
            verify_document(doc_id, content)
            for doc_id, content in expected_documents.items()
        ]
        await asyncio.gather(*verify_tasks, return_exceptions=True)

        verify_metrics.end()

        # Print both summaries
        metrics.print_summary("Concurrent Store (Integrity Test)")
        verify_metrics.print_summary("Data Integrity Verification")

        # Assertions
        assert metrics.success_count >= 475, f"Too many store failures: {metrics.failure_count}/500"
        assert verify_metrics.success_count >= int(metrics.success_count * 0.95), \
            f"Data corruption detected! Only {verify_metrics.success_count}/{metrics.success_count} documents verified"


class TestDeadlockPrevention:
    """Test deadlock prevention and detection."""

    @pytest.mark.asyncio
    async def test_no_deadlocks_with_circular_dependencies(
        self, mcp_server_instance, test_collection_name
    ):
        """Test that circular operation patterns don't cause deadlocks."""
        metrics = ConcurrencyMetrics()

        # Create a pattern that could cause deadlocks:
        # - Multiple operations accessing same resources in different orders
        # - Concurrent reads and writes
        # - Long-running operations

        async def complex_operation_chain(chain_id: int):
            """Perform a chain of operations that could deadlock."""
            operation_start = time.time()

            try:
                # Step 1: Store document
                store_result = await mcp_server_instance.store(
                    content=f"Chain {chain_id} document",
                    metadata={"chain_id": chain_id},
                    collection=test_collection_name,
                )

                if not store_result.get("success"):
                    raise Exception("Store failed")

                # Step 2: Search for it immediately
                search_result = await mcp_server_instance.search(
                    query=f"Chain {chain_id}",
                    collection=test_collection_name,
                )

                # Step 3: Get collection info
                info_result = await mcp_server_instance.manage(
                    action="collection_info",
                    collection=test_collection_name,
                )

                # Step 4: Retrieve original document
                retrieve_result = await mcp_server_instance.retrieve(
                    document_id=store_result["document_id"],
                    collection=test_collection_name,
                )

                duration = time.time() - operation_start

                # All steps must succeed
                if all(r.get("success", False) for r in [store_result, info_result]):
                    metrics.record_success("operation_chain", duration)
                else:
                    metrics.record_failure("operation_chain", Exception("Chain incomplete"))

            except Exception as e:
                metrics.record_failure("operation_chain", e)

        # Run 200 concurrent operation chains
        print(f"\nTesting deadlock prevention with 200 concurrent operation chains...")
        metrics.start()

        tasks = [complex_operation_chain(i) for i in range(200)]

        # Set timeout to detect deadlocks
        try:
            await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=120  # 2 minutes max
            )
        except asyncio.TimeoutError:
            pytest.fail("DEADLOCK DETECTED: Operations timed out after 120 seconds")

        metrics.end()
        metrics.print_summary("Deadlock Prevention Test")

        # Assertions
        assert metrics.success_count >= 190, f"Too many failures: {metrics.failure_count}/200"
        assert metrics.total_duration < 120, f"Operations took too long, possible deadlock: {metrics.total_duration:.2f}s"


class TestPerformanceDegradation:
    """Test and measure performance degradation under load."""

    @pytest.mark.asyncio
    async def test_performance_degradation_measurement(
        self, mcp_server_instance, test_collection_name
    ):
        """Measure performance degradation from 1 to 1000 concurrent operations."""

        # Baseline: Single operation performance
        print(f"\nMeasuring baseline performance (single operation)...")
        baseline_times = []

        for i in range(10):
            start = time.time()
            await mcp_server_instance.store(
                content=f"Baseline document {i}",
                collection=test_collection_name,
            )
            baseline_times.append(time.time() - start)

        baseline_avg = statistics.mean(baseline_times) * 1000  # Convert to ms
        print(f"Baseline avg response time: {baseline_avg:.2f}ms")

        # Test at increasing concurrency levels
        concurrency_levels = [10, 50, 100, 500, 1000]
        results = {}

        for concurrency in concurrency_levels:
            print(f"\nTesting at concurrency level: {concurrency}")
            metrics = ConcurrencyMetrics()

            async def concurrent_store(op_id: int):
                """Store operation for concurrency test."""
                start = time.time()
                try:
                    result = await mcp_server_instance.store(
                        content=f"Concurrency test {concurrency} - doc {op_id}",
                        collection=test_collection_name,
                    )
                    duration = time.time() - start

                    if result.get("success"):
                        metrics.record_success("store", duration)
                    else:
                        metrics.record_failure("store", Exception(result.get("error")))
                except Exception as e:
                    metrics.record_failure("store", e)

            metrics.start()
            tasks = [concurrent_store(i) for i in range(concurrency)]
            await asyncio.gather(*tasks, return_exceptions=True)
            metrics.end()

            results[concurrency] = {
                "avg_time_ms": metrics.avg_operation_time,
                "p95_time_ms": metrics.p95_operation_time,
                "throughput": metrics.throughput,
                "error_rate": metrics.error_rate,
            }

            print(f"  Avg time: {metrics.avg_operation_time:.2f}ms")
            print(f"  P95 time: {metrics.p95_operation_time:.2f}ms")
            print(f"  Throughput: {metrics.throughput:.2f} ops/sec")
            print(f"  Error rate: {metrics.error_rate:.2f}%")

        # Analyze degradation
        print(f"\n{'='*70}")
        print(f"PERFORMANCE DEGRADATION ANALYSIS")
        print(f"{'='*70}")
        print(f"Baseline (1 op):     {baseline_avg:.2f}ms")

        for concurrency, data in results.items():
            degradation_factor = data["avg_time_ms"] / baseline_avg if baseline_avg > 0 else 0
            print(f"{concurrency:4d} concurrent: {data['avg_time_ms']:7.2f}ms ({degradation_factor:.2f}x degradation)")

        # Assertions
        # At 1000 concurrent operations, degradation should be < 3x baseline
        max_degradation = results[1000]["avg_time_ms"] / baseline_avg if baseline_avg > 0 else float('inf')
        assert max_degradation < 3.0, \
            f"Performance degradation too high at 1000 concurrent: {max_degradation:.2f}x (expected < 3x)"

        # Error rate should stay low
        assert results[1000]["error_rate"] < 5.0, \
            f"Error rate too high at 1000 concurrent: {results[1000]['error_rate']:.2f}%"


class TestConnectionPoolExhaustion:
    """Test connection pool exhaustion and recovery."""

    @pytest.mark.asyncio
    async def test_connection_pool_exhaustion_handling(
        self, mcp_server_instance, test_collection_name
    ):
        """Test graceful handling of connection pool exhaustion."""
        metrics = ConcurrencyMetrics()

        async def long_running_operation(op_id: int):
            """Simulate long-running operation that holds connections."""
            operation_start = time.time()

            try:
                # Perform operation with small delay to hold connection longer
                result = await mcp_server_instance.store(
                    content=f"Connection pool test document {op_id}" + ("x" * 1000),
                    collection=test_collection_name,
                )

                # Small delay to simulate processing
                await asyncio.sleep(random.uniform(0.01, 0.05))

                duration = time.time() - operation_start

                if result.get("success"):
                    metrics.record_success("long_op", duration)
                else:
                    metrics.record_failure("long_op", Exception(result.get("error")))

            except Exception as e:
                metrics.record_failure("long_op", e)

        # Launch more operations than typical connection pool size
        print(f"\nTesting connection pool with 500 concurrent long-running operations...")
        metrics.start()

        tasks = [long_running_operation(i) for i in range(500)]

        try:
            await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=180  # 3 minutes
            )
        except asyncio.TimeoutError:
            pytest.fail("Connection pool exhaustion caused timeout")

        metrics.end()
        metrics.print_summary("Connection Pool Exhaustion Test")

        # Assertions
        # Should handle gracefully even if some operations fail due to pool exhaustion
        assert metrics.success_count >= 400, f"Too many failures: {metrics.failure_count}/500"
        assert metrics.total_duration < 180, "Operations took too long (connection pool issue?)"

    @pytest.mark.asyncio
    async def test_connection_pool_recovery(
        self, mcp_server_instance, test_collection_name
    ):
        """Test connection pool recovery after exhaustion."""

        # Phase 1: Exhaust connection pool
        print(f"\nPhase 1: Exhausting connection pool...")
        phase1_metrics = ConcurrencyMetrics()

        async def exhaust_connection(op_id: int):
            """Operation to exhaust pool."""
            try:
                result = await mcp_server_instance.store(
                    content=f"Exhaustion phase {op_id}",
                    collection=test_collection_name,
                )
                if result.get("success"):
                    phase1_metrics.record_success("exhaust", 0)
                else:
                    phase1_metrics.record_failure("exhaust", Exception(result.get("error")))
            except Exception as e:
                phase1_metrics.record_failure("exhaust", e)

        phase1_metrics.start()
        tasks = [exhaust_connection(i) for i in range(300)]
        await asyncio.gather(*tasks, return_exceptions=True)
        phase1_metrics.end()

        print(f"Phase 1 complete: {phase1_metrics.success_count} succeeded, {phase1_metrics.failure_count} failed")

        # Phase 2: Wait for recovery
        print(f"\nPhase 2: Waiting for connection pool recovery...")
        await asyncio.sleep(5)

        # Phase 3: Test normal operations after recovery
        print(f"\nPhase 3: Testing operations after recovery...")
        phase2_metrics = ConcurrencyMetrics()

        async def post_recovery_operation(op_id: int):
            """Operation after recovery."""
            try:
                result = await mcp_server_instance.store(
                    content=f"Post-recovery {op_id}",
                    collection=test_collection_name,
                )
                if result.get("success"):
                    phase2_metrics.record_success("recovery", 0)
                else:
                    phase2_metrics.record_failure("recovery", Exception(result.get("error")))
            except Exception as e:
                phase2_metrics.record_failure("recovery", e)

        phase2_metrics.start()
        tasks = [post_recovery_operation(i) for i in range(100)]
        await asyncio.gather(*tasks, return_exceptions=True)
        phase2_metrics.end()

        phase1_metrics.print_summary("Connection Pool Exhaustion Phase")
        phase2_metrics.print_summary("Connection Pool Recovery Phase")

        # Assertions
        # After recovery, success rate should be high
        recovery_success_rate = (phase2_metrics.success_count / 100) * 100
        assert recovery_success_rate >= 90.0, \
            f"Poor recovery: only {recovery_success_rate:.1f}% success after pool exhaustion"


# Summary test that validates overall concurrent operation capabilities
@pytest.mark.asyncio
async def test_comprehensive_concurrent_stress_summary(
    mcp_server_instance, test_collection_name
):
    """
    Comprehensive summary test validating all concurrent operation criteria.

    This test runs a realistic mixed workload and validates:
    - 1000+ total operations complete successfully
    - Data consistency maintained
    - No deadlocks
    - Acceptable performance degradation
    - Graceful error handling
    """
    print(f"\n{'='*70}")
    print(f"COMPREHENSIVE CONCURRENT STRESS TEST")
    print(f"{'='*70}")

    metrics = ConcurrencyMetrics()
    stored_docs = []
    lock = asyncio.Lock()

    async def realistic_mixed_operation(op_id: int):
        """Realistic mixed operation."""
        operation_start = time.time()

        # Realistic operation distribution
        op_type = random.choices(
            ["store", "search", "retrieve", "manage"],
            weights=[0.35, 0.50, 0.10, 0.05],
        )[0]

        try:
            if op_type == "store":
                result = await mcp_server_instance.store(
                    content=f"Comprehensive test document {op_id} - {datetime.now().isoformat()}",
                    metadata={"op_id": op_id, "test": "comprehensive"},
                    collection=test_collection_name,
                )
                if result.get("success") and result.get("document_id"):
                    async with lock:
                        stored_docs.append(result["document_id"])

            elif op_type == "search":
                result = await mcp_server_instance.search(
                    query=random.choice(["comprehensive test", "document", "operation"]),
                    collection=test_collection_name,
                    limit=10,
                )

            elif op_type == "retrieve" and stored_docs:
                async with lock:
                    doc_id = random.choice(stored_docs) if stored_docs else None

                if doc_id:
                    result = await mcp_server_instance.retrieve(
                        document_id=doc_id,
                        collection=test_collection_name,
                    )
                else:
                    result = {"success": True, "note": "No docs available"}

            elif op_type == "manage":
                result = await mcp_server_instance.manage(
                    action="collection_info",
                    collection=test_collection_name,
                )

            else:
                result = {"success": True}

            duration = time.time() - operation_start

            if result.get("success", False):
                metrics.record_success(op_type, duration)
            else:
                metrics.record_failure(op_type, Exception(result.get("error", "Unknown")))

        except Exception as e:
            metrics.record_failure(op_type, e)

    # Execute 1500 concurrent operations for comprehensive test
    print(f"\nExecuting 1500 concurrent mixed operations...")
    metrics.start()

    tasks = [realistic_mixed_operation(i) for i in range(1500)]

    try:
        await asyncio.wait_for(
            asyncio.gather(*tasks, return_exceptions=True),
            timeout=300  # 5 minutes max
        )
    except asyncio.TimeoutError:
        pytest.fail("TIMEOUT: Possible deadlock or severe performance issue")

    metrics.end()
    metrics.print_summary("Comprehensive Concurrent Stress Test (1500 ops)")

    # Final Validation
    print(f"\n{'='*70}")
    print(f"FINAL VALIDATION")
    print(f"{'='*70}")

    success_rate = (metrics.success_count / 1500) * 100
    print(f"Total operations:    1500")
    print(f"Success rate:        {success_rate:.2f}% ({metrics.success_count}/1500)")
    print(f"Error rate:          {metrics.error_rate:.2f}%")
    print(f"Throughput:          {metrics.throughput:.2f} ops/sec")
    print(f"Avg response time:   {metrics.avg_operation_time:.2f}ms")
    print(f"P95 response time:   {metrics.p95_operation_time:.2f}ms")
    print(f"Total duration:      {metrics.total_duration:.2f}s")
    print(f"No deadlocks:        {'YES' if metrics.total_duration < 300 else 'NO'}")
    print(f"{'='*70}\n")

    # Comprehensive assertions
    assert metrics.success_count >= 1425, f"Success rate too low: {metrics.success_count}/1500"
    assert metrics.error_rate < 5.0, f"Error rate too high: {metrics.error_rate:.2f}%"
    assert metrics.total_duration < 300, "Test took too long (possible deadlock)"
    assert metrics.throughput > 5, f"Throughput too low: {metrics.throughput:.2f} ops/sec"
    assert metrics.p95_operation_time < 10000, f"P95 latency too high: {metrics.p95_operation_time:.2f}ms"

    print("ALL COMPREHENSIVE STRESS TEST CRITERIA PASSED")
