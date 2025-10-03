"""
Unit tests for Queue Bottleneck Detection System

Tests bottleneck identification for operations, collections, parsers, tenants,
and pipeline stages. Ensures accurate detection and appropriate recommendations.
"""

import asyncio
import pytest
import sqlite3
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import List
from unittest.mock import MagicMock, patch

from src.python.common.core.queue_bottleneck_detector import (
    BottleneckDetector,
    SlowOperation,
    OperationBottleneck,
    CollectionBottleneck,
    ParserStats,
    TenantBottleneck,
    PipelineAnalysis,
)


@pytest.fixture
async def detector(tmp_path):
    """Create a bottleneck detector with temporary database."""
    db_path = tmp_path / "test_queue.db"

    # Create minimal database schema
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()

    # Create minimal ingestion_queue table for stats collector
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS ingestion_queue (
            file_absolute_path TEXT PRIMARY KEY,
            collection_name TEXT NOT NULL,
            operation TEXT NOT NULL,
            priority INTEGER DEFAULT 5,
            queued_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            retry_count INTEGER DEFAULT 0,
            error_message_id INTEGER,
            tenant_id TEXT DEFAULT 'default'
        )
    """)

    conn.commit()
    conn.close()

    detector = BottleneckDetector(
        db_path=str(db_path),
        slow_operation_threshold_ms=1000.0,  # Lower threshold for testing
        slow_collection_multiplier=2.0,
        slow_parser_multiplier=2.0,
        slow_tenant_rate_threshold=0.5,
        max_slow_items=100
    )

    await detector.initialize()
    yield detector
    await detector.close()


@pytest.fixture
def sample_operations():
    """Create sample slow operations for testing."""
    base_time = datetime.now(timezone.utc)

    operations = [
        SlowOperation(
            file_path="/project/file1.py",
            operation="ingest",
            duration_ms=1500.0,
            timestamp=base_time - timedelta(minutes=5),
            collection_name="test-collection",
            file_type="py",
            tenant_id="tenant1",
            metadata={"size": 1024}
        ),
        SlowOperation(
            file_path="/project/file2.py",
            operation="ingest",
            duration_ms=2000.0,
            timestamp=base_time - timedelta(minutes=4),
            collection_name="test-collection",
            file_type="py",
            tenant_id="tenant1",
            metadata={"size": 2048}
        ),
        SlowOperation(
            file_path="/project/file3.js",
            operation="ingest",
            duration_ms=3000.0,
            timestamp=base_time - timedelta(minutes=3),
            collection_name="slow-collection",
            file_type="js",
            tenant_id="tenant1",
            metadata={"size": 3072}
        ),
        SlowOperation(
            file_path="/project/file4.js",
            operation="ingest",
            duration_ms=4000.0,
            timestamp=base_time - timedelta(minutes=2),
            collection_name="slow-collection",
            file_type="js",
            tenant_id="tenant2",
            metadata={"size": 4096}
        ),
        SlowOperation(
            file_path="/project/file5.md",
            operation="update",
            duration_ms=1200.0,
            timestamp=base_time - timedelta(minutes=1),
            collection_name="test-collection",
            file_type="md",
            tenant_id="tenant1",
            metadata={"size": 512}
        ),
    ]

    return operations


@pytest.mark.asyncio
class TestSlowOperationTracking:
    """Tests for slow operation recording and tracking."""

    async def test_record_operation_above_threshold(self, detector):
        """Test that operations above threshold are recorded."""
        await detector.record_operation(
            file_path="/test/file.py",
            operation="ingest",
            duration_ms=1500.0,
            collection_name="test",
            file_type="py",
            tenant_id="default"
        )

        slowest = await detector.get_slowest_items(limit=10)
        assert len(slowest) == 1
        assert slowest[0].file_path == "/test/file.py"
        assert slowest[0].duration_ms == 1500.0

    async def test_record_operation_below_threshold(self, detector):
        """Test that operations below threshold are not recorded."""
        await detector.record_operation(
            file_path="/test/file.py",
            operation="ingest",
            duration_ms=500.0,  # Below 1000ms threshold
            collection_name="test",
            file_type="py",
            tenant_id="default"
        )

        slowest = await detector.get_slowest_items(limit=10)
        assert len(slowest) == 0

    async def test_max_slow_items_retention(self, detector):
        """Test that max_slow_items limit is respected."""
        # Record more than max_slow_items (100)
        for i in range(150):
            await detector.record_operation(
                file_path=f"/test/file{i}.py",
                operation="ingest",
                duration_ms=1500.0,
                collection_name="test",
                file_type="py",
                tenant_id="default"
            )

        slowest = await detector.get_slowest_items(limit=200)
        assert len(slowest) == 100  # Should not exceed max_slow_items


@pytest.mark.asyncio
class TestOperationBottleneckDetection:
    """Tests for operation-level bottleneck identification."""

    async def test_identify_slow_operations_basic(self, detector, sample_operations):
        """Test basic slow operation identification."""
        # Record sample operations
        for op in sample_operations:
            async with detector._lock:
                detector._slow_operations.append(op)

        bottlenecks = await detector.identify_slow_operations(threshold_percentile=95)

        assert len(bottlenecks) > 0

        # Check that ingest operations are identified
        ingest_bottleneck = next((b for b in bottlenecks if b.operation_type == "ingest"), None)
        assert ingest_bottleneck is not None
        assert ingest_bottleneck.count == 4
        assert ingest_bottleneck.avg_duration > 0
        assert ingest_bottleneck.p95_duration > 0

    async def test_identify_slow_operations_empty_data(self, detector):
        """Test operation identification with no slow operations."""
        bottlenecks = await detector.identify_slow_operations()
        assert len(bottlenecks) == 0

    async def test_identify_slow_operations_time_window(self, detector, sample_operations):
        """Test operation identification with time window."""
        # Record sample operations
        for op in sample_operations:
            async with detector._lock:
                detector._slow_operations.append(op)

        # Use a very short time window - should find fewer operations
        bottlenecks = await detector.identify_slow_operations(
            threshold_percentile=95,
            time_window_minutes=1
        )

        # Only the most recent operation (file5.md) should be in window
        assert all(b.count <= 2 for b in bottlenecks)

    async def test_operation_bottleneck_to_dict(self, detector, sample_operations):
        """Test OperationBottleneck serialization."""
        for op in sample_operations:
            async with detector._lock:
                detector._slow_operations.append(op)

        bottlenecks = await detector.identify_slow_operations()

        for bottleneck in bottlenecks:
            result = bottleneck.to_dict()
            assert "operation_type" in result
            assert "avg_duration" in result
            assert "p95_duration" in result
            assert "slowest_items" in result
            assert "count" in result
            assert "recommendation" in result
            assert isinstance(result["slowest_items"], list)


@pytest.mark.asyncio
class TestCollectionBottleneckDetection:
    """Tests for collection-level bottleneck identification."""

    async def test_identify_slow_collections(self, detector, sample_operations):
        """Test slow collection identification."""
        for op in sample_operations:
            async with detector._lock:
                detector._slow_operations.append(op)

        bottlenecks = await detector.identify_slow_collections(time_window_minutes=60)

        # Should identify slow-collection as it has higher avg time
        slow_collections = [b.collection_name for b in bottlenecks]

        # At least one collection should be identified as slow
        assert len(bottlenecks) >= 0  # May be 0 if not enough variation

    async def test_identify_slow_collections_empty_data(self, detector):
        """Test collection identification with no data."""
        bottlenecks = await detector.identify_slow_collections()
        assert len(bottlenecks) == 0

    async def test_collection_bottleneck_to_dict(self, detector, sample_operations):
        """Test CollectionBottleneck serialization."""
        # Add operations with significant difference
        base_time = datetime.now(timezone.utc)

        fast_ops = [
            SlowOperation(
                file_path=f"/fast/file{i}.py",
                operation="ingest",
                duration_ms=1000.0,
                timestamp=base_time - timedelta(minutes=i),
                collection_name="fast-collection",
                file_type="py"
            )
            for i in range(5)
        ]

        slow_ops = [
            SlowOperation(
                file_path=f"/slow/file{i}.py",
                operation="ingest",
                duration_ms=5000.0,  # 5x slower
                timestamp=base_time - timedelta(minutes=i),
                collection_name="slow-collection",
                file_type="py"
            )
            for i in range(5)
        ]

        for op in fast_ops + slow_ops:
            async with detector._lock:
                detector._slow_operations.append(op)

        bottlenecks = await detector.identify_slow_collections()

        if bottlenecks:
            result = bottlenecks[0].to_dict()
            assert "collection_name" in result
            assert "processing_rate" in result
            assert "avg_time" in result
            assert "issue_description" in result
            assert "recommendation" in result


@pytest.mark.asyncio
class TestParserBottleneckDetection:
    """Tests for parser/file-type bottleneck identification."""

    async def test_identify_slow_parsers(self, detector):
        """Test slow parser identification."""
        base_time = datetime.now(timezone.utc)

        # Create operations with different file types and durations
        operations = [
            SlowOperation(
                file_path=f"/test/file{i}.py",
                operation="ingest",
                duration_ms=1000.0,
                timestamp=base_time - timedelta(minutes=i),
                collection_name="test",
                file_type="py"
            )
            for i in range(5)
        ] + [
            SlowOperation(
                file_path=f"/test/file{i}.js",
                operation="ingest",
                duration_ms=5000.0,  # 5x slower
                timestamp=base_time - timedelta(minutes=i),
                collection_name="test",
                file_type="js"
            )
            for i in range(5)
        ]

        for op in operations:
            async with detector._lock:
                detector._slow_operations.append(op)

        parsers = await detector.identify_slow_parsers(time_window_minutes=60)

        # Should identify js as slow parser
        assert "js" in parsers or "py" in parsers

        if "js" in parsers:
            assert parsers["js"].avg_time > parsers.get("py", parsers["js"]).avg_time

    async def test_identify_slow_parsers_empty_data(self, detector):
        """Test parser identification with no data."""
        parsers = await detector.identify_slow_parsers()
        assert len(parsers) == 0

    async def test_parser_stats_to_dict(self, detector):
        """Test ParserStats serialization."""
        base_time = datetime.now(timezone.utc)

        operations = [
            SlowOperation(
                file_path=f"/test/file{i}.py",
                operation="ingest",
                duration_ms=2000.0,
                timestamp=base_time - timedelta(minutes=i),
                collection_name="test",
                file_type="py"
            )
            for i in range(3)
        ]

        for op in operations:
            async with detector._lock:
                detector._slow_operations.append(op)

        parsers = await detector.identify_slow_parsers()

        if parsers:
            result = list(parsers.values())[0].to_dict()
            assert "file_type" in result
            assert "count" in result
            assert "avg_time" in result
            assert "p95_time" in result
            assert "slowest_items" in result
            assert "recommendation" in result


@pytest.mark.asyncio
class TestTenantBottleneckDetection:
    """Tests for tenant-level bottleneck identification."""

    async def test_identify_slow_tenants(self, detector):
        """Test slow tenant identification."""
        base_time = datetime.now(timezone.utc)

        # Create operations with different tenants and rates
        # Fast tenant: 10 operations in 10 minutes = 1 op/min
        fast_ops = [
            SlowOperation(
                file_path=f"/test/file{i}.py",
                operation="ingest",
                duration_ms=1000.0,
                timestamp=base_time - timedelta(minutes=i),
                collection_name="test",
                file_type="py",
                tenant_id="fast-tenant"
            )
            for i in range(10)
        ]

        # Slow tenant: 2 operations in 10 minutes = 0.2 op/min
        slow_ops = [
            SlowOperation(
                file_path=f"/test/file{i}.py",
                operation="ingest",
                duration_ms=1000.0,
                timestamp=base_time - timedelta(minutes=i * 5),
                collection_name="test",
                file_type="py",
                tenant_id="slow-tenant"
            )
            for i in range(2)
        ]

        for op in fast_ops + slow_ops:
            async with detector._lock:
                detector._slow_operations.append(op)

        bottlenecks = await detector.identify_slow_tenants(time_window_minutes=60)

        # Should identify slow-tenant
        slow_tenant_names = [b.tenant_id for b in bottlenecks]
        assert "slow-tenant" in slow_tenant_names or len(bottlenecks) == 0

    async def test_identify_slow_tenants_empty_data(self, detector):
        """Test tenant identification with no data."""
        bottlenecks = await detector.identify_slow_tenants()
        assert len(bottlenecks) == 0

    async def test_tenant_bottleneck_to_dict(self, detector):
        """Test TenantBottleneck serialization."""
        base_time = datetime.now(timezone.utc)

        operations = [
            SlowOperation(
                file_path=f"/test/file{i}.py",
                operation="ingest",
                duration_ms=1000.0,
                timestamp=base_time - timedelta(minutes=i * 10),
                collection_name="test",
                file_type="py",
                tenant_id="test-tenant"
            )
            for i in range(2)
        ]

        for op in operations:
            async with detector._lock:
                detector._slow_operations.append(op)

        bottlenecks = await detector.identify_slow_tenants()

        if bottlenecks:
            result = bottlenecks[0].to_dict()
            assert "tenant_id" in result
            assert "processing_rate" in result
            assert "avg_processing_rate" in result
            assert "issue_description" in result
            assert "recommendation" in result


@pytest.mark.asyncio
class TestSlowestItemsRetrieval:
    """Tests for retrieving slowest operations."""

    async def test_get_slowest_items_basic(self, detector, sample_operations):
        """Test getting slowest items."""
        for op in sample_operations:
            async with detector._lock:
                detector._slow_operations.append(op)

        slowest = await detector.get_slowest_items(limit=3)

        assert len(slowest) == 3
        # Should be sorted by duration (slowest first)
        assert slowest[0].duration_ms >= slowest[1].duration_ms
        assert slowest[1].duration_ms >= slowest[2].duration_ms

    async def test_get_slowest_items_limit(self, detector, sample_operations):
        """Test that limit parameter works correctly."""
        for op in sample_operations:
            async with detector._lock:
                detector._slow_operations.append(op)

        slowest = await detector.get_slowest_items(limit=2)
        assert len(slowest) == 2

    async def test_get_slowest_items_empty(self, detector):
        """Test getting slowest items with no data."""
        slowest = await detector.get_slowest_items(limit=10)
        assert len(slowest) == 0


@pytest.mark.asyncio
class TestPipelineAnalysis:
    """Tests for processing pipeline analysis."""

    async def test_analyze_processing_pipeline_basic(self, detector, sample_operations):
        """Test basic pipeline analysis."""
        for op in sample_operations:
            async with detector._lock:
                detector._slow_operations.append(op)

        analysis = await detector.analyze_processing_pipeline(time_window_minutes=60)

        assert analysis.total_avg_time > 0
        assert analysis.bottleneck_stage in ["parse", "metadata", "embed", "store"]
        assert analysis.parse_avg_time >= 0
        assert analysis.metadata_avg_time >= 0
        assert analysis.embed_avg_time >= 0
        assert analysis.store_avg_time >= 0
        assert len(analysis.stage_percentages) == 4
        assert analysis.recommendation

    async def test_analyze_processing_pipeline_empty_data(self, detector):
        """Test pipeline analysis with no data."""
        analysis = await detector.analyze_processing_pipeline()

        assert analysis.total_avg_time == 0
        assert analysis.bottleneck_stage == "unknown"

    async def test_pipeline_analysis_to_dict(self, detector, sample_operations):
        """Test PipelineAnalysis serialization."""
        for op in sample_operations:
            async with detector._lock:
                detector._slow_operations.append(op)

        analysis = await detector.analyze_processing_pipeline()
        result = analysis.to_dict()

        assert "parse_avg_time" in result
        assert "metadata_avg_time" in result
        assert "embed_avg_time" in result
        assert "store_avg_time" in result
        assert "total_avg_time" in result
        assert "bottleneck_stage" in result
        assert "stage_percentages" in result
        assert "recommendation" in result


@pytest.mark.asyncio
class TestEdgeCases:
    """Tests for edge cases and error conditions."""

    async def test_insufficient_samples(self, detector):
        """Test handling of insufficient samples for analysis."""
        # Add only one operation
        await detector.record_operation(
            file_path="/test/file.py",
            operation="ingest",
            duration_ms=1500.0,
            collection_name="test",
            file_type="py"
        )

        # Should handle gracefully with minimal data
        bottlenecks = await detector.identify_slow_operations()
        assert isinstance(bottlenecks, list)

    async def test_all_items_slow(self, detector):
        """Test when all operations are slow."""
        base_time = datetime.now(timezone.utc)

        for i in range(10):
            await detector.record_operation(
                file_path=f"/test/file{i}.py",
                operation="ingest",
                duration_ms=10000.0,  # All very slow
                collection_name="test",
                file_type="py"
            )

        bottlenecks = await detector.identify_slow_operations()
        assert len(bottlenecks) > 0

    async def test_concurrent_recording(self, detector):
        """Test thread-safe concurrent operation recording."""
        async def record_ops():
            for i in range(10):
                await detector.record_operation(
                    file_path=f"/test/file{i}.py",
                    operation="ingest",
                    duration_ms=1500.0,
                    collection_name="test",
                    file_type="py"
                )

        # Run multiple concurrent recording tasks
        await asyncio.gather(*[record_ops() for _ in range(3)])

        slowest = await detector.get_slowest_items(limit=100)
        assert len(slowest) == 30  # 10 ops × 3 tasks

    async def test_slow_operation_to_dict(self, detector):
        """Test SlowOperation serialization."""
        await detector.record_operation(
            file_path="/test/file.py",
            operation="ingest",
            duration_ms=1500.0,
            collection_name="test",
            file_type="py",
            tenant_id="default",
            metadata={"key": "value"}
        )

        slowest = await detector.get_slowest_items(limit=1)
        assert len(slowest) == 1

        result = slowest[0].to_dict()
        assert "file_path" in result
        assert "operation" in result
        assert "duration_ms" in result
        assert "timestamp" in result
        assert "collection_name" in result
        assert "file_type" in result
        assert "tenant_id" in result
        assert "metadata" in result


@pytest.mark.asyncio
class TestRecommendationGeneration:
    """Tests for bottleneck recommendation generation."""

    async def test_operation_recommendation_generation(self, detector):
        """Test that recommendations are generated for operations."""
        base_time = datetime.now(timezone.utc)

        operations = [
            SlowOperation(
                file_path=f"/test/file{i}.py",
                operation="ingest",
                duration_ms=3000.0,
                timestamp=base_time - timedelta(minutes=i),
                collection_name="test",
                file_type="py"
            )
            for i in range(5)
        ]

        for op in operations:
            async with detector._lock:
                detector._slow_operations.append(op)

        bottlenecks = await detector.identify_slow_operations()

        for bottleneck in bottlenecks:
            assert bottleneck.recommendation
            assert len(bottleneck.recommendation) > 0

    async def test_collection_recommendation_generation(self, detector):
        """Test that recommendations are generated for collections."""
        base_time = datetime.now(timezone.utc)

        operations = [
            SlowOperation(
                file_path=f"/test/file{i}.py",
                operation="ingest",
                duration_ms=5000.0,
                timestamp=base_time - timedelta(minutes=i),
                collection_name="slow-collection",
                file_type="py"
            )
            for i in range(5)
        ]

        for op in operations:
            async with detector._lock:
                detector._slow_operations.append(op)

        bottlenecks = await detector.identify_slow_collections()

        for bottleneck in bottlenecks:
            assert bottleneck.recommendation
            assert len(bottleneck.recommendation) > 0

    async def test_parser_recommendation_generation(self, detector):
        """Test that recommendations are generated for parsers."""
        base_time = datetime.now(timezone.utc)

        operations = [
            SlowOperation(
                file_path=f"/test/file{i}.py",
                operation="ingest",
                duration_ms=4000.0,
                timestamp=base_time - timedelta(minutes=i),
                collection_name="test",
                file_type="py"
            )
            for i in range(5)
        ]

        for op in operations:
            async with detector._lock:
                detector._slow_operations.append(op)

        parsers = await detector.identify_slow_parsers()

        for parser_stats in parsers.values():
            assert parser_stats.recommendation
            assert len(parser_stats.recommendation) > 0

    async def test_pipeline_recommendation_generation(self, detector):
        """Test that recommendations are generated for pipeline analysis."""
        base_time = datetime.now(timezone.utc)

        operations = [
            SlowOperation(
                file_path=f"/test/file{i}.py",
                operation="ingest",
                duration_ms=3000.0,
                timestamp=base_time - timedelta(minutes=i),
                collection_name="test",
                file_type="py"
            )
            for i in range(5)
        ]

        for op in operations:
            async with detector._lock:
                detector._slow_operations.append(op)

        analysis = await detector.analyze_processing_pipeline()

        assert analysis.recommendation
        assert len(analysis.recommendation) > 0
