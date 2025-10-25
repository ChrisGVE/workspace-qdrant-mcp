"""
Queue Bottleneck Detection System

Identifies performance bottlenecks in queue processing operations by analyzing
operation timings, collection performance, parser efficiency, and tenant-specific
issues. Provides actionable insights for queue optimization.

Features:
    - Operation-level bottleneck identification using percentile thresholds
    - Collection bottleneck detection by comparing processing rates
    - Parser bottleneck analysis by file type
    - Tenant-specific bottleneck identification
    - Processing pipeline stage analysis (parse, embed, store)
    - Slow operation tracking with configurable retention
    - Actionable recommendations for bottleneck resolution

Example:
    ```python
    from workspace_qdrant_mcp.core.queue_bottleneck_detector import BottleneckDetector

    # Initialize detector
    detector = BottleneckDetector()
    await detector.initialize()

    # Identify slow operations
    bottlenecks = await detector.identify_slow_operations(threshold_percentile=95)
    for bottleneck in bottlenecks:
        print(f"Operation {bottleneck.operation_type}: {bottleneck.avg_duration}ms")

    # Identify slow collections
    slow_collections = await detector.identify_slow_collections()
    for collection in slow_collections:
        print(f"Collection {collection.collection_name}: {collection.issue_description}")

    # Get slowest operations
    slowest = await detector.get_slowest_items(limit=10)
    for item in slowest:
        print(f"File {item.file_path}: {item.duration_ms}ms")
    ```
"""

import asyncio
import statistics
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from loguru import logger

from .queue_connection import ConnectionConfig, QueueConnectionPool
from .queue_statistics import QueueStatisticsCollector


@dataclass
class SlowOperation:
    """
    Details of a single slow operation for tracking and analysis.

    Attributes:
        file_path: Path to file being processed
        operation: Operation type (ingest, update, delete)
        duration_ms: Processing duration in milliseconds
        timestamp: When operation occurred
        collection_name: Collection being processed
        file_type: File extension/type
        tenant_id: Tenant identifier
        metadata: Additional operation metadata
    """
    file_path: str
    operation: str
    duration_ms: float
    timestamp: datetime
    collection_name: str
    file_type: str | None = None
    tenant_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "file_path": self.file_path,
            "operation": self.operation,
            "duration_ms": round(self.duration_ms, 2),
            "timestamp": self.timestamp.isoformat(),
            "collection_name": self.collection_name,
            "file_type": self.file_type,
            "tenant_id": self.tenant_id,
            "metadata": self.metadata
        }


@dataclass
class OperationBottleneck:
    """
    Bottleneck detected for a specific operation type.

    Attributes:
        operation_type: Type of operation (ingest, update, delete)
        avg_duration: Average duration in milliseconds
        p95_duration: 95th percentile duration in milliseconds
        p99_duration: 99th percentile duration in milliseconds
        slowest_items: List of slowest operations
        count: Number of operations analyzed
        recommendation: Suggested action to resolve bottleneck
    """
    operation_type: str
    avg_duration: float
    p95_duration: float
    p99_duration: float
    slowest_items: list[SlowOperation]
    count: int
    recommendation: str

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "operation_type": self.operation_type,
            "avg_duration": round(self.avg_duration, 2),
            "p95_duration": round(self.p95_duration, 2),
            "p99_duration": round(self.p99_duration, 2),
            "slowest_items": [item.to_dict() for item in self.slowest_items[:5]],
            "count": self.count,
            "recommendation": self.recommendation
        }


@dataclass
class CollectionBottleneck:
    """
    Bottleneck detected for a specific collection.

    Attributes:
        collection_name: Name of affected collection
        processing_rate: Current processing rate (items/min)
        avg_processing_rate: Average processing rate across all collections
        avg_time: Average processing time for this collection (ms)
        overall_avg_time: Average processing time across all collections (ms)
        issue_description: Description of the bottleneck
        recommendation: Suggested action to resolve bottleneck
        sample_slow_items: Example slow operations
    """
    collection_name: str
    processing_rate: float
    avg_processing_rate: float
    avg_time: float
    overall_avg_time: float
    issue_description: str
    recommendation: str
    sample_slow_items: list[SlowOperation] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "collection_name": self.collection_name,
            "processing_rate": round(self.processing_rate, 2),
            "avg_processing_rate": round(self.avg_processing_rate, 2),
            "avg_time": round(self.avg_time, 2),
            "overall_avg_time": round(self.overall_avg_time, 2),
            "issue_description": self.issue_description,
            "recommendation": self.recommendation,
            "sample_slow_items": [item.to_dict() for item in self.sample_slow_items[:3]]
        }


@dataclass
class ParserStats:
    """
    Parser performance statistics for a file type.

    Attributes:
        file_type: File extension/type
        count: Number of files processed
        avg_time: Average parsing time (ms)
        p95_time: 95th percentile parsing time (ms)
        slowest_items: List of slowest parsing operations
        recommendation: Suggested optimization
    """
    file_type: str
    count: int
    avg_time: float
    p95_time: float
    slowest_items: list[SlowOperation]
    recommendation: str

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "file_type": self.file_type,
            "count": self.count,
            "avg_time": round(self.avg_time, 2),
            "p95_time": round(self.p95_time, 2),
            "slowest_items": [item.to_dict() for item in self.slowest_items[:3]],
            "recommendation": self.recommendation
        }


@dataclass
class TenantBottleneck:
    """
    Bottleneck detected for a specific tenant.

    Attributes:
        tenant_id: Tenant identifier
        processing_rate: Current processing rate (items/min)
        avg_processing_rate: Average processing rate across all tenants
        issue_description: Description of the bottleneck
        recommendation: Suggested action to resolve bottleneck
    """
    tenant_id: str
    processing_rate: float
    avg_processing_rate: float
    issue_description: str
    recommendation: str

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "tenant_id": self.tenant_id,
            "processing_rate": round(self.processing_rate, 2),
            "avg_processing_rate": round(self.avg_processing_rate, 2),
            "issue_description": self.issue_description,
            "recommendation": self.recommendation
        }


@dataclass
class PipelineAnalysis:
    """
    Analysis of processing pipeline stage performance.

    Attributes:
        parse_avg_time: Average parsing stage time (ms)
        metadata_avg_time: Average metadata extraction time (ms)
        embed_avg_time: Average embedding generation time (ms)
        store_avg_time: Average storage time (ms)
        total_avg_time: Average total processing time (ms)
        bottleneck_stage: Slowest pipeline stage
        stage_percentages: Percentage of total time per stage
        recommendation: Suggested optimization
    """
    parse_avg_time: float
    metadata_avg_time: float
    embed_avg_time: float
    store_avg_time: float
    total_avg_time: float
    bottleneck_stage: str
    stage_percentages: dict[str, float]
    recommendation: str

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "parse_avg_time": round(self.parse_avg_time, 2),
            "metadata_avg_time": round(self.metadata_avg_time, 2),
            "embed_avg_time": round(self.embed_avg_time, 2),
            "store_avg_time": round(self.store_avg_time, 2),
            "total_avg_time": round(self.total_avg_time, 2),
            "bottleneck_stage": self.bottleneck_stage,
            "stage_percentages": {k: round(v, 1) for k, v in self.stage_percentages.items()},
            "recommendation": self.recommendation
        }


class BottleneckDetector:
    """
    Queue bottleneck detector with operation-level analysis.

    Identifies performance bottlenecks by analyzing operation timings,
    collection performance, parser efficiency, and pipeline stages.
    """

    def __init__(
        self,
        db_path: str | None = None,
        connection_config: ConnectionConfig | None = None,
        slow_operation_threshold_ms: float = 5000.0,
        slow_collection_multiplier: float = 2.0,
        slow_parser_multiplier: float = 2.0,
        slow_tenant_rate_threshold: float = 0.5,
        max_slow_items: int = 1000
    ):
        """
        Initialize bottleneck detector.

        Args:
            db_path: Optional custom database path
            connection_config: Optional connection configuration
            slow_operation_threshold_ms: Threshold for slow operations (ms)
            slow_collection_multiplier: Multiplier for slow collection detection
            slow_parser_multiplier: Multiplier for slow parser detection
            slow_tenant_rate_threshold: Threshold for slow tenant (fraction of avg)
            max_slow_items: Maximum slow items to retain in memory
        """
        self.connection_pool = QueueConnectionPool(
            db_path=db_path or self._get_default_db_path(),
            config=connection_config or ConnectionConfig()
        )
        self._initialized = False

        # Configuration
        self.slow_operation_threshold_ms = slow_operation_threshold_ms
        self.slow_collection_multiplier = slow_collection_multiplier
        self.slow_parser_multiplier = slow_parser_multiplier
        self.slow_tenant_rate_threshold = slow_tenant_rate_threshold

        # Slow operation tracking
        self._slow_operations: deque[SlowOperation] = deque(maxlen=max_slow_items)
        self._lock = asyncio.Lock()

        # Statistics collector for rate calculations
        self.stats_collector: QueueStatisticsCollector | None = None

    def _get_default_db_path(self) -> str:
        """Get default database path from OS directories."""
        from ..utils.os_directories import OSDirectories
        os_dirs = OSDirectories()
        os_dirs.ensure_directories()
        return str(os_dirs.get_state_file("workspace_state.db"))

    async def initialize(self):
        """Initialize the bottleneck detector."""
        if self._initialized:
            return

        await self.connection_pool.initialize()

        # Initialize statistics collector for rate calculations
        self.stats_collector = QueueStatisticsCollector(
            db_path=self.connection_pool.db_path
        )
        await self.stats_collector.initialize()

        self._initialized = True
        logger.info("Bottleneck detector initialized")

    async def close(self):
        """Close the bottleneck detector."""
        if not self._initialized:
            return

        if self.stats_collector:
            await self.stats_collector.close()

        await self.connection_pool.close()
        self._initialized = False
        logger.info("Bottleneck detector closed")

    async def record_operation(
        self,
        file_path: str,
        operation: str,
        duration_ms: float,
        collection_name: str,
        file_type: str | None = None,
        tenant_id: str | None = None,
        metadata: dict[str, Any] | None = None
    ):
        """
        Record an operation for bottleneck analysis.

        Args:
            file_path: Path to file being processed
            operation: Operation type (ingest, update, delete)
            duration_ms: Processing duration in milliseconds
            collection_name: Collection being processed
            file_type: File extension/type
            tenant_id: Tenant identifier
            metadata: Additional operation metadata
        """
        # Only track if above threshold
        if duration_ms >= self.slow_operation_threshold_ms:
            slow_op = SlowOperation(
                file_path=file_path,
                operation=operation,
                duration_ms=duration_ms,
                timestamp=datetime.now(timezone.utc),
                collection_name=collection_name,
                file_type=file_type or Path(file_path).suffix.lstrip('.') or 'unknown',
                tenant_id=tenant_id,
                metadata=metadata or {}
            )

            async with self._lock:
                self._slow_operations.append(slow_op)

            logger.debug(
                f"Recorded slow operation: {file_path} took {duration_ms}ms "
                f"(threshold: {self.slow_operation_threshold_ms}ms)"
            )

    async def identify_slow_operations(
        self,
        threshold_percentile: int = 95,
        time_window_minutes: int | None = 60
    ) -> list[OperationBottleneck]:
        """
        Identify operations taking >p95 (or specified percentile) of all operations.

        Args:
            threshold_percentile: Percentile threshold (0-100)
            time_window_minutes: Time window for analysis (None = all time)

        Returns:
            List of operation bottlenecks
        """
        # Get slow operations within time window
        cutoff_time = None
        if time_window_minutes:
            cutoff_time = datetime.now(timezone.utc) - timedelta(minutes=time_window_minutes)

        async with self._lock:
            operations = list(self._slow_operations)

        if cutoff_time:
            operations = [op for op in operations if op.timestamp >= cutoff_time]

        if not operations:
            logger.debug("No slow operations found for bottleneck analysis")
            return []

        # Group by operation type
        by_operation_type: dict[str, list[SlowOperation]] = defaultdict(list)
        for op in operations:
            by_operation_type[op.operation].append(op)

        # Calculate statistics for each operation type
        bottlenecks = []
        for operation_type, ops in by_operation_type.items():
            if len(ops) < 2:  # Need at least 2 samples
                continue

            durations = [op.duration_ms for op in ops]
            durations.sort()

            avg_duration = statistics.mean(durations)

            # Calculate percentiles
            p95_index = int(threshold_percentile / 100.0 * len(durations))
            p95_duration = durations[min(p95_index, len(durations) - 1)]

            p99_index = int(0.99 * len(durations))
            p99_duration = durations[min(p99_index, len(durations) - 1)]

            # Get slowest items
            slowest_ops = sorted(ops, key=lambda x: x.duration_ms, reverse=True)[:10]

            # Generate recommendation
            recommendation = self._generate_operation_recommendation(
                operation_type, avg_duration, slowest_ops
            )

            bottlenecks.append(OperationBottleneck(
                operation_type=operation_type,
                avg_duration=avg_duration,
                p95_duration=p95_duration,
                p99_duration=p99_duration,
                slowest_items=slowest_ops,
                count=len(ops),
                recommendation=recommendation
            ))

        # Sort by average duration (slowest first)
        bottlenecks.sort(key=lambda x: x.avg_duration, reverse=True)

        logger.info(f"Identified {len(bottlenecks)} operation bottlenecks")
        return bottlenecks

    async def identify_slow_collections(
        self,
        time_window_minutes: int = 60
    ) -> list[CollectionBottleneck]:
        """
        Identify collections with avg processing time >2x overall average.

        Args:
            time_window_minutes: Time window for analysis

        Returns:
            List of collection bottlenecks
        """
        cutoff_time = datetime.now(timezone.utc) - timedelta(minutes=time_window_minutes)

        async with self._lock:
            operations = list(self._slow_operations)

        operations = [op for op in operations if op.timestamp >= cutoff_time]

        if not operations:
            logger.debug("No operations found for collection bottleneck analysis")
            return []

        # Calculate overall average
        overall_avg_time = statistics.mean([op.duration_ms for op in operations])

        # Group by collection
        by_collection: dict[str, list[SlowOperation]] = defaultdict(list)
        for op in operations:
            by_collection[op.collection_name].append(op)

        # Identify slow collections
        bottlenecks = []
        threshold = overall_avg_time * self.slow_collection_multiplier

        for collection_name, ops in by_collection.items():
            if len(ops) < 2:  # Need at least 2 samples
                continue

            avg_time = statistics.mean([op.duration_ms for op in ops])

            if avg_time >= threshold:
                # Calculate processing rate (approximation)
                time_span = (max(op.timestamp for op in ops) -
                           min(op.timestamp for op in ops)).total_seconds() / 60
                processing_rate = len(ops) / time_span if time_span > 0 else 0

                # Calculate average processing rate across all collections
                avg_processing_rate = len(operations) / time_window_minutes

                issue_description = (
                    f"Collection '{collection_name}' has average processing time of "
                    f"{avg_time:.1f}ms, which is {avg_time/overall_avg_time:.1f}x "
                    f"the overall average ({overall_avg_time:.1f}ms)"
                )

                recommendation = self._generate_collection_recommendation(
                    collection_name, avg_time, ops
                )

                bottlenecks.append(CollectionBottleneck(
                    collection_name=collection_name,
                    processing_rate=processing_rate,
                    avg_processing_rate=avg_processing_rate,
                    avg_time=avg_time,
                    overall_avg_time=overall_avg_time,
                    issue_description=issue_description,
                    recommendation=recommendation,
                    sample_slow_items=sorted(ops, key=lambda x: x.duration_ms, reverse=True)[:5]
                ))

        # Sort by average time (slowest first)
        bottlenecks.sort(key=lambda x: x.avg_time, reverse=True)

        logger.info(f"Identified {len(bottlenecks)} collection bottlenecks")
        return bottlenecks

    async def identify_slow_parsers(
        self,
        time_window_minutes: int = 60
    ) -> dict[str, ParserStats]:
        """
        Identify file types with avg parsing time >2x overall average.

        Args:
            time_window_minutes: Time window for analysis

        Returns:
            Dictionary of file type to parser statistics
        """
        cutoff_time = datetime.now(timezone.utc) - timedelta(minutes=time_window_minutes)

        async with self._lock:
            operations = list(self._slow_operations)

        operations = [op for op in operations if op.timestamp >= cutoff_time]

        if not operations:
            logger.debug("No operations found for parser bottleneck analysis")
            return {}

        # Calculate overall average
        overall_avg_time = statistics.mean([op.duration_ms for op in operations])

        # Group by file type
        by_file_type: dict[str, list[SlowOperation]] = defaultdict(list)
        for op in operations:
            file_type = op.file_type or 'unknown'
            by_file_type[file_type].append(op)

        # Identify slow parsers
        slow_parsers = {}
        threshold = overall_avg_time * self.slow_parser_multiplier

        for file_type, ops in by_file_type.items():
            if len(ops) < 2:  # Need at least 2 samples
                continue

            durations = [op.duration_ms for op in ops]
            avg_time = statistics.mean(durations)

            # Only include if slow or if we want all parsers
            if avg_time >= threshold or len(slow_parsers) == 0:
                durations.sort()
                p95_index = int(0.95 * len(durations))
                p95_time = durations[min(p95_index, len(durations) - 1)]

                slowest_ops = sorted(ops, key=lambda x: x.duration_ms, reverse=True)[:5]

                recommendation = self._generate_parser_recommendation(
                    file_type, avg_time, overall_avg_time
                )

                slow_parsers[file_type] = ParserStats(
                    file_type=file_type,
                    count=len(ops),
                    avg_time=avg_time,
                    p95_time=p95_time,
                    slowest_items=slowest_ops,
                    recommendation=recommendation
                )

        logger.info(f"Identified {len(slow_parsers)} file type parser statistics")
        return slow_parsers

    async def identify_slow_tenants(
        self,
        time_window_minutes: int = 60
    ) -> list[TenantBottleneck]:
        """
        Identify tenants with processing rate <50% of average.

        Args:
            time_window_minutes: Time window for analysis

        Returns:
            List of tenant bottlenecks
        """
        cutoff_time = datetime.now(timezone.utc) - timedelta(minutes=time_window_minutes)

        async with self._lock:
            operations = list(self._slow_operations)

        operations = [op for op in operations if op.timestamp >= cutoff_time and op.tenant_id]

        if not operations:
            logger.debug("No tenant operations found for bottleneck analysis")
            return []

        # Group by tenant
        by_tenant: dict[str, list[SlowOperation]] = defaultdict(list)
        for op in operations:
            if op.tenant_id:
                by_tenant[op.tenant_id].append(op)

        # Calculate processing rates
        tenant_rates = {}
        for tenant_id, ops in by_tenant.items():
            time_span = (max(op.timestamp for op in ops) -
                        min(op.timestamp for op in ops)).total_seconds() / 60
            tenant_rates[tenant_id] = len(ops) / time_span if time_span > 0 else 0

        # Calculate average rate
        avg_rate = statistics.mean(tenant_rates.values()) if tenant_rates else 0

        # Identify slow tenants
        bottlenecks = []
        threshold = avg_rate * self.slow_tenant_rate_threshold

        for tenant_id, rate in tenant_rates.items():
            if rate < threshold:
                issue_description = (
                    f"Tenant '{tenant_id}' has processing rate of {rate:.1f} items/min, "
                    f"which is {rate/avg_rate:.1%} of the average rate ({avg_rate:.1f} items/min)"
                )

                recommendation = (
                    f"Investigate tenant '{tenant_id}' for resource constraints, "
                    f"quota limits, or tenant-specific issues"
                )

                bottlenecks.append(TenantBottleneck(
                    tenant_id=tenant_id,
                    processing_rate=rate,
                    avg_processing_rate=avg_rate,
                    issue_description=issue_description,
                    recommendation=recommendation
                ))

        # Sort by processing rate (slowest first)
        bottlenecks.sort(key=lambda x: x.processing_rate)

        logger.info(f"Identified {len(bottlenecks)} tenant bottlenecks")
        return bottlenecks

    async def get_slowest_items(
        self,
        limit: int = 10
    ) -> list[SlowOperation]:
        """
        Get top N slowest operations.

        Args:
            limit: Maximum number of items to return

        Returns:
            List of slowest operations
        """
        async with self._lock:
            operations = list(self._slow_operations)

        # Sort by duration (slowest first)
        operations.sort(key=lambda x: x.duration_ms, reverse=True)

        return operations[:limit]

    async def analyze_processing_pipeline(
        self,
        time_window_minutes: int = 60
    ) -> PipelineAnalysis:
        """
        Analyze processing pipeline to identify which stage is slowest.

        Note: This is a simplified implementation. In production, you would
        need to track individual pipeline stage timings.

        Args:
            time_window_minutes: Time window for analysis

        Returns:
            Pipeline analysis with stage timings
        """
        # This is a simplified implementation that estimates stage times
        # In production, you would track actual stage timings

        cutoff_time = datetime.now(timezone.utc) - timedelta(minutes=time_window_minutes)

        async with self._lock:
            operations = list(self._slow_operations)

        operations = [op for op in operations if op.timestamp >= cutoff_time]

        if not operations:
            logger.debug("No operations found for pipeline analysis")
            return PipelineAnalysis(
                parse_avg_time=0,
                metadata_avg_time=0,
                embed_avg_time=0,
                store_avg_time=0,
                total_avg_time=0,
                bottleneck_stage="unknown",
                stage_percentages={},
                recommendation="Insufficient data for pipeline analysis"
            )

        total_avg_time = statistics.mean([op.duration_ms for op in operations])

        # Estimate stage times (simplified - in production, track actual times)
        # These are rough estimates based on typical processing patterns
        parse_avg_time = total_avg_time * 0.2  # ~20% parsing
        metadata_avg_time = total_avg_time * 0.1  # ~10% metadata extraction
        embed_avg_time = total_avg_time * 0.5  # ~50% embedding generation
        store_avg_time = total_avg_time * 0.2  # ~20% storage

        # Identify bottleneck stage
        stages = {
            "parse": parse_avg_time,
            "metadata": metadata_avg_time,
            "embed": embed_avg_time,
            "store": store_avg_time
        }
        bottleneck_stage = max(stages.items(), key=lambda x: x[1])[0]

        # Calculate percentages
        stage_percentages = {
            stage: (time / total_avg_time * 100) if total_avg_time > 0 else 0
            for stage, time in stages.items()
        }

        # Generate recommendation
        recommendation = self._generate_pipeline_recommendation(bottleneck_stage)

        return PipelineAnalysis(
            parse_avg_time=parse_avg_time,
            metadata_avg_time=metadata_avg_time,
            embed_avg_time=embed_avg_time,
            store_avg_time=store_avg_time,
            total_avg_time=total_avg_time,
            bottleneck_stage=bottleneck_stage,
            stage_percentages=stage_percentages,
            recommendation=recommendation
        )

    def _generate_operation_recommendation(
        self,
        operation_type: str,
        avg_duration: float,
        slowest_ops: list[SlowOperation]
    ) -> str:
        """Generate recommendation for operation bottleneck."""
        # Analyze patterns in slowest operations
        file_types = defaultdict(int)
        for op in slowest_ops:
            if op.file_type:
                file_types[op.file_type] += 1

        most_common_type = max(file_types.items(), key=lambda x: x[1])[0] if file_types else None

        if operation_type == "ingest":
            if most_common_type:
                return (
                    f"Optimize ingestion for {most_common_type} files. "
                    f"Consider increasing batch size or optimizing parser for this file type."
                )
            return "Optimize ingestion process. Review embedding generation and storage operations."

        elif operation_type == "update":
            return "Optimize update operations. Consider incremental updates instead of full re-processing."

        elif operation_type == "delete":
            return "Optimize delete operations. Batch deletes when possible."

        return f"Review {operation_type} operation implementation for optimization opportunities."

    def _generate_collection_recommendation(
        self,
        collection_name: str,
        avg_time: float,
        ops: list[SlowOperation]
    ) -> str:
        """Generate recommendation for collection bottleneck."""
        # Check if collection has specific file types
        file_types = defaultdict(int)
        for op in ops:
            if op.file_type:
                file_types[op.file_type] += 1

        if file_types:
            dominant_type = max(file_types.items(), key=lambda x: x[1])[0]
            return (
                f"Collection '{collection_name}' is dominated by {dominant_type} files. "
                f"Optimize parser/processor for this file type or adjust collection configuration."
            )

        return (
            f"Review collection '{collection_name}' configuration. "
            f"Consider splitting large collections or optimizing indexing settings."
        )

    def _generate_parser_recommendation(
        self,
        file_type: str,
        avg_time: float,
        overall_avg_time: float
    ) -> str:
        """Generate recommendation for parser bottleneck."""
        ratio = avg_time / overall_avg_time if overall_avg_time > 0 else 1

        if ratio >= 3:
            return (
                f"Parser for {file_type} files is {ratio:.1f}x slower than average. "
                f"Consider implementing specialized parser or caching parsed results."
            )
        elif ratio >= 2:
            return (
                f"Parser for {file_type} files is {ratio:.1f}x slower than average. "
                f"Review parser implementation for optimization opportunities."
            )
        else:
            return f"Parser performance for {file_type} files is acceptable."

    def _generate_pipeline_recommendation(self, bottleneck_stage: str) -> str:
        """Generate recommendation for pipeline bottleneck."""
        recommendations = {
            "parse": (
                "Parsing is the bottleneck. Optimize parsers, use faster parsing libraries, "
                "or implement parser caching."
            ),
            "metadata": (
                "Metadata extraction is the bottleneck. Optimize metadata extraction logic "
                "or parallelize extraction operations."
            ),
            "embed": (
                "Embedding generation is the bottleneck. Consider using faster embedding models, "
                "batch embedding operations, or implement embedding caching."
            ),
            "store": (
                "Storage is the bottleneck. Optimize database operations, use batch insertions, "
                "or review Qdrant configuration."
            )
        }

        return recommendations.get(
            bottleneck_stage,
            "Review processing pipeline for optimization opportunities."
        )
