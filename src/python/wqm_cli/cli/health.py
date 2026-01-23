"""
Health monitoring and system optimization utilities for workspace-qdrant-mcp.

This module provides comprehensive health monitoring, performance tracking,
and system optimization recommendations for workspace-qdrant-mcp installations.
It monitors system resources, collection health, configuration status, and
operational metrics to ensure optimal performance.

Key Features:
    - Real-time system resource monitoring
    - Collection health and statistics analysis
    - Configuration integrity validation
    - Performance trend tracking
    - Optimization recommendations
    - Alert system for critical issues

Monitoring Coverage:
    - System resources (CPU, memory, disk)
    - Qdrant database performance and health
    - Collection statistics and optimization status
    - Embedding service performance
    - Search latency and throughput
    - Configuration drift detection

Example:
    ```bash
    # Check current system health
    workspace-qdrant-health

    # Continuous monitoring mode
    workspace-qdrant-health --watch

    # Detailed analysis with recommendations
    workspace-qdrant-health --analyze

    # Generate health report
    workspace-qdrant-health --report health_report.json
    ```
"""

import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import typer
from common.core.client import QdrantWorkspaceClient
from common.core.config import Config, get_config
from common.utils.config_validator import ConfigValidator
from loguru import logger
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich.text import Text

# Configure logging
logging.basicConfig(level=logging.INFO)
# logger imported from loguru

# Rich console for beautiful output
console = Console()

# Typer app instance
app = typer.Typer(
    name="workspace-qdrant-health",
    help="Health monitoring and optimization for workspace-qdrant-mcp",
    no_args_is_help=False,
)


@dataclass
class ResourceStatus:
    """System resource status."""

    cpu_percent: float
    memory_used_gb: float
    memory_total_gb: float
    memory_percent: float
    disk_used_gb: float
    disk_total_gb: float
    disk_percent: float
    load_average: float | None = None


@dataclass
class CollectionHealth:
    """Health status of a collection."""

    name: str
    vector_count: int
    indexed_vectors_count: int
    points_count: int
    segments_count: int
    disk_data_size: int
    ram_data_size: int
    config_params: dict[str, Any]
    status: str
    optimization_status: str
    last_optimization: str | None = None


@dataclass
class PerformanceMetrics:
    """Performance metrics for the system."""

    embedding_avg_time_ms: float | None = None
    search_avg_time_ms: float | None = None
    throughput_docs_per_sec: float | None = None
    error_rate_percent: float | None = None
    uptime_hours: float | None = None


@dataclass
class HealthReport:
    """Complete health report."""

    timestamp: datetime
    overall_health: str  # "healthy", "warning", "critical"
    system_resources: ResourceStatus
    collections: list[CollectionHealth]
    performance_metrics: PerformanceMetrics
    configuration_status: dict[str, Any]
    alerts: list[str]
    recommendations: list[str]

    def to_dict(self) -> dict[str, Any]:
        """Convert report to dictionary for JSON serialization."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "overall_health": self.overall_health,
            "system_resources": {
                "cpu_percent": self.system_resources.cpu_percent,
                "memory_used_gb": self.system_resources.memory_used_gb,
                "memory_total_gb": self.system_resources.memory_total_gb,
                "memory_percent": self.system_resources.memory_percent,
                "disk_used_gb": self.system_resources.disk_used_gb,
                "disk_total_gb": self.system_resources.disk_total_gb,
                "disk_percent": self.system_resources.disk_percent,
                "load_average": self.system_resources.load_average,
            },
            "collections": [
                {
                    "name": col.name,
                    "vector_count": col.vector_count,
                    "indexed_vectors_count": col.indexed_vectors_count,
                    "points_count": col.points_count,
                    "segments_count": col.segments_count,
                    "disk_data_size": col.disk_data_size,
                    "ram_data_size": col.ram_data_size,
                    "status": col.status,
                    "optimization_status": col.optimization_status,
                    "last_optimization": col.last_optimization,
                }
                for col in self.collections
            ],
            "performance_metrics": {
                "embedding_avg_time_ms": self.performance_metrics.embedding_avg_time_ms,
                "search_avg_time_ms": self.performance_metrics.search_avg_time_ms,
                "throughput_docs_per_sec": self.performance_metrics.throughput_docs_per_sec,
                "error_rate_percent": self.performance_metrics.error_rate_percent,
                "uptime_hours": self.performance_metrics.uptime_hours,
            },
            "configuration_status": self.configuration_status,
            "alerts": self.alerts,
            "recommendations": self.recommendations,
        }


class HealthMonitor:
    """
    Comprehensive health monitoring system for workspace-qdrant-mcp.

    Provides real-time monitoring of system resources, collection health,
    performance metrics, and configuration status. Generates alerts and
    optimization recommendations based on collected data.

    The monitor tracks:
        1. System resources (CPU, memory, disk usage)
        2. Qdrant database health and performance
        3. Collection statistics and optimization status
        4. Embedding and search performance
        5. Configuration integrity and changes
        6. Error rates and system stability

    Attributes:
        config: System configuration
        client: Workspace client for Qdrant operations
        watch_mode: Whether running in continuous monitoring mode
        analyze_mode: Whether to perform deep analysis
    """

    def __init__(
        self, config: Config, watch_mode: bool = False, analyze_mode: bool = False
    ):
        self.config = config
        self.client: QdrantWorkspaceClient | None = None
        self.watch_mode = watch_mode
        self.analyze_mode = analyze_mode
        self.console = console

    async def check_system_health(self) -> HealthReport:
        """Perform comprehensive system health check.

        Returns:
            HealthReport: Complete health assessment
        """
        try:
            # Initialize client
            if not self.client:
                self.client = QdrantWorkspaceClient(self.config)
                await self.client.initialize()

            # Collect health data
            system_resources = self._check_system_resources()
            collections = await self._check_collection_health()
            performance_metrics = await self._check_performance_metrics()
            config_status = await self._check_configuration_validity()

            # Generate alerts and recommendations
            alerts = self._generate_alerts(
                system_resources, collections, performance_metrics
            )
            recommendations = self._generate_recommendations(
                system_resources, collections, performance_metrics, config_status
            )

            # Determine overall health
            overall_health = self._calculate_overall_health(
                alerts, system_resources, collections
            )

            return HealthReport(
                timestamp=datetime.now(),
                overall_health=overall_health,
                system_resources=system_resources,
                collections=collections,
                performance_metrics=performance_metrics,
                configuration_status=config_status,
                alerts=alerts,
                recommendations=recommendations,
            )

        except Exception as e:
            logger.error(f"Health check failed: {e}", exc_info=True)
            # Return minimal health report with error
            return HealthReport(
                timestamp=datetime.now(),
                overall_health="critical",
                system_resources=ResourceStatus(0, 0, 0, 0, 0, 0, 0),
                collections=[],
                performance_metrics=PerformanceMetrics(),
                configuration_status={"error": str(e)},
                alerts=[f"Health check failed: {e}"],
                recommendations=["Fix system configuration and retry health check"],
            )

    def _check_system_resources(self) -> ResourceStatus:
        """Check system resource usage."""
        try:
            import psutil

            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)

            # Memory usage
            memory = psutil.virtual_memory()
            memory_used_gb = (memory.total - memory.available) / (1024**3)
            memory_total_gb = memory.total / (1024**3)
            memory_percent = memory.percent

            # Disk usage
            disk = psutil.disk_usage(".")
            disk_used_gb = (disk.total - disk.free) / (1024**3)
            disk_total_gb = disk.total / (1024**3)
            disk_percent = (disk.used / disk.total) * 100

            # Load average (Unix systems only)
            load_average = None
            try:
                if hasattr(psutil, "getloadavg"):
                    load_average = psutil.getloadavg()[0]  # 1-minute load average
            except (AttributeError, OSError):
                pass

            return ResourceStatus(
                cpu_percent=round(cpu_percent, 1),
                memory_used_gb=round(memory_used_gb, 2),
                memory_total_gb=round(memory_total_gb, 2),
                memory_percent=round(memory_percent, 1),
                disk_used_gb=round(disk_used_gb, 2),
                disk_total_gb=round(disk_total_gb, 2),
                disk_percent=round(disk_percent, 1),
                load_average=round(load_average, 2) if load_average else None,
            )

        except Exception as e:
            logger.error(f"Failed to check system resources: {e}")
            return ResourceStatus(0, 0, 0, 0, 0, 0, 0)

    async def _check_collection_health(self) -> list[CollectionHealth]:
        """Check health of all collections."""
        collections_health = []

        try:
            if not self.client:
                return collections_health

            # Get all workspace collections
            collection_names = self.client.list_collections()

            for collection_name in collection_names:
                try:
                    # Get collection info
                    collection_info = await asyncio.get_event_loop().run_in_executor(
                        None, lambda: self.client.client.get_collection(collection_name)
                    )

                    # Extract health metrics
                    health = CollectionHealth(
                        name=collection_name,
                        vector_count=collection_info.vectors_count or 0,
                        indexed_vectors_count=collection_info.indexed_vectors_count
                        or 0,
                        points_count=collection_info.points_count or 0,
                        segments_count=collection_info.segments_count or 0,
                        disk_data_size=getattr(collection_info, "disk_data_size", 0)
                        or 0,
                        ram_data_size=getattr(collection_info, "ram_data_size", 0) or 0,
                        config_params=collection_info.config.dict()
                        if hasattr(collection_info.config, "dict")
                        else {},
                        status=collection_info.status.value
                        if hasattr(collection_info.status, "value")
                        else str(collection_info.status),
                        optimization_status="unknown",
                    )

                    # Check optimization status if available
                    try:
                        await asyncio.get_event_loop().run_in_executor(
                            None, lambda: self.client.client.get_cluster_info()
                        )
                        # Extract optimization information if available
                        health.optimization_status = "optimal"  # Placeholder
                    except Exception:
                        health.optimization_status = "unknown"

                    collections_health.append(health)

                except Exception as e:
                    logger.warning(
                        f"Failed to get health for collection {collection_name}: {e}"
                    )
                    # Add minimal health info for failed collection
                    collections_health.append(
                        CollectionHealth(
                            name=collection_name,
                            vector_count=0,
                            indexed_vectors_count=0,
                            points_count=0,
                            segments_count=0,
                            disk_data_size=0,
                            ram_data_size=0,
                            config_params={},
                            status="error",
                            optimization_status="error",
                        )
                    )

        except Exception as e:
            logger.error(f"Failed to check collection health: {e}")

        return collections_health

    async def _check_performance_metrics(self) -> PerformanceMetrics:
        """Check system performance metrics."""
        metrics = PerformanceMetrics()

        try:
            if not self.client:
                return metrics

            # Test embedding performance
            embedding_service = self.client.get_embedding_service()
            test_texts = ["Performance test document for health monitoring."]

            start_time = time.time()
            try:
                await embedding_service.generate_embeddings(test_texts)
                embedding_time = (time.time() - start_time) * 1000  # ms
                metrics.embedding_avg_time_ms = round(embedding_time, 2)
            except Exception as e:
                logger.warning(f"Embedding performance test failed: {e}")

            # Test search performance if collections exist
            collections = self.client.list_collections()
            if collections:
                try:
                    from workspace_qdrant_mcp.tools.search import semantic_search

                    start_time = time.time()
                    await semantic_search(
                        query="health monitoring test",
                        collection=collections[0],
                        limit=5,
                    )
                    search_time = (time.time() - start_time) * 1000  # ms
                    metrics.search_avg_time_ms = round(search_time, 2)
                except Exception as e:
                    logger.warning(f"Search performance test failed: {e}")

            # Calculate uptime (approximate based on process start)
            try:
                import psutil

                process = psutil.Process()
                uptime_seconds = time.time() - process.create_time()
                metrics.uptime_hours = round(uptime_seconds / 3600, 1)
            except Exception:
                pass

            # Throughput estimation (would need historical data in production)
            if metrics.embedding_avg_time_ms:
                # Rough estimate: 1000ms / avg_time_ms = docs per second
                metrics.throughput_docs_per_sec = round(
                    1000 / metrics.embedding_avg_time_ms, 1
                )

            # Error rate would be tracked over time in production
            metrics.error_rate_percent = 0.0  # Placeholder

        except Exception as e:
            logger.error(f"Failed to check performance metrics: {e}")

        return metrics

    async def _check_configuration_validity(self) -> dict[str, Any]:
        """Check configuration integrity and validity."""
        status = {
            "valid": False,
            "issues": [],
            "source": "unknown",
            "last_modified": None,
        }

        try:
            validator = ConfigValidator(self.config)
            is_valid, results = await asyncio.get_event_loop().run_in_executor(
                None, validator.validate_all
            )

            status["valid"] = is_valid
            status["issues"] = results.get("issues", [])

            # Check configuration source
            if Path(".env").exists():
                status["source"] = ".env file"
                env_path = Path(".env")
                status["last_modified"] = datetime.fromtimestamp(
                    env_path.stat().st_mtime
                ).isoformat()
            elif any(k.startswith("WORKSPACE_QDRANT_") for k in os.environ):
                status["source"] = "environment variables"
            else:
                status["source"] = "defaults"

        except Exception as e:
            status["issues"].append(f"Configuration validation failed: {e}")
            logger.error(f"Configuration validation failed: {e}")

        return status

    def _generate_alerts(
        self,
        resources: ResourceStatus,
        collections: list[CollectionHealth],
        performance: PerformanceMetrics,
    ) -> list[str]:
        """Generate system alerts based on health data."""
        alerts = []

        # Resource alerts
        if resources.cpu_percent > 90:
            alerts.append(f"Warning: High CPU usage: {resources.cpu_percent}%")
        elif resources.cpu_percent > 80:
            alerts.append(f"🟡 Elevated CPU usage: {resources.cpu_percent}%")

        if resources.memory_percent > 95:
            alerts.append(
                f"Warning: Critical memory usage: {resources.memory_percent}%"
            )
        elif resources.memory_percent > 85:
            alerts.append(f"🟡 High memory usage: {resources.memory_percent}%")

        if resources.disk_percent > 95:
            alerts.append(f"Warning: Critical disk space: {resources.disk_percent}%")
        elif resources.disk_percent > 90:
            alerts.append(f"🟡 Low disk space: {resources.disk_percent}%")

        if resources.load_average and resources.load_average > 5.0:
            alerts.append(f"Warning: High system load: {resources.load_average}")

        # Collection alerts
        for collection in collections:
            if collection.status == "error":
                alerts.append(f"Error: Collection '{collection.name}' in error state")
            elif collection.optimization_status == "error":
                alerts.append(
                    f"Warning: Collection '{collection.name}' needs optimization"
                )

            # Check for large unindexed vectors
            if (
                collection.vector_count > collection.indexed_vectors_count * 1.1
            ):  # 10% threshold
                unindexed = collection.vector_count - collection.indexed_vectors_count
                alerts.append(
                    f"🟡 Collection '{collection.name}' has {unindexed} unindexed vectors"
                )

        # Performance alerts
        if (
            performance.embedding_avg_time_ms
            and performance.embedding_avg_time_ms > 5000
        ):  # 5 seconds
            alerts.append(
                f"Warning: Slow embedding generation: {performance.embedding_avg_time_ms}ms"
            )

        if (
            performance.search_avg_time_ms and performance.search_avg_time_ms > 1000
        ):  # 1 second
            alerts.append(
                f"Warning: Slow search performance: {performance.search_avg_time_ms}ms"
            )

        if (
            performance.error_rate_percent and performance.error_rate_percent > 5
        ):  # 5% error rate
            alerts.append(
                f"Warning: High error rate: {performance.error_rate_percent}%"
            )

        return alerts

    def _generate_recommendations(
        self,
        resources: ResourceStatus,
        collections: list[CollectionHealth],
        performance: PerformanceMetrics,
        config_status: dict[str, Any],
    ) -> list[str]:
        """Generate optimization recommendations."""
        recommendations = []

        # Resource optimization
        if resources.memory_percent > 80:
            recommendations.append(
                "📏 Consider reducing embedding batch size to lower memory usage"
            )
            recommendations.append("Restart the system to free up memory if possible")

        if resources.disk_percent > 85:
            recommendations.append("📋 Clean up old logs and temporary files")
            recommendations.append("📁 Consider archiving old collections or data")

        # Collection optimization
        total_vectors = sum(col.vector_count for col in collections)
        if total_vectors > 1000000:  # 1M vectors
            recommendations.append(
                "⚡ Consider enabling collection sharding for better performance"
            )
            recommendations.append("Monitor collection segment optimization status")

        for collection in collections:
            if collection.segments_count > 20:
                recommendations.append(
                    f"🔧 Collection '{collection.name}' may benefit from optimization (many segments)"
                )

            if (
                collection.ram_data_size > collection.disk_data_size * 0.8
            ):  # Most data in RAM
                recommendations.append(
                    f"💾 Collection '{collection.name}' using significant RAM - consider disk optimization"
                )

        # Performance optimization
        if (
            performance.embedding_avg_time_ms
            and performance.embedding_avg_time_ms > 1000
        ):
            recommendations.append(
                "🧠 Consider using a smaller embedding model for better performance"
            )
            recommendations.append("⚡ Enable GPU acceleration if available")

        if performance.search_avg_time_ms and performance.search_avg_time_ms > 500:
            recommendations.append("Optimize search parameters and query construction")
            recommendations.append("📊 Consider using more specific search filters")

        # Configuration recommendations
        if not config_status.get("valid", False):
            recommendations.append("Fix configuration issues for optimal performance")
            for issue in config_status.get("issues", [])[:3]:
                recommendations.append(f"  - {issue}")

        if not self.config.embedding.enable_sparse_vectors:
            recommendations.append("Enable sparse vectors for improved search quality")

        if self.config.embedding.chunk_size > 2000:
            recommendations.append(
                "📏 Consider smaller chunk size for better search precision"
            )

        # General recommendations
        if not recommendations:
            recommendations.extend(
                [
                    "System is running optimally!",
                    "📊 Monitor performance trends over time",
                    "Regular system maintenance is recommended",
                ]
            )

        return recommendations

    def _calculate_overall_health(
        self,
        alerts: list[str],
        resources: ResourceStatus,
        collections: list[CollectionHealth],
    ) -> str:
        """Calculate overall system health status."""

        # Check for critical alerts
        critical_keywords = ["Warning:", "Error:", "critical", "error"]
        has_critical = any(
            any(keyword in alert for keyword in critical_keywords) for alert in alerts
        )

        if has_critical:
            return "critical"

        # Check for warning conditions
        warning_conditions = [
            resources.cpu_percent > 80,
            resources.memory_percent > 85,
            resources.disk_percent > 90,
            any(
                col.status != "green" for col in collections if col.status != "unknown"
            ),
            len(alerts) > 0,
        ]

        if any(warning_conditions):
            return "warning"

        return "healthy"

    async def run_continuous_monitoring(self, interval_seconds: int = 30) -> None:
        """Run continuous health monitoring with live updates."""

        # Create layout for live updates
        layout = Layout()
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="body"),
            Layout(name="footer", size=3),
        )

        layout["body"].split_row(
            Layout(name="left"),
            Layout(name="right"),
        )

        with Live(layout, refresh_per_second=1, screen=True):
            while True:
                try:
                    # Get current health report
                    report = await self.check_system_health()

                    # Update layout
                    self._update_live_layout(layout, report)

                    # Wait for next check
                    await asyncio.sleep(interval_seconds)

                except KeyboardInterrupt:
                    break
                except Exception as e:
                    logger.error(f"Monitoring error: {e}")
                    await asyncio.sleep(interval_seconds)

    def _update_live_layout(self, layout: Layout, report: HealthReport) -> None:
        """Update the live monitoring layout with current data."""

        # Header
        status_color = {"healthy": "green", "warning": "yellow", "critical": "red"}[
            report.overall_health
        ]

        header_text = Text()
        header_text.append("Workspace Qdrant MCP Health Monitor\n", style="bold cyan")
        header_text.append(
            f"Status: {report.overall_health.upper()} ", style=f"bold {status_color}"
        )
        header_text.append(
            f"| Last Update: {report.timestamp.strftime('%H:%M:%S')}", style="dim"
        )

        layout["header"].update(Panel(header_text, border_style=status_color))

        # System resources (left panel)
        resources_table = Table(
            title="System Resources", show_header=True, header_style="bold magenta"
        )
        resources_table.add_column("Resource", style="cyan", width=12)
        resources_table.add_column("Usage", style="white", width=20)
        resources_table.add_column("Status", style="white", width=8)

        # CPU
        cpu_status = (
            "[ERROR]"
            if report.system_resources.cpu_percent > 90
            else "[WARN]"
            if report.system_resources.cpu_percent > 80
            else "[OK]"
        )
        resources_table.add_row(
            "CPU", f"{report.system_resources.cpu_percent}%", cpu_status
        )

        # Memory
        mem_status = (
            "[ERROR]"
            if report.system_resources.memory_percent > 95
            else "[WARN]"
            if report.system_resources.memory_percent > 85
            else "[OK]"
        )
        resources_table.add_row(
            "Memory",
            f"{report.system_resources.memory_used_gb:.1f}GB / {report.system_resources.memory_total_gb:.1f}GB ({report.system_resources.memory_percent}%)",
            mem_status,
        )

        # Disk
        disk_status = (
            "[ERROR]"
            if report.system_resources.disk_percent > 95
            else "[WARN]"
            if report.system_resources.disk_percent > 90
            else "[OK]"
        )
        resources_table.add_row(
            "Disk",
            f"{report.system_resources.disk_used_gb:.1f}GB / {report.system_resources.disk_total_gb:.1f}GB ({report.system_resources.disk_percent}%)",
            disk_status,
        )

        if report.system_resources.load_average is not None:
            load_status = (
                "[ERROR]"
                if report.system_resources.load_average > 5
                else "[WARN]"
                if report.system_resources.load_average > 2
                else "[OK]"
            )
            resources_table.add_row(
                "Load Avg", f"{report.system_resources.load_average:.2f}", load_status
            )

        layout["left"].update(resources_table)

        # Collections and performance (right panel)
        right_content = ""

        # Collections summary
        if report.collections:
            right_content += f"Collections ({len(report.collections)}):\n"
            for col in report.collections[:5]:  # Show top 5
                status_icon = (
                    "[OK]"
                    if col.status == "green"
                    else "[WARN]"
                    if col.status != "error"
                    else "[ERROR]"
                )
                right_content += (
                    f"  {status_icon} {col.name}: {col.points_count:,} points\n"
                )

            if len(report.collections) > 5:
                right_content += f"  ... and {len(report.collections) - 5} more\n"

        right_content += "\n"

        # Performance metrics
        if report.performance_metrics:
            right_content += "Performance:\n"
            if report.performance_metrics.embedding_avg_time_ms:
                right_content += f"  Embedding: {report.performance_metrics.embedding_avg_time_ms:.0f}ms\n"
            if report.performance_metrics.search_avg_time_ms:
                right_content += (
                    f"  Search: {report.performance_metrics.search_avg_time_ms:.0f}ms\n"
                )
            if report.performance_metrics.throughput_docs_per_sec:
                right_content += f"  Throughput: {report.performance_metrics.throughput_docs_per_sec:.1f} docs/s\n"

        layout["right"].update(
            Panel(right_content, title="Collections & Performance", border_style="blue")
        )

        # Footer - alerts and recommendations
        footer_text = ""
        if report.alerts:
            footer_text = "Alerts: " + " | ".join(report.alerts[:3])
            if len(report.alerts) > 3:
                footer_text += f" | +{len(report.alerts) - 3} more"
        else:
            footer_text = "No active alerts - system running normally"

        layout["footer"].update(
            Panel(footer_text, border_style="yellow" if report.alerts else "green")
        )

    async def close(self) -> None:
        """Clean up resources."""
        if self.client:
            await self.client.close()


@app.command()
def main(
    watch: bool = typer.Option(
        False, "--watch", "-w", help="Continuous monitoring mode"
    ),
    interval: int = typer.Option(
        30, "--interval", "-i", help="Monitoring interval in seconds (watch mode)"
    ),
    analyze: bool = typer.Option(
        False, "--analyze", "-a", help="Perform detailed analysis"
    ),
    report: bool = typer.Option(
        False, "--report", "-r", help="Generate JSON report file"
    ),
    output: str | None = typer.Option(
        None, "--output", "-o", help="Output file for JSON report"
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
) -> None:
    """
    Monitor system health and performance of workspace-qdrant-mcp.

    This tool provides comprehensive health monitoring including system resources,
    collection status, performance metrics, and optimization recommendations.

    Examples:
        # One-time health check
        workspace-qdrant-health

        # Continuous monitoring
        workspace-qdrant-health --watch

        # Detailed analysis
        workspace-qdrant-health --analyze

        # Generate health report
        workspace-qdrant-health --report --output health_report.json

        # Watch mode with custom interval
        workspace-qdrant-health --watch --interval 60
    """

    async def run_health_monitor():
        monitor = None
        try:
            # Load configuration
            config = get_config()

            # Create health monitor
            monitor = HealthMonitor(
                config=config, watch_mode=watch, analyze_mode=analyze
            )

            if watch:
                # Run continuous monitoring
                console.print(
                    "👀 Starting continuous health monitoring...", style="blue"
                )
                console.print(f"Press Ctrl+C to stop. Update interval: {interval}s\n")
                await monitor.run_continuous_monitoring(interval)
            else:
                # Run single health check
                console.print("Running health check...", style="blue")

                with Progress(
                    SpinnerColumn(),
                    TextColumn("[blue]Checking system health..."),
                    console=console,
                    transient=True,
                ) as progress:
                    progress.add_task("health", total=None)
                    health_report = await monitor.check_system_health()

                # Display results
                display_health_report(health_report, analyze, verbose)

                # Generate JSON report if requested
                if report:
                    output_file = (
                        output
                        or f"workspace_qdrant_health_report_{int(time.time())}.json"
                    )
                    output_path = Path(output_file)

                    try:
                        output_path.write_text(
                            json.dumps(health_report.to_dict(), indent=2)
                        )
                        console.print(
                            f"\nHealth report saved to: {output_path.absolute()}",
                            style="green",
                        )
                    except Exception as e:
                        console.print(
                            f"\nWarning: Failed to save report: {e}", style="yellow"
                        )

                # Exit with appropriate code
                exit_code = {"healthy": 0, "warning": 1, "critical": 2}.get(
                    health_report.overall_health, 2
                )

                raise typer.Exit(exit_code)

        except KeyboardInterrupt:
            console.print("\nError: Health monitoring cancelled by user", style="red")
            raise typer.Exit(1)
        except Exception as e:
            console.print(f"\nError: Health monitoring failed: {e}", style="red")
            logger.error(f"Health monitoring failed: {e}", exc_info=True)
            raise typer.Exit(1)
        finally:
            if monitor:
                await monitor.close()

    # Run the async health monitor
    asyncio.run(run_health_monitor())


def display_health_report(report: HealthReport, analyze: bool, verbose: bool) -> None:
    """Display health report with rich formatting."""

    # Overall status header
    status_colors = {"healthy": "green", "warning": "yellow", "critical": "red"}

    status_icons = {"healthy": "[OK]", "warning": "[WARN]", "critical": "[ERROR]"}

    status_color = status_colors[report.overall_health]
    status_icon = status_icons[report.overall_health]

    header_text = Text()
    header_text.append(f"{status_icon} System Health: ", style="bold white")
    header_text.append(report.overall_health.upper(), style=f"bold {status_color}")
    header_text.append(
        f"\nLast checked: {report.timestamp.strftime('%Y-%m-%d %H:%M:%S')}", style="dim"
    )

    console.print(
        Panel(
            header_text,
            title="🏥 Health Status",
            border_style=status_color,
            padding=(1, 2),
        )
    )

    # System resources
    resources_table = Table(
        title="System Resources", show_header=True, header_style="bold magenta"
    )
    resources_table.add_column("Resource", style="cyan")
    resources_table.add_column("Current", style="white")
    resources_table.add_column("Status", style="white")

    # Add resource rows with status indicators
    resources = [
        (
            "CPU Usage",
            f"{report.system_resources.cpu_percent}%",
            "[ERROR]"
            if report.system_resources.cpu_percent > 90
            else "[WARN]"
            if report.system_resources.cpu_percent > 80
            else "[OK]",
        ),
        (
            "Memory Usage",
            f"{report.system_resources.memory_percent}% ({report.system_resources.memory_used_gb:.1f}GB / {report.system_resources.memory_total_gb:.1f}GB)",
            "[ERROR]"
            if report.system_resources.memory_percent > 95
            else "[WARN]"
            if report.system_resources.memory_percent > 85
            else "[OK]",
        ),
        (
            "Disk Usage",
            f"{report.system_resources.disk_percent}% ({report.system_resources.disk_used_gb:.1f}GB / {report.system_resources.disk_total_gb:.1f}GB)",
            "[ERROR]"
            if report.system_resources.disk_percent > 95
            else "[WARN]"
            if report.system_resources.disk_percent > 90
            else "[OK]",
        ),
    ]

    if report.system_resources.load_average is not None:
        resources.append(
            (
                "Load Average",
                f"{report.system_resources.load_average:.2f}",
                "[ERROR]"
                if report.system_resources.load_average > 5
                else "[WARN]"
                if report.system_resources.load_average > 2
                else "[OK]",
            )
        )

    for resource, value, status in resources:
        resources_table.add_row(resource, value, status)

    console.print(resources_table)

    # Collections summary
    if report.collections:
        collections_table = Table(
            title="Collections Health", show_header=True, header_style="bold magenta"
        )
        collections_table.add_column("Collection", style="cyan")
        collections_table.add_column("Points", style="white")
        collections_table.add_column("Vectors", style="white")
        collections_table.add_column("Status", style="white")

        for col in report.collections:
            status_icon = (
                "[OK]"
                if col.status == "green"
                else "[WARN]"
                if col.status != "error"
                else "[ERROR]"
            )
            collections_table.add_row(
                col.name,
                f"{col.points_count:,}",
                f"{col.vector_count:,}"
                + (
                    f" ({col.indexed_vectors_count:,} indexed)"
                    if col.vector_count != col.indexed_vectors_count
                    else ""
                ),
                f"{status_icon} {col.status}",
            )

        console.print(collections_table)

    # Performance metrics
    if any(
        [
            report.performance_metrics.embedding_avg_time_ms,
            report.performance_metrics.search_avg_time_ms,
            report.performance_metrics.throughput_docs_per_sec,
        ]
    ):
        perf_table = Table(title="Performance Metrics", show_header=True)
        perf_table.add_column("Metric", style="cyan")
        perf_table.add_column("Value", style="white")

        if report.performance_metrics.embedding_avg_time_ms:
            perf_table.add_row(
                "Embedding Generation",
                f"{report.performance_metrics.embedding_avg_time_ms:.0f}ms",
            )

        if report.performance_metrics.search_avg_time_ms:
            perf_table.add_row(
                "Search Response Time",
                f"{report.performance_metrics.search_avg_time_ms:.0f}ms",
            )

        if report.performance_metrics.throughput_docs_per_sec:
            perf_table.add_row(
                "Estimated Throughput",
                f"{report.performance_metrics.throughput_docs_per_sec:.1f} docs/sec",
            )

        if report.performance_metrics.uptime_hours:
            perf_table.add_row(
                "Process Uptime", f"{report.performance_metrics.uptime_hours:.1f} hours"
            )

        console.print(perf_table)

    # Alerts
    if report.alerts:
        alert_text = Text()
        for alert in report.alerts:
            alert_text.append(f"{alert}\n")

        console.print(
            Panel(alert_text, title="Active Alerts", border_style="red", padding=(1, 2))
        )

    # Recommendations
    if report.recommendations:
        rec_text = Text()
        for rec in report.recommendations:
            rec_text.append(f"{rec}\n")

        console.print(
            Panel(
                rec_text,
                title="Recommendations",
                border_style="yellow",
                padding=(1, 2),
            )
        )

    # Detailed analysis (if requested)
    if analyze and verbose:
        detail_text = Text()
        detail_text.append("Configuration Status:\n", style="bold")
        detail_text.append(
            f"  Valid: {report.configuration_status.get('valid', 'unknown')}\n"
        )
        detail_text.append(
            f"  Source: {report.configuration_status.get('source', 'unknown')}\n"
        )

        if report.configuration_status.get("issues"):
            detail_text.append("  Issues:\n", style="red")
            for issue in report.configuration_status["issues"][:5]:
                detail_text.append(f"    - {issue}\n", style="red")

        console.print(
            Panel(
                detail_text,
                title="Detailed Analysis",
                border_style="blue",
                padding=(1, 2),
            )
        )


if __name__ == "__main__":
    app()
