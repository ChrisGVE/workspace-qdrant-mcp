#!/usr/bin/env python3
"""
Command-line benchmarking tool for metadata filtering performance.

This CLI tool provides comprehensive benchmarking capabilities to compare
current performance against established baselines (2.18ms response time,
94.2% precision). Supports automated regression testing and performance
monitoring for production systems.

Usage Examples:
    # Run basic performance benchmark
    python performance_benchmark_cli.py benchmark --collection documents --queries 50

    # Run with custom baseline comparison
    python performance_benchmark_cli.py benchmark --baseline-response-time 3.0 --baseline-precision 90.0

    # Run multi-tenant isolation benchmark
    python performance_benchmark_cli.py multi-tenant --tenants 5 --queries-per-tenant 20

    # Generate performance report
    python performance_benchmark_cli.py report --output performance_report.json

    # Run continuous monitoring mode
    python performance_benchmark_cli.py monitor --interval 300 --alert-threshold 3.0

Task 233.6: CLI benchmarking tools for baseline comparison and automated monitoring.
"""

import asyncio
import json
import sys
import time
from datetime import datetime

import click
from loguru import logger
from qdrant_client import QdrantClient

from ..core.hybrid_search import HybridSearchEngine

# Import our performance monitoring components
from ..core.performance_monitoring import (
    MetadataFilteringPerformanceMonitor,
    PerformanceBenchmarkResult,
)
from ..core.ssl_config import suppress_qdrant_ssl_warnings


class PerformanceBenchmarkCLI:
    """Command-line interface for performance benchmarking."""

    def __init__(
        self,
        qdrant_url: str = "http://localhost:6333",
        qdrant_api_key: str | None = None,
        baseline_config: dict | None = None
    ):
        """Initialize CLI with Qdrant connection and baseline configuration."""
        self.qdrant_url = qdrant_url
        self.qdrant_api_key = qdrant_api_key

        # Initialize Qdrant client
        with suppress_qdrant_ssl_warnings():
            self.client = QdrantClient(
                url=qdrant_url,
                api_key=qdrant_api_key,
                timeout=30
            )

        # Initialize search engine and performance monitor
        self.search_engine = HybridSearchEngine(
            client=self.client,
            enable_optimizations=True,
            enable_multi_tenant_aggregation=True
        )

        self.performance_monitor = MetadataFilteringPerformanceMonitor(
            search_engine=self.search_engine,
            baseline_config=baseline_config
        )

        logger.info("Performance benchmark CLI initialized",
                   qdrant_url=qdrant_url,
                   baseline_response_time=self.performance_monitor.baseline.target_response_time)

    def generate_synthetic_test_data(self, count: int = 50, complexity_range: tuple = (1, 10)) -> list[dict]:
        """Generate synthetic test data for benchmarking."""
        import random

        test_queries = []
        project_names = ["project_alpha", "project_beta", "project_gamma", "project_delta"]
        collection_types = ["project", "scratchbook", "notes", "docs"]

        for i in range(count):
            # Generate embeddings
            dense_embedding = [random.gauss(0, 1) for _ in range(384)]
            sparse_indices = sorted(random.sample(range(1000), k=random.randint(5, 15)))
            sparse_values = [random.uniform(0.1, 1.0) for _ in range(len(sparse_indices))]

            query = {
                "id": f"bench_query_{i}",
                "embeddings": {
                    "dense": dense_embedding,
                    "sparse": {"indices": sparse_indices, "values": sparse_values}
                },
                "project_context": {
                    "project_name": random.choice(project_names),
                    "collection_type": random.choice(collection_types),
                    "tenant_namespace": f"tenant_{random.randint(1, 3)}",
                    "workspace_scope": random.choice(["project", "shared"])
                },
                "query_text": f"benchmark query {i} with random search terms",
                "expected_result_count": random.randint(5, 20),
                "complexity_score": random.randint(*complexity_range)
            }
            test_queries.append(query)

        logger.info(f"Generated {count} synthetic test queries")
        return test_queries

    async def run_performance_benchmark(
        self,
        collection_name: str,
        query_count: int = 50,
        iterations: int = 10,
        output_file: str | None = None
    ) -> PerformanceBenchmarkResult:
        """Run comprehensive performance benchmark."""
        click.echo(f"🚀 Starting performance benchmark: {collection_name}")
        click.echo(f"   Queries: {query_count}, Iterations per query: {iterations}")

        # Generate test queries
        test_queries = self.generate_synthetic_test_data(query_count)

        # Run benchmark
        start_time = time.time()
        result = await self.performance_monitor.benchmark_suite.run_metadata_filtering_benchmark(
            collection_name=collection_name,
            test_queries=test_queries,
            iterations=iterations
        )
        total_benchmark_time = time.time() - start_time

        # Display results
        self._display_benchmark_results(result, total_benchmark_time)

        # Save results if output file specified
        if output_file:
            await self._save_benchmark_results(result, output_file)

        return result

    async def run_multi_tenant_benchmark(
        self,
        collection_name: str,
        tenant_count: int = 3,
        queries_per_tenant: int = 10,
        output_file: str | None = None
    ) -> PerformanceBenchmarkResult:
        """Run multi-tenant isolation benchmark."""
        click.echo(f"🏢 Starting multi-tenant benchmark: {collection_name}")
        click.echo(f"   Tenants: {tenant_count}, Queries per tenant: {queries_per_tenant}")

        # Generate multi-tenant test data
        tenant_test_data = {}
        for tenant_id in range(tenant_count):
            tenant_namespace = f"tenant_{tenant_id}"
            tenant_queries = self.generate_synthetic_test_data(
                count=queries_per_tenant,
                complexity_range=(1, 5)
            )

            # Add tenant-specific context to queries
            for query in tenant_queries:
                query["project_context"]["tenant_namespace"] = tenant_namespace

            tenant_test_data[tenant_namespace] = tenant_queries

        # Run multi-tenant benchmark
        start_time = time.time()
        result = await self.performance_monitor.benchmark_suite.run_multi_tenant_isolation_benchmark(
            collection_name=collection_name,
            tenant_test_data=tenant_test_data
        )
        total_benchmark_time = time.time() - start_time

        # Display results
        self._display_multi_tenant_results(result, total_benchmark_time)

        # Save results if output file specified
        if output_file:
            await self._save_benchmark_results(result, output_file)

        return result

    def _display_benchmark_results(self, result: PerformanceBenchmarkResult, total_time: float):
        """Display benchmark results in formatted output."""
        baseline = self.performance_monitor.baseline

        click.echo("\n" + "="*60)
        click.echo("📊 PERFORMANCE BENCHMARK RESULTS")
        click.echo("="*60)

        # Response Time Results
        click.echo("\n⏱️  RESPONSE TIME ANALYSIS")
        click.echo(f"   Average:     {result.avg_response_time:.2f}ms")
        click.echo(f"   P50:         {result.p50_response_time:.2f}ms")
        click.echo(f"   P95:         {result.p95_response_time:.2f}ms")
        click.echo(f"   P99:         {result.p99_response_time:.2f}ms")

        # Baseline Comparison
        click.echo("\n📏 BASELINE COMPARISON")
        click.echo(f"   Target (2.18ms):     {'✅ PASS' if result.avg_response_time <= baseline.target_response_time else '❌ FAIL'} ({result.avg_response_time:.2f}ms)")
        click.echo(f"   Acceptable (3.0ms):  {'✅ PASS' if result.avg_response_time <= baseline.acceptable_response_time else '❌ FAIL'}")

        # Accuracy Results
        if result.avg_precision > 0:
            click.echo("\n🎯 ACCURACY ANALYSIS")
            click.echo(f"   Precision:   {result.avg_precision:.1f}% (target: {baseline.target_precision:.1f}%)")
            click.echo(f"   Recall:      {result.avg_recall:.1f}% (target: {baseline.target_recall:.1f}%)")
            click.echo(f"   F1 Score:    {result.avg_f1_score:.1f}%")

        # Performance Status
        status = "🟢 EXCELLENT" if result.passes_baseline(baseline) else \
                 "🟡 ACCEPTABLE" if not result.performance_regression else \
                 "🔴 DEGRADED"
        click.echo(f"\n🚦 OVERALL STATUS: {status}")

        # Test Configuration
        click.echo("\n⚙️  TEST CONFIGURATION")
        for key, value in result.test_config.items():
            click.echo(f"   {key}: {value}")

        click.echo(f"\n⏲️  Total benchmark time: {total_time:.1f}s")
        click.echo("="*60)

    def _display_multi_tenant_results(self, result: PerformanceBenchmarkResult, total_time: float):
        """Display multi-tenant benchmark results."""
        isolation_info = result.metadata.get("tenant_isolation", {})

        click.echo("\n" + "="*60)
        click.echo("🏢 MULTI-TENANT ISOLATION RESULTS")
        click.echo("="*60)

        # Isolation Results
        click.echo("\n🔒 TENANT ISOLATION")
        enforcement_rate = isolation_info.get("enforcement_rate", 0)
        violations = isolation_info.get("violations", 0)
        total_queries = isolation_info.get("total_queries", 0)

        click.echo(f"   Enforcement Rate:  {enforcement_rate:.1f}%")
        click.echo(f"   Violations:        {violations}")
        click.echo(f"   Total Queries:     {total_queries}")
        click.echo(f"   Isolation Status:  {'✅ PASS' if violations == 0 else '❌ FAIL'}")

        # Performance Results
        click.echo("\n⏱️  MULTI-TENANT PERFORMANCE")
        click.echo(f"   Average Response:  {result.avg_response_time:.2f}ms")
        click.echo(f"   P95 Response:      {result.p95_response_time:.2f}ms")

        # Overall Status
        isolation_pass = enforcement_rate >= self.performance_monitor.baseline.tenant_isolation_enforcement
        performance_pass = result.avg_response_time <= self.performance_monitor.baseline.acceptable_response_time

        if isolation_pass and performance_pass:
            status = "🟢 EXCELLENT"
        elif isolation_pass or performance_pass:
            status = "🟡 PARTIAL"
        else:
            status = "🔴 FAILED"

        click.echo(f"\n🚦 MULTI-TENANT STATUS: {status}")
        click.echo(f"⏲️  Total benchmark time: {total_time:.1f}s")
        click.echo("="*60)

    async def _save_benchmark_results(self, result: PerformanceBenchmarkResult, output_file: str):
        """Save benchmark results to file."""
        try:
            result_data = {
                "benchmark_id": result.benchmark_id,
                "timestamp": result.timestamp.isoformat(),
                "test_name": result.test_name,
                "performance_metrics": {
                    "avg_response_time": result.avg_response_time,
                    "p50_response_time": result.p50_response_time,
                    "p95_response_time": result.p95_response_time,
                    "p99_response_time": result.p99_response_time,
                },
                "accuracy_metrics": {
                    "avg_precision": result.avg_precision,
                    "avg_recall": result.avg_recall,
                    "avg_f1_score": result.avg_f1_score,
                },
                "baseline_comparison": result.baseline_comparison,
                "test_config": result.test_config,
                "metadata": result.metadata,
                "passes_baseline": result.passes_baseline(self.performance_monitor.baseline),
                "performance_regression": result.performance_regression,
                "accuracy_regression": result.accuracy_regression
            }

            with open(output_file, 'w') as f:
                json.dump(result_data, f, indent=2, default=str)

            click.echo(f"📄 Results saved to: {output_file}")

        except Exception as e:
            click.echo(f"❌ Failed to save results: {e}")

    async def generate_performance_report(self, output_file: str = "performance_report.json"):
        """Generate comprehensive performance report."""
        click.echo("📋 Generating comprehensive performance report...")

        # Get performance status
        status = self.performance_monitor.get_performance_status()

        # Get dashboard data
        dashboard_data = self.performance_monitor.dashboard.get_real_time_dashboard()

        # Generate report
        {
            "report_generated": datetime.now().isoformat(),
            "qdrant_connection": {
                "url": self.qdrant_url,
                "status": "connected"  # Would check actual connection status
            },
            "baseline_configuration": status["baseline_configuration"],
            "performance_status": status,
            "dashboard_snapshot": dashboard_data,
            "benchmark_history": [
                {
                    "benchmark_id": b.benchmark_id,
                    "timestamp": b.timestamp.isoformat(),
                    "test_name": b.test_name,
                    "avg_response_time": b.avg_response_time,
                    "avg_precision": b.avg_precision,
                    "passes_baseline": b.passes_baseline(self.performance_monitor.baseline)
                }
                for b in self.performance_monitor.benchmark_suite.get_benchmark_history()
            ]
        }

        # Export dashboard report
        export_result = self.performance_monitor.dashboard.export_performance_report(output_file)

        if export_result.get("success"):
            click.echo(f"✅ Performance report generated: {output_file}")
            return export_result
        else:
            click.echo(f"❌ Failed to generate report: {export_result.get('error')}")
            return None

    async def continuous_monitoring(
        self,
        collection_name: str,
        interval_seconds: int = 300,
        alert_threshold: float = 3.0,
        max_iterations: int = 100
    ):
        """Run continuous performance monitoring."""
        click.echo(f"🔄 Starting continuous monitoring: {collection_name}")
        click.echo(f"   Interval: {interval_seconds}s, Alert threshold: {alert_threshold}ms")

        iteration = 0
        while iteration < max_iterations:
            try:
                click.echo(f"\n⏰ Monitoring iteration {iteration + 1}/{max_iterations}")

                # Generate small test query for monitoring
                test_queries = self.generate_synthetic_test_data(count=3, complexity_range=(1, 5))

                # Run quick benchmark
                start_time = time.time()
                result = await self.performance_monitor.benchmark_suite.run_metadata_filtering_benchmark(
                    collection_name=collection_name,
                    test_queries=test_queries,
                    iterations=5
                )
                monitoring_time = time.time() - start_time

                # Check against alert threshold
                if result.avg_response_time > alert_threshold:
                    click.echo(f"🚨 PERFORMANCE ALERT: {result.avg_response_time:.2f}ms > {alert_threshold}ms")

                    # Generate alert report
                    alert_data = {
                        "timestamp": datetime.now().isoformat(),
                        "alert_type": "performance_degradation",
                        "measured_response_time": result.avg_response_time,
                        "threshold": alert_threshold,
                        "baseline_target": self.performance_monitor.baseline.target_response_time,
                        "monitoring_iteration": iteration + 1
                    }

                    alert_file = f"performance_alert_{int(time.time())}.json"
                    with open(alert_file, 'w') as f:
                        json.dump(alert_data, f, indent=2)

                    click.echo(f"📄 Alert details saved to: {alert_file}")
                else:
                    click.echo(f"✅ Performance normal: {result.avg_response_time:.2f}ms")

                # Record real-time metric
                self.performance_monitor.dashboard.record_real_time_metric(
                    operation_type="continuous_monitoring",
                    response_time=result.avg_response_time,
                    metadata={
                        "iteration": iteration + 1,
                        "monitoring_time": monitoring_time
                    }
                )

                iteration += 1

                # Wait for next iteration
                if iteration < max_iterations:
                    click.echo(f"⏳ Waiting {interval_seconds}s until next check...")
                    await asyncio.sleep(interval_seconds)

            except KeyboardInterrupt:
                click.echo("\n🛑 Monitoring stopped by user")
                break
            except Exception as e:
                click.echo(f"❌ Monitoring error: {e}")
                await asyncio.sleep(10)  # Short pause before retry

        click.echo(f"🏁 Continuous monitoring completed: {iteration} iterations")


# Click CLI interface
@click.group()
@click.option('--qdrant-url', default='http://localhost:6333', help='Qdrant server URL')
@click.option('--qdrant-api-key', help='Qdrant API key')
@click.option('--baseline-response-time', type=float, default=2.18, help='Baseline response time (ms)')
@click.option('--baseline-precision', type=float, default=94.2, help='Baseline precision (%)')
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
@click.pass_context
def cli(ctx, qdrant_url, qdrant_api_key, baseline_response_time, baseline_precision, verbose):
    """Performance benchmarking CLI for metadata filtering operations."""
    # Configure logging
    if verbose:
        logger.add(sys.stderr, level="DEBUG")
    else:
        logger.add(sys.stderr, level="INFO")

    # Create baseline configuration
    baseline_config = {
        "target_response_time": baseline_response_time,
        "target_precision": baseline_precision
    }

    # Initialize CLI context
    ctx.ensure_object(dict)
    ctx.obj['benchmark_cli'] = PerformanceBenchmarkCLI(
        qdrant_url=qdrant_url,
        qdrant_api_key=qdrant_api_key,
        baseline_config=baseline_config
    )


@cli.command()
@click.option('--collection', required=True, help='Collection name to benchmark')
@click.option('--queries', default=50, help='Number of test queries to generate')
@click.option('--iterations', default=10, help='Iterations per query')
@click.option('--output', help='Output file for benchmark results')
@click.pass_context
def benchmark(ctx, collection, queries, iterations, output):
    """Run performance benchmark against response time and accuracy baselines."""
    benchmark_cli = ctx.obj['benchmark_cli']

    async def run_benchmark():
        result = await benchmark_cli.run_performance_benchmark(
            collection_name=collection,
            query_count=queries,
            iterations=iterations,
            output_file=output
        )

        # Exit with error code if benchmark fails baseline
        if not result.passes_baseline(benchmark_cli.performance_monitor.baseline):
            sys.exit(1)

    asyncio.run(run_benchmark())


@cli.command()
@click.option('--collection', required=True, help='Collection name to benchmark')
@click.option('--tenants', default=3, help='Number of tenants to test')
@click.option('--queries-per-tenant', default=10, help='Queries per tenant')
@click.option('--output', help='Output file for benchmark results')
@click.pass_context
def multi_tenant(ctx, collection, tenants, queries_per_tenant, output):
    """Run multi-tenant isolation benchmark."""
    benchmark_cli = ctx.obj['benchmark_cli']

    async def run_multi_tenant():
        await benchmark_cli.run_multi_tenant_benchmark(
            collection_name=collection,
            tenant_count=tenants,
            queries_per_tenant=queries_per_tenant,
            output_file=output
        )

    asyncio.run(run_multi_tenant())


@cli.command()
@click.option('--output', default='performance_report.json', help='Output file for report')
@click.pass_context
def report(ctx, output):
    """Generate comprehensive performance report."""
    benchmark_cli = ctx.obj['benchmark_cli']

    async def generate_report():
        await benchmark_cli.generate_performance_report(output_file=output)

    asyncio.run(generate_report())


@cli.command()
@click.option('--collection', required=True, help='Collection name to monitor')
@click.option('--interval', default=300, help='Monitoring interval in seconds')
@click.option('--alert-threshold', default=3.0, help='Alert threshold in ms')
@click.option('--max-iterations', default=100, help='Maximum monitoring iterations')
@click.pass_context
def monitor(ctx, collection, interval, alert_threshold, max_iterations):
    """Run continuous performance monitoring with alerting."""
    benchmark_cli = ctx.obj['benchmark_cli']

    async def run_monitoring():
        await benchmark_cli.continuous_monitoring(
            collection_name=collection,
            interval_seconds=interval,
            alert_threshold=alert_threshold,
            max_iterations=max_iterations
        )

    asyncio.run(run_monitoring())


if __name__ == '__main__':
    cli()
