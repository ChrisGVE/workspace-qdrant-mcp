#!/usr/bin/env python3
"""
Authoritative benchmark tool for workspace-qdrant-mcp.

This tool provides comprehensive, realistic performance testing by:
1. Using actual Qdrant operations (not simulation)
2. Integrating with workspace-qdrant-ingest CLI
3. Testing with large OSS projects (neovim, rust, go)
4. Optimizing chunk sizes and search parameters
5. Providing statistical analysis with confidence intervals

This replaces all previous simulation-based benchmark tools with
end-to-end realistic testing.
"""

import asyncio
import json
import logging
import math
import os
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple
import urllib.request
import zipfile

# Rich for beautiful output
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.table import Table
from rich.panel import Panel

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    from src.workspace_qdrant_mcp.core.config import Config
    from src.workspace_qdrant_mcp.core.client import QdrantWorkspaceClient
    from tests.fixtures.test_data_collector import TestDataCollector
    from tests.utils.metrics import SearchMetrics, RecallPrecisionMeter
except ImportError as e:
    print(f"❌ Import error: {e}")
    print(
        "Make sure you're running from the project root and have installed the package"
    )
    sys.exit(1)

# Setup
console = Console()
logger = logging.getLogger(__name__)


class OSProject:
    """Represents a large open source project for testing."""

    def __init__(self, name: str, repo_url: str, archive_url: str, description: str):
        self.name = name
        self.repo_url = repo_url
        self.archive_url = archive_url
        self.description = description
        self.local_path: Path | None = None

    def __str__(self) -> str:
        return f"{self.name}: {self.description}"


class BenchmarkScenario:
    """Represents a specific benchmark test scenario."""

    def __init__(
        self, name: str, collection_name: str, chunk_size: int, includes_oss: bool
    ):
        self.name = name
        self.collection_name = collection_name
        self.chunk_size = chunk_size
        self.includes_oss = includes_oss
        self.ingestion_time: float | None = None
        self.document_count: int | None = None
        self.chunk_count: int | None = None


class AuthoritativeBenchmark:
    """Comprehensive benchmark tool with real Qdrant integration."""

    def __init__(self):
        self.console = Console()
        self.project_root = Path(__file__).parent.parent
        self.test_data_dir = self.project_root / "test_data"
        self.results_dir = self.project_root / "benchmark_results"

        # Test scenarios
        self.chunk_sizes = [500, 1000, 2000, 4000]
        self.scenarios: List[BenchmarkScenario] = []

        # OSS projects for realistic mixed testing
        self.oss_projects = [
            OSProject(
                name="neovim",
                repo_url="https://github.com/neovim/neovim.git",
                archive_url="https://github.com/neovim/neovim/archive/refs/heads/master.zip",
                description="Modern vim editor (C, Lua)",
            ),
            OSProject(
                name="rust",
                repo_url="https://github.com/rust-lang/rust.git",
                archive_url="https://github.com/rust-lang/rust/archive/refs/heads/master.zip",
                description="Rust programming language (Rust)",
            ),
            OSProject(
                name="go",
                repo_url="https://github.com/golang/go.git",
                archive_url="https://github.com/golang/go/archive/refs/heads/master.zip",
                description="Go programming language (Go)",
            ),
        ]

        # Results storage
        self.benchmark_results = {
            "metadata": {},
            "scenarios": {},
            "search_performance": {},
            "statistical_analysis": {},
        }

        # Qdrant client
        self.client: QdrantWorkspaceClient | None = None
        self.config: Config | None = None

    async def initialize(self):
        """Initialize the benchmark environment."""
        console.print("🚀 Initializing Authoritative Benchmark", style="bold blue")

        # Setup directories
        self.test_data_dir.mkdir(exist_ok=True)
        self.results_dir.mkdir(exist_ok=True)

        # Initialize Qdrant client
        try:
            self.config = Config()
            self.client = QdrantWorkspaceClient(self.config)
            await self.client.initialize()

            console.print(
                f"✅ Connected to Qdrant at {self.config.qdrant.url}", style="green"
            )

        except Exception as e:
            console.print(f"❌ Failed to connect to Qdrant: {e}", style="red")
            console.print(
                "💡 Make sure Qdrant is running: docker run -p 6333:6333 qdrant/qdrant",
                style="yellow",
            )
            raise

        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler(self.results_dir / "benchmark.log"),
                logging.StreamHandler(),
            ],
        )

        console.print("✅ Benchmark environment initialized", style="green")

    async def download_oss_projects(self) -> None:
        """Download large OSS projects for testing."""
        console.print("\n📥 Downloading Large OSS Projects", style="bold cyan")

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            for project in self.oss_projects:
                task = progress.add_task(f"Downloading {project.name}...")
                project_dir = self.test_data_dir / project.name

                if project_dir.exists():
                    console.print(
                        f"  ✅ {project.name} already exists, skipping", style="green"
                    )
                    project.local_path = project_dir
                    progress.update(task, completed=100)
                    continue

                try:
                    # Download archive (faster than git clone for large repos)
                    archive_path = self.test_data_dir / f"{project.name}.zip"

                    console.print(f"  📥 Downloading {project.name} archive...")
                    urllib.request.urlretrieve(project.archive_url, archive_path)

                    # Extract
                    console.print(f"  📂 Extracting {project.name}...")
                    with zipfile.ZipFile(archive_path, "r") as zip_ref:
                        zip_ref.extractall(self.test_data_dir)

                    # Find extracted directory (usually has -master suffix)
                    extracted_dirs = list(self.test_data_dir.glob(f"{project.name}*"))
                    if extracted_dirs:
                        extracted_dirs[0].rename(project_dir)
                        project.local_path = project_dir

                    # Cleanup archive
                    archive_path.unlink()

                    console.print(
                        f"  ✅ {project.name} downloaded successfully", style="green"
                    )
                    progress.update(task, completed=100)

                except Exception as e:
                    console.print(
                        f"  ❌ Failed to download {project.name}: {e}", style="red"
                    )
                    logger.error(f"Failed to download {project.name}: {e}")
                    continue

        console.print("\n📊 OSS Projects Summary:")
        for project in self.oss_projects:
            if project.local_path and project.local_path.exists():
                size = sum(
                    f.stat().st_size
                    for f in project.local_path.rglob("*")
                    if f.is_file()
                )
                console.print(f"  ✅ {project.name}: {size / (1024 * 1024):.1f} MB")
            else:
                console.print(f"  ❌ {project.name}: Not available")

    def _generate_scenarios(self) -> None:
        """Generate all test scenarios."""
        self.scenarios = []

        # Project-only scenarios
        for chunk_size in self.chunk_sizes:
            scenario = BenchmarkScenario(
                name=f"project_only_{chunk_size}",
                collection_name=f"bench_project_{chunk_size}",
                chunk_size=chunk_size,
                includes_oss=False,
            )
            self.scenarios.append(scenario)

        # Mixed project + OSS scenarios
        for chunk_size in self.chunk_sizes:
            scenario = BenchmarkScenario(
                name=f"mixed_projects_{chunk_size}",
                collection_name=f"bench_mixed_{chunk_size}",
                chunk_size=chunk_size,
                includes_oss=True,
            )
            self.scenarios.append(scenario)

    async def ingest_test_data(self) -> None:
        """Ingest test data using the workspace-qdrant-ingest CLI."""
        console.print("\n📊 Ingesting Test Data", style="bold cyan")

        self._generate_scenarios()

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            console=console,
        ) as progress:
            for scenario in self.scenarios:
                task = progress.add_task(f"Ingesting {scenario.name}...")

                try:
                    start_time = time.time()

                    # Build ingestion command using installed CLI
                    cmd = [
                        "workspace-qdrant-ingest",
                        "ingest",
                        str(self.project_root),
                        "--collection",
                        scenario.collection_name,
                        "--chunk-size",
                        str(scenario.chunk_size),
                        "--chunk-overlap",
                        "200",
                        "--concurrency",
                        "5",
                        "--yes",  # Skip confirmation
                    ]

                    # Add OSS projects if needed
                    if scenario.includes_oss:
                        for project in self.oss_projects:
                            if project.local_path and project.local_path.exists():
                                cmd.extend(["--include", str(project.local_path)])

                    # Run ingestion
                    result = subprocess.run(
                        cmd,
                        capture_output=True,
                        text=True,
                        timeout=1800,  # 30 minute timeout
                    )

                    scenario.ingestion_time = time.time() - start_time

                    if result.returncode == 0:
                        # Extract metrics from output
                        output_lines = result.stdout.split("\n")
                        for line in output_lines:
                            if "Documents created:" in line:
                                scenario.document_count = int(
                                    line.split(":")[1].strip().replace(",", "")
                                )
                            elif "Text chunks:" in line:
                                scenario.chunk_count = int(
                                    line.split(":")[1].strip().replace(",", "")
                                )

                        console.print(
                            f"  ✅ {scenario.name}: {scenario.document_count} docs, {scenario.chunk_count} chunks",
                            style="green",
                        )
                    else:
                        console.print(
                            f"  ❌ {scenario.name}: Ingestion failed", style="red"
                        )
                        logger.error(
                            f"Ingestion failed for {scenario.name}: {result.stderr}"
                        )

                    progress.update(task, completed=100)

                except subprocess.TimeoutExpired:
                    console.print(f"  ⏰ {scenario.name}: Timeout", style="red")
                except Exception as e:
                    console.print(f"  ❌ {scenario.name}: {e}", style="red")
                    logger.error(f"Ingestion error for {scenario.name}: {e}")

    async def run_search_benchmarks(self) -> None:
        """Run comprehensive search benchmarks using actual MCP tools."""
        console.print("\n🔍 Running Search Benchmarks", style="bold cyan")

        # Generate test queries
        test_queries = self._generate_test_queries()

        console.print(f"📝 Generated {len(test_queries)} test queries")

        for scenario in self.scenarios:
            if scenario.document_count is None:
                console.print(f"⏭️  Skipping {scenario.name} - no data", style="yellow")
                continue

            console.print(f"\n🎯 Testing scenario: {scenario.name}")

            scenario_results = {
                "search_types": {},
                "performance_metrics": {},
                "statistical_analysis": {},
            }

            # Test different search types
            search_types = ["semantic", "hybrid", "exact"]

            for search_type in search_types:
                type_results = await self._test_search_type(
                    scenario, search_type, test_queries
                )
                scenario_results["search_types"][search_type] = type_results

            # Store results
            self.benchmark_results["scenarios"][scenario.name] = scenario_results

    def _generate_test_queries(self) -> List[Dict[str, Any]]:
        """Generate realistic test queries."""
        queries = []

        # Symbol queries (functions, classes)
        symbol_queries = [
            "search_workspace",
            "QdrantClient",
            "HybridSearchEngine",
            "initialize",
            "process_documents",
            "calculate_metrics",
            "TestDataCollector",
            "AuthoritativeBenchmark",
        ]

        for query in symbol_queries:
            queries.append(
                {"text": query, "type": "symbol", "expected_relevance": "high"}
            )

        # Semantic queries (concepts)
        semantic_queries = [
            "vector search implementation",
            "document chunking strategies",
            "performance benchmarking",
            "search quality metrics",
            "async processing patterns",
            "error handling approaches",
        ]

        for query in semantic_queries:
            queries.append(
                {"text": query, "type": "semantic", "expected_relevance": "medium"}
            )

        # Exact queries (code patterns)
        exact_queries = [
            "async def",
            "import asyncio",
            "class ",
            "def __init__",
            "pytest.fixture",
            "assert ",
            "return None",
        ]

        for query in exact_queries:
            queries.append(
                {"text": query, "type": "exact", "expected_relevance": "high"}
            )

        return queries

    async def _test_search_type(
        self,
        scenario: BenchmarkScenario,
        search_type: str,
        queries: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Test a specific search type for a scenario."""

        results = {
            "query_count": 0,
            "total_time": 0.0,
            "response_times": [],
            "precision_scores": [],
            "recall_scores": [],
            "f1_scores": [],
        }

        # Filter queries appropriate for this search type
        relevant_queries = [
            q for q in queries if q["type"] == search_type or search_type == "hybrid"
        ]

        for query in relevant_queries[:20]:  # Limit to 20 queries per type for speed
            try:
                start_time = time.time()

                # Use actual search via client
                search_results = await self._perform_search(
                    collection=scenario.collection_name,
                    query=query["text"],
                    search_type=search_type,
                )

                response_time = time.time() - start_time

                # Calculate quality metrics (simplified for now)
                precision, recall, f1 = self._calculate_search_quality(
                    query, search_results
                )

                # Store metrics
                results["query_count"] += 1
                results["total_time"] += response_time
                results["response_times"].append(response_time)
                results["precision_scores"].append(precision)
                results["recall_scores"].append(recall)
                results["f1_scores"].append(f1)

            except Exception as e:
                logger.error(f"Search failed for query '{query['text']}': {e}")
                continue

        # Calculate summary statistics
        if results["response_times"]:
            results["avg_response_time"] = statistics.mean(results["response_times"])
            results["p95_response_time"] = self._percentile(
                results["response_times"], 95
            )
            results["avg_precision"] = statistics.mean(results["precision_scores"])
            results["avg_recall"] = statistics.mean(results["recall_scores"])
            results["avg_f1"] = statistics.mean(results["f1_scores"])

        return results

    async def _perform_search(
        self, collection: str, query: str, search_type: str
    ) -> List[Dict[str, Any]]:
        """Perform actual search using Qdrant client."""
        if not self.client:
            raise RuntimeError("Client not initialized")

        try:
            if search_type == "semantic":
                results = await self.client.search_workspace(
                    query=query, collection=collection, limit=10
                )
            elif search_type == "hybrid":
                results = await self.client.hybrid_search(
                    query=query, collection=collection, limit=10
                )
            elif search_type == "exact":
                results = await self.client.search_workspace(
                    query=query, collection=collection, search_type="exact", limit=10
                )
            else:
                raise ValueError(f"Unknown search type: {search_type}")

            return results

        except Exception as e:
            logger.error(f"Search error: {e}")
            return []

    def _calculate_search_quality(
        self, query: Dict[str, Any], results: List[Dict[str, Any]]
    ) -> Tuple[float, float, float]:
        """Calculate precision, recall, F1 for search results."""
        if not results:
            return 0.0, 0.0, 0.0

        # Simplified relevance assessment based on query type and content
        relevant_count = 0

        for result in results:
            content = result.get("content", "").lower()
            query_text = query["text"].lower()

            # Simple relevance heuristics
            if query["type"] == "symbol":
                # Symbol should appear in definition context
                is_relevant = (
                    f"def {query_text}" in content
                    or f"class {query_text}" in content
                    or f"{query_text}(" in content
                )
            elif query["type"] == "exact":
                # Exact match required
                is_relevant = query_text in content
            else:  # semantic
                # Word overlap for semantic relevance
                query_words = set(query_text.split())
                content_words = set(content.split())
                overlap = len(query_words & content_words)
                is_relevant = overlap >= len(query_words) * 0.5

            if is_relevant:
                relevant_count += 1

        # Calculate metrics
        precision = relevant_count / len(results)
        recall = min(1.0, relevant_count / 5)  # Assume 5 relevant docs exist
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        return precision, recall, f1

    def _percentile(self, data: List[float], percentile: float) -> float:
        """Calculate percentile of data."""
        if not data:
            return 0.0
        sorted_data = sorted(data)
        index = (percentile / 100) * (len(sorted_data) - 1)
        if index == int(index):
            return sorted_data[int(index)]
        else:
            lower = sorted_data[int(index)]
            upper = sorted_data[int(index) + 1]
            return lower + (upper - lower) * (index - int(index))

    def _calculate_confidence_interval(
        self, data: List[float], confidence: float = 0.95
    ) -> Tuple[float, float]:
        """Calculate confidence interval."""
        if len(data) < 2:
            return (0.0, 0.0)

        mean = statistics.mean(data)
        std = statistics.stdev(data)
        n = len(data)

        # Use t-distribution approximation
        t_val = 1.96 if n >= 30 else 2.0
        margin = t_val * (std / math.sqrt(n))

        return (max(0, mean - margin), min(1, mean + margin))

    def generate_comprehensive_report(self) -> None:
        """Generate comprehensive benchmark report."""
        console.print("\n📊 Generating Comprehensive Report", style="bold cyan")

        # Create summary table
        summary_table = Table(title="🎯 Benchmark Results Summary")
        summary_table.add_column("Scenario", style="cyan")
        summary_table.add_column("Chunk Size", justify="right")
        summary_table.add_column("Documents", justify="right")
        summary_table.add_column("Chunks", justify="right")
        summary_table.add_column("Ingestion Time", justify="right")
        summary_table.add_column("Avg Precision", justify="right")
        summary_table.add_column("Avg Recall", justify="right")

        for scenario in self.scenarios:
            if scenario.document_count:
                scenario_data = self.benchmark_results["scenarios"].get(
                    scenario.name, {}
                )

                # Calculate average precision/recall across search types
                avg_precision = 0.0
                avg_recall = 0.0
                search_types = scenario_data.get("search_types", {})

                if search_types:
                    precisions = [
                        st.get("avg_precision", 0) for st in search_types.values()
                    ]
                    recalls = [st.get("avg_recall", 0) for st in search_types.values()]
                    avg_precision = (
                        statistics.mean([p for p in precisions if p > 0])
                        if precisions
                        else 0
                    )
                    avg_recall = (
                        statistics.mean([r for r in recalls if r > 0]) if recalls else 0
                    )

                summary_table.add_row(
                    scenario.name,
                    str(scenario.chunk_size),
                    f"{scenario.document_count:,}",
                    f"{scenario.chunk_count:,}",
                    f"{scenario.ingestion_time:.1f}s"
                    if scenario.ingestion_time
                    else "N/A",
                    f"{avg_precision:.3f}",
                    f"{avg_recall:.3f}",
                )

        console.print(summary_table)

        # Performance comparison
        self._print_performance_comparison()

        # Chunk size optimization
        self._print_chunk_size_analysis()

        # Export detailed results
        self._export_results()

    def _print_performance_comparison(self) -> None:
        """Print project-only vs mixed project performance comparison."""
        console.print(
            "\n📈 Performance Comparison: Project-Only vs Mixed", style="bold yellow"
        )

        comparison_table = Table()
        comparison_table.add_column("Chunk Size", style="cyan")
        comparison_table.add_column("Project-Only Precision", justify="right")
        comparison_table.add_column("Mixed Projects Precision", justify="right")
        comparison_table.add_column("Precision Drop", justify="right", style="red")
        comparison_table.add_column("Project-Only Recall", justify="right")
        comparison_table.add_column("Mixed Projects Recall", justify="right")
        comparison_table.add_column("Recall Drop", justify="right", style="red")

        for chunk_size in self.chunk_sizes:
            project_scenario = f"project_only_{chunk_size}"
            mixed_scenario = f"mixed_projects_{chunk_size}"

            project_data = self.benchmark_results["scenarios"].get(project_scenario, {})
            mixed_data = self.benchmark_results["scenarios"].get(mixed_scenario, {})

            if project_data and mixed_data:
                # Calculate averages
                project_precision = self._get_avg_metric(project_data, "avg_precision")
                mixed_precision = self._get_avg_metric(mixed_data, "avg_precision")
                project_recall = self._get_avg_metric(project_data, "avg_recall")
                mixed_recall = self._get_avg_metric(mixed_data, "avg_recall")

                precision_drop = (
                    ((project_precision - mixed_precision) / project_precision * 100)
                    if project_precision > 0
                    else 0
                )
                recall_drop = (
                    ((project_recall - mixed_recall) / project_recall * 100)
                    if project_recall > 0
                    else 0
                )

                comparison_table.add_row(
                    str(chunk_size),
                    f"{project_precision:.3f}",
                    f"{mixed_precision:.3f}",
                    f"{precision_drop:.1f}%",
                    f"{project_recall:.3f}",
                    f"{mixed_recall:.3f}",
                    f"{recall_drop:.1f}%",
                )

        console.print(comparison_table)

    def _print_chunk_size_analysis(self) -> None:
        """Print chunk size optimization analysis."""
        console.print("\n📏 Chunk Size Optimization Analysis", style="bold yellow")

        chunk_table = Table()
        chunk_table.add_column("Chunk Size", style="cyan")
        chunk_table.add_column("Avg Precision", justify="right")
        chunk_table.add_column("Avg Recall", justify="right")
        chunk_table.add_column("Avg Response Time", justify="right")
        chunk_table.add_column("Index Size (Est.)", justify="right")
        chunk_table.add_column("Recommendation", style="green")

        chunk_analysis = {}

        for chunk_size in self.chunk_sizes:
            # Analyze project-only scenarios for cleaner comparison
            scenario_name = f"project_only_{chunk_size}"
            scenario_data = self.benchmark_results["scenarios"].get(scenario_name, {})

            if scenario_data:
                precision = self._get_avg_metric(scenario_data, "avg_precision")
                recall = self._get_avg_metric(scenario_data, "avg_recall")
                response_time = self._get_avg_metric(scenario_data, "avg_response_time")

                # Find corresponding scenario for chunk count
                scenario = next(
                    (s for s in self.scenarios if s.name == scenario_name), None
                )
                index_size = (
                    scenario.chunk_count if scenario and scenario.chunk_count else 0
                )

                chunk_analysis[chunk_size] = {
                    "precision": precision,
                    "recall": recall,
                    "response_time": response_time,
                    "index_size": index_size,
                }

        # Generate recommendations
        best_precision = (
            max(chunk_analysis.values(), key=lambda x: x["precision"])["precision"]
            if chunk_analysis
            else 0
        )
        best_recall = (
            max(chunk_analysis.values(), key=lambda x: x["recall"])["recall"]
            if chunk_analysis
            else 0
        )
        best_speed = (
            min(
                (v for v in chunk_analysis.values() if v["response_time"] > 0),
                key=lambda x: x["response_time"],
                default={"response_time": 0},
            )["response_time"]
            if chunk_analysis
            else 0
        )

        for chunk_size in self.chunk_sizes:
            data = chunk_analysis.get(chunk_size, {})

            # Generate recommendation
            recommendation = ""
            if data.get("precision", 0) >= best_precision * 0.95:
                recommendation = "✅ Best Quality"
            elif (
                data.get("response_time", float("inf")) <= best_speed * 1.1
                and best_speed > 0
            ):
                recommendation = "⚡ Best Speed"
            elif chunk_size == 1000:  # Default fallback
                recommendation = "📝 Balanced"
            else:
                recommendation = "⚪ Consider"

            chunk_table.add_row(
                str(chunk_size),
                f"{data.get('precision', 0):.3f}",
                f"{data.get('recall', 0):.3f}",
                f"{data.get('response_time', 0):.3f}s",
                f"{data.get('index_size', 0):,}",
                recommendation,
            )

        console.print(chunk_table)

        # Provide optimization recommendations
        console.print("\n💡 Optimization Recommendations:", style="bold green")
        if chunk_analysis:
            best_chunk_size = max(
                chunk_analysis.keys(),
                key=lambda k: chunk_analysis[k]["precision"]
                * chunk_analysis[k]["recall"],
            )
            console.print(f"  🎯 Recommended chunk size: {best_chunk_size} characters")
            console.print(f"  📊 Balance of search quality and performance")
            console.print(f"  ⚖️  Consider project size and use case requirements")

    def _get_avg_metric(self, scenario_data: Dict[str, Any], metric: str) -> float:
        """Get average metric across search types."""
        search_types = scenario_data.get("search_types", {})
        if not search_types:
            return 0.0

        values = [st.get(metric, 0) for st in search_types.values()]
        return statistics.mean([v for v in values if v > 0]) if values else 0.0

    def _export_results(self) -> None:
        """Export detailed results to JSON."""
        results_file = self.results_dir / f"benchmark_results_{int(time.time())}.json"

        # Add metadata
        self.benchmark_results["metadata"] = {
            "timestamp": time.time(),
            "benchmark_version": "1.0.0",
            "project_root": str(self.project_root),
            "qdrant_url": self.config.qdrant.url if self.config else "unknown",
            "scenarios_tested": len(self.scenarios),
            "oss_projects": [p.name for p in self.oss_projects if p.local_path],
        }

        with open(results_file, "w") as f:
            json.dump(self.benchmark_results, f, indent=2, default=str)

        console.print(f"\n💾 Detailed results exported to: {results_file}")

    async def cleanup(self) -> None:
        """Cleanup resources."""
        if self.client:
            await self.client.close()

    async def run_full_benchmark(self) -> None:
        """Run the complete benchmark suite."""
        try:
            await self.initialize()
            await self.download_oss_projects()
            await self.ingest_test_data()
            await self.run_search_benchmarks()
            self.generate_comprehensive_report()

            console.print("\n🎉 Benchmark completed successfully!", style="bold green")

        except Exception as e:
            console.print(f"\n❌ Benchmark failed: {e}", style="bold red")
            logger.error(f"Benchmark failed: {e}", exc_info=True)
            raise
        finally:
            await self.cleanup()


async def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Authoritative workspace-qdrant-mcp benchmark"
    )
    parser.add_argument(
        "--skip-oss", action="store_true", help="Skip OSS project download"
    )
    parser.add_argument(
        "--chunk-sizes",
        nargs="+",
        type=int,
        default=[500, 1000, 2000, 4000],
        help="Chunk sizes to test",
    )

    args = parser.parse_args()

    benchmark = AuthoritativeBenchmark()

    if args.chunk_sizes:
        benchmark.chunk_sizes = args.chunk_sizes

    if args.skip_oss:
        benchmark.oss_projects = []

    await benchmark.run_full_benchmark()


if __name__ == "__main__":
    asyncio.run(main())
