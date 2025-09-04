#!/usr/bin/env python3
"""
Comprehensive test runner for workspace-qdrant-mcp functional tests.

Orchestrates the complete test suite including data ingestion, search functionality,
MCP integration, performance benchmarking, and recall/precision measurements.
"""

import argparse
import asyncio
import json
import logging
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class ComprehensiveTestRunner:
    """Orchestrates comprehensive functional testing of workspace-qdrant-mcp."""

    def __init__(self, output_dir: Path = None):
        self.output_dir = output_dir or Path("test_results")
        self.output_dir.mkdir(exist_ok=True)

        self.test_results = {}
        self.start_time = time.time()

        # Test categories and their pytest markers
        self.test_categories = {
            "data_ingestion": {
                "module": "tests/functional/test_data_ingestion.py",
                "markers": ["integration", "slow"],
                "description": "Real codebase data ingestion with chunking and embedding",
            },
            "search_functionality": {
                "module": "tests/functional/test_search_functionality.py",
                "markers": ["integration", "performance"],
                "description": "Search quality across semantic, hybrid, and exact modes",
            },
            "mcp_integration": {
                "module": "tests/functional/test_mcp_integration.py",
                "markers": ["e2e", "integration"],
                "description": "End-to-end MCP server tool testing",
            },
            "performance": {
                "module": "tests/functional/test_performance.py",
                "markers": ["performance", "slow"],
                "description": "Performance benchmarking and stress testing",
            },
            "recall_precision": {
                "module": "tests/functional/test_recall_precision.py",
                "markers": ["integration", "slow"],
                "description": "Comprehensive quality measurement with ground truth",
            },
        }

    async def run_all_tests(self, categories: list[str] = None, verbose: bool = True):
        """Run all or specified test categories."""
        categories = categories or list(self.test_categories.keys())

        print("🚀 Starting Comprehensive Functional Testing")
        print("=" * 60)
        print(f"Output directory: {self.output_dir.absolute()}")
        print(f"Test categories: {', '.join(categories)}")
        print("=" * 60)

        overall_success = True

        for category in categories:
            if category not in self.test_categories:
                print(f"❌ Unknown test category: {category}")
                overall_success = False
                continue

            success = await self.run_test_category(category, verbose)
            overall_success = overall_success and success

        # Generate final report
        await self.generate_final_report()

        return overall_success

    async def run_test_category(self, category: str, verbose: bool = True) -> bool:
        """Run a specific test category."""
        config = self.test_categories[category]

        print(f"\n🔄 Running {category} tests...")
        print(f"📝 {config['description']}")
        print("-" * 40)

        start_time = time.time()

        # Construct pytest command
        cmd = [
            sys.executable,
            "-m",
            "pytest",
            config["module"],
            "-v" if verbose else "-q",
            "--tb=short",
            f"--junitxml={self.output_dir / f'{category}_junit.xml'}",
            f"--html={self.output_dir / f'{category}_report.html'}",
            "--self-contained-html",
            "--cov=src/workspace_qdrant_mcp",
            f"--cov-report=html:{self.output_dir / f'{category}_coverage'}",
            "--cov-report=json",
            "--benchmark-json=" + str(self.output_dir / f"{category}_benchmarks.json"),
            "--timeout=300",  # 5 minute timeout per test
        ]

        # Add marker filters
        for marker in config["markers"]:
            cmd.extend(["-m", marker])

        try:
            # Run pytest
            result = subprocess.run(
                cmd,
                cwd=Path.cwd(),
                capture_output=True,
                text=True,
                timeout=1800,  # 30 minute timeout for entire category
            )

            duration = time.time() - start_time
            success = result.returncode == 0

            # Store results
            self.test_results[category] = {
                "success": success,
                "duration": duration,
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "command": " ".join(cmd),
            }

            # Print summary
            status = "✅ PASSED" if success else "❌ FAILED"
            print(f"{status} {category} tests ({duration:.1f}s)")

            if not success or verbose:
                print(f"Return code: {result.returncode}")
                if result.stdout:
                    print("STDOUT:")
                    print(result.stdout[-1000:])  # Last 1000 chars
                if result.stderr:
                    print("STDERR:")
                    print(result.stderr[-1000:])  # Last 1000 chars

            return success

        except subprocess.TimeoutExpired:
            print(f"⏰ {category} tests timed out")
            self.test_results[category] = {
                "success": False,
                "duration": time.time() - start_time,
                "error": "Timeout",
                "command": " ".join(cmd),
            }
            return False

        except Exception as e:
            print(f"💥 Error running {category} tests: {e}")
            self.test_results[category] = {
                "success": False,
                "duration": time.time() - start_time,
                "error": str(e),
                "command": " ".join(cmd),
            }
            return False

    async def generate_final_report(self):
        """Generate comprehensive final report."""
        print("\n📋 Generating Final Test Report...")
        print("=" * 60)

        total_duration = time.time() - self.start_time
        successful_categories = sum(
            1 for r in self.test_results.values() if r["success"]
        )
        total_categories = len(self.test_results)

        # Create summary report
        summary_report = {
            "test_run_info": {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "total_duration_seconds": total_duration,
                "categories_run": total_categories,
                "categories_passed": successful_categories,
                "success_rate": successful_categories / total_categories
                if total_categories > 0
                else 0,
            },
            "category_results": {},
            "overall_success": successful_categories == total_categories,
        }

        # Process each category
        for category, result in self.test_results.items():
            config = self.test_categories[category]

            category_summary = {
                "description": config["description"],
                "success": result["success"],
                "duration": result["duration"],
                "markers": config["markers"],
            }

            # Add error info if failed
            if not result["success"]:
                category_summary["error"] = result.get("error", "Test failures")
                category_summary["returncode"] = result.get("returncode")

            # Try to extract additional metrics from test outputs
            if "stdout" in result:
                category_summary["metrics"] = self._extract_metrics_from_output(
                    result["stdout"], category
                )

            summary_report["category_results"][category] = category_summary

        # Export summary report
        summary_file = self.output_dir / "comprehensive_test_summary.json"
        with open(summary_file, "w") as f:
            json.dump(summary_report, f, indent=2)

        # Create human-readable report
        readme_file = self.output_dir / "TEST_RESULTS.md"
        with open(readme_file, "w") as f:
            f.write(self._generate_markdown_report(summary_report))

        # Print console summary
        print("📊 Test Run Summary:")
        print(f"  Duration: {total_duration:.1f} seconds")
        print(f"  Categories: {successful_categories}/{total_categories} passed")
        print(
            f"  Success Rate: {(successful_categories / total_categories * 100):.1f}%"
        )

        print("\n📁 Generated Reports:")
        print(f"  Summary: {summary_file}")
        print(f"  Readable: {readme_file}")
        print(f"  All outputs: {self.output_dir}/")

        # Category details
        print("\n📈 Category Results:")
        for category, result in self.test_results.items():
            status = "✅" if result["success"] else "❌"
            duration = result["duration"]
            print(f"  {status} {category}: {duration:.1f}s")

        if successful_categories == total_categories:
            print("\n🎉 ALL TESTS PASSED! Comprehensive testing successful.")
        else:
            print(
                f"\n⚠️  {total_categories - successful_categories} test categories failed."
            )
            print(f"Check individual reports in {self.output_dir}/ for details.")

        return summary_report

    def _extract_metrics_from_output(
        self, output: str, category: str
    ) -> dict[str, Any]:
        """Extract key metrics from test output."""
        metrics = {}

        try:
            # Look for common metric patterns
            lines = output.split("\n")

            for line in lines:
                line = line.strip()

                # Search quality metrics
                if "Average precision:" in line:
                    metrics["avg_precision"] = float(line.split(":")[1].strip())
                elif "Average recall:" in line:
                    metrics["avg_recall"] = float(line.split(":")[1].strip())
                elif "Average F1:" in line:
                    metrics["avg_f1"] = float(line.split(":")[1].strip())

                # Performance metrics
                elif "ops/sec" in line and "avg" in line:
                    # Extract throughput
                    parts = line.split()
                    for i, part in enumerate(parts):
                        if "ops/sec" in part and i > 0:
                            try:
                                metrics["throughput_ops_sec"] = float(parts[i - 1])
                            except (ValueError, IndexError):
                                pass

                # Coverage metrics (from pytest output)
                elif "TOTAL" in line and "%" in line:
                    parts = line.split()
                    for part in parts:
                        if part.endswith("%"):
                            try:
                                metrics["code_coverage"] = float(part.rstrip("%"))
                            except ValueError:
                                pass

                # Test counts
                elif "passed" in line and "failed" in line:
                    # Extract test counts
                    if "passed," in line:
                        parts = line.split()
                        for i, part in enumerate(parts):
                            if part == "passed,":
                                try:
                                    metrics["tests_passed"] = int(parts[i - 1])
                                except (ValueError, IndexError):
                                    pass

        except Exception as e:
            logger.warning(f"Failed to extract metrics from {category}: {e}")

        return metrics

    def _generate_markdown_report(self, summary: dict[str, Any]) -> str:
        """Generate human-readable markdown report."""
        report = []

        # Header
        report.append("# Workspace-Qdrant-MCP Comprehensive Test Results")
        report.append("")
        report.append(f"**Test Run:** {summary['test_run_info']['timestamp']}")
        report.append(
            f"**Duration:** {summary['test_run_info']['total_duration_seconds']:.1f} seconds"
        )
        report.append(
            f"**Success Rate:** {summary['test_run_info']['success_rate'] * 100:.1f}%"
        )
        report.append("")

        # Overall status
        if summary["overall_success"]:
            report.append("## ✅ Overall Status: PASSED")
        else:
            report.append("## ❌ Overall Status: FAILED")
        report.append("")

        # Category results
        report.append("## Test Categories")
        report.append("")

        for category, result in summary["category_results"].items():
            status = "✅ PASSED" if result["success"] else "❌ FAILED"
            report.append(f"### {category.replace('_', ' ').title()}")
            report.append(f"**Status:** {status}")
            report.append(f"**Duration:** {result['duration']:.1f}s")
            report.append(f"**Description:** {result['description']}")

            if "metrics" in result and result["metrics"]:
                report.append("**Key Metrics:**")
                for metric, value in result["metrics"].items():
                    if isinstance(value, float):
                        report.append(
                            f"- {metric.replace('_', ' ').title()}: {value:.3f}"
                        )
                    else:
                        report.append(f"- {metric.replace('_', ' ').title()}: {value}")

            if not result["success"] and "error" in result:
                report.append(f"**Error:** {result['error']}")

            report.append("")

        # Files generated
        report.append("## Generated Files")
        report.append("")
        report.append("- `comprehensive_test_summary.json` - Machine-readable summary")
        report.append("- `*_junit.xml` - JUnit XML reports for CI/CD")
        report.append("- `*_report.html` - HTML test reports")
        report.append("- `*_coverage/` - Code coverage reports")
        report.append("- `*_benchmarks.json` - Performance benchmark data")
        report.append("")

        # Next steps
        if not summary["overall_success"]:
            report.append("## Next Steps")
            report.append("")
            report.append("1. Review failed test category reports")
            report.append("2. Check error logs for specific failure causes")
            report.append("3. Run individual test categories for detailed debugging")
            report.append("4. Address failing tests and re-run comprehensive suite")
            report.append("")

        return "\n".join(report)


def main():
    """Main entry point for comprehensive test runner."""
    parser = argparse.ArgumentParser(description="Run comprehensive functional tests")
    parser.add_argument(
        "--categories",
        nargs="+",
        choices=[
            "data_ingestion",
            "search_functionality",
            "mcp_integration",
            "performance",
            "recall_precision",
        ],
        help="Specific test categories to run (default: all)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default="test_results",
        help="Output directory for test results",
    )
    parser.add_argument("--quiet", action="store_true", help="Reduce output verbosity")

    args = parser.parse_args()

    # Create test runner
    runner = ComprehensiveTestRunner(output_dir=args.output_dir)

    # Run tests
    try:
        success = asyncio.run(
            runner.run_all_tests(categories=args.categories, verbose=not args.quiet)
        )

        sys.exit(0 if success else 1)

    except KeyboardInterrupt:
        print("\n⚠️  Test run interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n💥 Test run failed with error: {e}")
        logger.exception("Test run failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
