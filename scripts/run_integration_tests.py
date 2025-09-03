#!/usr/bin/env python3
"""
Integration test runner script.

Provides comprehensive test execution with coverage measurement,
performance benchmarking, and CI/CD integration support.
"""

import argparse
import asyncio
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Any, List, Optional


class IntegrationTestRunner:
    """Comprehensive integration test runner."""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.test_results = {}
        self.coverage_threshold = 80
        
    def run_test_suite(
        self,
        test_categories: List[str] = None,
        coverage: bool = True,
        performance: bool = True,
        parallel: bool = False,
        verbose: bool = False
    ) -> Dict[str, Any]:
        """
        Run comprehensive integration test suite.
        
        Args:
            test_categories: Specific test categories to run
            coverage: Whether to measure code coverage
            performance: Whether to run performance tests
            parallel: Whether to run tests in parallel
            verbose: Verbose output
            
        Returns:
            Test execution results
        """
        
        start_time = time.time()
        results = {
            "start_time": start_time,
            "test_categories": test_categories or ["all"],
            "coverage_enabled": coverage,
            "performance_enabled": performance,
            "success": False,
            "test_results": {},
            "coverage_results": {},
            "performance_results": {},
            "errors": []
        }
        
        try:
            print("🚀 Starting integration test suite...")
            
            # Step 1: Environment validation
            print("\n📋 Validating test environment...")
            env_check = self._validate_environment()
            if not env_check["valid"]:
                results["errors"].extend(env_check["errors"])
                return results
            
            # Step 2: Setup test environment
            print("\n🔧 Setting up test environment...")
            setup_result = self._setup_test_environment()
            if not setup_result["success"]:
                results["errors"].extend(setup_result["errors"])
                return results
            
            # Step 3: Run test categories
            test_commands = self._build_test_commands(
                test_categories, coverage, performance, parallel, verbose
            )
            
            for category, command in test_commands.items():
                print(f"\n🧪 Running {category} tests...")
                category_result = self._run_test_command(command, category)
                results["test_results"][category] = category_result
                
                if not category_result["success"]:
                    results["errors"].append(f"{category} tests failed")
            
            # Step 4: Process coverage results
            if coverage:
                print("\n📊 Processing coverage results...")
                coverage_result = self._process_coverage_results()
                results["coverage_results"] = coverage_result
                
                if coverage_result["coverage_percentage"] < self.coverage_threshold:
                    results["errors"].append(
                        f"Coverage {coverage_result['coverage_percentage']:.1f}% "
                        f"below threshold {self.coverage_threshold}%"
                    )
            
            # Step 5: Process performance results
            if performance:
                print("\n⚡ Processing performance results...")
                perf_result = self._process_performance_results()
                results["performance_results"] = perf_result
                
                if perf_result.get("regressions"):
                    results["errors"].append(
                        f"Performance regressions detected: {perf_result['regressions']}"
                    )
            
            # Step 6: Generate reports
            print("\n📄 Generating test reports...")
            self._generate_reports(results)
            
            # Determine overall success
            results["success"] = len(results["errors"]) == 0
            
        except Exception as e:
            results["errors"].append(f"Test runner error: {str(e)}")
            results["success"] = False
        
        finally:
            results["end_time"] = time.time()
            results["duration_seconds"] = results["end_time"] - start_time
            
            # Cleanup
            print("\n🧹 Cleaning up test environment...")
            self._cleanup_test_environment()
        
        return results
    
    def _validate_environment(self) -> Dict[str, Any]:
        """Validate test environment prerequisites."""
        result = {"valid": True, "errors": [], "checks": {}}
        
        # Check Python version
        if sys.version_info < (3, 10):
            result["errors"].append("Python 3.10+ required")
            result["valid"] = False
        result["checks"]["python_version"] = f"{sys.version_info.major}.{sys.version_info.minor}"
        
        # Check dependencies
        required_packages = [
            "pytest", "pytest-cov", "pytest-asyncio", "pytest-benchmark",
            "testcontainers", "docker"
        ]
        
        missing_packages = []
        for package in required_packages:
            try:
                __import__(package.replace("-", "_"))
                result["checks"][f"package_{package}"] = "available"
            except ImportError:
                missing_packages.append(package)
                result["checks"][f"package_{package}"] = "missing"
        
        if missing_packages:
            result["errors"].append(f"Missing packages: {', '.join(missing_packages)}")
            result["valid"] = False
        
        # Check Docker availability
        try:
            subprocess.run(
                ["docker", "version"], 
                check=True, 
                capture_output=True
            )
            result["checks"]["docker"] = "available"
        except (subprocess.CalledProcessError, FileNotFoundError):
            result["errors"].append("Docker not available")
            result["checks"]["docker"] = "unavailable"
            # Don't fail validation - some tests can run without Docker
        
        # Check disk space
        import shutil
        disk_usage = shutil.disk_usage(self.project_root)
        free_gb = disk_usage.free / (1024**3)
        
        if free_gb < 2:  # Need at least 2GB for containers and test data
            result["errors"].append(f"Insufficient disk space: {free_gb:.1f}GB available")
            result["valid"] = False
        result["checks"]["disk_space_gb"] = free_gb
        
        return result
    
    def _setup_test_environment(self) -> Dict[str, Any]:
        """Set up test environment."""
        result = {"success": True, "errors": []}
        
        try:
            # Set environment variables
            os.environ["PYTHONPATH"] = str(self.project_root)
            os.environ["INTEGRATION_TESTING"] = "1"
            os.environ["TEST_ENV"] = "integration"
            
            # Create necessary directories
            test_dirs = [
                "htmlcov",
                ".pytest_cache",
                "test_results",
                "performance_results"
            ]
            
            for dir_name in test_dirs:
                dir_path = self.project_root / dir_name
                dir_path.mkdir(exist_ok=True)
            
        except Exception as e:
            result["success"] = False
            result["errors"].append(f"Environment setup failed: {str(e)}")
        
        return result
    
    def _build_test_commands(
        self,
        categories: List[str],
        coverage: bool,
        performance: bool,
        parallel: bool,
        verbose: bool
    ) -> Dict[str, List[str]]:
        """Build pytest commands for different test categories."""
        
        base_cmd = ["python", "-m", "pytest"]
        
        # Common options
        common_opts = []
        if verbose:
            common_opts.extend(["-v", "-s"])
        if parallel:
            common_opts.extend(["-n", "auto"])
        
        # Coverage options
        coverage_opts = []
        if coverage:
            coverage_opts.extend([
                "--cov=src/workspace_qdrant_mcp",
                "--cov-report=html",
                "--cov-report=xml",
                "--cov-report=term-missing",
                f"--cov-fail-under={self.coverage_threshold}",
                "--cov-branch"
            ])
        
        commands = {}
        
        if not categories or "all" in categories:
            # Full integration test suite
            commands["integration"] = (
                base_cmd + common_opts + coverage_opts +
                [
                    "tests/integration/",
                    "-m", "integration and not slow",
                    "--tb=short",
                    "--durations=10"
                ]
            )
            
            # Performance tests (if enabled)
            if performance:
                commands["performance"] = (
                    base_cmd + common_opts +
                    [
                        "tests/integration/",
                        "-m", "performance",
                        "--benchmark-only",
                        "--benchmark-json=performance_results/benchmarks.json",
                        "--tb=short"
                    ]
                )
            
            # Slow tests
            commands["slow"] = (
                base_cmd + common_opts +
                [
                    "tests/integration/",
                    "-m", "slow",
                    "--tb=short",
                    "--timeout=300"
                ]
            )
        else:
            # Specific categories
            for category in categories:
                if category == "smoke":
                    commands["smoke"] = (
                        base_cmd + common_opts + coverage_opts +
                        [
                            "tests/integration/",
                            "-m", "smoke",
                            "--tb=short"
                        ]
                    )
                elif category == "performance":
                    commands["performance"] = (
                        base_cmd + common_opts +
                        [
                            "tests/integration/",
                            "-m", "performance",
                            "--benchmark-only",
                            "--benchmark-json=performance_results/benchmarks.json"
                        ]
                    )
                elif category == "regression":
                    commands["regression"] = (
                        base_cmd + common_opts + coverage_opts +
                        [
                            "tests/integration/",
                            "-m", "regression",
                            "--tb=short"
                        ]
                    )
        
        return commands
    
    def _run_test_command(self, command: List[str], category: str) -> Dict[str, Any]:
        """Run a specific test command."""
        result = {
            "category": category,
            "command": " ".join(command),
            "success": False,
            "exit_code": None,
            "output": "",
            "error_output": "",
            "duration": 0
        }
        
        start_time = time.time()
        
        try:
            process = subprocess.run(
                command,
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=900  # 15 minute timeout
            )
            
            result["exit_code"] = process.returncode
            result["output"] = process.stdout
            result["error_output"] = process.stderr
            result["success"] = process.returncode == 0
            
        except subprocess.TimeoutExpired:
            result["error_output"] = "Test execution timed out"
        except Exception as e:
            result["error_output"] = f"Command execution failed: {str(e)}"
        
        result["duration"] = time.time() - start_time
        return result
    
    def _process_coverage_results(self) -> Dict[str, Any]:
        """Process coverage results."""
        result = {
            "coverage_percentage": 0,
            "line_coverage": 0,
            "branch_coverage": 0,
            "missing_lines": 0,
            "files_analyzed": 0,
            "report_path": "htmlcov/index.html"
        }
        
        try:
            # Parse coverage XML report
            coverage_xml = self.project_root / "coverage.xml"
            if coverage_xml.exists():
                import xml.etree.ElementTree as ET
                tree = ET.parse(coverage_xml)
                root = tree.getroot()
                
                # Extract overall coverage
                coverage_elem = root.find(".//coverage")
                if coverage_elem is not None:
                    result["line_coverage"] = float(coverage_elem.get("line-rate", 0)) * 100
                    result["branch_coverage"] = float(coverage_elem.get("branch-rate", 0)) * 100
                    result["coverage_percentage"] = result["line_coverage"]
                
                # Count files
                packages = root.findall(".//package")
                result["files_analyzed"] = sum(
                    len(pkg.findall(".//class")) for pkg in packages
                )
            
            # Parse text coverage output for missing lines
            # This would require parsing pytest output or coverage report
            
        except Exception as e:
            print(f"Warning: Could not process coverage results: {e}")
        
        return result
    
    def _process_performance_results(self) -> Dict[str, Any]:
        """Process performance benchmark results."""
        result = {
            "benchmarks_run": 0,
            "baseline_comparison": {},
            "regressions": [],
            "improvements": [],
            "report_path": "performance_results/benchmarks.json"
        }
        
        try:
            benchmark_file = self.project_root / "performance_results" / "benchmarks.json"
            if benchmark_file.exists():
                with open(benchmark_file) as f:
                    benchmark_data = json.load(f)
                
                result["benchmarks_run"] = len(benchmark_data.get("benchmarks", []))
                
                # Compare against baselines if available
                baseline_file = (
                    self.project_root / 
                    "tests" / "integration" / "performance_baselines.json"
                )
                
                if baseline_file.exists():
                    with open(baseline_file) as f:
                        baselines = json.load(f)
                    
                    # Detect regressions
                    for benchmark in benchmark_data.get("benchmarks", []):
                        name = benchmark.get("name", "")
                        current_time = benchmark.get("stats", {}).get("mean", 0)
                        
                        # Find matching baseline
                        baseline_time = None
                        for baseline_category, baseline_data in baselines.items():
                            if name in baseline_category or baseline_category in name:
                                baseline_time = baseline_data.get("avg_processing_time_ms", 0) / 1000
                                break
                        
                        if baseline_time and current_time > baseline_time * 1.2:  # 20% regression
                            result["regressions"].append({
                                "test": name,
                                "baseline": baseline_time,
                                "current": current_time,
                                "regression_percent": ((current_time - baseline_time) / baseline_time) * 100
                            })
            
        except Exception as e:
            print(f"Warning: Could not process performance results: {e}")
        
        return result
    
    def _generate_reports(self, results: Dict[str, Any]):
        """Generate comprehensive test reports."""
        
        # Generate summary report
        summary_report = {
            "test_run_summary": {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "duration_seconds": results.get("duration_seconds", 0),
                "success": results["success"],
                "errors": results["errors"]
            },
            "test_results": results["test_results"],
            "coverage": results["coverage_results"],
            "performance": results["performance_results"]
        }
        
        # Save JSON report
        report_file = self.project_root / "test_results" / "integration_test_report.json"
        report_file.parent.mkdir(exist_ok=True)
        
        with open(report_file, "w") as f:
            json.dump(summary_report, f, indent=2)
        
        print(f"📄 Test report saved to: {report_file}")
        
        # Generate markdown summary
        self._generate_markdown_summary(results, summary_report)
    
    def _generate_markdown_summary(self, results: Dict[str, Any], summary: Dict[str, Any]):
        """Generate markdown summary report."""
        
        md_content = f"""# Integration Test Results

**Test Run Date:** {summary['test_run_summary']['timestamp']}
**Duration:** {summary['test_run_summary']['duration_seconds']:.1f} seconds
**Overall Result:** {'✅ PASSED' if results['success'] else '❌ FAILED'}

## Test Categories

"""
        
        for category, result in results["test_results"].items():
            status = "✅ PASSED" if result["success"] else "❌ FAILED"
            duration = f"{result['duration']:.1f}s"
            md_content += f"- **{category.title()}**: {status} ({duration})\n"
        
        if results["coverage_results"]:
            cov = results["coverage_results"]
            md_content += f"""
## Coverage Report

- **Line Coverage:** {cov['line_coverage']:.1f}%
- **Branch Coverage:** {cov['branch_coverage']:.1f}%
- **Files Analyzed:** {cov['files_analyzed']}
- **Report:** [HTML Report](htmlcov/index.html)

"""
        
        if results["performance_results"]:
            perf = results["performance_results"]
            md_content += f"""
## Performance Results

- **Benchmarks Run:** {perf['benchmarks_run']}
- **Regressions:** {len(perf['regressions'])}
- **Improvements:** {len(perf['improvements'])}

"""
            
            if perf["regressions"]:
                md_content += "### Performance Regressions\n\n"
                for reg in perf["regressions"]:
                    md_content += f"- **{reg['test']}**: {reg['regression_percent']:.1f}% slower\n"
        
        if results["errors"]:
            md_content += f"""
## Errors

"""
            for error in results["errors"]:
                md_content += f"- {error}\n"
        
        # Save markdown report
        md_file = self.project_root / "test_results" / "integration_test_summary.md"
        with open(md_file, "w") as f:
            f.write(md_content)
        
        print(f"📄 Markdown summary saved to: {md_file}")
    
    def _cleanup_test_environment(self):
        """Clean up test environment."""
        try:
            # Stop any running containers
            subprocess.run(
                ["docker", "container", "prune", "-f"],
                capture_output=True
            )
            
            # Clean up temporary files
            temp_patterns = ["*.tmp", "*.temp", ".pytest_tmp*"]
            for pattern in temp_patterns:
                for temp_file in self.project_root.rglob(pattern):
                    if temp_file.is_file():
                        temp_file.unlink()
            
        except Exception as e:
            print(f"Cleanup warning: {e}")


def main():
    """Main entry point for integration test runner."""
    
    parser = argparse.ArgumentParser(
        description="Run comprehensive integration tests"
    )
    
    parser.add_argument(
        "--categories",
        nargs="+",
        choices=["all", "smoke", "integration", "performance", "regression", "slow"],
        default=["integration"],
        help="Test categories to run"
    )
    
    parser.add_argument(
        "--no-coverage",
        action="store_true",
        help="Disable coverage measurement"
    )
    
    parser.add_argument(
        "--no-performance",
        action="store_true", 
        help="Disable performance tests"
    )
    
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Run tests in parallel"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose output"
    )
    
    parser.add_argument(
        "--coverage-threshold",
        type=int,
        default=80,
        help="Coverage threshold percentage (default: 80)"
    )
    
    args = parser.parse_args()
    
    # Find project root
    project_root = Path(__file__).parent.parent
    if not (project_root / "pyproject.toml").exists():
        print("Error: Could not find project root")
        sys.exit(1)
    
    # Create test runner
    runner = IntegrationTestRunner(project_root)
    runner.coverage_threshold = args.coverage_threshold
    
    # Run tests
    results = runner.run_test_suite(
        test_categories=args.categories,
        coverage=not args.no_coverage,
        performance=not args.no_performance,
        parallel=args.parallel,
        verbose=args.verbose
    )
    
    # Print summary
    print(f"\n{'='*60}")
    print("INTEGRATION TEST SUMMARY")
    print(f"{'='*60}")
    print(f"Result: {'✅ PASSED' if results['success'] else '❌ FAILED'}")
    print(f"Duration: {results.get('duration_seconds', 0):.1f} seconds")
    
    if results.get("coverage_results", {}).get("coverage_percentage"):
        cov_pct = results["coverage_results"]["coverage_percentage"]
        print(f"Coverage: {cov_pct:.1f}%")
    
    if results["errors"]:
        print(f"Errors: {len(results['errors'])}")
        for error in results["errors"]:
            print(f"  - {error}")
    
    # Exit with appropriate code
    sys.exit(0 if results["success"] else 1)


if __name__ == "__main__":
    main()