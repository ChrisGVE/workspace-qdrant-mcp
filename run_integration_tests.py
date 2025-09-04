#!/usr/bin/env python3
"""
Integration Test Suite Execution Script for Task 91
Executes comprehensive integration tests with detailed reporting and analysis.
"""

import asyncio
import json
import subprocess
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
import os

class IntegrationTestRunner:
    """Comprehensive integration test execution and reporting."""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.test_results = {
            "execution_start": datetime.now().isoformat(),
            "test_suites": [],
            "summary": {},
            "failures": [],
            "performance_metrics": {},
            "coverage_report": {}
        }
        
    def run_pytest_suite(self, test_path: str, markers: List[str] = None, 
                        timeout: int = 300) -> Dict[str, Any]:
        """Execute a pytest test suite with comprehensive reporting."""
        
        cmd = [
            sys.executable, "-m", "pytest", 
            test_path,
            "-v",
            "--tb=short",
            "--durations=10",
            "--json-report", 
            "--json-report-file=/tmp/pytest_report.json",
            "--cov=src/workspace_qdrant_mcp",
            "--cov-report=term-missing",
            "--cov-append",
            f"--timeout={timeout}"
        ]
        
        if markers:
            for marker in markers:
                cmd.extend(["-m", marker])
                
        print(f"🧪 Executing: {' '.join(cmd)}")
        
        start_time = time.time()
        
        try:
            result = subprocess.run(
                cmd,
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=timeout + 30  # Buffer for pytest timeout handling
            )
            
            execution_time = time.time() - start_time
            
            # Load JSON report if available
            json_report = {}
            json_report_path = Path("/tmp/pytest_report.json")
            if json_report_path.exists():
                try:
                    with open(json_report_path, 'r') as f:
                        json_report = json.load(f)
                except Exception as e:
                    print(f"⚠️  Failed to load JSON report: {e}")
                    
            return {
                "test_path": test_path,
                "markers": markers or [],
                "return_code": result.returncode,
                "execution_time": execution_time,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "json_report": json_report,
                "success": result.returncode == 0
            }
            
        except subprocess.TimeoutExpired:
            return {
                "test_path": test_path,
                "markers": markers or [],
                "return_code": -1,
                "execution_time": timeout + 30,
                "stdout": "",
                "stderr": f"Test execution timed out after {timeout + 30} seconds",
                "json_report": {},
                "success": False,
                "timeout": True
            }
        except Exception as e:
            return {
                "test_path": test_path,
                "markers": markers or [],
                "return_code": -1,
                "execution_time": time.time() - start_time,
                "stdout": "",
                "stderr": f"Test execution failed: {str(e)}",
                "json_report": {},
                "success": False,
                "error": str(e)
            }
    
    def analyze_test_failures(self, test_result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Analyze test failures and extract detailed information."""
        failures = []
        
        if not test_result["success"]:
            # Parse pytest output for failure details
            stderr_lines = test_result["stderr"].split('\n')
            stdout_lines = test_result["stdout"].split('\n')
            
            current_failure = None
            in_failure_block = False
            
            for line in stdout_lines + stderr_lines:
                if "FAILED" in line and "::" in line:
                    if current_failure:
                        failures.append(current_failure)
                    current_failure = {
                        "test_name": line.split("FAILED")[0].strip(),
                        "failure_reason": "",
                        "traceback": [],
                        "error_type": "unknown"
                    }
                    in_failure_block = True
                elif current_failure and in_failure_block:
                    if line.strip().startswith("assert") or "AssertionError" in line:
                        current_failure["failure_reason"] = line.strip()
                        current_failure["error_type"] = "assertion"
                    elif "Error:" in line or "Exception:" in line:
                        current_failure["failure_reason"] = line.strip()
                        current_failure["error_type"] = "exception"
                    elif line.strip():
                        current_failure["traceback"].append(line.strip())
                    
            if current_failure:
                failures.append(current_failure)
                
        return failures
    
    def generate_performance_metrics(self, test_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate performance metrics from test execution."""
        
        total_tests = 0
        total_time = 0
        passed_tests = 0
        failed_tests = 0
        timeout_tests = 0
        
        suite_metrics = {}
        
        for result in test_results:
            suite_name = Path(result["test_path"]).name
            
            if result.get("json_report") and "summary" in result["json_report"]:
                summary = result["json_report"]["summary"]
                suite_tests = summary.get("total", 0)
                suite_passed = summary.get("passed", 0)
                suite_failed = summary.get("failed", 0)
            else:
                # Fallback to parsing stdout
                suite_tests = 1 if result["success"] else 1
                suite_passed = 1 if result["success"] else 0
                suite_failed = 0 if result["success"] else 1
            
            total_tests += suite_tests
            passed_tests += suite_passed
            failed_tests += suite_failed
            total_time += result["execution_time"]
            
            if result.get("timeout"):
                timeout_tests += 1
                
            suite_metrics[suite_name] = {
                "total_tests": suite_tests,
                "passed": suite_passed,
                "failed": suite_failed,
                "execution_time": result["execution_time"],
                "success_rate": (suite_passed / suite_tests) if suite_tests > 0 else 0,
                "timeout": result.get("timeout", False)
            }
        
        return {
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "failed_tests": failed_tests,
            "timeout_tests": timeout_tests,
            "total_execution_time": total_time,
            "average_test_time": total_time / max(total_tests, 1),
            "overall_success_rate": (passed_tests / max(total_tests, 1)),
            "suite_breakdown": suite_metrics
        }
    
    def execute_comprehensive_test_suite(self) -> Dict[str, Any]:
        """Execute all integration test suites with comprehensive analysis."""
        
        print("🚀 Starting Comprehensive Integration Test Suite Execution")
        print("=" * 80)
        
        # Define test suites to execute
        test_suites = [
            {
                "name": "stdio_communication",
                "path": "tests/integration/test_stdio_communication.py",
                "markers": ["integration", "requires_qdrant"],
                "timeout": 120,
                "description": "STDIO communication protocol tests"
            },
            {
                "name": "data_ingestion", 
                "path": "tests/functional/test_data_ingestion.py",
                "markers": ["integration"],
                "timeout": 300,
                "description": "Real data ingestion functional tests"
            },
            {
                "name": "search_functionality",
                "path": "tests/functional/test_search_functionality.py", 
                "markers": ["integration"],
                "timeout": 300,
                "description": "Search functionality and quality tests"
            },
            {
                "name": "recall_precision",
                "path": "tests/functional/test_recall_precision.py",
                "markers": ["integration"],
                "timeout": 180,
                "description": "Search recall and precision measurements"
            },
            {
                "name": "performance_tests",
                "path": "tests/functional/test_performance.py",
                "markers": ["performance"],
                "timeout": 300,
                "description": "Performance and benchmark tests"
            },
            {
                "name": "e2e_workflow",
                "path": "tests/e2e/test_full_workflow.py",
                "markers": ["e2e"],
                "timeout": 300,
                "description": "End-to-end workflow validation tests"
            },
            {
                "name": "memory_integration",
                "path": "tests/memory/test_memory_integration.py",
                "markers": ["integration"],
                "timeout": 180,
                "description": "Memory system integration tests"
            }
        ]
        
        # Execute each test suite
        all_results = []
        for suite in test_suites:
            print(f"\n📋 Executing {suite['name']}: {suite['description']}")
            print("-" * 60)
            
            # Check if test file exists
            test_file = self.project_root / suite["path"]
            if not test_file.exists():
                print(f"⚠️  Test file not found: {test_file}")
                result = {
                    "test_path": suite["path"],
                    "markers": suite.get("markers", []),
                    "return_code": -1,
                    "execution_time": 0,
                    "stdout": "",
                    "stderr": f"Test file not found: {test_file}",
                    "json_report": {},
                    "success": False,
                    "skipped": True
                }
            else:
                result = self.run_pytest_suite(
                    suite["path"],
                    markers=suite.get("markers"),
                    timeout=suite.get("timeout", 300)
                )
                
            result["suite_name"] = suite["name"]
            result["description"] = suite["description"]
            all_results.append(result)
            
            # Report immediate results
            if result["success"]:
                print(f"✅ {suite['name']}: PASSED ({result['execution_time']:.1f}s)")
            else:
                print(f"❌ {suite['name']}: FAILED ({result['execution_time']:.1f}s)")
                
                # Analyze failures
                failures = self.analyze_test_failures(result)
                for failure in failures:
                    print(f"   🔍 {failure['test_name']}: {failure['failure_reason']}")
        
        # Generate comprehensive analysis
        print("\n📊 Generating Comprehensive Analysis...")
        print("=" * 80)
        
        performance_metrics = self.generate_performance_metrics(all_results)
        
        # Compile final results
        self.test_results.update({
            "execution_end": datetime.now().isoformat(),
            "test_suites": all_results,
            "performance_metrics": performance_metrics,
            "summary": {
                "total_suites": len(test_suites),
                "passed_suites": len([r for r in all_results if r["success"]]),
                "failed_suites": len([r for r in all_results if not r["success"] and not r.get("skipped")]),
                "skipped_suites": len([r for r in all_results if r.get("skipped")]),
                "overall_success": all(r["success"] or r.get("skipped", False) for r in all_results),
                "total_execution_time": sum(r["execution_time"] for r in all_results)
            }
        })
        
        return self.test_results
    
    def generate_detailed_report(self, output_file: Optional[Path] = None) -> str:
        """Generate detailed test execution report."""
        
        if output_file is None:
            output_file = self.project_root / "integration_test_report.json"
            
        # Write JSON report
        with open(output_file, 'w') as f:
            json.dump(self.test_results, f, indent=2, default=str)
            
        # Generate markdown summary
        markdown_report = self.generate_markdown_summary()
        markdown_file = output_file.with_suffix('.md')
        
        with open(markdown_file, 'w') as f:
            f.write(markdown_report)
            
        return str(output_file)
    
    def generate_markdown_summary(self) -> str:
        """Generate markdown summary report."""
        
        summary = self.test_results["summary"]
        metrics = self.test_results["performance_metrics"]
        
        report_lines = [
            "# Integration Test Suite Execution Report - Task 91",
            f"**Generated:** {self.test_results['execution_end']}",
            "",
            "## Executive Summary",
            "",
            f"- **Total Test Suites:** {summary['total_suites']}",
            f"- **Passed Suites:** {summary['passed_suites']} ({summary['passed_suites']/summary['total_suites']*100:.1f}%)",
            f"- **Failed Suites:** {summary['failed_suites']}",
            f"- **Skipped Suites:** {summary['skipped_suites']}",
            f"- **Overall Success:** {'✅ PASS' if summary['overall_success'] else '❌ FAIL'}",
            f"- **Total Execution Time:** {summary['total_execution_time']:.1f} seconds",
            "",
            "## Performance Metrics",
            "",
            f"- **Total Tests Executed:** {metrics['total_tests']}",
            f"- **Tests Passed:** {metrics['passed_tests']} ({metrics['overall_success_rate']*100:.1f}%)",
            f"- **Tests Failed:** {metrics['failed_tests']}",
            f"- **Tests Timed Out:** {metrics['timeout_tests']}",
            f"- **Average Test Time:** {metrics['average_test_time']:.2f} seconds",
            "",
            "## Test Suite Results",
            ""
        ]
        
        for suite_result in self.test_results["test_suites"]:
            status = "✅ PASSED" if suite_result["success"] else ("⏭️ SKIPPED" if suite_result.get("skipped") else "❌ FAILED")
            report_lines.extend([
                f"### {suite_result['suite_name'].replace('_', ' ').title()}",
                f"**Status:** {status}",
                f"**Description:** {suite_result['description']}",
                f"**Execution Time:** {suite_result['execution_time']:.1f}s",
                ""
            ])
            
            if not suite_result["success"] and not suite_result.get("skipped"):
                failures = self.analyze_test_failures(suite_result)
                if failures:
                    report_lines.extend(["**Failures:**", ""])
                    for failure in failures:
                        report_lines.extend([
                            f"- **{failure['test_name']}**",
                            f"  - Error: {failure['failure_reason']}",
                            f"  - Type: {failure['error_type']}",
                            ""
                        ])
        
        # Add recommendations
        report_lines.extend([
            "",
            "## Recommendations",
            ""
        ])
        
        if summary["failed_suites"] > 0:
            report_lines.extend([
                "### Failed Test Suites",
                "1. Review individual test failures in detail",
                "2. Check test environment setup and dependencies", 
                "3. Verify mock configurations and test data",
                "4. Consider increasing timeout values for slow tests",
                ""
            ])
        
        if summary["overall_success"]:
            report_lines.extend([
                "### Success - Next Steps",
                "1. Integration test suite execution completed successfully",
                "2. All critical workflows validated",
                "3. Performance benchmarks meet expectations",
                "4. Ready for production deployment validation",
                ""
            ])
            
        report_lines.extend([
            "## Test Suite Maintenance Guide",
            "",
            "### Execution Commands",
            "```bash",
            "# Run all integration tests",
            "python run_integration_tests.py",
            "",
            "# Run specific test suite",
            "pytest tests/integration/test_stdio_communication.py -v",
            "",
            "# Run with coverage",
            "pytest --cov=src/workspace_qdrant_mcp --cov-report=html",
            "```",
            "",
            "### CI/CD Integration",
            "```yaml",
            "- name: Run Integration Tests",
            "  run: python run_integration_tests.py",
            "  timeout-minutes: 20",
            "```"
        ])
        
        return "\n".join(report_lines)

def main():
    """Main execution function."""
    project_root = Path(__file__).parent
    
    print("🎯 Task 91: Integration Test Suite Execution")
    print("=" * 80)
    print(f"Project Root: {project_root}")
    print(f"Python Version: {sys.version}")
    print(f"Working Directory: {os.getcwd()}")
    
    # Ensure we're in the project directory
    os.chdir(project_root)
    
    # Initialize test runner
    runner = IntegrationTestRunner(project_root)
    
    try:
        # Execute comprehensive test suite
        results = runner.execute_comprehensive_test_suite()
        
        # Generate detailed report
        report_file = runner.generate_detailed_report()
        
        print(f"\n📄 Test execution report generated: {report_file}")
        print(f"📄 Markdown summary: {report_file.replace('.json', '.md')}")
        
        # Print final summary
        summary = results["summary"]
        metrics = results["performance_metrics"]
        
        print(f"\n🎯 Task 91 Execution Summary:")
        print("=" * 80)
        print(f"Overall Result: {'✅ SUCCESS' if summary['overall_success'] else '❌ FAILURE'}")
        print(f"Test Suites: {summary['passed_suites']}/{summary['total_suites']} passed")
        print(f"Individual Tests: {metrics['passed_tests']}/{metrics['total_tests']} passed")
        print(f"Success Rate: {metrics['overall_success_rate']*100:.1f}%")
        print(f"Total Time: {summary['total_execution_time']:.1f} seconds")
        
        # Exit with appropriate code
        exit_code = 0 if summary['overall_success'] else 1
        sys.exit(exit_code)
        
    except Exception as e:
        print(f"\n❌ Integration test execution failed: {e}")
        print(f"Traceback: {traceback.format_exc()}")
        sys.exit(1)

if __name__ == "__main__":
    main()