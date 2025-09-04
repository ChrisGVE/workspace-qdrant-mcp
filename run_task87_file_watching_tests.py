#!/usr/bin/env python3
"""
Task 87 Test Runner - File Watching and Auto-Ingestion Testing

This script runs the comprehensive file watching and auto-ingestion tests
for Task 87, covering all required test areas:

1. File watching system validation
2. Automatic ingestion trigger testing
3. Real-time status update verification
4. Watch configuration testing
5. Error scenario handling and service persistence testing

Usage:
    python run_task87_file_watching_tests.py [--verbose] [--coverage]
"""

import asyncio
import logging
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class Task87TestRunner:
    """Test runner for Task 87 file watching and auto-ingestion testing."""
    
    def __init__(self, verbose: bool = False, coverage: bool = False):
        """Initialize test runner."""
        self.verbose = verbose
        self.coverage = coverage
        self.project_root = Path(__file__).parent
        self.test_file = self.project_root / "tests" / "test_file_watching_comprehensive.py"
        
        # Test results tracking
        self.test_results: Dict[str, Dict] = {}
        self.total_tests = 0
        self.passed_tests = 0
        self.failed_tests = 0
        self.skipped_tests = 0
    
    def check_prerequisites(self) -> bool:
        """Check that all prerequisites are available."""
        logger.info("Checking prerequisites for Task 87 testing...")
        
        # Check test file exists
        if not self.test_file.exists():
            logger.error(f"Test file not found: {self.test_file}")
            return False
        
        # Check required modules
        required_modules = [
            "workspace_qdrant_mcp.core.file_watcher",
            "pytest",
            "asyncio",
        ]
        
        missing_modules = []
        for module in required_modules:
            try:
                __import__(module)
            except ImportError:
                missing_modules.append(module)
        
        if missing_modules:
            logger.error(f"Missing required modules: {missing_modules}")
            return False
        
        logger.info("All prerequisites satisfied")
        return True
    
    async def run_test_area_1_file_watching_validation(self) -> Dict:
        """Run Test Area 1: File watching system validation."""
        logger.info("Running Test Area 1: File watching system validation...")
        
        cmd = [
            sys.executable, "-m", "pytest",
            str(self.test_file) + "::TestFileWatchingSystemValidation",
            "-v" if self.verbose else "-q",
            "--tb=short"
        ]
        
        if self.coverage:
            cmd.extend([
                "--cov=workspace_qdrant_mcp.core.file_watcher",
                "--cov-report=term-missing"
            ])
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        test_results = {
            "area": "File Watching System Validation",
            "return_code": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "passed": result.returncode == 0
        }
        
        self.test_results["area_1"] = test_results
        
        if result.returncode == 0:
            logger.info("✓ Test Area 1: File watching system validation - PASSED")
        else:
            logger.error("✗ Test Area 1: File watching system validation - FAILED")
            if self.verbose:
                logger.error(f"STDOUT: {result.stdout}")
                logger.error(f"STDERR: {result.stderr}")
        
        return test_results
    
    async def run_test_area_2_ingestion_triggers(self) -> Dict:
        """Run Test Area 2: Automatic ingestion trigger testing."""
        logger.info("Running Test Area 2: Automatic ingestion trigger testing...")
        
        cmd = [
            sys.executable, "-m", "pytest",
            str(self.test_file) + "::TestAutomaticIngestionTriggers",
            "-v" if self.verbose else "-q",
            "--tb=short"
        ]
        
        if self.coverage:
            cmd.extend([
                "--cov=workspace_qdrant_mcp.core.file_watcher",
                "--cov-report=term-missing"
            ])
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        test_results = {
            "area": "Automatic Ingestion Triggers",
            "return_code": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "passed": result.returncode == 0
        }
        
        self.test_results["area_2"] = test_results
        
        if result.returncode == 0:
            logger.info("✓ Test Area 2: Automatic ingestion triggers - PASSED")
        else:
            logger.error("✗ Test Area 2: Automatic ingestion triggers - FAILED")
            if self.verbose:
                logger.error(f"STDOUT: {result.stdout}")
                logger.error(f"STDERR: {result.stderr}")
        
        return test_results
    
    async def run_test_area_3_status_updates(self) -> Dict:
        """Run Test Area 3: Real-time status update verification."""
        logger.info("Running Test Area 3: Real-time status update verification...")
        
        cmd = [
            sys.executable, "-m", "pytest",
            str(self.test_file) + "::TestRealTimeStatusUpdates",
            "-v" if self.verbose else "-q",
            "--tb=short"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        test_results = {
            "area": "Real-time Status Updates",
            "return_code": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "passed": result.returncode == 0
        }
        
        self.test_results["area_3"] = test_results
        
        if result.returncode == 0:
            logger.info("✓ Test Area 3: Real-time status updates - PASSED")
        else:
            logger.error("✗ Test Area 3: Real-time status updates - FAILED")
            if self.verbose:
                logger.error(f"STDOUT: {result.stdout}")
                logger.error(f"STDERR: {result.stderr}")
        
        return test_results
    
    async def run_test_area_4_configuration_management(self) -> Dict:
        """Run Test Area 4: Watch configuration testing."""
        logger.info("Running Test Area 4: Watch configuration testing...")
        
        cmd = [
            sys.executable, "-m", "pytest",
            str(self.test_file) + "::TestWatchConfigurationManagement",
            "-v" if self.verbose else "-q",
            "--tb=short"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        test_results = {
            "area": "Watch Configuration Management",
            "return_code": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "passed": result.returncode == 0
        }
        
        self.test_results["area_4"] = test_results
        
        if result.returncode == 0:
            logger.info("✓ Test Area 4: Watch configuration management - PASSED")
        else:
            logger.error("✗ Test Area 4: Watch configuration management - FAILED")
            if self.verbose:
                logger.error(f"STDOUT: {result.stdout}")
                logger.error(f"STDERR: {result.stderr}")
        
        return test_results
    
    async def run_test_area_5_error_handling(self) -> Dict:
        """Run Test Area 5: Error scenario handling and service persistence."""
        logger.info("Running Test Area 5: Error scenario handling and service persistence...")
        
        cmd = [
            sys.executable, "-m", "pytest",
            str(self.test_file) + "::TestErrorScenarioHandling",
            "-v" if self.verbose else "-q",
            "--tb=short"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        test_results = {
            "area": "Error Scenario Handling",
            "return_code": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "passed": result.returncode == 0
        }
        
        self.test_results["area_5"] = test_results
        
        if result.returncode == 0:
            logger.info("✓ Test Area 5: Error scenario handling - PASSED")
        else:
            logger.error("✗ Test Area 5: Error scenario handling - FAILED")
            if self.verbose:
                logger.error(f"STDOUT: {result.stdout}")
                logger.error(f"STDERR: {result.stderr}")
        
        return test_results
    
    async def run_service_persistence_tests(self) -> Dict:
        """Run service persistence tests."""
        logger.info("Running service persistence tests...")
        
        cmd = [
            sys.executable, "-m", "pytest",
            str(self.test_file) + "::TestWatchServicePersistence",
            "-v" if self.verbose else "-q",
            "--tb=short"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        test_results = {
            "area": "Watch Service Persistence",
            "return_code": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "passed": result.returncode == 0
        }
        
        self.test_results["persistence"] = test_results
        
        if result.returncode == 0:
            logger.info("✓ Service persistence tests - PASSED")
        else:
            logger.error("✗ Service persistence tests - FAILED")
            if self.verbose:
                logger.error(f"STDOUT: {result.stdout}")
                logger.error(f"STDERR: {result.stderr}")
        
        return test_results
    
    async def run_performance_tests(self) -> Dict:
        """Run performance and stress tests."""
        logger.info("Running performance and stress tests...")
        
        cmd = [
            sys.executable, "-m", "pytest",
            str(self.test_file) + "::TestFileWatchingPerformance",
            "-v" if self.verbose else "-q",
            "--tb=short"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        test_results = {
            "area": "Performance Testing",
            "return_code": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "passed": result.returncode == 0
        }
        
        self.test_results["performance"] = test_results
        
        if result.returncode == 0:
            logger.info("✓ Performance tests - PASSED")
        else:
            logger.error("✗ Performance tests - FAILED")
            if self.verbose:
                logger.error(f"STDOUT: {result.stdout}")
                logger.error(f"STDERR: {result.stderr}")
        
        return test_results
    
    async def run_comprehensive_test_suite(self) -> Dict:
        """Run the complete comprehensive test suite."""
        logger.info("Starting Task 87 comprehensive file watching test suite...")
        
        if not self.check_prerequisites():
            return {"error": "Prerequisites not met"}
        
        # Run all test areas
        test_functions = [
            self.run_test_area_1_file_watching_validation,
            self.run_test_area_2_ingestion_triggers,
            self.run_test_area_3_status_updates,
            self.run_test_area_4_configuration_management,
            self.run_test_area_5_error_handling,
            self.run_service_persistence_tests,
            self.run_performance_tests
        ]
        
        all_passed = True
        for test_func in test_functions:
            try:
                result = await test_func()
                if not result["passed"]:
                    all_passed = False
            except Exception as e:
                logger.error(f"Error running {test_func.__name__}: {e}")
                all_passed = False
        
        return {
            "overall_success": all_passed,
            "test_results": self.test_results
        }
    
    def generate_test_report(self) -> str:
        """Generate a comprehensive test report."""
        report_lines = [
            "=" * 80,
            "Task 87: File Watching and Auto-Ingestion Testing Report",
            "=" * 80,
            "",
            "Test Areas Covered:",
            "1. File watching system validation",
            "2. Automatic ingestion trigger testing", 
            "3. Real-time status update verification",
            "4. Watch configuration testing",
            "5. Error scenario handling and service persistence testing",
            "",
            "Test Results Summary:",
            "-" * 40
        ]
        
        passed_areas = 0
        total_areas = len(self.test_results)
        
        for area_key, results in self.test_results.items():
            status = "PASSED" if results["passed"] else "FAILED"
            report_lines.append(f"{results['area']}: {status}")
            if results["passed"]:
                passed_areas += 1
        
        report_lines.extend([
            "",
            f"Overall Results: {passed_areas}/{total_areas} test areas passed",
            f"Success Rate: {(passed_areas/total_areas)*100:.1f}%" if total_areas > 0 else "No tests run",
            "",
            "=" * 80
        ])
        
        return "\n".join(report_lines)


async def main():
    """Main entry point for Task 87 testing."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Run Task 87 file watching and auto-ingestion tests"
    )
    parser.add_argument(
        "--verbose", "-v", 
        action="store_true",
        help="Enable verbose output"
    )
    parser.add_argument(
        "--coverage", "-c",
        action="store_true", 
        help="Enable coverage reporting"
    )
    
    args = parser.parse_args()
    
    # Create test runner
    runner = Task87TestRunner(
        verbose=args.verbose,
        coverage=args.coverage
    )
    
    # Run comprehensive test suite
    logger.info("Starting Task 87: File Watching and Auto-Ingestion Testing")
    
    results = await runner.run_comprehensive_test_suite()
    
    # Generate and display report
    report = runner.generate_test_report()
    print(report)
    
    # Exit with appropriate code
    if results.get("overall_success", False):
        logger.info("Task 87 testing completed successfully!")
        sys.exit(0)
    else:
        logger.error("Task 87 testing failed - see report above for details")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())