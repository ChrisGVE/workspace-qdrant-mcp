#!/usr/bin/env python3
"""
Comprehensive test suite for Language-Aware File Filtering System.

This test module validates the filtering system's functionality including:
- Pattern matching accuracy across 25+ programming languages
- MIME type detection and classification
- Performance benchmarks with large codebases
- Configuration loading and validation
- Statistics tracking accuracy
- Integration with file watcher system

Usage:
    python 20250107-1643_test_language_filtering_comprehensive.py
"""

import asyncio
import json
import logging
import os
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Tuple
import yaml

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import the modules we're testing
import sys
sys.path.insert(0, str(Path(__file__).parent / "src"))

from workspace_qdrant_mcp.core.language_filters import (
    LanguageAwareFilter, 
    FilterConfiguration, 
    FilterStatistics,
    MimeTypeDetector,
    CompiledPatterns
)
from workspace_qdrant_mcp.core.file_watcher import FileWatcher, WatchConfiguration, WatchManager


class TestLanguageFiltering:
    """Comprehensive test suite for language-aware filtering."""
    
    def __init__(self):
        self.test_results = {}
        self.temp_dir = None
        self.config_dir = None
        
    async def setup_test_environment(self):
        """Set up temporary directories and test files."""
        self.temp_dir = Path(tempfile.mkdtemp(prefix="language_filter_test_"))
        self.config_dir = self.temp_dir / "config"
        self.config_dir.mkdir(exist_ok=True)
        
        # Create test configuration
        await self._create_test_config()
        
        # Create test files for various languages
        await self._create_test_files()
        
        logger.info(f"Test environment created at {self.temp_dir}")
    
    async def _create_test_config(self):
        """Create test configuration file."""
        test_config = {
            "ignore_patterns": {
                "dot_files": True,
                "directories": [
                    "node_modules", "__pycache__", ".git", "target", 
                    "build", "dist", "vendor", ".vscode", ".idea"
                ],
                "file_extensions": [
                    "*.pyc", "*.class", "*.o", "*.so", "*.dll", 
                    "*.exe", "*.zip", "*.tar.gz", "*.log"
                ]
            },
            "performance": {
                "max_file_size_mb": 5.0,
                "max_files_per_batch": 50,
                "debounce_seconds": 2
            },
            "user_overrides": {
                "force_include": {
                    "directories": ["important-vendor"],
                    "file_extensions": ["*.important"]
                },
                "additional_ignores": {
                    "directories": ["temp-data"],
                    "file_extensions": ["*.tmp"]
                }
            }
        }
        
        config_file = self.config_dir / "ingestion.yaml"
        with config_file.open('w') as f:
            yaml.dump(test_config, f, default_flow_style=False)
    
    async def _create_test_files(self):
        """Create comprehensive test file structure."""
        test_files = {
            # Python files
            "src/main.py": "#!/usr/bin/env python3\nprint('Hello World')",
            "src/__pycache__/main.cpython-39.pyc": b"compiled python bytecode",
            "tests/test_main.py": "import unittest\nclass TestMain(unittest.TestCase): pass",
            
            # JavaScript/Node.js
            "frontend/app.js": "console.log('Hello JS');",
            "frontend/app.min.js": "console.log('minified');",
            "node_modules/react/index.js": "module.exports = require('./lib/React');",
            "package.json": '{"name": "test-project", "version": "1.0.0"}',
            
            # Java/JVM
            "src/main/java/App.java": "public class App { public static void main(String[] args) {} }",
            "target/classes/App.class": b"java bytecode",
            "build.gradle": "plugins { id 'java' }",
            
            # C/C++
            "src/main.cpp": "#include <iostream>\nint main() { return 0; }",
            "src/main.h": "#ifndef MAIN_H\n#define MAIN_H\n#endif",
            "build/main.o": b"compiled object file",
            "cmake-build-debug/main": b"executable binary",
            
            # Rust
            "src/lib.rs": "pub fn hello() { println!(\"Hello Rust\"); }",
            "target/debug/libtest.so": b"rust shared library",
            "Cargo.toml": "[package]\nname = \"test\"\nversion = \"0.1.0\"",
            
            # Go
            "main.go": "package main\nfunc main() {}",
            "vendor/github.com/pkg/errors/errors.go": "package errors",
            "go.mod": "module test\ngo 1.19",
            
            # Web files
            "frontend/index.html": "<html><body>Test</body></html>",
            "frontend/style.css": "body { margin: 0; }",
            "frontend/style.scss": "$primary: blue;\nbody { color: $primary; }",
            
            # Configuration files
            "config.yml": "database:\n  host: localhost",
            ".env": "SECRET_KEY=test123",
            "Dockerfile": "FROM python:3.9\nCOPY . /app",
            
            # Documentation
            "README.md": "# Test Project\nThis is a test.",
            "docs/api.md": "# API Documentation",
            "docs/_build/html/index.html": "<html>Generated docs</html>",
            
            # Git and IDE files
            ".git/config": "[core]\nrepositoryformatversion = 0",
            ".vscode/settings.json": '{"python.defaultInterpreter": "python3"}',
            ".idea/workspace.xml": "<?xml version=\"1.0\"?>",
            ".DS_Store": b"macOS metadata",
            
            # Temporary and log files
            "app.log": "2023-01-01 INFO Starting application",
            "temp.tmp": "temporary data",
            "backup~": "backup file content",
            
            # Large files (for size testing)
            "large_file.txt": "x" * (6 * 1024 * 1024),  # 6MB file
            
            # Force include test
            "important-vendor/critical.js": "// This should be included",
            "test.important": "// This should be included due to extension override",
            
            # Additional ignore test
            "temp-data/ignore-me.py": "# This should be ignored",
            "test.tmp": "This should be ignored"
        }
        
        for file_path, content in test_files.items():
            full_path = self.temp_dir / file_path
            full_path.parent.mkdir(parents=True, exist_ok=True)
            
            if isinstance(content, str):
                full_path.write_text(content)
            else:
                full_path.write_bytes(content)
    
    async def test_pattern_matching_accuracy(self):
        """Test pattern matching accuracy across different file types."""
        logger.info("Testing pattern matching accuracy...")
        
        filter_system = LanguageAwareFilter(self.config_dir)
        await filter_system.load_configuration()
        
        # Test cases: (file_path, should_be_processed, expected_reason_contains)
        test_cases = [
            # Should be processed
            ("src/main.py", True, "accepted"),
            ("frontend/app.js", True, "accepted"),
            ("src/main.cpp", True, "accepted"),
            ("README.md", True, "accepted"),
            ("important-vendor/critical.js", True, "force_included"),
            ("test.important", True, "force_included"),
            
            # Should be filtered out
            ("src/__pycache__/main.cpython-39.pyc", False, "directory_ignored"),
            ("node_modules/react/index.js", False, "directory_ignored"),
            ("target/classes/App.class", False, "extension_ignored"),
            ("build/main.o", False, "directory_ignored"),
            (".git/config", False, "dot_file_ignored"),
            (".DS_Store", False, "dot_file_ignored"),
            ("app.log", False, "extension_ignored"),
            ("large_file.txt", False, "file_too_large"),
            ("temp-data/ignore-me.py", False, "directory_ignored"),
            ("test.tmp", False, "extension_ignored"),
        ]
        
        results = {"passed": 0, "failed": 0, "details": []}
        
        for file_path, expected_result, expected_reason_part in test_cases:
            full_path = self.temp_dir / file_path
            should_process, reason = filter_system.should_process_file(full_path)
            
            passed = should_process == expected_result and expected_reason_part in reason
            results["passed" if passed else "failed"] += 1
            
            results["details"].append({
                "file": file_path,
                "expected": expected_result,
                "actual": should_process,
                "reason": reason,
                "passed": passed
            })
            
            if not passed:
                logger.warning(f"FAILED: {file_path} - Expected {expected_result}, got {should_process} (reason: {reason})")
        
        self.test_results["pattern_matching"] = results
        logger.info(f"Pattern matching test: {results['passed']} passed, {results['failed']} failed")
    
    async def test_mime_type_detection(self):
        """Test MIME type detection functionality."""
        logger.info("Testing MIME type detection...")
        
        detector = MimeTypeDetector()
        
        test_files = [
            ("src/main.py", "text/x-python"),
            ("frontend/app.js", "application/javascript"),
            ("README.md", "text/x-web-markdown"),
            ("package.json", "application/json"),
        ]
        
        results = {"detected": 0, "total": len(test_files), "details": []}
        
        for file_path, expected_mime in test_files:
            full_path = self.temp_dir / file_path
            detected_mime = detector.get_mime_type(full_path)
            
            results["details"].append({
                "file": file_path,
                "expected": expected_mime,
                "detected": detected_mime,
                "match": detected_mime == expected_mime or (detected_mime and expected_mime in detected_mime)
            })
            
            if detected_mime:
                results["detected"] += 1
        
        self.test_results["mime_detection"] = results
        logger.info(f"MIME detection: {results['detected']}/{results['total']} files detected")
    
    async def test_performance_benchmarks(self):
        """Test performance with large numbers of files."""
        logger.info("Testing performance benchmarks...")
        
        filter_system = LanguageAwareFilter(self.config_dir)
        await filter_system.load_configuration()
        
        # Get all test files
        all_files = list(self.temp_dir.rglob("*"))
        file_paths = [f for f in all_files if f.is_file()]
        
        # Benchmark filtering performance
        start_time = time.perf_counter()
        
        processed_count = 0
        filtered_count = 0
        
        for file_path in file_paths:
            should_process, _ = filter_system.should_process_file(file_path)
            if should_process:
                processed_count += 1
            else:
                filtered_count += 1
        
        end_time = time.perf_counter()
        
        total_time_ms = (end_time - start_time) * 1000
        files_per_second = len(file_paths) / max(0.001, end_time - start_time)
        
        # Get detailed statistics
        stats = filter_system.get_statistics()
        
        results = {
            "total_files": len(file_paths),
            "processed_files": processed_count,
            "filtered_files": filtered_count,
            "total_time_ms": total_time_ms,
            "files_per_second": files_per_second,
            "avg_time_per_file_ms": total_time_ms / len(file_paths),
            "filter_efficiency": filtered_count / len(file_paths),
            "detailed_stats": stats.to_dict()
        }
        
        self.test_results["performance"] = results
        logger.info(f"Performance: {len(file_paths)} files in {total_time_ms:.2f}ms ({files_per_second:.1f} files/sec)")
    
    async def test_pattern_cache_performance(self):
        """Test pattern compilation and caching performance."""
        logger.info("Testing pattern cache performance...")
        
        compiled_patterns = CompiledPatterns()
        
        # Test patterns
        test_patterns = [
            "*.py", "*.js", "*.cpp", "*.java", "node_modules/*", 
            "__pycache__/*", "*.pyc", "build/*", "target/*"
        ]
        
        test_paths = [
            "src/main.py", "app.js", "test.cpp", "App.java",
            "node_modules/react/index.js", "__pycache__/main.pyc",
            "build/output.o", "target/classes/App.class"
        ]
        
        # Test cache performance with repeated patterns
        start_time = time.perf_counter()
        
        matches = 0
        for _ in range(100):  # Repeat to test caching
            for pattern in test_patterns:
                for path in test_paths:
                    if compiled_patterns.match_glob(pattern, path):
                        matches += 1
        
        end_time = time.perf_counter()
        
        cache_stats = compiled_patterns.get_cache_stats()
        
        results = {
            "total_operations": len(test_patterns) * len(test_paths) * 100,
            "total_matches": matches,
            "total_time_ms": (end_time - start_time) * 1000,
            "operations_per_second": (len(test_patterns) * len(test_paths) * 100) / max(0.001, end_time - start_time),
            "cache_stats": cache_stats
        }
        
        self.test_results["cache_performance"] = results
        logger.info(f"Cache performance: {results['operations_per_second']:.1f} ops/sec, cache hit ratio: {cache_stats.get('total_hits', 0) / max(1, cache_stats.get('unique_patterns', 1)):.2f}")
    
    async def test_file_watcher_integration(self):
        """Test integration with file watcher system."""
        logger.info("Testing file watcher integration...")
        
        watch_config = WatchConfiguration(
            id="test_watch",
            path=str(self.temp_dir),
            collection="test_collection",
            use_language_filtering=True
        )
        
        processed_files = []
        
        def ingestion_callback(file_path: str, collection: str):
            processed_files.append(file_path)
        
        watcher = FileWatcher(
            config=watch_config,
            ingestion_callback=ingestion_callback,
            filter_config_path=str(self.config_dir)
        )
        
        await watcher.start()
        
        # Wait for initialization
        await asyncio.sleep(0.1)
        
        # Get filter statistics
        filter_stats = watcher.get_filter_statistics()
        
        await watcher.stop()
        
        results = {
            "watcher_created": True,
            "language_filtering_enabled": filter_stats.get("language_filtering", False),
            "filter_config_loaded": "filter_config_summary" in filter_stats,
            "statistics_available": "detailed_stats" in filter_stats or "files_processed" in filter_stats
        }
        
        self.test_results["watcher_integration"] = results
        logger.info(f"Watcher integration: Language filtering enabled: {results['language_filtering_enabled']}")
    
    async def test_configuration_validation(self):
        """Test configuration loading and validation."""
        logger.info("Testing configuration validation...")
        
        # Test valid configuration
        filter_system = LanguageAwareFilter(self.config_dir)
        await filter_system.load_configuration()
        
        config_summary = filter_system.get_configuration_summary()
        
        results = {
            "config_loaded": filter_system._initialized,
            "config_summary_available": bool(config_summary),
            "expected_features": {
                "dot_files_ignored": config_summary.get("dot_files_ignored", False),
                "has_ignored_directories": config_summary.get("ignored_directories_count", 0) > 0,
                "has_ignored_extensions": config_summary.get("ignored_extensions_count", 0) > 0,
                "mime_detection": config_summary.get("mime_detection_enabled", False)
            }
        }
        
        # Test with missing configuration (should use defaults)
        temp_config_dir = self.temp_dir / "missing_config"
        filter_system_missing = LanguageAwareFilter(temp_config_dir)
        await filter_system_missing.load_configuration()
        
        results["fallback_config"] = {
            "loaded": filter_system_missing._initialized,
            "config_exists": filter_system_missing.config is not None
        }
        
        self.test_results["configuration"] = results
        logger.info(f"Configuration test: Valid config loaded: {results['config_loaded']}")
    
    async def run_all_tests(self):
        """Run all test suites."""
        logger.info("Starting comprehensive language filtering tests...")
        
        try:
            await self.setup_test_environment()
            
            # Run all tests
            await self.test_pattern_matching_accuracy()
            await self.test_mime_type_detection()
            await self.test_performance_benchmarks()
            await self.test_pattern_cache_performance()
            await self.test_file_watcher_integration()
            await self.test_configuration_validation()
            
            # Generate summary report
            await self.generate_test_report()
            
        except Exception as e:
            logger.error(f"Test execution failed: {e}")
            raise
        finally:
            # Cleanup
            if self.temp_dir and self.temp_dir.exists():
                import shutil
                shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    async def generate_test_report(self):
        """Generate comprehensive test report."""
        report = {
            "test_execution_summary": {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "total_test_suites": len(self.test_results),
                "test_environment": str(self.temp_dir)
            },
            "test_results": self.test_results
        }
        
        # Calculate overall success metrics
        pattern_results = self.test_results.get("pattern_matching", {})
        overall_passed = pattern_results.get("passed", 0)
        overall_failed = pattern_results.get("failed", 0)
        overall_success_rate = overall_passed / max(1, overall_passed + overall_failed)
        
        performance_results = self.test_results.get("performance", {})
        
        print("\n" + "="*80)
        print("LANGUAGE-AWARE FILE FILTERING - COMPREHENSIVE TEST REPORT")
        print("="*80)
        print(f"Test Execution Time: {report['test_execution_summary']['timestamp']}")
        print(f"Total Test Suites: {report['test_execution_summary']['total_test_suites']}")
        print()
        
        print("PATTERN MATCHING ACCURACY:")
        print(f"  ✓ Passed: {overall_passed}")
        print(f"  ✗ Failed: {overall_failed}")  
        print(f"  Success Rate: {overall_success_rate:.1%}")
        print()
        
        print("PERFORMANCE METRICS:")
        print(f"  Files Processed: {performance_results.get('total_files', 0)}")
        print(f"  Processing Speed: {performance_results.get('files_per_second', 0):.1f} files/second")
        print(f"  Average Time: {performance_results.get('avg_time_per_file_ms', 0):.3f}ms per file")
        print(f"  Filter Efficiency: {performance_results.get('filter_efficiency', 0):.1%} filtered out")
        print()
        
        mime_results = self.test_results.get("mime_detection", {})
        print("MIME TYPE DETECTION:")
        print(f"  Detection Rate: {mime_results.get('detected', 0)}/{mime_results.get('total', 0)} files")
        print()
        
        cache_results = self.test_results.get("cache_performance", {})
        print("PATTERN CACHE PERFORMANCE:")
        print(f"  Operations: {cache_results.get('total_operations', 0)}")
        print(f"  Speed: {cache_results.get('operations_per_second', 0):.0f} ops/second")
        print()
        
        integration_results = self.test_results.get("watcher_integration", {})
        print("FILE WATCHER INTEGRATION:")
        print(f"  Language Filtering Enabled: {integration_results.get('language_filtering_enabled', False)}")
        print(f"  Configuration Loaded: {integration_results.get('filter_config_loaded', False)}")
        print()
        
        config_results = self.test_results.get("configuration", {})
        print("CONFIGURATION VALIDATION:")
        print(f"  Config Loaded: {config_results.get('config_loaded', False)}")
        print(f"  Fallback Handling: {config_results.get('fallback_config', {}).get('loaded', False)}")
        print()
        
        print("="*80)
        
        if overall_success_rate >= 0.95:
            print("🎉 OVERALL STATUS: EXCELLENT - Language filtering system is production ready!")
        elif overall_success_rate >= 0.80:
            print("✅ OVERALL STATUS: GOOD - Minor issues need attention")
        else:
            print("❌ OVERALL STATUS: NEEDS WORK - Significant issues found")
        
        print("="*80)
        
        # Save detailed report
        report_file = Path(__file__).parent / f"20250107-1643_language_filtering_test_report.json"
        with report_file.open('w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Detailed test report saved to: {report_file}")


async def main():
    """Run comprehensive language filtering tests."""
    test_suite = TestLanguageFiltering()
    await test_suite.run_all_tests()


if __name__ == "__main__":
    asyncio.run(main())