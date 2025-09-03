"""
Test configuration and fixtures for SQLite State Manager comprehensive testing.

This file provides shared test fixtures, configuration, and utilities
for running the comprehensive SQLite state management test suite.
"""

import pytest
import tempfile
import os
import shutil
import asyncio
import logging
from pathlib import Path


# Configure test logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Configure pytest markers
pytest_plugins = []


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "performance: marks tests as performance tests"
    )
    config.addinivalue_line(
        "markers", "crash_recovery: marks tests as crash recovery tests"
    )
    config.addinivalue_line(
        "markers", "concurrent: marks tests as concurrent access tests"
    )
    config.addinivalue_line(
        "markers", "acid: marks tests as ACID transaction tests"
    )
    config.addinivalue_line(
        "markers", "maintenance: marks tests as database maintenance tests"
    )
    config.addinivalue_line(
        "markers", "error_scenarios: marks tests as error scenario tests"
    )


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
def temp_test_dir():
    """Create a temporary directory for the entire test session."""
    temp_dir = tempfile.mkdtemp(prefix="sqlite_test_")
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
async def clean_temp_db():
    """Create a clean temporary database file for each test."""
    fd, path = tempfile.mkstemp(suffix='.db', prefix="test_")
    os.close(fd)
    
    yield path
    
    # Cleanup all SQLite files
    for ext in ['', '-wal', '-shm']:
        try:
            os.unlink(path + ext)
        except FileNotFoundError:
            pass


@pytest.fixture
def mock_large_dataset():
    """Create mock data for large dataset testing."""
    def create_records(count: int = 1000):
        """Generate test records."""
        import time
        from tests.test_sqlite_state_manager_comprehensive import StateRecord
        
        records = []
        for i in range(count):
            record = StateRecord(
                file_path=f"/mock/file_{i:04d}.txt",
                status=f"status_{i % 5}",
                last_modified=time.time(),
                checksum=f"mock_hash_{i}",
                metadata={"mock": True, "index": i},
                created_at=time.time(),
                updated_at=time.time()
            )
            records.append(record)
        return records
    
    return create_records


@pytest.fixture
def performance_monitor():
    """Monitor performance during tests."""
    import psutil
    import time
    
    class PerformanceMonitor:
        def __init__(self):
            self.process = psutil.Process()
            self.start_time = None
            self.start_memory = None
            self.start_cpu = None
            
        def start(self):
            """Start monitoring."""
            self.start_time = time.perf_counter()
            self.start_memory = self.process.memory_info().rss / 1024 / 1024  # MB
            self.start_cpu = self.process.cpu_percent()
            
        def stop(self):
            """Stop monitoring and return stats."""
            if self.start_time is None:
                return {}
                
            duration = time.perf_counter() - self.start_time
            memory_usage = self.process.memory_info().rss / 1024 / 1024  # MB
            memory_delta = memory_usage - self.start_memory
            cpu_usage = self.process.cpu_percent()
            
            return {
                'duration_seconds': duration,
                'memory_start_mb': self.start_memory,
                'memory_end_mb': memory_usage,
                'memory_delta_mb': memory_delta,
                'cpu_usage_percent': cpu_usage
            }
    
    return PerformanceMonitor()


# Test collection hooks
def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers based on test names."""
    for item in items:
        # Add markers based on test name patterns
        if "performance" in item.name:
            item.add_marker(pytest.mark.performance)
            item.add_marker(pytest.mark.slow)
        
        if "crash_recovery" in item.name or "crash" in item.name:
            item.add_marker(pytest.mark.crash_recovery)
            item.add_marker(pytest.mark.slow)
        
        if "concurrent" in item.name:
            item.add_marker(pytest.mark.concurrent)
            item.add_marker(pytest.mark.slow)
        
        if "transaction" in item.name:
            item.add_marker(pytest.mark.acid)
        
        if "vacuum" in item.name or "analyze" in item.name or "wal" in item.name:
            item.add_marker(pytest.mark.maintenance)
        
        if "disk_full" in item.name or "corruption" in item.name or "error" in item.name:
            item.add_marker(pytest.mark.error_scenarios)


# Custom test result reporting
def pytest_runtest_call(item):
    """Called to execute the test item."""
    # Add any pre-test setup here
    pass


def pytest_runtest_teardown(item, nextitem):
    """Called after test execution."""
    # Add any post-test cleanup here
    pass


# Skip tests based on conditions
def pytest_runtest_setup(item):
    """Called before each test runs."""
    # Skip performance tests if explicitly requested
    if item.config.getoption("--skip-performance", False):
        if "performance" in item.keywords:
            pytest.skip("Performance tests skipped")
    
    # Skip slow tests if explicitly requested
    if item.config.getoption("--skip-slow", False):
        if "slow" in item.keywords:
            pytest.skip("Slow tests skipped")


def pytest_addoption(parser):
    """Add custom command line options."""
    parser.addoption(
        "--skip-performance",
        action="store_true",
        default=False,
        help="Skip performance tests"
    )
    parser.addoption(
        "--skip-slow", 
        action="store_true",
        default=False,
        help="Skip slow tests"
    )
    parser.addoption(
        "--db-engine",
        action="store",
        default="sqlite",
        help="Database engine to test (default: sqlite)"
    )


# Test environment setup
@pytest.fixture(autouse=True)
def setup_test_environment(monkeypatch):
    """Setup test environment for each test."""
    # Set test environment variables
    monkeypatch.setenv("TESTING", "1")
    monkeypatch.setenv("LOG_LEVEL", "INFO")
    
    # Ensure tests don't interfere with production data
    monkeypatch.setenv("DB_PATH", ":memory:")
    
    yield


# Database connection fixtures
@pytest.fixture
def sqlite_memory_db():
    """Create an in-memory SQLite database."""
    import sqlite3
    conn = sqlite3.connect(":memory:")
    yield conn
    conn.close()


@pytest.fixture
async def isolated_state_manager(clean_temp_db):
    """Create an isolated state manager instance for testing."""
    from tests.test_sqlite_state_manager_comprehensive import SQLiteStateManagerComprehensive
    
    manager = SQLiteStateManagerComprehensive(clean_temp_db, enable_wal=True)
    await manager.initialize()
    
    yield manager
    
    await manager.close()


# Test data fixtures
@pytest.fixture
def sample_file_states():
    """Generate sample file state records."""
    import time
    from tests.test_sqlite_state_manager_comprehensive import StateRecord
    
    states = []
    statuses = ["pending", "processing", "completed", "failed", "retrying"]
    
    for i in range(20):
        state = StateRecord(
            file_path=f"/sample/file_{i:03d}.txt",
            status=statuses[i % len(statuses)],
            last_modified=time.time() - (i * 3600),  # Spread over time
            checksum=f"sample_hash_{i}",
            metadata={
                "size": 1024 * (i + 1),
                "type": "test_sample",
                "index": i
            },
            created_at=time.time() - (i * 3600),
            updated_at=time.time(),
            retry_count=0 if i % 5 != 3 else 1  # Some files have retries
        )
        states.append(state)
    
    return states


# Async test utilities
class AsyncTestHelper:
    """Helper class for async test operations."""
    
    @staticmethod
    async def wait_for_condition(condition_func, timeout=5.0, interval=0.1):
        """Wait for a condition to become true."""
        import asyncio
        
        elapsed = 0
        while elapsed < timeout:
            if await condition_func():
                return True
            await asyncio.sleep(interval)
            elapsed += interval
        return False
    
    @staticmethod
    async def run_concurrent_tasks(tasks, max_concurrent=10):
        """Run tasks concurrently with limited concurrency."""
        import asyncio
        
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def run_task(task):
            async with semaphore:
                return await task
        
        return await asyncio.gather(*[run_task(task) for task in tasks])


@pytest.fixture
def async_helper():
    """Provide async test helper utilities."""
    return AsyncTestHelper()


# Test result collection
test_results = []

def pytest_runtest_logreport(report):
    """Collect test results for analysis."""
    if report.when == "call":
        test_results.append({
            'nodeid': report.nodeid,
            'outcome': report.outcome,
            'duration': report.duration,
            'keywords': list(report.keywords.keys())
        })


def pytest_sessionfinish(session, exitstatus):
    """Called after the entire test session."""
    # Print summary statistics
    if test_results:
        total_tests = len(test_results)
        passed = len([r for r in test_results if r['outcome'] == 'passed'])
        failed = len([r for r in test_results if r['outcome'] == 'failed'])
        skipped = len([r for r in test_results if r['outcome'] == 'skipped'])
        
        print(f"\n=== SQLite State Manager Test Summary ===")
        print(f"Total tests: {total_tests}")
        print(f"Passed: {passed}")
        print(f"Failed: {failed}")
        print(f"Skipped: {skipped}")
        print(f"Success rate: {passed/total_tests*100:.1f}%" if total_tests > 0 else "No tests run")
        
        # Performance test summary
        perf_tests = [r for r in test_results if 'performance' in r['keywords']]
        if perf_tests:
            avg_duration = sum(r['duration'] for r in perf_tests) / len(perf_tests)
            print(f"Performance tests: {len(perf_tests)}, Avg duration: {avg_duration:.2f}s")


# Error handling fixtures
@pytest.fixture
def error_handler():
    """Provide error handling utilities for tests."""
    
    class TestErrorHandler:
        def __init__(self):
            self.errors = []
        
        def capture_error(self, error, context=""):
            """Capture an error for later analysis."""
            self.errors.append({
                'error': error,
                'context': context,
                'timestamp': time.time()
            })
        
        def get_errors(self):
            """Get all captured errors."""
            return self.errors.copy()
        
        def clear_errors(self):
            """Clear captured errors."""
            self.errors.clear()
    
    import time
    return TestErrorHandler()


# Resource monitoring
@pytest.fixture
def resource_monitor():
    """Monitor system resources during tests."""
    
    class ResourceMonitor:
        def __init__(self):
            self.snapshots = []
        
        def snapshot(self, label=""):
            """Take a resource usage snapshot."""
            import psutil
            import time
            
            process = psutil.Process()
            memory_info = process.memory_info()
            
            snapshot = {
                'label': label,
                'timestamp': time.time(),
                'memory_rss_mb': memory_info.rss / 1024 / 1024,
                'memory_vms_mb': memory_info.vms / 1024 / 1024,
                'cpu_percent': process.cpu_percent(),
                'open_files': len(process.open_files()),
                'threads': process.num_threads()
            }
            
            self.snapshots.append(snapshot)
            return snapshot
        
        def get_snapshots(self):
            """Get all resource snapshots."""
            return self.snapshots.copy()
        
        def get_memory_delta(self, start_label="start", end_label="end"):
            """Get memory usage delta between two snapshots."""
            start_snapshot = next((s for s in self.snapshots if s['label'] == start_label), None)
            end_snapshot = next((s for s in self.snapshots if s['label'] == end_label), None)
            
            if start_snapshot and end_snapshot:
                return end_snapshot['memory_rss_mb'] - start_snapshot['memory_rss_mb']
            return None
    
    return ResourceMonitor()