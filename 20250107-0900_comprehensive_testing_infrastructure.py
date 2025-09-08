#!/usr/bin/env python3
"""
Comprehensive Testing Infrastructure Framework for workspace-qdrant-mcp
===================================================================

Task #151: Setup comprehensive testing infrastructure and environment

This framework provides:
1. Testing infrastructure with monitoring and logging
2. Test data generation capabilities
3. Validation frameworks for all testing phases
4. Test environment management and cleanup systems
5. Performance monitoring and metrics collection
6. Emergency shutdown and safety procedures

Supports comprehensive testing campaign (Tasks 151-170) including:
- LSP integration testing
- Ingestion capabilities testing
- Retrieval accuracy testing
- Automation testing

Usage:
    python 20250107-0900_comprehensive_testing_infrastructure.py --setup
    python 20250107-0900_comprehensive_testing_infrastructure.py --run-basic-tests
    python 20250107-0900_comprehensive_testing_infrastructure.py --cleanup
"""

import asyncio
import json
import logging
import os
import sys
import time
import tempfile
import traceback
import threading
import psutil
import shutil
import signal
from pathlib import Path
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import subprocess

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    from workspace_qdrant_mcp.core.client import QdrantWorkspaceClient
    from workspace_qdrant_mcp.core.config import QdrantConfig
    from workspace_qdrant_mcp.tools.scratchbook import ScratchbookManager
    from workspace_qdrant_mcp.tools.search import search_workspace
except ImportError as e:
    print(f"⚠️  Import warning: {e}")
    print("Some tests may be limited without full package imports")

class TestPhase(Enum):
    """Test phases for the comprehensive testing campaign."""
    SETUP = "setup"
    INFRASTRUCTURE = "infrastructure"
    DATA_GENERATION = "data_generation"
    LSP_INTEGRATION = "lsp_integration"
    INGESTION_CAPABILITIES = "ingestion_capabilities"
    RETRIEVAL_ACCURACY = "retrieval_accuracy"
    AUTOMATION = "automation"
    PERFORMANCE = "performance"
    STRESS = "stress"
    CLEANUP = "cleanup"

class TestStatus(Enum):
    """Test execution status."""
    PENDING = "pending"
    RUNNING = "running"
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    ERROR = "error"

@dataclass
class TestMetrics:
    """Comprehensive test metrics."""
    start_time: float
    end_time: Optional[float] = None
    cpu_usage: List[float] = field(default_factory=list)
    memory_usage: List[float] = field(default_factory=list)
    disk_usage: List[float] = field(default_factory=list)
    network_io: List[Dict] = field(default_factory=list)
    custom_metrics: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def duration(self) -> float:
        if self.end_time:
            return self.end_time - self.start_time
        return time.time() - self.start_time
    
    @property
    def avg_cpu_usage(self) -> float:
        return sum(self.cpu_usage) / len(self.cpu_usage) if self.cpu_usage else 0.0
    
    @property
    def peak_memory_usage(self) -> float:
        return max(self.memory_usage) if self.memory_usage else 0.0

@dataclass
class TestResult:
    """Individual test result."""
    test_id: str
    phase: TestPhase
    status: TestStatus
    metrics: TestMetrics
    details: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    warnings: List[str] = field(default_factory=list)

class SafetyMonitor:
    """Emergency safety monitoring system."""
    
    def __init__(self, max_cpu_percent: float = 90.0, max_memory_percent: float = 85.0, 
                 max_disk_percent: float = 95.0, check_interval: float = 1.0):
        self.max_cpu_percent = max_cpu_percent
        self.max_memory_percent = max_memory_percent
        self.max_disk_percent = max_disk_percent
        self.check_interval = check_interval
        self.monitoring = False
        self.emergency_shutdown_callbacks: List[Callable] = []
        self.monitor_thread: Optional[threading.Thread] = None
        self.logger = logging.getLogger(f"{__name__}.SafetyMonitor")
        
    def add_emergency_callback(self, callback: Callable):
        """Add emergency shutdown callback."""
        self.emergency_shutdown_callbacks.append(callback)
        
    def start_monitoring(self):
        """Start safety monitoring in background thread."""
        if self.monitoring:
            return
            
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()
        self.logger.info("Safety monitoring started")
        
    def stop_monitoring(self):
        """Stop safety monitoring."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2.0)
        self.logger.info("Safety monitoring stopped")
        
    def _monitoring_loop(self):
        """Main monitoring loop."""
        while self.monitoring:
            try:
                # Check system resources
                cpu_percent = psutil.cpu_percent(interval=0.1)
                memory_percent = psutil.virtual_memory().percent
                disk_percent = psutil.disk_usage('/').percent
                
                # Check thresholds
                if cpu_percent > self.max_cpu_percent:
                    self.logger.critical(f"CPU usage exceeded threshold: {cpu_percent:.1f}% > {self.max_cpu_percent}%")
                    self._emergency_shutdown("CPU usage exceeded threshold")
                    return
                    
                if memory_percent > self.max_memory_percent:
                    self.logger.critical(f"Memory usage exceeded threshold: {memory_percent:.1f}% > {self.max_memory_percent}%")
                    self._emergency_shutdown("Memory usage exceeded threshold")
                    return
                    
                if disk_percent > self.max_disk_percent:
                    self.logger.critical(f"Disk usage exceeded threshold: {disk_percent:.1f}% > {self.max_disk_percent}%")
                    self._emergency_shutdown("Disk usage exceeded threshold")
                    return
                    
                time.sleep(self.check_interval)
                
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                time.sleep(self.check_interval)
                
    def _emergency_shutdown(self, reason: str):
        """Execute emergency shutdown procedures."""
        self.logger.critical(f"EMERGENCY SHUTDOWN TRIGGERED: {reason}")
        
        for callback in self.emergency_shutdown_callbacks:
            try:
                callback()
            except Exception as e:
                self.logger.error(f"Error in emergency callback: {e}")
                
        # Stop monitoring
        self.monitoring = False

class TestDataGenerator:
    """Generate realistic test data for comprehensive testing."""
    
    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.logger = logging.getLogger(f"{__name__}.TestDataGenerator")
        
    def generate_source_code_files(self, count: int = 50, languages: List[str] = None) -> List[Path]:
        """Generate realistic source code files."""
        if languages is None:
            languages = ['python', 'javascript', 'typescript', 'markdown', 'json']
            
        files = []
        templates = self._get_code_templates()
        
        for i in range(count):
            language = languages[i % len(languages)]
            template = templates.get(language, templates['python'])
            
            filename = f"test_file_{i:03d}.{self._get_extension(language)}"
            filepath = self.output_dir / "source_code" / filename
            filepath.parent.mkdir(parents=True, exist_ok=True)
            
            content = template.format(
                index=i,
                timestamp=datetime.now().isoformat(),
                language=language
            )
            
            filepath.write_text(content, encoding='utf-8')
            files.append(filepath)
            
        self.logger.info(f"Generated {len(files)} source code files")
        return files
        
    def generate_documentation_files(self, count: int = 20) -> List[Path]:
        """Generate documentation files with various content types."""
        files = []
        
        for i in range(count):
            filename = f"documentation_{i:03d}.md"
            filepath = self.output_dir / "documentation" / filename
            filepath.parent.mkdir(parents=True, exist_ok=True)
            
            content = self._generate_markdown_content(i)
            filepath.write_text(content, encoding='utf-8')
            files.append(filepath)
            
        self.logger.info(f"Generated {len(files)} documentation files")
        return files
        
    def generate_configuration_files(self, count: int = 10) -> List[Path]:
        """Generate configuration files in various formats."""
        files = []
        formats = ['json', 'yaml', 'toml', 'ini']
        
        for i in range(count):
            format_type = formats[i % len(formats)]
            filename = f"config_{i:03d}.{format_type}"
            filepath = self.output_dir / "config" / filename
            filepath.parent.mkdir(parents=True, exist_ok=True)
            
            content = self._generate_config_content(i, format_type)
            filepath.write_text(content, encoding='utf-8')
            files.append(filepath)
            
        self.logger.info(f"Generated {len(files)} configuration files")
        return files
        
    def generate_large_files(self, count: int = 5, size_mb: int = 10) -> List[Path]:
        """Generate large files for stress testing."""
        files = []
        
        for i in range(count):
            filename = f"large_file_{i:03d}.txt"
            filepath = self.output_dir / "large_files" / filename
            filepath.parent.mkdir(parents=True, exist_ok=True)
            
            # Generate content with specified size
            line_content = f"Large file line {i} - " + "x" * 100 + "\n"
            lines_needed = (size_mb * 1024 * 1024) // len(line_content.encode('utf-8'))
            
            with open(filepath, 'w', encoding='utf-8') as f:
                for line_num in range(lines_needed):
                    f.write(line_content.replace(f"line {i}", f"line {line_num}"))
                    
            files.append(filepath)
            
        self.logger.info(f"Generated {len(files)} large files ({size_mb}MB each)")
        return files
        
    def _get_code_templates(self) -> Dict[str, str]:
        """Get code templates for different languages."""
        return {
            'python': '''#!/usr/bin/env python3
"""
Test file {index} - {language}
Generated at: {timestamp}
"""

import asyncio
import json
from typing import Dict, List, Optional

class TestClass{index}:
    """Test class for comprehensive testing."""
    
    def __init__(self, name: str = "test_{index}"):
        self.name = name
        self.data = {{"index": {index}, "timestamp": "{timestamp}"}}
        
    async def process_data(self, input_data: Dict) -> Dict:
        """Process input data and return results."""
        result = {{
            "processed": True,
            "input_size": len(str(input_data)),
            "output_timestamp": "{timestamp}",
            "class_name": self.__class__.__name__
        }}
        
        # Simulate some async processing
        await asyncio.sleep(0.001)
        
        return result
        
    def get_info(self) -> str:
        return f"TestClass{index}: {{self.name}}"

if __name__ == "__main__":
    test = TestClass{index}()
    print(test.get_info())
''',
            'javascript': '''/**
 * Test file {index} - {language}
 * Generated at: {timestamp}
 */

const TestClass{index} = {{
    name: "test_{index}",
    data: {{
        index: {index},
        timestamp: "{timestamp}"
    }},
    
    processData: async function(inputData) {{
        const result = {{
            processed: true,
            inputSize: JSON.stringify(inputData).length,
            outputTimestamp: "{timestamp}",
            className: "TestClass{index}"
        }};
        
        // Simulate async processing
        await new Promise(resolve => setTimeout(resolve, 1));
        
        return result;
    }},
    
    getInfo: function() {{
        return `TestClass{index}: ${{this.name}}`;
    }}
}};

// Export for testing
if (typeof module !== 'undefined' && module.exports) {{
    module.exports = TestClass{index};
}}

console.log(TestClass{index}.getInfo());
''',
            'markdown': '''# Test Document {index}

Generated at: {timestamp}

## Overview

This is test document {index} for comprehensive testing of the workspace-qdrant-mcp system.

### Features Tested

- Document indexing and search
- Content extraction and processing  
- Metadata handling
- Cross-reference resolution

### Code Examples

```{language}
function testFunction{index}() {{
    return {{
        index: {index},
        timestamp: "{timestamp}",
        status: "active"
    }};
}}
```

### Search Keywords

Testing keywords: search, indexing, retrieval, accuracy, performance, {language}, test{index}

## Performance Notes

This document is designed to test:
- Semantic search capabilities
- Keyword search accuracy
- Hybrid search performance
- Content ranking algorithms

### Related Documents

See also: test_file_{{(index + 1) % 10}}.md for related content.
''',
            'json': '''{{
    "test_file_id": {index},
    "language": "{language}",
    "timestamp": "{timestamp}",
    "metadata": {{
        "version": "1.0",
        "type": "test_configuration",
        "environment": "comprehensive_testing"
    }},
    "test_configuration": {{
        "search_tests": {{
            "semantic_search": true,
            "keyword_search": true,
            "hybrid_search": true,
            "fuzzy_search": false
        }},
        "performance_tests": {{
            "indexing_speed": true,
            "query_latency": true,
            "memory_usage": true,
            "concurrent_queries": 10
        }},
        "data_validation": {{
            "schema_validation": true,
            "content_integrity": true,
            "encoding_verification": "utf-8"
        }}
    }},
    "test_data": [
        {{"id": {index}, "value": "test_value_{index}", "active": true}},
        {{"id": {index + 1000}, "value": "nested_test_value", "nested": {{"level": 2, "data": "deep_content"}}}},
        {{"id": {index + 2000}, "value": "search_test_content", "keywords": ["search", "test", "content", "indexing"]}}
    ]
}}'''
        }
        
    def _get_extension(self, language: str) -> str:
        """Get file extension for language."""
        extensions = {
            'python': 'py',
            'javascript': 'js', 
            'typescript': 'ts',
            'markdown': 'md',
            'json': 'json'
        }
        return extensions.get(language, 'txt')
        
    def _generate_markdown_content(self, index: int) -> str:
        """Generate markdown documentation content."""
        return f'''# Documentation File {index:03d}

Generated at: {datetime.now().isoformat()}

## Purpose

This documentation file is part of the comprehensive testing infrastructure for workspace-qdrant-mcp.

### Testing Areas Covered

1. **Document Processing**
   - Markdown parsing and indexing
   - Metadata extraction
   - Cross-reference handling

2. **Search Functionality** 
   - Semantic search accuracy
   - Keyword matching
   - Hybrid search performance

3. **Content Analysis**
   - Natural language understanding
   - Technical content recognition
   - Code snippet processing

### Performance Requirements

- Indexing latency: < 100ms per document
- Search response time: < 50ms
- Memory efficiency: < 10MB per 1000 documents
- Concurrent user support: 100+ simultaneous queries

### Integration Points

This documentation integrates with:
- LSP server integration (Task 152-157)
- Ingestion capabilities (Task 158-162)  
- Retrieval accuracy testing (Task 163-167)
- Automation frameworks (Task 168-170)

### Code Examples

```python
# Example search implementation
async def search_documentation(query: str, limit: int = 10):
    results = await client.search(
        query=query,
        collections=["documentation"],
        mode="hybrid",
        limit=limit
    )
    return results
```

### See Also

- Related file: documentation_{(index + 1) % 20:03d}.md
- Code implementation: test_file_{index:03d}.py
- Configuration: config_{index:03d}.json
'''

    def _generate_config_content(self, index: int, format_type: str) -> str:
        """Generate configuration content in specified format."""
        if format_type == 'json':
            return json.dumps({
                "config_id": f"test_config_{index:03d}",
                "timestamp": datetime.now().isoformat(),
                "environment": "comprehensive_testing",
                "qdrant": {
                    "url": "http://localhost:6333",
                    "collection_prefix": f"test_{index:03d}",
                    "timeout": 30
                },
                "search": {
                    "default_limit": 10,
                    "score_threshold": 0.7,
                    "enable_hybrid": True
                },
                "monitoring": {
                    "enabled": True,
                    "metrics_interval": 1.0,
                    "log_level": "INFO"
                }
            }, indent=2)
        else:
            # Simple key=value format for other types
            return f'''# Configuration file {index:03d}
# Generated at: {datetime.now().isoformat()}

config_id=test_config_{index:03d}
environment=comprehensive_testing
qdrant_url=http://localhost:6333
collection_prefix=test_{index:03d}
search_limit=10
score_threshold=0.7
'''

class PerformanceMonitor:
    """Real-time performance monitoring during tests."""
    
    def __init__(self, collection_interval: float = 0.5):
        self.collection_interval = collection_interval
        self.monitoring = False
        self.monitor_thread: Optional[threading.Thread] = None
        self.metrics = TestMetrics(start_time=time.time())
        self.logger = logging.getLogger(f"{__name__}.PerformanceMonitor")
        
    def start_monitoring(self):
        """Start performance monitoring."""
        if self.monitoring:
            return
            
        self.monitoring = True
        self.metrics.start_time = time.time()
        self.monitor_thread = threading.Thread(target=self._collection_loop, daemon=True)
        self.monitor_thread.start()
        self.logger.info("Performance monitoring started")
        
    def stop_monitoring(self) -> TestMetrics:
        """Stop monitoring and return collected metrics."""
        self.monitoring = False
        self.metrics.end_time = time.time()
        
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2.0)
            
        self.logger.info(f"Performance monitoring stopped. Duration: {self.metrics.duration:.2f}s")
        return self.metrics
        
    def _collection_loop(self):
        """Collect performance metrics in background."""
        while self.monitoring:
            try:
                # Collect system metrics
                cpu_percent = psutil.cpu_percent(interval=None)
                memory = psutil.virtual_memory()
                disk = psutil.disk_usage('/')
                
                self.metrics.cpu_usage.append(cpu_percent)
                self.metrics.memory_usage.append(memory.percent)
                self.metrics.disk_usage.append(disk.percent)
                
                # Network I/O (if available)
                try:
                    network = psutil.net_io_counters()
                    self.metrics.network_io.append({
                        'bytes_sent': network.bytes_sent,
                        'bytes_recv': network.bytes_recv,
                        'packets_sent': network.packets_sent,
                        'packets_recv': network.packets_recv
                    })
                except Exception:
                    pass  # Network stats not available on all systems
                    
                time.sleep(self.collection_interval)
                
            except Exception as e:
                self.logger.error(f"Error collecting performance metrics: {e}")
                time.sleep(self.collection_interval)

class TestEnvironmentManager:
    """Manage test environments, data, and cleanup."""
    
    def __init__(self, base_dir: Optional[Path] = None):
        self.base_dir = base_dir or Path(tempfile.mkdtemp(prefix="wqmcp_comprehensive_test_"))
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        self.test_data_dir = self.base_dir / "test_data"
        self.results_dir = self.base_dir / "results"
        self.logs_dir = self.base_dir / "logs"
        
        # Create subdirectories
        for dir_path in [self.test_data_dir, self.results_dir, self.logs_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
            
        self.logger = logging.getLogger(f"{__name__}.TestEnvironmentManager")
        self.cleanup_callbacks: List[Callable] = []
        self.temp_files: List[Path] = []
        
        # Setup logging
        self._setup_logging()
        
        self.logger.info(f"Test environment initialized at: {self.base_dir}")
        
    def _setup_logging(self):
        """Setup comprehensive logging for test execution."""
        log_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
        )
        
        # File handler for all logs
        file_handler = logging.FileHandler(self.logs_dir / "comprehensive_tests.log")
        file_handler.setFormatter(log_formatter)
        file_handler.setLevel(logging.DEBUG)
        
        # File handler for errors only
        error_handler = logging.FileHandler(self.logs_dir / "errors.log")
        error_handler.setFormatter(log_formatter)
        error_handler.setLevel(logging.ERROR)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        ))
        console_handler.setLevel(logging.INFO)
        
        # Configure root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.DEBUG)
        root_logger.addHandler(file_handler)
        root_logger.addHandler(error_handler)
        root_logger.addHandler(console_handler)
        
    def add_cleanup_callback(self, callback: Callable):
        """Add cleanup callback to be executed during environment cleanup."""
        self.cleanup_callbacks.append(callback)
        
    def add_temp_file(self, filepath: Path):
        """Register a temporary file for cleanup."""
        self.temp_files.append(filepath)
        
    def generate_test_data(self, data_profile: str = "comprehensive") -> Dict[str, List[Path]]:
        """Generate test data based on profile."""
        self.logger.info(f"Generating test data with profile: {data_profile}")
        
        generator = TestDataGenerator(self.test_data_dir)
        generated_files = {}
        
        if data_profile == "comprehensive":
            generated_files["source_code"] = generator.generate_source_code_files(count=100)
            generated_files["documentation"] = generator.generate_documentation_files(count=50)
            generated_files["configuration"] = generator.generate_configuration_files(count=20)
            generated_files["large_files"] = generator.generate_large_files(count=10, size_mb=5)
            
        elif data_profile == "minimal":
            generated_files["source_code"] = generator.generate_source_code_files(count=20)
            generated_files["documentation"] = generator.generate_documentation_files(count=10)
            generated_files["configuration"] = generator.generate_configuration_files(count=5)
            
        elif data_profile == "stress":
            generated_files["source_code"] = generator.generate_source_code_files(count=500)
            generated_files["documentation"] = generator.generate_documentation_files(count=200)
            generated_files["configuration"] = generator.generate_configuration_files(count=50)
            generated_files["large_files"] = generator.generate_large_files(count=20, size_mb=25)
            
        # Register all generated files for cleanup
        for file_list in generated_files.values():
            for file_path in file_list:
                self.add_temp_file(file_path)
                
        total_files = sum(len(files) for files in generated_files.values())
        self.logger.info(f"Generated {total_files} test files in {len(generated_files)} categories")
        
        return generated_files
        
    def save_test_result(self, result: TestResult):
        """Save individual test result."""
        result_file = self.results_dir / f"{result.test_id}_{int(time.time())}.json"
        
        result_data = {
            "test_id": result.test_id,
            "phase": result.phase.value,
            "status": result.status.value,
            "duration": result.metrics.duration,
            "avg_cpu_usage": result.metrics.avg_cpu_usage,
            "peak_memory_usage": result.metrics.peak_memory_usage,
            "details": result.details,
            "error": result.error,
            "warnings": result.warnings,
            "timestamp": datetime.now().isoformat()
        }
        
        with open(result_file, 'w') as f:
            json.dump(result_data, f, indent=2)
            
        self.add_temp_file(result_file)
        
    def save_comprehensive_report(self, results: List[TestResult]) -> Path:
        """Generate and save comprehensive test report."""
        report_file = self.results_dir / f"comprehensive_test_report_{int(time.time())}.json"
        
        # Calculate summary statistics
        total_tests = len(results)
        passed_tests = sum(1 for r in results if r.status == TestStatus.PASSED)
        failed_tests = sum(1 for r in results if r.status == TestStatus.FAILED)
        error_tests = sum(1 for r in results if r.status == TestStatus.ERROR)
        
        # Performance statistics
        total_duration = sum(r.metrics.duration for r in results)
        avg_cpu = sum(r.metrics.avg_cpu_usage for r in results) / total_tests if total_tests > 0 else 0
        peak_memory = max((r.metrics.peak_memory_usage for r in results), default=0)
        
        # Group by phase
        phase_summary = {}
        for phase in TestPhase:
            phase_results = [r for r in results if r.phase == phase]
            if phase_results:
                phase_summary[phase.value] = {
                    "total": len(phase_results),
                    "passed": sum(1 for r in phase_results if r.status == TestStatus.PASSED),
                    "failed": sum(1 for r in phase_results if r.status == TestStatus.FAILED),
                    "errors": sum(1 for r in phase_results if r.status == TestStatus.ERROR),
                    "avg_duration": sum(r.metrics.duration for r in phase_results) / len(phase_results)
                }
        
        report_data = {
            "report_metadata": {
                "timestamp": datetime.now().isoformat(),
                "test_environment": str(self.base_dir),
                "total_duration": total_duration
            },
            "summary": {
                "total_tests": total_tests,
                "passed_tests": passed_tests,
                "failed_tests": failed_tests,
                "error_tests": error_tests,
                "success_rate": (passed_tests / total_tests * 100) if total_tests > 0 else 0
            },
            "performance": {
                "total_duration_seconds": total_duration,
                "average_cpu_usage_percent": avg_cpu,
                "peak_memory_usage_percent": peak_memory,
                "tests_per_second": total_tests / total_duration if total_duration > 0 else 0
            },
            "phase_summary": phase_summary,
            "detailed_results": [
                {
                    "test_id": r.test_id,
                    "phase": r.phase.value,
                    "status": r.status.value,
                    "duration": r.metrics.duration,
                    "error": r.error,
                    "warnings_count": len(r.warnings)
                }
                for r in results
            ]
        }
        
        with open(report_file, 'w') as f:
            json.dump(report_data, f, indent=2)
            
        self.logger.info(f"Comprehensive report saved to: {report_file}")
        return report_file
        
    def cleanup(self, force: bool = False):
        """Cleanup test environment."""
        self.logger.info("Starting test environment cleanup...")
        
        # Execute cleanup callbacks
        for callback in self.cleanup_callbacks:
            try:
                callback()
            except Exception as e:
                self.logger.error(f"Error in cleanup callback: {e}")
                
        # Remove temporary files
        files_removed = 0
        for temp_file in self.temp_files:
            try:
                if temp_file.exists():
                    if temp_file.is_file():
                        temp_file.unlink()
                    else:
                        shutil.rmtree(temp_file)
                    files_removed += 1
            except Exception as e:
                self.logger.error(f"Error removing temp file {temp_file}: {e}")
                
        # Optionally remove entire test directory
        if force and self.base_dir.exists():
            try:
                shutil.rmtree(self.base_dir)
                self.logger.info(f"Removed test environment directory: {self.base_dir}")
            except Exception as e:
                self.logger.error(f"Error removing test directory: {e}")
        else:
            self.logger.info(f"Test environment preserved at: {self.base_dir}")
            
        self.logger.info(f"Cleanup completed. Removed {files_removed} temporary files")

class ComprehensiveTestingInfrastructure:
    """Main testing infrastructure orchestrator."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or self._get_default_config()
        self.env_manager = TestEnvironmentManager()
        self.safety_monitor = SafetyMonitor()
        self.performance_monitor = PerformanceMonitor()
        self.test_results: List[TestResult] = []
        
        self.logger = logging.getLogger(f"{__name__}.ComprehensiveTestingInfrastructure")
        
        # Setup emergency procedures
        self.safety_monitor.add_emergency_callback(self._emergency_shutdown)
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for testing infrastructure."""
        return {
            "safety": {
                "max_cpu_percent": 90.0,
                "max_memory_percent": 85.0,
                "max_disk_percent": 95.0,
                "monitor_interval": 1.0
            },
            "performance": {
                "collection_interval": 0.5,
                "enable_network_monitoring": True
            },
            "data_generation": {
                "default_profile": "comprehensive",
                "source_files": 100,
                "doc_files": 50,
                "config_files": 20,
                "large_files": 10,
                "large_file_size_mb": 5
            },
            "test_execution": {
                "continue_on_failure": True,
                "max_test_duration": 300,  # 5 minutes per test
                "parallel_execution": False
            }
        }
        
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        self.logger.warning(f"Received signal {signum}, initiating graceful shutdown...")
        self._emergency_shutdown()
        
    def _emergency_shutdown(self):
        """Execute emergency shutdown procedures."""
        self.logger.critical("EMERGENCY SHUTDOWN INITIATED")
        
        try:
            # Stop monitoring
            self.safety_monitor.stop_monitoring()
            self.performance_monitor.stop_monitoring()
            
            # Save any partial results
            if self.test_results:
                self.env_manager.save_comprehensive_report(self.test_results)
                
            # Cleanup environment
            self.env_manager.cleanup()
            
        except Exception as e:
            self.logger.error(f"Error during emergency shutdown: {e}")
            
        self.logger.critical("Emergency shutdown completed")
        
    async def setup_infrastructure(self) -> bool:
        """Setup the complete testing infrastructure."""
        self.logger.info("Setting up comprehensive testing infrastructure...")
        
        try:
            # Start safety monitoring
            self.safety_monitor.start_monitoring()
            
            # Generate test data
            test_data = self.env_manager.generate_test_data(
                self.config["data_generation"]["default_profile"]
            )
            
            self.logger.info("Infrastructure setup completed successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Infrastructure setup failed: {e}")
            return False
            
    async def run_basic_infrastructure_tests(self) -> List[TestResult]:
        """Run basic infrastructure validation tests."""
        self.logger.info("Running basic infrastructure tests...")
        
        test_cases = [
            ("data_generation_test", self._test_data_generation),
            ("safety_monitoring_test", self._test_safety_monitoring),
            ("performance_monitoring_test", self._test_performance_monitoring),
            ("environment_management_test", self._test_environment_management),
            ("cleanup_procedures_test", self._test_cleanup_procedures)
        ]
        
        results = []
        for test_id, test_func in test_cases:
            result = await self._run_single_test(test_id, TestPhase.INFRASTRUCTURE, test_func)
            results.append(result)
            self.test_results.append(result)
            
        return results
        
    async def _run_single_test(self, test_id: str, phase: TestPhase, 
                             test_func: Callable) -> TestResult:
        """Run a single test with full monitoring and error handling."""
        self.logger.info(f"Starting test: {test_id}")
        
        # Start performance monitoring for this test
        perf_monitor = PerformanceMonitor()
        perf_monitor.start_monitoring()
        
        result = TestResult(
            test_id=test_id,
            phase=phase,
            status=TestStatus.RUNNING,
            metrics=TestMetrics(start_time=time.time())
        )
        
        try:
            # Execute test with timeout
            test_output = await asyncio.wait_for(
                test_func(),
                timeout=self.config["test_execution"]["max_test_duration"]
            )
            
            result.status = TestStatus.PASSED
            result.details = test_output or {}
            
            self.logger.info(f"Test {test_id} PASSED")
            
        except asyncio.TimeoutError:
            result.status = TestStatus.FAILED
            result.error = f"Test timed out after {self.config['test_execution']['max_test_duration']} seconds"
            self.logger.error(f"Test {test_id} TIMED OUT")
            
        except Exception as e:
            result.status = TestStatus.ERROR
            result.error = str(e)
            self.logger.error(f"Test {test_id} ERROR: {e}")
            
        finally:
            # Stop monitoring and collect metrics
            result.metrics = perf_monitor.stop_monitoring()
            
        # Save individual test result
        self.env_manager.save_test_result(result)
        
        return result
        
    async def _test_data_generation(self) -> Dict[str, Any]:
        """Test data generation capabilities."""
        generator = TestDataGenerator(self.env_manager.test_data_dir / "generation_test")
        
        # Test different file types
        source_files = generator.generate_source_code_files(count=10)
        doc_files = generator.generate_documentation_files(count=5)
        config_files = generator.generate_configuration_files(count=3)
        
        # Verify files were created and have content
        total_files = len(source_files) + len(doc_files) + len(config_files)
        total_size = sum(f.stat().st_size for f in source_files + doc_files + config_files if f.exists())
        
        return {
            "files_generated": total_files,
            "total_size_bytes": total_size,
            "source_files": len(source_files),
            "doc_files": len(doc_files),
            "config_files": len(config_files),
            "avg_file_size": total_size / total_files if total_files > 0 else 0
        }
        
    async def _test_safety_monitoring(self) -> Dict[str, Any]:
        """Test safety monitoring system."""
        # Create a temporary safety monitor for testing
        test_monitor = SafetyMonitor(
            max_cpu_percent=95.0,  # Higher threshold for testing
            max_memory_percent=90.0,
            check_interval=0.1
        )
        
        callback_triggered = False
        
        def test_callback():
            nonlocal callback_triggered
            callback_triggered = True
            
        test_monitor.add_emergency_callback(test_callback)
        
        # Start monitoring briefly
        test_monitor.start_monitoring()
        await asyncio.sleep(0.5)  # Let it collect some metrics
        test_monitor.stop_monitoring()
        
        return {
            "monitoring_functional": True,
            "callback_system_ready": callback_triggered is False,  # Should not trigger under normal conditions
            "max_cpu_threshold": test_monitor.max_cpu_percent,
            "max_memory_threshold": test_monitor.max_memory_percent
        }
        
    async def _test_performance_monitoring(self) -> Dict[str, Any]:
        """Test performance monitoring system."""
        monitor = PerformanceMonitor(collection_interval=0.1)
        
        monitor.start_monitoring()
        
        # Simulate some work
        await asyncio.sleep(0.5)
        
        metrics = monitor.stop_monitoring()
        
        return {
            "monitoring_duration": metrics.duration,
            "cpu_samples_collected": len(metrics.cpu_usage),
            "memory_samples_collected": len(metrics.memory_usage),
            "avg_cpu_usage": metrics.avg_cpu_usage,
            "peak_memory_usage": metrics.peak_memory_usage,
            "collection_working": len(metrics.cpu_usage) > 0
        }
        
    async def _test_environment_management(self) -> Dict[str, Any]:
        """Test environment management capabilities."""
        # Create a temporary environment for testing
        temp_env = TestEnvironmentManager()
        
        # Test directory structure
        expected_dirs = [temp_env.test_data_dir, temp_env.results_dir, temp_env.logs_dir]
        dirs_exist = all(d.exists() for d in expected_dirs)
        
        # Test file registration
        test_file = temp_env.test_data_dir / "temp_test.txt"
        test_file.write_text("test content")
        temp_env.add_temp_file(test_file)
        
        # Test cleanup callback
        callback_executed = False
        
        def test_cleanup():
            nonlocal callback_executed
            callback_executed = True
            
        temp_env.add_cleanup_callback(test_cleanup)
        
        # Execute cleanup
        temp_env.cleanup(force=True)
        
        return {
            "directories_created": dirs_exist,
            "file_registration_working": len(temp_env.temp_files) > 0,
            "cleanup_callbacks_working": callback_executed,
            "temp_file_removed": not test_file.exists()
        }
        
    async def _test_cleanup_procedures(self) -> Dict[str, Any]:
        """Test cleanup procedures."""
        # Create temporary files and directories
        cleanup_test_dir = self.env_manager.test_data_dir / "cleanup_test"
        cleanup_test_dir.mkdir(exist_ok=True)
        
        test_files = []
        for i in range(5):
            test_file = cleanup_test_dir / f"cleanup_test_{i}.txt"
            test_file.write_text(f"cleanup test content {i}")
            test_files.append(test_file)
            self.env_manager.add_temp_file(test_file)
            
        # Register cleanup callback
        callback_executed = False
        
        def cleanup_callback():
            nonlocal callback_executed
            callback_executed = True
            
        self.env_manager.add_cleanup_callback(cleanup_callback)
        
        # Test that files exist before cleanup
        files_before = sum(1 for f in test_files if f.exists())
        
        # Execute partial cleanup (files only, not force)
        self.env_manager.cleanup(force=False)
        
        # Check results
        files_after = sum(1 for f in test_files if f.exists())
        
        return {
            "files_before_cleanup": files_before,
            "files_after_cleanup": files_after,
            "callback_executed": callback_executed,
            "cleanup_effective": files_after < files_before
        }
        
    async def shutdown(self):
        """Graceful shutdown of testing infrastructure."""
        self.logger.info("Shutting down testing infrastructure...")
        
        # Stop monitoring
        self.safety_monitor.stop_monitoring()
        self.performance_monitor.stop_monitoring()
        
        # Generate final report
        if self.test_results:
            report_file = self.env_manager.save_comprehensive_report(self.test_results)
            self.logger.info(f"Final test report saved to: {report_file}")
            
        # Cleanup environment (preserve results by default)
        self.env_manager.cleanup(force=False)
        
        self.logger.info("Infrastructure shutdown completed")

async def main():
    """Main entry point for testing infrastructure."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Comprehensive Testing Infrastructure for workspace-qdrant-mcp")
    parser.add_argument("--setup", action="store_true", help="Setup testing infrastructure")
    parser.add_argument("--run-basic-tests", action="store_true", help="Run basic infrastructure tests")
    parser.add_argument("--cleanup", action="store_true", help="Cleanup test environment")
    parser.add_argument("--data-profile", default="comprehensive", 
                       choices=["minimal", "comprehensive", "stress"],
                       help="Test data generation profile")
    parser.add_argument("--config", type=str, help="Path to custom configuration file")
    
    args = parser.parse_args()
    
    # Load custom config if provided
    config = None
    if args.config:
        try:
            with open(args.config, 'r') as f:
                config = json.load(f)
        except Exception as e:
            print(f"Error loading config file: {e}")
            return 1
            
    # Initialize infrastructure
    infrastructure = ComprehensiveTestingInfrastructure(config)
    
    try:
        if args.setup:
            print("🔧 Setting up comprehensive testing infrastructure...")
            success = await infrastructure.setup_infrastructure()
            if success:
                print("✅ Infrastructure setup completed successfully")
                return 0
            else:
                print("❌ Infrastructure setup failed")
                return 1
                
        elif args.run_basic_tests:
            print("🧪 Running basic infrastructure tests...")
            
            # Setup first
            setup_success = await infrastructure.setup_infrastructure()
            if not setup_success:
                print("❌ Infrastructure setup failed, cannot run tests")
                return 1
                
            # Run tests
            results = await infrastructure.run_basic_infrastructure_tests()
            
            # Display results
            passed = sum(1 for r in results if r.status == TestStatus.PASSED)
            total = len(results)
            
            print(f"\n📊 Test Results: {passed}/{total} tests passed")
            
            for result in results:
                status_emoji = "✅" if result.status == TestStatus.PASSED else "❌"
                print(f"   {status_emoji} {result.test_id}: {result.status.value}")
                if result.error:
                    print(f"      Error: {result.error}")
                    
            if passed == total:
                print("\n🎉 All infrastructure tests passed!")
                return 0
            else:
                print(f"\n⚠️  {total - passed} tests failed")
                return 1
                
        elif args.cleanup:
            print("🧹 Cleaning up test environment...")
            await infrastructure.shutdown()
            print("✅ Cleanup completed")
            return 0
            
        else:
            parser.print_help()
            return 0
            
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted by user")
        await infrastructure.shutdown()
        return 1
        
    except Exception as e:
        print(f"\n💥 Fatal error: {e}")
        print(traceback.format_exc())
        await infrastructure.shutdown()
        return 1
        
    finally:
        await infrastructure.shutdown()

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)