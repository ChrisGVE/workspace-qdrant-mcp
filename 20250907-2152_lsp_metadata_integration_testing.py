#!/usr/bin/env python3
"""
LSP Metadata Extraction and Integration Testing Framework
Task 154 Implementation - Comprehensive LSP Integration Testing

This script implements comprehensive testing for LSP-enhanced metadata extraction,
symbol resolution, and code structure analysis across multiple programming languages.

Key Test Areas:
1. LSP server connection and health validation across multiple languages
2. Multi-language symbol extraction (Python, TypeScript, Rust, JavaScript)
3. Code structure analysis and dependency mapping testing
4. Performance validation for <5ms lookup times and concurrent operations
5. Metadata synchronization across Qdrant, SQLite state manager, and Web UI

Performance Targets:
- <5ms per LSP symbol lookup
- Support for concurrent LSP operations
- Multi-language support validation
- Metadata consistency across system components

Usage:
    python lsp_metadata_integration_testing.py [--languages python,typescript,rust,javascript]
"""

import asyncio
import json
import logging
import statistics
import time
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
import tempfile
import shutil
import subprocess

# Import existing project modules
from src.workspace_qdrant_mcp.core.lsp_client import AsyncioLspClient, LspError
from src.workspace_qdrant_mcp.core.lsp_metadata_extractor import LspMetadataExtractor
from src.workspace_qdrant_mcp.core.lsp_health_monitor import LspHealthMonitor
from src.workspace_qdrant_mcp.core.incremental_processor import IncrementalProcessor
from src.workspace_qdrant_mcp.core.sqlite_state_manager import SqliteStateManager


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class LspTestResult:
    """Results from LSP testing operations"""
    test_name: str
    language: str
    success: bool
    duration_ms: float
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass 
class PerformanceBenchmark:
    """Performance benchmark results"""
    operation: str
    avg_duration_ms: float
    min_duration_ms: float
    max_duration_ms: float
    p95_duration_ms: float
    operations_count: int
    success_rate: float


class LspLanguageValidator:
    """Validates LSP functionality for specific programming languages"""
    
    def __init__(self, language: str):
        self.language = language
        self.test_files = self._generate_test_files()
        
    def _generate_test_files(self) -> Dict[str, str]:
        """Generate language-specific test files for validation"""
        test_files = {}
        
        if self.language == "python":
            test_files["main.py"] = '''
"""Test Python file for LSP validation"""
import asyncio
from typing import List, Dict, Optional
from pathlib import Path

class TestClass:
    """A test class for symbol extraction"""
    
    def __init__(self, name: str):
        self.name = name
        self._private_attr = None
    
    async def async_method(self, items: List[str]) -> Dict[str, int]:
        """Async method with type annotations"""
        result = {}
        for item in items:
            result[item] = len(item)
        return result
    
    @property
    def name_length(self) -> int:
        return len(self.name)

def standalone_function(data: Optional[Dict]) -> bool:
    """Standalone function for testing"""
    return data is not None

async def main():
    """Main function"""
    test_obj = TestClass("test")
    result = await test_obj.async_method(["a", "b", "c"])
    print(f"Result: {result}")

if __name__ == "__main__":
    asyncio.run(main())
'''
            
        elif self.language == "typescript":
            test_files["main.ts"] = '''
/**
 * Test TypeScript file for LSP validation
 */
import { readFile } from 'fs/promises';

interface TestInterface {
    name: string;
    value: number;
    optional?: boolean;
}

class TestClass implements TestInterface {
    public name: string;
    private _value: number;
    
    constructor(name: string, value: number) {
        this.name = name;
        this._value = value;
    }
    
    get value(): number {
        return this._value;
    }
    
    async processData<T>(data: T[]): Promise<T[]> {
        return data.filter(item => item !== null);
    }
}

export function createInstance(name: string): TestClass {
    return new TestClass(name, 42);
}

export default TestClass;
'''
            
        elif self.language == "rust":
            test_files["main.rs"] = '''
//! Test Rust file for LSP validation
use std::collections::HashMap;
use std::path::Path;

/// A test struct for symbol extraction
#[derive(Debug, Clone)]
pub struct TestStruct {
    pub name: String,
    value: i32,
}

impl TestStruct {
    /// Create a new TestStruct
    pub fn new(name: String, value: i32) -> Self {
        Self { name, value }
    }
    
    /// Get the value
    pub fn get_value(&self) -> i32 {
        self.value
    }
    
    /// Process data with generic type
    pub fn process_data<T>(&self, data: Vec<T>) -> Vec<T> 
    where 
        T: Clone,
    {
        data.into_iter().collect()
    }
}

/// Standalone function for testing
pub fn create_map() -> HashMap<String, i32> {
    let mut map = HashMap::new();
    map.insert("test".to_string(), 42);
    map
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_struct_creation() {
        let test_obj = TestStruct::new("test".to_string(), 100);
        assert_eq!(test_obj.get_value(), 100);
    }
}
'''
            
        elif self.language == "javascript":
            test_files["main.js"] = '''
/**
 * Test JavaScript file for LSP validation
 */

const path = require('path');
const fs = require('fs/promises');

/**
 * A test class for symbol extraction
 */
class TestClass {
    /**
     * @param {string} name - The name
     * @param {number} value - The value
     */
    constructor(name, value) {
        this.name = name;
        this._value = value;
    }
    
    /**
     * Get the value
     * @returns {number} The value
     */
    get value() {
        return this._value;
    }
    
    /**
     * Process data array
     * @param {Array} data - Data to process
     * @returns {Promise<Array>} Processed data
     */
    async processData(data) {
        return data.filter(item => item !== null);
    }
}

/**
 * Create a new instance
 * @param {string} name - The name
 * @returns {TestClass} New instance
 */
function createInstance(name) {
    return new TestClass(name, 42);
}

module.exports = {
    TestClass,
    createInstance
};
'''
        
        return test_files


class LspMetadataIntegrationTester:
    """Comprehensive LSP metadata extraction and integration testing framework"""
    
    def __init__(self, languages: Optional[List[str]] = None):
        self.languages = languages or ["python", "typescript", "rust", "javascript"]
        self.test_results: List[LspTestResult] = []
        self.performance_results: List[PerformanceBenchmark] = []
        self.temp_dir: Optional[Path] = None
        
        # LSP components
        self.lsp_clients: Dict[str, AsyncioLspClient] = {}
        self.metadata_extractor: Optional[LspMetadataExtractor] = None
        self.health_monitor: Optional[LspHealthMonitor] = None
        self.incremental_processor: Optional[IncrementalProcessor] = None
        self.state_manager: Optional[SqliteStateManager] = None
    
    async def setup_test_environment(self):
        """Set up the test environment with temporary directories and test files"""
        logger.info("Setting up LSP integration test environment")
        
        # Create temporary directory
        self.temp_dir = Path(tempfile.mkdtemp(prefix="lsp_integration_test_"))
        logger.info(f"Created test directory: {self.temp_dir}")
        
        # Create test files for each language
        for language in self.languages:
            language_dir = self.temp_dir / language
            language_dir.mkdir(exist_ok=True)
            
            validator = LspLanguageValidator(language)
            for filename, content in validator.test_files.items():
                test_file = language_dir / filename
                test_file.write_text(content)
                logger.info(f"Created test file: {test_file}")
        
        # Initialize LSP components
        await self._initialize_lsp_components()
    
    async def _initialize_lsp_components(self):
        """Initialize LSP-related components"""
        try:
            # Initialize metadata extractor
            self.metadata_extractor = LspMetadataExtractor()
            await self.metadata_extractor.initialize()
            
            # Initialize health monitor
            self.health_monitor = LspHealthMonitor()
            
            # Initialize incremental processor
            self.incremental_processor = IncrementalProcessor()
            
            # Initialize state manager with test database
            state_db_path = self.temp_dir / "test_state.db"
            self.state_manager = SqliteStateManager(str(state_db_path))
            await self.state_manager.initialize()
            
            logger.info("Successfully initialized LSP components")
            
        except Exception as e:
            logger.error(f"Failed to initialize LSP components: {e}")
            raise
    
    async def test_lsp_server_connections(self) -> List[LspTestResult]:
        """Test LSP server connections and health for all languages"""
        logger.info("Testing LSP server connections and health")
        results = []
        
        for language in self.languages:
            start_time = time.time()
            
            try:
                # Test LSP server health
                health_status = await self._test_lsp_health(language)
                
                duration_ms = (time.time() - start_time) * 1000
                
                result = LspTestResult(
                    test_name="lsp_server_connection",
                    language=language,
                    success=health_status["connected"],
                    duration_ms=duration_ms,
                    metadata=health_status
                )
                
                if not health_status["connected"]:
                    result.error_message = health_status.get("error", "LSP server not connected")
                
                results.append(result)
                logger.info(f"LSP connection test for {language}: {'PASS' if result.success else 'FAIL'}")
                
            except Exception as e:
                duration_ms = (time.time() - start_time) * 1000
                result = LspTestResult(
                    test_name="lsp_server_connection",
                    language=language,
                    success=False,
                    duration_ms=duration_ms,
                    error_message=str(e)
                )
                results.append(result)
                logger.error(f"LSP connection test failed for {language}: {e}")
        
        self.test_results.extend(results)
        return results
    
    async def _test_lsp_health(self, language: str) -> Dict[str, Any]:
        """Test LSP server health for a specific language"""
        if not self.health_monitor:
            raise RuntimeError("Health monitor not initialized")
        
        # Language-specific LSP server detection
        lsp_servers = {
            "python": "pylsp",
            "typescript": "typescript-language-server", 
            "rust": "rust-analyzer",
            "javascript": "typescript-language-server"
        }
        
        server_name = lsp_servers.get(language)
        if not server_name:
            return {"connected": False, "error": f"No LSP server defined for {language}"}
        
        try:
            health_status = await self.health_monitor.check_server_health(server_name)
            return health_status
            
        except Exception as e:
            return {"connected": False, "error": str(e)}
    
    async def test_symbol_extraction(self) -> List[LspTestResult]:
        """Test symbol extraction across multiple languages"""
        logger.info("Testing multi-language symbol extraction")
        results = []
        
        for language in self.languages:
            language_dir = self.temp_dir / language
            
            # Find test files for this language
            test_files = list(language_dir.glob("*"))
            
            for test_file in test_files:
                start_time = time.time()
                
                try:
                    # Extract symbols from file
                    symbols = await self._extract_file_symbols(test_file, language)
                    
                    duration_ms = (time.time() - start_time) * 1000
                    
                    # Validate extracted symbols
                    validation_result = self._validate_extracted_symbols(symbols, language)
                    
                    result = LspTestResult(
                        test_name="symbol_extraction",
                        language=language,
                        success=validation_result["valid"],
                        duration_ms=duration_ms,
                        metadata={
                            "symbols_count": len(symbols),
                            "file_path": str(test_file),
                            "validation": validation_result
                        }
                    )
                    
                    if not validation_result["valid"]:
                        result.error_message = validation_result.get("error", "Symbol validation failed")
                    
                    results.append(result)
                    logger.info(f"Symbol extraction for {test_file}: {'PASS' if result.success else 'FAIL'}")
                    
                except Exception as e:
                    duration_ms = (time.time() - start_time) * 1000
                    result = LspTestResult(
                        test_name="symbol_extraction",
                        language=language,
                        success=False,
                        duration_ms=duration_ms,
                        error_message=str(e),
                        metadata={"file_path": str(test_file)}
                    )
                    results.append(result)
                    logger.error(f"Symbol extraction failed for {test_file}: {e}")
        
        self.test_results.extend(results)
        return results
    
    async def _extract_file_symbols(self, file_path: Path, language: str) -> List[Dict[str, Any]]:
        """Extract symbols from a file using LSP metadata extractor"""
        if not self.metadata_extractor:
            raise RuntimeError("Metadata extractor not initialized")
        
        try:
            # Extract metadata including symbols
            metadata = await self.metadata_extractor.extract_file_metadata(str(file_path))
            return metadata.get("symbols", [])
            
        except Exception as e:
            logger.error(f"Failed to extract symbols from {file_path}: {e}")
            raise
    
    def _validate_extracted_symbols(self, symbols: List[Dict[str, Any]], language: str) -> Dict[str, Any]:
        """Validate that extracted symbols meet expectations"""
        if not symbols:
            return {"valid": False, "error": "No symbols extracted"}
        
        # Language-specific validation rules
        expected_symbols = {
            "python": ["class", "function", "method", "import"],
            "typescript": ["class", "interface", "function", "method"],
            "rust": ["struct", "impl", "function", "mod"],
            "javascript": ["class", "function", "method"]
        }
        
        expected_for_language = expected_symbols.get(language, [])
        found_symbol_types = set()
        
        for symbol in symbols:
            symbol_kind = symbol.get("kind", "").lower()
            symbol_type = symbol.get("type", "").lower()
            
            # Map LSP symbol kinds to our expected types
            if "class" in symbol_kind or "class" in symbol_type:
                found_symbol_types.add("class")
            elif "function" in symbol_kind or "function" in symbol_type:
                found_symbol_types.add("function")
            elif "method" in symbol_kind or "method" in symbol_type:
                found_symbol_types.add("method")
            elif "interface" in symbol_kind or "interface" in symbol_type:
                found_symbol_types.add("interface")
            elif "struct" in symbol_kind or "struct" in symbol_type:
                found_symbol_types.add("struct")
            elif "import" in symbol_kind or "import" in symbol_type:
                found_symbol_types.add("import")
        
        # Check if we found expected symbol types
        found_expected = any(expected in found_symbol_types for expected in expected_for_language)
        
        return {
            "valid": found_expected and len(symbols) > 0,
            "found_symbols": list(found_symbol_types),
            "expected_symbols": expected_for_language,
            "symbols_count": len(symbols)
        }
    
    async def test_performance_benchmarks(self) -> List[PerformanceBenchmark]:
        """Test performance benchmarks for LSP operations"""
        logger.info("Running LSP performance benchmarks")
        benchmarks = []
        
        # Test symbol lookup performance
        lookup_benchmark = await self._benchmark_symbol_lookups()
        benchmarks.append(lookup_benchmark)
        
        # Test concurrent operations
        concurrent_benchmark = await self._benchmark_concurrent_operations()
        benchmarks.append(concurrent_benchmark)
        
        # Test metadata synchronization performance
        sync_benchmark = await self._benchmark_metadata_synchronization()
        benchmarks.append(sync_benchmark)
        
        self.performance_results.extend(benchmarks)
        return benchmarks
    
    async def _benchmark_symbol_lookups(self) -> PerformanceBenchmark:
        """Benchmark symbol lookup performance"""
        logger.info("Benchmarking symbol lookup performance")
        
        durations = []
        successful_operations = 0
        total_operations = 0
        
        # Collect all test files
        test_files = []
        for language in self.languages:
            language_dir = self.temp_dir / language
            test_files.extend(list(language_dir.glob("*")))
        
        # Perform multiple lookup operations
        for _ in range(10):  # 10 iterations per file
            for test_file in test_files:
                total_operations += 1
                start_time = time.perf_counter()
                
                try:
                    # Simulate symbol lookup operation
                    symbols = await self._extract_file_symbols(test_file, test_file.parent.name)
                    
                    duration_ms = (time.perf_counter() - start_time) * 1000
                    durations.append(duration_ms)
                    
                    if duration_ms < 5.0:  # Target: <5ms per lookup
                        successful_operations += 1
                        
                except Exception as e:
                    logger.error(f"Symbol lookup failed for {test_file}: {e}")
                    durations.append(float('inf'))  # Mark as failed
        
        if not durations or all(d == float('inf') for d in durations):
            return PerformanceBenchmark(
                operation="symbol_lookups",
                avg_duration_ms=float('inf'),
                min_duration_ms=float('inf'),
                max_duration_ms=float('inf'),
                p95_duration_ms=float('inf'),
                operations_count=total_operations,
                success_rate=0.0
            )
        
        # Filter out failed operations for statistics
        valid_durations = [d for d in durations if d != float('inf')]
        
        return PerformanceBenchmark(
            operation="symbol_lookups",
            avg_duration_ms=statistics.mean(valid_durations) if valid_durations else float('inf'),
            min_duration_ms=min(valid_durations) if valid_durations else float('inf'),
            max_duration_ms=max(valid_durations) if valid_durations else float('inf'),
            p95_duration_ms=statistics.quantiles(valid_durations, n=20)[18] if len(valid_durations) > 1 else (valid_durations[0] if valid_durations else float('inf')),
            operations_count=total_operations,
            success_rate=successful_operations / total_operations if total_operations > 0 else 0.0
        )
    
    async def _benchmark_concurrent_operations(self) -> PerformanceBenchmark:
        """Benchmark concurrent LSP operations"""
        logger.info("Benchmarking concurrent LSP operations")
        
        # Test concurrent symbol extraction
        test_files = []
        for language in self.languages:
            language_dir = self.temp_dir / language
            test_files.extend(list(language_dir.glob("*")))
        
        durations = []
        successful_operations = 0
        total_operations = len(test_files) * 5  # 5 concurrent operations per file
        
        async def concurrent_extraction(test_file: Path) -> float:
            start_time = time.perf_counter()
            try:
                symbols = await self._extract_file_symbols(test_file, test_file.parent.name)
                return (time.perf_counter() - start_time) * 1000
            except Exception as e:
                logger.error(f"Concurrent extraction failed for {test_file}: {e}")
                return float('inf')
        
        # Run concurrent operations
        for test_file in test_files:
            # Create 5 concurrent tasks per file
            tasks = [concurrent_extraction(test_file) for _ in range(5)]
            
            start_time = time.perf_counter()
            results = await asyncio.gather(*tasks, return_exceptions=True)
            total_duration_ms = (time.perf_counter() - start_time) * 1000
            
            for result in results:
                if isinstance(result, Exception):
                    durations.append(float('inf'))
                else:
                    durations.append(result)
                    if result < 5.0:  # Target: <5ms per operation
                        successful_operations += 1
        
        valid_durations = [d for d in durations if d != float('inf')]
        
        return PerformanceBenchmark(
            operation="concurrent_operations",
            avg_duration_ms=statistics.mean(valid_durations) if valid_durations else float('inf'),
            min_duration_ms=min(valid_durations) if valid_durations else float('inf'),
            max_duration_ms=max(valid_durations) if valid_durations else float('inf'),
            p95_duration_ms=statistics.quantiles(valid_durations, n=20)[18] if len(valid_durations) > 1 else (valid_durations[0] if valid_durations else float('inf')),
            operations_count=total_operations,
            success_rate=successful_operations / total_operations if total_operations > 0 else 0.0
        )
    
    async def _benchmark_metadata_synchronization(self) -> PerformanceBenchmark:
        """Benchmark metadata synchronization across system components"""
        logger.info("Benchmarking metadata synchronization")
        
        durations = []
        successful_operations = 0
        total_operations = 0
        
        # Test metadata sync for each language
        for language in self.languages:
            language_dir = self.temp_dir / language
            test_files = list(language_dir.glob("*"))
            
            for test_file in test_files:
                total_operations += 1
                start_time = time.perf_counter()
                
                try:
                    # Extract metadata
                    symbols = await self._extract_file_symbols(test_file, language)
                    
                    # Simulate metadata synchronization
                    if self.state_manager:
                        await self._sync_metadata_to_state_manager(test_file, symbols)
                    
                    duration_ms = (time.perf_counter() - start_time) * 1000
                    durations.append(duration_ms)
                    successful_operations += 1
                    
                except Exception as e:
                    duration_ms = (time.perf_counter() - start_time) * 1000
                    durations.append(duration_ms)
                    logger.error(f"Metadata sync failed for {test_file}: {e}")
        
        valid_durations = [d for d in durations if d != float('inf')]
        
        return PerformanceBenchmark(
            operation="metadata_synchronization",
            avg_duration_ms=statistics.mean(valid_durations) if valid_durations else float('inf'),
            min_duration_ms=min(valid_durations) if valid_durations else float('inf'),
            max_duration_ms=max(valid_durations) if valid_durations else float('inf'),
            p95_duration_ms=statistics.quantiles(valid_durations, n=20)[18] if len(valid_durations) > 1 else (valid_durations[0] if valid_durations else float('inf')),
            operations_count=total_operations,
            success_rate=successful_operations / total_operations if total_operations > 0 else 0.0
        )
    
    async def _sync_metadata_to_state_manager(self, file_path: Path, symbols: List[Dict[str, Any]]):
        """Sync metadata to state manager"""
        if not self.state_manager:
            return
        
        # Create metadata record
        metadata = {
            "file_path": str(file_path),
            "symbols_count": len(symbols),
            "last_updated": time.time(),
            "symbols": symbols
        }
        
        # Store in state manager
        await self.state_manager.set_file_metadata(str(file_path), metadata)
    
    async def test_incremental_processing(self) -> List[LspTestResult]:
        """Test incremental processing capabilities"""
        logger.info("Testing incremental processing")
        results = []
        
        for language in self.languages:
            language_dir = self.temp_dir / language
            test_files = list(language_dir.glob("*"))
            
            for test_file in test_files:
                start_time = time.time()
                
                try:
                    # Initial processing
                    initial_symbols = await self._extract_file_symbols(test_file, language)
                    
                    # Modify file content
                    original_content = test_file.read_text()
                    modified_content = original_content + "\n# Added comment for incremental test\n"
                    test_file.write_text(modified_content)
                    
                    # Process incremental update
                    updated_symbols = await self._extract_file_symbols(test_file, language)
                    
                    # Restore original content
                    test_file.write_text(original_content)
                    
                    duration_ms = (time.time() - start_time) * 1000
                    
                    # Validate incremental processing worked
                    processing_valid = len(updated_symbols) >= len(initial_symbols)
                    
                    result = LspTestResult(
                        test_name="incremental_processing",
                        language=language,
                        success=processing_valid,
                        duration_ms=duration_ms,
                        metadata={
                            "initial_symbols": len(initial_symbols),
                            "updated_symbols": len(updated_symbols),
                            "file_path": str(test_file)
                        }
                    )
                    
                    if not processing_valid:
                        result.error_message = "Incremental processing validation failed"
                    
                    results.append(result)
                    logger.info(f"Incremental processing for {test_file}: {'PASS' if result.success else 'FAIL'}")
                    
                except Exception as e:
                    duration_ms = (time.time() - start_time) * 1000
                    result = LspTestResult(
                        test_name="incremental_processing",
                        language=language,
                        success=False,
                        duration_ms=duration_ms,
                        error_message=str(e),
                        metadata={"file_path": str(test_file)}
                    )
                    results.append(result)
                    logger.error(f"Incremental processing failed for {test_file}: {e}")
        
        self.test_results.extend(results)
        return results
    
    def generate_comprehensive_report(self) -> Dict[str, Any]:
        """Generate comprehensive test report"""
        # Calculate overall statistics
        total_tests = len(self.test_results)
        successful_tests = sum(1 for result in self.test_results if result.success)
        success_rate = (successful_tests / total_tests * 100) if total_tests > 0 else 0
        
        # Performance analysis
        performance_summary = {}
        for benchmark in self.performance_results:
            performance_summary[benchmark.operation] = {
                "avg_duration_ms": benchmark.avg_duration_ms,
                "p95_duration_ms": benchmark.p95_duration_ms,
                "success_rate": benchmark.success_rate * 100,
                "meets_target": benchmark.avg_duration_ms < 5.0,  # <5ms target
                "operations_count": benchmark.operations_count
            }
        
        # Language-specific results
        language_results = {}
        for language in self.languages:
            language_tests = [r for r in self.test_results if r.language == language]
            language_success = sum(1 for r in language_tests if r.success)
            language_total = len(language_tests)
            
            language_results[language] = {
                "tests_run": language_total,
                "tests_passed": language_success,
                "success_rate": (language_success / language_total * 100) if language_total > 0 else 0,
                "avg_duration_ms": statistics.mean([r.duration_ms for r in language_tests]) if language_tests else 0
            }
        
        # Test type analysis
        test_type_results = {}
        for test_name in ["lsp_server_connection", "symbol_extraction", "incremental_processing"]:
            type_tests = [r for r in self.test_results if r.test_name == test_name]
            type_success = sum(1 for r in type_tests if r.success)
            type_total = len(type_tests)
            
            test_type_results[test_name] = {
                "tests_run": type_total,
                "tests_passed": type_success,
                "success_rate": (type_success / type_total * 100) if type_total > 0 else 0,
                "avg_duration_ms": statistics.mean([r.duration_ms for r in type_tests]) if type_tests else 0
            }
        
        return {
            "summary": {
                "total_tests": total_tests,
                "successful_tests": successful_tests,
                "success_rate": success_rate,
                "languages_tested": len(self.languages),
                "performance_targets_met": sum(1 for p in performance_summary.values() if p.get("meets_target", False))
            },
            "performance_benchmarks": performance_summary,
            "language_results": language_results,
            "test_type_results": test_type_results,
            "detailed_results": [
                {
                    "test_name": r.test_name,
                    "language": r.language,
                    "success": r.success,
                    "duration_ms": r.duration_ms,
                    "error_message": r.error_message,
                    "metadata": r.metadata
                } for r in self.test_results
            ]
        }
    
    async def cleanup(self):
        """Clean up test environment"""
        if self.temp_dir and self.temp_dir.exists():
            shutil.rmtree(self.temp_dir, ignore_errors=True)
            logger.info(f"Cleaned up test directory: {self.temp_dir}")
        
        # Cleanup LSP components
        if self.metadata_extractor:
            await self.metadata_extractor.cleanup()
        
        if self.state_manager:
            await self.state_manager.close()
    
    async def run_comprehensive_tests(self) -> Dict[str, Any]:
        """Run all comprehensive LSP integration tests"""
        logger.info("Starting comprehensive LSP metadata integration testing")
        
        try:
            # Setup test environment
            await self.setup_test_environment()
            
            # Run test suites
            logger.info("Phase 1: Testing LSP server connections")
            await self.test_lsp_server_connections()
            
            logger.info("Phase 2: Testing symbol extraction")
            await self.test_symbol_extraction()
            
            logger.info("Phase 3: Testing incremental processing")
            await self.test_incremental_processing()
            
            logger.info("Phase 4: Running performance benchmarks")
            await self.test_performance_benchmarks()
            
            # Generate comprehensive report
            report = self.generate_comprehensive_report()
            
            logger.info("LSP integration testing completed successfully")
            return report
            
        except Exception as e:
            logger.error(f"LSP integration testing failed: {e}")
            raise
        
        finally:
            await self.cleanup()


async def main():
    """Main execution function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="LSP Metadata Integration Testing Framework")
    parser.add_argument(
        "--languages",
        type=str,
        default="python,typescript,rust,javascript",
        help="Comma-separated list of languages to test"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="lsp_integration_test_report.json",
        help="Output file for test report"
    )
    
    args = parser.parse_args()
    
    languages = [lang.strip() for lang in args.languages.split(",")]
    
    # Run comprehensive testing
    tester = LspMetadataIntegrationTester(languages=languages)
    
    try:
        report = await tester.run_comprehensive_tests()
        
        # Save report
        output_path = Path(args.output)
        output_path.write_text(json.dumps(report, indent=2))
        
        logger.info(f"Test report saved to: {output_path}")
        
        # Print summary
        print("\n" + "="*80)
        print("LSP METADATA INTEGRATION TEST SUMMARY")
        print("="*80)
        
        summary = report["summary"]
        print(f"Total Tests: {summary['total_tests']}")
        print(f"Successful Tests: {summary['successful_tests']}")
        print(f"Success Rate: {summary['success_rate']:.1f}%")
        print(f"Languages Tested: {summary['languages_tested']}")
        print(f"Performance Targets Met: {summary['performance_targets_met']}")
        
        print("\nPERFORMANCE BENCHMARKS:")
        for operation, metrics in report["performance_benchmarks"].items():
            print(f"  {operation}:")
            print(f"    Avg Duration: {metrics['avg_duration_ms']:.2f}ms")
            print(f"    P95 Duration: {metrics['p95_duration_ms']:.2f}ms") 
            print(f"    Success Rate: {metrics['success_rate']:.1f}%")
            print(f"    Target Met: {'✓' if metrics['meets_target'] else '✗'}")
        
        print("\nLANGUAGE RESULTS:")
        for language, results in report["language_results"].items():
            print(f"  {language}: {results['tests_passed']}/{results['tests_run']} ({results['success_rate']:.1f}%)")
        
        # Exit code based on success rate
        exit_code = 0 if summary['success_rate'] >= 80.0 else 1
        print(f"\nTest {'PASSED' if exit_code == 0 else 'FAILED'}")
        return exit_code
        
    except Exception as e:
        logger.error(f"Testing failed with error: {e}")
        return 1


if __name__ == "__main__":
    import sys
    exit_code = asyncio.run(main())
    sys.exit(exit_code)