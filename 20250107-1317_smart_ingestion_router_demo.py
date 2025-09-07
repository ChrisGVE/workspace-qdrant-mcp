#!/usr/bin/env python3
"""
Smart Ingestion Router Demonstration Script

This script demonstrates the key functionality of the Smart Ingestion Router
including file classification, processing strategies, batch operations, and
integration with LSP-based code metadata extraction.

Task #122 - Smart Ingestion Differentiation Logic Demo

Usage:
    python 20250107-1317_smart_ingestion_router_demo.py
"""

import asyncio
import json
import tempfile
import time
from pathlib import Path
from typing import List, Dict, Any

import structlog

from src.workspace_qdrant_mcp.core.smart_ingestion_router import (
    SmartIngestionRouter,
    RouterConfiguration,
    ProcessingStrategy,
    FileClassification
)
from src.workspace_qdrant_mcp.core.language_filters import LanguageAwareFilter

# Setup logging for demo
logging = structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger(__name__)


def create_demo_files(workspace_path: Path) -> List[Path]:
    """Create demonstration files of various types"""
    demo_files = []
    
    # Python code file
    python_file = workspace_path / "example.py"
    python_file.write_text("""
#!/usr/bin/env python3
\"\"\"Example Python module for Smart Ingestion Router demo\"\"\"

import asyncio
from typing import List, Optional

class DataProcessor:
    \"\"\"Process data with various algorithms\"\"\"
    
    def __init__(self, config: dict):
        self.config = config
        self.processed_count = 0
    
    async def process_data(self, data: List[str]) -> Optional[List[str]]:
        \"\"\"Process a list of data items asynchronously\"\"\"
        if not data:
            return None
        
        results = []
        for item in data:
            # Simulate processing
            await asyncio.sleep(0.01)
            processed_item = f"processed_{item}"
            results.append(processed_item)
            self.processed_count += 1
        
        return results
    
    def get_stats(self) -> dict:
        \"\"\"Get processing statistics\"\"\"
        return {
            "processed_count": self.processed_count,
            "config": self.config
        }

def main():
    \"\"\"Main function for demonstration\"\"\"
    processor = DataProcessor({"mode": "demo"})
    data = ["item1", "item2", "item3"]
    
    async def run_demo():
        result = await processor.process_data(data)
        print(f"Processing result: {result}")
        print(f"Statistics: {processor.get_stats()}")
    
    asyncio.run(run_demo())

if __name__ == "__main__":
    main()
""")
    demo_files.append(python_file)
    
    # JavaScript/TypeScript file
    js_file = workspace_path / "api.js"
    js_file.write_text("""
/**
 * API client for Smart Ingestion Router demo
 */

class ApiClient {
    constructor(baseUrl, options = {}) {
        this.baseUrl = baseUrl;
        this.timeout = options.timeout || 5000;
        this.retries = options.retries || 3;
    }
    
    async fetchData(endpoint, params = {}) {
        const url = new URL(endpoint, this.baseUrl);
        Object.keys(params).forEach(key => 
            url.searchParams.append(key, params[key])
        );
        
        for (let attempt = 1; attempt <= this.retries; attempt++) {
            try {
                const response = await fetch(url.toString(), {
                    timeout: this.timeout
                });
                
                if (!response.ok) {
                    throw new Error(`HTTP ${response.status}: ${response.statusText}`);
                }
                
                return await response.json();
            } catch (error) {
                console.warn(`Attempt ${attempt} failed:`, error.message);
                if (attempt === this.retries) {
                    throw error;
                }
                await this.delay(1000 * attempt);
            }
        }
    }
    
    delay(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }
}

module.exports = { ApiClient };
""")
    demo_files.append(js_file)
    
    # Rust file
    rust_file = workspace_path / "lib.rs"
    rust_file.write_text("""
//! Smart Ingestion Router - Rust demonstration module
//! 
//! This module demonstrates Rust code that would be processed
//! through LSP-enriched ingestion pipeline.

use std::collections::HashMap;
use std::error::Error;
use std::fmt;

/// Configuration for data processing
#[derive(Debug, Clone)]
pub struct ProcessorConfig {
    pub batch_size: usize,
    pub timeout_ms: u64,
    pub retries: u32,
}

impl Default for ProcessorConfig {
    fn default() -> Self {
        ProcessorConfig {
            batch_size: 100,
            timeout_ms: 5000,
            retries: 3,
        }
    }
}

/// Custom error type for processing operations
#[derive(Debug)]
pub enum ProcessingError {
    InvalidInput(String),
    TimeoutError,
    NetworkError(String),
}

impl fmt::Display for ProcessingError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            ProcessingError::InvalidInput(msg) => write!(f, "Invalid input: {}", msg),
            ProcessingError::TimeoutError => write!(f, "Operation timed out"),
            ProcessingError::NetworkError(msg) => write!(f, "Network error: {}", msg),
        }
    }
}

impl Error for ProcessingError {}

/// Main data processor struct
pub struct DataProcessor {
    config: ProcessorConfig,
    cache: HashMap<String, String>,
}

impl DataProcessor {
    /// Create a new processor with given configuration
    pub fn new(config: ProcessorConfig) -> Self {
        DataProcessor {
            config,
            cache: HashMap::new(),
        }
    }
    
    /// Process a batch of data items
    pub async fn process_batch(&mut self, items: Vec<&str>) -> Result<Vec<String>, ProcessingError> {
        if items.is_empty() {
            return Err(ProcessingError::InvalidInput("Empty batch".to_string()));
        }
        
        let mut results = Vec::with_capacity(items.len());
        
        for item in items {
            // Check cache first
            if let Some(cached) = self.cache.get(item) {
                results.push(cached.clone());
                continue;
            }
            
            // Simulate processing
            let processed = format!("processed_{}", item);
            self.cache.insert(item.to_string(), processed.clone());
            results.push(processed);
        }
        
        Ok(results)
    }
    
    /// Get cache statistics
    pub fn cache_stats(&self) -> (usize, usize) {
        (self.cache.len(), self.config.batch_size)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_process_batch() {
        let mut processor = DataProcessor::new(ProcessorConfig::default());
        let items = vec!["item1", "item2", "item3"];
        
        let result = processor.process_batch(items).await;
        assert!(result.is_ok());
        
        let processed = result.unwrap();
        assert_eq!(processed.len(), 3);
        assert_eq!(processed[0], "processed_item1");
    }
}
""")
    demo_files.append(rust_file)
    
    # Documentation file
    readme_file = workspace_path / "README.md"
    readme_file.write_text("""
# Smart Ingestion Router Demo

This demonstration showcases the Smart Ingestion Router's ability to intelligently
differentiate between different types of files and route them through appropriate
processing pipelines.

## Features

### File Classification
The router uses multiple detection methods:
- **Extension-based**: Fast classification using file extensions
- **MIME type detection**: Content-type analysis for accurate classification  
- **Content analysis**: Deep content inspection for ambiguous files

### Processing Strategies

1. **LSP Enriched Processing**
   - Used for code files (Python, JavaScript, Rust, etc.)
   - Extracts detailed metadata using Language Server Protocol
   - Provides symbol information, relationships, and code structure

2. **Standard Ingestion**
   - Used for documentation, data files, and configuration
   - Processes content as text with basic metadata
   - Fast and efficient for non-code content

3. **Fallback Processing**
   - Used when LSP servers are unavailable
   - Processes code files as standard text with basic syntax analysis
   - Ensures reliability and continuity

### Batch Processing

The router optimizes batch processing by:
- Grouping files by processing strategy
- Concurrent processing within strategy groups
- Resource management and timeout handling
- Performance monitoring and statistics

## Configuration

The router supports extensive configuration:
- Custom file type mappings
- Processing strategy overrides
- Performance tuning parameters
- Caching and resource management

## Statistics and Monitoring

Comprehensive statistics include:
- File classification accuracy
- Processing performance metrics
- Cache hit rates and efficiency
- Error rates and fallback triggers

## Example Usage

```python
from smart_ingestion_router import SmartIngestionRouter

# Initialize router
router = SmartIngestionRouter()
await router.initialize()

# Process single file
result = await router.process_single_file("example.py")

# Process batch of files
results = await router.process_batch(file_paths)

# Get processing statistics
stats = router.get_statistics()
```

See the demo script for complete examples and usage patterns.
""")
    demo_files.append(readme_file)
    
    # Configuration file
    config_file = workspace_path / "config.json"
    config_file.write_text(json.dumps({
        "database": {
            "host": "localhost",
            "port": 5432,
            "name": "smart_ingestion_demo",
            "pool_size": 10
        },
        "processing": {
            "batch_size": 50,
            "timeout_seconds": 30,
            "enable_caching": True,
            "max_concurrent": 5
        },
        "logging": {
            "level": "INFO",
            "format": "json",
            "include_timestamp": True
        }
    }, indent=2))
    demo_files.append(config_file)
    
    # Data file
    data_file = workspace_path / "sample_data.csv"
    data_file.write_text("""id,name,type,processed_at
1,example_file_1.py,code,2025-01-07T10:30:00Z
2,documentation.md,doc,2025-01-07T10:31:00Z
3,config.json,data,2025-01-07T10:32:00Z
4,library.rs,code,2025-01-07T10:33:00Z
5,api.js,code,2025-01-07T10:34:00Z
""")
    demo_files.append(data_file)
    
    # Binary-like file
    binary_file = workspace_path / "image.png"
    binary_file.write_bytes(b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x10\x00\x00\x00\x10\x08\x02')
    demo_files.append(binary_file)
    
    return demo_files


async def demonstrate_single_file_processing(router: SmartIngestionRouter, files: List[Path]):
    """Demonstrate single file processing"""
    print("\n=== Single File Processing Demo ===")
    
    for file_path in files[:4]:  # Process first 4 files
        print(f"\nProcessing: {file_path.name}")
        
        # Route file to determine strategy
        strategy, classification = await router.route_file(file_path)
        
        print(f"  Classification: {classification.classification.value if classification else 'N/A'}")
        print(f"  Strategy: {strategy.value}")
        print(f"  Confidence: {classification.confidence if classification else 'N/A'}")
        
        if classification and classification.detected_language:
            print(f"  Detected Language: {classification.detected_language}")
        
        # Process the file
        result = await router.process_single_file(file_path)
        
        if result:
            print(f"  Processing: SUCCESS")
            print(f"  Strategy Used: {result['processing_strategy']}")
            if 'lsp_metadata' in result:
                print(f"  LSP Metadata: Available")
            elif 'syntax_info' in result:
                print(f"  Syntax Info: {result['syntax_info']}")
            else:
                content_preview = result.get('content', '')[:100]
                print(f"  Content Preview: {content_preview}{'...' if len(content_preview) == 100 else ''}")
        else:
            print(f"  Processing: SKIPPED/FAILED")


async def demonstrate_batch_processing(router: SmartIngestionRouter, files: List[Path]):
    """Demonstrate batch processing optimization"""
    print("\n=== Batch Processing Demo ===")
    
    print(f"Processing {len(files)} files in batch...")
    
    start_time = time.perf_counter()
    results = await router.process_batch(files, batch_size=3)
    end_time = time.perf_counter()
    
    processing_time = (end_time - start_time) * 1000
    
    print(f"\nBatch Processing Results:")
    print(f"  Total Files: {len(files)}")
    print(f"  Successfully Processed: {len(results)}")
    print(f"  Processing Time: {processing_time:.2f} ms")
    print(f"  Average Time per File: {processing_time / len(files):.2f} ms")
    
    # Group results by processing strategy
    strategy_groups = {}
    for result in results:
        strategy = result['processing_strategy']
        strategy_groups[strategy] = strategy_groups.get(strategy, 0) + 1
    
    print(f"\nProcessing Strategy Distribution:")
    for strategy, count in strategy_groups.items():
        print(f"  {strategy}: {count} files")


async def demonstrate_statistics_and_capabilities(router: SmartIngestionRouter):
    """Demonstrate statistics and capabilities reporting"""
    print("\n=== Statistics and Capabilities Demo ===")
    
    # Get processing capabilities
    capabilities = await router.get_processing_capabilities()
    print("\nProcessing Capabilities:")
    print(f"  LSP Available: {capabilities['lsp_available']}")
    print(f"  Supported LSP Languages: {', '.join(capabilities['supported_lsp_languages']) if capabilities['supported_lsp_languages'] else 'None'}")
    print(f"  File Filter Initialized: {capabilities['file_filter_initialized']}")
    print(f"  Classification Cache Size: {capabilities['classification_cache_size']}")
    
    # Get detailed statistics
    stats = router.get_statistics()
    stats_dict = stats.to_dict()
    
    print("\nProcessing Statistics:")
    
    # Classification stats
    classification = stats_dict['classification']
    print(f"  Files Classified: {classification['files_classified']}")
    print(f"  Average Classification Time: {classification['avg_classification_time_ms']:.2f} ms")
    
    if classification['classification_by_type']:
        print("  Classification Breakdown:")
        for file_type, count in classification['classification_by_type'].items():
            print(f"    {file_type}: {count}")
    
    # Processing stats
    processing = stats_dict['processing']
    print(f"  Files Processed: {processing['files_processed']}")
    print(f"  Success Rate: {processing['success_rate']:.1%}")
    print(f"  LSP Processed: {processing['lsp_processed']}")
    print(f"  Standard Processed: {processing['standard_processed']}")
    print(f"  Fallback Processed: {processing['fallback_processed']}")
    
    # Performance stats
    performance = stats_dict['performance']
    print(f"  Average Processing Time: {performance['avg_processing_time_ms']:.2f} ms")
    print(f"  Total Processing Time: {performance['total_processing_time_ms']:.2f} ms")
    
    # Cache stats
    cache = stats_dict['cache']
    print(f"  Cache Hit Rate: {cache['cache_hit_rate']:.1%}")
    print(f"  Cache Hits: {cache['cache_hits']}")
    print(f"  Cache Misses: {cache['cache_misses']}")


async def demonstrate_configuration_options():
    """Demonstrate configuration customization"""
    print("\n=== Configuration Demo ===")
    
    # Create custom configuration
    config = RouterConfiguration()
    
    # Customize extensions
    config.force_lsp_extensions.add('.custom')
    config.force_standard_extensions.add('.special')
    config.custom_language_map['.custom'] = 'custom_language'
    
    # Performance settings
    config.batch_size_limit = 50
    config.enable_caching = True
    config.cache_ttl_seconds = 1800.0  # 30 minutes
    
    print("Custom Configuration:")
    print(f"  Force LSP Extensions: {len(config.force_lsp_extensions)}")
    print(f"  Force Standard Extensions: {len(config.force_standard_extensions)}")
    print(f"  Custom Language Mappings: {len(config.custom_language_map)}")
    print(f"  Batch Size Limit: {config.batch_size_limit}")
    print(f"  Caching Enabled: {config.enable_caching}")
    print(f"  Cache TTL: {config.cache_ttl_seconds} seconds")
    
    # Create router with custom config
    custom_router = SmartIngestionRouter(config=config)
    await custom_router.initialize()
    
    print("  Custom Router Initialized: SUCCESS")
    
    await custom_router.shutdown()


async def main():
    """Main demonstration function"""
    print("Smart Ingestion Router - Comprehensive Demo")
    print("=" * 50)
    
    # Create temporary workspace
    with tempfile.TemporaryDirectory() as temp_dir:
        workspace = Path(temp_dir)
        print(f"Demo Workspace: {workspace}")
        
        # Create demonstration files
        demo_files = create_demo_files(workspace)
        print(f"Created {len(demo_files)} demo files")
        
        # Initialize router with default configuration
        router = SmartIngestionRouter()
        
        try:
            await router.initialize(workspace)
            print("Router Initialized: SUCCESS")
            
            # Run demonstrations
            await demonstrate_single_file_processing(router, demo_files)
            await demonstrate_batch_processing(router, demo_files)
            await demonstrate_statistics_and_capabilities(router)
            
        finally:
            await router.shutdown()
            print("\nRouter Shutdown: SUCCESS")
        
        # Demonstrate configuration options
        await demonstrate_configuration_options()
    
    print("\n" + "=" * 50)
    print("Smart Ingestion Router Demo Completed Successfully!")
    print("\nKey Features Demonstrated:")
    print("• Intelligent file classification using multiple detection methods")
    print("• Smart routing between LSP-enriched and standard processing")
    print("• Batch processing optimization with strategy grouping")
    print("• Comprehensive statistics and performance monitoring")
    print("• Flexible configuration and customization options")
    print("• Robust error handling and fallback mechanisms")


if __name__ == "__main__":
    asyncio.run(main())