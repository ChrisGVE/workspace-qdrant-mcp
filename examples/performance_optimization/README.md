# Performance Optimization Examples

Comprehensive performance optimization strategies for workspace-qdrant-mcp, including large dataset handling, memory optimization, and search performance tuning.

## 🎯 Overview

This section provides:

- **Large Dataset Strategies** - Techniques for handling massive document collections
- **Memory Optimization** - Efficient memory usage patterns and garbage collection
- **Search Performance** - Query optimization and indexing strategies
- **Batch Processing** - High-throughput document ingestion and updates
- **Monitoring & Profiling** - Performance monitoring and bottleneck identification
- **Scaling Strategies** - Horizontal and vertical scaling approaches

## 🏗️ Optimization Structure

```
performance_optimization/
├── README.md                    # This file
├── large_datasets/              # Large-scale data handling
│   ├── chunked_ingestion.py    # Chunked document processing
│   ├── streaming_processor.py   # Memory-efficient streaming
│   ├── parallel_indexing.py    # Parallel processing strategies
│   └── dataset_analyzer.py     # Dataset analysis and planning
├── memory_optimization/         # Memory efficiency techniques
│   ├── memory_profiler.py      # Memory usage monitoring
│   ├── garbage_collection.py   # GC optimization strategies
│   ├── object_pooling.py       # Object reuse patterns
│   └── cache_management.py     # Intelligent caching systems
├── search_optimization/         # Search performance tuning
│   ├── query_optimizer.py      # Query optimization techniques
│   ├── index_management.py     # Index optimization strategies
│   ├── result_caching.py       # Search result caching
│   └── hybrid_search_tuning.py # Hybrid search optimization
├── batch_operations/            # High-performance batch processing
│   ├── bulk_indexer.py         # Optimized bulk indexing
│   ├── concurrent_processor.py # Concurrent processing patterns
│   ├── pipeline_optimizer.py   # Processing pipeline optimization
│   └── throughput_monitor.py   # Throughput monitoring and tuning
└── monitoring/                  # Performance monitoring tools
    ├── performance_monitor.py   # Real-time performance monitoring
    ├── benchmark_suite.py       # Performance benchmarking tools
    ├── bottleneck_detector.py   # Automatic bottleneck detection
    └── optimization_advisor.py  # Performance optimization recommendations
```

## 🚀 Quick Setup

### 1. Install Performance Dependencies

```bash
# Navigate to performance optimization directory
cd examples/performance_optimization

# Install performance monitoring dependencies
pip install -r requirements.txt

# Additional optional dependencies for advanced features
pip install memory-profiler psutil line-profiler
```

### 2. Configure Performance Monitoring

```bash
# Set up performance monitoring configuration
cp config/performance_config.example.yaml config/performance_config.yaml

# Configure environment for performance testing
export WQ_PERFORMANCE_CONFIG="config/performance_config.yaml"
export WQ_ENABLE_PROFILING="true"
```

### 3. Run Performance Baseline

```bash
# Run baseline performance test
python monitoring/benchmark_suite.py --baseline

# Analyze current system performance
python monitoring/performance_monitor.py --analyze

# Get optimization recommendations
python monitoring/optimization_advisor.py --system-analysis
```

## 📚 Large Dataset Strategies

### Chunked Ingestion System

**Memory-efficient processing of massive datasets:**

```python
#!/usr/bin/env python3
"""
chunked_ingestion.py - Memory-efficient chunked document ingestion

Handles massive datasets by processing documents in manageable chunks,
preventing memory exhaustion and providing progress tracking.
"""

import gc
import os
import sys
import time
import psutil
from datetime import datetime
from pathlib import Path
from typing import Iterator, List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import queue

@dataclass
class ChunkStats:
    """Statistics for a processed chunk."""
    chunk_id: int
    documents_processed: int
    success_count: int
    error_count: int
    processing_time: float
    memory_usage_mb: float
    throughput_docs_per_sec: float

@dataclass
class IngestionConfig:
    """Configuration for chunked ingestion."""
    chunk_size: int = 1000
    max_workers: int = 4
    memory_limit_mb: int = 2048
    gc_threshold: int = 100  # Run GC every N chunks
    progress_interval: int = 10  # Report progress every N chunks
    error_tolerance: float = 0.05  # Max 5% error rate
    batch_commit_size: int = 100
    enable_compression: bool = True
    prefetch_chunks: int = 2

class MemoryMonitor:
    """Monitor memory usage during processing."""
    
    def __init__(self, limit_mb: int):
        self.limit_mb = limit_mb
        self.peak_usage = 0.0
        self.measurements = []
        self._lock = threading.Lock()
    
    def check_memory(self) -> Tuple[float, bool]:
        """Check current memory usage and return (usage_mb, over_limit)."""
        process = psutil.Process()
        memory_info = process.memory_info()
        current_mb = memory_info.rss / 1024 / 1024
        
        with self._lock:
            self.peak_usage = max(self.peak_usage, current_mb)
            self.measurements.append((datetime.now(), current_mb))
            
            # Keep only last 100 measurements
            if len(self.measurements) > 100:
                self.measurements = self.measurements[-100:]
        
        return current_mb, current_mb > self.limit_mb
    
    def get_stats(self) -> Dict[str, Any]:
        """Get memory usage statistics."""
        with self._lock:
            if not self.measurements:
                return {"error": "No measurements taken"}
            
            recent_usage = [m[1] for m in self.measurements[-10:]]
            return {
                "current_mb": self.measurements[-1][1] if self.measurements else 0,
                "peak_mb": self.peak_usage,
                "average_mb": sum(recent_usage) / len(recent_usage),
                "limit_mb": self.limit_mb,
                "measurement_count": len(self.measurements)
            }

class ChunkedIngestionProcessor:
    """
    High-performance chunked document ingestion system.
    
    Processes large datasets in memory-efficient chunks with parallel processing,
    memory monitoring, and automatic garbage collection.
    """
    
    def __init__(self, client, config: IngestionConfig = None):
        self.client = client
        self.config = config or IngestionConfig()
        self.memory_monitor = MemoryMonitor(self.config.memory_limit_mb)
        
        # Processing state
        self.total_processed = 0
        self.total_errors = 0
        self.start_time = None
        self.chunk_stats = []
        self.error_log = []
        
        # Thread safety
        self._stats_lock = threading.Lock()
        self._error_lock = threading.Lock()
    
    def process_dataset(self, data_source: Iterator[Dict[str, Any]], 
                       collection: str, total_estimate: int = None) -> Dict[str, Any]:
        """
        Process large dataset with chunked ingestion strategy.
        
        Args:
            data_source: Iterator yielding document dictionaries
            collection: Target collection name
            total_estimate: Estimated total documents (for progress tracking)
            
        Returns:
            Processing results and statistics
        """
        self.start_time = datetime.now()
        print(f"🚀 Starting chunked ingestion to collection '{collection}'")
        
        if total_estimate:
            print(f"📊 Estimated total documents: {total_estimate:,}")
        
        try:
            # Process data in chunks
            chunk_id = 0
            chunk_buffer = []
            
            for document in data_source:
                chunk_buffer.append(document)
                
                # Process chunk when buffer is full
                if len(chunk_buffer) >= self.config.chunk_size:
                    self._process_chunk(chunk_buffer, chunk_id, collection)
                    chunk_buffer = []
                    chunk_id += 1
                    
                    # Memory management
                    self._handle_memory_pressure()
                    
                    # Progress reporting
                    if chunk_id % self.config.progress_interval == 0:
                        self._report_progress(chunk_id, total_estimate)
            
            # Process final partial chunk
            if chunk_buffer:
                self._process_chunk(chunk_buffer, chunk_id, collection)
            
            # Final processing
            return self._generate_final_report()
            
        except Exception as e:
            print(f"❌ Ingestion failed: {str(e)}")
            return {"error": str(e), "partial_results": self._generate_partial_report()}
    
    def _process_chunk(self, documents: List[Dict[str, Any]], 
                      chunk_id: int, collection: str) -> ChunkStats:
        """Process a single chunk of documents."""
        chunk_start = time.time()
        memory_before, _ = self.memory_monitor.check_memory()
        
        print(f"📦 Processing chunk {chunk_id} ({len(documents)} documents)")
        
        # Process documents in parallel within the chunk
        success_count = 0
        error_count = 0
        
        with ThreadPoolExecutor(max_workers=min(self.config.max_workers, len(documents))) as executor:
            # Submit document processing tasks
            future_to_doc = {
                executor.submit(self._process_single_document, doc, collection): i
                for i, doc in enumerate(documents)
            }
            
            # Collect results
            for future in as_completed(future_to_doc):
                doc_index = future_to_doc[future]
                
                try:
                    success = future.result()
                    if success:
                        success_count += 1
                    else:
                        error_count += 1
                except Exception as e:
                    error_count += 1
                    with self._error_lock:
                        self.error_log.append(f"Chunk {chunk_id}, Doc {doc_index}: {str(e)}")
        
        # Calculate chunk statistics
        processing_time = time.time() - chunk_start
        memory_after, _ = self.memory_monitor.check_memory()
        throughput = len(documents) / processing_time if processing_time > 0 else 0
        
        chunk_stats = ChunkStats(
            chunk_id=chunk_id,
            documents_processed=len(documents),
            success_count=success_count,
            error_count=error_count,
            processing_time=processing_time,
            memory_usage_mb=memory_after,
            throughput_docs_per_sec=throughput
        )
        
        # Update global statistics
        with self._stats_lock:
            self.total_processed += len(documents)
            self.total_errors += error_count
            self.chunk_stats.append(chunk_stats)
        
        # Check error tolerance
        error_rate = error_count / len(documents)
        if error_rate > self.config.error_tolerance:
            print(f"⚠️  High error rate in chunk {chunk_id}: {error_rate:.2%}")
        
        print(f"✅ Chunk {chunk_id} complete: {success_count} success, {error_count} errors ({throughput:.1f} docs/sec)")
        
        return chunk_stats
    
    def _process_single_document(self, document: Dict[str, Any], 
                                collection: str) -> bool:
        """Process a single document."""
        try:
            # Extract content and metadata from document
            content = document.get('content', '')
            metadata = {k: v for k, v in document.items() if k != 'content'}
            
            # Add processing metadata
            metadata.update({
                'ingestion_timestamp': datetime.now().isoformat(),
                'processor_version': '2.0',
                'processing_mode': 'chunked_ingestion'
            })
            
            # Store document
            result = self.client.store(
                content=content,
                metadata=metadata,
                collection=collection
            )
            
            return result is not None
            
        except Exception as e:
            return False
    
    def _handle_memory_pressure(self):
        """Handle memory pressure during processing."""
        current_mb, over_limit = self.memory_monitor.check_memory()
        
        if over_limit:
            print(f"🔥 Memory pressure detected: {current_mb:.1f}MB (limit: {self.config.memory_limit_mb}MB)")
            
            # Force garbage collection
            collected = gc.collect()
            print(f"🧹 Garbage collection freed {collected} objects")
            
            # Check memory again after GC
            new_mb, still_over = self.memory_monitor.check_memory()
            if still_over:
                print(f"⚠️  Memory still high after GC: {new_mb:.1f}MB")
                # Could implement additional strategies here:
                # - Reduce worker count
                # - Reduce chunk size
                # - Trigger emergency cleanup
    
    def _report_progress(self, chunk_id: int, total_estimate: int = None):
        """Report processing progress."""
        elapsed = (datetime.now() - self.start_time).total_seconds()
        rate = self.total_processed / elapsed if elapsed > 0 else 0
        
        memory_stats = self.memory_monitor.get_stats()
        
        print(f"📊 Progress Report (Chunk {chunk_id})")
        print(f"   Documents processed: {self.total_processed:,}")
        print(f"   Processing rate: {rate:.1f} docs/sec")
        print(f"   Memory usage: {memory_stats['current_mb']:.1f}MB (peak: {memory_stats['peak_mb']:.1f}MB)")
        print(f"   Error rate: {(self.total_errors / self.total_processed * 100):.2f}%")
        
        if total_estimate:
            progress = (self.total_processed / total_estimate) * 100
            remaining = total_estimate - self.total_processed
            eta_seconds = remaining / rate if rate > 0 else 0
            print(f"   Progress: {progress:.1f}% ({remaining:,} remaining)")
            print(f"   ETA: {eta_seconds/60:.1f} minutes")
    
    def _generate_final_report(self) -> Dict[str, Any]:
        """Generate comprehensive final processing report."""
        total_time = (datetime.now() - self.start_time).total_seconds()
        overall_rate = self.total_processed / total_time if total_time > 0 else 0
        
        # Calculate chunk statistics
        chunk_times = [cs.processing_time for cs in self.chunk_stats]
        chunk_rates = [cs.throughput_docs_per_sec for cs in self.chunk_stats]
        
        memory_stats = self.memory_monitor.get_stats()
        
        report = {
            "success": True,
            "summary": {
                "total_processed": self.total_processed,
                "total_errors": self.total_errors,
                "error_rate": self.total_errors / self.total_processed if self.total_processed > 0 else 0,
                "processing_time_seconds": total_time,
                "overall_throughput_docs_per_sec": overall_rate,
                "chunks_processed": len(self.chunk_stats)
            },
            "performance": {
                "average_chunk_time": sum(chunk_times) / len(chunk_times) if chunk_times else 0,
                "fastest_chunk_rate": max(chunk_rates) if chunk_rates else 0,
                "slowest_chunk_rate": min(chunk_rates) if chunk_rates else 0,
                "memory_efficiency": memory_stats
            },
            "configuration": {
                "chunk_size": self.config.chunk_size,
                "max_workers": self.config.max_workers,
                "memory_limit_mb": self.config.memory_limit_mb,
                "error_tolerance": self.config.error_tolerance
            },
            "errors": self.error_log[:100],  # First 100 errors
            "chunk_details": [
                {
                    "chunk_id": cs.chunk_id,
                    "documents": cs.documents_processed,
                    "success_rate": cs.success_count / cs.documents_processed if cs.documents_processed > 0 else 0,
                    "throughput": cs.throughput_docs_per_sec,
                    "processing_time": cs.processing_time
                }
                for cs in self.chunk_stats
            ]
        }
        
        print(f"\n🎉 Ingestion Complete!")
        print(f"   Total processed: {self.total_processed:,} documents")
        print(f"   Processing time: {total_time:.1f} seconds")
        print(f"   Average throughput: {overall_rate:.1f} docs/sec")
        print(f"   Error rate: {(self.total_errors / self.total_processed * 100):.2f}%")
        print(f"   Peak memory usage: {memory_stats['peak_mb']:.1f}MB")
        
        return report
    
    def _generate_partial_report(self) -> Dict[str, Any]:
        """Generate partial report in case of failure."""
        return {
            "total_processed": self.total_processed,
            "total_errors": self.total_errors,
            "chunks_completed": len(self.chunk_stats),
            "last_error_log": self.error_log[-10:] if self.error_log else []
        }

# Data source generators for different input types

def file_document_generator(file_paths: List[Path], 
                           extract_metadata: bool = True) -> Iterator[Dict[str, Any]]:
    """Generate documents from a list of files."""
    for file_path in file_paths:
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            document = {"content": content}
            
            if extract_metadata:
                document.update({
                    "filename": file_path.name,
                    "file_path": str(file_path),
                    "file_size": file_path.stat().st_size,
                    "file_extension": file_path.suffix.lower(),
                    "created_date": datetime.fromtimestamp(file_path.stat().st_ctime).isoformat(),
                    "modified_date": datetime.fromtimestamp(file_path.stat().st_mtime).isoformat()
                })
            
            yield document
            
        except Exception as e:
            print(f"⚠️  Error reading {file_path}: {e}")
            continue

def json_lines_generator(file_path: Path) -> Iterator[Dict[str, Any]]:
    """Generate documents from a JSON Lines file."""
    import json
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                document = json.loads(line.strip())
                yield document
            except json.JSONDecodeError as e:
                print(f"⚠️  JSON decode error at line {line_num}: {e}")
                continue

def csv_document_generator(file_path: Path, content_column: str,
                          metadata_columns: List[str] = None) -> Iterator[Dict[str, Any]]:
    """Generate documents from CSV file."""
    try:
        import pandas as pd
        
        df = pd.read_csv(file_path)
        
        for _, row in df.iterrows():
            document = {"content": str(row[content_column])}
            
            if metadata_columns:
                for col in metadata_columns:
                    if col in row:
                        document[col] = row[col]
            
            yield document
            
    except ImportError:
        print("❌ pandas required for CSV processing")
        return

# Example usage and CLI interface
if __name__ == "__main__":
    import argparse
    from workspace_qdrant_mcp.client import WorkspaceClient
    
    parser = argparse.ArgumentParser(description="Chunked ingestion processor for large datasets")
    parser.add_argument("--source-type", choices=["files", "jsonl", "csv"], required=True,
                       help="Type of data source")
    parser.add_argument("--source-path", required=True, help="Path to data source")
    parser.add_argument("--collection", "-c", required=True, help="Target collection")
    parser.add_argument("--chunk-size", type=int, default=1000, help="Documents per chunk")
    parser.add_argument("--workers", type=int, default=4, help="Worker threads")
    parser.add_argument("--memory-limit", type=int, default=2048, help="Memory limit in MB")
    parser.add_argument("--total-estimate", type=int, help="Estimated total documents")
    
    # CSV-specific arguments
    parser.add_argument("--content-column", help="CSV column containing document content")
    parser.add_argument("--metadata-columns", help="Comma-separated list of metadata columns")
    
    args = parser.parse_args()
    
    # Create configuration
    config = IngestionConfig(
        chunk_size=args.chunk_size,
        max_workers=args.workers,
        memory_limit_mb=args.memory_limit
    )
    
    # Initialize client and processor
    client = WorkspaceClient()
    processor = ChunkedIngestionProcessor(client, config)
    
    # Create data generator based on source type
    if args.source_type == "files":
        source_path = Path(args.source_path)
        if source_path.is_file():
            file_paths = [source_path]
        else:
            file_paths = list(source_path.rglob("*"))
            file_paths = [f for f in file_paths if f.is_file()]
        
        data_generator = file_document_generator(file_paths)
        
    elif args.source_type == "jsonl":
        data_generator = json_lines_generator(Path(args.source_path))
        
    elif args.source_type == "csv":
        if not args.content_column:
            print("❌ --content-column required for CSV processing")
            sys.exit(1)
        
        metadata_cols = args.metadata_columns.split(',') if args.metadata_columns else None
        data_generator = csv_document_generator(
            Path(args.source_path),
            args.content_column,
            metadata_cols
        )
    
    # Process the dataset
    results = processor.process_dataset(
        data_source=data_generator,
        collection=args.collection,
        total_estimate=args.total_estimate
    )
    
    # Save detailed results
    if results.get("success"):
        results_file = f"ingestion_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        import json
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"📄 Detailed results saved to {results_file}")
    else:
        print("❌ Ingestion failed - check error logs")
```

### Search Performance Optimization

**Advanced search optimization and query tuning:**

```python
#!/usr/bin/env python3
"""
search_optimization.py - Advanced search performance optimization

Comprehensive search optimization including query analysis, result caching,
and performance tuning for high-traffic search scenarios.
"""

import time
import hashlib
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from collections import OrderedDict
import statistics

@dataclass
class SearchMetrics:
    """Metrics for search performance analysis."""
    query: str
    response_time_ms: float
    result_count: int
    cache_hit: bool
    timestamp: datetime
    collection: str
    filters: Dict[str, Any] = field(default_factory=dict)
    hybrid_weights: Optional[Tuple[float, float]] = None

@dataclass
class QueryPattern:
    """Analysis of query patterns for optimization."""
    query_hash: str
    frequency: int
    avg_response_time: float
    result_count_variance: float
    collections_used: Set[str]
    common_filters: Dict[str, Any]
    optimization_suggestions: List[str] = field(default_factory=list)

class SearchCache:
    """High-performance search result cache with TTL and LRU eviction."""
    
    def __init__(self, max_size: int = 1000, ttl_minutes: int = 30):
        self.max_size = max_size
        self.ttl_delta = timedelta(minutes=ttl_minutes)
        self.cache = OrderedDict()
        self.timestamps = {}
        self._lock = threading.RLock()
        self.stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "size": 0
        }
    
    def _generate_key(self, query: str, collection: str, 
                     metadata_filter: Dict[str, Any] = None,
                     limit: int = 10) -> str:
        """Generate cache key for search parameters."""
        key_components = [
            query,
            collection or "all",
            str(sorted((metadata_filter or {}).items())),
            str(limit)
        ]
        key_string = "|".join(key_components)
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def get(self, query: str, collection: str, 
            metadata_filter: Dict[str, Any] = None,
            limit: int = 10) -> Optional[List[Dict[str, Any]]]:
        """Get cached search results if available and not expired."""
        key = self._generate_key(query, collection, metadata_filter, limit)
        
        with self._lock:
            # Check if key exists and is not expired
            if key in self.cache and key in self.timestamps:
                if datetime.now() - self.timestamps[key] < self.ttl_delta:
                    # Move to end (LRU)
                    self.cache.move_to_end(key)
                    self.stats["hits"] += 1
                    return self.cache[key]
                else:
                    # Expired - remove
                    del self.cache[key]
                    del self.timestamps[key]
                    self.stats["evictions"] += 1
            
            self.stats["misses"] += 1
            return None
    
    def put(self, query: str, collection: str, results: List[Dict[str, Any]],
            metadata_filter: Dict[str, Any] = None, limit: int = 10):
        """Cache search results."""
        key = self._generate_key(query, collection, metadata_filter, limit)
        
        with self._lock:
            # Remove oldest entries if at capacity
            while len(self.cache) >= self.max_size:
                oldest_key = next(iter(self.cache))
                del self.cache[oldest_key]
                del self.timestamps[oldest_key]
                self.stats["evictions"] += 1
            
            self.cache[key] = results
            self.timestamps[key] = datetime.now()
            self.stats["size"] = len(self.cache)
    
    def clear(self):
        """Clear all cached results."""
        with self._lock:
            self.cache.clear()
            self.timestamps.clear()
            self.stats = {"hits": 0, "misses": 0, "evictions": 0, "size": 0}
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics."""
        with self._lock:
            total_requests = self.stats["hits"] + self.stats["misses"]
            hit_rate = self.stats["hits"] / total_requests if total_requests > 0 else 0
            
            return {
                **self.stats,
                "hit_rate": hit_rate,
                "total_requests": total_requests,
                "ttl_minutes": self.ttl_delta.total_seconds() / 60
            }

class QueryOptimizer:
    """Advanced query optimization and analysis system."""
    
    def __init__(self):
        self.stop_words = {
            "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
            "of", "with", "by", "is", "are", "was", "were", "be", "been", "have",
            "has", "had", "do", "does", "did", "will", "would", "could", "should"
        }
        self.query_expansions = {
            "ml": "machine learning",
            "ai": "artificial intelligence",
            "nlp": "natural language processing",
            "db": "database",
            "api": "application programming interface"
        }
    
    def optimize_query(self, query: str) -> Tuple[str, Dict[str, Any]]:
        """Optimize search query for better performance and results."""
        original_query = query
        optimizations = []
        
        # 1. Expand abbreviations
        expanded_query = self._expand_abbreviations(query)
        if expanded_query != query:
            optimizations.append("abbreviation_expansion")
            query = expanded_query
        
        # 2. Remove excessive stop words (but keep some for context)
        filtered_query = self._filter_stop_words(query)
        if filtered_query != query:
            optimizations.append("stop_word_filtering")
            query = filtered_query
        
        # 3. Add quotes for exact phrases (heuristic)
        phrase_query = self._identify_phrases(query)
        if phrase_query != query:
            optimizations.append("phrase_identification")
            query = phrase_query
        
        # 4. Boost important terms
        boosted_query, boosts = self._add_term_boosts(query)
        if boosted_query != query:
            optimizations.append("term_boosting")
            query = boosted_query
        
        optimization_info = {
            "original_query": original_query,
            "optimized_query": query,
            "optimizations_applied": optimizations,
            "term_boosts": boosts if 'term_boosting' in optimizations else {},
            "estimated_improvement": len(optimizations) * 0.1  # Rough estimate
        }
        
        return query, optimization_info
    
    def _expand_abbreviations(self, query: str) -> str:
        """Expand common abbreviations in query."""
        words = query.lower().split()
        expanded_words = []
        
        for word in words:
            if word in self.query_expansions:
                expanded_words.append(self.query_expansions[word])
            else:
                expanded_words.append(word)
        
        return " ".join(expanded_words)
    
    def _filter_stop_words(self, query: str) -> str:
        """Remove excessive stop words while preserving meaning."""
        words = query.lower().split()
        
        # Keep stop words if query is very short
        if len(words) <= 3:
            return query
        
        # Remove stop words but keep at least 2 content words
        content_words = [w for w in words if w not in self.stop_words]
        
        if len(content_words) >= 2:
            return " ".join(content_words)
        else:
            return query  # Don't over-filter
    
    def _identify_phrases(self, query: str) -> str:
        """Identify and quote likely phrases."""
        # Simple heuristics for phrase identification
        phrases_patterns = [
            "machine learning",
            "artificial intelligence", 
            "natural language processing",
            "deep learning",
            "data science",
            "software engineering"
        ]
        
        query_lower = query.lower()
        for phrase in phrases_patterns:
            if phrase in query_lower and f'"{phrase}"' not in query_lower:
                query = query.replace(phrase, f'"{phrase}"')
        
        return query
    
    def _add_term_boosts(self, query: str) -> Tuple[str, Dict[str, float]]:
        """Add term-specific boosts for better relevance."""
        # Terms that should be boosted for technical searches
        boost_terms = {
            "python": 1.5,
            "javascript": 1.5,
            "machine learning": 1.3,
            "algorithm": 1.3,
            "performance": 1.2,
            "optimization": 1.2
        }
        
        boosts_applied = {}
        boosted_query = query
        
        for term, boost in boost_terms.items():
            if term.lower() in query.lower():
                # This is a simplified boost notation - actual implementation
                # would depend on the search engine's query syntax
                boosted_query = boosted_query.replace(term, f"{term}^{boost}")
                boosts_applied[term] = boost
        
        return boosted_query, boosts_applied

class SearchPerformanceOptimizer:
    """
    Comprehensive search performance optimization system.
    
    Provides query optimization, result caching, performance monitoring,
    and automatic tuning for high-performance search scenarios.
    """
    
    def __init__(self, client, cache_size: int = 1000, cache_ttl_minutes: int = 30):
        self.client = client
        self.cache = SearchCache(cache_size, cache_ttl_minutes)
        self.query_optimizer = QueryOptimizer()
        
        # Performance tracking
        self.search_metrics = []
        self.query_patterns = {}
        self._metrics_lock = threading.Lock()
        
        # Configuration
        self.enable_caching = True
        self.enable_query_optimization = True
        self.enable_metrics_collection = True
        self.slow_query_threshold_ms = 1000
    
    def search(self, query: str, collection: str = None,
               metadata_filter: Dict[str, Any] = None,
               limit: int = 10, force_refresh: bool = False) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Optimized search with caching and performance monitoring.
        
        Args:
            query: Search query
            collection: Collection to search (None for all)
            metadata_filter: Metadata filtering criteria
            limit: Maximum results to return
            force_refresh: Skip cache and force fresh search
            
        Returns:
            Tuple of (results, performance_info)
        """
        search_start = time.time()
        performance_info = {
            "cache_hit": False,
            "query_optimized": False,
            "response_time_ms": 0,
            "result_count": 0
        }
        
        original_query = query
        
        # 1. Query Optimization
        if self.enable_query_optimization:
            optimized_query, optimization_info = self.query_optimizer.optimize_query(query)
            if optimized_query != query:
                query = optimized_query
                performance_info["query_optimized"] = True
                performance_info["optimization_info"] = optimization_info
        
        # 2. Cache Check
        results = None
        if self.enable_caching and not force_refresh:
            results = self.cache.get(query, collection or "all", metadata_filter, limit)
            if results is not None:
                performance_info["cache_hit"] = True
                performance_info["result_count"] = len(results)
                performance_info["response_time_ms"] = (time.time() - search_start) * 1000
                
                # Record metrics
                self._record_metrics(original_query, performance_info, collection)
                return results, performance_info
        
        # 3. Execute Search
        search_execution_start = time.time()
        
        try:
            results = self.client.search(
                query=query,
                collection=collection,
                metadata_filter=metadata_filter,
                limit=limit
            )
            
            search_execution_time = (time.time() - search_execution_start) * 1000
            performance_info["search_execution_time_ms"] = search_execution_time
            
        except Exception as e:
            performance_info["error"] = str(e)
            total_time = (time.time() - search_start) * 1000
            performance_info["response_time_ms"] = total_time
            return [], performance_info
        
        # 4. Cache Results
        if self.enable_caching and results:
            self.cache.put(query, collection or "all", results, metadata_filter, limit)
        
        # 5. Calculate Final Performance Metrics
        total_time = (time.time() - search_start) * 1000
        performance_info["response_time_ms"] = total_time
        performance_info["result_count"] = len(results)
        
        # 6. Record Metrics
        self._record_metrics(original_query, performance_info, collection)
        
        # 7. Check for Slow Queries
        if total_time > self.slow_query_threshold_ms:
            self._handle_slow_query(original_query, total_time, collection, metadata_filter)
        
        return results, performance_info
    
    def _record_metrics(self, query: str, performance_info: Dict[str, Any], 
                       collection: str = None):
        """Record search metrics for analysis."""
        if not self.enable_metrics_collection:
            return
        
        metric = SearchMetrics(
            query=query,
            response_time_ms=performance_info["response_time_ms"],
            result_count=performance_info["result_count"],
            cache_hit=performance_info["cache_hit"],
            timestamp=datetime.now(),
            collection=collection or "all"
        )
        
        with self._metrics_lock:
            self.search_metrics.append(metric)
            
            # Keep only last 10000 metrics to prevent memory growth
            if len(self.search_metrics) > 10000:
                self.search_metrics = self.search_metrics[-5000:]
            
            # Update query patterns
            self._update_query_patterns(query, performance_info["response_time_ms"])
    
    def _update_query_patterns(self, query: str, response_time: float):
        """Update query pattern analysis."""
        # Create query hash for pattern matching (normalize query)
        normalized_query = " ".join(sorted(query.lower().split()))
        query_hash = hashlib.md5(normalized_query.encode()).hexdigest()
        
        if query_hash not in self.query_patterns:
            self.query_patterns[query_hash] = {
                "frequency": 0,
                "response_times": [],
                "example_query": query
            }
        
        pattern = self.query_patterns[query_hash]
        pattern["frequency"] += 1
        pattern["response_times"].append(response_time)
        
        # Keep only last 100 response times per pattern
        if len(pattern["response_times"]) > 100:
            pattern["response_times"] = pattern["response_times"][-50:]
    
    def _handle_slow_query(self, query: str, response_time: float,
                          collection: str = None, metadata_filter: Dict[str, Any] = None):
        """Handle slow query detection and logging."""
        print(f"🐌 Slow query detected ({response_time:.1f}ms): {query[:100]}...")
        
        # Log slow query details
        slow_query_info = {
            "query": query,
            "response_time_ms": response_time,
            "collection": collection,
            "metadata_filter": metadata_filter,
            "timestamp": datetime.now().isoformat(),
            "threshold_ms": self.slow_query_threshold_ms
        }
        
        # In production, you'd want to log this to a file or monitoring system
        # For now, just print it
        print(f"   Collection: {collection}")
        print(f"   Filters: {metadata_filter}")
        print(f"   Suggestion: Consider query optimization or indexing")
    
    def get_performance_report(self, hours: int = 24) -> Dict[str, Any]:
        """Generate comprehensive performance report."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        with self._metrics_lock:
            recent_metrics = [m for m in self.search_metrics if m.timestamp >= cutoff_time]
        
        if not recent_metrics:
            return {"error": "No metrics available for the specified time period"}
        
        # Calculate statistics
        response_times = [m.response_time_ms for m in recent_metrics]
        cache_hits = sum(1 for m in recent_metrics if m.cache_hit)
        
        # Query pattern analysis
        popular_patterns = []
        for query_hash, pattern in self.query_patterns.items():
            if pattern["frequency"] >= 5:  # Patterns with 5+ occurrences
                avg_time = statistics.mean(pattern["response_times"])
                popular_patterns.append({
                    "example_query": pattern["example_query"],
                    "frequency": pattern["frequency"],
                    "avg_response_time_ms": avg_time,
                    "performance_grade": "Good" if avg_time < 500 else "Needs Optimization"
                })
        
        popular_patterns.sort(key=lambda x: x["frequency"], reverse=True)
        
        report = {
            "time_period_hours": hours,
            "summary": {
                "total_searches": len(recent_metrics),
                "avg_response_time_ms": statistics.mean(response_times),
                "median_response_time_ms": statistics.median(response_times),
                "p95_response_time_ms": self._percentile(response_times, 95),
                "p99_response_time_ms": self._percentile(response_times, 99),
                "cache_hit_rate": cache_hits / len(recent_metrics),
                "slow_queries": len([m for m in recent_metrics if m.response_time_ms > self.slow_query_threshold_ms])
            },
            "cache_performance": self.cache.get_stats(),
            "popular_query_patterns": popular_patterns[:10],
            "performance_trends": self._analyze_performance_trends(recent_metrics),
            "optimization_recommendations": self._generate_optimization_recommendations(recent_metrics)
        }
        
        return report
    
    def _percentile(self, values: List[float], percentile: int) -> float:
        """Calculate percentile of values."""
        if not values:
            return 0.0
        
        sorted_values = sorted(values)
        k = (len(sorted_values) - 1) * (percentile / 100)
        f = int(k)
        c = k - f
        
        if f == len(sorted_values) - 1:
            return sorted_values[f]
        
        return sorted_values[f] * (1 - c) + sorted_values[f + 1] * c
    
    def _analyze_performance_trends(self, metrics: List[SearchMetrics]) -> Dict[str, Any]:
        """Analyze performance trends over time."""
        # Group metrics by hour
        hourly_performance = {}
        
        for metric in metrics:
            hour_key = metric.timestamp.strftime('%Y-%m-%d %H:00')
            if hour_key not in hourly_performance:
                hourly_performance[hour_key] = []
            hourly_performance[hour_key].append(metric.response_time_ms)
        
        # Calculate hourly averages
        hourly_averages = {}
        for hour, times in hourly_performance.items():
            hourly_averages[hour] = statistics.mean(times)
        
        # Determine trend
        if len(hourly_averages) >= 2:
            recent_hours = sorted(hourly_averages.keys())[-2:]
            if len(recent_hours) == 2:
                trend = "improving" if hourly_averages[recent_hours[1]] < hourly_averages[recent_hours[0]] else "degrading"
            else:
                trend = "stable"
        else:
            trend = "insufficient_data"
        
        return {
            "trend": trend,
            "hourly_averages": hourly_averages,
            "peak_hour": max(hourly_averages, key=hourly_averages.get) if hourly_averages else None,
            "best_hour": min(hourly_averages, key=hourly_averages.get) if hourly_averages else None
        }
    
    def _generate_optimization_recommendations(self, metrics: List[SearchMetrics]) -> List[str]:
        """Generate optimization recommendations based on metrics."""
        recommendations = []
        
        response_times = [m.response_time_ms for m in metrics]
        cache_hit_rate = sum(1 for m in metrics if m.cache_hit) / len(metrics)
        avg_response_time = statistics.mean(response_times)
        slow_query_rate = len([m for m in metrics if m.response_time_ms > self.slow_query_threshold_ms]) / len(metrics)
        
        # Cache optimization
        if cache_hit_rate < 0.3:
            recommendations.append("Increase cache size or TTL - low cache hit rate detected")
        
        # Response time optimization
        if avg_response_time > 800:
            recommendations.append("Consider query optimization or index tuning - high average response time")
        
        # Slow query optimization
        if slow_query_rate > 0.1:
            recommendations.append("Optimize slow queries - high percentage of queries exceed threshold")
        
        # Collection-specific recommendations
        collection_performance = {}
        for metric in metrics:
            if metric.collection not in collection_performance:
                collection_performance[metric.collection] = []
            collection_performance[metric.collection].append(metric.response_time_ms)
        
        for collection, times in collection_performance.items():
            avg_time = statistics.mean(times)
            if avg_time > 1000:
                recommendations.append(f"Review indexing strategy for collection '{collection}' - poor performance detected")
        
        # Memory optimization
        if len(self.search_metrics) > 8000:
            recommendations.append("Consider reducing metrics retention to prevent memory growth")
        
        if not recommendations:
            recommendations.append("Performance looks good! Continue monitoring for any changes.")
        
        return recommendations
    
    def benchmark_search_performance(self, test_queries: List[str], 
                                   iterations: int = 10) -> Dict[str, Any]:
        """Run comprehensive search performance benchmark."""
        print(f"🏁 Running search benchmark ({iterations} iterations per query)")
        
        benchmark_results = {
            "test_queries": len(test_queries),
            "iterations_per_query": iterations,
            "start_time": datetime.now().isoformat(),
            "query_results": [],
            "summary": {}
        }
        
        all_response_times = []
        
        for i, query in enumerate(test_queries):
            print(f"   Testing query {i+1}/{len(test_queries)}: {query[:50]}...")
            
            query_times = []
            
            # Run multiple iterations
            for iteration in range(iterations):
                start_time = time.time()
                results, perf_info = self.search(query, force_refresh=True)
                response_time = (time.time() - start_time) * 1000
                
                query_times.append(response_time)
                all_response_times.append(response_time)
            
            # Calculate statistics for this query
            query_stats = {
                "query": query,
                "avg_time_ms": statistics.mean(query_times),
                "min_time_ms": min(query_times),
                "max_time_ms": max(query_times),
                "std_dev_ms": statistics.stdev(query_times) if len(query_times) > 1 else 0,
                "median_time_ms": statistics.median(query_times),
                "result_count": len(results) if 'results' in locals() else 0
            }
            
            benchmark_results["query_results"].append(query_stats)
        
        # Calculate overall statistics
        benchmark_results["summary"] = {
            "total_queries_tested": len(test_queries) * iterations,
            "overall_avg_time_ms": statistics.mean(all_response_times),
            "overall_median_time_ms": statistics.median(all_response_times),
            "overall_min_time_ms": min(all_response_times),
            "overall_max_time_ms": max(all_response_times),
            "p95_time_ms": self._percentile(all_response_times, 95),
            "p99_time_ms": self._percentile(all_response_times, 99),
            "queries_per_second": len(all_response_times) / (sum(all_response_times) / 1000),
            "end_time": datetime.now().isoformat()
        }
        
        print(f"✅ Benchmark complete!")
        print(f"   Overall average: {benchmark_results['summary']['overall_avg_time_ms']:.1f}ms")
        print(f"   P95 response time: {benchmark_results['summary']['p95_time_ms']:.1f}ms")
        print(f"   Throughput: {benchmark_results['summary']['queries_per_second']:.1f} queries/sec")
        
        return benchmark_results

# Example usage and testing
if __name__ == "__main__":
    from workspace_qdrant_mcp.client import WorkspaceClient
    
    # Initialize client and optimizer
    client = WorkspaceClient()
    optimizer = SearchPerformanceOptimizer(client, cache_size=500, cache_ttl_minutes=15)
    
    # Example test queries for benchmarking
    test_queries = [
        "machine learning algorithms",
        "python data processing",
        "API documentation examples",
        "database optimization techniques",
        "software architecture patterns",
        "natural language processing models",
        "web development frameworks",
        "cloud computing services",
        "cybersecurity best practices",
        "mobile app development"
    ]
    
    # Run benchmark
    results = optimizer.benchmark_search_performance(test_queries, iterations=5)
    
    # Display results
    print(f"\n📊 Benchmark Summary:")
    summary = results["summary"]
    print(f"   Total queries: {summary['total_queries_tested']}")
    print(f"   Average response time: {summary['overall_avg_time_ms']:.1f}ms")
    print(f"   P95 response time: {summary['p95_time_ms']:.1f}ms")
    print(f"   Throughput: {summary['queries_per_second']:.1f} queries/sec")
    
    # Generate performance report
    print(f"\n📈 Performance Report:")
    report = optimizer.get_performance_report(hours=1)
    print(f"   Cache hit rate: {report['cache_performance']['hit_rate']:.1%}")
    print(f"   Recommendations:")
    for rec in report['optimization_recommendations']:
        print(f"     • {rec}")
```

## 💡 Best Practices

### Performance Optimization Guidelines

**Large Dataset Processing:**
1. **Use chunked ingestion** for datasets > 10,000 documents
2. **Monitor memory usage** and implement garbage collection
3. **Process in parallel** with appropriate worker thread counts
4. **Implement progress tracking** for long-running operations
5. **Use streaming processors** for very large datasets

**Search Performance:**
1. **Implement result caching** for frequently accessed queries
2. **Optimize query structure** with query analysis and rewriting
3. **Use hybrid search strategically** based on content type
4. **Monitor slow queries** and optimize problematic patterns
5. **Implement connection pooling** for high-traffic scenarios

**Memory Management:**
1. **Set appropriate memory limits** for processing operations
2. **Use object pooling** for frequently created/destroyed objects
3. **Implement proper cleanup** in long-running processes
4. **Monitor garbage collection** performance
5. **Use memory profiling** to identify leaks

### Monitoring and Alerting

**Performance Monitoring Setup:**
```bash
# Daily performance monitoring
0 9 * * * python monitoring/performance_monitor.py --daily-report

# Real-time alerting for slow queries
* * * * * python monitoring/bottleneck_detector.py --alert-threshold 2000

# Weekly optimization recommendations
0 8 * * 1 python monitoring/optimization_advisor.py --weekly-review
```

### Scaling Strategies

**Horizontal Scaling:**
1. **Distribute collections** across multiple Qdrant instances
2. **Load balance search queries** across replicas
3. **Partition large collections** by metadata criteria
4. **Implement read replicas** for high-read scenarios
5. **Use consistent hashing** for data distribution

**Vertical Scaling:**
1. **Optimize hardware resources** (CPU, RAM, SSD)
2. **Tune Qdrant configuration** for your workload
3. **Use appropriate vector dimensions** for embeddings
4. **Optimize batch sizes** for your hardware
5. **Configure proper indexing** strategies

## 🔗 Integration Examples

- **[Automation Scripts](../automation/README.md)** - Automated performance optimization
- **[VS Code Integration](../vscode/README.md)** - Development environment optimization  
- **[Software Development](../../software_development/README.md)** - Code performance patterns

---

**Next Steps:**
1. Implement [Large Dataset Processing](large_datasets/) for your use case
2. Set up [Performance Monitoring](monitoring/) for production systems
3. Configure [Search Optimization](search_optimization/) for high-traffic scenarios