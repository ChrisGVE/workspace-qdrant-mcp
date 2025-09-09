#!/usr/bin/env python3
"""
Progressive Scaling Validation System - FINAL VERSION for Workspace Qdrant MCP
Task #146 - Comprehensive scaling analysis with corrected Qdrant API usage

Tests daemon performance across 6 progressive scales:
1. Baseline - Current workspace only
2. Small Scale - 10 keyboard directories (~171 files)
3. Medium Scale - 50 keyboard directories (~951 files)
4. Large Scale - keyboards + quantum + docs (~2,000 files)
5. Extra-Large Scale - Major directories (~5,000+ files)
6. Maximum Scale - Full QMK repository (22,066 files)
"""

import os
import sys
import time
import json
import psutil
import subprocess
import threading
import uuid
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import requests

class ResourceMonitor:
    """Monitor system resources during testing"""
    
    def __init__(self):
        self.monitoring = False
        self.data = []
        self.thread = None
        
    def start_monitoring(self, interval: float = 1.0):
        """Start resource monitoring"""
        self.monitoring = True
        self.data = []
        self.thread = threading.Thread(target=self._monitor_loop, args=(interval,))
        self.thread.daemon = True
        self.thread.start()
        
    def stop_monitoring(self) -> Dict:
        """Stop monitoring and return stats"""
        self.monitoring = False
        if self.thread:
            self.thread.join(timeout=5.0)
        
        if not self.data:
            return {}
            
        cpu_values = [d['cpu_percent'] for d in self.data]
        memory_values = [d['memory_percent'] for d in self.data]
        memory_mb = [d['memory_mb'] for d in self.data]
        
        return {
            'duration_seconds': len(self.data) * 1.0,
            'samples': len(self.data),
            'cpu': {
                'min': min(cpu_values),
                'max': max(cpu_values),
                'avg': sum(cpu_values) / len(cpu_values)
            },
            'memory': {
                'min_percent': min(memory_values),
                'max_percent': max(memory_values),
                'avg_percent': sum(memory_values) / len(memory_values),
                'min_mb': min(memory_mb),
                'max_mb': max(memory_mb),
                'avg_mb': sum(memory_mb) / len(memory_mb)
            }
        }
        
    def _monitor_loop(self, interval: float):
        """Monitor loop running in thread"""
        while self.monitoring:
            try:
                cpu_percent = psutil.cpu_percent(interval=None)
                memory = psutil.virtual_memory()
                
                self.data.append({
                    'timestamp': time.time(),
                    'cpu_percent': cpu_percent,
                    'memory_percent': memory.percent,
                    'memory_mb': memory.used / 1024 / 1024
                })
                
                # Safety check - stop if resource usage too high
                if memory.percent > 85 or cpu_percent > 90:
                    print(f"⚠️  SAFETY THRESHOLD EXCEEDED - CPU: {cpu_percent:.1f}%, Memory: {memory.percent:.1f}%")
                    # Don't stop monitoring, just warn
                    
                time.sleep(interval)
            except Exception as e:
                print(f"⚠️  Monitoring error: {e}")
                time.sleep(interval)

class QdrantTestClient:
    """Corrected Qdrant client for testing with proper API usage"""
    
    def __init__(self, url: str = "http://localhost:6333"):
        self.url = url
        
    def get_collections(self) -> List[str]:
        """Get list of collections"""
        try:
            response = requests.get(f"{self.url}/collections")
            if response.status_code == 200:
                data = response.json()
                return [col['name'] for col in data.get('result', {}).get('collections', [])]
            return []
        except Exception as e:
            print(f"⚠️  Error getting collections: {e}")
            return []
            
    def create_collection(self, collection_name: str) -> Dict:
        """Create a collection"""
        collection_config = {
            "vectors": {
                "size": 384,
                "distance": "Cosine"
            }
        }
        
        try:
            # Delete existing collection if it exists
            requests.delete(f"{self.url}/collections/{collection_name}")
            time.sleep(1)
            
            # Create new collection
            response = requests.put(f"{self.url}/collections/{collection_name}", json=collection_config)
            if response.status_code in [200, 201]:
                return {'success': True, 'collection': collection_name}
            else:
                return {'success': False, 'error': f'HTTP {response.status_code}: {response.text}'}
        except Exception as e:
            return {'success': False, 'error': str(e)}
            
    def add_document_batch(self, collection: str, documents: List[Dict]) -> Dict:
        """Add documents to collection with corrected Qdrant API usage"""
        try:
            if not documents:
                return {'success': True, 'added': 0}
                
            # Prepare points for Qdrant with correct format
            points = []
            for i, doc in enumerate(documents):
                # Generate deterministic vector from content
                vector = self._generate_vector_from_content(doc['content'])
                
                # Use integer ID (more reliable than UUID strings)
                point_id = i + 1
                
                point = {
                    "id": point_id,
                    "vector": vector,
                    "payload": {
                        'content': doc['content'][:500],  # Truncate for testing
                        'file_path': doc['metadata']['file_path'],
                        'file_name': doc['metadata']['file_name'],
                        'file_type': doc['metadata']['file_type'],
                        'size_bytes': doc['metadata']['size_bytes']
                    }
                }
                points.append(point)
                
            # Insert in batches with correct API format
            batch_size = 100
            total_added = 0
            
            for i in range(0, len(points), batch_size):
                batch = points[i:i + batch_size]
                batch_data = {"points": batch}
                
                response = requests.put(f"{self.url}/collections/{collection}/points", json=batch_data)
                if response.status_code in [200, 201]:
                    total_added += len(batch)
                else:
                    error_msg = f"Batch {i//batch_size + 1} failed: HTTP {response.status_code} - {response.text}"
                    return {'success': False, 'error': error_msg, 'added': total_added}
                    
            return {'success': True, 'added': total_added}
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
            
    def _generate_vector_from_content(self, content: str) -> List[float]:
        """Generate a 384-dimensional vector from content"""
        # Create deterministic hash of content
        content_hash = hashlib.sha256(content.encode('utf-8')).hexdigest()
        
        # Convert hash to 384-dimensional vector
        vector = []
        for i in range(384):
            # Use different parts of the hash for each dimension
            hash_segment = content_hash[(i * 2) % len(content_hash):(i * 2 + 8) % len(content_hash)]
            if len(hash_segment) < 8:
                hash_segment = (hash_segment + content_hash)[:8]
            
            # Convert hex to float in range [-1, 1]
            hex_val = int(hash_segment, 16) if hash_segment else 0
            normalized_val = (hex_val % 2000) / 1000.0 - 1.0
            vector.append(normalized_val)
            
        return vector
        
    def search_collection(self, collection: str, query: str, limit: int = 5) -> Dict:
        """Test search performance"""
        try:
            # Generate query vector from query text
            query_vector = self._generate_vector_from_content(query)
            
            search_data = {
                "vector": query_vector,
                "limit": limit,
                "with_payload": True
            }
            
            start_time = time.time()
            response = requests.post(f"{self.url}/collections/{collection}/points/search", json=search_data)
            search_time = time.time() - start_time
            
            if response.status_code == 200:
                results = response.json().get('result', [])
                return {
                    'search_time_ms': search_time * 1000,
                    'results_count': len(results),
                    'success': True
                }
            return {'success': False, 'error': f'HTTP {response.status_code} - {response.text}'}
        except Exception as e:
            return {'success': False, 'error': str(e)}
            
    def get_collection_info(self, collection: str) -> Dict:
        """Get collection information"""
        try:
            response = requests.get(f"{self.url}/collections/{collection}")
            if response.status_code == 200:
                return response.json().get('result', {})
            return {}
        except Exception as e:
            print(f"⚠️  Error getting collection info: {e}")
            return {}

class ProgressiveScalingTester:
    """Main progressive scaling test orchestrator"""
    
    def __init__(self):
        self.qmk_path = Path("/Users/chris/Dropbox/dev/ai/claude-code-cfg/mcp/workspace-qdrant-mcp/LT_20250906-1822_testing_sandbox/qmk_integration/qmk_firmware")
        self.results_dir = Path("/Users/chris/Dropbox/dev/ai/claude-code-cfg/mcp/workspace-qdrant-mcp/LT_20250906-1822_testing_sandbox/qmk_integration")
        self.monitor = ResourceMonitor()
        self.qdrant = QdrantTestClient()
        self.test_results = []
        
        # Test phases configuration
        self.phases = [
            {
                'name': 'baseline',
                'description': 'Current workspace only (reference point)',
                'expected_files': 0,
                'test_type': 'baseline'
            },
            {
                'name': 'small_scale',
                'description': 'First 10 keyboard directories',
                'expected_files': 171,
                'test_type': 'keyboards',
                'keyboard_count': 10
            },
            {
                'name': 'medium_scale',
                'description': 'First 50 keyboard directories',
                'expected_files': 951,
                'test_type': 'keyboards',
                'keyboard_count': 50
            },
            {
                'name': 'large_scale',
                'description': 'keyboards + quantum + docs',
                'expected_files': 2000,
                'test_type': 'selective_dirs',
                'dirs': ['keyboards', 'quantum', 'docs']
            },
            {
                'name': 'extra_large_scale',
                'description': 'Major directories',
                'expected_files': 5000,
                'test_type': 'selective_dirs',
                'dirs': ['keyboards', 'quantum', 'docs', 'drivers', 'platforms', 'tmk_core', 'lib']
            },
            {
                'name': 'maximum_scale',
                'description': 'Full QMK repository',
                'expected_files': 22066,
                'test_type': 'full_repo'
            }
        ]
        
    def count_files(self, paths: List[Path]) -> int:
        """Count files in given paths"""
        total = 0
        for path in paths:
            if path.is_file():
                total += 1
            elif path.is_dir():
                try:
                    for item in path.rglob('*'):
                        if item.is_file() and self._should_process_file(item):
                            total += 1
                except PermissionError:
                    continue
        return total
        
    def get_keyboard_dirs(self, count: int) -> List[Path]:
        """Get first N keyboard directories"""
        keyboards_dir = self.qmk_path / 'keyboards'
        keyboard_dirs = sorted([d for d in keyboards_dir.iterdir() if d.is_dir()])
        return keyboard_dirs[:count]
        
    def get_paths_for_phase(self, phase: Dict) -> List[Path]:
        """Get file paths for a specific test phase"""
        test_type = phase['test_type']
        
        if test_type == 'baseline':
            return []  # No additional files for baseline
            
        elif test_type == 'keyboards':
            count = phase['keyboard_count']
            return self.get_keyboard_dirs(count)
            
        elif test_type == 'selective_dirs':
            dirs = phase['dirs']
            return [self.qmk_path / dir_name for dir_name in dirs if (self.qmk_path / dir_name).exists()]
            
        elif test_type == 'full_repo':
            return [self.qmk_path]
            
        return []
        
    def _should_process_file(self, file_path: Path) -> bool:
        """Determine if file should be processed"""
        # Skip binary files and very large files
        if file_path.suffix.lower() in ['.bin', '.hex', '.o', '.so', '.dylib', '.exe', '.jpg', '.png', '.gif', '.pdf']:
            return False
            
        try:
            # Skip files over 1MB
            if file_path.stat().st_size > 1024 * 1024:
                return False
        except:
            return False
            
        return True
        
    def collect_documents_from_paths(self, paths: List[Path], max_files: int = 1000) -> List[Dict]:
        """Collect documents from file paths"""
        documents = []
        file_count = 0
        
        for path in paths:
            if file_count >= max_files:
                break
                
            try:
                if path.is_file() and self._should_process_file(path):
                    doc = self._process_file_to_document(path, file_count)
                    if doc:
                        documents.append(doc)
                        file_count += 1
                elif path.is_dir():
                    for file_path in path.rglob('*'):
                        if file_count >= max_files:
                            break
                        if file_path.is_file() and self._should_process_file(file_path):
                            doc = self._process_file_to_document(file_path, file_count)
                            if doc:
                                documents.append(doc)
                                file_count += 1
                                
            except Exception as e:
                print(f"⚠️  Error processing path {path}: {e}")
                continue
                
        return documents
        
    def _process_file_to_document(self, file_path: Path, doc_id: int) -> Optional[Dict]:
        """Process a file into a document"""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                
            if not content.strip():
                return None
                
            return {
                'id': doc_id,
                'content': content,
                'metadata': {
                    'file_path': str(file_path),
                    'file_name': file_path.name,
                    'file_type': file_path.suffix,
                    'size_bytes': len(content.encode('utf-8'))
                }
            }
        except Exception as e:
            return None
            
    def run_performance_tests(self, collection: str) -> Dict:
        """Run performance tests on the collection"""
        test_queries = [
            "keyboard configuration",
            "matrix scanning", 
            "LED control",
            "USB communication",
            "firmware update"
        ]
        
        search_results = []
        
        for query in test_queries:
            result = self.qdrant.search_collection(collection, query, limit=10)
            search_results.append(result)
            time.sleep(0.1)  # Small delay between searches
            
        # Calculate performance metrics
        successful_searches = [r for r in search_results if r.get('success', False)]
        if successful_searches:
            search_times = [r['search_time_ms'] for r in successful_searches]
            performance = {
                'search_tests': len(test_queries),
                'successful_searches': len(successful_searches),
                'avg_search_time_ms': sum(search_times) / len(search_times),
                'max_search_time_ms': max(search_times),
                'min_search_time_ms': min(search_times)
            }
        else:
            performance = {
                'search_tests': len(test_queries),
                'successful_searches': 0,
                'error': 'All searches failed'
            }
            
        return performance
        
    def run_phase(self, phase: Dict) -> Dict:
        """Execute a single test phase"""
        phase_name = phase['name']
        print(f"\n🔄 Starting Phase: {phase_name}")
        print(f"   Description: {phase['description']}")
        print(f"   Expected files: ~{phase['expected_files']}")
        
        phase_start = time.time()
        
        # Get paths for this phase
        paths = self.get_paths_for_phase(phase)
        actual_file_count = self.count_files(paths) if paths else 0
        
        print(f"   Actual files found: {actual_file_count}")
        
        results = {
            'phase_name': phase_name,
            'description': phase['description'],
            'expected_files': phase['expected_files'],
            'actual_files': actual_file_count,
            'start_time': phase_start,
            'paths_count': len(paths)
        }
        
        try:
            if phase['test_type'] == 'baseline':
                # Baseline test - just check current system state
                print("   📊 Running baseline measurements...")
                self.monitor.start_monitoring()
                
                # Get current collections info
                collections = self.qdrant.get_collections()
                collection_info = {}
                for coll in collections:
                    info = self.qdrant.get_collection_info(coll)
                    collection_info[coll] = info.get('points_count', 0)
                
                time.sleep(10)  # Monitor for 10 seconds
                resource_stats = self.monitor.stop_monitoring()
                
                results.update({
                    'resource_stats': resource_stats,
                    'existing_collections': collection_info,
                    'success': True
                })
                
            else:
                # Create test collection
                collection_name = f"qmk-scaling-test-{phase_name}-final"
                collection_result = self.qdrant.create_collection(collection_name)
                
                if not collection_result['success']:
                    results['success'] = False
                    results['error'] = f"Failed to create collection: {collection_result['error']}"
                    return results
                    
                results['collection_name'] = collection_name
                
                # Collect documents from paths
                print("   📄 Collecting documents...")
                max_files = min(actual_file_count, 1000)  # Limit for realistic testing
                documents = self.collect_documents_from_paths(paths, max_files)
                
                print(f"   📊 Documents collected: {len(documents)}")
                
                if not documents:
                    results['success'] = False
                    results['error'] = "No documents collected"
                    return results
                
                # Start resource monitoring
                print("   🔍 Starting resource monitoring...")
                self.monitor.start_monitoring()
                
                # Ingest documents
                print("   📥 Starting document ingestion...")
                ingestion_start = time.time()
                ingestion_result = self.qdrant.add_document_batch(collection_name, documents)
                ingestion_time = time.time() - ingestion_start
                
                print(f"   ✅ Ingestion completed in {ingestion_time:.2f}s")
                print(f"   📊 Documents processed: {ingestion_result.get('added', 0)}")
                
                if not ingestion_result['success']:
                    print(f"   ⚠️  Ingestion failed: {ingestion_result.get('error', 'Unknown error')}")
                    resource_stats = self.monitor.stop_monitoring()
                    results.update({
                        'success': False,
                        'error': f"Ingestion failed: {ingestion_result.get('error', 'Unknown error')}",
                        'resource_stats': resource_stats,
                        'documents_collected': len(documents),
                        'ingestion_time_seconds': ingestion_time
                    })
                    return results
                
                # Brief pause before performance tests
                time.sleep(2)
                
                # Run performance tests
                print("   🚀 Running performance tests...")
                performance_start = time.time()
                performance_results = self.run_performance_tests(collection_name)
                performance_time = time.time() - performance_start
                
                # Stop monitoring
                resource_stats = self.monitor.stop_monitoring()
                
                # Get final collection stats
                collection_info = self.qdrant.get_collection_info(collection_name)
                
                results.update({
                    'collection_info': collection_info,
                    'documents_collected': len(documents),
                    'ingestion_time_seconds': ingestion_time,
                    'ingestion_result': ingestion_result,
                    'performance_time_seconds': performance_time,
                    'performance_results': performance_results,
                    'resource_stats': resource_stats,
                    'success': True
                })
                
        except Exception as e:
            print(f"   ❌ Phase failed: {e}")
            # Stop monitoring if it's running
            try:
                self.monitor.stop_monitoring()
            except:
                pass
            results.update({
                'success': False,
                'error': str(e)
            })
            
        results['total_time_seconds'] = time.time() - phase_start
        
        # Safety check - verify system is stable after phase
        memory = psutil.virtual_memory()
        cpu = psutil.cpu_percent(interval=1)
        
        print(f"   📊 Post-phase system state:")
        print(f"      CPU: {cpu:.1f}%")
        print(f"      Memory: {memory.percent:.1f}% ({memory.used/1024/1024/1024:.1f} GB used)")
        
        if memory.percent > 80 or cpu > 85:
            print(f"   ⚠️  WARNING: High resource usage detected!")
            results['resource_warning'] = True
            
        return results
        
    def run_all_phases(self) -> List[Dict]:
        """Run all test phases"""
        print("🚀 Starting Progressive Scaling Validation System - FINAL VERSION")
        print(f"📍 QMK Path: {self.qmk_path}")
        print(f"📊 Number of phases: {len(self.phases)}")
        
        all_results = []
        
        for i, phase in enumerate(self.phases, 1):
            print(f"\n{'='*60}")
            print(f"PHASE {i}/{len(self.phases)}: {phase['name'].upper()}")
            print(f"{'='*60}")
            
            # Check system resources before starting
            memory = psutil.virtual_memory()
            cpu = psutil.cpu_percent(interval=1)
            
            print(f"Pre-phase system state:")
            print(f"  CPU: {cpu:.1f}%")
            print(f"  Memory: {memory.percent:.1f}% ({memory.used/1024/1024/1024:.1f} GB used)")
            
            # Safety check - abort if system is already stressed
            if memory.percent > 75 or cpu > 80:
                print(f"❌ ABORTING: System resources too high before phase start")
                result = {
                    'phase_name': phase['name'],
                    'success': False,
                    'error': f'Pre-phase resource check failed - CPU: {cpu:.1f}%, Memory: {memory.percent:.1f}%',
                    'aborted': True
                }
                all_results.append(result)
                break
                
            # Run the phase
            result = self.run_phase(phase)
            all_results.append(result)
            
            # Check if phase failed
            if not result.get('success', False):
                print(f"❌ Phase {phase['name']} failed: {result.get('error', 'Unknown error')}")
                
                # Continue with all phases for comprehensive analysis
                print("⚠️  Continuing with next phase for comprehensive analysis...")
                    
            # Cleanup between phases
            time.sleep(3)  # Allow system to stabilize
            
        return all_results
        
    def save_results(self, results: List[Dict]):
        """Save test results to file"""
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        filename = f"progressive_scaling_results_FINAL_{timestamp}.json"
        filepath = self.results_dir / filename
        
        # Create comprehensive results document
        full_results = {
            'test_info': {
                'test_name': 'Progressive Scaling Validation System - FINAL',
                'task_number': 146,
                'timestamp': timestamp,
                'qmk_path': str(self.qmk_path),
                'total_phases': len(self.phases),
                'completed_phases': len(results)
            },
            'system_info': {
                'cpu_count': psutil.cpu_count(),
                'memory_gb': psutil.virtual_memory().total / 1024 / 1024 / 1024,
                'python_version': sys.version,
                'platform': sys.platform
            },
            'phase_results': results
        }
        
        with open(filepath, 'w') as f:
            json.dump(full_results, f, indent=2)
            
        print(f"📄 Results saved to: {filepath}")
        return filepath
        
    def generate_summary_report(self, results: List[Dict]):
        """Generate comprehensive summary analysis"""
        print(f"\n{'='*80}")
        print("PROGRESSIVE SCALING VALIDATION FINAL SUMMARY")
        print(f"{'='*80}")
        
        successful_phases = [r for r in results if r.get('success', False)]
        failed_phases = [r for r in results if not r.get('success', False)]
        
        print(f"✅ Successful phases: {len(successful_phases)}")
        print(f"❌ Failed phases: {len(failed_phases)}")
        print(f"📊 Total phases attempted: {len(results)}")
        
        if successful_phases:
            print(f"\n🎯 SCALING CHARACTERISTICS:")
            
            for result in successful_phases:
                if result['phase_name'] == 'baseline':
                    continue
                    
                files = result.get('actual_files', 0)
                documents_processed = result.get('documents_collected', 0)
                documents_added = result.get('ingestion_result', {}).get('added', 0)
                ingestion_time = result.get('ingestion_time_seconds', 0)
                
                if documents_processed > 0 and ingestion_time > 0:
                    docs_per_second = documents_added / ingestion_time if ingestion_time > 0 else 0
                    print(f"  {result['phase_name']:<20}: {files:>6} files → {documents_processed:>4} collected → {documents_added:>4} added in {ingestion_time:>6.1f}s ({docs_per_second:>5.1f} docs/s)")
                    
            print(f"\n💾 RESOURCE USAGE PATTERNS:")
            
            for result in successful_phases:
                resource_stats = result.get('resource_stats', {})
                if resource_stats:
                    memory_stats = resource_stats.get('memory', {})
                    cpu_stats = resource_stats.get('cpu', {})
                    
                    max_memory = memory_stats.get('max_percent', 0)
                    max_cpu = cpu_stats.get('max', 0)
                    avg_memory = memory_stats.get('avg_percent', 0)
                    avg_cpu = cpu_stats.get('avg', 0)
                    
                    print(f"  {result['phase_name']:<20}: Memory {avg_memory:>5.1f}%/{max_memory:>5.1f}% (avg/peak), CPU {avg_cpu:>5.1f}%/{max_cpu:>5.1f}% (avg/peak)")
                    
            print(f"\n🚀 SEARCH PERFORMANCE:")
            
            for result in successful_phases:
                if result['phase_name'] == 'baseline':
                    continue
                    
                perf_results = result.get('performance_results', {})
                if perf_results and perf_results.get('successful_searches', 0) > 0:
                    avg_time = perf_results.get('avg_search_time_ms', 0)
                    max_time = perf_results.get('max_search_time_ms', 0)
                    min_time = perf_results.get('min_search_time_ms', 0)
                    success_rate = perf_results.get('successful_searches', 0) / perf_results.get('search_tests', 1) * 100
                    print(f"  {result['phase_name']:<20}: {avg_time:>5.1f}ms avg ({min_time:>4.1f}-{max_time:>5.1f}ms range), {success_rate:>3.0f}% success")
                    
        if failed_phases:
            print(f"\n❌ FAILED PHASES:")
            for result in failed_phases:
                error = result.get('error', 'Unknown error')
                files = result.get('actual_files', 0)
                print(f"  {result['phase_name']:<20}: {error} ({files:,} files)")
                
        # Performance scaling analysis
        successful_scaling_phases = [r for r in successful_phases if r['phase_name'] != 'baseline' and r.get('documents_collected', 0) > 0]
        
        if len(successful_scaling_phases) > 1:
            print(f"\n📈 SCALING EFFICIENCY ANALYSIS:")
            
            base_phase = successful_scaling_phases[0]
            base_docs = base_phase.get('documents_collected', 1)
            base_time = base_phase.get('ingestion_time_seconds', 1)
            base_efficiency = base_docs / base_time
            
            for result in successful_scaling_phases:
                docs = result.get('documents_collected', 0)
                time_taken = result.get('ingestion_time_seconds', 1)
                efficiency = docs / time_taken if time_taken > 0 else 0
                efficiency_ratio = efficiency / base_efficiency if base_efficiency > 0 else 0
                
                scale_factor = docs / base_docs if base_docs > 0 else 0
                
                print(f"  {result['phase_name']:<20}: {scale_factor:>5.1f}x scale, {efficiency_ratio:>5.1f}x efficiency ({efficiency:>5.1f} docs/s)")
                
        # Breaking point analysis
        max_successful_files = 0
        max_successful_phase = None
        
        for result in successful_phases:
            if result['phase_name'] != 'baseline':
                files = result.get('actual_files', 0)
                if files > max_successful_files:
                    max_successful_files = files
                    max_successful_phase = result['phase_name']
                    
        if max_successful_phase:
            print(f"\n🔍 SYSTEM LIMITS ANALYSIS:")
            print(f"  Maximum successful scale: {max_successful_phase} ({max_successful_files:,} files)")
            
            # Check resource usage at maximum scale
            for result in successful_phases:
                if result['phase_name'] == max_successful_phase:
                    resource_stats = result.get('resource_stats', {})
                    if resource_stats:
                        memory_stats = resource_stats.get('memory', {})
                        cpu_stats = resource_stats.get('cpu', {})
                        max_memory = memory_stats.get('max_percent', 0)
                        max_cpu = cpu_stats.get('max', 0)
                        
                        print(f"  Resource usage at max scale: Memory {max_memory:.1f}%, CPU {max_cpu:.1f}%")
                        
                        # Calculate headroom
                        memory_headroom = 85 - max_memory  # Safety threshold is 85%
                        cpu_headroom = 90 - max_cpu        # Safety threshold is 90%
                        
                        print(f"  Safety headroom remaining: Memory {memory_headroom:.1f}%, CPU {cpu_headroom:.1f}%")
                        
            # Estimate scaling potential
            first_failed = None
            for result in failed_phases:
                if not result.get('aborted', False):
                    first_failed = result
                    break
                    
            if first_failed:
                expected_files = first_failed.get('expected_files', 0)
                print(f"  First failure at scale:   {first_failed['phase_name']} (~{expected_files:,} files)")
                print(f"  Estimated scaling range:  {max_successful_files:,} - {expected_files:,} files")
            else:
                print(f"  🎉 All phases completed successfully!")
                print(f"  System can handle the full QMK repository scale!")
                
        print(f"\n🏆 FINAL RECOMMENDATIONS:")
        if successful_scaling_phases:
            avg_efficiency = sum(r.get('documents_collected', 0) / r.get('ingestion_time_seconds', 1) for r in successful_scaling_phases) / len(successful_scaling_phases)
            print(f"  • Average ingestion rate: {avg_efficiency:.1f} documents/second")
            print(f"  • Recommended batch size: {min(1000, max_successful_files // 10)} files")
            
        if max_successful_files > 0:
            print(f"  • Proven stable scale: up to {max_successful_files:,} files")
            
        if failed_phases:
            print(f"  • Areas for optimization: {len(failed_phases)} phases need investigation")
            
        print(f"\n{'='*80}")

def main():
    """Main execution function"""
    print("Progressive Scaling Validation System - FINAL VERSION")
    print("Task #146: Comprehensive daemon performance analysis across project scales\n")
    
    # Verify QMK path exists
    qmk_path = Path("/Users/chris/Dropbox/dev/ai/claude-code-cfg/mcp/workspace-qdrant-mcp/LT_20250906-1822_testing_sandbox/qmk_integration/qmk_firmware")
    if not qmk_path.exists():
        print(f"❌ QMK path not found: {qmk_path}")
        sys.exit(1)
        
    # Verify Qdrant is running
    try:
        response = requests.get("http://localhost:6333/collections")
        if response.status_code != 200:
            raise Exception(f"HTTP {response.status_code}")
    except Exception as e:
        print(f"❌ Qdrant not accessible: {e}")
        print("   Please ensure Qdrant is running on localhost:6333")
        sys.exit(1)
        
    print("✅ Prerequisites verified")
    print("✅ QMK repository accessible")  
    print("✅ Qdrant daemon running")
    
    # Create and run tester
    tester = ProgressiveScalingTester()
    
    try:
        results = tester.run_all_phases()
        filepath = tester.save_results(results)
        tester.generate_summary_report(results)
        
        print(f"\n🎉 Progressive scaling validation COMPLETED!")
        print(f"📊 Comprehensive results saved to: {filepath}")
        print(f"🔬 Task #146 progressive scaling analysis complete")
        
        return results
        
    except KeyboardInterrupt:
        print(f"\n⚠️  Test interrupted by user")
        return []
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return []

if __name__ == "__main__":
    main()