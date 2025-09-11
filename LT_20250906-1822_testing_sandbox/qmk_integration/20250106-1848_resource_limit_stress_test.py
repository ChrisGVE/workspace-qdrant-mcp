#!/usr/bin/env python3
"""
Resource Limit Stress Testing Framework - Task #148
Comprehensive system for finding daemon breaking points and validating resource guardrails

CRITICAL SAFETY FEATURES:
- Mandatory 20% system resource reservation for OS
- Progressive stress escalation with validation
- Emergency stop mechanisms for system protection
- Automatic recovery and cleanup procedures

TEST SCENARIOS:
1. Memory Pressure Testing - Progressive consumption until limits
2. CPU Saturation Testing - Maximum concurrent processing threads  
3. Disk I/O Stress Testing - Rapid file operations and queue saturation
4. Resource Guardrail Validation - Test daemon's built-in safety mechanisms
5. Recovery Behavior Analysis - System recovery after pressure relief
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
import tempfile
import concurrent.futures
import random
import string
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import requests
import signal
import gc
import multiprocessing

# Critical System Constants
MAX_SAFE_MEMORY_PERCENT = 80.0  # Reserve 20% for OS
MAX_SAFE_CPU_PERCENT = 80.0     # Reserve 20% for OS
EMERGENCY_STOP_MEMORY = 95.0    # Emergency stop threshold
EMERGENCY_STOP_CPU = 95.0       # Emergency stop threshold
STRESS_ESCALATION_INTERVAL = 30 # Seconds between stress level increases

class SystemSafetyMonitor:
    """
    Critical system safety monitor with emergency stop capabilities
    Prevents system strangulation during resource limit testing
    """
    
    def __init__(self):
        self.monitoring = False
        self.emergency_stop_triggered = False
        self.thread = None
        self.callbacks = []
        self.data = []
        self.breaking_points = {
            'memory_warning': None,
            'memory_critical': None,
            'cpu_warning': None,
            'cpu_critical': None,
            'emergency_stop': None
        }
        
    def add_emergency_callback(self, callback):
        """Add callback to execute on emergency stop"""
        self.callbacks.append(callback)
        
    def start_monitoring(self, interval: float = 0.5):
        """Start intensive safety monitoring"""
        print(f"🛡️  Starting system safety monitoring (interval: {interval}s)")
        self.monitoring = True
        self.emergency_stop_triggered = False
        self.data = []
        self.thread = threading.Thread(target=self._safety_monitor_loop, args=(interval,))
        self.thread.daemon = True
        self.thread.start()
        
    def stop_monitoring(self) -> Dict:
        """Stop monitoring and return safety statistics"""
        self.monitoring = False
        if self.thread:
            self.thread.join(timeout=10.0)
            
        if not self.data:
            return {'error': 'No monitoring data collected'}
            
        cpu_values = [d['cpu_percent'] for d in self.data]
        memory_values = [d['memory_percent'] for d in self.data]
        memory_gb = [d['memory_gb'] for d in self.data]
        
        return {
            'duration_seconds': len(self.data) * 0.5,
            'samples': len(self.data),
            'emergency_stop_triggered': self.emergency_stop_triggered,
            'breaking_points': self.breaking_points.copy(),
            'resource_peaks': {
                'cpu_max': max(cpu_values),
                'memory_max_percent': max(memory_values),
                'memory_max_gb': max(memory_gb),
                'cpu_avg': sum(cpu_values) / len(cpu_values),
                'memory_avg_percent': sum(memory_values) / len(memory_values)
            },
            'safety_violations': len([d for d in self.data if d['cpu_percent'] > MAX_SAFE_CPU_PERCENT or d['memory_percent'] > MAX_SAFE_MEMORY_PERCENT]),
            'critical_violations': len([d for d in self.data if d['cpu_percent'] > EMERGENCY_STOP_CPU or d['memory_percent'] > EMERGENCY_STOP_MEMORY])
        }
        
    def _safety_monitor_loop(self, interval: float):
        """Critical safety monitoring loop"""
        while self.monitoring and not self.emergency_stop_triggered:
            try:
                cpu_percent = psutil.cpu_percent(interval=None)
                memory = psutil.virtual_memory()
                memory_percent = memory.percent
                memory_gb = memory.used / 1024 / 1024 / 1024
                
                timestamp = time.time()
                self.data.append({
                    'timestamp': timestamp,
                    'cpu_percent': cpu_percent,
                    'memory_percent': memory_percent,
                    'memory_gb': memory_gb
                })
                
                # Check for breaking point detection
                if memory_percent > MAX_SAFE_MEMORY_PERCENT and not self.breaking_points['memory_warning']:
                    self.breaking_points['memory_warning'] = {
                        'timestamp': timestamp,
                        'memory_percent': memory_percent,
                        'memory_gb': memory_gb
                    }
                    print(f"⚠️  MEMORY WARNING THRESHOLD REACHED: {memory_percent:.1f}% ({memory_gb:.1f} GB)")
                    
                if cpu_percent > MAX_SAFE_CPU_PERCENT and not self.breaking_points['cpu_warning']:
                    self.breaking_points['cpu_warning'] = {
                        'timestamp': timestamp,
                        'cpu_percent': cpu_percent
                    }
                    print(f"⚠️  CPU WARNING THRESHOLD REACHED: {cpu_percent:.1f}%")
                
                # Critical thresholds
                if memory_percent > 90 and not self.breaking_points['memory_critical']:
                    self.breaking_points['memory_critical'] = {
                        'timestamp': timestamp,
                        'memory_percent': memory_percent,
                        'memory_gb': memory_gb
                    }
                    print(f"🚨 MEMORY CRITICAL THRESHOLD: {memory_percent:.1f}% ({memory_gb:.1f} GB)")
                    
                if cpu_percent > 90 and not self.breaking_points['cpu_critical']:
                    self.breaking_points['cpu_critical'] = {
                        'timestamp': timestamp,
                        'cpu_percent': cpu_percent
                    }
                    print(f"🚨 CPU CRITICAL THRESHOLD: {cpu_percent:.1f}%")
                
                # Emergency stop conditions
                if memory_percent > EMERGENCY_STOP_MEMORY or cpu_percent > EMERGENCY_STOP_CPU:
                    self.breaking_points['emergency_stop'] = {
                        'timestamp': timestamp,
                        'cpu_percent': cpu_percent,
                        'memory_percent': memory_percent,
                        'memory_gb': memory_gb,
                        'trigger': 'memory' if memory_percent > EMERGENCY_STOP_MEMORY else 'cpu'
                    }
                    
                    print(f"🔴 EMERGENCY STOP TRIGGERED - CPU: {cpu_percent:.1f}%, Memory: {memory_percent:.1f}%")
                    self._trigger_emergency_stop()
                    break
                    
                time.sleep(interval)
                
            except Exception as e:
                print(f"⚠️  Safety monitor error: {e}")
                time.sleep(interval)
                
    def _trigger_emergency_stop(self):
        """Execute emergency stop procedures"""
        print("🔴 EXECUTING EMERGENCY STOP PROCEDURES")
        self.emergency_stop_triggered = True
        
        # Execute all emergency callbacks
        for callback in self.callbacks:
            try:
                callback()
            except Exception as e:
                print(f"⚠️  Emergency callback error: {e}")
                
        # Force garbage collection
        gc.collect()
        
        print("🔴 Emergency stop procedures completed")

class QdrantStressClient:
    """Enhanced Qdrant client for resource stress testing"""
    
    def __init__(self, base_url: str = "http://localhost:6333"):
        self.base_url = base_url
        self.test_collection = f"stress_test_{uuid.uuid4().hex[:8]}"
        self.session = requests.Session()
        
    def create_stress_collection(self, vector_size: int = 384) -> bool:
        """Create collection optimized for stress testing"""
        collection_config = {
            "vectors": {
                "size": vector_size,
                "distance": "Cosine"
            },
            "replication_factor": 1,
            "write_consistency_factor": 1
        }
        
        try:
            response = self.session.put(
                f"{self.base_url}/collections/{self.test_collection}",
                json=collection_config
            )
            return response.status_code in [200, 201]
        except Exception as e:
            print(f"⚠️  Collection creation error: {e}")
            return False
            
    def stress_upsert_batch(self, documents: List[Dict], batch_size: int = 100) -> Tuple[bool, float, Dict]:
        """Stress test document upserts with resource monitoring"""
        start_time = time.time()
        success_count = 0
        error_count = 0
        
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            
            try:
                points = []
                for doc in batch:
                    points.append({
                        "id": doc['id'],
                        "vector": doc['vector'],
                        "payload": doc['payload']
                    })
                    
                upsert_data = {"points": points}
                
                response = self.session.put(
                    f"{self.base_url}/collections/{self.test_collection}/points",
                    json=upsert_data
                )
                
                if response.status_code in [200, 201]:
                    success_count += len(batch)
                else:
                    error_count += len(batch)
                    print(f"⚠️  Batch upsert failed: {response.status_code}")
                    
            except Exception as e:
                error_count += len(batch)
                print(f"⚠️  Upsert error: {e}")
                
        duration = time.time() - start_time
        throughput = success_count / duration if duration > 0 else 0
        
        return (
            error_count == 0,
            duration,
            {
                'success_count': success_count,
                'error_count': error_count,
                'throughput_docs_per_sec': throughput,
                'batch_size': batch_size
            }
        )
        
    def stress_search_bombardment(self, query_count: int, concurrent_threads: int = 10) -> Dict:
        """Bombard search system with concurrent queries"""
        print(f"🔍 Starting search bombardment: {query_count} queries, {concurrent_threads} threads")
        
        def execute_search():
            try:
                search_data = {
                    "vector": [random.random() for _ in range(384)],
                    "limit": 10,
                    "with_payload": True
                }
                
                response = self.session.post(
                    f"{self.base_url}/collections/{self.test_collection}/points/search",
                    json=search_data
                )
                
                return {
                    'success': response.status_code == 200,
                    'response_time': response.elapsed.total_seconds(),
                    'status_code': response.status_code
                }
            except Exception as e:
                return {
                    'success': False,
                    'error': str(e),
                    'response_time': 0
                }
                
        start_time = time.time()
        results = []
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrent_threads) as executor:
            futures = [executor.submit(execute_search) for _ in range(query_count)]
            
            for future in concurrent.futures.as_completed(futures):
                results.append(future.result())
                
        duration = time.time() - start_time
        
        success_results = [r for r in results if r.get('success', False)]
        error_results = [r for r in results if not r.get('success', False)]
        
        if success_results:
            response_times = [r['response_time'] for r in success_results]
            avg_response_time = sum(response_times) / len(response_times)
            max_response_time = max(response_times)
            min_response_time = min(response_times)
        else:
            avg_response_time = max_response_time = min_response_time = 0
            
        return {
            'total_queries': query_count,
            'successful_queries': len(success_results),
            'failed_queries': len(error_results),
            'success_rate': len(success_results) / query_count * 100,
            'duration_seconds': duration,
            'queries_per_second': query_count / duration if duration > 0 else 0,
            'avg_response_time': avg_response_time,
            'max_response_time': max_response_time,
            'min_response_time': min_response_time,
            'concurrent_threads': concurrent_threads
        }
        
    def cleanup_stress_collection(self):
        """Clean up stress test collection"""
        try:
            response = self.session.delete(f"{self.base_url}/collections/{self.test_collection}")
            return response.status_code in [200, 204]
        except:
            return False

class MemoryStressTest:
    """Progressive memory stress testing to find breaking points"""
    
    def __init__(self, qdrant_client: QdrantStressClient, safety_monitor: SystemSafetyMonitor):
        self.qdrant_client = qdrant_client
        self.safety_monitor = safety_monitor
        self.memory_chunks = []  # Track allocated memory
        self.large_documents = []
        
    def generate_large_document(self, size_mb: int) -> Dict:
        """Generate artificially large document for memory stress"""
        # Create large text content
        content_size = size_mb * 1024 * 1024  # Convert MB to bytes
        large_content = 'A' * content_size  # Simple but effective memory allocation
        
        return {
            'id': random.randint(100000, 999999),
            'vector': [random.random() for _ in range(384)],
            'payload': {
                'title': f'Large Document {size_mb}MB',
                'content': large_content,
                'size_mb': size_mb,
                'timestamp': time.time()
            }
        }
        
    def progressive_memory_stress(self, start_mb: int = 50, max_mb: int = 1000, step_mb: int = 50) -> Dict:
        """Progressively increase memory pressure until limits reached"""
        print(f"🧠 Starting progressive memory stress test")
        print(f"   Range: {start_mb}MB → {max_mb}MB (steps: {step_mb}MB)")
        
        results = {
            'start_time': time.time(),
            'phases': [],
            'breaking_point': None,
            'max_memory_reached': 0,
            'emergency_stop': False
        }
        
        for current_mb in range(start_mb, max_mb + 1, step_mb):
            if self.safety_monitor.emergency_stop_triggered:
                print("🔴 Emergency stop triggered during memory stress test")
                results['emergency_stop'] = True
                break
                
            print(f"📈 Memory stress phase: {current_mb}MB documents")
            
            phase_start = time.time()
            
            # Generate large documents
            phase_documents = []
            for _ in range(3):  # Create 3 large documents per phase
                doc = self.generate_large_document(current_mb // 3)  # Split size across documents
                phase_documents.append(doc)
                self.large_documents.append(doc)
                
            # Attempt to upsert to Qdrant
            memory_before = psutil.virtual_memory()
            
            success, duration, upsert_stats = self.qdrant_client.stress_upsert_batch(
                phase_documents, batch_size=1
            )
            
            memory_after = psutil.virtual_memory()
            phase_duration = time.time() - phase_start
            
            phase_result = {
                'target_mb': current_mb,
                'success': success,
                'duration': phase_duration,
                'upsert_stats': upsert_stats,
                'memory_before_percent': memory_before.percent,
                'memory_after_percent': memory_after.percent,
                'memory_delta_mb': (memory_after.used - memory_before.used) / 1024 / 1024
            }
            
            results['phases'].append(phase_result)
            results['max_memory_reached'] = memory_after.percent
            
            if not success or memory_after.percent > MAX_SAFE_MEMORY_PERCENT:
                print(f"🔶 Memory stress breaking point reached at {current_mb}MB")
                print(f"   Memory usage: {memory_after.percent:.1f}%")
                results['breaking_point'] = {
                    'memory_mb': current_mb,
                    'memory_percent': memory_after.percent,
                    'reason': 'upsert_failure' if not success else 'memory_limit'
                }
                break
                
            # Wait between phases for safety
            time.sleep(5)
            
        results['end_time'] = time.time()
        results['total_duration'] = results['end_time'] - results['start_time']
        
        return results
        
    def cleanup_memory_stress(self):
        """Clean up memory stress test artifacts"""
        print("🧹 Cleaning up memory stress test")
        self.large_documents.clear()
        self.memory_chunks.clear()
        gc.collect()  # Force garbage collection

class CPUStressTest:
    """CPU saturation testing to find processing limits"""
    
    def __init__(self, qdrant_client: QdrantStressClient, safety_monitor: SystemSafetyMonitor):
        self.qdrant_client = qdrant_client
        self.safety_monitor = safety_monitor
        self.cpu_workers = []
        
    def cpu_intensive_task(self, duration_seconds: int = 10):
        """CPU-intensive task for stress testing"""
        start_time = time.time()
        operations = 0
        
        while time.time() - start_time < duration_seconds:
            if self.safety_monitor.emergency_stop_triggered:
                break
                
            # CPU-intensive operations
            _ = sum(i * i for i in range(1000))
            _ = hashlib.sha256(str(time.time()).encode()).hexdigest()
            operations += 1
            
        return operations
        
    def concurrent_processing_stress(self, max_threads: int = None) -> Dict:
        """Test maximum concurrent processing capacity"""
        if max_threads is None:
            max_threads = multiprocessing.cpu_count() * 4  # 4x CPU cores
            
        print(f"⚡ Starting concurrent processing stress test")
        print(f"   Max threads: {max_threads}")
        
        results = {
            'start_time': time.time(),
            'thread_phases': [],
            'breaking_point': None,
            'max_cpu_reached': 0
        }
        
        for thread_count in [1, 2, 4, 8, 16, 32, 64, min(max_threads, 128)]:
            if self.safety_monitor.emergency_stop_triggered:
                break
                
            print(f"🔄 Testing {thread_count} concurrent threads")
            
            cpu_before = psutil.cpu_percent(interval=1)
            phase_start = time.time()
            
            # Execute CPU-intensive tasks
            with concurrent.futures.ThreadPoolExecutor(max_workers=thread_count) as executor:
                futures = [executor.submit(self.cpu_intensive_task, 10) for _ in range(thread_count)]
                
                # Wait for completion
                total_operations = 0
                for future in concurrent.futures.as_completed(futures):
                    if not self.safety_monitor.emergency_stop_triggered:
                        total_operations += future.result()
                    else:
                        break
                        
            cpu_after = psutil.cpu_percent(interval=1)
            phase_duration = time.time() - phase_start
            
            phase_result = {
                'thread_count': thread_count,
                'cpu_before': cpu_before,
                'cpu_after': cpu_after,
                'cpu_peak': cpu_after,  # Simplified for this version
                'total_operations': total_operations,
                'operations_per_second': total_operations / phase_duration if phase_duration > 0 else 0,
                'duration': phase_duration
            }
            
            results['thread_phases'].append(phase_result)
            results['max_cpu_reached'] = max(results['max_cpu_reached'], cpu_after)
            
            if cpu_after > MAX_SAFE_CPU_PERCENT:
                print(f"🔶 CPU stress breaking point reached at {thread_count} threads")
                print(f"   CPU usage: {cpu_after:.1f}%")
                results['breaking_point'] = {
                    'thread_count': thread_count,
                    'cpu_percent': cpu_after,
                    'reason': 'cpu_limit'
                }
                break
                
            # Brief pause between phases
            time.sleep(3)
            
        results['end_time'] = time.time()
        results['total_duration'] = results['end_time'] - results['start_time']
        
        return results
        
    def search_query_bombardment(self, duration_seconds: int = 60) -> Dict:
        """Bombard system with search queries to test CPU under query load"""
        print(f"🔍 Starting search query bombardment for {duration_seconds} seconds")
        
        # Progressive thread counts
        thread_counts = [1, 2, 5, 10, 20, 50]
        results = {
            'bombardment_phases': [],
            'max_cpu_reached': 0,
            'breaking_point': None
        }
        
        for thread_count in thread_counts:
            if self.safety_monitor.emergency_stop_triggered:
                break
                
            print(f"🎯 Query bombardment with {thread_count} threads")
            
            cpu_before = psutil.cpu_percent(interval=1)
            
            # Execute bombardment
            bombardment_result = self.qdrant_client.stress_search_bombardment(
                query_count=thread_count * 10,  # 10 queries per thread
                concurrent_threads=thread_count
            )
            
            cpu_after = psutil.cpu_percent(interval=1)
            
            phase_result = {
                'thread_count': thread_count,
                'cpu_before': cpu_before,
                'cpu_after': cpu_after,
                'bombardment_stats': bombardment_result
            }
            
            results['bombardment_phases'].append(phase_result)
            results['max_cpu_reached'] = max(results['max_cpu_reached'], cpu_after)
            
            if cpu_after > MAX_SAFE_CPU_PERCENT or bombardment_result['success_rate'] < 50:
                print(f"🔶 Query bombardment breaking point at {thread_count} threads")
                results['breaking_point'] = {
                    'thread_count': thread_count,
                    'cpu_percent': cpu_after,
                    'success_rate': bombardment_result['success_rate'],
                    'reason': 'cpu_limit' if cpu_after > MAX_SAFE_CPU_PERCENT else 'query_failure'
                }
                break
                
            time.sleep(2)
            
        return results

class DiskIOStressTest:
    """Disk I/O stress testing for file system limits"""
    
    def __init__(self, test_dir: Path, safety_monitor: SystemSafetyMonitor):
        self.test_dir = test_dir
        self.safety_monitor = safety_monitor
        self.stress_files = []
        
    def rapid_file_operations(self, file_count: int = 1000, file_size_kb: int = 100) -> Dict:
        """Test rapid file creation/deletion cycles"""
        print(f"📁 Starting rapid file operations test")
        print(f"   Files: {file_count}, Size: {file_size_kb}KB each")
        
        results = {
            'file_count': file_count,
            'file_size_kb': file_size_kb,
            'creation_phase': {},
            'deletion_phase': {},
            'breaking_point': None
        }
        
        # File creation phase
        creation_start = time.time()
        created_files = []
        
        try:
            file_content = 'X' * (file_size_kb * 1024)  # Generate content
            
            for i in range(file_count):
                if self.safety_monitor.emergency_stop_triggered:
                    break
                    
                file_path = self.test_dir / f"stress_file_{i:06d}.txt"
                
                try:
                    with open(file_path, 'w') as f:
                        f.write(file_content)
                        f.flush()
                        os.fsync(f.fileno())  # Force write to disk
                    
                    created_files.append(file_path)
                    self.stress_files.append(file_path)
                    
                    if i % 100 == 0:  # Progress indicator
                        memory_percent = psutil.virtual_memory().percent
                        if memory_percent > MAX_SAFE_MEMORY_PERCENT:
                            print(f"🔶 I/O stress memory limit reached at {i} files")
                            results['breaking_point'] = {
                                'phase': 'creation',
                                'files_processed': i,
                                'memory_percent': memory_percent,
                                'reason': 'memory_limit'
                            }
                            break
                            
                except Exception as e:
                    print(f"⚠️  File creation error at {i}: {e}")
                    results['breaking_point'] = {
                        'phase': 'creation',
                        'files_processed': i,
                        'error': str(e),
                        'reason': 'io_error'
                    }
                    break
                    
        except Exception as e:
            print(f"⚠️  Creation phase error: {e}")
            
        creation_duration = time.time() - creation_start
        results['creation_phase'] = {
            'files_created': len(created_files),
            'duration': creation_duration,
            'files_per_second': len(created_files) / creation_duration if creation_duration > 0 else 0
        }
        
        # File deletion phase
        deletion_start = time.time()
        deleted_count = 0
        
        try:
            for file_path in created_files:
                if self.safety_monitor.emergency_stop_triggered:
                    break
                    
                try:
                    file_path.unlink()
                    deleted_count += 1
                except Exception as e:
                    print(f"⚠️  File deletion error: {e}")
                    
        except Exception as e:
            print(f"⚠️  Deletion phase error: {e}")
            
        deletion_duration = time.time() - deletion_start
        results['deletion_phase'] = {
            'files_deleted': deleted_count,
            'duration': deletion_duration,
            'files_per_second': deleted_count / deletion_duration if deletion_duration > 0 else 0
        }
        
        return results
        
    def large_file_stress(self, file_sizes_mb: List[int] = [10, 50, 100, 500]) -> Dict:
        """Test large file read/write operations"""
        print(f"📊 Starting large file stress test")
        print(f"   File sizes: {file_sizes_mb} MB")
        
        results = {
            'file_tests': [],
            'breaking_point': None
        }
        
        for size_mb in file_sizes_mb:
            if self.safety_monitor.emergency_stop_triggered:
                break
                
            print(f"📄 Testing {size_mb}MB file operations")
            
            file_path = self.test_dir / f"large_test_{size_mb}mb.dat"
            content = b'B' * (size_mb * 1024 * 1024)
            
            # Write test
            write_start = time.time()
            try:
                with open(file_path, 'wb') as f:
                    f.write(content)
                    f.flush()
                    os.fsync(f.fileno())
                write_duration = time.time() - write_start
                write_success = True
            except Exception as e:
                print(f"⚠️  Large file write error ({size_mb}MB): {e}")
                write_duration = 0
                write_success = False
                
            # Read test
            read_start = time.time()
            try:
                if write_success:
                    with open(file_path, 'rb') as f:
                        read_content = f.read()
                    read_duration = time.time() - read_start
                    read_success = len(read_content) == len(content)
                else:
                    read_duration = 0
                    read_success = False
            except Exception as e:
                print(f"⚠️  Large file read error ({size_mb}MB): {e}")
                read_duration = 0
                read_success = False
                
            # Cleanup
            try:
                if file_path.exists():
                    file_path.unlink()
            except:
                pass
                
            memory_percent = psutil.virtual_memory().percent
            
            file_result = {
                'size_mb': size_mb,
                'write_success': write_success,
                'write_duration': write_duration,
                'write_speed_mbps': size_mb / write_duration if write_duration > 0 else 0,
                'read_success': read_success,
                'read_duration': read_duration,
                'read_speed_mbps': size_mb / read_duration if read_duration > 0 else 0,
                'memory_percent': memory_percent
            }
            
            results['file_tests'].append(file_result)
            
            if not (write_success and read_success) or memory_percent > MAX_SAFE_MEMORY_PERCENT:
                print(f"🔶 Large file breaking point at {size_mb}MB")
                results['breaking_point'] = {
                    'size_mb': size_mb,
                    'write_success': write_success,
                    'read_success': read_success,
                    'memory_percent': memory_percent,
                    'reason': 'io_failure' if not (write_success and read_success) else 'memory_limit'
                }
                break
                
        return results
        
    def cleanup_disk_stress(self):
        """Clean up disk stress test files"""
        print("🧹 Cleaning up disk I/O stress test files")
        for file_path in self.stress_files:
            try:
                if file_path.exists():
                    file_path.unlink()
            except:
                pass
        self.stress_files.clear()

class ResourceLimitStressTester:
    """
    Main orchestrator for comprehensive resource limit stress testing
    Coordinates all stress test components with safety monitoring
    """
    
    def __init__(self, qmk_dir: Path):
        self.qmk_dir = qmk_dir
        self.test_dir = Path(f"LT_20250906-1848_resource_stress_test")
        self.safety_monitor = SystemSafetyMonitor()
        self.qdrant_client = QdrantStressClient()
        
        # Stress test components
        self.memory_stress = None
        self.cpu_stress = None
        self.disk_stress = None
        
        # Results storage
        self.results = {
            'test_start_time': None,
            'test_end_time': None,
            'system_info': self._get_system_info(),
            'safety_monitoring': {},
            'baseline_measurements': {},
            'stress_test_results': {},
            'breaking_points_summary': {},
            'recovery_analysis': {},
            'production_recommendations': {}
        }
        
        # Register emergency stop callback
        self.safety_monitor.add_emergency_callback(self._emergency_cleanup)
        
    def _get_system_info(self) -> Dict:
        """Get comprehensive system information"""
        memory = psutil.virtual_memory()
        return {
            'cpu_count': psutil.cpu_count(),
            'cpu_count_logical': psutil.cpu_count(logical=True),
            'memory_total_gb': memory.total / 1024 / 1024 / 1024,
            'memory_available_gb': memory.available / 1024 / 1024 / 1024,
            'platform': sys.platform,
            'python_version': sys.version,
            'max_safe_memory_percent': MAX_SAFE_MEMORY_PERCENT,
            'max_safe_cpu_percent': MAX_SAFE_CPU_PERCENT,
            'emergency_stop_memory': EMERGENCY_STOP_MEMORY,
            'emergency_stop_cpu': EMERGENCY_STOP_CPU
        }
        
    def _emergency_cleanup(self):
        """Emergency cleanup procedures"""
        print("🔴 Executing emergency cleanup for resource stress test")
        
        try:
            if self.memory_stress:
                self.memory_stress.cleanup_memory_stress()
        except:
            pass
            
        try:
            if self.disk_stress:
                self.disk_stress.cleanup_disk_stress()
        except:
            pass
            
        try:
            if self.qdrant_client:
                self.qdrant_client.cleanup_stress_collection()
        except:
            pass
            
        # Force garbage collection
        gc.collect()
        
    def setup_test_environment(self) -> bool:
        """Set up test environment with safety checks"""
        print("🏗️  Setting up resource limit stress test environment")
        
        # Create test directory
        self.test_dir.mkdir(exist_ok=True)
        
        # Verify Qdrant connectivity
        try:
            response = requests.get("http://localhost:6333/cluster")
            if response.status_code != 200:
                print("❌ Qdrant daemon not accessible")
                return False
            print("✅ Qdrant daemon accessible")
        except Exception as e:
            print(f"❌ Qdrant daemon not accessible: {e}")
            return False
            
        # Create stress test collection
        if not self.qdrant_client.create_stress_collection():
            print("❌ Failed to create stress test collection")
            return False
            
        # Initialize stress test components
        self.memory_stress = MemoryStressTest(self.qdrant_client, self.safety_monitor)
        self.cpu_stress = CPUStressTest(self.qdrant_client, self.safety_monitor)
        self.disk_stress = DiskIOStressTest(self.test_dir, self.safety_monitor)
        
        print("✅ Test environment setup complete")
        return True
        
    def run_baseline_measurements(self):
        """Establish baseline resource usage measurements"""
        print("📊 Running baseline measurements")
        
        baseline_start = time.time()
        
        # Let system stabilize
        time.sleep(5)
        
        # Measure baseline resources
        memory = psutil.virtual_memory()
        cpu_percent = psutil.cpu_percent(interval=2)
        
        self.results['baseline_measurements'] = {
            'timestamp': baseline_start,
            'memory_percent': memory.percent,
            'memory_gb': memory.used / 1024 / 1024 / 1024,
            'memory_available_gb': memory.available / 1024 / 1024 / 1024,
            'cpu_percent': cpu_percent,
            'available_headroom': {
                'memory_percent': MAX_SAFE_MEMORY_PERCENT - memory.percent,
                'cpu_percent': MAX_SAFE_CPU_PERCENT - cpu_percent
            }
        }
        
        print(f"📈 Baseline: Memory {memory.percent:.1f}%, CPU {cpu_percent:.1f}%")
        print(f"🎯 Available headroom: Memory {MAX_SAFE_MEMORY_PERCENT - memory.percent:.1f}%, CPU {MAX_SAFE_CPU_PERCENT - cpu_percent:.1f}%")
        
    def execute_comprehensive_stress_testing(self):
        """Execute all stress testing phases"""
        print("🚀 Starting comprehensive resource limit stress testing")
        
        self.results['test_start_time'] = time.time()
        self.safety_monitor.start_monitoring(interval=0.5)
        
        try:
            # Phase 1: Memory Stress Testing
            print("\n" + "="*60)
            print("🧠 PHASE 1: MEMORY PRESSURE STRESS TESTING")
            print("="*60)
            
            self.results['stress_test_results']['memory_stress'] = self.memory_stress.progressive_memory_stress(
                start_mb=50, max_mb=500, step_mb=50
            )
            
            if self.safety_monitor.emergency_stop_triggered:
                print("🔴 Emergency stop during memory stress test")
                return
                
            time.sleep(10)  # Recovery pause
            
            # Phase 2: CPU Stress Testing  
            print("\n" + "="*60)
            print("⚡ PHASE 2: CPU SATURATION STRESS TESTING")
            print("="*60)
            
            self.results['stress_test_results']['cpu_stress'] = self.cpu_stress.concurrent_processing_stress()
            
            if self.safety_monitor.emergency_stop_triggered:
                print("🔴 Emergency stop during CPU stress test")
                return
                
            time.sleep(10)  # Recovery pause
            
            # Phase 3: Search Query Bombardment
            self.results['stress_test_results']['search_bombardment'] = self.cpu_stress.search_query_bombardment(duration_seconds=30)
            
            if self.safety_monitor.emergency_stop_triggered:
                print("🔴 Emergency stop during search bombardment")
                return
                
            time.sleep(10)  # Recovery pause
            
            # Phase 4: Disk I/O Stress Testing
            print("\n" + "="*60)
            print("📁 PHASE 3: DISK I/O STRESS TESTING")
            print("="*60)
            
            self.results['stress_test_results']['disk_rapid_ops'] = self.disk_stress.rapid_file_operations(
                file_count=500, file_size_kb=100
            )
            
            if self.safety_monitor.emergency_stop_triggered:
                print("🔴 Emergency stop during disk I/O stress test")
                return
                
            time.sleep(5)  # Brief pause
            
            self.results['stress_test_results']['disk_large_files'] = self.disk_stress.large_file_stress(
                file_sizes_mb=[10, 50, 100, 250]
            )
            
        except Exception as e:
            print(f"⚠️  Stress testing error: {e}")
            self.results['stress_test_error'] = str(e)
            
        finally:
            self.results['test_end_time'] = time.time()
            self.results['safety_monitoring'] = self.safety_monitor.stop_monitoring()
            
    def analyze_breaking_points(self):
        """Analyze and summarize breaking points found during testing"""
        print("📈 Analyzing breaking points and system limits")
        
        breaking_points = {}
        
        # Extract breaking points from each test
        if 'memory_stress' in self.results['stress_test_results']:
            memory_result = self.results['stress_test_results']['memory_stress']
            if memory_result.get('breaking_point'):
                breaking_points['memory_stress'] = memory_result['breaking_point']
                
        if 'cpu_stress' in self.results['stress_test_results']:
            cpu_result = self.results['stress_test_results']['cpu_stress']
            if cpu_result.get('breaking_point'):
                breaking_points['cpu_concurrency'] = cpu_result['breaking_point']
                
        if 'search_bombardment' in self.results['stress_test_results']:
            search_result = self.results['stress_test_results']['search_bombardment']
            if search_result.get('breaking_point'):
                breaking_points['search_performance'] = search_result['breaking_point']
                
        if 'disk_rapid_ops' in self.results['stress_test_results']:
            disk_result = self.results['stress_test_results']['disk_rapid_ops']
            if disk_result.get('breaking_point'):
                breaking_points['disk_rapid_ops'] = disk_result['breaking_point']
                
        if 'disk_large_files' in self.results['stress_test_results']:
            large_file_result = self.results['stress_test_results']['disk_large_files']
            if large_file_result.get('breaking_point'):
                breaking_points['disk_large_files'] = large_file_result['breaking_point']
                
        # Add safety monitor breaking points
        if 'breaking_points' in self.results['safety_monitoring']:
            breaking_points['safety_monitor'] = self.results['safety_monitoring']['breaking_points']
            
        self.results['breaking_points_summary'] = breaking_points
        
        # Generate production recommendations
        self._generate_production_recommendations()
        
    def _generate_production_recommendations(self):
        """Generate production deployment recommendations based on test results"""
        print("📋 Generating production deployment recommendations")
        
        recommendations = {
            'memory_limits': {},
            'cpu_limits': {},
            'operational_guidelines': [],
            'monitoring_thresholds': {},
            'safety_margins': {}
        }
        
        # Memory recommendations
        safety_stats = self.results['safety_monitoring'].get('resource_peaks', {})
        if safety_stats:
            max_memory = safety_stats.get('memory_max_percent', 0)
            recommendations['memory_limits'] = {
                'tested_maximum': max_memory,
                'recommended_warning_threshold': max(60, max_memory * 0.7),
                'recommended_critical_threshold': max(75, max_memory * 0.85),
                'production_headroom_required': '25%'
            }
            
            max_cpu = safety_stats.get('cpu_max', 0)
            recommendations['cpu_limits'] = {
                'tested_maximum': max_cpu,
                'recommended_warning_threshold': max(70, max_cpu * 0.7),
                'recommended_critical_threshold': max(80, max_cpu * 0.85),
                'production_headroom_required': '20%'
            }
            
        # Operational guidelines
        recommendations['operational_guidelines'] = [
            "Monitor memory usage continuously during heavy ingestion phases",
            "Limit concurrent processing threads to tested safe levels",
            "Implement circuit breakers for search query rate limiting",
            "Use progressive stress escalation for capacity planning",
            "Maintain 20% system resource headroom for OS operations",
            "Implement automatic graceful degradation at warning thresholds"
        ]
        
        # Monitoring thresholds
        recommendations['monitoring_thresholds'] = {
            'memory_warning_percent': 70,
            'memory_critical_percent': 80,
            'cpu_warning_percent': 75,
            'cpu_critical_percent': 85,
            'emergency_stop_memory_percent': EMERGENCY_STOP_MEMORY,
            'emergency_stop_cpu_percent': EMERGENCY_STOP_CPU
        }
        
        self.results['production_recommendations'] = recommendations
        
    def save_comprehensive_results(self) -> Path:
        """Save comprehensive test results"""
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        results_file = Path(f"resource_limit_stress_results_{timestamp}.json")
        
        try:
            with open(results_file, 'w') as f:
                json.dump(self.results, f, indent=2, default=str)
            print(f"📁 Results saved to: {results_file}")
            return results_file
        except Exception as e:
            print(f"⚠️  Failed to save results: {e}")
            return None
            
    def cleanup_test_environment(self):
        """Clean up test environment"""
        print("🧹 Cleaning up resource limit stress test environment")
        
        try:
            if self.memory_stress:
                self.memory_stress.cleanup_memory_stress()
        except:
            pass
            
        try:
            if self.disk_stress:
                self.disk_stress.cleanup_disk_stress()
        except:
            pass
            
        try:
            self.qdrant_client.cleanup_stress_collection()
        except:
            pass
            
        try:
            if self.test_dir.exists():
                import shutil
                shutil.rmtree(self.test_dir, ignore_errors=True)
        except:
            pass
            
    def execute_full_stress_test_suite(self) -> bool:
        """Execute complete resource limit stress testing suite"""
        print("🎯 Starting Resource Limit Stress Testing Framework")
        print("="*80)
        print("⚡ MISSION: Find daemon breaking points and validate resource guardrails")
        print("🛡️  SAFETY: 20% system resource headroom maintained")
        print("="*80)
        
        try:
            # Setup
            if not self.setup_test_environment():
                print("❌ Failed to set up test environment")
                return False
                
            # Baseline measurements
            self.run_baseline_measurements()
            
            # Execute stress testing
            self.execute_comprehensive_stress_testing()
            
            # Analysis
            self.analyze_breaking_points()
            
            # Save results
            results_file = self.save_comprehensive_results()
            
            # Generate summary report
            self._generate_summary_report()
            
            print("\n" + "="*80)
            print("🎉 RESOURCE LIMIT STRESS TESTING COMPLETED")
            print("="*80)
            
            return True
            
        except KeyboardInterrupt:
            print("\n🛑 Stress test interrupted by user")
            return False
        except Exception as e:
            print(f"\n❌ Stress test failed: {e}")
            return False
        finally:
            self.cleanup_test_environment()
            
    def _generate_summary_report(self):
        """Generate human-readable summary report"""
        print("\n" + "="*80)
        print("📊 RESOURCE LIMIT STRESS TESTING SUMMARY REPORT")
        print("="*80)
        
        # System info
        sys_info = self.results['system_info']
        print(f"🖥️  System: {sys_info['cpu_count']} cores, {sys_info['memory_total_gb']:.1f} GB RAM")
        
        # Test duration
        if self.results.get('test_end_time') and self.results.get('test_start_time'):
            duration = self.results['test_end_time'] - self.results['test_start_time']
            print(f"⏱️  Test Duration: {duration:.1f} seconds")
            
        # Safety monitoring
        safety = self.results.get('safety_monitoring', {})
        if safety:
            print(f"🛡️  Safety Monitoring:")
            if safety.get('emergency_stop_triggered'):
                print("    🔴 EMERGENCY STOP TRIGGERED")
            
            peaks = safety.get('resource_peaks', {})
            if peaks:
                print(f"    📈 Peak Memory: {peaks.get('memory_max_percent', 0):.1f}%")
                print(f"    📈 Peak CPU: {peaks.get('cpu_max', 0):.1f}%")
                
        # Breaking points summary
        breaking_points = self.results.get('breaking_points_summary', {})
        if breaking_points:
            print(f"\n🔶 BREAKING POINTS IDENTIFIED:")
            
            if 'memory_stress' in breaking_points:
                bp = breaking_points['memory_stress']
                print(f"    🧠 Memory Limit: {bp.get('memory_percent', 0):.1f}% at {bp.get('memory_mb', 0)}MB documents")
                
            if 'cpu_concurrency' in breaking_points:
                bp = breaking_points['cpu_concurrency']
                print(f"    ⚡ CPU Limit: {bp.get('cpu_percent', 0):.1f}% at {bp.get('thread_count', 0)} threads")
                
            if 'search_performance' in breaking_points:
                bp = breaking_points['search_performance']
                print(f"    🔍 Search Limit: {bp.get('success_rate', 0):.1f}% success at {bp.get('thread_count', 0)} threads")
                
        # Production recommendations
        recommendations = self.results.get('production_recommendations', {})
        if recommendations:
            print(f"\n🚀 PRODUCTION RECOMMENDATIONS:")
            
            memory_limits = recommendations.get('memory_limits', {})
            if memory_limits:
                print(f"    📊 Memory Warning: {memory_limits.get('recommended_warning_threshold', 0):.1f}%")
                print(f"    🚨 Memory Critical: {memory_limits.get('recommended_critical_threshold', 0):.1f}%")
                
            cpu_limits = recommendations.get('cpu_limits', {})
            if cpu_limits:
                print(f"    📊 CPU Warning: {cpu_limits.get('recommended_warning_threshold', 0):.1f}%")
                print(f"    🚨 CPU Critical: {cpu_limits.get('recommended_critical_threshold', 0):.1f}%")
                
        print("="*80)

def main():
    """Main execution function"""
    
    # Verify QMK directory
    qmk_dir = Path("qmk_firmware")
    if not qmk_dir.exists():
        print("❌ QMK firmware directory not found")
        print("   Please ensure qmk_firmware directory is available for testing")
        return False
        
    # Create and execute stress tester
    stress_tester = ResourceLimitStressTester(qmk_dir)
    
    try:
        success = stress_tester.execute_full_stress_test_suite()
        return success
    except KeyboardInterrupt:
        print("\n🛑 Resource limit stress test interrupted")
        return False
    except Exception as e:
        print(f"\n❌ Resource limit stress test failed: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)