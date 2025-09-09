#!/usr/bin/env python3
"""
Focused Resource Limit Stress Testing - Task #148
Streamlined version for finding daemon breaking points and resource limits

MISSION: Find exact breaking points for memory, CPU, and I/O while validating guardrails
SAFETY: Mandatory 20% system resource reservation with emergency stops
"""

import os
import sys
import time
import json
import psutil
import threading
import uuid
import random
import tempfile
import concurrent.futures
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import requests
import gc

# Safety Constants
MAX_SAFE_MEMORY = 80.0
MAX_SAFE_CPU = 80.0
EMERGENCY_MEMORY = 95.0
EMERGENCY_CPU = 95.0

class SafetyMonitor:
    """Lightweight safety monitoring with emergency stop"""
    
    def __init__(self):
        self.monitoring = False
        self.emergency_stop = False
        self.data = []
        self.breaking_points = {}
        
    def start(self):
        self.monitoring = True
        self.thread = threading.Thread(target=self._monitor)
        self.thread.daemon = True
        self.thread.start()
        
    def stop(self):
        self.monitoring = False
        if hasattr(self, 'thread'):
            self.thread.join(timeout=5)
        return self._get_stats()
        
    def _monitor(self):
        while self.monitoring:
            try:
                memory = psutil.virtual_memory()
                cpu = psutil.cpu_percent()
                
                self.data.append({
                    'time': time.time(),
                    'memory': memory.percent,
                    'cpu': cpu
                })
                
                # Breaking point detection
                if memory.percent > MAX_SAFE_MEMORY and 'memory_warning' not in self.breaking_points:
                    self.breaking_points['memory_warning'] = {
                        'memory_percent': memory.percent,
                        'timestamp': time.time()
                    }
                    print(f"⚠️  MEMORY WARNING: {memory.percent:.1f}%")
                    
                if cpu > MAX_SAFE_CPU and 'cpu_warning' not in self.breaking_points:
                    self.breaking_points['cpu_warning'] = {
                        'cpu_percent': cpu,
                        'timestamp': time.time()
                    }
                    print(f"⚠️  CPU WARNING: {cpu:.1f}%")
                
                # Emergency stop
                if memory.percent > EMERGENCY_MEMORY or cpu > EMERGENCY_CPU:
                    print(f"🔴 EMERGENCY STOP - Memory: {memory.percent:.1f}%, CPU: {cpu:.1f}%")
                    self.emergency_stop = True
                    self.breaking_points['emergency_stop'] = {
                        'memory_percent': memory.percent,
                        'cpu_percent': cpu,
                        'timestamp': time.time()
                    }
                    break
                    
                time.sleep(0.5)
            except:
                break
                
    def _get_stats(self):
        if not self.data:
            return {}
        
        memory_vals = [d['memory'] for d in self.data]
        cpu_vals = [d['cpu'] for d in self.data]
        
        return {
            'peak_memory': max(memory_vals),
            'peak_cpu': max(cpu_vals),
            'avg_memory': sum(memory_vals) / len(memory_vals),
            'avg_cpu': sum(cpu_vals) / len(cpu_vals),
            'breaking_points': self.breaking_points,
            'emergency_stop': self.emergency_stop,
            'samples': len(self.data)
        }

class QuickQdrantClient:
    """Simplified Qdrant client for stress testing"""
    
    def __init__(self):
        self.base_url = "http://localhost:6333"
        self.collection = f"stress_test_{uuid.uuid4().hex[:8]}"
        
    def create_collection(self):
        try:
            config = {
                "vectors": {"size": 384, "distance": "Cosine"},
                "replication_factor": 1
            }
            response = requests.put(f"{self.base_url}/collections/{self.collection}", json=config)
            return response.status_code in [200, 201]
        except:
            return False
            
    def stress_upsert(self, doc_count: int, doc_size_kb: int = 10):
        """Stress test document upserts"""
        print(f"📝 Upserting {doc_count} documents ({doc_size_kb}KB each)")
        
        docs = []
        content = 'X' * (doc_size_kb * 1024)  # Generate content
        
        for i in range(doc_count):
            docs.append({
                "id": i,
                "vector": [random.random() for _ in range(384)],
                "payload": {"content": content, "size_kb": doc_size_kb}
            })
            
        start_time = time.time()
        try:
            response = requests.put(
                f"{self.base_url}/collections/{self.collection}/points",
                json={"points": docs}
            )
            success = response.status_code in [200, 201]
            duration = time.time() - start_time
            
            return {
                'success': success,
                'duration': duration,
                'throughput': doc_count / duration if duration > 0 else 0,
                'status_code': response.status_code
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'duration': time.time() - start_time
            }
            
    def stress_search(self, query_count: int, threads: int = 5):
        """Stress test search queries"""
        print(f"🔍 Running {query_count} searches with {threads} threads")
        
        def run_search():
            try:
                query = {
                    "vector": [random.random() for _ in range(384)],
                    "limit": 10
                }
                response = requests.post(
                    f"{self.base_url}/collections/{self.collection}/points/search",
                    json=query
                )
                return {
                    'success': response.status_code == 200,
                    'time': response.elapsed.total_seconds()
                }
            except:
                return {'success': False, 'time': 0}
                
        start_time = time.time()
        with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as executor:
            futures = [executor.submit(run_search) for _ in range(query_count)]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]
            
        duration = time.time() - start_time
        successful = [r for r in results if r['success']]
        
        return {
            'total_queries': query_count,
            'successful': len(successful),
            'success_rate': len(successful) / query_count * 100,
            'duration': duration,
            'qps': query_count / duration if duration > 0 else 0,
            'avg_response_time': sum(r['time'] for r in successful) / len(successful) if successful else 0
        }
        
    def cleanup(self):
        try:
            requests.delete(f"{self.base_url}/collections/{self.collection}")
        except:
            pass

class ResourceStressTester:
    """Focused resource limit stress tester"""
    
    def __init__(self):
        self.safety = SafetyMonitor()
        self.qdrant = QuickQdrantClient()
        self.test_dir = Path(f"LT_stress_test_{int(time.time())}")
        self.results = {
            'start_time': time.time(),
            'system_info': self._get_system_info(),
            'tests': {},
            'breaking_points': {},
            'recommendations': {}
        }
        
    def _get_system_info(self):
        memory = psutil.virtual_memory()
        return {
            'cpu_count': psutil.cpu_count(),
            'memory_gb': memory.total / 1024 / 1024 / 1024,
            'baseline_memory_percent': memory.percent,
            'baseline_cpu_percent': psutil.cpu_percent(interval=1)
        }
        
    def setup(self):
        print("🏗️  Setting up focused resource stress test")
        
        # Create test directory
        self.test_dir.mkdir(exist_ok=True)
        
        # Setup Qdrant collection
        if not self.qdrant.create_collection():
            print("❌ Failed to create Qdrant collection")
            return False
            
        # Start safety monitoring
        self.safety.start()
        
        print("✅ Setup complete")
        return True
        
    def test_memory_pressure(self):
        """Progressive memory stress testing"""
        print("\n" + "="*50)
        print("🧠 MEMORY PRESSURE STRESS TEST")
        print("="*50)
        
        results = {'phases': []}
        
        # Progressive document sizes: 50KB, 100KB, 500KB, 1MB, 5MB
        sizes_kb = [50, 100, 500, 1000, 5000]
        doc_counts = [100, 200, 100, 50, 20]  # Adjust counts for larger sizes
        
        for size_kb, doc_count in zip(sizes_kb, doc_counts):
            if self.safety.emergency_stop:
                break
                
            print(f"📈 Phase: {doc_count} documents × {size_kb}KB = {doc_count * size_kb / 1024:.1f}MB")
            
            memory_before = psutil.virtual_memory().percent
            
            result = self.qdrant.stress_upsert(doc_count, size_kb)
            
            memory_after = psutil.virtual_memory().percent
            memory_delta = memory_after - memory_before
            
            phase_result = {
                'size_kb': size_kb,
                'doc_count': doc_count,
                'total_mb': doc_count * size_kb / 1024,
                'memory_before': memory_before,
                'memory_after': memory_after,
                'memory_delta': memory_delta,
                'upsert_result': result
            }
            
            results['phases'].append(phase_result)
            
            print(f"   Memory: {memory_before:.1f}% → {memory_after:.1f}% (Δ{memory_delta:+.1f}%)")
            if result['success']:
                print(f"   Throughput: {result['throughput']:.1f} docs/sec")
            else:
                print(f"   ❌ Upsert failed")
                break
                
            # Check for breaking point
            if memory_after > MAX_SAFE_MEMORY or not result['success']:
                print(f"🔶 Memory breaking point reached")
                self.results['breaking_points']['memory'] = {
                    'size_kb': size_kb,
                    'memory_percent': memory_after,
                    'reason': 'memory_limit' if memory_after > MAX_SAFE_MEMORY else 'upsert_failure'
                }
                break
                
            time.sleep(5)  # Brief recovery pause
            
        self.results['tests']['memory_pressure'] = results
        
    def test_cpu_saturation(self):
        """CPU saturation stress testing"""
        print("\n" + "="*50)
        print("⚡ CPU SATURATION STRESS TEST")
        print("="*50)
        
        results = {'phases': []}
        
        # Progressive thread counts and query loads
        test_configs = [
            (50, 2),    # 50 queries, 2 threads
            (100, 5),   # 100 queries, 5 threads
            (200, 10),  # 200 queries, 10 threads
            (500, 20),  # 500 queries, 20 threads
            (1000, 30), # 1000 queries, 30 threads
        ]
        
        for query_count, threads in test_configs:
            if self.safety.emergency_stop:
                break
                
            print(f"🔄 Phase: {query_count} queries × {threads} threads")
            
            cpu_before = psutil.cpu_percent(interval=1)
            
            result = self.qdrant.stress_search(query_count, threads)
            
            cpu_after = psutil.cpu_percent(interval=1)
            
            phase_result = {
                'query_count': query_count,
                'threads': threads,
                'cpu_before': cpu_before,
                'cpu_after': cpu_after,
                'search_result': result
            }
            
            results['phases'].append(phase_result)
            
            print(f"   CPU: {cpu_before:.1f}% → {cpu_after:.1f}%")
            print(f"   Success rate: {result['success_rate']:.1f}%")
            print(f"   QPS: {result['qps']:.1f}")
            
            # Check for breaking point
            if cpu_after > MAX_SAFE_CPU or result['success_rate'] < 80:
                print(f"🔶 CPU breaking point reached")
                self.results['breaking_points']['cpu'] = {
                    'threads': threads,
                    'cpu_percent': cpu_after,
                    'success_rate': result['success_rate'],
                    'reason': 'cpu_limit' if cpu_after > MAX_SAFE_CPU else 'performance_degradation'
                }
                break
                
            time.sleep(3)  # Brief pause
            
        self.results['tests']['cpu_saturation'] = results
        
    def test_disk_io_stress(self):
        """Disk I/O stress testing"""
        print("\n" + "="*50)
        print("📁 DISK I/O STRESS TEST")
        print("="*50)
        
        results = {'phases': []}
        
        # Progressive file operations
        test_configs = [
            (100, 10),   # 100 files × 10KB
            (500, 50),   # 500 files × 50KB
            (1000, 100), # 1000 files × 100KB
            (2000, 200), # 2000 files × 200KB
        ]
        
        for file_count, file_size_kb in test_configs:
            if self.safety.emergency_stop:
                break
                
            print(f"📄 Phase: {file_count} files × {file_size_kb}KB")
            
            start_time = time.time()
            memory_before = psutil.virtual_memory().percent
            
            # File creation phase
            files_created = []
            content = 'Y' * (file_size_kb * 1024)
            
            creation_start = time.time()
            try:
                for i in range(file_count):
                    if self.safety.emergency_stop:
                        break
                        
                    file_path = self.test_dir / f"test_{i:06d}.dat"
                    with open(file_path, 'w') as f:
                        f.write(content)
                    files_created.append(file_path)
                    
                creation_time = time.time() - creation_start
                
                # File deletion phase
                deletion_start = time.time()
                for file_path in files_created:
                    file_path.unlink(missing_ok=True)
                deletion_time = time.time() - deletion_start
                
                memory_after = psutil.virtual_memory().percent
                total_time = time.time() - start_time
                
                phase_result = {
                    'file_count': file_count,
                    'file_size_kb': file_size_kb,
                    'total_mb': file_count * file_size_kb / 1024,
                    'files_created': len(files_created),
                    'creation_time': creation_time,
                    'deletion_time': deletion_time,
                    'total_time': total_time,
                    'memory_before': memory_before,
                    'memory_after': memory_after,
                    'create_rate': len(files_created) / creation_time if creation_time > 0 else 0,
                    'delete_rate': len(files_created) / deletion_time if deletion_time > 0 else 0
                }
                
                results['phases'].append(phase_result)
                
                print(f"   Created: {len(files_created)} files in {creation_time:.1f}s ({phase_result['create_rate']:.1f} files/sec)")
                print(f"   Deleted: {len(files_created)} files in {deletion_time:.1f}s ({phase_result['delete_rate']:.1f} files/sec)")
                print(f"   Memory: {memory_before:.1f}% → {memory_after:.1f}%")
                
                # Check for breaking point
                if memory_after > MAX_SAFE_MEMORY or phase_result['create_rate'] < 10:
                    print(f"🔶 Disk I/O breaking point reached")
                    self.results['breaking_points']['disk_io'] = {
                        'file_count': file_count,
                        'memory_percent': memory_after,
                        'create_rate': phase_result['create_rate'],
                        'reason': 'memory_limit' if memory_after > MAX_SAFE_MEMORY else 'performance_degradation'
                    }
                    break
                    
            except Exception as e:
                print(f"❌ Disk I/O error: {e}")
                self.results['breaking_points']['disk_io'] = {
                    'file_count': file_count,
                    'error': str(e),
                    'reason': 'io_error'
                }
                break
                
            time.sleep(2)  # Brief pause
            
        self.results['tests']['disk_io_stress'] = results
        
    def generate_recommendations(self):
        """Generate production recommendations from test results"""
        print("\n" + "="*50)
        print("📋 GENERATING PRODUCTION RECOMMENDATIONS")
        print("="*50)
        
        safety_stats = self.safety._get_stats()
        recommendations = {
            'resource_limits': {
                'memory_warning_threshold': 70,
                'memory_critical_threshold': 80,
                'cpu_warning_threshold': 70,
                'cpu_critical_threshold': 80
            },
            'operational_guidelines': [
                f"Peak memory observed: {safety_stats.get('peak_memory', 0):.1f}%",
                f"Peak CPU observed: {safety_stats.get('peak_cpu', 0):.1f}%",
                "Maintain 20% system resource headroom for OS operations",
                "Implement progressive stress escalation for capacity planning",
                "Use circuit breakers for query rate limiting during high load"
            ],
            'breaking_points_found': len(self.results['breaking_points']),
            'emergency_stops': 1 if safety_stats.get('emergency_stop') else 0
        }
        
        # Add specific breaking point recommendations
        for bp_type, bp_data in self.results['breaking_points'].items():
            if bp_type == 'memory':
                recommendations['operational_guidelines'].append(
                    f"Memory limit reached at {bp_data.get('memory_percent', 0):.1f}% with {bp_data.get('size_kb', 0)}KB documents"
                )
            elif bp_type == 'cpu':
                recommendations['operational_guidelines'].append(
                    f"CPU limit reached at {bp_data.get('cpu_percent', 0):.1f}% with {bp_data.get('threads', 0)} concurrent threads"
                )
                
        self.results['recommendations'] = recommendations
        
    def run_complete_test(self):
        """Execute complete focused stress testing"""
        print("🎯 FOCUSED RESOURCE LIMIT STRESS TESTING")
        print("="*60)
        print("Mission: Find daemon breaking points with safety guardrails")
        print("="*60)
        
        try:
            if not self.setup():
                return False
                
            # Run all stress tests
            self.test_memory_pressure()
            if not self.safety.emergency_stop:
                self.test_cpu_saturation()
            if not self.safety.emergency_stop:
                self.test_disk_io_stress()
                
            # Analysis
            self.generate_recommendations()
            
            # Save results
            self.results['end_time'] = time.time()
            self.results['duration'] = self.results['end_time'] - self.results['start_time']
            self.results['safety_monitoring'] = self.safety.stop()
            
            # Save to file
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            results_file = f"focused_resource_stress_results_{timestamp}.json"
            
            with open(results_file, 'w') as f:
                json.dump(self.results, f, indent=2, default=str)
                
            # Generate summary
            self._print_summary()
            
            print(f"💾 Results saved to: {results_file}")
            return True
            
        except KeyboardInterrupt:
            print("\n🛑 Test interrupted by user")
            return False
        except Exception as e:
            print(f"\n❌ Test failed: {e}")
            return False
        finally:
            self._cleanup()
            
    def _print_summary(self):
        """Print test summary"""
        print("\n" + "="*60)
        print("📊 RESOURCE LIMIT STRESS TEST SUMMARY")
        print("="*60)
        
        sys_info = self.results['system_info']
        print(f"🖥️  System: {sys_info['cpu_count']} cores, {sys_info['memory_gb']:.1f} GB RAM")
        print(f"⏱️  Duration: {self.results['duration']:.1f} seconds")
        
        safety = self.results['safety_monitoring']
        print(f"📈 Peak Usage: Memory {safety.get('peak_memory', 0):.1f}%, CPU {safety.get('peak_cpu', 0):.1f}%")
        
        if self.results['breaking_points']:
            print(f"\n🔶 BREAKING POINTS FOUND: {len(self.results['breaking_points'])}")
            for bp_type, bp_data in self.results['breaking_points'].items():
                print(f"   {bp_type.upper()}: {bp_data}")
        else:
            print(f"\n✅ NO BREAKING POINTS REACHED (within test parameters)")
            
        if safety.get('emergency_stop'):
            print(f"\n🔴 EMERGENCY STOP TRIGGERED")
            
        rec = self.results['recommendations']
        print(f"\n📋 RECOMMENDED PRODUCTION LIMITS:")
        print(f"   Memory Warning: {rec['resource_limits']['memory_warning_threshold']}%")
        print(f"   Memory Critical: {rec['resource_limits']['memory_critical_threshold']}%")
        print(f"   CPU Warning: {rec['resource_limits']['cpu_warning_threshold']}%")
        print(f"   CPU Critical: {rec['resource_limits']['cpu_critical_threshold']}%")
        
    def _cleanup(self):
        """Cleanup test resources"""
        print("\n🧹 Cleaning up test resources")
        
        try:
            self.qdrant.cleanup()
        except:
            pass
            
        try:
            if self.test_dir.exists():
                import shutil
                shutil.rmtree(self.test_dir, ignore_errors=True)
        except:
            pass
            
        try:
            self.safety.stop()
        except:
            pass
            
        gc.collect()

def main():
    tester = ResourceStressTester()
    return tester.run_complete_test()

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)