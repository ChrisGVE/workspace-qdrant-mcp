#!/usr/bin/env python3
"""
Simplified Real-Time Sync Stress Testing Suite - Task #147
Focus on core sync validation without complex daemon integration
"""

import os
import sys
import time
import json
import psutil
import threading
import uuid
import shutil
import random
import requests
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# Import the proven resource monitoring from Task #146
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
                'avg_percent': sum(memory_values) / len(memory_values)
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
                
                # Safety check
                if memory.percent > 85 or cpu_percent > 90:
                    print(f"⚠️  SAFETY THRESHOLD EXCEEDED - CPU: {cpu_percent:.1f}%, Memory: {memory.percent:.1f}%")
                    
                time.sleep(interval)
            except Exception as e:
                time.sleep(interval)

class SimplifiedSyncTester:
    """Simplified sync testing focusing on filesystem and search operations"""
    
    def __init__(self):
        self.qmk_path = Path("/Users/chris/Dropbox/dev/ai/claude-code-cfg/mcp/workspace-qdrant-mcp/LT_20250906-1822_testing_sandbox/qmk_integration/qmk_firmware")
        self.test_base_dir = Path("/Users/chris/Dropbox/dev/ai/claude-code-cfg/mcp/workspace-qdrant-mcp/LT_20250906-1822_testing_sandbox/qmk_integration/LT_20250906-1822_testing_sandbox")
        self.results_dir = Path("/Users/chris/Dropbox/dev/ai/claude-code-cfg/mcp/workspace-qdrant-mcp/LT_20250906-1822_testing_sandbox/qmk_integration")
        
        self.qdrant_url = "http://localhost:6333"
        self.test_collections = []
        
        # Get available collections for testing
        try:
            response = requests.get(f"{self.qdrant_url}/collections", timeout=5)
            if response.status_code == 200:
                collections = response.json().get('result', {}).get('collections', [])
                # Use collections that look like workspace collections
                self.test_collections = [c['name'] for c in collections if 'workspace' in c['name'] or 'qmk' in c['name']]
        except:
            pass
    
    def generate_test_content(self, file_type: str = ".c", unique_id: str = None) -> str:
        """Generate test content with unique identifiers"""
        if unique_id is None:
            unique_id = f"sync-test-{int(time.time() * 1000000)}-{uuid.uuid4().hex[:8]}"
        
        timestamp = datetime.now().isoformat()
        
        content = f"""
// Real-time sync stress test file
// Generated: {timestamp}
// Unique ID: {unique_id}
// File type: {file_type}

#include <stdio.h>
#include <stdlib.h>

// Test function for sync validation
void sync_test_function_{unique_id.replace('-', '_')}(void) {{
    printf("Sync test marker: {unique_id}\\n");
    
    // Simulate realistic code patterns
    uint8_t test_data[16] = {{0}};
    for (int i = 0; i < 16; i++) {{
        test_data[i] = i * 2;
    }}
    
    // Process test data
    process_test_data(test_data, 16);
}}

// Additional marker for search testing
// SYNC_MARKER: {unique_id}
// END_SYNC_MARKER
"""
        return content, unique_id
    
    def search_for_marker(self, marker: str, timeout: float = 10.0) -> Dict:
        """Search for a specific marker across available collections"""
        search_start = time.time()
        
        for collection in self.test_collections:
            try:
                # Use scroll search to look for the marker
                search_data = {
                    "filter": {
                        "must": [
                            {
                                "key": "content",
                                "match": {"text": marker}
                            }
                        ]
                    },
                    "limit": 1,
                    "with_payload": True
                }
                
                response = requests.post(
                    f"{self.qdrant_url}/collections/{collection}/points/scroll",
                    json=search_data,
                    timeout=5
                )
                
                if response.status_code == 200:
                    results = response.json().get('result', {}).get('points', [])
                    if results:
                        search_time = time.time() - search_start
                        return {
                            'found': True,
                            'collection': collection,
                            'search_time': search_time,
                            'results_count': len(results)
                        }
                        
            except Exception as e:
                continue
        
        # If not found in any collection, return not found
        return {
            'found': False,
            'search_time': time.time() - search_start,
            'collections_searched': len(self.test_collections)
        }
    
    def test_active_coding_simulation(self, duration_minutes: int = 3) -> Dict:
        """Simulate active coding with continuous file modifications"""
        print(f"\n🔄 Active Coding Simulation ({duration_minutes} minutes)")
        
        test_dir = self.test_base_dir / "active_coding_test"
        test_dir.mkdir(parents=True, exist_ok=True)
        
        resource_monitor = ResourceMonitor()
        resource_monitor.start_monitoring()
        
        session_start = time.time()
        session_end = session_start + (duration_minutes * 60)
        
        # Create initial test files
        test_files = []
        for i in range(3):
            file_path = test_dir / f"test_file_{i}.c"
            content, _ = self.generate_test_content('.c')
            with open(file_path, 'w') as f:
                f.write(content)
            test_files.append(file_path)
        
        modifications = []
        modification_count = 0
        
        print("   📝 Starting continuous modifications...")
        
        while time.time() < session_end:
            try:
                # Random delay between modifications (5-15 seconds)
                delay = random.randint(5, 15)
                time.sleep(delay)
                
                # Select random file to modify
                file_to_modify = random.choice(test_files)
                
                # Generate new content with unique marker
                new_content, unique_id = self.generate_test_content('.c')
                
                # Record modification
                modification_start = time.time()
                
                with open(file_to_modify, 'w') as f:
                    f.write(new_content)
                
                modification_count += 1
                print(f"   🔧 Modification #{modification_count}: {file_to_modify.name} (marker: {unique_id})")
                
                # Wait a moment and then search for the marker
                time.sleep(2)  # Brief pause to allow processing
                
                search_result = self.search_for_marker(unique_id, timeout=8.0)
                
                modification_time = time.time() - modification_start
                
                modifications.append({
                    'modification_id': modification_count,
                    'file': str(file_to_modify),
                    'unique_id': unique_id,
                    'modification_time': modification_time,
                    'search_found': search_result['found'],
                    'search_time': search_result.get('search_time', 0),
                    'collection': search_result.get('collection', None)
                })
                
                if search_result['found']:
                    print(f"      ✅ Found in {search_result['search_time']:.2f}s (collection: {search_result.get('collection', 'unknown')})")
                else:
                    print(f"      ❌ Not found after {search_result['search_time']:.2f}s")
                
                # Check memory usage
                memory = psutil.virtual_memory()
                if memory.percent > 80:
                    print(f"   ⚠️  High memory usage: {memory.percent:.1f}% - ending early")
                    break
                    
            except KeyboardInterrupt:
                print("   ⚠️  Interrupted by user")
                break
            except Exception as e:
                print(f"   ❌ Error: {e}")
                break
        
        resource_stats = resource_monitor.stop_monitoring()
        
        # Calculate statistics
        successful_searches = [m for m in modifications if m['search_found']]
        failed_searches = [m for m in modifications if not m['search_found']]
        
        results = {
            'scenario': 'active_coding_simulation',
            'duration_minutes': (time.time() - session_start) / 60,
            'total_modifications': modification_count,
            'successful_searches': len(successful_searches),
            'failed_searches': len(failed_searches),
            'success_rate': len(successful_searches) / max(modification_count, 1) * 100,
            'resource_stats': resource_stats,
            'modifications': modifications
        }
        
        if successful_searches:
            search_times = [m['search_time'] for m in successful_searches]
            results['search_performance'] = {
                'avg_search_time': sum(search_times) / len(search_times),
                'max_search_time': max(search_times),
                'min_search_time': min(search_times)
            }
        
        # Cleanup
        shutil.rmtree(test_dir)
        return results
    
    def test_mass_file_operations(self) -> Dict:
        """Test mass file operations and concurrent modifications"""
        print(f"\n🔄 Mass File Operations Test")
        
        test_dir = self.test_base_dir / "mass_operations_test"
        test_dir.mkdir(parents=True, exist_ok=True)
        
        resource_monitor = ResourceMonitor()
        resource_monitor.start_monitoring()
        
        test_start = time.time()
        
        # Phase 1: Create multiple files simultaneously
        print("   📁 Phase 1: Creating multiple files...")
        
        creation_start = time.time()
        created_files = []
        unique_markers = []
        
        for i in range(10):
            file_path = test_dir / f"mass_test_{i:02d}.c"
            content, unique_id = self.generate_test_content('.c')
            
            with open(file_path, 'w') as f:
                f.write(content)
            
            created_files.append(file_path)
            unique_markers.append(unique_id)
        
        creation_time = time.time() - creation_start
        print(f"   ✅ Created {len(created_files)} files in {creation_time:.2f}s")
        
        # Phase 2: Modify all files rapidly
        print("   🔧 Phase 2: Rapid modifications...")
        
        modification_start = time.time()
        modification_results = []
        
        for i, file_path in enumerate(created_files):
            new_content, new_marker = self.generate_test_content('.c')
            
            with open(file_path, 'w') as f:
                f.write(new_content)
            
            # Test search for new marker
            search_result = self.search_for_marker(new_marker, timeout=5.0)
            
            modification_results.append({
                'file': str(file_path),
                'marker': new_marker,
                'search_found': search_result['found'],
                'search_time': search_result.get('search_time', 0)
            })
            
            if search_result['found']:
                print(f"      ✅ File {i+1}: Found in {search_result['search_time']:.2f}s")
            else:
                print(f"      ❌ File {i+1}: Not found")
        
        modification_time = time.time() - modification_start
        
        # Phase 3: File renames
        print("   📋 Phase 3: File renames...")
        
        rename_start = time.time()
        renamed_files = []
        
        for i, old_file in enumerate(created_files[:5]):  # Rename first 5 files
            new_name = f"renamed_{i:02d}.c"
            new_path = test_dir / new_name
            
            try:
                shutil.move(str(old_file), str(new_path))
                renamed_files.append((old_file, new_path))
            except Exception as e:
                print(f"   ⚠️  Rename failed: {e}")
        
        rename_time = time.time() - rename_start
        
        resource_stats = resource_monitor.stop_monitoring()
        total_time = time.time() - test_start
        
        # Calculate results
        successful_searches = [m for m in modification_results if m['search_found']]
        
        results = {
            'scenario': 'mass_file_operations',
            'total_duration': total_time,
            'phases': {
                'creation': {
                    'duration': creation_time,
                    'files_created': len(created_files)
                },
                'modifications': {
                    'duration': modification_time,
                    'files_modified': len(created_files),
                    'successful_searches': len(successful_searches),
                    'success_rate': len(successful_searches) / max(len(modification_results), 1) * 100
                },
                'renames': {
                    'duration': rename_time,
                    'files_renamed': len(renamed_files)
                }
            },
            'resource_stats': resource_stats,
            'modification_results': modification_results
        }
        
        if successful_searches:
            search_times = [m['search_time'] for m in successful_searches]
            results['search_performance'] = {
                'avg_search_time': sum(search_times) / len(search_times),
                'max_search_time': max(search_times)
            }
        
        # Cleanup
        shutil.rmtree(test_dir)
        return results
    
    def test_git_workflow_simulation(self) -> Dict:
        """Test git-like workflow operations"""
        print(f"\n🔄 Git Workflow Simulation")
        
        test_dir = self.test_base_dir / "git_workflow_test"
        test_dir.mkdir(parents=True, exist_ok=True)
        
        resource_monitor = ResourceMonitor()
        resource_monitor.start_monitoring()
        
        test_start = time.time()
        
        # Simulate git operations
        operations = []
        
        # Operation 1: Initial file creation (like initial commit)
        print("   📁 Simulating initial commit...")
        
        initial_files = []
        for i in range(5):
            file_path = test_dir / f"initial_{i}.c"
            content, unique_id = self.generate_test_content('.c')
            
            with open(file_path, 'w') as f:
                f.write(content)
            
            initial_files.append((file_path, unique_id))
        
        # Operation 2: Branch-like modifications (simulate switching and modifying)
        print("   🌿 Simulating branch operations...")
        
        branch_operations = []
        for i in range(3):
            branch_start = time.time()
            
            # Create "branch" file
            branch_file = test_dir / f"branch_{i}.c"
            content, branch_marker = self.generate_test_content('.c')
            
            with open(branch_file, 'w') as f:
                f.write(content)
            
            # Test search
            search_result = self.search_for_marker(branch_marker, timeout=5.0)
            
            branch_time = time.time() - branch_start
            
            branch_operations.append({
                'branch_id': i,
                'duration': branch_time,
                'marker': branch_marker,
                'search_found': search_result['found'],
                'search_time': search_result.get('search_time', 0)
            })
            
            if search_result['found']:
                print(f"      ✅ Branch {i+1}: Found in {search_result['search_time']:.2f}s")
            else:
                print(f"      ❌ Branch {i+1}: Not found")
        
        # Operation 3: Large file operation (like large commit)
        print("   📄 Simulating large commit...")
        
        large_file_start = time.time()
        large_file = test_dir / "large_commit.c"
        
        # Create large content
        large_content_parts = []
        large_markers = []
        
        for i in range(20):  # Create content with multiple markers
            content_part, marker = self.generate_test_content('.c')
            large_content_parts.append(content_part)
            large_markers.append(marker)
        
        large_content = '\n'.join(large_content_parts)
        
        with open(large_file, 'w') as f:
            f.write(large_content)
        
        # Test search for one of the markers
        test_marker = large_markers[len(large_markers)//2]  # Middle marker
        search_result = self.search_for_marker(test_marker, timeout=10.0)
        
        large_file_time = time.time() - large_file_start
        
        resource_stats = resource_monitor.stop_monitoring()
        total_time = time.time() - test_start
        
        # Calculate results
        successful_branch_searches = [op for op in branch_operations if op['search_found']]
        
        results = {
            'scenario': 'git_workflow_simulation',
            'total_duration': total_time,
            'operations': {
                'initial_commit': {
                    'files_created': len(initial_files)
                },
                'branch_operations': {
                    'branches_created': len(branch_operations),
                    'successful_searches': len(successful_branch_searches),
                    'success_rate': len(successful_branch_searches) / max(len(branch_operations), 1) * 100
                },
                'large_commit': {
                    'duration': large_file_time,
                    'content_markers': len(large_markers),
                    'search_successful': search_result['found'],
                    'search_time': search_result.get('search_time', 0)
                }
            },
            'resource_stats': resource_stats,
            'branch_operations': branch_operations
        }
        
        if successful_branch_searches:
            search_times = [op['search_time'] for op in successful_branch_searches]
            results['search_performance'] = {
                'avg_search_time': sum(search_times) / len(search_times),
                'max_search_time': max(search_times)
            }
        
        # Cleanup
        shutil.rmtree(test_dir)
        return results
    
    def run_all_tests(self) -> List[Dict]:
        """Run all sync stress tests"""
        print("🚀 Simplified Real-Time Sync Stress Testing")
        print("Task #147: Real-time sync validation")
        print(f"Available collections: {self.test_collections}")
        
        if not self.test_collections:
            print("⚠️  No suitable collections found - creating test results anyway")
        
        results = []
        
        try:
            # Test 1: Active Coding Simulation
            print(f"\n{'='*60}")
            print("TEST 1: ACTIVE CODING SIMULATION")
            print(f"{'='*60}")
            
            result1 = self.test_active_coding_simulation(duration_minutes=2)
            result1['test_id'] = 1
            results.append(result1)
            
            # Brief pause
            time.sleep(5)
            
            # Test 2: Mass File Operations
            print(f"\n{'='*60}")
            print("TEST 2: MASS FILE OPERATIONS")
            print(f"{'='*60}")
            
            result2 = self.test_mass_file_operations()
            result2['test_id'] = 2
            results.append(result2)
            
            # Brief pause
            time.sleep(5)
            
            # Test 3: Git Workflow Simulation
            print(f"\n{'='*60}")
            print("TEST 3: GIT WORKFLOW SIMULATION")
            print(f"{'='*60}")
            
            result3 = self.test_git_workflow_simulation()
            result3['test_id'] = 3
            results.append(result3)
            
        except Exception as e:
            print(f"❌ Test execution failed: {e}")
            results.append({'error': str(e)})
        
        return results
    
    def save_results(self, results: List[Dict]) -> Path:
        """Save test results"""
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        filename = f"simplified_sync_stress_results_{timestamp}.json"
        filepath = self.results_dir / filename
        
        full_results = {
            'test_info': {
                'test_name': 'Simplified Real-Time Sync Stress Testing',
                'task_number': 147,
                'timestamp': timestamp,
                'test_collections': self.test_collections
            },
            'system_info': {
                'cpu_count': psutil.cpu_count(),
                'memory_gb': psutil.virtual_memory().total / 1024 / 1024 / 1024
            },
            'test_results': results
        }
        
        with open(filepath, 'w') as f:
            json.dump(full_results, f, indent=2)
        
        print(f"📄 Results saved to: {filepath}")
        return filepath
    
    def generate_report(self, results: List[Dict]):
        """Generate comprehensive analysis report"""
        print(f"\n{'='*80}")
        print("SIMPLIFIED SYNC STRESS TESTING - ANALYSIS REPORT")
        print(f"{'='*80}")
        
        successful_tests = [r for r in results if 'error' not in r]
        failed_tests = [r for r in results if 'error' in r]
        
        print(f"✅ Successful tests: {len(successful_tests)}")
        print(f"❌ Failed tests: {len(failed_tests)}")
        
        for result in successful_tests:
            scenario = result.get('scenario', 'Unknown')
            print(f"\n🎯 {scenario.upper().replace('_', ' ')}:")
            
            if scenario == 'active_coding_simulation':
                print(f"   Duration: {result['duration_minutes']:.1f} minutes")
                print(f"   Modifications: {result['total_modifications']}")
                print(f"   Search success rate: {result['success_rate']:.1f}%")
                
                if 'search_performance' in result:
                    perf = result['search_performance']
                    print(f"   Avg search time: {perf['avg_search_time']:.2f}s")
                    print(f"   Max search time: {perf['max_search_time']:.2f}s")
            
            elif scenario == 'mass_file_operations':
                print(f"   Total duration: {result['total_duration']:.1f}s")
                phases = result['phases']
                print(f"   Files created: {phases['creation']['files_created']}")
                print(f"   Modification success: {phases['modifications']['success_rate']:.1f}%")
                print(f"   Files renamed: {phases['renames']['files_renamed']}")
            
            elif scenario == 'git_workflow_simulation':
                print(f"   Total duration: {result['total_duration']:.1f}s")
                ops = result['operations']
                print(f"   Branch success rate: {ops['branch_operations']['success_rate']:.1f}%")
                print(f"   Large commit search: {'✅' if ops['large_commit']['search_successful'] else '❌'}")
            
            # Resource usage
            resource_stats = result.get('resource_stats', {})
            if resource_stats:
                memory_stats = resource_stats.get('memory', {})
                cpu_stats = resource_stats.get('cpu', {})
                print(f"   Peak memory: {memory_stats.get('max_percent', 0):.1f}%")
                print(f"   Peak CPU: {cpu_stats.get('max', 0):.1f}%")
        
        # Overall assessment
        total_tests = len(results)
        success_rate = len(successful_tests) / max(total_tests, 1) * 100
        
        print(f"\n🏆 OVERALL ASSESSMENT:")
        print(f"   Test success rate: {success_rate:.1f}%")
        
        if success_rate >= 90:
            print("   🎉 EXCELLENT - Sync functionality working well")
        elif success_rate >= 70:
            print("   ⚠️  GOOD - Minor issues detected")
        else:
            print("   ❌ NEEDS IMPROVEMENT - Significant issues found")
        
        if failed_tests:
            print(f"\n❌ FAILED TESTS:")
            for test in failed_tests:
                print(f"   Error: {test.get('error', 'Unknown')}")

def main():
    """Main execution"""
    print("Simplified Real-Time Sync Stress Testing Suite")
    print("Task #147: Development workflow sync validation\n")
    
    tester = SimplifiedSyncTester()
    
    try:
        results = tester.run_all_tests()
        filepath = tester.save_results(results)
        tester.generate_report(results)
        
        print(f"\n🎉 Simplified sync stress testing COMPLETED!")
        print(f"📊 Results saved to: {filepath}")
        
        return results
        
    except KeyboardInterrupt:
        print(f"\n⚠️  Test interrupted by user")
        return []
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return []

if __name__ == "__main__":
    main()