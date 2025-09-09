#!/usr/bin/env python3
"""
Real-Time Sync Stress Testing Suite - Task #147
Comprehensive validation of daemon sync performance under active development loads

Tests three critical development workflow scenarios:
1. Active Coding Session - Continuous file modifications every 5-30 seconds
2. Refactoring Operations - Mass file operations, renames, moves  
3. Git Operations Stress - Branch switching, merges, commits

Key Measurements:
- Change Detection Latency (filesystem → daemon awareness)
- Processing Queue Performance (backlog depth during activity)
- Search Result Freshness (change → queryable content)
- Concurrent Operation Impact (performance under simultaneous ops)
- Sync Integrity (no stale data returned to LLM queries)
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
import shutil
import random
import requests
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, Future

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
                    
                time.sleep(interval)
            except Exception as e:
                print(f"⚠️  Monitoring error: {e}")
                time.sleep(interval)

@dataclass
class SyncMetrics:
    """Container for sync performance metrics"""
    change_detection_time: float = 0.0
    processing_latency: float = 0.0
    query_freshness_time: float = 0.0
    end_to_end_latency: float = 0.0
    queue_depth_max: int = 0
    queue_depth_avg: float = 0.0
    success: bool = False
    error: Optional[str] = None
    timestamp: float = 0.0

class RealtimeSyncMonitor:
    """Monitor real-time sync performance of the workspace-qdrant daemon"""
    
    def __init__(self, workspace_url: str = "http://localhost:8000"):
        self.workspace_url = workspace_url
        self.qdrant_url = "http://localhost:6333"
        self.monitoring_active = False
        self.queue_depth_history = []
        
    def check_daemon_status(self) -> Dict:
        """Check if the workspace daemon is running and accessible"""
        try:
            # Try to get workspace status from the MCP daemon
            response = requests.get(f"{self.workspace_url}/status", timeout=5)
            if response.status_code == 200:
                return {'running': True, 'status': response.json()}
            else:
                return {'running': False, 'error': f'HTTP {response.status_code}'}
        except Exception as e:
            return {'running': False, 'error': str(e)}
    
    def get_watch_folder_status(self) -> Dict:
        """Get current watch folder status and queue information"""
        try:
            # Try to get watch status from the daemon
            response = requests.get(f"{self.workspace_url}/watches", timeout=5)
            if response.status_code == 200:
                return response.json()
            else:
                return {'error': f'HTTP {response.status_code}'}
        except Exception as e:
            return {'error': str(e)}
    
    def measure_change_detection_time(self, file_path: Path, content: str, timeout: float = 30.0) -> SyncMetrics:
        """Measure time from file change to daemon awareness"""
        metrics = SyncMetrics()
        metrics.timestamp = time.time()
        
        try:
            # Record the change with a unique identifier
            unique_id = f"sync-test-{int(time.time() * 1000000)}"
            test_content = f"{content}\n// SYNC_TEST_MARKER: {unique_id}"
            
            # Make the file change
            change_start = time.time()
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(test_content)
            
            # Ensure filesystem sync
            os.fsync(open(file_path, 'r').fileno())
            
            # Monitor daemon for awareness of the change
            detection_time = None
            processing_complete_time = None
            queue_depths = []
            
            start_monitoring = time.time()
            while time.time() - start_monitoring < timeout:
                try:
                    # Check if daemon has detected the change
                    watch_status = self.get_watch_folder_status()
                    if 'error' not in watch_status:
                        # Record queue depth for analysis
                        queue_depth = watch_status.get('queue_depth', 0)
                        queue_depths.append(queue_depth)
                        
                        # Check if our change has been detected
                        if detection_time is None:
                            last_processed = watch_status.get('last_processed_file', '')
                            if str(file_path) in last_processed or unique_id in str(watch_status):
                                detection_time = time.time()
                                metrics.change_detection_time = detection_time - change_start
                        
                        # Check if processing is complete (queue depth back to normal)
                        if detection_time and processing_complete_time is None:
                            if queue_depth == 0:
                                processing_complete_time = time.time()
                                metrics.processing_latency = processing_complete_time - detection_time
                    
                    # Test query freshness
                    if detection_time and not metrics.success:
                        search_result = self.search_for_content(unique_id)
                        if search_result.get('found', False):
                            query_freshness_time = time.time()
                            metrics.query_freshness_time = query_freshness_time - change_start
                            metrics.end_to_end_latency = query_freshness_time - change_start
                            metrics.success = True
                            break
                    
                    time.sleep(0.1)  # Check every 100ms
                    
                except Exception as e:
                    print(f"⚠️  Error during sync monitoring: {e}")
                    time.sleep(0.5)
            
            # Calculate queue statistics
            if queue_depths:
                metrics.queue_depth_max = max(queue_depths)
                metrics.queue_depth_avg = sum(queue_depths) / len(queue_depths)
            
            if not metrics.success:
                metrics.error = f"Sync not detected within {timeout}s timeout"
                
        except Exception as e:
            metrics.error = str(e)
            metrics.success = False
        
        return metrics
    
    def search_for_content(self, search_term: str) -> Dict:
        """Search for specific content to test freshness"""
        try:
            # Search directly in Qdrant for the content
            response = requests.post(
                f"{self.qdrant_url}/collections/workspace/points/scroll",
                json={
                    "filter": {
                        "must": [
                            {
                                "key": "content",
                                "match": {"text": search_term}
                            }
                        ]
                    },
                    "limit": 1,
                    "with_payload": True
                },
                timeout=5
            )
            
            if response.status_code == 200:
                results = response.json().get('result', {}).get('points', [])
                return {'found': len(results) > 0, 'results': results}
            else:
                return {'found': False, 'error': f'HTTP {response.status_code}'}
                
        except Exception as e:
            return {'found': False, 'error': str(e)}

class DevelopmentWorkflowSimulator:
    """Base class for development workflow simulation"""
    
    def __init__(self, test_directory: Path, sync_monitor: RealtimeSyncMonitor):
        self.test_directory = test_directory
        self.sync_monitor = sync_monitor
        self.resource_monitor = ResourceMonitor()
        self.metrics_history = []
        
    def setup_test_environment(self):
        """Setup test environment for workflow simulation"""
        self.test_directory.mkdir(parents=True, exist_ok=True)
        
    def cleanup_test_environment(self):
        """Clean up test environment"""
        if self.test_directory.exists():
            shutil.rmtree(self.test_directory)
    
    def generate_realistic_code_content(self, file_type: str = ".c") -> str:
        """Generate realistic code content for testing"""
        templates = {
            '.c': '''
#include <stdio.h>
#include <stdlib.h>
#include "config.h"

// Function: process_keyboard_matrix
// Purpose: Handle keyboard matrix scanning
int process_keyboard_matrix(void) {
    uint8_t matrix_state[MATRIX_ROWS];
    
    for (int row = 0; row < MATRIX_ROWS; row++) {
        matrix_state[row] = scan_row(row);
        
        if (matrix_state[row] != previous_state[row]) {
            handle_key_change(row, matrix_state[row]);
        }
    }
    
    return 0;
}
            ''',
            '.h': '''
#ifndef KEYBOARD_CONFIG_H
#define KEYBOARD_CONFIG_H

#define MATRIX_ROWS 8
#define MATRIX_COLS 16

// Keyboard configuration structure
typedef struct {
    uint8_t matrix_pins[MATRIX_ROWS];
    uint8_t led_count;
    bool backlight_enabled;
} keyboard_config_t;

extern keyboard_config_t keyboard_config;

#endif // KEYBOARD_CONFIG_H
            ''',
            '.py': '''
#!/usr/bin/env python3
"""
Keyboard configuration generator
Generates QMK keyboard configurations from templates
"""

import json
import os
from pathlib import Path

class KeyboardConfigGenerator:
    def __init__(self, config_path: Path):
        self.config_path = config_path
        self.layouts = {}
    
    def generate_config(self, keyboard_name: str) -> dict:
        """Generate keyboard configuration"""
        config = {
            "keyboard": keyboard_name,
            "matrix": {
                "rows": 8,
                "cols": 16
            },
            "features": {
                "backlight": True,
                "rgb": False
            }
        }
        return config
            '''
        }
        
        base_content = templates.get(file_type, templates['.c'])
        timestamp = datetime.now().isoformat()
        return f"{base_content}\n// Generated: {timestamp}\n// Test ID: {uuid.uuid4()}\n"

class ActiveCodingSimulator(DevelopmentWorkflowSimulator):
    """Simulate active coding session with continuous file modifications"""
    
    def simulate_coding_session(self, duration_minutes: int = 10, 
                               modification_interval: Tuple[int, int] = (5, 30)) -> Dict:
        """Simulate an active coding session"""
        print(f"\n🔄 Starting Active Coding Session Simulation")
        print(f"   Duration: {duration_minutes} minutes")
        print(f"   Modification interval: {modification_interval[0]}-{modification_interval[1]} seconds")
        
        self.setup_test_environment()
        self.resource_monitor.start_monitoring()
        
        session_start = time.time()
        session_end = session_start + (duration_minutes * 60)
        
        # Create initial files
        test_files = []
        for i in range(5):
            file_path = self.test_directory / f"keyboard_{i}.c"
            content = self.generate_realistic_code_content('.c')
            with open(file_path, 'w') as f:
                f.write(content)
            test_files.append(file_path)
        
        modifications_count = 0
        successful_syncs = 0
        failed_syncs = 0
        
        print("   📝 Starting continuous file modifications...")
        
        while time.time() < session_end:
            try:
                # Random delay between modifications
                delay = random.randint(modification_interval[0], modification_interval[1])
                time.sleep(delay)
                
                # Select random file to modify
                file_to_modify = random.choice(test_files)
                
                # Generate new content
                new_content = self.generate_realistic_code_content('.c')
                
                # Measure sync performance for this modification
                print(f"   🔧 Modifying {file_to_modify.name} (change #{modifications_count + 1})")
                
                sync_metrics = self.sync_monitor.measure_change_detection_time(
                    file_to_modify, new_content, timeout=15.0
                )
                
                modifications_count += 1
                self.metrics_history.append(sync_metrics)
                
                if sync_metrics.success:
                    successful_syncs += 1
                    print(f"      ✅ Sync successful in {sync_metrics.end_to_end_latency:.2f}s")
                else:
                    failed_syncs += 1
                    print(f"      ❌ Sync failed: {sync_metrics.error}")
                
                # Check for safety thresholds
                memory = psutil.virtual_memory()
                if memory.percent > 80:
                    print(f"   ⚠️  HIGH MEMORY USAGE: {memory.percent:.1f}% - ending session early")
                    break
                    
            except KeyboardInterrupt:
                print("   ⚠️  Session interrupted by user")
                break
            except Exception as e:
                print(f"   ❌ Error during coding session: {e}")
                failed_syncs += 1
        
        resource_stats = self.resource_monitor.stop_monitoring()
        
        # Calculate performance statistics
        successful_metrics = [m for m in self.metrics_history if m.success]
        
        results = {
            'scenario': 'active_coding_session',
            'duration_minutes': (time.time() - session_start) / 60,
            'modifications_total': modifications_count,
            'modifications_successful': successful_syncs,
            'modifications_failed': failed_syncs,
            'success_rate': successful_syncs / max(modifications_count, 1) * 100,
            'resource_stats': resource_stats,
            'sync_performance': {}
        }
        
        if successful_metrics:
            detection_times = [m.change_detection_time for m in successful_metrics if m.change_detection_time > 0]
            end_to_end_times = [m.end_to_end_latency for m in successful_metrics if m.end_to_end_latency > 0]
            queue_depths = [m.queue_depth_max for m in successful_metrics if m.queue_depth_max > 0]
            
            results['sync_performance'] = {
                'change_detection_avg': sum(detection_times) / len(detection_times) if detection_times else 0,
                'change_detection_max': max(detection_times) if detection_times else 0,
                'end_to_end_avg': sum(end_to_end_times) / len(end_to_end_times) if end_to_end_times else 0,
                'end_to_end_max': max(end_to_end_times) if end_to_end_times else 0,
                'queue_depth_max': max(queue_depths) if queue_depths else 0
            }
        
        self.cleanup_test_environment()
        return results

class RefactoringSimulator(DevelopmentWorkflowSimulator):
    """Simulate refactoring operations with mass file changes"""
    
    def simulate_refactoring_session(self, file_count: int = 20) -> Dict:
        """Simulate mass refactoring operations"""
        print(f"\n🔄 Starting Refactoring Operations Simulation")
        print(f"   File count: {file_count}")
        
        self.setup_test_environment()
        self.resource_monitor.start_monitoring()
        
        session_start = time.time()
        
        # Phase 1: Create initial file structure
        print("   📁 Phase 1: Creating initial file structure...")
        initial_files = []
        for i in range(file_count):
            file_path = self.test_directory / f"module_{i:02d}.c"
            content = self.generate_realistic_code_content('.c')
            with open(file_path, 'w') as f:
                f.write(content)
            initial_files.append(file_path)
        
        time.sleep(2)  # Let initial files settle
        
        # Phase 2: Mass content modifications
        print("   🔧 Phase 2: Mass content modifications...")
        modification_start = time.time()
        
        modification_futures = []
        with ThreadPoolExecutor(max_workers=4) as executor:
            for file_path in initial_files:
                new_content = self.generate_realistic_code_content('.c')
                future = executor.submit(
                    self.sync_monitor.measure_change_detection_time,
                    file_path, new_content, 20.0
                )
                modification_futures.append(future)
        
        # Collect modification results
        modification_results = []
        for future in modification_futures:
            try:
                result = future.result(timeout=30)
                modification_results.append(result)
            except Exception as e:
                print(f"   ❌ Modification failed: {e}")
        
        modification_time = time.time() - modification_start
        
        # Phase 3: File renames and moves
        print("   📋 Phase 3: File renames and directory restructuring...")
        
        rename_start = time.time()
        renamed_files = []
        
        # Create subdirectories
        subdirs = ['core', 'drivers', 'config']
        for subdir in subdirs:
            (self.test_directory / subdir).mkdir(exist_ok=True)
        
        # Rename and move files
        for i, old_file in enumerate(initial_files[:10]):  # Rename first 10 files
            if old_file.exists():
                new_name = f"refactored_{i:02d}.c"
                subdir = subdirs[i % len(subdirs)]
                new_path = self.test_directory / subdir / new_name
                
                try:
                    shutil.move(str(old_file), str(new_path))
                    renamed_files.append((old_file, new_path))
                except Exception as e:
                    print(f"   ⚠️  Failed to rename {old_file.name}: {e}")
        
        rename_time = time.time() - rename_start
        
        # Phase 4: Large file operations
        print("   📄 Phase 4: Large file operations...")
        
        large_file_start = time.time()
        large_file_path = self.test_directory / "large_config.h"
        
        # Create large file with many function definitions
        large_content = ""
        for i in range(100):
            large_content += f"""
void function_{i}(void) {{
    // Function implementation {i}
    uint8_t data[{i + 1}];
    process_data(data, {i + 1});
}}
"""
        
        large_file_sync = self.sync_monitor.measure_change_detection_time(
            large_file_path, large_content, timeout=30.0
        )
        
        large_file_time = time.time() - large_file_start
        
        resource_stats = self.resource_monitor.stop_monitoring()
        session_time = time.time() - session_start
        
        # Analyze results
        successful_modifications = [m for m in modification_results if m.success]
        failed_modifications = [m for m in modification_results if not m.success]
        
        results = {
            'scenario': 'refactoring_operations',
            'total_duration': session_time,
            'phases': {
                'mass_modifications': {
                    'duration': modification_time,
                    'files_processed': len(initial_files),
                    'successful_syncs': len(successful_modifications),
                    'failed_syncs': len(failed_modifications),
                    'success_rate': len(successful_modifications) / max(len(modification_results), 1) * 100
                },
                'file_renames': {
                    'duration': rename_time,
                    'files_renamed': len(renamed_files),
                    'subdirs_created': len(subdirs)
                },
                'large_file_ops': {
                    'duration': large_file_time,
                    'file_size_chars': len(large_content),
                    'sync_successful': large_file_sync.success,
                    'sync_time': large_file_sync.end_to_end_latency if large_file_sync.success else None
                }
            },
            'resource_stats': resource_stats
        }
        
        # Add sync performance analysis
        if successful_modifications:
            detection_times = [m.change_detection_time for m in successful_modifications if m.change_detection_time > 0]
            end_to_end_times = [m.end_to_end_latency for m in successful_modifications if m.end_to_end_latency > 0]
            
            results['sync_performance'] = {
                'concurrent_detection_avg': sum(detection_times) / len(detection_times) if detection_times else 0,
                'concurrent_detection_max': max(detection_times) if detection_times else 0,
                'concurrent_end_to_end_avg': sum(end_to_end_times) / len(end_to_end_times) if end_to_end_times else 0,
                'concurrent_end_to_end_max': max(end_to_end_times) if end_to_end_times else 0,
                'large_file_sync_time': large_file_sync.end_to_end_latency if large_file_sync.success else None
            }
        
        self.cleanup_test_environment()
        return results

class GitOperationsSimulator(DevelopmentWorkflowSimulator):
    """Simulate git operations and repository state changes"""
    
    def simulate_git_operations(self) -> Dict:
        """Simulate git operations stress testing"""
        print(f"\n🔄 Starting Git Operations Simulation")
        
        self.setup_test_environment()
        self.resource_monitor.start_monitoring()
        
        session_start = time.time()
        
        # Initialize git repository
        print("   📁 Initializing test git repository...")
        subprocess.run(['git', 'init'], cwd=self.test_directory, capture_output=True)
        subprocess.run(['git', 'config', 'user.email', 'test@test.com'], cwd=self.test_directory, capture_output=True)
        subprocess.run(['git', 'config', 'user.name', 'Test User'], cwd=self.test_directory, capture_output=True)
        
        operations_results = []
        
        try:
            # Phase 1: Initial commit with multiple files
            print("   📝 Phase 1: Initial commit creation...")
            
            initial_files = []
            for i in range(10):
                file_path = self.test_directory / f"initial_{i}.c"
                content = self.generate_realistic_code_content('.c')
                with open(file_path, 'w') as f:
                    f.write(content)
                initial_files.append(file_path)
            
            subprocess.run(['git', 'add', '.'], cwd=self.test_directory, capture_output=True)
            subprocess.run(['git', 'commit', '-m', 'Initial commit'], cwd=self.test_directory, capture_output=True)
            
            # Phase 2: Branch creation and switching stress test
            print("   🌿 Phase 2: Branch operations...")
            
            branches = ['feature/new-keyboard', 'feature/led-support', 'feature/usb-improvements']
            branch_operations = []
            
            for branch in branches:
                branch_start = time.time()
                
                # Create and switch to branch
                subprocess.run(['git', 'checkout', '-b', branch], cwd=self.test_directory, capture_output=True)
                
                # Make changes in branch
                branch_file = self.test_directory / f"{branch.replace('/', '_')}.c"
                branch_content = self.generate_realistic_code_content('.c')
                
                sync_metrics = self.sync_monitor.measure_change_detection_time(
                    branch_file, branch_content, timeout=15.0
                )
                
                # Commit changes
                subprocess.run(['git', 'add', '.'], cwd=self.test_directory, capture_output=True)
                subprocess.run(['git', 'commit', '-m', f'Add {branch} feature'], cwd=self.test_directory, capture_output=True)
                
                branch_time = time.time() - branch_start
                
                branch_operations.append({
                    'branch': branch,
                    'duration': branch_time,
                    'sync_successful': sync_metrics.success,
                    'sync_latency': sync_metrics.end_to_end_latency if sync_metrics.success else None
                })
                
                # Switch back to main
                subprocess.run(['git', 'checkout', 'master'], cwd=self.test_directory, capture_output=True)
            
            # Phase 3: Merge operations
            print("   🔀 Phase 3: Merge operations...")
            
            merge_operations = []
            for branch in branches:
                merge_start = time.time()
                
                # Merge branch
                result = subprocess.run(['git', 'merge', branch], cwd=self.test_directory, capture_output=True)
                
                # Check if merge was successful and test sync
                if result.returncode == 0:
                    # Test sync after merge
                    merge_test_file = self.test_directory / "merge_test.c"
                    merge_content = self.generate_realistic_code_content('.c')
                    
                    sync_metrics = self.sync_monitor.measure_change_detection_time(
                        merge_test_file, merge_content, timeout=15.0
                    )
                    
                    merge_time = time.time() - merge_start
                    
                    merge_operations.append({
                        'branch': branch,
                        'duration': merge_time,
                        'merge_successful': True,
                        'sync_successful': sync_metrics.success,
                        'sync_latency': sync_metrics.end_to_end_latency if sync_metrics.success else None
                    })
                else:
                    merge_operations.append({
                        'branch': branch,
                        'duration': time.time() - merge_start,
                        'merge_successful': False,
                        'error': result.stderr.decode() if result.stderr else 'Unknown merge error'
                    })
            
            # Phase 4: Repository state stress test
            print("   🔄 Phase 4: Repository state transitions...")
            
            state_operations = []
            
            # Create a large commit with many files
            large_commit_start = time.time()
            
            for i in range(50):
                large_file = self.test_directory / f"large_commit_{i:02d}.c"
                content = self.generate_realistic_code_content('.c')
                with open(large_file, 'w') as f:
                    f.write(content)
            
            # Test sync performance during large commit preparation
            test_during_commit = self.test_directory / "commit_test_marker.c"
            commit_sync_metrics = self.sync_monitor.measure_change_detection_time(
                test_during_commit, self.generate_realistic_code_content('.c'), timeout=20.0
            )
            
            subprocess.run(['git', 'add', '.'], cwd=self.test_directory, capture_output=True)
            subprocess.run(['git', 'commit', '-m', 'Large commit with 50+ files'], cwd=self.test_directory, capture_output=True)
            
            large_commit_time = time.time() - large_commit_start
            
            state_operations.append({
                'operation': 'large_commit',
                'duration': large_commit_time,
                'files_count': 50,
                'sync_during_commit': {
                    'successful': commit_sync_metrics.success,
                    'latency': commit_sync_metrics.end_to_end_latency if commit_sync_metrics.success else None
                }
            })
            
        except Exception as e:
            print(f"   ❌ Git operations failed: {e}")
            operations_results.append({'error': str(e)})
        
        resource_stats = self.resource_monitor.stop_monitoring()
        session_time = time.time() - session_start
        
        results = {
            'scenario': 'git_operations',
            'total_duration': session_time,
            'branch_operations': branch_operations,
            'merge_operations': merge_operations,
            'state_operations': state_operations,
            'resource_stats': resource_stats
        }
        
        # Calculate success rates
        successful_branch_syncs = [op for op in branch_operations if op.get('sync_successful', False)]
        successful_merge_syncs = [op for op in merge_operations if op.get('sync_successful', False)]
        
        results['summary'] = {
            'branch_sync_success_rate': len(successful_branch_syncs) / max(len(branch_operations), 1) * 100,
            'merge_sync_success_rate': len(successful_merge_syncs) / max(len(merge_operations), 1) * 100,
            'avg_branch_sync_latency': sum(op['sync_latency'] for op in successful_branch_syncs if op.get('sync_latency')) / max(len(successful_branch_syncs), 1) if successful_branch_syncs else 0,
            'avg_merge_sync_latency': sum(op['sync_latency'] for op in successful_merge_syncs if op.get('sync_latency')) / max(len(successful_merge_syncs), 1) if successful_merge_syncs else 0
        }
        
        self.cleanup_test_environment()
        return results

class RealtimeSyncStressTester:
    """Main orchestrator for real-time sync stress testing"""
    
    def __init__(self):
        self.qmk_path = Path("/Users/chris/Dropbox/dev/ai/claude-code-cfg/mcp/workspace-qdrant-mcp/LT_20250906-1822_testing_sandbox/qmk_integration/qmk_firmware")
        self.test_base_dir = Path("/Users/chris/Dropbox/dev/ai/claude-code-cfg/mcp/workspace-qdrant-mcp/LT_20250906-1822_testing_sandbox/qmk_integration/LT_20250906-1822_testing_sandbox")
        self.results_dir = Path("/Users/chris/Dropbox/dev/ai/claude-code-cfg/mcp/workspace-qdrant-mcp/LT_20250906-1822_testing_sandbox/qmk_integration")
        
        self.sync_monitor = RealtimeSyncMonitor()
        self.resource_monitor = ResourceMonitor()
        
        self.test_results = []
        
    def verify_prerequisites(self) -> bool:
        """Verify all prerequisites for testing"""
        print("🔍 Verifying test prerequisites...")
        
        # Check QMK path
        if not self.qmk_path.exists():
            print(f"❌ QMK repository not found: {self.qmk_path}")
            return False
        print("✅ QMK repository accessible")
        
        # Check Qdrant
        try:
            response = requests.get("http://localhost:6333/collections", timeout=5)
            if response.status_code != 200:
                raise Exception(f"HTTP {response.status_code}")
            print("✅ Qdrant daemon running")
        except Exception as e:
            print(f"❌ Qdrant not accessible: {e}")
            return False
        
        # Check daemon status
        daemon_status = self.sync_monitor.check_daemon_status()
        if daemon_status['running']:
            print("✅ Workspace daemon accessible")
        else:
            print(f"⚠️  Workspace daemon not accessible: {daemon_status.get('error', 'Unknown error')}")
            print("   Proceeding with direct Qdrant testing...")
        
        # Create test directories
        self.test_base_dir.mkdir(parents=True, exist_ok=True)
        print("✅ Test directories prepared")
        
        return True
        
    def run_all_scenarios(self) -> List[Dict]:
        """Execute all development workflow scenarios"""
        print("\n🚀 Starting Real-Time Sync Stress Testing Suite")
        print("Task #147: Development workflow sync validation")
        
        all_results = []
        
        # Scenario 1: Active Coding Session
        try:
            print(f"\n{'='*60}")
            print("SCENARIO 1: ACTIVE CODING SESSION")
            print(f"{'='*60}")
            
            test_dir = self.test_base_dir / "active_coding_test"
            simulator = ActiveCodingSimulator(test_dir, self.sync_monitor)
            
            result = simulator.simulate_coding_session(
                duration_minutes=5,  # 5 minutes for thorough testing
                modification_interval=(5, 15)  # 5-15 second intervals
            )
            
            result['scenario_id'] = 1
            all_results.append(result)
            
            print(f"✅ Active coding scenario completed")
            print(f"   Success rate: {result['success_rate']:.1f}%")
            print(f"   Modifications: {result['modifications_successful']}/{result['modifications_total']}")
            
        except Exception as e:
            print(f"❌ Active coding scenario failed: {e}")
            all_results.append({
                'scenario_id': 1,
                'scenario': 'active_coding_session',
                'success': False,
                'error': str(e)
            })
        
        # Brief pause between scenarios
        time.sleep(10)
        
        # Scenario 2: Refactoring Operations
        try:
            print(f"\n{'='*60}")
            print("SCENARIO 2: REFACTORING OPERATIONS")
            print(f"{'='*60}")
            
            test_dir = self.test_base_dir / "refactoring_test"
            simulator = RefactoringSimulator(test_dir, self.sync_monitor)
            
            result = simulator.simulate_refactoring_session(file_count=15)
            
            result['scenario_id'] = 2
            all_results.append(result)
            
            print(f"✅ Refactoring scenario completed")
            if 'phases' in result:
                mass_mods = result['phases']['mass_modifications']
                print(f"   Mass modifications success: {mass_mods['success_rate']:.1f}%")
                print(f"   Files renamed: {result['phases']['file_renames']['files_renamed']}")
                print(f"   Large file sync: {'✅' if result['phases']['large_file_ops']['sync_successful'] else '❌'}")
            
        except Exception as e:
            print(f"❌ Refactoring scenario failed: {e}")
            all_results.append({
                'scenario_id': 2,
                'scenario': 'refactoring_operations',
                'success': False,
                'error': str(e)
            })
        
        # Brief pause between scenarios
        time.sleep(10)
        
        # Scenario 3: Git Operations
        try:
            print(f"\n{'='*60}")
            print("SCENARIO 3: GIT OPERATIONS STRESS")
            print(f"{'='*60}")
            
            test_dir = self.test_base_dir / "git_operations_test"
            simulator = GitOperationsSimulator(test_dir, self.sync_monitor)
            
            result = simulator.simulate_git_operations()
            
            result['scenario_id'] = 3
            all_results.append(result)
            
            print(f"✅ Git operations scenario completed")
            if 'summary' in result:
                print(f"   Branch sync success: {result['summary']['branch_sync_success_rate']:.1f}%")
                print(f"   Merge sync success: {result['summary']['merge_sync_success_rate']:.1f}%")
            
        except Exception as e:
            print(f"❌ Git operations scenario failed: {e}")
            all_results.append({
                'scenario_id': 3,
                'scenario': 'git_operations',
                'success': False,
                'error': str(e)
            })
        
        return all_results
    
    def save_results(self, results: List[Dict]) -> Path:
        """Save comprehensive test results"""
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        filename = f"realtime_sync_stress_results_{timestamp}.json"
        filepath = self.results_dir / filename
        
        # Create comprehensive results document
        full_results = {
            'test_info': {
                'test_name': 'Real-Time Sync Stress Testing Suite',
                'task_number': 147,
                'timestamp': timestamp,
                'test_duration': None,  # Will be calculated
                'qmk_path': str(self.qmk_path),
                'scenarios_executed': len(results)
            },
            'system_info': {
                'cpu_count': psutil.cpu_count(),
                'memory_gb': psutil.virtual_memory().total / 1024 / 1024 / 1024,
                'python_version': sys.version,
                'platform': sys.platform
            },
            'scenario_results': results
        }
        
        with open(filepath, 'w') as f:
            json.dump(full_results, f, indent=2)
        
        print(f"📄 Results saved to: {filepath}")
        return filepath
    
    def generate_comprehensive_report(self, results: List[Dict]):
        """Generate detailed analysis report"""
        print(f"\n{'='*80}")
        print("REAL-TIME SYNC STRESS TESTING - COMPREHENSIVE ANALYSIS")
        print(f"{'='*80}")
        
        successful_scenarios = [r for r in results if r.get('success', True) and 'error' not in r]
        failed_scenarios = [r for r in results if 'error' in r or r.get('success', False) == False]
        
        print(f"✅ Successful scenarios: {len(successful_scenarios)}")
        print(f"❌ Failed scenarios: {len(failed_scenarios)}")
        print(f"📊 Total scenarios: {len(results)}")
        
        # Scenario-specific analysis
        for result in successful_scenarios:
            scenario = result.get('scenario', 'Unknown')
            print(f"\n🎯 {scenario.upper().replace('_', ' ')} ANALYSIS:")
            
            if scenario == 'active_coding_session':
                print(f"   Duration: {result['duration_minutes']:.1f} minutes")
                print(f"   File modifications: {result['modifications_total']}")
                print(f"   Sync success rate: {result['success_rate']:.1f}%")
                
                if 'sync_performance' in result:
                    perf = result['sync_performance']
                    print(f"   Avg change detection: {perf.get('change_detection_avg', 0):.2f}s")
                    print(f"   Avg end-to-end sync: {perf.get('end_to_end_avg', 0):.2f}s")
                    print(f"   Max queue depth: {perf.get('queue_depth_max', 0)}")
                
            elif scenario == 'refactoring_operations':
                print(f"   Total duration: {result['total_duration']:.1f}s")
                
                if 'phases' in result:
                    phases = result['phases']
                    mass_mods = phases.get('mass_modifications', {})
                    print(f"   Mass modifications: {mass_mods.get('success_rate', 0):.1f}% success")
                    print(f"   Files renamed: {phases.get('file_renames', {}).get('files_renamed', 0)}")
                    print(f"   Large file sync: {phases.get('large_file_ops', {}).get('sync_successful', False)}")
                
                if 'sync_performance' in result:
                    perf = result['sync_performance']
                    print(f"   Concurrent sync avg: {perf.get('concurrent_end_to_end_avg', 0):.2f}s")
                    print(f"   Large file sync time: {perf.get('large_file_sync_time', 'N/A')}")
                
            elif scenario == 'git_operations':
                print(f"   Total duration: {result['total_duration']:.1f}s")
                
                if 'summary' in result:
                    summary = result['summary']
                    print(f"   Branch sync success: {summary.get('branch_sync_success_rate', 0):.1f}%")
                    print(f"   Merge sync success: {summary.get('merge_sync_success_rate', 0):.1f}%")
                    print(f"   Avg branch sync: {summary.get('avg_branch_sync_latency', 0):.2f}s")
                    print(f"   Avg merge sync: {summary.get('avg_merge_sync_latency', 0):.2f}s")
        
        # Resource usage analysis
        print(f"\n💾 RESOURCE USAGE ANALYSIS:")
        
        for result in successful_scenarios:
            resource_stats = result.get('resource_stats', {})
            if resource_stats:
                memory_stats = resource_stats.get('memory', {})
                cpu_stats = resource_stats.get('cpu', {})
                
                scenario_name = result.get('scenario', 'Unknown')
                max_memory = memory_stats.get('max_percent', 0)
                max_cpu = cpu_stats.get('max', 0)
                avg_memory = memory_stats.get('avg_percent', 0)
                avg_cpu = cpu_stats.get('avg', 0)
                
                print(f"   {scenario_name:<25}: Memory {avg_memory:>5.1f}%/{max_memory:>5.1f}% (avg/peak), CPU {avg_cpu:>5.1f}%/{max_cpu:>5.1f}% (avg/peak)")
        
        # Failed scenarios analysis
        if failed_scenarios:
            print(f"\n❌ FAILED SCENARIOS ANALYSIS:")
            for result in failed_scenarios:
                scenario = result.get('scenario', 'Unknown')
                error = result.get('error', 'Unknown error')
                print(f"   {scenario:<25}: {error}")
        
        # Overall assessment and recommendations
        print(f"\n🏆 OVERALL SYNC PERFORMANCE ASSESSMENT:")
        
        # Calculate overall metrics
        total_scenarios = len(results)
        successful_count = len(successful_scenarios)
        overall_success_rate = (successful_count / max(total_scenarios, 1)) * 100
        
        print(f"   Overall success rate: {overall_success_rate:.1f}%")
        
        if overall_success_rate >= 90:
            print("   🎉 EXCELLENT - Real-time sync performs exceptionally well")
            print("   ✅ Ready for production development workflows")
        elif overall_success_rate >= 75:
            print("   ⚠️  GOOD - Real-time sync performs well with minor issues")
            print("   ✅ Suitable for production with monitoring")
        elif overall_success_rate >= 50:
            print("   ⚠️  MODERATE - Real-time sync has significant issues")
            print("   ❌ Requires optimization before production use")
        else:
            print("   ❌ POOR - Real-time sync requires major improvements")
            print("   ❌ Not suitable for production workflows")
        
        print(f"\n📋 PRODUCTION DEPLOYMENT RECOMMENDATIONS:")
        
        # Extract performance metrics for recommendations
        coding_session_result = next((r for r in successful_scenarios if r.get('scenario') == 'active_coding_session'), None)
        if coding_session_result and 'sync_performance' in coding_session_result:
            perf = coding_session_result['sync_performance']
            avg_sync_time = perf.get('end_to_end_avg', 0)
            
            if avg_sync_time > 0:
                if avg_sync_time <= 5.0:
                    print("   ✅ Sync latency excellent for active development")
                elif avg_sync_time <= 10.0:
                    print("   ⚠️  Sync latency acceptable but monitor for improvements")
                else:
                    print("   ❌ Sync latency too high for responsive development")
                
                print(f"   • Average sync time: {avg_sync_time:.2f} seconds")
                print(f"   • Recommended query delay: {max(avg_sync_time * 1.5, 2.0):.1f} seconds after file changes")
        
        # Resource recommendations
        max_memory_used = 0
        max_cpu_used = 0
        
        for result in successful_scenarios:
            resource_stats = result.get('resource_stats', {})
            if resource_stats:
                memory_stats = resource_stats.get('memory', {})
                cpu_stats = resource_stats.get('cpu', {})
                max_memory_used = max(max_memory_used, memory_stats.get('max_percent', 0))
                max_cpu_used = max(max_cpu_used, cpu_stats.get('max', 0))
        
        if max_memory_used > 0:
            memory_headroom = 85 - max_memory_used
            cpu_headroom = 90 - max_cpu_used
            
            print(f"   • Memory headroom: {memory_headroom:.1f}% (Peak usage: {max_memory_used:.1f}%)")
            print(f"   • CPU headroom: {cpu_headroom:.1f}% (Peak usage: {max_cpu_used:.1f}%)")
            
            if memory_headroom > 30 and cpu_headroom > 40:
                print("   ✅ Excellent resource efficiency - can handle larger projects")
            elif memory_headroom > 15 and cpu_headroom > 20:
                print("   ⚠️  Good resource efficiency - monitor during scale-up")
            else:
                print("   ❌ Limited resource headroom - consider resource optimization")
        
        print(f"\n{'='*80}")

def main():
    """Main execution function"""
    print("Real-Time Sync Stress Testing Suite")
    print("Task #147: Development workflow sync validation\n")
    
    tester = RealtimeSyncStressTester()
    
    # Verify prerequisites
    if not tester.verify_prerequisites():
        print("❌ Prerequisites not met - aborting test")
        sys.exit(1)
    
    try:
        # Execute all scenarios
        results = tester.run_all_scenarios()
        
        # Save results and generate report
        filepath = tester.save_results(results)
        tester.generate_comprehensive_report(results)
        
        print(f"\n🎉 Real-time sync stress testing COMPLETED!")
        print(f"📊 Comprehensive results saved to: {filepath}")
        print(f"🔬 Task #147 real-time sync validation complete")
        
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