#!/usr/bin/env python3
"""
Resource Monitor - Comprehensive system resource tracking
Monitors memory, CPU, disk I/O, and network activity during testing
"""

import psutil
import time
import json
import logging
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional
import subprocess

class ResourceMonitor:
    def __init__(self, monitoring_interval: int = 10):
        self.monitoring_interval = monitoring_interval
        self.running = False
        self.monitor_thread = None
        self.metrics_history = []
        self.max_history_size = 1000  # Keep last 1000 measurements
        
        # Setup logging
        self.setup_logging()
        
        # Initial baseline
        self.baseline_metrics = None
        
        # Monitoring targets
        self.process_whitelist = []  # Specific processes to monitor
        self.mcp_processes = []  # Track MCP server processes
        
    def setup_logging(self):
        """Setup resource monitoring logging"""
        log_dir = Path(__file__).parent.parent / "monitoring_logs"
        log_dir.mkdir(exist_ok=True)
        
        log_file = log_dir / f"resource_monitor_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def capture_baseline(self) -> Dict[str, Any]:
        """Capture baseline system metrics before testing"""
        self.logger.info("Capturing baseline system metrics")
        
        baseline = {}
        
        try:
            # System-wide metrics
            memory = psutil.virtual_memory()
            cpu_percent = psutil.cpu_percent(interval=2)  # 2-second sample
            disk = psutil.disk_usage('/')
            network = psutil.net_io_counters()
            
            # Disk I/O metrics
            disk_io = psutil.disk_io_counters()
            
            baseline = {
                "timestamp": datetime.now().isoformat(),
                "memory": {
                    "total_gb": memory.total / (1024**3),
                    "available_gb": memory.available / (1024**3),
                    "percent_used": memory.percent,
                    "free_gb": memory.free / (1024**3),
                    "cached_gb": memory.cached / (1024**3) if hasattr(memory, 'cached') else 0
                },
                "cpu": {
                    "percent_used": cpu_percent,
                    "core_count": psutil.cpu_count(),
                    "core_count_logical": psutil.cpu_count(logical=True),
                    "frequency_mhz": psutil.cpu_freq().current if psutil.cpu_freq() else None,
                    "load_average": list(psutil.getloadavg()) if hasattr(psutil, 'getloadavg') else None
                },
                "disk": {
                    "total_gb": disk.total / (1024**3),
                    "free_gb": disk.free / (1024**3),
                    "used_gb": disk.used / (1024**3),
                    "percent_used": (disk.used / disk.total) * 100
                },
                "disk_io": {
                    "read_count": disk_io.read_count if disk_io else 0,
                    "write_count": disk_io.write_count if disk_io else 0,
                    "read_bytes": disk_io.read_bytes if disk_io else 0,
                    "write_bytes": disk_io.write_bytes if disk_io else 0,
                    "read_time": disk_io.read_time if disk_io else 0,
                    "write_time": disk_io.write_time if disk_io else 0
                },
                "network": {
                    "bytes_sent": network.bytes_sent,
                    "bytes_recv": network.bytes_recv,
                    "packets_sent": network.packets_sent,
                    "packets_recv": network.packets_recv,
                    "errors_in": network.errin,
                    "errors_out": network.errout,
                    "drops_in": network.dropin,
                    "drops_out": network.dropout
                },
                "processes": {
                    "total_count": len(psutil.pids()),
                    "python_processes": self.count_python_processes()
                }
            }
            
            self.baseline_metrics = baseline
            
            # Save baseline to file
            baseline_dir = Path(__file__).parent.parent / "baseline_metrics"
            baseline_file = baseline_dir / f"baseline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            with open(baseline_file, 'w') as f:
                json.dump(baseline, f, indent=2)
            
            self.logger.info(f"Baseline metrics captured and saved to {baseline_file}")
            
        except Exception as e:
            self.logger.error(f"Error capturing baseline metrics: {e}")
        
        return baseline
    
    def count_python_processes(self) -> int:
        """Count currently running Python processes"""
        python_count = 0
        try:
            for proc in psutil.process_iter(['name']):
                if proc.info['name'] in ['python', 'python3']:
                    python_count += 1
        except:
            pass
        return python_count
    
    def get_process_metrics(self, pid: int) -> Dict[str, Any]:
        """Get detailed metrics for a specific process"""
        try:
            process = psutil.Process(pid)
            
            # Get memory info
            memory_info = process.memory_info()
            memory_percent = process.memory_percent()
            
            # Get CPU info
            cpu_percent = process.cpu_percent()
            cpu_times = process.cpu_times()
            
            # Get I/O info (if available)
            try:
                io_counters = process.io_counters()
            except (psutil.AccessDenied, AttributeError):
                io_counters = None
            
            # Get file descriptors count (if available)
            try:
                num_fds = process.num_fds()
            except (psutil.AccessDenied, AttributeError):
                num_fds = None
            
            return {
                "pid": pid,
                "name": process.name(),
                "status": process.status(),
                "create_time": process.create_time(),
                "memory": {
                    "rss_mb": memory_info.rss / (1024**2),  # Resident Set Size
                    "vms_mb": memory_info.vms / (1024**2),  # Virtual Memory Size
                    "percent": memory_percent
                },
                "cpu": {
                    "percent": cpu_percent,
                    "user_time": cpu_times.user,
                    "system_time": cpu_times.system
                },
                "io": {
                    "read_count": io_counters.read_count if io_counters else 0,
                    "write_count": io_counters.write_count if io_counters else 0,
                    "read_bytes": io_counters.read_bytes if io_counters else 0,
                    "write_bytes": io_counters.write_bytes if io_counters else 0
                } if io_counters else None,
                "file_descriptors": num_fds,
                "num_threads": process.num_threads()
            }
            
        except psutil.NoSuchProcess:
            return None
        except Exception as e:
            self.logger.error(f"Error getting process metrics for PID {pid}: {e}")
            return None
    
    def collect_current_metrics(self) -> Dict[str, Any]:
        """Collect comprehensive current system metrics"""
        try:
            # System-wide metrics (similar to baseline but with current values)
            memory = psutil.virtual_memory()
            cpu_percent = psutil.cpu_percent(interval=1)
            disk = psutil.disk_usage('/')
            network = psutil.net_io_counters()
            disk_io = psutil.disk_io_counters()
            
            # Per-CPU metrics
            cpu_per_core = psutil.cpu_percent(percpu=True, interval=1)
            
            metrics = {
                "timestamp": datetime.now().isoformat(),
                "memory": {
                    "total_gb": memory.total / (1024**3),
                    "available_gb": memory.available / (1024**3),
                    "percent_used": memory.percent,
                    "free_gb": memory.free / (1024**3),
                    "cached_gb": memory.cached / (1024**3) if hasattr(memory, 'cached') else 0,
                    "buffers_gb": memory.buffers / (1024**3) if hasattr(memory, 'buffers') else 0
                },
                "cpu": {
                    "percent_used": cpu_percent,
                    "per_core_percent": cpu_per_core,
                    "core_count": len(cpu_per_core),
                    "load_average": list(psutil.getloadavg()) if hasattr(psutil, 'getloadavg') else None
                },
                "disk": {
                    "total_gb": disk.total / (1024**3),
                    "free_gb": disk.free / (1024**3),
                    "used_gb": disk.used / (1024**3),
                    "percent_used": (disk.used / disk.total) * 100
                },
                "disk_io": {
                    "read_count": disk_io.read_count if disk_io else 0,
                    "write_count": disk_io.write_count if disk_io else 0,
                    "read_bytes": disk_io.read_bytes if disk_io else 0,
                    "write_bytes": disk_io.write_bytes if disk_io else 0,
                    "read_time": disk_io.read_time if disk_io else 0,
                    "write_time": disk_io.write_time if disk_io else 0
                } if disk_io else None,
                "network": {
                    "bytes_sent": network.bytes_sent,
                    "bytes_recv": network.bytes_recv,
                    "packets_sent": network.packets_sent,
                    "packets_recv": network.packets_recv,
                    "errors_in": network.errin,
                    "errors_out": network.errout
                },
                "processes": {
                    "total_count": len(psutil.pids()),
                    "python_processes": self.count_python_processes()
                }
            }
            
            # Add specific process metrics if monitoring specific processes
            if self.process_whitelist:
                metrics["monitored_processes"] = {}
                for pid in self.process_whitelist:
                    proc_metrics = self.get_process_metrics(pid)
                    if proc_metrics:
                        metrics["monitored_processes"][str(pid)] = proc_metrics
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error collecting current metrics: {e}")
            return {}
    
    def calculate_deltas(self, current_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate deltas from baseline metrics"""
        if not self.baseline_metrics:
            return {}
        
        try:
            deltas = {
                "timestamp": current_metrics["timestamp"],
                "memory": {
                    "percent_change": current_metrics["memory"]["percent_used"] - self.baseline_metrics["memory"]["percent_used"],
                    "available_gb_change": current_metrics["memory"]["available_gb"] - self.baseline_metrics["memory"]["available_gb"]
                },
                "cpu": {
                    "percent_change": current_metrics["cpu"]["percent_used"] - self.baseline_metrics["cpu"]["percent_used"]
                },
                "disk": {
                    "percent_change": current_metrics["disk"]["percent_used"] - self.baseline_metrics["disk"]["percent_used"],
                    "free_gb_change": current_metrics["disk"]["free_gb"] - self.baseline_metrics["disk"]["free_gb"]
                },
                "network": {
                    "bytes_sent_delta": current_metrics["network"]["bytes_sent"] - self.baseline_metrics["network"]["bytes_sent"],
                    "bytes_recv_delta": current_metrics["network"]["bytes_recv"] - self.baseline_metrics["network"]["bytes_recv"],
                    "packets_sent_delta": current_metrics["network"]["packets_sent"] - self.baseline_metrics["network"]["packets_sent"],
                    "packets_recv_delta": current_metrics["network"]["packets_recv"] - self.baseline_metrics["network"]["packets_recv"]
                },
                "processes": {
                    "count_change": current_metrics["processes"]["total_count"] - self.baseline_metrics["processes"]["total_count"],
                    "python_count_change": current_metrics["processes"]["python_processes"] - self.baseline_metrics["processes"]["python_processes"]
                }
            }
            
            # Add disk I/O deltas if available
            if current_metrics.get("disk_io") and self.baseline_metrics.get("disk_io"):
                deltas["disk_io"] = {
                    "read_bytes_delta": current_metrics["disk_io"]["read_bytes"] - self.baseline_metrics["disk_io"]["read_bytes"],
                    "write_bytes_delta": current_metrics["disk_io"]["write_bytes"] - self.baseline_metrics["disk_io"]["write_bytes"],
                    "read_count_delta": current_metrics["disk_io"]["read_count"] - self.baseline_metrics["disk_io"]["read_count"],
                    "write_count_delta": current_metrics["disk_io"]["write_count"] - self.baseline_metrics["disk_io"]["write_count"]
                }
            
            return deltas
            
        except Exception as e:
            self.logger.error(f"Error calculating deltas: {e}")
            return {}
    
    def start_monitoring(self):
        """Start continuous resource monitoring"""
        if self.running:
            self.logger.warning("Resource monitoring already running")
            return
        
        self.running = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()
        self.logger.info(f"Resource monitoring started with {self.monitoring_interval}s intervals")
    
    def stop_monitoring(self):
        """Stop resource monitoring"""
        self.running = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=self.monitoring_interval + 5)
        self.logger.info("Resource monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop"""
        metrics_file = Path(__file__).parent.parent / "monitoring_logs" / f"resource_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        while self.running:
            try:
                # Collect metrics
                metrics = self.collect_current_metrics()
                if metrics:
                    # Calculate deltas from baseline
                    deltas = self.calculate_deltas(metrics)
                    
                    # Add to history (limited size)
                    self.metrics_history.append(metrics)
                    if len(self.metrics_history) > self.max_history_size:
                        self.metrics_history.pop(0)
                    
                    # Log to file
                    log_entry = {
                        "metrics": metrics,
                        "deltas": deltas
                    }
                    
                    with open(metrics_file, 'a') as f:
                        f.write(json.dumps(log_entry) + '\n')
                    
                    # Log significant changes
                    if deltas.get("memory", {}).get("percent_change", 0) > 10:
                        self.logger.warning(f"Memory usage increased by {deltas['memory']['percent_change']:.1f}%")
                    
                    if deltas.get("cpu", {}).get("percent_change", 0) > 20:
                        self.logger.warning(f"CPU usage increased by {deltas['cpu']['percent_change']:.1f}%")
                
                # Sleep until next collection
                time.sleep(self.monitoring_interval)
                
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                time.sleep(self.monitoring_interval)
    
    def add_process_to_monitor(self, pid: int):
        """Add a process to the monitoring whitelist"""
        if pid not in self.process_whitelist:
            self.process_whitelist.append(pid)
            self.logger.info(f"Added PID {pid} to monitoring whitelist")
    
    def remove_process_from_monitor(self, pid: int):
        """Remove a process from the monitoring whitelist"""
        if pid in self.process_whitelist:
            self.process_whitelist.remove(pid)
            self.logger.info(f"Removed PID {pid} from monitoring whitelist")
    
    def get_summary_report(self) -> Dict[str, Any]:
        """Generate summary report of monitoring session"""
        if not self.metrics_history:
            return {"error": "No metrics collected"}
        
        try:
            # Calculate summary statistics
            memory_usage = [m["memory"]["percent_used"] for m in self.metrics_history]
            cpu_usage = [m["cpu"]["percent_used"] for m in self.metrics_history]
            
            summary = {
                "monitoring_period": {
                    "start": self.metrics_history[0]["timestamp"],
                    "end": self.metrics_history[-1]["timestamp"],
                    "duration_minutes": len(self.metrics_history) * self.monitoring_interval / 60,
                    "sample_count": len(self.metrics_history)
                },
                "memory": {
                    "min_percent": min(memory_usage),
                    "max_percent": max(memory_usage),
                    "avg_percent": sum(memory_usage) / len(memory_usage),
                    "peak_usage_gb": max([m["memory"]["total_gb"] - m["memory"]["available_gb"] for m in self.metrics_history])
                },
                "cpu": {
                    "min_percent": min(cpu_usage),
                    "max_percent": max(cpu_usage),
                    "avg_percent": sum(cpu_usage) / len(cpu_usage)
                },
                "baseline_comparison": {
                    "memory_peak_increase": max(memory_usage) - self.baseline_metrics["memory"]["percent_used"] if self.baseline_metrics else None,
                    "cpu_peak_increase": max(cpu_usage) - self.baseline_metrics["cpu"]["percent_used"] if self.baseline_metrics else None
                }
            }
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Error generating summary report: {e}")
            return {"error": str(e)}


def main():
    """Main entry point for standalone operation"""
    monitor = ResourceMonitor(monitoring_interval=5)  # 5-second intervals for testing
    
    try:
        # Capture baseline
        monitor.capture_baseline()
        
        # Start monitoring
        monitor.start_monitoring()
        
        print("Resource monitoring started. Press Ctrl+C to stop and generate report.")
        
        # Keep running until interrupted
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\nStopping resource monitoring...")
        monitor.stop_monitoring()
        
        # Generate and display summary
        summary = monitor.get_summary_report()
        print("\nMonitoring Summary:")
        print(json.dumps(summary, indent=2))
        
        # Save summary report
        report_dir = Path(__file__).parent.parent / "results_summary"
        report_dir.mkdir(exist_ok=True)
        report_file = report_dir / f"resource_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(report_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\nSummary saved to: {report_file}")


if __name__ == "__main__":
    main()