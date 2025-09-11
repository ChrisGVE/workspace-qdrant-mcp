#!/usr/bin/env python3
"""
System Guardian - Main safety orchestrator for autonomous testing
Monitors system resources and enforces safety limits during overnight testing
"""

import psutil
import time
import json
import logging
import threading
import signal
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
import subprocess

class SystemGuardian:
    def __init__(self, config_path: Optional[str] = None):
        self.config = self._load_config(config_path)
        self.running = True
        self.emergency_stop = False
        self.monitoring_thread = None
        self.last_check = datetime.now()
        
        # Setup logging
        self._setup_logging()
        
        # Safety thresholds
        self.memory_threshold = self.config.get('memory_threshold_percent', 80)
        self.cpu_threshold = self.config.get('cpu_threshold_percent', 85)
        self.disk_threshold = self.config.get('disk_threshold_percent', 90)
        self.check_interval = self.config.get('check_interval_seconds', 30)
        
        # Emergency processes to kill
        self.emergency_processes = []
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        self.logger.info("System Guardian initialized with safety thresholds")
        self.logger.info(f"Memory: {self.memory_threshold}%, CPU: {self.cpu_threshold}%, Disk: {self.disk_threshold}%")
    
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load configuration from JSON file"""
        if config_path is None:
            config_path = Path(__file__).parent / "safety_config.json"
        
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            # Return default config
            return {
                "memory_threshold_percent": 80,
                "cpu_threshold_percent": 85,
                "disk_threshold_percent": 90,
                "check_interval_seconds": 30,
                "emergency_stop_on_consecutive_violations": 3,
                "reserved_memory_mb": 2048,
                "reserved_cpu_percent": 20,
                "max_test_duration_hours": 12
            }
    
    def _setup_logging(self):
        """Setup comprehensive logging"""
        log_dir = Path(__file__).parent.parent / "monitoring_logs"
        log_dir.mkdir(exist_ok=True)
        
        log_file = log_dir / f"system_guardian_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        self.logger.warning(f"Received signal {signum}, initiating shutdown")
        self.emergency_stop = True
        self.stop_monitoring()
    
    def get_system_metrics(self) -> Dict[str, Any]:
        """Get current system resource metrics"""
        try:
            # Memory metrics
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            memory_available_gb = memory.available / (1024**3)
            
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_count = psutil.cpu_count()
            
            # Disk metrics
            disk = psutil.disk_usage('/')
            disk_percent = (disk.used / disk.total) * 100
            disk_free_gb = disk.free / (1024**3)
            
            # Network metrics (for MCP monitoring)
            network = psutil.net_io_counters()
            
            # Process count
            process_count = len(psutil.pids())
            
            return {
                "timestamp": datetime.now().isoformat(),
                "memory": {
                    "percent_used": memory_percent,
                    "available_gb": memory_available_gb,
                    "total_gb": memory.total / (1024**3)
                },
                "cpu": {
                    "percent_used": cpu_percent,
                    "core_count": cpu_count,
                    "load_average": psutil.getloadavg() if hasattr(psutil, 'getloadavg') else None
                },
                "disk": {
                    "percent_used": disk_percent,
                    "free_gb": disk_free_gb,
                    "total_gb": disk.total / (1024**3)
                },
                "network": {
                    "bytes_sent": network.bytes_sent,
                    "bytes_recv": network.bytes_recv,
                    "packets_sent": network.packets_sent,
                    "packets_recv": network.packets_recv
                },
                "processes": {
                    "count": process_count
                }
            }
        except Exception as e:
            self.logger.error(f"Error getting system metrics: {e}")
            return {}
    
    def check_safety_thresholds(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Check if system metrics exceed safety thresholds"""
        violations = []
        warnings = []
        
        # Memory check
        memory_percent = metrics.get("memory", {}).get("percent_used", 0)
        if memory_percent > self.memory_threshold:
            violations.append(f"Memory usage {memory_percent:.1f}% exceeds threshold {self.memory_threshold}%")
        elif memory_percent > self.memory_threshold - 10:  # Warning zone
            warnings.append(f"Memory usage {memory_percent:.1f}% approaching threshold")
        
        # CPU check
        cpu_percent = metrics.get("cpu", {}).get("percent_used", 0)
        if cpu_percent > self.cpu_threshold:
            violations.append(f"CPU usage {cpu_percent:.1f}% exceeds threshold {self.cpu_threshold}%")
        elif cpu_percent > self.cpu_threshold - 10:  # Warning zone
            warnings.append(f"CPU usage {cpu_percent:.1f}% approaching threshold")
        
        # Disk check
        disk_percent = metrics.get("disk", {}).get("percent_used", 0)
        if disk_percent > self.disk_threshold:
            violations.append(f"Disk usage {disk_percent:.1f}% exceeds threshold {self.disk_threshold}%")
        elif disk_percent > self.disk_threshold - 5:  # Warning zone
            warnings.append(f"Disk usage {disk_percent:.1f}% approaching threshold")
        
        return {
            "violations": violations,
            "warnings": warnings,
            "safe": len(violations) == 0
        }
    
    def register_test_process(self, pid: int, name: str):
        """Register a test process for emergency shutdown"""
        self.emergency_processes.append({"pid": pid, "name": name})
        self.logger.info(f"Registered test process {name} (PID: {pid})")
    
    def emergency_shutdown(self):
        """Emergency shutdown of all registered test processes"""
        self.logger.critical("EMERGENCY SHUTDOWN INITIATED")
        
        for process_info in self.emergency_processes:
            try:
                pid = process_info["pid"]
                name = process_info["name"]
                
                if psutil.pid_exists(pid):
                    process = psutil.Process(pid)
                    self.logger.warning(f"Terminating process {name} (PID: {pid})")
                    process.terminate()
                    
                    # Wait for graceful shutdown
                    try:
                        process.wait(timeout=10)
                    except psutil.TimeoutExpired:
                        self.logger.error(f"Force killing process {name} (PID: {pid})")
                        process.kill()
                else:
                    self.logger.info(f"Process {name} (PID: {pid}) already terminated")
                    
            except Exception as e:
                self.logger.error(f"Error shutting down process {process_info}: {e}")
        
        self.emergency_processes.clear()
        self.emergency_stop = True
    
    def start_monitoring(self):
        """Start continuous system monitoring"""
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.logger.warning("Monitoring already running")
            return
        
        self.running = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        self.logger.info("System monitoring started")
    
    def stop_monitoring(self):
        """Stop system monitoring"""
        self.running = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        self.logger.info("System monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop"""
        consecutive_violations = 0
        max_violations = self.config.get("emergency_stop_on_consecutive_violations", 3)
        
        while self.running and not self.emergency_stop:
            try:
                # Get system metrics
                metrics = self.get_system_metrics()
                
                # Check safety thresholds
                safety_check = self.check_safety_thresholds(metrics)
                
                # Log metrics to file
                self._log_metrics(metrics, safety_check)
                
                # Handle violations
                if not safety_check["safe"]:
                    consecutive_violations += 1
                    self.logger.error(f"Safety violation #{consecutive_violations}: {safety_check['violations']}")
                    
                    if consecutive_violations >= max_violations:
                        self.logger.critical(f"Maximum consecutive violations ({max_violations}) reached")
                        self.emergency_shutdown()
                        break
                else:
                    consecutive_violations = 0
                    if safety_check["warnings"]:
                        self.logger.warning(f"System warnings: {safety_check['warnings']}")
                
                # Update last check time
                self.last_check = datetime.now()
                
                # Sleep until next check
                time.sleep(self.check_interval)
                
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                time.sleep(self.check_interval)
    
    def _log_metrics(self, metrics: Dict[str, Any], safety_check: Dict[str, Any]):
        """Log metrics to structured log file"""
        log_dir = Path(__file__).parent.parent / "monitoring_logs"
        metrics_file = log_dir / f"system_metrics_{datetime.now().strftime('%Y%m%d')}.json"
        
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "metrics": metrics,
            "safety_check": safety_check
        }
        
        try:
            # Append to daily metrics file
            with open(metrics_file, 'a') as f:
                f.write(json.dumps(log_entry) + '\n')
        except Exception as e:
            self.logger.error(f"Error logging metrics: {e}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get current guardian status"""
        return {
            "running": self.running,
            "emergency_stop": self.emergency_stop,
            "last_check": self.last_check.isoformat(),
            "registered_processes": len(self.emergency_processes),
            "config": self.config
        }


def main():
    """Main entry point for standalone operation"""
    guardian = SystemGuardian()
    
    try:
        guardian.start_monitoring()
        
        # Keep running until stopped
        while guardian.running and not guardian.emergency_stop:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\nShutting down System Guardian...")
    finally:
        guardian.stop_monitoring()


if __name__ == "__main__":
    main()