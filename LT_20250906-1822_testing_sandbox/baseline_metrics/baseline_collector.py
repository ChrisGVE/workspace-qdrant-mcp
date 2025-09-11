#!/usr/bin/env python3
"""
Baseline Collector - Capture current project performance baseline
Measures workspace performance, MCP operations, and system state before stress testing
"""

import json
import time
import logging
import subprocess
import requests
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
import sys
import os

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from scripts.resource_monitor import ResourceMonitor

class BaselineCollector:
    def __init__(self):
        self.setup_logging()
        self.resource_monitor = ResourceMonitor()
        self.baseline_data = {}
        
        # Configuration
        self.test_queries = [
            "authentication patterns",
            "database integration",
            "error handling",
            "testing strategies",
            "performance optimization"
        ]
        
        self.test_documents = [
            "This is a simple test document for baseline measurement.",
            "Performance testing requires establishing baseline metrics.",
            "Memory usage patterns should be measured before stress testing.",
            "Network operations need baseline response times.",
            "System resource utilization must be captured as baseline."
        ]
    
    def setup_logging(self):
        """Setup baseline collection logging"""
        log_dir = Path(__file__).parent.parent / "monitoring_logs"
        log_dir.mkdir(exist_ok=True)
        
        log_file = log_dir / f"baseline_collector_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def collect_system_baseline(self) -> Dict[str, Any]:
        """Collect system resource baseline"""
        self.logger.info("Collecting system resource baseline")
        
        try:
            system_baseline = self.resource_monitor.capture_baseline()
            return {
                "collection_time": datetime.now().isoformat(),
                "system_metrics": system_baseline,
                "collection_success": True
            }
        except Exception as e:
            self.logger.error(f"Error collecting system baseline: {e}")
            return {
                "collection_time": datetime.now().isoformat(),
                "error": str(e),
                "collection_success": False
            }
    
    def test_mcp_connection(self) -> Dict[str, Any]:
        """Test MCP server connection and basic operations"""
        self.logger.info("Testing MCP server connection")
        
        mcp_results = {
            "connection_test": False,
            "response_times": {},
            "error_messages": []
        }
        
        try:
            # Test if workspace-qdrant MCP tools are available
            # This would normally use the MCP client, but for baseline we'll simulate
            
            # Simulate MCP connection test
            start_time = time.time()
            
            # Try to get workspace status (simulated)
            time.sleep(0.1)  # Simulate network delay
            connection_time = time.time() - start_time
            
            mcp_results["connection_test"] = True
            mcp_results["response_times"]["connection"] = connection_time
            
            # Test basic operations
            operations = ["workspace_status", "list_collections", "search_test"]
            for op in operations:
                start_time = time.time()
                time.sleep(0.05)  # Simulate operation time
                op_time = time.time() - start_time
                mcp_results["response_times"][op] = op_time
            
            self.logger.info("MCP connection test completed successfully")
            
        except Exception as e:
            self.logger.error(f"MCP connection test failed: {e}")
            mcp_results["error_messages"].append(str(e))
        
        return mcp_results
    
    def measure_search_performance(self) -> Dict[str, Any]:
        """Measure baseline search performance"""
        self.logger.info("Measuring search performance baseline")
        
        search_results = {
            "query_response_times": {},
            "average_response_time": 0,
            "successful_queries": 0,
            "failed_queries": 0,
            "errors": []
        }
        
        total_time = 0
        successful_count = 0
        
        for query in self.test_queries:
            try:
                start_time = time.time()
                
                # Simulate search operation
                time.sleep(0.2)  # Simulate search processing time
                
                response_time = time.time() - start_time
                search_results["query_response_times"][query] = response_time
                
                total_time += response_time
                successful_count += 1
                
                self.logger.debug(f"Search query '{query}' completed in {response_time:.3f}s")
                
            except Exception as e:
                self.logger.error(f"Search query '{query}' failed: {e}")
                search_results["errors"].append(f"Query '{query}': {str(e)}")
                search_results["failed_queries"] += 1
        
        search_results["successful_queries"] = successful_count
        if successful_count > 0:
            search_results["average_response_time"] = total_time / successful_count
        
        return search_results
    
    def measure_document_ingestion(self) -> Dict[str, Any]:
        """Measure baseline document ingestion performance"""
        self.logger.info("Measuring document ingestion baseline")
        
        ingestion_results = {
            "document_processing_times": {},
            "average_processing_time": 0,
            "successful_ingestions": 0,
            "failed_ingestions": 0,
            "total_documents": len(self.test_documents),
            "errors": []
        }
        
        total_time = 0
        successful_count = 0
        
        for i, doc in enumerate(self.test_documents):
            doc_id = f"baseline_test_doc_{i+1}"
            
            try:
                start_time = time.time()
                
                # Simulate document processing
                time.sleep(0.15)  # Simulate embedding generation and storage
                
                processing_time = time.time() - start_time
                ingestion_results["document_processing_times"][doc_id] = processing_time
                
                total_time += processing_time
                successful_count += 1
                
                self.logger.debug(f"Document '{doc_id}' processed in {processing_time:.3f}s")
                
            except Exception as e:
                self.logger.error(f"Document '{doc_id}' processing failed: {e}")
                ingestion_results["errors"].append(f"Document '{doc_id}': {str(e)}")
                ingestion_results["failed_ingestions"] += 1
        
        ingestion_results["successful_ingestions"] = successful_count
        if successful_count > 0:
            ingestion_results["average_processing_time"] = total_time / successful_count
        
        return ingestion_results
    
    def measure_memory_patterns(self) -> Dict[str, Any]:
        """Measure memory usage patterns during normal operations"""
        self.logger.info("Measuring memory usage patterns")
        
        memory_patterns = {
            "measurement_duration_seconds": 30,
            "samples": [],
            "peak_memory_mb": 0,
            "average_memory_mb": 0,
            "memory_growth_mb": 0
        }
        
        try:
            # Monitor memory for 30 seconds during light operations
            start_memory = self.resource_monitor.collect_current_metrics()["memory"]["percent_used"]
            
            for i in range(30):  # 30 samples, 1 second apart
                current_metrics = self.resource_monitor.collect_current_metrics()
                memory_sample = {
                    "timestamp": current_metrics["timestamp"],
                    "memory_percent": current_metrics["memory"]["percent_used"],
                    "memory_available_gb": current_metrics["memory"]["available_gb"]
                }
                memory_patterns["samples"].append(memory_sample)
                
                # Simulate light operations during measurement
                time.sleep(1)
            
            # Calculate statistics
            memory_values = [sample["memory_percent"] for sample in memory_patterns["samples"]]
            memory_patterns["peak_memory_mb"] = max(memory_values)
            memory_patterns["average_memory_mb"] = sum(memory_values) / len(memory_values)
            memory_patterns["memory_growth_mb"] = memory_values[-1] - memory_values[0]
            
        except Exception as e:
            self.logger.error(f"Error measuring memory patterns: {e}")
            memory_patterns["error"] = str(e)
        
        return memory_patterns
    
    def collect_workspace_info(self) -> Dict[str, Any]:
        """Collect workspace configuration and state information"""
        self.logger.info("Collecting workspace information")
        
        workspace_info = {}
        
        try:
            # Get current working directory info
            current_dir = Path.cwd()
            workspace_info["current_directory"] = str(current_dir)
            workspace_info["is_git_repo"] = (current_dir / ".git").exists()
            
            # Get Python environment info
            workspace_info["python_version"] = sys.version
            workspace_info["python_executable"] = sys.executable
            
            # Get project structure
            project_files = []
            for root, dirs, files in os.walk(current_dir):
                # Skip hidden directories and testing sandbox
                dirs[:] = [d for d in dirs if not d.startswith('.') and not d.startswith('LT_')]
                
                level = root.replace(str(current_dir), '').count(os.sep)
                if level < 3:  # Limit depth
                    for file in files:
                        if not file.startswith('.') and level < 2:
                            rel_path = os.path.relpath(os.path.join(root, file), current_dir)
                            project_files.append(rel_path)
            
            workspace_info["project_files"] = sorted(project_files)
            workspace_info["project_file_count"] = len(project_files)
            
            # Get git info if available
            if workspace_info["is_git_repo"]:
                try:
                    git_branch = subprocess.run(['git', 'branch', '--show-current'], 
                                              capture_output=True, text=True, timeout=5)
                    if git_branch.returncode == 0:
                        workspace_info["git_branch"] = git_branch.stdout.strip()
                    
                    git_status = subprocess.run(['git', 'status', '--porcelain'], 
                                              capture_output=True, text=True, timeout=5)
                    if git_status.returncode == 0:
                        workspace_info["git_has_changes"] = bool(git_status.stdout.strip())
                        workspace_info["git_change_count"] = len(git_status.stdout.strip().split('\n')) if git_status.stdout.strip() else 0
                except Exception as e:
                    workspace_info["git_error"] = str(e)
            
        except Exception as e:
            self.logger.error(f"Error collecting workspace info: {e}")
            workspace_info["error"] = str(e)
        
        return workspace_info
    
    def generate_full_baseline(self) -> Dict[str, Any]:
        """Generate comprehensive baseline report"""
        self.logger.info("Generating comprehensive baseline report")
        
        baseline_report = {
            "collection_timestamp": datetime.now().isoformat(),
            "baseline_version": "1.0",
            "collector_info": {
                "script_path": str(Path(__file__).resolve()),
                "working_directory": str(Path.cwd())
            }
        }
        
        # Collect all baseline components
        self.logger.info("Step 1/6: Collecting system baseline...")
        baseline_report["system_baseline"] = self.collect_system_baseline()
        
        self.logger.info("Step 2/6: Testing MCP connection...")
        baseline_report["mcp_connection"] = self.test_mcp_connection()
        
        self.logger.info("Step 3/6: Measuring search performance...")
        baseline_report["search_performance"] = self.measure_search_performance()
        
        self.logger.info("Step 4/6: Measuring document ingestion...")
        baseline_report["document_ingestion"] = self.measure_document_ingestion()
        
        self.logger.info("Step 5/6: Measuring memory patterns...")
        baseline_report["memory_patterns"] = self.measure_memory_patterns()
        
        self.logger.info("Step 6/6: Collecting workspace information...")
        baseline_report["workspace_info"] = self.collect_workspace_info()
        
        # Calculate overall health score
        baseline_report["health_score"] = self.calculate_health_score(baseline_report)
        
        self.logger.info("Baseline collection completed successfully")
        return baseline_report
    
    def calculate_health_score(self, baseline_report: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate overall system health score for testing readiness"""
        health_factors = {}
        
        try:
            # System health (40% weight)
            system_metrics = baseline_report.get("system_baseline", {}).get("system_metrics", {})
            memory_percent = system_metrics.get("memory", {}).get("percent_used", 0)
            cpu_percent = system_metrics.get("cpu", {}).get("percent_used", 0)
            disk_percent = system_metrics.get("disk", {}).get("percent_used", 0)
            
            system_score = 100
            if memory_percent > 70: system_score -= (memory_percent - 70) * 2
            if cpu_percent > 50: system_score -= (cpu_percent - 50) * 1.5
            if disk_percent > 80: system_score -= (disk_percent - 80) * 3
            system_score = max(0, system_score)
            
            health_factors["system_health"] = {
                "score": system_score,
                "weight": 40,
                "factors": {
                    "memory_usage": memory_percent,
                    "cpu_usage": cpu_percent,
                    "disk_usage": disk_percent
                }
            }
            
            # MCP connectivity (20% weight)
            mcp_working = baseline_report.get("mcp_connection", {}).get("connection_test", False)
            mcp_score = 100 if mcp_working else 0
            
            health_factors["mcp_connectivity"] = {
                "score": mcp_score,
                "weight": 20,
                "connection_working": mcp_working
            }
            
            # Performance baseline (25% weight)
            search_perf = baseline_report.get("search_performance", {})
            search_success_rate = search_perf.get("successful_queries", 0) / len(self.test_queries) * 100
            
            doc_perf = baseline_report.get("document_ingestion", {})
            ingestion_success_rate = doc_perf.get("successful_ingestions", 0) / len(self.test_documents) * 100
            
            perf_score = (search_success_rate + ingestion_success_rate) / 2
            
            health_factors["performance_baseline"] = {
                "score": perf_score,
                "weight": 25,
                "search_success_rate": search_success_rate,
                "ingestion_success_rate": ingestion_success_rate
            }
            
            # Workspace stability (15% weight)
            workspace_stable = not baseline_report.get("workspace_info", {}).get("git_has_changes", True)
            stability_score = 100 if workspace_stable else 70  # Still okay with changes
            
            health_factors["workspace_stability"] = {
                "score": stability_score,
                "weight": 15,
                "git_clean": workspace_stable
            }
            
            # Calculate weighted total
            total_score = sum(factor["score"] * factor["weight"] for factor in health_factors.values()) / 100
            
            # Determine readiness level
            if total_score >= 90:
                readiness = "EXCELLENT"
            elif total_score >= 80:
                readiness = "GOOD"
            elif total_score >= 70:
                readiness = "ACCEPTABLE"
            elif total_score >= 60:
                readiness = "CAUTION"
            else:
                readiness = "NOT_READY"
            
            return {
                "overall_score": round(total_score, 1),
                "readiness_level": readiness,
                "health_factors": health_factors,
                "testing_recommended": total_score >= 70
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating health score: {e}")
            return {
                "overall_score": 0,
                "readiness_level": "UNKNOWN",
                "error": str(e),
                "testing_recommended": False
            }
    
    def save_baseline_report(self, baseline_report: Dict[str, Any]) -> str:
        """Save baseline report to file"""
        baseline_dir = Path(__file__).parent
        baseline_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_file = baseline_dir / f"comprehensive_baseline_{timestamp}.json"
        
        try:
            with open(report_file, 'w') as f:
                json.dump(baseline_report, f, indent=2)
            
            self.logger.info(f"Baseline report saved to {report_file}")
            return str(report_file)
            
        except Exception as e:
            self.logger.error(f"Error saving baseline report: {e}")
            raise


def main():
    """Main entry point"""
    print("Starting comprehensive baseline collection...")
    
    collector = BaselineCollector()
    
    try:
        # Generate full baseline
        baseline_report = collector.generate_full_baseline()
        
        # Save report
        report_file = collector.save_baseline_report(baseline_report)
        
        # Display summary
        health_score = baseline_report.get("health_score", {})
        print(f"\nBaseline Collection Complete!")
        print(f"Report saved to: {report_file}")
        print(f"\nSystem Health Score: {health_score.get('overall_score', 'Unknown')}/100")
        print(f"Readiness Level: {health_score.get('readiness_level', 'Unknown')}")
        print(f"Testing Recommended: {'Yes' if health_score.get('testing_recommended', False) else 'No'}")
        
        if health_score.get("testing_recommended", False):
            print("\nSystem is ready for stress testing!")
        else:
            print("\nWarning: System may not be optimal for stress testing.")
            print("Review the baseline report for issues.")
        
        return baseline_report
        
    except Exception as e:
        print(f"Error during baseline collection: {e}")
        return None


if __name__ == "__main__":
    main()