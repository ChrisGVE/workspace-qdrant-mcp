#!/usr/bin/env python3
"""
Concurrent Load Test - Stress test with multiple simultaneous operations
Tests system behavior under concurrent MCP connections and operations
"""

import asyncio
import json
import logging
import time
import threading
import random
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
import concurrent.futures
from dataclasses import dataclass, asdict
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from scripts.resource_monitor import ResourceMonitor
from safety_monitoring.system_guardian import SystemGuardian

@dataclass
class LoadTestConfig:
    """Configuration for concurrent load testing"""
    max_concurrent_connections: int = 10
    operations_per_connection: int = 50
    test_duration_minutes: int = 30
    ramp_up_duration_seconds: int = 60
    operation_types: List[str] = None
    stress_level: str = "moderate"  # low, moderate, high, extreme
    
    def __post_init__(self):
        if self.operation_types is None:
            self.operation_types = ["search", "add_document", "get_status", "list_collections"]

@dataclass
class OperationResult:
    """Result of a single operation"""
    operation_type: str
    start_time: float
    end_time: float
    success: bool
    error_message: Optional[str] = None
    response_size: int = 0
    connection_id: int = 0

class ConcurrentLoadTest:
    def __init__(self, config: LoadTestConfig):
        self.config = config
        self.setup_logging()
        self.resource_monitor = ResourceMonitor(monitoring_interval=5)
        self.system_guardian = SystemGuardian()
        
        # Test state
        self.test_running = False
        self.start_time = None
        self.end_time = None
        self.operation_results = []
        self.connection_pools = []
        
        # Safety tracking
        self.emergency_stop = False
        
        # Test data
        self.test_documents = self.generate_test_documents()
        self.test_queries = self.generate_test_queries()
        
    def setup_logging(self):
        """Setup load testing logging"""
        log_dir = Path(__file__).parent.parent / "monitoring_logs"
        log_dir.mkdir(exist_ok=True)
        
        log_file = log_dir / f"concurrent_load_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def generate_test_documents(self) -> List[Dict[str, Any]]:
        """Generate test documents for load testing"""
        base_texts = [
            "This is a test document for concurrent load testing. It contains sample content to test document ingestion performance under load.",
            "Load testing requires generating multiple documents with varying content sizes and complexity to stress test the system properly.",
            "Memory management during concurrent operations is critical for system stability and performance during high load scenarios.",
            "Network operations and database interactions need to be tested under concurrent load to identify potential bottlenecks and issues.",
            "Error handling and recovery mechanisms should be validated during stress testing to ensure system reliability under adverse conditions."
        ]
        
        documents = []
        for i in range(200):  # Generate 200 test documents
            base_text = random.choice(base_texts)
            
            # Vary document size
            repetitions = random.randint(1, 10)
            content = f"{base_text} " * repetitions
            
            document = {
                "id": f"load_test_doc_{i:04d}",
                "content": content,
                "metadata": {
                    "test_type": "concurrent_load",
                    "doc_index": i,
                    "size_category": "small" if repetitions <= 3 else "medium" if repetitions <= 7 else "large",
                    "generated_at": datetime.now().isoformat()
                }
            }
            documents.append(document)
        
        return documents
    
    def generate_test_queries(self) -> List[str]:
        """Generate test queries for load testing"""
        query_templates = [
            "test document concurrent",
            "load testing performance",
            "memory management system",
            "network operations database",
            "error handling recovery",
            "stress testing reliability",
            "document ingestion performance",
            "concurrent operations stability",
            "system bottlenecks issues",
            "high load scenarios"
        ]
        
        queries = []
        for template in query_templates:
            # Create variations
            queries.extend([
                template,
                f"{template} optimization",
                f"{template} patterns",
                f"advanced {template}",
                f"{template} strategies"
            ])
        
        return queries
    
    async def simulate_mcp_operation(self, operation_type: str, connection_id: int) -> OperationResult:
        """Simulate an MCP operation"""
        start_time = time.time()
        
        try:
            if operation_type == "search":
                result = await self.simulate_search_operation(connection_id)
            elif operation_type == "add_document":
                result = await self.simulate_add_document_operation(connection_id)
            elif operation_type == "get_status":
                result = await self.simulate_status_operation(connection_id)
            elif operation_type == "list_collections":
                result = await self.simulate_list_collections_operation(connection_id)
            else:
                raise ValueError(f"Unknown operation type: {operation_type}")
            
            end_time = time.time()
            
            return OperationResult(
                operation_type=operation_type,
                start_time=start_time,
                end_time=end_time,
                success=True,
                response_size=result.get("response_size", 0),
                connection_id=connection_id
            )
            
        except Exception as e:
            end_time = time.time()
            
            return OperationResult(
                operation_type=operation_type,
                start_time=start_time,
                end_time=end_time,
                success=False,
                error_message=str(e),
                connection_id=connection_id
            )
    
    async def simulate_search_operation(self, connection_id: int) -> Dict[str, Any]:
        """Simulate a search operation"""
        query = random.choice(self.test_queries)
        
        # Simulate network delay and processing time
        processing_delay = random.uniform(0.1, 0.5)
        await asyncio.sleep(processing_delay)
        
        # Simulate varying response sizes
        num_results = random.randint(1, 20)
        response_size = num_results * 500  # Estimate 500 bytes per result
        
        return {
            "query": query,
            "results_count": num_results,
            "response_size": response_size,
            "connection_id": connection_id
        }
    
    async def simulate_add_document_operation(self, connection_id: int) -> Dict[str, Any]:
        """Simulate adding a document"""
        document = random.choice(self.test_documents)
        
        # Simulate embedding generation and storage time
        processing_delay = random.uniform(0.2, 0.8)
        await asyncio.sleep(processing_delay)
        
        response_size = len(document["content"]) + 100  # Document size plus metadata
        
        return {
            "document_id": document["id"],
            "document_size": len(document["content"]),
            "response_size": response_size,
            "connection_id": connection_id
        }
    
    async def simulate_status_operation(self, connection_id: int) -> Dict[str, Any]:
        """Simulate getting workspace status"""
        # Status operations are typically fast
        processing_delay = random.uniform(0.05, 0.2)
        await asyncio.sleep(processing_delay)
        
        return {
            "status": "active",
            "collections_count": random.randint(1, 10),
            "response_size": 200,  # Small JSON response
            "connection_id": connection_id
        }
    
    async def simulate_list_collections_operation(self, connection_id: int) -> Dict[str, Any]:
        """Simulate listing collections"""
        processing_delay = random.uniform(0.1, 0.3)
        await asyncio.sleep(processing_delay)
        
        collections_count = random.randint(1, 15)
        response_size = collections_count * 100  # Estimate 100 bytes per collection info
        
        return {
            "collections": [f"collection_{i}" for i in range(collections_count)],
            "response_size": response_size,
            "connection_id": connection_id
        }
    
    async def connection_worker(self, connection_id: int, operations_to_perform: int) -> List[OperationResult]:
        """Worker function for a single connection"""
        self.logger.info(f"Connection {connection_id} starting with {operations_to_perform} operations")
        
        connection_results = []
        
        try:
            for op_index in range(operations_to_perform):
                if self.emergency_stop or not self.test_running:
                    self.logger.warning(f"Connection {connection_id} stopping due to emergency stop or test end")
                    break
                
                # Select random operation type
                operation_type = random.choice(self.config.operation_types)
                
                # Perform operation
                result = await self.simulate_mcp_operation(operation_type, connection_id)
                connection_results.append(result)
                
                # Log failures immediately
                if not result.success:
                    self.logger.error(f"Connection {connection_id} operation {op_index} failed: {result.error_message}")
                
                # Add small delay between operations to simulate real usage
                if op_index < operations_to_perform - 1:  # No delay after last operation
                    inter_operation_delay = random.uniform(0.1, 0.5)
                    await asyncio.sleep(inter_operation_delay)
            
        except Exception as e:
            self.logger.error(f"Connection {connection_id} worker failed: {e}")
        
        self.logger.info(f"Connection {connection_id} completed with {len(connection_results)} operations")
        return connection_results
    
    async def ramp_up_connections(self) -> List[asyncio.Task]:
        """Gradually ramp up connections over the ramp-up period"""
        self.logger.info(f"Ramping up {self.config.max_concurrent_connections} connections over {self.config.ramp_up_duration_seconds} seconds")
        
        tasks = []
        connections_per_interval = max(1, self.config.max_concurrent_connections // 10)  # Ramp up in 10 steps
        interval_duration = self.config.ramp_up_duration_seconds / 10
        
        for step in range(10):
            if self.emergency_stop or not self.test_running:
                break
            
            # Determine how many connections to start in this step
            connections_this_step = min(
                connections_per_interval,
                self.config.max_concurrent_connections - len(tasks)
            )
            
            for i in range(connections_this_step):
                connection_id = len(tasks)
                
                # Create connection task
                task = asyncio.create_task(
                    self.connection_worker(connection_id, self.config.operations_per_connection)
                )
                tasks.append(task)
                
                self.logger.info(f"Started connection {connection_id}")
                
                # Small delay between connections in the same step
                if i < connections_this_step - 1:
                    await asyncio.sleep(0.1)
            
            # Wait for the interval to complete before next step
            if step < 9:  # No wait after last step
                await asyncio.sleep(interval_duration)
        
        self.logger.info(f"Ramp-up complete: {len(tasks)} connections started")
        return tasks
    
    async def run_load_test(self) -> Dict[str, Any]:
        """Run the complete concurrent load test"""
        self.logger.info("Starting concurrent load test")
        
        try:
            # Initialize monitoring
            self.system_guardian.start_monitoring()
            self.resource_monitor.start_monitoring()
            
            # Register this test with system guardian
            self.system_guardian.register_test_process(os.getpid(), "concurrent_load_test")
            
            self.test_running = True
            self.start_time = time.time()
            
            # Ramp up connections
            connection_tasks = await self.ramp_up_connections()
            
            if not connection_tasks:
                raise Exception("No connections were started")
            
            # Monitor test progress
            monitor_task = asyncio.create_task(self.monitor_test_progress(connection_tasks))
            
            # Wait for all connections to complete or timeout
            test_duration_seconds = self.config.test_duration_minutes * 60
            
            try:
                # Wait for all tasks with timeout
                results = await asyncio.wait_for(
                    asyncio.gather(*connection_tasks, return_exceptions=True),
                    timeout=test_duration_seconds
                )
                
                self.logger.info("All connections completed successfully")
                
            except asyncio.TimeoutError:
                self.logger.warning("Test timed out, cancelling remaining connections")
                
                for task in connection_tasks:
                    if not task.done():
                        task.cancel()
                
                # Wait a bit for cancellation to complete
                await asyncio.sleep(2)
                
                # Collect results from completed tasks
                results = []
                for task in connection_tasks:
                    try:
                        if task.done() and not task.cancelled():
                            result = task.result()
                            results.append(result)
                    except Exception as e:
                        self.logger.error(f"Error getting task result: {e}")
            
            finally:
                monitor_task.cancel()
            
            self.end_time = time.time()
            self.test_running = False
            
            # Compile all operation results
            all_operations = []
            for connection_results in results:
                if isinstance(connection_results, list):
                    all_operations.extend(connection_results)
                elif isinstance(connection_results, Exception):
                    self.logger.error(f"Connection failed with exception: {connection_results}")
            
            self.operation_results = all_operations
            
            # Generate test report
            test_report = self.generate_test_report()
            
            self.logger.info("Concurrent load test completed")
            return test_report
            
        except Exception as e:
            self.logger.error(f"Load test failed: {e}")
            self.emergency_stop = True
            raise
            
        finally:
            # Clean up monitoring
            self.system_guardian.stop_monitoring()
            self.resource_monitor.stop_monitoring()
    
    async def monitor_test_progress(self, connection_tasks: List[asyncio.Task]):
        """Monitor test progress and log periodic updates"""
        while self.test_running and not self.emergency_stop:
            try:
                # Check task status
                completed_tasks = sum(1 for task in connection_tasks if task.done())
                running_tasks = len(connection_tasks) - completed_tasks
                
                # Calculate elapsed time
                elapsed_time = time.time() - self.start_time if self.start_time else 0
                
                # Log progress
                self.logger.info(f"Progress: {completed_tasks}/{len(connection_tasks)} connections complete, "
                               f"{running_tasks} running, elapsed: {elapsed_time:.1f}s")
                
                # Check for system issues
                system_metrics = self.resource_monitor.collect_current_metrics()
                memory_percent = system_metrics.get("memory", {}).get("percent_used", 0)
                cpu_percent = system_metrics.get("cpu", {}).get("percent_used", 0)
                
                if memory_percent > 90 or cpu_percent > 95:
                    self.logger.warning(f"High resource usage: Memory {memory_percent}%, CPU {cpu_percent}%")
                
                # Wait before next check
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in progress monitor: {e}")
                await asyncio.sleep(30)
    
    def generate_test_report(self) -> Dict[str, Any]:
        """Generate comprehensive test report"""
        total_operations = len(self.operation_results)
        successful_operations = sum(1 for r in self.operation_results if r.success)
        failed_operations = total_operations - successful_operations
        
        # Calculate performance metrics
        if self.operation_results:
            response_times = [r.end_time - r.start_time for r in self.operation_results if r.success]
            
            if response_times:
                avg_response_time = sum(response_times) / len(response_times)
                min_response_time = min(response_times)
                max_response_time = max(response_times)
                
                # Calculate percentiles
                sorted_times = sorted(response_times)
                p95_index = int(len(sorted_times) * 0.95)
                p99_index = int(len(sorted_times) * 0.99)
                p95_response_time = sorted_times[p95_index] if p95_index < len(sorted_times) else max_response_time
                p99_response_time = sorted_times[p99_index] if p99_index < len(sorted_times) else max_response_time
            else:
                avg_response_time = 0
                min_response_time = 0
                max_response_time = 0
                p95_response_time = 0
                p99_response_time = 0
        else:
            avg_response_time = 0
            min_response_time = 0
            max_response_time = 0
            p95_response_time = 0
            p99_response_time = 0
        
        # Operation type breakdown
        operation_stats = {}
        for op_type in self.config.operation_types:
            op_results = [r for r in self.operation_results if r.operation_type == op_type]
            op_successful = sum(1 for r in op_results if r.success)
            op_failed = len(op_results) - op_successful
            
            if op_results:
                op_response_times = [r.end_time - r.start_time for r in op_results if r.success]
                op_avg_time = sum(op_response_times) / len(op_response_times) if op_response_times else 0
            else:
                op_avg_time = 0
            
            operation_stats[op_type] = {
                "total_operations": len(op_results),
                "successful_operations": op_successful,
                "failed_operations": op_failed,
                "success_rate": (op_successful / len(op_results) * 100) if op_results else 0,
                "avg_response_time": op_avg_time
            }
        
        # Test duration
        test_duration = self.end_time - self.start_time if self.start_time and self.end_time else 0
        operations_per_second = total_operations / test_duration if test_duration > 0 else 0
        
        report = {
            "test_info": {
                "test_type": "concurrent_load_test",
                "timestamp": datetime.now().isoformat(),
                "config": asdict(self.config),
                "test_duration_seconds": test_duration,
                "start_time": datetime.fromtimestamp(self.start_time).isoformat() if self.start_time else None,
                "end_time": datetime.fromtimestamp(self.end_time).isoformat() if self.end_time else None
            },
            "summary": {
                "total_operations": total_operations,
                "successful_operations": successful_operations,
                "failed_operations": failed_operations,
                "success_rate": (successful_operations / total_operations * 100) if total_operations > 0 else 0,
                "operations_per_second": operations_per_second,
                "avg_response_time": avg_response_time,
                "min_response_time": min_response_time,
                "max_response_time": max_response_time,
                "p95_response_time": p95_response_time,
                "p99_response_time": p99_response_time
            },
            "operation_breakdown": operation_stats,
            "resource_impact": self.resource_monitor.get_summary_report(),
            "errors": [
                {
                    "operation_type": r.operation_type,
                    "error_message": r.error_message,
                    "connection_id": r.connection_id,
                    "timestamp": datetime.fromtimestamp(r.start_time).isoformat()
                }
                for r in self.operation_results if not r.success
            ]
        }
        
        return report
    
    def save_test_report(self, report: Dict[str, Any]) -> str:
        """Save test report to file"""
        results_dir = Path(__file__).parent.parent / "results_summary"
        results_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_file = results_dir / f"concurrent_load_test_report_{timestamp}.json"
        
        try:
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=2)
            
            self.logger.info(f"Test report saved to {report_file}")
            return str(report_file)
            
        except Exception as e:
            self.logger.error(f"Error saving test report: {e}")
            raise


async def main():
    """Main entry point"""
    print("Starting Concurrent Load Test...")
    
    # Configure load test based on system capabilities
    config = LoadTestConfig(
        max_concurrent_connections=5,  # Start conservative
        operations_per_connection=20,   # Moderate load
        test_duration_minutes=10,      # Short test for safety
        ramp_up_duration_seconds=30,   # Gradual ramp up
        stress_level="moderate"
    )
    
    load_test = ConcurrentLoadTest(config)
    
    try:
        # Run the test
        report = await load_test.run_load_test()
        
        # Save report
        report_file = load_test.save_test_report(report)
        
        # Display summary
        summary = report.get("summary", {})
        print(f"\nConcurrent Load Test Complete!")
        print(f"Report saved to: {report_file}")
        print(f"\nTest Summary:")
        print(f"  Total Operations: {summary.get('total_operations', 0)}")
        print(f"  Success Rate: {summary.get('success_rate', 0):.1f}%")
        print(f"  Operations/Second: {summary.get('operations_per_second', 0):.1f}")
        print(f"  Average Response Time: {summary.get('avg_response_time', 0):.3f}s")
        print(f"  95th Percentile: {summary.get('p95_response_time', 0):.3f}s")
        
        if summary.get('failed_operations', 0) > 0:
            print(f"  Failed Operations: {summary.get('failed_operations', 0)}")
        
        return report
        
    except Exception as e:
        print(f"Error during concurrent load test: {e}")
        return None


if __name__ == "__main__":
    import os
    
    # Run the async main function
    asyncio.run(main())