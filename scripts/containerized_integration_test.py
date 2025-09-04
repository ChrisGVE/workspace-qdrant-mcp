#!/usr/bin/env python3
"""
Containerized Integration Test for Workspace Qdrant MCP
Tests the actual workspace-qdrant-mcp services using containerized deployment
"""

import asyncio
import json
import logging
import os
import subprocess
import sys
import tempfile
import time
import requests
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import yaml

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ContainerizedIntegrationTest:
    """Test the actual workspace-qdrant-mcp services in containers"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.docker_dir = self.project_root / "docker"
        self.compose_file = self.docker_dir / "integration-tests" / "docker-compose.yml"
        self.test_results = {}
        self.start_time = datetime.now()
        self.test_data_dir = self.project_root / "test_results" / "containerized_integration"
        self.test_data_dir.mkdir(parents=True, exist_ok=True)
        
    def setup_test_environment(self):
        """Setup containerized test environment"""
        logger.info("Setting up containerized integration test environment...")
        
        # Check if docker-compose is available
        try:
            result = subprocess.run(["docker-compose", "--version"], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                logger.info(f"Docker Compose available: {result.stdout.strip()}")
            else:
                logger.error("Docker Compose not available")
                return False
        except FileNotFoundError:
            logger.error("Docker Compose not found")
            return False
            
        # Verify compose file exists
        if not self.compose_file.exists():
            logger.error(f"Compose file not found: {self.compose_file}")
            return False
            
        # Clean up any existing test containers
        self.cleanup_test_environment()
        
        return True
        
    def cleanup_test_environment(self):
        """Clean up test environment"""
        logger.info("Cleaning up test environment...")
        
        try:
            subprocess.run([
                "docker-compose", "-f", str(self.compose_file), "down", "-v"
            ], capture_output=True, cwd=self.compose_file.parent)
            logger.info("Test environment cleaned up")
        except Exception as e:
            logger.warning(f"Cleanup warning: {e}")
            
    async def test_service_startup_sequence(self):
        """Test service startup sequence and dependencies"""
        logger.info("Testing service startup sequence...")
        
        test_results = {
            "startup_order": [],
            "health_checks": {},
            "startup_times": {},
            "dependency_satisfaction": {}
        }
        
        try:
            # Start services and monitor startup
            start_time = time.time()
            
            process = subprocess.Popen([
                "docker-compose", "-f", str(self.compose_file), "up", "-d"
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, 
               cwd=self.compose_file.parent, text=True)
            
            stdout, stderr = process.communicate()
            
            if process.returncode == 0:
                startup_duration = time.time() - start_time
                test_results["startup_times"]["total_startup"] = startup_duration
                logger.info(f"Services started in {startup_duration:.2f} seconds")
                
                # Wait for services to be ready
                await asyncio.sleep(10)
                
                # Check service health
                services = ["qdrant"]  # Primary service to test
                
                for service in services:
                    health_status = await self.check_service_health(service)
                    test_results["health_checks"][service] = health_status
                    
                test_results["startup_success"] = True
                
            else:
                test_results["startup_success"] = False
                test_results["startup_error"] = stderr
                logger.error(f"Service startup failed: {stderr}")
                
        except Exception as e:
            test_results["startup_success"] = False
            test_results["startup_error"] = str(e)
            logger.error(f"Service startup test failed: {e}")
            
        self.test_results["service_startup"] = test_results
        return test_results
        
    async def check_service_health(self, service_name: str, timeout: int = 60):
        """Check individual service health"""
        logger.info(f"Checking health for service: {service_name}")
        
        health_status = {
            "service": service_name,
            "healthy": False,
            "response_time": None,
            "health_endpoint": None,
            "error": None
        }
        
        # Service-specific health check endpoints
        health_endpoints = {
            "qdrant": "http://localhost:6333/health",
            "workspace-qdrant-mcp": "http://localhost:8000/health"
        }
        
        endpoint = health_endpoints.get(service_name)
        if not endpoint:
            health_status["error"] = f"No health endpoint defined for {service_name}"
            return health_status
            
        health_status["health_endpoint"] = endpoint
        
        start_time = time.time()
        end_time = start_time + timeout
        
        while time.time() < end_time:
            try:
                response = requests.get(endpoint, timeout=5)
                if response.status_code == 200:
                    health_status["healthy"] = True
                    health_status["response_time"] = time.time() - start_time
                    logger.info(f"Service {service_name} is healthy")
                    return health_status
            except requests.RequestException:
                pass
                
            await asyncio.sleep(2)
            
        health_status["error"] = f"Service {service_name} failed to become healthy within {timeout}s"
        logger.error(health_status["error"])
        return health_status
        
    async def test_qdrant_functionality(self):
        """Test Qdrant functionality in container"""
        logger.info("Testing Qdrant functionality...")
        
        test_results = {
            "collection_creation": {},
            "vector_operations": {},
            "search_functionality": {}
        }
        
        qdrant_url = "http://localhost:6333"
        collection_name = "test_collection"
        
        try:
            # Test collection creation
            collection_config = {
                "vectors": {
                    "size": 128,
                    "distance": "Cosine"
                }
            }
            
            response = requests.put(
                f"{qdrant_url}/collections/{collection_name}",
                json=collection_config,
                timeout=10
            )
            
            if response.status_code in [200, 201]:
                test_results["collection_creation"]["success"] = True
                logger.info(f"Collection {collection_name} created successfully")
            else:
                test_results["collection_creation"]["success"] = False
                test_results["collection_creation"]["error"] = f"Status: {response.status_code}"
                logger.error(f"Failed to create collection: {response.status_code}")
                
            # Test vector operations
            test_vector = [0.1] * 128
            point_data = {
                "points": [{
                    "id": 1,
                    "vector": test_vector,
                    "payload": {"test": "data"}
                }]
            }
            
            response = requests.put(
                f"{qdrant_url}/collections/{collection_name}/points",
                json=point_data,
                timeout=10
            )
            
            if response.status_code in [200, 201]:
                test_results["vector_operations"]["upsert_success"] = True
                logger.info("Vector upsert successful")
            else:
                test_results["vector_operations"]["upsert_success"] = False
                test_results["vector_operations"]["upsert_error"] = f"Status: {response.status_code}"
                
            # Test search functionality
            search_data = {
                "vector": test_vector,
                "limit": 5
            }
            
            response = requests.post(
                f"{qdrant_url}/collections/{collection_name}/points/search",
                json=search_data,
                timeout=10
            )
            
            if response.status_code == 200:
                search_results = response.json()
                test_results["search_functionality"]["search_success"] = True
                test_results["search_functionality"]["results_count"] = len(search_results.get("result", []))
                logger.info(f"Search returned {len(search_results.get('result', []))} results")
            else:
                test_results["search_functionality"]["search_success"] = False
                test_results["search_functionality"]["search_error"] = f"Status: {response.status_code}"
                
        except Exception as e:
            test_results["qdrant_test_error"] = str(e)
            logger.error(f"Qdrant functionality test failed: {e}")
            
        self.test_results["qdrant_functionality"] = test_results
        return test_results
        
    async def test_volume_data_persistence(self):
        """Test data persistence across container restarts"""
        logger.info("Testing volume data persistence...")
        
        test_results = {
            "data_before_restart": {},
            "restart_operation": {},
            "data_after_restart": {}
        }
        
        qdrant_url = "http://localhost:6333"
        collection_name = "persistence_test_collection"
        
        try:
            # Create collection and insert test data
            collection_config = {
                "vectors": {
                    "size": 64,
                    "distance": "Cosine"
                }
            }
            
            response = requests.put(
                f"{qdrant_url}/collections/{collection_name}",
                json=collection_config,
                timeout=10
            )
            
            if response.status_code in [200, 201]:
                # Insert test data
                test_points = {
                    "points": [
                        {
                            "id": i,
                            "vector": [0.1 + i * 0.01] * 64,
                            "payload": {"test_id": i, "test_type": "persistence"}
                        } for i in range(10)
                    ]
                }
                
                response = requests.put(
                    f"{qdrant_url}/collections/{collection_name}/points",
                    json=test_points,
                    timeout=10
                )
                
                if response.status_code in [200, 201]:
                    test_results["data_before_restart"]["points_inserted"] = 10
                    logger.info("Test data inserted successfully")
                else:
                    test_results["data_before_restart"]["insert_failed"] = True
                    logger.error("Failed to insert test data")
                    return test_results
                    
            # Restart the containers
            logger.info("Restarting containers to test persistence...")
            
            restart_start = time.time()
            
            # Stop containers
            stop_result = subprocess.run([
                "docker-compose", "-f", str(self.compose_file), "stop"
            ], capture_output=True, cwd=self.compose_file.parent)
            
            if stop_result.returncode == 0:
                logger.info("Containers stopped successfully")
            else:
                test_results["restart_operation"]["stop_failed"] = True
                logger.error("Failed to stop containers")
                return test_results
                
            # Start containers again
            start_result = subprocess.run([
                "docker-compose", "-f", str(self.compose_file), "start"
            ], capture_output=True, cwd=self.compose_file.parent)
            
            if start_result.returncode == 0:
                restart_duration = time.time() - restart_start
                test_results["restart_operation"]["restart_duration"] = restart_duration
                logger.info(f"Containers restarted in {restart_duration:.2f} seconds")
            else:
                test_results["restart_operation"]["start_failed"] = True
                logger.error("Failed to start containers")
                return test_results
                
            # Wait for services to be ready
            await asyncio.sleep(15)
            
            # Check if data persisted
            health_status = await self.check_service_health("qdrant", timeout=30)
            
            if health_status["healthy"]:
                # Query the collection to check if data persisted
                response = requests.get(
                    f"{qdrant_url}/collections/{collection_name}",
                    timeout=10
                )
                
                if response.status_code == 200:
                    collection_info = response.json()
                    points_count = collection_info.get("result", {}).get("points_count", 0)
                    
                    test_results["data_after_restart"]["points_count"] = points_count
                    test_results["data_after_restart"]["data_persisted"] = points_count == 10
                    
                    if points_count == 10:
                        logger.info("Data successfully persisted across restart")
                    else:
                        logger.warning(f"Data persistence issue: expected 10 points, found {points_count}")
                else:
                    test_results["data_after_restart"]["collection_query_failed"] = True
                    logger.error("Failed to query collection after restart")
            else:
                test_results["data_after_restart"]["service_unhealthy"] = True
                logger.error("Qdrant service not healthy after restart")
                
        except Exception as e:
            test_results["persistence_test_error"] = str(e)
            logger.error(f"Volume persistence test failed: {e}")
            
        self.test_results["volume_persistence"] = test_results
        return test_results
        
    async def test_resource_monitoring(self):
        """Test resource usage monitoring"""
        logger.info("Testing resource monitoring...")
        
        test_results = {
            "container_stats": {},
            "resource_usage": {},
            "performance_metrics": {}
        }
        
        try:
            # Get container statistics
            stats_result = subprocess.run([
                "docker", "stats", "--no-stream", "--format", 
                "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.NetIO}}\t{{.BlockIO}}"
            ], capture_output=True, text=True)
            
            if stats_result.returncode == 0:
                test_results["container_stats"]["stats_available"] = True
                test_results["container_stats"]["stats_output"] = stats_result.stdout
                logger.info("Container statistics collected")
            else:
                test_results["container_stats"]["stats_available"] = False
                test_results["container_stats"]["error"] = stats_result.stderr
                
            # Test performance under load
            qdrant_url = "http://localhost:6333"
            
            # Create a larger test collection
            load_collection = "load_test_collection"
            collection_config = {
                "vectors": {
                    "size": 256,
                    "distance": "Cosine"
                }
            }
            
            response = requests.put(
                f"{qdrant_url}/collections/{load_collection}",
                json=collection_config,
                timeout=10
            )
            
            if response.status_code in [200, 201]:
                # Insert many points to create load
                load_start = time.time()
                
                batch_size = 100
                total_points = 1000
                
                for batch_start in range(0, total_points, batch_size):
                    batch_points = {
                        "points": [
                            {
                                "id": i,
                                "vector": [(i % 100) * 0.01] * 256,
                                "payload": {"batch": batch_start // batch_size, "id": i}
                            } for i in range(batch_start, min(batch_start + batch_size, total_points))
                        ]
                    }
                    
                    response = requests.put(
                        f"{qdrant_url}/collections/{load_collection}/points",
                        json=batch_points,
                        timeout=30
                    )
                    
                    if response.status_code not in [200, 201]:
                        logger.warning(f"Failed to insert batch starting at {batch_start}")
                        break
                        
                load_duration = time.time() - load_start
                test_results["performance_metrics"]["load_test_duration"] = load_duration
                test_results["performance_metrics"]["points_per_second"] = total_points / load_duration
                
                logger.info(f"Load test completed: {total_points} points in {load_duration:.2f}s")
                logger.info(f"Rate: {total_points / load_duration:.2f} points/second")
                
        except Exception as e:
            test_results["resource_monitoring_error"] = str(e)
            logger.error(f"Resource monitoring test failed: {e}")
            
        self.test_results["resource_monitoring"] = test_results
        return test_results
        
    def generate_integration_report(self):
        """Generate integration test report"""
        logger.info("Generating containerized integration test report...")
        
        end_time = datetime.now()
        duration = end_time - self.start_time
        
        report = {
            "test_metadata": {
                "test_type": "containerized_integration",
                "start_time": self.start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "duration_seconds": duration.total_seconds(),
                "compose_file": str(self.compose_file)
            },
            "test_results": self.test_results,
            "summary": self._generate_integration_summary()
        }
        
        # Save detailed JSON report
        report_file = self.test_data_dir / f"integration_test_report_{int(time.time())}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
            
        # Generate markdown summary
        self._generate_integration_markdown(report, report_file.with_suffix('.md'))
        
        logger.info(f"Integration test report saved to: {report_file}")
        return report_file
        
    def _generate_integration_summary(self):
        """Generate integration test summary"""
        summary = {
            "total_test_categories": len(self.test_results),
            "passed_categories": 0,
            "failed_categories": 0,
            "category_status": {}
        }
        
        for category, results in self.test_results.items():
            if self._is_integration_category_passed(results):
                summary["passed_categories"] += 1
                summary["category_status"][category] = "PASSED"
            else:
                summary["failed_categories"] += 1
                summary["category_status"][category] = "FAILED"
                
        return summary
        
    def _is_integration_category_passed(self, results):
        """Determine if an integration test category passed"""
        if isinstance(results, dict):
            for key, value in results.items():
                if key.endswith("_error") or key == "error":
                    return False
                if key.endswith("_failed") and value is True:
                    return False
                if isinstance(value, dict):
                    if "success" in value and not value["success"]:
                        return False
                    if "healthy" in value and not value["healthy"]:
                        return False
                        
        return True
        
    def _generate_integration_markdown(self, report, output_path):
        """Generate markdown integration report"""
        with open(output_path, 'w') as f:
            f.write("# Containerized Integration Test Report\n\n")
            
            # Metadata
            f.write("## Test Metadata\n\n")
            metadata = report["test_metadata"]
            f.write(f"- **Test Type**: {metadata['test_type']}\n")
            f.write(f"- **Start Time**: {metadata['start_time']}\n")
            f.write(f"- **End Time**: {metadata['end_time']}\n")
            f.write(f"- **Duration**: {metadata['duration_seconds']:.2f} seconds\n")
            f.write(f"- **Compose File**: {metadata['compose_file']}\n\n")
            
            # Summary
            f.write("## Test Summary\n\n")
            summary = report["summary"]
            f.write(f"- **Total Categories**: {summary['total_test_categories']}\n")
            f.write(f"- **Passed**: {summary['passed_categories']}\n")
            f.write(f"- **Failed**: {summary['failed_categories']}\n\n")
            
            f.write("### Category Results\n\n")
            for category, status in summary["category_status"].items():
                emoji = "✅" if status == "PASSED" else "❌"
                f.write(f"- {emoji} **{category.replace('_', ' ').title()}**: {status}\n")
            f.write("\n")
            
            # Detailed results
            f.write("## Detailed Results\n\n")
            for category, results in report["test_results"].items():
                f.write(f"### {category.replace('_', ' ').title()}\n\n")
                f.write("```json\n")
                f.write(json.dumps(results, indent=2))
                f.write("\n```\n\n")

async def main():
    """Main integration test execution"""
    logger.info("Starting containerized integration testing...")
    
    suite = ContainerizedIntegrationTest()
    
    if not suite.setup_test_environment():
        logger.error("Failed to setup test environment")
        return 1
        
    try:
        # Run integration tests
        await suite.test_service_startup_sequence()
        await suite.test_qdrant_functionality()
        await suite.test_volume_data_persistence()
        await suite.test_resource_monitoring()
        
        # Generate report
        report_file = suite.generate_integration_report()
        
        logger.info("Containerized integration tests completed!")
        logger.info(f"Test report: {report_file}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Integration test execution failed: {e}")
        return 1
        
    finally:
        # Cleanup
        suite.cleanup_test_environment()

if __name__ == "__main__":
    sys.exit(asyncio.run(main()))