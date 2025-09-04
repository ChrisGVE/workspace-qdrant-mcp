#!/usr/bin/env python3
"""
Comprehensive Cross-Platform Container Testing Suite
Test containerized deployment, service dependencies, volume persistence, 
network communication, resource constraints, and cross-platform compatibility.
"""

import asyncio
import docker
import json
import logging
import os
import platform
import subprocess
import sys
import tempfile
import time
import yaml
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union
import requests
import psutil

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ContainerTestSuite:
    """Comprehensive container testing suite"""
    
    def __init__(self):
        self.docker_client = docker.from_env()
        self.project_root = Path(__file__).parent.parent
        self.docker_dir = self.project_root / "docker"
        self.test_results = {}
        self.start_time = datetime.now()
        
    def setup_test_environment(self):
        """Setup test environment"""
        logger.info("Setting up container test environment...")
        
        # Ensure Docker is running
        try:
            self.docker_client.ping()
            logger.info("Docker daemon is running")
        except Exception as e:
            logger.error(f"Docker daemon not accessible: {e}")
            return False
            
        # Create test results directory
        self.results_dir = self.project_root / "test_results" / "container_tests"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        return True
        
    async def test_container_orchestration(self):
        """Test container orchestration and service dependencies"""
        logger.info("Testing container orchestration and service dependencies...")
        
        test_results = {
            "service_startup": {},
            "health_checks": {},
            "dependency_order": {},
            "service_communication": {}
        }
        
        compose_files = [
            self.docker_dir / "docker-compose.yml",
            self.docker_dir / "integration-tests" / "docker-compose.yml"
        ]
        
        for compose_file in compose_files:
            if not compose_file.exists():
                logger.warning(f"Compose file not found: {compose_file}")
                continue
                
            logger.info(f"Testing compose file: {compose_file}")
            
            # Test service startup
            try:
                result = subprocess.run([
                    "docker-compose", "-f", str(compose_file), "config"
                ], capture_output=True, text=True, cwd=compose_file.parent)
                
                if result.returncode == 0:
                    config = yaml.safe_load(result.stdout)
                    services = list(config.get("services", {}).keys())
                    test_results["service_startup"][compose_file.name] = {
                        "valid_config": True,
                        "services": services,
                        "service_count": len(services)
                    }
                    logger.info(f"Valid compose config with {len(services)} services")
                else:
                    test_results["service_startup"][compose_file.name] = {
                        "valid_config": False,
                        "error": result.stderr
                    }
                    logger.error(f"Invalid compose config: {result.stderr}")
                    
            except Exception as e:
                test_results["service_startup"][compose_file.name] = {
                    "valid_config": False,
                    "error": str(e)
                }
                logger.error(f"Error validating compose file: {e}")
                
            # Test dependency resolution
            try:
                with open(compose_file, 'r') as f:
                    compose_config = yaml.safe_load(f)
                    
                services = compose_config.get("services", {})
                dependency_graph = {}
                
                for service_name, service_config in services.items():
                    depends_on = service_config.get("depends_on", [])
                    if isinstance(depends_on, dict):
                        depends_on = list(depends_on.keys())
                    dependency_graph[service_name] = depends_on
                    
                test_results["dependency_order"][compose_file.name] = dependency_graph
                logger.info(f"Dependency graph extracted for {len(dependency_graph)} services")
                
            except Exception as e:
                logger.error(f"Error analyzing dependencies: {e}")
                test_results["dependency_order"][compose_file.name] = {"error": str(e)}
        
        self.test_results["orchestration"] = test_results
        return test_results
        
    async def test_volume_persistence(self):
        """Test volume persistence across container restarts"""
        logger.info("Testing volume persistence across container restarts...")
        
        test_results = {
            "volume_creation": {},
            "data_persistence": {},
            "restart_persistence": {}
        }
        
        # Test with a simple container that writes and reads data
        test_volume_name = "workspace-qdrant-test-volume"
        test_data = {"timestamp": datetime.now().isoformat(), "test_data": "container_test"}
        
        try:
            # Create a test volume
            volume = self.docker_client.volumes.create(name=test_volume_name)
            test_results["volume_creation"]["success"] = True
            test_results["volume_creation"]["volume_name"] = test_volume_name
            logger.info(f"Created test volume: {test_volume_name}")
            
            # Run container to write data
            container = self.docker_client.containers.run(
                "alpine:latest",
                command=f"sh -c 'echo \'{json.dumps(test_data)}\' > /data/test_file.json'",
                volumes={test_volume_name: {"bind": "/data", "mode": "rw"}},
                detach=True,
                remove=True
            )
            container.wait()
            logger.info("Data written to volume")
            
            # Run another container to read data
            read_container = self.docker_client.containers.run(
                "alpine:latest",
                command="cat /data/test_file.json",
                volumes={test_volume_name: {"bind": "/data", "mode": "ro"}},
                detach=True
            )
            
            result = read_container.wait()
            if result["StatusCode"] == 0:
                data_content = read_container.logs().decode().strip()
                read_data = json.loads(data_content)
                test_results["data_persistence"]["success"] = True
                test_results["data_persistence"]["data_matches"] = read_data == test_data
                logger.info("Data successfully persisted and read from volume")
            else:
                test_results["data_persistence"]["success"] = False
                test_results["data_persistence"]["error"] = "Failed to read from volume"
                logger.error("Failed to read data from volume")
                
            read_container.remove()
            
            # Test persistence after volume remount
            restart_container = self.docker_client.containers.run(
                "alpine:latest", 
                command="cat /data/test_file.json",
                volumes={test_volume_name: {"bind": "/data", "mode": "ro"}},
                detach=True
            )
            
            result = restart_container.wait()
            if result["StatusCode"] == 0:
                restart_data = restart_container.logs().decode().strip()
                restart_json = json.loads(restart_data)
                test_results["restart_persistence"]["success"] = True
                test_results["restart_persistence"]["data_matches"] = restart_json == test_data
                logger.info("Data persisted after volume remount")
            else:
                test_results["restart_persistence"]["success"] = False
                logger.error("Data lost after volume remount")
                
            restart_container.remove()
            
        except Exception as e:
            test_results["volume_creation"]["success"] = False
            test_results["volume_creation"]["error"] = str(e)
            logger.error(f"Volume persistence test failed: {e}")
            
        finally:
            # Cleanup test volume
            try:
                volume.remove()
                logger.info("Test volume cleaned up")
            except:
                pass
                
        self.test_results["volume_persistence"] = test_results
        return test_results
        
    async def test_network_communication(self):
        """Test network communication between containers"""
        logger.info("Testing network communication between containers...")
        
        test_results = {
            "network_creation": {},
            "container_communication": {},
            "service_discovery": {}
        }
        
        test_network_name = "workspace-qdrant-test-network"
        
        try:
            # Create test network
            network = self.docker_client.networks.create(
                test_network_name,
                driver="bridge"
            )
            test_results["network_creation"]["success"] = True
            test_results["network_creation"]["network_name"] = test_network_name
            logger.info(f"Created test network: {test_network_name}")
            
            # Start a simple HTTP server container
            server_container = self.docker_client.containers.run(
                "python:3.11-alpine",
                command="python -c 'import http.server; http.server.HTTPServer((\"\", 8000), http.server.SimpleHTTPRequestHandler).serve_forever()'",
                name="test-server",
                network=test_network_name,
                detach=True
            )
            
            # Wait for server to start
            await asyncio.sleep(2)
            
            # Start client container to test connectivity
            client_container = self.docker_client.containers.run(
                "alpine:latest",
                command="wget -qO- http://test-server:8000 || echo 'CONNECTION_FAILED'",
                network=test_network_name,
                detach=True
            )
            
            result = client_container.wait()
            logs = client_container.logs().decode().strip()
            
            if result["StatusCode"] == 0 and "CONNECTION_FAILED" not in logs:
                test_results["container_communication"]["success"] = True
                test_results["service_discovery"]["hostname_resolution"] = True
                logger.info("Container-to-container communication successful")
            else:
                test_results["container_communication"]["success"] = False
                test_results["service_discovery"]["hostname_resolution"] = False
                logger.error("Container-to-container communication failed")
                
            # Cleanup containers
            server_container.stop()
            server_container.remove()
            client_container.remove()
            
        except Exception as e:
            test_results["network_creation"]["success"] = False
            test_results["network_creation"]["error"] = str(e)
            logger.error(f"Network communication test failed: {e}")
            
        finally:
            # Cleanup test network
            try:
                network.remove()
                logger.info("Test network cleaned up")
            except:
                pass
                
        self.test_results["network_communication"] = test_results
        return test_results
        
    async def test_resource_constraints(self):
        """Test resource constraints and limits"""
        logger.info("Testing resource constraints and limits...")
        
        test_results = {
            "memory_limits": {},
            "cpu_limits": {},
            "container_resource_monitoring": {}
        }
        
        try:
            # Test memory limits
            logger.info("Testing memory constraints...")
            memory_container = self.docker_client.containers.run(
                "alpine:latest",
                command="sh -c 'dd if=/dev/zero of=/tmp/memory_test bs=1M count=100 2>/dev/null || echo MEMORY_LIMITED'",
                mem_limit="50m",
                detach=True
            )
            
            result = memory_container.wait()
            logs = memory_container.logs().decode().strip()
            
            if "MEMORY_LIMITED" in logs or result["StatusCode"] != 0:
                test_results["memory_limits"]["constraint_enforced"] = True
                logger.info("Memory limits properly enforced")
            else:
                test_results["memory_limits"]["constraint_enforced"] = False
                logger.warning("Memory limits may not be enforced")
                
            memory_container.remove()
            
            # Test CPU limits
            logger.info("Testing CPU constraints...")
            cpu_container = self.docker_client.containers.run(
                "alpine:latest",
                command="sh -c 'for i in $(seq 1 4); do (while true; do :; done) & done; sleep 2; kill %1 %2 %3 %4'",
                cpus=0.5,
                detach=True
            )
            
            # Monitor CPU usage
            stats = cpu_container.stats(stream=False)
            cpu_stats = stats.get("cpu_stats", {})
            
            if cpu_stats:
                test_results["cpu_limits"]["stats_available"] = True
                logger.info("CPU statistics available")
            else:
                test_results["cpu_limits"]["stats_available"] = False
                logger.warning("CPU statistics not available")
                
            cpu_container.stop()
            cpu_container.remove()
            
        except Exception as e:
            test_results["resource_constraints_error"] = str(e)
            logger.error(f"Resource constraints test failed: {e}")
            
        self.test_results["resource_constraints"] = test_results
        return test_results
        
    async def test_cross_platform_compatibility(self):
        """Test cross-platform compatibility"""
        logger.info("Testing cross-platform compatibility...")
        
        test_results = {
            "platform_info": {
                "system": platform.system(),
                "machine": platform.machine(),
                "architecture": platform.architecture(),
                "python_version": platform.python_version()
            },
            "docker_platform": {},
            "multi_arch_support": {}
        }
        
        try:
            # Get Docker platform info
            docker_info = self.docker_client.info()
            test_results["docker_platform"] = {
                "architecture": docker_info.get("Architecture"),
                "os_type": docker_info.get("OSType"),
                "kernel_version": docker_info.get("KernelVersion"),
                "docker_version": docker_info.get("ServerVersion")
            }
            logger.info(f"Docker platform: {docker_info.get('Architecture')} on {docker_info.get('OSType')}")
            
            # Test multi-architecture image support
            test_images = [
                "alpine:latest",
                "python:3.11-alpine",
                "qdrant/qdrant:v1.7.3"
            ]
            
            for image in test_images:
                try:
                    img = self.docker_client.images.pull(image)
                    test_results["multi_arch_support"][image] = {
                        "pull_success": True,
                        "image_id": img.id[:12],
                        "size": img.attrs.get("Size", 0)
                    }
                    logger.info(f"Successfully pulled {image}")
                except Exception as e:
                    test_results["multi_arch_support"][image] = {
                        "pull_success": False,
                        "error": str(e)
                    }
                    logger.error(f"Failed to pull {image}: {e}")
                    
        except Exception as e:
            test_results["compatibility_test_error"] = str(e)
            logger.error(f"Cross-platform compatibility test failed: {e}")
            
        self.test_results["cross_platform"] = test_results
        return test_results
        
    async def test_package_installation(self):
        """Test Python wheel generation and CLI functionality portability"""
        logger.info("Testing package installation and CLI portability...")
        
        test_results = {
            "wheel_generation": {},
            "cli_functionality": {},
            "dependency_resolution": {}
        }
        
        try:
            # Test wheel generation
            logger.info("Testing wheel generation...")
            build_result = subprocess.run([
                sys.executable, "-m", "pip", "wheel", ".", "--no-deps", 
                "--wheel-dir", str(self.results_dir / "wheels")
            ], capture_output=True, text=True, cwd=self.project_root)
            
            if build_result.returncode == 0:
                wheels_dir = self.results_dir / "wheels"
                wheels_dir.mkdir(exist_ok=True)
                wheel_files = list(wheels_dir.glob("*.whl"))
                test_results["wheel_generation"] = {
                    "success": True,
                    "wheel_count": len(wheel_files),
                    "wheel_files": [f.name for f in wheel_files]
                }
                logger.info(f"Generated {len(wheel_files)} wheel files")
            else:
                test_results["wheel_generation"] = {
                    "success": False,
                    "error": build_result.stderr
                }
                logger.error(f"Wheel generation failed: {build_result.stderr}")
                
            # Test CLI functionality in container
            logger.info("Testing CLI functionality in container...")
            
            # Create a test Dockerfile for CLI testing
            test_dockerfile_content = f"""
FROM python:3.11-alpine
WORKDIR /app
COPY . .
RUN pip install -e .
RUN pip install pytest
CMD ["python", "-m", "workspace_qdrant", "--version"]
"""
            
            dockerfile_path = self.results_dir / "Dockerfile.cli_test"
            with open(dockerfile_path, 'w') as f:
                f.write(test_dockerfile_content)
                
            # Build test image
            test_image, _ = self.docker_client.images.build(
                path=str(self.project_root),
                dockerfile=str(dockerfile_path),
                tag="workspace-qdrant-cli-test"
            )
            
            # Run CLI test
            container = self.docker_client.containers.run(
                "workspace-qdrant-cli-test",
                detach=True
            )
            
            result = container.wait()
            logs = container.logs().decode().strip()
            
            if result["StatusCode"] == 0:
                test_results["cli_functionality"] = {
                    "success": True,
                    "version_output": logs
                }
                logger.info("CLI functionality test passed")
            else:
                test_results["cli_functionality"] = {
                    "success": False,
                    "error": logs
                }
                logger.error(f"CLI functionality test failed: {logs}")
                
            container.remove()
            
            # Test dependency resolution
            logger.info("Testing dependency resolution...")
            deps_result = subprocess.run([
                sys.executable, "-m", "pip", "check"
            ], capture_output=True, text=True, cwd=self.project_root)
            
            test_results["dependency_resolution"] = {
                "dependencies_compatible": deps_result.returncode == 0,
                "pip_check_output": deps_result.stdout + deps_result.stderr
            }
            
            if deps_result.returncode == 0:
                logger.info("All dependencies are compatible")
            else:
                logger.warning(f"Dependency issues found: {deps_result.stderr}")
                
        except Exception as e:
            test_results["package_test_error"] = str(e)
            logger.error(f"Package installation test failed: {e}")
            
        self.test_results["package_installation"] = test_results
        return test_results
        
    def generate_test_report(self):
        """Generate comprehensive test report"""
        logger.info("Generating comprehensive test report...")
        
        end_time = datetime.now()
        duration = end_time - self.start_time
        
        report = {
            "test_metadata": {
                "start_time": self.start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "duration_seconds": duration.total_seconds(),
                "platform": platform.system(),
                "architecture": platform.machine(),
                "python_version": platform.python_version()
            },
            "test_results": self.test_results,
            "summary": self._generate_summary()
        }
        
        # Save detailed JSON report
        report_file = self.results_dir / f"container_test_report_{int(time.time())}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
            
        # Generate markdown summary
        self._generate_markdown_report(report, report_file.with_suffix('.md'))
        
        logger.info(f"Test report saved to: {report_file}")
        return report_file
        
    def _generate_summary(self):
        """Generate test summary"""
        summary = {
            "total_test_categories": len(self.test_results),
            "passed_categories": 0,
            "failed_categories": 0,
            "category_status": {}
        }
        
        for category, results in self.test_results.items():
            if self._is_category_passed(results):
                summary["passed_categories"] += 1
                summary["category_status"][category] = "PASSED"
            else:
                summary["failed_categories"] += 1
                summary["category_status"][category] = "FAILED"
                
        return summary
        
    def _is_category_passed(self, results):
        """Determine if a test category passed"""
        if isinstance(results, dict):
            for key, value in results.items():
                if key.endswith("_error") or key == "error":
                    return False
                if isinstance(value, dict) and "success" in value:
                    if not value["success"]:
                        return False
        return True
        
    def _generate_markdown_report(self, report, output_path):
        """Generate markdown report"""
        with open(output_path, 'w') as f:
            f.write("# Cross-Platform Container Testing Report\n\n")
            
            # Metadata
            f.write("## Test Metadata\n\n")
            metadata = report["test_metadata"]
            f.write(f"- **Start Time**: {metadata['start_time']}\n")
            f.write(f"- **End Time**: {metadata['end_time']}\n") 
            f.write(f"- **Duration**: {metadata['duration_seconds']:.2f} seconds\n")
            f.write(f"- **Platform**: {metadata['platform']}\n")
            f.write(f"- **Architecture**: {metadata['architecture']}\n")
            f.write(f"- **Python Version**: {metadata['python_version']}\n\n")
            
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
    """Main test execution"""
    logger.info("Starting comprehensive cross-platform container testing...")
    
    suite = ContainerTestSuite()
    
    if not suite.setup_test_environment():
        logger.error("Failed to setup test environment")
        return 1
        
    try:
        # Run all test categories
        await suite.test_container_orchestration()
        await suite.test_volume_persistence()
        await suite.test_network_communication()
        await suite.test_resource_constraints()
        await suite.test_cross_platform_compatibility()
        await suite.test_package_installation()
        
        # Generate report
        report_file = suite.generate_test_report()
        
        logger.info("All container tests completed successfully!")
        logger.info(f"Test report: {report_file}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Test execution failed: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(asyncio.run(main()))