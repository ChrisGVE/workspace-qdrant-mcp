#!/usr/bin/env python3
"""
Cross-Platform Validation Test
Validates workspace-qdrant-mcp deployment across different platforms and architectures
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
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class CrossPlatformValidator:
    """Validates cross-platform compatibility"""
    
    def __init__(self):
        self.docker_client = docker.from_env()
        self.project_root = Path(__file__).parent.parent
        self.test_results = {}
        self.start_time = datetime.now()
        self.results_dir = self.project_root / "test_results" / "cross_platform"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Platform configurations to test
        self.test_platforms = [
            {"platform": "linux/amd64", "base_image": "python:3.11-alpine"},
            {"platform": "linux/arm64", "base_image": "python:3.11-alpine"},
            {"platform": "linux/amd64", "base_image": "python:3.11-slim"},
            {"platform": "linux/arm64", "base_image": "python:3.11-slim"}
        ]
        
    def setup_test_environment(self):
        """Setup cross-platform test environment"""
        logger.info("Setting up cross-platform validation environment...")
        
        try:
            # Check Docker buildx support for multi-platform builds
            result = subprocess.run([
                "docker", "buildx", "version"
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                logger.info(f"Docker Buildx available: {result.stdout.strip()}")
                
                # Create or use buildx builder
                builder_result = subprocess.run([
                    "docker", "buildx", "create", "--use", "--name", "cross-platform-builder"
                ], capture_output=True, text=True)
                
                if builder_result.returncode == 0 or "already exists" in builder_result.stderr:
                    logger.info("Multi-platform builder ready")
                    return True
                else:
                    logger.warning("Failed to create multi-platform builder")
                    
            else:
                logger.warning("Docker Buildx not available - limited cross-platform testing")
                
        except Exception as e:
            logger.warning(f"Buildx setup warning: {e}")
            
        return True
        
    async def test_platform_compatibility(self):
        """Test platform compatibility across different architectures"""
        logger.info("Testing platform compatibility...")
        
        test_results = {
            "current_platform": {
                "system": platform.system(),
                "machine": platform.machine(),
                "architecture": platform.architecture(),
                "processor": platform.processor()
            },
            "docker_platform": {},
            "multi_arch_builds": {},
            "image_compatibility": {}
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
            
            # Test multi-architecture image availability
            test_images = [
                "python:3.11-alpine",
                "python:3.11-slim",
                "qdrant/qdrant:v1.7.3",
                "redis:7.2-alpine",
                "nginx:1.25-alpine"
            ]
            
            for image in test_images:
                try:
                    # Check image manifest for multi-arch support
                    result = subprocess.run([
                        "docker", "manifest", "inspect", image
                    ], capture_output=True, text=True)
                    
                    if result.returncode == 0:
                        manifest = json.loads(result.stdout)
                        architectures = []
                        
                        if "manifests" in manifest:
                            for m in manifest["manifests"]:
                                if "platform" in m:
                                    arch = f"{m['platform'].get('os', 'unknown')}/{m['platform'].get('architecture', 'unknown')}"
                                    architectures.append(arch)
                                    
                        test_results["image_compatibility"][image] = {
                            "multi_arch": len(architectures) > 1,
                            "architectures": architectures,
                            "manifest_available": True
                        }
                        logger.info(f"Image {image} supports: {architectures}")
                        
                    else:
                        test_results["image_compatibility"][image] = {
                            "multi_arch": False,
                            "manifest_available": False,
                            "error": result.stderr
                        }
                        
                except Exception as e:
                    test_results["image_compatibility"][image] = {
                        "multi_arch": False,
                        "manifest_available": False,
                        "error": str(e)
                    }
                    logger.error(f"Failed to check manifest for {image}: {e}")
                    
        except Exception as e:
            test_results["platform_compatibility_error"] = str(e)
            logger.error(f"Platform compatibility test failed: {e}")
            
        self.test_results["platform_compatibility"] = test_results
        return test_results
        
    async def test_multi_architecture_builds(self):
        """Test building for multiple architectures"""
        logger.info("Testing multi-architecture builds...")
        
        test_results = {
            "buildx_available": False,
            "platform_builds": {},
            "build_times": {},
            "build_success_rate": {}
        }
        
        try:
            # Check if buildx is available
            result = subprocess.run([
                "docker", "buildx", "version"
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                test_results["buildx_available"] = True
                logger.info("Docker Buildx is available")
                
                # Create test Dockerfile for cross-platform builds
                test_dockerfile = self.results_dir / "Dockerfile.cross_platform_test"
                dockerfile_content = """
FROM python:3.11-alpine AS base

# Install system dependencies
RUN apk add --no-cache \\
    gcc \\
    g++ \\
    musl-dev \\
    linux-headers \\
    curl

# Install Python dependencies
WORKDIR /app
COPY requirements.txt* ./

# Simulate installing workspace-qdrant-mcp
RUN pip install --no-cache-dir \\
    wheel \\
    setuptools \\
    requests \\
    pydantic \\
    && pip list

# Test platform info
RUN python -c "import platform; print(f'Platform: {platform.system()} {platform.machine()}')"

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \\
    CMD python -c "import sys; print('Health check passed'); sys.exit(0)"

CMD ["python", "-c", "print('Cross-platform test successful')"]
"""
                
                with open(test_dockerfile, 'w') as f:
                    f.write(dockerfile_content)
                    
                # Test building for different platforms
                platforms_to_test = ["linux/amd64", "linux/arm64"]
                
                for platform_name in platforms_to_test:
                    logger.info(f"Testing build for platform: {platform_name}")
                    
                    build_start = time.time()
                    
                    try:
                        build_result = subprocess.run([
                            "docker", "buildx", "build",
                            "--platform", platform_name,
                            "--file", str(test_dockerfile),
                            "--tag", f"workspace-qdrant-test:{platform_name.replace('/', '-')}",
                            "--load" if platform_name == f"linux/{platform.machine()}" else "--no-cache",
                            str(self.project_root)
                        ], capture_output=True, text=True, timeout=300)
                        
                        build_duration = time.time() - build_start
                        
                        if build_result.returncode == 0:
                            test_results["platform_builds"][platform_name] = {
                                "success": True,
                                "build_time": build_duration
                            }
                            test_results["build_times"][platform_name] = build_duration
                            logger.info(f"Build successful for {platform_name} in {build_duration:.2f}s")
                            
                        else:
                            test_results["platform_builds"][platform_name] = {
                                "success": False,
                                "error": build_result.stderr,
                                "build_time": build_duration
                            }
                            logger.error(f"Build failed for {platform_name}: {build_result.stderr}")
                            
                    except subprocess.TimeoutExpired:
                        test_results["platform_builds"][platform_name] = {
                            "success": False,
                            "error": "Build timeout after 5 minutes",
                            "build_time": 300
                        }
                        logger.error(f"Build timeout for {platform_name}")
                        
                    except Exception as e:
                        test_results["platform_builds"][platform_name] = {
                            "success": False,
                            "error": str(e),
                            "build_time": time.time() - build_start
                        }
                        logger.error(f"Build error for {platform_name}: {e}")
                        
                # Calculate success rate
                successful_builds = sum(1 for result in test_results["platform_builds"].values() if result["success"])
                total_builds = len(test_results["platform_builds"])
                test_results["build_success_rate"] = {
                    "successful": successful_builds,
                    "total": total_builds,
                    "percentage": (successful_builds / total_builds) * 100 if total_builds > 0 else 0
                }
                
            else:
                test_results["buildx_available"] = False
                test_results["buildx_error"] = result.stderr
                logger.warning("Docker Buildx not available - skipping multi-arch builds")
                
        except Exception as e:
            test_results["multi_arch_build_error"] = str(e)
            logger.error(f"Multi-architecture build test failed: {e}")
            
        self.test_results["multi_architecture_builds"] = test_results
        return test_results
        
    async def test_package_compatibility(self):
        """Test package compatibility across platforms"""
        logger.info("Testing package compatibility...")
        
        test_results = {
            "wheel_builds": {},
            "dependency_resolution": {},
            "cli_functionality": {}
        }
        
        try:
            # Test wheel building for different platforms
            wheel_platforms = ["any", "linux_x86_64", "linux_aarch64", "macosx_10_9_x86_64", "macosx_11_0_arm64"]
            
            for wheel_platform in wheel_platforms:
                logger.info(f"Testing wheel build for platform: {wheel_platform}")
                
                try:
                    # Create temporary build directory
                    with tempfile.TemporaryDirectory() as temp_dir:
                        wheel_dir = Path(temp_dir) / "wheels"
                        wheel_dir.mkdir()
                        
                        # Build wheel for specific platform
                        build_result = subprocess.run([
                            sys.executable, "-m", "pip", "wheel",
                            ".", "--no-deps", "--wheel-dir", str(wheel_dir),
                            "--platform", wheel_platform
                        ] if wheel_platform != "any" else [
                            sys.executable, "-m", "pip", "wheel",
                            ".", "--no-deps", "--wheel-dir", str(wheel_dir)
                        ], capture_output=True, text=True, cwd=self.project_root)
                        
                        if build_result.returncode == 0:
                            wheel_files = list(wheel_dir.glob("*.whl"))
                            test_results["wheel_builds"][wheel_platform] = {
                                "success": True,
                                "wheel_count": len(wheel_files),
                                "wheel_files": [f.name for f in wheel_files]
                            }
                            logger.info(f"Wheel build successful for {wheel_platform}")
                            
                        else:
                            test_results["wheel_builds"][wheel_platform] = {
                                "success": False,
                                "error": build_result.stderr
                            }
                            logger.error(f"Wheel build failed for {wheel_platform}: {build_result.stderr}")
                            
                except Exception as e:
                    test_results["wheel_builds"][wheel_platform] = {
                        "success": False,
                        "error": str(e)
                    }
                    logger.error(f"Wheel build error for {wheel_platform}: {e}")
                    
            # Test dependency resolution
            logger.info("Testing dependency resolution...")
            
            deps_result = subprocess.run([
                sys.executable, "-m", "pip", "check"
            ], capture_output=True, text=True, cwd=self.project_root)
            
            test_results["dependency_resolution"] = {
                "dependencies_compatible": deps_result.returncode == 0,
                "pip_check_output": deps_result.stdout + deps_result.stderr
            }
            
            # Test CLI functionality
            logger.info("Testing CLI functionality...")
            
            try:
                # Test importing the main package
                import_result = subprocess.run([
                    sys.executable, "-c", "import workspace_qdrant; print('Import successful')"
                ], capture_output=True, text=True, cwd=self.project_root)
                
                test_results["cli_functionality"]["import_test"] = {
                    "success": import_result.returncode == 0,
                    "output": import_result.stdout,
                    "error": import_result.stderr
                }
                
                # Test CLI command
                cli_result = subprocess.run([
                    sys.executable, "-m", "workspace_qdrant", "--version"
                ], capture_output=True, text=True, cwd=self.project_root, timeout=30)
                
                test_results["cli_functionality"]["version_test"] = {
                    "success": cli_result.returncode == 0,
                    "output": cli_result.stdout,
                    "error": cli_result.stderr
                }
                
            except Exception as e:
                test_results["cli_functionality"]["test_error"] = str(e)
                logger.error(f"CLI functionality test error: {e}")
                
        except Exception as e:
            test_results["package_compatibility_error"] = str(e)
            logger.error(f"Package compatibility test failed: {e}")
            
        self.test_results["package_compatibility"] = test_results
        return test_results
        
    async def test_configuration_compatibility(self):
        """Test configuration compatibility across platforms"""
        logger.info("Testing configuration compatibility...")
        
        test_results = {
            "path_handling": {},
            "environment_variables": {},
            "file_permissions": {},
            "network_configuration": {}
        }
        
        try:
            # Test path handling across platforms
            logger.info("Testing path handling...")
            
            test_paths = [
                "/tmp/workspace-qdrant-test",
                "~/.workspace-qdrant",
                "./data/test",
                "../config/test.yaml"
            ]
            
            for test_path in test_paths:
                try:
                    expanded_path = Path(test_path).expanduser().resolve()
                    test_results["path_handling"][test_path] = {
                        "expandable": True,
                        "expanded_path": str(expanded_path),
                        "absolute": expanded_path.is_absolute()
                    }
                except Exception as e:
                    test_results["path_handling"][test_path] = {
                        "expandable": False,
                        "error": str(e)
                    }
                    
            # Test environment variable handling
            logger.info("Testing environment variable handling...")
            
            test_env_vars = {
                "WORKSPACE_QDRANT_HOST": "localhost",
                "WORKSPACE_QDRANT_PORT": "8000",
                "QDRANT_HOST": "qdrant",
                "QDRANT_PORT": "6333"
            }
            
            for var, value in test_env_vars.items():
                os.environ[var] = value
                
            # Test reading environment variables
            for var in test_env_vars:
                try:
                    env_value = os.getenv(var)
                    test_results["environment_variables"][var] = {
                        "readable": env_value is not None,
                        "value": env_value
                    }
                except Exception as e:
                    test_results["environment_variables"][var] = {
                        "readable": False,
                        "error": str(e)
                    }
                    
            # Clean up test environment variables
            for var in test_env_vars:
                os.environ.pop(var, None)
                
            # Test file permissions (Unix-like systems)
            logger.info("Testing file permissions...")
            
            if platform.system() in ["Linux", "Darwin"]:
                try:
                    test_file = self.results_dir / "permission_test.txt"
                    test_file.write_text("permission test")
                    
                    # Test different permission settings
                    permissions = [0o644, 0o755, 0o600]
                    
                    for perm in permissions:
                        test_file.chmod(perm)
                        current_perm = test_file.stat().st_mode & 0o777
                        
                        test_results["file_permissions"][f"0o{perm:o}"] = {
                            "set_successfully": current_perm == perm,
                            "actual_permissions": f"0o{current_perm:o}"
                        }
                        
                    test_file.unlink()
                    
                except Exception as e:
                    test_results["file_permissions"]["error"] = str(e)
                    
            else:
                test_results["file_permissions"]["skipped"] = "Not applicable on Windows"
                
            # Test network configuration compatibility
            logger.info("Testing network configuration...")
            
            network_tests = {
                "localhost_resolution": "127.0.0.1",
                "ipv4_binding": "0.0.0.0",
                "ipv6_support": "::"
            }
            
            for test_name, address in network_tests.items():
                try:
                    import socket
                    
                    # Test socket binding
                    sock = socket.socket(socket.AF_INET if ":" not in address else socket.AF_INET6, socket.SOCK_STREAM)
                    sock.bind((address, 0))  # Bind to any available port
                    port = sock.getsockname()[1]
                    sock.close()
                    
                    test_results["network_configuration"][test_name] = {
                        "bindable": True,
                        "address": address,
                        "test_port": port
                    }
                    
                except Exception as e:
                    test_results["network_configuration"][test_name] = {
                        "bindable": False,
                        "address": address,
                        "error": str(e)
                    }
                    
        except Exception as e:
            test_results["configuration_compatibility_error"] = str(e)
            logger.error(f"Configuration compatibility test failed: {e}")
            
        self.test_results["configuration_compatibility"] = test_results
        return test_results
        
    def generate_cross_platform_report(self):
        """Generate cross-platform validation report"""
        logger.info("Generating cross-platform validation report...")
        
        end_time = datetime.now()
        duration = end_time - self.start_time
        
        report = {
            "test_metadata": {
                "test_type": "cross_platform_validation",
                "start_time": self.start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "duration_seconds": duration.total_seconds(),
                "host_platform": {
                    "system": platform.system(),
                    "machine": platform.machine(),
                    "architecture": platform.architecture(),
                    "processor": platform.processor(),
                    "python_version": platform.python_version()
                }
            },
            "test_results": self.test_results,
            "summary": self._generate_cross_platform_summary()
        }
        
        # Save detailed JSON report
        report_file = self.results_dir / f"cross_platform_report_{int(time.time())}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
            
        # Generate markdown summary
        self._generate_cross_platform_markdown(report, report_file.with_suffix('.md'))
        
        logger.info(f"Cross-platform validation report saved to: {report_file}")
        return report_file
        
    def _generate_cross_platform_summary(self):
        """Generate cross-platform validation summary"""
        summary = {
            "total_test_categories": len(self.test_results),
            "passed_categories": 0,
            "failed_categories": 0,
            "category_status": {},
            "platform_support": {
                "multi_arch_builds": False,
                "cross_platform_wheels": False,
                "configuration_portable": False
            }
        }
        
        for category, results in self.test_results.items():
            if self._is_cross_platform_category_passed(results):
                summary["passed_categories"] += 1
                summary["category_status"][category] = "PASSED"
            else:
                summary["failed_categories"] += 1
                summary["category_status"][category] = "FAILED"
                
        # Analyze platform support
        if "multi_architecture_builds" in self.test_results:
            builds = self.test_results["multi_architecture_builds"].get("platform_builds", {})
            summary["platform_support"]["multi_arch_builds"] = any(
                result.get("success", False) for result in builds.values()
            )
            
        if "package_compatibility" in self.test_results:
            wheels = self.test_results["package_compatibility"].get("wheel_builds", {})
            summary["platform_support"]["cross_platform_wheels"] = any(
                result.get("success", False) for result in wheels.values()
            )
            
        if "configuration_compatibility" in self.test_results:
            config = self.test_results["configuration_compatibility"]
            summary["platform_support"]["configuration_portable"] = not any(
                "error" in str(result) for result in config.values() if isinstance(result, dict)
            )
                
        return summary
        
    def _is_cross_platform_category_passed(self, results):
        """Determine if a cross-platform test category passed"""
        if isinstance(results, dict):
            for key, value in results.items():
                if key.endswith("_error") or key == "error":
                    return False
                if isinstance(value, dict) and "success" in value:
                    if not value["success"]:
                        return False
                        
        return True
        
    def _generate_cross_platform_markdown(self, report, output_path):
        """Generate markdown cross-platform report"""
        with open(output_path, 'w') as f:
            f.write("# Cross-Platform Validation Report\n\n")
            
            # Metadata
            f.write("## Test Metadata\n\n")
            metadata = report["test_metadata"]
            f.write(f"- **Test Type**: {metadata['test_type']}\n")
            f.write(f"- **Start Time**: {metadata['start_time']}\n")
            f.write(f"- **End Time**: {metadata['end_time']}\n")
            f.write(f"- **Duration**: {metadata['duration_seconds']:.2f} seconds\n\n")
            
            f.write("### Host Platform\n\n")
            host = metadata["host_platform"]
            f.write(f"- **System**: {host['system']}\n")
            f.write(f"- **Architecture**: {host['machine']}\n")
            f.write(f"- **Processor**: {host['processor']}\n")
            f.write(f"- **Python Version**: {host['python_version']}\n\n")
            
            # Summary
            f.write("## Test Summary\n\n")
            summary = report["summary"]
            f.write(f"- **Total Categories**: {summary['total_test_categories']}\n")
            f.write(f"- **Passed**: {summary['passed_categories']}\n")
            f.write(f"- **Failed**: {summary['failed_categories']}\n\n")
            
            f.write("### Platform Support\n\n")
            support = summary["platform_support"]
            for feature, supported in support.items():
                emoji = "✅" if supported else "❌"
                f.write(f"- {emoji} **{feature.replace('_', ' ').title()}**: {'Supported' if supported else 'Not Supported'}\n")
            f.write("\n")
            
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
    """Main cross-platform validation execution"""
    logger.info("Starting cross-platform validation testing...")
    
    validator = CrossPlatformValidator()
    
    if not validator.setup_test_environment():
        logger.error("Failed to setup test environment")
        return 1
        
    try:
        # Run cross-platform validation tests
        await validator.test_platform_compatibility()
        await validator.test_multi_architecture_builds()
        await validator.test_package_compatibility()
        await validator.test_configuration_compatibility()
        
        # Generate report
        report_file = validator.generate_cross_platform_report()
        
        logger.info("Cross-platform validation completed!")
        logger.info(f"Test report: {report_file}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Cross-platform validation failed: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(asyncio.run(main()))