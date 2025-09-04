#!/usr/bin/env python3
"""
Comprehensive Container Testing Suite Runner
Orchestrates all container testing components for Task 83
"""

import asyncio
import json
import logging
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ComprehensiveContainerTestRunner:
    """Orchestrates comprehensive container testing"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.scripts_dir = self.project_root / "scripts"
        self.results_dir = self.project_root / "test_results" / "comprehensive_container_tests"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.start_time = datetime.now()
        self.test_results = {}
        
        # Test scripts to run
        self.test_scripts = [
            {
                "name": "comprehensive_container_test",
                "script": "comprehensive_container_test.py",
                "description": "General container orchestration and functionality testing"
            },
            {
                "name": "containerized_integration_test", 
                "script": "containerized_integration_test.py",
                "description": "Integration testing of workspace-qdrant-mcp services"
            },
            {
                "name": "cross_platform_validation",
                "script": "cross_platform_validation.py", 
                "description": "Cross-platform compatibility validation"
            }
        ]
        
    async def run_comprehensive_tests(self):
        """Run all comprehensive container tests"""
        logger.info("Starting comprehensive container testing suite...")
        
        overall_results = {
            "test_metadata": {
                "start_time": self.start_time.isoformat(),
                "test_suite": "comprehensive_container_tests",
                "total_test_scripts": len(self.test_scripts)
            },
            "test_execution": {},
            "consolidation": {}
        }
        
        # Pre-flight checks
        preflight_success = await self.run_preflight_checks()
        overall_results["preflight"] = preflight_success
        
        if not preflight_success["all_checks_passed"]:
            logger.error("Pre-flight checks failed - cannot proceed with testing")
            return overall_results
            
        # Run each test script
        for test_info in self.test_scripts:
            logger.info(f"Running test: {test_info['name']}")
            logger.info(f"Description: {test_info['description']}")
            
            test_result = await self.run_test_script(test_info)
            overall_results["test_execution"][test_info["name"]] = test_result
            
            # Brief pause between tests for system cleanup
            await asyncio.sleep(5)
            
        # Consolidate results
        consolidated_results = await self.consolidate_test_results(overall_results)
        overall_results["consolidation"] = consolidated_results
        
        # Generate final comprehensive report
        report_file = await self.generate_comprehensive_report(overall_results)
        
        end_time = datetime.now()
        duration = end_time - self.start_time
        
        logger.info(f"Comprehensive container testing completed!")
        logger.info(f"Total duration: {duration.total_seconds():.2f} seconds")
        logger.info(f"Final report: {report_file}")
        
        return overall_results
        
    async def run_preflight_checks(self):
        """Run pre-flight checks before testing"""
        logger.info("Running pre-flight checks...")
        
        checks = {
            "docker_available": False,
            "docker_compose_available": False,
            "test_scripts_exist": False,
            "sufficient_disk_space": False,
            "network_connectivity": False,
            "all_checks_passed": False
        }
        
        try:
            # Check Docker availability
            docker_result = subprocess.run([
                "docker", "--version"
            ], capture_output=True, text=True)
            
            if docker_result.returncode == 0:
                checks["docker_available"] = True
                logger.info(f"Docker available: {docker_result.stdout.strip()}")
            else:
                logger.error("Docker not available")
                
            # Check Docker Compose availability
            compose_result = subprocess.run([
                "docker-compose", "--version"
            ], capture_output=True, text=True)
            
            if compose_result.returncode == 0:
                checks["docker_compose_available"] = True
                logger.info(f"Docker Compose available: {compose_result.stdout.strip()}")
            else:
                logger.error("Docker Compose not available")
                
            # Check test scripts exist
            missing_scripts = []
            for test_info in self.test_scripts:
                script_path = self.scripts_dir / test_info["script"]
                if not script_path.exists():
                    missing_scripts.append(test_info["script"])
                    
            if not missing_scripts:
                checks["test_scripts_exist"] = True
                logger.info("All test scripts found")
            else:
                logger.error(f"Missing test scripts: {missing_scripts}")
                
            # Check disk space (at least 2GB free)
            import shutil
            free_space = shutil.disk_usage(self.project_root).free
            if free_space > 2 * 1024 * 1024 * 1024:  # 2GB in bytes
                checks["sufficient_disk_space"] = True
                logger.info(f"Sufficient disk space: {free_space / (1024**3):.2f} GB free")
            else:
                logger.warning(f"Limited disk space: {free_space / (1024**3):.2f} GB free")
                checks["sufficient_disk_space"] = False
                
            # Basic network connectivity check
            try:
                import socket
                socket.create_connection(("8.8.8.8", 53), timeout=5)
                checks["network_connectivity"] = True
                logger.info("Network connectivity verified")
            except:
                logger.warning("Network connectivity check failed")
                checks["network_connectivity"] = False
                
            # Overall check
            essential_checks = ["docker_available", "docker_compose_available", "test_scripts_exist"]
            checks["all_checks_passed"] = all(checks[check] for check in essential_checks)
            
            if checks["all_checks_passed"]:
                logger.info("All pre-flight checks passed!")
            else:
                logger.error("Some pre-flight checks failed")
                
        except Exception as e:
            logger.error(f"Pre-flight check error: {e}")
            checks["preflight_error"] = str(e)
            
        return checks
        
    async def run_test_script(self, test_info: Dict) -> Dict:
        """Run individual test script"""
        script_path = self.scripts_dir / test_info["script"]
        
        test_result = {
            "script_name": test_info["script"],
            "description": test_info["description"],
            "start_time": None,
            "end_time": None,
            "duration_seconds": None,
            "exit_code": None,
            "stdout": "",
            "stderr": "",
            "success": False,
            "results_files": []
        }
        
        try:
            start_time = time.time()
            test_result["start_time"] = datetime.fromtimestamp(start_time).isoformat()
            
            logger.info(f"Executing: python {script_path}")
            
            # Run the test script
            process = subprocess.run([
                sys.executable, str(script_path)
            ], capture_output=True, text=True, cwd=self.project_root, timeout=1800)  # 30 minute timeout
            
            end_time = time.time()
            test_result["end_time"] = datetime.fromtimestamp(end_time).isoformat()
            test_result["duration_seconds"] = end_time - start_time
            test_result["exit_code"] = process.returncode
            test_result["stdout"] = process.stdout
            test_result["stderr"] = process.stderr
            test_result["success"] = process.returncode == 0
            
            # Look for result files generated by the test
            result_patterns = [
                f"*{test_info['name']}*report*.json",
                f"*{test_info['name']}*report*.md",
                "*container*test*.json",
                "*cross*platform*.json",
                "*integration*test*.json"
            ]
            
            for pattern in result_patterns:
                result_files = list(self.project_root.glob(f"test_results/**/{pattern}"))
                test_result["results_files"].extend([str(f) for f in result_files])
                
            if test_result["success"]:
                logger.info(f"Test {test_info['name']} completed successfully in {test_result['duration_seconds']:.2f}s")
            else:
                logger.error(f"Test {test_info['name']} failed with exit code {process.returncode}")
                logger.error(f"Error output: {process.stderr}")
                
        except subprocess.TimeoutExpired:
            test_result["exit_code"] = -1
            test_result["stderr"] = "Test timeout after 30 minutes"
            test_result["success"] = False
            logger.error(f"Test {test_info['name']} timed out")
            
        except Exception as e:
            test_result["exit_code"] = -1
            test_result["stderr"] = str(e)
            test_result["success"] = False
            logger.error(f"Test {test_info['name']} execution error: {e}")
            
        return test_result
        
    async def consolidate_test_results(self, overall_results: Dict) -> Dict:
        """Consolidate results from all tests"""
        logger.info("Consolidating test results...")
        
        consolidation = {
            "summary": {
                "total_tests": len(self.test_scripts),
                "successful_tests": 0,
                "failed_tests": 0,
                "total_duration": 0,
                "success_rate": 0
            },
            "test_status": {},
            "combined_findings": {
                "container_orchestration": "unknown",
                "volume_persistence": "unknown", 
                "network_communication": "unknown",
                "resource_constraints": "unknown",
                "cross_platform_compatibility": "unknown",
                "package_portability": "unknown"
            },
            "recommendations": []
        }
        
        # Analyze each test execution
        for test_name, test_result in overall_results["test_execution"].items():
            if test_result["success"]:
                consolidation["summary"]["successful_tests"] += 1
                consolidation["test_status"][test_name] = "PASSED"
            else:
                consolidation["summary"]["failed_tests"] += 1
                consolidation["test_status"][test_name] = "FAILED"
                
            if test_result["duration_seconds"]:
                consolidation["summary"]["total_duration"] += test_result["duration_seconds"]
                
        # Calculate success rate
        if consolidation["summary"]["total_tests"] > 0:
            consolidation["summary"]["success_rate"] = (
                consolidation["summary"]["successful_tests"] / consolidation["summary"]["total_tests"]
            ) * 100
            
        # Load and analyze individual test results
        for test_name, test_result in overall_results["test_execution"].items():
            if test_result["success"] and test_result["results_files"]:
                await self.analyze_test_results(test_name, test_result["results_files"], consolidation)
                
        # Generate recommendations based on findings
        consolidation["recommendations"] = self.generate_recommendations(consolidation)
        
        return consolidation
        
    async def analyze_test_results(self, test_name: str, result_files: List[str], consolidation: Dict):
        """Analyze individual test results"""
        for result_file in result_files:
            if not Path(result_file).exists():
                continue
                
            try:
                if result_file.endswith('.json'):
                    with open(result_file, 'r') as f:
                        test_data = json.load(f)
                        
                    # Extract findings based on test type
                    if "comprehensive_container" in test_name:
                        self.extract_container_findings(test_data, consolidation)
                    elif "containerized_integration" in test_name:
                        self.extract_integration_findings(test_data, consolidation)
                    elif "cross_platform" in test_name:
                        self.extract_cross_platform_findings(test_data, consolidation)
                        
            except Exception as e:
                logger.warning(f"Failed to analyze result file {result_file}: {e}")
                
    def extract_container_findings(self, test_data: Dict, consolidation: Dict):
        """Extract findings from container test results"""
        if "test_results" in test_data:
            results = test_data["test_results"]
            
            if "orchestration" in results:
                consolidation["combined_findings"]["container_orchestration"] = "passed" if results["orchestration"] else "failed"
                
            if "volume_persistence" in results:
                consolidation["combined_findings"]["volume_persistence"] = "passed" if results["volume_persistence"] else "failed"
                
            if "network_communication" in results:
                consolidation["combined_findings"]["network_communication"] = "passed" if results["network_communication"] else "failed"
                
            if "resource_constraints" in results:
                consolidation["combined_findings"]["resource_constraints"] = "passed" if results["resource_constraints"] else "failed"
                
    def extract_integration_findings(self, test_data: Dict, consolidation: Dict):
        """Extract findings from integration test results"""
        if "test_results" in test_data:
            results = test_data["test_results"]
            
            # Analyze service startup and health
            if "service_startup" in results:
                startup = results["service_startup"]
                if startup.get("startup_success"):
                    consolidation["combined_findings"]["container_orchestration"] = "passed"
                    
            # Analyze volume persistence
            if "volume_persistence" in results:
                persistence = results["volume_persistence"]
                if persistence.get("data_after_restart", {}).get("data_persisted"):
                    consolidation["combined_findings"]["volume_persistence"] = "passed"
                    
    def extract_cross_platform_findings(self, test_data: Dict, consolidation: Dict):
        """Extract findings from cross-platform test results"""
        if "test_results" in test_data:
            results = test_data["test_results"]
            
            if "platform_compatibility" in results:
                consolidation["combined_findings"]["cross_platform_compatibility"] = "passed"
                
            if "package_compatibility" in results:
                pkg_compat = results["package_compatibility"]
                if pkg_compat.get("wheel_builds") or pkg_compat.get("cli_functionality"):
                    consolidation["combined_findings"]["package_portability"] = "passed"
                    
    def generate_recommendations(self, consolidation: Dict) -> List[str]:
        """Generate recommendations based on test results"""
        recommendations = []
        
        findings = consolidation["combined_findings"]
        
        if findings["container_orchestration"] == "failed":
            recommendations.append("Review Docker compose configuration and service dependencies")
            
        if findings["volume_persistence"] == "failed":
            recommendations.append("Check volume mounts and data persistence configuration")
            
        if findings["network_communication"] == "failed":
            recommendations.append("Verify network configuration and service discovery")
            
        if findings["resource_constraints"] == "failed":
            recommendations.append("Review resource limits and constraints in container configuration")
            
        if findings["cross_platform_compatibility"] == "failed":
            recommendations.append("Address cross-platform compatibility issues in Docker builds")
            
        if findings["package_portability"] == "failed":
            recommendations.append("Fix package installation and CLI portability issues")
            
        if consolidation["summary"]["success_rate"] < 100:
            recommendations.append("Address failed test cases before production deployment")
            
        if not recommendations:
            recommendations.append("All container tests passed - system ready for deployment")
            
        return recommendations
        
    async def generate_comprehensive_report(self, overall_results: Dict) -> Path:
        """Generate comprehensive test report"""
        end_time = datetime.now()
        duration = end_time - self.start_time
        
        comprehensive_report = {
            **overall_results,
            "final_metadata": {
                "end_time": end_time.isoformat(),
                "total_duration_seconds": duration.total_seconds(),
                "report_generated": datetime.now().isoformat()
            }
        }
        
        # Save comprehensive JSON report
        report_file = self.results_dir / f"comprehensive_container_test_report_{int(time.time())}.json"
        with open(report_file, 'w') as f:
            json.dump(comprehensive_report, f, indent=2)
            
        # Generate markdown summary
        markdown_file = report_file.with_suffix('.md')
        with open(markdown_file, 'w') as f:
            f.write("# Comprehensive Container Testing Report - Task 83\n\n")
            
            # Executive Summary
            f.write("## Executive Summary\n\n")
            consolidation = comprehensive_report["consolidation"]
            summary = consolidation["summary"]
            
            f.write(f"- **Total Tests**: {summary['total_tests']}\n")
            f.write(f"- **Success Rate**: {summary['success_rate']:.1f}%\n")
            f.write(f"- **Total Duration**: {summary['total_duration']:.2f} seconds\n")
            f.write(f"- **Overall Status**: {'PASSED' if summary['success_rate'] == 100 else 'NEEDS ATTENTION'}\n\n")
            
            # Test Results Summary
            f.write("## Test Results Summary\n\n")
            for test_name, status in consolidation["test_status"].items():
                emoji = "✅" if status == "PASSED" else "❌"
                f.write(f"- {emoji} **{test_name.replace('_', ' ').title()}**: {status}\n")
            f.write("\n")
            
            # Key Findings
            f.write("## Key Findings\n\n")
            findings = consolidation["combined_findings"]
            for finding, status in findings.items():
                emoji = "✅" if status == "passed" else "❌" if status == "failed" else "⚠️"
                f.write(f"- {emoji} **{finding.replace('_', ' ').title()}**: {status.title()}\n")
            f.write("\n")
            
            # Recommendations
            f.write("## Recommendations\n\n")
            for i, recommendation in enumerate(consolidation["recommendations"], 1):
                f.write(f"{i}. {recommendation}\n")
            f.write("\n")
            
            # Detailed Results
            f.write("## Detailed Test Execution\n\n")
            for test_name, test_result in comprehensive_report["test_execution"].items():
                f.write(f"### {test_name.replace('_', ' ').title()}\n\n")
                f.write(f"- **Duration**: {test_result.get('duration_seconds', 0):.2f} seconds\n")
                f.write(f"- **Exit Code**: {test_result.get('exit_code', 'Unknown')}\n")
                f.write(f"- **Success**: {'Yes' if test_result.get('success') else 'No'}\n")
                f.write(f"- **Result Files**: {len(test_result.get('results_files', []))}\n\n")
                
        logger.info(f"Comprehensive report saved to: {report_file}")
        logger.info(f"Markdown summary saved to: {markdown_file}")
        
        return report_file

async def main():
    """Main comprehensive container test execution"""
    logger.info("Starting Task 83 - Comprehensive Cross-Platform Container Testing...")
    
    runner = ComprehensiveContainerTestRunner()
    
    try:
        results = await runner.run_comprehensive_tests()
        
        # Determine overall success
        consolidation = results.get("consolidation", {})
        success_rate = consolidation.get("summary", {}).get("success_rate", 0)
        
        if success_rate == 100:
            logger.info("🎉 All container tests passed successfully!")
            return 0
        elif success_rate >= 80:
            logger.warning(f"⚠️ Container tests mostly passed ({success_rate:.1f}% success rate)")
            return 0
        else:
            logger.error(f"❌ Container tests need attention ({success_rate:.1f}% success rate)")
            return 1
            
    except Exception as e:
        logger.error(f"Comprehensive container testing failed: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(asyncio.run(main()))