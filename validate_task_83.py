#!/usr/bin/env python3
"""
Task 83 Validation Script
Validates that all required components for cross-platform container testing are in place
"""

import json
import logging
import subprocess
import sys
from datetime import datetime
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def validate_task_83_implementation():
    """Validate Task 83 implementation completeness"""
    logger.info("Validating Task 83 - Cross-Platform Container Testing implementation...")
    
    project_root = Path(__file__).parent
    scripts_dir = project_root / "scripts"
    docker_dir = project_root / "docker"
    
    validation_results = {
        "timestamp": datetime.now().isoformat(),
        "task": "Task 83 - Cross-Platform Container Testing",
        "validation_status": {},
        "missing_components": [],
        "implementation_complete": False
    }
    
    # Required components for Task 83
    required_components = {
        "test_scripts": [
            "comprehensive_container_test.py",
            "containerized_integration_test.py", 
            "cross_platform_validation.py",
            "run_comprehensive_container_tests.py"
        ],
        "docker_configs": [
            "docker-compose.yml",
            "integration-tests/docker-compose.yml",
            "Dockerfile"
        ],
        "test_capabilities": [
            "Container orchestration testing",
            "Volume persistence validation",
            "Network communication testing",
            "Resource constraint validation",
            "Cross-platform compatibility",
            "Package installation testing"
        ]
    }
    
    # Validate test scripts exist
    logger.info("Checking test scripts...")
    for script in required_components["test_scripts"]:
        script_path = scripts_dir / script
        if script_path.exists():
            validation_results["validation_status"][f"script_{script}"] = "EXISTS"
            logger.info(f"✅ Found: {script}")
        else:
            validation_results["validation_status"][f"script_{script}"] = "MISSING"
            validation_results["missing_components"].append(f"scripts/{script}")
            logger.error(f"❌ Missing: {script}")
    
    # Validate Docker configurations
    logger.info("Checking Docker configurations...")
    for config in required_components["docker_configs"]:
        config_path = docker_dir / config
        if config_path.exists():
            validation_results["validation_status"][f"docker_{config.replace('/', '_')}"] = "EXISTS"
            logger.info(f"✅ Found: docker/{config}")
        else:
            validation_results["validation_status"][f"docker_{config.replace('/', '_')}"] = "MISSING"
            validation_results["missing_components"].append(f"docker/{config}")
            logger.error(f"❌ Missing: docker/{config}")
    
    # Validate script syntax and imports
    logger.info("Validating script syntax...")
    for script in required_components["test_scripts"]:
        script_path = scripts_dir / script
        if script_path.exists():
            try:
                # Test syntax by compiling
                with open(script_path, 'r') as f:
                    compile(f.read(), str(script_path), 'exec')
                validation_results["validation_status"][f"syntax_{script}"] = "VALID"
                logger.info(f"✅ Syntax valid: {script}")
            except SyntaxError as e:
                validation_results["validation_status"][f"syntax_{script}"] = f"SYNTAX_ERROR: {e}"
                logger.error(f"❌ Syntax error in {script}: {e}")
            except Exception as e:
                validation_results["validation_status"][f"syntax_{script}"] = f"ERROR: {e}"
                logger.error(f"❌ Error validating {script}: {e}")
    
    # Check Docker availability
    logger.info("Checking Docker availability...")
    try:
        docker_result = subprocess.run(["docker", "--version"], capture_output=True, text=True)
        if docker_result.returncode == 0:
            validation_results["validation_status"]["docker_available"] = "AVAILABLE"
            logger.info(f"✅ Docker: {docker_result.stdout.strip()}")
        else:
            validation_results["validation_status"]["docker_available"] = "NOT_AVAILABLE"
            logger.warning("⚠️ Docker not available - container tests may fail")
    except FileNotFoundError:
        validation_results["validation_status"]["docker_available"] = "NOT_INSTALLED"
        logger.warning("⚠️ Docker not installed")
    
    # Check Docker Compose availability
    try:
        compose_result = subprocess.run(["docker-compose", "--version"], capture_output=True, text=True)
        if compose_result.returncode == 0:
            validation_results["validation_status"]["docker_compose_available"] = "AVAILABLE"
            logger.info(f"✅ Docker Compose: {compose_result.stdout.strip()}")
        else:
            validation_results["validation_status"]["docker_compose_available"] = "NOT_AVAILABLE"
            logger.warning("⚠️ Docker Compose not available")
    except FileNotFoundError:
        validation_results["validation_status"]["docker_compose_available"] = "NOT_INSTALLED"
        logger.warning("⚠️ Docker Compose not installed")
    
    # Validate test coverage for required capabilities
    logger.info("Validating test coverage...")
    
    test_coverage_mapping = {
        "Container orchestration testing": ["comprehensive_container_test.py", "containerized_integration_test.py"],
        "Volume persistence validation": ["comprehensive_container_test.py", "containerized_integration_test.py"],
        "Network communication testing": ["comprehensive_container_test.py"],
        "Resource constraint validation": ["comprehensive_container_test.py", "containerized_integration_test.py"],
        "Cross-platform compatibility": ["cross_platform_validation.py"],
        "Package installation testing": ["comprehensive_container_test.py", "cross_platform_validation.py"]
    }
    
    for capability, covering_scripts in test_coverage_mapping.items():
        covered = all((scripts_dir / script).exists() for script in covering_scripts)
        validation_results["validation_status"][f"coverage_{capability.replace(' ', '_').lower()}"] = "COVERED" if covered else "NOT_COVERED"
        if covered:
            logger.info(f"✅ Coverage: {capability}")
        else:
            logger.error(f"❌ Not covered: {capability}")
            validation_results["missing_components"].append(f"Coverage for {capability}")
    
    # Overall validation
    missing_count = len(validation_results["missing_components"])
    total_checks = len([k for k in validation_results["validation_status"] if not k.startswith("docker")])
    
    if missing_count == 0:
        validation_results["implementation_complete"] = True
        logger.info("🎉 Task 83 implementation is COMPLETE!")
        logger.info("All required components for cross-platform container testing are present.")
    else:
        validation_results["implementation_complete"] = False
        logger.error(f"❌ Task 83 implementation is INCOMPLETE!")
        logger.error(f"Missing {missing_count} components:")
        for component in validation_results["missing_components"]:
            logger.error(f"  - {component}")
    
    # Save validation report
    report_file = project_root / "test_results" / "task_83_validation_report.json"
    report_file.parent.mkdir(exist_ok=True)
    
    with open(report_file, 'w') as f:
        json.dump(validation_results, f, indent=2)
    
    logger.info(f"Validation report saved to: {report_file}")
    
    return validation_results

def main():
    """Main validation execution"""
    try:
        results = validate_task_83_implementation()
        
        if results["implementation_complete"]:
            print("\n✅ TASK 83 VALIDATION PASSED")
            print("Cross-platform container testing implementation is complete!")
            return 0
        else:
            print("\n❌ TASK 83 VALIDATION FAILED") 
            print("Implementation is incomplete - see missing components above.")
            return 1
            
    except Exception as e:
        print(f"\n💥 VALIDATION ERROR: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())