#!/usr/bin/env python3
"""
Configuration System Test Execution Script
Executes unit tests for the configuration system and reports results.
"""

import sys
import os
import subprocess
from pathlib import Path
import json
from datetime import datetime

def run_command(cmd, cwd=None):
    """Run a command and return result details."""
    try:
        result = subprocess.run(
            cmd, 
            shell=True, 
            cwd=cwd,
            capture_output=True, 
            text=True, 
            timeout=60
        )
        return {
            'success': result.returncode == 0,
            'returncode': result.returncode,
            'stdout': result.stdout,
            'stderr': result.stderr,
            'command': cmd
        }
    except subprocess.TimeoutExpired:
        return {
            'success': False,
            'returncode': -1,
            'stdout': '',
            'stderr': 'Command timed out after 60 seconds',
            'command': cmd
        }
    except Exception as e:
        return {
            'success': False,
            'returncode': -2,
            'stdout': '',
            'stderr': str(e),
            'command': cmd
        }

def check_dependencies():
    """Check if required dependencies are available."""
    print("Checking dependencies...")
    
    # Check Python packages
    packages = ['pytest', 'toml']
    missing = []
    
    for package in packages:
        result = run_command(f"{sys.executable} -c 'import {package}; print(f\"{package} OK\")'")
        if result['success']:
            print(f"✓ {package} available")
        else:
            print(f"❌ {package} missing")
            missing.append(package)
    
    return missing

def run_import_test():
    """Test if our configuration module can be imported."""
    print("\nTesting configuration module imports...")
    
    import_test_code = '''
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path.cwd()))

try:
    from core.config import (
        resolve_config_directory,
        McpConfig,
        DaemonConfig,
        validate_config
    )
    
    # Test basic functionality
    config_dir = resolve_config_directory()
    mcp_config = McpConfig()
    daemon_config = DaemonConfig()
    
    errors = validate_config(mcp_config)
    if not errors:
        print("SUCCESS: All imports and basic functions work")
    else:
        print(f"VALIDATION_ERROR: {errors}")
        
except Exception as e:
    print(f"IMPORT_ERROR: {e}")
    import traceback
    traceback.print_exc()
'''
    
    result = run_command(f'{sys.executable} -c "{import_test_code}"')
    
    if result['success'] and 'SUCCESS' in result['stdout']:
        print("✓ Configuration module imports successfully")
        return True
    else:
        print("❌ Configuration module import failed")
        print(f"STDOUT: {result['stdout']}")
        print(f"STDERR: {result['stderr']}")
        return False

def run_pytest_tests():
    """Run the actual pytest tests."""
    print("\nRunning pytest test suite...")
    
    project_dir = Path(__file__).parent
    
    # Command to run tests
    pytest_cmd = f"{sys.executable} -m pytest tests/test_config.py -v --tb=short --no-header -x"
    
    result = run_command(pytest_cmd, cwd=project_dir)
    
    print(f"Command executed: {pytest_cmd}")
    print(f"Return code: {result['returncode']}")
    print("\nSTDOUT:")
    print(result['stdout'])
    
    if result['stderr']:
        print("\nSTDERR:")
        print(result['stderr'])
    
    return result

def generate_report(test_result):
    """Generate a test execution report."""
    report = {
        "timestamp": datetime.now().isoformat(),
        "test_execution": {
            "success": test_result['success'],
            "return_code": test_result['returncode'],
            "command": test_result['command']
        },
        "output": {
            "stdout": test_result['stdout'],
            "stderr": test_result['stderr']
        }
    }
    
    # Save report to file
    report_file = Path(__file__).parent / "20250108-0240_config_test_report.json"
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\nTest report saved to: {report_file}")
    return report

def main():
    """Main execution function."""
    print("Configuration System Test Execution")
    print("=" * 60)
    print(f"Execution time: {datetime.now()}")
    print(f"Working directory: {Path.cwd()}")
    print(f"Python executable: {sys.executable}")
    
    # Step 1: Check dependencies
    missing_deps = check_dependencies()
    if missing_deps:
        print(f"\n❌ Missing dependencies: {missing_deps}")
        print("Please install with: pip install " + " ".join(missing_deps))
        return 1
    
    # Step 2: Test imports
    if not run_import_test():
        print("\n❌ Import test failed. Cannot proceed with pytest tests.")
        return 1
    
    # Step 3: Run actual tests
    test_result = run_pytest_tests()
    
    # Step 4: Generate report
    report = generate_report(test_result)
    
    # Step 5: Summary
    print("\n" + "=" * 60)
    if test_result['success']:
        print("🎉 ALL CONFIGURATION TESTS PASSED!")
        
        # Count test results from stdout
        stdout = test_result['stdout']
        if 'passed' in stdout:
            print(f"✓ Test results: {stdout.split('passed')[0].strip().split()[-1]} tests passed")
        
        print("✓ Configuration system validation completed successfully")
        return 0
    else:
        print("❌ CONFIGURATION TESTS FAILED!")
        print(f"Return code: {test_result['returncode']}")
        return test_result['returncode']

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)