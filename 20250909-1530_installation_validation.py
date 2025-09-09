#!/usr/bin/env python3
"""
Installation Validation Test for Task 112
Tests the complete installation cycle to ensure it works for new users
Date: 2025-09-09 15:30
"""

import subprocess
import os
import json
import time
from pathlib import Path

def run_command(cmd, timeout=30):
    """Run a command and capture results."""
    print(f"Running: {cmd}")
    try:
        result = subprocess.run(
            cmd, 
            shell=True, 
            capture_output=True, 
            text=True,
            timeout=timeout
        )
        return {
            'returncode': result.returncode,
            'stdout': result.stdout,
            'stderr': result.stderr,
            'success': result.returncode == 0
        }
    except subprocess.TimeoutExpired:
        return {
            'returncode': -1,
            'stdout': '',
            'stderr': 'Command timed out',
            'success': False
        }
    except Exception as e:
        return {
            'returncode': -1,
            'stdout': '',
            'stderr': str(e),
            'success': False
        }

def test_installation_validation():
    """Test the complete installation and functionality."""
    
    results = {}
    
    print("=== Installation Validation Test ===")
    
    # Step 1: Check current installation status
    print("1. Checking current installation status...")
    results['current_status'] = run_command("uv tool list | grep workspace-qdrant-mcp")
    
    # Step 2: Test reinstallation (force)
    print("2. Testing reinstallation (force)...")
    results['reinstall_force'] = run_command("uv tool install --force workspace-qdrant-mcp")
    
    # Step 3: Verify commands are available after reinstall
    print("3. Verifying commands after reinstall...")
    results['which_workspace_after'] = run_command("which workspace-qdrant-mcp")
    results['which_wqm_after'] = run_command("which wqm")
    
    # Step 4: Test basic functionality after reinstall
    print("4. Testing basic functionality...")
    results['workspace_help_after'] = run_command("workspace-qdrant-mcp --help", timeout=15)
    results['wqm_help_after'] = run_command("wqm --help", timeout=15)
    results['wqm_version_after'] = run_command("wqm --version", timeout=10)
    
    # Step 5: Test from different directories
    print("5. Testing from different directories...")
    test_dirs = ['/tmp', str(Path.home())]
    
    for i, test_dir in enumerate(test_dirs):
        if os.path.exists(test_dir):
            print(f"   Testing from {test_dir}...")
            # Change to directory and test
            cmd = f"cd {test_dir} && wqm --version"
            results[f'dir_test_{i}'] = run_command(cmd, timeout=10)
    
    # Step 6: Test with different transport modes
    print("6. Testing transport modes...")
    transports = ['stdio', 'http']
    for transport in transports:
        print(f"   Testing {transport} transport...")
        cmd = f"workspace-qdrant-mcp --transport {transport} --help"
        results[f'transport_{transport}_after'] = run_command(cmd, timeout=15)
    
    # Step 7: Test configuration loading
    print("7. Testing configuration loading...")
    results['config_test'] = run_command("wqm --debug --version", timeout=10)
    
    return results

def test_edge_cases():
    """Test edge cases and error conditions."""
    
    results = {}
    
    print("=== Testing Edge Cases ===")
    
    # Test 1: Invalid arguments
    print("1. Testing invalid arguments...")
    results['invalid_transport'] = run_command("workspace-qdrant-mcp --transport invalid_transport --help", timeout=10)
    results['invalid_port'] = run_command("workspace-qdrant-mcp --port -1 --help", timeout=10)
    results['invalid_config'] = run_command("wqm --config /nonexistent/config.yaml --help", timeout=10)
    
    # Test 2: Help system responsiveness
    print("2. Testing help system...")
    start_time = time.time()
    results['help_speed'] = run_command("wqm --help", timeout=10)
    results['help_speed']['execution_time'] = round(time.time() - start_time, 3)
    
    # Test 3: Version consistency
    print("3. Testing version consistency...")
    results['wqm_version_check'] = run_command("wqm --version", timeout=10)
    
    return results

def analyze_installation_results(all_results):
    """Analyze installation validation results."""
    
    analysis = {
        'installation_success': 'unknown',
        'command_availability': 'unknown',
        'functionality_working': 'unknown',
        'cross_directory_working': 'unknown',
        'transport_modes_working': 'unknown',
        'edge_cases_handled': 'unknown',
        'overall_grade': 'unknown',
        'issues_found': [],
        'recommendations': []
    }
    
    validation_results = all_results.get('validation', {})
    edge_results = all_results.get('edge_cases', {})
    
    # Check installation success
    if validation_results.get('reinstall_force', {}).get('success', False):
        analysis['installation_success'] = 'working'
    else:
        analysis['installation_success'] = 'failing'
        analysis['issues_found'].append("uv tool installation is failing")
    
    # Check command availability
    workspace_available = validation_results.get('which_workspace_after', {}).get('success', False)
    wqm_available = validation_results.get('which_wqm_after', {}).get('success', False)
    
    if workspace_available and wqm_available:
        analysis['command_availability'] = 'both_available'
    elif workspace_available or wqm_available:
        analysis['command_availability'] = 'partial'
        analysis['issues_found'].append("Not all commands are globally available")
    else:
        analysis['command_availability'] = 'none_available'
        analysis['issues_found'].append("No commands are globally available")
    
    # Check basic functionality
    help_working = validation_results.get('wqm_help_after', {}).get('success', False)
    version_working = validation_results.get('wqm_version_after', {}).get('success', False)
    server_help_working = validation_results.get('workspace_help_after', {}).get('success', False)
    
    if help_working and version_working and server_help_working:
        analysis['functionality_working'] = 'working'
    else:
        analysis['functionality_working'] = 'issues'
        analysis['issues_found'].append("Basic functionality not working after installation")
    
    # Check cross-directory functionality
    dir_tests = [k for k in validation_results.keys() if k.startswith('dir_test_')]
    dir_success = sum(1 for k in dir_tests if validation_results[k].get('success', False))
    
    if len(dir_tests) > 0 and dir_success >= len(dir_tests) * 0.8:
        analysis['cross_directory_working'] = 'working'
    else:
        analysis['cross_directory_working'] = 'issues'
        analysis['issues_found'].append("Cross-directory functionality issues")
    
    # Check transport modes
    transport_tests = [k for k in validation_results.keys() if k.startswith('transport_')]
    transport_success = sum(1 for k in transport_tests if validation_results[k].get('success', False))
    
    if len(transport_tests) > 0 and transport_success >= len(transport_tests) * 0.8:
        analysis['transport_modes_working'] = 'working'
    else:
        analysis['transport_modes_working'] = 'issues'
        analysis['issues_found'].append("Transport modes not working correctly")
    
    # Check edge cases (should handle gracefully)
    invalid_transport = edge_results.get('invalid_transport', {}).get('returncode', 0)
    invalid_port = edge_results.get('invalid_port', {}).get('returncode', 0)
    
    # These should return non-zero (error) but not crash
    if invalid_transport != 0 and invalid_port != 0:
        analysis['edge_cases_handled'] = 'good'
    else:
        analysis['edge_cases_handled'] = 'issues'
        analysis['issues_found'].append("Edge cases not handled properly")
    
    # Overall grade
    if len(analysis['issues_found']) == 0:
        analysis['overall_grade'] = 'A+'
        analysis['recommendations'].append("Perfect! Global installation is working flawlessly")
    elif len(analysis['issues_found']) <= 1:
        analysis['overall_grade'] = 'A-'
        analysis['recommendations'].append("Excellent! Minor issues that don't affect core functionality")
    elif len(analysis['issues_found']) <= 2:
        analysis['overall_grade'] = 'B+'
        analysis['recommendations'].append("Good! Some issues need attention but overall working well")
    else:
        analysis['overall_grade'] = 'C'
        analysis['recommendations'].append("Needs improvement! Several issues affecting functionality")
    
    return analysis

def main():
    """Run complete installation validation."""
    
    print("Installation Validation Suite for Task 112")
    print("=" * 60)
    
    all_results = {}
    
    # Run validation tests
    all_results['validation'] = test_installation_validation()
    all_results['edge_cases'] = test_edge_cases()
    
    # Analyze results
    analysis = analyze_installation_results(all_results)
    
    # Save results
    with open("20250909-1530_installation_validation_results.json", "w") as f:
        json.dump({
            'test_results': all_results,
            'analysis': analysis,
            'timestamp': time.time()
        }, f, indent=2)
    
    # Print summary
    print("\n" + "=" * 60)
    print("INSTALLATION VALIDATION RESULTS")
    print("=" * 60)
    
    print(f"Installation Success: {analysis['installation_success']}")
    print(f"Command Availability: {analysis['command_availability']}")
    print(f"Basic Functionality: {analysis['functionality_working']}")
    print(f"Cross-Directory Support: {analysis['cross_directory_working']}")
    print(f"Transport Modes: {analysis['transport_modes_working']}")
    print(f"Edge Case Handling: {analysis['edge_cases_handled']}")
    print(f"\nOVERALL GRADE: {analysis['overall_grade']}")
    
    if analysis['issues_found']:
        print(f"\nIssues Found ({len(analysis['issues_found'])}):")
        for i, issue in enumerate(analysis['issues_found'], 1):
            print(f"  {i}. {issue}")
    
    if analysis['recommendations']:
        print(f"\nRecommendations ({len(analysis['recommendations'])}):")
        for i, rec in enumerate(analysis['recommendations'], 1):
            print(f"  {i}. {rec}")
    
    print(f"\nDetailed results saved to: 20250909-1530_installation_validation_results.json")

if __name__ == "__main__":
    main()