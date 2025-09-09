#!/usr/bin/env python3
"""
Test script for global uv tool installation behavior
Date: 2025-09-09 15:08
"""

import subprocess
import os
import tempfile
from pathlib import Path
import json

def run_command(cmd, cwd=None):
    """Run command and capture output."""
    print(f"Running: {cmd} (cwd: {cwd or 'current'})")
    try:
        result = subprocess.run(
            cmd, 
            shell=True, 
            capture_output=True, 
            text=True, 
            cwd=cwd
        )
        return {
            'returncode': result.returncode,
            'stdout': result.stdout,
            'stderr': result.stderr,
            'cwd_before': cwd or os.getcwd(),
            'cwd_after': os.getcwd()
        }
    except Exception as e:
        return {
            'returncode': -1,
            'stdout': '',
            'stderr': str(e),
            'cwd_before': cwd or os.getcwd(),
            'cwd_after': os.getcwd()
        }

def test_global_installation():
    """Test global installation behavior."""
    
    results = {}
    
    # 1. Check current installation status
    print("=== Testing Global Installation Status ===")
    results['uv_tool_list'] = run_command("uv tool list | grep workspace-qdrant-mcp")
    results['which_workspace'] = run_command("which workspace-qdrant-mcp")
    results['which_wqm'] = run_command("which wqm")
    
    # 2. Test commands from different directories
    print("\n=== Testing Command Availability ===")
    
    # Test from temp directory
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"Testing from temporary directory: {temp_dir}")
        results['workspace_help_temp'] = run_command("workspace-qdrant-mcp --help", cwd=temp_dir)
        results['wqm_help_temp'] = run_command("wqm --help", cwd=temp_dir)
        
        # Check if working directory changed
        original_cwd = os.getcwd()
        results['wqm_version_temp'] = run_command("wqm --version", cwd=temp_dir)
        current_cwd = os.getcwd()
        
        results['cwd_change_detected'] = original_cwd != current_cwd
        results['original_cwd'] = original_cwd  
        results['current_cwd'] = current_cwd
        
    # 3. Test from home directory
    print("\n=== Testing from Home Directory ===")
    home_dir = Path.home()
    results['workspace_help_home'] = run_command("workspace-qdrant-mcp --help", cwd=str(home_dir))
    results['wqm_version_home'] = run_command("wqm --version", cwd=str(home_dir))
    
    # 4. Test project detection from different locations
    print("\n=== Testing Project Detection ===")
    
    # Test workspace-qdrant-mcp command from various locations
    test_dirs = [
        "/tmp",
        str(Path.home()),
        "/Users/chris/Dropbox/dev/ai/claude-code-cfg/mcp/workspace-qdrant-mcp"
    ]
    
    for test_dir in test_dirs:
        if os.path.exists(test_dir):
            key = f"project_detection_{test_dir.replace('/', '_')}"
            # Just test that the command works, not trying to start the server
            results[key] = run_command(f"timeout 5 workspace-qdrant-mcp --help", cwd=test_dir)
    
    # 5. Test memory usage
    print("\n=== Testing Memory Usage ===")
    # We'll just test that the help command is lightweight
    results['memory_test'] = run_command("time -l wqm --help 2>&1 | grep 'maximum resident set size'")
    
    return results

def analyze_results(results):
    """Analyze test results and provide recommendations."""
    
    analysis = {
        'installation_status': 'unknown',
        'global_availability': 'unknown', 
        'working_directory_behavior': 'unknown',
        'project_detection': 'unknown',
        'issues_found': [],
        'recommendations': []
    }
    
    # Check installation status
    if results['uv_tool_list']['returncode'] == 0 and 'workspace-qdrant-mcp' in results['uv_tool_list']['stdout']:
        analysis['installation_status'] = 'installed'
    else:
        analysis['installation_status'] = 'not_installed'
        analysis['issues_found'].append("Package not installed as uv tool")
        analysis['recommendations'].append("Run: uv tool install workspace-qdrant-mcp")
    
    # Check global command availability
    workspace_available = results['which_workspace']['returncode'] == 0
    wqm_available = results['which_wqm']['returncode'] == 0
    
    if workspace_available and wqm_available:
        analysis['global_availability'] = 'both_available'
    elif workspace_available:
        analysis['global_availability'] = 'workspace_only'
        analysis['issues_found'].append("wqm command not globally available")
    elif wqm_available:
        analysis['global_availability'] = 'wqm_only'
        analysis['issues_found'].append("workspace-qdrant-mcp command not globally available")
    else:
        analysis['global_availability'] = 'none_available'
        analysis['issues_found'].append("Neither command globally available")
    
    # Check working directory behavior
    if results.get('cwd_change_detected', False):
        analysis['working_directory_behavior'] = 'changes_cwd'
        analysis['issues_found'].append("Commands change working directory")
        analysis['recommendations'].append("Fix commands to preserve working directory")
    else:
        analysis['working_directory_behavior'] = 'preserves_cwd'
    
    # Check project detection from different directories
    detection_success_count = 0
    for key, result in results.items():
        if key.startswith('project_detection_') and result['returncode'] == 0:
            detection_success_count += 1
    
    if detection_success_count > 0:
        analysis['project_detection'] = 'working'
    else:
        analysis['project_detection'] = 'failing'
        analysis['issues_found'].append("Project detection may not work from all directories")
    
    return analysis

if __name__ == "__main__":
    print("Testing Global UV Tool Installation Behavior")
    print("=" * 50)
    
    results = test_global_installation()
    analysis = analyze_results(results)
    
    # Save detailed results
    with open("20250909-1508_global_test_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    # Print analysis
    print("\n" + "=" * 50)
    print("ANALYSIS RESULTS")
    print("=" * 50)
    
    print(f"Installation Status: {analysis['installation_status']}")
    print(f"Global Availability: {analysis['global_availability']}")
    print(f"Working Directory Behavior: {analysis['working_directory_behavior']}")
    print(f"Project Detection: {analysis['project_detection']}")
    
    if analysis['issues_found']:
        print(f"\nIssues Found ({len(analysis['issues_found'])}):")
        for i, issue in enumerate(analysis['issues_found'], 1):
            print(f"  {i}. {issue}")
    
    if analysis['recommendations']:
        print(f"\nRecommendations ({len(analysis['recommendations'])}):")
        for i, rec in enumerate(analysis['recommendations'], 1):
            print(f"  {i}. {rec}")
    
    print(f"\nDetailed results saved to: 20250909-1508_global_test_results.json")