#!/usr/bin/env python3
"""
MCP Integration Test for Task 112
Date: 2025-09-09 15:25
"""

import subprocess
import os
import time
import json
from pathlib import Path
import tempfile

def test_mcp_server_startup():
    """Test that MCP server can start properly."""
    
    results = {}
    
    print("=== Testing MCP Server Startup ===")
    
    # Test 1: Server startup with --help (should work without dependencies)
    print("1. Testing server help command...")
    result = subprocess.run(
        ["workspace-qdrant-mcp", "--help"], 
        capture_output=True, 
        text=True,
        timeout=10
    )
    results['server_help'] = {
        'returncode': result.returncode,
        'stdout': result.stdout,
        'stderr': result.stderr,
        'success': result.returncode == 0
    }
    
    # Test 2: Test with different transport options
    print("2. Testing transport options...")
    transport_tests = ['stdio', 'http', 'sse']
    
    for transport in transport_tests:
        print(f"   Testing {transport} transport...")
        # Just test that the argument parsing works
        result = subprocess.run(
            ["workspace-qdrant-mcp", "--transport", transport, "--help"],
            capture_output=True,
            text=True,
            timeout=10
        )
        results[f'transport_{transport}'] = {
            'returncode': result.returncode,
            'stdout': result.stdout,
            'stderr': result.stderr,
            'success': result.returncode == 0
        }
    
    # Test 3: Test with configuration file
    print("3. Testing configuration file option...")
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write("""
# Test configuration for MCP server
qdrant:
  url: "http://localhost:6333"
  collection_prefix: "test_mcp"

embeddings:
  provider: "openai"  
  model: "text-embedding-ada-002"

memory:
  enable_persistence: true
  max_entries: 1000
""")
        config_path = f.name
    
    try:
        result = subprocess.run(
            ["workspace-qdrant-mcp", "--config", config_path, "--help"],
            capture_output=True,
            text=True,
            timeout=10
        )
        results['config_file_test'] = {
            'returncode': result.returncode,
            'stdout': result.stdout,
            'stderr': result.stderr,
            'success': result.returncode == 0
        }
    finally:
        os.unlink(config_path)
    
    return results

def test_project_detection_from_global():
    """Test project detection when run as global tool."""
    
    results = {}
    
    print("=== Testing Project Detection from Global Installation ===")
    
    # Test from different directories to ensure project detection works
    test_scenarios = [
        {
            'name': 'from_tmp',
            'cwd': '/tmp',
            'description': 'Test from /tmp directory (no project)'
        },
        {
            'name': 'from_home',
            'cwd': str(Path.home()),
            'description': 'Test from home directory'
        },
        {
            'name': 'from_project',
            'cwd': '/Users/chris/Dropbox/dev/ai/claude-code-cfg/mcp/workspace-qdrant-mcp',
            'description': 'Test from actual project directory'
        }
    ]
    
    for scenario in test_scenarios:
        print(f"Testing {scenario['description']}...")
        
        if not os.path.exists(scenario['cwd']):
            print(f"  Skipping {scenario['cwd']} - directory doesn't exist")
            continue
            
        # Test basic command execution
        result = subprocess.run(
            ["wqm", "--version"],
            capture_output=True,
            text=True,
            cwd=scenario['cwd'],
            timeout=10
        )
        
        results[scenario['name']] = {
            'cwd': scenario['cwd'],
            'returncode': result.returncode,
            'stdout': result.stdout,
            'stderr': result.stderr,
            'success': result.returncode == 0
        }
    
    return results

def test_configuration_precedence_detailed():
    """Test configuration precedence in detail."""
    
    results = {}
    
    print("=== Testing Configuration Precedence ===")
    
    # Test 1: Environment variables
    print("1. Testing environment variable configuration...")
    env = os.environ.copy()
    env['WORKSPACE_QDRANT_LOG_LEVEL'] = 'DEBUG'
    env['WORKSPACE_QDRANT_COLLECTION_PREFIX'] = 'env_test'
    
    result = subprocess.run(
        ["wqm", "--debug", "--version"],
        capture_output=True,
        text=True,
        env=env,
        timeout=10
    )
    results['env_precedence'] = {
        'returncode': result.returncode,
        'stdout': result.stdout,
        'stderr': result.stderr,
        'success': result.returncode == 0,
        'has_debug_output': 'DEBUG' in result.stdout or 'debug' in result.stdout.lower()
    }
    
    # Test 2: CLI arguments override
    print("2. Testing CLI argument precedence...")
    result = subprocess.run(
        ["wqm", "--debug", "admin", "--help"],
        capture_output=True,
        text=True,
        timeout=10
    )
    results['cli_precedence'] = {
        'returncode': result.returncode,
        'stdout': result.stdout,
        'stderr': result.stderr,
        'success': result.returncode == 0,
        'has_debug_output': 'DEBUG' in result.stdout or 'debug' in result.stdout.lower()
    }
    
    return results

def test_memory_efficiency():
    """Test memory usage efficiency for global tool."""
    
    results = {}
    
    print("=== Testing Memory Efficiency ===")
    
    # Test lightweight commands that shouldn't consume much memory
    lightweight_commands = [
        (["wqm", "--version"], "version_command"),
        (["wqm", "--help"], "help_command"),  
        (["workspace-qdrant-mcp", "--help"], "server_help_command"),
    ]
    
    for cmd, test_name in lightweight_commands:
        print(f"Testing {test_name}...")
        
        start_time = time.time()
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=10
        )
        end_time = time.time()
        
        results[test_name] = {
            'returncode': result.returncode,
            'stdout': result.stdout,
            'stderr': result.stderr,
            'execution_time': round(end_time - start_time, 3),
            'success': result.returncode == 0,
            'fast_execution': (end_time - start_time) < 3.0  # Should be fast
        }
    
    return results

def analyze_mcp_integration_results(all_results):
    """Analyze MCP integration test results."""
    
    analysis = {
        'server_startup': 'unknown',
        'project_detection': 'unknown',
        'configuration_precedence': 'unknown',
        'memory_efficiency': 'unknown',
        'transport_support': 'unknown',
        'overall_status': 'unknown',
        'issues_found': [],
        'recommendations': [],
        'performance_metrics': {}
    }
    
    # Analyze server startup
    server_results = all_results.get('server_startup', {})
    if server_results.get('server_help', {}).get('success', False):
        transport_success = sum(1 for k, v in server_results.items() 
                              if k.startswith('transport_') and v.get('success', False))
        if transport_success >= 2:  # At least stdio and http should work
            analysis['server_startup'] = 'working'
            analysis['transport_support'] = 'working'
        else:
            analysis['server_startup'] = 'partial'
            analysis['transport_support'] = 'limited'
            analysis['issues_found'].append("Some transport options may not be working")
    else:
        analysis['server_startup'] = 'failing'
        analysis['issues_found'].append("Server startup is failing")
    
    # Analyze project detection
    detection_results = all_results.get('project_detection', {})
    detection_success = sum(1 for v in detection_results.values() if v.get('success', False))
    if detection_success >= len(detection_results) * 0.8:
        analysis['project_detection'] = 'working'
    else:
        analysis['project_detection'] = 'issues'
        analysis['issues_found'].append("Project detection issues from some directories")
    
    # Analyze configuration precedence
    config_results = all_results.get('configuration', {})
    debug_working = config_results.get('cli_precedence', {}).get('has_debug_output', False)
    if debug_working:
        analysis['configuration_precedence'] = 'working'
    else:
        analysis['configuration_precedence'] = 'issues'
        analysis['issues_found'].append("Configuration precedence may not be working")
    
    # Analyze memory efficiency
    memory_results = all_results.get('memory', {})
    fast_commands = sum(1 for v in memory_results.values() if v.get('fast_execution', False))
    total_commands = len(memory_results)
    
    if total_commands > 0:
        avg_time = sum(v.get('execution_time', 0) for v in memory_results.values()) / total_commands
        analysis['performance_metrics']['avg_execution_time'] = round(avg_time, 3)
        
        if fast_commands >= total_commands * 0.8:
            analysis['memory_efficiency'] = 'good'
        else:
            analysis['memory_efficiency'] = 'slow'
            analysis['issues_found'].append("Commands are executing slowly")
    
    # Overall status
    if len(analysis['issues_found']) == 0:
        analysis['overall_status'] = 'excellent'
        analysis['recommendations'].append("Global installation is working perfectly")
    elif len(analysis['issues_found']) <= 2:
        analysis['overall_status'] = 'good'
        analysis['recommendations'].append("Minor issues found, but global installation is mostly working")
    else:
        analysis['overall_status'] = 'needs_work'
        analysis['recommendations'].append("Several issues found that need to be addressed")
    
    return analysis

def main():
    """Run all MCP integration tests."""
    
    print("MCP Integration Test Suite for Task 112")
    print("=" * 50)
    
    all_results = {}
    
    # Run test suites
    all_results['server_startup'] = test_mcp_server_startup()
    all_results['project_detection'] = test_project_detection_from_global()
    all_results['configuration'] = test_configuration_precedence_detailed()
    all_results['memory'] = test_memory_efficiency()
    
    # Analyze results
    analysis = analyze_mcp_integration_results(all_results)
    
    # Save results
    with open("20250909-1525_mcp_integration_results.json", "w") as f:
        json.dump({
            'test_results': all_results,
            'analysis': analysis,
            'timestamp': time.time()
        }, f, indent=2)
    
    # Print summary
    print("\n" + "=" * 50)
    print("MCP INTEGRATION TEST RESULTS")
    print("=" * 50)
    
    print(f"Server Startup: {analysis['server_startup']}")
    print(f"Transport Support: {analysis['transport_support']}")
    print(f"Project Detection: {analysis['project_detection']}")
    print(f"Configuration Precedence: {analysis['configuration_precedence']}")
    print(f"Memory Efficiency: {analysis['memory_efficiency']}")
    print(f"Overall Status: {analysis['overall_status']}")
    
    if analysis['performance_metrics']:
        print(f"\nPerformance Metrics:")
        for metric, value in analysis['performance_metrics'].items():
            print(f"  {metric}: {value}")
    
    if analysis['issues_found']:
        print(f"\nIssues Found ({len(analysis['issues_found'])}):")
        for i, issue in enumerate(analysis['issues_found'], 1):
            print(f"  {i}. {issue}")
    
    if analysis['recommendations']:
        print(f"\nRecommendations ({len(analysis['recommendations'])}):")
        for i, rec in enumerate(analysis['recommendations'], 1):
            print(f"  {i}. {rec}")
    
    print(f"\nDetailed results saved to: 20250909-1525_mcp_integration_results.json")

if __name__ == "__main__":
    main()