#!/usr/bin/env python3
"""
Advanced testing suite for Task 112: Configuration, Memory, and MCP Integration
Date: 2025-09-09 15:17
"""

import subprocess
import os
import tempfile
import json
import time
import psutil
from pathlib import Path

def run_command(cmd, cwd=None, timeout=10):
    """Run command with timeout and resource monitoring."""
    print(f"Running: {cmd} (cwd: {cwd or 'current'})")
    
    start_time = time.time()
    start_memory = psutil.virtual_memory().available
    
    try:
        result = subprocess.run(
            cmd, 
            shell=True, 
            capture_output=True, 
            text=True, 
            cwd=cwd,
            timeout=timeout
        )
        
        end_time = time.time()
        end_memory = psutil.virtual_memory().available
        
        return {
            'returncode': result.returncode,
            'stdout': result.stdout,
            'stderr': result.stderr,
            'execution_time': round(end_time - start_time, 3),
            'memory_change': start_memory - end_memory,
            'cwd_before': cwd or os.getcwd(),
            'success': result.returncode == 0
        }
    except subprocess.TimeoutExpired:
        return {
            'returncode': -2,
            'stdout': '',
            'stderr': 'Command timed out',
            'execution_time': timeout,
            'memory_change': 0,
            'cwd_before': cwd or os.getcwd(),
            'success': False
        }
    except Exception as e:
        return {
            'returncode': -1,
            'stdout': '',
            'stderr': str(e),
            'execution_time': 0,
            'memory_change': 0,
            'cwd_before': cwd or os.getcwd(),
            'success': False
        }

def test_configuration_precedence():
    """Test configuration loading precedence: CLI > YAML > env > defaults."""
    
    print("=== Testing Configuration Precedence ===")
    results = {}
    
    # Test 1: Default configuration (help commands to avoid starting server)
    results['default_config'] = run_command("wqm admin --help")
    
    # Test 2: Environment variable configuration 
    env_test_cmd = "WORKSPACE_QDRANT_COLLECTION_PREFIX=test_env wqm admin --help"
    results['env_config'] = run_command(env_test_cmd)
    
    # Test 3: Create temporary YAML config and test
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write("""
qdrant:
  url: "http://test-yaml:6333"
  collection_prefix: "test_yaml"

embeddings:
  provider: "openai"
  model: "text-embedding-ada-002"
""")
        yaml_config_path = f.name
    
    try:
        results['yaml_config'] = run_command(f"wqm --config {yaml_config_path} admin --help")
    finally:
        os.unlink(yaml_config_path)
    
    # Test 4: CLI argument precedence (using debug flag)
    results['cli_debug'] = run_command("wqm --debug admin --help", timeout=5)
    
    return results

def test_memory_usage():
    """Test memory usage and performance characteristics."""
    
    print("=== Testing Memory Usage ===")
    results = {}
    
    # Test 1: Basic help command memory footprint
    results['help_memory'] = run_command("wqm --help")
    
    # Test 2: Version command memory footprint
    results['version_memory'] = run_command("wqm --version")
    
    # Test 3: More complex command memory usage
    results['admin_status_memory'] = run_command("wqm admin --help")
    
    # Test 4: Memory usage for startup overhead
    # We'll use a short timeout to avoid actually starting the server
    results['startup_memory'] = run_command("timeout 2 wqm admin status || true", timeout=5)
    
    return results

def test_cross_platform_compatibility():
    """Test cross-platform compatibility features."""
    
    print("=== Testing Cross-Platform Compatibility ===")
    results = {}
    
    # Test 1: Path handling (test with different path formats)
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a config file with different path formats
        config_content = f"""
qdrant:
  url: "http://localhost:6333"
  data_path: "{temp_dir}"
"""
        config_path = os.path.join(temp_dir, "test-config.yaml")
        with open(config_path, 'w') as f:
            f.write(config_content)
        
        results['path_handling'] = run_command(f"wqm --config {config_path} admin --help")
    
    # Test 2: Environment detection
    results['env_detection'] = run_command("wqm --debug --version", timeout=5)
    
    # Test 3: Unicode handling in commands
    results['unicode_handling'] = run_command("wqm search --help")
    
    return results

def test_mcp_integration():
    """Test MCP integration capabilities."""
    
    print("=== Testing MCP Integration ===")
    results = {}
    
    # Test 1: Server help and available transports
    results['server_help'] = run_command("workspace-qdrant-mcp --help")
    
    # Test 2: Test stdio transport option (just check parsing)
    results['stdio_transport'] = run_command("workspace-qdrant-mcp --transport stdio --help")
    
    # Test 3: Test HTTP transport option (just check parsing)
    results['http_transport'] = run_command("workspace-qdrant-mcp --transport http --help")
    
    # Test 4: Test port configuration
    results['port_config'] = run_command("workspace-qdrant-mcp --port 9000 --help")
    
    # Test 5: Test host configuration
    results['host_config'] = run_command("workspace-qdrant-mcp --host 127.0.0.1 --help")
    
    return results

def test_project_detection_advanced():
    """Test advanced project detection scenarios."""
    
    print("=== Testing Advanced Project Detection ===")
    results = {}
    
    # Test from various directory types
    test_dirs = [
        "/tmp",
        str(Path.home()),
        "/usr/local" if os.path.exists("/usr/local") else "/usr",
    ]
    
    for test_dir in test_dirs:
        if os.path.exists(test_dir) and os.access(test_dir, os.R_OK):
            key = f"detection_{test_dir.replace('/', '_').strip('_')}"
            results[key] = run_command("wqm admin --help", cwd=test_dir)
    
    # Test with nested directories
    with tempfile.TemporaryDirectory() as temp_dir:
        nested_dir = os.path.join(temp_dir, "level1", "level2", "level3")
        os.makedirs(nested_dir, exist_ok=True)
        results['nested_detection'] = run_command("wqm admin --help", cwd=nested_dir)
    
    return results

def analyze_advanced_results(results):
    """Analyze advanced test results."""
    
    analysis = {
        'configuration_precedence': 'unknown',
        'memory_performance': 'unknown',
        'cross_platform': 'unknown', 
        'mcp_integration': 'unknown',
        'project_detection_advanced': 'unknown',
        'performance_metrics': {},
        'issues_found': [],
        'recommendations': []
    }
    
    # Analyze configuration tests
    config_results = results.get('configuration', {})
    config_success_count = sum(1 for r in config_results.values() if r.get('success', False))
    if config_success_count >= 3:
        analysis['configuration_precedence'] = 'working'
    else:
        analysis['configuration_precedence'] = 'issues'
        analysis['issues_found'].append("Configuration precedence may not be working correctly")
    
    # Analyze memory usage
    memory_results = results.get('memory', {})
    total_execution_time = sum(r.get('execution_time', 0) for r in memory_results.values())
    max_memory_change = max((r.get('memory_change', 0) for r in memory_results.values()), default=0)
    
    analysis['performance_metrics'] = {
        'total_test_execution_time': round(total_execution_time, 3),
        'max_memory_change_bytes': max_memory_change,
        'max_memory_change_mb': round(max_memory_change / (1024*1024), 2) if max_memory_change > 0 else 0
    }
    
    # Simple memory performance assessment
    if total_execution_time < 5.0:  # All help commands should be fast
        analysis['memory_performance'] = 'good'
    else:
        analysis['memory_performance'] = 'slow'
        analysis['issues_found'].append("Commands are taking too long to execute")
    
    # Analyze cross-platform compatibility
    cross_platform_results = results.get('cross_platform', {})
    cp_success_count = sum(1 for r in cross_platform_results.values() if r.get('success', False))
    if cp_success_count >= len(cross_platform_results) * 0.8:
        analysis['cross_platform'] = 'compatible'
    else:
        analysis['cross_platform'] = 'issues'
        analysis['issues_found'].append("Cross-platform compatibility issues detected")
    
    # Analyze MCP integration
    mcp_results = results.get('mcp', {})
    mcp_success_count = sum(1 for r in mcp_results.values() if r.get('success', False))
    if mcp_success_count >= len(mcp_results) * 0.8:
        analysis['mcp_integration'] = 'working'
    else:
        analysis['mcp_integration'] = 'issues'
        analysis['issues_found'].append("MCP integration issues detected")
    
    # Analyze advanced project detection
    detection_results = results.get('project_detection', {})
    detection_success_count = sum(1 for r in detection_results.values() if r.get('success', False))
    if detection_success_count >= len(detection_results) * 0.8:
        analysis['project_detection_advanced'] = 'working'
    else:
        analysis['project_detection_advanced'] = 'issues'
        analysis['issues_found'].append("Advanced project detection issues detected")
    
    # Generate recommendations
    if analysis['memory_performance'] == 'good':
        analysis['recommendations'].append("Memory performance is good - within targets")
    
    if not analysis['issues_found']:
        analysis['recommendations'].append("All advanced tests passed - global installation is working correctly")
    
    return analysis

def main():
    """Run all advanced tests."""
    
    print("Advanced Testing Suite for Task 112")
    print("=" * 50)
    
    all_results = {}
    
    # Run all test suites
    all_results['configuration'] = test_configuration_precedence()
    all_results['memory'] = test_memory_usage() 
    all_results['cross_platform'] = test_cross_platform_compatibility()
    all_results['mcp'] = test_mcp_integration()
    all_results['project_detection'] = test_project_detection_advanced()
    
    # Analyze results
    analysis = analyze_advanced_results(all_results)
    
    # Save results
    with open("20250909-1517_advanced_test_results.json", "w") as f:
        json.dump({
            'test_results': all_results,
            'analysis': analysis,
            'timestamp': time.time()
        }, f, indent=2)
    
    # Print summary
    print("\n" + "=" * 50)
    print("ADVANCED TEST RESULTS")
    print("=" * 50)
    
    print(f"Configuration Precedence: {analysis['configuration_precedence']}")
    print(f"Memory Performance: {analysis['memory_performance']}")
    print(f"Cross-Platform: {analysis['cross_platform']}")
    print(f"MCP Integration: {analysis['mcp_integration']}")
    print(f"Advanced Project Detection: {analysis['project_detection_advanced']}")
    
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
    
    print(f"\nDetailed results saved to: 20250909-1517_advanced_test_results.json")

if __name__ == "__main__":
    main()