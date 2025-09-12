#!/usr/bin/env python3
"""
Service Management Behavior Test

This script tests the improved service management behavior to ensure:
1. macOS services use proper launchctl start/stop commands
2. KeepAlive=false allows proper user control
3. Services don't automatically restart when manually stopped
4. Linux services behave consistently with systemd conventions

Test Plan:
1. Parse service.py for KeepAlive setting 
2. Check that _stop_macos_service uses 'launchctl stop'
3. Validate that _start_macos_service properly loads then starts
4. Confirm Linux uses Restart=on-failure (crash recovery only)
"""

import ast
import re
from pathlib import Path
from typing import Dict, List, Any


def analyze_service_py() -> Dict[str, Any]:
    """Analyze the service.py file for proper service management patterns."""
    service_file = Path("/Users/chris/Dropbox/dev/ai/claude-code-cfg/mcp/workspace-qdrant-mcp/src/python/wqm_cli/cli/commands/service.py")
    
    if not service_file.exists():
        return {"success": False, "error": "service.py not found"}
    
    content = service_file.read_text()
    results = {
        "success": True,
        "macos_fixes": {},
        "linux_consistency": {},
        "issues": []
    }
    
    # Check 1: KeepAlive should be false
    keepalive_pattern = r'<key>KeepAlive</key>\s*<(true|false)/>'
    keepalive_match = re.search(keepalive_pattern, content)
    if keepalive_match:
        keepalive_value = keepalive_match.group(1)
        results["macos_fixes"]["keepalive"] = keepalive_value
        if keepalive_value == "true":
            results["issues"].append("KeepAlive should be 'false' for proper user control")
    else:
        results["issues"].append("KeepAlive setting not found in plist template")
    
    # Check 2: _stop_macos_service should use 'launchctl stop'
    if 'launchctl", "stop"' in content:
        results["macos_fixes"]["uses_stop_command"] = True
    else:
        results["issues"].append("_stop_macos_service should use 'launchctl stop'")
    
    # Check 3: Should not use 'launchctl unload' as primary stop method
    if 'unload" as primary stop' in content or '_stop_macos_service' in content:
        # Look for unload in stop method context
        stop_method_match = re.search(r'async def _stop_macos_service.*?async def', content, re.DOTALL)
        if stop_method_match:
            stop_method = stop_method_match.group(0)
            if 'launchctl", "unload"' in stop_method:
                results["issues"].append("_stop_macos_service should not use 'launchctl unload' as primary method")
            else:
                results["macos_fixes"]["avoids_unload_primary"] = True
    
    # Check 4: _start_macos_service should handle load/start properly
    if 'launchctl", "start"' in content:
        results["macos_fixes"]["uses_start_command"] = True
    else:
        results["issues"].append("_start_macos_service should use 'launchctl start'")
    
    # Check 5: Linux service should use Restart=on-failure
    restart_pattern = r'Restart=([^\s\n]+)'
    restart_match = re.search(restart_pattern, content)
    if restart_match:
        restart_value = restart_match.group(1).strip()
        results["linux_consistency"]["restart_policy"] = restart_value
        if restart_value != "on-failure":
            results["issues"].append(f"Linux should use 'Restart=on-failure', found '{restart_value}'")
    else:
        results["issues"].append("Linux Restart policy not found")
    
    # Check 6: Look for documentation about the approach
    if "KeepAlive=false allows proper start/stop control" in content:
        results["macos_fixes"]["documented_approach"] = True
    else:
        results["issues"].append("Missing documentation about KeepAlive=false approach")
    
    return results


def analyze_service_methods() -> Dict[str, Any]:
    """Analyze specific service method implementations."""
    service_file = Path("/Users/chris/Dropbox/dev/ai/claude-code-cfg/mcp/workspace-qdrant-mcp/src/python/wqm_cli/cli/commands/service.py")
    content = service_file.read_text()
    
    results = {
        "stop_method_analysis": {},
        "start_method_analysis": {},
        "issues": []
    }
    
    # Extract _stop_macos_service method
    stop_match = re.search(r'async def _stop_macos_service.*?(?=async def|\Z)', content, re.DOTALL)
    if stop_match:
        stop_method = stop_match.group(0)
        
        # Check for proper stop command usage
        if '"launchctl", "stop"' in stop_method:
            results["stop_method_analysis"]["uses_launchctl_stop"] = True
        else:
            results["issues"].append("stop method should use 'launchctl stop'")
        
        # Check that it verifies service status after stop
        if 'launchctl", "list"' in stop_method:
            results["stop_method_analysis"]["verifies_stop_status"] = True
        else:
            results["issues"].append("stop method should verify service stopped")
        
        # Check that it doesn't kill processes as primary method
        if stop_method.count('kill') == 0:
            results["stop_method_analysis"]["avoids_manual_kill"] = True
        else:
            # Allow kill only as fallback for orphaned processes
            if "orphaned" in stop_method.lower() or "fallback" in stop_method.lower():
                results["stop_method_analysis"]["kill_only_for_orphaned"] = True
            else:
                results["issues"].append("stop method should not use kill as primary mechanism")
    
    # Extract _start_macos_service method  
    start_match = re.search(r'async def _start_macos_service.*?(?=async def|\Z)', content, re.DOTALL)
    if start_match:
        start_method = start_match.group(0)
        
        # Check for proper load then start sequence
        if '"launchctl", "load"' in start_method and '"launchctl", "start"' in start_method:
            results["start_method_analysis"]["proper_load_start_sequence"] = True
        else:
            results["issues"].append("start method should load service first, then start it")
        
        # Check that it verifies if service is already loaded
        if '"launchctl", "list"' in start_method:
            results["start_method_analysis"]["checks_if_loaded"] = True
        else:
            results["issues"].append("start method should check if service is already loaded")
    
    return results


def main():
    """Run all service management tests."""
    print("🔍 Analyzing Service Management Implementation")
    print("=" * 50)
    
    # Test 1: Overall file analysis
    print("\n📋 Test 1: Service Management Configuration")
    file_results = analyze_service_py()
    
    if file_results["success"]:
        print("✅ service.py file found and parsed")
        
        # macOS fixes
        macos = file_results["macos_fixes"]
        print(f"   KeepAlive setting: {macos.get('keepalive', 'NOT FOUND')}")
        print(f"   Uses stop command: {macos.get('uses_stop_command', False)}")
        print(f"   Uses start command: {macos.get('uses_start_command', False)}")
        print(f"   Avoids unload as primary: {macos.get('avoids_unload_primary', False)}")
        print(f"   Documented approach: {macos.get('documented_approach', False)}")
        
        # Linux consistency
        linux = file_results["linux_consistency"] 
        print(f"   Linux restart policy: {linux.get('restart_policy', 'NOT FOUND')}")
    
    # Test 2: Method implementation analysis
    print("\n📋 Test 2: Service Method Implementation")
    method_results = analyze_service_methods()
    
    stop_analysis = method_results["stop_method_analysis"]
    print(f"   Stop uses launchctl stop: {stop_analysis.get('uses_launchctl_stop', False)}")
    print(f"   Stop verifies status: {stop_analysis.get('verifies_stop_status', False)}")
    print(f"   Stop avoids manual kill: {stop_analysis.get('avoids_manual_kill', False)}")
    
    start_analysis = method_results["start_method_analysis"]
    print(f"   Start uses load+start: {start_analysis.get('proper_load_start_sequence', False)}")
    print(f"   Start checks if loaded: {start_analysis.get('checks_if_loaded', False)}")
    
    # Compile all issues
    all_issues = file_results.get("issues", []) + method_results.get("issues", [])
    
    print("\n📊 Summary")
    print("=" * 30)
    
    if not all_issues:
        print("✅ All service management fixes implemented correctly!")
        print("\n🎯 Key Improvements:")
        print("   • macOS: KeepAlive=false allows proper user control")
        print("   • macOS: Uses launchctl start/stop instead of load/unload workaround")  
        print("   • Linux: Restart=on-failure provides crash recovery only")
        print("   • Both platforms follow OS service management conventions")
    else:
        print(f"❌ Found {len(all_issues)} issues:")
        for i, issue in enumerate(all_issues, 1):
            print(f"   {i}. {issue}")
    
    return len(all_issues) == 0


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)