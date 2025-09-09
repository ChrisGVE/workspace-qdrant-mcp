#!/usr/bin/env python3
"""
Resilience Testing Execution Wrapper
Safely executes the comprehensive resilience testing suite with monitoring and safety controls
"""

import subprocess
import sys
import time
import psutil
from datetime import datetime

def check_system_readiness():
    """Check if system is ready for resilience testing"""
    print("🔍 SYSTEM READINESS CHECK")
    print("="*30)
    
    # Check memory availability
    memory = psutil.virtual_memory()
    cpu_percent = psutil.cpu_percent(interval=1)
    
    print(f"💾 Memory: {memory.percent:.1f}% used ({memory.available/1024**3:.1f}GB available)")
    print(f"🖥️  CPU: {cpu_percent:.1f}% average")
    
    # Safety thresholds - more conservative for resilience testing
    if memory.percent > 85:
        print("❌ INSUFFICIENT MEMORY - System using >85% memory")
        return False
        
    if cpu_percent > 75:
        print("❌ HIGH CPU LOAD - System using >75% CPU") 
        return False
        
    if memory.available < 8 * 1024**3:  # Less than 8GB available
        print("❌ INSUFFICIENT MEMORY - Less than 8GB available")
        return False
        
    print("✅ System ready for resilience testing")
    return True

def check_qdrant_availability():
    """Check if Qdrant is accessible"""
    print("\n🔗 QDRANT CONNECTIVITY CHECK")
    print("="*30)
    
    try:
        import requests
        response = requests.get("http://localhost:6333/cluster", timeout=5)
        if response.status_code == 200:
            print("✅ Qdrant accessible at localhost:6333")
            return True
        else:
            print(f"❌ Qdrant returned status {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Qdrant connection failed: {e}")
        return False

def main():
    """Execute resilience testing with safety controls"""
    print("🚀 RESILIENCE TESTING EXECUTION WRAPPER")
    print(f"🕒 Start Time: {datetime.now().isoformat()}")
    print("="*60)
    
    # Pre-flight checks
    if not check_system_readiness():
        print("\n🚨 SYSTEM NOT READY - Aborting resilience tests")
        sys.exit(1)
        
    if not check_qdrant_availability():
        print("\n🚨 QDRANT NOT AVAILABLE - Aborting resilience tests")
        sys.exit(1)
        
    print("\n🎯 LAUNCHING RESILIENCE TESTING SUITE")
    print("="*45)
    
    # Execute resilience testing suite
    script_path = "/Users/chris/Dropbox/dev/ai/claude-code-cfg/mcp/workspace-qdrant-mcp/LT_20250906-1822_testing_sandbox/qmk_integration/20250106-1854_resilience_recovery_testing_suite.py"
    
    try:
        # Run resilience tests with real-time output
        process = subprocess.Popen(
            [sys.executable, script_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )
        
        # Stream output in real-time
        for line in process.stdout:
            print(line.rstrip())
            
        # Wait for completion
        process.wait()
        
        if process.returncode == 0:
            print(f"\n🎉 RESILIENCE TESTING COMPLETED SUCCESSFULLY")
        else:
            print(f"\n❌ RESILIENCE TESTING FAILED (exit code: {process.returncode})")
            
    except KeyboardInterrupt:
        print(f"\n🛑 RESILIENCE TESTING INTERRUPTED BY USER")
        if process:
            process.terminate()
        sys.exit(1)
        
    except Exception as e:
        print(f"\n🚨 RESILIENCE TESTING ERROR: {e}")
        sys.exit(1)
        
    print(f"\n🕒 End Time: {datetime.now().isoformat()}")
    print("="*60)

if __name__ == "__main__":
    main()