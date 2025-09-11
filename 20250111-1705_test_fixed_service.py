#!/usr/bin/env python3
"""Test the fixed service implementation."""
import subprocess
import time

def test_command(cmd):
    """Test a single command."""
    print(f"🔧 Testing: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        print(f"   Exit code: {result.returncode}")
        if result.returncode == 0:
            print(f"   ✅ SUCCESS")
        else:
            print(f"   ❌ FAILED")
            if result.stderr:
                print(f"   STDERR: {result.stderr}")
        print()
        return result.returncode == 0
    except Exception as e:
        print(f"   💥 ERROR: {e}")
        print()
        return False

def main():
    wqm = "/Users/chris/.local/bin/wqm"
    print("🚀 Testing fixed service implementation...")
    print("="*60)
    
    # Test each command in sequence
    test_cases = [
        ([wqm, "service", "uninstall"], "Cleanup any existing service"),
        ([wqm, "service", "status"], "Check status when not installed"),
        ([wqm, "service", "install"], "Install service"),
        ([wqm, "service", "status"], "Check status after install"),
        ([wqm, "service", "start"], "Start service"),
        ([wqm, "service", "status"], "Check status after start"),
        ([wqm, "service", "logs"], "Check service logs"),
        ([wqm, "service", "stop"], "Stop service"),
        ([wqm, "service", "status"], "Check status after stop"),
        ([wqm, "service", "restart"], "Restart service"),
        ([wqm, "service", "status"], "Check status after restart"),
        ([wqm, "service", "uninstall"], "Uninstall service"),
        ([wqm, "service", "status"], "Check status after uninstall"),
    ]
    
    success_count = 0
    total_count = len(test_cases)
    
    for cmd, description in test_cases:
        print(f"📋 {description}")
        success = test_command(cmd)
        if success:
            success_count += 1
        time.sleep(2)  # Brief pause between commands
    
    print("="*60)
    print(f"📊 TEST RESULTS:")
    print(f"   Successful: {success_count}/{total_count}")
    print(f"   Failed: {total_count - success_count}/{total_count}")
    print(f"   Success rate: {success_count/total_count*100:.1f}%")
    
    if success_count == total_count:
        print("\n🎉 ALL TESTS PASSED! Service is working correctly!")
        return 0
    else:
        print(f"\n🔥 {total_count - success_count} TESTS FAILED! Service needs more work.")
        return 1

if __name__ == "__main__":
    exit(main())