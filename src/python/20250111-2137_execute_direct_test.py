#!/usr/bin/env python3
"""
Direct execution test for WQM service commands
"""
import subprocess
import sys
import os

# Ensure we're in the right directory
project_root = "/Users/chris/Dropbox/dev/ai/claude-code-cfg/mcp/workspace-qdrant-mcp"
os.chdir(project_root)

def test_command(cmd_list, description):
    """Test a single command with detailed output"""
    print(f"\n{'='*60}")
    print(f"TEST: {description}")
    print(f"COMMAND: {' '.join(cmd_list)}")
    print('='*60)
    
    try:
        # Run the command
        process = subprocess.Popen(
            cmd_list,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=project_root
        )
        
        stdout, stderr = process.communicate(timeout=30)
        
        print(f"EXIT CODE: {process.returncode}")
        
        if stdout.strip():
            print(f"STDOUT:\n{stdout}")
        
        if stderr.strip():
            print(f"STDERR:\n{stderr}")
            
        # Check for running processes after each command
        ps_proc = subprocess.run(["ps", "aux"], capture_output=True, text=True)
        memexd_procs = [line for line in ps_proc.stdout.split('\n') 
                       if 'memexd' in line.lower() and 'grep' not in line]
        
        print(f"MEMEXD PROCESSES: {len(memexd_procs)}")
        for proc in memexd_procs:
            print(f"  {proc}")
            
        return process.returncode, stdout, stderr
        
    except subprocess.TimeoutExpired:
        print("TIMEOUT: Command exceeded 30 seconds")
        process.kill()
        return -1, "", "Timeout"
    except Exception as e:
        print(f"EXCEPTION: {e}")
        return -2, "", str(e)

def main():
    print("WQM SERVICE TESTING - DIRECT EXECUTION")
    print(f"Time: {subprocess.run(['date'], capture_output=True, text=True).stdout.strip()}")
    print(f"Directory: {os.getcwd()}")
    
    # Test sequence with all critical commands
    test_sequence = [
        (["uv", "run", "wqm", "service", "status"], "Initial status check"),
        (["uv", "run", "wqm", "service", "stop"], "Stop any existing service"), 
        (["uv", "run", "wqm", "service", "uninstall"], "Uninstall any existing service"),
        (["uv", "run", "wqm", "service", "status"], "Status after cleanup"),
        (["uv", "run", "wqm", "service", "install"], "Install service"),
        (["uv", "run", "wqm", "service", "status"], "Status after install"),
        (["uv", "run", "wqm", "service", "start"], "Start service"),
        (["uv", "run", "wqm", "service", "status"], "Status after start"),
        (["uv", "run", "wqm", "service", "stop"], "Stop service"),
        (["uv", "run", "wqm", "service", "status"], "Status after stop"),
        (["uv", "run", "wqm", "service", "restart"], "Restart service"),
        (["uv", "run", "wqm", "service", "status"], "Status after restart"),
        (["uv", "run", "wqm", "service", "start"], "Start when already running"),
        (["uv", "run", "wqm", "service", "install"], "Install when already installed"),
        (["uv", "run", "wqm", "service", "stop"], "Final stop"),
        (["uv", "run", "wqm", "service", "uninstall"], "Final uninstall")
    ]
    
    results = []
    
    for cmd_list, description in test_sequence:
        exit_code, stdout, stderr = test_command(cmd_list, description)
        results.append({
            'command': ' '.join(cmd_list),
            'description': description,
            'exit_code': exit_code,
            'stdout': stdout,
            'stderr': stderr,
            'success': exit_code == 0
        })
    
    # Summary
    print(f"\n{'='*80}")
    print("TESTING SUMMARY")
    print(f"{'='*80}")
    
    total = len(results)
    successes = sum(1 for r in results if r['success'])
    failures = total - successes
    
    print(f"Total commands tested: {total}")
    print(f"Successful commands: {successes}")
    print(f"Failed commands: {failures}")
    print(f"Success rate: {(successes/total*100):.1f}%")
    
    if failures > 0:
        print(f"\nFAILED COMMANDS:")
        for r in results:
            if not r['success']:
                print(f"  FAIL: {r['command']} (exit {r['exit_code']})")
                if r['stderr']:
                    print(f"    Error: {r['stderr'][:100]}...")
    
    # Write detailed results to file
    report_file = f"{project_root}/src/python/20250111-2137_wqm_test_results.txt"
    with open(report_file, 'w') as f:
        f.write("WQM SERVICE TESTING RESULTS\n")
        f.write("="*50 + "\n\n")
        
        for r in results:
            f.write(f"Command: {r['command']}\n")
            f.write(f"Description: {r['description']}\n")
            f.write(f"Exit Code: {r['exit_code']}\n")
            f.write(f"Success: {r['success']}\n")
            if r['stdout']:
                f.write(f"STDOUT:\n{r['stdout']}\n")
            if r['stderr']:
                f.write(f"STDERR:\n{r['stderr']}\n")
            f.write("-" * 40 + "\n\n")
        
        f.write(f"\nSUMMARY:\n")
        f.write(f"Total: {total}, Success: {successes}, Failed: {failures}\n")
        f.write(f"Success Rate: {(successes/total*100):.1f}%\n")
    
    print(f"\nDetailed results saved to: {report_file}")

if __name__ == "__main__":
    main()