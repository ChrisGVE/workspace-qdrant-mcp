#!/usr/bin/env python3
"""Check if memexd binary exists anywhere."""
import os
import subprocess
from pathlib import Path

def check_binary():
    print("🔍 Checking for memexd binary...")
    
    # Check with which
    try:
        result = subprocess.run(["which", "memexd"], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ Found via which: {result.stdout.strip()}")
            return True
        else:
            print("❌ Not found via which")
    except:
        print("❌ which command failed")
    
    # Check common locations
    locations = [
        "/Users/chris/.local/bin/memexd",
        Path.cwd() / "rust-engine" / "target" / "release" / "memexd",
        Path.cwd() / "rust-engine" / "target" / "debug" / "memexd",
        Path.cwd() / "target" / "release" / "memexd", 
        Path.cwd() / "target" / "debug" / "memexd",
    ]
    
    for loc in locations:
        if Path(loc).exists():
            print(f"✅ Found at: {loc}")
            return True
        else:
            print(f"❌ Not found at: {loc}")
    
    print("\n🚨 memexd binary not found anywhere!")
    print("This is the root cause of service installation failures.")
    
    # Check if rust-engine exists
    rust_engine_dir = Path.cwd() / "rust-engine"
    if rust_engine_dir.exists():
        print(f"\n💡 rust-engine directory exists at: {rust_engine_dir}")
        print("Try building the daemon with:")
        print("  cd rust-engine && cargo build --release --bin memexd")
    else:
        print(f"\n❌ rust-engine directory not found at: {rust_engine_dir}")
        print("The daemon binary needs to be built or the service implementation needs to be changed.")
    
    return False

if __name__ == "__main__":
    check_binary()