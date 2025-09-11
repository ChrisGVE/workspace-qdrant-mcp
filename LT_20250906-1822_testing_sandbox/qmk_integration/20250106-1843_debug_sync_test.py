#!/usr/bin/env python3
"""
Debug Real-Time Sync Test - Simplified version to identify issues
"""

import os
import sys
import time
import json
import requests
from pathlib import Path
from datetime import datetime

def test_basic_connectivity():
    """Test basic connectivity to required services"""
    print("🔍 Testing basic connectivity...")
    
    # Test Qdrant
    try:
        response = requests.get("http://localhost:6333/collections", timeout=5)
        print(f"✅ Qdrant: HTTP {response.status_code}")
        if response.status_code == 200:
            collections = response.json().get('result', {}).get('collections', [])
            print(f"   Collections: {[c['name'] for c in collections]}")
    except Exception as e:
        print(f"❌ Qdrant: {e}")
        return False
    
    # Test workspace daemon (if available)
    workspace_urls = [
        "http://localhost:8000",
        "http://localhost:3000", 
        "http://localhost:8080"
    ]
    
    daemon_found = False
    for url in workspace_urls:
        try:
            response = requests.get(f"{url}/status", timeout=2)
            print(f"✅ Workspace daemon at {url}: HTTP {response.status_code}")
            daemon_found = True
            break
        except Exception:
            continue
    
    if not daemon_found:
        print("⚠️  Workspace daemon not found at common ports")
        print("   Will test with direct Qdrant access")
    
    return True

def test_file_creation_and_detection():
    """Test basic file creation and timing"""
    print("\n🔍 Testing file operations...")
    
    test_dir = Path("/Users/chris/Dropbox/dev/ai/claude-code-cfg/mcp/workspace-qdrant-mcp/LT_20250906-1822_testing_sandbox/qmk_integration/LT_20250906-1822_testing_sandbox/debug_test")
    test_dir.mkdir(parents=True, exist_ok=True)
    
    test_file = test_dir / "debug_test.c"
    
    # Create test file
    start_time = time.time()
    content = f"""
// Debug test file
// Created: {datetime.now().isoformat()}
// Test marker: DEBUG_SYNC_TEST_{int(time.time() * 1000000)}

#include <stdio.h>

void debug_function(void) {{
    printf("Debug sync test function\\n");
}}
"""
    
    with open(test_file, 'w') as f:
        f.write(content)
    
    creation_time = time.time() - start_time
    print(f"✅ File created in {creation_time*1000:.1f}ms")
    
    # Test file access
    if test_file.exists():
        size = test_file.stat().st_size
        print(f"✅ File accessible: {size} bytes")
    else:
        print("❌ File not accessible")
        return False
    
    # Cleanup
    test_file.unlink()
    test_dir.rmdir()
    
    return True

def test_search_functionality():
    """Test basic search against Qdrant"""
    print("\n🔍 Testing search functionality...")
    
    try:
        # Get available collections
        response = requests.get("http://localhost:6333/collections", timeout=5)
        if response.status_code != 200:
            print("❌ Cannot access collections")
            return False
        
        collections = response.json().get('result', {}).get('collections', [])
        if not collections:
            print("⚠️  No collections found")
            return True
        
        # Try to search in the first collection
        collection_name = collections[0]['name']
        print(f"   Testing search in collection: {collection_name}")
        
        # Simple scroll search
        search_data = {
            "limit": 1,
            "with_payload": True
        }
        
        start_time = time.time()
        response = requests.post(
            f"http://localhost:6333/collections/{collection_name}/points/scroll",
            json=search_data,
            timeout=10
        )
        search_time = time.time() - start_time
        
        if response.status_code == 200:
            results = response.json().get('result', {}).get('points', [])
            print(f"✅ Search successful in {search_time*1000:.1f}ms")
            print(f"   Found {len(results)} results")
            return True
        else:
            print(f"❌ Search failed: HTTP {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Search error: {e}")
        return False

def main():
    """Main debug test"""
    print("Debug Real-Time Sync Test")
    print("=" * 40)
    
    # Test 1: Basic connectivity
    if not test_basic_connectivity():
        print("\n❌ Basic connectivity failed")
        return False
    
    # Test 2: File operations
    if not test_file_creation_and_detection():
        print("\n❌ File operations failed")
        return False
    
    # Test 3: Search functionality
    if not test_search_functionality():
        print("\n❌ Search functionality failed")
        return False
    
    print("\n✅ All basic tests passed!")
    print("🔄 Ready for full sync stress testing")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)