#!/usr/bin/env python3
"""
Quick debug test for resource limit stress testing
"""

import requests
import psutil
import time
import json

def test_basic_functionality():
    print("🔍 Testing basic functionality for stress testing")
    
    # Test Qdrant connectivity
    try:
        response = requests.get("http://localhost:6333/cluster")
        print(f"✅ Qdrant accessible: {response.status_code}")
        print(f"   Response: {response.json()}")
    except Exception as e:
        print(f"❌ Qdrant error: {e}")
        return False
        
    # Test system resource monitoring
    try:
        memory = psutil.virtual_memory()
        cpu = psutil.cpu_percent(interval=1)
        print(f"✅ System monitoring:")
        print(f"   Memory: {memory.percent:.1f}% ({memory.used/1024/1024/1024:.1f} GB used)")
        print(f"   CPU: {cpu:.1f}%")
    except Exception as e:
        print(f"❌ System monitoring error: {e}")
        return False
        
    # Test collection creation
    test_collection = f"debug_test_collection"
    try:
        collection_config = {
            "vectors": {
                "size": 384,
                "distance": "Cosine"
            }
        }
        response = requests.put(
            f"http://localhost:6333/collections/{test_collection}",
            json=collection_config
        )
        print(f"✅ Collection creation: {response.status_code}")
        
        # Clean up
        requests.delete(f"http://localhost:6333/collections/{test_collection}")
        
    except Exception as e:
        print(f"❌ Collection creation error: {e}")
        return False
        
    print("✅ All basic functionality tests passed")
    return True

if __name__ == "__main__":
    success = test_basic_functionality()
    if success:
        print("🎯 Ready to run resource limit stress testing")
    else:
        print("❌ Issues found - debug required")