#!/usr/bin/env python3
"""
Single module test to verify file-by-file approach works
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_import_server():
    """Test importing server module"""
    try:
        import workspace_qdrant_mcp.server
        print("✅ server.py imported successfully")
        return True
    except Exception as e:
        print(f"❌ server.py import failed: {e}")
        return False

def test_import_client():
    """Test importing client module"""
    try:
        from workspace_qdrant_mcp.core import client
        print("✅ client.py imported successfully")
        return True
    except Exception as e:
        print(f"❌ client.py import failed: {e}")
        return False

def test_import_embeddings():
    """Test importing embeddings module"""
    try:
        from workspace_qdrant_mcp.core import embeddings
        print("✅ embeddings.py imported successfully")
        return True
    except Exception as e:
        print(f"❌ embeddings.py import failed: {e}")
        return False

if __name__ == "__main__":
    print("Testing file-by-file approach...")

    tests = [test_import_server, test_import_client, test_import_embeddings]
    passed = 0

    for test in tests:
        if test():
            passed += 1

    print(f"\nResults: {passed}/{len(tests)} tests passed")
    print(f"Success rate: {passed/len(tests)*100:.1f}%")