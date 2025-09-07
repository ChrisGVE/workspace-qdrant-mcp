#!/usr/bin/env python3
"""
Test file for auto-ingestion system verification.

This file tests the automatic file ingestion capabilities of the workspace-qdrant-mcp system.
It should be automatically detected and processed by the file watcher.

Features tested:
- Python code detection
- Automatic embedding generation
- Semantic search integration
"""

def test_function():
    """Simple test function for auto-ingestion verification."""
    return "Auto-ingestion is working!"

def fibonacci(n):
    """Calculate fibonacci number using recursion."""
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

class TestClass:
    """Test class for object-oriented code detection."""
    
    def __init__(self, name):
        self.name = name
        
    def greet(self):
        return f"Hello from {self.name}!"

if __name__ == "__main__":
    print(test_function())
    test_obj = TestClass("Auto-Ingestion System")
    print(test_obj.greet())