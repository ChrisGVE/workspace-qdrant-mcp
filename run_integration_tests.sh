#!/bin/bash

# Run multi-component integration tests for Task 80

echo "🔧 Setting up test environment..."

# Ensure test directory exists
mkdir -p tests/integration

echo "🧪 Running multi-component communication tests..."

# Run the specific integration test
python -m pytest tests/integration/test_multi_component_communication.py -v --tb=short -s

echo "✅ Integration tests completed"