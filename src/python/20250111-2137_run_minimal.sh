#!/bin/bash

cd "/Users/chris/Dropbox/dev/ai/claude-code-cfg/mcp/workspace-qdrant-mcp"

echo "Running minimal WQM service tests..."
echo "Working directory: $(pwd)"
echo "Time: $(date)"
echo ""

# Run the minimal test
python3 src/python/20250111-2137_minimal_test.py 2>&1 | tee src/python/20250111-2137_test_output.log

echo ""
echo "Test completed. Checking for result files..."

if [ -f "src/python/20250111-2137_minimal_test_results.txt" ]; then
    echo "Results file created:"
    cat src/python/20250111-2137_minimal_test_results.txt
else
    echo "No results file found."
fi

if [ -f "src/python/20250111-2137_test_output.log" ]; then
    echo ""
    echo "Full test output available in: src/python/20250111-2137_test_output.log"
fi