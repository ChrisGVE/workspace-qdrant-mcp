#!/bin/bash

echo "Running WQM Service Tests..."
cd "/Users/chris/Dropbox/dev/ai/claude-code-cfg/mcp/workspace-qdrant-mcp"

echo "Step 1: Quick WQM Check"
python3 src/python/20250111-2137_quick_wqm_check.py > src/python/20250111-2137_quick_check_results.txt 2>&1
cat src/python/20250111-2137_quick_check_results.txt

echo ""
echo "Step 2: Direct Test Execution" 
python3 src/python/20250111-2137_execute_direct_test.py > src/python/20250111-2137_direct_test_output.txt 2>&1

echo "Direct test completed. Checking results..."
if [ -f "src/python/20250111-2137_wqm_test_results.txt" ]; then
    echo "Results file generated successfully"
    echo "Preview of results:"
    tail -20 src/python/20250111-2137_wqm_test_results.txt
else
    echo "No results file found. Checking output..."
    if [ -f "src/python/20250111-2137_direct_test_output.txt" ]; then
        echo "Test output:"
        cat src/python/20250111-2137_direct_test_output.txt
    fi
fi