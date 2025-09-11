#!/bin/bash
# Execute WQM Service Tests

echo "=== WQM SERVICE COMMAND TESTING ==="
echo "Starting comprehensive test of all wqm service commands..."
echo "Date: $(date)"
echo ""

# Change to project root
cd "/Users/chris/Dropbox/dev/ai/claude-code-cfg/mcp/workspace-qdrant-mcp"
echo "Working directory: $(pwd)"
echo ""

# Run the Python test script
echo "Executing test script..."
python3 src/python/20250111-2137_wqm_service_testing.py

echo ""
echo "Test execution complete."
echo "Check for result files..."

# Look for result files
if [ -f "20250111-2137_wqm_service_detailed_results.txt" ]; then
    echo "Main results file found: 20250111-2137_wqm_service_detailed_results.txt"
    echo ""
    echo "=== RESULTS PREVIEW ==="
    tail -20 "20250111-2137_wqm_service_detailed_results.txt"
elif [ -f "src/python/20250111-2137_wqm_service_detailed_results.txt" ]; then
    echo "Results file found in src/python/"
    echo ""
    echo "=== RESULTS PREVIEW ==="
    tail -20 "src/python/20250111-2137_wqm_service_detailed_results.txt"
else
    echo "Warning: No results file found"
fi

echo ""
echo "Test execution completed."