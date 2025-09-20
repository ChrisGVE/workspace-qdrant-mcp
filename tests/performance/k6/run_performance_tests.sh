#!/bin/bash

# K6 Performance Test Runner for Workspace-Qdrant-MCP
# Tests all 11 core MCP tools for sub-200ms response time requirements

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../../" && pwd)"
TEST_SCRIPT="${SCRIPT_DIR}/mcp_performance_tests.js"
CONFIG_FILE="${SCRIPT_DIR}/config.json"
RESULTS_DIR="${SCRIPT_DIR}/results"
LOG_FILE="${RESULTS_DIR}/performance_test_$(date +%Y%m%d_%H%M%S).log"

# Server configuration
MCP_SERVER_URL="${MCP_SERVER_URL:-http://127.0.0.1:8000}"
MCP_SERVER_PID=""
QDRANT_REQUIRED="${QDRANT_REQUIRED:-true}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${1}" | tee -a "${LOG_FILE}"
}

# Error handling
cleanup() {
    local exit_code=$?
    log "${YELLOW}🧹 Cleaning up...${NC}"

    if [[ -n "${MCP_SERVER_PID}" ]] && kill -0 "${MCP_SERVER_PID}" 2>/dev/null; then
        log "${YELLOW}   Stopping MCP server (PID: ${MCP_SERVER_PID})${NC}"
        kill "${MCP_SERVER_PID}" || true
        wait "${MCP_SERVER_PID}" 2>/dev/null || true
    fi

    # Clean up test collections and temp files
    rm -rf /tmp/k6_test_* 2>/dev/null || true

    exit $exit_code
}

trap cleanup EXIT

# Check prerequisites
check_prerequisites() {
    log "${BLUE}🔍 Checking prerequisites...${NC}"

    # Check k6 installation
    if ! command -v k6 &> /dev/null; then
        log "${RED}❌ k6 is not installed. Please install k6 first.${NC}"
        log "${YELLOW}   Install with: brew install k6${NC}"
        exit 1
    fi

    log "${GREEN}   ✅ k6 found: $(k6 version)${NC}"

    # Check Python virtual environment
    if [[ -z "${VIRTUAL_ENV:-}" ]] && [[ ! -f "${PROJECT_ROOT}/.venv/bin/activate" ]]; then
        log "${RED}❌ No Python virtual environment found${NC}"
        exit 1
    fi

    # Activate virtual environment if not already active
    if [[ -z "${VIRTUAL_ENV:-}" ]]; then
        source "${PROJECT_ROOT}/.venv/bin/activate"
        log "${GREEN}   ✅ Virtual environment activated${NC}"
    fi

    # Check required Python packages
    if ! python -c "import workspace_qdrant_mcp" 2>/dev/null; then
        log "${RED}❌ workspace_qdrant_mcp package not found${NC}"
        log "${YELLOW}   Run: uv sync --dev${NC}"
        exit 1
    fi

    log "${GREEN}   ✅ workspace_qdrant_mcp package found${NC}"

    # Check Qdrant if required
    if [[ "${QDRANT_REQUIRED}" == "true" ]]; then
        if ! curl -s http://localhost:6333/health >/dev/null 2>&1; then
            log "${YELLOW}⚠️  Qdrant server not running on localhost:6333${NC}"
            log "${YELLOW}   Starting Qdrant with Docker...${NC}"

            if ! command -v docker &> /dev/null; then
                log "${RED}❌ Docker not found. Please start Qdrant manually or install Docker.${NC}"
                exit 1
            fi

            docker run -d --name qdrant-k6-test -p 6333:6333 qdrant/qdrant:latest >/dev/null 2>&1 || {
                log "${RED}❌ Failed to start Qdrant container${NC}"
                exit 1
            }

            # Wait for Qdrant to be ready
            log "${YELLOW}   Waiting for Qdrant to be ready...${NC}"
            for i in {1..30}; do
                if curl -s http://localhost:6333/health >/dev/null 2>&1; then
                    log "${GREEN}   ✅ Qdrant is ready${NC}"
                    break
                fi
                sleep 1
                if [[ $i -eq 30 ]]; then
                    log "${RED}❌ Qdrant failed to start within 30 seconds${NC}"
                    exit 1
                fi
            done
        else
            log "${GREEN}   ✅ Qdrant is running${NC}"
        fi
    fi
}

# Start MCP server
start_mcp_server() {
    log "${BLUE}🚀 Starting MCP server...${NC}"

    cd "${PROJECT_ROOT}"

    # Start server in HTTP mode for k6 testing
    python -m workspace_qdrant_mcp.web.server --host 127.0.0.1 --port 8000 > "${RESULTS_DIR}/server.log" 2>&1 &
    MCP_SERVER_PID=$!

    log "${YELLOW}   Server PID: ${MCP_SERVER_PID}${NC}"

    # Wait for server to be ready
    log "${YELLOW}   Waiting for server to be ready...${NC}"
    for i in {1..30}; do
        if curl -s "${MCP_SERVER_URL}/health" >/dev/null 2>&1; then
            log "${GREEN}   ✅ MCP server is ready${NC}"
            return 0
        fi
        sleep 1
        if [[ $i -eq 30 ]]; then
            log "${RED}❌ MCP server failed to start within 30 seconds${NC}"
            log "${RED}   Server logs:${NC}"
            tail -20 "${RESULTS_DIR}/server.log" | while read line; do
                log "${RED}   ${line}${NC}"
            done
            exit 1
        fi
    done
}

# Run performance tests
run_k6_tests() {
    log "${BLUE}📊 Running K6 performance tests...${NC}"

    local test_name="${1:-full}"
    local output_file="${RESULTS_DIR}/k6_results_${test_name}_$(date +%Y%m%d_%H%M%S).json"

    # Set environment variables for k6
    export MCP_SERVER_URL

    case "${test_name}" in
        "load")
            log "${YELLOW}   Running load test (10 VUs, 30s)...${NC}"
            k6 run --out json="${output_file}" \
                   --scenario load_test \
                   "${TEST_SCRIPT}"
            ;;
        "stress")
            log "${YELLOW}   Running stress test (ramp to 50 VUs)...${NC}"
            k6 run --out json="${output_file}" \
                   --scenario stress_test \
                   "${TEST_SCRIPT}"
            ;;
        "spike")
            log "${YELLOW}   Running spike test (burst to 200 req/s)...${NC}"
            k6 run --out json="${output_file}" \
                   --scenario spike_test \
                   "${TEST_SCRIPT}"
            ;;
        "full")
            log "${YELLOW}   Running all test scenarios...${NC}"
            k6 run --out json="${output_file}" \
                   "${TEST_SCRIPT}"
            ;;
        *)
            log "${RED}❌ Unknown test type: ${test_name}${NC}"
            log "${YELLOW}   Available tests: load, stress, spike, full${NC}"
            exit 1
            ;;
    esac

    local exit_code=$?

    if [[ $exit_code -eq 0 ]]; then
        log "${GREEN}✅ K6 tests completed successfully${NC}"
        log "${BLUE}📄 Results saved to: ${output_file}${NC}"

        # Parse and display key metrics
        if command -v jq &> /dev/null && [[ -f "${output_file}" ]]; then
            log "${BLUE}📈 Key Performance Metrics:${NC}"

            # Extract summary metrics
            local p95_duration=$(jq -r '.metrics.http_req_duration.values.p95' "${output_file}" 2>/dev/null || echo "N/A")
            local p99_duration=$(jq -r '.metrics.http_req_duration.values.p99' "${output_file}" 2>/dev/null || echo "N/A")
            local error_rate=$(jq -r '.metrics.http_req_failed.values.rate' "${output_file}" 2>/dev/null || echo "N/A")
            local total_requests=$(jq -r '.metrics.http_reqs.values.count' "${output_file}" 2>/dev/null || echo "N/A")

            log "${YELLOW}   Response Time P95: ${p95_duration}ms${NC}"
            log "${YELLOW}   Response Time P99: ${p99_duration}ms${NC}"
            log "${YELLOW}   Error Rate: ${error_rate}${NC}"
            log "${YELLOW}   Total Requests: ${total_requests}${NC}"

            # Check if sub-200ms target was met
            if [[ "${p95_duration}" != "N/A" ]] && (( $(echo "${p95_duration} < 200" | bc -l) )); then
                log "${GREEN}🎯 SUCCESS: Sub-200ms target achieved!${NC}"
            else
                log "${RED}❌ FAILED: Sub-200ms target not met${NC}"
            fi
        fi
    else
        log "${RED}❌ K6 tests failed with exit code: ${exit_code}${NC}"
        return $exit_code
    fi
}

# Generate performance report
generate_report() {
    log "${BLUE}📋 Generating performance report...${NC}"

    local report_file="${RESULTS_DIR}/performance_report_$(date +%Y%m%d_%H%M%S).md"

    cat > "${report_file}" << EOF
# Workspace-Qdrant-MCP Performance Test Report

Generated: $(date)
Test Duration: $(date)
Server: ${MCP_SERVER_URL}

## Test Configuration

- **Target Response Time**: < 200ms (P95)
- **Maximum Acceptable**: < 500ms (P99)
- **Error Rate Limit**: < 1%
- **Success Rate Target**: >= 95%

## Core MCP Tools Tested

1. list_workspace_collections
2. create_collection
3. add_document_tool
4. get_document_tool
5. search_by_metadata_tool
6. update_scratchbook_tool
7. search_scratchbook_tool
8. list_scratchbook_notes_tool
9. hybrid_search_advanced_tool
10. add_watch_folder
11. list_watched_folders

## Test Scenarios

### Load Test
- 10 Virtual Users
- 30 seconds duration
- Sustained constant load

### Stress Test
- Ramp up from 0 to 50 VUs
- 2 minutes total duration
- Tests system limits

### Spike Test
- Sudden traffic spikes
- Up to 200 requests/second
- Tests resilience

## Results

EOF

    # Add results from latest test files
    for result_file in "${RESULTS_DIR}"/k6_results_*.json; do
        if [[ -f "${result_file}" ]] && command -v jq &> /dev/null; then
            echo "### $(basename "${result_file}")" >> "${report_file}"
            echo "" >> "${report_file}"

            local p95=$(jq -r '.metrics.http_req_duration.values.p95' "${result_file}" 2>/dev/null || echo "N/A")
            local p99=$(jq -r '.metrics.http_req_duration.values.p99' "${result_file}" 2>/dev/null || echo "N/A")
            local error_rate=$(jq -r '.metrics.http_req_failed.values.rate' "${result_file}" 2>/dev/null || echo "N/A")
            local requests=$(jq -r '.metrics.http_reqs.values.count' "${result_file}" 2>/dev/null || echo "N/A")

            cat >> "${report_file}" << EOF
- **Response Time P95**: ${p95}ms
- **Response Time P99**: ${p99}ms
- **Error Rate**: ${error_rate}
- **Total Requests**: ${requests}

EOF
        fi
    done

    cat >> "${report_file}" << EOF
## Conclusion

$(if [[ -f "${RESULTS_DIR}"/k6_results_*.json ]]; then
    local latest_result=$(ls -t "${RESULTS_DIR}"/k6_results_*.json | head -1)
    local p95=$(jq -r '.metrics.http_req_duration.values.p95' "${latest_result}" 2>/dev/null || echo "999")
    if (( $(echo "${p95} < 200" | bc -l) )); then
        echo "✅ **PASSED**: All MCP tools meet the sub-200ms response time requirement."
    else
        echo "❌ **FAILED**: Sub-200ms target not achieved. Performance optimization needed."
    fi
else
    echo "⚠️  **INCOMPLETE**: Test results not available for analysis."
fi)

## Recommendations

- Monitor response times continuously in production
- Consider implementing caching for frequently accessed data
- Optimize database queries for search operations
- Scale horizontally if load increases beyond current capacity

---
*Generated by workspace-qdrant-mcp k6 performance test suite*
EOF

    log "${GREEN}📋 Report generated: ${report_file}${NC}"
}

# Main execution
main() {
    local test_type="${1:-full}"

    log "${BLUE}🎯 K6 Performance Testing for Workspace-Qdrant-MCP${NC}"
    log "${YELLOW}   Target: Sub-200ms response times for all MCP tools${NC}"

    # Create results directory
    mkdir -p "${RESULTS_DIR}"

    # Check prerequisites
    check_prerequisites

    # Start MCP server
    start_mcp_server

    # Run performance tests
    run_k6_tests "${test_type}"

    # Generate report
    generate_report

    log "${GREEN}🏁 Performance testing completed successfully!${NC}"
    log "${BLUE}📁 Results available in: ${RESULTS_DIR}${NC}"
}

# Parse command line arguments
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    case "${1:-}" in
        "-h"|"--help")
            echo "Usage: $0 [test_type]"
            echo ""
            echo "Test types:"
            echo "  load    - Load test (10 VUs, 30s)"
            echo "  stress  - Stress test (ramp to 50 VUs)"
            echo "  spike   - Spike test (burst to 200 req/s)"
            echo "  full    - All test scenarios (default)"
            echo ""
            echo "Environment variables:"
            echo "  MCP_SERVER_URL      - Server URL (default: http://127.0.0.1:8000)"
            echo "  QDRANT_REQUIRED     - Require Qdrant server (default: true)"
            exit 0
            ;;
        *)
            main "$@"
            ;;
    esac
fi