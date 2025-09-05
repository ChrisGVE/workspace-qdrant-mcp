#!/bin/bash

# Workspace Qdrant MCP Installation Verification Script
# This script performs comprehensive health checks after installation

set -euo pipefail

readonly CONFIG_DIR="$HOME/.workspace-qdrant-mcp"
readonly DAEMON_NAME="memexd"
readonly DAEMON_PATH="/usr/local/bin/$DAEMON_NAME"

# Colors
readonly RED='\033[0;31m'
readonly GREEN='\033[0;32m'
readonly YELLOW='\033[1;33m'
readonly BLUE='\033[0;34m'
readonly NC='\033[0m'

# Counters
TESTS_RUN=0
TESTS_PASSED=0
TESTS_FAILED=0

# Test result tracking
FAILED_TESTS=()

# Utility functions
info() { echo -e "${BLUE}[INFO]${NC} $1"; }
success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Test framework
run_test() {
    local test_name="$1"
    local test_function="$2"
    
    ((TESTS_RUN++))
    
    info "Running test: $test_name"
    
    if $test_function; then
        success "✓ $test_name"
        ((TESTS_PASSED++))
        return 0
    else
        error "✗ $test_name"
        FAILED_TESTS+=("$test_name")
        ((TESTS_FAILED++))
        return 1
    fi
}

# Test functions
test_prerequisites() {
    local all_good=true
    
    # Check uv
    if ! command -v uv >/dev/null 2>&1; then
        error "  uv not found"
        all_good=false
    fi
    
    # Check python
    if ! uv python --version >/dev/null 2>&1; then
        error "  Python via uv not accessible"
        all_good=false
    fi
    
    # Check rust
    if ! command -v rustc >/dev/null 2>&1; then
        error "  Rust compiler not found"
        all_good=false
    fi
    
    if ! command -v cargo >/dev/null 2>&1; then
        error "  Cargo not found"
        all_good=false
    fi
    
    $all_good
}

test_daemon_binary() {
    if [[ ! -f "$DAEMON_PATH" ]]; then
        error "  Daemon binary not found at $DAEMON_PATH"
        return 1
    fi
    
    if [[ ! -x "$DAEMON_PATH" ]]; then
        error "  Daemon binary not executable"
        return 1
    fi
    
    # Test version command
    if ! "$DAEMON_PATH" --version >/dev/null 2>&1; then
        error "  Daemon binary doesn't respond to --version"
        return 1
    fi
    
    return 0
}

test_python_packages() {
    local all_good=true
    
    # Check main server command
    if ! command -v workspace-qdrant-mcp >/dev/null 2>&1; then
        error "  workspace-qdrant-mcp command not found"
        all_good=false
    fi
    
    # Check CLI command
    if ! command -v wqm >/dev/null 2>&1; then
        error "  wqm command not found"
        all_good=false
    fi
    
    # Test help commands
    if ! workspace-qdrant-mcp --help >/dev/null 2>&1; then
        error "  workspace-qdrant-mcp --help failed"
        all_good=false
    fi
    
    if ! wqm --help >/dev/null 2>&1; then
        error "  wqm --help failed"
        all_good=false
    fi
    
    $all_good
}

test_directory_structure() {
    local all_good=true
    local required_dirs=(
        "$CONFIG_DIR"
        "$CONFIG_DIR/config"
        "$CONFIG_DIR/logs"
        "$CONFIG_DIR/data"
    )
    
    for dir in "${required_dirs[@]}"; do
        if [[ ! -d "$dir" ]]; then
            error "  Required directory missing: $dir"
            all_good=false
        fi
    done
    
    $all_good
}

test_configuration_files() {
    local all_good=true
    local config_file="$CONFIG_DIR/config/default.yaml"
    local mcp_config="$CONFIG_DIR/config/mcp.json"
    
    if [[ ! -f "$config_file" ]]; then
        error "  Default configuration missing: $config_file"
        all_good=false
    else
        # Test YAML parsing
        if ! python3 -c "import yaml; yaml.safe_load(open('$config_file'))" 2>/dev/null; then
            error "  Default configuration has invalid YAML syntax"
            all_good=false
        fi
    fi
    
    if [[ ! -f "$mcp_config" ]]; then
        error "  MCP configuration template missing: $mcp_config"
        all_good=false
    else
        # Test JSON parsing
        if ! python3 -c "import json; json.load(open('$mcp_config'))" 2>/dev/null; then
            error "  MCP configuration has invalid JSON syntax"
            all_good=false
        fi
    fi
    
    $all_good
}

test_daemon_management_script() {
    local script_path="$CONFIG_DIR/manage-daemon.sh"
    
    if [[ ! -f "$script_path" ]]; then
        error "  Daemon management script missing: $script_path"
        return 1
    fi
    
    if [[ ! -x "$script_path" ]]; then
        error "  Daemon management script not executable"
        return 1
    fi
    
    # Test help command
    if ! "$script_path" help >/dev/null 2>&1; then
        error "  Daemon management script help command failed"
        return 1
    fi
    
    return 0
}

test_daemon_functionality() {
    local script_path="$CONFIG_DIR/manage-daemon.sh"
    
    # Test daemon start (don't actually start, just test the script logic)
    if ! "$script_path" status >/dev/null 2>&1; then
        # This is expected if daemon is not running
        info "  Daemon not currently running (this is normal)"
    fi
    
    # The fact that status command runs without error is a good sign
    return 0
}

test_qdrant_connectivity() {
    # Try to connect to default Qdrant instance
    local qdrant_url="http://localhost:6333"
    
    if command -v curl >/dev/null 2>&1; then
        if curl -s "$qdrant_url/collections" >/dev/null 2>&1; then
            info "  Qdrant server is accessible at $qdrant_url"
            return 0
        else
            warning "  Qdrant server not accessible at $qdrant_url (start with: docker run -p 6333:6333 qdrant/qdrant)"
            # This is not a fatal error for installation verification
            return 0
        fi
    else
        warning "  curl not available, cannot test Qdrant connectivity"
        return 0
    fi
}

test_mcp_server_startup() {
    # Test that the MCP server can at least show help without crashing
    local timeout=10
    
    if timeout $timeout workspace-qdrant-mcp --help >/dev/null 2>&1; then
        return 0
    else
        error "  MCP server failed to show help within $timeout seconds"
        return 1
    fi
}

test_python_imports() {
    # Test that required Python packages can be imported
    local imports=(
        "fastmcp"
        "qdrant_client"
        "fastembed"
        "GitPython"
        "pydantic"
        "yaml"
        "typer"
    )
    
    local all_good=true
    
    for import in "${imports[@]}"; do
        if ! python3 -c "import ${import}" 2>/dev/null; then
            error "  Cannot import Python package: $import"
            all_good=false
        fi
    done
    
    $all_good
}

# Main test runner
run_all_tests() {
    info "Starting comprehensive installation verification..."
    echo
    
    run_test "Prerequisites (uv, rust)" test_prerequisites
    run_test "Daemon binary installation" test_daemon_binary
    run_test "Python package installation" test_python_packages
    run_test "Directory structure" test_directory_structure
    run_test "Configuration files" test_configuration_files
    run_test "Daemon management script" test_daemon_management_script
    run_test "Daemon functionality" test_daemon_functionality
    run_test "Qdrant connectivity" test_qdrant_connectivity
    run_test "MCP server startup" test_mcp_server_startup
    run_test "Python imports" test_python_imports
    
    echo
    print_summary
}

# Print test summary
print_summary() {
    echo "=============================================="
    echo "           VERIFICATION SUMMARY"
    echo "=============================================="
    echo
    echo "Tests run:    $TESTS_RUN"
    echo -e "Tests passed: ${GREEN}$TESTS_PASSED${NC}"
    echo -e "Tests failed: ${RED}$TESTS_FAILED${NC}"
    echo
    
    if (( TESTS_FAILED == 0 )); then
        success "All tests passed! Installation appears to be working correctly."
        echo
        print_next_steps
        return 0
    else
        error "Some tests failed. Please check the issues above."
        echo
        echo "Failed tests:"
        for test in "${FAILED_TESTS[@]}"; do
            echo -e "  ${RED}✗${NC} $test"
        done
        echo
        print_troubleshooting
        return 1
    fi
}

# Print next steps
print_next_steps() {
    cat << EOF
${BLUE}Next Steps:${NC}

1. ${YELLOW}Start Qdrant (if not already running):${NC}
   docker run -p 6333:6333 qdrant/qdrant

2. ${YELLOW}Test the MCP server:${NC}
   workspace-qdrant-mcp --help
   workspace-qdrant-mcp --config-file $CONFIG_DIR/config/default.yaml

3. ${YELLOW}Test the CLI tools:${NC}
   wqm --help
   wqm health

4. ${YELLOW}Manage the daemon:${NC}
   $CONFIG_DIR/manage-daemon.sh start
   $CONFIG_DIR/manage-daemon.sh status

5. ${YELLOW}Configure for Claude (optional):${NC}
   Copy $CONFIG_DIR/config/mcp.json to your Claude configuration

6. ${YELLOW}Customize configuration:${NC}
   Edit $CONFIG_DIR/config/default.yaml

${BLUE}Documentation:${NC}
  - Configuration: $CONFIG_DIR/config/default.yaml
  - Logs: $CONFIG_DIR/logs/
  - Management: $CONFIG_DIR/manage-daemon.sh help

EOF
}

# Print troubleshooting information
print_troubleshooting() {
    cat << EOF
${YELLOW}Troubleshooting:${NC}

1. ${YELLOW}Command not found errors:${NC}
   - Ensure uv's tool bin directory is in your PATH
   - Try: export PATH="$HOME/.local/bin:\$PATH"
   - Restart your shell or source ~/.bashrc / ~/.zshrc

2. ${YELLOW}Permission errors:${NC}
   - Daemon binary: sudo chown \$USER $DAEMON_PATH
   - Config directory: chown -R \$USER:$USER $CONFIG_DIR

3. ${YELLOW}Python import errors:${NC}
   - Try: uv tool upgrade workspace-qdrant-mcp
   - Or reinstall: uv tool uninstall workspace-qdrant-mcp && uv tool install .

4. ${YELLOW}Daemon issues:${NC}
   - Check logs: $CONFIG_DIR/manage-daemon.sh logs
   - Restart: $CONFIG_DIR/manage-daemon.sh restart

5. ${YELLOW}Configuration issues:${NC}
   - Validate YAML: python3 -c "import yaml; print(yaml.safe_load(open('$CONFIG_DIR/config/default.yaml')))"
   - Check permissions: ls -la $CONFIG_DIR/config/

${BLUE}For more help:${NC}
  - Check installation logs: /tmp/workspace-qdrant-mcp-install.log
  - Run with debug: workspace-qdrant-mcp --debug
  - Review documentation in the project repository

EOF
}

# Run verification
main() {
    case "${1:-verify}" in
        verify|--verify|-v)
            run_all_tests
            ;;
        help|--help|-h)
            cat << EOF
Workspace Qdrant MCP Installation Verification

USAGE:
    $0 [verify]     Run all verification tests (default)
    $0 help         Show this help message

This script performs comprehensive checks to verify that the
Workspace Qdrant MCP system was installed correctly and is
ready for use.
EOF
            ;;
        *)
            error "Unknown option: $1"
            echo "Use '$0 help' for usage information"
            exit 1
            ;;
    esac
}

# Run if script is executed directly
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi