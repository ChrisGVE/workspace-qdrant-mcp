#!/bin/bash

# Comprehensive test runner for workspace-qdrant-mcp Rust workspace
# This script runs all tests, benchmarks, and generates coverage reports

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
WORKSPACE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COVERAGE_DIR="${WORKSPACE_ROOT}/coverage"
BENCHMARK_DIR="${WORKSPACE_ROOT}/benchmark-results"

# Default options
RUN_UNIT_TESTS=true
RUN_INTEGRATION_TESTS=true
RUN_DOCTESTS=true
RUN_BENCHMARKS=false
RUN_COVERAGE=false
RUN_CLIPPY=true
RUN_FORMAT_CHECK=false
VERBOSE=false
FAIL_FAST=false

# Function to print colored output
print_status() {
    local color=$1
    local message=$2
    echo -e "${color}[$(date +'%H:%M:%S')] ${message}${NC}"
}

print_success() {
    print_status "$GREEN" "✓ $1"
}

print_error() {
    print_status "$RED" "✗ $1"
}

print_warning() {
    print_status "$YELLOW" "⚠ $1"
}

print_info() {
    print_status "$BLUE" "ℹ $1"
}

# Function to show usage
show_usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Test runner for workspace-qdrant-mcp Rust workspace

OPTIONS:
    --unit              Run unit tests only
    --integration       Run integration tests only
    --doctests          Run doctests only
    --benchmarks        Run benchmarks
    --coverage          Generate coverage report
    --clippy            Run clippy lints (default: true)
    --format-check      Check code formatting
    --no-clippy         Skip clippy lints
    --verbose           Show verbose output
    --fail-fast         Stop on first test failure
    --all               Run all tests and checks
    --help              Show this help message

EXAMPLES:
    $0                          # Run default test suite
    $0 --all                    # Run everything including benchmarks and coverage
    $0 --unit --clippy          # Run unit tests and clippy
    $0 --integration --verbose  # Run integration tests with verbose output
    $0 --coverage               # Generate coverage report only

EOF
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --unit)
            RUN_UNIT_TESTS=true
            RUN_INTEGRATION_TESTS=false
            RUN_DOCTESTS=false
            ;;
        --integration)
            RUN_UNIT_TESTS=false
            RUN_INTEGRATION_TESTS=true
            RUN_DOCTESTS=false
            ;;
        --doctests)
            RUN_UNIT_TESTS=false
            RUN_INTEGRATION_TESTS=false
            RUN_DOCTESTS=true
            ;;
        --benchmarks)
            RUN_BENCHMARKS=true
            ;;
        --coverage)
            RUN_COVERAGE=true
            ;;
        --clippy)
            RUN_CLIPPY=true
            ;;
        --no-clippy)
            RUN_CLIPPY=false
            ;;
        --format-check)
            RUN_FORMAT_CHECK=true
            ;;
        --verbose)
            VERBOSE=true
            ;;
        --fail-fast)
            FAIL_FAST=true
            ;;
        --all)
            RUN_UNIT_TESTS=true
            RUN_INTEGRATION_TESTS=true
            RUN_DOCTESTS=true
            RUN_BENCHMARKS=true
            RUN_COVERAGE=true
            RUN_CLIPPY=true
            RUN_FORMAT_CHECK=true
            ;;
        --help)
            show_usage
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
    shift
done

# Change to workspace root
cd "$WORKSPACE_ROOT"

print_info "Starting test suite for workspace-qdrant-mcp"
print_info "Workspace root: $WORKSPACE_ROOT"

# Check if cargo is available
if ! command -v cargo &> /dev/null; then
    print_error "cargo not found. Please install Rust."
    exit 1
fi

# Build options
BUILD_OPTS=""
if [[ "$VERBOSE" == "true" ]]; then
    BUILD_OPTS="--verbose"
fi

if [[ "$FAIL_FAST" == "true" ]]; then
    BUILD_OPTS="$BUILD_OPTS --"
fi

# Create output directories
mkdir -p "$COVERAGE_DIR"
mkdir -p "$BENCHMARK_DIR"

# Function to run a command and handle errors
run_command() {
    local description=$1
    shift
    local command=("$@")

    print_info "Running: $description"

    if [[ "$VERBOSE" == "true" ]]; then
        print_info "Command: ${command[*]}"
    fi

    if "${command[@]}"; then
        print_success "$description completed successfully"
        return 0
    else
        print_error "$description failed"
        if [[ "$FAIL_FAST" == "true" ]]; then
            exit 1
        fi
        return 1
    fi
}

# Initialize error tracking
ERRORS=()

# Ensure dependencies are up to date
print_info "Updating dependencies..."
run_command "Dependency update" cargo update

# Check formatting if requested
if [[ "$RUN_FORMAT_CHECK" == "true" ]]; then
    if ! run_command "Code formatting check" cargo fmt --check; then
        ERRORS+=("Format check failed")
    fi
fi

# Run clippy if enabled
if [[ "$RUN_CLIPPY" == "true" ]]; then
    print_info "Running clippy lints..."
    CLIPPY_OPTS="--workspace --all-targets"
    if [[ "$VERBOSE" == "true" ]]; then
        CLIPPY_OPTS="$CLIPPY_OPTS --verbose"
    fi

    if ! run_command "Clippy lints" cargo clippy $CLIPPY_OPTS -- -D warnings; then
        ERRORS+=("Clippy checks failed")
    fi
fi

# Build the workspace first
print_info "Building workspace..."
if ! run_command "Workspace build" cargo build --workspace $BUILD_OPTS; then
    ERRORS+=("Build failed")
    if [[ "$FAIL_FAST" == "true" ]]; then
        print_error "Build failed, stopping early"
        exit 1
    fi
fi

# Run unit tests
if [[ "$RUN_UNIT_TESTS" == "true" ]]; then
    print_info "Running unit tests..."
    TEST_OPTS="--workspace --lib"
    if [[ "$VERBOSE" == "true" ]]; then
        TEST_OPTS="$TEST_OPTS --verbose"
    fi

    if ! run_command "Unit tests" cargo test $TEST_OPTS; then
        ERRORS+=("Unit tests failed")
    fi
fi

# Run integration tests
if [[ "$RUN_INTEGRATION_TESTS" == "true" ]]; then
    print_info "Running integration tests..."
    TEST_OPTS="--workspace --test '*'"
    if [[ "$VERBOSE" == "true" ]]; then
        TEST_OPTS="$TEST_OPTS --verbose"
    fi

    # Note: Integration tests might require external services
    if ! run_command "Integration tests" cargo test $TEST_OPTS; then
        ERRORS+=("Integration tests failed")
    fi
fi

# Run doctests
if [[ "$RUN_DOCTESTS" == "true" ]]; then
    print_info "Running doctests..."
    if ! run_command "Doctests" cargo test --workspace --doc; then
        ERRORS+=("Doctests failed")
    fi
fi

# Run benchmarks
if [[ "$RUN_BENCHMARKS" == "true" ]]; then
    print_info "Running benchmarks..."

    # Check if criterion is available
    if cargo bench --help &> /dev/null; then
        BENCH_OPTS="--workspace"
        if [[ "$VERBOSE" == "true" ]]; then
            BENCH_OPTS="$BENCH_OPTS --verbose"
        fi

        if ! run_command "Benchmarks" cargo bench $BENCH_OPTS; then
            ERRORS+=("Benchmarks failed")
        fi

        # Move benchmark results to our directory
        if [[ -d "target/criterion" ]]; then
            cp -r target/criterion/* "$BENCHMARK_DIR/" 2>/dev/null || true
            print_success "Benchmark results saved to $BENCHMARK_DIR"
        fi
    else
        print_warning "Benchmarks not available (criterion not found)"
    fi
fi

# Generate coverage report
if [[ "$RUN_COVERAGE" == "true" ]]; then
    print_info "Generating coverage report..."

    # Check if tarpaulin is installed
    if command -v cargo-tarpaulin &> /dev/null; then
        TARPAULIN_OPTS="--config tarpaulin.toml"
        if [[ "$VERBOSE" == "true" ]]; then
            TARPAULIN_OPTS="$TARPAULIN_OPTS --verbose"
        fi

        if ! run_command "Coverage generation" cargo tarpaulin $TARPAULIN_OPTS; then
            ERRORS+=("Coverage generation failed")
        else
            print_success "Coverage report generated in $COVERAGE_DIR"
        fi
    else
        print_warning "cargo-tarpaulin not installed. Install with: cargo install cargo-tarpaulin"
        ERRORS+=("Coverage tool not available")
    fi
fi

# Summary
print_info "Test suite completed"

if [[ ${#ERRORS[@]} -eq 0 ]]; then
    print_success "All checks passed! ✨"
    exit 0
else
    print_error "Some checks failed:"
    for error in "${ERRORS[@]}"; do
        print_error "  - $error"
    done
    exit 1
fi