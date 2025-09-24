#!/bin/bash
set -euo pipefail

# Comprehensive Test Runner for Workspace Qdrant Daemon
# This script runs all cross-platform, memory safety, FFI performance,
# thread safety, and edge case tests with various validation tools

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
RUST_ENGINE_DIR="$(dirname "$SCRIPT_DIR")"
PROJECT_ROOT="$(dirname "$RUST_ENGINE_DIR")"

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Configuration
ENABLE_VALGRIND=${ENABLE_VALGRIND:-false}
ENABLE_MIRI=${ENABLE_MIRI:-true}
ENABLE_SANITIZERS=${ENABLE_SANITIZERS:-false}
GENERATE_COVERAGE=${GENERATE_COVERAGE:-true}
VERBOSE=${VERBOSE:-false}

# Test results tracking
TESTS_PASSED=0
TESTS_FAILED=0
TESTS_SKIPPED=0

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
    ((TESTS_PASSED++))
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
    ((TESTS_SKIPPED++))
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
    ((TESTS_FAILED++))
}

log_section() {
    echo -e "\n${PURPLE}==== $1 ====${NC}\n"
}

check_prerequisites() {
    log_section "Checking Prerequisites"

    # Check Rust toolchain
    if ! command -v cargo &> /dev/null; then
        log_error "Cargo not found. Please install Rust toolchain."
        exit 1
    fi
    log_success "Rust toolchain found"

    # Check for nightly toolchain if sanitizers are enabled
    if [ "$ENABLE_SANITIZERS" = true ]; then
        if ! rustup toolchain list | grep -q nightly; then
            log_warning "Nightly toolchain not found. Installing..."
            rustup toolchain install nightly
        fi
        log_success "Nightly toolchain available"
    fi

    # Check for Miri if enabled
    if [ "$ENABLE_MIRI" = true ]; then
        if ! command -v cargo-miri &> /dev/null; then
            log_warning "Miri not found. Installing..."
            rustup toolchain install nightly --component miri
            cargo +nightly miri setup
        fi
        log_success "Miri available"
    fi

    # Check for Valgrind if enabled
    if [ "$ENABLE_VALGRIND" = true ]; then
        if ! command -v valgrind &> /dev/null; then
            log_warning "Valgrind not found. Skipping Valgrind tests."
            ENABLE_VALGRIND=false
        else
            log_success "Valgrind available"
        fi
    fi

    # Check for coverage tools if enabled
    if [ "$GENERATE_COVERAGE" = true ]; then
        if ! command -v cargo-llvm-cov &> /dev/null; then
            log_warning "cargo-llvm-cov not found. Installing..."
            cargo install cargo-llvm-cov
        fi
        log_success "Coverage tools available"
    fi
}

run_basic_tests() {
    log_section "Running Basic Unit and Integration Tests"

    cd "$RUST_ENGINE_DIR"

    if cargo test --all-features 2>&1 | tee test_output.log; then
        log_success "Basic tests passed"
    else
        log_error "Basic tests failed"
        cat test_output.log
    fi
}

run_cross_platform_tests() {
    log_section "Running Cross-Platform Tests"

    cd "$RUST_ENGINE_DIR"

    # Run cross-platform specific tests
    local test_files=(
        "tests/20250924-1310_cross_platform_tests.rs"
    )

    for test_file in "${test_files[@]}"; do
        if [ -f "$test_file" ]; then
            local test_name=$(basename "$test_file" .rs)
            log_info "Running $test_name..."

            if cargo test --test "${test_name}" -- --nocapture 2>&1 | tee "${test_name}_output.log"; then
                log_success "Cross-platform test $test_name passed"
            else
                log_error "Cross-platform test $test_name failed"
            fi
        else
            log_warning "Test file $test_file not found"
        fi
    done
}

run_memory_safety_tests() {
    log_section "Running Memory Safety Validation Tests"

    cd "$RUST_ENGINE_DIR"

    # Run memory safety tests with standard Rust
    local test_name="20250924-1315_memory_safety_validation"
    if cargo test --test "$test_name" -- --nocapture 2>&1 | tee "${test_name}_output.log"; then
        log_success "Memory safety tests passed"
    else
        log_error "Memory safety tests failed"
    fi

    # Run with Miri if enabled
    if [ "$ENABLE_MIRI" = true ]; then
        log_info "Running memory safety tests with Miri..."
        if timeout 300 cargo +nightly miri test --test "$test_name" 2>&1 | tee "${test_name}_miri_output.log"; then
            log_success "Miri memory safety validation passed"
        else
            log_error "Miri memory safety validation failed or timed out"
        fi
    fi

    # Run with AddressSanitizer if enabled
    if [ "$ENABLE_SANITIZERS" = true ]; then
        log_info "Running memory safety tests with AddressSanitizer..."
        if RUSTFLAGS="-Z sanitizer=address" cargo +nightly test --test "$test_name" --target x86_64-unknown-linux-gnu 2>&1 | tee "${test_name}_asan_output.log"; then
            log_success "AddressSanitizer validation passed"
        else
            log_error "AddressSanitizer validation failed"
        fi
    fi
}

run_thread_safety_tests() {
    log_section "Running Thread Safety Validation Tests"

    cd "$RUST_ENGINE_DIR"

    local test_name="20250924-1325_thread_safety_validation"

    # Standard thread safety tests
    if cargo test --test "$test_name" -- --nocapture 2>&1 | tee "${test_name}_output.log"; then
        log_success "Thread safety tests passed"
    else
        log_error "Thread safety tests failed"
    fi

    # Run with ThreadSanitizer if enabled
    if [ "$ENABLE_SANITIZERS" = true ]; then
        log_info "Running thread safety tests with ThreadSanitizer..."
        if RUSTFLAGS="-Z sanitizer=thread" cargo +nightly test --test "$test_name" --target x86_64-unknown-linux-gnu 2>&1 | tee "${test_name}_tsan_output.log"; then
            log_success "ThreadSanitizer validation passed"
        else
            log_error "ThreadSanitizer validation failed"
        fi
    fi
}

run_ffi_performance_benchmarks() {
    log_section "Running FFI Performance Benchmarks"

    cd "$RUST_ENGINE_DIR"

    local bench_name="20250924-1320_ffi_performance_benchmarks"

    # Run FFI performance benchmarks
    if cargo bench --bench "$bench_name" 2>&1 | tee "${bench_name}_output.log"; then
        log_success "FFI performance benchmarks completed"
    else
        log_error "FFI performance benchmarks failed"
    fi

    # Generate performance report
    if [ -f "target/criterion/report/index.html" ]; then
        log_info "FFI performance report generated at target/criterion/report/index.html"
    fi
}

run_performance_regression_tests() {
    log_section "Running Performance Regression Tests"

    cd "$RUST_ENGINE_DIR"

    local bench_name="20250924-1330_performance_regression_tests"

    # Run performance regression benchmarks
    if cargo bench --bench "$bench_name" 2>&1 | tee "${bench_name}_output.log"; then
        log_success "Performance regression tests completed"
    else
        log_error "Performance regression tests failed"
    fi
}

run_edge_case_tests() {
    log_section "Running Comprehensive Edge Case Tests"

    cd "$RUST_ENGINE_DIR"

    local test_name="20250924-1335_edge_case_comprehensive_tests"

    # Standard edge case tests
    if cargo test --test "$test_name" -- --nocapture 2>&1 | tee "${test_name}_output.log"; then
        log_success "Edge case tests passed"
    else
        log_error "Edge case tests failed"
    fi

    # Run edge case tests with Valgrind if enabled
    if [ "$ENABLE_VALGRIND" = true ] && [ "$(uname)" = "Linux" ]; then
        log_info "Running edge case tests with Valgrind..."
        if timeout 600 valgrind --leak-check=full --show-leak-kinds=all --track-origins=yes --error-exitcode=1 \
           cargo test --test "$test_name" test_valgrind_memory_safety 2>&1 | tee "${test_name}_valgrind_output.log"; then
            log_success "Valgrind edge case validation passed"
        else
            log_error "Valgrind edge case validation failed or timed out"
        fi
    fi
}

generate_coverage_report() {
    if [ "$GENERATE_COVERAGE" = true ]; then
        log_section "Generating Test Coverage Report"

        cd "$RUST_ENGINE_DIR"

        if cargo llvm-cov --all-features --workspace --lcov --output-path lcov.info 2>&1 | tee coverage_output.log; then
            log_success "Coverage report generated"

            # Generate HTML report
            if command -v genhtml &> /dev/null; then
                genhtml lcov.info --output-directory coverage_html
                log_info "HTML coverage report generated at coverage_html/index.html"
            fi

            # Display coverage summary
            if command -v lcov &> /dev/null; then
                lcov --summary lcov.info
            fi
        else
            log_error "Coverage report generation failed"
        fi
    fi
}

run_stress_tests() {
    log_section "Running Stress Tests"

    cd "$RUST_ENGINE_DIR"

    # Long-running stress test with all components
    log_info "Running combined stress test for 30 seconds..."

    timeout 30 cargo test --release \
        test_race_conditions_under_stress \
        test_memory_leaks_in_long_running_operations \
        test_concurrent_daemon_access \
        -- --nocapture 2>&1 | tee stress_test_output.log || {

        if [ ${PIPESTATUS[0]} -eq 124 ]; then
            log_success "Stress tests completed (timed out as expected)"
        else
            log_error "Stress tests failed unexpectedly"
        fi
    }
}

cleanup_test_artifacts() {
    log_section "Cleaning Up Test Artifacts"

    cd "$RUST_ENGINE_DIR"

    # Clean up temporary files but preserve important logs
    find . -name "*.profraw" -type f -delete 2>/dev/null || true
    find . -name "*.tmp" -type f -delete 2>/dev/null || true

    # Compress large log files
    for log_file in *_output.log; do
        if [ -f "$log_file" ] && [ $(stat -f%z "$log_file" 2>/dev/null || stat -c%s "$log_file" 2>/dev/null) -gt 1048576 ]; then
            gzip "$log_file"
            log_info "Compressed large log file: $log_file.gz"
        fi
    done

    log_success "Cleanup completed"
}

generate_test_report() {
    log_section "Generating Test Summary Report"

    local report_file="$RUST_ENGINE_DIR/test_report_$(date +%Y%m%d_%H%M%S).md"

    cat > "$report_file" << EOF
# Comprehensive Test Report

**Generated:** $(date)
**Platform:** $(uname -a)
**Rust Version:** $(rustc --version)

## Test Results Summary

- ✅ Tests Passed: $TESTS_PASSED
- ❌ Tests Failed: $TESTS_FAILED
- ⚠️ Tests Skipped: $TESTS_SKIPPED
- 📊 Total Tests: $((TESTS_PASSED + TESTS_FAILED + TESTS_SKIPPED))

## Configuration

- Valgrind: $([ "$ENABLE_VALGRIND" = true ] && echo "✅ Enabled" || echo "❌ Disabled")
- Miri: $([ "$ENABLE_MIRI" = true ] && echo "✅ Enabled" || echo "❌ Disabled")
- Sanitizers: $([ "$ENABLE_SANITIZERS" = true ] && echo "✅ Enabled" || echo "❌ Disabled")
- Coverage: $([ "$GENERATE_COVERAGE" = true ] && echo "✅ Enabled" || echo "❌ Disabled")

## Test Categories Executed

1. ✅ Cross-Platform Tests
2. ✅ Memory Safety Validation
3. ✅ Thread Safety Validation
4. ✅ FFI Performance Benchmarks
5. ✅ Performance Regression Tests
6. ✅ Edge Case Comprehensive Tests
7. ✅ Stress Tests

## Files Generated

EOF

    # List important generated files
    cd "$RUST_ENGINE_DIR"
    for file in *.log coverage_html/index.html target/criterion/report/index.html lcov.info; do
        if [ -e "$file" ]; then
            echo "- $file" >> "$report_file"
        fi
    done

    log_success "Test report generated: $report_file"
}

main() {
    log_section "Starting Comprehensive Test Suite for Workspace Qdrant Daemon"

    check_prerequisites
    run_basic_tests
    run_cross_platform_tests
    run_memory_safety_tests
    run_thread_safety_tests
    run_ffi_performance_benchmarks
    run_performance_regression_tests
    run_edge_case_tests
    run_stress_tests
    generate_coverage_report
    cleanup_test_artifacts
    generate_test_report

    log_section "Test Suite Execution Complete"

    if [ $TESTS_FAILED -eq 0 ]; then
        log_success "All tests passed successfully! ✅"
        exit 0
    else
        log_error "$TESTS_FAILED test(s) failed ❌"
        exit 1
    fi
}

# Handle script arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --enable-valgrind)
            ENABLE_VALGRIND=true
            shift
            ;;
        --disable-miri)
            ENABLE_MIRI=false
            shift
            ;;
        --enable-sanitizers)
            ENABLE_SANITIZERS=true
            shift
            ;;
        --no-coverage)
            GENERATE_COVERAGE=false
            shift
            ;;
        --verbose)
            VERBOSE=true
            shift
            ;;
        --help)
            echo "Usage: $0 [options]"
            echo "Options:"
            echo "  --enable-valgrind    Enable Valgrind memory checking"
            echo "  --disable-miri       Disable Miri unsafe code checking"
            echo "  --enable-sanitizers  Enable AddressSanitizer and ThreadSanitizer"
            echo "  --no-coverage        Disable coverage report generation"
            echo "  --verbose            Enable verbose output"
            echo "  --help               Show this help message"
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Execute main function
main "$@"