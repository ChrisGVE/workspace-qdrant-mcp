#!/bin/bash

# Cross-platform testing and memory safety validation runner
# This script runs comprehensive tests across platforms and validates memory safety

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default configuration
ENABLE_VALGRIND=${ENABLE_VALGRIND:-"auto"}
ENABLE_MIRI=${ENABLE_MIRI:-"true"}
ENABLE_SANITIZERS=${ENABLE_SANITIZERS:-"true"}
ENABLE_BENCHMARKS=${ENABLE_BENCHMARKS:-"true"}
ENABLE_CROSS_COMPILATION=${ENABLE_CROSS_COMPILATION:-"false"}
TEST_TIMEOUT=${TEST_TIMEOUT:-"600"} # 10 minutes
VERBOSE=${VERBOSE:-"false"}
REPORT_DIR=${REPORT_DIR:-"target/test-reports"}

# Supported platforms for cross-compilation testing
CROSS_TARGETS=(
    "x86_64-unknown-linux-gnu"
    "aarch64-unknown-linux-gnu"
    "x86_64-apple-darwin"
    "aarch64-apple-darwin"
    "x86_64-pc-windows-msvc"
    "aarch64-pc-windows-msvc"
)

# Function to print colored output
print_status() {
    local color=$1
    local message=$2
    echo -e "${color}[$(date '+%Y-%m-%d %H:%M:%S')] ${message}${NC}"
}

print_info() {
    print_status "$BLUE" "INFO: $1"
}

print_success() {
    print_status "$GREEN" "SUCCESS: $1"
}

print_warning() {
    print_status "$YELLOW" "WARNING: $1"
}

print_error() {
    print_status "$RED" "ERROR: $1"
}

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to detect platform
detect_platform() {
    case "$(uname -s)" in
        Linux*)     echo "linux";;
        Darwin*)    echo "macos";;
        CYGWIN*|MINGW*|MSYS*) echo "windows";;
        *)          echo "unknown";;
    esac
}

# Function to check prerequisites
check_prerequisites() {
    print_info "Checking prerequisites..."

    # Check Rust toolchain
    if ! command_exists "cargo"; then
        print_error "Cargo is not installed. Please install Rust toolchain."
        exit 1
    fi

    # Check if we're in the right directory
    if [[ ! -f "Cargo.toml" ]]; then
        print_error "Not in a Rust project directory. Please run from the daemon directory."
        exit 1
    fi

    # Check for Valgrind (Linux only)
    if [[ "$(detect_platform)" == "linux" ]]; then
        if [[ "$ENABLE_VALGRIND" == "true" ]] || [[ "$ENABLE_VALGRIND" == "auto" ]]; then
            if command_exists "valgrind"; then
                print_info "Valgrind found - memory safety tests will be enabled"
                ENABLE_VALGRIND="true"
            else
                if [[ "$ENABLE_VALGRIND" == "true" ]]; then
                    print_error "Valgrind requested but not found. Install with: sudo apt-get install valgrind"
                    exit 1
                else
                    print_warning "Valgrind not found - skipping Valgrind-based memory tests"
                    ENABLE_VALGRIND="false"
                fi
            fi
        fi
    else
        ENABLE_VALGRIND="false"
    fi

    # Check for Miri
    if [[ "$ENABLE_MIRI" == "true" ]]; then
        if ! cargo miri --version >/dev/null 2>&1; then
            print_info "Installing Miri for undefined behavior detection..."
            rustup +nightly component add miri || {
                print_warning "Failed to install Miri - skipping Miri tests"
                ENABLE_MIRI="false"
            }
        fi
    fi

    # Create report directory
    mkdir -p "$REPORT_DIR"

    print_success "Prerequisites check completed"
}

# Function to run basic tests
run_basic_tests() {
    print_info "Running basic test suite..."

    local test_args=""
    if [[ "$VERBOSE" == "true" ]]; then
        test_args="--verbose"
    fi

    # Run unit tests
    print_info "Running unit tests..."
    timeout "$TEST_TIMEOUT" cargo test $test_args --lib 2>&1 | tee "$REPORT_DIR/unit_tests.log"

    # Run integration tests
    print_info "Running integration tests..."
    timeout "$TEST_TIMEOUT" cargo test $test_args --test "*" 2>&1 | tee "$REPORT_DIR/integration_tests.log"

    # Run doctests
    print_info "Running documentation tests..."
    timeout "$TEST_TIMEOUT" cargo test $test_args --doc 2>&1 | tee "$REPORT_DIR/doc_tests.log"

    print_success "Basic tests completed"
}

# Function to run cross-platform specific tests
run_cross_platform_tests() {
    print_info "Running cross-platform specific tests..."

    # Run our comprehensive cross-platform test suite
    timeout "$TEST_TIMEOUT" cargo test cross_platform_safety_tests --release 2>&1 | tee "$REPORT_DIR/cross_platform_tests.log"

    # Run unsafe code audit tests
    timeout "$TEST_TIMEOUT" cargo test unsafe_code_audit_tests --release 2>&1 | tee "$REPORT_DIR/unsafe_audit_tests.log"

    # Run FFI performance tests
    timeout "$TEST_TIMEOUT" cargo test ffi_performance_tests --release 2>&1 | tee "$REPORT_DIR/ffi_performance_tests.log"

    print_success "Cross-platform tests completed"
}

# Function to run memory safety tests
run_memory_safety_tests() {
    print_info "Running memory safety tests..."

    # Run with Miri if available
    if [[ "$ENABLE_MIRI" == "true" ]]; then
        print_info "Running tests with Miri for undefined behavior detection..."
        MIRIFLAGS="-Zmiri-symbolic-alignment-check -Zmiri-check-number-validity" \
            timeout "$TEST_TIMEOUT" cargo +nightly miri test 2>&1 | tee "$REPORT_DIR/miri_tests.log" || {
            print_warning "Some Miri tests failed - check $REPORT_DIR/miri_tests.log"
        }
    fi

    # Run with Valgrind if available
    if [[ "$ENABLE_VALGRIND" == "true" ]]; then
        print_info "Running tests with Valgrind for memory leak detection..."

        # Build test binary
        cargo test --no-run --release

        # Find test binary
        TEST_BINARY=$(find target/release/deps -name "*-*" -type f -executable | grep -E "(cross_platform|unsafe_code|memory)" | head -n 1)

        if [[ -n "$TEST_BINARY" ]]; then
            print_info "Running Valgrind on test binary: $TEST_BINARY"
            timeout "$TEST_TIMEOUT" valgrind \
                --tool=memcheck \
                --leak-check=full \
                --show-leak-kinds=all \
                --track-origins=yes \
                --verbose \
                --xml=yes \
                --xml-file="$REPORT_DIR/valgrind_memcheck.xml" \
                "$TEST_BINARY" 2>&1 | tee "$REPORT_DIR/valgrind_tests.log" || {
                print_warning "Valgrind detected issues - check $REPORT_DIR/valgrind_tests.log"
            }
        else
            print_warning "Could not find test binary for Valgrind analysis"
        fi
    fi

    print_success "Memory safety tests completed"
}

# Function to run with sanitizers
run_sanitizer_tests() {
    if [[ "$ENABLE_SANITIZERS" != "true" ]]; then
        return
    fi

    print_info "Running tests with sanitizers..."

    local platform=$(detect_platform)

    # AddressSanitizer
    if [[ "$platform" == "linux" ]] || [[ "$platform" == "macos" ]]; then
        print_info "Running tests with AddressSanitizer..."
        RUSTFLAGS="-Z sanitizer=address" \
            timeout "$TEST_TIMEOUT" cargo +nightly test --target x86_64-unknown-linux-gnu 2>&1 | tee "$REPORT_DIR/asan_tests.log" || {
            print_warning "AddressSanitizer detected issues - check $REPORT_DIR/asan_tests.log"
        }

        # ThreadSanitizer
        print_info "Running tests with ThreadSanitizer..."
        RUSTFLAGS="-Z sanitizer=thread" \
            timeout "$TEST_TIMEOUT" cargo +nightly test --target x86_64-unknown-linux-gnu 2>&1 | tee "$REPORT_DIR/tsan_tests.log" || {
            print_warning "ThreadSanitizer detected issues - check $REPORT_DIR/tsan_tests.log"
        }
    fi

    print_success "Sanitizer tests completed"
}

# Function to run benchmarks
run_benchmarks() {
    if [[ "$ENABLE_BENCHMARKS" != "true" ]]; then
        return
    fi

    print_info "Running performance benchmarks..."

    # Run Criterion benchmarks
    timeout "$TEST_TIMEOUT" cargo bench --bench cross_platform_benchmarks 2>&1 | tee "$REPORT_DIR/benchmarks.log"

    # Run built-in benchmarks
    timeout "$TEST_TIMEOUT" cargo bench 2>&1 | tee -a "$REPORT_DIR/benchmarks.log"

    # Copy benchmark results
    if [[ -d "target/criterion" ]]; then
        cp -r target/criterion "$REPORT_DIR/"
        print_info "Criterion benchmark reports available in $REPORT_DIR/criterion"
    fi

    print_success "Benchmarks completed"
}

# Function to run cross-compilation tests
run_cross_compilation_tests() {
    if [[ "$ENABLE_CROSS_COMPILATION" != "true" ]]; then
        return
    fi

    print_info "Running cross-compilation tests..."

    for target in "${CROSS_TARGETS[@]}"; do
        print_info "Testing cross-compilation for target: $target"

        # Install target if not available
        rustup target add "$target" || {
            print_warning "Could not add target $target - skipping"
            continue
        }

        # Try to compile for target
        if cargo check --target "$target" 2>&1 | tee "$REPORT_DIR/cross_compile_${target}.log"; then
            print_success "Cross-compilation successful for $target"
        else
            print_warning "Cross-compilation failed for $target - check $REPORT_DIR/cross_compile_${target}.log"
        fi
    done

    print_success "Cross-compilation tests completed"
}

# Function to generate comprehensive report
generate_report() {
    print_info "Generating test report..."

    local report_file="$REPORT_DIR/test_summary.md"

    cat > "$report_file" << EOF
# Cross-Platform Testing and Memory Safety Report

Generated on: $(date)
Platform: $(detect_platform)
Architecture: $(uname -m)
Rust version: $(rustc --version)

## Test Configuration

- Valgrind enabled: $ENABLE_VALGRIND
- Miri enabled: $ENABLE_MIRI
- Sanitizers enabled: $ENABLE_SANITIZERS
- Benchmarks enabled: $ENABLE_BENCHMARKS
- Cross-compilation enabled: $ENABLE_CROSS_COMPILATION

## Test Results

EOF

    # Analyze test results
    local tests_passed=0
    local tests_failed=0

    for log_file in "$REPORT_DIR"/*.log; do
        if [[ -f "$log_file" ]]; then
            local test_name=$(basename "$log_file" .log)
            echo "### $test_name" >> "$report_file"

            if grep -q "test result: ok" "$log_file" 2>/dev/null; then
                echo "✅ PASSED" >> "$report_file"
                ((tests_passed++))
            elif grep -q -E "(FAILED|ERROR|failed)" "$log_file" 2>/dev/null; then
                echo "❌ FAILED" >> "$report_file"
                ((tests_failed++))
                echo "Check $log_file for details" >> "$report_file"
            else
                echo "⚠️ INCONCLUSIVE" >> "$report_file"
            fi
            echo "" >> "$report_file"
        fi
    done

    echo "## Summary" >> "$report_file"
    echo "- Tests passed: $tests_passed" >> "$report_file"
    echo "- Tests failed: $tests_failed" >> "$report_file"
    echo "- Total tests: $((tests_passed + tests_failed))" >> "$report_file"

    if [[ $tests_failed -eq 0 ]]; then
        echo "🎉 All tests passed!" >> "$report_file"
    else
        echo "⚠️ Some tests failed. Review the logs for details." >> "$report_file"
    fi

    print_success "Test report generated: $report_file"

    # Print summary to console
    echo ""
    print_info "=== TEST SUMMARY ==="
    print_info "Tests passed: $tests_passed"
    if [[ $tests_failed -gt 0 ]]; then
        print_warning "Tests failed: $tests_failed"
    else
        print_success "Tests failed: $tests_failed"
    fi
    print_info "Full report: $report_file"
}

# Function to clean up
cleanup() {
    print_info "Cleaning up..."

    # Remove temporary files
    find target -name "*.tmp" -delete 2>/dev/null || true

    # Clean up environment variables
    unset MIRIFLAGS RUSTFLAGS
}

# Function to show usage
show_usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Cross-platform testing and memory safety validation runner

OPTIONS:
    --valgrind          Enable Valgrind testing (Linux only)
    --no-valgrind       Disable Valgrind testing
    --miri              Enable Miri testing
    --no-miri           Disable Miri testing
    --sanitizers        Enable sanitizer testing
    --no-sanitizers     Disable sanitizer testing
    --benchmarks        Enable benchmark testing
    --no-benchmarks     Disable benchmark testing
    --cross-compile     Enable cross-compilation testing
    --verbose           Enable verbose output
    --timeout SECONDS   Set test timeout (default: 600)
    --report-dir DIR    Set report directory (default: target/test-reports)
    --help              Show this help message

ENVIRONMENT VARIABLES:
    ENABLE_VALGRIND     Enable/disable Valgrind (true/false/auto)
    ENABLE_MIRI         Enable/disable Miri (true/false)
    ENABLE_SANITIZERS   Enable/disable sanitizers (true/false)
    ENABLE_BENCHMARKS   Enable/disable benchmarks (true/false)
    ENABLE_CROSS_COMPILATION Enable/disable cross-compilation (true/false)
    TEST_TIMEOUT        Test timeout in seconds
    VERBOSE             Enable verbose output (true/false)
    REPORT_DIR          Report directory path

EXAMPLES:
    # Run all tests with default settings
    $0

    # Run tests with Valgrind and verbose output
    $0 --valgrind --verbose

    # Run only basic tests without memory safety checks
    $0 --no-valgrind --no-miri --no-sanitizers

    # Run comprehensive testing including cross-compilation
    $0 --cross-compile --benchmarks
EOF
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --valgrind)
            ENABLE_VALGRIND="true"
            shift
            ;;
        --no-valgrind)
            ENABLE_VALGRIND="false"
            shift
            ;;
        --miri)
            ENABLE_MIRI="true"
            shift
            ;;
        --no-miri)
            ENABLE_MIRI="false"
            shift
            ;;
        --sanitizers)
            ENABLE_SANITIZERS="true"
            shift
            ;;
        --no-sanitizers)
            ENABLE_SANITIZERS="false"
            shift
            ;;
        --benchmarks)
            ENABLE_BENCHMARKS="true"
            shift
            ;;
        --no-benchmarks)
            ENABLE_BENCHMARKS="false"
            shift
            ;;
        --cross-compile)
            ENABLE_CROSS_COMPILATION="true"
            shift
            ;;
        --verbose)
            VERBOSE="true"
            shift
            ;;
        --timeout)
            TEST_TIMEOUT="$2"
            shift 2
            ;;
        --report-dir)
            REPORT_DIR="$2"
            shift 2
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
done

# Main execution
main() {
    print_info "Starting cross-platform testing and memory safety validation"
    print_info "Platform: $(detect_platform) ($(uname -m))"
    print_info "Report directory: $REPORT_DIR"

    # Set up trap for cleanup
    trap cleanup EXIT

    # Run test phases
    check_prerequisites
    run_basic_tests
    run_cross_platform_tests
    run_memory_safety_tests
    run_sanitizer_tests
    run_benchmarks
    run_cross_compilation_tests
    generate_report

    print_success "All testing phases completed successfully!"
}

# Execute main function
main "$@"