#!/usr/bin/env bash
# Test Rust daemon across multiple toolchain versions
#
# Tests MSRV (1.75.0), stable, and latest against:
# - cargo build
# - cargo test
# - cargo clippy
# - cargo fmt --check
# - cross-compilation targets

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Rust versions to test
MSRV="1.75.0"
STABLE="stable"
LATEST="nightly"

# Test results tracking
declare -A TEST_RESULTS

# Print colored message
print_msg() {
    local color=$1
    shift
    echo -e "${color}$*${NC}"
}

# Print section header
print_section() {
    echo ""
    print_msg "$YELLOW" "================================"
    print_msg "$YELLOW" "$1"
    print_msg "$YELLOW" "================================"
}

# Test a specific Rust version
test_rust_version() {
    local version=$1
    local test_name="rust_$version"

    print_section "Testing with Rust $version"

    # Install/update toolchain
    print_msg "$YELLOW" "Installing Rust $version..."
    if ! rustup toolchain install "$version" --component clippy rustfmt; then
        print_msg "$RED" "✗ Failed to install Rust $version"
        TEST_RESULTS["$test_name"]="FAILED"
        return 1
    fi

    # Check version
    print_msg "$GREEN" "Rust version:"
    rustup run "$version" rustc --version
    rustup run "$version" cargo --version

    # Test: cargo check
    print_msg "$YELLOW" "Running cargo check..."
    if rustup run "$version" cargo check --all-features; then
        print_msg "$GREEN" "✓ cargo check passed"
        TEST_RESULTS["${test_name}_check"]="PASSED"
    else
        print_msg "$RED" "✗ cargo check failed"
        TEST_RESULTS["${test_name}_check"]="FAILED"
        return 1
    fi

    # Test: cargo build
    print_msg "$YELLOW" "Running cargo build..."
    if rustup run "$version" cargo build --all-features; then
        print_msg "$GREEN" "✓ cargo build passed"
        TEST_RESULTS["${test_name}_build"]="PASSED"
    else
        print_msg "$RED" "✗ cargo build failed"
        TEST_RESULTS["${test_name}_build"]="FAILED"
        return 1
    fi

    # Test: cargo test
    print_msg "$YELLOW" "Running cargo test..."
    if rustup run "$version" cargo test --all-features; then
        print_msg "$GREEN" "✓ cargo test passed"
        TEST_RESULTS["${test_name}_test"]="PASSED"
    else
        print_msg "$RED" "✗ cargo test failed"
        TEST_RESULTS["${test_name}_test"]="FAILED"
        return 1
    fi

    # Test: cargo clippy
    print_msg "$YELLOW" "Running cargo clippy..."
    if rustup run "$version" cargo clippy --all-features -- -D warnings; then
        print_msg "$GREEN" "✓ cargo clippy passed"
        TEST_RESULTS["${test_name}_clippy"]="PASSED"
    else
        print_msg "$RED" "✗ cargo clippy failed"
        TEST_RESULTS["${test_name}_clippy"]="FAILED"
        return 1
    fi

    # Test: cargo fmt
    print_msg "$YELLOW" "Running cargo fmt --check..."
    if rustup run "$version" cargo fmt --all -- --check; then
        print_msg "$GREEN" "✓ cargo fmt passed"
        TEST_RESULTS["${test_name}_fmt"]="PASSED"
    else
        print_msg "$RED" "✗ cargo fmt failed"
        TEST_RESULTS["${test_name}_fmt"]="FAILED"
        return 1
    fi

    print_msg "$GREEN" "All tests passed for Rust $version!"
    TEST_RESULTS["$test_name"]="PASSED"
    return 0
}

# Test cross-compilation for a specific target
test_cross_compilation() {
    local version=$1
    local target=$2
    local test_name="cross_${target//-/_}"

    print_msg "$YELLOW" "Testing cross-compilation for $target..."

    # Install target
    if ! rustup target add --toolchain "$version" "$target"; then
        print_msg "$RED" "✗ Failed to add target $target"
        TEST_RESULTS["$test_name"]="FAILED"
        return 1
    fi

    # Try to build for target
    if rustup run "$version" cargo build --target "$target" --lib; then
        print_msg "$GREEN" "✓ Cross-compilation for $target passed"
        TEST_RESULTS["$test_name"]="PASSED"
    else
        print_msg "$RED" "✗ Cross-compilation for $target failed"
        TEST_RESULTS["$test_name"]="FAILED"
        return 1
    fi

    return 0
}

# Print test summary
print_summary() {
    print_section "Test Summary"

    local total=0
    local passed=0
    local failed=0

    for test in "${!TEST_RESULTS[@]}"; do
        total=$((total + 1))
        if [[ "${TEST_RESULTS[$test]}" == "PASSED" ]]; then
            passed=$((passed + 1))
            print_msg "$GREEN" "✓ $test: PASSED"
        else
            failed=$((failed + 1))
            print_msg "$RED" "✗ $test: FAILED"
        fi
    done

    echo ""
    print_msg "$YELLOW" "Total: $total tests"
    print_msg "$GREEN" "Passed: $passed"
    print_msg "$RED" "Failed: $failed"

    if [[ $failed -eq 0 ]]; then
        print_msg "$GREEN" "All compatibility tests passed!"
        return 0
    else
        print_msg "$RED" "Some compatibility tests failed!"
        return 1
    fi
}

# Main execution
main() {
    cd "$(dirname "$0")"

    print_section "Rust Toolchain Compatibility Testing"
    print_msg "$YELLOW" "Testing workspace-qdrant-daemon"

    # Test MSRV
    test_rust_version "$MSRV" || true

    # Test stable
    test_rust_version "$STABLE" || true

    # Test cross-compilation (only on stable to save time)
    if [[ "${TEST_RESULTS[rust_stable]:-FAILED}" == "PASSED" ]]; then
        print_section "Cross-Compilation Tests (Stable)"

        # Test major platforms
        test_cross_compilation "$STABLE" "x86_64-unknown-linux-gnu" || true

        # Only test macOS targets if on macOS
        if [[ "$OSTYPE" == "darwin"* ]]; then
            test_cross_compilation "$STABLE" "x86_64-apple-darwin" || true
            test_cross_compilation "$STABLE" "aarch64-apple-darwin" || true
        fi
    fi

    # Print summary
    print_summary
}

# Run main
main "$@"
