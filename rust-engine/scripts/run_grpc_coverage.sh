#!/bin/bash
# Script to run gRPC protocol integration tests with coverage measurement

set -e

echo "==================================="
echo "gRPC Protocol Integration Testing"
echo "==================================="

echo "Checking cargo tarpaulin availability..."
if ! command -v cargo-tarpaulin &> /dev/null; then
    echo "Installing cargo-tarpaulin..."
    cargo install cargo-tarpaulin
fi

echo "Running gRPC protocol integration tests with coverage..."

# Run basic protocol tests
echo "Running basic gRPC protocol tests..."
cargo tarpaulin \
    --test grpc_protocol_basic \
    --out Html \
    --output-dir target/tarpaulin-grpc-basic \
    --verbose \
    --timeout 120 \
    --follow-exec \
    --force-clean

# Run comprehensive protocol tests
echo "Running comprehensive gRPC protocol tests..."
cargo tarpaulin \
    --test grpc_protocol_comprehensive \
    --out Html \
    --output-dir target/tarpaulin-grpc-comprehensive \
    --verbose \
    --timeout 180 \
    --follow-exec \
    --force-clean

# Run combined coverage for gRPC modules
echo "Running combined gRPC module coverage..."
cargo tarpaulin \
    --test grpc_protocol_basic \
    --test grpc_protocol_comprehensive \
    --test grpc_server \
    --test grpc_middleware \
    --out Html \
    --output-dir target/tarpaulin-grpc-combined \
    --verbose \
    --timeout 300 \
    --follow-exec \
    --force-clean \
    --exclude-files="tests/*" \
    --include="src/grpc/*"

echo "Generating coverage summary..."
echo "Coverage reports generated:"
echo "  - Basic tests: target/tarpaulin-grpc-basic/tarpaulin-report.html"
echo "  - Comprehensive tests: target/tarpaulin-grpc-comprehensive/tarpaulin-report.html"
echo "  - Combined gRPC coverage: target/tarpaulin-grpc-combined/tarpaulin-report.html"

echo "==================================="
echo "gRPC Protocol Testing Complete"
echo "==================================="