#!/usr/bin/env bash
#
# CLI Startup Performance Benchmarks
#
# Compares Rust CLI (wqm) startup time against Python CLI (wqm-py)
# to validate the <100ms target for Rust CLI.
#
# Requirements:
#   - hyperfine: brew install hyperfine (macOS) or cargo install hyperfine
#   - jq: brew install jq (for JSON processing)
#   - Rust CLI built: cargo build --release in src/rust/cli
#   - Python CLI installed: uv sync
#
# Usage:
#   ./benchmarks/cli_startup.sh [--quick|--full]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
REPORT_DIR="$PROJECT_ROOT/tmp"
REPORT_FILE="$REPORT_DIR/cli_benchmark_$(date +%Y%m%d-%H%M).md"

# Check for hyperfine
if ! command -v hyperfine &> /dev/null; then
    echo "Error: hyperfine not found. Install with: brew install hyperfine"
    exit 1
fi

# CLI paths
RUST_CLI="$PROJECT_ROOT/src/rust/cli/target/release/wqm"
PYTHON_CLI="wqm-py"

# Check Rust CLI exists
if [[ ! -f "$RUST_CLI" ]]; then
    echo "Error: Rust CLI not found at $RUST_CLI"
    echo "Build with: cd src/rust/cli && cargo build --release"
    exit 1
fi

# Check Python CLI exists
if ! command -v $PYTHON_CLI &> /dev/null; then
    echo "Warning: Python CLI (wqm-py) not found in PATH"
    echo "Install with: uv sync"
    PYTHON_CLI=""
fi

# Benchmark parameters
WARMUP=${WARMUP:-3}
RUNS=${RUNS:-10}

# Quick mode uses fewer runs
if [[ "$1" == "--quick" ]]; then
    WARMUP=1
    RUNS=5
    echo "Running in quick mode (warmup=$WARMUP, runs=$RUNS)"
fi

# Full mode uses more runs for accuracy
if [[ "$1" == "--full" ]]; then
    WARMUP=5
    RUNS=50
    echo "Running in full mode (warmup=$WARMUP, runs=$RUNS)"
fi

mkdir -p "$REPORT_DIR"

echo "================================================================"
echo "CLI Startup Performance Benchmarks"
echo "================================================================"
echo ""
echo "Platform: $(uname -ms)"
echo "Rust CLI: $RUST_CLI"
echo "Python CLI: ${PYTHON_CLI:-'not available'}"
echo "Warmup runs: $WARMUP"
echo "Measurement runs: $RUNS"
echo ""

# Function to run benchmark and extract results
run_benchmark() {
    local name="$1"
    local cmd="$2"
    local json_file="$REPORT_DIR/${name}.json"

    echo "Benchmarking: $name"
    echo "  Command: $cmd"

    hyperfine \
        --warmup "$WARMUP" \
        --runs "$RUNS" \
        --export-json "$json_file" \
        --shell=none \
        "$cmd" 2>/dev/null

    # Extract results
    local mean=$(jq -r '.results[0].mean * 1000 | floor' "$json_file")
    local stddev=$(jq -r '.results[0].stddev * 1000 | floor' "$json_file")
    local min=$(jq -r '.results[0].min * 1000 | floor' "$json_file")
    local max=$(jq -r '.results[0].max * 1000 | floor' "$json_file")

    echo "  Mean: ${mean}ms (±${stddev}ms)"
    echo "  Min: ${min}ms, Max: ${max}ms"
    echo ""

    # Return results as string
    echo "$mean|$stddev|$min|$max"
}

# Function to run comparison benchmark
run_comparison() {
    local name="$1"
    local rust_cmd="$2"
    local python_cmd="$3"
    local json_file="$REPORT_DIR/${name}_comparison.json"

    echo "Comparing: $name"
    echo "  Rust:   $rust_cmd"
    echo "  Python: $python_cmd"

    hyperfine \
        --warmup "$WARMUP" \
        --runs "$RUNS" \
        --export-json "$json_file" \
        --shell=none \
        "$rust_cmd" \
        "$python_cmd" 2>/dev/null

    # Extract comparison
    local rust_mean=$(jq -r '.results[0].mean * 1000 | floor' "$json_file")
    local python_mean=$(jq -r '.results[1].mean * 1000 | floor' "$json_file")
    local speedup=$(echo "scale=1; $python_mean / $rust_mean" | bc)

    echo "  Rust mean:   ${rust_mean}ms"
    echo "  Python mean: ${python_mean}ms"
    echo "  Speedup:     ${speedup}x"
    echo ""
}

echo "================================================================"
echo "1. Version Command (--version)"
echo "================================================================"
echo ""

# Benchmark Rust CLI --version
RUST_VERSION_RESULT=$(run_benchmark "rust_version" "$RUST_CLI --version")
RUST_VERSION_MEAN=$(echo "$RUST_VERSION_RESULT" | tail -1 | cut -d'|' -f1)

if [[ -n "$PYTHON_CLI" ]]; then
    # Benchmark Python CLI --version
    PYTHON_VERSION_RESULT=$(run_benchmark "python_version" "$PYTHON_CLI --version")
    PYTHON_VERSION_MEAN=$(echo "$PYTHON_VERSION_RESULT" | tail -1 | cut -d'|' -f1)

    # Calculate speedup
    VERSION_SPEEDUP=$(echo "scale=1; $PYTHON_VERSION_MEAN / $RUST_VERSION_MEAN" | bc)
    echo "Version command speedup: ${VERSION_SPEEDUP}x"
    echo ""
fi

echo "================================================================"
echo "2. Help Command (--help)"
echo "================================================================"
echo ""

# Benchmark Rust CLI --help
RUST_HELP_RESULT=$(run_benchmark "rust_help" "$RUST_CLI --help")
RUST_HELP_MEAN=$(echo "$RUST_HELP_RESULT" | tail -1 | cut -d'|' -f1)

if [[ -n "$PYTHON_CLI" ]]; then
    # Benchmark Python CLI --help
    PYTHON_HELP_RESULT=$(run_benchmark "python_help" "$PYTHON_CLI --help")
    PYTHON_HELP_MEAN=$(echo "$PYTHON_HELP_RESULT" | tail -1 | cut -d'|' -f1)

    # Calculate speedup
    HELP_SPEEDUP=$(echo "scale=1; $PYTHON_HELP_MEAN / $RUST_HELP_MEAN" | bc)
    echo "Help command speedup: ${HELP_SPEEDUP}x"
    echo ""
fi

echo "================================================================"
echo "3. Service Command Help (service --help)"
echo "================================================================"
echo ""

# Benchmark Rust CLI service --help
RUST_SERVICE_RESULT=$(run_benchmark "rust_service" "$RUST_CLI service --help")
RUST_SERVICE_MEAN=$(echo "$RUST_SERVICE_RESULT" | tail -1 | cut -d'|' -f1)

if [[ -n "$PYTHON_CLI" ]]; then
    # Benchmark Python CLI service --help (if command exists)
    if $PYTHON_CLI service --help &> /dev/null; then
        PYTHON_SERVICE_RESULT=$(run_benchmark "python_service" "$PYTHON_CLI service --help")
        PYTHON_SERVICE_MEAN=$(echo "$PYTHON_SERVICE_RESULT" | tail -1 | cut -d'|' -f1)

        # Calculate speedup
        SERVICE_SPEEDUP=$(echo "scale=1; $PYTHON_SERVICE_MEAN / $RUST_SERVICE_MEAN" | bc)
        echo "Service command speedup: ${SERVICE_SPEEDUP}x"
        echo ""
    else
        echo "Python CLI 'service' command not available for comparison"
        echo ""
    fi
fi

echo "================================================================"
echo "4. Admin Command Help (admin --help)"
echo "================================================================"
echo ""

# Benchmark Rust CLI admin --help
RUST_ADMIN_RESULT=$(run_benchmark "rust_admin" "$RUST_CLI admin --help")
RUST_ADMIN_MEAN=$(echo "$RUST_ADMIN_RESULT" | tail -1 | cut -d'|' -f1)

echo "================================================================"
echo "Performance Summary"
echo "================================================================"
echo ""
echo "Target: Rust CLI <100ms for --version, <200ms for commands"
echo ""
echo "Rust CLI Results:"
echo "  --version:      ${RUST_VERSION_MEAN}ms  $(if [[ $RUST_VERSION_MEAN -lt 100 ]]; then echo '[PASS]'; else echo '[FAIL]'; fi)"
echo "  --help:         ${RUST_HELP_MEAN}ms  $(if [[ $RUST_HELP_MEAN -lt 100 ]]; then echo '[PASS]'; else echo '[FAIL]'; fi)"
echo "  service --help: ${RUST_SERVICE_MEAN}ms  $(if [[ $RUST_SERVICE_MEAN -lt 100 ]]; then echo '[PASS]'; else echo '[FAIL]'; fi)"
echo "  admin --help:   ${RUST_ADMIN_MEAN}ms  $(if [[ $RUST_ADMIN_MEAN -lt 100 ]]; then echo '[PASS]'; else echo '[FAIL]'; fi)"
echo ""

if [[ -n "$PYTHON_CLI" ]]; then
    echo "Python CLI Results:"
    echo "  --version:      ${PYTHON_VERSION_MEAN}ms"
    echo "  --help:         ${PYTHON_HELP_MEAN}ms"
    echo ""
    echo "Speedup (Python/Rust):"
    echo "  --version: ${VERSION_SPEEDUP:-N/A}x"
    echo "  --help:    ${HELP_SPEEDUP:-N/A}x"
    echo ""
fi

# Generate markdown report
cat > "$REPORT_FILE" << EOF
# CLI Startup Performance Benchmark Report

**Date:** $(date +"%Y-%m-%d %H:%M:%S")
**Platform:** $(uname -ms)
**Rust CLI:** $RUST_CLI
**Python CLI:** ${PYTHON_CLI:-'not available'}

## Configuration

- Warmup runs: $WARMUP
- Measurement runs: $RUNS

## Results

### Rust CLI Performance

| Command | Mean | Target | Status |
|---------|------|--------|--------|
| \`--version\` | ${RUST_VERSION_MEAN}ms | <100ms | $(if [[ $RUST_VERSION_MEAN -lt 100 ]]; then echo 'PASS'; else echo 'FAIL'; fi) |
| \`--help\` | ${RUST_HELP_MEAN}ms | <100ms | $(if [[ $RUST_HELP_MEAN -lt 100 ]]; then echo 'PASS'; else echo 'FAIL'; fi) |
| \`service --help\` | ${RUST_SERVICE_MEAN}ms | <100ms | $(if [[ $RUST_SERVICE_MEAN -lt 100 ]]; then echo 'PASS'; else echo 'FAIL'; fi) |
| \`admin --help\` | ${RUST_ADMIN_MEAN}ms | <100ms | $(if [[ $RUST_ADMIN_MEAN -lt 100 ]]; then echo 'PASS'; else echo 'FAIL'; fi) |

EOF

if [[ -n "$PYTHON_CLI" ]]; then
    cat >> "$REPORT_FILE" << EOF
### Python CLI Performance (for comparison)

| Command | Mean |
|---------|------|
| \`--version\` | ${PYTHON_VERSION_MEAN}ms |
| \`--help\` | ${PYTHON_HELP_MEAN}ms |

### Speedup Summary

| Command | Speedup |
|---------|---------|
| \`--version\` | ${VERSION_SPEEDUP:-N/A}x |
| \`--help\` | ${HELP_SPEEDUP:-N/A}x |

EOF
fi

cat >> "$REPORT_FILE" << EOF
## Conclusion

The Rust CLI $(if [[ $RUST_VERSION_MEAN -lt 100 && $RUST_HELP_MEAN -lt 100 ]]; then echo '**meets**'; else echo '**does not meet**'; fi) the <100ms startup target.

$(if [[ -n "$PYTHON_CLI" ]]; then echo "The Rust CLI is approximately **${VERSION_SPEEDUP:-N/A}x faster** than the Python CLI for basic operations."; fi)

## Raw Data

JSON benchmark results are stored in: \`$REPORT_DIR/\`
EOF

echo ""
echo "Report saved to: $REPORT_FILE"
echo ""
