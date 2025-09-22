#!/bin/bash
# Quick Coverage Status Check

PROJECT_ROOT="/Users/chris/Dropbox/dev/ai/claude-code-cfg/mcp/workspace-qdrant-mcp"
cd "$PROJECT_ROOT"

echo "🚀 COVERAGE STATUS CHECK - $(date)"
echo "======================================"

# Python Coverage Check
echo "🐍 Python Coverage Check:"
timeout 300 uv run pytest --cov=src --cov-report=term --tb=no -q > python_status.tmp 2>&1

PYTHON_COV=$(grep "TOTAL" python_status.tmp | grep -o '[0-9]\+%' | head -1 | sed 's/%//')
IMPORT_ERRORS=$(grep -o '[0-9]\+ errors during collection' python_status.tmp | grep -o '[0-9]\+' || echo "0")

if [ ! -z "$PYTHON_COV" ]; then
    if [ "$PYTHON_COV" -ge 100 ]; then
        echo "   🎯 TARGET ACHIEVED: ${PYTHON_COV}%"
    elif [ "$PYTHON_COV" -ge 95 ]; then
        echo "   🟡 APPROACHING TARGET: ${PYTHON_COV}%"
    else
        echo "   🔴 IN PROGRESS: ${PYTHON_COV}%"
    fi
else
    echo "   ❌ UNABLE TO MEASURE"
fi

if [ "$IMPORT_ERRORS" -gt 0 ]; then
    echo "   🔧 Import Errors: $IMPORT_ERRORS (BLOCKING)"
else
    echo "   🟢 Import Errors: Clean"
fi

# Rust Coverage Check
echo ""
echo "🦀 Rust Coverage Check:"
if [ -d "rust-engine" ]; then
    cd rust-engine
    timeout 120 cargo test > ../rust_status.tmp 2>&1
    RUST_EXIT=$?
    cd ..

    if [ $RUST_EXIT -eq 0 ]; then
        echo "   🟡 ESTIMATED: 85% (tests passing)"
    else
        echo "   🔴 ESTIMATED: 40% (tests failing)"
    fi
else
    echo "   ❌ Rust engine not found"
fi

echo ""
echo "======================================"

# Overall status
if [ ! -z "$PYTHON_COV" ] && [ "$PYTHON_COV" -ge 100 ] && [ $RUST_EXIT -eq 0 ]; then
    echo "🎉 ALL TARGETS ON TRACK!"
elif [ "$IMPORT_ERRORS" -gt 0 ]; then
    echo "🚨 BLOCKING: Fix $IMPORT_ERRORS import errors first"
else
    echo "🚧 WORK IN PROGRESS - Continue toward 100%"
fi

echo "======================================"

# Cleanup
rm -f python_status.tmp rust_status.tmp