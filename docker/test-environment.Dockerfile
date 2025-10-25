# Test environment Dockerfile for testcontainers integration
# Provides isolated testing environment with all dependencies

FROM ubuntu:22.04

LABEL maintainer="Christian C. Berclaz <christian.berclaz@mac.com>"
LABEL description="Test environment for workspace-qdrant-mcp CI/CD pipeline"
LABEL version="1.0.0"

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV RUST_BACKTRACE=1
ENV CARGO_TERM_COLOR=always

# Install system dependencies
RUN apt-get update && apt-get install -y \
    # Build essentials
    build-essential \
    pkg-config \
    cmake \
    curl \
    wget \
    git \
    # Python dependencies
    python3.11 \
    python3.11-dev \
    python3.11-venv \
    python3-pip \
    # Rust dependencies
    protobuf-compiler \
    libprotobuf-dev \
    # Database and networking
    sqlite3 \
    netcat-openbsd \
    # Testing utilities
    jq \
    xmlstarlet \
    # System utilities
    htop \
    procps \
    psmisc \
    tree \
    && rm -rf /var/lib/apt/lists/*

# Install Rust toolchain
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y \
    && /root/.cargo/bin/rustup default stable \
    && /root/.cargo/bin/rustup component add rustfmt clippy \
    && /root/.cargo/bin/cargo install cargo-audit cargo-deny cargo-geiger

# Add Rust to PATH
ENV PATH="/root/.cargo/bin:${PATH}"

# Install uv (Python package manager)
RUN pip3 install uv

# Create application directory
WORKDIR /app

# Copy project files
COPY . .

# Create Python virtual environment and install dependencies
RUN uv venv --python python3.11 .venv \
    && . .venv/bin/activate \
    && uv pip install -e ".[dev]" \
    && uv pip install pytest-testmon pytest-sugar pytest-clarity

# Build Rust components
RUN cd src/rust/daemon \
    && cargo build --release \
    && cargo test --release --no-run

# Create test data directory
RUN mkdir -p /app/test-data

# Set up test environment configuration
COPY docker/test-config.yaml /app/test-config.yaml

# Create test script
RUN cat > /app/run-tests.sh << 'EOF'
#!/bin/bash
set -e

echo "🚀 Starting test environment..."

# Activate Python environment
source .venv/bin/activate

# Print environment info
echo "📋 Environment Information:"
echo "Python version: $(python --version)"
echo "Rust version: $(rustc --version)"
echo "UV version: $(uv --version)"
echo "Pytest version: $(pytest --version)"

# Wait for external services if needed
if [[ -n "$QDRANT_URL" ]]; then
    echo "⏳ Waiting for Qdrant service..."
    timeout 60s bash -c 'until curl -f "$QDRANT_URL/healthz" >/dev/null 2>&1; do sleep 2; done'
    echo "✅ Qdrant service is ready"
fi

# Run tests based on environment
case "${TEST_SUITE:-all}" in
    "unit")
        echo "🧪 Running unit tests..."
        pytest tests/unit/ -v --tb=short
        ;;
    "integration")
        echo "🔗 Running integration tests..."
        pytest tests/integration/ -v --tb=short
        ;;
    "edge-cases")
        echo "⚠️ Running edge case tests..."
        pytest tests/edge_cases/ -v --tb=short
        ;;
    "performance")
        echo "🚀 Running performance tests..."
        pytest tests/ -m performance -v --tb=short
        ;;
    "all")
        echo "🌟 Running comprehensive test suite..."
        pytest --cov=src/python --cov-report=xml --cov-report=html --cov-branch --junitxml=test-results.xml -v
        ;;
    *)
        echo "❓ Unknown test suite: ${TEST_SUITE}"
        exit 1
        ;;
esac

echo "✅ Tests completed successfully"
EOF

RUN chmod +x /app/run-tests.sh

# Create health check script
RUN cat > /app/healthcheck.sh << 'EOF'
#!/bin/bash
# Health check for test environment

# Check Python environment
source .venv/bin/activate
python -c "import workspace_qdrant_mcp; print('Python imports OK')" || exit 1

# Check Rust components
cd src/rust/daemon
cargo check --release >/dev/null 2>&1 || exit 1
cd /app

echo "Health check passed"
EOF

RUN chmod +x /app/healthcheck.sh

# Expose ports for potential services
EXPOSE 8000 8080

# Set up health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD ["/app/healthcheck.sh"]

# Default command
CMD ["/app/run-tests.sh"]

# Add metadata for testcontainers
LABEL testcontainers.test-env.version="1.0.0"
LABEL testcontainers.test-env.python="3.11"
LABEL testcontainers.test-env.rust="stable"