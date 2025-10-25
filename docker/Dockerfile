# Multi-stage Dockerfile for Workspace Qdrant MCP
# Security-hardened with minimal attack surface and non-root execution

# Stage 1: Rust builder for performance-critical components
FROM rust:1.75-slim-bullseye AS rust-builder

LABEL stage=rust-builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    pkg-config \
    libssl-dev \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Set up Rust environment
WORKDIR /app/rust-engine-legacy

# Copy Rust source code and build
COPY rust-engine-legacy/ .
RUN cargo build --release --locked

# Stage 2: Python base with system dependencies
FROM python:3.11-slim-bullseye AS python-base

# Environment variables for Python
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    # Core dependencies
    ca-certificates \
    curl \
    git \
    # Document processing dependencies
    libmagic1 \
    libxml2 \
    libxslt1.1 \
    # Clean up
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Stage 3: Development environment (includes dev tools)
FROM python-base AS development

LABEL stage=development

# Install development dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    make \
    pkg-config \
    libffi-dev \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Set up Python environment
WORKDIR /app

# Install uv for faster dependency management
RUN pip install uv

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Install dependencies including development tools
RUN uv sync --dev

# Copy source code
COPY src/ src/
COPY tests/ tests/
COPY --from=rust-builder /app/rust-engine-legacy/target/release/ rust-engine-legacy/target/release/

# Create non-root user for development
RUN groupadd --gid 1000 appuser \
    && useradd --uid 1000 --gid 1000 --create-home --shell /bin/bash appuser \
    && chown -R appuser:appuser /app

USER appuser

# Development server command
CMD ["python", "-m", "uvicorn", "workspace_qdrant_mcp.server:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]

# Stage 4: Testing environment with coverage tools
FROM development AS testing

LABEL stage=testing

USER root

# Install testing dependencies
RUN apt-get update && apt-get install -y \
    # For integration testing
    docker.io \
    docker-compose \
    # For security scanning
    wget \
    && rm -rf /var/lib/apt/lists/*

# Switch back to app user for running tests
USER appuser

# Test command
CMD ["python", "-m", "pytest", "tests/", "-v", "--cov=src/workspace_qdrant_mcp", "--cov-report=html", "--cov-report=term-missing"]

# Stage 5: Production base with minimal footprint
FROM python-base AS production-base

LABEL stage=production-base

# Create application directory
WORKDIR /app

# Install uv for dependency management
RUN pip install --no-cache-dir uv==0.4.15

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Install only production dependencies
RUN uv sync --no-dev --frozen

# Stage 6: Final production image
FROM python-base AS production

LABEL maintainer="christian.berclaz@mac.com" \
      version="0.3.0" \
      description="Workspace Qdrant MCP Server - Production Ready" \
      org.opencontainers.image.title="workspace-qdrant-mcp" \
      org.opencontainers.image.description="Project-scoped Qdrant MCP server with scratchbook functionality" \
      org.opencontainers.image.version="0.3.0" \
      org.opencontainers.image.authors="Christian C. Berclaz <christian.berclaz@mac.com>" \
      org.opencontainers.image.url="https://github.com/ChrisGVE/workspace-qdrant-mcp" \
      org.opencontainers.image.source="https://github.com/ChrisGVE/workspace-qdrant-mcp" \
      org.opencontainers.image.licenses="MIT"

# Security: Create non-root user with minimal privileges
RUN groupadd --system --gid 65534 appuser \
    && useradd --system --uid 65534 --gid 65534 --no-create-home --shell /sbin/nologin appuser

# Set up application directory with proper ownership
WORKDIR /app

# Copy virtual environment from production-base
COPY --from=production-base --chown=root:root /app/.venv /app/.venv

# Copy Rust binaries if available
COPY --from=rust-builder --chown=root:root /app/rust-engine-legacy/target/release/ /app/rust-engine-legacy/target/release/

# Copy application source code
COPY --chown=root:root src/ src/
COPY --chown=root:root README.md LICENSE CHANGELOG.md ./

# Create necessary directories with proper permissions
RUN mkdir -p /app/data /app/logs /app/tmp \
    && chown -R appuser:appuser /app/data /app/logs /app/tmp \
    && chmod 750 /app/data /app/logs /app/tmp

# Security hardening
RUN chmod -R 755 /app/src \
    && find /app -type f -name "*.py" -exec chmod 644 {} \; \
    && find /app -type d -exec chmod 755 {} \;

# Add virtual environment to PATH
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONPATH="/app/src:$PYTHONPATH"

# Application configuration
ENV WORKSPACE_QDRANT_HOST=0.0.0.0 \
    WORKSPACE_QDRANT_PORT=8000 \
    WORKSPACE_QDRANT_LOG_LEVEL=INFO \
    WORKSPACE_QDRANT_DATA_DIR=/app/data \
    WORKSPACE_QDRANT_LOG_DIR=/app/logs \
    WORKSPACE_QDRANT_TMP_DIR=/app/tmp

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')" || exit 1

# Expose port
EXPOSE 8000

# Switch to non-root user
USER appuser

# Set up entry point script
COPY --chown=root:root docker/entrypoint.sh /entrypoint.sh
USER root
RUN chmod +x /entrypoint.sh
USER appuser

ENTRYPOINT ["/entrypoint.sh"]
CMD ["workspace-qdrant-mcp"]

# Stage 7: Security scanning stage
FROM production AS security-scan

LABEL stage=security-scan

USER root

# Install security scanning tools
RUN apt-get update && apt-get install -y \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install grype for vulnerability scanning
RUN curl -sSfL https://raw.githubusercontent.com/anchore/grype/main/install.sh | sh -s -- -b /usr/local/bin

# Install syft for SBOM generation
RUN curl -sSfL https://raw.githubusercontent.com/anchore/syft/main/install.sh | sh -s -- -b /usr/local/bin

USER appuser

# Security scan command
CMD ["grype", "dir:/app", "--output", "json"]