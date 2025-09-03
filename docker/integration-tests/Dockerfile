FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    protobuf-compiler \
    libprotobuf-dev \
    pkg-config \
    docker.io \
    && rm -rf /var/lib/apt/lists/*

# Install Rust for building extensions
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY pyproject.toml ./
RUN pip install --upgrade pip && \
    pip install -e .[dev] && \
    pip install testcontainers pytest-benchmark pytest-timeout pytest-xdist

# Copy source code
COPY . .

# Create directories for test results
RUN mkdir -p test_results coverage_reports performance_results

# Set environment variables
ENV PYTHONPATH=/app
ENV TEST_ENV=integration
ENV INTEGRATION_TESTING=1

# Default command
CMD ["python", "scripts/run_integration_tests.py", "--categories", "integration", "--verbose"]