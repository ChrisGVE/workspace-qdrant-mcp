# Cross-Platform Compilation Guide

This document describes the cross-platform compilation support in workspace-qdrant-mcp, including platform-specific optimizations, build configurations, and CI/CD integration.

## Supported Platforms

### macOS
- **x86_64-apple-darwin** - Intel-based Macs
- **aarch64-apple-darwin** - Apple Silicon Macs (M1/M2/M3+)

**Platform-specific features:**
- FSEvents integration for high-performance file watching
- kqueue support for fine-grained event monitoring  
- Optimized memory alignment for Apple Silicon
- Native code generation with `target-cpu=native`

### Linux
- **x86_64-unknown-linux-gnu** - Standard Linux x86_64 with glibc
- **aarch64-unknown-linux-gnu** - Linux ARM64 with glibc  
- **x86_64-unknown-linux-musl** - Alpine/musl-based x86_64 Linux
- **aarch64-unknown-linux-musl** - Alpine/musl-based ARM64 Linux

**Platform-specific features:**
- inotify integration for efficient file system monitoring
- epoll support for high-performance event handling
- SIMD optimizations for vector operations
- Static linking support with musl targets

### Windows
- **x86_64-pc-windows-msvc** - Standard Windows x86_64
- **aarch64-pc-windows-msvc** - Windows ARM64 (Surface Pro X, etc.)
- **x86_64-pc-windows-gnu** - Windows with GNU toolchain (MinGW)

**Platform-specific features:**
- ReadDirectoryChangesW for native file watching
- I/O Completion Ports (IOCP) for async operations
- Unicode path handling
- Windows-specific optimizations

## Building Locally

### Prerequisites

#### All Platforms
- Rust 1.70+ with `rustup`
- Protocol Buffers compiler (`protoc`)
- Python 3.10+

#### Platform-Specific Tools

**macOS:**
```bash
# Install Xcode command line tools
xcode-select --install

# Install Homebrew dependencies
brew install protobuf

# Add targets
rustup target add x86_64-apple-darwin aarch64-apple-darwin
```

**Linux:**
```bash
# Install system dependencies
sudo apt-get update
sudo apt-get install -y protobuf-compiler build-essential

# For cross-compilation to ARM64
sudo apt-get install -y gcc-aarch64-linux-gnu g++-aarch64-linux-gnu

# For musl targets
sudo apt-get install -y musl-tools musl-dev

# Add targets
rustup target add x86_64-unknown-linux-gnu aarch64-unknown-linux-gnu
rustup target add x86_64-unknown-linux-musl aarch64-unknown-linux-musl
```

**Windows:**
```powershell
# Install Visual Studio Build Tools or Visual Studio Community

# Install Protocol Buffers
choco install protoc

# Add targets  
rustup target add x86_64-pc-windows-msvc aarch64-pc-windows-msvc x86_64-pc-windows-gnu
```

### Building the Rust Engine

```bash
# Build for current platform
cd rust-engine
cargo build --release

# Build for specific target
cargo build --release --target x86_64-unknown-linux-gnu

# Cross-compile using the cross tool
cargo install cross
cross build --release --target aarch64-unknown-linux-gnu

# Build with platform-specific features
cargo build --release --features macos-optimizations  # macOS
cargo build --release --features linux-optimizations  # Linux  
cargo build --release --features windows-optimizations # Windows
```

### Building Python Wheels

Using maturin:
```bash
# Install maturin
pip install maturin

# Build wheel for current platform
cd rust-engine/python-bindings
maturin build --release

# Build for specific target
maturin build --release --target x86_64-unknown-linux-gnu

# Build with platform optimizations
RUSTFLAGS="-C target-cpu=native" maturin build --release
```

Using cibuildwheel (for all platforms):
```bash
pip install cibuildwheel
cibuildwheel --platform linux    # Build Linux wheels
cibuildwheel --platform macos     # Build macOS wheels  
cibuildwheel --platform windows   # Build Windows wheels
```

## CI/CD Integration

### GitHub Actions Workflows

The project includes two main workflows for cross-platform builds:

#### 1. rust-wheels.yml
- Builds production-ready Python wheels for all platforms
- Runs comprehensive testing and integration tests
- Publishes to PyPI on releases
- Includes security scanning and benchmarks

#### 2. cross-platform-build.yml  
- Comprehensive cross-compilation matrix testing
- Platform-specific integration tests
- Performance benchmarking
- Security auditing with cargo-audit, cargo-deny, and cargo-geiger

### Triggering Builds

Builds are triggered automatically on:
- Push to `main`, `develop`, or `release/*` branches
- Pull requests targeting `main` or `develop`
- Manual workflow dispatch

### Build Matrix

The CI system tests the following combinations:

| Platform | Architecture | Toolchain | Test Execution |
|----------|-------------|-----------|----------------|
| macOS Latest | x86_64 | Stable Rust | ✅ Full tests |
| macOS Latest | ARM64 | Stable Rust | ⚠️ Build only |  
| Ubuntu Latest | x86_64 | Stable Rust | ✅ Full tests |
| Ubuntu Latest | ARM64 | Cross-compile | ⚠️ Build only |
| Ubuntu Latest | x86_64 musl | Cross-compile | ✅ Limited tests |
| Ubuntu Latest | ARM64 musl | Cross-compile | ⚠️ Build only |
| Windows Latest | x86_64 MSVC | Stable Rust | ✅ Full tests |
| Windows Latest | ARM64 MSVC | Stable Rust | ⚠️ Build only |
| Windows Latest | x86_64 GNU | Stable Rust | ✅ Full tests |

## Performance Optimizations

### Compilation Flags

The build system automatically applies platform-specific optimizations:

**All Platforms:**
- Link-time optimization (LTO)
- Dead code elimination
- Single codegen unit for better optimization

**macOS:**
```rust
RUSTFLAGS="-C target-cpu=native -C link-arg=-Wl,-dead_strip"
```

**Linux:**
```rust  
RUSTFLAGS="-C target-cpu=native -C link-arg=-Wl,--gc-sections -C link-arg=-Wl,--strip-debug"
```

**Windows:**
```rust
RUSTFLAGS="-C target-cpu=native"
```

### Platform-Specific Features

Enable platform optimizations with feature flags:

```bash
# macOS optimizations (FSEvents + kqueue)
cargo build --features macos-optimizations

# Linux optimizations (inotify + epoll)  
cargo build --features linux-optimizations

# Windows optimizations (ReadDirectoryChangesW + IOCP)
cargo build --features windows-optimizations
```

## Benchmarking

The project includes comprehensive benchmarks:

### Running Benchmarks

```bash
cd rust-engine
cargo bench --features benchmarks

# Platform-specific benchmarks
cargo bench --bench platform_benchmarks --features benchmarks

# File watching benchmarks
cargo bench --bench watching_benchmarks --features benchmarks

# Processing benchmarks  
cargo bench --bench processing_benchmarks --features benchmarks
```

### Benchmark Categories

1. **Processing Benchmarks** - Document parsing, embedding generation, I/O operations
2. **Watching Benchmarks** - File system monitoring, event processing, pattern matching
3. **Platform Benchmarks** - Platform-specific optimizations, memory alignment, API calls

## Testing

### Integration Tests

Platform-specific integration tests validate functionality:

```bash
# Run all tests
cargo test --workspace --release

# Platform-specific tests (require respective platform)
cargo test --features macos-integration-tests    # macOS
cargo test --features linux-integration-tests    # Linux
cargo test --features windows-integration-tests  # Windows

# Cross-platform compatibility tests
cargo test file_watching_tests
cargo test memory_usage_tests
cargo test performance_baseline_tests
```

### Test Categories

1. **Unit Tests** - Core functionality testing
2. **Integration Tests** - Platform-specific feature testing  
3. **Cross-Platform Tests** - Compatibility and behavior consistency
4. **Performance Tests** - Baseline performance validation
5. **Memory Tests** - Memory usage and leak detection

## Troubleshooting

### Common Build Issues

**Protoc Not Found:**
```bash
# Install protobuf compiler for your platform
# macOS: brew install protobuf  
# Linux: apt-get install protobuf-compiler
# Windows: choco install protoc
```

**Cross-compilation Failures:**
```bash
# Install target toolchain
rustup target add <target-triple>

# For Linux ARM64 cross-compilation
sudo apt-get install gcc-aarch64-linux-gnu g++-aarch64-linux-gnu
export CC_aarch64_unknown_linux_gnu=aarch64-linux-gnu-gcc
export AR_aarch64_unknown_linux_gnu=aarch64-linux-gnu-ar
```

**Linking Errors:**
```bash
# Update RUSTFLAGS for your target
export RUSTFLAGS="-C linker=<appropriate-linker>"

# Or use the cross tool for easier cross-compilation
cargo install cross
cross build --target <target-triple>
```

### Performance Issues

If you experience performance issues:

1. **Enable platform optimizations:**
   ```bash
   cargo build --release --features <platform>-optimizations
   ```

2. **Use native CPU features:**
   ```bash
   RUSTFLAGS="-C target-cpu=native" cargo build --release
   ```

3. **Profile with benchmarks:**
   ```bash
   cargo bench --features benchmarks
   ```

### Platform-Specific Issues

**macOS:**
- Ensure Xcode command line tools are installed
- For Apple Silicon, verify both x86_64 and aarch64 targets work
- FSEvents may require additional permissions in sandboxed environments

**Linux:**  
- Install appropriate cross-compilation toolchains
- Musl targets may require static linking adjustments
- Check inotify limits: `cat /proc/sys/fs/inotify/max_user_watches`

**Windows:**
- Verify Visual Studio Build Tools are properly installed
- ARM64 Windows requires Windows 10 version 1709+
- Path length limitations may require long path support

## Contributing

When contributing cross-platform features:

1. **Test on multiple platforms** using the provided CI workflows
2. **Add platform-specific tests** to validate functionality
3. **Update benchmarks** if performance characteristics change
4. **Document platform differences** in code comments and documentation
5. **Follow the established patterns** for conditional compilation

### Adding New Platforms

To add support for a new platform:

1. Add the target triple to workspace metadata in `Cargo.toml`
2. Add platform-specific dependencies with `[target.'cfg(...)'.dependencies]`
3. Implement platform-specific features in the `platform.rs` module
4. Add CI matrix entries in both workflow files
5. Update documentation and benchmarks

The cross-platform architecture is designed to be extensible, so adding new platforms should integrate smoothly with the existing infrastructure.