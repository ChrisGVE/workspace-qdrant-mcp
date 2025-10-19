# Rust Toolchain Compatibility

This document describes the Rust toolchain compatibility testing strategy for workspace-qdrant-daemon.

## Minimum Supported Rust Version (MSRV)

**MSRV: 1.75.0**

This version is required by our dependencies:
- `sqlx 0.8` requires Rust 1.75+
- `tonic 0.12` requires Rust 1.70+
- `tokio 1.x` with full features requires Rust 1.70+

The MSRV is declared in:
- `Cargo.toml` (`rust-version = "1.75.0"`)
- `rust-toolchain.toml` (`channel = "1.75.0"`)

## Tested Rust Versions

We test against:
1. **MSRV (1.75.0)** - Minimum supported version
2. **Stable** - Current stable Rust release
3. **Nightly** (optional) - Latest nightly for experimental features

## Test Coverage

For each Rust version, we validate:

### Build Tests
- `cargo check --all-features` - Fast compilation check
- `cargo build --all-features` - Full build
- `cargo build --release` - Release build with optimizations

### Code Quality
- `cargo test --all-features` - Unit and integration tests
- `cargo clippy --all-features -- -D warnings` - Linting with warnings as errors
- `cargo fmt --all -- --check` - Code formatting validation

### Cross-Compilation
- `x86_64-unknown-linux-gnu` - Linux x86_64
- `x86_64-apple-darwin` - macOS Intel
- `aarch64-apple-darwin` - macOS Apple Silicon
- `x86_64-pc-windows-msvc` - Windows x86_64 (when available)

## Running Compatibility Tests

### Automatic Testing (All Versions)

```bash
cd rust-engine
./test-rust-versions.sh
```

This script will:
1. Install/update required Rust toolchains
2. Run all tests for each version
3. Test cross-compilation targets
4. Provide a summary of results

### Manual Testing (Specific Version)

```bash
cd rust-engine

# Test with MSRV
rustup toolchain install 1.75.0
cargo +1.75.0 build --all-features
cargo +1.75.0 test --all-features
cargo +1.75.0 clippy --all-features -- -D warnings

# Test with stable
cargo build --all-features
cargo test --all-features
cargo clippy --all-features -- -D warnings

# Test cross-compilation
rustup target add x86_64-unknown-linux-gnu
cargo build --target x86_64-unknown-linux-gnu --lib
```

## CI/CD Integration

### GitHub Actions

Add to `.github/workflows/rust-compatibility.yml`:

```yaml
name: Rust Compatibility

on: [push, pull_request]

jobs:
  test-rust-versions:
    name: Test Rust ${{ matrix.rust }}
    runs-on: ${{ matrix.os }}
    strategy:
      matrix:
        rust: ['1.75.0', 'stable']
        os: [ubuntu-latest, macos-latest, windows-latest]
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@master
        with:
          toolchain: ${{ matrix.rust }}
          components: clippy, rustfmt
      - name: Check
        run: cargo check --all-features
        working-directory: rust-engine
      - name: Test
        run: cargo test --all-features
        working-directory: rust-engine
      - name: Clippy
        run: cargo clippy --all-features -- -D warnings
        working-directory: rust-engine
```

## Upgrading MSRV

When upgrading the MSRV:

1. **Update version in both files:**
   - `Cargo.toml`: `rust-version = "X.Y.Z"`
   - `rust-toolchain.toml`: `channel = "X.Y.Z"`

2. **Test thoroughly:**
   ```bash
   ./test-rust-versions.sh
   ```

3. **Update documentation:**
   - This file (RUST_COMPATIBILITY.md)
   - Main README.md
   - CHANGELOG.md

4. **Consider:**
   - Are all dependencies compatible with new MSRV?
   - Do any new Rust features benefit the codebase?
   - Does the upgrade break any downstream users?

## Dependency Compatibility

Key dependencies and their Rust requirements:

| Dependency | Version | Min Rust | Notes |
|------------|---------|----------|-------|
| tonic | 0.12 | 1.70 | gRPC framework |
| tokio | 1.x | 1.70 | Async runtime |
| sqlx | 0.8 | 1.75 | **MSRV driver** |
| serde | 1.0 | 1.56 | Serialization |
| anyhow | 1.0 | 1.56 | Error handling |

## Edition Compatibility

Current edition: **2021**

Rust 2021 edition requires:
- Rust 1.56+ to compile
- Includes disjoint capture in closures
- New panic macro behavior
- IntoIterator for arrays

## Platform Support

Tested platforms:
- **Linux**: Ubuntu 20.04+, Debian 11+, Alpine 3.17+
- **macOS**: macOS 12+ (Intel and Apple Silicon)
- **Windows**: Windows 10+, Windows Server 2019+

## Troubleshooting

### MSRV Build Failures

If builds fail on MSRV but pass on stable:

1. Check for accidental use of newer Rust features
2. Verify all dependencies support MSRV
3. Look for `#![feature(...)]` gates that require nightly
4. Check for newer Cargo.lock dependencies

### Cross-Compilation Issues

If cross-compilation fails:

1. Install target: `rustup target add <target-triple>`
2. Install system linker/toolchain if needed
3. Check for platform-specific dependencies
4. Consider using `cross` tool for complex targets

## Resources

- [Rust Edition Guide](https://doc.rust-lang.org/edition-guide/)
- [Cargo Book - rust-version](https://doc.rust-lang.org/cargo/reference/manifest.html#the-rust-version-field)
- [Cross-compilation Guide](https://rust-lang.github.io/rustup/cross-compilation.html)
