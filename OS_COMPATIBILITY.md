# Operating System Compatibility

This document describes the operating system compatibility testing strategy for workspace-qdrant-mcp.

## Supported Operating Systems

**Supported Platforms:**
- **Linux**: Ubuntu 20.04+, Debian 11+, Alpine 3.17+
- **macOS**: macOS 12+ (Monterey and later) - Intel and Apple Silicon
- **Windows**: Windows 10+, Windows Server 2019+

## Platform Requirements

### Linux
- **Kernel**: 5.4+ (Ubuntu 20.04 baseline)
- **Python**: 3.10+
- **Rust**: 1.75.0+
- **Permissions**: Standard user permissions for file operations
- **Dependencies**:
  - `libc6` (glibc 2.31+)
  - `libsqlite3` for SQLite operations
  - `libssl` for secure connections

### macOS
- **Version**: macOS 12+ (Darwin kernel 21+)
- **Architectures**: x86_64 (Intel) and arm64 (Apple Silicon)
- **Python**: 3.10+
- **Rust**: 1.75.0+
- **XCode Command Line Tools**: Required for native compilation
- **Permissions**: File system permissions, no sandboxing issues

### Windows
- **Version**: Windows 10 (build 19041+) or Windows Server 2019+
- **Architecture**: x86_64 (AMD64)
- **Python**: 3.10+
- **Rust**: 1.75.0+ with MSVC toolchain
- **Visual Studio Build Tools**: Required for compilation
- **Permissions**: Standard user permissions, no admin required (except for symlinks)

## Compatibility Test Coverage

### Platform Detection Tests
- System identification (Linux/Darwin/Windows)
- Release version validation
- Architecture detection (x86_64, arm64, etc.)
- Python implementation verification

### Path Handling Tests
- Path separator handling (`/` vs `\`)
- Absolute path detection
- Home directory expansion
- Temporary directory access
- Path normalization across platforms

### File System Permission Tests
- File creation permissions
- Directory creation permissions
- Unix-style permissions (chmod 0644, 0755)
- Windows read-only attributes
- Executable bit (Unix)
- Symbolic link creation

### Process Operations Tests
- Process ID retrieval
- Current working directory
- User information (Unix: pwd, Windows: USERNAME)
- Environment variable handling

### OS-Specific Feature Tests
- **Linux**: `/proc` filesystem access
- **macOS**: System directories (/Applications, /Library)
- **Windows**: System paths (SYSTEMROOT, PROGRAMFILES)
- Line ending handling (LF vs CRLF)

## Running Compatibility Tests

### Manual Testing

```bash
# Run all OS compatibility tests
uv run pytest tests/compatibility/test_os_platforms.py -v

# Run specific test class
uv run pytest tests/compatibility/test_os_platforms.py::TestPlatformDetection -v

# Run on specific platform only
uv run pytest tests/compatibility/test_os_platforms.py -v -m "not skipif"
```

### Platform-Specific Tests

Tests are automatically skipped on unsupported platforms using pytest markers:
- `@pytest.mark.skipif(platform.system() == "Windows", reason="Unix only")`
- `@pytest.mark.skipif(platform.system() != "Windows", reason="Windows only")`
- `@pytest.mark.skipif(platform.system() != "Linux", reason="Linux only")`
- `@pytest.mark.skipif(platform.system() != "Darwin", reason="macOS only")`

### CI/CD Integration

Example GitHub Actions workflow:

```yaml
name: OS Compatibility

on: [push, pull_request]

jobs:
  test-os-compatibility:
    name: Test ${{ matrix.os }}
    runs-on: ${{ matrix.os }}
    strategy:
      matrix:
        os: [ubuntu-20.04, ubuntu-22.04, macos-12, macos-13, windows-2019, windows-2022]
        python-version: ['3.10', '3.11', '3.12']

    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: ${{ matrix.python-version }}

      - name: Install dependencies
        run: |
          pip install uv
          uv sync --dev

      - name: Run OS compatibility tests
        run: uv run pytest tests/compatibility/test_os_platforms.py -v
```

## Docker-Based Testing

For consistent Linux testing across platforms:

```dockerfile
# Ubuntu 20.04 LTS
FROM ubuntu:20.04

RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    curl \
    build-essential \
    libsqlite3-dev

COPY . /workspace
WORKDIR /workspace

RUN pip install uv
RUN uv sync --dev

CMD ["uv", "run", "pytest", "tests/compatibility/test_os_platforms.py", "-v"]
```

Build and run:
```bash
docker build -t wqm-ubuntu-test .
docker run --rm wqm-ubuntu-test
```

## Known Platform-Specific Limitations

### Linux
- Requires kernel 5.4+ for modern file watching features
- Some tests require `/proc` filesystem (standard on Linux)

### macOS
- Symbolic links work without admin privileges
- Case-insensitive filesystem by default (APFS can be case-sensitive)
- Different process management compared to Linux

### Windows
- Symbolic links require admin privileges or Developer Mode enabled
- Path length limit of 260 characters (unless long path support enabled)
- Different line endings (CRLF vs LF)
- Case-insensitive filesystem (always)

## Troubleshooting

### macOS Specific Issues

**Issue**: `Operation not permitted` errors
- **Cause**: macOS privacy restrictions
- **Solution**: Grant Full Disk Access in System Settings → Privacy & Security

**Issue**: `xcrun: error: invalid active developer path`
- **Cause**: XCode Command Line Tools not installed
- **Solution**: Run `xcode-select --install`

### Windows Specific Issues

**Issue**: Symlink creation fails
- **Cause**: Insufficient privileges
- **Solution**: Enable Developer Mode or run as administrator

**Issue**: Long path errors
- **Cause**: Windows path length limit (260 chars)
- **Solution**: Enable long path support in Windows 10+
  - Registry: `HKLM\SYSTEM\CurrentControlSet\Control\FileSystem\LongPathsEnabled = 1`

**Issue**: Line ending differences
- **Cause**: Git auto-converts CRLF ↔ LF
- **Solution**: Configure `.gitattributes` for consistent line endings

### Linux Specific Issues

**Issue**: Permission denied on file operations
- **Cause**: Insufficient permissions
- **Solution**: Check file ownership and permissions with `ls -la`

**Issue**: SQLite locked database
- **Cause**: Multiple processes accessing database
- **Solution**: Ensure WAL mode enabled, check for zombie processes

## Version Compatibility Matrix

| OS | Version | Python 3.10 | Python 3.11 | Python 3.12 | Python 3.13 | Rust 1.75+ |
|----|---------|-------------|-------------|-------------|-------------|------------|
| Ubuntu 20.04 | LTS | ✅ | ✅ | ✅ | ✅ | ✅ |
| Ubuntu 22.04 | LTS | ✅ | ✅ | ✅ | ✅ | ✅ |
| Ubuntu 24.04 | LTS | ✅ | ✅ | ✅ | ✅ | ✅ |
| Debian 11 | Stable | ✅ | ✅ | ✅ | ✅ | ✅ |
| Debian 12 | Stable | ✅ | ✅ | ✅ | ✅ | ✅ |
| macOS 12 | Monterey | ✅ | ✅ | ✅ | ✅ | ✅ |
| macOS 13 | Ventura | ✅ | ✅ | ✅ | ✅ | ✅ |
| macOS 14 | Sonoma | ✅ | ✅ | ✅ | ✅ | ✅ |
| Windows 10 | 19041+ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Windows 11 | 22000+ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Windows Server 2019 | - | ✅ | ✅ | ✅ | ✅ | ✅ |
| Windows Server 2022 | - | ✅ | ✅ | ✅ | ✅ | ✅ |

## Resources

- [Python Platform Module Documentation](https://docs.python.org/3/library/platform.html)
- [pathlib Cross-Platform Paths](https://docs.python.org/3/library/pathlib.html)
- [Windows Long Path Support](https://docs.microsoft.com/en-us/windows/win32/fileio/maximum-file-path-limitation)
- [macOS File System Events](https://developer.apple.com/documentation/coreservices/file_system_events)
- [Linux inotify Documentation](https://man7.org/linux/man-pages/man7/inotify.7.html)
