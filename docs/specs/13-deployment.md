## Deployment and Installation

This section documents the deployment architecture, platform support, installation methods, and operational procedures for workspace-qdrant-mcp.

### Deployment Architecture Overview

The system consists of three primary components that work together:

```
┌─────────────────────────────────────────────────────────────────┐
│                      Claude Desktop / Claude Code               │
│                         (MCP Client)                            │
└─────────────────────┬───────────────────────────────────────────┘
                      │ MCP Protocol (stdio)
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│                     MCP Server (TypeScript)                     │
│                  src/typescript/mcp-server/                     │
│         Exposes: search, retrieve, rules, store tools           │
└──────────┬──────────────────────────────────────────┬───────────┘
           │ gRPC (localhost:50051)                   │ HTTP REST
           ▼                                          ▼
┌─────────────────────────┐              ┌────────────────────────┐
│    Daemon (memexd)      │              │   Qdrant Vector DB     │
│    Rust binary          │◄────────────►│   (localhost:6333)     │
│                         │  Qdrant API  │                        │
│  - File watching        │              │  Collections:          │
│  - Queue processing     │              │  - projects (unified)  │
│  - Code intelligence    │              │  - libraries (unified) │
│  - Embedding generation │              │  - memory              │
└──────────┬──────────────┘              └────────────────────────┘
           │
           ▼
┌─────────────────────────┐
│    SQLite Database      │
│  ~/.workspace-qdrant/   │
│      state.db           │
│                         │
│  - unified_queue        │
│  - watch_folders        │
│  - tracked_files        │
│  -   qdrant_chunks      │
│  - schema_version       │
└─────────────────────────┘
```

**Deployment Modes:**

| Mode | Qdrant Location | Use Case |
|------|-----------------|----------|
| All-in-One | Bundled in Docker | Development, testing |
| Qdrant-External | Separate Docker container | Production self-hosted |
| Qdrant-Local | Host-installed Qdrant | Development with persistence |
| Qdrant-Cloud | Qdrant Cloud service | Production managed |

### Platform Support Matrix

The system supports 6 platform/architecture combinations:

| Platform | Architecture | Target Triple | Notes |
|----------|--------------|---------------|-------|
| Linux | x86_64 | `x86_64-unknown-linux-gnu` | Primary CI/CD target |
| Linux | ARM64 | `aarch64-unknown-linux-gnu` | AWS Graviton, Raspberry Pi |
| macOS | Apple Silicon | `aarch64-apple-darwin` | M1/M2/M3 Macs |
| macOS | Intel | `x86_64-apple-darwin` | Pre-2020 Macs |
| Windows | x86_64 | `x86_64-pc-windows-msvc` | Windows 10/11 |
| Windows | ARM64 | `aarch64-pc-windows-msvc` | Surface Pro X, ARM laptops |

**Platform-Specific Notes:**

- **Linux**: Uses `inotify` for file watching. Systemd user service for daemon management.
- **macOS**: Uses `FSEvents` for file watching. Launchd plist for daemon management.
- **Windows**: Uses `ReadDirectoryChangesW` for file watching. Service support planned.

**ONNX Runtime Static Linking:**

The Rust daemon statically links ONNX Runtime for embedding generation. This ensures self-contained binaries that work without external dependencies:

| Platform | Build Approach |
|----------|----------------|
| Linux x86_64 | Static linking via `ort` crate prebuilt binaries |
| Linux ARM64 | Static linking via `ort` crate prebuilt binaries |
| macOS ARM64 | Static linking via `ort` crate prebuilt binaries |
| macOS Intel | **Special case**: ONNX Runtime compiled standalone first, then statically linked (no prebuilt binaries available from `ort` crate) |
| Windows x86_64 | Static linking via `ort` crate prebuilt binaries |
| Windows ARM64 | Static linking via `ort` crate prebuilt binaries |

**Critical Requirements:**
- Release binaries MUST be self-contained (no external ONNX Runtime dependency)
- Source builds MUST also produce self-contained binaries
- Users should NEVER need to install ONNX Runtime separately (e.g., via Homebrew)
- The `ort-load-dynamic` feature MUST NOT be used in production builds

**Allowed External Dependencies (per platform):**
- **macOS**: System libraries in `/usr/lib/` and `/System/Library/Frameworks/` only
- **Linux**: `libc`, `libm`, `libdl`, `libpthread`, `libgcc_s`, `libstdc++`, `librt` only
- **Windows**: Windows system DLLs (`KERNEL32`, `ADVAPI32`, `WS2_32`, `bcrypt`, etc.) only
- **All platforms**: Tree-sitter grammar files (external `.so`/`.dylib`/`.dll` in cache dir) and LSP binaries (external executables)

**Runtime Prerequisites (user-installed):**

Tree-sitter grammars are distributed as C source code and compiled locally on first use. Users must have a C compiler available:

| Platform | Compiler | Install Command |
|----------|----------|-----------------|
| macOS | Clang (Xcode CLT) | `xcode-select --install` |
| Linux (Debian/Ubuntu) | GCC | `apt install build-essential` |
| Linux (Fedora/RHEL) | GCC | `dnf groupinstall "Development Tools"` |
| Linux (Arch) | GCC | `pacman -S base-devel` |
| Windows | MSVC | Visual Studio Build Tools with C++ workload |

A C++ compiler is additionally required for grammars with C++ external scanners (tracked via the `has_cpp_scanner` field in the Language Registry). The above packages include both C and C++ compilers.

If no compiler is found, the daemon logs a warning and falls back to text-based chunking for that language.

**Release Verification:**
The CI release workflow (`release.yml`) enforces self-contained binaries via:
1. Per-platform dependency verification scripts (`scripts/verify-deps-{linux,macos,windows}.*`) that **fail the build** if unexpected dynamic dependencies are found
2. Smoke tests (`--version`) that verify binaries are runnable before release

**Intel Mac Build Pipeline:**
Since the `ort` crate doesn't provide prebuilt static libraries for `x86_64-apple-darwin`:
1. CI downloads ONNX Runtime source or prebuilt static library
2. Compiles/extracts static library for Intel Mac target
3. Links statically with the daemon during cargo build
4. Produces self-contained `memexd` binary

**Tree-sitter Grammar Compatibility:**

Pre-compiled grammar libraries are included for each platform. Custom grammars can be loaded from `~/.workspace-qdrant/grammars/`.

### Installation Methods

#### Binary Releases (Recommended)

Download pre-built binaries from GitHub Releases:

**Quick Install (Recommended):**

```bash
# Linux/macOS - one-liner
curl -fsSL https://raw.githubusercontent.com/ChrisGVE/workspace-qdrant-mcp/main/scripts/download-install.sh | bash

# Windows (PowerShell) - one-liner
irm https://raw.githubusercontent.com/ChrisGVE/workspace-qdrant-mcp/main/scripts/download-install.ps1 | iex
```

**Manual Download:**

```bash
# Linux/macOS
VERSION="v0.4.0"  # Or use 'latest'
PLATFORM="$(uname -s | tr '[:upper:]' '[:lower:]')-$(uname -m | sed 's/x86_64/x64/' | sed 's/aarch64/arm64/')"
curl -fsSL "https://github.com/ChrisGVE/workspace-qdrant-mcp/releases/download/${VERSION}/workspace-qdrant-mcp-${PLATFORM}.tar.gz" | tar xz
sudo mv wqm memexd /usr/local/bin/

# Windows (PowerShell)
$Version = "v0.4.0"
Invoke-WebRequest -Uri "https://github.com/ChrisGVE/workspace-qdrant-mcp/releases/download/$Version/workspace-qdrant-mcp-windows-x64.zip" -OutFile wqm.zip
Expand-Archive wqm.zip -DestinationPath "$env:LOCALAPPDATA\Programs\wqm"
# Add to PATH manually
```

**Release Assets:**

Each release includes:
- `wqm` - CLI binary
- `memexd` - Daemon binary
- `grammars/` - Pre-compiled tree-sitter grammars
- `assets/` - Default configuration files

#### Source Build

For development or custom builds:

```bash
# Prerequisites
# - Rust 1.75+ (rustup recommended)
# - Node.js 18+ with npm
# - Protocol Buffers compiler (protoc)

# Clone and build
git clone https://github.com/ChrisGVE/workspace-qdrant-mcp.git
cd workspace-qdrant-mcp

# Use the install script
./install.sh                    # Interactive mode
./install.sh --release          # Release build
./install.sh --prefix=/usr/local # Custom install location
./install.sh --no-daemon        # CLI only, no daemon

# Or build manually
cd src/rust/daemon && cargo build --release
cd src/rust/cli && cargo build --release
cd src/typescript/mcp-server && npm install && npm run build
```

**Install Script Options:**

| Option | Description |
|--------|-------------|
| `--prefix PATH` | Installation directory (default: `~/.local`) |
| `--force` | Clean rebuild from scratch (cargo clean) |
| `--cli-only` | Build only CLI, skip daemon |
| `--no-service` | Skip daemon service setup instructions |
| `--no-verify` | Skip verification steps |

#### npm Global Install (MCP Server Only)

For MCP server without daemon:

```bash
npm install -g workspace-qdrant-mcp

# Verify installation
npx workspace-qdrant-mcp --version
```

> **Note:** The npm package only includes the MCP server. For full functionality, install the Rust daemon separately.

#### Package Managers

**Homebrew (macOS/Linux):**

A Homebrew formula is available in the repository:

```bash
# Install from local formula (when building from source)
brew install --build-from-source ./Formula/workspace-qdrant-mcp.rb

# Or install from tap (when tap is configured)
brew tap ChrisGVE/workspace-qdrant-mcp
brew install workspace-qdrant-mcp
```

The formula downloads pre-built binaries and configures the daemon as a Homebrew service.

> **Note:** Homebrew tap publication is planned for a future release.

**apt (Debian/Ubuntu):**
```bash
# Planned for future release
# sudo add-apt-repository ppa:workspace-qdrant-mcp/stable
# sudo apt update
# sudo apt install workspace-qdrant-mcp
```

**winget (Windows):**
```powershell
# Planned for future release
# winget install workspace-qdrant-mcp
```

### Docker Deployment Options

Four Docker deployment modes are supported:

#### Mode 1: All-in-One (Development)

Bundled Qdrant for quick setup:

```yaml
# docker-compose.dev.yml
services:
  memexd:
    image: workspace-qdrant-mcp/daemon:latest
    volumes:
      - ~/.workspace-qdrant:/data
      - ~/projects:/projects:ro
    environment:
      - QDRANT_URL=http://qdrant:6333
    depends_on:
      - qdrant

  qdrant:
    image: qdrant/qdrant:v1.7.3
    volumes:
      - qdrant_data:/qdrant/storage
```

```bash
docker-compose -f docker/docker-compose.dev.yml up -d
```

#### Mode 2: Qdrant-External (Production)

Separate Qdrant container for isolation:

```yaml
# docker-compose.prod.yml
services:
  memexd:
    image: workspace-qdrant-mcp/daemon:latest
    environment:
      - QDRANT_URL=http://qdrant:6333
      - QDRANT_API_KEY=${QDRANT_API_KEY}
```

#### Mode 3: Qdrant-Local

Use host-installed Qdrant:

```bash
# Install Qdrant on host
docker run -p 6333:6333 qdrant/qdrant:v1.7.3

# Run daemon pointing to host
docker run -e QDRANT_URL=http://host.docker.internal:6333 workspace-qdrant-mcp/daemon
```

#### Mode 4: Qdrant-Cloud

Connect to managed Qdrant Cloud:

```bash
# Set environment variables
export QDRANT_URL=https://your-cluster.qdrant.io:6333
export QDRANT_API_KEY=your-api-key

# Run daemon
docker run -e QDRANT_URL -e QDRANT_API_KEY workspace-qdrant-mcp/daemon
```

**Docker Environment Variables:**

| Variable | Default | Description |
|----------|---------|-------------|
| `QDRANT_URL` | `http://localhost:6333` | Qdrant server URL |
| `QDRANT_API_KEY` | (none) | Qdrant API key |
| `WQM_DATABASE_PATH` | `~/.workspace-qdrant/state.db` | SQLite database path |
| `WQM_LOG_LEVEL` | `INFO` | Log level (DEBUG, INFO, WARN, ERROR) |
| `WQM_GRPC_PORT` | `50051` | gRPC server port |

#### Docker Images

Docker images are published to GitHub Container Registry on each release:

```bash
# Pull latest version
docker pull ghcr.io/chrisgve/workspace-qdrant-mcp:latest

# Pull specific version
docker pull ghcr.io/chrisgve/workspace-qdrant-mcp:0.4.0
docker pull ghcr.io/chrisgve/workspace-qdrant-mcp:0.4
docker pull ghcr.io/chrisgve/workspace-qdrant-mcp:0

# Run the daemon
docker run -d \
  -p 50051:50051 \
  -e QDRANT_URL=http://host.docker.internal:6333 \
  -v ~/.workspace-qdrant:/data \
  ghcr.io/chrisgve/workspace-qdrant-mcp:latest
```

Available platforms: `linux/amd64`, `linux/arm64`

### Initial Configuration

#### First-Run Setup

On first run, the system initializes automatically:

1. **Config file generation:**
   ```bash
   wqm config init                 # Generate default config
   wqm config show                 # Display current config
   wqm config edit                 # Open config in editor
   ```

2. **Config file locations (search order):**
   - `~/.workspace-qdrant/config.yaml`
   - `~/.config/workspace-qdrant/config.yaml`
   - `~/Library/Application Support/workspace-qdrant/config.yaml` (macOS)
   - `%APPDATA%\workspace-qdrant\config.yaml` (Windows)

3. **Qdrant connection validation:**
   ```bash
   wqm admin health               # Check Qdrant connectivity
   wqm admin collections          # List collections
   ```

4. **SQLite database initialization:**
   The daemon automatically creates `~/.workspace-qdrant/state.db` on first run with all required tables.

#### MCP Client Configuration

**Claude Desktop (`~/.config/claude/config.json`):**

```json
{
  "mcpServers": {
    "workspace-qdrant": {
      "command": "npx",
      "args": ["workspace-qdrant-mcp"],
      "env": {
        "QDRANT_URL": "http://localhost:6333"
      }
    }
  }
}
```

**Claude Code (`.mcp.json` in project root):**

```json
{
  "mcpServers": {
    "workspace-qdrant": {
      "command": "npx",
      "args": ["workspace-qdrant-mcp"],
      "env": {
        "QDRANT_URL": "http://localhost:6333"
      }
    }
  }
}
```

### Service Management

The daemon (`memexd`) runs as a background service for continuous file watching and processing.

#### CLI Commands

```bash
wqm service install              # Install service (platform-specific)
wqm service start                # Start daemon
wqm service stop                 # Stop daemon
wqm service restart              # Restart daemon
wqm service status               # Check daemon status
wqm service logs                 # View daemon logs
wqm service logs --follow        # Follow logs in real-time
```

#### macOS (launchd)

The daemon installs as a launchd user agent:

**Plist location:** `~/Library/LaunchAgents/com.workspace-qdrant.memexd.plist`

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.workspace-qdrant.memexd</string>
    <key>ProgramArguments</key>
    <array>
        <string>/usr/local/bin/memexd</string>
    </array>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
    <key>StandardOutPath</key>
    <string>~/Library/Logs/workspace-qdrant/daemon.log</string>
    <key>StandardErrorPath</key>
    <string>~/Library/Logs/workspace-qdrant/daemon.err</string>
</dict>
</plist>
```

**Manual control:**
```bash
launchctl load ~/Library/LaunchAgents/com.workspace-qdrant.memexd.plist
launchctl unload ~/Library/LaunchAgents/com.workspace-qdrant.memexd.plist
launchctl list | grep memexd
```

#### Linux (systemd)

The daemon installs as a systemd user service:

**Service file:** `~/.config/systemd/user/memexd.service`

```ini
[Unit]
Description=Workspace Qdrant Daemon
After=network.target

[Service]
Type=simple
ExecStart=/usr/local/bin/memexd
Restart=on-failure
RestartSec=5
Environment=QDRANT_URL=http://localhost:6333

# Logging: stdout/stderr go to journald
StandardOutput=journal
StandardError=journal
SyslogIdentifier=memexd

# State directory for file-based logs
StateDirectory=workspace-qdrant

[Install]
WantedBy=default.target
```

**Manual control:**
```bash
systemctl --user daemon-reload
systemctl --user enable memexd
systemctl --user start memexd
systemctl --user status memexd

# View logs via journalctl
journalctl --user -u memexd -f

# Or via file (daemon writes to both)
tail -f ~/.local/state/workspace-qdrant/logs/daemon.jsonl
```

#### Windows

Windows service support is available via `wqm service` commands:

```powershell
# Install as Windows service (requires Administrator)
wqm service install

# Start/stop/status
wqm service start
wqm service stop
wqm service status

# View logs
wqm service logs --lines 50

# Uninstall service
wqm service uninstall
```

**Manual startup (without service):**
```powershell
# Run in foreground
memexd.exe --foreground

# Run as background process
Start-Process -NoNewWindow memexd.exe
```

> **Note:** Windows service management requires Administrator privileges. The service runs as LocalSystem by default.

#### Health Checks

```bash
# Quick health check
wqm admin health

# Detailed diagnostics
wqm admin health --verbose

# Check specific components
wqm admin health --component qdrant
wqm admin health --component daemon
wqm admin health --component database
```

**Health check output:**
```
Component       Status    Latency   Details
─────────────────────────────────────────────
Qdrant          healthy   12ms      v1.7.3, 4 collections
Daemon          healthy   2ms       pid=12345, uptime=2h15m
Database        healthy   1ms       23 watch folders, 156 queue items
```

### Logging and Observability

The daemon produces structured logs using the `tracing` crate with JSON output for machine parsing. Logs follow OS-canonical paths and integrate with platform service managers.

#### Canonical Log Paths

Log files follow platform-specific conventions:

| OS | Log Directory | Environment Override |
|----|---------------|---------------------|
| **Linux** | `$XDG_STATE_HOME/workspace-qdrant/logs/` (default: `~/.local/state/workspace-qdrant/logs/`) | `WQM_LOG_DIR` |
| **macOS** | `~/Library/Logs/workspace-qdrant/` | `WQM_LOG_DIR` |
| **Windows** | `%LOCALAPPDATA%\workspace-qdrant\logs\` | `WQM_LOG_DIR` |

**Log files:**

| File | Component | Format | Description |
|------|-----------|--------|-------------|
| `daemon.jsonl` | Rust daemon | JSON Lines | Primary daemon structured logs |
| `mcp-server.jsonl` | MCP Server | JSON Lines | TypeScript MCP server logs |

**Note:** Daemon and MCP Server logs are kept in **separate files** to prevent corruption if one component crashes while writing. The CLI merges them for unified viewing.

**Important:** The `~/.workspace-qdrant/` directory is reserved for **configuration only**, not logs. On Linux, if `$XDG_CONFIG_HOME` is set, configuration moves to `$XDG_CONFIG_HOME/workspace-qdrant/`.

#### Service Manager Integration

When running as a managed service, logs are captured by the platform service manager in addition to file output:

**Linux (systemd):**
```bash
# Primary access via journalctl
journalctl --user -u memexd -f              # Follow logs
journalctl --user -u memexd -n 100          # Last 100 entries
journalctl --user -u memexd --since "1 hour ago" --output=json

# File-based logs also available
cat ~/.local/state/workspace-qdrant/logs/daemon.jsonl | jq .
```

**macOS (launchctl):**
```bash
# LaunchAgent captures stdout/stderr to plist-specified paths
tail -f ~/Library/Logs/workspace-qdrant/daemon.log

# macOS unified log (limited - mainly for crashes)
log show --predicate 'process == "memexd"' --last 1h
```

**Windows:**
```powershell
# File-based logs
Get-Content "$env:LOCALAPPDATA\workspace-qdrant\logs\daemon.jsonl" -Tail 100

# Windows Event Log (for service events)
Get-EventLog -LogName Application -Source memexd -Newest 50
```

#### Log Format (JSON Lines)

Each log entry is a single JSON object:

```json
{"timestamp":"2026-02-05T10:30:45.123Z","level":"INFO","target":"memexd::processing","message":"Document processed","fields":{"document_id":"abc123","duration_ms":45.2,"collection":"projects","tenant_id":"proj_xyz"}}
```

**Standard fields:**

| Field | Type | Description |
|-------|------|-------------|
| `timestamp` | ISO 8601 | UTC timestamp |
| `level` | string | TRACE, DEBUG, INFO, WARN, ERROR |
| `target` | string | Rust module path |
| `message` | string | Human-readable message |
| `fields` | object | Structured context (varies by log type) |
| `span` | object | Active tracing span (if any) |

#### MCP Server Logging

The MCP Server uses `pino` for structured JSON logging directly to file. **No stderr output** is used to avoid potential future MCP protocol conflicts.

**Why file-only (no stderr):**

- MCP stdio transport uses stdout for protocol messages
- stderr could be used by future MCP protocol extensions
- File logging is reliable and doesn't interfere with any transport

**MCP Server log format:**

```json
{"level":30,"time":1707134445123,"pid":12345,"hostname":"workstation","name":"mcp-server","msg":"Tool called","session_id":"abc123","tool":"search","duration_ms":45}
```

**MCP Server log fields:**

| Field | Type | Description |
|-------|------|-------------|
| `level` | number | pino level (10=trace, 20=debug, 30=info, 40=warn, 50=error) |
| `time` | number | Unix timestamp (milliseconds) |
| `name` | string | Always `"mcp-server"` |
| `msg` | string | Log message |
| `session_id` | string | MCP session identifier (for correlation) |

**TypeScript implementation:**

```typescript
import pino from 'pino';
import { getLogDirectory } from './utils/paths';

const logger = pino({
  name: 'mcp-server',
  level: process.env.WQM_LOG_LEVEL || 'info',
  transport: {
    target: 'pino/file',
    options: { destination: `${getLogDirectory()}/mcp-server.jsonl` }
  }
});

// Usage
logger.info({ session_id, tool: 'search', duration_ms: 45 }, 'Tool called');
```

**Log rotation:** MCP Server logs follow the same rotation settings as daemon logs.

#### CLI Log Access

The CLI provides unified log access across all platforms, merging daemon and MCP server logs:

```bash
wqm debug logs                       # Show recent logs from all components
wqm debug logs -n 100                # Last 100 entries
wqm debug logs --follow              # Follow in real-time (both files)
wqm debug logs --errors-only         # Filter to WARN and ERROR
wqm debug logs --json                # Output raw JSON (for piping to jq)
wqm debug logs --since "1 hour ago"

# Component filtering
wqm debug logs --component daemon      # Daemon logs only
wqm debug logs --component mcp-server  # MCP Server logs only
wqm debug logs --component all         # Both (default)

# Correlation
wqm debug logs --session <session_id>  # Filter by MCP session
```

**Log merging behavior:**

- CLI reads both `daemon.jsonl` and `mcp-server.jsonl`
- Entries are merged and sorted by timestamp
- Component origin is indicated in output (unless `--json`)

**Log source priority:**

1. Canonical log files (daemon.jsonl, mcp-server.jsonl)
2. Service manager logs (journalctl on Linux) - daemon only
3. Fallback locations (for backwards compatibility)

#### Log Rotation

Log rotation is handled by the daemon:

| Setting | Default | Description |
|---------|---------|-------------|
| `max_log_size` | 50 MB | Rotate when file exceeds this size |
| `max_log_files` | 5 | Number of rotated files to keep |
| `compress_rotated` | true | Gzip rotated files |

**Rotated file naming:** `daemon.jsonl.1`, `daemon.jsonl.2.gz`, etc.

#### OpenTelemetry Integration (Optional)

For production deployments with distributed tracing infrastructure:

```bash
# Enable OTLP export
export OTEL_EXPORTER_OTLP_ENDPOINT=http://jaeger:4317
export OTEL_SERVICE_NAME=memexd
export OTEL_TRACES_SAMPLER_ARG=0.1  # 10% sampling

# Traces are exported to the configured backend
# View in Jaeger, Zipkin, Grafana Tempo, etc.
```

**Note:** OpenTelemetry is for distributed tracing correlation, not log viewing. It requires external infrastructure and is optional.

#### Environment Variables

**Shared (Daemon and MCP Server):**

| Variable | Default | Description |
|----------|---------|-------------|
| `WQM_LOG_DIR` | (OS-canonical) | Override log directory for both components |
| `WQM_LOG_LEVEL` | `info` | Minimum log level (trace, debug, info, warn, error) |

**Daemon-specific:**

| Variable | Default | Description |
|----------|---------|-------------|
| `WQM_LOG_JSON` | `true` | Enable JSON output (daemon) |
| `WQM_LOG_CONSOLE` | `false` (service) / `true` (foreground) | Console output (daemon) |
| `RUST_LOG` | - | Fine-grained module filtering (e.g., `memexd=debug,hyper=warn`) |

**MCP Server-specific:**

| Variable | Default | Description |
|----------|---------|-------------|
| `WQM_MCP_LOG_LEVEL` | `WQM_LOG_LEVEL` | Override log level for MCP Server only |

**Note:** MCP Server does not support console output to avoid protocol interference.

---

### CI/CD and Release Process

#### Version Tagging

Semantic versioning with optional pre-release tags:

| Pattern | Example | Description |
|---------|---------|-------------|
| `vX.Y.Z` | `v0.4.1` | Stable release |
| `vX.Y.Z-rc.N` | `v0.5.0-rc.1` | Release candidate |
| `vX.Y.Z-beta.N` | `v0.5.0-beta.1` | Beta release |
| `vX.Y.Z-alpha.N` | `v0.5.0-alpha.1` | Alpha release |

#### Release Workflow

```
Tag Push (vX.Y.Z)
       │
       ▼
┌──────────────────┐
│  Build Matrix    │  6 platform builds in parallel
│  (GitHub Actions)│
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  Test Suite      │  Unit tests, integration tests
│                  │  per platform
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  Create Release  │  Draft release with
│                  │  changelog generation
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  Upload Assets   │  Binaries, checksums,
│                  │  documentation
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  Publish Release │  Make release public
└──────────────────┘
```

**GitHub Workflows:**

| Workflow | Trigger | Purpose |
|----------|---------|---------|
| `ci.yml` | Push, PR | Build and test |
| `release.yml` | Tag push | Create release |
| `tree-sitter-version-bump.yml` | Schedule | Update tree-sitter grammars |
| `onnx-runtime-version-bump.yml` | Schedule | Update ONNX Runtime |

#### Automated Dependency Updates

- **Tree-sitter grammars:** Weekly check for grammar updates
- **ONNX Runtime:** Monthly check for new versions
- **Rust dependencies:** Dependabot PRs for security updates

### Upgrade and Migration

#### Release Packaging (Placeholder)

Define per-OS release packaging and installer expectations, and how the CLI performs updates.

- **macOS**: Specify installer format (e.g., `.pkg` or `.dmg`) and signing/notarization requirements.
- **Windows**: Specify installer format (e.g., `.msi` or `.exe`) and code signing requirements.
- **Linux**: Specify package formats (e.g., `.deb`, `.rpm`, AppImage, or tarball) and repository strategy.
- **CLI-driven updates**: `wqm update` is the primary mechanism for installing updates across all OSes and should handle download, verification, install, and service restart steps.

> Placeholder: fill in exact packaging formats, install locations, and service integration per OS.

#### Standard Upgrade

```bash
# Stop the daemon
wqm service stop

# Download new version
curl -fsSL https://github.com/[org]/workspace-qdrant-mcp/releases/latest/download/wqm-$(uname -s)-$(uname -m).tar.gz | tar xz
sudo mv wqm /usr/local/bin/
sudo mv memexd /usr/local/bin/

# Start the daemon (auto-migrates database)
wqm service start

# Verify
wqm --version
wqm admin health
```

#### Database Migration

The daemon automatically applies database migrations on startup:

1. Reads `schema_version` table to determine current version
2. Applies pending migrations in order
3. Updates `schema_version` table

**Manual migration check:**
```bash
wqm admin db-version              # Show current schema version
wqm admin db-migrate --dry-run    # Preview pending migrations
```

#### Rollback Procedure

If an upgrade causes issues:

```bash
# Stop the daemon
wqm service stop

# Restore previous binaries (keep backups!)
sudo mv /usr/local/bin/wqm.backup /usr/local/bin/wqm
sudo mv /usr/local/bin/memexd.backup /usr/local/bin/memexd

# Start the daemon
wqm service start
```

> **Warning:** Database schema rollbacks are not supported. Create SQLite backups before upgrading.

#### Self-Update Command

The `wqm update` command provides in-place binary updates:

```bash
# Check for updates
wqm update check

# Update to latest stable version
wqm update

# Update to latest version in specific channel
wqm update --channel stable       # Default: stable releases only
wqm update --channel beta         # Include beta releases
wqm update --channel rc           # Include release candidates
wqm update --channel alpha        # Include alpha releases

# Update to specific version
wqm update --version v0.5.0

# Force reinstall current version
wqm update --force

# Install specific version with force
wqm update install --version v0.4.0 --force
```

**Update process:**
1. Fetches release info from GitHub API
2. Downloads platform-specific binary
3. Verifies SHA256 checksum
4. Stops running daemon
5. Replaces binary (with backup)
6. Restarts daemon

> **Note:** Updates require write permission to the installation directory. The daemon is automatically restarted after update.

### Troubleshooting

#### Common Issues

**Issue: Daemon won't start**

```bash
# Check if already running
pgrep memexd

# Check port availability
lsof -i :50051

# Check logs
wqm service logs --lines 50

# Run in foreground for debugging
memexd --foreground --log-level debug
```

**Issue: Qdrant connection failed**

```bash
# Test Qdrant connectivity
curl http://localhost:6333/health

# Check environment
echo $QDRANT_URL

# Verify config
wqm config show | grep qdrant
```

**Issue: File changes not detected**

```bash
# Check watch folders
wqm watch list

# Verify file system events
wqm admin debug --watch-events /path/to/project

# Check inotify limits (Linux)
cat /proc/sys/fs/inotify/max_user_watches
# Increase if needed:
echo 524288 | sudo tee /proc/sys/fs/inotify/max_user_watches
```

**Issue: High memory usage**

```bash
# Check queue size
wqm queue stats

# Clear old queue items
wqm queue clean --days 7

# Reduce batch size in config
wqm config set daemon.queue_batch_size 5
```

#### Diagnostic Commands

```bash
# Full system diagnostic
wqm admin diagnose

# Export diagnostic report
wqm admin diagnose --output report.json

# Check specific subsystems
wqm admin diagnose --component file-watching
wqm admin diagnose --component embedding
wqm admin diagnose --component grpc
```

---

