# Installation Reference

workspace-qdrant-mcp consists of three components:

- **memexd** — Rust daemon for file watching, embedding generation, and queue processing
- **wqm** — Rust CLI for service management, configuration, and queue inspection
- **MCP Server** — TypeScript server exposing the 6 MCP tools to LLM clients

All three components must be installed for full functionality. The MCP Server and CLI can operate without the daemon for limited read-only queries, but file indexing and write operations require `memexd`.

---

## Platform Support

| Platform | Architecture | Target Triple |
|----------|--------------|---------------|
| macOS | Apple Silicon (M1/M2/M3) | `aarch64-apple-darwin` |
| macOS | Intel | `x86_64-apple-darwin` |
| Linux | x86_64 | `x86_64-unknown-linux-gnu` |
| Linux | ARM64 | `aarch64-unknown-linux-gnu` |
| Windows | x86_64 | `x86_64-pc-windows-msvc` |
| Windows | ARM64 | `aarch64-pc-windows-msvc` |

Pre-built release binaries are self-contained and require no external runtime dependencies (ONNX Runtime is statically linked).

---

## Prerequisites

Before installing workspace-qdrant-mcp, a running Qdrant instance is required. See [Qdrant Setup](#qdrant-setup) below.

---

## Installation Methods

### Homebrew (macOS/Linux)

A Homebrew formula is available in the repository. Tap-based installation is planned for a future release.

**Local formula install (current):**

```bash
brew install --build-from-source ./Formula/workspace-qdrant-mcp.rb
```

**Tap-based install (planned — not yet available):**

```bash
brew tap chrisgve/tap
brew install workspace-qdrant-mcp
brew services start workspace-qdrant-mcp
```

The formula installs both `wqm` and `memexd` binaries and registers the daemon as a Homebrew service.

---

### Pre-built Binaries (Recommended)

Downloads the latest release binaries from GitHub, verifies SHA256 checksums, and installs to `~/.local/bin` (Linux/macOS) or `%LOCALAPPDATA%\wqm\bin` (Windows).

#### Linux and macOS

**One-liner (installs both `wqm` and `memexd`):**

```bash
curl -fsSL https://raw.githubusercontent.com/ChrisGVE/workspace-qdrant-mcp/main/scripts/download-install.sh | bash
```

**With options:**

```bash
# Download the script first for inspection
curl -fsSL https://raw.githubusercontent.com/ChrisGVE/workspace-qdrant-mcp/main/scripts/download-install.sh -o install.sh
chmod +x install.sh

# Install latest to default prefix (~/.local)
./install.sh

# Install to a custom prefix
./install.sh --prefix /usr/local

# Install a specific version
./install.sh --version v0.4.0

# Install CLI only (no daemon)
./install.sh --cli-only
```

**Installer options:**

| Option | Description | Default |
|--------|-------------|---------|
| `--prefix PATH` | Installation prefix; binaries go to `PATH/bin` | `~/.local` |
| `--version VERSION` | Specific release tag to install | latest |
| `--cli-only` | Skip daemon (`memexd`), install `wqm` only | false |

**Environment variable equivalents:**

| Variable | Equivalent option |
|----------|-------------------|
| `INSTALL_PREFIX` | `--prefix` |
| `WQM_VERSION` | `--version` |

After installation, ensure `~/.local/bin` is on your `PATH`:

```bash
# Add to ~/.bashrc or ~/.zshrc
export PATH="$HOME/.local/bin:$PATH"
```

#### Windows (PowerShell)

**One-liner:**

```powershell
irm https://raw.githubusercontent.com/ChrisGVE/workspace-qdrant-mcp/main/scripts/download-install.ps1 | iex
```

**With options:**

```powershell
# Download and run with parameters
$script = Invoke-WebRequest -Uri "https://raw.githubusercontent.com/ChrisGVE/workspace-qdrant-mcp/main/scripts/download-install.ps1" -UseBasicParsing
$sb = [ScriptBlock]::Create($script.Content)

# Install specific version
& $sb -Version v0.4.0

# Custom prefix
& $sb -Prefix "C:\Tools\wqm"

# CLI only
& $sb -CliOnly
```

**PowerShell installer parameters:**

| Parameter | Description | Default |
|-----------|-------------|---------|
| `-Prefix` | Installation directory | `%LOCALAPPDATA%\wqm` |
| `-Version` | Release tag to install | latest |
| `-CliOnly` | Skip daemon (`memexd.exe`) | false |

After installation, add the bin directory to your `PATH`:

```powershell
# Temporary (current session)
$env:PATH += ";$env:LOCALAPPDATA\wqm\bin"

# Permanent (via System Properties > Environment Variables)
# Add: %LOCALAPPDATA%\wqm\bin
```

#### What the installer does

1. Detects OS and CPU architecture, maps to the correct Rust target triple
2. Queries the GitHub Releases API to resolve the latest version (or uses `--version`)
3. Downloads `wqm` and `memexd` binaries (and `.sha256` checksum files)
4. Verifies SHA256 checksums (warns and continues if checksum file is unavailable)
5. Sets executable permissions and moves binaries to `$BIN_DIR`
6. Prints next-step instructions

---

### npm Global Install (MCP Server only)

The MCP Server can be installed independently via npm. This does not install the daemon or CLI.

```bash
npm install -g workspace-qdrant-mcp

# Verify
npx workspace-qdrant-mcp --version
```

The npm package is the recommended installation method when you only need the MCP Server and manage the daemon separately.

---

### Build from Source

Use this method for development or when pre-built binaries are unavailable for your platform.

#### Prerequisites

| Requirement | Minimum Version | Notes |
|-------------|-----------------|-------|
| Rust (via rustup) | 1.75 | `curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \| sh` |
| ONNX Runtime static library | 1.23.x | See below |
| Node.js | 18 | Required for MCP Server only |
| protoc | Any recent | Required for gRPC code generation |

#### ONNX Runtime static library

The daemon statically links ONNX Runtime. You must provide the static library at build time via `ORT_LIB_LOCATION`.

**Option 1 — Pre-built static library (recommended for macOS Intel and all platforms):**

```bash
mkdir -p ~/.onnxruntime-static
curl -L "https://github.com/supertone-inc/onnxruntime-build/releases/download/v1.23.2/onnxruntime-osx-universal2-static_lib-1.23.2.tgz" \
  -o ~/.onnxruntime-static/ort.tgz
tar xzf ~/.onnxruntime-static/ort.tgz -C ~/.onnxruntime-static
rm ~/.onnxruntime-static/ort.tgz
```

Replace the download URL with the correct platform variant from [supertone-inc/onnxruntime-build releases](https://github.com/supertone-inc/onnxruntime-build/releases).

**Option 2 — Homebrew (macOS development only, dynamic linking):**

```bash
brew install onnxruntime
# Note: produces a dynamically-linked binary; not suitable for distribution
ORT_LIB_LOCATION=/usr/local/Cellar/onnxruntime/1.23.2_2 ORT_PREFER_DYNAMIC_LINK=1 cargo build --release ...
```

#### Build commands

```bash
git clone https://github.com/ChrisGVE/workspace-qdrant-mcp.git
cd workspace-qdrant-mcp

# Build daemon (memexd)
ORT_LIB_LOCATION=/path/to/onnxruntime/lib \
  cargo build --release \
  --manifest-path src/rust/Cargo.toml \
  --package memexd

# Build CLI (wqm)
ORT_LIB_LOCATION=/path/to/onnxruntime/lib \
  cargo build --release \
  --manifest-path src/rust/Cargo.toml \
  --package wqm-cli

# Build MCP Server
cd src/typescript/mcp-server
npm install
npm run build
```

Replace `/path/to/onnxruntime/lib` with the absolute path to your ONNX Runtime library directory (e.g. `$HOME/.onnxruntime-static/lib`). Do not use `$HOME` in the value — shell variable expansion may not occur in all invocation contexts. Use an explicit absolute path.

#### Deploy built binaries

```bash
# Copy to ~/.local/bin (or any directory on your PATH)
cp src/rust/target/release/memexd ~/.local/bin/memexd
cp src/rust/target/release/wqm ~/.local/bin/wqm
```

---

## Qdrant Setup

workspace-qdrant-mcp requires a running Qdrant instance. Three deployment options are available.

### Docker (local, recommended for development)

```bash
docker run -d \
  --name qdrant \
  -p 6333:6333 \
  -p 6334:6334 \
  -v "$HOME/.qdrant/storage:/qdrant/storage" \
  qdrant/qdrant:v1.7.3
```

The `-v` mount persists vector data across container restarts. Without it, all indexed content is lost when the container stops.

**Verify Qdrant is running:**

```bash
curl http://localhost:6333/health
```

Expected response: `{"title":"qdrant - vector search engine","version":"..."}`

### Docker Compose (development stack)

A `docker-compose.dev.yml` is provided that starts Qdrant and the daemon together:

```bash
docker compose -f docker/docker-compose.dev.yml up -d
```

This starts:
- `qdrant` — vector database on ports 6333 (HTTP) and 6334 (gRPC)
- `memexd` — daemon connecting to the Qdrant container on port 50051
- `qdrant-web-ui` — Qdrant dashboard on port 8080

### Qdrant Cloud

Obtain your cluster URL and API key from [cloud.qdrant.io](https://cloud.qdrant.io).

Set the connection details before starting the daemon:

```bash
export QDRANT_URL=https://your-cluster.qdrant.io:6333
export QDRANT_API_KEY=your-api-key
```

Or specify them in the configuration file (see [Configuration Reference](configuration.md)):

```yaml
qdrant:
  url: "https://your-cluster.qdrant.io:6333"
  api_key: "your-api-key"
```

---

## Service Configuration

The daemon (`memexd`) runs as a background service and must be started before the MCP Server can process write operations.

### Install and start the service

After installing the binaries, use the CLI to register the daemon as a platform service:

```bash
wqm service install    # Register platform service (launchd / systemd / Windows SCM)
wqm service start      # Start the daemon
wqm service status     # Check status
```

### macOS (launchd)

`wqm service install` writes a launchd user agent plist and loads it:

**Plist location:** `~/Library/LaunchAgents/com.workspace-qdrant.memexd.plist`

**Log paths:**
- stdout: `~/Library/Logs/workspace-qdrant/daemon.log`
- stderr: `~/Library/Logs/workspace-qdrant/daemon.err`

**Manual launchctl control:**

```bash
# Load (start at login + start now)
launchctl load ~/Library/LaunchAgents/com.workspace-qdrant.memexd.plist

# Unload (stop and disable)
launchctl unload ~/Library/LaunchAgents/com.workspace-qdrant.memexd.plist

# Check if running
launchctl list | grep memexd
```

### Linux (systemd)

`wqm service install` writes a systemd user service unit and enables it:

**Service file:** `~/.config/systemd/user/memexd.service`

**Log access:**

```bash
# Follow logs via journalctl
journalctl --user -u memexd -f

# View last 100 entries
journalctl --user -u memexd -n 100

# File-based logs
tail -f ~/.local/state/workspace-qdrant/logs/daemon.jsonl
```

**Manual systemctl control:**

```bash
systemctl --user daemon-reload
systemctl --user enable memexd
systemctl --user start memexd
systemctl --user status memexd
systemctl --user stop memexd
```

### Windows

```powershell
# Install as Windows service (requires Administrator)
wqm service install
wqm service start
wqm service stop
wqm service status

# View logs
wqm service logs --lines 50
```

**Run in foreground (for debugging):**

```powershell
memexd.exe --foreground
```

### All service management commands

```bash
wqm service install      # Register platform service
wqm service start        # Start daemon
wqm service stop         # Stop daemon
wqm service restart      # Restart daemon
wqm service status       # Check daemon status
wqm service logs         # View daemon logs
wqm service logs --follow  # Stream logs in real-time
wqm service uninstall    # Remove service registration
```

---

## MCP Client Configuration

After installing and starting the daemon, configure your MCP client to use the server.

### Claude Desktop

Edit `~/.config/claude/config.json` (Linux/macOS) or `%APPDATA%\Claude\config.json` (Windows):

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

### Claude Code

Add a `.mcp.json` file at the project root:

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

---

## Verification

After installation, confirm all components are working:

```bash
# Check binary versions
wqm --version
memexd --version

# Run health check (requires Qdrant and daemon to be running)
wqm admin health

# List Qdrant collections (empty on first run)
wqm admin collections
```

**Expected `wqm admin health` output:**

```
Component       Status    Latency   Details
─────────────────────────────────────────────
Qdrant          healthy   12ms      v1.7.3, 4 collections
Daemon          healthy   2ms       pid=12345, uptime=2h15m
Database        healthy   1ms       23 watch folders, 156 queue items
```

If any component reports unhealthy, see the troubleshooting section in [docs/specs/13-deployment.md](../specs/13-deployment.md).

---

## First Run

On first run, the daemon creates the SQLite database automatically:

- **macOS/Linux:** `~/.workspace-qdrant/state.db`
- **Windows:** `%APPDATA%\workspace-qdrant\state.db`

Generate and inspect the default configuration file:

```bash
wqm config init    # Write default config to ~/.workspace-qdrant/config.yaml
wqm config show    # Display active configuration
wqm config edit    # Open config in $EDITOR
```

No manual configuration is required for a local Qdrant setup at the default address (`http://localhost:6333`). All defaults are embedded in the binaries and applied without a config file present.
