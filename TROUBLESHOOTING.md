# Troubleshooting Guide

Comprehensive troubleshooting guide for workspace-qdrant-mcp installation, configuration, and operation issues.

## Table of Contents

- [Installation Issues](#installation-issues)
- [Qdrant Connection Problems](#qdrant-connection-problems)
- [MCP Server Debugging](#mcp-server-debugging)
- [Daemon Startup Issues](#daemon-startup-issues)
- [Performance Troubleshooting](#performance-troubleshooting)
- [Configuration Guide](#configuration-guide)
- [Debugging Commands](#debugging-commands)
- [Log Locations](#log-locations)
- [Common Error Messages](#common-error-messages)

## Installation Issues

### Package Installation Fails

**Problem:** `uv tool install workspace-qdrant-mcp` fails with dependency errors

**Solutions:**

```bash
# Update uv to latest version
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clean cache and reinstall
uv cache clean
uv tool install workspace-qdrant-mcp

# Install with verbose output for debugging
uv tool install workspace-qdrant-mcp -v
```

**Common causes:**
- Outdated uv version (requires uv 0.1.0+)
- Python version incompatibility (requires Python 3.10+)
- Conflicting dependencies in environment
- Network issues downloading packages

**Verification:**
```bash
# Check Python version
python --version  # Should be 3.10+

# Check uv version
uv --version

# Verify installation
which workspace-qdrant-mcp
workspace-qdrant-mcp --help
```

### Missing System Dependencies

**Problem:** Installation succeeds but runtime errors due to missing system libraries

**Solutions:**

**macOS:**
```bash
# Install system dependencies
brew install openssl sqlite3

# For LSP support
brew install node  # Required for language servers
```

**Linux (Ubuntu/Debian):**
```bash
# Install system dependencies
sudo apt-get update
sudo apt-get install -y \
    libssl-dev \
    libsqlite3-dev \
    build-essential \
    pkg-config

# For LSP support
sudo apt-get install -y nodejs npm
```

**Linux (RHEL/CentOS/Fedora):**
```bash
# Install system dependencies
sudo dnf install -y \
    openssl-devel \
    sqlite-devel \
    gcc \
    gcc-c++ \
    make

# For LSP support
sudo dnf install -y nodejs npm
```

### Rust Engine Build Failures

**Problem:** Rust daemon compilation fails during installation

**Solutions:**

```bash
# Install/Update Rust toolchain
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
rustup update stable

# Build manually to see detailed errors
cd src/rust/daemon/core
cargo clean
cargo build --release

# Check for specific errors
cargo check
```

**Common causes:**
- Missing Rust compiler (install rustup)
- Outdated Rust version (update with `rustup update`)
- Missing C compiler or system headers
- Insufficient disk space for compilation

## Qdrant Connection Problems

### Connection Refused

**Problem:** `Connection refused` when attempting to connect to Qdrant

**Diagnosis:**
```bash
# Check if Qdrant is running
curl http://localhost:6333/collections

# Expected: JSON response with collections list
# If refused: Qdrant is not running
```

**Solutions:**

**Local Qdrant with Docker:**
```bash
# Start Qdrant container
docker run -d \
  --name qdrant \
  -p 6333:6333 \
  -p 6334:6334 \
  -v $(pwd)/qdrant_storage:/qdrant/storage \
  qdrant/qdrant

# Verify Qdrant is running
docker ps | grep qdrant
curl http://localhost:6333/healthz
```

**Local Qdrant with docker-compose:**
```bash
# Start services
docker-compose up -d qdrant

# Check logs
docker-compose logs qdrant
```

**Verify configuration:**
```bash
# Check QDRANT_URL environment variable
echo $QDRANT_URL  # Should be http://localhost:6333

# Test connection
wqm admin status
```

### Authentication Failures

**Problem:** `Unauthorized` or `403 Forbidden` errors with Qdrant Cloud

**Diagnosis:**
```bash
# Check if API key is set
echo $QDRANT_API_KEY  # Should not be empty

# Verify API key works
curl -H "api-key: $QDRANT_API_KEY" \
  https://your-cluster.qdrant.io:6333/collections
```

**Solutions:**

```bash
# Set API key in environment
export QDRANT_API_KEY="your-api-key-here"
export QDRANT_URL="https://your-cluster.qdrant.io:6333"

# Verify connection with CLI
wqm admin status

# For Claude Desktop, update config
# File: ~/Library/Application Support/Claude/claude_desktop_config.json
{
  "mcpServers": {
    "workspace-qdrant-mcp": {
      "command": "workspace-qdrant-mcp",
      "env": {
        "QDRANT_URL": "https://your-cluster.qdrant.io:6333",
        "QDRANT_API_KEY": "your-api-key-here"
      }
    }
  }
}
```

### Timeout Errors

**Problem:** Operations timeout before completing

**Diagnosis:**
```bash
# Check Qdrant response time
time curl http://localhost:6333/collections

# Check network latency to Qdrant Cloud
ping your-cluster.qdrant.io
```

**Solutions:**

**Increase timeout in configuration:**
```yaml
# config.yaml
qdrant:
  timeout: 60s  # Increase from default 30s

  pool:
    acquisition_timeout: 60s  # Increase connection timeout
```

**Environment variable override:**
```bash
export QDRANT_TIMEOUT=60
```

**Check network:**
```bash
# Test connection quality
curl -v http://localhost:6333/healthz

# For Qdrant Cloud, check SSL/TLS
openssl s_client -connect your-cluster.qdrant.io:6333
```

## MCP Server Debugging

### Claude Desktop Not Showing Tools

**Problem:** workspace-qdrant-mcp tools not appearing in Claude Desktop

**Diagnosis:**

```bash
# 1. Verify installation
which workspace-qdrant-mcp
# Should output path like: /Users/you/.local/bin/workspace-qdrant-mcp

# 2. Test server manually
workspace-qdrant-mcp --transport http --port 8000
# Should start without errors

# 3. Check Claude config file exists and is valid JSON
cat ~/Library/Application\ Support/Claude/claude_desktop_config.json | jq .
```

**Solutions:**

**1. Fix configuration syntax:**
```json
{
  "mcpServers": {
    "workspace-qdrant-mcp": {
      "command": "workspace-qdrant-mcp",
      "env": {
        "QDRANT_URL": "http://localhost:6333"
      }
    }
  }
}
```

**2. Verify server starts:**
```bash
# Test in HTTP mode (shows errors)
workspace-qdrant-mcp --transport http

# Check for startup errors
# Server should start on port 8000
```

**3. Restart Claude Desktop:**
- Completely quit Claude Desktop (⌘Q on macOS)
- Relaunch Claude Desktop
- Open new conversation
- Tools should appear

**4. Check Claude Desktop logs:**
```bash
# macOS
cat ~/Library/Logs/Claude/mcp.log

# Windows
type %APPDATA%\Claude\Logs\mcp.log

# Linux
cat ~/.config/Claude/logs/mcp.log
```

### MCP Tools Return Errors

**Problem:** Tools appear but return errors when used

**Diagnosis:**

```bash
# Test with HTTP mode for detailed errors
workspace-qdrant-mcp --transport http --port 8000

# In another terminal, test tools with curl
curl -X POST http://localhost:8000/manage \
  -H "Content-Type: application/json" \
  -d '{"action":"workspace_status"}'
```

**Solutions:**

**Check daemon connection:**
```bash
# Verify daemon is running
wqm service status

# Start daemon if not running
wqm service start

# Check daemon health
wqm admin status
```

**Check Qdrant connection:**
```bash
# Test Qdrant directly
curl http://localhost:6333/collections

# Verify environment variables
echo $QDRANT_URL
echo $QDRANT_API_KEY  # For cloud only
```

**Enable debug mode:**
```bash
# Start server with debug logging
workspace-qdrant-mcp --transport http --debug

# Check detailed error messages in output
```

### stdio Mode Issues

**Problem:** Server works in HTTP mode but not stdio mode

**Diagnosis:**

stdio mode is completely silent by design - no output means it's working correctly.

**Verification:**
```bash
# Check configuration is correct
cat ~/Library/Application\ Support/Claude/claude_desktop_config.json

# Ensure "command" uses absolute path or is in PATH
which workspace-qdrant-mcp

# Test that command works
workspace-qdrant-mcp --help
```

**Solutions:**

**Use absolute path in configuration:**
```json
{
  "mcpServers": {
    "workspace-qdrant-mcp": {
      "command": "/Users/yourname/.local/bin/workspace-qdrant-mcp",
      "env": {
        "QDRANT_URL": "http://localhost:6333"
      }
    }
  }
}
```

**Add PATH in environment:**
```json
{
  "mcpServers": {
    "workspace-qdrant-mcp": {
      "command": "workspace-qdrant-mcp",
      "env": {
        "PATH": "/Users/yourname/.local/bin:/usr/local/bin:/usr/bin:/bin",
        "QDRANT_URL": "http://localhost:6333"
      }
    }
  }
}
```

## Daemon Startup Issues

### Daemon Won't Start

**Problem:** `wqm service start` fails or daemon crashes immediately

**Diagnosis:**

```bash
# Check daemon status
wqm service status

# View daemon logs
wqm service logs

# Try starting manually to see errors
/path/to/src/rust/daemon/core/target/release/memexd --foreground --log-level debug
```

**Common errors and solutions:**

**1. Port already in use:**
```
Error: Address already in use (os error 48)
```

```bash
# Find process using port 50051
lsof -i :50051

# Kill the process
kill <PID>

# Or change port in configuration
# config.yaml
grpc:
  port: 50052  # Use different port
```

**2. Permission denied:**
```
Permission denied (os error 13)
```

```bash
# Check file permissions
ls -la ~/.local/share/workspace-qdrant-mcp/

# Fix permissions
chmod 755 ~/.local/share/workspace-qdrant-mcp
chmod 644 ~/.local/share/workspace-qdrant-mcp/*.db
```

**3. Missing database:**
```
Error: Unable to open database file
```

```bash
# Reinitialize state database
wqm config init-unified

# Verify database exists
ls -la ~/.local/share/workspace-qdrant-mcp/state.db
```

### Daemon Crashes Repeatedly

**Problem:** Daemon starts but crashes after a few seconds

**Diagnosis:**

```bash
# Run in foreground with debug logging
memexd --foreground --log-level debug

# Check system resource usage
top | grep memexd

# Check for out-of-memory kills
dmesg | grep -i kill
```

**Solutions:**

**Reduce resource usage:**
```yaml
# config.yaml
performance:
  max_concurrent_tasks: 2  # Reduce from default 4

embedding:
  batch_size: 25  # Reduce from default 50

auto_ingestion:
  max_files_per_batch: 3  # Reduce from default 5
```

**Check file limits:**
```bash
# Increase file descriptor limit
ulimit -n 4096

# Make permanent (add to ~/.bashrc or ~/.zshrc)
echo "ulimit -n 4096" >> ~/.bashrc
```

**Disable problematic features:**
```yaml
# config.yaml
auto_ingestion:
  enabled: false  # Disable auto-ingestion temporarily

observability:
  telemetry:
    enabled: false  # Reduce monitoring overhead
```

### gRPC Connection Failures

**Problem:** MCP server can't connect to daemon via gRPC

**Diagnosis:**

```bash
# Check if daemon is listening on gRPC port
lsof -i :50051
netstat -an | grep 50051

# Test gRPC connection
grpcurl -plaintext localhost:50051 list
```

**Solutions:**

**1. Verify daemon is running:**
```bash
wqm service status
# Should show "running"

# If not running, start it
wqm service start
```

**2. Check firewall settings:**
```bash
# macOS - allow connections
sudo /usr/libexec/ApplicationFirewall/socketfilterfw --add /path/to/memexd
sudo /usr/libexec/ApplicationFirewall/socketfilterfw --unblockapp /path/to/memexd
```

**3. Enable fallback mode:**
```yaml
# config.yaml
grpc:
  fallback_to_direct: true  # Enable direct Qdrant access fallback
```

## Performance Troubleshooting

### Slow Search Responses

**Problem:** Search queries taking longer than 150ms

**Diagnosis:**

```bash
# Run performance diagnostics
wqm observability diagnostics

# Check specific search performance
time wqm search "test query" --collection myproject

# Monitor system resources
top
htop  # If available
```

**Solutions:**

**1. Optimize Qdrant configuration:**
```yaml
# config.yaml
qdrant:
  prefer_grpc: true  # Use faster gRPC protocol

  default_collection:
    enable_indexing: true  # Ensure indexes are enabled

    hnsw:
      ef: 64  # Balance between speed and accuracy
      m: 16   # Good default connectivity
```

**2. Check collection health:**
```bash
# View collection statistics
wqm collections list --verbose

# Optimize collections
wqm admin collections
```

**3. Reduce query complexity:**
```bash
# Use smaller result limits
wqm search "query" --limit 5  # Instead of default 10

# Disable sparse vectors if not needed
# config.yaml
embedding:
  enable_sparse_vectors: false
```

**4. Check Qdrant performance:**
```bash
# Monitor Qdrant resource usage
docker stats qdrant

# Check Qdrant disk I/O
docker exec qdrant df -h
```

### High Memory Usage

**Problem:** Daemon or MCP server consuming excessive memory

**Diagnosis:**

```bash
# Check memory usage
ps aux | grep memexd
ps aux | grep workspace-qdrant-mcp

# Get detailed memory stats
wqm observability metrics --component memory
```

**Solutions:**

**1. Reduce batch sizes:**
```yaml
# config.yaml
embedding:
  batch_size: 25  # Reduce from 50
  chunk_size: 600  # Reduce from 800

auto_ingestion:
  max_files_per_batch: 3  # Reduce from 5
```

**2. Limit concurrent operations:**
```yaml
# config.yaml
performance:
  max_concurrent_tasks: 2  # Reduce parallelism

qdrant:
  pool:
    max_connections: 5  # Reduce from 10
```

**3. Disable telemetry:**
```yaml
# config.yaml
observability:
  telemetry:
    enabled: false  # Reduce memory overhead
```

**4. Use disk-based vectors:**
```yaml
# config.yaml
qdrant:
  default_collection:
    on_disk_vectors: true  # Store vectors on disk
```

### Slow Document Ingestion

**Problem:** File processing slower than 1000 docs/minute

**Diagnosis:**

```bash
# Check ingestion queue status
wqm status --verbose

# Monitor processing rate
wqm observability diagnostics

# Check for bottlenecks
wqm observability monitor
```

**Solutions:**

**1. Increase parallelism:**
```yaml
# config.yaml
performance:
  max_concurrent_tasks: 8  # Increase for multi-core systems

auto_ingestion:
  max_files_per_batch: 10  # Process more files per batch
```

**2. Optimize embedding generation:**
```yaml
# config.yaml
embedding:
  batch_size: 100  # Larger batches for better throughput

  # Use faster model
  model: "sentence-transformers/all-MiniLM-L6-v2"  # Fastest option
```

**3. Skip unnecessary files:**
```yaml
# config.yaml
workspace:
  custom_exclude_patterns:
    - "node_modules/**"
    - "venv/**"
    - "*.min.js"
    - "*.map"
```

**4. Increase debounce delay:**
```yaml
# config.yaml
auto_ingestion:
  debounce: 30s  # Wait longer before processing file changes
```

## Configuration Guide

### Configuration File Locations

**Priority order** (highest to lowest):
1. Command-line arguments
2. `--config` file specified on command line
3. Environment variables
4. `~/.config/workspace-qdrant-mcp/config.yaml`
5. Default values from `assets/default_configuration.yaml`

**System locations:**
- **Linux**: `~/.config/workspace-qdrant-mcp/`
- **macOS**: `~/Library/Application Support/workspace-qdrant-mcp/`
- **Windows**: `%APPDATA%\workspace-qdrant-mcp\`

### Environment Variables

Complete list of environment variables affecting behavior:

| Variable | Default | Description |
|----------|---------|-------------|
| `QDRANT_URL` | `http://localhost:6333` | Qdrant server URL |
| `QDRANT_API_KEY` | _(none)_ | Qdrant API key (cloud only) |
| `FASTEMBED_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Embedding model |
| `WQM_CONFIG_PATH` | Platform-specific | Configuration file path |
| `WQM_DATA_DIR` | `~/.local/share/workspace-qdrant-mcp` | Data directory |
| `WQM_DAEMON_HOST` | `127.0.0.1` | Daemon gRPC host |
| `WQM_DAEMON_PORT` | `50051` | Daemon gRPC port |
| `WQM_LOG_LEVEL` | `INFO` | Logging level |
| `WQM_STDIO_MODE` | Auto-detected | Force stdio mode |
| `WQM_CLI_MODE` | Auto-detected | Force CLI mode |

**Setting environment variables:**

```bash
# Temporary (current session only)
export QDRANT_URL="https://my-cloud.qdrant.io"
export FASTEMBED_MODEL="BAAI/bge-base-en-v1.5"

# Permanent (add to ~/.bashrc or ~/.zshrc)
echo 'export QDRANT_URL="https://my-cloud.qdrant.io"' >> ~/.bashrc
source ~/.bashrc

# For Claude Desktop, set in config.json:
{
  "mcpServers": {
    "workspace-qdrant-mcp": {
      "command": "workspace-qdrant-mcp",
      "env": {
        "QDRANT_URL": "https://my-cloud.qdrant.io",
        "QDRANT_API_KEY": "your-key"
      }
    }
  }
}
```

### Key Configuration Sections

For complete configuration documentation, see [`assets/default_configuration.yaml`](assets/default_configuration.yaml).

**Critical sections:**

**Qdrant Connection:**
```yaml
qdrant:
  url: "http://localhost:6333"
  api_key: null  # Set via QDRANT_API_KEY env var
  timeout: 30s
  prefer_grpc: true
```

**Embedding Settings:**
```yaml
embedding:
  model: "sentence-transformers/all-MiniLM-L6-v2"
  enable_sparse_vectors: true
  chunk_size: 800
  chunk_overlap: 120
  batch_size: 50
```

**Performance Tuning:**
```yaml
performance:
  max_concurrent_tasks: 4
  default_timeout: 30s
  enable_preemption: true
  chunk_size: 1000
```

**Auto-Ingestion:**
```yaml
auto_ingestion:
  enabled: true
  auto_create_watches: true
  max_files_per_batch: 5
  batch_delay: 2s
  max_file_size: 50MB
  debounce: 10s
```

## Debugging Commands

### Comprehensive Diagnostics

```bash
# Full system diagnostics
wqm observability diagnostics

# Generate diagnostic report
workspace-qdrant-test --report diagnostic_report.json

# Health check with analysis
workspace-qdrant-health --analyze
```

### Service Management

```bash
# Check daemon status
wqm service status

# View daemon logs
wqm service logs

# Restart daemon
wqm service restart
```

### Collection Management

```bash
# List all collections
wqm collections list --verbose

# Check collection health
wqm admin collections

# View workspace status
wqm admin status
```

### LSP Diagnostics

```bash
# LSP status
wqm lsp status

# Diagnose LSP issues
wqm lsp diagnose

# Restart LSP servers
wqm lsp restart
```

### Configuration Validation

```bash
# Validate configuration
wqm config validate

# Show current configuration
wqm config show

# Check environment variables
wqm config env-vars
```

### Performance Monitoring

```bash
# View real-time metrics
wqm observability monitor

# Check specific components
wqm admin status --component daemon
wqm admin status --component qdrant

# Run benchmarks
workspace-qdrant-test --benchmark
```

### Failed Processing

```bash
# View failed operations
wqm status --status failed --verbose

# Retry failed operations
wqm retry-failed

# Clear error queue
wqm admin clear-errors
```

## Log Locations

### System Logs

**macOS (launchd):**
```bash
# Daemon logs
cat ~/Library/Logs/workspace-qdrant-mcp/daemon.log
tail -f ~/Library/Logs/workspace-qdrant-mcp/daemon.log

# Service logs
log show --predicate 'subsystem == "com.workspace-qdrant-mcp"' --last 1h
```

**Linux (systemd):**
```bash
# Daemon logs
journalctl -u workspace-qdrant-daemon.service -f

# Last 100 lines
journalctl -u workspace-qdrant-daemon.service -n 100

# Logs since today
journalctl -u workspace-qdrant-daemon.service --since today
```

**Windows (Windows Service):**
```powershell
# Event Viewer
Get-EventLog -LogName Application -Source workspace-qdrant-mcp -Newest 50

# Log file
type "%APPDATA%\workspace-qdrant-mcp\logs\daemon.log"
```

### Application Logs

**Data directory logs:**
```bash
# Default location: ~/.local/share/workspace-qdrant-mcp/logs/

# View recent logs
cat ~/.local/share/workspace-qdrant-mcp/logs/app.log

# Follow logs in real-time
tail -f ~/.local/share/workspace-qdrant-mcp/logs/app.log

# Search for errors
grep ERROR ~/.local/share/workspace-qdrant-mcp/logs/app.log
```

### Claude Desktop Logs

**MCP-specific logs:**
```bash
# macOS
cat ~/Library/Logs/Claude/mcp.log
tail -f ~/Library/Logs/Claude/mcp.log

# Windows
type %APPDATA%\Claude\Logs\mcp.log

# Linux
cat ~/.config/Claude/logs/mcp.log
```

### Enable Debug Logging

**Environment variable:**
```bash
export WQM_LOG_LEVEL="DEBUG"
```

**Configuration file:**
```yaml
# config.yaml
logging:
  level: "debug"
  use_file_logging: true
  log_file: "/var/log/workspace-qdrant-mcp/debug.log"
```

**Daemon foreground mode:**
```bash
# Run daemon in foreground with debug logging
memexd --foreground --log-level debug
```

## Common Error Messages

### "Failed to connect to daemon"

**Cause:** gRPC daemon is not running or not accessible

**Solution:**
```bash
# Start daemon
wqm service start

# Verify it's running
wqm service status

# Check daemon logs for startup errors
wqm service logs
```

### "Collection not found"

**Cause:** Attempting to access non-existent collection

**Solution:**
```bash
# List available collections
wqm collections list

# Create collection
wqm collections create myproject-code

# Or initialize project collection
curl -X POST http://localhost:8000/manage \
  -H "Content-Type: application/json" \
  -d '{"action":"init_project"}'
```

### "Embedding generation failed"

**Cause:** FastEmbed model download or initialization error

**Solution:**
```bash
# Check model is downloaded
ls ~/.cache/fastembed/

# Try different model
export FASTEMBED_MODEL="sentence-transformers/all-MiniLM-L6-v2"

# Clear cache and retry
rm -rf ~/.cache/fastembed/
# Model will auto-download on next use
```

### "Circuit breaker open"

**Cause:** Qdrant unavailable, circuit breaker preventing further attempts

**Solution:**
```bash
# Check Qdrant status
curl http://localhost:6333/healthz

# Wait for circuit breaker to reset (60s default)
# Or restart service
wqm service restart

# Reduce circuit breaker sensitivity
# config.yaml
qdrant:
  circuit_breaker:
    failure_threshold: 10  # Increase from 5
```

### "Token limit exceeded"

**Cause:** Attempting to embed text exceeding model limits

**Solution:**
```yaml
# config.yaml
embedding:
  chunk_size: 600  # Reduce from 800
  chunk_overlap: 90  # Reduce from 120
```

### "Permission denied" errors

**Cause:** Insufficient permissions for data directory or log files

**Solution:**
```bash
# Check permissions
ls -la ~/.local/share/workspace-qdrant-mcp/

# Fix ownership
chown -R $USER ~/.local/share/workspace-qdrant-mcp/

# Fix permissions
chmod 755 ~/.local/share/workspace-qdrant-mcp/
chmod 644 ~/.local/share/workspace-qdrant-mcp/*.db
```

---

## Getting More Help

If issues persist after following this guide:

1. **Run comprehensive diagnostics:**
   ```bash
   workspace-qdrant-test --report diagnostic_report.json
   workspace-qdrant-health --report health_report.json
   ```

2. **Gather relevant information:**
   - Error messages from logs
   - Configuration files (redact sensitive information)
   - System information (OS, Python version, etc.)
   - Steps to reproduce the issue

3. **Check documentation:**
   - [README.md](README.md) - Installation and setup
   - [API.md](API.md) - MCP tools reference
   - [CLI.md](CLI.md) - Command-line reference
   - [Architecture](docs/ARCHITECTURE.md) - System architecture

4. **Open an issue:**
   - Visit: https://github.com/ChrisGVE/workspace-qdrant-mcp/issues
   - Include diagnostic reports
   - Provide clear reproduction steps
   - Attach relevant log excerpts

---

**Related Documentation:**
- [Installation Guide](README.md#installation)
- [Configuration Reference](assets/default_configuration.yaml)
- [CLI Commands](CLI.md)
- [MCP Tools](API.md)
- [Architecture](docs/ARCHITECTURE.md)
