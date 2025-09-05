# Global Installation Guide

This guide covers the global installation of the Workspace Qdrant MCP system, which includes three main components:

1. **Rust Daemon (memexd)** - Background service for memory management
2. **Python MCP Server** - Main application server
3. **Configuration System** - Default configurations and management scripts

## Quick Installation

### One-Command Install

```bash
curl -fsSL https://raw.githubusercontent.com/ChrisGVE/workspace-qdrant-mcp/main/install.sh | bash
```

Or download and run locally:

```bash
git clone https://github.com/ChrisGVE/workspace-qdrant-mcp.git
cd workspace-qdrant-mcp
./install.sh
```

## Manual Installation

### Prerequisites

Ensure you have the following installed:

- **Python 3.10+**
- **uv package manager** - [Installation guide](https://docs.astral.sh/uv/getting-started/installation/)
- **Rust toolchain** - [Installation guide](https://rustup.rs/)
- **Docker** (optional, for Qdrant) - [Installation guide](https://docs.docker.com/get-docker/)

### Step-by-Step Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/ChrisGVE/workspace-qdrant-mcp.git
   cd workspace-qdrant-mcp
   ```

2. **Run the installation script:**
   ```bash
   ./install.sh
   ```

3. **Verify installation:**
   ```bash
   ./verify-installation.sh
   ```

## What Gets Installed

### Binaries and Commands

- **`/usr/local/bin/memexd`** - Rust daemon binary (requires sudo)
- **`workspace-qdrant-mcp`** - Main MCP server command
- **`wqm`** - Workspace Qdrant Manager CLI

### Directory Structure

```
~/.workspace-qdrant-mcp/
├── config/
│   ├── default.yaml          # Main configuration file
│   └── mcp.json              # Claude MCP configuration template
├── logs/
│   ├── workspace-qdrant-mcp.log  # Server logs
│   └── memexd.log            # Daemon logs
├── data/
│   ├── memexd.pid            # Daemon process ID
│   └── memexd.lock           # Daemon lock file
└── manage-daemon.sh          # Daemon management script
```

### Configuration Files

- **Default Configuration**: `~/.workspace-qdrant-mcp/config/default.yaml`
- **MCP Template**: `~/.workspace-qdrant-mcp/config/mcp.json`
- **Environment Variables**: Set in shell profiles

## Post-Installation Setup

### 1. Start Qdrant Database

If you don't have Qdrant running:

```bash
# Using Docker (recommended)
docker run -d --name qdrant -p 6333:6333 qdrant/qdrant

# Or using Docker Compose
echo 'version: "3.8"
services:
  qdrant:
    image: qdrant/qdrant:latest
    ports:
      - "6333:6333"
    volumes:
      - qdrant_storage:/qdrant/storage
volumes:
  qdrant_storage:' > docker-compose.yml
docker-compose up -d
```

### 2. Configure the System

Edit the default configuration:

```bash
# Open configuration file
nano ~/.workspace-qdrant-mcp/config/default.yaml

# Key settings to review:
# - qdrant.url: Qdrant server URL
# - workspace.github_user: Your GitHub username
# - auto_ingestion.enabled: Enable automatic file ingestion
# - logging.level: Log verbosity
```

### 3. Start the Daemon

```bash
# Start the daemon
~/.workspace-qdrant-mcp/manage-daemon.sh start

# Check status
~/.workspace-qdrant-mcp/manage-daemon.sh status

# View logs
~/.workspace-qdrant-mcp/manage-daemon.sh logs
```

### 4. Test the Installation

```bash
# Test MCP server
workspace-qdrant-mcp --help
workspace-qdrant-mcp --version

# Test CLI tools
wqm --help
wqm health

# Run comprehensive verification
./verify-installation.sh
```

## Claude Integration

### MCP Configuration

1. **Copy the MCP configuration template:**
   ```bash
   cp ~/.workspace-qdrant-mcp/config/mcp.json ~/your-claude-config/
   ```

2. **Update Claude's configuration** to include the MCP server
3. **Restart Claude** to load the new MCP server

### Usage with Claude

Once configured, Claude will have access to:
- Semantic search across your project files
- Automatic file ingestion and monitoring
- Collection management for different projects
- Advanced document processing capabilities

## Management Commands

### Daemon Management

```bash
# Daemon control
~/.workspace-qdrant-mcp/manage-daemon.sh start
~/.workspace-qdrant-mcp/manage-daemon.sh stop
~/.workspace-qdrant-mcp/manage-daemon.sh restart
~/.workspace-qdrant-mcp/manage-daemon.sh status
~/.workspace-qdrant-mcp/manage-daemon.sh health

# View logs
~/.workspace-qdrant-mcp/manage-daemon.sh logs        # Last 50 lines
~/.workspace-qdrant-mcp/manage-daemon.sh logs 100   # Last 100 lines
~/.workspace-qdrant-mcp/manage-daemon.sh logs follow # Follow logs
```

### Server Commands

```bash
# Start MCP server
workspace-qdrant-mcp
workspace-qdrant-mcp --config-file /path/to/config.yaml
workspace-qdrant-mcp --host 0.0.0.0 --port 8080

# CLI management
wqm health                    # Check system health
wqm list-collections         # List all collections
wqm search "query text"      # Search across collections
wqm clean --dry-run         # Preview cleanup operations
```

## Platform Support

### Supported Platforms

- **macOS** (Intel x86_64 and Apple Silicon ARM64)
- **Linux** (x86_64 and ARM64)

### Platform-Specific Notes

**macOS:**
- Installation requires admin privileges for `/usr/local/bin/`
- Homebrew users may need to ensure PATH includes `/usr/local/bin`
- Apple Silicon users: Rust will automatically target ARM64

**Linux:**
- Uses standard system directories
- Systemd integration available (see advanced configuration)
- Supports both glibc and musl-based distributions

## Troubleshooting

### Common Issues

1. **Command not found**
   ```bash
   # Add uv tool bin to PATH
   export PATH="$HOME/.local/bin:$PATH"
   echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
   ```

2. **Permission denied on daemon**
   ```bash
   # Fix daemon permissions
   sudo chown $USER /usr/local/bin/memexd
   sudo chmod +x /usr/local/bin/memexd
   ```

3. **Python import errors**
   ```bash
   # Reinstall Python package
   uv tool uninstall workspace-qdrant-mcp
   uv tool install workspace-qdrant-mcp
   ```

4. **Configuration errors**
   ```bash
   # Validate YAML configuration
   python3 -c "import yaml; yaml.safe_load(open('~/.workspace-qdrant-mcp/config/default.yaml'))"
   ```

5. **Qdrant connection issues**
   ```bash
   # Test Qdrant connectivity
   curl http://localhost:6333/collections
   
   # Start Qdrant if not running
   docker run -d -p 6333:6333 qdrant/qdrant
   ```

### Debug Mode

Run components in debug mode for detailed logging:

```bash
# Debug MCP server
workspace-qdrant-mcp --debug

# Debug daemon (check logs)
~/.workspace-qdrant-mcp/manage-daemon.sh logs follow

# Run verification with details
./verify-installation.sh
```

### Log Locations

- **Installation logs**: `/tmp/workspace-qdrant-mcp-install.log`
- **Server logs**: `~/.workspace-qdrant-mcp/logs/workspace-qdrant-mcp.log`
- **Daemon logs**: `~/.workspace-qdrant-mcp/logs/memexd.log`

## Advanced Configuration

### Custom Installation Paths

The installation script supports environment variables:

```bash
# Custom configuration directory
export WORKSPACE_QDRANT_CONFIG_DIR="/opt/workspace-qdrant"

# Custom daemon path
export WORKSPACE_QDRANT_DAEMON_PATH="/opt/bin/memexd"

./install.sh
```

### Systemd Service (Linux)

Create a systemd service for the daemon:

```bash
# Create service file
sudo tee /etc/systemd/system/memexd.service << 'EOF'
[Unit]
Description=Workspace Qdrant MCP Daemon
After=network.target

[Service]
Type=simple
User=your-user
ExecStart=/usr/local/bin/memexd
Restart=always
RestartSec=5
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF

# Enable and start service
sudo systemctl daemon-reload
sudo systemctl enable memexd
sudo systemctl start memexd
```

### LaunchD Service (macOS)

Create a LaunchAgent for the daemon:

```bash
# Create plist file
mkdir -p ~/Library/LaunchAgents
tee ~/Library/LaunchAgents/com.workspace-qdrant.memexd.plist << 'EOF'
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
    <string>/Users/$(whoami)/.workspace-qdrant-mcp/logs/memexd.log</string>
    <key>StandardErrorPath</key>
    <string>/Users/$(whoami)/.workspace-qdrant-mcp/logs/memexd.log</string>
</dict>
</plist>
EOF

# Load the service
launchctl load ~/Library/LaunchAgents/com.workspace-qdrant.memexd.plist
launchctl start com.workspace-qdrant.memexd
```

## Uninstallation

To completely remove the installation:

```bash
# Stop services
~/.workspace-qdrant-mcp/manage-daemon.sh stop

# Remove Python packages
uv tool uninstall workspace-qdrant-mcp

# Remove daemon binary (requires sudo)
sudo rm -f /usr/local/bin/memexd

# Remove configuration directory
rm -rf ~/.workspace-qdrant-mcp

# Remove environment variables from shell profiles
# Edit ~/.bashrc, ~/.zshrc to remove WORKSPACE_QDRANT_CONFIG
```

## Support

For issues and support:

1. **Check the troubleshooting section above**
2. **Run the verification script**: `./verify-installation.sh`
3. **Review logs** in `~/.workspace-qdrant-mcp/logs/`
4. **Open an issue** on the GitHub repository
5. **Check the documentation** for advanced configuration options

## Contributing

The installation system is designed to be:
- **Cross-platform** (macOS and Linux)
- **Idempotent** (safe to run multiple times)
- **Comprehensive** (includes verification and management tools)
- **User-friendly** (clear feedback and error messages)

Contributions to improve the installation experience are welcome!