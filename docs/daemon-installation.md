# Daemon Service Installation and Startup Guide

This guide covers installing and starting the workspace-qdrant-mcp document processing daemon (`memexd`) as a system service across different platforms.

## Overview

The `memexd` daemon provides:
- Continuous document monitoring and processing
- Background embedding generation
- File watching for real-time updates
- IPC communication for Python integration
- Automatic startup and recovery
- Structured logging and health monitoring

## Prerequisites

Before installing the daemon service, ensure:

1. **System Requirements**:
   - Linux: systemd-enabled distribution
   - macOS: macOS 10.10+ (launchd support)
   - Windows: Windows 10+ with service manager
   - Minimum 2GB RAM and 1GB disk space

2. **Dependencies Installed**:
   - Rust toolchain (`cargo` command available)
   - Python 3.10+ with `uv` package manager
   - Qdrant server running (local or cloud)

3. **User Permissions**:
   - Linux: User must have `sudo` access for system service installation
   - macOS: Admin user permissions for `/Library/LaunchDaemons`
   - Windows: Administrator privileges for service installation

## Quick Installation

For immediate setup with sensible defaults:

```bash
# Install the package and daemon binary
./install.sh

# Install and start as system service
wqm service install
wqm service start

# Verify service is running
wqm service status
```

## Platform-Specific Installation

### Linux (systemd)

#### 1. Install Daemon Binary

```bash
# Build and install daemon
cd rust-engine
cargo build --release --bin memexd
sudo cp target/release/memexd /usr/local/bin/memexd
sudo chmod +x /usr/local/bin/memexd
```

#### 2. Create Service User

```bash
# Create dedicated service user for security
sudo useradd --system --home /var/lib/memexd --shell /bin/false memexd
sudo mkdir -p /var/lib/memexd
sudo chown memexd:memexd /var/lib/memexd
```

#### 3. Create Service Configuration

Create `/etc/systemd/system/memexd.service` (see [service-configs/systemd/memexd.service](service-configs/systemd/memexd.service) for the complete file):

```ini
[Unit]
Description=Memory eXchange Daemon - Document processing service
Documentation=https://github.com/ChrisGVE/workspace-qdrant-mcp
After=network-online.target
Wants=network-online.target
RequiresMountsFor=/var/lib/memexd

[Service]
Type=simple
User=memexd
Group=memexd
WorkingDirectory=/var/lib/memexd
ExecStart=/usr/local/bin/memexd --config=/etc/memexd/config.toml --pid-file=/var/run/memexd.pid
ExecReload=/bin/kill -HUP $MAINPID

# Restart configuration
Restart=always
RestartSec=10
StartLimitInterval=300
StartLimitBurst=3

# Security settings
NoNewPrivileges=true
ProtectSystem=strict
ProtectHome=true
ReadWritePaths=/var/lib/memexd /var/log/memexd
PrivateTmp=true
ProtectKernelTunables=true
ProtectKernelModules=true
ProtectControlGroups=true

# Resource limits
MemoryHigh=1G
MemoryMax=2G
TasksMax=100

# Logging
StandardOutput=journal
StandardError=journal
SyslogIdentifier=memexd

[Install]
WantedBy=multi-user.target
```

#### 4. Create Configuration Directory

```bash
# Create configuration directory
sudo mkdir -p /etc/memexd
sudo mkdir -p /var/log/memexd
sudo chown memexd:memexd /var/log/memexd

# Create default configuration
sudo tee /etc/memexd/config.toml << EOF
# See service-configs/config/memexd.toml for complete configuration options
[daemon]
port = 8765
log_level = "info"
worker_threads = 4

[qdrant]
url = "http://localhost:6333"
timeout = 30
prefer_grpc = false

[embedding]
model = "sentence-transformers/all-MiniLM-L6-v2"
chunk_size = 800
batch_size = 50

[workspace]
collections = ["project"]
max_collections = 100
EOF

sudo chown memexd:memexd /etc/memexd/config.toml
sudo chmod 640 /etc/memexd/config.toml
```

#### 5. Enable and Start Service

```bash
# Enable service for automatic startup
sudo systemctl enable memexd.service

# Start the service
sudo systemctl start memexd.service

# Check service status
sudo systemctl status memexd.service
```

### macOS (launchd)

The macOS installation uses the native launchd system for service management, providing robust daemon lifecycle management on macOS Darwin systems.

#### 1. Install Daemon Binary

```bash
# Build and install daemon on macOS
cd rust-engine
cargo build --release --bin memexd
sudo cp target/release/memexd /usr/local/bin/memexd
sudo chmod +x /usr/local/bin/memexd
```

#### 2. Create Launch Daemon Configuration

Create `/Library/LaunchDaemons/com.workspace-qdrant.memexd.plist` (see [service-configs/launchd/com.workspace-qdrant.memexd.plist](service-configs/launchd/com.workspace-qdrant.memexd.plist) for the complete file):

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
        <string>--config</string>
        <string>/usr/local/etc/memexd/config.toml</string>
        <string>--pid-file</string>
        <string>/usr/local/var/run/memexd.pid</string>
    </array>
    
    <key>WorkingDirectory</key>
    <string>/usr/local/var/lib/memexd</string>
    
    <key>StandardOutPath</key>
    <string>/usr/local/var/log/memexd/memexd.log</string>
    
    <key>StandardErrorPath</key>
    <string>/usr/local/var/log/memexd/memexd.error.log</string>
    
    <key>KeepAlive</key>
    <dict>
        <key>SuccessfulExit</key>
        <false/>
        <key>Crashed</key>
        <true/>
    </dict>
    
    <key>ThrottleInterval</key>
    <integer>10</integer>
    
    <key>RunAtLoad</key>
    <true/>
    
    <key>ProcessType</key>
    <string>Background</string>
    
    <key>UserName</key>
    <string>_memexd</string>
    
    <key>GroupName</key>
    <string>_memexd</string>
    
    <key>Nice</key>
    <integer>10</integer>
</dict>
</plist>
```

#### 3. Create Service User and Directories

```bash
# Create dedicated service user
sudo dscl . -create /Users/_memexd
sudo dscl . -create /Users/_memexd UserShell /usr/bin/false
sudo dscl . -create /Users/_memexd RealName "memexd daemon"
sudo dscl . -create /Users/_memexd UniqueID 499
sudo dscl . -create /Users/_memexd PrimaryGroupID 499
sudo dscl . -create /Users/_memexd NFSHomeDirectory /usr/local/var/lib/memexd

# Create group
sudo dscl . -create /Groups/_memexd
sudo dscl . -create /Groups/_memexd PrimaryGroupID 499

# Create directories
sudo mkdir -p /usr/local/etc/memexd
sudo mkdir -p /usr/local/var/lib/memexd
sudo mkdir -p /usr/local/var/log/memexd
sudo mkdir -p /usr/local/var/run

# Set permissions
sudo chown -R _memexd:_memexd /usr/local/var/lib/memexd
sudo chown -R _memexd:_memexd /usr/local/var/log/memexd
```

#### 4. Create Configuration File

```bash
sudo tee /usr/local/etc/memexd/config.toml << EOF
# See service-configs/config/memexd.toml for complete configuration options
[daemon]
port = 8765
log_level = "info"
worker_threads = 4

[qdrant]
url = "http://localhost:6333"
timeout = 30
prefer_grpc = false

[embedding]
model = "sentence-transformers/all-MiniLM-L6-v2"
chunk_size = 800
batch_size = 50

[workspace]
collections = ["project"]
max_collections = 100
EOF

sudo chown _memexd:_memexd /usr/local/etc/memexd/config.toml
sudo chmod 640 /usr/local/etc/memexd/config.toml
```

#### 5. Load and Start Service

```bash
# Load the service
sudo launchctl load /Library/LaunchDaemons/com.workspace-qdrant.memexd.plist

# Start the service
sudo launchctl start com.workspace-qdrant.memexd

# Check service status
sudo launchctl list | grep memexd
```

### Windows Service

#### 1. Install Daemon Binary

```powershell
# Build daemon (run in PowerShell as Administrator)
cd rust-engine
cargo build --release --bin memexd.exe

# Create program directory
New-Item -ItemType Directory -Path "C:\Program Files\MemexD" -Force

# Copy binary
Copy-Item "target\release\memexd.exe" -Destination "C:\Program Files\MemexD\memexd.exe"
```

#### 2. Install Service using sc.exe

```powershell
# Install service (as Administrator)
sc.exe create MemexD `
    binPath= "C:\Program Files\MemexD\memexd.exe --config=C:\ProgramData\MemexD\config.toml --pid-file=C:\ProgramData\MemexD\memexd.pid" `
    start= auto `
    obj= "NT SERVICE\MemexD" `
    DisplayName= "Memory eXchange Daemon" `
    description= "Document processing and embedding generation service"

# Grant logon as service right
sc.exe sidtype MemexD unrestricted
```

#### 3. Create Configuration Directory

```powershell
# Create directories
New-Item -ItemType Directory -Path "C:\ProgramData\MemexD" -Force
New-Item -ItemType Directory -Path "C:\ProgramData\MemexD\logs" -Force

# Create configuration file
@"
[daemon]
port = 8765
log_level = "info"
worker_threads = 4

[qdrant]
url = "http://localhost:6333"
timeout = 30
prefer_grpc = false

[embedding]
model = "sentence-transformers/all-MiniLM-L6-v2"
chunk_size = 800
batch_size = 50

[workspace]
collections = ["project"]
max_collections = 100
"@ | Out-File -FilePath "C:\ProgramData\MemexD\config.toml" -Encoding UTF8

# Set permissions
$acl = Get-Acl "C:\ProgramData\MemexD"
$accessRule = New-Object System.Security.AccessControl.FileSystemAccessRule("NT SERVICE\MemexD","FullControl","ContainerInherit,ObjectInherit","None","Allow")
$acl.SetAccessRule($accessRule)
Set-Acl "C:\ProgramData\MemexD" $acl
```

#### 4. Start Service

```powershell
# Start the service
Start-Service MemexD

# Check service status
Get-Service MemexD

# Set to start automatically
Set-Service MemexD -StartupType Automatic
```

## Service Management Commands

### Universal CLI (wqm)

The `wqm` CLI tool provides cross-platform service management:

```bash
# Install service for current platform
wqm service install

# Start/stop/restart service
wqm service start
wqm service stop  
wqm service restart

# Check service status
wqm service status

# View service logs
wqm service logs

# Uninstall service
wqm service uninstall
```

### Platform-Specific Commands

#### Linux (systemd)

```bash
# Service control
sudo systemctl start memexd.service
sudo systemctl stop memexd.service
sudo systemctl restart memexd.service
sudo systemctl reload memexd.service

# Status and logs
sudo systemctl status memexd.service
sudo journalctl -u memexd.service -f

# Enable/disable automatic startup
sudo systemctl enable memexd.service
sudo systemctl disable memexd.service

# Configuration reload
sudo systemctl daemon-reload
```

#### macOS (launchd)

```bash
# Service control
sudo launchctl start com.workspace-qdrant.memexd
sudo launchctl stop com.workspace-qdrant.memexd

# Load/unload service
sudo launchctl load /Library/LaunchDaemons/com.workspace-qdrant.memexd.plist
sudo launchctl unload /Library/LaunchDaemons/com.workspace-qdrant.memexd.plist

# Status and logs
sudo launchctl list | grep memexd
tail -f /usr/local/var/log/memexd/memexd.log
```

#### Windows Service

```powershell
# Service control
Start-Service MemexD
Stop-Service MemexD
Restart-Service MemexD

# Status
Get-Service MemexD

# Startup configuration
Set-Service MemexD -StartupType Automatic
Set-Service MemexD -StartupType Manual
Set-Service MemexD -StartupType Disabled

# View logs
Get-EventLog -LogName Application -Source MemexD -Newest 10
```

## Post-Installation Verification

### 1. Service Status Check

```bash
# Using wqm (cross-platform)
wqm service status

# Expected output:
# ✅ Service Status: Running
# ✅ Process ID: 12345
# ✅ Uptime: 5 minutes
# ✅ Memory Usage: 45MB
# ✅ Log Level: info
```

### 2. Health Check

```bash
# Run comprehensive health check
workspace-qdrant-health

# Check daemon connectivity
curl -f http://localhost:8765/health || echo "Daemon not responding"
```

### 3. Log Verification

```bash
# Check logs for startup confirmation
wqm service logs | grep "memexd daemon is running"

# Look for error messages
wqm service logs | grep -i error
```

### 4. IPC Communication Test

```bash
# Test Python-Rust IPC connection
python3 -c "
from workspace_qdrant_mcp.daemon_client import DaemonClient
client = DaemonClient()
print('IPC Status:', client.ping())
"
```

## Troubleshooting

### Common Issues

#### Service Won't Start

**Symptoms**: Service shows failed status or exits immediately

**Solutions**:

1. **Check configuration file**:
   ```bash
   # Validate configuration syntax
   memexd --config=/etc/memexd/config.toml --foreground --log-level=debug
   ```

2. **Check file permissions**:
   ```bash
   # Linux/macOS
   ls -la /etc/memexd/config.toml
   ls -la /usr/local/bin/memexd
   
   # Windows
   Get-Acl "C:\ProgramData\MemexD\config.toml"
   ```

3. **Check dependencies**:
   ```bash
   # Test Qdrant connectivity
   curl http://localhost:6333/collections
   
   # Check port availability
   netstat -ln | grep 8765
   ```

#### High Memory Usage

**Symptoms**: Service consuming excessive memory

**Solutions**:

1. **Adjust batch size in configuration**:
   ```toml
   [embedding]
   batch_size = 20  # Reduce from default 50
   chunk_size = 600  # Reduce from default 800
   ```

2. **Set resource limits** (Linux):
   ```ini
   # In /etc/systemd/system/memexd.service
   MemoryHigh=512M
   MemoryMax=1G
   ```

#### Permission Errors

**Symptoms**: "Permission denied" errors in logs

**Solutions**:

1. **Fix file ownership**:
   ```bash
   # Linux
   sudo chown -R memexd:memexd /var/lib/memexd
   
   # macOS  
   sudo chown -R _memexd:_memexd /usr/local/var/lib/memexd
   ```

2. **Check SELinux context** (RHEL/CentOS):
   ```bash
   sudo setsebool -P daemons_enable_cluster_mode 1
   sudo semanage fcontext -a -t bin_t "/usr/local/bin/memexd"
   sudo restorecon /usr/local/bin/memexd
   ```

#### IPC Connection Failed

**Symptoms**: Python client cannot connect to daemon

**Solutions**:

1. **Check daemon is listening**:
   ```bash
   netstat -ln | grep 8765
   lsof -i :8765
   ```

2. **Verify firewall settings**:
   ```bash
   # Linux (ufw)
   sudo ufw allow 8765/tcp
   
   # Linux (firewalld)
   sudo firewall-cmd --add-port=8765/tcp --permanent
   sudo firewall-cmd --reload
   ```

3. **Test with telnet**:
   ```bash
   telnet localhost 8765
   ```

### Log Analysis

#### Enable Debug Logging

```bash
# Temporary debug mode (Linux)
sudo systemctl edit memexd.service

# Add override:
[Service]
ExecStart=
ExecStart=/usr/local/bin/memexd --config=/etc/memexd/config.toml --log-level=debug
```

#### Key Log Messages

**Successful startup**:
```
INFO memexd daemon is running. Send SIGTERM or SIGINT to stop.
INFO ProcessingEngine started successfully
INFO IPC client available for Python integration
```

**Configuration errors**:
```
ERROR Failed to load configuration: invalid TOML syntax
ERROR Qdrant connection failed: connection refused
```

**Resource issues**:
```
WARN High memory usage detected: 1.2GB
ERROR Failed to process document: out of memory
```

## Security Considerations

### User Permissions

1. **Dedicated Service User**:
   - Never run as root/Administrator
   - Create dedicated service account with minimal permissions
   - Restrict access to configuration files (640 permissions)

2. **File System Access**:
   - Service user only needs read access to document directories
   - Write access only to log and data directories
   - Use `ReadWritePaths` in systemd for additional protection

3. **Network Security**:
   - Bind daemon to localhost only by default
   - Use firewall rules to restrict access
   - Consider TLS for production deployments

### Configuration Security

1. **Sensitive Data**:
   ```toml
   # Store API keys in environment variables
   [qdrant]
   api_key = "${QDRANT_API_KEY}"  # Resolved from environment
   ```

2. **File Permissions**:
   ```bash
   # Configuration files should be read-only for service user
   chmod 640 /etc/memexd/config.toml
   chown root:memexd /etc/memexd/config.toml
   ```

### System Hardening

1. **Systemd Security Features** (Linux):
   ```ini
   # Comprehensive security settings in service file
   NoNewPrivileges=true
   ProtectSystem=strict
   ProtectHome=true
   PrivateTmp=true
   ProtectKernelTunables=true
   ProtectKernelModules=true
   ProtectControlGroups=true
   RestrictNamespaces=true
   RestrictSUIDSGID=true
   ```

2. **Resource Limits**:
   ```ini
   # Prevent resource exhaustion
   MemoryMax=2G
   TasksMax=100
   CPUQuota=80%
   ```

## Auto-Start Configuration

### System Boot Integration

All service configurations above include automatic startup on system boot:

- **Linux**: `WantedBy=multi-user.target` in systemd service
- **macOS**: `RunAtLoad=true` in LaunchDaemon plist
- **Windows**: `start=auto` in service configuration

### Crash Recovery

Services are configured to automatically restart on failure:

- **Linux**: `Restart=always` with exponential backoff
- **macOS**: `KeepAlive` with crash detection
- **Windows**: Service recovery settings via Service Manager

### Dependencies

Services wait for network availability before starting:

- **Linux**: `After=network-online.target`
- **macOS**: Can add network dependency in plist
- **Windows**: Service dependencies can be configured via `sc.exe`

## Performance Tuning

### Resource Optimization

1. **Memory Settings**:
   ```toml
   [embedding]
   batch_size = 20      # Smaller batches for limited memory
   chunk_size = 600     # Smaller chunks reduce memory peaks
   
   [daemon]
   worker_threads = 2   # Match CPU cores available
   ```

2. **Disk I/O**:
   ```toml
   # Use SSD storage for data directory
   # Separate data and logs on different volumes for performance
   ```

3. **Network Configuration**:
   ```toml
   [qdrant]
   prefer_grpc = true   # Better performance than HTTP for large datasets
   timeout = 60         # Longer timeout for large operations
   ```

### Monitoring Integration

1. **System Metrics**:
   ```bash
   # Monitor resource usage
   systemctl status memexd.service
   journalctl -u memexd.service --since "1 hour ago" | grep -i memory
   ```

2. **Health Endpoints**:
   ```bash
   # Built-in health monitoring
   curl http://localhost:8765/health
   curl http://localhost:8765/metrics
   ```

3. **Log Analysis**:
   ```bash
   # Performance metrics in logs
   wqm service logs | grep "processing_time"
   wqm service logs | grep "documents_processed"
   ```

This comprehensive guide covers all aspects of daemon service installation, from basic setup to production deployment with security and monitoring considerations.