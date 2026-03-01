# Windows Installation Guide

workspace-qdrant-mcp on Windows requires three components:

- **memexd.exe** — Rust daemon for file watching, embedding, and queue processing
- **wqm.exe** — Rust CLI for service management and configuration
- **MCP Server** — TypeScript server exposing tools to Claude (requires Node.js)

---

## Prerequisites

- Windows 10 (1809+) or Windows 11
- [Docker Desktop](https://www.docker.com/products/docker-desktop/) for Qdrant (or a remote Qdrant instance)
- [Node.js](https://nodejs.org/) 18+ (for the MCP server only)

---

## 1. Install Binaries

### Option A: One-liner (PowerShell)

```powershell
irm https://raw.githubusercontent.com/ChrisGVE/workspace-qdrant-mcp/main/scripts/download-install.ps1 | iex
```

This installs `wqm.exe` and `memexd.exe` to `%LOCALAPPDATA%\wqm\bin`.

**Options** (when running the script directly):

```powershell
# Install to a custom location
.\scripts\download-install.ps1 -Prefix "C:\tools\wqm"

# Install a specific version
.\scripts\download-install.ps1 -Version v0.4.0

# Install CLI only (skip daemon)
.\scripts\download-install.ps1 -CliOnly
```

### Option B: Manual download

1. Go to [GitHub Releases](https://github.com/ChrisGVE/workspace-qdrant-mcp/releases/latest)
2. Download `workspace-qdrant-mcp-windows-x64.zip` (or `windows-arm64` for ARM devices)
3. Extract to a permanent location, e.g.:

```powershell
# Create directory
New-Item -ItemType Directory -Force -Path "$env:LOCALAPPDATA\wqm\bin"

# Extract
Expand-Archive workspace-qdrant-mcp-windows-x64.zip -DestinationPath "$env:LOCALAPPDATA\wqm\bin"
```

### Add to PATH

If the installer reported that the directory is not in your PATH:

**PowerShell (current session):**
```powershell
$env:PATH += ";$env:LOCALAPPDATA\wqm\bin"
```

**PowerShell (permanent, user-level):**
```powershell
$binDir = "$env:LOCALAPPDATA\wqm\bin"
$userPath = [Environment]::GetEnvironmentVariable("PATH", "User")
if ($userPath -notlike "*$binDir*") {
    [Environment]::SetEnvironmentVariable("PATH", "$userPath;$binDir", "User")
    Write-Host "Added $binDir to user PATH. Restart your terminal to apply."
}
```

**CMD (permanent):**
```cmd
setx PATH "%PATH%;%LOCALAPPDATA%\wqm\bin"
```

### Verify

```powershell
wqm --version
memexd --version
```

---

## 2. Set Up Qdrant

### Option A: Docker Desktop

```powershell
docker run -d --name qdrant -p 6333:6333 -p 6334:6334 `
  -v qdrant_storage:/qdrant/storage qdrant/qdrant
```

### Option B: Qdrant Cloud

Create a free cluster at [cloud.qdrant.io](https://cloud.qdrant.io/) and set the connection:

```powershell
# PowerShell
$env:QDRANT_URL = "https://your-cluster.cloud.qdrant.io:6333"
$env:QDRANT_API_KEY = "your-api-key"

# Or permanently
[Environment]::SetEnvironmentVariable("QDRANT_URL", "https://your-cluster.cloud.qdrant.io:6333", "User")
[Environment]::SetEnvironmentVariable("QDRANT_API_KEY", "your-api-key", "User")
```

---

## 3. Start the Daemon

### Option A: Windows Service (recommended)

```powershell
wqm service install
wqm service start
```

The service starts automatically on login and restarts on failure.

**Management:**
```powershell
wqm service status    # Check status
wqm service stop      # Stop service
wqm service start     # Start service
```

### Option B: Task Scheduler

Create a scheduled task that starts `memexd` on login:

```powershell
$action = New-ScheduledTaskAction -Execute "$env:LOCALAPPDATA\wqm\bin\memexd.exe"
$trigger = New-ScheduledTaskTrigger -AtLogOn -User $env:USERNAME
$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -RestartCount 3 -RestartInterval (New-TimeSpan -Minutes 1)
Register-ScheduledTask -TaskName "workspace-qdrant-daemon" -Action $action -Trigger $trigger -Settings $settings -Description "workspace-qdrant-mcp daemon"
```

**Start immediately:**
```powershell
Start-ScheduledTask -TaskName "workspace-qdrant-daemon"
```

### Option C: Manual startup

Run the daemon in a background process:

```powershell
Start-Process -NoNewWindow -FilePath memexd
```

Or in a separate terminal window:
```cmd
start /B memexd
```

### Verify

```powershell
wqm admin health
```

Expected output: `healthy` with Qdrant version and collection information.

---

## 4. Configuration

### Config file location

```
%APPDATA%\workspace-qdrant\config.yaml
```

Create the directory and file if they don't exist:

```powershell
New-Item -ItemType Directory -Force -Path "$env:APPDATA\workspace-qdrant"
```

### Example configuration

```yaml
qdrant:
  url: "http://localhost:6333"

daemon:
  resource_limits:
    max_memory_mb: 512
    max_concurrent_jobs: 2

logging:
  level: info
```

### Environment variables

Set via PowerShell:

```powershell
# Session only
$env:QDRANT_URL = "http://localhost:6333"
$env:WQM_LOG_LEVEL = "DEBUG"

# Persistent (user-level)
[Environment]::SetEnvironmentVariable("QDRANT_URL", "http://localhost:6333", "User")
```

Set via CMD:

```cmd
set QDRANT_URL=http://localhost:6333
setx QDRANT_URL http://localhost:6333
```

See the [Configuration Reference](configuration.md) for all options.

---

## 5. MCP Server Setup

### Install the MCP server

```powershell
npm install -g workspace-qdrant-mcp
```

Or install from source:

```powershell
cd src\typescript\mcp-server
npm install
npm run build
```

### Claude Desktop configuration

Add to `%APPDATA%\Claude\claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "workspace-qdrant": {
      "command": "node",
      "args": ["C:\\Users\\YOU\\AppData\\Local\\wqm\\mcp-server\\dist\\index.js"],
      "env": {
        "QDRANT_URL": "http://localhost:6333"
      }
    }
  }
}
```

Replace `C:\\Users\\YOU` with your actual user directory. Use double backslashes in JSON.

### Claude Code configuration

```powershell
claude mcp add workspace-qdrant -- node "C:\Users\YOU\AppData\Local\wqm\mcp-server\dist\index.js"
```

---

## 6. Troubleshooting

### Daemon won't start

**Check logs:**
```powershell
wqm debug logs --tail 50
```

Log location: `%LOCALAPPDATA%\workspace-qdrant\logs\`

**Check port availability:**
```powershell
netstat -an | findstr "50051"
netstat -an | findstr "6333"
```

### Qdrant connection fails

**Verify Docker is running:**
```powershell
docker ps | findstr qdrant
```

**Verify Qdrant is accessible:**
```powershell
Invoke-RestMethod http://localhost:6333/healthz
```

### Windows Firewall

If Qdrant or the daemon are blocked, add firewall rules:

```powershell
# Allow Qdrant
New-NetFirewallRule -DisplayName "Qdrant" -Direction Inbound -LocalPort 6333,6334 -Protocol TCP -Action Allow

# Allow memexd gRPC
New-NetFirewallRule -DisplayName "workspace-qdrant daemon" -Direction Inbound -LocalPort 50051 -Protocol TCP -Action Allow
```

### Antivirus exclusions

Some antivirus software may flag `memexd.exe` as suspicious (it's a locally-built Rust binary). Add exclusions for:

- `%LOCALAPPDATA%\wqm\bin\memexd.exe`
- `%LOCALAPPDATA%\wqm\bin\wqm.exe`
- `%LOCALAPPDATA%\workspace-qdrant\` (state database)

### Binary not found after install

Verify the directory is in your PATH:

```powershell
$env:PATH -split ";" | Select-String "wqm"
```

If empty, re-add to PATH (see step 1 above) and restart your terminal.

---

## Future Distribution

Package manager support is planned for future releases:

- **winget** — Windows Package Manager
- **scoop** — Command-line installer for Windows
- **chocolatey** — Software management for Windows

For now, use the PowerShell installer or manual download.
