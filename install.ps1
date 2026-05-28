#Requires -Version 5.1
<#
.SYNOPSIS
    workspace-qdrant-mcp installer for Windows

.DESCRIPTION
    Builds and installs the CLI (wqm), daemon (memexd), and TypeScript MCP server.
    This is the source-build installer (compiles from this checkout). For a
    pre-built binary download, use scripts\download-install.ps1 instead.

.PARAMETER Prefix
    Installation prefix (default: $env:LOCALAPPDATA\wqm)

.PARAMETER Force
    Clean rebuild from scratch (cargo clean + npm reinstall)

.PARAMETER NoService
    Skip daemon service installation hints

.PARAMETER NoVerify
    Skip verification steps

.PARAMETER CliOnly
    Build only the CLI (skip daemon and MCP server)

.EXAMPLE
    .\install.ps1
    # Install to default location

.EXAMPLE
    .\install.ps1 -Prefix "C:\Program Files\wqm"
    # Install to custom location

.EXAMPLE
    .\install.ps1 -Force
    # Clean rebuild from scratch
#>

[CmdletBinding()]
param(
    [string]$Prefix = "$env:LOCALAPPDATA\wqm",
    [switch]$Force,
    [switch]$NoService,
    [switch]$NoVerify,
    [switch]$CliOnly
)

$ErrorActionPreference = "Stop"

# Use environment variable override if set
if ($env:INSTALL_PREFIX) {
    $Prefix = $env:INSTALL_PREFIX
}

$BinDir = if ($env:BIN_DIR) { $env:BIN_DIR } else { Join-Path $Prefix "bin" }

# Helper functions
function Write-Info {
    param([string]$Message)
    Write-Host "==> " -ForegroundColor Blue -NoNewline
    Write-Host $Message
}

function Write-Success {
    param([string]$Message)
    Write-Host "==> " -ForegroundColor Green -NoNewline
    Write-Host $Message
}

function Write-Warning {
    param([string]$Message)
    Write-Host "Warning: " -ForegroundColor Yellow -NoNewline
    Write-Host $Message
}

function Write-Error {
    param([string]$Message)
    Write-Host "Error: " -ForegroundColor Red -NoNewline
    Write-Host $Message
    exit 1
}

# Check prerequisites
function Test-Prerequisites {
    Write-Info "Checking prerequisites..."

    # cargo is required to build the Rust binaries
    try {
        $cargoVersion = (cargo --version) -replace 'cargo ', ''
    } catch {
        Write-Error "cargo not found. Please install Rust toolchain: https://rustup.rs"
    }

    $prereq = "cargo: $cargoVersion"

    # npm is optional — only needed for the TypeScript MCP server
    try {
        $npmVersion = (npm --version)
        $prereq = "$prereq, npm: $npmVersion"
    } catch {
        Write-Warning "npm not found - MCP server installation will be skipped"
        Write-Warning "Install Node.js 18+ from https://nodejs.org to enable the MCP server"
    }

    Write-Success "Prerequisites: $prereq"
}

# Create directories
function New-Directories {
    Write-Info "Creating directories..."

    if (-not (Test-Path $BinDir)) {
        New-Item -ItemType Directory -Path $BinDir -Force | Out-Null
    }

    Write-Success "Created $BinDir"
}

# Build Rust binaries
function Build-Rust {
    Write-Info "Building Rust binaries from unified workspace..."

    Push-Location "src\rust"

    try {
        if ($Force) {
            Write-Info "Force rebuild: cleaning previous build artifacts..."
            cargo clean
        }

        if ($CliOnly) {
            Write-Info "Building CLI only (-CliOnly specified)..."
            cargo build --release -p wqm-cli
            if ($LASTEXITCODE -ne 0) { throw "CLI build failed" }
        } else {
            Write-Info "Building CLI..."
            cargo build --release -p wqm-cli
            if ($LASTEXITCODE -ne 0) { throw "CLI build failed" }

            Write-Info "Building daemon..."
            cargo build --release -p memexd
            if ($LASTEXITCODE -eq 0) {
                Write-Success "Daemon built successfully"
            } else {
                Write-Warning "Daemon build failed (may require ONNX Runtime setup)"
                Write-Warning "CLI will still be installed. See docs/TROUBLESHOOTING.md"
                $script:CliOnly = $true
            }
        }
    } finally {
        Pop-Location
    }

    Write-Success "Rust build complete"
}

# Install binaries
function Install-Binaries {
    Write-Info "Installing binaries to $BinDir..."

    # CLI
    $wqmSrc = "src\rust\target\release\wqm.exe"
    if (Test-Path $wqmSrc) {
        Copy-Item $wqmSrc -Destination $BinDir -Force
        Write-Success "Installed wqm.exe"
    } else {
        Write-Error "wqm.exe not found at $wqmSrc"
    }

    # Daemon (if built)
    $memexdSrc = "src\rust\target\release\memexd.exe"
    if (-not $CliOnly -and (Test-Path $memexdSrc)) {
        Copy-Item $memexdSrc -Destination $BinDir -Force
        Write-Success "Installed memexd.exe"
    }
}

# Install TypeScript MCP server
function Install-TypeScriptMcp {
    if ($CliOnly) { return }

    Write-Info "Installing TypeScript MCP server..."

    # npm is required for this step
    try {
        npm --version | Out-Null
    } catch {
        Write-Warning "npm not found - skipping TypeScript MCP server installation"
        Write-Warning "Install Node.js 18+ to enable the MCP server"
        return
    }

    Push-Location "src\typescript\mcp-server"

    try {
        if ($Force -or -not (Test-Path "node_modules")) {
            Write-Info "Installing npm dependencies..."
            npm install
            if ($LASTEXITCODE -ne 0) { throw "npm install failed" }
        } else {
            Write-Info "Node modules exist, skipping npm install (use -Force to reinstall)"
        }

        Write-Info "Building TypeScript MCP server..."
        npm run build
        if ($LASTEXITCODE -ne 0) { throw "npm run build failed" }
    } finally {
        Pop-Location
    }

    Write-Success "TypeScript MCP server built"
}

# Verify installation
function Test-Installation {
    if ($NoVerify) { return }

    Write-Info "Verifying installation..."

    # Check if BinDir is in PATH
    $pathDirs = $env:PATH -split ';'
    if ($BinDir -notin $pathDirs) {
        Write-Warning "$BinDir is not in your PATH"
        Write-Host "  Add to your PATH with:"
        Write-Host "    `$env:PATH += `";$BinDir`""
        Write-Host "  Or add permanently via System Properties > Environment Variables"
    }

    # Test wqm
    $wqmPath = Join-Path $BinDir "wqm.exe"
    if (Test-Path $wqmPath) {
        try {
            $wqmVersion = & $wqmPath --version 2>$null
            Write-Success "wqm version: $wqmVersion"
        } catch {
            Write-Warning "wqm.exe found but could not get version"
        }
    } else {
        Write-Warning "wqm.exe not found at $wqmPath"
    }

    # Test memexd
    $memexdPath = Join-Path $BinDir "memexd.exe"
    if (Test-Path $memexdPath) {
        try {
            $memexdVersion = & $memexdPath --version 2>$null
            Write-Success "memexd version: $memexdVersion"
        } catch {
            Write-Warning "memexd.exe found but could not get version"
        }
    }

    # Test TypeScript MCP server
    if (Test-Path "src\typescript\mcp-server\dist\index.js") {
        Write-Success "MCP server built at src\typescript\mcp-server\dist\"
    } elseif (-not $CliOnly) {
        Write-Warning "MCP server not built - run 'npm run build' in src\typescript\mcp-server\"
    }
}

# Setup daemon service
function Set-DaemonService {
    if ($NoService -or $CliOnly) { return }

    if (-not (Test-Path (Join-Path $BinDir "memexd.exe"))) { return }

    Write-Info "Setting up daemon service..."
    Write-Host ""
    Write-Host "To install and start the daemon service, run:"
    Write-Host "  $BinDir\wqm.exe service install"
    Write-Host "  $BinDir\wqm.exe service start"
    Write-Host ""
}

# Print summary
function Write-Summary {
    Write-Host ""
    Write-Host "======================================"
    Write-Success "Installation complete!"
    Write-Host "======================================"
    Write-Host ""
    Write-Host "Installed components:"
    Write-Host "  - wqm (CLI): $BinDir\wqm.exe"
    if (Test-Path (Join-Path $BinDir "memexd.exe")) {
        Write-Host "  - memexd (daemon): $BinDir\memexd.exe"
    }
    if (-not $CliOnly) {
        Write-Host "  - MCP server: node src\typescript\mcp-server\dist\index.js"
    }
    Write-Host ""

    $pathDirs = $env:PATH -split ';'
    if ($BinDir -notin $pathDirs) {
        Write-Host "Add to your PATH:"
        Write-Host "  `$env:PATH += `";$BinDir`""
        Write-Host ""
    }

    Write-Host "Quick start:"
    Write-Host "  1. Start Qdrant: docker run -p 6333:6333 qdrant/qdrant"
    Write-Host "  2. Start daemon: $BinDir\memexd.exe"
    if (-not $CliOnly) {
        Write-Host "  3. Run MCP server: node src\typescript\mcp-server\dist\index.js"
    }
    Write-Host "  4. Use CLI: wqm --help"
    Write-Host ""
}

# Main installation flow
function Main {
    Write-Host ""
    Write-Host "workspace-qdrant-mcp installer"
    Write-Host "=============================="
    Write-Host ""
    Write-Host "Installation prefix: $Prefix"
    Write-Host "Binary directory: $BinDir"
    Write-Host ""

    Test-Prerequisites
    New-Directories
    Build-Rust
    Install-Binaries
    Install-TypeScriptMcp
    Test-Installation
    Set-DaemonService
    Write-Summary
}

Main
