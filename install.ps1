#Requires -Version 5.1
<#
.SYNOPSIS
    workspace-qdrant-mcp installer for Windows

.DESCRIPTION
    Builds and installs the CLI (wqm), daemon (memexd), and Python MCP server.

.PARAMETER Prefix
    Installation prefix (default: $env:LOCALAPPDATA\wqm)

.PARAMETER NoService
    Skip daemon service installation

.PARAMETER NoVerify
    Skip verification steps

.PARAMETER CliOnly
    Build only CLI (skip daemon)

.EXAMPLE
    .\install.ps1
    # Install to default location

.EXAMPLE
    .\install.ps1 -Prefix "C:\Program Files\wqm"
    # Install to custom location
#>

[CmdletBinding()]
param(
    [string]$Prefix = "$env:LOCALAPPDATA\wqm",
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

    # Check cargo
    try {
        $cargoVersion = (cargo --version) -replace 'cargo ', ''
    } catch {
        Write-Error "cargo not found. Please install Rust toolchain: https://rustup.rs"
    }

    # Check uv
    try {
        $uvVersion = (uv --version) -replace 'uv ', ''
    } catch {
        Write-Error "uv not found. Please install uv: https://github.com/astral-sh/uv"
    }

    Write-Success "Prerequisites OK (cargo: $cargoVersion, uv: $uvVersion)"
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
        if ($CliOnly) {
            Write-Info "Building CLI only (-CliOnly specified)..."
            cargo build --release -p wqm-cli
            if ($LASTEXITCODE -ne 0) { throw "CLI build failed" }
        } else {
            Write-Info "Building CLI..."
            cargo build --release -p wqm-cli
            if ($LASTEXITCODE -ne 0) { throw "CLI build failed" }

            Write-Info "Attempting to build daemon..."
            cargo build --release -p memexd 2>$null
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

# Install Python components
function Install-Python {
    Write-Info "Installing Python MCP server..."
    uv sync
    if ($LASTEXITCODE -ne 0) { throw "Python installation failed" }
    Write-Success "Python dependencies installed"
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

    # Test Python server
    try {
        uv run workspace-qdrant-mcp --help 2>$null | Out-Null
        Write-Success "MCP server ready"
    } catch {
        Write-Warning "MCP server not responding (may need Qdrant running)"
    }
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
    Write-Host "  - MCP server: uv run workspace-qdrant-mcp"
    Write-Host ""

    $pathDirs = $env:PATH -split ';'
    if ($BinDir -notin $pathDirs) {
        Write-Host "Add to your PATH:"
        Write-Host "  `$env:PATH += `";$BinDir`""
        Write-Host ""
    }

    Write-Host "Quick start:"
    Write-Host "  1. Start Qdrant: docker run -p 6333:6333 qdrant/qdrant"
    Write-Host "  2. Run MCP server: uv run workspace-qdrant-mcp"
    Write-Host "  3. Use CLI: wqm --help"
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
    Install-Python
    Test-Installation
    Write-Summary
}

Main
