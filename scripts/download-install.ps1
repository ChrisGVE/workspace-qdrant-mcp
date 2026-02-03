#Requires -Version 5.1
<#
.SYNOPSIS
    workspace-qdrant-mcp binary installer for Windows

.DESCRIPTION
    Downloads and installs pre-built binaries from GitHub releases.
    For source builds, use .\install.ps1 instead.

.PARAMETER Prefix
    Installation prefix (default: $env:LOCALAPPDATA\wqm)

.PARAMETER Version
    Specific version to install (default: latest)

.PARAMETER CliOnly
    Install only CLI (skip daemon)

.EXAMPLE
    .\scripts\download-install.ps1
    # Install latest version to default location

.EXAMPLE
    .\scripts\download-install.ps1 -Version v0.4.0
    # Install specific version

.EXAMPLE
    irm https://raw.githubusercontent.com/ChrisGVE/workspace-qdrant-mcp/main/scripts/download-install.ps1 | iex
    # One-liner installation
#>

[CmdletBinding()]
param(
    [string]$Prefix = "$env:LOCALAPPDATA\wqm",
    [string]$Version = "latest",
    [switch]$CliOnly
)

$ErrorActionPreference = "Stop"

# Configuration
$Repo = "ChrisGVE/workspace-qdrant-mcp"
$GitHubApi = "https://api.github.com/repos/$Repo"
$GitHubReleases = "https://github.com/$Repo/releases"

$BinDir = Join-Path $Prefix "bin"

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

function Write-Warn {
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

# Detect platform
function Get-Platform {
    $arch = if ([Environment]::Is64BitOperatingSystem) {
        if ([System.Runtime.InteropServices.RuntimeInformation]::OSArchitecture -eq "Arm64") {
            "arm64"
        } else {
            "x64"
        }
    } else {
        Write-Error "32-bit systems are not supported"
    }

    # Map to Rust target triple
    $target = switch ($arch) {
        "x64"   { "x86_64-pc-windows-msvc" }
        "arm64" { "aarch64-pc-windows-msvc" }
    }

    return @{
        Arch = $arch
        Target = $target
    }
}

# Get latest version from GitHub API
function Get-LatestVersion {
    try {
        $response = Invoke-RestMethod -Uri "$GitHubApi/releases/latest" -UseBasicParsing
        return $response.tag_name
    } catch {
        Write-Error "Could not determine latest version. Check your internet connection or specify -Version"
    }
}

# Verify checksum
function Test-Checksum {
    param(
        [string]$FilePath,
        [string]$Expected
    )

    $actual = (Get-FileHash -Path $FilePath -Algorithm SHA256).Hash.ToLower()

    if ($actual -ne $Expected.ToLower()) {
        Write-Error "Checksum mismatch for $(Split-Path $FilePath -Leaf)`n  Expected: $Expected`n  Actual:   $actual"
    }

    Write-Success "Checksum verified for $(Split-Path $FilePath -Leaf)"
}

# Download binary
function Get-Binary {
    param(
        [string]$Name,
        [string]$Target,
        [string]$Version,
        [string]$DestDir
    )

    $exeName = "$Name.exe"
    $url = "$GitHubReleases/download/$Version/$Name-$Target.exe"
    $checksumUrl = "$url.sha256"
    $destPath = Join-Path $DestDir $exeName

    Write-Info "Downloading $Name for $Target..."

    try {
        Invoke-WebRequest -Uri $url -OutFile $destPath -UseBasicParsing
    } catch {
        Write-Error "Failed to download $Name from $url"
    }

    # Download and verify checksum
    try {
        $expectedChecksum = (Invoke-WebRequest -Uri $checksumUrl -UseBasicParsing).Content.Trim()
        Test-Checksum -FilePath $destPath -Expected $expectedChecksum
    } catch {
        Write-Warn "Checksum file not available, skipping verification"
    }
}

# Main installation
function Main {
    Write-Host ""
    Write-Host "workspace-qdrant-mcp binary installer"
    Write-Host "======================================"
    Write-Host ""

    # Detect platform
    $platform = Get-Platform
    Write-Info "Detected platform: Windows $($platform.Arch) ($($platform.Target))"

    # Resolve version
    if ($Version -eq "latest") {
        Write-Info "Fetching latest version..."
        $Version = Get-LatestVersion
    }
    Write-Info "Installing version: $Version"

    # Create directories
    if (-not (Test-Path $BinDir)) {
        New-Item -ItemType Directory -Path $BinDir -Force | Out-Null
    }

    # Create temp directory
    $tempDir = Join-Path $env:TEMP "wqm-install-$(Get-Random)"
    New-Item -ItemType Directory -Path $tempDir -Force | Out-Null

    try {
        # Download CLI
        Get-Binary -Name "wqm" -Target $platform.Target -Version $Version -DestDir $tempDir

        # Download daemon (unless -CliOnly)
        if (-not $CliOnly) {
            Get-Binary -Name "memexd" -Target $platform.Target -Version $Version -DestDir $tempDir
        }

        # Install binaries
        Write-Info "Installing to $BinDir..."
        Move-Item (Join-Path $tempDir "wqm.exe") -Destination $BinDir -Force
        Write-Success "Installed wqm.exe"

        if (-not $CliOnly) {
            $memexdPath = Join-Path $tempDir "memexd.exe"
            if (Test-Path $memexdPath) {
                Move-Item $memexdPath -Destination $BinDir -Force
                Write-Success "Installed memexd.exe"
            }
        }

        # Verify installation
        $wqmPath = Join-Path $BinDir "wqm.exe"
        if (Test-Path $wqmPath) {
            try {
                $wqmVersion = & $wqmPath --version 2>$null
                Write-Success "wqm version: $wqmVersion"
            } catch {
                Write-Warn "wqm.exe installed but could not get version"
            }
        }

        $memexdPath = Join-Path $BinDir "memexd.exe"
        if (Test-Path $memexdPath) {
            try {
                $memexdVersion = & $memexdPath --version 2>$null
                Write-Success "memexd version: $memexdVersion"
            } catch {
                Write-Warn "memexd.exe installed but could not get version"
            }
        }

        # PATH check
        $pathDirs = $env:PATH -split ';'
        if ($BinDir -notin $pathDirs) {
            Write-Warn "$BinDir is not in your PATH"
            Write-Host ""
            Write-Host "Add to your PATH with:"
            Write-Host "  `$env:PATH += `";$BinDir`""
            Write-Host ""
            Write-Host "Or add permanently via System Properties > Environment Variables"
            Write-Host ""
        }

        Write-Host ""
        Write-Host "======================================"
        Write-Success "Installation complete!"
        Write-Host "======================================"
        Write-Host ""
        Write-Host "Quick start:"
        Write-Host "  1. Start Qdrant: docker run -p 6333:6333 qdrant/qdrant"
        Write-Host "  2. Check health: wqm admin health"
        Write-Host "  3. View help: wqm --help"
        Write-Host ""

        if (-not $CliOnly -and (Test-Path (Join-Path $BinDir "memexd.exe"))) {
            Write-Host "To start the daemon service:"
            Write-Host "  wqm service install"
            Write-Host "  wqm service start"
            Write-Host ""
        }
    } finally {
        # Cleanup temp directory
        if (Test-Path $tempDir) {
            Remove-Item -Path $tempDir -Recurse -Force -ErrorAction SilentlyContinue
        }
    }
}

Main
