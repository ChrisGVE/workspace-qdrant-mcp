# Windows Service Installation Script for memexd
# Run as Administrator in PowerShell

param(
    [string]$ServiceName = "MemexD",
    [string]$DisplayName = "Memory eXchange Daemon",
    [string]$Description = "Document processing and embedding generation service for workspace-qdrant-mcp",
    [string]$BinaryPath = "C:\Program Files\MemexD\memexd.exe",
    [string]$ConfigPath = "C:\ProgramData\MemexD\config.toml",
    [string]$StartupType = "Automatic"
)

Write-Host "Installing MemexD Windows Service..." -ForegroundColor Green

# Check if running as Administrator
if (-NOT ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole] "Administrator")) {
    Write-Error "This script must be run as Administrator!"
    exit 1
}

try {
    # Create program directory
    $ProgramDir = Split-Path $BinaryPath -Parent
    if (-not (Test-Path $ProgramDir)) {
        Write-Host "Creating program directory: $ProgramDir"
        New-Item -ItemType Directory -Path $ProgramDir -Force
    }

    # Create data directory
    $DataDir = Split-Path $ConfigPath -Parent
    if (-not (Test-Path $DataDir)) {
        Write-Host "Creating data directory: $DataDir"
        New-Item -ItemType Directory -Path $DataDir -Force
        New-Item -ItemType Directory -Path "$DataDir\logs" -Force
    }

    # Check if binary exists
    if (-not (Test-Path $BinaryPath)) {
        Write-Error "Binary not found at $BinaryPath. Please build and copy memexd.exe first."
        Write-Host "Build command: cd rust-engine && cargo build --release --bin memexd"
        Write-Host "Copy command: Copy-Item 'target\release\memexd.exe' -Destination '$BinaryPath'"
        exit 1
    }

    # Create default configuration if it doesn't exist
    if (-not (Test-Path $ConfigPath)) {
        Write-Host "Creating default configuration at $ConfigPath"
        
        $ConfigContent = @"
[daemon]
port = 8765
log_level = "info"
worker_threads = 4
enable_file_watching = true
batch_size = 50

[qdrant]
url = "http://localhost:6333"
api_key = ""
timeout = 30
prefer_grpc = false

[embedding]
model = "sentence-transformers/all-MiniLM-L6-v2"
enable_sparse_vectors = true
chunk_size = 800
chunk_overlap = 120
batch_size = 50

[workspace]
collections = ["project"]
global_collections = []
max_collections = 100

[logging]
file_logging = true
log_file_path = "$DataDir\logs\memexd.log"
json_format = true
max_file_size_mb = 100
max_backup_files = 5

[security]
bind_address = "127.0.0.1"
max_request_size_mb = 10
rate_limit_requests_per_minute = 1000

[performance]
max_memory_mb = 2048
document_timeout_seconds = 300
health_check_interval_seconds = 60
"@
        
        $ConfigContent | Out-File -FilePath $ConfigPath -Encoding UTF8
    }

    # Stop existing service if running
    $ExistingService = Get-Service -Name $ServiceName -ErrorAction SilentlyContinue
    if ($ExistingService) {
        Write-Host "Stopping existing service..."
        Stop-Service -Name $ServiceName -Force
        Write-Host "Removing existing service..."
        sc.exe delete $ServiceName
        Start-Sleep -Seconds 2
    }

    # Install the service
    Write-Host "Installing service: $ServiceName"
    $BinPathArg = "`"$BinaryPath`" --config=`"$ConfigPath`" --pid-file=`"$DataDir\memexd.pid`""
    
    $Result = sc.exe create $ServiceName `
        binPath= $BinPathArg `
        start= auto `
        obj= "NT SERVICE\$ServiceName" `
        DisplayName= $DisplayName `
        description= $Description

    if ($LASTEXITCODE -ne 0) {
        Write-Error "Failed to install service. Exit code: $LASTEXITCODE"
        Write-Error "Output: $Result"
        exit 1
    }

    # Configure service recovery options
    Write-Host "Configuring service recovery options..."
    sc.exe failure $ServiceName reset= 3600 actions= restart/10000/restart/30000/restart/60000

    # Grant logon as service right
    Write-Host "Granting logon as service right..."
    sc.exe sidtype $ServiceName unrestricted

    # Set service description
    sc.exe description $ServiceName $Description

    # Set permissions for service account
    Write-Host "Setting permissions for service account..."
    $ServiceSid = (sc.exe showsid $ServiceName | Select-String "SERVICE SID:").ToString().Split(":")[1].Trim()
    
    # Grant permissions to data directory
    $Acl = Get-Acl $DataDir
    $AccessRule = New-Object System.Security.AccessControl.FileSystemAccessRule("NT SERVICE\$ServiceName", "FullControl", "ContainerInherit,ObjectInherit", "None", "Allow")
    $Acl.SetAccessRule($AccessRule)
    Set-Acl $DataDir $Acl

    # Start the service
    Write-Host "Starting service..."
    Start-Service -Name $ServiceName

    # Verify service is running
    Start-Sleep -Seconds 5
    $Service = Get-Service -Name $ServiceName
    if ($Service.Status -eq "Running") {
        Write-Host "✅ Service installed and started successfully!" -ForegroundColor Green
        Write-Host ""
        Write-Host "Service Details:" -ForegroundColor Cyan
        Write-Host "  Name: $ServiceName"
        Write-Host "  Display Name: $DisplayName"
        Write-Host "  Status: $($Service.Status)"
        Write-Host "  Startup Type: $($Service.StartType)"
        Write-Host "  Binary: $BinaryPath"
        Write-Host "  Config: $ConfigPath"
        Write-Host "  Logs: $DataDir\logs\memexd.log"
        Write-Host ""
        Write-Host "Management Commands:" -ForegroundColor Yellow
        Write-Host "  Start:   Start-Service $ServiceName"
        Write-Host "  Stop:    Stop-Service $ServiceName"
        Write-Host "  Restart: Restart-Service $ServiceName"
        Write-Host "  Status:  Get-Service $ServiceName"
        Write-Host "  Logs:    Get-Content '$DataDir\logs\memexd.log' -Wait"
    } else {
        Write-Warning "Service installed but not running. Check logs for errors."
        Write-Host "Check logs: Get-Content '$DataDir\logs\memexd.log'"
    }

} catch {
    Write-Error "Failed to install service: $_"
    exit 1
}