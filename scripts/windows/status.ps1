[CmdletBinding()]
param(
  [string]$RepoDir = (Get-Location).Path,
  [string]$QdrantUrl = "http://localhost:6333",
  [string]$DaemonEndpoint = "localhost:50051"
)

$ErrorActionPreference = "Stop"

function Resolve-HttpEndpoint {
  param([string]$Endpoint)

  if ($Endpoint -match '^https?://') {
    return $Endpoint
  }

  return "http://$Endpoint"
}

function Resolve-DockerDatabasePath {
  param([string]$BaseDir)

  $candidates = @(
    (Join-Path $BaseDir "state\memexd\memexd.db"),
    (Join-Path $BaseDir "state\memexd\state.db"),
    (Join-Path $BaseDir "state\state.db")
  )

  foreach ($candidate in $candidates) {
    if (Test-Path $candidate) {
      return (Resolve-Path $candidate).Path
    }
  }

  return (Join-Path $BaseDir "state\memexd\memexd.db")
}

$wqm = Get-Command wqm -ErrorAction SilentlyContinue
if (-not $wqm) {
  throw "wqm não encontrado no PATH"
}

$previousWqmDaemonAddr = $env:WQM_DAEMON_ADDR
$previousWqmQdrantUrl = $env:WQM_QDRANT_URL
$previousQdrantUrl = $env:QDRANT_URL
$previousWqmDatabasePath = $env:WQM_DATABASE_PATH

try {
  $env:WQM_DAEMON_ADDR = (Resolve-HttpEndpoint -Endpoint $DaemonEndpoint)
  $env:WQM_QDRANT_URL = $QdrantUrl
  $env:QDRANT_URL = $QdrantUrl
  $env:WQM_DATABASE_PATH = (Resolve-DockerDatabasePath -BaseDir $RepoDir)

  Write-Host "== workspace-qdrant Docker status =="
  Write-Host "RepoDir: $RepoDir"
  Write-Host "DaemonEndpoint: $($env:WQM_DAEMON_ADDR)"
  Write-Host "QdrantUrl: $QdrantUrl"
  Write-Host "DatabasePath: $($env:WQM_DATABASE_PATH)"
  Write-Host ""

  $exitCode = 0

  wqm status health
  if ($LASTEXITCODE -ne 0) {
    $exitCode = $LASTEXITCODE
  }

  Write-Host ""
  wqm project list
  if ($LASTEXITCODE -ne 0 -and $exitCode -eq 0) {
    $exitCode = $LASTEXITCODE
  }

  exit $exitCode
} finally {
  if ($null -eq $previousWqmDaemonAddr) {
    Remove-Item Env:WQM_DAEMON_ADDR -ErrorAction SilentlyContinue
  } else {
    $env:WQM_DAEMON_ADDR = $previousWqmDaemonAddr
  }

  if ($null -eq $previousWqmQdrantUrl) {
    Remove-Item Env:WQM_QDRANT_URL -ErrorAction SilentlyContinue
  } else {
    $env:WQM_QDRANT_URL = $previousWqmQdrantUrl
  }

  if ($null -eq $previousQdrantUrl) {
    Remove-Item Env:QDRANT_URL -ErrorAction SilentlyContinue
  } else {
    $env:QDRANT_URL = $previousQdrantUrl
  }

  if ($null -eq $previousWqmDatabasePath) {
    Remove-Item Env:WQM_DATABASE_PATH -ErrorAction SilentlyContinue
  } else {
    $env:WQM_DATABASE_PATH = $previousWqmDatabasePath
  }
}
