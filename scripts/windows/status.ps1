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

function Invoke-PowerShellFile {
  param(
    [string]$FilePath,
    [string[]]$Arguments = @()
  )

  $payload = & powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File $FilePath @Arguments 2>&1
  [pscustomobject]@{
    ok = ($LASTEXITCODE -eq 0)
    exitCode = $LASTEXITCODE
    output = (@($payload) -join "`n").Trim()
  }
}

function Get-HookStatus {
  param([string]$RepoDir)

  $hookScript = Join-Path $RepoDir "scripts\windows\indexed-projects-hooks.ps1"
  $report = [ordered]@{
    available = (Test-Path -LiteralPath $hookScript)
    installed = $false
    healthy = $false
    scriptPath = $hookScript
  }

  if (-not $report.available) {
    $report.reason = 'hook script missing'
    return $report
  }

  try {
    $result = Invoke-PowerShellFile -FilePath $hookScript -Arguments @('-Action', 'status', '-RepoDir', $RepoDir)
    $report.lastCheck = $result
    if (-not $result.ok) {
      $report.reason = $(if ($result.output) { $result.output } elseif ($null -ne $result.exitCode) { "hook status command failed (exitCode $($result.exitCode))" } else { 'hook status command failed' })
      return $report
    }
    $output = $result.output
    if (-not $output) {
      $report.reason = 'empty hook status'
      return $report
    }

    $status = $output | ConvertFrom-Json -ErrorAction Stop
    $hookCount = 0
    if ($null -ne $status.hookCount) {
      $hookCount = [int]$status.hookCount
    }

    $report.status = $status
    if ($null -ne $status.installed) {
      $report.installed = [bool]$status.installed
    } else {
      $report.installed = [bool]$status.hooksPathMatches -and [bool]$status.runnerExists -and ($hookCount -ge 5)
    }

    if ($null -ne $status.healthy) {
      $report.healthy = [bool]$status.healthy
    } else {
      $report.healthy = $report.installed
    }

    if (-not $report.installed) {
      $report.reason = $(if ($status.reason) { $status.reason } else { 'hooks not fully installed' })
    }
  } catch {
    $report.reason = $_.Exception.Message
  }

  return $report
}

function Format-HookSummary {
  param([object]$HookStatus)

  if (-not $HookStatus.available) {
    return 'off (hook script missing)'
  }

  if ($HookStatus.healthy) {
    $hookCount = if ($HookStatus.status -and $null -ne $HookStatus.status.hookCount) { [int]$HookStatus.status.hookCount } else { 0 }
    $pathInfo = if ($HookStatus.status -and $HookStatus.status.configuredHooksPath) { " at $($HookStatus.status.configuredHooksPath)" } else { '' }
    return "ok ($hookCount hooks$pathInfo)"
  }

  if ($HookStatus.reason) {
    return "warn ($($HookStatus.reason))"
  }

  return 'warn (hook status incomplete)'
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
  $hookStatus = Get-HookStatus -RepoDir $RepoDir
  Write-Host "Hooks: $(Format-HookSummary $hookStatus)"
  Write-Host ""

  $exitCode = 0
  if (-not $hookStatus.healthy) {
    $exitCode = 1
  }

  wqm status health
  if ($LASTEXITCODE -ne 0) {
    $exitCode = if ($exitCode -eq 0) { $LASTEXITCODE } else { [Math]::Max($exitCode, $LASTEXITCODE) }
  }

  Write-Host ""
  wqm project list
  if ($LASTEXITCODE -ne 0) {
    $exitCode = if ($exitCode -eq 0) { $LASTEXITCODE } else { [Math]::Max($exitCode, $LASTEXITCODE) }
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
