[CmdletBinding()]
param(
  [string]$RepoDir = (Get-Location).Path,
  [string]$ProjectDir = (Get-Location).Path,
  [string]$QdrantUrl = "http://localhost:6333",
  [string]$DaemonEndpoint = "localhost:50051",
  [string]$LogDir = ".wqm-fork/logs",
  [switch]$FastEmbed,
  [string]$FastEmbedCacheDir = ".fastembed_cache",
  [string]$FastEmbedQdrantUrl = "http://127.0.0.1:6334",
  [int]$FastEmbedGrpcPort = 55151,
  [int]$FastEmbedControlPort = 7798,
  [int]$FastEmbedMetricsPort = 9091,
  [string]$FastEmbedLogLevel = "info"
)

$ErrorActionPreference = "Continue"

function Invoke-Step {
  param([string]$Name, [scriptblock]$Body)
  Write-Host "==> $Name"
  try {
    & $Body
    if ($LASTEXITCODE -and $LASTEXITCODE -ne 0) { Write-Host "Aviso: $Name retornou codigo $LASTEXITCODE" -ForegroundColor Yellow }
  } catch {
    Write-Host "Aviso: $Name falhou: $($_.Exception.Message)" -ForegroundColor Yellow
  }
}

function Resolve-LocalPath {
  param(
    [string]$BaseDir,
    [string]$Value
  )

  if ([System.IO.Path]::IsPathRooted($Value)) {
    return $Value
  }

  return (Join-Path $BaseDir $Value)
}

function Resolve-HttpEndpoint {
  param([string]$Endpoint)

  if ($Endpoint -match '^https?://') {
    return $Endpoint
  }

  return "http://$Endpoint"
}

$ResolvedFastEmbedCacheDir = Resolve-LocalPath -BaseDir $RepoDir -Value $FastEmbedCacheDir
$ResolvedDaemonEndpoint = if ($FastEmbed) {
  "localhost:$FastEmbedGrpcPort"
} else {
  $DaemonEndpoint
}
$ResolvedDaemonAddr = Resolve-HttpEndpoint -Endpoint $ResolvedDaemonEndpoint

$PreviousWqmDaemonAddr = $env:WQM_DAEMON_ADDR
$PreviousWqmQdrantUrl = $env:WQM_QDRANT_URL

Push-Location $RepoDir
try {
  $env:WQM_DAEMON_ADDR = $ResolvedDaemonAddr
  $env:WQM_QDRANT_URL = $QdrantUrl

  Invoke-Step "snapshot inicial" { .\scripts\windows\service-observe.ps1 -RepoDir $RepoDir -ProjectDir $ProjectDir -QdrantUrl $QdrantUrl -DaemonEndpoint $ResolvedDaemonEndpoint -LogDir $LogDir -Once }
  Invoke-Step "subir Qdrant se necessario" {
    if ($FastEmbed) {
      .\scripts\windows\start-qdrant.ps1 -QdrantUrl $QdrantUrl -EnsureGrpc
    } else {
      .\scripts\windows\start-qdrant.ps1 -QdrantUrl $QdrantUrl
    }
  }
  Invoke-Step "iniciar daemon se necessario" {
    if ($FastEmbed) {
      .\scripts\windows\start-daemon-fastembed.ps1 `
        -RepoDir $RepoDir `
        -CacheDir $ResolvedFastEmbedCacheDir `
        -QdrantUrl $FastEmbedQdrantUrl `
        -GrpcPort $FastEmbedGrpcPort `
        -ControlPort $FastEmbedControlPort `
        -MetricsPort $FastEmbedMetricsPort `
        -LogLevel $FastEmbedLogLevel
    } else {
      .\scripts\windows\start-daemon.ps1 -RepoDir $RepoDir
    }
  }
  Start-Sleep -Seconds 3
  Invoke-Step "snapshot final" { .\scripts\windows\service-observe.ps1 -RepoDir $RepoDir -ProjectDir $ProjectDir -QdrantUrl $QdrantUrl -DaemonEndpoint $ResolvedDaemonEndpoint -LogDir $LogDir -Once }
  Write-Host "service-stabilize concluido. Veja logs em $LogDir"
} finally {
  if ($null -eq $PreviousWqmDaemonAddr) {
    Remove-Item Env:WQM_DAEMON_ADDR -ErrorAction SilentlyContinue
  } else {
    $env:WQM_DAEMON_ADDR = $PreviousWqmDaemonAddr
  }
  if ($null -eq $PreviousWqmQdrantUrl) {
    Remove-Item Env:WQM_QDRANT_URL -ErrorAction SilentlyContinue
  } else {
    $env:WQM_QDRANT_URL = $PreviousWqmQdrantUrl
  }
  Pop-Location
}
