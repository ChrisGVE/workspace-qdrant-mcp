[CmdletBinding()]
param(
  [string]$RepoDir = (Get-Location).Path,
  [string]$QdrantUrl = "http://localhost:6333",
  [string]$DaemonEndpoint = "localhost:50051"
)

$ErrorActionPreference = "Stop"
$script:Failures = 0
$script:Warnings = 0

function Write-Pass([string]$Message) { Write-Host "[PASS] $Message" -ForegroundColor Green }
function Write-Warn2([string]$Message) { $script:Warnings++; Write-Host "[WARN] $Message" -ForegroundColor Yellow }
function Write-Fail2([string]$Message) { $script:Failures++; Write-Host "[FAIL] $Message" -ForegroundColor Red }

function Test-Command2([string]$Name, [bool]$Required = $true) {
  $cmd = Get-Command $Name -ErrorAction SilentlyContinue
  if ($cmd) {
    Write-Pass "$Name encontrado: $($cmd.Source)"
    return $true
  }
  if ($Required) { Write-Fail2 "$Name não encontrado no PATH" } else { Write-Warn2 "$Name não encontrado no PATH" }
  return $false
}

function Test-TcpPort([string]$HostName, [int]$Port, [int]$TimeoutMs = 1200) {
  try {
    $client = New-Object System.Net.Sockets.TcpClient
    $iar = $client.BeginConnect($HostName, $Port, $null, $null)
    $ok = $iar.AsyncWaitHandle.WaitOne($TimeoutMs, $false)
    if (-not $ok) {
      $client.Close()
      return $false
    }
    $client.EndConnect($iar)
    $client.Close()
    return $true
  } catch {
    return $false
  }
}

Write-Host "== workspace-qdrant fork doctor =="
Write-Host "RepoDir: $RepoDir"
Write-Host "QdrantUrl: $QdrantUrl"
Write-Host "DaemonEndpoint: $DaemonEndpoint"
Write-Host ""

Test-Command2 git $true | Out-Null
Test-Command2 node $true | Out-Null
Test-Command2 npm $true | Out-Null
Test-Command2 cargo $false | Out-Null
Test-Command2 rustc $false | Out-Null
Test-Command2 docker $false | Out-Null
Test-Command2 wqm $false | Out-Null
Test-Command2 codex $false | Out-Null
Test-Command2 gh $false | Out-Null

Write-Host ""
Write-Host "== Repositório =="

if (-not (Test-Path $RepoDir)) {
  Write-Fail2 "RepoDir não existe: $RepoDir"
} elseif (-not (Test-Path (Join-Path $RepoDir ".git"))) {
  Write-Fail2 "RepoDir não tem .git: $RepoDir"
} else {
  Write-Pass "Repo Git detectado"
  Push-Location $RepoDir
  try {
    $branch = git branch --show-current
    $origin = git remote get-url origin 2>$null
    $upstream = git remote get-url upstream 2>$null
    Write-Pass "Branch atual: $branch"
    if ($origin) { Write-Pass "origin: $origin" } else { Write-Warn2 "remote origin não configurado" }
    if ($upstream) { Write-Pass "upstream: $upstream" } else { Write-Warn2 "remote upstream não configurado" }
  } finally {
    Pop-Location
  }
}

$pkg = Join-Path $RepoDir "src\typescript\mcp-server\package.json"
$dist = Join-Path $RepoDir "src\typescript\mcp-server\dist\index.js"
if (Test-Path $pkg) { Write-Pass "package.json do MCP encontrado" } else { Write-Fail2 "package.json do MCP não encontrado: $pkg" }
if (Test-Path $dist) { Write-Pass "dist/index.js existe" } else { Write-Warn2 "dist/index.js ainda não existe; rode make -f Makefile.win build-ts" }

Write-Host ""
Write-Host "== Serviços =="

try {
  $resp = Invoke-WebRequest -Uri "$QdrantUrl/collections" -UseBasicParsing -TimeoutSec 3
  if ($resp.StatusCode -ge 200 -and $resp.StatusCode -lt 300) {
    Write-Pass "Qdrant respondeu em $QdrantUrl"
  } else {
    Write-Warn2 "Qdrant respondeu com HTTP $($resp.StatusCode)"
  }
} catch {
  Write-Warn2 "Qdrant não respondeu em $QdrantUrl; rode make -f Makefile.win qdrant-up"
}

$parts = $DaemonEndpoint -replace "^https?://", ""
$hostPort = $parts.Split(":")
$daemonHost = $hostPort[0]
$daemonPort = 50051
if ($hostPort.Count -gt 1) { [int]$daemonPort = $hostPort[1] }

if (Test-TcpPort $daemonHost $daemonPort) {
  Write-Pass "Porta do daemon aberta: ${daemonHost}:${daemonPort}"
} else {
  Write-Warn2 "Porta do daemon não respondeu: ${daemonHost}:${daemonPort}; rode make -f Makefile.win start-daemon"
}

Write-Host ""
Write-Host "== Configs de clientes =="

$claudeConfig = Join-Path $env:APPDATA "Claude\claude_desktop_config.json"
$codexConfig = Join-Path $env:USERPROFILE ".codex\config.toml"
if (Test-Path $claudeConfig) { Write-Pass "Claude config existe: $claudeConfig" } else { Write-Warn2 "Claude config não existe: $claudeConfig" }
if (Test-Path $codexConfig) { Write-Pass "Codex config existe: $codexConfig" } else { Write-Warn2 "Codex config não existe: $codexConfig" }

Write-Host ""
Write-Host "Resultado: $script:Failures falha(s), $script:Warnings aviso(s)."
if ($script:Failures -gt 0) { exit 1 }
exit 0
