[CmdletBinding()]
param(
  [string]$RepoDir = (Get-Location).Path,
  [string]$ProjectDir = (Get-Location).Path,
  [string]$QdrantUrl = "http://localhost:6333",
  [string]$DaemonEndpoint = "localhost:50051"
)

$ErrorActionPreference = "Stop"

function Step([string]$Message) {
  Write-Host ""
  Write-Host "== $Message ==" -ForegroundColor Cyan
}

function Test-TcpPort([string]$HostName, [int]$Port, [int]$TimeoutMs = 1200) {
  try {
    $client = New-Object System.Net.Sockets.TcpClient
    $iar = $client.BeginConnect($HostName, $Port, $null, $null)
    $ok = $iar.AsyncWaitHandle.WaitOne($TimeoutMs, $false)
    if (-not $ok) { $client.Close(); return $false }
    $client.EndConnect($iar); $client.Close(); return $true
  } catch { return $false }
}

function Convert-ToWqmPath([string]$PathValue) {
  if (-not $PathValue) { return $PathValue }

  $absolute = [System.IO.Path]::GetFullPath($PathValue)
  $normalized = $absolute -replace '\\', '/'

  if ($normalized -match '^/mnt/[a-z]/') {
    return $normalized
  }

  if ($normalized -match '^([A-Za-z]):/(.*)$') {
    $drive = $Matches[1].ToLower()
    $rest = $Matches[2].TrimStart('/')
    return "/mnt/$drive/$rest"
  }

  return $normalized
}

Step "Qdrant"
Invoke-WebRequest -Uri "$QdrantUrl/collections" -UseBasicParsing -TimeoutSec 5 | Out-Null
Write-Host "Qdrant OK"

Step "Daemon"
$parts = ($DaemonEndpoint -replace "^https?://", "").Split(":")
$daemonHost = $parts[0]
$daemonPort = 50051
if ($parts.Count -gt 1) { [int]$daemonPort = $parts[1] }
if (-not (Test-TcpPort $daemonHost $daemonPort)) {
  throw "Daemon não respondeu em ${daemonHost}:${daemonPort}"
}
Write-Host "Daemon port OK"

Step "wqm health"
if (-not (Get-Command wqm -ErrorAction SilentlyContinue)) {
  throw "wqm não encontrado no PATH"
}
wqm status health

Step "TypeScript MCP build"
$dist = Join-Path $RepoDir "src\typescript\mcp-server\dist\index.js"
if (-not (Test-Path $dist)) {
  Push-Location (Join-Path $RepoDir "src\typescript\mcp-server")
  try {
    npm install
    npm run build
  } finally {
    Pop-Location
  }
}
Write-Host "MCP dist OK: $dist"

Step "Project registration"
if (Test-Path $ProjectDir) {
  $WqmProjectDir = Convert-ToWqmPath $ProjectDir
  wqm project register $WqmProjectDir --yes
  wqm project status $WqmProjectDir
} else {
  Write-Warning "ProjectDir não existe; pulando registro: $ProjectDir"
}

Step "Rules add/list/remove"
$label = "wqm-smoke"
$content = "Smoke test rule created by workspace-qdrant fork kit. Safe to remove."
try {
  wqm rules add --label $label --content $content --global
  wqm rules list --global
} finally {
  try { wqm rules remove --label $label --global } catch { Write-Warning "Não foi possível remover regra smoke: $_" }
}

Step "Search smoke"
try {
  wqm search global "smoke test" -n 1
} catch {
  Write-Warning "Busca global falhou ou não retornou dados; valide ingestão depois. Erro: $_"
}

Write-Host ""
Write-Host "Smoke test concluído."
