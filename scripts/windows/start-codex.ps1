[CmdletBinding()]
param(
  [string]$RepoDir = (Get-Location).Path,
  [Parameter(ValueFromRemainingArguments = $true)]
  [string[]]$CodexArgs = @()
)

$ErrorActionPreference = "Stop"

. (Join-Path $PSScriptRoot "semantic-mcp-common.ps1")

function Resolve-CodexToken {
  param([string]$BaseDir)

  $token = Resolve-SettingValue -RepoDir $BaseDir -Name "MCP_HTTP_TOKEN" -DefaultValue ""
  if (-not $token) {
    throw "MCP_HTTP_TOKEN is required. Put it in docker\.env, .env, or export it before running scripts/windows/start-codex.ps1."
  }

  $token = $token.Trim()
  if ($token.Length -lt 16) {
    throw "MCP_HTTP_TOKEN must be at least 16 characters (got $($token.Length)). Update docker\.env, .env, or the exported environment before launching Codex."
  }

  return $token
}

if (-not (Test-Path -LiteralPath $RepoDir)) {
  throw "RepoDir não existe: $RepoDir"
}

$RepoDir = (Resolve-Path -LiteralPath $RepoDir).Path
$tokenSource = if ([Environment]::GetEnvironmentVariable("MCP_HTTP_TOKEN")) {
  "ambiente atual"
} else {
  "docker\.env or .env"
}
$env:MCP_HTTP_TOKEN = Resolve-CodexToken -BaseDir $RepoDir

if (-not (Get-Command codex -ErrorAction SilentlyContinue)) {
  throw "Não encontrei o comando codex no PATH."
}

$codexExitCode = 0
Push-Location $RepoDir
try {
  Write-Host "Abrindo Codex em $RepoDir com MCP_HTTP_TOKEN carregado de $tokenSource."
  & codex @CodexArgs
  $codexExitCode = $LASTEXITCODE
} finally {
  Pop-Location
}

if ($codexExitCode -ne 0) {
  exit $codexExitCode
}
