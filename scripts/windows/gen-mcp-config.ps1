[CmdletBinding()]
param(
  [string]$RepoDir = (Get-Location).Path,
  [string]$ServerName = "workspace-qdrant",
  [string]$QdrantUrl = "http://localhost:6333",
  [string]$DaemonEndpoint = "http://localhost:50051",
  [ValidateSet("http", "stdio")]
  [string]$CodexTransport = "http",
  [string]$CodexHttpUrl = "http://localhost:6335/mcp",
  [string]$CodexBearerTokenEnvVar = "MCP_HTTP_TOKEN",
  [string]$ToolCsv = "search,retrieve,grep,list,store,rules,workspace_index",
  [string]$ConfigSuffix = "",
  [string]$ClaudeConfigPath = "",
  [string]$CodexConfigPath = "",
  [switch]$ApplyClaude,
  [switch]$ApplyCodex
)

$ErrorActionPreference = "Stop"

function Escape-TomlString([string]$Value) {
  # Use forward slashes in TOML strings so Windows paths do not need backslash escaping.
  $safe = $Value -replace "\\", "/"
  return ($safe -replace '"', '\"')
}

function Json-ServerObject([string]$IndexPath) {
  $indexPathForJson = $IndexPath -replace "\\", "/"
  return [pscustomobject]@{
    command = "node"
    args = @($indexPathForJson)
    env = [pscustomobject]@{
      QDRANT_URL = $QdrantUrl
      WQM_DAEMON_ENDPOINT = $DaemonEndpoint
    }
  }
}

function Write-Utf8NoBomFile([string]$Path, [string]$Content) {
  $utf8NoBom = New-Object System.Text.UTF8Encoding $false
  [System.IO.File]::WriteAllText($Path, $Content, $utf8NoBom)
}

. (Join-Path $PSScriptRoot "semantic-mcp-common.ps1")

$IndexPath = Join-Path $RepoDir "src\typescript\mcp-server\dist\index.js"
if (-not (Test-Path $IndexPath)) {
  Write-Warning "MCP dist/index.js não encontrado: $IndexPath"
  Write-Warning "Rode: make -f Makefile.win build-ts"
}

$GeneratedDir = Join-Path $RepoDir "generated"
New-Item -ItemType Directory -Force -Path $GeneratedDir | Out-Null

if (-not $ClaudeConfigPath) {
  $ClaudeConfigPath = Join-Path $env:APPDATA "Claude\claude_desktop_config.json"
}
if (-not $CodexConfigPath) {
  $CodexConfigPath = Join-Path $env:USERPROFILE ".codex\config.toml"
}

$claudeSnippet = [pscustomobject]@{
  mcpServers = [pscustomobject]@{
    $ServerName = (Json-ServerObject $IndexPath)
  }
}
$claudeOut = Join-Path $GeneratedDir "claude_desktop_config.$ServerName$ConfigSuffix.json"
Write-Utf8NoBomFile $claudeOut ($claudeSnippet | ConvertTo-Json -Depth 12)

$tools = $ToolCsv.Split(",") | ForEach-Object { $_.Trim() } | Where-Object { $_ }
$toolsToml = ($tools | ForEach-Object { '"' + (Escape-TomlString $_) + '"' }) -join ", "
$idxToml = Escape-TomlString $IndexPath
$qdrantToml = Escape-TomlString $QdrantUrl
$daemonToml = Escape-TomlString $DaemonEndpoint
$codexHttpToml = Escape-TomlString $CodexHttpUrl
$codexBearerTokenEnvVarToml = Escape-TomlString $CodexBearerTokenEnvVar

if ($CodexTransport -eq "http") {
  $codexTokenEnv = [Environment]::GetEnvironmentVariable($CodexBearerTokenEnvVar)
  $codexTokenFile = Get-EnvFileValue -RepoDir $RepoDir -Name $CodexBearerTokenEnvVar
  if ($codexTokenEnv) {
    if ($codexTokenEnv.Trim().Length -lt 16) {
      Write-Warning "$CodexBearerTokenEnvVar tem menos de 16 caracteres; o Codex HTTP vai falhar com required=true."
    }
  } elseif ($codexTokenFile) {
    if ($codexTokenFile.Trim().Length -lt 16) {
      Write-Warning "$CodexBearerTokenEnvVar em docker\.env ou .env tem menos de 16 caracteres; o Codex HTTP vai falhar com required=true."
    } else {
      Write-Warning "$CodexBearerTokenEnvVar está em docker\.env ou .env, mas não está exportado neste shell; use scripts/windows/start-codex.ps1 ou make -f Makefile.win codex-open para abrir o Codex."
    }
  } else {
    Write-Warning "$CodexBearerTokenEnvVar não foi encontrado no ambiente nem em docker\.env/.env; o Codex HTTP vai falhar com required=true."
  }
}

$codexBlock = if ($CodexTransport -eq "http") {
@"
# BEGIN workspace-qdrant-fork-kit
[mcp_servers.$ServerName]
url = "$codexHttpToml"
bearer_token_env_var = "$codexBearerTokenEnvVarToml"
startup_timeout_sec = 20
tool_timeout_sec = 120
required = true
enabled_tools = [$toolsToml]

# END workspace-qdrant-fork-kit
"@
} else {
@"
# BEGIN workspace-qdrant-fork-kit
[mcp_servers.$ServerName]
command = "node"
args = ["$idxToml"]
startup_timeout_sec = 20
tool_timeout_sec = 120
required = true
enabled_tools = [$toolsToml]

[mcp_servers.$ServerName.env]
QDRANT_URL = "$qdrantToml"
WQM_DAEMON_ENDPOINT = "$daemonToml"
# END workspace-qdrant-fork-kit
"@
}

$codexOut = Join-Path $GeneratedDir "codex_config.$ServerName$ConfigSuffix.toml"
Write-Utf8NoBomFile $codexOut $codexBlock

Write-Host "Gerado:"
Write-Host "  $claudeOut"
Write-Host "  $codexOut"

if ($ApplyClaude) {
  $dir = Split-Path -Parent $ClaudeConfigPath
  New-Item -ItemType Directory -Force -Path $dir | Out-Null

  if (Test-Path $ClaudeConfigPath) {
    $config = Get-Content -Raw $ClaudeConfigPath | ConvertFrom-Json
  } else {
    $config = [pscustomobject]@{}
  }

  if (-not $config.PSObject.Properties["mcpServers"]) {
    $config | Add-Member -NotePropertyName "mcpServers" -NotePropertyValue ([pscustomobject]@{})
  }

  if ($config.mcpServers.PSObject.Properties[$ServerName]) {
    $config.mcpServers.PSObject.Properties.Remove($ServerName)
  }

  $config.mcpServers | Add-Member -NotePropertyName $ServerName -NotePropertyValue (Json-ServerObject $IndexPath)
  Write-Utf8NoBomFile $ClaudeConfigPath ($config | ConvertTo-Json -Depth 12)
  Write-Host "Aplicado Claude config: $ClaudeConfigPath"
}

if ($ApplyCodex) {
  $dir = Split-Path -Parent $CodexConfigPath
  New-Item -ItemType Directory -Force -Path $dir | Out-Null

  $existing = ""
  if (Test-Path $CodexConfigPath) {
    $existing = Get-Content -Raw $CodexConfigPath
  }
  $pattern = "(?s)# BEGIN workspace-qdrant-fork-kit.*?# END workspace-qdrant-fork-kit\s*"
  $clean = [regex]::Replace($existing, $pattern, "")
  Write-Utf8NoBomFile $CodexConfigPath ($clean.TrimEnd() + "`r`n`r`n" + $codexBlock.TrimEnd() + "`r`n")
  Write-Host "Aplicado Codex config: $CodexConfigPath"
}
