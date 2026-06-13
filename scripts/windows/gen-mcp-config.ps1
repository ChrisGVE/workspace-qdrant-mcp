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
  [ValidateSet("stdio", "http")]
  [string]$ClaudeTransport = "stdio",
  [string]$ClaudeHttpUrl = "http://localhost:6335/mcp",
  [string]$ClaudeBearerTokenEnvVar = "MCP_HTTP_TOKEN",
  [string]$ClaudeMcpRemotePath = "",
  # Default = "(default)" sentinel; resolved from mcp-public-config.json below.
  # Pass an explicit CSV (e.g. "search,grep") to override the canonical list.
  [string]$ToolCsv = "(default)",
  [string]$ConfigSuffix = "",
  [string]$ClaudeConfigPath = "",
  [string]$CodexConfigPath = "",
  [switch]$ApplyClaude,
  [switch]$ApplyCodex
)

$ErrorActionPreference = "Stop"

# Single source of truth for tool list + Codex/Claude defaults.
# DO NOT hardcode publicTools or timeouts in this script — read from JSON.
$publicConfigPath = Join-Path $RepoDir "src/typescript/mcp-server/src/constants/mcp-public-config.json"
if (-not (Test-Path -LiteralPath $publicConfigPath)) {
  throw "mcp-public-config.json not found at: $publicConfigPath"
}
$mcpPublicConfig = Get-Content -LiteralPath $publicConfigPath -Raw | ConvertFrom-Json

if ($ToolCsv -eq "(default)") {
  $ToolCsv = ($mcpPublicConfig.publicTools -join ",")
}
$codexStartupTimeoutSec = [int]$mcpPublicConfig.codex.startup_timeout_sec
$codexToolTimeoutSec = [int]$mcpPublicConfig.codex.tool_timeout_sec

function Escape-TomlString([string]$Value) {
  # Use forward slashes in TOML strings so Windows paths do not need backslash escaping.
  $safe = $Value -replace "\\", "/"
  return ($safe -replace '"', '\"')
}

function Escape-TomlLiteralKey([string]$Value) {
  if ($Value -match "'") {
    throw "TOML literal project keys do not support single quotes: $Value"
  }
  return $Value
}

function Escape-TomlBasicKey([string]$Value) {
  return $Value.Replace('\', '\\').Replace('"', '\"')
}

function Add-UniquePath([System.Collections.Generic.List[string]]$Paths, [string]$PathValue) {
  if (-not $PathValue) { return }
  foreach ($existing in $Paths) {
    if ($existing -eq $PathValue) { return }
  }
  $Paths.Add($PathValue) | Out-Null
}

function Convert-WslUncToPosix([string]$PathValue) {
  if (-not $PathValue) { return $null }
  $plain = $PathValue
  if ($plain.StartsWith("\\?\UNC\", [System.StringComparison]::OrdinalIgnoreCase)) {
    $plain = "\\" + $plain.Substring(8)
  }
  if ($plain -notmatch '^\\\\(?:wsl\.localhost|wsl\$)\\([^\\]+)\\(.+)$') {
    return $null
  }

  $relative = $Matches[2] -replace "\\", "/"
  return "/" + $relative.TrimStart("/")
}

function Get-CodexProjectTrustPaths([string]$PathValue) {
  $paths = [System.Collections.Generic.List[string]]::new()
  $resolved = Resolve-Path -LiteralPath $PathValue -ErrorAction SilentlyContinue
  $rawPath = if ($resolved) {
    if ($resolved.ProviderPath) { $resolved.ProviderPath } else { $resolved.Path }
  } else {
    $PathValue
  }
  if ($rawPath -match '^Microsoft\.PowerShell\.Core\\FileSystem::(.+)$') {
    $rawPath = $Matches[1]
  }

  $abs = if ($rawPath.StartsWith("\\", [System.StringComparison]::Ordinal)) {
    $rawPath
  } else {
    [System.IO.Path]::GetFullPath($rawPath)
  }

  Add-UniquePath $paths $abs

  if ($abs.StartsWith("\\?\UNC\", [System.StringComparison]::OrdinalIgnoreCase)) {
    Add-UniquePath $paths ("\\" + $abs.Substring(8))
  } elseif ($abs.StartsWith("\\", [System.StringComparison]::Ordinal)) {
    Add-UniquePath $paths ("\\?\UNC\" + $abs.Substring(2))
  }

  $posix = Convert-WslUncToPosix $abs
  if ($posix) {
    Add-UniquePath $paths $posix
  }

  return @($paths)
}

function New-CodexProjectTrustBlock([string[]]$ProjectPaths) {
  if (-not $ProjectPaths -or $ProjectPaths.Count -eq 0) { return "" }

  $items = $ProjectPaths | ForEach-Object {
@"
[projects.'$(Escape-TomlLiteralKey $_)']
trust_level = "trusted"
"@
  }

  return (($items -join "`r`n`r`n") + "`r`n`r`n")
}

function Remove-CodexProjectTrustTables([string]$Config, [string[]]$ProjectPaths) {
  $clean = $Config
  foreach ($path in $ProjectPaths) {
    $literalPath = [regex]::Escape((Escape-TomlLiteralKey $path))
    $literalPattern = "(?ms)^\[projects\.'$literalPath'\]\s*\r?\ntrust_level\s*=\s*""trusted""\s*(?=^\[|\z)"
    $clean = [regex]::Replace($clean, $literalPattern, "")

    $basicPath = [regex]::Escape((Escape-TomlBasicKey $path))
    $basicPattern = "(?ms)^\[projects\.""$basicPath""\]\s*\r?\ntrust_level\s*=\s*""trusted""\s*(?=^\[|\z)"
    $clean = [regex]::Replace($clean, $basicPattern, "")
  }
  return $clean
}

function Resolve-McpRemoteProxyPath {
  if ($ClaudeMcpRemotePath) {
    if (-not (Test-Path -LiteralPath $ClaudeMcpRemotePath)) {
      throw "ClaudeMcpRemotePath nao existe: $ClaudeMcpRemotePath"
    }
    return ([System.IO.Path]::GetFullPath($ClaudeMcpRemotePath) -replace "\\", "/")
  }

  $npmRootRaw = & npm root -g 2>$null
  if ($LASTEXITCODE -ne 0 -or -not $npmRootRaw) {
    return $null
  }

  $npmRoot = (@($npmRootRaw) -join "`n").Trim()
  if (-not $npmRoot) { return $null }

  $candidate = Join-Path $npmRoot "mcp-remote\dist\proxy.js"
  if (Test-Path -LiteralPath $candidate) {
    return ([System.IO.Path]::GetFullPath($candidate) -replace "\\", "/")
  }

  return $null
}

function Resolve-ClaudeHttpToken {
  $envValue = [Environment]::GetEnvironmentVariable($ClaudeBearerTokenEnvVar)
  if ($envValue) { return $envValue.Trim() }

  $fileValue = Get-EnvFileValue -RepoDir $RepoDir -Name $ClaudeBearerTokenEnvVar
  if ($fileValue) { return $fileValue.Trim() }

  return $null
}

function Json-ServerObject([string]$IndexPath) {
  if ($ClaudeTransport -eq "http") {
    $proxyPath = Resolve-McpRemoteProxyPath
    if (-not $proxyPath) {
      Write-Warning "mcp-remote nao encontrado em (npm root -g)\mcp-remote\dist\proxy.js"
      Write-Warning "Instale com: npm install -g mcp-remote"
      Write-Warning "Em redes com MITM SSL corporativo, uma instalacao one-shot:"
      Write-Warning "  npm install -g mcp-remote --strict-ssl=false"
      Write-Warning "Ou aponte explicitamente com: -ClaudeMcpRemotePath '<abs-path>/dist/proxy.js'"
      throw "mcp-remote nao encontrado; instale antes de gerar a config Claude HTTP (ou use -ClaudeTransport stdio)."
    }

    $token = Resolve-ClaudeHttpToken
    $argsList = [System.Collections.Generic.List[string]]::new()
    $argsList.Add($proxyPath) | Out-Null
    $argsList.Add($ClaudeHttpUrl) | Out-Null
    if ($token) {
      $argsList.Add("--header") | Out-Null
      $argsList.Add("Authorization: Bearer $token") | Out-Null
    }

    return [pscustomobject]@{
      command = "node"
      args = @($argsList)
    }
  }

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
if ($ClaudeTransport -eq "stdio" -and -not (Test-Path $IndexPath)) {
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
$codexProjectTrustPaths = @(Get-CodexProjectTrustPaths $RepoDir)
$codexProjectTrustBlock = New-CodexProjectTrustBlock $codexProjectTrustPaths

if ($ClaudeTransport -eq "http") {
  $claudeTokenEnv = [Environment]::GetEnvironmentVariable($ClaudeBearerTokenEnvVar)
  $claudeTokenFile = Get-EnvFileValue -RepoDir $RepoDir -Name $ClaudeBearerTokenEnvVar
  $claudeResolved = if ($claudeTokenEnv) { $claudeTokenEnv } else { $claudeTokenFile }
  if (-not $claudeResolved) {
    Write-Warning "$ClaudeBearerTokenEnvVar nao encontrado em env nem em docker\.env/.env; o header Authorization sera omitido. Funciona apenas se MCP_HTTP_TRUST_LOCALHOST=1 no container."
  } elseif ($claudeResolved.Trim().Length -lt 16) {
    Write-Warning "$ClaudeBearerTokenEnvVar tem menos de 16 caracteres; vai falhar se MCP_HTTP_TRUST_LOCALHOST=0."
  }
}

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
$codexProjectTrustBlock[mcp_servers.$ServerName]
url = "$codexHttpToml"
bearer_token_env_var = "$codexBearerTokenEnvVarToml"
startup_timeout_sec = $codexStartupTimeoutSec
tool_timeout_sec = $codexToolTimeoutSec
required = true
enabled_tools = [$toolsToml]

# END workspace-qdrant-fork-kit
"@
} else {
@"
# BEGIN workspace-qdrant-fork-kit
$codexProjectTrustBlock[mcp_servers.$ServerName]
command = "node"
args = ["$idxToml"]
startup_timeout_sec = $codexStartupTimeoutSec
tool_timeout_sec = $codexToolTimeoutSec
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
  $clean = Remove-CodexProjectTrustTables $clean $codexProjectTrustPaths
  Write-Utf8NoBomFile $CodexConfigPath ($clean.TrimEnd() + "`r`n`r`n" + $codexBlock.TrimEnd() + "`r`n")
  Write-Host "Aplicado Codex config: $CodexConfigPath"
}
