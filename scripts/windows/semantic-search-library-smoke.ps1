[CmdletBinding()]
param(
  [string]$RepoDir = (Get-Location).Path,
  [string]$McpUrl = "",
  [string]$Token = "",
  [string]$LibraryName = "semantic-smoke",
  [int]$Attempts = 30,
  [int]$PollSeconds = 2
)

$ErrorActionPreference = "Stop"

. (Join-Path $PSScriptRoot "semantic-mcp-common.ps1")

$McpUrl = Resolve-McpUrl -RepoDir $RepoDir -McpUrl $McpUrl
$Token = Resolve-McpToken -RepoDir $RepoDir -Token $Token

$runner = Join-Path $PSScriptRoot "semantic-mcp-smoke.mjs"
$nodeArgs = @(
  '--mode', 'library',
  '--repo-dir', $RepoDir,
  '--mcp-url', $McpUrl,
  '--token', $Token,
  '--library-name', $LibraryName,
  '--attempts', "$Attempts",
  '--poll-seconds', "$PollSeconds"
)

& node $runner @nodeArgs

if ($LASTEXITCODE -ne 0) {
  exit $LASTEXITCODE
}
