[CmdletBinding()]
param(
  [string]$RepoDir = (Get-Location).Path,
  [string]$McpUrl = "",
  [string]$Token = "",
  [string]$ProjectDir = "",
  [string]$ProjectName = "semantic-project",
  [int]$Attempts = 40,
  [int]$PollSeconds = 2
)

$ErrorActionPreference = "Stop"

. (Join-Path $PSScriptRoot "semantic-mcp-common.ps1")

$McpUrl = Resolve-McpUrl -RepoDir $RepoDir -McpUrl $McpUrl
$Token = Resolve-McpToken -RepoDir $RepoDir -Token $Token

$runner = Join-Path $PSScriptRoot "semantic-mcp-smoke.mjs"
$nodeArgs = @(
  '--mode', 'project',
  '--repo-dir', $RepoDir,
  '--mcp-url', $McpUrl,
  '--token', $Token,
  '--project-name', $ProjectName,
  '--attempts', "$Attempts",
  '--poll-seconds', "$PollSeconds"
)

if ($ProjectDir) {
  $nodeArgs += @('--project-dir', $ProjectDir)
}

& node $runner @nodeArgs

if ($LASTEXITCODE -ne 0) {
  exit $LASTEXITCODE
}
