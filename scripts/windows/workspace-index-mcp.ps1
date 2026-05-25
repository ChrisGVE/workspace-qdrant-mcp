[CmdletBinding()]
param(
  [string]$Action = "list-projects",
  [string]$RegistryPath = ".wqm-fork\indexed-projects.json",
  [string]$ProjectName = "",
  [string]$ProjectId = "",
  [string]$ProjectDir = "",
  [string]$BranchName = "",
  [string]$BaseBranch = "main",
  [string]$ReturnBranch = "",
  [string]$WorktreePath = "",
  [string]$WorktreeRoot = "",
  [string]$UseWorktree = "true",
  [string]$Purpose = "agent change",
  [string]$CreatedBy = "mcp-agent",
  [string]$AllowMutation = "false"
)

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$registryScript = Join-Path $scriptDir "indexed-projects-registry.ps1"
$mutate = "false"
if (($AllowMutation -match '^(1|true|yes|y)$') -and ($env:WQM_INDEX_MANAGER_ALLOW_MUTATION -match '^(1|true|yes|y)$')) {
  $mutate = "true"
}
& $registryScript -Action $Action -RegistryPath $RegistryPath -ProjectName $ProjectName -ProjectId $ProjectId -ProjectDir $ProjectDir -BranchName $BranchName -BaseBranch $BaseBranch -ReturnBranch $ReturnBranch -WorktreePath $WorktreePath -WorktreeRoot $WorktreeRoot -UseWorktree $UseWorktree -Purpose $Purpose -CreatedBy $CreatedBy -Mutate $mutate -Json true
exit $LASTEXITCODE
