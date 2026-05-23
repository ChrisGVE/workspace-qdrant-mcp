[CmdletBinding()]
param(
  [string]$RegistryPath = ".wqm-fork\projects.json",
  [ValidateSet("init", "list", "add", "remove", "status", "status-all", "sync-project", "sync-all", "start-fix", "promote-fix", "observe-all")]
  [string]$Action = "list",
  [string]$Name = "",
  [string]$ProjectDir = "",
  [string]$MainBranch = "main",
  [string]$OverlayBranch = "fork/overlay",
  [string]$FixesBranch = "fork/fixes",
  [string]$UseBranch = "personal/use-in-projects",
  [string]$UpstreamRemote = "upstream",
  [string]$UpstreamUrl = "https://github.com/ChrisGVE/workspace-qdrant-mcp.git",
  [string]$UpstreamRef = "upstream/main",
  [string]$FixBranch = "",
  [string]$QdrantUrl = "http://localhost:6333",
  [string]$DaemonEndpoint = "localhost:50051",
  [string]$LogDir = ".wqm-fork\logs",
  [string]$Push = "false",
  [string]$Mutate = "false",
  [int]$IntervalSeconds = 30
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path

function Test-TrueValue([string]$Value) {
  return $Value -match '^(1|true|yes|y)$'
}

function Assert-MutationAllowed {
  if (-not (Test-TrueValue $Mutate)) {
    throw "Acao mutavel bloqueada. Reexecute com -Mutate true apos confirmacao humana explicita."
  }
}

function Convert-ToAbsolutePath([string]$PathValue) {
  if (-not $PathValue) { return $PathValue }
  return [System.IO.Path]::GetFullPath((Resolve-Path -LiteralPath $PathValue -ErrorAction SilentlyContinue) ?? $PathValue)
}

function New-EmptyRegistry {
  return [ordered]@{
    schemaVersion = 1
    updatedAt = (Get-Date).ToUniversalTime().ToString("o")
    projects = @()
  }
}

function Read-Registry {
  if (-not (Test-Path -LiteralPath $RegistryPath)) {
    return New-EmptyRegistry
  }
  $raw = Get-Content -LiteralPath $RegistryPath -Raw
  if (-not $raw.Trim()) { return New-EmptyRegistry }
  return $raw | ConvertFrom-Json
}

function Write-Registry($Registry) {
  $Registry.updatedAt = (Get-Date).ToUniversalTime().ToString("o")
  $parent = Split-Path -Parent $RegistryPath
  if ($parent -and -not (Test-Path -LiteralPath $parent)) {
    New-Item -ItemType Directory -Path $parent -Force | Out-Null
  }
  $Registry | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $RegistryPath -Encoding UTF8
}

function Normalize-Project($Project) {
  $defaults = [ordered]@{
    name = $Project.name
    path = $Project.path
    enabled = if ($null -ne $Project.enabled) { [bool]$Project.enabled } else { $true }
    mainBranch = if ($Project.mainBranch) { $Project.mainBranch } else { $MainBranch }
    overlayBranch = if ($Project.overlayBranch) { $Project.overlayBranch } else { $OverlayBranch }
    fixesBranch = if ($Project.fixesBranch) { $Project.fixesBranch } else { $FixesBranch }
    useBranch = if ($Project.useBranch) { $Project.useBranch } else { $UseBranch }
    upstreamRemote = if ($Project.upstreamRemote) { $Project.upstreamRemote } else { $UpstreamRemote }
    upstreamUrl = if ($Project.upstreamUrl) { $Project.upstreamUrl } else { $UpstreamUrl }
    upstreamRef = if ($Project.upstreamRef) { $Project.upstreamRef } else { $UpstreamRef }
    qdrantUrl = if ($Project.qdrantUrl) { $Project.qdrantUrl } else { $QdrantUrl }
    daemonEndpoint = if ($Project.daemonEndpoint) { $Project.daemonEndpoint } else { $DaemonEndpoint }
  }
  return [pscustomobject]$defaults
}

function Get-Projects {
  param($Registry)
  @($Registry.projects) | ForEach-Object { Normalize-Project $_ }
}

function Find-Project {
  param($Registry)
  $projects = Get-Projects $Registry
  if ($Name) {
    $match = @($projects | Where-Object { $_.name -eq $Name })
    if ($match.Count -eq 0) { throw "Projeto nao encontrado no registry: $Name" }
    if ($match.Count -gt 1) { throw "Nome ambiguo no registry: $Name" }
    return $match[0]
  }
  if ($ProjectDir) {
    $abs = Convert-ToAbsolutePath $ProjectDir
    $match = @($projects | Where-Object { (Convert-ToAbsolutePath $_.path) -eq $abs })
    if ($match.Count -eq 0) { throw "ProjectDir nao encontrado no registry: $ProjectDir" }
    return $match[0]
  }
  throw "Informe -Name ou -ProjectDir."
}

function Invoke-Git {
  param(
    [Parameter(Mandatory=$true)][string]$Repo,
    [Parameter(ValueFromRemainingArguments=$true)][string[]]$Args
  )
  & git -C $Repo @Args
  if ($LASTEXITCODE -ne 0) {
    throw "git -C $Repo $($Args -join ' ') falhou com codigo $LASTEXITCODE"
  }
}

function Invoke-GitText {
  param([string]$Repo, [string[]]$Args)
  $out = & git -C $Repo @Args 2>&1
  $code = $LASTEXITCODE
  return [pscustomobject]@{ code = $code; output = @($out) }
}

function Assert-RepoClean {
  param([string]$Repo)
  $status = & git -C $Repo status --porcelain
  if ($status) {
    Write-Host "Working tree suja em $Repo" -ForegroundColor Yellow
    $status | ForEach-Object { Write-Host "  $_" }
    throw "Faça commit/stash antes de sincronizar branches deste projeto."
  }
}

function Ensure-Remote {
  param($Project)
  $remotes = @(& git -C $Project.path remote)
  if ($remotes -notcontains $Project.upstreamRemote) {
    Invoke-Git -Repo $Project.path remote add $Project.upstreamRemote $Project.upstreamUrl
  }
}

function Test-BranchExists {
  param([string]$Repo, [string]$Branch)
  & git -C $Repo show-ref --verify --quiet "refs/heads/$Branch"
  return $LASTEXITCODE -eq 0
}

function Checkout-OrCreate {
  param([string]$Repo, [string]$Branch, [string]$Base)
  if (Test-BranchExists -Repo $Repo -Branch $Branch) {
    Invoke-Git -Repo $Repo checkout $Branch
  } else {
    Invoke-Git -Repo $Repo checkout -b $Branch $Base
  }
}

function Push-IfRequested {
  param([string]$Repo, [string]$Branch)
  if (Test-TrueValue $Push) {
    Invoke-Git -Repo $Repo push -u origin $Branch
  }
}

function Get-ProjectStatus {
  param($Project)
  if (-not (Test-Path -LiteralPath (Join-Path $Project.path ".git"))) {
    return [pscustomobject]@{ name=$Project.name; path=$Project.path; ok=$false; error="Not a git repository" }
  }
  $branch = Invoke-GitText -Repo $Project.path -Args @("branch", "--show-current")
  $status = Invoke-GitText -Repo $Project.path -Args @("status", "--short", "--branch")
  $branches = Invoke-GitText -Repo $Project.path -Args @("branch", "--list", $Project.mainBranch, $Project.overlayBranch, $Project.fixesBranch, $Project.useBranch)
  return [pscustomobject]@{
    name = $Project.name
    path = $Project.path
    enabled = $Project.enabled
    ok = $true
    currentBranch = (($branch.output -join "`n").Trim())
    clean = -not (@($status.output | Where-Object { $_ -notmatch '^## ' }).Count)
    status = @($status.output)
    managedBranches = @($branches.output)
    mainBranch = $Project.mainBranch
    overlayBranch = $Project.overlayBranch
    fixesBranch = $Project.fixesBranch
    useBranch = $Project.useBranch
    upstreamRef = $Project.upstreamRef
  }
}

function Sync-ProjectChain {
  param($Project)
  Assert-MutationAllowed
  Assert-RepoClean -Repo $Project.path
  Ensure-Remote $Project
  Invoke-Git -Repo $Project.path fetch $Project.upstreamRemote

  # Importante: este fluxo MCP/registry nao faz checkout nem merge na main.
  # Ele usa upstream/main como base para overlay e preserva main como espelho humano.
  Checkout-OrCreate -Repo $Project.path -Branch $Project.overlayBranch -Base $Project.upstreamRef
  Invoke-Git -Repo $Project.path merge --no-edit $Project.upstreamRef
  Push-IfRequested -Repo $Project.path -Branch $Project.overlayBranch

  Checkout-OrCreate -Repo $Project.path -Branch $Project.fixesBranch -Base $Project.overlayBranch
  Invoke-Git -Repo $Project.path merge --no-edit $Project.overlayBranch
  Push-IfRequested -Repo $Project.path -Branch $Project.fixesBranch

  Checkout-OrCreate -Repo $Project.path -Branch $Project.useBranch -Base $Project.fixesBranch
  Invoke-Git -Repo $Project.path merge --no-edit $Project.fixesBranch
  Push-IfRequested -Repo $Project.path -Branch $Project.useBranch

  return Get-ProjectStatus $Project
}

function Assert-SafeFixBranch {
  param($Project)
  if (-not $FixBranch) { throw "Informe -FixBranch, por exemplo fix/minha-correcao." }
  $forbidden = @($Project.mainBranch, "main", "master", $Project.overlayBranch, $Project.fixesBranch, $Project.useBranch)
  if ($forbidden -contains $FixBranch) { throw "FixBranch proibida: $FixBranch" }
  if ($FixBranch -notmatch '^(fix|chore|refactor|docs|test|local)/[A-Za-z0-9._/-]+$') {
    throw "FixBranch deve começar com fix/, chore/, refactor/, docs/, test/ ou local/: $FixBranch"
  }
}

function Start-FixBranch {
  param($Project)
  Assert-MutationAllowed
  Assert-SafeFixBranch $Project
  Assert-RepoClean -Repo $Project.path
  Ensure-Remote $Project
  Invoke-Git -Repo $Project.path fetch $Project.upstreamRemote
  if (Test-BranchExists -Repo $Project.path -Branch $FixBranch) {
    Invoke-Git -Repo $Project.path checkout $FixBranch
  } else {
    Invoke-Git -Repo $Project.path checkout -b $FixBranch $Project.upstreamRef
  }
  return Get-ProjectStatus $Project
}

function Promote-FixBranch {
  param($Project)
  Assert-MutationAllowed
  Assert-SafeFixBranch $Project
  Assert-RepoClean -Repo $Project.path
  if (-not (Test-BranchExists -Repo $Project.path -Branch $FixBranch)) {
    throw "Branch de correcao nao encontrada: $FixBranch"
  }
  Checkout-OrCreate -Repo $Project.path -Branch $Project.fixesBranch -Base $Project.overlayBranch
  Invoke-Git -Repo $Project.path merge --no-edit $FixBranch
  Push-IfRequested -Repo $Project.path -Branch $Project.fixesBranch
  Checkout-OrCreate -Repo $Project.path -Branch $Project.useBranch -Base $Project.fixesBranch
  Invoke-Git -Repo $Project.path merge --no-edit $Project.fixesBranch
  Push-IfRequested -Repo $Project.path -Branch $Project.useBranch
  return Get-ProjectStatus $Project
}

function Observe-All {
  param($Registry)
  $projects = Get-Projects $Registry | Where-Object { $_.enabled }
  foreach ($p in $projects) {
    Write-Host "== Observando $($p.name) ==" -ForegroundColor Cyan
    & (Join-Path $ScriptDir "service-observe.ps1") -RepoDir $p.path -ProjectDir $p.path -QdrantUrl $p.qdrantUrl -DaemonEndpoint $p.daemonEndpoint -LogDir $LogDir -Once
  }
}

$registry = Read-Registry

switch ($Action) {
  "init" {
    Assert-MutationAllowed
    if (-not (Test-Path -LiteralPath $RegistryPath)) { Write-Registry $registry }
    Write-Host "Registry pronto: $RegistryPath"
  }
  "list" {
    Get-Projects $registry | ConvertTo-Json -Depth 12
  }
  "add" {
    Assert-MutationAllowed
    if (-not $Name) { throw "Informe -Name." }
    if (-not $ProjectDir) { throw "Informe -ProjectDir." }
    $abs = Convert-ToAbsolutePath $ProjectDir
    if (-not (Test-Path -LiteralPath (Join-Path $abs ".git"))) { throw "ProjectDir nao parece ser repo git: $abs" }
    $projects = @(Get-Projects $registry | Where-Object { $_.name -ne $Name })
    $entry = [pscustomobject]@{
      name = $Name
      path = $abs
      enabled = $true
      mainBranch = $MainBranch
      overlayBranch = $OverlayBranch
      fixesBranch = $FixesBranch
      useBranch = $UseBranch
      upstreamRemote = $UpstreamRemote
      upstreamUrl = $UpstreamUrl
      upstreamRef = $UpstreamRef
      qdrantUrl = $QdrantUrl
      daemonEndpoint = $DaemonEndpoint
    }
    $registry.projects = @($projects + $entry)
    Write-Registry $registry
    Write-Host "Projeto registrado: $Name -> $abs"
  }
  "remove" {
    Assert-MutationAllowed
    if (-not $Name) { throw "Informe -Name." }
    $registry.projects = @(Get-Projects $registry | Where-Object { $_.name -ne $Name })
    Write-Registry $registry
    Write-Host "Projeto removido do registry: $Name"
  }
  "status" {
    $p = Find-Project $registry
    Get-ProjectStatus $p | ConvertTo-Json -Depth 12
  }
  "status-all" {
    @(Get-Projects $registry | Where-Object { $_.enabled } | ForEach-Object { Get-ProjectStatus $_ }) | ConvertTo-Json -Depth 12
  }
  "sync-project" {
    $p = Find-Project $registry
    Sync-ProjectChain $p | ConvertTo-Json -Depth 12
  }
  "sync-all" {
    $results = @()
    foreach ($p in (Get-Projects $registry | Where-Object { $_.enabled })) {
      $results += Sync-ProjectChain $p
    }
    $results | ConvertTo-Json -Depth 12
  }
  "start-fix" {
    $p = Find-Project $registry
    Start-FixBranch $p | ConvertTo-Json -Depth 12
  }
  "promote-fix" {
    $p = Find-Project $registry
    Promote-FixBranch $p | ConvertTo-Json -Depth 12
  }
  "observe-all" {
    Observe-All $registry
  }
}
