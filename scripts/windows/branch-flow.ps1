[CmdletBinding()]
param(
  [string]$RepoDir = (Get-Location).Path,
  [ValidateSet("init-chain", "sync-main", "sync-overlay", "sync-fixes", "sync-use", "sync-chain", "status-chain", "start-fix", "merge-fix", "tag-use")]
  [string]$Action = "status-chain",
  [string]$UpstreamUrl = "https://github.com/ChrisGVE/workspace-qdrant-mcp.git",
  [string]$MainBranch = "main",
  [string]$OverlayBranch = "fork/overlay",
  [string]$FixesBranch = "fork/fixes",
  [string]$UseBranch = "personal/use-in-projects",
  [string]$FixBranch = "",
  [string]$Tag = "",
  [ValidateSet("merge", "rebase")]
  [string]$Strategy = "merge",
  [string]$Push = "false"
)

$ErrorActionPreference = "Stop"

function Should-Push {
  return $Push -match '^(1|true|yes|y)$'
}

function Invoke-Git {
  param([Parameter(ValueFromRemainingArguments = $true)][string[]]$Args)
  & git @Args
  if ($LASTEXITCODE -ne 0) {
    throw "git $($Args -join ' ') falhou com código $LASTEXITCODE"
  }
}

function Assert-CleanTree {
  $status = git status --porcelain
  if ($status) {
    Write-Host "Working tree não está limpa:" -ForegroundColor Yellow
    $status | ForEach-Object { Write-Host "  $_" }
    throw "Faça commit/stash antes de trocar/sincronizar branches."
  }
}

function Ensure-Repo {
  if (-not (Test-Path (Join-Path $RepoDir ".git"))) {
    throw "RepoDir não parece ser um repositório Git: $RepoDir"
  }
}

function Ensure-UpstreamRemote {
  $remotes = git remote
  if ($remotes -notcontains "upstream") {
    Invoke-Git remote add upstream $UpstreamUrl
  }
}

function Assert-FixBranchNameIsSafe {
  param([string]$Name)
  $protected = @($MainBranch, $OverlayBranch, $FixesBranch, $UseBranch)
  if ($protected -contains $Name) {
    throw "Nome de FixBranch bloqueado: $Name e branch protegida. Use fix/<nome-da-correcao>."
  }
  if (-not ($Name -like "fix/*")) {
    Write-Host "Aviso: recomenda-se usar prefixo fix/* para correcoes upstreamaveis." -ForegroundColor Yellow
  }
}

function Branch-Exists {
  param([string]$Name)
  & git show-ref --verify --quiet "refs/heads/$Name"
  return $LASTEXITCODE -eq 0
}

function Checkout-OrCreate {
  param(
    [string]$Branch,
    [string]$Base
  )
  if (Branch-Exists $Branch) {
    Invoke-Git checkout $Branch
  } else {
    Invoke-Git checkout -b $Branch $Base
  }
}

function Push-BranchIfRequested {
  param([string]$Branch)
  if (Should-Push) {
    Invoke-Git push -u origin $Branch
  }
}

function Sync-MainBranch {
  Assert-CleanTree
  Ensure-UpstreamRemote
  Invoke-Git fetch upstream
  if (Branch-Exists $MainBranch) {
    Invoke-Git checkout $MainBranch
  } else {
    Invoke-Git checkout -b $MainBranch "upstream/$MainBranch"
  }
  Invoke-Git merge --ff-only "upstream/$MainBranch"
  Push-BranchIfRequested $MainBranch
}

function Sync-ChildBranch {
  param(
    [string]$Child,
    [string]$Base
  )
  Assert-CleanTree
  Checkout-OrCreate $Child $Base
  if ($Strategy -eq "rebase") {
    Invoke-Git rebase $Base
  } else {
    Invoke-Git merge --no-edit $Base
  }
  Push-BranchIfRequested $Child
}

function Sync-Chain {
  Sync-MainBranch
  Sync-ChildBranch -Child $OverlayBranch -Base $MainBranch
  Sync-ChildBranch -Child $FixesBranch -Base $OverlayBranch
  Sync-ChildBranch -Child $UseBranch -Base $FixesBranch
}

function Start-FixBranch {
  if (-not $FixBranch) {
    throw "Informe -FixBranch, por exemplo: -FixBranch fix/rules-tenant-scope"
  }
  Assert-FixBranchNameIsSafe $FixBranch
  Assert-CleanTree
  Ensure-UpstreamRemote
  Invoke-Git fetch upstream
  if (-not (Branch-Exists $MainBranch)) {
    Invoke-Git checkout -b $MainBranch "upstream/$MainBranch"
  } else {
    Invoke-Git checkout $MainBranch
    Invoke-Git merge --ff-only "upstream/$MainBranch"
  }
  Invoke-Git checkout -b $FixBranch $MainBranch
  Write-Host "Branch de correção criada a partir da main: $FixBranch"
  Write-Host "Aplique o patch, rode testes, faça commit e depois use:"
  Write-Host "  make -f Makefile.win fix-promote FIX_BRANCH=$FixBranch"
}

function Merge-FixIntoPersonalLine {
  if (-not $FixBranch) {
    throw "Informe -FixBranch, por exemplo: -FixBranch fix/rules-tenant-scope"
  }
  Assert-FixBranchNameIsSafe $FixBranch
  Assert-CleanTree
  if (-not (Branch-Exists $FixBranch)) {
    throw "Branch de correção não encontrada: $FixBranch"
  }
  Sync-ChildBranch -Child $OverlayBranch -Base $MainBranch
  Sync-ChildBranch -Child $FixesBranch -Base $OverlayBranch
  Invoke-Git checkout $FixesBranch
  Invoke-Git merge --no-edit $FixBranch
  Push-BranchIfRequested $FixesBranch
  Sync-ChildBranch -Child $UseBranch -Base $FixesBranch
}

function Tag-UseBranch {
  if (-not $Tag) {
    throw "Informe -Tag, por exemplo: -Tag fork-claude-codex-ready-v0.1.0"
  }
  Assert-CleanTree
  Checkout-OrCreate $UseBranch $FixesBranch
  Invoke-Git tag -a $Tag -m $Tag
  if (Should-Push) {
    Invoke-Git push origin $Tag
  }
}

function Show-StatusChain {
  Write-Host "Repo: $RepoDir"
  Write-Host "Branches:"
  Write-Host "  Main:    $MainBranch"
  Write-Host "  Overlay: $OverlayBranch"
  Write-Host "  Fixes:   $FixesBranch"
  Write-Host "  Use:     $UseBranch"
  Write-Host "  Strategy:$Strategy"
  Write-Host "  Push:    $Push"
  Write-Host "  Main guard: agentes nao devem commitar, mergear ou abrir PR para main."
  Write-Host ""
  Invoke-Git status --short --branch
  Write-Host ""
  Invoke-Git branch --list $MainBranch $OverlayBranch $FixesBranch $UseBranch
  Write-Host ""
  Invoke-Git log --oneline --graph --decorate --all -n 24
}

Ensure-Repo
Push-Location $RepoDir
try {
  switch ($Action) {
    "init-chain"   { Sync-Chain }
    "sync-main"    { Sync-MainBranch }
    "sync-overlay" { Sync-ChildBranch -Child $OverlayBranch -Base $MainBranch }
    "sync-fixes"   { Sync-ChildBranch -Child $FixesBranch -Base $OverlayBranch }
    "sync-use"     { Sync-ChildBranch -Child $UseBranch -Base $FixesBranch }
    "sync-chain"   { Sync-Chain }
    "status-chain" { Show-StatusChain }
    "start-fix"    { Start-FixBranch }
    "merge-fix"    { Merge-FixIntoPersonalLine }
    "tag-use"      { Tag-UseBranch }
  }
} finally {
  Pop-Location
}
