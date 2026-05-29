[CmdletBinding()]
param(
  [string]$RepoDir = (Get-Location).Path,
  [ValidateSet("status", "sync-upstream", "sync-dev", "promote", "start-fix", "finish-fix", "tag-main")]
  [string]$Action = "status",
  [string]$UpstreamUrl = "https://github.com/ChrisGVE/workspace-qdrant-mcp.git",
  [string]$MainBranch = "main",
  [string]$DevBranch = "dev",
  [string]$UpstreamSyncBranch = "upstream-sync",
  [string]$FixBranch = "",
  [string]$Tag = "",
  [ValidateSet("merge", "rebase")]
  [string]$Strategy = "merge",
  [string]$Push = "false"
)

$ErrorActionPreference = "Stop"

# Branch model (GitFlow-lite, since 2026-05-29). See AGENTS.md › "Modelo de branches".
#   upstream/main --(ff)----> upstream-sync   clean fetch-only mirror; never commit
#   upstream-sync --(merge)-> dev             active work line (CI runs on dev)
#   dev --(promote when stable)-> main        stable working version
# Feature work: fix/* branches cut from dev. The `upstream` remote push URL is
# intentionally DISABLED (git remote set-url --push upstream DISABLED); this
# script only ever pushes to `origin`, never to `upstream`.

function Should-Push {
  return $Push -match '^(1|true|yes|y)$'
}

function Invoke-Git {
  param([Parameter(ValueFromRemainingArguments = $true)][string[]]$GitArgs)
  & git @GitArgs
  if ($LASTEXITCODE -ne 0) {
    throw "git $($GitArgs -join ' ') falhou com código $LASTEXITCODE"
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

function Branch-Exists {
  param([string]$Name)
  & git show-ref --verify --quiet "refs/heads/$Name"
  return $LASTEXITCODE -eq 0
}

function Checkout-OrCreate {
  param([string]$Branch, [string]$Base)
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

# Branches that must never be reused as a fix/feature branch name.
function Assert-FixBranchNameIsSafe {
  param([string]$Name)
  $protected = @(
    $MainBranch, $DevBranch, $UpstreamSyncBranch,
    "main", "master", "dev", "upstream-sync",
    "fork/overlay", "fork/fixes", "personal/use-in-projects"
  )
  if ($protected -contains $Name) {
    throw "Nome de FixBranch bloqueado: $Name é branch protegida. Use fix/<nome-da-correção>."
  }
  if ($Name -notmatch '^(fix|chore|refactor|docs|test|local)/[A-Za-z0-9._/-]+$') {
    Write-Host "Aviso: recomenda-se prefixo fix/ chore/ refactor/ docs/ test/ ou local/ para a branch de trabalho." -ForegroundColor Yellow
  }
}

# upstream/main --(ff)--> upstream-sync. Never commits; never pushes to upstream.
function Sync-UpstreamSync {
  Assert-CleanTree
  Ensure-UpstreamRemote
  Invoke-Git fetch upstream
  if (Branch-Exists $UpstreamSyncBranch) {
    Invoke-Git checkout $UpstreamSyncBranch
  } else {
    Invoke-Git checkout -b $UpstreamSyncBranch "upstream/$MainBranch"
  }
  Invoke-Git merge --ff-only "upstream/$MainBranch"
  Push-BranchIfRequested $UpstreamSyncBranch
}

# upstream-sync --(merge|rebase)--> dev. Brings upstream updates into the work line.
function Sync-Dev {
  Sync-UpstreamSync
  Assert-CleanTree
  if (Branch-Exists $DevBranch) {
    Invoke-Git checkout $DevBranch
  } else {
    Invoke-Git checkout -b $DevBranch $MainBranch
  }
  if ($Strategy -eq "rebase") {
    Invoke-Git rebase $UpstreamSyncBranch
  } else {
    Invoke-Git merge --no-edit $UpstreamSyncBranch
  }
  Push-BranchIfRequested $DevBranch
}

# dev --(promote when stable)--> main. Run only after dev is validated (CI green).
function Promote-DevToMain {
  Assert-CleanTree
  if (-not (Branch-Exists $DevBranch)) {
    throw "Branch de trabalho não encontrada: $DevBranch"
  }
  Write-Host "Promovendo $DevBranch -> $MainBranch. Faça isso só quando $DevBranch estiver estável (CI verde)." -ForegroundColor Cyan
  if (Branch-Exists $MainBranch) {
    Invoke-Git checkout $MainBranch
  } else {
    Invoke-Git checkout -b $MainBranch $DevBranch
  }
  Invoke-Git merge --no-ff --no-edit $DevBranch
  Push-BranchIfRequested $MainBranch
  Write-Host "Alternativa via PR: gh pr create --base $MainBranch --head $DevBranch" -ForegroundColor DarkGray
}

# fix/* cut from dev.
function Start-FixBranch {
  if (-not $FixBranch) {
    throw "Informe -FixBranch, por exemplo: -FixBranch fix/minha-correcao"
  }
  Assert-FixBranchNameIsSafe $FixBranch
  Assert-CleanTree
  if (-not (Branch-Exists $DevBranch)) {
    throw "Branch de trabalho não encontrada: $DevBranch. Rode sync-dev primeiro."
  }
  Invoke-Git checkout $DevBranch
  Invoke-Git checkout -b $FixBranch $DevBranch
  Write-Host "Branch de trabalho criada a partir de ${DevBranch}: $FixBranch"
  Write-Host "Ao terminar: make -f Makefile.win fix-promote FIX_BRANCH=$FixBranch"
}

# fix/* --(merge)--> dev.
function Finish-FixBranch {
  if (-not $FixBranch) {
    throw "Informe -FixBranch, por exemplo: -FixBranch fix/minha-correcao"
  }
  Assert-FixBranchNameIsSafe $FixBranch
  Assert-CleanTree
  if (-not (Branch-Exists $FixBranch)) {
    throw "Branch de correção não encontrada: $FixBranch"
  }
  Checkout-OrCreate $DevBranch $MainBranch
  Invoke-Git merge --no-edit $FixBranch
  Push-BranchIfRequested $DevBranch
}

# Tag a stable point on main.
function Tag-Main {
  if (-not $Tag) {
    throw "Informe -Tag, por exemplo: -Tag fork-claude-codex-ready-v0.1.0"
  }
  Assert-CleanTree
  if (-not (Branch-Exists $MainBranch)) {
    throw "Branch principal não encontrada: $MainBranch"
  }
  Invoke-Git checkout $MainBranch
  Invoke-Git tag -a $Tag -m $Tag
  if (Should-Push) {
    Invoke-Git push origin $Tag
  }
}

function Show-Status {
  Write-Host "Repo: $RepoDir"
  Write-Host "Modelo: main (estável) <- dev (trabalho, CI) <- upstream-sync (espelho fetch-only de upstream/main)"
  Write-Host "Branches:"
  Write-Host "  Main:          $MainBranch"
  Write-Host "  Dev:           $DevBranch"
  Write-Host "  Upstream-sync: $UpstreamSyncBranch"
  Write-Host "  Strategy:      $Strategy"
  Write-Host "  Push:          $Push"
  Write-Host "  Nota: trabalho vai em dev (ou fix/*); main só recebe promoções estáveis; upstream é fetch-only (push desabilitado)."
  Write-Host ""
  Invoke-Git status --short --branch
  Write-Host ""
  Invoke-Git branch --list $MainBranch $DevBranch $UpstreamSyncBranch
  Write-Host ""
  Invoke-Git log --oneline --graph --decorate --all -n 24
}

Ensure-Repo
Push-Location $RepoDir
try {
  switch ($Action) {
    "status"        { Show-Status }
    "sync-upstream" { Sync-UpstreamSync }
    "sync-dev"      { Sync-Dev }
    "promote"       { Promote-DevToMain }
    "start-fix"     { Start-FixBranch }
    "finish-fix"    { Finish-FixBranch }
    "tag-main"      { Tag-Main }
  }
} finally {
  Pop-Location
}
