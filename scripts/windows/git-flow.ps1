[CmdletBinding()]
param(
  [string]$RepoDir = (Get-Location).Path,
  [string]$Branch = "personal/windows-hardening",
  [ValidateSet("branch", "sync", "push", "pr")]
  [string]$Action = "branch",
  [string]$Upstream = "https://github.com/ChrisGVE/workspace-qdrant-mcp.git"
)

$ErrorActionPreference = "Stop"

function Assert-NotMainBranchName {
  param([string]$BranchName)
  if ($BranchName -eq "main") {
    throw "Operacao bloqueada: agentes nao devem trabalhar, fazer push ou abrir PR a partir de/para main."
  }
}

if (-not (Test-Path (Join-Path $RepoDir ".git"))) {
  throw "RepoDir não parece repo Git: $RepoDir"
}

Push-Location $RepoDir
try {
  $remotes = git remote
  if ($remotes -notcontains "upstream") {
    git remote add upstream $Upstream
  }

  switch ($Action) {
    "branch" {
      Assert-NotMainBranchName $Branch
      git fetch upstream
      git checkout -B $Branch upstream/main
      git status --short
    }
    "sync" {
      git fetch upstream
      git checkout $Branch
      git merge --ff-only upstream/main
      git status --short
    }
    "push" {
      Assert-NotMainBranchName $Branch
      git checkout $Branch
      git push -u origin $Branch
    }
    "pr" {
      throw "PR automatizado para main esta bloqueado neste fork. Use branch fix/*, gere o commit/patch e deixe o usuario humano abrir PR explicitamente."
    }
  }
} finally {
  Pop-Location
}
