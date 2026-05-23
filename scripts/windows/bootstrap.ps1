[CmdletBinding()]
param(
  [string]$RepoDir = (Get-Location).Path,
  [string]$Branch = "personal/windows-hardening",
  [string]$Upstream = "https://github.com/ChrisGVE/workspace-qdrant-mcp.git",
  [string]$ForkUrl = "",
  [switch]$CreateForkWithGh,
  [switch]$BuildRust
)

$ErrorActionPreference = "Stop"

function Ensure-Command([string]$Name) {
  if (-not (Get-Command $Name -ErrorAction SilentlyContinue)) {
    throw "$Name não encontrado no PATH"
  }
}

Ensure-Command git
Ensure-Command node
Ensure-Command npm

if ($CreateForkWithGh) {
  Ensure-Command gh
  Write-Host "Criando fork com gh, se necessário..."
  gh repo fork ChrisGVE/workspace-qdrant-mcp --clone=false
}

if (-not (Test-Path $RepoDir)) {
  if (-not $ForkUrl) {
    throw "RepoDir não existe e ForkUrl não foi informado. Clone seu fork ou chame com -ForkUrl."
  }
  git clone $ForkUrl $RepoDir
}

if (-not (Test-Path (Join-Path $RepoDir ".git"))) {
  throw "RepoDir não parece um repo Git: $RepoDir"
}

Push-Location $RepoDir
try {
  $remotes = git remote
  if ($remotes -notcontains "upstream") {
    git remote add upstream $Upstream
  }

  git fetch upstream
  git fetch origin

  Write-Host "Criando/resetando branch local $Branch a partir de upstream/main"
  git checkout -B $Branch upstream/main

  $mcpDir = Join-Path $RepoDir "src\typescript\mcp-server"
  if (Test-Path $mcpDir) {
    Push-Location $mcpDir
    try {
      npm install
      npm run build
    } finally {
      Pop-Location
    }
  } else {
    Write-Warning "Diretório TypeScript do MCP não encontrado: $mcpDir"
  }

  if ($BuildRust) {
    Ensure-Command cargo
    Push-Location (Join-Path $RepoDir "src\rust")
    try {
      cargo check --manifest-path Cargo.toml --workspace
    } finally {
      Pop-Location
    }
  }

  Write-Host "Bootstrap concluído."
  git status --short
} finally {
  Pop-Location
}
