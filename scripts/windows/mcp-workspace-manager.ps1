[CmdletBinding()]
param(
  [string]$RepoDir = (Get-Location).Path,
  [string]$PatchPath = "patches\mcp\fork-workspace-mcp-tool.patch",
  [switch]$CheckOnly,
  [switch]$ValidateOnly
)

$ErrorActionPreference = "Stop"

function Invoke-Git {
  param([Parameter(ValueFromRemainingArguments = $true)][string[]]$Args)
  & git @Args
  if ($LASTEXITCODE -ne 0) { throw "git $($Args -join ' ') falhou com codigo $LASTEXITCODE" }
}

Push-Location $RepoDir
try {
  if (-not (Test-Path ".git")) { throw "RepoDir nao parece ser git repo: $RepoDir" }

  if ($ValidateOnly) {
    $markers = @(
      "src/typescript/mcp-server/src/tools/fork-workspace.ts",
      "src/typescript/mcp-server/src/tool-definitions/fork-workspace.ts"
    )
    foreach ($m in $markers) {
      if (-not (Test-Path $m)) { throw "Arquivo MCP manager ausente: $m" }
    }
    git grep "fork_workspace" -- src/typescript/mcp-server/src | Out-Host
    if ($LASTEXITCODE -ne 0) { throw "Marcador fork_workspace nao encontrado." }
    Write-Host "MCP workspace manager parece instalado."
    return
  }

  if (git grep -q "fork_workspace" -- src/typescript/mcp-server/src 2>$null) {
    Write-Host "MCP workspace manager ja parece estar aplicado."
    return
  }

  if (-not (Test-Path $PatchPath)) { throw "Patch nao encontrado: $PatchPath" }

  Invoke-Git apply --check $PatchPath
  if ($CheckOnly) {
    Write-Host "Patch aplicavel: $PatchPath"
    return
  }
  Invoke-Git apply $PatchPath
  Write-Host "Patch aplicado: $PatchPath"
  Write-Host "Proximos passos: make -f Makefile.win typecheck-ts; make -f Makefile.win test-ts; git add src/typescript/mcp-server/src; git commit -m 'feat(mcp): add safe fork workspace manager tool'"
} finally {
  Pop-Location
}
