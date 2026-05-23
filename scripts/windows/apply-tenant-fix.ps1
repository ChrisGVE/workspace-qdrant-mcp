[CmdletBinding()]
param(
  [string]$RepoDir = (Get-Location).Path,
  [string]$PatchPath = "patches/fixes/fix-rules-tenant-scope.patch",
  [string]$ExtraPatchPath = "patches/fixes/harden-rules-add-scope-error.patch",
  [switch]$CheckOnly,
  [switch]$ValidateOnly
)

$ErrorActionPreference = "Stop"

function Invoke-Git {
  param([Parameter(ValueFromRemainingArguments = $true)][string[]]$Args)
  & git @Args
  if ($LASTEXITCODE -ne 0) {
    throw "git $($Args -join ' ') falhou com codigo $LASTEXITCODE"
  }
}

function Resolve-RepoPath {
  param([string]$Base, [string]$PathValue)
  if ([string]::IsNullOrWhiteSpace($PathValue)) { return $null }
  if ([System.IO.Path]::IsPathRooted($PathValue)) { return $PathValue }
  return (Join-Path $Base $PathValue)
}

function Assert-Repo {
  if (-not (Test-Path (Join-Path $RepoDir ".git"))) {
    throw "RepoDir nao parece ser um repositorio Git: $RepoDir"
  }
}

function Assert-NotMain {
  $branch = (git branch --show-current).Trim()
  if ($branch -eq "main") {
    throw "Bloqueado: nao aplique tenant fix diretamente na main. Use make -f Makefile.win fix-start FIX_BRANCH=fix/rules-tenant-scope."
  }
}

function Assert-CleanTree {
  $status = git status --porcelain
  if ($status) {
    Write-Host "Working tree nao esta limpa:" -ForegroundColor Yellow
    $status | ForEach-Object { Write-Host "  $_" }
    throw "Faca commit/stash antes de aplicar patches de tenant."
  }
}

function Test-FileContains {
  param([string]$PathValue, [string]$Needle)
  if (-not (Test-Path $PathValue)) { return $false }
  $content = Get-Content -Raw -Path $PathValue
  return $content.Contains($Needle)
}

function Test-CoreTenantFixApplied {
  $rules = Join-Path $RepoDir "src/typescript/mcp-server/src/tools/rules.ts"
  $helpers = Join-Path $RepoDir "src/typescript/mcp-server/src/tools/rules-mutation-helpers.ts"
  $mutations = Join-Path $RepoDir "src/typescript/mcp-server/src/tools/rules-mutations.ts"

  $hasDuplicateScope = Test-FileContains $rules "buildDuplicateScopeFilter"
  $hasProjectField = Test-FileContains $rules "FIELD_PROJECT_ID"
  $hasTenantUpdate = Test-FileContains $helpers "const tenantId = resolvedProjectId ?? TENANT_GLOBAL"
  $hasScopedUpdate = Test-FileContains $mutations "projectDetector: ProjectDetector"
  $hasScopedRemove = Test-FileContains $mutations "scope = 'project', projectId"

  return ($hasDuplicateScope -and $hasProjectField -and $hasTenantUpdate -and $hasScopedUpdate -and $hasScopedRemove)
}

function Test-TenantHardeningApplied {
  $rules = Join-Path $RepoDir "src/typescript/mcp-server/src/tools/rules.ts"
  return (
    (Test-FileContains $rules "scopeError") -and
    (Test-FileContains $rules "if (scopeError) return scopeError")
  )
}

function Test-PatchApplies {
  param([string]$PatchFullPath)
  if (-not (Test-Path $PatchFullPath)) { throw "Patch nao encontrado: $PatchFullPath" }
  Invoke-Git apply --check $PatchFullPath
}

function Apply-PatchIfNeeded {
  param(
    [string]$Name,
    [string]$PatchFullPath,
    [scriptblock]$AlreadyApplied
  )

  if (& $AlreadyApplied) {
    Write-Host "$Name ja parece aplicado." -ForegroundColor Green
    return
  }

  if ($ValidateOnly) {
    throw "$Name ainda NAO parece aplicado."
  }

  Test-PatchApplies -PatchFullPath $PatchFullPath

  if ($CheckOnly) {
    Write-Host "$Name: patch valido e aplicavel." -ForegroundColor Green
    return
  }

  Write-Host "Aplicando $Name: $PatchFullPath"
  Invoke-Git apply $PatchFullPath

  if (-not (& $AlreadyApplied)) {
    throw "$Name aplicado, mas a validacao semantica nao encontrou os marcadores esperados. Revise git diff."
  }

  Write-Host "$Name aplicado com sucesso." -ForegroundColor Green
}

Assert-Repo
$PatchFullPath = Resolve-RepoPath -Base $RepoDir -PathValue $PatchPath
$ExtraPatchFullPath = Resolve-RepoPath -Base $RepoDir -PathValue $ExtraPatchPath

Push-Location $RepoDir
try {
  Assert-NotMain

  $needsCore = -not (Test-CoreTenantFixApplied)
  $needsHardening = -not (Test-TenantHardeningApplied)

  if (-not $needsCore -and -not $needsHardening) {
    Write-Host "Tenant fix + hardening ja parecem aplicados. Nenhuma alteracao necessaria." -ForegroundColor Green
    exit 0
  }

  if ($CheckOnly) {
    if ($needsCore) {
      Test-PatchApplies -PatchFullPath $PatchFullPath
      Write-Host "tenant core fix: patch valido e aplicavel." -ForegroundColor Green
      if ($needsHardening) {
        Write-Host "tenant hardening depende do core fix; rode tenant-apply ou reexecute tenant-check depois do core." -ForegroundColor Yellow
      }
      exit 0
    }
    if ($needsHardening) {
      Test-PatchApplies -PatchFullPath $ExtraPatchFullPath
      Write-Host "tenant hardening: patch valido e aplicavel." -ForegroundColor Green
      exit 0
    }
  }

  if ($ValidateOnly) {
    if ($needsCore) { throw "tenant core fix ainda NAO parece aplicado." }
    if ($needsHardening) { throw "tenant hardening ainda NAO parece aplicado." }
  }

  Assert-CleanTree

  Apply-PatchIfNeeded -Name "tenant core fix" -PatchFullPath $PatchFullPath -AlreadyApplied { Test-CoreTenantFixApplied }
  Apply-PatchIfNeeded -Name "tenant hardening" -PatchFullPath $ExtraPatchFullPath -AlreadyApplied { Test-TenantHardeningApplied }

  Write-Host "Tenant fix/hardening prontos." -ForegroundColor Green
  Write-Host "Proximos passos sugeridos:"
  Write-Host "  make -f Makefile.win typecheck-ts"
  Write-Host "  make -f Makefile.win test-ts"
  Write-Host "  git add src/typescript/mcp-server/src/tools"
  Write-Host "  git commit -m \"fix(mcp/rules): scope rule mutations by tenant\""
} finally {
  Pop-Location
}
