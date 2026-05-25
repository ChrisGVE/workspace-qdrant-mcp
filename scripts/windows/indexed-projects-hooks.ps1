[CmdletBinding()]
param(
  [ValidateSet("install","status")]
  [string]$Action = "install",
  [string]$RepoDir = (Get-Location).Path
)

$ErrorActionPreference = "Stop"

function Resolve-AbsolutePath([string]$BaseDir, [string]$PathValue) {
  if (-not $PathValue) { return $PathValue }

  if ([System.IO.Path]::IsPathRooted($PathValue)) {
    return [System.IO.Path]::GetFullPath($PathValue)
  }

  return [System.IO.Path]::GetFullPath((Join-Path $BaseDir $PathValue))
}

function Ensure-Dir([string]$PathValue) {
  if (-not (Test-Path -LiteralPath $PathValue)) {
    New-Item -ItemType Directory -Path $PathValue -Force | Out-Null
  }
}

function Invoke-GitText([string]$Repo, [string[]]$Args) {
  $out = & git -C $Repo @Args 2>&1
  [pscustomobject]@{
    code = $LASTEXITCODE
    ok = ($LASTEXITCODE -eq 0)
    output = (@($out) -join "`n")
  }
}

function Test-GitMarker([string]$RepoAbs) {
  return (Test-Path -LiteralPath (Join-Path $RepoAbs '.git'))
}

function Get-CommonGitDirFromMarker([string]$RepoAbs) {
  $gitMarker = Join-Path $RepoAbs '.git'
  if (-not (Test-Path -LiteralPath $gitMarker)) {
    return $null
  }

  $item = Get-Item -LiteralPath $gitMarker -Force
  if ($item.PSIsContainer) {
    return [System.IO.Path]::GetFullPath($gitMarker)
  }

  $raw = Get-Content -LiteralPath $gitMarker -Raw
  if (-not $raw) {
    return $null
  }

  if ($raw -notmatch '^gitdir:\s*(.+)$') {
    return $null
  }

  $gitDir = Resolve-AbsolutePath $RepoAbs $Matches[1].Trim()
  return Split-Path -Parent (Split-Path -Parent $gitDir)
}

function Write-TextFile([string]$PathValue, [string]$Content) {
  $encoding = [System.Text.UTF8Encoding]::new($false)
  [System.IO.File]::WriteAllText($PathValue, $Content, $encoding)
}

function Get-TopLevelRoot([string]$RepoDirValue) {
  $repoAbs = Resolve-AbsolutePath (Get-Location).Path $RepoDirValue
  if (Test-GitMarker $repoAbs) {
    return $repoAbs
  }

  $candidates = @($repoAbs, (Get-Location).Path) | Select-Object -Unique

  foreach ($candidate in $candidates) {
    $result = Invoke-GitText $candidate @('rev-parse','--show-toplevel')
    if ($result.ok -and $result.output.Trim()) {
      return [System.IO.Path]::GetFullPath($result.output.Trim())
    }
  }

  throw "Nao foi possivel resolver o root do repo: $RepoDirValue"
}

function Get-CommonGitDir([string]$RepoRoot) {
  $markerDir = Get-CommonGitDirFromMarker $RepoRoot
  if ($markerDir) {
    return $markerDir
  }

  $candidates = @($RepoRoot, (Get-Location).Path) | Select-Object -Unique

  foreach ($candidate in $candidates) {
    $result = Invoke-GitText $candidate @('rev-parse','--git-common-dir')
    if ($result.ok -and $result.output.Trim()) {
      return Resolve-AbsolutePath $candidate $result.output.Trim()
    }
  }

  throw "Nao foi possivel resolver o git-common-dir de: $RepoRoot"
}

function Get-SharedRoot([string]$RepoRoot) {
  return Split-Path -Parent (Get-CommonGitDir $RepoRoot)
}

function Write-HookRunner([string]$PathValue, [string]$RegistryScriptSource) {
  $content = @'
[CmdletBinding()]
param(
  [string]$HookName = "",
  [string]$RepoDir = (Get-Location).Path
)

$ErrorActionPreference = "Stop"

function Ensure-Dir([string]$PathValue) {
  if (-not (Test-Path -LiteralPath $PathValue)) {
    New-Item -ItemType Directory -Path $PathValue -Force | Out-Null
  }
}

function Resolve-RepoRoot([string]$RepoDirValue) {
  $repoAbs = if ([System.IO.Path]::IsPathRooted($RepoDirValue)) { [System.IO.Path]::GetFullPath($RepoDirValue) } else { [System.IO.Path]::GetFullPath((Join-Path (Get-Location).Path $RepoDirValue)) }
  if (Test-Path -LiteralPath (Join-Path $repoAbs '.git')) {
    return $repoAbs
  }
  $result = & git -C $repoAbs rev-parse --show-toplevel 2>&1
  if ($LASTEXITCODE -ne 0) { return $null }
  $root = (@($result) -join "`n").Trim()
  if (-not $root) { return $null }
  return [System.IO.Path]::GetFullPath($root)
}

function Write-Log([string]$LogFile, [object]$Value) {
  Ensure-Dir (Split-Path -Parent $LogFile)
  $line = if ($Value -is [string]) { $Value } else { $Value | ConvertTo-Json -Depth 20 -Compress }
  Add-Content -LiteralPath $LogFile -Value $line -Encoding UTF8
}

try {
  $repoRoot = Resolve-RepoRoot $RepoDir
  if (-not $repoRoot) { exit 0 }

  $scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
  $sharedRoot = Split-Path -Parent (Split-Path -Parent $scriptDir)
  $registryPath = Join-Path $sharedRoot '.wqm-fork\indexed-projects.json'
  $logFile = Join-Path $sharedRoot '.wqm-fork\logs\git-hooks.jsonl'
  $registryScript = "__REGISTRY_SCRIPT__"

  if (-not (Test-Path -LiteralPath $registryScript)) {
    exit 0
  }

  $payload = & $registryScript -Action sync-current-branch -RepoDir $repoRoot -RegistryPath $registryPath -LogDir (Join-Path $sharedRoot '.wqm-fork\logs') -Mutate true 2>&1
  $payloadText = (@($payload) -join "`n").Trim()
  if ($payloadText) {
    $decoded = $payloadText | ConvertFrom-Json -ErrorAction SilentlyContinue
    if ($decoded) {
      Write-Log $logFile ([ordered]@{
        timestamp = (Get-Date).ToUniversalTime().ToString("o")
        hook = $HookName
        repo = $repoRoot
        result = $decoded
      })
    } else {
      Write-Log $logFile ([ordered]@{
        timestamp = (Get-Date).ToUniversalTime().ToString("o")
        hook = $HookName
        repo = $repoRoot
        result = $payloadText
      })
    }
  }
} catch {
  try {
    $fallbackRepo = if ($repoRoot) { $repoRoot } else { $RepoDir }
    $sharedRoot = if ($scriptDir) { Split-Path -Parent (Split-Path -Parent $scriptDir) } else { $fallbackRepo }
    $logFile = Join-Path $sharedRoot '.wqm-fork\logs\git-hooks.jsonl'
    Write-Log $logFile ([ordered]@{
      timestamp = (Get-Date).ToUniversalTime().ToString("o")
      hook = $HookName
      repo = $fallbackRepo
      error = $_.Exception.Message
    })
  } catch { }
} finally {
  exit 0
}
'@
  $content = $content.Replace("__REGISTRY_SCRIPT__", $RegistryScriptSource)
  Write-TextFile $PathValue $content
}

function Write-HookScript([string]$PathValue, [string]$HookName, [bool]$CheckoutOnly) {
  $checkoutBlock = if ($CheckoutOnly) {
    @'
if [ "${3:-}" = "0" ]; then
  exit 0
fi
'@
  } else {
    ''
  }

  $content = @'
#!/bin/sh
set -eu
__CHECKOUT_BLOCK__
if ! command -v powershell.exe >/dev/null 2>&1; then
  exit 0
fi
HOOK_PATH="$0"
HOOK_PATH="${HOOK_PATH//\\//}"
case "$HOOK_PATH" in
  */*) HOOK_DIR_WIN="${HOOK_PATH%/*}" ;;
  *) HOOK_DIR_WIN="$(pwd -W 2>/dev/null || true)" ;;
esac
if [ -z "$HOOK_DIR_WIN" ]; then
  exit 0
fi
if REPO_DIR_WIN="$(pwd -W 2>/dev/null)"; then
  :
else
  exit 0
fi
HOOK_SCRIPT_WIN="$HOOK_DIR_WIN\\wqm-git-hook.ps1"
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "$HOOK_SCRIPT_WIN" -HookName "__HOOK_NAME__" -RepoDir "$REPO_DIR_WIN" >/dev/null 2>&1 || true
exit 0
'@.Replace("__HOOK_NAME__", $HookName).Replace("__CHECKOUT_BLOCK__", $checkoutBlock)

  Write-TextFile $PathValue $content
}

function Get-HookStatus([string]$RepoRoot, [string]$SharedRoot, [string]$HooksDir, [string]$CommonConfig) {
  $configured = $null
  if (Test-Path -LiteralPath $CommonConfig) {
    try {
      $rawConfig = Get-Content -LiteralPath $CommonConfig -Raw
      if ($rawConfig -match '(?im)^\s*hooksPath\s*=\s*(.+?)\s*$') {
        $configured = [System.IO.Path]::GetFullPath((($Matches[1].Trim()) -replace '\\\\', '\'))
      } else {
        $result = Invoke-GitText $RepoRoot @('config','--file',$CommonConfig,'--get','core.hooksPath')
        if ($result.ok) {
          $configured = [System.IO.Path]::GetFullPath((($result.output.Trim()) -replace '\\\\', '\'))
        }
      }
    } catch {
      $result = Invoke-GitText $RepoRoot @('config','--file',$CommonConfig,'--get','core.hooksPath')
      if ($result.ok) {
        $configured = [System.IO.Path]::GetFullPath((($result.output.Trim()) -replace '\\\\', '\'))
      }
    }
  }

  $hookNames = @('post-checkout','post-commit','post-merge','post-rewrite','post-worktree-add')

  [pscustomobject]@{
    repoRoot = $RepoRoot
    sharedRoot = $SharedRoot
    hooksDir = $HooksDir
    configuredHooksPath = $configured
    hooksPathMatches = ($configured -eq $HooksDir)
    runnerExists = (Test-Path -LiteralPath (Join-Path $HooksDir 'wqm-git-hook.ps1'))
    hookCount = @($hookNames | Where-Object { Test-Path -LiteralPath (Join-Path $HooksDir $_) }).Count
  }
}

$repoRoot = Get-TopLevelRoot $RepoDir
$sharedRoot = Get-SharedRoot $repoRoot
$hooksDir = Join-Path $sharedRoot '.wqm-fork\git-hooks'
$commonGitDir = Get-CommonGitDir $repoRoot
$commonConfig = Join-Path $commonGitDir 'config'
$registryScript = Join-Path $repoRoot 'scripts\windows\indexed-projects-registry.ps1'
$registryPath = Join-Path $sharedRoot '.wqm-fork\indexed-projects.json'
$installerScriptDir = Split-Path -Parent $PSCommandPath
$registryScriptSource = Join-Path $installerScriptDir 'indexed-projects-registry.ps1'

switch ($Action) {
  'install' {
    Ensure-Dir $hooksDir
    Ensure-Dir (Join-Path $sharedRoot '.wqm-fork\logs')
    Write-HookRunner (Join-Path $hooksDir 'wqm-git-hook.ps1') $registryScriptSource
    Write-HookScript (Join-Path $hooksDir 'post-checkout') 'post-checkout' $true
    Write-HookScript (Join-Path $hooksDir 'post-commit') 'post-commit' $false
    Write-HookScript (Join-Path $hooksDir 'post-merge') 'post-merge' $false
    Write-HookScript (Join-Path $hooksDir 'post-rewrite') 'post-rewrite' $false
    Write-HookScript (Join-Path $hooksDir 'post-worktree-add') 'post-worktree-add' $false
    & git config --file $commonConfig core.hooksPath $hooksDir
    if ($LASTEXITCODE -ne 0) {
      throw "Nao foi possivel configurar core.hooksPath para $hooksDir"
    }

    if (Test-Path -LiteralPath $registryScript) {
      $null = & $registryScript -Action sync-current-branch -RepoDir $repoRoot -RegistryPath $registryPath -LogDir (Join-Path $sharedRoot '.wqm-fork\logs') -Mutate true 2>&1
    }

    Get-HookStatus $repoRoot $sharedRoot $hooksDir $commonConfig | ConvertTo-Json -Depth 12
  }
  'status' {
    Get-HookStatus $repoRoot $sharedRoot $hooksDir $commonConfig | ConvertTo-Json -Depth 12
  }
  default {
    throw "Acao nao implementada: $Action"
  }
}
