# path-resolver.ps1 — PowerShell git/path detection helpers.
#
# Dot-source from any script that needs path or git state:
#
#   $libDir = Join-Path $PSScriptRoot 'lib'
#   . (Join-Path $libDir 'path-resolver.ps1')
#
# Mirrors scripts/lib/path-resolver.sh and src/typescript/mcp-server/src/utils/git-utils.ts
# so all three implementations agree on the same rules.
#
# Conventions:
#   - Every function returns `$null` (or empty string for textual fields)
#     when it can't resolve a value. No exceptions for "not a repo" etc.
#   - Git-specific functions always shell out to `git` so worktrees and
#     other edge cases are handled by git itself.
#   - Path functions are pure string manipulation — no fs access except
#     when absolutely necessary (Test-Path on a single literal).

$script:WqmPathResolverVersion = '1.0'

# ── Tool availability ────────────────────────────────────────────────────────

function Test-WqmTool {
  param([Parameter(Mandatory)][string]$Name)
  return [bool](Get-Command -Name $Name -ErrorAction SilentlyContinue)
}

# ── Path utilities ───────────────────────────────────────────────────────────

function ConvertTo-WqmAbsolutePath {
  param(
    [Parameter(Mandatory)][string]$Path,
    [string]$BaseDir
  )
  if (-not $Path) { return $null }

  if ([System.IO.Path]::IsPathRooted($Path)) {
    return [System.IO.Path]::GetFullPath($Path)
  }

  if (-not $BaseDir) { $BaseDir = (Get-Location).Path }
  return [System.IO.Path]::GetFullPath((Join-Path $BaseDir $Path))
}

function ConvertTo-WqmPosixPath {
  # Replace backslashes with forward slashes; leave drive letter intact.
  # Example: C:\Users\alber\dev -> C:/Users/alber/dev
  param([Parameter(Mandatory)][string]$Path)
  return ($Path -replace '\\', '/')
}

function ConvertTo-WqmWslPath {
  # Translate a Windows absolute path to WSL-style mount path.
  # Example: C:\Users\alber\dev -> /mnt/c/Users/alber/dev
  # Returns the input unchanged if it's already POSIX or /mnt/ form.
  param([Parameter(Mandatory)][string]$Path)
  if (-not $Path) { return $Path }

  $absolute = ConvertTo-WqmAbsolutePath -Path $Path
  $normalized = ConvertTo-WqmPosixPath -Path $absolute

  if ($normalized -match '^/mnt/[a-z]/') { return $normalized }

  if ($normalized -match '^([A-Za-z]):/(.*)$') {
    $drive = $Matches[1].ToLower()
    $rest  = $Matches[2].TrimStart('/')
    return "/mnt/$drive/$rest"
  }

  return $normalized
}

function Test-WqmPathIsWindowsAbsolute {
  param([Parameter(Mandatory)][string]$Path)
  return ($Path -match '^[A-Za-z]:[\\/]')
}

function Test-WqmPathIsAbsolute {
  param([Parameter(Mandatory)][string]$Path)
  if ($Path.StartsWith('/')) { return $true }
  return (Test-WqmPathIsWindowsAbsolute -Path $Path)
}

# ── Git repository detection ─────────────────────────────────────────────────

function Test-WqmGitMarker {
  # True if the directory has a .git entry (file or directory).
  param([Parameter(Mandatory)][string]$RepoRoot)
  return (Test-Path -LiteralPath (Join-Path $RepoRoot '.git'))
}

function Get-WqmGitRepoRoot {
  # Resolve the repo root via `git rev-parse --show-toplevel`.
  # Returns $null when not in a repo or git is unavailable.
  param([string]$Path = (Get-Location).Path)
  if (-not (Test-WqmTool 'git')) { return $null }

  $result = & git -C $Path rev-parse --show-toplevel 2>$null
  if ($LASTEXITCODE -ne 0 -or -not $result) { return $null }
  return [System.IO.Path]::GetFullPath(($result | Select-Object -First 1).Trim())
}

function Test-WqmGitIsWorktree {
  # True if .git in RepoRoot is a file (linked worktree).
  param([Parameter(Mandatory)][string]$RepoRoot)
  $gitMarker = Join-Path $RepoRoot '.git'
  if (-not (Test-Path -LiteralPath $gitMarker)) { return $false }
  $item = Get-Item -LiteralPath $gitMarker -Force
  return -not $item.PSIsContainer
}

function Get-WqmGitCommonDir {
  # Shared git directory. For a main repo: <root>/.git. For a worktree:
  # the parent repo's .git, resolved via git itself.
  param([Parameter(Mandatory)][string]$RepoRoot)
  if (-not (Test-WqmTool 'git')) { return $null }

  $result = & git -C $RepoRoot rev-parse --git-common-dir 2>$null
  if ($LASTEXITCODE -ne 0 -or -not $result) { return $null }

  $value = ($result | Select-Object -First 1).Trim()
  if ([System.IO.Path]::IsPathRooted($value)) {
    return [System.IO.Path]::GetFullPath($value)
  }
  return [System.IO.Path]::GetFullPath((Join-Path $RepoRoot $value))
}

function Get-WqmGitCurrentBranch {
  # Branch name, or "HEAD" in detached state. Empty string if not a repo.
  param([Parameter(Mandatory)][string]$RepoRoot)
  if (-not (Test-WqmTool 'git')) { return '' }

  $result = & git -C $RepoRoot rev-parse --abbrev-ref HEAD 2>$null
  if ($LASTEXITCODE -ne 0 -or -not $result) { return '' }
  return ($result | Select-Object -First 1).Trim()
}

function Get-WqmGitHeadCommit {
  # HEAD commit SHA, or empty for empty repos / non-repos.
  param([Parameter(Mandatory)][string]$RepoRoot)
  if (-not (Test-WqmTool 'git')) { return '' }

  $result = & git -C $RepoRoot rev-parse HEAD 2>$null
  if ($LASTEXITCODE -ne 0 -or -not $result) { return '' }
  return ($result | Select-Object -First 1).Trim()
}

function Get-WqmGitRemoteUrl {
  # remote.origin.url, or empty when not configured.
  param([Parameter(Mandatory)][string]$RepoRoot)
  if (-not (Test-WqmTool 'git')) { return '' }

  $result = & git -C $RepoRoot config --get remote.origin.url 2>$null
  if ($LASTEXITCODE -ne 0 -or -not $result) { return '' }
  return ($result | Select-Object -First 1).Trim()
}

function Get-WqmGitState {
  # Aggregate object with branch, commit, remote, worktree, common dir.
  # Returns $null when RepoRoot is not a git repo.
  param([Parameter(Mandatory)][string]$RepoRoot)

  if (-not (Test-WqmGitMarker -RepoRoot $RepoRoot)) { return $null }

  $isWorktree = Test-WqmGitIsWorktree -RepoRoot $RepoRoot

  return [pscustomobject]@{
    repoRoot     = $RepoRoot
    branch       = Get-WqmGitCurrentBranch -RepoRoot $RepoRoot
    commit       = Get-WqmGitHeadCommit -RepoRoot $RepoRoot
    remoteUrl    = Get-WqmGitRemoteUrl -RepoRoot $RepoRoot
    isWorktree   = $isWorktree
    worktreePath = $(if ($isWorktree) { $RepoRoot } else { '' })
    commonDir    = Get-WqmGitCommonDir -RepoRoot $RepoRoot
  }
}
