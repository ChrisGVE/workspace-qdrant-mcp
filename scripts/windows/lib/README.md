# scripts/windows/lib

Shared PowerShell helpers for path and git state detection. Mirrors the
sister implementations in `scripts/lib/` (POSIX sh) and
`src/typescript/mcp-server/src/utils/git-utils.ts` (TS) so every entry
point — Windows host scripts, POSIX hooks, dockerized MCP — agrees on
the same rules for what a git repo is, what a worktree is, and how a
path should be normalized.

## Usage

```powershell
$libDir = Join-Path $PSScriptRoot 'lib'
. (Join-Path $libDir 'path-resolver.ps1')

# Use any helper:
$root = Get-WqmGitRepoRoot
$state = Get-WqmGitState -RepoRoot $root
```

For scripts in `scripts/windows/*.ps1`, the lib path is
`Join-Path $PSScriptRoot 'lib\path-resolver.ps1'`.

## What's available

### Tool detection
- `Test-WqmTool <name>` — `git` on PATH?

### Path utilities
- `ConvertTo-WqmAbsolutePath -Path <p> [-BaseDir <d>]` — resolve relative input.
- `ConvertTo-WqmPosixPath -Path <p>` — `C:\foo` → `C:/foo`.
- `ConvertTo-WqmWslPath -Path <p>` — `C:\foo` → `/mnt/c/foo`.
- `Test-WqmPathIsWindowsAbsolute -Path <p>`
- `Test-WqmPathIsAbsolute -Path <p>` — POSIX or Windows absolute.

### Git detection
- `Test-WqmGitMarker -RepoRoot <r>` — has a `.git` entry (file or dir).
- `Get-WqmGitRepoRoot [-Path <p>]` — `git rev-parse --show-toplevel`.
- `Test-WqmGitIsWorktree -RepoRoot <r>` — true if `.git` is a file.
- `Get-WqmGitCommonDir -RepoRoot <r>` — shared git dir (worktree-aware).
- `Get-WqmGitCurrentBranch -RepoRoot <r>` — branch name or `HEAD`.
- `Get-WqmGitHeadCommit -RepoRoot <r>` — HEAD SHA.
- `Get-WqmGitRemoteUrl -RepoRoot <r>` — `remote.origin.url`.
- `Get-WqmGitState -RepoRoot <r>` — `[pscustomobject]` with all of the above.

## Migration policy

Existing scripts (`indexed-projects-hooks.ps1`,
`indexed-projects-registry.ps1`, `register-project.ps1`,
`smoke-test.ps1`, `workspace-registry.ps1`) keep their inline helpers
for now. **New** scripts must dot-source this lib instead of copying
helpers. Migrating existing scripts is opt-in, file by file, when other
work touches them.

## Cross-language parity

Functions in this file are 1:1 with:

| PowerShell                       | POSIX sh                       | TypeScript                        |
|----------------------------------|--------------------------------|-----------------------------------|
| `Get-WqmGitRepoRoot`             | `wqm_git_repo_root`            | `findGitRoot`                     |
| `Test-WqmGitMarker`              | (inline `[ -e .git ]`)         | `isGitRepository`                 |
| `Test-WqmGitIsWorktree`          | `wqm_git_is_worktree`          | `isWorktree`                      |
| `Get-WqmGitCommonDir`            | `wqm_git_common_dir`           | `getGitCommonDir`                 |
| `Get-WqmGitCurrentBranch`        | `wqm_git_current_branch`       | `getCurrentBranch`                |
| `Get-WqmGitHeadCommit`           | `wqm_git_head_commit`          | `getHeadCommit`                   |
| `Get-WqmGitRemoteUrl`            | `wqm_git_remote_url`           | `getGitRemoteUrl`                 |
| `Get-WqmGitState`                | `wqm_git_state` (JSON)         | `getGitState`                     |
| `ConvertTo-WqmPosixPath`         | `wqm_path_normalize_slashes`   | (built into `normalizePath`)      |

When you change the contract on one side, change it on the other two.
The test fixtures in `tests/path-fixtures/` exist to catch drift.
