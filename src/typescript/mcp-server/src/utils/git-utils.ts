/**
 * Git utility functions for repository detection and remote URL parsing.
 *
 * Behavior:
 * - `.git` is accepted both as a directory (main repo) and as a file
 *   (linked worktree — contains `gitdir: <path>` pointing to the
 *   worktree's git dir under the main repo's `.git/worktrees/<n>/`).
 * - Remote URL resolution shells out to `git config --get` so worktrees
 *   resolve via the shared common config without us having to follow
 *   the `gitdir:` indirection manually.
 */

import { execFileSync } from 'node:child_process';
import { existsSync, statSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';

/**
 * Check if a path is a git repository.
 *
 * A repo root is identified by the presence of a `.git` entry, which may
 * be a directory (main worktree) or a file (linked worktree). Both forms
 * are valid — we do not require it to be a directory.
 */
export function isGitRepository(path: string): boolean {
  try {
    return existsSync(join(path, '.git'));
  } catch {
    return false;
  }
}

/**
 * Find git repository root by walking up from a path.
 * Returns the path containing .git (file or directory), or null if not found.
 */
export function findGitRoot(startPath: string): string | null {
  let currentPath = resolve(startPath);
  const MAX_DEPTH = 20;

  for (let i = 0; i < MAX_DEPTH; i++) {
    if (existsSync(join(currentPath, '.git'))) {
      return currentPath;
    }
    const parent = dirname(currentPath);
    if (parent === currentPath) break;
    currentPath = parent;
  }
  return null;
}

/**
 * Get the git remote URL for a repository.
 *
 * Uses `git config --get remote.origin.url` so linked worktrees resolve
 * via the shared `.git` indirection automatically. Returns `null` when:
 * - `git` is not on PATH;
 * - the path is not a repo;
 * - no `remote.origin` is configured;
 * - the git command times out or fails for any other reason.
 *
 * This is for informational purposes only — the daemon computes the
 * authoritative project_id.
 */
export function getGitRemoteUrl(repoPath: string): string | null {
  return runGit(repoPath, ['config', '--get', 'remote.origin.url']);
}

/**
 * Detect whether a repo root is a linked worktree.
 *
 * In a linked worktree, `.git` is a regular file whose contents are
 * `gitdir: <path>` pointing to the worktree's git dir under the main
 * repo's `.git/worktrees/<n>/`. In a main worktree (or a non-worktree
 * clone), `.git` is a directory.
 *
 * Returns `false` when `.git` is missing.
 */
export function isWorktree(repoRoot: string): boolean {
  try {
    return statSync(join(repoRoot, '.git')).isFile();
  } catch {
    return false;
  }
}

/**
 * Resolve the shared git directory (the `--git-common-dir`).
 *
 * For a main worktree, this is `<repo>/.git`. For a linked worktree,
 * it's the main repo's `.git`, reached by following the `gitdir:`
 * pointer in the worktree's `.git` file and then walking up.
 *
 * Returns `null` when the path is not a repo or `git` is unavailable.
 */
export function getGitCommonDir(repoRoot: string): string | null {
  const out = runGit(repoRoot, ['rev-parse', '--git-common-dir']);
  if (out === null) return null;
  // `git --git-common-dir` may return a relative path; resolve against repoRoot.
  return resolve(repoRoot, out);
}

/**
 * Current branch name reported by `git rev-parse --abbrev-ref HEAD`.
 *
 * Returns `"HEAD"` in detached-HEAD state (mirroring git's own behavior).
 * Returns `null` when `git` is unavailable or the path is not a repo.
 */
export function getCurrentBranch(repoRoot: string): string | null {
  return runGit(repoRoot, ['rev-parse', '--abbrev-ref', 'HEAD']);
}

/**
 * SHA of the current HEAD commit.
 *
 * Returns `null` on empty repos (no commits yet) or when `git` is
 * unavailable / the path is not a repo.
 */
export function getHeadCommit(repoRoot: string): string | null {
  return runGit(repoRoot, ['rev-parse', 'HEAD']);
}

/**
 * Aggregate git state for a repo root.
 *
 * Combines the primitives above into a single object suitable for passing
 * over the wire (gRPC, MCP). All fields are best-effort: any individual
 * field may be `null` if its underlying git command fails, without
 * affecting the others. Returns `null` only when `repoRoot` is not a git
 * repo at all.
 */
export interface GitState {
  readonly repoRoot: string;
  readonly branch: string | null;
  readonly commit: string | null;
  readonly remoteUrl: string | null;
  readonly isWorktree: boolean;
  readonly worktreePath: string | null;
  readonly commonDir: string | null;
}

export function getGitState(repoRoot: string): GitState | null {
  if (!isGitRepository(repoRoot)) return null;
  const worktree = isWorktree(repoRoot);
  return {
    repoRoot,
    branch: getCurrentBranch(repoRoot),
    commit: getHeadCommit(repoRoot),
    remoteUrl: getGitRemoteUrl(repoRoot),
    isWorktree: worktree,
    worktreePath: worktree ? repoRoot : null,
    commonDir: getGitCommonDir(repoRoot),
  };
}

/**
 * Run a `git` command in the given repo, returning trimmed stdout or
 * `null` on any failure. Internal helper used by the typed accessors
 * above.
 */
function runGit(repoRoot: string, args: ReadonlyArray<string>): string | null {
  try {
    const out = execFileSync('git', ['-C', repoRoot, ...args], {
      encoding: 'utf-8',
      stdio: ['ignore', 'pipe', 'ignore'],
      timeout: 2000,
      windowsHide: true,
    });
    const trimmed = out.trim();
    return trimmed.length > 0 ? trimmed : null;
  } catch {
    return null;
  }
}
