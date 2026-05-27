/**
 * Project discovery for the admin UI.
 *
 * Walks the configured `devRoot` looking for git repositories (either main
 * working trees, with `.git/` as a directory, or linked worktrees with
 * `.git` as a file). Each match is returned as a candidate the user can
 * approve from the UI. We do NOT call the daemon during discovery —
 * candidates are just metadata until the user clicks register.
 */

import { existsSync, readdirSync, statSync } from 'node:fs';
import { join, resolve } from 'node:path';

import { getCurrentBranch, getGitRemoteUrl, isWorktree } from '../utils/git-utils.js';

export interface ProjectCandidate {
  /** Absolute path of the candidate repo (canonical separators preserved). */
  path: string;
  /** Display name = basename of the path. */
  name: string;
  /** True when `.git` at this path is a file (linked worktree). */
  isWorktree: boolean;
  /** Current branch reported by git, or `null` when unavailable. */
  branch: string | null;
  /** Configured `remote.origin.url`, or empty string. */
  remoteUrl: string;
  /** Depth at which this candidate was found (1 = direct child of devRoot). */
  depth: number;
}

export interface ScanResult {
  /** Absolute path that was scanned. */
  root: string;
  /** Maximum walk depth applied. */
  maxDepth: number;
  /** Number of directories visited (for diagnostics). */
  visited: number;
  /** Number of directories skipped because they were too deep or excluded. */
  skipped: number;
  /** Candidates found, ordered by path. */
  candidates: ProjectCandidate[];
  /** ISO 8601 timestamp the scan finished. */
  finishedAt: string;
}

/**
 * Directories that are never worth descending into. Match against the
 * basename; case-sensitive. Mirrors the daemon's `IGNORED_DIRS` for
 * consistency.
 */
const SKIP_DIRS = new Set<string>([
  'node_modules',
  'target',
  'dist',
  'build',
  '.next',
  '.nuxt',
  '.svelte-kit',
  '.cache',
  '.venv',
  'venv',
  '__pycache__',
  '.pytest_cache',
  '.mypy_cache',
  '.tox',
  '.eggs',
  '.idea',
  '.vscode',
  '.DS_Store',
  'coverage',
  '.git',
]);

function readDirSafe(path: string): string[] {
  try {
    return readdirSync(path);
  } catch {
    return [];
  }
}

function isDirSafe(path: string): boolean {
  try {
    return statSync(path).isDirectory();
  } catch {
    return false;
  }
}

/**
 * Whether `path` is a git repository root (main or linked worktree).
 * `.git` may be a directory (main) or a file (worktree) — both qualify.
 */
function isGitRepoRoot(path: string): boolean {
  return existsSync(join(path, '.git'));
}

/**
 * Collect candidate info for a confirmed git repo path.
 * Best-effort: any failed git call returns null/empty without throwing.
 */
function describeCandidate(path: string, depth: number): ProjectCandidate {
  return {
    path: path.replace(/\\/g, '/'),
    name: path.split(/[\\/]/).pop() ?? path,
    isWorktree: isWorktree(path),
    branch: getCurrentBranch(path),
    remoteUrl: getGitRemoteUrl(path) ?? '',
    depth,
  };
}

interface WalkAccumulator {
  visited: number;
  skipped: number;
  candidates: ProjectCandidate[];
}

function walk(path: string, currentDepth: number, maxDepth: number, acc: WalkAccumulator): void {
  acc.visited += 1;

  if (isGitRepoRoot(path)) {
    acc.candidates.push(describeCandidate(path, currentDepth));
    // Found a repo — don't descend further. Sub-worktrees / submodules
    // are surfaced by the daemon's own discovery (see daemon spec 06),
    // not by this scan.
    return;
  }

  if (currentDepth >= maxDepth) {
    acc.skipped += 1;
    return;
  }

  for (const entry of readDirSafe(path)) {
    if (SKIP_DIRS.has(entry)) continue;
    const child = join(path, entry);
    if (!isDirSafe(child)) continue;
    walk(child, currentDepth + 1, maxDepth, acc);
  }
}

/**
 * Scan `devRoot` for git repositories up to `maxDepth` levels deep.
 *
 * `maxDepth` is clamped to [1, 5] — depth 1 = direct children only,
 * depth 5 already visits a lot of directories on a typical dev tree.
 * Going deeper without a more targeted strategy invites surprises.
 */
export function scanForGitProjects(devRoot: string, maxDepth = 1): ScanResult {
  const finishedAt = () => new Date().toISOString();
  if (!devRoot || !existsSync(devRoot) || !isDirSafe(devRoot)) {
    return {
      root: devRoot,
      maxDepth,
      visited: 0,
      skipped: 0,
      candidates: [],
      finishedAt: finishedAt(),
    };
  }

  const clampedDepth = Math.max(1, Math.min(5, maxDepth));
  const absoluteRoot = resolve(devRoot);
  const acc: WalkAccumulator = { visited: 0, skipped: 0, candidates: [] };

  walk(absoluteRoot, 0, clampedDepth, acc);
  acc.candidates.sort((a, b) => a.path.localeCompare(b.path));

  return {
    root: absoluteRoot.replace(/\\/g, '/'),
    maxDepth: clampedDepth,
    visited: acc.visited,
    skipped: acc.skipped,
    candidates: acc.candidates,
    finishedAt: finishedAt(),
  };
}
