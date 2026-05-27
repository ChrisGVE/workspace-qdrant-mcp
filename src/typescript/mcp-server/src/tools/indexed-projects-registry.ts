/**
 * indexed-projects-registry.ts — TypeScript port of the Phase 1 surface of
 * `scripts/windows/indexed-projects-registry.ps1`.
 *
 * Implements the registry actions that don't require the `wqm` CLI binary,
 * so the dockerized MCP container (Linux, no PowerShell, no wqm.exe) can
 * serve them. Actions that integrate with `wqm` (observe, incremental-check,
 * register-wqm) stay PowerShell-only for now and fall back to spawn(pwsh).
 *
 * Schema is byte-compatible with the PS1 implementation: same field names,
 * same enum values for `kind` / `status`, same `[ordered]` insertion order in
 * the serialized JSON.
 */

import { execFileSync } from 'node:child_process';
import { existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { dirname, isAbsolute, join, resolve } from 'node:path';

import {
  findGitRoot,
  getCurrentBranch,
  getGitRemoteUrl,
  getHeadCommit,
  isGitRepository,
  isWorktree,
} from './../utils/git-utils.js';

// ── Types ───────────────────────────────────────────────────────────────────

export interface RegistryBranch {
  name: string;
  kind: string; // 'manual' | 'manual-worktree' | 'agent' | 'primary'
  path: string;
  baseBranch?: string;
  returnBranch?: string;
  status: string; // 'active' | 'ready_for_review' | 'abandoned'
  createdBy?: string;
  createdAt: string;
  lastSeenAt: string;
  baseCommit?: string | null;
  headCommit?: string | null;
  lastIndexedCommit?: string | null;
  watchEnabled?: boolean;
  indexed?: boolean;
  purpose?: string;
  useWorktree?: boolean;
  note?: string;
}

export interface RegistryProject {
  name: string;
  root: string;
  projectId?: string | null;
  qdrantUrl?: string;
  daemonEndpoint?: string;
  defaultBranch?: string;
  tenantStrategy?: string;
  enabled?: boolean;
  createdAt?: string;
  updatedAt?: string;
  branches: RegistryBranch[];
}

export interface Registry {
  schemaVersion: number;
  kind: string;
  updatedAt: string;
  projects: RegistryProject[];
}

export interface ProjectSelector {
  projectName?: string;
  projectId?: string;
  projectDir?: string;
}

// ── Constants ───────────────────────────────────────────────────────────────

const SCHEMA_VERSION = 2;
const REGISTRY_KIND = 'indexed-projects';

// ── Time + path helpers ─────────────────────────────────────────────────────

function utcNow(): string {
  return new Date().toISOString();
}

/**
 * Translate a host-style path (`C:\Users\alber\...` or `C:/Users/alber/...`)
 * to the bind-mount path visible inside the container (`/run/desktop/...`),
 * using the WQM_HOST_DEV_ROOT → WQM_DEV_ROOT translation declared in
 * `docker/.env`. Returns the original path unchanged when:
 *   - env vars are unset (host-side execution: paths are already native);
 *   - the input doesn't sit under WQM_HOST_DEV_ROOT;
 *   - the input is already in container form.
 */
function translateHostPath(pathValue: string): string {
  if (!pathValue) return pathValue;
  const hostRoot = process.env['WQM_HOST_DEV_ROOT'];
  const devRoot = process.env['WQM_DEV_ROOT'];
  if (!hostRoot || !devRoot) return pathValue;

  // Normalize separators on the input AND the host-root marker so the prefix
  // match works regardless of which slash style the JSON happens to use.
  const inputForward = pathValue.replace(/\\/g, '/');
  const hostForward = hostRoot.replace(/\\/g, '/').replace(/\/+$/, '');

  // Case-insensitive comparison covers `C:` vs `c:` on Windows-origin paths.
  const lowerInput = inputForward.toLowerCase();
  const lowerHost = hostForward.toLowerCase();
  if (lowerInput === lowerHost) return devRoot;
  if (lowerInput.startsWith(lowerHost + '/')) {
    return devRoot + inputForward.slice(hostForward.length);
  }
  return pathValue;
}

function toAbs(pathValue: string): string {
  if (!pathValue) return pathValue;
  // First, translate host→container if we're running inside the docker MCP.
  const translated = translateHostPath(pathValue);
  // Then resolve relative components against cwd. We only call resolve() for
  // already-absolute paths in practice; this also handles posix normalization.
  if (translated.match(/^([A-Za-z]:[\\/]|\/)/)) {
    // Already absolute (POSIX or Windows). Just normalize separators.
    return translated.replace(/\\/g, '/');
  }
  return resolve(translated);
}

export function defaultRegistryPath(repoDir: string): string {
  return resolve(repoDir, '.wqm-fork', 'indexed-projects.json');
}

function ensureDir(dirPath: string): void {
  if (!existsSync(dirPath)) {
    mkdirSync(dirPath, { recursive: true });
  }
}

// ── Branch slug (matches PS Safe-BranchSlug) ────────────────────────────────

export function safeBranchSlug(name: string): string {
  return name.replace(/[^A-Za-z0-9._-]+/g, '-').replace(/^-+|-+$/g, '');
}

// ── Registry IO ─────────────────────────────────────────────────────────────

export function newRegistry(): Registry {
  return {
    schemaVersion: SCHEMA_VERSION,
    kind: REGISTRY_KIND,
    updatedAt: utcNow(),
    projects: [],
  };
}

export function readRegistry(registryPath: string): Registry {
  if (!existsSync(registryPath)) return newRegistry();
  const raw = readFileSync(registryPath, 'utf-8').trim();
  if (!raw) return newRegistry();
  const parsed = JSON.parse(raw) as Registry;
  // Defensive: tolerate missing schema fields from older registries.
  if (!parsed.projects) parsed.projects = [];
  if (!parsed.schemaVersion) parsed.schemaVersion = SCHEMA_VERSION;
  if (!parsed.kind) parsed.kind = REGISTRY_KIND;
  return parsed;
}

export function writeRegistry(registryPath: string, registry: Registry): void {
  registry.updatedAt = utcNow();
  ensureDir(dirname(registryPath));
  writeFileSync(registryPath, JSON.stringify(registry, null, 4) + '\n', 'utf-8');
}

// ── Lookup helpers ──────────────────────────────────────────────────────────

function normalizeProject(p: RegistryProject): RegistryProject {
  return {
    ...p,
    root: toAbs(p.root),
    qdrantUrl: p.qdrantUrl ?? 'http://localhost:6333',
    daemonEndpoint: p.daemonEndpoint ?? 'localhost:50051',
    defaultBranch: p.defaultBranch ?? 'main',
    tenantStrategy: p.tenantStrategy ?? 'project',
    enabled: p.enabled ?? true,
    branches: p.branches ?? [],
  };
}

export function findProject(registry: Registry, sel: ProjectSelector): RegistryProject {
  const projects = registry.projects.map(normalizeProject);
  let candidates = projects;

  if (sel.projectDir) {
    const target = resolveProjectRoot(sel.projectDir);
    const exact = projects.filter((p) => toAbs(p.root) === target);
    if (exact.length === 1) return exact[0] as RegistryProject;
    if (exact.length > 1) {
      throw new Error(`Projeto ambiguo (root match): ${sel.projectDir}`);
    }
    // Fall through to name/id matching if root didn't pin it.
  }

  if (sel.projectName) {
    candidates = candidates.filter((p) => p.name === sel.projectName);
  }
  if (sel.projectId) {
    candidates = candidates.filter((p) => p.projectId === sel.projectId);
  }

  if (!sel.projectName && !sel.projectId && !sel.projectDir) {
    throw new Error('Informe projectName, projectId ou projectDir.');
  }
  if (candidates.length === 0) {
    throw new Error(
      `Projeto indexado nao encontrado: ${sel.projectName ?? ''} ${sel.projectId ?? ''} ${sel.projectDir ?? ''}`.trim()
    );
  }
  if (candidates.length > 1) {
    throw new Error(
      `Projeto ambiguo: ${sel.projectName ?? ''} ${sel.projectId ?? ''} ${sel.projectDir ?? ''}`.trim()
    );
  }
  return candidates[0] as RegistryProject;
}

export function findProjectByRoot(
  registry: Registry,
  rootPath: string
): RegistryProject | null {
  if (!rootPath) return null;
  const abs = resolveProjectRoot(rootPath);
  return (
    registry.projects.map(normalizeProject).find((p) => toAbs(p.root) === abs) ?? null
  );
}

export function findBranch(
  project: RegistryProject,
  branchName: string
): RegistryBranch | undefined {
  return (project.branches ?? []).find((b) => b.name === branchName);
}

export function upsertProject(registry: Registry, project: RegistryProject): void {
  const idx = registry.projects.findIndex((p) => p.name === project.name);
  if (idx >= 0) {
    registry.projects[idx] = project;
  } else {
    registry.projects.push(project);
  }
}

export function upsertBranch(
  registry: Registry,
  projectName: string,
  branch: RegistryBranch
): void {
  const project = registry.projects.find((p) => p.name === projectName);
  if (!project) throw new Error(`Projeto nao encontrado: ${projectName}`);
  if (!project.branches) project.branches = [];
  const idx = project.branches.findIndex((b) => b.name === branch.name);
  if (idx >= 0) {
    project.branches[idx] = branch;
  } else {
    project.branches.push(branch);
  }
  project.updatedAt = utcNow();
}

// ── Git helpers (using local git CLI) ───────────────────────────────────────

function resolveProjectRoot(pathValue: string): string {
  // For worktrees, walk up until we hit the *main* repo (the one whose .git
  // is a directory). The PS impl uses git --git-common-dir; we mirror it.
  if (!pathValue) return pathValue;
  const root = findGitRoot(pathValue);
  if (!root) return toAbs(pathValue);
  // If linked worktree, traverse to common dir's parent.
  if (isWorktree(root)) {
    try {
      const commonDir = execFileSync('git', ['-C', root, 'rev-parse', '--git-common-dir'], {
        encoding: 'utf-8',
        timeout: 5000,
      }).trim();
      const absCommon = isAbsolute(commonDir) ? commonDir : resolve(root, commonDir);
      return resolve(dirname(absCommon));
    } catch {
      return toAbs(root);
    }
  }
  return toAbs(root);
}

function branchExists(repo: string, branch: string): boolean {
  try {
    execFileSync('git', ['-C', repo, 'rev-parse', '--verify', branch], {
      encoding: 'utf-8',
      timeout: 5000,
      stdio: ['ignore', 'pipe', 'ignore'],
    });
    return true;
  } catch {
    return false;
  }
}

function gitRevParse(repo: string, ref: string): string | null {
  try {
    return execFileSync('git', ['-C', repo, 'rev-parse', ref], {
      encoding: 'utf-8',
      timeout: 5000,
    }).trim();
  } catch {
    return null;
  }
}

// ── Action handlers ─────────────────────────────────────────────────────────

export interface BaseArgs {
  registryPath: string;
}

export interface ProjectArgs extends BaseArgs, ProjectSelector {}

export interface BranchArgs extends ProjectArgs {
  branchName: string;
}

export interface StartAgentBranchArgs extends ProjectArgs {
  branchName: string;
  baseBranch?: string;
  returnBranch?: string;
  worktreePath?: string;
  worktreeRoot?: string;
  useWorktree?: boolean;
  purpose?: string;
  createdBy?: string;
}

export interface AbandonAgentBranchArgs extends BranchArgs {
  removeWorktree?: boolean;
}

export function runInit(args: BaseArgs): unknown {
  if (!existsSync(args.registryPath)) {
    writeRegistry(args.registryPath, newRegistry());
  }
  return { success: true, action: 'init', registry: args.registryPath };
}

export function runListProjects(args: BaseArgs): unknown {
  const registry = readRegistry(args.registryPath);
  return {
    success: true,
    registry: args.registryPath,
    projects: registry.projects.map((p) => ({
      name: p.name,
      projectId: p.projectId ?? null,
      root: toAbs(p.root),
      defaultBranch: p.defaultBranch ?? 'main',
      tenantStrategy: p.tenantStrategy ?? 'project',
      enabled: p.enabled ?? true,
    })),
  };
}

export function runListBranches(args: ProjectArgs): unknown {
  const registry = readRegistry(args.registryPath);
  const project = findProject(registry, args);
  return {
    success: true,
    project: project.name,
    branches: project.branches ?? [],
  };
}

export function runAgentBranchStatus(args: BranchArgs): unknown {
  const registry = readRegistry(args.registryPath);
  const project = findProject(registry, args);
  const branch = findBranch(project, args.branchName);
  if (!branch) throw new Error(`Branch nao registrada: ${args.branchName}`);
  return {
    success: true,
    project: project.name,
    branch,
  };
}

export function runStartAgentBranch(args: StartAgentBranchArgs): unknown {
  if (!args.branchName) throw new Error('branchName obrigatorio');

  const registry = readRegistry(args.registryPath);

  // Find project (may auto-create from worktreePath via sync semantics).
  let project: RegistryProject;
  try {
    project = findProject(registry, args);
  } catch (err) {
    // If projectDir was provided and we can detect a git repo there,
    // bootstrap a project entry. Otherwise propagate.
    if (args.projectDir && isGitRepository(args.projectDir)) {
      const root = resolveProjectRoot(args.projectDir);
      const remote = getGitRemoteUrl(root);
      const nameOpt = args.projectName ?? root.split(/[\\/]/).pop() ?? 'unknown';
      project = {
        name: nameOpt,
        root,
        projectId: null,
        defaultBranch: args.baseBranch ?? 'main',
        tenantStrategy: 'project',
        enabled: true,
        createdAt: utcNow(),
        updatedAt: utcNow(),
        branches: [],
      };
      // remote URL noted but not yet wired into registry schema
      void remote;
      upsertProject(registry, project);
    } else {
      throw err;
    }
  }

  const repo = toAbs(project.root);
  const baseBranch = args.baseBranch ?? project.defaultBranch ?? 'main';
  const useWt = args.useWorktree === true;
  const slug = safeBranchSlug(args.branchName);

  let branchPath: string;
  let baseCommit: string | null = null;
  const wtPath = args.worktreePath
    ? toAbs(args.worktreePath)
    : args.worktreeRoot
      ? join(toAbs(args.worktreeRoot), `${repo.split(/[\\/]/).pop()}-${slug}`)
      : join(dirname(repo), `${repo.split(/[\\/]/).pop()}-${slug}`);

  if (useWt) {
    // Worktree mode: create new worktree, OR adopt existing one (backfill).
    if (existsSync(wtPath)) {
      // Backfill: worktree already exists on disk. Just register it.
      if (!isGitRepository(wtPath)) {
        throw new Error(
          `worktreePath existe mas nao e um worktree git valido: ${wtPath}`
        );
      }
      baseCommit = gitRevParse(repo, baseBranch);
      branchPath = wtPath;
    } else {
      // Fresh worktree creation.
      baseCommit = gitRevParse(repo, baseBranch);
      if (!baseCommit) throw new Error(`baseBranch nao encontrada: ${baseBranch}`);
      if (branchExists(repo, args.branchName)) {
        execFileSync('git', ['-C', repo, 'worktree', 'add', wtPath, args.branchName], {
          stdio: 'inherit',
          timeout: 60000,
        });
      } else {
        execFileSync(
          'git',
          [
            '-C',
            repo,
            'worktree',
            'add',
            '-b',
            args.branchName,
            wtPath,
            baseBranch,
          ],
          { stdio: 'inherit', timeout: 60000 }
        );
      }
      branchPath = wtPath;
    }
  } else {
    // In-place checkout mode. Refuse on dirty tree (matches PS Assert-Clean).
    const status = execFileSync('git', ['-C', repo, 'status', '--porcelain'], {
      encoding: 'utf-8',
      timeout: 5000,
    });
    if (status.trim().length > 0) {
      throw new Error('working tree nao esta limpo; commit ou stash antes');
    }
    execFileSync('git', ['-C', repo, 'checkout', baseBranch], {
      stdio: 'inherit',
      timeout: 15000,
    });
    baseCommit = getHeadCommit(repo);
    if (branchExists(repo, args.branchName)) {
      execFileSync('git', ['-C', repo, 'checkout', args.branchName], {
        stdio: 'inherit',
        timeout: 15000,
      });
    } else {
      execFileSync('git', ['-C', repo, 'checkout', '-b', args.branchName], {
        stdio: 'inherit',
        timeout: 15000,
      });
    }
    branchPath = repo;
  }

  const head = getHeadCommit(branchPath);
  const returnBranch = args.returnBranch ?? getCurrentBranch(repo) ?? baseBranch;

  const branch: RegistryBranch = {
    name: args.branchName,
    kind: 'agent',
    path: toAbs(branchPath),
    baseBranch,
    returnBranch,
    status: 'active',
    createdBy: args.createdBy ?? 'ai-agent',
    createdAt: utcNow(),
    lastSeenAt: utcNow(),
    baseCommit,
    headCommit: head,
    lastIndexedCommit: null,
    watchEnabled: true,
    indexed: false,
    purpose: args.purpose ?? 'agent change',
    useWorktree: useWt,
  };

  upsertBranch(registry, project.name, branch);
  writeRegistry(args.registryPath, registry);

  return {
    success: true,
    action: 'start_agent_branch',
    project: project.name,
    branch,
    message:
      'Branch de agente registrada. Daemon registration acontece via hook ou sync_current_branch.',
  };
}

export function runFinishAgentBranch(args: BranchArgs): unknown {
  if (!args.branchName) throw new Error('branchName obrigatorio');
  const registry = readRegistry(args.registryPath);
  const project = findProject(registry, args);
  const branch = findBranch(project, args.branchName);
  if (!branch) throw new Error(`Branch nao registrada: ${args.branchName}`);

  const path = toAbs(branch.path);
  if (existsSync(path)) {
    const head = getHeadCommit(path);
    if (head) branch.headCommit = head;
  }
  branch.lastSeenAt = utcNow();
  branch.status = 'ready_for_review';
  branch.note = 'Pronta para revisao humana. Merge nao executado.';

  upsertBranch(registry, project.name, branch);
  writeRegistry(args.registryPath, registry);

  return {
    success: true,
    action: 'finish_agent_branch',
    project: project.name,
    branch,
    message: 'Marcada como ready_for_review sem merge.',
  };
}

export function runAbandonAgentBranch(args: AbandonAgentBranchArgs): unknown {
  if (!args.branchName) throw new Error('branchName obrigatorio');
  const registry = readRegistry(args.registryPath);
  const project = findProject(registry, args);
  const branch = findBranch(project, args.branchName);
  if (!branch) throw new Error(`Branch nao registrada: ${args.branchName}`);

  branch.status = 'abandoned';
  branch.lastSeenAt = utcNow();
  branch.note = 'Abandonada no registry. Worktree/branch nao deletados automaticamente.';

  if (args.removeWorktree === true && branch.useWorktree) {
    try {
      execFileSync(
        'git',
        ['-C', toAbs(project.root), 'worktree', 'remove', toAbs(branch.path)],
        { stdio: 'inherit', timeout: 30000 }
      );
      branch.note = 'Abandonada e worktree removida por solicitacao explicita.';
    } catch (err) {
      branch.note = `Abandonada; worktree remove falhou: ${(err as Error).message}`;
    }
  }

  upsertBranch(registry, project.name, branch);
  writeRegistry(args.registryPath, registry);

  return {
    success: true,
    action: 'abandon_agent_branch',
    project: project.name,
    branch,
  };
}
