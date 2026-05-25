import { afterEach, beforeEach, describe, expect, it } from 'vitest';
import { execFileSync } from 'node:child_process';
import { existsSync, mkdtempSync, mkdirSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { tmpdir } from 'node:os';
import { fileURLToPath } from 'node:url';
import { setTimeout as delay } from 'node:timers/promises';

const suite = process.platform === 'win32' ? describe : describe.skip;
const testDir = dirname(fileURLToPath(import.meta.url));
const repoRoot = resolve(testDir, '../../../../..');
const installScript = join(repoRoot, 'scripts', 'windows', 'indexed-projects-hooks.ps1');
const registryScript = join(repoRoot, 'scripts', 'windows', 'indexed-projects-registry.ps1');

type RegistryProject = {
  name: string;
  root: string;
  defaultBranch?: string;
  branches?: Array<Record<string, unknown> & { name: string; path: string }>;
};

type RegistryState = {
  projects?: RegistryProject[];
};

function normalizePath(value: string) {
  return resolve(value).toLowerCase();
}

function run(command: string, args: string[], options: { cwd?: string; env?: NodeJS.ProcessEnv } = {}) {
  return execFileSync(command, args, {
    cwd: options.cwd,
    env: options.env,
    encoding: 'utf8',
    stdio: ['ignore', 'pipe', 'pipe'],
  }).trim();
}

function runGit(repo: string, args: string[], env: NodeJS.ProcessEnv) {
  return run('git', args, { cwd: repo, env });
}

function runPowerShellScript(script: string, args: string[], cwd: string, env: NodeJS.ProcessEnv) {
  const command = ['powershell.exe', '-NoLogo', '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', script, ...args].join(' ');

  return run('cmd.exe', ['/d', '/s', '/c', command], {
    cwd,
    env,
  });
}

function readRegistry(registryPath: string): RegistryState | null {
  if (!existsSync(registryPath)) {
    return null;
  }

  return JSON.parse(readFileSync(registryPath, 'utf8')) as RegistryState;
}

function findProject(registry: RegistryState | null, root: string) {
  const projects = registry?.projects ?? [];
  const target = normalizePath(root);
  return projects.find((project) => normalizePath(project.root) === target);
}

function findBranch(project: RegistryProject | undefined, name: string) {
  return project?.branches?.find((branch) => branch.name === name);
}

async function waitFor<T>(factory: () => T | Promise<T>, timeoutMs = 20000, intervalMs = 200) {
  const deadline = Date.now() + timeoutMs;
  let lastError: unknown = null;

  while (Date.now() < deadline) {
    try {
      const value = await factory();
      if (value) {
        return value;
      }
    } catch (error) {
      lastError = error;
    }

    await delay(intervalMs);
  }

  if (lastError instanceof Error) {
    throw lastError;
  }

  throw new Error('Timed out waiting for registry update');
}

suite('Indexed projects git hooks', () => {
  let tempRoot: string;
  let repoDir: string;
  let registryPath: string;
  let hooksDir: string;
  let fakeWqmDir: string;
  let fakeWqmMarker: string;
  let env: NodeJS.ProcessEnv;

  beforeEach(() => {
    tempRoot = mkdtempSync(join(tmpdir(), 'wqm-hooks-test-'));
    repoDir = join(tempRoot, 'repo');
    hooksDir = join(repoDir, '.wqm-fork', 'git-hooks');
    registryPath = join(repoDir, '.wqm-fork', 'indexed-projects.json');

    mkdirSync(repoDir);
    fakeWqmDir = join(tempRoot, 'bin');
    mkdirSync(fakeWqmDir);
    fakeWqmMarker = join(tempRoot, 'wqm-called.txt');
    writeFileSync(
      join(fakeWqmDir, 'wqm.cmd'),
      '@echo off\r\necho CALLED>%WQM_MARKER%\r\nexit /b 0\r\n',
      'utf8'
    );

    env = {
      ...process.env,
      PATH: `${fakeWqmDir};${process.env.PATH ?? ''}`,
      WQM_PATH: join(fakeWqmDir, 'wqm.cmd'),
      WQM_MARKER: fakeWqmMarker,
    };

    try {
      runGit(repoDir, ['init', '-b', 'main'], env);
    } catch {
      runGit(repoDir, ['init'], env);
      runGit(repoDir, ['checkout', '-b', 'main'], env);
    }

    runGit(repoDir, ['config', 'user.name', 'Codex Test'], env);
    runGit(repoDir, ['config', 'user.email', 'codex@example.com'], env);

    writeFileSync(join(repoDir, 'README.md'), 'initial\n', 'utf8');
    runGit(repoDir, ['add', 'README.md'], env);
    runGit(repoDir, ['commit', '-m', 'initial'], env);
  });

  afterEach(() => {
    rmSync(tempRoot, { recursive: true, force: true });
  });

  it('installs hooks and registers the current branch immediately', async () => {
    const output = runPowerShellScript(installScript, ['-Action', 'install', '-RepoDir', repoDir], repoDir, env);
    const status = JSON.parse(output) as {
      hooksPathMatches: boolean;
      configuredHooksPath: string;
      hooksDir: string;
      runnerExists: boolean;
      hookCount: number;
    };

    expect(status.hooksPathMatches).toBe(true);
    expect(normalizePath(status.configuredHooksPath)).toBe(normalizePath(hooksDir));
    expect(status.runnerExists).toBe(true);
    expect(status.hookCount).toBe(5);

    runGit(repoDir, ['commit', '--allow-empty', '-m', 'trigger initial hook'], env);

    const created = await waitFor(() => {
      const registry = readRegistry(registryPath);
      const project = findProject(registry, repoDir);
      const mainBranch = findBranch(project, 'main');
      return project && mainBranch ? { project, mainBranch } : null;
    });

    expect(created.project).toBeDefined();
    expect(created.mainBranch).toBeDefined();
    expect(created.mainBranch?.indexed).toBe(true);
    expect(normalizePath(created.mainBranch?.path ?? '')).toBe(normalizePath(repoDir));
    expect(created.mainBranch?.headCommit).toBe(runGit(repoDir, ['rev-parse', 'HEAD'], env));
  }, 30000);

  it('registers a manual checkout branch and updates it after commits', async () => {
    runPowerShellScript(installScript, ['-Action', 'install', '-RepoDir', repoDir], repoDir, env);

    runGit(repoDir, ['checkout', '-b', 'feature/manual-checkout'], env);

    const created = await waitFor(() => {
      const registry = readRegistry(registryPath);
      const project = findProject(registry, repoDir);
      const branch = findBranch(project, 'feature/manual-checkout');
      return branch && project ? { project, branch } : null;
    });

    expect(created.branch.kind).toBe('manual');
    expect(created.branch.useWorktree).toBe(false);
    expect(normalizePath(created.branch.path)).toBe(normalizePath(repoDir));

    runGit(repoDir, ['commit', '--allow-empty', '-m', 'manual branch update'], env);
    const headCommit = runGit(repoDir, ['rev-parse', 'HEAD'], env);

    const updated = await waitFor(() => {
      const registry = readRegistry(registryPath);
      const project = findProject(registry, repoDir);
      const branch = findBranch(project, 'feature/manual-checkout');
      return branch?.headCommit === headCommit ? branch : null;
    });

    expect(updated.headCommit).toBe(headCommit);
    expect(updated.lastIndexedCommit).toBe(headCommit);
    expect(typeof updated.lastSeenAt).toBe('string');
  }, 30000);

  it('removes deleted branch entries with cleanup-orphans', async () => {
    runPowerShellScript(installScript, ['-Action', 'install', '-RepoDir', repoDir], repoDir, env);

    runGit(repoDir, ['checkout', '-b', 'feature/delete-me'], env);
    await waitFor(() => {
      const registry = readRegistry(registryPath);
      const project = findProject(registry, repoDir);
      return findBranch(project, 'feature/delete-me') ?? null;
    });

    runGit(repoDir, ['checkout', 'main'], env);
    runGit(repoDir, ['branch', '-D', 'feature/delete-me'], env);

    const cleanupOutput = runPowerShellScript(
      registryScript,
      ['-Action', 'cleanup-orphans', '-RepoDir', repoDir, '-RegistryPath', registryPath, '-Mutate', 'true'],
      repoDir,
      env
    );
    const cleanup = JSON.parse(cleanupOutput) as {
      mutated: boolean;
      removedBranchCount: number;
      removedBranches: Array<{ branch: string }>;
    };

    expect(cleanup.mutated).toBe(true);
    expect(cleanup.removedBranchCount).toBeGreaterThanOrEqual(1);
    expect(cleanup.removedBranches.some((entry) => entry.branch === 'feature/delete-me')).toBe(true);

    const registry = readRegistry(registryPath);
    const project = findProject(registry, repoDir);

    expect(findBranch(project, 'feature/delete-me')).toBeUndefined();
    expect(findBranch(project, 'main')).toBeDefined();
  }, 30000);

  it('registers a manual worktree branch and removes it when the worktree is deleted', async () => {
    runPowerShellScript(installScript, ['-Action', 'install', '-RepoDir', repoDir], repoDir, env);

    runGit(repoDir, ['commit', '--allow-empty', '-m', 'trigger initial hook'], env);
    await waitFor(() => {
      const registry = readRegistry(registryPath);
      const project = findProject(registry, repoDir);
      const mainBranch = findBranch(project, 'main');
      return project && mainBranch ? { project, mainBranch } : null;
    });

    const worktreeDir = join(tempRoot, 'feature-worktree');
    runGit(repoDir, ['worktree', 'add', '-b', 'feature/worktree', worktreeDir, 'main'], env);

    const created = await waitFor(() => {
      const registry = readRegistry(registryPath);
      const project = findProject(registry, repoDir);
      const branch = findBranch(project, 'feature/worktree');
      return branch && project ? { project, branch } : null;
    });

    expect(created.branch.kind).toBe('manual-worktree');
    expect(created.branch.useWorktree).toBe(true);
    expect(normalizePath(created.branch.path)).toBe(normalizePath(worktreeDir));

    runGit(repoDir, ['worktree', 'remove', '--force', worktreeDir], env);

    const cleanupOutput = runPowerShellScript(
      registryScript,
      ['-Action', 'cleanup-orphans', '-RepoDir', repoDir, '-RegistryPath', registryPath, '-Mutate', 'true'],
      repoDir,
      env
    );
    const cleanup = JSON.parse(cleanupOutput) as {
      mutated: boolean;
      removedBranchCount: number;
      removedBranches: Array<{ branch: string; path: string }>;
    };

    expect(cleanup.mutated).toBe(true);
    expect(cleanup.removedBranches.some((entry) => entry.branch === 'feature/worktree')).toBe(true);

    const registry = readRegistry(registryPath);
    const project = findProject(registry, repoDir);

    expect(findBranch(project, 'feature/worktree')).toBeUndefined();
    expect(findBranch(project, 'main')).toBeDefined();
  }, 30000);
});
