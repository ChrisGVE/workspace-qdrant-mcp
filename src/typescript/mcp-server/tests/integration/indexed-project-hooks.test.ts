import { afterEach, beforeEach, describe, expect, it } from 'vitest';
import { execFileSync, spawnSync } from 'node:child_process';
import { existsSync, mkdtempSync, mkdirSync, readFileSync, readdirSync, rmSync, writeFileSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { tmpdir } from 'node:os';
import { fileURLToPath } from 'node:url';
import { setTimeout as delay } from 'node:timers/promises';

const suite = process.platform === 'win32' ? describe : describe.skip;
const testDir = dirname(fileURLToPath(import.meta.url));
const repoRoot = resolve(testDir, '../../../../..');
const installScript = join(repoRoot, 'scripts', 'windows', 'indexed-projects-hooks.ps1');
const registryScript = join(repoRoot, 'scripts', 'windows', 'indexed-projects-registry.ps1');
const statusScript = join(repoRoot, 'scripts', 'windows', 'status.ps1');

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

function runPowerShellScriptResult(script: string, args: string[], cwd: string, env: NodeJS.ProcessEnv) {
  const command = ['powershell.exe', '-NoLogo', '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', script, ...args].join(' ');

  return spawnSync('cmd.exe', ['/d', '/s', '/c', command], {
    cwd,
    env,
    encoding: 'utf8',
    stdio: ['ignore', 'pipe', 'pipe'],
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

function readLatestServiceSnapshot(logDir: string) {
  const files = readdirSync(logDir)
    .filter((entry) => entry.startsWith('service-observe-') && entry.endsWith('.jsonl'))
    .sort();

  expect(files.length).toBeGreaterThan(0);

  const logPath = join(logDir, files[files.length - 1]);
  const lines = readFileSync(logPath, 'utf8')
    .trim()
    .split(/\r?\n/)
    .filter(Boolean);

  expect(lines.length).toBeGreaterThan(0);
  return JSON.parse(lines[lines.length - 1]);
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
      installed: boolean;
      healthy: boolean;
    };

    expect(status.hooksPathMatches).toBe(true);
    expect(normalizePath(status.configuredHooksPath)).toBe(normalizePath(hooksDir));
    expect(status.runnerExists).toBe(true);
    expect(status.hookCount).toBe(5);
    expect(status.installed).toBe(true);
    expect(status.healthy).toBe(true);

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

  it('includes hook status in the observability snapshot', async () => {
    runPowerShellScript(installScript, ['-Action', 'install', '-RepoDir', repoDir], repoDir, env);

    const observeLogDir = join(tempRoot, 'observe-logs');
    mkdirSync(observeLogDir);

    const serviceObserveScript = join(repoRoot, 'scripts', 'windows', 'service-observe.ps1');
    runPowerShellScript(
      serviceObserveScript,
      [
        '-RepoDir',
        repoDir,
        '-ProjectDir',
        repoDir,
        '-QdrantUrl',
        'http://localhost:6333',
        '-DaemonEndpoint',
        'localhost:50051',
        '-LogDir',
        observeLogDir,
        '-Once',
      ],
      repoDir,
      env
    );

    const snapshot = readLatestServiceSnapshot(observeLogDir) as {
      hooks?: {
        available?: boolean;
        installed?: boolean;
        healthy?: boolean;
        status?: {
          hooksPathMatches?: boolean;
          runnerExists?: boolean;
          hookCount?: number;
          installed?: boolean;
          healthy?: boolean;
          reason?: string;
        };
      };
    };

    expect(snapshot.hooks).toBeDefined();
    expect(snapshot.hooks?.available).toBe(true);
    expect(snapshot.hooks?.installed).toBe(true);
    expect(snapshot.hooks?.healthy).toBe(true);
    expect(snapshot.hooks?.status?.hooksPathMatches).toBe(true);
    expect(snapshot.hooks?.status?.runnerExists).toBe(true);
    expect(snapshot.hooks?.status?.hookCount).toBe(5);
    expect(snapshot.hooks?.status?.installed).toBe(true);
    expect(snapshot.hooks?.status?.healthy).toBe(true);
  }, 30000);

  it('treats watch probes as optional when the CLI does not expose them', async () => {
    writeFileSync(
      join(fakeWqmDir, 'wqm.cmd'),
      [
        '@echo off',
        'setlocal',
        'set "first=%~1"',
        'set "second=%~2"',
        'if /I "%first%"=="watch" (',
        "  echo error: unrecognized subcommand 'watch' 1>&2",
        '  exit /b 2',
        ')',
        'if /I "%first% %second%"=="project watch" (',
        "  echo error: unrecognized subcommand 'watch' 1>&2",
        '  exit /b 2',
        ')',
        'echo CALLED>%WQM_MARKER%',
        'exit /b 0',
      ].join('\r\n'),
      'utf8'
    );

    runPowerShellScript(installScript, ['-Action', 'install', '-RepoDir', repoDir], repoDir, env);

    const observeLogDir = join(tempRoot, 'observe-logs-watch-optional');
    mkdirSync(observeLogDir);

    const serviceObserveScript = join(repoRoot, 'scripts', 'windows', 'service-observe.ps1');
    runPowerShellScript(
      serviceObserveScript,
      [
        '-RepoDir',
        repoDir,
        '-ProjectDir',
        repoDir,
        '-QdrantUrl',
        'http://localhost:6333',
        '-DaemonEndpoint',
        'localhost:50051',
        '-LogDir',
        observeLogDir,
        '-Once',
      ],
      repoDir,
      env
    );

    const snapshot = readLatestServiceSnapshot(observeLogDir) as {
      wqm?: {
        watchList?: {
          ok?: boolean;
          skipped?: boolean;
          available?: boolean;
          reason?: string;
        };
      };
    };

    expect(snapshot.wqm?.watchList).toBeDefined();
    expect(snapshot.wqm?.watchList?.ok).toBe(true);
    expect(snapshot.wqm?.watchList?.skipped).toBe(true);
    expect(snapshot.wqm?.watchList?.available).toBe(false);
    expect(snapshot.wqm?.watchList?.reason).toContain('watch subcommand unavailable');
  }, 30000);

  it('includes hook status in the status command output', async () => {
    runPowerShellScript(installScript, ['-Action', 'install', '-RepoDir', repoDir], repoDir, env);

    const output = runPowerShellScript(
      statusScript,
      ['-RepoDir', repoDir, '-QdrantUrl', 'http://localhost:6333', '-DaemonEndpoint', 'localhost:50051'],
      repoDir,
      env
    );

    expect(output).toContain('== workspace-qdrant Docker status ==');
    expect(output).toContain('Hooks: ok');
  }, 30000);

  it('returns a non-zero status when hooks are not installed', () => {
    const result = runPowerShellScriptResult(
      statusScript,
      ['-RepoDir', repoDir, '-QdrantUrl', 'http://localhost:6333', '-DaemonEndpoint', 'localhost:50051'],
      repoDir,
      env
    );

    expect(result.error).toBeUndefined();
    expect(result.status).not.toBe(0);
    expect(result.stdout).toContain('Hooks: warn');
  });

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

  it('bootstraps an unindexed repo when starting an agent branch', async () => {
    const output = runPowerShellScript(
      registryScript,
      [
        '-Action',
        'start-agent-branch',
        '-BranchName',
        'agent/bootstrap-index',
        '-ProjectName',
        'bootstrap-project',
        '-Mutate',
        'true',
      ],
      repoDir,
      env
    );

    const result = JSON.parse(output) as {
      success: boolean;
      createdProject?: boolean;
      project?: string;
      root?: string;
      branch?: {
        name?: string;
        path?: string;
        useWorktree?: boolean;
        indexed?: boolean;
      };
    };

    expect(result.success).toBe(true);
    expect(result.createdProject).toBe(true);
    expect(result.project).toBe('bootstrap-project');
    expect(result.branch?.name).toBe('agent/bootstrap-index');
    expect(result.branch?.useWorktree).toBe(true);
    expect(result.branch?.indexed).toBe(true);

    const registry = readRegistry(registryPath);
    const project = findProject(registry, repoDir);
    const branch = findBranch(project, 'agent/bootstrap-index');

    expect(project?.name).toBe('bootstrap-project');
    expect(branch?.indexed).toBe(true);
    expect(branch?.useWorktree).toBe(true);
    expect(normalizePath(branch?.path ?? '')).not.toBe(normalizePath(repoDir));
  }, 30000);

  it('ignores detached HEAD commits instead of rebinding them to the default branch', async () => {
    runPowerShellScript(installScript, ['-Action', 'install', '-RepoDir', repoDir], repoDir, env);

    const baseHead = runGit(repoDir, ['rev-parse', 'HEAD'], env);

    await waitFor(() => {
      const registry = readRegistry(registryPath);
      const project = findProject(registry, repoDir);
      const mainBranch = findBranch(project, 'main');
      return mainBranch && normalizePath(mainBranch.path) === normalizePath(repoDir) ? mainBranch : null;
    });

    runGit(repoDir, ['checkout', '--detach', 'HEAD'], env);
    runGit(repoDir, ['commit', '--allow-empty', '-m', 'detached HEAD commit'], env);
    const detachedHead = runGit(repoDir, ['rev-parse', 'HEAD'], env);

    const preserved = await waitFor(() => {
      const registry = readRegistry(registryPath);
      const project = findProject(registry, repoDir);
      const mainBranch = findBranch(project, 'main');
      return mainBranch?.headCommit === baseHead ? mainBranch : null;
    });

    expect(preserved.headCommit).toBe(baseHead);
    expect(preserved.headCommit).not.toBe(detachedHead);
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
