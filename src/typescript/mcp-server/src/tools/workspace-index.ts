/**
 * workspace_index MCP tool implementation.
 *
 * Most actions delegate to the PowerShell registry helper (host-only). The
 * `sync_current_branch` action is implemented natively in TypeScript so it can
 * run inside the dockerized MCP container — it receives git state from a host
 * hook (which has access to the local git CLI) and forwards a RegisterProject
 * gRPC call to the daemon.
 */

import { spawn } from 'node:child_process';
import { existsSync } from 'node:fs';
import { basename, dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

import type { DaemonClient } from './../clients/daemon-client.js';
import { startGitHookTimer } from './../telemetry/metrics.js';
import { getGitState } from './../utils/git-utils.js';
import {
  defaultRegistryPath,
  runAbandonAgentBranch,
  runAgentBranchStatus,
  runFinishAgentBranch,
  runIncrementalCheck,
  runIncrementalCheckAll,
  runInit,
  runListBranches,
  runListProjects,
  runObserveAll,
  runObserveProject,
  runProjectStatus,
  runStartAgentBranch,
  runStatusAll,
  type AbandonAgentBranchArgs,
  type BaseArgs,
  type BranchArgs,
  type ProjectArgs,
  type StartAgentBranchArgs,
} from './indexed-projects-registry.js';

type JsonObject = Record<string, unknown>;

const MUTATING_ACTIONS = new Set([
  'init',
  'add_project',
  'start_agent_branch',
  'finish_agent_branch',
  'abandon_agent_branch',
  'register_wqm',
  'register_all_wqm',
  'cleanup_orphans',
]);

const ACTION_TO_PS: Record<string, string> = {
  init: 'init',
  list_projects: 'list-projects',
  project_status: 'project-status',
  status_all: 'status-all',
  list_branches: 'list-branches',
  agent_branch_status: 'agent-branch-status',
  observe_project: 'observe-project',
  observe_all: 'observe-all',
  incremental_check: 'incremental-check',
  incremental_check_all: 'incremental-check-all',
  add_project: 'add-project',
  start_agent_branch: 'start-agent-branch',
  finish_agent_branch: 'finish-agent-branch',
  abandon_agent_branch: 'abandon-agent-branch',
  register_wqm: 'register-wqm',
  register_all_wqm: 'register-all-wqm',
  cleanup_orphans: 'cleanup-orphans',
};

function payloadObject(args: JsonObject): JsonObject {
  const payload = args['payload'];
  if (payload && typeof payload === 'object' && !Array.isArray(payload)) {
    return payload as JsonObject;
  }
  return {};
}

function argValues(args: JsonObject, key: string, aliases: string[] = []): unknown[] {
  const payload = payloadObject(args);
  const keys = [key, ...aliases];
  const values: unknown[] = [];

  for (const candidate of keys) {
    if (Object.prototype.hasOwnProperty.call(args, candidate)) values.push(args[candidate]);
  }
  for (const candidate of keys) {
    if (Object.prototype.hasOwnProperty.call(payload, candidate)) values.push(payload[candidate]);
  }

  return values;
}

function stringArg(args: JsonObject, key: string, aliases: string[] = []): string | undefined {
  for (const value of argValues(args, key, aliases)) {
    if (typeof value === 'string' && value.trim().length > 0) return value;
  }
  return undefined;
}

function boolArg(args: JsonObject, key: string, aliases: string[] = []): string | undefined {
  for (const value of argValues(args, key, aliases)) {
    if (typeof value === 'boolean') return String(value);
    if (typeof value === 'string' && value.trim().match(/^(1|0|true|false|yes|no|y|n)$/i)) {
      return value.trim();
    }
  }
  return undefined;
}

function assertMutationAllowed(action: string, args: JsonObject): void {
  if (!MUTATING_ACTIONS.has(action)) return;

  const envAllowed = process.env['WQM_INDEX_MANAGER_ALLOW_MUTATION'] === '1';
  const callAllowed = args['allowMutation'] === true;
  if (!envAllowed || !callAllowed) {
    throw new Error(
      `workspace_index action '${action}' mutates local workspace state. ` +
        'Set WQM_INDEX_MANAGER_ALLOW_MUTATION=1 and pass allowMutation=true to continue.'
    );
  }
}

let autoDetectedRepoDir: string | null | undefined;

function autoDetectRepoDir(): string | undefined {
  if (autoDetectedRepoDir !== undefined) {
    return autoDetectedRepoDir ?? undefined;
  }
  try {
    let cur = dirname(fileURLToPath(import.meta.url));
    for (let i = 0; i < 8; i++) {
      const marker = resolve(cur, 'scripts', 'windows', 'workspace-index-mcp.ps1');
      if (existsSync(marker)) {
        autoDetectedRepoDir = cur;
        return cur;
      }
      const parent = dirname(cur);
      if (parent === cur) break;
      cur = parent;
    }
  } catch {
    // import.meta.url unavailable (CJS bundle) — fall through.
  }
  autoDetectedRepoDir = null;
  return undefined;
}

function resolveRepoDir(args: JsonObject): string {
  return resolve(
    stringArg(args, 'repoDir') ??
      process.env['WQM_REPO_DIR'] ??
      autoDetectRepoDir() ??
      process.cwd()
  );
}

function buildScriptArgs(args: JsonObject, repoDir: string): string[] {
  const action = stringArg(args, 'action');
  if (!action) throw new Error('workspace_index requires action');
  const psAction = ACTION_TO_PS[action];
  if (!psAction) throw new Error(`Unsupported workspace_index action: ${action}`);

  assertMutationAllowed(action, args);

  const registryPath = resolve(repoDir, '.wqm-fork', 'indexed-projects.json');
  const scriptArgs = [
    '-NoProfile',
    '-ExecutionPolicy',
    'Bypass',
    '-File',
    resolve(repoDir, 'scripts', 'windows', 'workspace-index-mcp.ps1'),
    '-Action',
    psAction,
    '-RegistryPath',
    registryPath,
    '-AllowMutation',
    args['allowMutation'] === true ? 'true' : 'false',
  ];

  const pairs: Array<[string, string | undefined]> = [
    ['-ProjectName', stringArg(args, 'projectName')],
    ['-ProjectId', stringArg(args, 'projectId')],
    ['-ProjectDir', stringArg(args, 'projectPath')],
    ['-BranchName', stringArg(args, 'branchName', ['branch'])],
    ['-BaseBranch', stringArg(args, 'baseBranch')],
    ['-ReturnBranch', stringArg(args, 'returnBranch')],
    ['-WorktreePath', stringArg(args, 'worktreePath', ['worktree'])],
    ['-WorktreeRoot', stringArg(args, 'worktreeRoot')],
    ['-UseWorktree', boolArg(args, 'useWorktree')],
    ['-Purpose', stringArg(args, 'purpose')],
    ['-CreatedBy', stringArg(args, 'createdBy')],
  ];

  for (const [name, value] of pairs) {
    if (value !== undefined) scriptArgs.push(name, value);
  }

  return scriptArgs;
}

/**
 * sync_current_branch — TypeScript-native handler.
 *
 * Forwards a RegisterProject gRPC call to the daemon with
 * `register_if_new: true` so a fresh branch or worktree becomes searchable
 * without manual wqm commands.
 *
 * Git state can come from two sources:
 *  - **Hook-provided fields** (`currentBranch`, `commitHash`, `worktreePath`,
 *    `isWorktree`, `gitRemote`) — authoritative; populated by the host-side
 *    hook that has direct access to the local `git` CLI.
 *  - **Local detection via `getGitState(repoDir)`** — fallback when the MCP
 *    server can see the path on its own filesystem. This works when the
 *    container has `git` and the same path string resolves to the same repo
 *    (e.g. dockerized MCP with `${WQM_DEV_ROOT}:${WQM_DEV_ROOT}` bind mount).
 *
 * Hook values always win; local detection only fills in the gaps. The daemon
 * does its own worktree detection and tenant_id calculation — we just deliver
 * the path and (optionally) the remote URL.
 */
async function handleSyncCurrentBranch(
  args: JsonObject,
  daemonClient: DaemonClient
): Promise<unknown> {
  const repoDir = stringArg(args, 'repoDir');
  const hookNameEarly = stringArg(args, 'hookName', ['hook_name']) ?? 'manual';
  if (!repoDir) {
    const recordResult = startGitHookTimer(hookNameEarly, false);
    recordResult('bad_request');
    return {
      success: false,
      action: 'sync_current_branch',
      error: 'repoDir is required (absolute path to the target repo)',
    };
  }

  const hookBranch = stringArg(args, 'currentBranch', ['branch', 'branchName']);
  const hookCommit = stringArg(args, 'commitHash', ['commit_hash', 'commit']);
  const hookWorktreePath = stringArg(args, 'worktreePath', ['worktree']);
  const hookIsWorktreeRaw = boolArg(args, 'isWorktree', ['is_worktree']);
  const hookRemote = stringArg(args, 'gitRemote', ['git_remote']);
  const hookName = hookNameEarly;
  const projectName =
    stringArg(args, 'projectName', ['name']) ?? (basename(repoDir) || 'unknown');

  // Best-effort local detection. Returns null when the MCP server can't see
  // the path (e.g. container without bind mount). Hook values still apply.
  const localState = getGitState(repoDir);

  const branch = hookBranch ?? localState?.branch ?? null;
  const commitHash = hookCommit ?? localState?.commit ?? null;
  const gitRemote = hookRemote ?? localState?.remoteUrl ?? undefined;
  const isWorktree =
    hookIsWorktreeRaw === 'true' || hookIsWorktreeRaw === '1' || hookIsWorktreeRaw === 'yes'
      ? true
      : localState?.isWorktree ?? false;
  const worktreePath = hookWorktreePath ?? localState?.worktreePath ?? null;

  const watchPath = isWorktree && worktreePath ? worktreePath : repoDir;
  const recordResult = startGitHookTimer(hookName, isWorktree);

  try {
    const response = await daemonClient.registerProject({
      path: watchPath,
      project_id: '',
      name: projectName,
      register_if_new: true,
      priority: 'high',
      ...(gitRemote ? { git_remote: gitRemote } : {}),
    });

    recordResult('success');
    return {
      success: true,
      action: 'sync_current_branch',
      hook: hookName,
      repo_dir: repoDir,
      watch_path: response.watch_path ?? watchPath,
      project_id: response.project_id,
      newly_registered: response.newly_registered,
      created: response.created,
      is_active: response.is_active,
      is_worktree: response.is_worktree ?? isWorktree,
      branch: branch ?? null,
      commit_hash: commitHash ?? null,
    };
  } catch (error) {
    recordResult('error');
    const errorMessage = error instanceof Error ? error.message : String(error);
    return {
      success: false,
      action: 'sync_current_branch',
      hook: hookName,
      repo_dir: repoDir,
      error: errorMessage,
    };
  }
}

/** Render an ETA in seconds as a coarse human-readable string ("3s", "12m",
 *  "2h 14m"). Kept intentionally crude — exact precision is meaningless
 *  for an estimate built from a 5-minute rate window. */
function formatEtaSeconds(seconds: number): string {
  if (!Number.isFinite(seconds) || seconds < 0) return 'unknown';
  if (seconds < 60) return `${Math.round(seconds)}s`;
  const minutes = Math.round(seconds / 60);
  if (minutes < 60) return `${minutes}m`;
  const hours = Math.floor(minutes / 60);
  const rem = minutes % 60;
  return rem === 0 ? `${hours}h` : `${hours}h ${rem}m`;
}

/**
 * `indexing_status` action — report per-project indexing progress without
 * shelling out to wqm. Reads the daemon's `GetProjectStatus` (which fills
 * pending/in_progress/failed/done/total/percent_complete).
 *
 * Args:
 *   - `projectId` (optional): explicit tenant. If missing, the daemon's
 *     `ListProjects` is consulted and we pick the first active project.
 *     This keeps the action useful in the dockerized MCP container where
 *     cwd-based detection isn't available.
 */
async function handleIndexingStatus(
  args: JsonObject,
  daemonClient: DaemonClient,
  actionLabel: 'indexing_status' | 'project_status' = 'indexing_status'
): Promise<unknown> {
  let projectId = stringArg(args, 'projectId');

  if (!projectId) {
    try {
      const projects = await daemonClient.listProjects({ active_only: true });
      const first = projects.projects[0];
      if (first) projectId = first.project_id;
    } catch (err) {
      return {
        success: false,
        action: actionLabel,
        error: `Failed to enumerate active projects: ${err instanceof Error ? err.message : String(err)}`,
      };
    }
  }

  if (!projectId) {
    return {
      success: false,
      action: actionLabel,
      error:
        'No projectId provided and no active project found. Pass `projectId` explicitly or register a project first.',
    };
  }

  try {
    const status = await daemonClient.getProjectStatus({ project_id: projectId });
    if (!status.found) {
      return {
        success: false,
        action: actionLabel,
        project_id: projectId,
        error: 'Project not registered with daemon',
      };
    }
    const pending = status.pending_count ?? 0;
    const inProgress = status.in_progress_count ?? 0;
    const failed = status.failed_count ?? 0;
    const done = status.done_count ?? 0;
    const total = status.total_count ?? 0;
    const percent = status.percent_complete ?? 100;
    const inFlight = pending + inProgress;
    const eta = typeof status.eta_seconds === 'number' ? status.eta_seconds : undefined;
    const etaSummary =
      eta === undefined
        ? 'ETA unknown (warming up)'
        : `ETA ~${formatEtaSeconds(eta)}`;
    const summary =
      inFlight === 0
        ? `Indexing complete (${done} files indexed; ${failed} failed)`
        : `Indexing in progress: ${inFlight} files in flight, ${done}/${total} done (${percent.toFixed(1)}%) · ${etaSummary}`;

    const indexing: {
      pending: number;
      in_progress: number;
      failed: number;
      done: number;
      total: number;
      percent: number;
      eta_seconds?: number;
    } = { pending, in_progress: inProgress, failed, done, total, percent };
    if (eta !== undefined) indexing.eta_seconds = eta;

    return {
      success: true,
      action: actionLabel,
      project_id: projectId,
      project_name: status.project_name,
      project_root: status.project_root,
      is_active: status.is_active,
      indexing,
      summary,
    };
  } catch (err) {
    return {
      success: false,
      action: actionLabel,
      project_id: projectId,
      error: `Failed to fetch project status: ${err instanceof Error ? err.message : String(err)}`,
    };
  }
}

function parseOutput(stdout: string, stderr: string): unknown {
  const trimmed = stdout.trim();
  if (trimmed.length === 0) {
    return { success: false, error: 'workspace_index produced no JSON output', stderr };
  }
  try {
    return JSON.parse(trimmed) as unknown;
  } catch {
    return { success: false, error: 'workspace_index output was not valid JSON', stdout, stderr };
  }
}

/**
 * TypeScript-native action handlers. These operate directly on
 * `.wqm-fork/indexed-projects.json` via the indexed-projects-registry module
 * and don't require PowerShell, so they work inside the dockerized MCP
 * container as well as on Windows hosts.
 *
 * Schema (field names, enum values, JSON shape) matches the legacy PS1
 * implementation byte-for-byte.
 */
const TS_NATIVE_ACTIONS = new Set([
  'init',
  'list_projects',
  'list_branches',
  'agent_branch_status',
  'start_agent_branch',
  'finish_agent_branch',
  'abandon_agent_branch',
  // Phase 2: observation/status surface — runs in the dockerized MCP
  // container without PowerShell. wqm/git absence is handled gracefully
  // (stub probe results) rather than failing the call.
  'project_status',
  'status_all',
  'observe_project',
  'observe_all',
  'incremental_check',
  'incremental_check_all',
]);

function projectSelectorFromArgs(args: JsonObject): {
  projectName?: string;
  projectId?: string;
  projectDir?: string;
} {
  const name = stringArg(args, 'projectName', ['name']);
  const id = stringArg(args, 'projectId');
  const dir = stringArg(args, 'projectPath', ['projectDir']);
  return {
    ...(name !== undefined ? { projectName: name } : {}),
    ...(id !== undefined ? { projectId: id } : {}),
    ...(dir !== undefined ? { projectDir: dir } : {}),
  };
}

function buildStartAgentBranchArgs(
  base: ProjectArgs,
  args: JsonObject,
  branchName: string
): StartAgentBranchArgs {
  const baseBranch = stringArg(args, 'baseBranch');
  const returnBranch = stringArg(args, 'returnBranch');
  const worktreePath = stringArg(args, 'worktreePath', ['worktree']);
  const worktreeRoot = stringArg(args, 'worktreeRoot');
  const purpose = stringArg(args, 'purpose');
  const createdBy = stringArg(args, 'createdBy');
  const useWtRaw = boolArg(args, 'useWorktree');
  const useWorktree =
    useWtRaw === 'true' || useWtRaw === '1' || useWtRaw === 'yes' || useWtRaw === 'y';
  return {
    ...base,
    branchName,
    useWorktree,
    ...(baseBranch !== undefined ? { baseBranch } : {}),
    ...(returnBranch !== undefined ? { returnBranch } : {}),
    ...(worktreePath !== undefined ? { worktreePath } : {}),
    ...(worktreeRoot !== undefined ? { worktreeRoot } : {}),
    ...(purpose !== undefined ? { purpose } : {}),
    ...(createdBy !== undefined ? { createdBy } : {}),
  };
}

function dispatchTsAction(
  action: string,
  args: JsonObject,
  repoDir: string,
  daemonClient: DaemonClient | undefined
): unknown | Promise<unknown> {
  const registryPath = stringArg(args, 'registryPath') ?? defaultRegistryPath(repoDir);
  const base: BaseArgs = { registryPath };
  const projectSel = projectSelectorFromArgs(args);
  const projectArgs: ProjectArgs = { ...base, ...projectSel };

  switch (action) {
    case 'init':
      return runInit(base);
    case 'list_projects':
      // Pass the daemon client so the listing can surface projects the daemon
      // indexes but that aren't in indexed-projects.json (eval item #5).
      return runListProjects(base, daemonClient);
    case 'list_branches':
      return runListBranches(projectArgs);
    case 'agent_branch_status': {
      const branchName = stringArg(args, 'branchName', ['branch']);
      if (!branchName) throw new Error('branchName obrigatorio');
      const arg: BranchArgs = { ...projectArgs, branchName };
      return runAgentBranchStatus(arg);
    }
    case 'start_agent_branch': {
      const branchName = stringArg(args, 'branchName', ['branch']);
      if (!branchName) throw new Error('branchName obrigatorio');
      return runStartAgentBranch(buildStartAgentBranchArgs(projectArgs, args, branchName));
    }
    case 'finish_agent_branch': {
      const branchName = stringArg(args, 'branchName', ['branch']);
      if (!branchName) throw new Error('branchName obrigatorio');
      const arg: BranchArgs = { ...projectArgs, branchName };
      return runFinishAgentBranch(arg);
    }
    case 'abandon_agent_branch': {
      const branchName = stringArg(args, 'branchName', ['branch']);
      if (!branchName) throw new Error('branchName obrigatorio');
      const rmWtRaw = boolArg(args, 'removeWorktree');
      const removeWorktree =
        rmWtRaw === 'true' || rmWtRaw === '1' || rmWtRaw === 'yes' || rmWtRaw === 'y';
      const arg: AbandonAgentBranchArgs = { ...projectArgs, branchName, removeWorktree };
      return runAbandonAgentBranch(arg);
    }
    // ── Phase 2: observation/status surface ─────────────────────────────
    case 'project_status':
      // Prefer the daemon-direct status (source of truth; works in the
      // container, where the `.wqm-fork/indexed-projects.json` registry the
      // PowerShell path reads is empty → "project not found" for a valid
      // tenant). Fall back to the registry/PowerShell path only without a daemon.
      return daemonClient
        ? handleIndexingStatus(args, daemonClient, 'project_status')
        : runProjectStatus(projectArgs, daemonClient);
    case 'status_all':
      return runStatusAll(base, daemonClient);
    case 'observe_project':
      return runObserveProject(projectArgs, daemonClient);
    case 'observe_all':
      return runObserveAll(base, daemonClient);
    case 'incremental_check':
      return runIncrementalCheck(projectArgs, daemonClient);
    case 'incremental_check_all':
      return runIncrementalCheckAll(base, daemonClient);
    default:
      throw new Error(`TS-native handler missing for action: ${action}`);
  }
}

export async function handleWorkspaceIndex(
  rawArgs: Record<string, unknown> | undefined,
  daemonClient?: DaemonClient
): Promise<unknown> {
  const args = rawArgs ?? {};

  // Native-TS action: runs inside the dockerized MCP container without
  // requiring PowerShell or git on the container. Reaches the daemon over
  // gRPC, identical path used by session-start registration.
  const action = stringArg(args, 'action');
  if (action === 'sync_current_branch') {
    if (!daemonClient) {
      throw new Error(
        'sync_current_branch requires a connected daemon client (gRPC unavailable)'
      );
    }
    return handleSyncCurrentBranch(args, daemonClient);
  }

  if (action === 'indexing_status') {
    if (!daemonClient) {
      throw new Error(
        'indexing_status requires a connected daemon client (gRPC unavailable)'
      );
    }
    return handleIndexingStatus(args, daemonClient);
  }

  // TS-native registry actions (Phases 1 + 2 ports from
  // indexed-projects-registry.ps1). These run inside the container; they
  // only need a writable .wqm-fork/indexed-projects.json on disk and a
  // `git` CLI in PATH. The wqm CLI is optional — when missing, probe
  // results carry a structured "not installed" stub instead of failing.
  if (action && TS_NATIVE_ACTIONS.has(action)) {
    assertMutationAllowed(action, args);
    const repoDir = resolveRepoDir(args);
    return dispatchTsAction(action, args, repoDir, daemonClient);
  }

  // PowerShell-backed actions: require the workspace-qdrant-mcp checkout to
  // be on the same filesystem (host deployment only).
  const repoDir = resolveRepoDir(args);
  const bridgePath = resolve(repoDir, 'scripts', 'windows', 'workspace-index-mcp.ps1');
  if (!existsSync(bridgePath)) {
    throw new Error(`workspace_index bridge script not found: ${bridgePath}`);
  }

  const psExe = process.env['WQM_POWERSHELL'] ?? 'pwsh';
  const scriptArgs = buildScriptArgs(args, repoDir);

  return new Promise((resolvePromise, reject) => {
    const child = spawn(psExe, scriptArgs, { cwd: repoDir, env: process.env });
    let stdout = '';
    let stderr = '';

    child.stdout.on('data', (chunk: Buffer) => {
      stdout += chunk.toString('utf8');
    });
    child.stderr.on('data', (chunk: Buffer) => {
      stderr += chunk.toString('utf8');
    });
    child.on('error', (error) => {
      reject(error);
    });
    child.on('close', (code) => {
      const parsed = parseOutput(stdout, stderr);
      if (code === 0) {
        resolvePromise(parsed);
        return;
      }
      reject(new Error(`workspace_index failed with exit code ${code}: ${stderr || stdout}`));
    });
  });
}
