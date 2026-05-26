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
import { getGitState } from './../utils/git-utils.js';

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
  if (!repoDir) {
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
  const hookName = stringArg(args, 'hookName', ['hook_name']) ?? 'manual';
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

  try {
    const response = await daemonClient.registerProject({
      path: watchPath,
      project_id: '',
      name: projectName,
      register_if_new: true,
      priority: 'high',
      ...(gitRemote ? { git_remote: gitRemote } : {}),
    });

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
