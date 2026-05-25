/**
 * workspace_index MCP tool implementation.
 *
 * The registry/worktree logic lives in the Windows PowerShell helper scripts
 * shipped by the fork kit. This TypeScript layer validates the double opt-in
 * for mutating actions, runs the bridge script, and returns JSON.
 */

import { spawn } from 'node:child_process';
import { existsSync } from 'node:fs';
import { resolve } from 'node:path';

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

function stringArg(args: JsonObject, key: string): string | undefined {
  const value = args[key];
  return typeof value === 'string' && value.trim().length > 0 ? value : undefined;
}

function boolArg(args: JsonObject, key: string): string | undefined {
  const value = args[key];
  return typeof value === 'boolean' ? String(value) : undefined;
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

function resolveRepoDir(args: JsonObject): string {
  return resolve(
    stringArg(args, 'repoDir') ?? process.env['WQM_REPO_DIR'] ?? process.cwd()
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
    ['-ProjectDir', stringArg(args, 'projectPath')],
    ['-BranchName', stringArg(args, 'branchName')],
    ['-BaseBranch', stringArg(args, 'baseBranch')],
    ['-ReturnBranch', stringArg(args, 'returnBranch')],
    ['-WorktreePath', stringArg(args, 'worktreePath')],
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
  rawArgs: Record<string, unknown> | undefined
): Promise<unknown> {
  const args = rawArgs ?? {};
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
