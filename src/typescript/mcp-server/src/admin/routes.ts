/**
 * Admin REST routes mounted under `/admin/api/*`.
 *
 * All routes return JSON, use the same Bearer auth as the MCP transport,
 * and never touch the MCP protocol surface. Daemon calls go through the
 * existing `DaemonClient`; SQLite reads go through `SqliteStateManager`
 * so the admin code never opens its own DB handle.
 */

import { spawn } from 'node:child_process';
import { existsSync, readdirSync, readFileSync, statSync, writeFileSync } from 'node:fs';
import type { IncomingMessage, ServerResponse } from 'node:http';
import { basename, dirname, join, resolve as resolvePath } from 'node:path';

import type { AuthConfig } from '../auth-middleware.js';
import type { DaemonClient } from '../clients/daemon-client.js';
import type { SqliteStateManager } from '../clients/sqlite-state-manager.js';
import mcpPublicConfig from '../constants/mcp-public-config.json' with { type: 'json' };
import { DEFAULT_HTTP_PORT } from '../server-types.js';
import { logError, logInfo } from '../utils/logger.js';

import { scanForGitProjects, type ProjectCandidate } from './discovery.js';
import {
  loadSettings,
  saveSettings,
  setProjectApproval,
  updateSettings,
  type AdminSettings,
} from './settings-store.js';

export interface AdminDeps {
  daemonClient: DaemonClient;
  stateManager: SqliteStateManager;
  authConfig: AuthConfig;
}

interface RouteHandler {
  (req: IncomingMessage, res: ServerResponse, deps: AdminDeps): Promise<void>;
}

// ── Helpers ──────────────────────────────────────────────────────────────────

function writeJson(res: ServerResponse, status: number, body: unknown): void {
  res.writeHead(status, { 'Content-Type': 'application/json; charset=utf-8' });
  res.end(JSON.stringify(body));
}

function writeError(res: ServerResponse, status: number, message: string, detail?: unknown): void {
  writeJson(res, status, { error: message, detail: detail ?? null });
}

async function readJsonBody(req: IncomingMessage): Promise<Record<string, unknown>> {
  const chunks: Buffer[] = [];
  for await (const chunk of req) {
    chunks.push(chunk as Buffer);
  }
  if (chunks.length === 0) return {};
  const raw = Buffer.concat(chunks).toString('utf-8');
  if (!raw.trim()) return {};
  try {
    const parsed = JSON.parse(raw) as unknown;
    return typeof parsed === 'object' && parsed !== null
      ? (parsed as Record<string, unknown>)
      : {};
  } catch {
    return {};
  }
}

// ── /api/snapshot — consolidated real-time state for the dashboard ──────────

const handleSnapshot: RouteHandler = async (_req, res, { daemonClient, stateManager }) => {
  const settings = loadSettings();
  const snapshotAt = new Date().toISOString();

  // Daemon health + status. Each call has its own timeout via the gRPC
  // client; we tolerate failures and surface them in the JSON.
  let daemon: unknown = { ok: false, reason: 'unknown' };
  try {
    const status = await daemonClient.getStatus();
    daemon = {
      ok: true,
      activeProjects: status.active_projects ?? [],
      totalDocuments: status.total_documents ?? 0,
      totalCollections: status.total_collections ?? 0,
      uptimeSince: status.uptime_since ?? null,
      metrics: status.metrics ?? null,
    };
  } catch (error) {
    daemon = {
      ok: false,
      reason: error instanceof Error ? error.message : String(error),
    };
  }

  // SQLite queue stats. Local read; if the MCP container is missing
  // the bind-mounted state.db (Docker Desktop: 9P doesn't honor SQLite
  // WAL locks → SQLITE_CANTOPEN), we report zeros rather than crashing
  // the whole snapshot. `getQueueStats` re-throws on non-"no such table"
  // errors, so we need an explicit try/catch around it here.
  let queueStats: unknown = { pending: 0, in_progress: 0, completed: 0, failed: 0 };
  try {
    const queueResult = stateManager.getQueueStats();
    if (queueResult.status === 'ok' && queueResult.data) {
      queueStats = queueResult.data;
    }
  } catch {
    // Already initialized to zeros above.
  }

  // Project inventory comes from the daemon's `ListProjects` RPC rather
  // than direct SQLite reads. The daemon owns the DB file (ADR-003)
  // and is the only process that can read it through bind mounts on
  // Docker Desktop. gRPC avoids the SQLITE_CANTOPEN failures we'd hit
  // pointing better-sqlite3 at a 9P-mounted state.db.
  interface RegisteredProject {
    tenantId: string;
    path: string;
    remoteUrl: string;
    isActive: boolean;
    lastActivityAt: string | null;
    indexing: {
      pending: number;
      in_progress: number;
      failed: number;
      done: number;
      total: number;
      percent: number;
      eta_seconds?: number;
    } | null;
  }
  let registered: RegisteredProject[] = [];
  try {
    const listResp = await daemonClient.listProjects({});
    const base: RegisteredProject[] = (listResp.projects ?? []).map((p) => ({
      tenantId: p.project_id,
      path: p.project_root,
      remoteUrl: '',
      isActive: p.is_active,
      lastActivityAt: p.last_active
        ? new Date(p.last_active.seconds * 1000).toISOString()
        : null,
      indexing: null,
    }));

    // Enrich each project with per-tenant indexing progress, in parallel.
    // Falls back to `null` on per-project error so the dashboard can render
    // the row without the progress bar instead of blanking the whole page.
    const progressResults = await Promise.allSettled(
      base.map((proj) => daemonClient.getProjectStatus({ project_id: proj.tenantId }))
    );
    progressResults.forEach((result, idx) => {
      const proj = base[idx];
      if (!proj || result.status !== 'fulfilled' || !result.value.found) return;
      const s = result.value;
      const indexing: NonNullable<RegisteredProject['indexing']> = {
        pending: s.pending_count ?? 0,
        in_progress: s.in_progress_count ?? 0,
        failed: s.failed_count ?? 0,
        done: s.done_count ?? 0,
        total: s.total_count ?? 0,
        percent: s.percent_complete ?? 100,
      };
      if (typeof s.eta_seconds === 'number') indexing.eta_seconds = s.eta_seconds;
      proj.indexing = indexing;
    });
    registered = base;
  } catch {
    // Daemon offline / not yet started — leave the list empty.
    registered = [];
  }

  writeJson(res, 200, {
    snapshotAt,
    settings,
    daemon,
    queue: queueStats,
    projects: {
      registered,
      registeredCount: registered.length,
      approvedCount: settings.approvedProjects.length,
    },
  });
};

// ── /api/projects/scan — runs discovery, returns candidates ─────────────────

const handleScan: RouteHandler = async (req, res) => {
  const body = await readJsonBody(req);
  const overrideRoot = typeof body['devRoot'] === 'string' ? (body['devRoot'] as string) : null;
  const overrideDepth =
    typeof body['scanDepth'] === 'number' ? (body['scanDepth'] as number) : null;

  const currentSettings = loadSettings();
  const devRoot = overrideRoot ?? currentSettings.devRoot;
  const scanDepth = overrideDepth ?? currentSettings.scanDepth;

  if (!devRoot) {
    writeError(res, 400, 'devRoot not configured', {
      hint: 'PUT /admin/api/settings with { devRoot: "<absolute path>" } first.',
    });
    return;
  }

  const result = scanForGitProjects(devRoot, scanDepth);

  // Persist the latest scan timestamp + (optionally) the override so the
  // user does not have to re-enter it next time.
  const patch: Partial<AdminSettings> = { lastScanAt: result.finishedAt };
  if (overrideRoot) patch.devRoot = overrideRoot;
  if (overrideDepth) patch.scanDepth = overrideDepth;
  const settings = updateSettings(patch);

  writeJson(res, 200, { scan: result, settings });
};

// ── /api/projects/register — register one or more candidates ────────────────

interface RegisterRequest {
  path: string;
  /** If true, sends `register_if_new=true` so the daemon enqueues new tenants. */
  registerIfNew?: boolean;
}

const handleRegister: RouteHandler = async (req, res, { daemonClient }) => {
  const body = (await readJsonBody(req)) as Partial<RegisterRequest>;
  const path = body.path;
  if (!path || typeof path !== 'string') {
    writeError(res, 400, 'path required');
    return;
  }

  try {
    const response = await daemonClient.registerProject({
      path,
      project_id: '',
      name: basename(path) || 'project',
      register_if_new: body.registerIfNew !== false,
      priority: 'high',
    });
    setProjectApproval(path, true);
    writeJson(res, 200, {
      ok: true,
      projectId: response.project_id,
      created: response.created,
      newlyRegistered: response.newly_registered,
      isActive: response.is_active,
      isWorktree: response.is_worktree ?? false,
      watchPath: response.watch_path ?? null,
    });
  } catch (error) {
    logError('admin register failed', error, { path });
    writeError(res, 502, 'register failed', error instanceof Error ? error.message : String(error));
  }
};

const handleDeregister: RouteHandler = async (req, res, { daemonClient }) => {
  const body = (await readJsonBody(req)) as Partial<{ projectId: string; path: string }>;
  const projectId = body.projectId;
  if (!projectId || typeof projectId !== 'string') {
    writeError(res, 400, 'projectId required');
    return;
  }
  try {
    const response = await daemonClient.deprioritizeProject({
      project_id: projectId,
      ...(body.path ? { watch_path: body.path } : {}),
    });
    if (body.path) setProjectApproval(body.path, false);
    writeJson(res, 200, {
      ok: true,
      isActive: response.is_active,
      newPriority: response.new_priority,
    });
  } catch (error) {
    logError('admin deregister failed', error, { projectId });
    writeError(res, 502, 'deregister failed', error instanceof Error ? error.message : String(error));
  }
};

// ── /api/health — aggregate health snapshot for the dashboard ───────────────

/**
 * Find the workspace-qdrant repo root visible to this process.
 *
 * On dockerized deployments WQM_PROJECT_ROOT points at the bind-mounted
 * *dev root* (parent directory holding several repos) rather than at the
 * workspace-qdrant checkout itself, so we (a) descend into immediate
 * children looking for `scripts/git-hooks/install.sh`, then (b) walk up
 * from cwd as a fallback for host-native invocations.
 */
function resolveRepoRoot(): string | null {
  const projectRoot = process.env['WQM_PROJECT_ROOT'];
  if (projectRoot && existsSync(projectRoot)) {
    const abs = resolvePath(projectRoot);
    // Direct: WQM_PROJECT_ROOT *is* the repo root.
    if (existsSync(join(abs, 'scripts', 'git-hooks', 'install.sh'))) return abs;
    // One level down: WQM_PROJECT_ROOT is the dev root, find the checkout.
    try {
      for (const entry of readdirSync(abs, { withFileTypes: true })) {
        if (!entry.isDirectory()) continue;
        const candidate = join(abs, entry.name);
        if (existsSync(join(candidate, 'scripts', 'git-hooks', 'install.sh'))) return candidate;
      }
    } catch {
      // ignore — permissions or transient FS errors
    }
  }
  // Host-native fallback: walk up from cwd.
  let cur = resolvePath(process.cwd());
  for (let i = 0; i < 8; i++) {
    if (existsSync(join(cur, 'scripts', 'git-hooks', 'install.sh'))) return cur;
    const parent = resolvePath(cur, '..');
    if (parent === cur) break;
    cur = parent;
  }
  return null;
}

interface HooksHealth {
  ok: boolean;
  kind: 'posix' | 'powershell' | 'mixed' | 'none';
  path: string | null;
  installed: string[];
  legacyArtifacts: string[];
}

function probeHooks(repoRoot: string | null): HooksHealth {
  if (!repoRoot) return { ok: false, kind: 'none', path: null, installed: [], legacyArtifacts: [] };
  const hooksDir = join(repoRoot, '.wqm-fork', 'git-hooks');
  if (!existsSync(hooksDir)) {
    return { ok: false, kind: 'none', path: hooksDir, installed: [], legacyArtifacts: [] };
  }
  const HOOK_NAMES = ['post-checkout', 'post-commit', 'post-merge', 'post-rewrite', 'post-worktree-add'];
  const installed: string[] = [];
  let sawPosix = false;
  let sawPs = false;
  for (const name of HOOK_NAMES) {
    const file = join(hooksDir, name);
    if (!existsSync(file)) continue;
    installed.push(name);
    try {
      const content = readFileSync(file, 'utf8');
      if (content.includes('# WQM_SYNC_BRANCH_HOOK')) sawPosix = true;
      else if (content.includes('wqm-git-hook.ps1') || content.includes('powershell.exe')) sawPs = true;
    } catch {
      // ignore read errors
    }
  }
  const legacyArtifacts: string[] = [];
  if (existsSync(join(hooksDir, 'wqm-git-hook.ps1'))) legacyArtifacts.push('wqm-git-hook.ps1');
  const backupPattern = /^git-hooks\.backup-/;
  try {
    for (const entry of readdirSync(join(repoRoot, '.wqm-fork'))) {
      if (backupPattern.test(entry)) legacyArtifacts.push(entry);
    }
  } catch {
    // ignore
  }
  let kind: HooksHealth['kind'] = 'none';
  if (sawPosix && sawPs) kind = 'mixed';
  else if (sawPosix) kind = 'posix';
  else if (sawPs) kind = 'powershell';
  return {
    ok: kind === 'posix' && installed.length === HOOK_NAMES.length,
    kind,
    path: hooksDir,
    installed,
    legacyArtifacts,
  };
}

const handleHealth: RouteHandler = async (_req, res, { daemonClient }) => {
  const repoRoot = resolveRepoRoot();
  const hooks = probeHooks(repoRoot);

  let daemon: unknown = { ok: false, reason: 'unknown' };
  try {
    const status = await daemonClient.getStatus();
    daemon = {
      ok: true,
      activeProjects: (status.active_projects ?? []).length,
      totalDocuments: status.total_documents ?? 0,
      totalCollections: status.total_collections ?? 0,
      uptimeSince: status.uptime_since ?? null,
    };
  } catch (error) {
    daemon = { ok: false, reason: error instanceof Error ? error.message : String(error) };
  }

  let qdrant: unknown = { ok: false };
  try {
    const url = process.env['QDRANT_URL'] ?? 'http://localhost:6333';
    const probe = await fetch(`${url.replace(/\/$/, '')}/collections`, {
      signal: AbortSignal.timeout(3000),
    });
    qdrant = { ok: probe.ok, statusCode: probe.status, endpoint: `${url}/collections` };
  } catch (error) {
    qdrant = { ok: false, reason: error instanceof Error ? error.message : String(error) };
  }

  const mcp = {
    version: process.env['npm_package_version'] ?? null,
    repoRoot,
    mode: process.env['MCP_SERVER_MODE'] ?? null,
    pid: process.pid,
    uptimeSeconds: Math.round(process.uptime()),
  };

  writeJson(res, 200, { hooks, daemon, qdrant, mcp });
};

// ── /api/hooks/install — re-install the POSIX git hooks ─────────────────────

/**
 * Run `scripts/git-hooks/install.sh --force` against the configured repo.
 * Returns stdout/stderr/exit code so the UI can show what happened. The
 * server doesn't reach out to the host directly — it shells out from
 * wherever the MCP process happens to be running. On dockerized
 * deployments that means the install.sh visible at WQM_PROJECT_ROOT
 * inside the container, which is the same file the host edits (bind
 * mount, same path on both sides).
 */
const handleInstallHooks: RouteHandler = async (req, res) => {
  const repoRoot = resolveRepoRoot();
  if (!repoRoot) {
    writeError(res, 500, 'repo root not found', {
      hint: 'Set WQM_PROJECT_ROOT to the repo root or run from inside the checkout.',
    });
    return;
  }
  const script = join(repoRoot, 'scripts', 'git-hooks', 'install.sh');
  if (!existsSync(script)) {
    writeError(res, 500, 'install.sh not found', { script });
    return;
  }
  const body = (await readJsonBody(req)) as Partial<{ force: boolean }>;
  const useForce = body.force !== false;

  const args = [
    script,
    '--repo',
    repoRoot,
    '--hooks-dir',
    join(repoRoot, '.wqm-fork', 'git-hooks'),
  ];
  // When the MCP runs in a container, install.sh sees its own path under the
  // bind mount (e.g. `/run/desktop/...`), but the generated hooks must embed
  // a path the host shell can execute. Forwarding the dev-root pair lets
  // install.sh translate the prefix before writing the hooks.
  const hostDevRoot = process.env['WQM_HOST_DEV_ROOT'];
  const containerDevRoot = process.env['WQM_DEV_ROOT'];
  if (hostDevRoot && containerDevRoot) {
    args.push('--host-dev-root', hostDevRoot, '--container-dev-root', containerDevRoot);
  }
  if (useForce) args.push('--force');

  await new Promise<void>((resolveOp) => {
    const child = spawn('sh', args, { cwd: repoRoot, env: process.env });
    let stdout = '';
    let stderr = '';
    child.stdout.on('data', (chunk: Buffer) => {
      stdout += chunk.toString('utf8');
    });
    child.stderr.on('data', (chunk: Buffer) => {
      stderr += chunk.toString('utf8');
    });
    child.on('error', (err) => {
      logError('admin hooks install spawn failed', err, { script });
      writeError(res, 500, 'spawn failed', err.message);
      resolveOp();
    });
    child.on('close', (code) => {
      const ok = code === 0;
      logInfo('admin hooks install finished', { code, force: useForce });
      writeJson(res, ok ? 200 : 500, { ok, exitCode: code, stdout, stderr, force: useForce });
      resolveOp();
    });
  });
};

// ── /api/ignore/global — read + write global.wqmignore ──────────────────────

/**
 * Resolve the directory that holds daemon state files (global.wqmignore,
 * memexd.db, …). Derived from `WQM_DATABASE_PATH` so both the daemon and
 * the MCP server agree on the location regardless of deployment mode.
 */
function getStateDirForIgnore(): string | null {
  const dbPath = process.env['WQM_DATABASE_PATH'];
  if (!dbPath) return null;
  return dirname(dbPath);
}

const handleGetGlobalIgnore: RouteHandler = async (_req, res) => {
  const stateDir = getStateDirForIgnore();
  if (!stateDir) {
    writeError(res, 500, 'WQM_DATABASE_PATH not set — cannot locate global.wqmignore');
    return;
  }
  const ignoreFile = join(stateDir, 'global.wqmignore');
  const content = existsSync(ignoreFile) ? readFileSync(ignoreFile, 'utf-8') : '';
  writeJson(res, 200, { content, path: ignoreFile });
};

const handlePutGlobalIgnore: RouteHandler = async (req, res) => {
  const body = await readJsonBody(req);
  if (typeof body['content'] !== 'string') {
    writeError(res, 400, '`content` (string) is required');
    return;
  }
  const content = body['content'] as string;
  const stateDir = getStateDirForIgnore();
  if (!stateDir) {
    writeError(res, 500, 'WQM_DATABASE_PATH not set — cannot locate global.wqmignore');
    return;
  }
  const ignoreFile = join(stateDir, 'global.wqmignore');
  try {
    writeFileSync(ignoreFile, content, 'utf-8');
  } catch (err) {
    logError('admin global.wqmignore write failed', err, { ignoreFile });
    writeError(res, 500, 'write failed', err instanceof Error ? err.message : String(err));
    return;
  }
  logInfo('admin global.wqmignore updated', { bytes: content.length, path: ignoreFile });
  writeJson(res, 200, { ok: true, bytes: content.length, path: ignoreFile });
};

// ── /api/settings — read + write the JSON settings file ─────────────────────

const handleGetSettings: RouteHandler = async (_req, res) => {
  writeJson(res, 200, loadSettings());
};

const handlePutSettings: RouteHandler = async (req, res) => {
  const body = (await readJsonBody(req)) as Partial<AdminSettings>;
  const next: AdminSettings = {
    ...loadSettings(),
    ...(typeof body.devRoot === 'string' ? { devRoot: body.devRoot } : {}),
    ...(typeof body.scanDepth === 'number' ? { scanDepth: body.scanDepth } : {}),
  };
  saveSettings(next);
  logInfo('admin settings updated', {
    devRoot: next.devRoot,
    scanDepth: next.scanDepth,
  });
  writeJson(res, 200, next);
};

// ── /api/config/clients — generate client config snippets ───────────────────

const handleGetClientConfigs: RouteHandler = async (_req, res) => {
  const port = process.env['MCP_HTTP_PORT'] ?? String(DEFAULT_HTTP_PORT);
  const token = process.env['MCP_HTTP_TOKEN'] ?? '';
  const mcpUrl = `http://localhost:${port}/mcp`;

  const claudeConfig = {
    mcpServers: {
      'workspace-qdrant': {
        command: 'npx',
        args: ['mcp-remote', mcpUrl, '--header', `Authorization: Bearer ${token}`],
      },
    },
  };

  const claudeHttpConfig = {
    mcpServers: {
      'workspace-qdrant': {
        url: mcpUrl,
        headers: { Authorization: `Bearer ${token}` },
      },
    },
  };

  // All Codex-related defaults (tool list + timeouts) come from
  // src/constants/mcp-public-config.json. Do not hardcode here — see file header.
  const toolsToml = mcpPublicConfig.publicTools.map((t) => `"${t}"`).join(', ');
  const codexConfig = [
    `# workspace-qdrant MCP`,
    `[mcp_servers.workspace-qdrant]`,
    `url = "${mcpUrl}"`,
    `bearer_token_env_var = "MCP_HTTP_TOKEN"`,
    `startup_timeout_sec = ${mcpPublicConfig.codex.startup_timeout_sec}`,
    `tool_timeout_sec = ${mcpPublicConfig.codex.tool_timeout_sec}`,
    `required = true`,
    `enabled_tools = [${toolsToml}]`,
  ].join('\n');

  writeJson(res, 200, {
    claudeDesktop: {
      mcp_remote: JSON.stringify(claudeConfig, null, 2),
      http_native: JSON.stringify(claudeHttpConfig, null, 2),
    },
    codex: codexConfig,
    mcpUrl,
    port,
    hasToken: !!token,
  });
};

// ── /api/logs/mcp — tail the MCP server JSONL log file ──────────────────────

const handleGetMcpLogs: RouteHandler = async (req, res) => {
  const url = new URL(req.url ?? '/', 'http://x');
  const lines = Math.min(parseInt(url.searchParams.get('lines') ?? '100', 10), 500);

  const logDir = process.env['WQM_LOG_DIR'] ?? null;
  if (!logDir || !existsSync(logDir)) {
    writeJson(res, 200, { lines: [], note: 'Log directory not found' });
    return;
  }

  let logFile: string | null = null;
  try {
    const files = readdirSync(logDir)
      .filter(f => f.startsWith('mcp-server') && f.endsWith('.jsonl'))
      .map(f => ({ name: f, mtime: statSync(join(logDir, f)).mtimeMs }))
      .sort((a, b) => b.mtime - a.mtime);
    if (files.length > 0 && files[0]) logFile = join(logDir, files[0].name);
  } catch { /* ignore */ }

  if (!logFile || !existsSync(logFile)) {
    writeJson(res, 200, { lines: [], note: 'No log file found' });
    return;
  }

  try {
    const content = readFileSync(logFile, 'utf-8');
    const allLines = content.split('\n').filter(l => l.trim());
    const tail = allLines.slice(-lines);
    const parsed = tail.map(l => {
      try { return JSON.parse(l); } catch { return { msg: l }; }
    });
    writeJson(res, 200, { lines: parsed, file: logFile, total: allLines.length });
  } catch (err) {
    writeError(res, 500, 'read failed', err instanceof Error ? err.message : String(err));
  }
};

// ── /api/daemon/raw-health — proxy to memexd metrics /health ────────────────

const handleDaemonRawHealth: RouteHandler = async (_req, res) => {
  const metricsHost = process.env['WQM_DAEMON_METRICS_HOST'] ?? 'memexd';
  const metricsPort = process.env['WQM_DAEMON_METRICS_PORT'] ?? '9091';
  try {
    const resp = await fetch(`http://${metricsHost}:${metricsPort}/health`, {
      signal: AbortSignal.timeout(3000),
    });
    const text = await resp.text();
    writeJson(res, 200, { ok: resp.ok, statusCode: resp.status, body: text });
  } catch (err) {
    writeJson(res, 200, { ok: false, reason: err instanceof Error ? err.message : String(err) });
  }
};

// ── Route table ──────────────────────────────────────────────────────────────

type Route = { method: string; path: string; handler: RouteHandler };

const ROUTES: ReadonlyArray<Route> = [
  { method: 'GET', path: '/admin/api/snapshot', handler: handleSnapshot },
  { method: 'GET', path: '/admin/api/health', handler: handleHealth },
  { method: 'POST', path: '/admin/api/projects/scan', handler: handleScan },
  { method: 'POST', path: '/admin/api/projects/register', handler: handleRegister },
  { method: 'POST', path: '/admin/api/projects/deregister', handler: handleDeregister },
  { method: 'POST', path: '/admin/api/hooks/install', handler: handleInstallHooks },
  { method: 'GET', path: '/admin/api/settings', handler: handleGetSettings },
  { method: 'PUT', path: '/admin/api/settings', handler: handlePutSettings },
  { method: 'GET', path: '/admin/api/ignore/global', handler: handleGetGlobalIgnore },
  { method: 'PUT', path: '/admin/api/ignore/global', handler: handlePutGlobalIgnore },
  { method: 'GET', path: '/admin/api/config/clients', handler: handleGetClientConfigs },
  { method: 'GET', path: '/admin/api/logs/mcp', handler: handleGetMcpLogs },
  { method: 'GET', path: '/admin/api/daemon/raw-health', handler: handleDaemonRawHealth },
];

/**
 * Dispatch one HTTP request to the matching admin route. Returns `true`
 * when the request was handled (so the HTTP server doesn't fall through
 * to the MCP transport), `false` when the path is not an admin route.
 */
export async function dispatchAdminApi(
  req: IncomingMessage,
  res: ServerResponse,
  urlPath: string,
  deps: AdminDeps
): Promise<boolean> {
  const match = ROUTES.find((r) => r.method === req.method && r.path === urlPath);
  if (!match) return false;

  try {
    await match.handler(req, res, deps);
  } catch (error) {
    logError('admin route handler crashed', error, { url: urlPath });
    if (!res.headersSent) {
      writeError(res, 500, 'internal error', error instanceof Error ? error.message : String(error));
    }
  }
  return true;
}

export type { ProjectCandidate };
