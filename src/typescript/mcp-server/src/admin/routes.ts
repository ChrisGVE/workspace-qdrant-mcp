/**
 * Admin REST routes mounted under `/admin/api/*`.
 *
 * All routes return JSON, use the same Bearer auth as the MCP transport,
 * and never touch the MCP protocol surface. Daemon calls go through the
 * existing `DaemonClient`; SQLite reads go through `SqliteStateManager`
 * so the admin code never opens its own DB handle.
 */

import type { IncomingMessage, ServerResponse } from 'node:http';
import { basename } from 'node:path';

import type { DaemonClient } from '../clients/daemon-client.js';
import type { SqliteStateManager } from '../clients/sqlite-state-manager.js';
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
  let registered: Array<{
    tenantId: string;
    path: string;
    remoteUrl: string;
    isActive: boolean;
    lastActivityAt: string | null;
  }> = [];
  try {
    const listResp = await daemonClient.listProjects({});
    registered = (listResp.projects ?? []).map((p) => ({
      tenantId: p.project_id,
      path: p.project_root,
      remoteUrl: '',
      isActive: p.is_active,
      lastActivityAt: p.last_active
        ? new Date(p.last_active.seconds * 1000).toISOString()
        : null,
    }));
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

// ── Route table ──────────────────────────────────────────────────────────────

type Route = { method: string; path: string; handler: RouteHandler };

const ROUTES: ReadonlyArray<Route> = [
  { method: 'GET', path: '/admin/api/snapshot', handler: handleSnapshot },
  { method: 'POST', path: '/admin/api/projects/scan', handler: handleScan },
  { method: 'POST', path: '/admin/api/projects/register', handler: handleRegister },
  { method: 'POST', path: '/admin/api/projects/deregister', handler: handleDeregister },
  { method: 'GET', path: '/admin/api/settings', handler: handleGetSettings },
  { method: 'PUT', path: '/admin/api/settings', handler: handlePutSettings },
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
