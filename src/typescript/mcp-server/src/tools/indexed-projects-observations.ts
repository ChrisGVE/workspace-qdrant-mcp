/**
 * indexed-projects-observations.ts — TS port of the observation/health-check
 * surface of `scripts/windows/indexed-projects-registry.ps1`.
 *
 * Used by the `project_status`, `status_all`, `observe_project`, `observe_all`,
 * `incremental_check`, and `incremental_check_all` workspace_index actions.
 *
 * Mirrors the PS implementation byte-for-byte:
 *   - same JSON field names and insertion order;
 *   - same fallback values when the `wqm` CLI is absent
 *     ("wqm CLI not installed on host" / "executable not found");
 *   - same sanitization rules for CLI output (strip ANSI, drop control chars
 *     except LF/CR, replace non-ASCII with '?').
 *
 * Designed to run inside the dockerized MCP container: `wqm` may be absent,
 * `git` is expected on PATH, network probes use stdlib `net`/`fetch`.
 */

import { spawnSync } from 'node:child_process';
import { appendFileSync, existsSync, mkdirSync, writeFileSync } from 'node:fs';
import { connect as netConnect } from 'node:net';
import { dirname, resolve } from 'node:path';

import type { RegistryBranch, RegistryProject } from './indexed-projects-registry.js';
import type { DaemonClient } from '../clients/daemon-client.js';
import { ServiceStatus } from '../clients/grpc-types.js';

// ── Time + small helpers ───────────────────────────────────────────────────

function utcNow(): string {
  return new Date().toISOString();
}

function safeBranchSlug(name: string): string {
  return name.replace(/[^A-Za-z0-9._-]+/g, '-').replace(/^-+|-+$/g, '');
}

function ensureDir(dirPath: string): void {
  if (!existsSync(dirPath)) {
    mkdirSync(dirPath, { recursive: true });
  }
}

// ── git snapshot ───────────────────────────────────────────────────────────

export interface GitSnapshot {
  ok: boolean;
  currentBranch: string | null;
  head: string | null;
  clean: boolean;
  dirtyCount: number;
  statusPreview: string;
  ahead: number | null;
  behind: number | null;
  base: string;
  error?: string;
}

interface CapturedGit {
  ok: boolean;
  stdout: string;
  stderr: string;
}

function runGitSync(repo: string, args: string[], timeoutMs = 5000): CapturedGit {
  const result = spawnSyncCaptured('git', ['-C', repo, ...args], undefined, timeoutMs);
  return { ok: result.exitCode === 0, stdout: result.stdout, stderr: result.stderr };
}

/**
 * Build the per-repo git snapshot used by `New-Observation` and
 * `agent-branch-status` in the PS reference. All git invocations are 5s.
 */
export function getGitSnapshot(repo: string, base = ''): GitSnapshot {
  // Repo path may not exist (e.g. stale registry entry) — caller checks first.
  const branchRes = runGitSync(repo, ['rev-parse', '--abbrev-ref', 'HEAD']);
  const branch = branchRes.ok ? branchRes.stdout.trim() : null;

  const headRes = runGitSync(repo, ['rev-parse', 'HEAD']);
  const head = headRes.ok ? headRes.stdout.trim() : null;

  const statusRes = runGitSync(repo, ['status', '--short', '--branch']);
  const statusLines = statusRes.ok
    ? statusRes.stdout.split('\n').filter((line) => line.length > 0)
    : [];
  const dirtyLines = statusLines.filter((line) => !line.startsWith('## '));
  const statusPreview = statusLines.slice(0, 20).join('\n');

  let ahead: number | null = null;
  let behind: number | null = null;
  if (base) {
    const ab = runGitSync(repo, ['rev-list', '--left-right', '--count', `${base}...HEAD`]);
    if (ab.ok) {
      const parts = ab.stdout.trim().split(/\s+/);
      if (parts.length >= 2) {
        const b = Number.parseInt(parts[0] ?? '', 10);
        const a = Number.parseInt(parts[1] ?? '', 10);
        if (Number.isFinite(b)) behind = b;
        if (Number.isFinite(a)) ahead = a;
      }
    }
  }

  const snap: GitSnapshot = {
    ok: statusRes.ok,
    currentBranch: branch,
    head,
    clean: dirtyLines.length === 0,
    dirtyCount: dirtyLines.length,
    statusPreview,
    ahead,
    behind,
    base,
  };
  if (!statusRes.ok && statusRes.stderr) snap.error = statusRes.stderr.trim();
  return snap;
}

// ── Health probes ──────────────────────────────────────────────────────────

export interface TcpProbe {
  ok: boolean;
  host: string;
  port: number;
  error?: string;
}

/**
 * Best-effort TCP connect within 2s. Matches PS Test-TcpEndpoint. Returns ok
 * even when the underlying socket errors out asynchronously — we only need to
 * know whether the daemon endpoint is reachable.
 */
export function testTcpEndpoint(endpoint: string | null | undefined): Promise<TcpProbe> {
  if (!endpoint) {
    return Promise.resolve({ ok: false, host: '', port: 0, error: 'endpoint not configured' });
  }
  const clean = endpoint.replace(/^https?:\/\//, '');
  const parts = clean.split(':');
  const host = parts[0] ?? '';
  const port = parts[1] ? Number.parseInt(parts[1], 10) || 50051 : 50051;

  return new Promise<TcpProbe>((resolvePromise) => {
    let settled = false;
    const settle = (result: TcpProbe): void => {
      if (settled) return;
      settled = true;
      try {
        socket.destroy();
      } catch {
        // ignore
      }
      resolvePromise(result);
    };
    const socket = netConnect({ host, port, family: 4 });
    socket.setTimeout(2000);
    socket.once('connect', () => settle({ ok: true, host, port }));
    socket.once('timeout', () => settle({ ok: false, host, port, error: 'timeout (2s)' }));
    socket.once('error', (err: Error) => settle({ ok: false, host, port, error: err.message }));
  });
}

export interface QdrantProbe {
  ok: boolean;
  statusCode?: number;
  endpoint: string;
  error?: string;
}

/**
 * HTTP GET <url>/collections within 5s. Matches PS Test-Qdrant.
 */
export async function testQdrant(url: string | null | undefined): Promise<QdrantProbe> {
  const base = (url ?? '').replace(/\/+$/, '');
  const endpoint = `${base}/collections`;
  if (!base) return { ok: false, endpoint, error: 'qdrantUrl not configured' };
  try {
    const resp = await fetch(endpoint, {
      method: 'GET',
      signal: AbortSignal.timeout(5000),
    });
    return { ok: resp.status >= 200 && resp.status < 300, statusCode: resp.status, endpoint };
  } catch (err) {
    return { ok: false, endpoint, error: (err as Error).message };
  }
}

// ── captured spawn (used by git snapshots) ─────────────────────────────────────


/**
 * Spawn a command and capture stdout/stderr, with a hard timeout. Used by the
 * git-snapshot helper (runGitSync); daemon health/queue/watch probes use gRPC.
 */
function spawnSyncCaptured(
  file: string,
  args: string[],
  cwd: string | undefined,
  timeoutMs: number
): { exitCode: number; stdout: string; stderr: string; timedOut: boolean } {
  // Sync helper so the rest of the observation builder stays linear. The wqm
  // CLI never produces enough output to overrun the buffer (max ~10kB per
  // invocation in practice), so a single 8MB buffer is fine.
  //
  // We avoid execFileSync because it throws on non-zero exit codes, which we
  // need to inspect (the PS bridge surfaces exitCode + stderr in the result).
  const res = spawnSync(file, args, {
    cwd,
    encoding: 'utf-8',
    timeout: timeoutMs,
    maxBuffer: 8 * 1024 * 1024,
    windowsHide: true,
  });
  return {
    exitCode: res.status ?? -1,
    stdout: res.stdout ?? '',
    stderr: res.stderr ?? '',
    timedOut: res.signal === 'SIGTERM',
  };
}


// ── Observation builder ────────────────────────────────────────────────────

export interface ObservationBranch {
  name: string;
  kind: string;
  status: string;
  path: string;
  baseBranch?: string;
  returnBranch?: string;
  git: GitSnapshot | { ok: false; error: string };
  lastIndexedCommit?: string | null;
  watchEnabled?: boolean;
}

/**
 * Daemon health snapshot sourced from SystemService.Health over gRPC.
 * `ok` means the RPC succeeded (daemon reachable); `status` carries the
 * daemon's own assessment (HEALTHY / DEGRADED / ...). On an unreachable
 * daemon `ok` is false and `error` holds the gRPC failure message.
 */
export interface DaemonHealthResult {
  ok: boolean;
  source: 'daemon-grpc';
  status?: string;
  components?: Array<{ component: string; status: string; message: string }>;
  error?: string;
}

/**
 * Queue-depth snapshot sourced from SystemService.GetQueueStats over gRPC.
 * The daemon reads its own SQLite (authoritative DB on the daemon volume), so
 * counts are correct regardless of where the MCP server runs.
 */
export interface DaemonQueueResult {
  ok: boolean;
  source: 'daemon-grpc';
  pending_count?: number;
  in_progress_count?: number;
  completed_count?: number;
  failed_count?: number;
  stale_items_count?: number;
  by_item_type?: Record<string, number>;
  by_collection?: Record<string, number>;
  error?: string;
}

export interface Observation {
  timestamp: string;
  project: string;
  root: string;
  qdrant: QdrantProbe;
  daemonTcp: TcpProbe;
  wqmHealth: DaemonHealthResult;
  queue: DaemonQueueResult;
  branches: ObservationBranch[];
}

function serviceStatusLabel(value: number): string {
  return ServiceStatus[value] ?? String(value);
}

/**
 * Probe daemon health over gRPC (SystemService.Health). Never throws — an
 * unreachable daemon yields `{ ok: false, error }` so the observation still
 * serialises cleanly.
 */
async function probeDaemonHealth(
  client: DaemonClient | null | undefined
): Promise<DaemonHealthResult> {
  if (!client) {
    return { ok: false, source: 'daemon-grpc', error: 'daemon gRPC client unavailable' };
  }
  try {
    const r = await client.healthCheck();
    return {
      ok: true,
      source: 'daemon-grpc',
      status: serviceStatusLabel(r.status),
      components: (r.components ?? []).map((c) => ({
        component: c.component_name,
        status: serviceStatusLabel(c.status),
        message: c.message,
      })),
    };
  } catch (err) {
    return {
      ok: false,
      source: 'daemon-grpc',
      error: err instanceof Error ? err.message : String(err),
    };
  }
}

/**
 * Probe queue depth over gRPC (SystemService.GetQueueStats). Never throws.
 */
export async function probeDaemonQueue(
  client: DaemonClient | null | undefined
): Promise<DaemonQueueResult> {
  if (!client) {
    return { ok: false, source: 'daemon-grpc', error: 'daemon gRPC client unavailable' };
  }
  try {
    const r = await client.getQueueStats();
    return {
      ok: true,
      source: 'daemon-grpc',
      pending_count: r.pending_count,
      in_progress_count: r.in_progress_count,
      completed_count: r.completed_count,
      failed_count: r.failed_count,
      stale_items_count: r.stale_items_count,
      by_item_type: r.by_item_type,
      by_collection: r.by_collection,
    };
  } catch (err) {
    return {
      ok: false,
      source: 'daemon-grpc',
      error: err instanceof Error ? err.message : String(err),
    };
  }
}

// ── Daemon project-status + watch-list probes (used by incremental_check) ──

export interface DaemonProjectStatusResult {
  ok: boolean;
  source: 'daemon-grpc';
  found?: boolean;
  project_id?: string;
  is_active?: boolean;
  pending_count?: number;
  in_progress_count?: number;
  failed_count?: number;
  done_count?: number;
  total_count?: number;
  percent_complete?: number;
  error?: string;
}

/**
 * Probe per-project indexing status over gRPC (ProjectService.GetProjectStatus).
 * Replaces `wqm project status` / `wqm project check` — the pending/in_progress/
 * done counts convey what still needs indexing. Never throws.
 */
export async function probeDaemonProjectStatus(
  client: DaemonClient | null | undefined,
  projectId: string | undefined
): Promise<DaemonProjectStatusResult> {
  if (!client) {
    return { ok: false, source: 'daemon-grpc', error: 'daemon gRPC client unavailable' };
  }
  if (!projectId) {
    return {
      ok: false,
      source: 'daemon-grpc',
      error: 'no project_id (project has no registered tenant id)',
    };
  }
  try {
    const r = await client.getProjectStatus({ project_id: projectId });
    return {
      ok: true,
      source: 'daemon-grpc',
      found: r.found,
      project_id: r.project_id,
      is_active: r.is_active,
      // int64 count fields arrive as strings (proto-loader longs:String); coerce
      // so the result honors the `number` type instead of leaking "0"/"159".
      pending_count: Number(r.pending_count ?? 0),
      in_progress_count: Number(r.in_progress_count ?? 0),
      failed_count: Number(r.failed_count ?? 0),
      done_count: Number(r.done_count ?? 0),
      total_count: Number(r.total_count ?? 0),
      percent_complete: Number(r.percent_complete ?? 0),
    };
  } catch (err) {
    return {
      ok: false,
      source: 'daemon-grpc',
      error: err instanceof Error ? err.message : String(err),
    };
  }
}

export interface DaemonWatchInfo {
  watch_id: string;
  path: string;
  collection: string;
  tenant_id: string;
  enabled: boolean;
  is_active: boolean;
  is_paused: boolean;
  is_archived: boolean;
}

export interface DaemonWatchListResult {
  ok: boolean;
  source: 'daemon-grpc';
  total_count?: number;
  error?: string;
  watches?: DaemonWatchInfo[];
}

/**
 * Probe watched folders over gRPC (ProjectService.ListWatches). Replaces
 * `wqm watch list`. Never throws.
 */
export async function probeDaemonWatches(
  client: DaemonClient | null | undefined,
  collection?: string
): Promise<DaemonWatchListResult> {
  if (!client) {
    return { ok: false, source: 'daemon-grpc', error: 'daemon gRPC client unavailable' };
  }
  try {
    const r = await client.listWatches(collection ? { collection } : {});
    return {
      ok: true,
      source: 'daemon-grpc',
      total_count: r.total_count,
      watches: (r.watches ?? []).map((w) => ({
        watch_id: w.watch_id,
        path: w.path,
        collection: w.collection,
        tenant_id: w.tenant_id,
        enabled: w.enabled,
        is_active: w.is_active,
        is_paused: w.is_paused,
        is_archived: w.is_archived,
      })),
    };
  } catch (err) {
    return {
      ok: false,
      source: 'daemon-grpc',
      error: err instanceof Error ? err.message : String(err),
    };
  }
}

/**
 * Build the per-project observation. The `wqmHealth` and `queue` fields are
 * sourced from the daemon over gRPC (SystemService.Health / GetQueueStats)
 * instead of shelling out to the `wqm` CLI — so the observation is correct
 * inside the dockerized MCP container, which has no wqm binary and (for the
 * queue) no direct access to the daemon's SQLite volume. Field set/order is
 * otherwise unchanged from the PS New-Observation surface.
 */
export async function newObservation(
  project: RegistryProject,
  daemonClient: DaemonClient | null | undefined
): Promise<Observation> {
  const root = project.root;

  // Per-branch git snapshots. PS uses the branch.baseBranch when computing
  // ahead/behind; we mirror that.
  const branchEntries: ObservationBranch[] = (project.branches ?? []).map((b: RegistryBranch) => {
    const path = b.path ? b.path : root;
    const git: ObservationBranch['git'] = existsSync(path)
      ? getGitSnapshot(path, b.baseBranch ?? '')
      : { ok: false, error: 'path missing' };
    return {
      name: b.name,
      kind: b.kind,
      status: b.status,
      path,
      ...(b.baseBranch !== undefined ? { baseBranch: b.baseBranch } : {}),
      ...(b.returnBranch !== undefined ? { returnBranch: b.returnBranch } : {}),
      git,
      ...(b.lastIndexedCommit !== undefined ? { lastIndexedCommit: b.lastIndexedCommit } : {}),
      ...(b.watchEnabled !== undefined ? { watchEnabled: b.watchEnabled } : {}),
    };
  });

  // Health + queue come from the daemon over gRPC (SystemService). The wqm
  // CLI shell-out is gone: it was absent in the dockerized container and, for
  // queue stats, read the wrong SQLite anyway. All four probes are independent.
  const [qdrant, daemonTcp, wqmHealth, queue] = await Promise.all([
    testQdrant(project.qdrantUrl),
    testTcpEndpoint(project.daemonEndpoint),
    probeDaemonHealth(daemonClient),
    probeDaemonQueue(daemonClient),
  ]);

  return {
    timestamp: utcNow(),
    project: project.name,
    root,
    qdrant,
    daemonTcp,
    wqmHealth,
    queue,
    branches: branchEntries,
  };
}

/**
 * Persist the observation to two places (matching PS Save-Observation):
 *   - `<dirname(registryPath)>/observability/<safe-slug>-latest.json`
 *     (overwritten on every save — gives the LLM a stable "most recent" file
 *     to grep without timestamp guessing);
 *   - `<dirname(registryPath)>/logs/indexed-projects-YYYYMMDD.jsonl`
 *     (appended; compressed one-line JSON).
 *
 * Returns the absolute path of the per-project latest.json so the action
 * result can include `savedTo` for the LLM to reference.
 */
export function saveObservation(registryPath: string, obs: Observation): string {
  const baseDir = dirname(registryPath);
  const obsDir = resolve(baseDir, 'observability');
  const logDir = resolve(baseDir, 'logs');
  ensureDir(obsDir);
  ensureDir(logDir);

  const safe = safeBranchSlug(obs.project);
  const latestFile = resolve(obsDir, `${safe}-latest.json`);
  writeFileSync(latestFile, `${JSON.stringify(obs, null, 4)}\n`, 'utf-8');

  const dayStamp = new Date().toISOString().slice(0, 10).replace(/-/g, '');
  const logFile = resolve(logDir, `indexed-projects-${dayStamp}.jsonl`);
  appendFileSync(logFile, `${JSON.stringify(obs)}\n`, 'utf-8');

  return latestFile;
}
