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
import { delimiter, dirname, isAbsolute, join, resolve } from 'node:path';

import type { RegistryBranch, RegistryProject } from './indexed-projects-registry.js';

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

// ── Output sanitization ────────────────────────────────────────────────────

/**
 * Strip ANSI CSI escapes, drop control chars except LF/CR, replace non-ASCII
 * with '?'. Matches PS Sanitize-CommandOutput. Without this, JSON.stringify
 * downstream may emit bytes that break a strict consumer (the PS version had
 * to deal with the same problem on PowerShell 5.1's ConvertTo-Json).
 */
export function sanitizeCommandOutput(text: string): string {
  if (!text) return text;
  // ESC [ ... letter (covers most CSI sequences including SGR colors).
  // Use a raw string with hex escape to avoid embedding a literal ESC.
  const noAnsi = text.replace(/\x1B\[[0-9;?]*[A-Za-z]/g, '');
  let out = '';
  for (let i = 0; i < noAnsi.length; i++) {
    const code = noAnsi.charCodeAt(i);
    if (code === 9) {
      out += ' ';
      continue;
    } // tab → space (JSON-safe)
    if (code === 10 || code === 13) {
      out += noAnsi[i];
      continue;
    } // keep LF/CR
    if (code < 32 || code === 0xfffd) continue; // drop other control + U+FFFD
    if (code > 126) {
      out += '?';
      continue;
    } // drop non-ASCII (box-drawing etc.)
    out += noAnsi[i];
  }
  return out;
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

// ── wqm CLI resolver + captured spawn ─────────────────────────────────────

/**
 * Locate the `wqm` binary. Mirrors PS Resolve-WqmPath order:
 *   1. WQM_PATH / WQM_EXECUTABLE env vars (absolute or PATH-resolvable);
 *   2. <baseDir>/src/rust/target/{debug,release}/wqm[.exe];
 *   3. PATH lookup with .exe/.cmd/.bat fallbacks on Windows.
 * Returns null when unfound (e.g. dockerized daemon-only setup).
 */
export function resolveWqmPath(baseDir: string): string | null {
  for (const envName of ['WQM_PATH', 'WQM_EXECUTABLE'] as const) {
    const configured = process.env[envName];
    if (configured) {
      const r = resolveCommandOnPath(configured);
      if (r) return r;
    }
  }
  const exeSuffix = process.platform === 'win32' ? '.exe' : '';
  const candidates = [
    join(baseDir, 'src', 'rust', 'target', 'debug', `wqm${exeSuffix}`),
    join(baseDir, 'src', 'rust', 'target', 'release', `wqm${exeSuffix}`),
  ];
  for (const cand of candidates) {
    if (existsSync(cand)) return cand;
  }
  const names =
    process.platform === 'win32'
      ? ['wqm.exe', 'wqm.cmd', 'wqm.bat', 'wqm']
      : ['wqm'];
  for (const name of names) {
    const r = resolveCommandOnPath(name);
    if (r) return r;
  }
  return null;
}

function resolveCommandOnPath(command: string): string | null {
  if (!command) return null;
  if (isAbsolute(command)) return existsSync(command) ? command : null;

  const isWin = process.platform === 'win32';
  const searchNames = [command];
  if (isWin && !/\.[A-Za-z0-9]{1,4}$/.test(command)) {
    searchNames.push(`${command}.exe`, `${command}.cmd`, `${command}.bat`);
  }

  const PATH = process.env['PATH'] ?? process.env['Path'] ?? '';
  const entries = PATH.split(delimiter).filter((p) => p.length > 0);
  for (const entry of entries) {
    for (const name of searchNames) {
      const cand = join(entry, name);
      if (existsSync(cand)) return cand;
    }
  }
  return null;
}

export interface CapturedResult {
  ok: boolean;
  exitCode: number;
  file?: string;
  stdout: string;
  stderr: string;
  skipped?: boolean;
  available?: boolean;
  reason?: string;
}

/**
 * Spawn a command and capture stdout/stderr, with a hard timeout. Matches
 * PS Invoke-Captured signature/shape closely so the JSON shipped to the LLM
 * looks identical to the PS bridge output.
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

export function invokeCaptured(
  file: string | null,
  args: string[],
  cwd: string,
  timeoutSeconds = 20
): CapturedResult {
  if (!file) {
    return {
      ok: false,
      exitCode: -1,
      stdout: '',
      stderr: 'executable not found (resolveWqmPath returned null)',
    };
  }
  const cleanArgs = args.filter((a) => a !== null && a !== undefined && a !== '');
  let resolved = file;
  if (!isAbsolute(resolved)) {
    const r = resolveCommandOnPath(resolved);
    if (r) resolved = r;
  }
  if (!existsSync(resolved)) {
    return {
      ok: false,
      exitCode: -1,
      file: resolved,
      stdout: '',
      stderr: `executable missing at: ${resolved}`,
    };
  }
  try {
    const res = spawnSyncCaptured(resolved, cleanArgs, cwd, timeoutSeconds * 1000);
    if (res.timedOut) {
      return {
        ok: false,
        exitCode: -1,
        file: resolved,
        stdout: sanitizeCommandOutput(res.stdout),
        stderr: `timed out after ${timeoutSeconds}s`,
      };
    }
    return {
      ok: res.exitCode === 0,
      exitCode: res.exitCode,
      file: resolved,
      stdout: sanitizeCommandOutput(res.stdout),
      stderr: sanitizeCommandOutput(res.stderr),
    };
  } catch (err) {
    return {
      ok: false,
      exitCode: -1,
      file: resolved,
      stdout: '',
      stderr: (err as Error).message,
    };
  }
}

/**
 * `wqm watch list --json` is only present on builds with the watch subcommand.
 * When missing, surface a structured skip so callers see "available: false"
 * instead of a hard error. Mirrors PS Invoke-OptionalWatchCapture.
 */
export function invokeOptionalWatchCapture(
  file: string | null,
  args: string[],
  cwd: string,
  timeoutSeconds = 20
): CapturedResult {
  const result = invokeCaptured(file, args, cwd, timeoutSeconds);
  if (result.ok) return result;
  const combined = `${result.stdout}\n${result.stderr}`;
  if (/unrecognized subcommand 'watch'/.test(combined)) {
    return {
      ok: true,
      skipped: true,
      available: false,
      reason: 'watch subcommand unavailable',
      exitCode: 0,
      stdout: result.stdout,
      stderr: result.stderr,
    };
  }
  return result;
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

export interface Observation {
  timestamp: string;
  project: string;
  root: string;
  qdrant: QdrantProbe;
  daemonTcp: TcpProbe;
  wqmHealth: CapturedResult;
  queue: CapturedResult;
  branches: ObservationBranch[];
}

/**
 * Build the per-project observation. Mirrors PS New-Observation:
 * field set, field order, fallback shapes when wqm is absent.
 */
export async function newObservation(
  project: RegistryProject,
  wqmPath: string | null
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

  // wqm probes. Container without wqm: surface a stable "not installed" stub
  // identical to the PS impl so downstream callers don't have to special-case.
  const wqmHealth: CapturedResult = wqmPath
    ? invokeCaptured(wqmPath, ['status', 'health'], root, 20)
    : {
        ok: false,
        exitCode: -1,
        stdout: '',
        stderr: 'wqm CLI not installed on host',
      };
  const queue: CapturedResult = wqmPath
    ? invokeCaptured(wqmPath, ['queue', 'stats'], root, 20)
    : {
        ok: false,
        exitCode: -1,
        stdout: '',
        stderr: 'wqm CLI not installed on host',
      };

  // Network probes (concurrent — neither depends on the other).
  const [qdrant, daemonTcp] = await Promise.all([
    testQdrant(project.qdrantUrl),
    testTcpEndpoint(project.daemonEndpoint),
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
