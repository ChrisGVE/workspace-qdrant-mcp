/**
 * workspace-qdrant admin UI client.
 *
 * Vanilla JS — no framework, no build step. Auth via Bearer token kept
 * in sessionStorage. Real-time refresh is plain polling (5s) so the
 * server stays single-protocol; no SSE / websocket complexity.
 */

const TOKEN_KEY = 'wqm.admin.token';
const REFRESH_MS = 5000;

const els = {
  loginPanel: document.getElementById('loginPanel'),
  loginForm: document.getElementById('loginForm'),
  tokenInput: document.getElementById('tokenInput'),
  loginError: document.getElementById('loginError'),
  appView: document.getElementById('appView'),
  connectionStatus: document.getElementById('connectionStatus'),
  lastUpdated: document.getElementById('lastUpdated'),
  statDaemon: document.getElementById('statDaemon'),
  statDaemonDetail: document.getElementById('statDaemonDetail'),
  statQueue: document.getElementById('statQueue'),
  statQueueDetail: document.getElementById('statQueueDetail'),
  statDocs: document.getElementById('statDocs'),
  statDocsDetail: document.getElementById('statDocsDetail'),
  healthHooks: document.getElementById('healthHooks'),
  healthHooksDetail: document.getElementById('healthHooksDetail'),
  healthQdrant: document.getElementById('healthQdrant'),
  healthQdrantDetail: document.getElementById('healthQdrantDetail'),
  healthMcp: document.getElementById('healthMcp'),
  healthMcpDetail: document.getElementById('healthMcpDetail'),
  reinstallHooksBtn: document.getElementById('reinstallHooksBtn'),
  hooksInstallLog: document.getElementById('hooksInstallLog'),
  settingsForm: document.getElementById('settingsForm'),
  devRootInput: document.getElementById('devRootInput'),
  scanDepthInput: document.getElementById('scanDepthInput'),
  scanBtn: document.getElementById('scanBtn'),
  settingsMsg: document.getElementById('settingsMsg'),
  candidatesTable: document.getElementById('candidatesTable'),
  candidatesBody: document.getElementById('candidatesBody'),
  candidatesEmpty: document.getElementById('candidatesEmpty'),
  candidatesMeta: document.getElementById('candidatesMeta'),
  registeredTable: document.getElementById('registeredTable'),
  registeredBody: document.getElementById('registeredBody'),
  registeredEmpty: document.getElementById('registeredEmpty'),
  registeredMeta: document.getElementById('registeredMeta'),
  debugRaw: document.getElementById('debugRaw'),
  toast: document.getElementById('toast'),
  globalIgnoreText: document.getElementById('globalIgnoreText'),
  saveIgnoreBtn: document.getElementById('saveIgnoreBtn'),
  reloadIgnoreBtn: document.getElementById('reloadIgnoreBtn'),
  ignoreMsg: document.getElementById('ignoreMsg'),
  largestFilesTable: document.getElementById('largestFilesTable'),
  largestFilesBody: document.getElementById('largestFilesBody'),
  largestFilesEmpty: document.getElementById('largestFilesEmpty'),
  largestFilesMeta: document.getElementById('largestFilesMeta'),
  largestFilesSkippedOnly: document.getElementById('largestFilesSkippedOnly'),
  reloadLargestFilesBtn: document.getElementById('reloadLargestFilesBtn'),
  showClaudeConfigBtn: document.getElementById('showClaudeConfigBtn'),
  showCodexConfigBtn: document.getElementById('showCodexConfigBtn'),
  configHint: document.getElementById('configHint'),
  configDisplay: document.getElementById('configDisplay'),
  configLabel: document.getElementById('configLabel'),
  configPre: document.getElementById('configPre'),
  copyConfigBtn: document.getElementById('copyConfigBtn'),
  copyMsg: document.getElementById('copyMsg'),
  logLinesSelect: document.getElementById('logLinesSelect'),
  loadMcpLogsBtn: document.getElementById('loadMcpLogsBtn'),
  clearLogsBtn: document.getElementById('clearLogsBtn'),
  logsEmpty: document.getElementById('logsEmpty'),
  logsMeta: document.getElementById('logsMeta'),
  logsTable: document.getElementById('logsTable'),
  logsBody: document.getElementById('logsBody'),
  checkDaemonMetricsBtn: document.getElementById('checkDaemonMetricsBtn'),
  daemonMetricsVal: document.getElementById('daemonMetricsVal'),
  daemonMetricsDetail: document.getElementById('daemonMetricsDetail'),
  forceReconcileBtn: document.getElementById('forceReconcileBtn'),
  stackActionsStatus: document.getElementById('stackActionsStatus'),
  stackActionsLog: document.getElementById('stackActionsLog'),
  refreshHealthBtn: document.getElementById('refreshHealthBtn'),
  adminPidVal: document.getElementById('adminPidVal'),
};

let token = sessionStorage.getItem(TOKEN_KEY) || '';
let pollTimer = null;
let lastCandidates = [];
let lastRegisteredPaths = new Set();

// ── Networking ──────────────────────────────────────────────────────

async function api(path, opts = {}) {
  const init = {
    method: opts.method || 'GET',
    headers: {
      'Accept': 'application/json',
      'Authorization': `Bearer ${token}`,
      ...(opts.body ? { 'Content-Type': 'application/json' } : {}),
      ...(opts.headers || {}),
    },
  };
  if (opts.body) init.body = JSON.stringify(opts.body);
  const res = await fetch(path, init);
  const text = await res.text();
  let json = null;
  try { json = text ? JSON.parse(text) : null; } catch { /* ignore */ }
  if (!res.ok) {
    const detail = json?.detail || json?.error || text || res.statusText;
    const err = new Error(`HTTP ${res.status}: ${detail}`);
    err.status = res.status;
    throw err;
  }
  return json;
}

// ── Toast ──────────────────────────────────────────────────────────

let toastTimer = null;
function toast(msg, kind = 'ok') {
  els.toast.textContent = msg;
  els.toast.className = `toast${kind === 'error' ? ' error' : ''}`;
  els.toast.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { els.toast.hidden = true; }, 3200);
}

// ── Render ─────────────────────────────────────────────────────────

function pill(text, kind) {
  return `<span class="pill pill-${kind}">${escapeHtml(text)}</span>`;
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({
    '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;',
  })[c]);
}

function fmtTime(iso) {
  if (!iso) return '—';
  const d = new Date(iso);
  if (isNaN(d.getTime())) return iso;
  const sec = Math.floor((Date.now() - d.getTime()) / 1000);
  if (sec < 60) return `${sec}s ago`;
  if (sec < 3600) return `${Math.floor(sec/60)}m ago`;
  if (sec < 86400) return `${Math.floor(sec/3600)}h ago`;
  return d.toISOString().slice(0, 16).replace('T', ' ');
}

function renderHealth(snap) {
  const d = snap.daemon || {};
  if (d.ok) {
    els.statDaemon.innerHTML = pill('healthy', 'ok');
    els.statDaemonDetail.textContent =
      `${(d.activeProjects || []).length} active · ${d.totalCollections || 0} collections`;
  } else {
    els.statDaemon.innerHTML = pill('unhealthy', 'err');
    els.statDaemonDetail.textContent = d.reason || 'unknown';
  }

  const q = snap.queue || {};
  els.statQueue.textContent = q.pending ?? 0;
  els.statQueueDetail.textContent =
    `${q.pending || 0} pending · ${q.in_progress || 0} in-progress · ${q.failed || 0} failed`;

  els.statDocs.textContent = (d.totalDocuments ?? 0).toLocaleString();
  els.statDocsDetail.textContent =
    `${d.totalCollections || 0} collections · ${snap.projects?.registeredCount || 0} watch folders`;
}

function renderSettings(snap) {
  if (document.activeElement !== els.devRootInput) {
    els.devRootInput.value = snap.settings?.devRoot || '';
  }
  if (document.activeElement !== els.scanDepthInput) {
    els.scanDepthInput.value = snap.settings?.scanDepth || 1;
  }
}

function renderCandidates() {
  const registered = lastRegisteredPaths;
  const cands = lastCandidates.filter((c) => !registered.has(c.path));
  els.candidatesMeta.textContent = `${cands.length} candidate(s)`;
  if (cands.length === 0) {
    els.candidatesTable.hidden = true;
    els.candidatesEmpty.hidden = false;
    return;
  }
  els.candidatesTable.hidden = false;
  els.candidatesEmpty.hidden = true;
  els.candidatesBody.innerHTML = cands.map((c) => `
    <tr>
      <td><span class="path">${escapeHtml(c.path)}</span>
          <span class="sub">${escapeHtml(c.name)}</span></td>
      <td>${escapeHtml(c.branch || '—')}</td>
      <td><span class="path">${escapeHtml(c.remoteUrl || '—')}</span></td>
      <td class="num">${c.depth}</td>
      <td>${c.isWorktree ? pill('worktree', 'warn') : pill('repo', 'muted')}</td>
      <td>
        <button class="primary small" data-action="register" data-path="${escapeHtml(c.path)}">Register</button>
      </td>
    </tr>
  `).join('');
}

/** Render an ETA in seconds as a coarse human-readable string ("3s",
 *  "12m", "2h 14m"). The daemon's rate-window resolution is 5 minutes,
 *  so any extra precision would be theatre. */
function formatEta(seconds) {
  const s = Number(seconds);
  if (!Number.isFinite(s) || s < 0) return 'unknown';
  if (s < 60) return `${Math.round(s)}s`;
  const m = Math.round(s / 60);
  if (m < 60) return `${m}m`;
  const h = Math.floor(m / 60);
  const rem = m % 60;
  return rem === 0 ? `${h}h` : `${h}h ${rem}m`;
}

/**
 * Render the indexing-progress cell for one registered project.
 * Returns a small HTML snippet showing a progress bar + counts + ETA.
 * When the daemon couldn't report (`indexing == null`), shows a dim "—".
 */
function renderIndexingCell(indexing) {
  if (!indexing) return '<span class="dim small">—</span>';
  const pct = Math.max(0, Math.min(100, Number(indexing.percent ?? 0)));
  const inFlight = (indexing.pending ?? 0) + (indexing.in_progress ?? 0);
  const failed = indexing.failed ?? 0;
  const done = indexing.done ?? 0;
  const total = indexing.total ?? 0;
  // Three states: idle (queue drained), active (in-flight > 0), failed-some.
  let state = 'idle';
  if (inFlight > 0) state = 'active';
  else if (failed > 0) state = 'warn';
  const labelMain =
    inFlight > 0
      ? `${inFlight} in flight · ${done}/${total}`
      : `${done} indexed`;
  const labelFailed = failed > 0 ? ` · <span class="warn">${failed} failed</span>` : '';
  // ETA only renders when the queue is still draining — once idle the
  // value is uninformative ("0s") and noisy. "Warming up" reflects the
  // daemon's cold-start window.
  let etaLine = '';
  if (inFlight > 0) {
    const etaText =
      typeof indexing.eta_seconds === 'number'
        ? `ETA ~${formatEta(indexing.eta_seconds)}`
        : 'ETA — warming up';
    etaLine = `<div class="indexing-eta dim small">${escapeHtml(etaText)}</div>`;
  }
  return `
    <div class="indexing-cell">
      <div class="indexing-bar indexing-bar-${state}" role="progressbar"
           aria-valuenow="${pct.toFixed(1)}" aria-valuemin="0" aria-valuemax="100">
        <div class="indexing-bar-fill" style="width:${pct.toFixed(1)}%"></div>
      </div>
      <div class="indexing-meta dim small">
        ${escapeHtml(labelMain)}${labelFailed} · ${pct.toFixed(1)}%
      </div>
      ${etaLine}
    </div>
  `;
}

function renderRegistered(snap) {
  const registered = snap.projects?.registered || [];
  lastRegisteredPaths = new Set(registered.map((r) => r.path));
  els.registeredMeta.textContent = `${registered.length} registered`;
  if (registered.length === 0) {
    els.registeredTable.hidden = true;
    els.registeredEmpty.hidden = false;
    return;
  }
  els.registeredTable.hidden = false;
  els.registeredEmpty.hidden = true;
  els.registeredBody.innerHTML = registered.map((r) => `
    <tr>
      <td><span class="path">${escapeHtml(r.path)}</span></td>
      <td><code>${escapeHtml(r.tenantId)}</code></td>
      <td>${r.isActive ? pill('active', 'ok') : pill('idle', 'muted')}</td>
      <td>${renderIndexingCell(r.indexing)}</td>
      <td class="dim">${escapeHtml(fmtTime(r.lastActivityAt))}</td>
      <td style="white-space:nowrap">
        <button class="secondary small"
                data-action="watch-pause"
                data-watch-id="${escapeHtml(r.path)}"
                title="Set is_paused=1 for this watch folder">Pause</button>
        <button class="secondary small"
                data-action="watch-resume"
                data-watch-id="${escapeHtml(r.path)}"
                title="Set is_paused=0 for this watch folder">Resume</button>
        <button class="secondary small"
                data-action="project-reindex"
                data-id="${escapeHtml(r.tenantId)}"
                title="Rebuild FTS5/tags/sparse/components for this project (no re-embed)">Reindex</button>
        <button class="secondary small"
                data-action="project-reembed"
                data-id="${escapeHtml(r.tenantId)}"
                title="Re-read & re-embed all of this project's files (regenerates vectors; runs in the queue)">Re-embed</button>
        <button class="danger small"
                data-action="deregister"
                data-id="${escapeHtml(r.tenantId)}"
                data-path="${escapeHtml(r.path)}">Deactivate</button>
      </td>
    </tr>
  `).join('');
}

function renderDebug(snap) {
  els.debugRaw.textContent = JSON.stringify(snap, null, 2);
}

function renderConnection(ok, err) {
  if (ok) {
    els.connectionStatus.className = 'pill pill-ok';
    els.connectionStatus.textContent = 'online';
  } else {
    els.connectionStatus.className = 'pill pill-err';
    els.connectionStatus.textContent = err || 'offline';
  }
  els.lastUpdated.textContent = `updated ${new Date().toLocaleTimeString()}`;
}

// ── Polling ────────────────────────────────────────────────────────

async function refresh() {
  try {
    const snap = await api('/admin/api/snapshot');
    renderHealth(snap);
    renderSettings(snap);
    renderRegistered(snap);
    renderCandidates();
    renderDebug(snap);
    renderConnection(true);
  } catch (e) {
    renderConnection(false, e.status === 401 ? 'auth failed' : 'offline');
    if (e.status === 401) {
      logout(e.message);
    }
  }
  // Host integrations health — refreshed on the same cadence but tolerant
  // of failures (the snapshot can still succeed even if /health hiccups).
  try {
    const health = await api('/admin/api/health');
    renderHostHealth(health);
    if (health.mcp) {
      els.adminPidVal.textContent = `pid ${health.mcp.pid ?? '—'} · up ${formatUptime(health.mcp.uptimeSeconds ?? 0)}`;
    }
  } catch {
    // Leave previous values in place; the snapshot path already drives
    // the "connecting…" pill so we don't double-flag.
  }
}

function renderHostHealth(h) {
  if (!h) return;
  const hooks = h.hooks || {};
  if (hooks.kind === 'posix' && hooks.ok) {
    els.healthHooks.innerHTML = pill('POSIX · OK', 'ok');
  } else if (hooks.kind === 'powershell') {
    els.healthHooks.innerHTML = pill('PowerShell (legacy)', 'warn');
  } else if (hooks.kind === 'mixed') {
    els.healthHooks.innerHTML = pill('Mixed PS+POSIX', 'warn');
  } else if (hooks.kind === 'posix' && !hooks.ok) {
    els.healthHooks.innerHTML = pill('POSIX · incomplete', 'warn');
  } else {
    els.healthHooks.innerHTML = pill('not installed', 'err');
  }
  const installedTxt = (hooks.installed || []).length + ' installed';
  const legacyTxt = (hooks.legacyArtifacts || []).length
    ? ` · ${(hooks.legacyArtifacts || []).length} legacy artifact(s)`
    : '';
  els.healthHooksDetail.textContent = `${installedTxt}${legacyTxt}`;

  const qdrant = h.qdrant || {};
  els.healthQdrant.innerHTML = qdrant.ok
    ? pill('reachable', 'ok')
    : pill('offline', 'err');
  els.healthQdrantDetail.textContent = qdrant.endpoint || qdrant.reason || '—';

  const mcp = h.mcp || {};
  els.healthMcp.innerHTML = pill(mcp.version || 'running', 'ok');
  const uptime = mcp.uptimeSeconds ? `${formatUptime(mcp.uptimeSeconds)} uptime` : '—';
  els.healthMcpDetail.textContent = `${mcp.mode || 'http'} · pid ${mcp.pid} · ${uptime}`;
}

function formatUptime(seconds) {
  if (seconds < 60) return `${seconds}s`;
  const mins = Math.floor(seconds / 60);
  if (mins < 60) return `${mins}m`;
  const hours = Math.floor(mins / 60);
  return `${hours}h${mins % 60}m`;
}

function startPolling() {
  if (pollTimer) clearInterval(pollTimer);
  refresh();
  pollTimer = setInterval(refresh, REFRESH_MS);
}

function stopPolling() {
  if (pollTimer) { clearInterval(pollTimer); pollTimer = null; }
}

// ── Auth ───────────────────────────────────────────────────────────

function showLogin() {
  els.loginPanel.hidden = false;
  els.appView.hidden = true;
  els.tokenInput.value = '';
  setTimeout(() => els.tokenInput.focus(), 50);
}

function showApp() {
  els.loginPanel.hidden = true;
  els.appView.hidden = false;
  loadGlobalIgnore();
  loadLargestFiles();
}

function logout(reason) {
  stopPolling();
  sessionStorage.removeItem(TOKEN_KEY);
  token = '';
  els.loginError.textContent = reason || '';
  els.loginError.hidden = !reason;
  showLogin();
}

els.loginForm.addEventListener('submit', async (e) => {
  e.preventDefault();
  token = els.tokenInput.value.trim();
  if (!token) return;
  try {
    await api('/admin/api/snapshot');
    sessionStorage.setItem(TOKEN_KEY, token);
    els.loginError.hidden = true;
    showApp();
    startPolling();
  } catch (e) {
    els.loginError.textContent = e.message || 'authentication failed';
    els.loginError.hidden = false;
  }
});

// ── Actions ────────────────────────────────────────────────────────

els.settingsForm.addEventListener('submit', async (e) => {
  e.preventDefault();
  try {
    const next = await api('/admin/api/settings', {
      method: 'PUT',
      body: {
        devRoot: els.devRootInput.value.trim(),
        scanDepth: Number(els.scanDepthInput.value) || 1,
      },
    });
    els.settingsMsg.textContent = `Saved at ${new Date().toLocaleTimeString()}`;
    toast('Settings saved');
    refresh();
  } catch (e) {
    toast(e.message, 'error');
  }
});

els.reinstallHooksBtn.addEventListener('click', async () => {
  els.reinstallHooksBtn.disabled = true;
  const originalLabel = els.reinstallHooksBtn.textContent;
  els.reinstallHooksBtn.textContent = 'Installing…';
  els.hooksInstallLog.hidden = true;
  try {
    const result = await api('/admin/api/hooks/install', {
      method: 'POST',
      body: { force: true },
    });
    const lines = [
      `exitCode: ${result.exitCode}`,
      result.stdout ? `--- stdout ---\n${result.stdout}` : '',
      result.stderr ? `--- stderr ---\n${result.stderr}` : '',
    ].filter(Boolean);
    els.hooksInstallLog.textContent = lines.join('\n');
    els.hooksInstallLog.hidden = false;
    if (result.ok) {
      toast('Hooks reinstalled');
    } else {
      toast(`Install failed (exit ${result.exitCode})`, 'error');
    }
    refresh();
  } catch (e) {
    toast(e.message, 'error');
  } finally {
    els.reinstallHooksBtn.disabled = false;
    els.reinstallHooksBtn.textContent = originalLabel;
  }
});

els.scanBtn.addEventListener('click', async () => {
  els.scanBtn.disabled = true;
  els.scanBtn.textContent = 'Scanning…';
  try {
    const result = await api('/admin/api/projects/scan', {
      method: 'POST',
      body: {
        devRoot: els.devRootInput.value.trim(),
        scanDepth: Number(els.scanDepthInput.value) || 1,
      },
    });
    lastCandidates = result.scan?.candidates || [];
    renderCandidates();
    toast(`Found ${lastCandidates.length} candidate(s) in ${result.scan?.visited || 0} dirs`);
    refresh();
  } catch (e) {
    toast(e.message, 'error');
  } finally {
    els.scanBtn.disabled = false;
    els.scanBtn.textContent = 'Scan now';
  }
});

document.addEventListener('click', async (e) => {
  const btn = e.target.closest('button[data-action]');
  if (!btn) return;
  const action = btn.dataset.action;
  btn.disabled = true;
  try {
    if (action === 'register') {
      await api('/admin/api/projects/register', {
        method: 'POST',
        body: { path: btn.dataset.path },
      });
      toast(`Registered ${btn.dataset.path}`);
      refresh();
    } else if (action === 'deregister') {
      await api('/admin/api/projects/deregister', {
        method: 'POST',
        body: { projectId: btn.dataset.id, path: btn.dataset.path },
      });
      toast(`Deactivated ${btn.dataset.id}`);
      refresh();
    } else if (action === 'watch-pause') {
      const result = await api('/admin/api/watches/pause', {
        method: 'POST',
        body: { watchId: btn.dataset.watchId },
      });
      if (result.affectedCount > 0) {
        toast(`Paused watch: ${btn.dataset.watchId}`);
      } else {
        toast(`No change (already paused, disabled, or not found)`, 'info');
      }
      refresh();
    } else if (action === 'watch-resume') {
      const result = await api('/admin/api/watches/resume', {
        method: 'POST',
        body: { watchId: btn.dataset.watchId },
      });
      if (result.affectedCount > 0) {
        toast(`Resumed watch: ${btn.dataset.watchId}`);
      } else {
        toast(`No change (not currently paused, disabled, or not found)`, 'info');
      }
      refresh();
    } else if (action === 'project-reindex') {
      const result = await api('/admin/api/projects/reindex', {
        method: 'POST',
        body: { tenantId: btn.dataset.id },
      });
      if (result.ok) {
        const ms = result.durationMs ? ` (${result.durationMs}ms)` : '';
        toast(`Reindexed project ${btn.dataset.id}${ms}`);
      } else {
        toast(`Reindex reported failure: ${result.message || 'unknown'}`, 'error');
      }
      refresh();
    } else if (action === 'project-reembed') {
      if (!confirm(`Re-embed all files for project ${btn.dataset.id}? This re-reads and re-embeds the whole project in the background.`)) {
        return;
      }
      const result = await api('/admin/api/projects/reembed', {
        method: 'POST',
        body: { tenantId: btn.dataset.id },
      });
      toast(`Re-embed queued for ${btn.dataset.id}: ${result.filesEnqueued ?? 0} folder scan(s)`);
      refresh();
    }
  } catch (e) {
    toast(e.message, 'error');
  } finally {
    btn.disabled = false;
  }
});

// ── Global ignore rules ────────────────────────────────────────────

async function loadGlobalIgnore() {
  try {
    const data = await api('/admin/api/ignore/global');
    // Don't overwrite if the user is actively editing
    if (document.activeElement !== els.globalIgnoreText) {
      els.globalIgnoreText.value = data.content || '';
    }
    els.ignoreMsg.textContent = `Loaded · ${data.path || ''}`;
  } catch (e) {
    els.ignoreMsg.textContent = `Load failed: ${e.message}`;
  }
}

els.saveIgnoreBtn.addEventListener('click', async () => {
  els.saveIgnoreBtn.disabled = true;
  const originalLabel = els.saveIgnoreBtn.textContent;
  els.saveIgnoreBtn.textContent = 'Saving…';
  try {
    const result = await api('/admin/api/ignore/global', {
      method: 'PUT',
      body: { content: els.globalIgnoreText.value },
    });
    els.ignoreMsg.textContent = `Saved ${result.bytes} bytes · ${new Date().toLocaleTimeString()}`;
    toast('Global ignore rules saved');
  } catch (e) {
    toast(e.message, 'error');
  } finally {
    els.saveIgnoreBtn.disabled = false;
    els.saveIgnoreBtn.textContent = originalLabel;
  }
});

els.reloadIgnoreBtn.addEventListener('click', () => loadGlobalIgnore());

// ── Largest files (search.db file_metadata) ────────────────────────

function formatBytes(n) {
  if (n === null || n === undefined) return '—';
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
  if (n < 1024 * 1024 * 1024) return `${(n / (1024 * 1024)).toFixed(1)} MB`;
  return `${(n / (1024 * 1024 * 1024)).toFixed(2)} GB`;
}

async function loadLargestFiles() {
  if (!els.largestFilesTable) return; // safety: section not in DOM
  const skipped = els.largestFilesSkippedOnly?.checked ? '&skipped=1' : '';
  try {
    const data = await api(`/admin/api/files/large?limit=20${skipped}`);
    const files = data?.files ?? [];
    if (data?.degraded) {
      els.largestFilesEmpty.textContent = `search.db unavailable: ${data.degraded.message}`;
      els.largestFilesEmpty.hidden = false;
      els.largestFilesTable.hidden = true;
      els.largestFilesMeta.textContent = '';
      return;
    }
    if (files.length === 0) {
      els.largestFilesEmpty.textContent = skipped
        ? 'No files currently marked fts5_skipped=1.'
        : 'No files indexed yet — daemon may still be walking.';
      els.largestFilesEmpty.hidden = false;
      els.largestFilesTable.hidden = true;
      els.largestFilesMeta.textContent = '';
      return;
    }
    els.largestFilesEmpty.hidden = true;
    els.largestFilesTable.hidden = false;
    els.largestFilesBody.innerHTML = files.map((f) => {
      const skippedBadge = f.fts5_skipped
        ? '<span class="pill" style="background:#fef3c7;color:#92400e;">skipped</span>'
        : '<span class="dim small">indexed</span>';
      // Show the trailing portion of the path to keep rows compact;
      // full path is in the title attribute on hover.
      const shortPath = f.file_path.length > 80
        ? '…' + f.file_path.slice(-78)
        : f.file_path;
      return `<tr>
        <td><code title="${escapeHtml(f.file_path)}">${escapeHtml(shortPath)}</code></td>
        <td><span class="dim small">${escapeHtml(f.tenant_id)}</span></td>
        <td><span class="dim small">${escapeHtml(f.branch)}</span></td>
        <td style="text-align:right;"><strong>${formatBytes(f.size_bytes)}</strong></td>
        <td>${skippedBadge}</td>
      </tr>`;
    }).join('');
    els.largestFilesMeta.textContent = `· ${files.length} rows · source: ${data.source ?? 'search.db'}`;
  } catch (err) {
    els.largestFilesEmpty.textContent = `Failed to load: ${err.message}`;
    els.largestFilesEmpty.hidden = false;
    els.largestFilesTable.hidden = true;
  }
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({
    '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;',
  })[c]);
}

els.reloadLargestFilesBtn?.addEventListener('click', () => loadLargestFiles());
els.largestFilesSkippedOnly?.addEventListener('change', () => loadLargestFiles());

// ── Client configs ─────────────────────────────────────────────────

let clientConfigs = null;

async function loadClientConfigs() {
  if (!clientConfigs) {
    clientConfigs = await api('/admin/api/config/clients');
  }
  return clientConfigs;
}

function showConfig(label, text) {
  els.configHint.hidden = true;
  els.configDisplay.hidden = false;
  els.configLabel.textContent = label;
  els.configPre.textContent = text;
  els.copyMsg.textContent = '';
}

els.showClaudeConfigBtn.addEventListener('click', async () => {
  try {
    const cfg = await loadClientConfigs();
    showConfig('Paste into claude_desktop_config.json → mcpServers', cfg.claudeDesktop.mcp_remote);
  } catch (e) { toast(e.message, 'error'); }
});

els.showCodexConfigBtn.addEventListener('click', async () => {
  try {
    const cfg = await loadClientConfigs();
    showConfig('Paste into ~/.codex/config.toml', cfg.codex);
  } catch (e) { toast(e.message, 'error'); }
});

els.copyConfigBtn.addEventListener('click', async () => {
  try {
    await navigator.clipboard.writeText(els.configPre.textContent);
    els.copyMsg.textContent = 'Copied!';
    setTimeout(() => { els.copyMsg.textContent = ''; }, 2000);
  } catch { els.copyMsg.textContent = 'Copy failed'; }
});

// ── Logs ────────────────────────────────────────────────────────────

const LOG_LEVEL_CLASSES = { trace: 'muted', debug: 'muted', info: '', warn: 'warn', error: 'err', fatal: 'err' };

function fmtLogTime(ts) {
  if (!ts) return '—';
  try {
    const d = new Date(ts);
    return d.toISOString().slice(11, 23); // HH:MM:SS.mmm
  } catch { return String(ts); }
}

function renderLogs(data) {
  const rows = data.lines ?? [];
  if (rows.length === 0) {
    els.logsEmpty.hidden = false;
    els.logsTable.hidden = true;
    els.logsMeta.hidden = true;
    return;
  }
  els.logsEmpty.hidden = true;
  els.logsMeta.hidden = false;
  els.logsMeta.textContent = `${rows.length} lines · ${data.file ?? ''}`;
  els.logsTable.hidden = false;
  els.logsBody.innerHTML = rows.map(row => {
    const level = row.level ?? row.severity ?? '';
    const levelName = typeof level === 'number'
      ? (level >= 50 ? 'error' : level >= 40 ? 'warn' : level >= 30 ? 'info' : level >= 20 ? 'debug' : 'trace')
      : String(level).toLowerCase();
    const cls = LOG_LEVEL_CLASSES[levelName] ?? '';
    const msg = escapeHtml(row.msg ?? row.message ?? JSON.stringify(row));
    const ctx = Object.entries(row)
      .filter(([k]) => !['level','time','msg','name','pid','hostname','component','v'].includes(k))
      .map(([k,v]) => `<span class="dim">${escapeHtml(k)}=</span>${escapeHtml(String(v))}`)
      .join(' ');
    return `<tr class="${cls ? 'log-' + cls : ''}">
      <td class="mono dim" style="white-space:nowrap">${escapeHtml(fmtLogTime(row.time))}</td>
      <td class="${cls ? 'pill-' + cls : ''}" style="font-size:10px;font-weight:600;text-transform:uppercase">${escapeHtml(levelName)}</td>
      <td>${msg}${ctx ? '<br><span class="dim small mono">' + ctx + '</span>' : ''}</td>
    </tr>`;
  }).join('');
}

els.loadMcpLogsBtn.addEventListener('click', async () => {
  els.loadMcpLogsBtn.disabled = true;
  els.loadMcpLogsBtn.textContent = 'Loading…';
  try {
    const lines = els.logLinesSelect.value;
    const data = await api(`/admin/api/logs/mcp?lines=${lines}`);
    renderLogs(data);
    if (data.note) toast(data.note, 'error');
  } catch (e) { toast(e.message, 'error'); }
  finally {
    els.loadMcpLogsBtn.disabled = false;
    els.loadMcpLogsBtn.textContent = 'MCP logs';
  }
});

els.clearLogsBtn.addEventListener('click', () => {
  els.logsEmpty.hidden = false;
  els.logsTable.hidden = true;
  els.logsMeta.hidden = true;
  els.logsEmpty.textContent = 'Click "MCP logs" to load recent server log entries.';
});

// ── Stack actions ───────────────────────────────────────────────────

els.checkDaemonMetricsBtn.addEventListener('click', async () => {
  els.checkDaemonMetricsBtn.disabled = true;
  try {
    const data = await api('/admin/api/daemon/raw-health');
    els.daemonMetricsVal.innerHTML = data.ok ? pill('healthy', 'ok') : pill('unreachable', 'err');
    els.daemonMetricsDetail.textContent = data.body ?? data.reason ?? '—';
  } catch (e) {
    els.daemonMetricsVal.innerHTML = pill('error', 'err');
    els.daemonMetricsDetail.textContent = e.message;
  } finally {
    els.checkDaemonMetricsBtn.disabled = false;
  }
});

els.forceReconcileBtn.addEventListener('click', async () => {
  els.forceReconcileBtn.disabled = true;
  const originalLabel = els.forceReconcileBtn.textContent;
  els.forceReconcileBtn.textContent = 'Reapplying…';
  try {
    const result = await api('/admin/api/ignore/reapply', { method: 'POST' });
    els.stackActionsStatus.textContent = `Ignore reapplied · ${new Date().toLocaleTimeString()}`;
    els.stackActionsLog.hidden = false;
    els.stackActionsLog.textContent =
      `Projects processed : ${result.projectsProcessed}\n` +
      `Stale deletes      : ${result.staleDeleted}\n` +
      `Missing adds       : ${result.missingAdded}\n` +
      `\nEnqueued items drain through the normal queue processor.`;
    toast(
      result.staleDeleted + result.missingAdded > 0
        ? `Reapplied ignore: ${result.staleDeleted} deletes, ${result.missingAdded} adds`
        : 'Reapplied ignore — no changes',
    );
  } catch (e) {
    toast(e.message, 'error');
  } finally {
    els.forceReconcileBtn.disabled = false;
    els.forceReconcileBtn.textContent = originalLabel;
  }
});

els.refreshHealthBtn.addEventListener('click', async () => {
  els.refreshHealthBtn.disabled = true;
  try {
    const h = await api('/admin/api/health');
    renderHostHealth(h);
    els.adminPidVal.textContent = `pid ${h.mcp?.pid ?? '—'} · up ${formatUptime(h.mcp?.uptimeSeconds ?? 0)}`;
    toast('Health refreshed');
  } catch (e) {
    toast(e.message, 'error');
  } finally {
    els.refreshHealthBtn.disabled = false;
  }
});

// ── Boot ───────────────────────────────────────────────────────────

/**
 * Try to bootstrap the auth token from `/admin/init`. The server returns
 * the configured MCP_HTTP_TOKEN ONLY when the request comes from a
 * loopback peer (127.0.0.1 / ::1). Anywhere else it 403s and we fall
 * back to the manual login prompt.
 *
 * When the server runs with `MCP_HTTP_TRUST_LOCALHOST=1`, the bearer
 * check is also bypassed for loopback clients, so we can skip straight
 * to the app even if the init endpoint hadn't existed.
 */
async function tryAutoInit() {
  try {
    const resp = await fetch('/admin/init', { method: 'GET' });
    if (!resp.ok) return false;
    const data = await resp.json().catch(() => null);
    if (!data) return false;
    if (typeof data.token === 'string' && data.token.length > 0) {
      token = data.token;
      sessionStorage.setItem(TOKEN_KEY, token);
      return true;
    }
    if (data.trustLocalhost === true) {
      // Token may legitimately be empty in trust-localhost setups; the
      // server will accept the requests without an Authorization header.
      token = '';
      return true;
    }
    return false;
  } catch {
    return false;
  }
}

(async () => {
  if (!token) {
    const ok = await tryAutoInit();
    if (!ok) {
      showLogin();
      return;
    }
  }
  showApp();
  startPolling();
})();

document.addEventListener('visibilitychange', () => {
  if (document.visibilityState === 'visible' && token !== undefined) refresh();
});
