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
      <td class="dim">${escapeHtml(fmtTime(r.lastActivityAt))}</td>
      <td>
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
    }
  } catch (e) {
    toast(e.message, 'error');
  } finally {
    btn.disabled = false;
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
