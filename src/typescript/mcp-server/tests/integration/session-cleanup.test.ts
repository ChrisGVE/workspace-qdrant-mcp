/**
 * Tests for cleanupSession idempotence (F-049).
 *
 * Verifies that the `cleaned` flag on SessionState prevents double-cleanup:
 * - sessionCount metric is decremented exactly once regardless of invocation order
 * - daemon deprioritizeProject is called at most once
 * - heartbeat interval is cleared at most once
 * - no errors when daemon is unavailable
 */

import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { mkdtempSync, rmSync, mkdirSync, realpathSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import Database from 'better-sqlite3';

import { WorkspaceQdrantMcpServer } from '../../src/server.js';
import { register } from '../../src/telemetry/metrics.js';
import type { ServerConfig } from '../../src/types/index.js';
import { TEST_SCHEMA, createTestConfig } from './shared-setup.js';

// ── Helpers ──────────────────────────────────────────────────────────────────

/** Read the current value of the wqm_mcp_session_count Gauge from the registry. */
async function getSessionCountMetric(): Promise<number> {
  const metrics = await register.getMetricsAsJSON();
  const metric = metrics.find((m) => m.name === 'wqm_mcp_session_count');
  if (!metric) return 0;
  const values = metric.values as Array<{ value: number }>;
  return values[0]?.value ?? 0;
}

// ── Mocks ─────────────────────────────────────────────────────────────────────

// We need per-test control of the daemon mock, so define a factory that can be
// replaced before each test via mockImplementation.
let mockDeprioritize = vi
  .fn()
  .mockResolvedValue({ success: true, is_active: false, new_priority: 'normal' });
let mockConnect = vi.fn().mockResolvedValue(undefined);

vi.mock('../../src/clients/daemon-client.js', () => ({
  DaemonClient: vi.fn().mockImplementation(() => ({
    connect: (...args: unknown[]) => mockConnect(...args),
    close: vi.fn(),
    isConnected: vi.fn().mockReturnValue(true),
    registerProject: vi.fn().mockResolvedValue({
      created: true,
      project_id: 'test-project-id',
      priority: 'high',
      is_active: true,
      newly_registered: true,
    }),
    deprioritizeProject: (...args: unknown[]) => mockDeprioritize(...args),
    heartbeat: vi.fn().mockResolvedValue({ acknowledged: true }),
    healthCheck: vi.fn().mockResolvedValue({ status: 1, components: [] }),
    ingestText: vi.fn().mockResolvedValue({ documentIds: [] }),
    embedText: vi.fn().mockResolvedValue({ success: true, embedding: new Array(384).fill(0.1) }),
    generateSparseVector: vi.fn().mockResolvedValue({ success: true, indices_values: {} }),
    enqueueItem: vi.fn().mockResolvedValue({ queue_id: 'q', is_new: true, idempotency_key: 'k' }),
    notifyServerStatus: vi.fn().mockResolvedValue({}),
    getStatus: vi.fn().mockResolvedValue({}),
    getMetrics: vi.fn().mockResolvedValue({}),
    logSearchEvent: vi.fn().mockResolvedValue({}),
    updateSearchEvent: vi.fn().mockResolvedValue({}),
    getConnectionState: vi.fn().mockReturnValue('connected'),
  })),
}));

vi.mock('@modelcontextprotocol/sdk/server/index.js', () => ({
  Server: vi.fn().mockImplementation(() => ({
    connect: vi.fn().mockResolvedValue(undefined),
    close: vi.fn().mockResolvedValue(undefined),
    setRequestHandler: vi.fn(),
    onerror: null,
    onclose: null,
  })),
}));

vi.mock('@modelcontextprotocol/sdk/server/stdio.js', () => ({
  StdioServerTransport: vi.fn().mockImplementation(() => ({})),
}));

vi.mock('@qdrant/js-client-rest', () => ({
  QdrantClient: vi.fn().mockImplementation(() => ({
    search: vi.fn().mockResolvedValue([]),
    retrieve: vi.fn().mockResolvedValue([]),
    scroll: vi.fn().mockResolvedValue({ points: [], next_page_offset: null }),
    upsert: vi.fn().mockResolvedValue({ status: 'completed' }),
    delete: vi.fn().mockResolvedValue({ status: 'completed' }),
    getCollections: vi.fn().mockResolvedValue({ collections: [] }),
  })),
}));

// ── Test helpers ──────────────────────────────────────────────────────────────

/** Create a temp dir, seed the DB, and return { tempDir, config }. */
function createTempEnv(): { tempDir: string; config: ServerConfig } {
  const tempDir = mkdtempSync(join(tmpdir(), 'mcp-cleanup-test-'));
  const dbPath = join(tempDir, 'state.db');
  const db = new Database(dbPath);
  db.exec(TEST_SCHEMA);
  db.close();
  return { tempDir, config: createTestConfig(tempDir) };
}

/** Create a temp git project dir and register it in the DB, return realpath. */
function createRegisteredProject(tempDir: string, name: string, tenantId: string): string {
  const projectPath = join(tempDir, name);
  mkdirSync(projectPath);
  mkdirSync(join(projectPath, '.git'));
  const realPath = realpathSync(projectPath);
  const db = new Database(join(tempDir, 'state.db'));
  db.prepare(
    `INSERT INTO watch_folders
     (watch_id, path, collection, tenant_id, is_active, created_at, updated_at)
     VALUES (?, ?, 'projects', ?, 1, datetime('now'), datetime('now'))`
  ).run(`watch-${name}`, realPath, tenantId);
  db.close();
  return realPath;
}

/**
 * Trigger the `onclose` callback registered on the MCP SDK Server mock.
 * The server sets `server.onclose = () => this.cleanupSession()` in setupHandlers().
 * We retrieve the mock instance and invoke that callback directly.
 */
function triggerOnclose(server: WorkspaceQdrantMcpServer): void {
  const mcpServer = server.getMcpServer();
  const closeFn = (mcpServer as unknown as { onclose: (() => void) | null }).onclose;
  if (closeFn) closeFn();
}

// ── Suite ─────────────────────────────────────────────────────────────────────

describe('cleanupSession idempotence (F-049)', () => {
  let tempDir: string;
  let config: ServerConfig;
  let server: WorkspaceQdrantMcpServer;
  let originalCwd: string;

  beforeEach(() => {
    register.resetMetrics();
    mockDeprioritize = vi
      .fn()
      .mockResolvedValue({ success: true, is_active: false, new_priority: 'normal' });
    mockConnect = vi.fn().mockResolvedValue(undefined);
    vi.clearAllMocks();
    originalCwd = process.cwd();
    ({ tempDir, config } = createTempEnv());
  });

  afterEach(async () => {
    process.chdir(originalCwd);
    // Ensure server is stopped if a test left it running (avoid resource leak)
    if (server) {
      try {
        await server.stop();
      } catch {
        /* already stopped */
      }
    }
    if (tempDir) rmSync(tempDir, { recursive: true, force: true });
  });

  // ── Scenario 1: stop() after onclose ───────────────────────────────────────

  it('stop() after onclose: sessionCount decremented exactly once', async () => {
    server = new WorkspaceQdrantMcpServer({ config, stdio: false });
    await server.start();

    const countAfterStart = await getSessionCountMetric();
    expect(countAfterStart).toBe(1);

    // Fire onclose (first cleanup)
    triggerOnclose(server);
    // Give microtasks a tick to settle
    await new Promise((r) => setImmediate(r));

    const countAfterClose = await getSessionCountMetric();
    expect(countAfterClose).toBe(0);

    // Now call stop() — second cleanup attempt should no-op
    await server.stop();

    const countAfterStop = await getSessionCountMetric();
    expect(countAfterStop).toBe(0);
  });

  // ── Scenario 2: onclose after stop() ──────────────────────────────────────

  it('onclose after stop(): sessionCount decremented exactly once', async () => {
    server = new WorkspaceQdrantMcpServer({ config, stdio: false });
    await server.start();

    expect(await getSessionCountMetric()).toBe(1);

    // stop() runs first cleanup
    await server.stop();
    expect(await getSessionCountMetric()).toBe(0);

    // onclose fires afterwards — should no-op
    triggerOnclose(server);
    await new Promise((r) => setImmediate(r));

    expect(await getSessionCountMetric()).toBe(0);
  });

  // ── Scenario 3: double stop() ─────────────────────────────────────────────

  it('double stop(): sessionCount decremented exactly once', async () => {
    server = new WorkspaceQdrantMcpServer({ config, stdio: false });
    await server.start();

    expect(await getSessionCountMetric()).toBe(1);

    // Fire two stop() calls in rapid succession
    await Promise.all([server.stop(), server.stop()]);

    expect(await getSessionCountMetric()).toBe(0);
  });

  // ── Scenario 4: cleanup with registered project ───────────────────────────

  it('cleanup with registered project: deprioritizeProject called exactly once', async () => {
    const realProjectPath = createRegisteredProject(tempDir, 'my-project', 'proj-id-1');
    process.chdir(realProjectPath);

    server = new WorkspaceQdrantMcpServer({ config, stdio: false });
    await server.start();

    const state = server.getSessionState();
    expect(state.projectId).not.toBeNull();

    // First cleanup via onclose
    triggerOnclose(server);
    await new Promise((r) => setImmediate(r));

    const callsAfterFirst = mockDeprioritize.mock.calls.length;
    expect(callsAfterFirst).toBe(1);

    // Second cleanup via stop() — should not call deprioritize again
    await server.stop();
    expect(mockDeprioritize.mock.calls.length).toBe(1);
  });

  // ── Scenario 5: cleanup without daemon ────────────────────────────────────

  it('cleanup without daemon: both onclose and stop() no-op gracefully', async () => {
    mockConnect = vi.fn().mockRejectedValue(new Error('daemon not available'));

    server = new WorkspaceQdrantMcpServer({ config, stdio: false });
    await server.start();

    expect(server.isDaemonConnected()).toBe(false);

    // onclose fires — no project registered, daemon not connected; should not throw
    let error: unknown;
    try {
      triggerOnclose(server);
      await new Promise((r) => setImmediate(r));
    } catch (e) {
      error = e;
    }
    expect(error).toBeUndefined();

    // stop() also safe
    await expect(server.stop()).resolves.toBeUndefined();

    // sessionCount decremented exactly once (from the first cleanup)
    expect(await getSessionCountMetric()).toBe(0);
  });

  // ── Scenario 6: heartbeat clearing ────────────────────────────────────────

  it('heartbeat clearing: interval cleared on first cleanup, null stays null on second', async () => {
    const realProjectPath = createRegisteredProject(tempDir, 'hb-project', 'hb-proj-id');
    process.chdir(realProjectPath);

    server = new WorkspaceQdrantMcpServer({ config, stdio: false });
    await server.start();

    const stateBefore = server.getSessionState();
    expect(stateBefore.heartbeatInterval).not.toBeNull();

    // First cleanup via stop()
    await server.stop();

    const stateAfterFirst = server.getSessionState();
    expect(stateAfterFirst.heartbeatInterval).toBeNull();

    // Second cleanup via onclose — clearInterval(null) must not throw
    let error: unknown;
    try {
      triggerOnclose(server);
      await new Promise((r) => setImmediate(r));
    } catch (e) {
      error = e;
    }
    expect(error).toBeUndefined();
    expect(server.getSessionState().heartbeatInterval).toBeNull();
  });
});
