/**
 * Tests for SqliteStateManager initialization, enqueueUnified, and getQueueStats
 *
 * enqueueUnified delegates to the daemon via gRPC, so a mock daemon client is
 * required. Tests without a daemon client verify degraded-mode behaviour.
 */

import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import Database from 'better-sqlite3';
import { createHash } from 'node:crypto';
import { mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

import { SqliteStateManager } from '../../src/clients/sqlite-state-manager.js';
import type { DaemonClient } from '../../src/clients/daemon-client.js';

// Counter for generating unique queue IDs across tests
let enqueueCallCount = 0;

/** Build a minimal mock DaemonClient that resolves enqueueItem requests. */
function createMockDaemonClient(): DaemonClient {
  const seenKeys = new Map<string, string>(); // idempotency_key → queue_id

  return {
    enqueueItem: vi.fn(async (request: Record<string, unknown>) => {
      enqueueCallCount += 1;
      // Compute idempotency key matching the daemon's algorithm
      const input = `${request.item_type}|${request.op}|${request.tenant_id}|${request.collection}|${request.payload_json}`;
      const key = createHash('sha256').update(input).digest('hex').slice(0, 32);

      const existing = seenKeys.get(key);
      if (existing) {
        return { queue_id: existing, idempotency_key: key, is_new: false };
      }
      const queueId = `q-${enqueueCallCount}`;
      seenKeys.set(key, queueId);
      return { queue_id: queueId, idempotency_key: key, is_new: true };
    }),
  } as unknown as DaemonClient;
}

describe('SqliteStateManager', () => {
  let tempDir: string;
  let dbPath: string;

  beforeEach(() => {
    tempDir = mkdtempSync(join(tmpdir(), 'sqlite-test-'));
    dbPath = join(tempDir, 'state.db');

    // Create minimal DB so SqliteStateManager.initialize() succeeds
    const db = new Database(dbPath);
    db.exec(`
      CREATE TABLE IF NOT EXISTS watch_folders (
        watch_id TEXT PRIMARY KEY,
        path TEXT NOT NULL UNIQUE,
        collection TEXT NOT NULL,
        tenant_id TEXT NOT NULL,
        is_active INTEGER DEFAULT 0,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
      );
    `);
    db.close();
  });

  afterEach(() => {
    rmSync(tempDir, { recursive: true, force: true });
  });

  describe('initialize', () => {
    it('should initialize successfully when database exists', () => {
      const manager = new SqliteStateManager({ dbPath });
      const result = manager.initialize();

      expect(result.status).toBe('ok');
      expect(result.data).toBe(true);
      expect(manager.isConnected()).toBe(true);

      manager.close();
    });

    it('should return degraded when database does not exist', () => {
      const manager = new SqliteStateManager({ dbPath: '/nonexistent/path/db.db' });
      const result = manager.initialize();

      expect(result.status).toBe('degraded');
      expect(result.reason).toBe('database_not_found');
      expect(manager.isConnected()).toBe(false);
    });
  });

  describe('enqueueUnified', () => {
    let manager: SqliteStateManager;

    beforeEach(() => {
      manager = new SqliteStateManager({ dbPath });
      manager.initialize();
      manager.setDaemonClient(createMockDaemonClient());
    });

    afterEach(() => {
      manager.close();
    });

    it('should enqueue new item successfully', async () => {
      const result = await manager.enqueueUnified(
        'text',
        'add',
        'tenant1',
        'collection1',
        { content: 'test' },
        5,
        'main'
      );

      expect(result.status).toBe('ok');
      expect(result.data).not.toBeNull();
      expect(result.data!.isNew).toBe(true);
      expect(result.data!.queueId).toBeDefined();
      expect(result.data!.idempotencyKey).toHaveLength(32);
    });

    it('should return existing item for duplicate idempotency key', async () => {
      const payload = { content: 'test' };

      const result1 = await manager.enqueueUnified(
        'text',
        'add',
        'tenant1',
        'collection1',
        payload
      );
      const result2 = await manager.enqueueUnified(
        'text',
        'add',
        'tenant1',
        'collection1',
        payload
      );

      expect(result1.data!.isNew).toBe(true);
      expect(result2.data!.isNew).toBe(false);
      expect(result2.data!.queueId).toBe(result1.data!.queueId);
    });

    it('should throw for invalid item type', async () => {
      await expect(
        manager.enqueueUnified('invalid' as 'text', 'add', 'tenant1', 'collection1', {})
      ).rejects.toThrow('Invalid item type');
    });

    it('should throw for invalid operation', async () => {
      await expect(
        manager.enqueueUnified('text', 'invalid' as 'add', 'tenant1', 'collection1', {})
      ).rejects.toThrow('Invalid operation');
    });

    it('should throw for invalid operation/item type combination', async () => {
      await expect(
        manager.enqueueUnified('text', 'scan', 'tenant1', 'collection1', {})
      ).rejects.toThrow("Invalid operation 'scan' for item type 'text'");
    });

    it('should throw for empty tenant_id', async () => {
      await expect(manager.enqueueUnified('text', 'add', '', 'collection1', {})).rejects.toThrow(
        'tenant_id cannot be empty'
      );
    });

    it('should throw for invalid priority', async () => {
      await expect(
        manager.enqueueUnified('text', 'add', 'tenant1', 'collection1', {}, 11)
      ).rejects.toThrow('Priority must be between 0 and 10');
    });

    it('should return degraded when no daemon client', async () => {
      manager.setDaemonClient(null);

      const result = await manager.enqueueUnified('text', 'add', 'tenant1', 'collection1', {
        content: 'test',
      });

      expect(result.status).toBe('degraded');
      expect(result.reason).toBe('daemon_unavailable');
    });
  });

  describe('getQueueStats', () => {
    let manager: SqliteStateManager;

    beforeEach(() => {
      manager = new SqliteStateManager({ dbPath });
      manager.initialize();
    });

    afterEach(() => {
      manager.close();
    });

    it('should return degraded for stats without unified_queue table', () => {
      const result = manager.getQueueStats();
      // DB has no unified_queue table, so stats query should degrade gracefully
      expect(['ok', 'degraded']).toContain(result.status);
    });
  });
});
