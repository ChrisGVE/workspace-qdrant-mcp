/**
 * Tests for SqliteStateManager initialization, enqueueUnified, and getQueueStats
 */

import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import Database from 'better-sqlite3';
import { mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

import { SqliteStateManager } from '../../src/clients/sqlite-state-manager.js';

// Create test schema (minimal version matching daemon's watch_folders table)
const TEST_SCHEMA = `
CREATE TABLE IF NOT EXISTS unified_queue (
    queue_id TEXT PRIMARY KEY,
    idempotency_key TEXT UNIQUE NOT NULL,
    item_type TEXT NOT NULL,
    op TEXT NOT NULL,
    tenant_id TEXT NOT NULL,
    collection TEXT NOT NULL,
    priority INTEGER DEFAULT 5,
    status TEXT DEFAULT 'pending',
    branch TEXT,
    payload_json TEXT,
    metadata TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    retry_count INTEGER DEFAULT 0,
    max_retries INTEGER DEFAULT 3,
    last_error TEXT,
    leased_by TEXT,
    lease_expires_at TEXT
);

CREATE TABLE IF NOT EXISTS watch_folders (
    watch_id TEXT PRIMARY KEY,
    path TEXT NOT NULL UNIQUE,
    collection TEXT NOT NULL,
    tenant_id TEXT NOT NULL,
    parent_watch_id TEXT,
    submodule_path TEXT,
    git_remote_url TEXT,
    remote_hash TEXT,
    disambiguation_path TEXT,
    is_active INTEGER DEFAULT 0,
    last_activity_at TEXT,
    library_mode TEXT,
    follow_symlinks INTEGER DEFAULT 0,
    enabled INTEGER DEFAULT 1,
    cleanup_on_disable INTEGER DEFAULT 0,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    last_scan TEXT
);
`;

describe('SqliteStateManager', () => {
  let tempDir: string;
  let dbPath: string;

  beforeEach(() => {
    tempDir = mkdtempSync(join(tmpdir(), 'sqlite-test-'));
    dbPath = join(tempDir, 'state.db');

    // Create database with test schema
    const db = new Database(dbPath);
    db.exec(TEST_SCHEMA);
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
    });

    afterEach(() => {
      manager.close();
    });

    it('should enqueue new item successfully', () => {
      const result = manager.enqueueUnified(
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

    it('should return existing item for duplicate idempotency key', () => {
      const payload = { content: 'test' };

      const result1 = manager.enqueueUnified(
        'text',
        'add',
        'tenant1',
        'collection1',
        payload
      );
      const result2 = manager.enqueueUnified(
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

    it('should throw for invalid item type', () => {
      expect(() => {
        manager.enqueueUnified(
          'invalid' as 'text',
          'add',
          'tenant1',
          'collection1',
          {}
        );
      }).toThrow('Invalid item type');
    });

    it('should throw for invalid operation', () => {
      expect(() => {
        manager.enqueueUnified(
          'text',
          'invalid' as 'add',
          'tenant1',
          'collection1',
          {}
        );
      }).toThrow('Invalid operation');
    });

    it('should throw for invalid operation/item type combination', () => {
      expect(() => {
        // scan is not valid for text type
        manager.enqueueUnified('text', 'scan', 'tenant1', 'collection1', {});
      }).toThrow("Invalid operation 'scan' for item type 'text'");
    });

    it('should throw for empty tenant_id', () => {
      expect(() => {
        manager.enqueueUnified('text', 'add', '', 'collection1', {});
      }).toThrow('tenant_id cannot be empty');
    });

    it('should throw for invalid priority', () => {
      expect(() => {
        manager.enqueueUnified('text', 'add', 'tenant1', 'collection1', {}, 11);
      }).toThrow('Priority must be between 0 and 10');
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

    it('should return empty stats for empty queue', () => {
      const result = manager.getQueueStats();

      expect(result.status).toBe('ok');
      expect(result.data!.total_pending).toBe(0);
      expect(result.data!.total_in_progress).toBe(0);
    });

    it('should return correct stats after enqueueing', () => {
      manager.enqueueUnified('text', 'add', 'tenant1', 'collection1', { a: 1 });
      manager.enqueueUnified('file', 'add', 'tenant1', 'collection1', { b: 2 });

      const result = manager.getQueueStats();

      expect(result.status).toBe('ok');
      expect(result.data!.total_pending).toBe(2);
      expect(result.data!.by_item_type.text).toBe(1);
      expect(result.data!.by_item_type.file).toBe(1);
    });
  });
});
