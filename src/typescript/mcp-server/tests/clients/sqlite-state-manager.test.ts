/**
 * Tests for SqliteStateManager
 */

import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import Database from 'better-sqlite3';
import { mkdtempSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

import {
  SqliteStateManager,
  generateIdempotencyKey,
  buildContentPayload,
  buildMemoryPayload,
  buildLibraryPayload,
} from '../../src/clients/sqlite-state-manager.js';

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

describe('generateIdempotencyKey', () => {
  it('should generate consistent 32-character hex key', () => {
    const key = generateIdempotencyKey('content', 'ingest', 'tenant1', 'collection1', {
      content: 'test',
    });

    expect(key).toHaveLength(32);
    expect(key).toMatch(/^[a-f0-9]+$/);
  });

  it('should generate same key for same inputs', () => {
    const payload = { content: 'test', source: 'user' };

    const key1 = generateIdempotencyKey('content', 'ingest', 'tenant1', 'collection1', payload);
    const key2 = generateIdempotencyKey('content', 'ingest', 'tenant1', 'collection1', payload);

    expect(key1).toBe(key2);
  });

  it('should generate different keys for different inputs', () => {
    const key1 = generateIdempotencyKey('content', 'ingest', 'tenant1', 'collection1', {
      a: 1,
    });
    const key2 = generateIdempotencyKey('content', 'ingest', 'tenant1', 'collection1', {
      a: 2,
    });

    expect(key1).not.toBe(key2);
  });

  it('should generate different keys for different operations', () => {
    const payload = { content: 'test' };

    const key1 = generateIdempotencyKey('content', 'ingest', 'tenant1', 'collection1', payload);
    const key2 = generateIdempotencyKey('content', 'update', 'tenant1', 'collection1', payload);

    expect(key1).not.toBe(key2);
  });

  it('should sort payload keys for consistency', () => {
    const key1 = generateIdempotencyKey('content', 'ingest', 'tenant1', 'collection1', {
      b: 2,
      a: 1,
    });
    const key2 = generateIdempotencyKey('content', 'ingest', 'tenant1', 'collection1', {
      a: 1,
      b: 2,
    });

    expect(key1).toBe(key2);
  });
});

describe('payload builders', () => {
  describe('buildContentPayload', () => {
    it('should build content payload with required fields', () => {
      const payload = buildContentPayload('test content', 'user_input');

      expect(payload).toEqual({
        content: 'test content',
        source_type: 'user_input',
        main_tag: undefined,
        full_tag: undefined,
      });
    });

    it('should build content payload with optional fields', () => {
      const payload = buildContentPayload('test', 'web', 'main_tag', 'full_tag');

      expect(payload).toEqual({
        content: 'test',
        source_type: 'web',
        main_tag: 'main_tag',
        full_tag: 'full_tag',
      });
    });
  });

  describe('buildMemoryPayload', () => {
    it('should build memory payload for global scope', () => {
      const payload = buildMemoryPayload('prefer-uv', 'Use uv for Python packages', 'global');

      expect(payload).toEqual({
        content: 'Use uv for Python packages',
        source_type: 'memory_rule',
        label: 'prefer-uv',
        scope: 'global',
        project_id: undefined,
      });
    });

    it('should build memory payload for project scope', () => {
      const payload = buildMemoryPayload('use-pytest', 'Use pytest', 'project', 'abc123');

      expect(payload).toEqual({
        content: 'Use pytest',
        source_type: 'memory_rule',
        label: 'use-pytest',
        scope: 'project',
        project_id: 'abc123',
      });
    });
  });

  describe('buildLibraryPayload', () => {
    it('should build library payload with required fields', () => {
      const payload = buildLibraryPayload('numpy');

      expect(payload).toEqual({
        library_name: 'numpy',
        content: undefined,
        source: undefined,
        url: undefined,
      });
    });

    it('should build library payload with optional fields', () => {
      const payload = buildLibraryPayload('numpy', 'doc content', 'web', 'https://numpy.org');

      expect(payload).toEqual({
        library_name: 'numpy',
        content: 'doc content',
        source: 'web',
        url: 'https://numpy.org',
      });
    });
  });
});

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
        'content',
        'ingest',
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
        'content',
        'ingest',
        'tenant1',
        'collection1',
        payload
      );
      const result2 = manager.enqueueUnified(
        'content',
        'ingest',
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
          'invalid' as 'content',
          'ingest',
          'tenant1',
          'collection1',
          {}
        );
      }).toThrow('Invalid item type');
    });

    it('should throw for invalid operation', () => {
      expect(() => {
        manager.enqueueUnified(
          'content',
          'invalid' as 'ingest',
          'tenant1',
          'collection1',
          {}
        );
      }).toThrow('Invalid operation');
    });

    it('should throw for invalid operation/item type combination', () => {
      expect(() => {
        // scan is not valid for content type
        manager.enqueueUnified('content', 'scan', 'tenant1', 'collection1', {});
      }).toThrow("Invalid operation 'scan' for item type 'content'");
    });

    it('should throw for empty tenant_id', () => {
      expect(() => {
        manager.enqueueUnified('content', 'ingest', '', 'collection1', {});
      }).toThrow('tenant_id cannot be empty');
    });

    it('should throw for invalid priority', () => {
      expect(() => {
        manager.enqueueUnified('content', 'ingest', 'tenant1', 'collection1', {}, 11);
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
      manager.enqueueUnified('content', 'ingest', 'tenant1', 'collection1', { a: 1 });
      manager.enqueueUnified('file', 'ingest', 'tenant1', 'collection1', { b: 2 });

      const result = manager.getQueueStats();

      expect(result.status).toBe('ok');
      expect(result.data!.total_pending).toBe(2);
      expect(result.data!.by_item_type.content).toBe(1);
      expect(result.data!.by_item_type.file).toBe(1);
    });
  });

  describe('project methods', () => {
    let manager: SqliteStateManager;
    let db: Database.Database;

    beforeEach(() => {
      // Insert test project into watch_folders
      db = new Database(dbPath);
      db.prepare(
        `
        INSERT INTO watch_folders
        (watch_id, path, collection, tenant_id, is_active, created_at, updated_at)
        VALUES ('watch-1', '/test/project', 'projects', 'abc123456789', 1, datetime('now'), datetime('now'))
      `
      ).run();
      db.close();

      manager = new SqliteStateManager({ dbPath });
      manager.initialize();
    });

    afterEach(() => {
      manager.close();
    });

    it('should get project by path', () => {
      const result = manager.getProjectByPath('/test/project');

      expect(result.status).toBe('ok');
      expect(result.data).not.toBeNull();
      expect(result.data!.project_id).toBe('abc123456789');
      expect(result.data!.is_active).toBe(true);
    });

    it('should return null for unknown path', () => {
      const result = manager.getProjectByPath('/unknown/path');

      expect(result.status).toBe('ok');
      expect(result.data).toBeNull();
    });

    it('should match subdirectory via longest-prefix', () => {
      const result = manager.getProjectByPath('/test/project/src/lib');

      expect(result.status).toBe('ok');
      expect(result.data).not.toBeNull();
      expect(result.data!.project_id).toBe('abc123456789');
      expect(result.data!.project_path).toBe('/test/project');
    });

    it('should return longest match when multiple projects match', () => {
      // Insert a parent project
      const db2 = new Database(dbPath);
      db2.prepare(
        `
        INSERT INTO watch_folders
        (watch_id, path, collection, tenant_id, is_active, created_at, updated_at)
        VALUES ('watch-parent', '/test', 'projects', 'parent000000', 1, datetime('now'), datetime('now'))
      `
      ).run();
      db2.close();

      const result = manager.getProjectByPath('/test/project/src');

      expect(result.status).toBe('ok');
      expect(result.data).not.toBeNull();
      // Should match the deeper /test/project, not /test
      expect(result.data!.project_id).toBe('abc123456789');
      expect(result.data!.project_path).toBe('/test/project');
    });

    it('should not match false prefix', () => {
      // /test/project-extra should NOT match /test/project
      const result = manager.getProjectByPath('/test/project-extra');

      expect(result.status).toBe('ok');
      expect(result.data).toBeNull();
    });

    it('should get project by ID', () => {
      const result = manager.getProjectById('abc123456789');

      expect(result.status).toBe('ok');
      expect(result.data).not.toBeNull();
      expect(result.data!.project_path).toBe('/test/project');
    });

    it('should list active projects', () => {
      const result = manager.listActiveProjects();

      expect(result.status).toBe('ok');
      expect(result.data).toHaveLength(1);
      expect(result.data[0].project_id).toBe('abc123456789');
    });
  });

  describe('graceful degradation', () => {
    it('should return degraded for missing unified_queue table', () => {
      // Create database without unified_queue table
      const noDB = new Database(dbPath);
      noDB.exec('DROP TABLE unified_queue');
      noDB.close();

      const manager = new SqliteStateManager({ dbPath });
      manager.initialize();

      const result = manager.enqueueUnified('content', 'ingest', 'tenant1', 'collection1', {});

      expect(result.status).toBe('degraded');
      expect(result.reason).toBe('table_not_found');

      manager.close();
    });

    it('should return degraded for missing watch_folders table', () => {
      // Create database without watch_folders table
      const noDB = new Database(dbPath);
      noDB.exec('DROP TABLE watch_folders');
      noDB.close();

      const manager = new SqliteStateManager({ dbPath });
      manager.initialize();

      const result = manager.getProjectByPath('/test');

      expect(result.status).toBe('degraded');
      expect(result.reason).toBe('table_not_found');

      manager.close();
    });

    it('should return degraded when not connected', () => {
      const manager = new SqliteStateManager({ dbPath: '/nonexistent/db.db' });
      // Don't initialize

      const result = manager.enqueueUnified('content', 'ingest', 'tenant1', 'collection1', {});

      expect(result.status).toBe('degraded');
      expect(result.reason).toBe('database_not_found');
    });
  });
});
