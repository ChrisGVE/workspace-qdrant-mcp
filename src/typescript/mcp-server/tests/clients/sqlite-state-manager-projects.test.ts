/**
 * Tests for SqliteStateManager project methods and graceful degradation
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

      const result = manager.enqueueUnified('text', 'add', 'tenant1', 'collection1', {});

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

      const result = manager.enqueueUnified('text', 'add', 'tenant1', 'collection1', {});

      expect(result.status).toBe('degraded');
      expect(result.reason).toBe('database_not_found');
    });
  });
});
