/**
 * Tests for ProjectDetector: getCurrentProject, getCurrentProjectId, and isGitRepository
 */

import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import Database from 'better-sqlite3';
import { mkdtempSync, rmSync, mkdirSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

import { ProjectDetector, isGitRepository } from '../../src/utils/project-detector.js';
import { SqliteStateManager } from '../../src/clients/sqlite-state-manager.js';

// Create test schema (watch_folders table from daemon)
const TEST_SCHEMA = `
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

describe('ProjectDetector', () => {
  let tempDir: string;
  let dbPath: string;
  let stateManager: SqliteStateManager;

  beforeEach(() => {
    tempDir = mkdtempSync(join(tmpdir(), 'project-detector-test-'));
    dbPath = join(tempDir, 'state.db');

    // Create database with test schema
    const db = new Database(dbPath);
    db.exec(TEST_SCHEMA);
    db.close();

    stateManager = new SqliteStateManager({ dbPath });
    stateManager.initialize();
  });

  afterEach(() => {
    stateManager.close();
    rmSync(tempDir, { recursive: true, force: true });
  });

  describe('getCurrentProject', () => {
    it('should find and return current project info', async () => {
      const projectPath = join(tempDir, 'current-project');
      mkdirSync(projectPath);
      mkdirSync(join(projectPath, '.git'));
      const srcPath = join(projectPath, 'src');
      mkdirSync(srcPath);

      // Register project
      const db = new Database(dbPath);
      db.prepare(
        `
        INSERT INTO watch_folders
        (watch_id, path, collection, tenant_id, is_active, created_at, updated_at)
        VALUES ('watch-3', ?, 'projects', 'current12345', 1, datetime('now'), datetime('now'))
      `
      ).run(projectPath);
      db.close();

      stateManager.close();
      stateManager = new SqliteStateManager({ dbPath });
      stateManager.initialize();

      const detector = new ProjectDetector({ stateManager });
      const info = await detector.getCurrentProject(srcPath);

      expect(info).not.toBeNull();
      expect(info!.projectId).toBe('current12345');
    });

    it('should resolve subdirectory without project markers', async () => {
      // No .git or other markers — database longest-prefix matching handles resolution
      const projectPath = join(tempDir, 'markerless-project');
      mkdirSync(projectPath);
      const deepPath = join(projectPath, 'src', 'deep', 'nested');
      mkdirSync(deepPath, { recursive: true });

      // Register project
      const db = new Database(dbPath);
      db.prepare(
        `
        INSERT INTO watch_folders
        (watch_id, path, collection, tenant_id, is_active, created_at, updated_at)
        VALUES ('watch-5', ?, 'projects', 'markerless123', 1, datetime('now'), datetime('now'))
      `
      ).run(projectPath);
      db.close();

      stateManager.close();
      stateManager = new SqliteStateManager({ dbPath });
      stateManager.initialize();

      const detector = new ProjectDetector({ stateManager });
      const info = await detector.getCurrentProject(deepPath);

      expect(info).not.toBeNull();
      expect(info!.projectId).toBe('markerless123');
      expect(info!.projectPath).toBe(projectPath);
    });

    it('should return null when path not registered', async () => {
      const unregisteredPath = join(tempDir, 'not-registered');
      mkdirSync(unregisteredPath);

      const detector = new ProjectDetector({ stateManager });
      const info = await detector.getCurrentProject(unregisteredPath);

      expect(info).toBeNull();
    });
  });

  describe('getCurrentProjectId', () => {
    it('should return just the project_id', async () => {
      const projectPath = join(tempDir, 'id-project');
      mkdirSync(projectPath);
      mkdirSync(join(projectPath, '.git'));

      // Register project
      const db = new Database(dbPath);
      db.prepare(
        `
        INSERT INTO watch_folders
        (watch_id, path, collection, tenant_id, is_active, created_at, updated_at)
        VALUES ('watch-4', ?, 'projects', 'justid123456', 1, datetime('now'), datetime('now'))
      `
      ).run(projectPath);
      db.close();

      stateManager.close();
      stateManager = new SqliteStateManager({ dbPath });
      stateManager.initialize();

      const detector = new ProjectDetector({ stateManager });
      const projectId = await detector.getCurrentProjectId(projectPath);

      expect(projectId).toBe('justid123456');
    });
  });
});

describe('isGitRepository', () => {
  let tempDir: string;

  beforeEach(() => {
    tempDir = mkdtempSync(join(tmpdir(), 'git-test-'));
  });

  afterEach(() => {
    rmSync(tempDir, { recursive: true, force: true });
  });

  it('should return true for git repository', () => {
    const repoPath = join(tempDir, 'repo');
    mkdirSync(repoPath);
    mkdirSync(join(repoPath, '.git'));

    expect(isGitRepository(repoPath)).toBe(true);
  });

  it('should return false for non-git directory', () => {
    const nonRepoPath = join(tempDir, 'not-repo');
    mkdirSync(nonRepoPath);

    expect(isGitRepository(nonRepoPath)).toBe(false);
  });

  it('should return false for .git file (worktrees)', () => {
    const worktreePath = join(tempDir, 'worktree');
    mkdirSync(worktreePath);
    writeFileSync(join(worktreePath, '.git'), 'gitdir: ../main/.git/worktrees/worktree');

    // .git is a file, not a directory
    expect(isGitRepository(worktreePath)).toBe(false);
  });
});
