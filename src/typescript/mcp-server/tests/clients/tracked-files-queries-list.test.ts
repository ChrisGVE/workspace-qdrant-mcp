/**
 * Tests for tracked-files-queries: listTrackedFiles
 */

import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import Database, { type Database as DatabaseType } from 'better-sqlite3';

import {
  listTrackedFiles,
} from '../../src/clients/tracked-files-queries/index.js';

const TRACKED_FILES_SCHEMA = `
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
    last_scan TEXT,
    FOREIGN KEY (parent_watch_id) REFERENCES watch_folders(watch_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS tracked_files (
    file_id INTEGER PRIMARY KEY AUTOINCREMENT,
    watch_folder_id TEXT NOT NULL,
    file_path TEXT NOT NULL,
    branch TEXT,
    file_type TEXT,
    language TEXT,
    file_mtime TEXT NOT NULL,
    file_hash TEXT NOT NULL,
    chunk_count INTEGER DEFAULT 0,
    chunking_method TEXT,
    lsp_status TEXT DEFAULT 'none',
    treesitter_status TEXT DEFAULT 'none',
    last_error TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    collection TEXT NOT NULL DEFAULT 'projects',
    extension TEXT,
    is_test INTEGER DEFAULT 0,
    base_point TEXT,
    relative_path TEXT,
    FOREIGN KEY (watch_folder_id) REFERENCES watch_folders(watch_id),
    UNIQUE(watch_folder_id, file_path, branch)
);
`;

const WATCH_ID = 'watch-001';
const NOW = '2026-02-24T12:00:00Z';

function seedProject(db: DatabaseType): void {
  db.prepare(
    `INSERT INTO watch_folders (watch_id, path, collection, tenant_id, is_active, created_at, updated_at)
     VALUES (?, ?, 'projects', 'tenant-001', 1, ?, ?)`,
  ).run(WATCH_ID, '/home/user/project', NOW, NOW);
}

function seedFile(
  db: DatabaseType,
  relativePath: string,
  opts: {
    fileType?: string;
    language?: string;
    extension?: string;
    isTest?: boolean;
    branch?: string;
  } = {},
): void {
  const ext = opts.extension ?? relativePath.split('.').pop() ?? null;
  db.prepare(
    `INSERT INTO tracked_files
     (watch_folder_id, file_path, relative_path, file_type, language, extension, is_test, branch, file_mtime, file_hash, created_at, updated_at)
     VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
  ).run(
    WATCH_ID,
    `/home/user/project/${relativePath}`,
    relativePath,
    opts.fileType ?? 'code',
    opts.language ?? null,
    ext,
    opts.isTest ? 1 : 0,
    opts.branch ?? 'main',
    NOW,
    'hash-' + relativePath,
    NOW,
    NOW,
  );
}

describe('listTrackedFiles', () => {
  let db: DatabaseType;

  beforeEach(() => {
    db = new Database(':memory:');
    db.exec(TRACKED_FILES_SCHEMA);
    seedProject(db);

    // Seed a realistic file set
    seedFile(db, 'src/main.rs', { language: 'rust', fileType: 'code' });
    seedFile(db, 'src/lib.rs', { language: 'rust', fileType: 'code' });
    seedFile(db, 'src/utils/helpers.rs', { language: 'rust', fileType: 'code' });
    seedFile(db, 'src/server.ts', { language: 'typescript', fileType: 'code' });
    seedFile(db, 'tests/test_main.rs', { language: 'rust', fileType: 'code', isTest: true });
    seedFile(db, 'README.md', { fileType: 'text', extension: 'md' });
    seedFile(db, 'Cargo.toml', { fileType: 'build', extension: 'toml' });
    seedFile(db, 'config.yaml', { fileType: 'config', extension: 'yaml' });
  });

  afterEach(() => {
    db.close();
  });

  it('should list all files for a watch folder', () => {
    const result = listTrackedFiles(db, { watchFolderId: WATCH_ID });
    expect(result.status).toBe('ok');
    expect(result.data).toHaveLength(8);
    expect(result.data[0].relativePath).toBe('Cargo.toml'); // sorted ASC
  });

  it('should filter by path prefix', () => {
    const result = listTrackedFiles(db, { watchFolderId: WATCH_ID, path: 'src' });
    expect(result.status).toBe('ok');
    expect(result.data).toHaveLength(4);
    for (const f of result.data) {
      expect(f.relativePath).toMatch(/^src\//);
    }
  });

  it('should filter by nested path prefix', () => {
    const result = listTrackedFiles(db, { watchFolderId: WATCH_ID, path: 'src/utils' });
    expect(result.status).toBe('ok');
    expect(result.data).toHaveLength(1);
    expect(result.data[0].relativePath).toBe('src/utils/helpers.rs');
  });

  it('should filter by fileType', () => {
    const result = listTrackedFiles(db, { watchFolderId: WATCH_ID, fileType: 'code' });
    expect(result.status).toBe('ok');
    expect(result.data).toHaveLength(5);
  });

  it('should filter by language', () => {
    const result = listTrackedFiles(db, { watchFolderId: WATCH_ID, language: 'rust' });
    expect(result.status).toBe('ok');
    expect(result.data).toHaveLength(4); // main.rs, lib.rs, helpers.rs, test_main.rs
  });

  it('should filter by extension', () => {
    const result = listTrackedFiles(db, { watchFolderId: WATCH_ID, extension: 'rs' });
    expect(result.status).toBe('ok');
    expect(result.data).toHaveLength(4); // main.rs, lib.rs, helpers.rs, test_main.rs
  });

  it('should exclude test files when includeTests is false', () => {
    const result = listTrackedFiles(db, { watchFolderId: WATCH_ID, includeTests: false });
    expect(result.status).toBe('ok');
    expect(result.data).toHaveLength(7);
    for (const f of result.data) {
      expect(f.isTest).toBe(false);
    }
  });

  it('should respect limit', () => {
    const result = listTrackedFiles(db, { watchFolderId: WATCH_ID, limit: 3 });
    expect(result.status).toBe('ok');
    expect(result.data).toHaveLength(3);
  });

  it('should combine filters', () => {
    const result = listTrackedFiles(db, {
      watchFolderId: WATCH_ID,
      path: 'src',
      language: 'rust',
    });
    expect(result.status).toBe('ok');
    expect(result.data).toHaveLength(3); // main.rs, lib.rs, helpers.rs
  });

  it('should return empty for non-existent path', () => {
    const result = listTrackedFiles(db, { watchFolderId: WATCH_ID, path: 'nonexistent' });
    expect(result.status).toBe('ok');
    expect(result.data).toHaveLength(0);
  });

  it('should return degraded when db is null', () => {
    const result = listTrackedFiles(null, { watchFolderId: WATCH_ID });
    expect(result.status).toBe('degraded');
    expect(result.reason).toBe('database_not_found');
    expect(result.data).toEqual([]);
  });

  it('should map fields correctly', () => {
    const result = listTrackedFiles(db, { watchFolderId: WATCH_ID, path: 'src', language: 'typescript' });
    expect(result.data).toHaveLength(1);
    const file = result.data[0];
    expect(file.relativePath).toBe('src/server.ts');
    expect(file.fileType).toBe('code');
    expect(file.language).toBe('typescript');
    expect(file.extension).toBe('ts');
    expect(file.isTest).toBe(false);
  });
});
