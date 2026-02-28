/**
 * Tests for tracked-files-queries: countTrackedFiles, listSubmodules, extractRepoName
 */

import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import Database, { type Database as DatabaseType } from 'better-sqlite3';

import {
  countTrackedFiles,
  listSubmodules,
  extractRepoName,
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

describe('countTrackedFiles', () => {
  let db: DatabaseType;

  beforeEach(() => {
    db = new Database(':memory:');
    db.exec(TRACKED_FILES_SCHEMA);
    seedProject(db);
    seedFile(db, 'src/main.rs', { language: 'rust', fileType: 'code' });
    seedFile(db, 'src/lib.rs', { language: 'rust', fileType: 'code' });
    seedFile(db, 'README.md', { fileType: 'text' });
  });

  afterEach(() => {
    db.close();
  });

  it('should count all files', () => {
    expect(countTrackedFiles(db, { watchFolderId: WATCH_ID })).toBe(3);
  });

  it('should count with path filter', () => {
    expect(countTrackedFiles(db, { watchFolderId: WATCH_ID, path: 'src' })).toBe(2);
  });

  it('should return 0 for null db', () => {
    expect(countTrackedFiles(null, { watchFolderId: WATCH_ID })).toBe(0);
  });
});

describe('listSubmodules', () => {
  let db: DatabaseType;

  beforeEach(() => {
    db = new Database(':memory:');
    db.exec(TRACKED_FILES_SCHEMA);
    seedProject(db);
  });

  afterEach(() => {
    db.close();
  });

  it('should list submodules with repo names', () => {
    db.prepare(
      `INSERT INTO watch_folders (watch_id, path, collection, tenant_id, parent_watch_id, submodule_path, git_remote_url, created_at, updated_at)
       VALUES (?, ?, 'projects', 'tenant-sub1', ?, ?, ?, ?, ?)`,
    ).run(
      'watch-sub1',
      '/home/user/project/vendor/lib-a',
      WATCH_ID,
      'vendor/lib-a',
      'https://github.com/org/lib-a.git',
      NOW,
      NOW,
    );

    db.prepare(
      `INSERT INTO watch_folders (watch_id, path, collection, tenant_id, parent_watch_id, submodule_path, git_remote_url, created_at, updated_at)
       VALUES (?, ?, 'projects', 'tenant-sub2', ?, ?, ?, ?, ?)`,
    ).run(
      'watch-sub2',
      '/home/user/project/vendor/lib-b',
      WATCH_ID,
      'vendor/lib-b',
      'git@github.com:org/lib-b.git',
      NOW,
      NOW,
    );

    const result = listSubmodules(db, WATCH_ID);
    expect(result.status).toBe('ok');
    expect(result.data).toHaveLength(2);
    expect(result.data[0]).toEqual({ submodulePath: 'vendor/lib-a', repoName: 'lib-a' });
    expect(result.data[1]).toEqual({ submodulePath: 'vendor/lib-b', repoName: 'lib-b' });
  });

  it('should return empty when no submodules', () => {
    const result = listSubmodules(db, WATCH_ID);
    expect(result.status).toBe('ok');
    expect(result.data).toHaveLength(0);
  });

  it('should skip entries without submodule_path', () => {
    db.prepare(
      `INSERT INTO watch_folders (watch_id, path, collection, tenant_id, parent_watch_id, created_at, updated_at)
       VALUES (?, ?, 'projects', 'tenant-sub3', ?, ?, ?)`,
    ).run('watch-sub3', '/home/user/project/other', WATCH_ID, NOW, NOW);

    const result = listSubmodules(db, WATCH_ID);
    expect(result.status).toBe('ok');
    expect(result.data).toHaveLength(0);
  });

  it('should return degraded when db is null', () => {
    const result = listSubmodules(null, WATCH_ID);
    expect(result.status).toBe('degraded');
    expect(result.data).toEqual([]);
  });
});

describe('extractRepoName', () => {
  it('should extract from HTTPS URL with .git', () => {
    expect(extractRepoName('https://github.com/user/my-repo.git', 'vendor/x')).toBe('my-repo');
  });

  it('should extract from HTTPS URL without .git', () => {
    expect(extractRepoName('https://github.com/user/my-repo', 'vendor/x')).toBe('my-repo');
  });

  it('should extract from SSH URL', () => {
    expect(extractRepoName('git@github.com:user/my-repo.git', 'vendor/x')).toBe('my-repo');
  });

  it('should extract from URL with trailing slash', () => {
    expect(extractRepoName('https://github.com/user/my-repo/', 'vendor/x')).toBe('my-repo');
  });

  it('should fall back to submodule path when URL is null', () => {
    expect(extractRepoName(null, 'vendor/some-lib')).toBe('some-lib');
  });

  it('should fall back to submodule path when URL is empty', () => {
    expect(extractRepoName('', 'deps/toolkit')).toBe('toolkit');
  });
});
