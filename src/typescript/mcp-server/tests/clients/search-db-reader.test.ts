/**
 * Unit tests for SearchDbReader.
 *
 * Builds a real on-disk SQLite database with the v8 file_metadata schema
 * (mirroring `code_lines_schema::CREATE_FILE_METADATA_SQL` on the daemon
 * side), inserts a fixture, and exercises every filter combo. The aim is
 * to catch regressions in the WHERE-clause composition the admin route
 * relies on — particularly the "(none)" → IS NULL translation and the
 * skippedOnly filter, both of which are easy to break silently.
 */

import { afterEach, beforeEach, describe, expect, it } from 'vitest';
import Database from 'better-sqlite3';
import { mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

import { SearchDbReader } from '../../src/clients/search-db-reader.js';

/** Daemon-side `CREATE_FILE_METADATA_SQL` at search.db v8. */
const CREATE_FILE_METADATA_V8 = `
  CREATE TABLE IF NOT EXISTS file_metadata (
    file_id INTEGER PRIMARY KEY,
    tenant_id TEXT NOT NULL,
    branch TEXT,
    file_path TEXT NOT NULL,
    size_bytes INTEGER,
    fts5_skipped INTEGER NOT NULL DEFAULT 0,
    base_point TEXT,
    relative_path TEXT,
    file_hash TEXT
  )
`;

/** A handful of files spanning tenant/branch/skipped combinations. */
const FIXTURE: Array<[number, string, string | null, string, number | null, number]> = [
  [1, 'proj-a', 'main', '/a.rs', 100, 0],
  [2, 'proj-a', 'main', '/b.rs', 200, 0],
  [3, 'proj-a', 'main', '/big.csv', 50_000, 1],
  [4, 'proj-a', 'feature/x', '/c.rs', 75, 0],
  [5, 'proj-b', 'main', '/d.rs', 999, 0],
  // NULL branch — exercises the "(none)" filter path
  [6, 'proj-b', null, '/orphan.md', 42, 0],
  // NULL size_bytes — should sort last under DESC NULLS LAST
  [7, 'proj-c', 'main', '/legacy.md', null, 0],
];

function seed(dbPath: string): void {
  const db = new Database(dbPath);
  db.exec(CREATE_FILE_METADATA_V8);
  const stmt = db.prepare(
    'INSERT INTO file_metadata (file_id, tenant_id, branch, file_path, size_bytes, fts5_skipped) VALUES (?, ?, ?, ?, ?, ?)'
  );
  for (const [id, tenant, branch, path, size, skipped] of FIXTURE) {
    stmt.run(id, tenant, branch, path, size, skipped);
  }
  db.close();
}

describe('SearchDbReader', () => {
  let tmpDir: string;
  let dbPath: string;
  let reader: SearchDbReader;

  beforeEach(() => {
    tmpDir = mkdtempSync(join(tmpdir(), 'wqm-search-db-reader-'));
    dbPath = join(tmpDir, 'search.db');
    seed(dbPath);
    reader = new SearchDbReader({ dbPath });
  });

  afterEach(() => {
    reader.close();
    rmSync(tmpDir, { recursive: true, force: true });
  });

  it('lists files ordered by size_bytes DESC with NULLs last', () => {
    const rows = reader.listLargestFiles({ limit: 10 });
    expect(rows.map((r) => r.file_id)).toEqual([
      3, // 50000
      5, // 999
      2, // 200
      1, // 100
      4, // 75
      6, // 42
      7, // NULL → last
    ]);
  });

  it('respects the limit parameter and clamps to DEFAULT', () => {
    expect(reader.listLargestFiles({ limit: 3 })).toHaveLength(3);
    expect(reader.listLargestFiles({ limit: 0 }).length).toBeGreaterThan(0); // clamped to >=1
  });

  it('filters by tenant_id', () => {
    const rows = reader.listLargestFiles({ tenantId: 'proj-a' });
    expect(rows.every((r) => r.tenant_id === 'proj-a')).toBe(true);
    expect(rows.map((r) => r.file_id).sort()).toEqual([1, 2, 3, 4]);
  });

  it('filters by branch (literal value)', () => {
    const rows = reader.listLargestFiles({ branch: 'feature/x' });
    expect(rows.map((r) => r.file_id)).toEqual([4]);
  });

  it('translates branch "(none)" to IS NULL', () => {
    // Exact match path would never find the NULL row; the reader must
    // recognise the sentinel and rewrite to IS NULL.
    const rows = reader.listLargestFiles({ branch: '(none)' });
    expect(rows.map((r) => r.file_id)).toEqual([6]);
    expect(rows[0]?.branch).toBe('(none)'); // COALESCE projects this back out
  });

  it('combines tenant_id + branch filters', () => {
    const rows = reader.listLargestFiles({ tenantId: 'proj-a', branch: 'main' });
    expect(rows.map((r) => r.file_id).sort()).toEqual([1, 2, 3]);
  });

  it('returns only fts5_skipped=1 rows when skippedOnly is true', () => {
    const rows = reader.listLargestFiles({ skippedOnly: true });
    expect(rows.map((r) => r.file_id)).toEqual([3]);
    expect(rows[0]?.fts5_skipped).toBe(1);
  });

  it('returns degraded status when search.db is missing', () => {
    const missing = new SearchDbReader({ dbPath: join(tmpDir, 'does-not-exist.db') });
    const status = missing.initialize();
    expect(status.status).toBe('degraded');
    if (status.status === 'degraded') {
      expect(status.reason).toBe('database_not_found');
    }
    expect(missing.listLargestFiles()).toEqual([]);
  });

  it('returns [] when file_metadata table is missing', () => {
    // Open the db and drop the table to simulate a pre-v4 search.db.
    const db = new Database(dbPath);
    db.exec('DROP TABLE file_metadata');
    db.close();
    const noTable = new SearchDbReader({ dbPath });
    expect(noTable.listLargestFiles()).toEqual([]);
    noTable.close();
  });
});
