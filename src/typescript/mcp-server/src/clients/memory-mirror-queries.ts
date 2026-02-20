/**
 * Memory mirror read/write operations for SqliteStateManager.
 *
 * Provides write-through caching for memory rules (Task 16).
 * The mirror enables fallback when Qdrant is unavailable.
 */

import type { Database as DatabaseType } from 'better-sqlite3';

export interface MemoryMirrorEntry {
  memoryId: string;
  ruleText: string;
  scope: string | null;
  tenantId: string | null;
  createdAt: string;
  updatedAt: string;
}

/**
 * Upsert a memory rule into the memory_mirror table.
 * Called after successful Qdrant writes to enable rebuild recovery.
 */
export function upsertMemoryMirror(
  db: DatabaseType | null,
  entry: MemoryMirrorEntry,
): void {
  if (!db) return;

  try {
    db.prepare(
      `INSERT INTO memory_mirror (memory_id, rule_text, scope, tenant_id, created_at, updated_at)
       VALUES (?, ?, ?, ?, ?, ?)
       ON CONFLICT(memory_id) DO UPDATE SET
           rule_text = excluded.rule_text,
           scope = excluded.scope,
           tenant_id = excluded.tenant_id,
           updated_at = excluded.updated_at`
    ).run(
      entry.memoryId,
      entry.ruleText,
      entry.scope,
      entry.tenantId,
      entry.createdAt,
      entry.updatedAt
    );
  } catch {
    // memory_mirror table may not exist yet (daemon not initialized)
  }
}

/**
 * Delete a memory rule from the memory_mirror table.
 * Called after successful Qdrant deletes.
 */
export function deleteMemoryMirror(
  db: DatabaseType | null,
  memoryId: string,
): void {
  if (!db) return;

  try {
    db.prepare(
      'DELETE FROM memory_mirror WHERE memory_id = ?'
    ).run(memoryId);
  } catch {
    // memory_mirror table may not exist yet
  }
}

/**
 * List memory rules from the memory_mirror table.
 * Used as fallback when Qdrant is unavailable.
 */
export function listMemoryMirror(
  db: DatabaseType | null,
  scope?: string,
  tenantId?: string,
  limit = 50,
): MemoryMirrorEntry[] {
  if (!db) return [];

  try {
    let sql = `SELECT memory_id AS memoryId, rule_text AS ruleText,
                      scope, tenant_id AS tenantId,
                      created_at AS createdAt, updated_at AS updatedAt
               FROM memory_mirror`;
    const conditions: string[] = [];
    const params: (string | number)[] = [];

    if (scope === 'global') {
      conditions.push("(scope = 'global' OR scope IS NULL)");
    } else if (scope === 'project' && tenantId) {
      conditions.push("scope = 'project'");
      conditions.push('tenant_id = ?');
      params.push(tenantId);
    }

    if (conditions.length > 0) {
      sql += ' WHERE ' + conditions.join(' AND ');
    }

    sql += ' ORDER BY updatedAt DESC LIMIT ?';
    params.push(limit);

    return db.prepare(sql).all(...params) as MemoryMirrorEntry[];
  } catch {
    return [];
  }
}
