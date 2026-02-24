/**
 * Rules mirror read/write operations for SqliteStateManager.
 *
 * Provides write-through caching for rules (Task 16).
 * The mirror enables fallback when Qdrant is unavailable.
 */

import type { Database as DatabaseType } from 'better-sqlite3';

export interface RulesMirrorEntry {
  ruleId: string;
  ruleText: string;
  scope: string | null;
  tenantId: string | null;
  createdAt: string;
  updatedAt: string;
}

/**
 * Upsert a rule into the rules_mirror table.
 * Called after successful Qdrant writes to enable rebuild recovery.
 */
export function upsertRulesMirror(
  db: DatabaseType | null,
  entry: RulesMirrorEntry,
): void {
  if (!db) return;

  try {
    db.prepare(
      `INSERT INTO rules_mirror (rule_id, rule_text, scope, tenant_id, created_at, updated_at)
       VALUES (?, ?, ?, ?, ?, ?)
       ON CONFLICT(rule_id) DO UPDATE SET
           rule_text = excluded.rule_text,
           scope = excluded.scope,
           tenant_id = excluded.tenant_id,
           updated_at = excluded.updated_at`
    ).run(
      entry.ruleId,
      entry.ruleText,
      entry.scope,
      entry.tenantId,
      entry.createdAt,
      entry.updatedAt
    );
  } catch {
    // rules_mirror table may not exist yet (daemon not initialized)
  }
}

/**
 * Delete a rule from the rules_mirror table.
 * Called after successful Qdrant deletes.
 */
export function deleteRulesMirror(
  db: DatabaseType | null,
  ruleId: string,
): void {
  if (!db) return;

  try {
    db.prepare(
      'DELETE FROM rules_mirror WHERE rule_id = ?'
    ).run(ruleId);
  } catch {
    // rules_mirror table may not exist yet
  }
}

/**
 * List rules from the rules_mirror table.
 * Used as fallback when Qdrant is unavailable.
 */
export function listRulesMirror(
  db: DatabaseType | null,
  scope?: string,
  tenantId?: string,
  limit = 50,
): RulesMirrorEntry[] {
  if (!db) return [];

  try {
    let sql = `SELECT rule_id AS ruleId, rule_text AS ruleText,
                      scope, tenant_id AS tenantId,
                      created_at AS createdAt, updated_at AS updatedAt
               FROM rules_mirror`;
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

    return db.prepare(sql).all(...params) as RulesMirrorEntry[];
  } catch {
    return [];
  }
}
