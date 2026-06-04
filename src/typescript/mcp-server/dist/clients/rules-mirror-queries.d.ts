/**
 * Rules mirror read/write operations for SqliteStateManager.
 *
 * Write operations (upsert, delete) go through daemon gRPC.
 * Read operations (list) query SQLite directly for fallback support.
 */
import type { Database as DatabaseType } from 'better-sqlite3';
import type { DaemonClient } from './daemon-client.js';
export interface RulesMirrorEntry {
    ruleId: string;
    ruleText: string;
    scope: string | null;
    tenantId: string | null;
    createdAt: string;
    updatedAt: string;
}
/**
 * Upsert a rule into the rules_mirror table via daemon gRPC.
 * Called after successful Qdrant writes to enable rebuild recovery.
 * Fire-and-forget: errors are swallowed since the mirror is advisory.
 */
export declare function upsertRulesMirror(daemonClient: DaemonClient | null, entry: RulesMirrorEntry): void;
/**
 * Delete a rule from the rules_mirror table via daemon gRPC.
 * Called after successful Qdrant deletes.
 * Fire-and-forget: errors are swallowed.
 */
export declare function deleteRulesMirror(daemonClient: DaemonClient | null, ruleId: string): void;
/**
 * List rules from the rules_mirror table.
 * Read-only — queries SQLite directly. Used as fallback when Qdrant is unavailable.
 */
export declare function listRulesMirror(db: DatabaseType | null, scope?: string, tenantId?: string, limit?: number): RulesMirrorEntry[];
//# sourceMappingURL=rules-mirror-queries.d.ts.map