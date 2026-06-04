/**
 * Rules mirror read/write operations for SqliteStateManager.
 *
 * Write operations (upsert, delete) go through daemon gRPC.
 * Read operations (list) query SQLite directly for fallback support.
 */
import { TENANT_GLOBAL } from '../constants/tenants.js';
/**
 * Upsert a rule into the rules_mirror table via daemon gRPC.
 * Called after successful Qdrant writes to enable rebuild recovery.
 * Fire-and-forget: errors are swallowed since the mirror is advisory.
 */
export function upsertRulesMirror(daemonClient, entry) {
    if (!daemonClient)
        return;
    const request = {
        rule_id: entry.ruleId,
        rule_text: entry.ruleText,
        created_at: entry.createdAt,
        updated_at: entry.updatedAt,
    };
    if (entry.scope !== null)
        request.scope = entry.scope;
    if (entry.tenantId !== null)
        request.tenant_id = entry.tenantId;
    daemonClient.upsertRuleMirror(request).catch((err) => {
        // rules_mirror is advisory — errors must not break rule operations
        console.warn('upsertRuleMirror failed:', err instanceof Error ? err.message : err);
    });
}
/**
 * Delete a rule from the rules_mirror table via daemon gRPC.
 * Called after successful Qdrant deletes.
 * Fire-and-forget: errors are swallowed.
 */
export function deleteRulesMirror(daemonClient, ruleId) {
    if (!daemonClient)
        return;
    daemonClient.deleteRuleMirror({ rule_id: ruleId }).catch((err) => {
        // rules_mirror is advisory — errors must not break rule operations
        console.warn('deleteRuleMirror failed:', err instanceof Error ? err.message : err);
    });
}
/**
 * List rules from the rules_mirror table.
 * Read-only — queries SQLite directly. Used as fallback when Qdrant is unavailable.
 */
export function listRulesMirror(db, scope, tenantId, limit = 50) {
    if (!db)
        return [];
    try {
        let sql = `SELECT rule_id AS ruleId, rule_text AS ruleText,
                      scope, tenant_id AS tenantId,
                      created_at AS createdAt, updated_at AS updatedAt
               FROM rules_mirror`;
        const conditions = [];
        const params = [];
        if (scope === TENANT_GLOBAL) {
            conditions.push("(scope = 'global' OR scope IS NULL)");
        }
        else if (scope === 'project' && tenantId) {
            conditions.push("scope = 'project'");
            conditions.push('tenant_id = ?');
            params.push(tenantId);
        }
        if (conditions.length > 0) {
            sql += ' WHERE ' + conditions.join(' AND ');
        }
        sql += ' ORDER BY updatedAt DESC LIMIT ?';
        params.push(limit);
        return db.prepare(sql).all(...params);
    }
    catch (err) {
        const msg = err instanceof Error ? err.message : String(err);
        if (msg.includes('no such table')) {
            return [];
        }
        throw err;
    }
}
//# sourceMappingURL=rules-mirror-queries.js.map