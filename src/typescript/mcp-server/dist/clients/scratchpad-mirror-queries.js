/**
 * Scratchpad mirror read/write operations for SqliteStateManager.
 *
 * Write operations (upsert, delete) go through daemon gRPC.
 * Read operations (list) query SQLite directly for fallback/rebuild support.
 */
/**
 * Upsert a scratchpad entry into the scratchpad_mirror table via daemon gRPC.
 * Fire-and-forget: errors are swallowed since the mirror is advisory.
 */
export function upsertScratchpadMirror(daemonClient, entry) {
    if (!daemonClient)
        return;
    const request = {
        scratchpad_id: entry.scratchpadId,
        content: entry.content,
        tenant_id: entry.tenantId,
        created_at: entry.createdAt,
        updated_at: entry.updatedAt,
    };
    if (entry.title !== null)
        request.title = entry.title;
    if (entry.tags !== '[]')
        request.tags = entry.tags;
    daemonClient.upsertScratchpadMirror(request).catch((err) => {
        console.warn('upsertScratchpadMirror failed:', err instanceof Error ? err.message : err);
    });
}
/**
 * Delete a scratchpad entry from the scratchpad_mirror table via daemon gRPC.
 * Fire-and-forget: errors are swallowed.
 */
export function deleteScratchpadMirror(daemonClient, scratchpadId) {
    if (!daemonClient)
        return;
    daemonClient.deleteScratchpadMirror({ scratchpad_id: scratchpadId }).catch((err) => {
        console.warn('deleteScratchpadMirror failed:', err instanceof Error ? err.message : err);
    });
}
/**
 * List scratchpad entries from the scratchpad_mirror table.
 * Read-only — queries SQLite directly.
 */
export function listScratchpadMirror(db, tenantId, limit = 100) {
    if (!db)
        return [];
    try {
        let sql = `SELECT scratchpad_id AS scratchpadId, title, content, tags,
                      tenant_id AS tenantId,
                      created_at AS createdAt, updated_at AS updatedAt
               FROM scratchpad_mirror`;
        const params = [];
        if (tenantId) {
            sql += ' WHERE tenant_id = ?';
            params.push(tenantId);
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
//# sourceMappingURL=scratchpad-mirror-queries.js.map