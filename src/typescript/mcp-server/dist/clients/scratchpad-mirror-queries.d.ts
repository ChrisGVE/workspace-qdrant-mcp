/**
 * Scratchpad mirror read/write operations for SqliteStateManager.
 *
 * Write operations (upsert, delete) go through daemon gRPC.
 * Read operations (list) query SQLite directly for fallback/rebuild support.
 */
import type { Database as DatabaseType } from 'better-sqlite3';
import type { DaemonClient } from './daemon-client.js';
export interface ScratchpadMirrorEntry {
    scratchpadId: string;
    title: string | null;
    content: string;
    tags: string;
    tenantId: string;
    createdAt: string;
    updatedAt: string;
}
/**
 * Upsert a scratchpad entry into the scratchpad_mirror table via daemon gRPC.
 * Fire-and-forget: errors are swallowed since the mirror is advisory.
 */
export declare function upsertScratchpadMirror(daemonClient: DaemonClient | null, entry: ScratchpadMirrorEntry): void;
/**
 * Delete a scratchpad entry from the scratchpad_mirror table via daemon gRPC.
 * Fire-and-forget: errors are swallowed.
 */
export declare function deleteScratchpadMirror(daemonClient: DaemonClient | null, scratchpadId: string): void;
/**
 * List scratchpad entries from the scratchpad_mirror table.
 * Read-only — queries SQLite directly.
 */
export declare function listScratchpadMirror(db: DatabaseType | null, tenantId?: string, limit?: number): ScratchpadMirrorEntry[];
//# sourceMappingURL=scratchpad-mirror-queries.d.ts.map