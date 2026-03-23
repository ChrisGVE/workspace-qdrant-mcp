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
  tags: string; // JSON array as string
  tenantId: string;
  createdAt: string;
  updatedAt: string;
}

/**
 * Upsert a scratchpad entry into the scratchpad_mirror table via daemon gRPC.
 * Fire-and-forget: errors are swallowed since the mirror is advisory.
 */
export function upsertScratchpadMirror(
  daemonClient: DaemonClient | null,
  entry: ScratchpadMirrorEntry
): void {
  if (!daemonClient) return;

  const request: {
    scratchpad_id: string;
    content: string;
    tenant_id: string;
    created_at: string;
    updated_at: string;
    title?: string;
    tags?: string;
  } = {
    scratchpad_id: entry.scratchpadId,
    content: entry.content,
    tenant_id: entry.tenantId,
    created_at: entry.createdAt,
    updated_at: entry.updatedAt,
  };
  if (entry.title !== null) request.title = entry.title;
  if (entry.tags !== '[]') request.tags = entry.tags;

  daemonClient.upsertScratchpadMirror(request).catch((err: unknown) => {
    console.warn('upsertScratchpadMirror failed:', err instanceof Error ? err.message : err);
  });
}

/**
 * Delete a scratchpad entry from the scratchpad_mirror table via daemon gRPC.
 * Fire-and-forget: errors are swallowed.
 */
export function deleteScratchpadMirror(
  daemonClient: DaemonClient | null,
  scratchpadId: string
): void {
  if (!daemonClient) return;

  daemonClient.deleteScratchpadMirror({ scratchpad_id: scratchpadId }).catch((err: unknown) => {
    console.warn('deleteScratchpadMirror failed:', err instanceof Error ? err.message : err);
  });
}

/**
 * List scratchpad entries from the scratchpad_mirror table.
 * Read-only — queries SQLite directly.
 */
export function listScratchpadMirror(
  db: DatabaseType | null,
  tenantId?: string,
  limit = 100
): ScratchpadMirrorEntry[] {
  if (!db) return [];

  try {
    let sql = `SELECT scratchpad_id AS scratchpadId, title, content, tags,
                      tenant_id AS tenantId,
                      created_at AS createdAt, updated_at AS updatedAt
               FROM scratchpad_mirror`;
    const params: (string | number)[] = [];

    if (tenantId) {
      sql += ' WHERE tenant_id = ?';
      params.push(tenantId);
    }

    sql += ' ORDER BY updatedAt DESC LIMIT ?';
    params.push(limit);

    return db.prepare(sql).all(...params) as ScratchpadMirrorEntry[];
  } catch (err: unknown) {
    const msg = err instanceof Error ? err.message : String(err);
    if (msg.includes('no such table')) {
      return [];
    }
    throw err;
  }
}
