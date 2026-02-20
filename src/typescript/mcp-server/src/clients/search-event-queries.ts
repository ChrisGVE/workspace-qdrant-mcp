/**
 * Search event instrumentation queries for SqliteStateManager.
 *
 * Logs and updates search events in the search_events table (schema v12+).
 * Errors are swallowed so instrumentation never blocks search execution.
 */

import type { Database as DatabaseType } from 'better-sqlite3';
import { utcNow } from '../utils/timestamps.js';

export interface SearchEventInput {
  id: string;
  sessionId?: string | undefined;
  projectId?: string | undefined;
  actor: string;
  tool: string;
  op: string;
  queryText?: string | undefined;
  filters?: string | undefined;
  topK?: number | undefined;
  resultCount?: number | undefined;
  latencyMs?: number | undefined;
  topResultRefs?: string | undefined;
  outcome?: string | undefined;
  parentEventId?: string | undefined;
}

export interface SearchEventUpdate {
  resultCount: number;
  latencyMs: number;
  topResultRefs?: string | undefined;
  outcome?: string | undefined;
}

/**
 * Log a search event to the search_events table.
 *
 * Called at the start of a search to create the initial record.
 */
export function logSearchEvent(
  db: DatabaseType | null,
  event: SearchEventInput,
): void {
  if (!db) return;

  try {
    const now = utcNow();
    const stmt = db.prepare(`
      INSERT INTO search_events (
        id, ts, session_id, project_id, actor, tool, op,
        query_text, filters, top_k, result_count, latency_ms,
        top_result_refs, outcome, parent_event_id, created_at
      ) VALUES (
        ?, ?, ?, ?, ?, ?, ?,
        ?, ?, ?, ?, ?,
        ?, ?, ?, ?
      )
    `);
    stmt.run(
      event.id,
      now,
      event.sessionId ?? null,
      event.projectId ?? null,
      event.actor,
      event.tool,
      event.op,
      event.queryText ?? null,
      event.filters ?? null,
      event.topK ?? null,
      event.resultCount ?? null,
      event.latencyMs ?? null,
      event.topResultRefs ?? null,
      event.outcome ?? null,
      event.parentEventId ?? null,
      now,
    );
  } catch {
    // Instrumentation must never break search. Table may not exist if
    // daemon hasn't migrated to v12+ yet.
  }
}

/**
 * Update a search event with post-search results.
 *
 * Updates result_count, latency_ms, top_result_refs, and outcome
 * for a previously created search event.
 */
export function updateSearchEvent(
  db: DatabaseType | null,
  eventId: string,
  update: SearchEventUpdate,
): void {
  if (!db) return;

  try {
    const stmt = db.prepare(`
      UPDATE search_events
      SET result_count = ?, latency_ms = ?, top_result_refs = ?, outcome = ?
      WHERE id = ?
    `);
    stmt.run(
      update.resultCount,
      update.latencyMs,
      update.topResultRefs ?? null,
      update.outcome ?? null,
      eventId,
    );
  } catch {
    // Instrumentation must never break search
  }
}
