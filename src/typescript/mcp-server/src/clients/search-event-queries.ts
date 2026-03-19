/**
 * Search event instrumentation via daemon gRPC.
 *
 * Logs and updates search events through the TrackingWriteService.
 * Errors are swallowed so instrumentation never blocks search execution.
 */

import type { DaemonClient } from './daemon-client.js';

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
 * Log a search event via daemon gRPC.
 *
 * Called at the start of a search to create the initial record.
 * Fire-and-forget: errors are swallowed so instrumentation never breaks search.
 */
export function logSearchEvent(daemonClient: DaemonClient | null, event: SearchEventInput): void {
  if (!daemonClient) return;

  // Fire-and-forget — catch errors to avoid breaking search
  const request: {
    id: string;
    actor: string;
    tool: string;
    op: string;
    session_id?: string;
    project_id?: string;
    query_text?: string;
    filters?: string;
    top_k?: number;
    result_count?: number;
    latency_ms?: number;
    top_result_refs?: string;
    outcome?: string;
    parent_event_id?: string;
  } = {
    id: event.id,
    actor: event.actor,
    tool: event.tool,
    op: event.op,
  };
  if (event.sessionId !== undefined) request.session_id = event.sessionId;
  if (event.projectId !== undefined) request.project_id = event.projectId;
  if (event.queryText !== undefined) request.query_text = event.queryText;
  if (event.filters !== undefined) request.filters = event.filters;
  if (event.topK !== undefined) request.top_k = event.topK;
  if (event.resultCount !== undefined) request.result_count = event.resultCount;
  if (event.latencyMs !== undefined) request.latency_ms = event.latencyMs;
  if (event.topResultRefs !== undefined) request.top_result_refs = event.topResultRefs;
  if (event.outcome !== undefined) request.outcome = event.outcome;
  if (event.parentEventId !== undefined) request.parent_event_id = event.parentEventId;

  daemonClient.logSearchEvent(request).catch((err: unknown) => {
    // Instrumentation must never break search, but log for diagnostics
    console.warn(
      'logSearchEvent instrumentation failed:',
      err instanceof Error ? err.message : err
    );
  });
}

/**
 * Update a search event with post-search results via daemon gRPC.
 *
 * Updates result_count, latency_ms, top_result_refs, and outcome
 * for a previously created search event.
 * Fire-and-forget: errors are swallowed.
 */
export function updateSearchEvent(
  daemonClient: DaemonClient | null,
  eventId: string,
  update: SearchEventUpdate
): void {
  if (!daemonClient) return;

  const request: {
    event_id: string;
    result_count: number;
    latency_ms: number;
    top_result_refs?: string;
    outcome?: string;
  } = {
    event_id: eventId,
    result_count: update.resultCount,
    latency_ms: update.latencyMs,
  };
  if (update.topResultRefs !== undefined) request.top_result_refs = update.topResultRefs;
  if (update.outcome !== undefined) request.outcome = update.outcome;

  daemonClient.updateSearchEvent(request).catch((err: unknown) => {
    // Instrumentation must never break search, but log for diagnostics
    console.warn(
      'updateSearchEvent instrumentation failed:',
      err instanceof Error ? err.message : err
    );
  });
}
