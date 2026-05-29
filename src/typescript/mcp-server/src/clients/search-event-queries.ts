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
 * Token-economy metrics from the post-execution shaping pass.
 * Spec: docs/specs/20-token-economy-instrumentation.md
 */
export interface SearchEventEconomyInput {
  bytesIn: number;
  bytesOut: number;
  hitsTruncated: number;
  shapeMode: 'truncate' | 'summary' | 'none';
  toolVersion?: string | undefined;
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

/**
 * Record token-economy metrics for a previously logged search event.
 * Fire-and-forget: errors are swallowed so instrumentation never blocks
 * the search response.
 */
export function updateSearchEventEconomy(
  daemonClient: DaemonClient | null,
  eventId: string,
  update: SearchEventEconomyInput
): void {
  if (!daemonClient) return;

  const request: {
    event_id: string;
    bytes_in: number;
    bytes_out: number;
    hits_truncated: number;
    shape_mode: 'truncate' | 'summary' | 'none';
    tool_version?: string;
  } = {
    event_id: eventId,
    bytes_in: update.bytesIn,
    bytes_out: update.bytesOut,
    hits_truncated: update.hitsTruncated,
    shape_mode: update.shapeMode,
  };
  if (update.toolVersion !== undefined) request.tool_version = update.toolVersion;

  daemonClient.updateSearchEventEconomy(request).catch((err: unknown) => {
    console.warn(
      'updateSearchEventEconomy instrumentation failed:',
      err instanceof Error ? err.message : err
    );
  });
}

/**
 * Combined finish-instrumentation for tools that don't have a shaping
 * pass of their own (grep / retrieve / list). Records both the
 * post-execution `result_count` / `latency_ms` and the token-economy
 * sidecar in a single call site so the tool's own code stays minimal.
 *
 * Spec: `docs/specs/20-token-economy-instrumentation.md` §3.2–§3.4.
 *
 * Fire-and-forget for both sides — never raises to the caller.
 */
export interface ToolEventFinish {
  resultCount: number;
  latencyMs: number;
  bytesIn: number;
  bytesOut: number;
  toolVersion?: string | undefined;
  outcome?: string | undefined;
  /**
   * Optional shaping mode. Defaults to `'none'` — these tools don't
   * currently shape responses. If a tool later grows a shaping pass,
   * pass `'truncate'` or `'summary'` accordingly.
   */
  shapeMode?: 'truncate' | 'summary' | 'none' | undefined;
  /** Optional truncated-hit count. Defaults to 0 when shapeMode is 'none'. */
  hitsTruncated?: number | undefined;
}

export function finishToolEvent(
  daemonClient: DaemonClient | null,
  eventId: string,
  finish: ToolEventFinish
): void {
  if (!daemonClient) return;
  const updateArgs: SearchEventUpdate = {
    resultCount: finish.resultCount,
    latencyMs: finish.latencyMs,
  };
  if (finish.outcome !== undefined) updateArgs.outcome = finish.outcome;
  updateSearchEvent(daemonClient, eventId, updateArgs);

  const economyArgs: SearchEventEconomyInput = {
    bytesIn: finish.bytesIn,
    bytesOut: finish.bytesOut,
    hitsTruncated: finish.hitsTruncated ?? 0,
    shapeMode: finish.shapeMode ?? 'none',
  };
  if (finish.toolVersion !== undefined) economyArgs.toolVersion = finish.toolVersion;
  updateSearchEventEconomy(daemonClient, eventId, economyArgs);
}
