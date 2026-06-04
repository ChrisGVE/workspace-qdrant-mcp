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
export declare function logSearchEvent(daemonClient: DaemonClient | null, event: SearchEventInput): void;
/**
 * Update a search event with post-search results via daemon gRPC.
 *
 * Updates result_count, latency_ms, top_result_refs, and outcome
 * for a previously created search event.
 * Fire-and-forget: errors are swallowed.
 */
export declare function updateSearchEvent(daemonClient: DaemonClient | null, eventId: string, update: SearchEventUpdate): void;
//# sourceMappingURL=search-event-queries.d.ts.map