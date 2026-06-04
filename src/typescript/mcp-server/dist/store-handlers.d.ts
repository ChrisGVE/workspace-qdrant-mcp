/**
 * Store handler helpers for URL and scratchpad store types
 */
import type { SqliteStateManager } from './clients/sqlite-state-manager.js';
import type { SessionState } from './server-types.js';
type StoreResult = {
    success: boolean;
    message: string;
    queue_id?: string;
    collection: string;
};
/**
 * Pre-enqueue URL validation. Rejects malformed input, non-http(s) schemes,
 * and obviously bad hostnames so the daemon does not waste a queue cycle on
 * URLs it would reject at fetch time. Full SSRF policy (private-network
 * denylist, DNS rebinding defense, redirect re-validation) is enforced
 * daemon-side; this is a fast-fail surface for user-facing error messages.
 */
export declare function validateUrlInput(raw: unknown): {
    ok: true;
} | {
    ok: false;
    message: string;
};
/**
 * Store a URL for daemon-side fetch and ingestion.
 *
 * Queues the URL as item_type 'url' in the unified queue.
 * The daemon will fetch the page, extract text, generate embeddings,
 * and store in Qdrant.
 */
export declare function storeUrl(args: Record<string, unknown> | undefined, stateManager: SqliteStateManager, sessionState: Pick<SessionState, 'projectId'>): Promise<StoreResult>;
export declare function storeScratchpad(args: Record<string, unknown> | undefined, stateManager: SqliteStateManager, sessionState: Pick<SessionState, 'projectId'>): Promise<StoreResult>;
export {};
//# sourceMappingURL=store-handlers.d.ts.map