/**
 * Daemon availability check with TTL-based caching.
 *
 * Verifies daemon connectivity before write operations, with TTL-based caching.
 * Positive results cached for 5s, negative results cached for 1s to allow fast recovery.
 */
import type { DaemonClient } from '../clients/daemon-client.js';
/**
 * Ensure the daemon is available for write operations.
 *
 * Returns the validated DaemonClient or throws if unavailable.
 * Caches positive results for 5s, negative for 1s to reduce health check overhead.
 */
export declare function ensureDaemonAvailable(client: DaemonClient | null): Promise<DaemonClient>;
/** Reset the cached check (useful for testing). */
export declare function resetDaemonCheck(): void;
//# sourceMappingURL=daemon-check.d.ts.map