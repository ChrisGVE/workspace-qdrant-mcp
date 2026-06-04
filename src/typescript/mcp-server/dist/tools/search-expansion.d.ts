/**
 * Tag-based query expansion for BM25 sparse search.
 */
import type { DaemonClient } from '../clients/daemon-client.js';
import type { SqliteStateManager } from '../clients/sqlite-state-manager.js';
/**
 * Expand sparse vector with keywords from matching tag baskets.
 *
 * 1. Query SQLite for tags matching the query text
 * 2. Retrieve keyword baskets for matching tags
 * 3. Generate a sparse vector for the expanded keywords
 * 4. Merge into the original sparse vector at reduced weight
 */
export declare function expandSparseWithTags(daemonClient: DaemonClient, stateManager: SqliteStateManager, query: string, originalSparse: Record<number, number>, collections: string[], expansionWeight: number, maxExpandedKeywords: number, tenantId?: string): Promise<Record<number, number>>;
//# sourceMappingURL=search-expansion.d.ts.map