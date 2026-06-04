/**
 * Graph context enrichment for search results.
 *
 * When `includeGraphContext` is enabled, fetches 1-hop code relationships
 * (callers/callees) from the daemon's GraphService for each search result
 * that contains a code symbol.
 */
import type { DaemonClient } from '../clients/daemon-client.js';
import type { SearchResult } from './search-types.js';
/**
 * Enrich search results with 1-hop graph context for code symbols.
 *
 * For each result that has chunk_symbol_name and chunk_chunk_type metadata,
 * queries the daemon GraphService for callers and callees. Queries run in
 * parallel with a per-query timeout. Failures are silently ignored — the
 * result simply won't have a graph_context field.
 */
export declare function expandGraphContext(daemonClient: DaemonClient, results: SearchResult[]): Promise<void>;
//# sourceMappingURL=search-graph-context.d.ts.map