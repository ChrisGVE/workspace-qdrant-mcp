/**
 * Graph-augmented RAG expansion for search results.
 *
 * 4-step pipeline:
 * 1. Vector search (existing RRF) — results arrive pre-fused
 * 2. Graph expansion — 1-2 hops, max 5 per result, max 50 total
 * 3. Score fusion — alpha * vector_score + (1-alpha) * graph_proximity
 * 4. Convergence bonus — +0.1 for results found in both vector and graph
 *
 * Gated behind the `includeGraphContext` parameter.
 */
import type { DaemonClient } from '../clients/daemon-client.js';
import type { SearchResult } from './search-types.js';
/**
 * Expand search results using graph traversal and fuse scores.
 *
 * Modifies results in-place: adds graph-expanded results and adjusts scores
 * using the alpha-blended fusion formula with convergence bonus.
 */
export declare function expandAndFuseWithGraph(daemonClient: DaemonClient, results: SearchResult[], collection: string): Promise<void>;
//# sourceMappingURL=search-graph-expansion.d.ts.map