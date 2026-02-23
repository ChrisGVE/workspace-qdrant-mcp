/**
 * Graph context enrichment for search results.
 *
 * When `includeGraphContext` is enabled, fetches 1-hop code relationships
 * (callers/callees) from the daemon's GraphService for each search result
 * that contains a code symbol.
 */

import { createHash } from 'node:crypto';
import type { DaemonClient } from '../clients/daemon-client.js';
import type { TraversalNodeProto } from '../clients/grpc-types.js';
import type { SearchResult, GraphContextNode } from './search-types.js';
import { logDebug } from '../utils/logger.js';

/** Qdrant payload keys for semantic chunk metadata */
const CHUNK_SYMBOL_NAME = 'chunk_symbol_name';
const CHUNK_CHUNK_TYPE = 'chunk_chunk_type';

/** Chunk types that have graph entries */
const CODE_CHUNK_TYPES = new Set([
  'function', 'async_function', 'method', 'class', 'struct',
  'trait', 'interface', 'enum', 'impl', 'module', 'constant',
  'type_alias', 'macro',
]);

/** Timeout for a single graph query (ms) */
const GRAPH_QUERY_TIMEOUT_MS = 200;

/**
 * Compute node_id matching Rust's `compute_node_id`:
 * SHA256(tenant_id|file_path|symbol_name|symbol_type)[..16] as hex
 */
function computeNodeId(
  tenantId: string,
  filePath: string,
  symbolName: string,
  symbolType: string,
): string {
  const input = `${tenantId}|${filePath}|${symbolName}|${symbolType}`;
  const hash = createHash('sha256').update(input).digest('hex');
  return hash.slice(0, 32); // 16 bytes = 32 hex chars
}

/**
 * Convert a TraversalNodeProto to a GraphContextNode.
 */
function toGraphContextNode(node: TraversalNodeProto): GraphContextNode {
  const result: GraphContextNode = {
    symbol: node.symbol_name,
    file_path: node.file_path,
  };
  return result;
}

/**
 * Enrich search results with 1-hop graph context for code symbols.
 *
 * For each result that has chunk_symbol_name and chunk_chunk_type metadata,
 * queries the daemon GraphService for callers and callees. Queries run in
 * parallel with a per-query timeout. Failures are silently ignored — the
 * result simply won't have a graph_context field.
 */
export async function expandGraphContext(
  daemonClient: DaemonClient,
  results: SearchResult[],
): Promise<void> {
  const enrichments: Array<{ result: SearchResult; tenantId: string; nodeId: string; filePath: string; symbolName: string }> = [];

  for (const result of results) {
    const symbolName = result.metadata[CHUNK_SYMBOL_NAME] as string | undefined;
    const chunkType = result.metadata[CHUNK_CHUNK_TYPE] as string | undefined;
    const tenantId = result.metadata['tenant_id'] as string | undefined;
    const filePath = (result.metadata['relative_path'] as string) ?? (result.metadata['file_path'] as string);

    if (!symbolName || !chunkType || !tenantId || !filePath) continue;
    if (!CODE_CHUNK_TYPES.has(chunkType)) continue;

    const nodeId = computeNodeId(tenantId, filePath, symbolName, chunkType);
    enrichments.push({ result, tenantId, nodeId, filePath, symbolName });
  }

  if (enrichments.length === 0) return;

  logDebug('Fetching graph context', { count: enrichments.length });

  // Fetch graph context for all eligible results in parallel
  await Promise.all(
    enrichments.map(async ({ result, tenantId, nodeId, filePath, symbolName }) => {
      try {
        const response = await Promise.race([
          daemonClient.queryRelated({
            tenant_id: tenantId,
            node_id: nodeId,
            max_hops: 1,
          }),
          new Promise<never>((_, reject) =>
            setTimeout(() => reject(new Error('Graph query timeout')), GRAPH_QUERY_TIMEOUT_MS),
          ),
        ]);

        if (!response.nodes || response.nodes.length === 0) return;

        // Separate callers (nodes calling us via CALLS where we are target)
        // and callees (nodes we call). In a 1-hop query from our node,
        // CALLS edges point to callees; reverse CALLS = callers.
        // The QueryRelated response includes edge_type for each node.
        const callers: GraphContextNode[] = [];
        const callees: GraphContextNode[] = [];

        for (const node of response.nodes) {
          // Skip self
          if (node.node_id === nodeId) continue;

          const ctxNode = toGraphContextNode(node);

          // Edge type tells us the relationship direction:
          // CALLS from source to target means source calls target
          // In our 1-hop traversal from a node, all CALLS edges mean
          // the node calls these targets (callees).
          // Reverse CALLS edges (callers) show up via reverse traversal.
          if (node.edge_type === 'CALLS') {
            callees.push(ctxNode);
          } else if (node.edge_type === 'CALLS_REVERSE' || node.edge_type === 'CONTAINS') {
            callers.push(ctxNode);
          } else {
            // For other edge types (IMPORTS, USES_TYPE, etc.), put in callees
            callees.push(ctxNode);
          }
        }

        result.graph_context = {
          symbol: symbolName,
          file_path: filePath,
          callers,
          callees,
        };
      } catch {
        // Silently ignore graph query failures
      }
    }),
  );
}
