/**
 * Graph context enrichment for search results.
 *
 * When `includeGraphContext` is enabled, fetches 1-hop code relationships
 * (callers/callees) from the daemon's GraphService for each search result
 * that contains a code symbol.
 */
import { createHash } from 'node:crypto';
import { logDebug } from '../utils/logger.js';
/** Qdrant payload keys for semantic chunk metadata */
const CHUNK_SYMBOL_NAME = 'chunk_symbol_name';
const CHUNK_CHUNK_TYPE = 'chunk_chunk_type';
/** Chunk types that have graph entries */
const CODE_CHUNK_TYPES = new Set([
    'function',
    'async_function',
    'method',
    'class',
    'struct',
    'trait',
    'interface',
    'enum',
    'impl',
    'module',
    'constant',
    'type_alias',
    'macro',
]);
/** Timeout for a single graph query (ms) */
const GRAPH_QUERY_TIMEOUT_MS = 200;
/**
 * Compute node_id matching Rust's `compute_node_id`:
 * SHA256(tenant_id|file_path|symbol_name|symbol_type)[..16] as hex
 */
function computeNodeId(tenantId, filePath, symbolName, symbolType) {
    const input = `${tenantId}|${filePath}|${symbolName}|${symbolType}`;
    const hash = createHash('sha256').update(input).digest('hex');
    return hash.slice(0, 32); // 16 bytes = 32 hex chars
}
/**
 * Convert a TraversalNodeProto to a GraphContextNode.
 */
function toGraphContextNode(node) {
    const result = {
        symbol: node.symbol_name,
        file_path: node.file_path,
    };
    return result;
}
/** Collect results eligible for graph context enrichment. */
function collectEnrichmentTargets(results) {
    const targets = [];
    for (const result of results) {
        const symbolName = result.metadata[CHUNK_SYMBOL_NAME];
        const chunkType = result.metadata[CHUNK_CHUNK_TYPE];
        const tenantId = result.metadata['tenant_id'];
        const filePath = result.metadata['relative_path'] ?? result.metadata['file_path'];
        if (!symbolName || !chunkType || !tenantId || !filePath)
            continue;
        if (!CODE_CHUNK_TYPES.has(chunkType))
            continue;
        targets.push({
            result,
            tenantId,
            nodeId: computeNodeId(tenantId, filePath, symbolName, chunkType),
            filePath,
            symbolName,
        });
    }
    return targets;
}
/** Fetch and attach graph context for one search result. */
async function enrichOneResult(daemonClient, target) {
    try {
        const response = await Promise.race([
            daemonClient.queryRelated({
                tenant_id: target.tenantId,
                node_id: target.nodeId,
                max_hops: 1,
            }),
            new Promise((_, reject) => setTimeout(() => reject(new Error('Graph query timeout')), GRAPH_QUERY_TIMEOUT_MS)),
        ]);
        if (!response.nodes || response.nodes.length === 0)
            return;
        const callers = [];
        const callees = [];
        for (const node of response.nodes) {
            if (node.node_id === target.nodeId)
                continue;
            const ctxNode = toGraphContextNode(node);
            if (node.edge_type === 'CALLS_REVERSE' || node.edge_type === 'CONTAINS') {
                callers.push(ctxNode);
            }
            else {
                callees.push(ctxNode);
            }
        }
        target.result.graph_context = {
            symbol: target.symbolName,
            file_path: target.filePath,
            callers,
            callees,
        };
    }
    catch {
        // Silently ignore graph query failures
    }
}
/**
 * Enrich search results with 1-hop graph context for code symbols.
 *
 * For each result that has chunk_symbol_name and chunk_chunk_type metadata,
 * queries the daemon GraphService for callers and callees. Queries run in
 * parallel with a per-query timeout. Failures are silently ignored — the
 * result simply won't have a graph_context field.
 */
export async function expandGraphContext(daemonClient, results) {
    const targets = collectEnrichmentTargets(results);
    if (targets.length === 0)
        return;
    logDebug('Fetching graph context', { count: targets.length });
    await Promise.all(targets.map((t) => enrichOneResult(daemonClient, t)));
}
//# sourceMappingURL=search-graph-context.js.map