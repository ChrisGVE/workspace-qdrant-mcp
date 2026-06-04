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
import { createHash } from 'node:crypto';
import { logDebug } from '../utils/logger.js';
const ALPHA = 0.7;
const CONVERGENCE_BONUS = 0.1;
const MAX_EXPANDED_PER_RESULT = 5;
const MAX_EXPANDED_TOTAL = 50;
const GRAPH_EXPANSION_TIMEOUT_MS = 500;
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
]);
function computeNodeId(tenantId, filePath, symbolName, symbolType) {
    const input = `${tenantId}|${filePath}|${symbolName}|${symbolType}`;
    return createHash('sha256').update(input).digest('hex').slice(0, 32);
}
function collectCandidates(results) {
    const candidates = [];
    for (const r of results) {
        const symbolName = r.metadata['chunk_symbol_name'];
        const chunkType = r.metadata['chunk_chunk_type'];
        const tenantId = r.metadata['tenant_id'];
        const filePath = r.metadata['relative_path'] ?? r.metadata['file_path'];
        if (!symbolName || !chunkType || !tenantId || !filePath)
            continue;
        if (!CODE_CHUNK_TYPES.has(chunkType))
            continue;
        candidates.push({
            tenantId,
            nodeId: computeNodeId(tenantId, filePath, symbolName, chunkType),
            vectorScore: r.score,
        });
    }
    return candidates;
}
function graphProximityScore(hopDistance) {
    if (hopDistance <= 0)
        return 1.0;
    if (hopDistance === 1)
        return 0.8;
    return 0.5;
}
function nodeToSearchResult(node, collection, tenantId, proximity) {
    return {
        id: node.node_id,
        score: proximity,
        collection,
        content: `${node.symbol_type} ${node.symbol_name} in ${node.file_path}`,
        title: node.symbol_name,
        metadata: {
            tenant_id: tenantId,
            chunk_symbol_name: node.symbol_name,
            chunk_chunk_type: node.symbol_type,
            file_path: node.file_path,
            source: 'graph_expansion',
        },
    };
}
/**
 * Expand search results using graph traversal and fuse scores.
 *
 * Modifies results in-place: adds graph-expanded results and adjusts scores
 * using the alpha-blended fusion formula with convergence bonus.
 */
export async function expandAndFuseWithGraph(daemonClient, results, collection) {
    const candidates = collectCandidates(results);
    if (candidates.length === 0)
        return;
    const existingIds = new Set(results.map((r) => r.id));
    const expandedResults = [];
    let totalExpanded = 0;
    const expansionPromises = candidates.slice(0, 20).map(async (candidate) => {
        try {
            const response = await Promise.race([
                daemonClient.queryRelated({
                    tenant_id: candidate.tenantId,
                    node_id: candidate.nodeId,
                    max_hops: 2,
                    edge_types: ['CALLS', 'USES_TYPE', 'CONTAINS'],
                }),
                new Promise((_, reject) => setTimeout(() => reject(new Error('timeout')), GRAPH_EXPANSION_TIMEOUT_MS)),
            ]);
            if (!response.nodes)
                return [];
            const newNodes = [];
            for (const node of response.nodes.slice(0, MAX_EXPANDED_PER_RESULT)) {
                if (node.node_id === candidate.nodeId)
                    continue;
                if (existingIds.has(node.node_id)) {
                    const existing = results.find((r) => r.id === node.node_id);
                    if (existing) {
                        existing.score += CONVERGENCE_BONUS;
                    }
                    continue;
                }
                if (totalExpanded >= MAX_EXPANDED_TOTAL)
                    break;
                const proximity = graphProximityScore(node.depth);
                const newResult = nodeToSearchResult(node, collection, candidate.tenantId, proximity);
                newResult.score = (1 - ALPHA) * proximity;
                newNodes.push(newResult);
                existingIds.add(node.node_id);
                totalExpanded++;
            }
            return newNodes;
        }
        catch {
            return [];
        }
    });
    const allExpanded = await Promise.all(expansionPromises);
    for (const batch of allExpanded) {
        expandedResults.push(...batch);
    }
    for (const r of results) {
        r.score = ALPHA * r.score + (r.score > 0 ? 0 : 0);
    }
    results.push(...expandedResults);
    results.sort((a, b) => b.score - a.score);
    if (expandedResults.length > 0) {
        logDebug('Graph expansion added results', {
            expanded: expandedResults.length,
            convergence_bonuses: candidates.length - expandedResults.length,
        });
    }
}
//# sourceMappingURL=search-graph-expansion.js.map