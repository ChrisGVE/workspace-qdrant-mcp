/**
 * Search tool helper functions — phase-level operations extracted from SearchTool.
 *
 * Each function corresponds to one phase of the hybrid search pipeline:
 *   resolveProjectContext  → project/instance disambiguation
 *   logSearchEventPre      → pre-execution telemetry
 *   generateEmbeddings     → dense + sparse vector generation
 *   searchAllCollections   → per-collection fan-out
 *   finalizeResults        → RRF fusion, context expansion, telemetry update
 */
import { QdrantClient } from '@qdrant/js-client-rest';
import { PROJECTS_COLLECTION } from './search-types.js';
import { buildFilter } from './search-filters.js';
import { searchCollection, applyRRFFusion, expandParentContext, fallbackSearch, } from './search-qdrant.js';
import { expandGraphContext } from './search-graph-context.js';
import { expandAndFuseWithGraph } from './search-graph-expansion.js';
import { diversifyResults, DEFAULT_DIVERSITY_CONFIG } from './search-diversity.js';
/** Maximum active base_points we still attach as a Qdrant filter. Above
 * this the filter clause would blow past server-side limits; we instead
 * surface a degradation flag so the caller reports it explicitly. */
const BASE_POINTS_FILTER_CAP = 500;
/** Resolve current project ID and base_points for instance-aware filtering.
 *
 * Project detection runs for all scopes because the current project ID is
 * needed both for tenant filtering (project/group) and relevance decay (group/all). */
export async function resolveProjectContext(projectId, scope, projectDetector, stateManager) {
    let currentProjectId = projectId;
    if (!currentProjectId) {
        const projectInfo = await projectDetector.getProjectInfo(process.cwd(), false);
        currentProjectId = projectInfo?.projectId;
    }
    let basePoints;
    let basePointsDegraded = false;
    let basePointsActiveCount;
    if (currentProjectId && scope === 'project') {
        const watchFolderId = stateManager.getWatchFolderIdByTenantId(currentProjectId);
        if (watchFolderId) {
            const points = stateManager.getActiveBasePoints(watchFolderId, false);
            if (points.length > 0 && points.length <= BASE_POINTS_FILTER_CAP) {
                basePoints = points;
            }
            else if (points.length > BASE_POINTS_FILTER_CAP) {
                // F-012: Fall back to "primary base-point only" filter.
                // Instead of dropping instance isolation entirely, narrow to
                // the single base point matching the caller's working directory.
                // Tenant filter still applies for project-level scoping.
                const cwd = process.cwd();
                const primaryPoint = points.find((bp) => cwd.startsWith(bp));
                if (primaryPoint) {
                    basePoints = [primaryPoint];
                }
                else {
                    // No base point matches cwd — degrade gracefully.
                    basePointsDegraded = true;
                    basePointsActiveCount = points.length;
                }
            }
        }
    }
    const result = { currentProjectId, basePoints };
    if (basePointsDegraded) {
        result.basePointsDegraded = true;
        if (basePointsActiveCount !== undefined) {
            result.basePointsActiveCount = basePointsActiveCount;
        }
    }
    return result;
}
/** Log the pre-execution search event. */
export function logSearchEventPre(stateManager, eventId, projectId, query, limit, opts) {
    const filters = {};
    if (opts.collection)
        filters.collection = opts.collection;
    if (opts.scope !== 'project')
        filters.scope = opts.scope;
    if (opts.branch)
        filters.branch = opts.branch;
    if (opts.fileType)
        filters.file_type = opts.fileType;
    if (opts.libraryName)
        filters.library_name = opts.libraryName;
    if (opts.tag)
        filters.tag = opts.tag;
    stateManager.logSearchEvent({
        id: eventId,
        projectId,
        actor: 'claude',
        tool: 'mcp_qdrant',
        op: 'search',
        queryText: query,
        filters: Object.keys(filters).length > 0 ? JSON.stringify(filters) : undefined,
        topK: limit,
    });
}
/** Generate dense and sparse embeddings. Returns `{ fallback }` on daemon error. */
export async function generateEmbeddings(daemonClient, qdrantClient, query, mode, options, collectionsToSearch, fallbackContext) {
    let denseEmbedding;
    let sparseVector;
    try {
        if (mode === 'hybrid' || mode === 'semantic') {
            const r = await daemonClient.embedText({ text: query });
            if (r.success)
                denseEmbedding = r.embedding;
        }
        if (mode === 'hybrid' || mode === 'keyword') {
            const r = await daemonClient.generateSparseVector({ text: query });
            if (r.success)
                sparseVector = r.indices_values;
        }
    }
    catch {
        return {
            fallback: await fallbackSearch(qdrantClient, options, collectionsToSearch, fallbackContext),
        };
    }
    return { denseEmbedding, sparseVector };
}
/** Build SearchCollectionParams for one collection in a fan-out search. */
function buildCollectionSearchParams(coll, params) {
    const filterParams = {
        collection: coll,
        scope: params.scope,
        projectId: params.currentProjectId,
        groupTenantIds: params.groupTenantIds,
        branch: params.branch,
        fileType: params.fileType,
        libraryName: params.libraryName,
        libraryPath: params.libraryPath,
        tag: params.tag,
        tags: params.tags,
        pathGlob: params.options.pathGlob,
        component: params.options.component,
        basePoints: coll === PROJECTS_COLLECTION ? params.basePoints : undefined,
    };
    return {
        collection: coll,
        mode: params.mode,
        denseEmbedding: params.denseEmbedding,
        sparseVector: params.sparseVector,
        filter: buildFilter(filterParams),
        limit: params.limit * 2,
        scoreThreshold: params.scoreThreshold,
    };
}
/** Search all target collections and collect results, tolerating partial failures. */
export async function searchAllCollections(qdrantClient, params) {
    const allResults = [];
    let status = 'ok';
    let statusReason;
    for (const coll of params.collectionsToSearch) {
        try {
            allResults.push(...(await searchCollection(qdrantClient, buildCollectionSearchParams(coll, params))));
        }
        catch (error) {
            status = 'uncertain';
            statusReason = `Some collections unavailable: ${error instanceof Error ? error.message : 'unknown'}`;
        }
    }
    return { allResults, status, statusReason };
}
/** Record search telemetry and build the final SearchResponse. */
function recordAndBuildResponse(stateManager, finalResults, params, searchStartMs, diversityScore) {
    const latencyMs = Date.now() - searchStartMs;
    const topRefs = finalResults
        .slice(0, 5)
        .map((r) => ({ id: r.id, score: Math.round(r.score * 1000) / 1000, collection: r.collection }));
    stateManager.updateSearchEvent(params.eventId, {
        resultCount: finalResults.length,
        latencyMs,
        topResultRefs: JSON.stringify(topRefs),
    });
    const response = {
        results: finalResults,
        total: finalResults.length,
        query: params.query,
        mode: params.mode,
        scope: params.scope,
        collections_searched: params.collectionsToSearch,
        status: params.status,
    };
    if (params.statusReason)
        response.status_reason = params.statusReason;
    // Expose the branch filter that was applied so callers can see which branch
    // scoped the search (absent when cross-branch via "*" or no filter).
    if (params.options.branch)
        response.branch = params.options.branch;
    if (diversityScore !== undefined)
        response.diversity_score = diversityScore;
    return response;
}
/** Fuse, sort, apply diversity re-ranking, expand context, update event, and assemble the final response. */
export async function finalizeResults(qdrantClient, daemonClient, stateManager, params) {
    const fusedResults = applyRRFFusion(params.allResults, params.mode);
    fusedResults.sort((a, b) => b.score - a.score);
    if (params.options.includeGraphContext) {
        const primaryCollection = params.collectionsToSearch[0] ?? 'projects';
        await expandAndFuseWithGraph(daemonClient, fusedResults, primaryCollection);
    }
    // Apply source diversity re-ranking when searching across multiple collections
    // (projects + libraries) so no single source dominates the top results.
    // Skip when the caller has explicitly set diverse=false.
    let diversityScore;
    let reranked = fusedResults;
    const diverseEnabled = params.options.diverse !== false;
    if (diverseEnabled && params.collectionsToSearch.length > 1) {
        const diversified = diversifyResults(fusedResults, DEFAULT_DIVERSITY_CONFIG);
        reranked = diversified.results;
        diversityScore = diversified.diversityScore;
    }
    const finalResults = reranked.slice(0, params.limit);
    if (params.options.expandContext)
        await expandParentContext(qdrantClient, finalResults);
    if (params.options.includeGraphContext)
        await expandGraphContext(daemonClient, finalResults);
    return recordAndBuildResponse(stateManager, finalResults, params, params.searchStartMs, diversityScore);
}
//# sourceMappingURL=search-helpers.js.map