import { DENSE_VECTOR_NAME, SPARSE_VECTOR_NAME, RRF_K, DEFAULT_LIMIT, PROJECTS_COLLECTION, LIBRARIES_COLLECTION, SCRATCHPAD_COLLECTION, RULES_COLLECTION, } from './search-types.js';
import { buildFilter } from './search-filters.js';
import { FIELD_CONTENT, FIELD_TITLE, FIELD_PARENT_UNIT_ID } from '../common/native-bridge.js';
/**
 * Build a Provenance object from a Qdrant payload and collection name.
 *
 * The `source` field is derived from the canonical collection name constants
 * so it is always one of the three known values.  Unknown collection names
 * fall back to 'projects' to avoid a type error.
 */
function buildProvenance(payload, collection) {
    let source;
    if (collection === LIBRARIES_COLLECTION) {
        source = 'libraries';
    }
    else if (collection === SCRATCHPAD_COLLECTION) {
        source = 'scratchpad';
    }
    else if (collection === RULES_COLLECTION) {
        source = 'rules';
    }
    else {
        source = 'projects';
    }
    const provenance = { source };
    const libraryName = payload?.library_name;
    if (libraryName)
        provenance.library_name = libraryName;
    const libraryPath = payload?.library_path;
    if (libraryPath)
        provenance.library_path = libraryPath;
    // document_name takes precedence over title for doc_title
    const docTitle = payload?.document_name ??
        payload?.[FIELD_TITLE];
    if (docTitle)
        provenance.doc_title = docTitle;
    const tenantId = payload?.tenant_id;
    if (tenantId)
        provenance.source_project_id = tenantId;
    return provenance;
}
/** Map a Qdrant search hit to a SearchResult. */
function hitToResult(hit, collection, searchType) {
    const result = {
        id: String(hit.id),
        score: hit.score,
        collection,
        content: hit.payload?.[FIELD_CONTENT] ?? '',
        metadata: { ...hit.payload, _search_type: searchType },
        provenance: buildProvenance(hit.payload, collection),
    };
    const title = hit.payload?.[FIELD_TITLE];
    if (title)
        result.title = title;
    return result;
}
/** Run the dense (semantic) search leg. */
async function searchDense(qdrantClient, params) {
    if (!(params.mode === 'hybrid' || params.mode === 'semantic') || !params.denseEmbedding)
        return [];
    try {
        const req = {
            vector: { name: DENSE_VECTOR_NAME, vector: params.denseEmbedding },
            limit: params.limit,
            score_threshold: params.scoreThreshold,
            with_payload: true,
        };
        if (params.filter)
            req.filter = params.filter;
        const hits = await qdrantClient.search(params.collection, req);
        return hits.map((h) => hitToResult(h, params.collection, 'semantic'));
    }
    catch {
        return [];
    }
}
/** Run the sparse (keyword) search leg. */
async function searchSparse(qdrantClient, params) {
    if (!(params.mode === 'hybrid' || params.mode === 'keyword') || !params.sparseVector)
        return [];
    try {
        const indices = Object.keys(params.sparseVector).map(Number);
        const values = Object.values(params.sparseVector);
        if (indices.length === 0)
            return [];
        const req = {
            vector: { name: SPARSE_VECTOR_NAME, vector: { indices, values } },
            limit: params.limit,
            score_threshold: params.scoreThreshold * 0.5,
            with_payload: true,
        };
        if (params.filter)
            req.filter = params.filter;
        const hits = await qdrantClient.search(params.collection, req);
        return hits.map((h) => hitToResult(h, params.collection, 'keyword'));
    }
    catch {
        return [];
    }
}
/**
 * Search a single collection with dense and/or sparse vectors.
 */
export async function searchCollection(qdrantClient, params) {
    const [dense, sparse] = await Promise.all([
        searchDense(qdrantClient, params),
        searchSparse(qdrantClient, params),
    ]);
    return [...dense, ...sparse];
}
/**
 * Apply Reciprocal Rank Fusion to combine results.
 * RRF score = sum(1 / (k + rank_i)) for each result across rankings.
 */
export function applyRRFFusion(results, mode) {
    if (mode !== 'hybrid' || results.length === 0)
        return results;
    const semanticResults = results.filter((r) => r.metadata['_search_type'] === 'semantic');
    const keywordResults = results.filter((r) => r.metadata['_search_type'] === 'keyword');
    if (semanticResults.length === 0 || keywordResults.length === 0)
        return results;
    const rrfScores = new Map();
    semanticResults.forEach((result, rank) => {
        const key = `${result.collection}:${result.id}`;
        const rrfScore = 1 / (RRF_K + rank + 1);
        const existing = rrfScores.get(key);
        if (existing)
            existing.score += rrfScore;
        else
            rrfScores.set(key, { score: rrfScore, result: { ...result } });
    });
    keywordResults.forEach((result, rank) => {
        const key = `${result.collection}:${result.id}`;
        const rrfScore = 1 / (RRF_K + rank + 1);
        const existing = rrfScores.get(key);
        if (existing)
            existing.score += rrfScore;
        else
            rrfScores.set(key, { score: rrfScore, result: { ...result } });
    });
    return Array.from(rrfScores.values()).map(({ score, result }) => ({
        ...result,
        score,
        metadata: { ...result.metadata, _search_type: 'hybrid' },
    }));
}
/** Index results by collection and parent ID for batch retrieval. */
function indexByParent(results) {
    const parentsByCollection = new Map();
    for (const result of results) {
        const parentId = result.metadata[FIELD_PARENT_UNIT_ID];
        if (!parentId)
            continue;
        let collMap = parentsByCollection.get(result.collection);
        if (!collMap) {
            collMap = new Map();
            parentsByCollection.set(result.collection, collMap);
        }
        let bucket = collMap.get(parentId);
        if (!bucket) {
            bucket = [];
            collMap.set(parentId, bucket);
        }
        bucket.push(result);
    }
    return parentsByCollection;
}
/** Fetch parent points for one collection and attach context to linked results. */
async function attachParentsForCollection(qdrantClient, collection, parentMap) {
    const parentIds = Array.from(parentMap.keys());
    if (parentIds.length === 0)
        return;
    try {
        const points = await qdrantClient.retrieve(collection, { ids: parentIds, with_payload: true });
        for (const point of points) {
            const linked = parentMap.get(String(point.id));
            if (!linked)
                continue;
            const ctx = {
                parent_unit_id: String(point.id),
                unit_type: point.payload?.['unit_type'] ?? 'unknown',
                unit_text: point.payload?.['unit_text'] ?? '',
            };
            const locator = point.payload?.['locator'];
            if (locator)
                ctx.locator = locator;
            for (const r of linked)
                r.parent_context = ctx;
        }
    }
    catch {
        // Parent records may not exist yet (pre-migration data)
    }
}
/** Expand parent context for search results (fetches parent unit records). */
export async function expandParentContext(qdrantClient, results) {
    const parentsByCollection = indexByParent(results);
    for (const [collection, parentMap] of parentsByCollection) {
        await attachParentsForCollection(qdrantClient, collection, parentMap);
    }
}
/** Retrieve a single parent unit by ID for on-demand expansion. */
export async function retrieveParent(qdrantClient, parentUnitId, collection) {
    try {
        const points = await qdrantClient.retrieve(collection, {
            ids: [parentUnitId],
            with_payload: true,
        });
        const point = points[0];
        if (!point)
            return null;
        const context = {
            parent_unit_id: String(point.id),
            unit_type: point.payload?.['unit_type'] ?? 'unknown',
            unit_text: point.payload?.['unit_text'] ?? '',
        };
        const locator = point.payload?.['locator'];
        if (locator)
            context.locator = locator;
        return context;
    }
    catch {
        return null;
    }
}
/** Text-match points from a scroll result against a query string. */
function matchScrollPoints(points, collection, queryLower) {
    const matched = [];
    for (const point of points) {
        const content = point.payload?.[FIELD_CONTENT] ?? '';
        const titlePayload = point.payload?.[FIELD_TITLE] ?? '';
        if (!content.toLowerCase().includes(queryLower) &&
            !titlePayload.toLowerCase().includes(queryLower))
            continue;
        const result = {
            id: String(point.id),
            score: 0.5,
            collection,
            content,
            metadata: { ...point.payload, _search_type: 'fallback' },
        };
        if (titlePayload)
            result.title = titlePayload;
        matched.push(result);
    }
    return matched;
}
/**
 * Build the per-collection scroll filter for the fallback path. Returns
 * `null` when scope = `'project'` but the tenant could not be resolved —
 * the caller MUST refuse to scroll in that case rather than broaden to
 * every tenant in the collection (F-001).
 */
function buildFallbackFilter(collection, options, context) {
    const scope = options.scope ?? 'project';
    // F-001: project-scope fallback REQUIRES a resolved tenant. Without
    // one, refuse to scroll — broad scroll + local substring match would
    // leak documents from foreign tenants.
    if (scope === 'project' && !context.currentProjectId)
        return null;
    const filterParams = {
        collection,
        scope,
        projectId: context.currentProjectId,
        groupTenantIds: undefined,
        branch: options.branch,
        fileType: options.fileType,
        libraryName: options.libraryName,
        libraryPath: options.libraryPath,
        tag: options.tag,
        tags: options.tags,
        pathGlob: options.pathGlob,
        component: options.component,
        basePoints: collection === PROJECTS_COLLECTION ? context.basePoints : undefined,
    };
    return buildFilter(filterParams);
}
/**
 * Fallback search when daemon is unavailable.
 *
 * Closes F-001: scrolls Qdrant with a tenant/project/library filter built
 * from the same `buildFilter` helper as the primary search path. If the
 * filter cannot be assembled (project-scope with unresolved project), no
 * scroll is performed and the caller receives a degraded, empty result
 * with a `status_reason` explaining why. Local substring matching on raw
 * scroll output has been removed — when filtering is in place the scroll
 * itself is the matcher; when it isn't, we refuse to read.
 */
export async function fallbackSearch(qdrantClient, options, collections, context) {
    const results = [];
    const queryLower = options.query.toLowerCase();
    const refusedCollections = [];
    let attemptedCollections = 0;
    for (const collection of collections) {
        const filter = buildFallbackFilter(collection, options, context);
        if (!filter) {
            // F-001: project-scope fallback with unresolved tenant — never scroll.
            refusedCollections.push(collection);
            continue;
        }
        attemptedCollections += 1;
        try {
            const scrollResult = await qdrantClient.scroll(collection, {
                limit: (options.limit ?? DEFAULT_LIMIT) * 3,
                with_payload: true,
                filter,
            });
            // Substring-match within the tenant-scoped page only (the filter
            // already enforces isolation; this is a coarse keyword filter).
            results.push(...matchScrollPoints(scrollResult.points, collection, queryLower));
        }
        catch {
            // Collection may not exist
        }
    }
    const limitedResults = results.slice(0, options.limit ?? DEFAULT_LIMIT);
    const scope = options.scope ?? 'project';
    const isDegraded = attemptedCollections === 0 && refusedCollections.length > 0;
    const statusReason = isDegraded
        ? `Daemon unavailable and project scope unresolved - cannot run cross-tenant fallback. Refused collections: ${refusedCollections.join(', ')}`
        : 'Daemon unavailable - using fallback text search';
    return {
        results: limitedResults,
        total: limitedResults.length,
        query: options.query,
        mode: options.mode ?? 'hybrid',
        scope,
        collections_searched: collections,
        status: 'uncertain',
        status_reason: statusReason,
    };
}
/** Check if a collection exists. */
export async function collectionExists(qdrantClient, collectionName) {
    try {
        await qdrantClient.getCollection(collectionName);
        return true;
    }
    catch {
        return false;
    }
}
//# sourceMappingURL=search-qdrant.js.map