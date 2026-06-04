/**
 * Search tool facade — delegates to domain-specific modules.
 *
 * - search-types.ts: Types, interfaces, constants
 * - search-filters.ts: Filter construction, collection determination
 * - search-qdrant.ts: Qdrant search, RRF fusion, parent context, fallback
 * - search-exact.ts: FTS5 exact/substring search via daemon
 * - search-expansion.ts: Tag-based BM25 query expansion
 * - search-helpers.ts: Phase helpers (project context, embeddings, fan-out, finalize)
 */
import { randomUUID } from 'node:crypto';
import { QdrantClient } from '@qdrant/js-client-rest';
import { DEFAULT_LIMIT, DEFAULT_SCORE_THRESHOLD, DEFAULT_EXPANSION_WEIGHT, DEFAULT_MAX_EXPANDED_KEYWORDS, } from './search-types.js';
import { determineCollections } from './search-filters.js';
import { retrieveParent, collectionExists } from './search-qdrant.js';
import { searchExact } from './search-exact.js';
import { expandSparseWithTags } from './search-expansion.js';
import { resolveProjectContext, logSearchEventPre, generateEmbeddings, searchAllCollections, finalizeResults, } from './search-helpers.js';
/** Format an explicit status_reason for the F-014 base-point degradation
 * case. Used in both the embedding-fallback path and the normal
 * fan-out path so callers always see why instance isolation was
 * bypassed. */
function formatBasePointsDegradedReason(activeCount) {
    const count = activeCount ?? 'too many';
    return (`Worktree/instance isolation degraded: project has ${count} active base points, ` +
        `exceeding the 500-filter cap; tenant filter still applies but base-point ` +
        `narrowing was bypassed. Narrow further with pathGlob, branch, or component to ` +
        `restore worktree-level isolation.`);
}
/**
 * Search tool for hybrid semantic + keyword search
 */
export class SearchTool {
    qdrantClient;
    daemonClient;
    _stateManager;
    projectDetector;
    enableTagExpansion;
    expansionWeight;
    maxExpandedKeywords;
    constructor(config, daemonClient, stateManager, projectDetector) {
        const clientConfig = {
            url: config.qdrantUrl,
            timeout: config.qdrantTimeout ?? 5000,
        };
        if (config.qdrantApiKey) {
            clientConfig.apiKey = config.qdrantApiKey;
        }
        this.qdrantClient = new QdrantClient(clientConfig);
        this.daemonClient = daemonClient;
        this._stateManager = stateManager;
        this.projectDetector = projectDetector;
        this.enableTagExpansion = config.enableTagExpansion ?? true;
        this.expansionWeight = config.expansionWeight ?? DEFAULT_EXPANSION_WEIGHT;
        this.maxExpandedKeywords = config.maxExpandedKeywords ?? DEFAULT_MAX_EXPANDED_KEYWORDS;
    }
    get stateManager() {
        return this._stateManager;
    }
    async prepareEmbeddings(options, query, mode, collectionsToSearch, currentProjectId, basePoints) {
        const embeddings = await generateEmbeddings(this.daemonClient, this.qdrantClient, query, mode, options, collectionsToSearch, { currentProjectId, basePoints });
        if ('fallback' in embeddings)
            return embeddings;
        let { denseEmbedding, sparseVector } = embeddings;
        if (sparseVector && this.enableTagExpansion && (mode === 'hybrid' || mode === 'keyword')) {
            sparseVector = await expandSparseWithTags(this.daemonClient, this._stateManager, query, sparseVector, collectionsToSearch, this.expansionWeight, this.maxExpandedKeywords, currentProjectId);
        }
        return { denseEmbedding, sparseVector };
    }
    async resolveContextAndLog(options, query, limit, scope, projectId) {
        const eventId = randomUUID();
        const searchStartMs = Date.now();
        const resolution = await resolveProjectContext(projectId, scope, this.projectDetector, this._stateManager);
        logSearchEventPre(this._stateManager, eventId, resolution.currentProjectId, query, limit, {
            collection: options.collection,
            scope,
            branch: options.branch,
            fileType: options.fileType,
            libraryName: options.libraryName,
            tag: options.tag,
        });
        return {
            eventId,
            searchStartMs,
            currentProjectId: resolution.currentProjectId,
            basePoints: resolution.basePoints,
            basePointsDegraded: resolution.basePointsDegraded ?? false,
            basePointsActiveCount: resolution.basePointsActiveCount,
        };
    }
    /**
     * Resolve scope to tenant filter and relevance decay weights via the daemon.
     *
     * - scope=project: no daemon call needed, returns no filter/decay.
     * - scope=group: daemon returns grouped tenant_ids + decay weights.
     * - scope=all: daemon returns no tenant filter + decay map for the
     *   current project (1.0) so other-project results get penalized (0.4).
     */
    async resolveScopeFilter(currentProjectId, scope) {
        if (scope === 'project' || !currentProjectId) {
            return { groupTenantIds: undefined, decayMap: undefined };
        }
        try {
            const response = await this.daemonClient.resolveSearchScope({
                tenant_id: currentProjectId,
                scope,
            });
            const decayMap = new Map();
            for (const entry of response.decay_map ?? []) {
                decayMap.set(entry.tenant_id, entry.multiplier);
            }
            return {
                groupTenantIds: response.filter_by_tenant ? response.tenant_ids : undefined,
                decayMap: decayMap.size > 0 ? decayMap : undefined,
            };
        }
        catch {
            return { groupTenantIds: undefined, decayMap: undefined };
        }
    }
    async search(options) {
        if (options.exact) {
            return searchExact(this.daemonClient, this._stateManager, this.projectDetector, options);
        }
        const mode = options.mode ?? 'hybrid';
        const limit = options.limit ?? DEFAULT_LIMIT;
        const scope = options.scope ?? 'project';
        const collectionsToSearch = determineCollections(options.collection, scope, options.includeLibraries ?? false);
        const { eventId, searchStartMs, currentProjectId, basePoints, basePointsDegraded, basePointsActiveCount, } = await this.resolveContextAndLog(options, options.query, limit, scope, options.projectId);
        const { groupTenantIds, decayMap } = await this.resolveScopeFilter(currentProjectId, scope);
        if (scope === 'group' && (!groupTenantIds || groupTenantIds.length === 0)) {
            return {
                results: [],
                total: 0,
                query: options.query,
                mode,
                scope,
                collections_searched: collectionsToSearch,
                status: 'error',
                status_reason: 'Group scope requires a resolved project context. Could not determine project group membership.',
            };
        }
        const embeddings = await this.prepareEmbeddings(options, options.query, mode, collectionsToSearch, currentProjectId, basePoints);
        if ('fallback' in embeddings) {
            if (basePointsDegraded) {
                embeddings.fallback.status = 'uncertain';
                const reason = formatBasePointsDegradedReason(basePointsActiveCount);
                embeddings.fallback.status_reason = embeddings.fallback.status_reason
                    ? `${embeddings.fallback.status_reason}; ${reason}`
                    : reason;
            }
            return embeddings.fallback;
        }
        return this.runSearchAndFinalize(options, mode, limit, scope, collectionsToSearch, eventId, searchStartMs, currentProjectId, basePoints, basePointsDegraded, basePointsActiveCount, embeddings.denseEmbedding, embeddings.sparseVector, groupTenantIds, decayMap);
    }
    async runSearchCollections(options, mode, limit, scope, collectionsToSearch, currentProjectId, basePoints, denseEmbedding, sparseVector, groupTenantIds) {
        const scoreThreshold = options.scoreThreshold ?? DEFAULT_SCORE_THRESHOLD;
        return searchAllCollections(this.qdrantClient, {
            collectionsToSearch,
            scope,
            currentProjectId,
            groupTenantIds,
            basePoints,
            branch: options.branch,
            fileType: options.fileType,
            libraryName: options.libraryName,
            libraryPath: options.libraryPath,
            tag: options.tag,
            tags: options.tags,
            options,
            mode,
            denseEmbedding,
            sparseVector,
            limit,
            scoreThreshold,
        });
    }
    applyRelevanceDecay(results, currentProjectId, decayMap) {
        if (!decayMap || !currentProjectId)
            return;
        for (const result of results) {
            const tenantId = result.metadata?.tenant_id;
            if (!tenantId)
                continue;
            const multiplier = decayMap.get(tenantId) ?? 0.4;
            result.score *= multiplier;
        }
        results.sort((a, b) => b.score - a.score);
    }
    async runSearchAndFinalize(options, mode, limit, scope, collectionsToSearch, eventId, searchStartMs, currentProjectId, basePoints, basePointsDegraded, basePointsActiveCount, denseEmbedding, sparseVector, groupTenantIds, decayMap) {
        const collectionsResult = await this.runSearchCollections(options, mode, limit, scope, collectionsToSearch, currentProjectId, basePoints, denseEmbedding, sparseVector, groupTenantIds);
        if (decayMap && decayMap.size > 0) {
            this.applyRelevanceDecay(collectionsResult.allResults, currentProjectId, decayMap);
        }
        let { status, statusReason } = collectionsResult;
        if (basePointsDegraded) {
            status = 'uncertain';
            const reason = formatBasePointsDegradedReason(basePointsActiveCount);
            statusReason = statusReason ? `${statusReason}; ${reason}` : reason;
        }
        return finalizeResults(this.qdrantClient, this.daemonClient, this._stateManager, {
            allResults: collectionsResult.allResults,
            mode,
            limit,
            options,
            eventId,
            searchStartMs,
            query: options.query,
            scope,
            collectionsToSearch,
            status,
            statusReason,
        });
    }
    async retrieveParent(parentUnitId, collection) {
        return retrieveParent(this.qdrantClient, parentUnitId, collection);
    }
    async collectionExists(collectionName) {
        return collectionExists(this.qdrantClient, collectionName);
    }
}
//# sourceMappingURL=search.js.map