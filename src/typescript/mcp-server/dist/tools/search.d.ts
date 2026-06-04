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
import type { DaemonClient } from '../clients/daemon-client.js';
import type { SqliteStateManager } from '../clients/sqlite-state-manager.js';
import type { ProjectDetector } from '../utils/project-detector.js';
export type { SearchMode, SearchScope, SearchOptions, ParentContext, Provenance, SearchResult, SearchResponse, SearchToolConfig, FilterParams, SearchCollectionParams, GraphContext, GraphContextNode, } from './search-types.js';
import type { SearchOptions, SearchResponse, SearchToolConfig, ParentContext } from './search-types.js';
/**
 * Search tool for hybrid semantic + keyword search
 */
export declare class SearchTool {
    private readonly qdrantClient;
    private readonly daemonClient;
    private readonly _stateManager;
    private readonly projectDetector;
    private readonly enableTagExpansion;
    private readonly expansionWeight;
    private readonly maxExpandedKeywords;
    constructor(config: SearchToolConfig, daemonClient: DaemonClient, stateManager: SqliteStateManager, projectDetector: ProjectDetector);
    get stateManager(): SqliteStateManager;
    private prepareEmbeddings;
    private resolveContextAndLog;
    /**
     * Resolve scope to tenant filter and relevance decay weights via the daemon.
     *
     * - scope=project: no daemon call needed, returns no filter/decay.
     * - scope=group: daemon returns grouped tenant_ids + decay weights.
     * - scope=all: daemon returns no tenant filter + decay map for the
     *   current project (1.0) so other-project results get penalized (0.4).
     */
    private resolveScopeFilter;
    search(options: SearchOptions): Promise<SearchResponse>;
    private runSearchCollections;
    private applyRelevanceDecay;
    private runSearchAndFinalize;
    retrieveParent(parentUnitId: string, collection: string): Promise<ParentContext | null>;
    collectionExists(collectionName: string): Promise<boolean>;
}
//# sourceMappingURL=search.d.ts.map