/**
 * FTS5 exact/substring search via daemon's TextSearchService.
 */
import type { DaemonClient } from '../clients/daemon-client.js';
import type { SqliteStateManager } from '../clients/sqlite-state-manager.js';
import type { ProjectDetector } from '../utils/project-detector.js';
import type { SearchOptions, SearchResponse } from './search-types.js';
/**
 * Execute FTS5 exact/substring search via daemon's TextSearchService.
 * Maps TextSearchResponse to the standard SearchResponse format.
 */
export declare function searchExact(daemonClient: DaemonClient, stateManager: SqliteStateManager, projectDetector: ProjectDetector, options: SearchOptions): Promise<SearchResponse>;
//# sourceMappingURL=search-exact.d.ts.map