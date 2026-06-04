/**
 * Query operations for project components from the project_components table.
 *
 * Returns detected workspace components (Cargo, npm, directory fallback)
 * that the daemon persists during file processing.
 */
import type { Database as DatabaseType } from 'better-sqlite3';
import type { DegradedQueryResult } from '../sqlite-state-manager.js';
export interface ComponentEntry {
    componentName: string;
    basePath: string;
    source: string;
}
/**
 * List project components from the daemon's project_components table.
 *
 * Returns detected workspace components (Cargo, npm, directory fallback)
 * that the daemon persists during file processing.
 */
export declare function listProjectComponents(db: DatabaseType | null, watchFolderId: string): DegradedQueryResult<ComponentEntry[]>;
//# sourceMappingURL=components.d.ts.map