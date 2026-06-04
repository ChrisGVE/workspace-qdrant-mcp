/**
 * Project query operations for SqliteStateManager.
 *
 * All functions accept the db handle as first parameter for delegation.
 */
import type { Database as DatabaseType } from 'better-sqlite3';
import type { RegisteredProject } from '../types/state.js';
import type { DegradedQueryResult } from './sqlite-state-manager.js';
/**
 * Get project by path from watch_folders table.
 *
 * Uses longest-prefix matching to find the closest enclosing project.
 */
export declare function getProjectByPath(db: DatabaseType | null, projectPath: string): DegradedQueryResult<RegisteredProject | null>;
/**
 * Get project by tenant_id from watch_folders table.
 */
export declare function getProjectById(db: DatabaseType | null, projectId: string): DegradedQueryResult<RegisteredProject | null>;
/**
 * List all active projects from watch_folders table.
 */
export declare function listActiveProjects(db: DatabaseType | null): DegradedQueryResult<RegisteredProject[]>;
//# sourceMappingURL=project-queries.d.ts.map