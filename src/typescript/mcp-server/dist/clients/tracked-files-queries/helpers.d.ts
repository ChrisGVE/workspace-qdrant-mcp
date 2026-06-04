/**
 * Shared helper utilities for tracked-files query modules.
 */
import type { DegradedQueryResult } from '../sqlite-state-manager.js';
/**
 * Handle "no such table" SQLite errors by returning a degraded result.
 * Re-throws all other errors.
 */
export declare function handleTableNotFound<T>(error: unknown, fallbackData: T, tableName: string): DegradedQueryResult<T>;
//# sourceMappingURL=helpers.d.ts.map