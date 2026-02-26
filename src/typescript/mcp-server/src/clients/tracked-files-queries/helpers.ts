/**
 * Shared helper utilities for tracked-files query modules.
 */

import type { DegradedQueryResult } from '../sqlite-state-manager.js';

/**
 * Handle "no such table" SQLite errors by returning a degraded result.
 * Re-throws all other errors.
 */
export function handleTableNotFound<T>(
  error: unknown,
  fallbackData: T,
  tableName: string,
): DegradedQueryResult<T> {
  const errorMessage = error instanceof Error ? error.message : String(error);
  if (errorMessage.includes('no such table')) {
    return {
      data: fallbackData,
      status: 'degraded',
      reason: 'table_not_found',
      message: `Table ${tableName} not found. Daemon has not initialized database.`,
    };
  }
  throw error;
}
