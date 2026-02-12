/**
 * Canonical UTC timestamp formatting for workspace-qdrant-mcp.
 *
 * All timestamps stored in SQLite MUST use ISO 8601 format with the `Z` suffix.
 * JavaScript's `Date.toISOString()` already produces this format, but centralizing
 * here ensures a single source of truth and makes the convention discoverable.
 *
 * Format: `YYYY-MM-DDTHH:MM:SS.mmmZ`
 */

/**
 * Return the current UTC time as an ISO 8601 string with Z suffix.
 */
export function utcNow(): string {
  return new Date().toISOString();
}
