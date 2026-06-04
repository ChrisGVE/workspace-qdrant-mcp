/**
 * Search filter construction and collection determination.
 */
import type { SearchScope, FilterParams } from './search-types.js';
/**
 * Extract the deterministic path prefix from a glob pattern.
 * Returns everything before the first glob metacharacter (* ? [ {).
 * Example: "src/**\/*.rs" → "src/", "**\/*.rs" → ""
 */
export declare function extractGlobPrefix(glob: string): string;
/**
 * Determine which collections to search based on scope.
 */
export declare function determineCollections(collection: string | undefined, scope: SearchScope, includeLibraries: boolean): string[];
/**
 * Build Qdrant filter based on search parameters.
 */
export declare function buildFilter(params: FilterParams): Record<string, unknown> | null;
//# sourceMappingURL=search-filters.d.ts.map