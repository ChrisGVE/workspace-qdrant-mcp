/**
 * Tag and keyword basket query operations for SqliteStateManager.
 *
 * Provides tag matching, basket retrieval, listing, and hierarchy queries
 * against the tags/keyword_baskets/canonical_tags tables (schema v16+).
 */
import type { Database as DatabaseType } from 'better-sqlite3';
/**
 * Find tags matching query terms.
 *
 * Tokenizes the query into words and searches the `tags` table for
 * matching concept tags within the given collection and tenant.
 * Returns distinct tag IDs with their names, ordered by score.
 */
export declare function getMatchingTags(db: DatabaseType | null, query: string, collection: string, tenantId?: string): {
    tagId: number;
    tag: string;
    score: number;
}[];
/**
 * Retrieve keyword baskets for a set of tag IDs.
 *
 * Returns the keywords_json content (an array of keyword strings)
 * for each basket associated with the given tag IDs.
 */
export declare function getKeywordBasketsForTags(db: DatabaseType | null, tagIds: number[]): {
    tagId: number;
    keywords: string[];
}[];
/**
 * List concept tags for a collection, optionally filtered by tenant.
 *
 * Returns distinct tag names with document count and average score,
 * ordered by frequency (most common first).
 */
export declare function listTags(db: DatabaseType | null, collection: string, tenantId?: string, limit?: number): {
    tag: string;
    docCount: number;
    avgScore: number;
}[];
/**
 * Get the canonical tag hierarchy for a collection.
 *
 * Returns canonical tags with their parent-child relationships.
 */
export declare function getTagHierarchy(db: DatabaseType | null, collection: string, tenantId?: string): {
    name: string;
    level: number;
    parentName: string | null;
    childCount: number;
}[];
//# sourceMappingURL=tag-queries.d.ts.map