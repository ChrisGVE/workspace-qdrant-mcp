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
export function getMatchingTags(
  db: DatabaseType | null,
  query: string,
  collection: string,
  tenantId?: string,
): { tagId: number; tag: string; score: number }[] {
  if (!db) return [];

  try {
    // Tokenize query into meaningful words (3+ chars, lowercase)
    const tokens = query
      .toLowerCase()
      .split(/\s+/)
      .map((t) => t.replace(/[^a-z0-9_-]/g, ''))
      .filter((t) => t.length >= 3);

    if (tokens.length === 0) return [];

    // Build LIKE conditions for each token
    const likeConditions = tokens.map(() => 'LOWER(t.tag) LIKE ?').join(' OR ');
    const likeParams = tokens.map((t) => `%${t}%`);

    const params: (string | number)[] = [collection, ...likeParams];
    let tenantClause = '';
    if (tenantId) {
      tenantClause = 'AND t.tenant_id = ?';
      params.push(tenantId);
    }

    const rows = db
      .prepare(
        `
        SELECT DISTINCT t.tag_id, t.tag, t.score
        FROM tags t
        WHERE t.collection = ?
          AND t.tag_type = 'concept'
          AND (${likeConditions})
          ${tenantClause}
        ORDER BY t.score DESC
        LIMIT 10
        `,
      )
      .all(...params) as Array<{ tag_id: number; tag: string; score: number }>;

    return rows.map((r) => ({ tagId: r.tag_id, tag: r.tag, score: r.score }));
  } catch {
    // Tags table may not exist if daemon hasn't migrated to v16+
    return [];
  }
}

/**
 * Retrieve keyword baskets for a set of tag IDs.
 *
 * Returns the keywords_json content (an array of keyword strings)
 * for each basket associated with the given tag IDs.
 */
export function getKeywordBasketsForTags(
  db: DatabaseType | null,
  tagIds: number[],
): { tagId: number; keywords: string[] }[] {
  if (!db || tagIds.length === 0) return [];

  try {
    const placeholders = tagIds.map(() => '?').join(',');
    const rows = db
      .prepare(
        `
        SELECT kb.tag_id, kb.keywords_json
        FROM keyword_baskets kb
        WHERE kb.tag_id IN (${placeholders})
        `,
      )
      .all(...tagIds) as Array<{ tag_id: number; keywords_json: string }>;

    return rows.map((r) => {
      let keywords: string[] = [];
      try {
        keywords = JSON.parse(r.keywords_json) as string[];
      } catch {
        // Malformed JSON - skip
      }
      return { tagId: r.tag_id, keywords };
    });
  } catch {
    // Table may not exist
    return [];
  }
}

/**
 * List concept tags for a collection, optionally filtered by tenant.
 *
 * Returns distinct tag names with document count and average score,
 * ordered by frequency (most common first).
 */
export function listTags(
  db: DatabaseType | null,
  collection: string,
  tenantId?: string,
  limit = 50,
): { tag: string; docCount: number; avgScore: number }[] {
  if (!db) return [];

  try {
    const params: (string | number)[] = [collection];
    let tenantClause = '';
    if (tenantId) {
      tenantClause = 'AND t.tenant_id = ?';
      params.push(tenantId);
    }
    params.push(limit);

    const rows = db
      .prepare(
        `
        SELECT t.tag,
               COUNT(DISTINCT t.doc_id) as doc_count,
               ROUND(AVG(t.score), 4) as avg_score
        FROM tags t
        WHERE t.collection = ?
          AND t.tag_type = 'concept'
          ${tenantClause}
        GROUP BY t.tag
        ORDER BY doc_count DESC, avg_score DESC
        LIMIT ?
        `,
      )
      .all(...params) as Array<{ tag: string; doc_count: number; avg_score: number }>;

    return rows.map((r) => ({
      tag: r.tag,
      docCount: r.doc_count,
      avgScore: r.avg_score,
    }));
  } catch {
    return [];
  }
}

/**
 * Get the canonical tag hierarchy for a collection.
 *
 * Returns canonical tags with their parent-child relationships.
 */
export function getTagHierarchy(
  db: DatabaseType | null,
  collection: string,
  tenantId?: string,
): { name: string; level: number; parentName: string | null; childCount: number }[] {
  if (!db) return [];

  try {
    const params: string[] = [collection];
    let tenantClause = '';
    if (tenantId) {
      tenantClause = 'AND ct.tenant_id = ?';
      params.push(tenantId);
    }

    const rows = db
      .prepare(
        `
        SELECT ct.canonical_name,
               ct.level,
               parent.canonical_name as parent_name,
               (SELECT COUNT(*) FROM canonical_tags child
                WHERE child.parent_id = ct.canonical_id) as child_count
        FROM canonical_tags ct
        LEFT JOIN canonical_tags parent ON ct.parent_id = parent.canonical_id
        WHERE ct.collection = ?
          ${tenantClause}
        ORDER BY ct.level ASC, ct.canonical_name ASC
        `,
      )
      .all(...params) as Array<{
      canonical_name: string;
      level: number;
      parent_name: string | null;
      child_count: number;
    }>;

    return rows.map((r) => ({
      name: r.canonical_name,
      level: r.level,
      parentName: r.parent_name,
      childCount: r.child_count,
    }));
  } catch {
    return [];
  }
}
