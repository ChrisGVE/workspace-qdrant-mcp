/**
 * Tag-based query expansion for BM25 sparse search.
 */

import type { DaemonClient } from '../clients/daemon-client.js';
import type { SqliteStateManager } from '../clients/sqlite-state-manager.js';

/**
 * Expand sparse vector with keywords from matching tag baskets.
 *
 * 1. Query SQLite for tags matching the query text
 * 2. Retrieve keyword baskets for matching tags
 * 3. Generate a sparse vector for the expanded keywords
 * 4. Merge into the original sparse vector at reduced weight
 */
export async function expandSparseWithTags(
  daemonClient: DaemonClient,
  stateManager: SqliteStateManager,
  query: string,
  originalSparse: Record<number, number>,
  collections: string[],
  expansionWeight: number,
  maxExpandedKeywords: number,
  tenantId?: string,
): Promise<Record<number, number>> {
  try {
    // Collect expanded keywords from all searched collections
    const allKeywords = new Set<string>();

    for (const coll of collections) {
      const matchingTags = stateManager.getMatchingTags(query, coll, tenantId);
      if (matchingTags.length === 0) continue;

      const tagIds = matchingTags.map((t) => t.tagId);
      const baskets = stateManager.getKeywordBasketsForTags(tagIds);

      for (const basket of baskets) {
        for (const kw of basket.keywords) {
          allKeywords.add(kw);
        }
      }
    }

    if (allKeywords.size === 0) return originalSparse;

    // Limit expanded keywords
    const expandedKeywords = Array.from(allKeywords).slice(0, maxExpandedKeywords);

    // Generate sparse vector for expanded keywords
    const expansionText = expandedKeywords.join(' ');
    const expansionResponse = await daemonClient.generateSparseVector({ text: expansionText });
    if (!expansionResponse.success || !expansionResponse.indices_values) {
      return originalSparse;
    }

    // Merge: add expansion terms at reduced weight
    const merged = { ...originalSparse };
    for (const [indexStr, value] of Object.entries(expansionResponse.indices_values)) {
      const index = Number(indexStr);
      const weightedValue = value * expansionWeight;

      if (index in merged) {
        // Original term already present - keep original weight (don't dilute exact match)
        continue;
      }
      merged[index] = weightedValue;
    }

    return merged;
  } catch {
    // Expansion is best-effort; never block search
    return originalSparse;
  }
}
