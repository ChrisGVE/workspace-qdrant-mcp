/**
 * Tag-based query expansion for BM25 sparse search.
 */
/** Collect unique expansion keywords from tag baskets across all collections. */
function collectTagKeywords(stateManager, query, collections, tenantId, maxKeywords) {
    const allKeywords = new Set();
    for (const coll of collections) {
        const matchingTags = stateManager.getMatchingTags(query, coll, tenantId);
        if (matchingTags.length === 0)
            continue;
        const baskets = stateManager.getKeywordBasketsForTags(matchingTags.map((t) => t.tagId));
        for (const basket of baskets) {
            for (const kw of basket.keywords)
                allKeywords.add(kw);
        }
    }
    return Array.from(allKeywords).slice(0, maxKeywords);
}
/** Merge an expansion sparse vector into the original at reduced weight (no-overwrite). */
function mergeSparseVectors(original, expansion, weight) {
    const merged = { ...original };
    for (const [indexStr, value] of Object.entries(expansion)) {
        const index = Number(indexStr);
        if (!(index in merged))
            merged[index] = value * weight;
    }
    return merged;
}
/**
 * Expand sparse vector with keywords from matching tag baskets.
 *
 * 1. Query SQLite for tags matching the query text
 * 2. Retrieve keyword baskets for matching tags
 * 3. Generate a sparse vector for the expanded keywords
 * 4. Merge into the original sparse vector at reduced weight
 */
export async function expandSparseWithTags(daemonClient, stateManager, query, originalSparse, collections, expansionWeight, maxExpandedKeywords, tenantId) {
    try {
        const keywords = collectTagKeywords(stateManager, query, collections, tenantId, maxExpandedKeywords);
        if (keywords.length === 0)
            return originalSparse;
        const expansionResponse = await daemonClient.generateSparseVector({ text: keywords.join(' ') });
        if (!expansionResponse.success || !expansionResponse.indices_values)
            return originalSparse;
        return mergeSparseVectors(originalSparse, expansionResponse.indices_values, expansionWeight);
    }
    catch {
        return originalSparse;
    }
}
//# sourceMappingURL=search-expansion.js.map