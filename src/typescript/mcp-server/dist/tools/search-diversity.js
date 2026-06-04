/**
 * Source diversity re-ranking for search results.
 *
 * Mirrors the Rust `diversify_results()` in
 * src/rust/daemon/core/src/source_diversity/tier.rs.
 *
 * Algorithm:
 *   1. Walk results sorted by score descending (expected from caller).
 *   2. Group consecutive results within `scoreTierThreshold` of each other
 *      into tiers (threshold measured from the top of the tier).
 *   3. Within each tier, round-robin across sources to interleave them.
 *   4. Enforce `maxPerSource` globally — skip any result that would push
 *      a source beyond the cap.
 *   5. Return the re-ranked list and a diversity score in [0, 1].
 */
export const DEFAULT_DIVERSITY_CONFIG = {
    enabled: true,
    maxPerSource: 3,
    scoreTierThreshold: 0.05,
};
/**
 * Derive a stable source key for a result.
 *
 * Uses `collection:library_name` for library results, and
 * `collection:tenant_id` for project results.  Falls back to
 * `collection:unknown` when neither field is present.
 */
export function extractSource(result) {
    const collection = result.collection;
    const libraryName = result.metadata?.library_name;
    const tenantId = result.metadata?.tenant_id;
    return libraryName ? `${collection}:${libraryName}` : `${collection}:${tenantId ?? 'unknown'}`;
}
/** Group sorted results into tiers by score proximity. */
function buildScoreTiers(results, threshold) {
    if (results.length === 0)
        return [];
    const first = results[0];
    if (first === undefined)
        return [];
    const tiers = [];
    let currentTier = [first];
    let tierTopScore = first.score;
    for (let i = 1; i < results.length; i++) {
        const r = results[i];
        if (r === undefined)
            continue;
        if (Math.abs(tierTopScore - r.score) <= threshold) {
            currentTier.push(r);
        }
        else {
            tiers.push(currentTier);
            currentTier = [r];
            tierTopScore = r.score;
        }
    }
    if (currentTier.length > 0)
        tiers.push(currentTier);
    return tiers;
}
/** Round-robin interleave a single tier by source. */
function interleaveTier(tier) {
    if (tier.length <= 1)
        return tier;
    const sourceOrder = [];
    const groups = new Map();
    for (const r of tier) {
        const source = extractSource(r);
        if (!groups.has(source)) {
            sourceOrder.push(source);
            groups.set(source, []);
        }
        groups.get(source).push(r);
    }
    const indices = new Array(sourceOrder.length).fill(0);
    let exhausted = 0;
    const output = [];
    while (exhausted < sourceOrder.length) {
        for (let i = 0; i < sourceOrder.length; i++) {
            const source = sourceOrder[i];
            if (source === undefined)
                continue;
            const group = groups.get(source);
            const idx = indices[i] ?? 0;
            if (idx < group.length) {
                const item = group[idx];
                if (item !== undefined)
                    output.push(item);
                indices[i] = idx + 1;
                if (idx + 1 === group.length)
                    exhausted++;
            }
        }
    }
    return output;
}
/**
 * Calculate the diversity score for a result list.
 *
 * Returns `unique_sources / total_results` in [0, 1].
 * An empty list returns 1.0 (no diversity concern).
 */
export function computeDiversityScore(results) {
    if (results.length === 0)
        return 1.0;
    const unique = new Set(results.map(extractSource));
    return unique.size / results.length;
}
/**
 * Apply source diversity re-ranking.
 *
 * Input must already be sorted by score descending (as produced by
 * `applyRRFFusion` / `fusedResults.sort`).
 */
export function diversifyResults(results, config = DEFAULT_DIVERSITY_CONFIG) {
    if (!config.enabled || results.length === 0) {
        return { results, diversityScore: computeDiversityScore(results) };
    }
    const tiers = buildScoreTiers(results, config.scoreTierThreshold);
    const sourceCounts = new Map();
    const output = [];
    const spillover = [];
    const targetCount = results.length;
    for (const tier of tiers) {
        const interleaved = interleaveTier(tier);
        for (const r of interleaved) {
            const source = extractSource(r);
            const count = sourceCounts.get(source) ?? 0;
            if (count < config.maxPerSource) {
                sourceCounts.set(source, count + 1);
                output.push(r);
            }
            else {
                spillover.push(r);
            }
        }
    }
    // Backfill from spillover to preserve the requested result count.
    for (const r of spillover) {
        if (output.length >= targetCount)
            break;
        output.push(r);
    }
    return { results: output, diversityScore: computeDiversityScore(output) };
}
//# sourceMappingURL=search-diversity.js.map