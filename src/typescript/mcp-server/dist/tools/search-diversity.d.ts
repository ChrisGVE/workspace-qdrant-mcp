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
import type { SearchResult } from './search-types.js';
export interface DiversityConfig {
    enabled: boolean;
    /** Maximum results from one source in the final output. */
    maxPerSource: number;
    /** Score delta within which results are grouped into the same tier. */
    scoreTierThreshold: number;
}
export declare const DEFAULT_DIVERSITY_CONFIG: DiversityConfig;
/**
 * Derive a stable source key for a result.
 *
 * Uses `collection:library_name` for library results, and
 * `collection:tenant_id` for project results.  Falls back to
 * `collection:unknown` when neither field is present.
 */
export declare function extractSource(result: SearchResult): string;
/**
 * Calculate the diversity score for a result list.
 *
 * Returns `unique_sources / total_results` in [0, 1].
 * An empty list returns 1.0 (no diversity concern).
 */
export declare function computeDiversityScore(results: SearchResult[]): number;
/**
 * Apply source diversity re-ranking.
 *
 * Input must already be sorted by score descending (as produced by
 * `applyRRFFusion` / `fusedResults.sort`).
 */
export declare function diversifyResults(results: SearchResult[], config?: DiversityConfig): {
    results: SearchResult[];
    diversityScore: number;
};
//# sourceMappingURL=search-diversity.d.ts.map