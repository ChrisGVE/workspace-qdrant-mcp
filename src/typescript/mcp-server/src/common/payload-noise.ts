/**
 * Indexing / ranking-aid payload fields the daemon injects on every code chunk.
 *
 * `inject_extraction_results` in
 * `src/rust/daemon/core/src/strategies/processing/file/keyword_extract.rs`
 * writes these onto each Qdrant point so retrieval scoring (BM25/keyword
 * baskets, tag overlap) can use them. They are pure *indexing* signal: a
 * reading agent never consumes them, yet they add ~1.5–2k tokens per hit —
 * ~15–20k on a 10-hit `search`/`retrieve` response.
 *
 * Both the `search` truncate path ({@link ../tools/search-shaping.ts}) and the
 * `retrieve` metadata extractor ({@link ../tools/retrieve-types.ts}) strip them
 * from served payloads. Keep this list as the single source of truth so the two
 * paths cannot drift; it must mirror the keys inserted in `keyword_extract.rs`.
 */
export const RANKING_AID_KEYS: readonly string[] = [
  'keywords',
  'keyword_baskets',
  'concept_tags',
  'structural_tags',
];
