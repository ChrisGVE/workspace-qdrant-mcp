//! SQLite-bound expansion adapter: tag-basket keyword collection.
//!
//! Mirrors `collectTagKeywords` in `search-expansion.ts:9-26`. The pure
//! sparse-vector merge (`merge_sparse_vectors`) now lives in the shared
//! `wqm-client` crate (`wqm_client::search::expansion`, WI-d4 #82) and is
//! re-exported here so existing `crate::tools::search::expansion::…` paths keep
//! resolving. The keyword collection reads the `tags` table, so it stays local.

use rusqlite::Connection;

use super::options::DEFAULT_MAX_EXPANDED_KEYWORDS;
use crate::sqlite::tag_queries::{get_keyword_baskets_for_tags, get_matching_tags};

// The pure, SQLite-free vector merge lives in the shared client.
pub use wqm_client::search::expansion::merge_sparse_vectors;

/// Collect unique expansion keywords from tag baskets across all collections.
///
/// Pre-computed synchronously by `search_tool` while holding the SQLite lock,
/// then passed into the pipeline as an owned `Vec<String>` so the pipeline
/// future stays `Send`.
///
/// Mirrors `collectTagKeywords` in `search-expansion.ts:9-26`.
pub fn collect_expansion_keywords(
    conn: Option<&Connection>,
    query: &str,
    collections: &[String],
    tenant_id: Option<&str>,
) -> Vec<String> {
    collect_tag_keywords(
        conn,
        query,
        collections,
        tenant_id,
        DEFAULT_MAX_EXPANDED_KEYWORDS,
    )
}

fn collect_tag_keywords(
    conn: Option<&Connection>,
    query: &str,
    collections: &[String],
    tenant_id: Option<&str>,
    max_keywords: usize,
) -> Vec<String> {
    let mut seen = std::collections::HashSet::new();
    let mut all_keywords: Vec<String> = Vec::new();

    for coll in collections {
        let matching = get_matching_tags(conn, query, coll, tenant_id);
        if matching.is_empty() {
            continue;
        }
        let tag_ids: Vec<i64> = matching.iter().map(|t| t.tag_id).collect();
        let baskets = get_keyword_baskets_for_tags(conn, &tag_ids);
        for basket in baskets {
            for kw in basket.keywords {
                if seen.insert(kw.clone()) {
                    all_keywords.push(kw);
                }
            }
        }
    }

    all_keywords.into_iter().take(max_keywords).collect()
}
