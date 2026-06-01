//! Sparse vector expansion via tag baskets.
//!
//! Mirrors `expandSparseWithTags` in `search-expansion.ts` lines 50-77.
//!
//! ## Algorithm (search-expansion.ts:50-77)
//! 1. Tokenise query, find matching tags from SQLite `tags` table.
//! 2. Retrieve keyword baskets for matched tag IDs.
//! 3. Join all keywords, generate sparse vector via daemon.
//! 4. Merge expansion vector into original at reduced weight (no-overwrite).
//!
//! The `weight` parameter defaults to `DEFAULT_EXPANSION_WEIGHT = 0.5`.
//! New indices not already in the original sparse vector are added scaled by
//! `weight`; existing indices are NOT modified (search-expansion.ts:36-39).

use std::collections::HashMap;

use rusqlite::Connection;

use super::options::DEFAULT_MAX_EXPANDED_KEYWORDS;
use crate::sqlite::tag_queries::{get_keyword_baskets_for_tags, get_matching_tags};

// ---------------------------------------------------------------------------
// Public function
// ---------------------------------------------------------------------------

/// Expand a sparse vector with keywords from matching tag baskets.
///
/// Mirrors `expandSparseWithTags` in `search-expansion.ts:50-77`.
/// Returns the original sparse map unchanged on any error or if no keywords found.
///
/// The `generate_sparse` closure mirrors `daemonClient.generateSparseVector`.
pub async fn expand_sparse_with_tags<F, Fut>(
    generate_sparse: &mut F,
    conn: Option<&Connection>,
    query: &str,
    original: HashMap<u32, f32>,
    collections: &[String],
    weight: f64,
    max_keywords: usize,
    tenant_id: Option<&str>,
) -> HashMap<u32, f32>
where
    F: FnMut(String) -> Fut,
    Fut: std::future::Future<Output = Option<HashMap<u32, f32>>>,
{
    let keywords = collect_tag_keywords(conn, query, collections, tenant_id, max_keywords);
    if keywords.is_empty() {
        return original;
    }

    let expansion = generate_sparse(keywords.join(" ")).await;
    match expansion {
        Some(exp) if !exp.is_empty() => merge_sparse_vectors(&original, &exp, weight),
        _ => original,
    }
}

/// Collect unique expansion keywords from tag baskets across all collections.
///
/// Exposed as `pub` so `flow.rs` can call it before a separate daemon call,
/// avoiding the closure-captures-`daemon` borrow problem.
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

/// Merge expansion sparse vector into original at reduced weight (no-overwrite).
///
/// Mirrors `mergeSparseVectors` in `search-expansion.ts:29-39`:
/// Only adds indices NOT already in the original; existing indices unchanged.
pub fn merge_sparse_vectors(
    original: &HashMap<u32, f32>,
    expansion: &HashMap<u32, f32>,
    weight: f64,
) -> HashMap<u32, f32> {
    let mut merged = original.clone();
    for (&index, &value) in expansion {
        merged
            .entry(index)
            .or_insert_with(|| (value as f64 * weight) as f32);
    }
    merged
}
