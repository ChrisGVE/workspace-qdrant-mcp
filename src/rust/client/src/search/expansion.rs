//! SQLite-free sparse-vector merge for tag-basket expansion (WI-d4, #82).
//!
//! Mirrors `mergeSparseVectors` in `search-expansion.ts:29-39`.
//!
//! The SQLite-bound parts of tag expansion (`collect_expansion_keywords` /
//! `collect_tag_keywords`, reading the `tags` table) live in the MCP server's
//! expansion adapter, which pre-resolves the keyword baskets and passes the
//! joined keywords to the pipeline. Only the pure vector merge stays here.

use std::collections::HashMap;

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
