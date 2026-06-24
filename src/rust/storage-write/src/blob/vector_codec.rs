//! Little-endian byte codec for dense and sparse vectors (arch §4.5, AC-F11).
//!
//! File: `wqm-storage-write/src/blob/vector_codec.rs`
//! Location: `src/rust/storage-write/src/blob/` (write-crate blob layer)
//! Context: The four encode/decode functions share a single responsibility --
//!   serialising Qdrant vector types to/from the compact byte columns
//!   `blobs.dense_vec` and `blobs.sparse_vec` in `store.db`. They are used by
//!   two callers: [`crate::blob::ladder`] (write path, ingest_miss) and
//!   [`crate::qdrant::recover`] (read path, rebuild_qdrant).
//!
//!   Splitting them out of `ladder.rs` keeps that module focused on the two
//!   write-cycle cases (hit / miss) and keeps this "codec" story self-contained
//!   (coding.md §X responsibility boundaries).
//!
//! Format spec (stable, little-endian):
//!   - dense:  `[f32_le; N]`  -- element count recoverable as `bytes.len() / 4`.
//!   - sparse: `u32_le count` followed by `count x (u32_le term, f32_le weight)`,
//!     terms sorted ascending so the encoding is deterministic for a given map.

use std::collections::HashMap;

/// Encode a dense vector as a little-endian `f32` byte array (4 bytes per element).
/// The element count is recoverable as `bytes.len() / 4`.
pub fn encode_dense(dense: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(dense.len() * 4);
    for v in dense {
        out.extend_from_slice(&v.to_le_bytes());
    }
    out
}

/// Encode a sparse vector as `u32_le count` followed by `count` x (`u32_le term`,
/// `f32_le weight`) pairs. Terms are sorted so the encoding is deterministic for a
/// given map (a `HashMap` has no inherent order).
pub fn encode_sparse(sparse: &HashMap<u32, f32>) -> Vec<u8> {
    let mut entries: Vec<(&u32, &f32)> = sparse.iter().collect();
    entries.sort_unstable_by_key(|(term, _)| **term);
    let mut out = Vec::with_capacity(4 + entries.len() * 8);
    out.extend_from_slice(&(entries.len() as u32).to_le_bytes());
    for (term, weight) in entries {
        out.extend_from_slice(&term.to_le_bytes());
        out.extend_from_slice(&weight.to_le_bytes());
    }
    out
}

/// Decode a dense vector from the little-endian `f32` byte array produced by
/// [`encode_dense`]. Returns an empty `Vec` for empty input. Panics if
/// `bytes.len()` is not a multiple of 4 (caller-enforced invariant: the SQLite
/// column stores exactly `encode_dense` output).
pub fn decode_dense(bytes: &[u8]) -> Vec<f32> {
    assert!(
        bytes.len() % 4 == 0,
        "decode_dense: byte length {} is not a multiple of 4 -- corrupted blobs.dense_vec",
        bytes.len()
    );
    bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
        .collect()
}

/// Decode a sparse vector from the byte format produced by [`encode_sparse`]:
/// `u32_le count` followed by `count` x (`u32_le term`, `f32_le weight`) pairs.
/// Returns an empty `HashMap` for empty input. Panics if the byte stream is
/// malformed (caller-enforced invariant: the SQLite column stores exactly
/// `encode_sparse` output).
pub fn decode_sparse(bytes: &[u8]) -> HashMap<u32, f32> {
    if bytes.is_empty() {
        return HashMap::new();
    }
    assert!(
        bytes.len() >= 4,
        "decode_sparse: too short ({} bytes) -- corrupted blobs.sparse_vec",
        bytes.len()
    );
    let count = u32::from_le_bytes(bytes[..4].try_into().unwrap()) as usize;
    let expected_len = 4 + count * 8;
    assert!(
        bytes.len() == expected_len,
        "decode_sparse: expected {} bytes for {count} entries, got {} -- corrupted blobs.sparse_vec",
        expected_len,
        bytes.len()
    );
    let mut map = HashMap::with_capacity(count);
    for i in 0..count {
        let base = 4 + i * 8;
        let term = u32::from_le_bytes(bytes[base..base + 4].try_into().unwrap());
        let weight = f32::from_le_bytes(bytes[base + 4..base + 8].try_into().unwrap());
        map.insert(term, weight);
    }
    map
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dense_encoding_round_trips_count() {
        let bytes = encode_dense(&[1.0, 2.5, -3.0]);
        assert_eq!(bytes.len(), 12, "3 f32 -> 12 bytes");
    }

    #[test]
    fn sparse_encoding_is_deterministic() {
        let mut m = HashMap::new();
        m.insert(7u32, 1.0f32);
        m.insert(3u32, 2.0f32);
        // Same map encodes identically regardless of insertion order.
        assert_eq!(encode_sparse(&m), encode_sparse(&m));
        // Count prefix = 2.
        assert_eq!(&encode_sparse(&m)[..4], &2u32.to_le_bytes());
    }
}
