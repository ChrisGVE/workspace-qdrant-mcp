//! Property-based tests for base_point and point_id uniqueness (Task 20)
//!
//! Verifies that compute_base_point and compute_point_id produce no collisions
//! across random input combinations using proptest.

use proptest::prelude::*;

// ============================================================================
// Property: base_point uniqueness
//
// With SHA256 truncated to 32 hex chars and varying inputs across tenant,
// branch, path, and hash dimensions, collisions should be astronomically
// unlikely.
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(500))]

    #[test]
    fn prop_base_point_no_collisions(
        tenant_ids in proptest::collection::vec("[a-z0-9_-]{4,16}", 3..6),
        branches in proptest::collection::vec("[a-z0-9/._-]{3,20}", 2..4),
        paths in proptest::collection::vec("[a-z0-9_/.-]{5,60}", 5..15),
        hashes in proptest::collection::vec("[a-f0-9]{8,64}", 3..8),
    ) {
        let mut seen = std::collections::HashSet::new();
        let mut collisions = 0u64;

        for tenant in &tenant_ids {
            for branch in &branches {
                for path in &paths {
                    for hash in &hashes {
                        let bp = wqm_common::hashing::compute_base_point(
                            tenant, branch, path, hash,
                        );
                        if !seen.insert(bp.clone()) {
                            collisions += 1;
                        }
                    }
                }
            }
        }

        // With SHA256 and varying inputs, collisions should be astronomically unlikely
        prop_assert_eq!(collisions, 0, "No base_point collisions should occur");
    }
}

// ============================================================================
// Property: point_id determinism
//
// Same base_point + chunk_index must always produce the same point_id.
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(5_000))]

    #[test]
    fn prop_point_id_deterministic(
        base_point in "[a-f0-9]{32}",
        chunk_index in 0u32..100,
    ) {
        let id1 = wqm_common::hashing::compute_point_id(&base_point, chunk_index);
        let id2 = wqm_common::hashing::compute_point_id(&base_point, chunk_index);
        prop_assert_eq!(id1, id2, "point_id must be deterministic");
    }
}

// ============================================================================
// Property: different chunks get different point_ids
//
// Given the same base_point, different chunk indices must produce different
// point_ids.
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(5_000))]

    #[test]
    fn prop_different_chunks_get_different_point_ids(
        base_point in "[a-f0-9]{32}",
        idx_a in 0u32..1000,
        idx_b in 0u32..1000,
    ) {
        if idx_a != idx_b {
            let id_a = wqm_common::hashing::compute_point_id(&base_point, idx_a);
            let id_b = wqm_common::hashing::compute_point_id(&base_point, idx_b);
            prop_assert_ne!(id_a, id_b,
                "Different chunk indices must produce different point IDs");
        }
    }
}
