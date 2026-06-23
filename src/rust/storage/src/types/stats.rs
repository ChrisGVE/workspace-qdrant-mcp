//! Facade outcome/stat DTOs — what the write methods report (§6.1).
//!
//! Location: `wqm-storage/src/types/stats.rs`. Logical context: the structured
//! return values of `ingest_file`/`branch_onboard`/`branch_delete`/`rebuild_qdrant`.
//! Each counts the work a call performed so callers can log progress and tests can
//! assert the dedup ladder and GC behaved. All derive `Default` so an empty result
//! (a no-op call) is `Stats::default()`.

use serde::{Deserialize, Serialize};

/// Result of [`ingest_file`](super) — the dedup-ladder outcome for one file (§4.1).
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct IngestOutcome {
    /// Chunks processed for the file.
    pub chunks_ingested: usize,
    /// Chunk-blobs newly embedded and stored (`content_key` misses).
    pub blobs_created: usize,
    /// Chunk-blobs already present, reused via membership only (`content_key` hits).
    pub blobs_reused: usize,
}

/// Result of `branch_onboard` — onboarding a previously unknown branch (§4.2).
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct BranchOnboardStats {
    /// Files that differed from the prior state and were ingested.
    pub files_changed: usize,
    /// Total chunks ingested across those files.
    pub chunks_ingested: usize,
    /// Chunk-blobs newly embedded and stored.
    pub blobs_created: usize,
    /// Chunk-blobs reused via membership only.
    pub blobs_reused: usize,
}

/// Result of `branch_delete` — membership removal + orphan blob GC (§4.3).
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct BranchDeleteStats {
    /// Blobs garbage-collected (no remaining referrer after the branch dropped).
    pub blobs_gc: usize,
    /// `files` rows removed for the deleted branch.
    pub files_removed: usize,
    /// `fts_branch_membership` / `blob_refs` membership rows removed.
    pub memberships_removed: usize,
}

/// Result of `rebuild_qdrant` — re-projecting SQLite durable vectors to Qdrant.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct RebuildStats {
    /// Qdrant points re-upserted from durable vectors (no embedding API calls).
    pub points_rebuilt: usize,
    /// Blobs scanned while rebuilding.
    pub blobs_scanned: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_are_zeroed() {
        assert_eq!(IngestOutcome::default().chunks_ingested, 0);
        assert_eq!(BranchOnboardStats::default().files_changed, 0);
        assert_eq!(BranchDeleteStats::default().blobs_gc, 0);
        assert_eq!(RebuildStats::default().points_rebuilt, 0);
    }

    #[test]
    fn ingest_outcome_serde_round_trip() {
        let o = IngestOutcome {
            chunks_ingested: 5,
            blobs_created: 2,
            blobs_reused: 3,
        };
        let json = serde_json::to_string(&o).unwrap();
        let back: IngestOutcome = serde_json::from_str(&json).unwrap();
        assert_eq!(back, o);
        assert_eq!(back.blobs_created + back.blobs_reused, 5);
    }
}
