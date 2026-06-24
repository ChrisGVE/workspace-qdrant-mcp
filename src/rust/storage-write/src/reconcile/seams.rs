//! Injectable seams for reconcile offline testing (AC-F15.1, #175 deferral).
//!
//! File: `wqm-storage-write/src/reconcile/seams.rs`
//! Location: `src/rust/storage-write/src/reconcile/`
//! Context: The reconcile pass needs two live capabilities that cannot be exercised
//!   in offline tests (no live Qdrant, no live git repo):
//!
//!   1. **`QdrantPointReader`** -- can a given `point_id` be found in Qdrant? Used by
//!      case-1 (re-upsert from durable vectors when the point is absent) and case-5
//!      (scroll candidate points sourced from the migration journal).
//!
//!   2. **`GitRefReader`** -- what branches still exist in the git topology? Used by
//!      case-3 to identify confirmed-deleted branches.
//!
//! Both are traits with an `async fn` (via `async_trait`). Offline tests supply mocks
//! that return canned responses without network or filesystem I/O. The LIVE wiring
//! (real Qdrant scroll + git2 call) RIDES #175 -- the daemon assembles the seams when
//! it wires the reconcile task.
//!
//! ## #175 deferral note
//!
//! The real implementations of both traits are NOT in this file. They live in the
//! daemon crate (`daemon/core` or `daemon/memexd`) and are assembled when the daemon
//! cutover task (#175) wires `wqm-storage-write` into `memexd`. This seam design
//! means #175 only has to implement the traits and pass them in -- it does NOT have to
//! redesign the reconcile logic.
//!
//! Neighbors: [`super::case1`] (uses `QdrantPointReader`), [`super::case3`] (uses
//!   `GitRefReader`), [`super::case5`] (uses `QdrantPointReader` for candidate scroll).

use async_trait::async_trait;

/// Can a Qdrant point be found in the collection?
///
/// The real implementation scrolls the collection via `QdrantReadClient::scroll`
/// (or a point-fetch by ID) to check existence. The offline mock returns a
/// pre-configured set of known point IDs.
///
/// RIDES #175 for the live implementation.
#[async_trait]
pub trait QdrantPointReader: Send + Sync {
    /// Return true if the point with `point_id` currently exists in Qdrant.
    async fn point_exists(&self, point_id: &str) -> bool;

    /// Return the `tenant_id` from the Qdrant payload for `point_id`, or None if
    /// the point does not exist or has no `tenant_id` field.
    ///
    /// Used by case-5 to read the current payload tenant and compare against the
    /// owning store's `store_meta.tenant_id`.
    async fn payload_tenant_id(&self, point_id: &str) -> Option<String>;
}

/// Which git branches still exist in the repository?
///
/// The real implementation calls `git2::Repository::branches` (or equivalent
/// `for-each-ref`) inside the daemon where a git repo path is known. The offline
/// mock returns a pre-configured set of live branch names.
///
/// RIDES #175 for the live implementation.
#[async_trait]
pub trait GitRefReader: Send + Sync {
    /// Return true if `branch_name` still exists in the git topology
    /// (i.e., `git for-each-ref` would return an entry for it).
    async fn branch_exists(&self, branch_name: &str) -> bool;
}

// ---------------------------------------------------------------------------
// Offline mocks (for tests -- NOT the live #175 implementations)
// ---------------------------------------------------------------------------

/// Mock `QdrantPointReader` that returns presence based on a pre-seeded set of
/// known point IDs. Points NOT in the set are treated as absent from Qdrant.
pub struct MockQdrantReader {
    /// Point IDs that "exist" in Qdrant (in the mock).
    pub existing_ids: std::collections::HashSet<String>,
    /// Per-point payload tenant_id overrides (for case-5 tests).
    pub payload_tenants: std::collections::HashMap<String, String>,
}

impl MockQdrantReader {
    /// All points are absent (Qdrant appears empty).
    pub fn all_absent() -> Self {
        Self {
            existing_ids: Default::default(),
            payload_tenants: Default::default(),
        }
    }

    /// Only the given `point_ids` exist.
    pub fn with_existing(point_ids: impl IntoIterator<Item = impl Into<String>>) -> Self {
        Self {
            existing_ids: point_ids.into_iter().map(|s| s.into()).collect(),
            payload_tenants: Default::default(),
        }
    }

    /// All points exist, with optional payload tenant overrides.
    pub fn all_present(
        point_ids: impl IntoIterator<Item = impl Into<String>>,
        payload_tenants: impl IntoIterator<Item = (impl Into<String>, impl Into<String>)>,
    ) -> Self {
        Self {
            existing_ids: point_ids.into_iter().map(|s| s.into()).collect(),
            payload_tenants: payload_tenants
                .into_iter()
                .map(|(k, v)| (k.into(), v.into()))
                .collect(),
        }
    }
}

#[async_trait]
impl QdrantPointReader for MockQdrantReader {
    async fn point_exists(&self, point_id: &str) -> bool {
        self.existing_ids.contains(point_id)
    }

    async fn payload_tenant_id(&self, point_id: &str) -> Option<String> {
        self.payload_tenants.get(point_id).cloned()
    }
}

/// Mock `GitRefReader` that treats a pre-seeded set of branch names as live.
pub struct MockGitRefReader {
    /// Branch names that "exist" in the git topology (in the mock).
    pub live_branches: std::collections::HashSet<String>,
}

impl MockGitRefReader {
    /// All branches provided are live.
    pub fn with_live(names: impl IntoIterator<Item = impl Into<String>>) -> Self {
        Self {
            live_branches: names.into_iter().map(|s| s.into()).collect(),
        }
    }

    /// No branches are live (all deleted).
    pub fn all_deleted() -> Self {
        Self {
            live_branches: Default::default(),
        }
    }
}

#[async_trait]
impl GitRefReader for MockGitRefReader {
    async fn branch_exists(&self, branch_name: &str) -> bool {
        self.live_branches.contains(branch_name)
    }
}
