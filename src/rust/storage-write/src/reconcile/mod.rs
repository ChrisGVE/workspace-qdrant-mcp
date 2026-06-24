//! Reconcile pass -- periodic background + daemon-startup (arch §4.7, F15).
//!
//! File: `wqm-storage-write/src/reconcile/mod.rs`
//! Location: `src/rust/storage-write/src/reconcile/`
//! Context: Implements the five-case reconcile pass that corrects state drift
//!   from crashes or missed events (arch §4.7). Two run modes:
//!
//!   **Incremental (default):** watermark-scoped + migration-journal-bounded.
//!   Does NOT scan the whole corpus on every pass -- bounded by `max_seen_blob_id`
//!   from `maintenance_meta` (AC-F15.4 / PERF-03). A longer-cadence FULL pass
//!   catches anything the incremental scan misses.
//!
//!   **FULL/systematic (migration acceptance gate):** not watermark-bounded; all
//!   five cases over the whole corpus; verifies functional 1-to-1 coverage; reports
//!   a coverage delta (REPORTED, not FAILED) for non-checked-out branches.
//!   Idempotent on a second pass (arch §5.6 step 4).
//!
//! ## Case ordering (invariant)
//!
//! Case 5 MUST run before case 2. Without this ordering, a mis-tenanted point is
//! seen by case 2 as a zero-referrer orphan and wrongly culled (DATA-R7-04).
//! Additive-first (cases 1/4/5 add before cases 2/3 prune).
//!
//! ## #175 deferral
//!
//! The LIVE wiring -- periodic scheduler, daemon-startup hook, `wqm admin reconcile`
//! CLI surface, real git-ref reader, real Qdrant scroll client -- ALL RIDE #175.
//! This module provides the reconcile LOGIC and SEAMS; #175 assembles and wires
//! them. See `docs/ARCHITECTURE.md` (F15 Path note).
//!
//! Neighbors: [`super::blob`], [`super::branch`], [`super::qdrant`],
//!   [`super::schema`] (all reused without re-implementation, FP-2 / DR GP-1).

pub mod branches_sync;
pub mod case1;
pub mod case2;
pub mod case3;
pub mod case4;
pub mod case5;
pub mod full_mode;
pub mod seams;
pub mod watermark;

pub use branches_sync::{run_branches_sync, sync_branches_from_state, ProjectLocation};
pub use case5::TenantMismatchCandidate;
pub use full_mode::{run_full_mode, FullModeReport};
pub use seams::{GitRefReader, MockGitRefReader, MockQdrantReader, QdrantPointReader};
pub use watermark::{
    ensure_maintenance_meta, read_watermark, record_tenant_move, update_watermark,
    ReconcileWatermark, CREATE_MAINTENANCE_META,
};
