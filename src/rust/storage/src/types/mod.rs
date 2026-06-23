//! Shared domain types for the storage facade (read + write crates).
//!
//! Location: `wqm-storage/src/types/`. Logical context: the public type surface
//! (AC-F2.1/F2.2) both storage crates and their callers (`mcp-server`, `wqm-cli`,
//! daemon-core) speak. It has two parts:
//!
//!   * **Consumed from `wqm-common` (F0 canonical homes, NOT redefined here)** —
//!     [`StorageError`], [`SearchResult`], [`FileChange`], [`FileChangeStatus`].
//!     The facade's outward error is always `wqm_common::StorageError` (DR GP-9 —
//!     one error type, no parallel definition). The cross-crate identifier
//!     newtypes [`TenantId`]/[`BranchId`]/[`ContentKey`]/[`PointId`] are also
//!     re-exported from `wqm_common::domain` for one import path.
//!   * **Facade-specific, net-new here** — the request/result/stat DTOs the
//!     `ReadStoreFacade`/`WriteStoreFacade` methods exchange (§6). Split by
//!     responsibility into [`binding`], [`requests`], [`results`], and [`stats`]
//!     so no single file exceeds the arch §9 budget.
//!
//! Field sets follow the §6 facade method semantics; later features (the facade
//! impl) consume these types and may extend a struct additively as a method gains
//! detail — the names and roles here are the stable F2 vocabulary.

pub mod binding;
pub mod requests;
pub mod results;
pub mod stats;

// F0 canonical types — consumed, never redefined (AC-F2.1, DR GP-9).
pub use wqm_common::domain::{BranchId, ContentKey, PointId, TenantId};
pub use wqm_common::error::StorageError;
pub use wqm_common::git::file_change::{FileChange, FileChangeStatus};
pub use wqm_common::search::types::SearchResult;

pub use binding::ProjectBinding;
pub use requests::{ChunkInput, IngestFileRequest, SearchQuery};
pub use results::{FileEntry, FtsResult};
pub use stats::{BranchDeleteStats, BranchOnboardStats, IngestOutcome, RebuildStats};
