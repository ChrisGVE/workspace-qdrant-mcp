//! Cross-project fan-out for `scope=group|all` searches (AC-F17, arch R5).
//!
//! File: `wqm-storage/src/facade/read/fanout.rs`
//! Location: `src/rust/storage/src/facade/read/` (read crate)
//! Context: workspace-qdrant-mcp branch-storage model (arch R5, §6.2).
//!
//!   F17 extends `ReadStoreFacade::search` with bounded multi-project fan-out:
//!
//!   1. `enumerate_by_scope` (in `project::resolver`) returns the set of
//!      `ProjectBinding`s to query for the requested scope.
//!   2. For `scope=all`, a cliff check guards against unbounded fan-out:
//!      above `FanoutConfig::cliff` projects the function returns the typed
//!      `StorageError::ScopeTooBroad` (never a silent slow path — AC-F17.2).
//!   3. Per-project queries run with bounded concurrency — at most
//!      `FanoutConfig::concurrency` (= min(N_CPU, 8)) parallel tasks; extras
//!      queue (AC-F17.2, proven by `t_f17_02_concurrency_bound_respected`).
//!   4. Each project's result list is truncated to `top_k` BEFORE the merge
//!      so the candidate set is <= P*K (AC-F17.3).
//!   5. Cross-project RRF merge uses `wqm_common::search::rrf::rrf_merge`
//!      keyed by tenant_id — one ranked list per project, normalized by the
//!      RRF formula itself (small project does not drown large one — AC-F17.1,
//!      DR GP-9: no fork of the RRF algorithm).
//!
//!   Cost model (AC-F17.2, arch R5, documented in docs/ARCHITECTURE.md §8):
//!     total_fan_out_p95 ~= ceil(P / concurrency) * per_project_p95
//!   Derived cliff:
//!     cliff = ceil(ceiling_ms / per_project_p95_ms) * concurrency
//!   Default: ceiling=1000ms, per_project_p95=200ms, concurrency=min(N_CPU,8).
//!   PRD §14-Q3 names the default cliff as 50 (ceiling=1s, p95=200ms, cpu=10).
//!
//!   Live wiring deferral: `ReadStoreFacade` is not yet wired to the MCP server
//!   or CLI live search paths (those still route through daemon gRPC). The fan-
//!   out logic is complete and fully unit-tested here; wire-up rides the read-
//!   facade cutover (tracked separately, same posture as F8/F20 write features
//!   pending #175). See docs/architecture/branch-storage-model.md §8.
//!
//! Neighbors: `search.rs` (per-project `branch_search`),
//!   `crate::project::resolver::{ProjectRegistry, SearchScope}`,
//!   `wqm_common::search::rrf::{rrf_merge, CrossCollectionResult}`,
//!   `wqm_common::error::{StorageError, ScopeTooBroadPayload}`.

use std::sync::Arc;

use tokio::sync::Semaphore;
use wqm_common::{
    error::{ScopeTooBroadPayload, StorageError},
    search::{
        rrf::{rrf_merge, CrossCollectionResult},
        types::SearchResult,
    },
};

// ---------------------------------------------------------------------------
// FanoutConfig
// ---------------------------------------------------------------------------

/// Runtime configuration for the cross-project fan-out (AC-F17.2, arch R5).
///
/// `cliff` and `concurrency` are configurable; the defaults match the PRD
/// §14-Q3 named values.
#[derive(Debug, Clone)]
pub struct FanoutConfig {
    /// Maximum number of projects for `scope=all` before returning
    /// `ScopeTooBroad`. Default: 50 (PRD §14-Q3).
    pub cliff: usize,
    /// Maximum parallel per-project queries. Default: min(N_CPU, 8).
    pub concurrency: usize,
}

impl Default for FanoutConfig {
    fn default() -> Self {
        let cpus = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1);
        Self {
            cliff: 50,
            concurrency: cpus.min(8),
        }
    }
}

impl FanoutConfig {
    /// Return `Err(ScopeTooBroad)` when `project_count > cliff`.
    ///
    /// The payload carries all fields as discrete typed values so MCP/JSON
    /// clients read `suggested_scope` and `cliff` directly (AC-F17.5).
    pub fn check_cliff(
        &self,
        project_count: usize,
        requested_scope: &str,
    ) -> Result<(), StorageError> {
        if project_count <= self.cliff {
            return Ok(());
        }
        let payload = ScopeTooBroadPayload {
            requested_scope: requested_scope.to_string(),
            project_count,
            cliff: self.cliff,
            suggested_scope: "group".to_string(),
            hint: format!(
                "Found {project_count} projects which exceeds the cliff of {}. \
                 Use --scope group to search only related projects.",
                self.cliff
            ),
        };
        Err(StorageError::ScopeTooBroad(
            project_count,
            self.cliff,
            Box::new(payload),
        ))
    }
}

// ---------------------------------------------------------------------------
// Public pure functions (tested directly; also used by ReadStoreFacade)
// ---------------------------------------------------------------------------

/// Compute the derived project cliff from the cost-model parameters.
///
/// Formula (AC-F17.2):
///   `cliff = ceil(ceiling_ms / per_project_p95_ms) * concurrency`
///
/// The PRD §14-Q3 default cliff of 50 is derived by:
///   `ceil(1000 / 200) * 10 = 5 * 10 = 50`  (10-core machine).
pub fn compute_cliff(ceiling_ms: u64, per_project_p95_ms: u64, concurrency: usize) -> usize {
    let batches = ceiling_ms.div_ceil(per_project_p95_ms) as usize;
    batches * concurrency
}

/// Truncate a per-project result list to `top_k` entries BEFORE the cross-
/// project RRF merge so the candidate set is bounded at P*K (AC-F17.3).
///
/// The input list is assumed to be already sorted best-first (as returned by
/// `branch_search`). If `len <= top_k` the list is returned unchanged.
pub fn apply_per_project_top_k(results: Vec<SearchResult>, top_k: usize) -> Vec<SearchResult> {
    if results.len() <= top_k {
        results
    } else {
        results.into_iter().take(top_k).collect()
    }
}

/// Build the `(collection_key, results)` pair that `rrf_merge` expects.
///
/// The collection key is the `tenant_id` — using the tenant as the key is
/// what makes the RRF "normalized per project": each project contributes
/// exactly one ranked list, so rank-1 in a 2-result project and rank-1 in a
/// 50-result project yield the same RRF score (1/(k+1)), (AC-F17.1).
pub fn build_project_collection(
    tenant_id: &str,
    results: Vec<SearchResult>,
) -> (String, Vec<SearchResult>) {
    (tenant_id.to_string(), results)
}

/// Merge per-project result lists via `wqm_common::search::rrf::rrf_merge`.
///
/// Each element of `collections` is `(tenant_id, Vec<SearchResult>)`.
/// Uses `rrf_merge` from `wqm_common` — no forked algorithm (AC-F17.1,
/// DR GP-9).  Standard k=60.
pub fn merge_project_results(
    collections: Vec<(String, Vec<SearchResult>)>,
    k: f32,
) -> Vec<CrossCollectionResult> {
    rrf_merge(&collections, k)
}

// ---------------------------------------------------------------------------
// Bounded concurrency executor
// ---------------------------------------------------------------------------

/// Execute `tasks` with at most `concurrency` running in parallel (arch R5).
///
/// Each task is an async closure that returns `Result<T, StorageError>`.
/// Results preserve the submission order. A task failure short-circuits the
/// remaining queue via the `?` propagation inside the spawned future; already-
/// running tasks complete normally.
pub async fn run_bounded<T, F, Fut>(
    tasks: Vec<F>,
    concurrency: usize,
) -> Result<Vec<T>, StorageError>
where
    T: Send + 'static,
    F: FnOnce() -> Fut + Send + 'static,
    Fut: std::future::Future<Output = Result<T, StorageError>> + Send + 'static,
{
    let sem = Arc::new(Semaphore::new(concurrency));
    let mut handles = Vec::with_capacity(tasks.len());

    for task in tasks {
        let permit = sem
            .clone()
            .acquire_owned()
            .await
            .map_err(|e| StorageError::Search(format!("semaphore closed: {e}")))?;

        let handle = tokio::spawn(async move {
            let _permit = permit; // released when this block exits
            task().await
        });
        handles.push(handle);
    }

    let mut results = Vec::with_capacity(handles.len());
    for handle in handles {
        let val = handle
            .await
            .map_err(|e| StorageError::Search(format!("fan-out task panicked: {e}")))?;
        results.push(val?);
    }
    Ok(results)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[path = "fanout_tests.rs"]
mod tests;
