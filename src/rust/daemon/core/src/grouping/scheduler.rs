//! Grouping scheduler -- unified coordinator for all project grouping strategies.
//!
//! Orchestrates the five grouping strategies in the correct order:
//!
//! 1. **Phase 1** (input data, independent):
//!    - Dependency-based (Jaccard on dep sets)
//!    - Workspace-based (Cargo/npm/Go workspaces)
//!    - Git-org-based (shared remote URL org)
//!
//! 2. **Phase 2** (derived from phase 1 data):
//!    - Embedding-affinity-based (cosine similarity on aggregate embeddings)
//!    - Tag-affinity-based (Jaccard on tag profiles)
//!
//! Each strategy is tracked independently for staleness. A strategy only
//! reruns if its cooldown has expired since its last successful run.

use std::sync::Arc;
use std::time::{Duration, Instant};

use sqlx::SqlitePool;
use tracing::{debug, info, warn};

use super::affinity::tag_affinity::{compute_tag_affinity_groups, TagAffinityConfig};
use super::affinity::{AffinityConfig, AffinityGrouper};
use super::{dependency, git_org, workspace};

/// Default cooldown between full recomputation cycles per strategy (1 hour).
const DEFAULT_COOLDOWN_SECS: u64 = 3600;

/// Result of running all grouping strategies.
#[derive(Debug, Clone)]
pub struct GroupingResult {
    pub dependency_groups: Option<usize>,
    pub workspace_groups: Option<usize>,
    pub git_org_groups: Option<usize>,
    pub affinity_groups: Option<usize>,
    pub tag_affinity_groups: Option<usize>,
    pub skipped_strategies: Vec<String>,
    pub failed_strategies: Vec<(String, String)>,
}

impl GroupingResult {
    /// Total number of groups created across all strategies.
    pub fn total_groups(&self) -> usize {
        self.dependency_groups.unwrap_or(0)
            + self.workspace_groups.unwrap_or(0)
            + self.git_org_groups.unwrap_or(0)
            + self.affinity_groups.unwrap_or(0)
            + self.tag_affinity_groups.unwrap_or(0)
    }
}

/// Per-strategy staleness tracking.
#[derive(Debug)]
struct StrategyState {
    name: &'static str,
    last_run: Option<Instant>,
    cooldown: Duration,
}

impl StrategyState {
    fn new(name: &'static str, cooldown_secs: u64) -> Self {
        Self {
            name,
            last_run: None,
            cooldown: Duration::from_secs(cooldown_secs),
        }
    }

    /// Returns `true` if this strategy is stale and should be rerun.
    fn is_stale(&self) -> bool {
        match self.last_run {
            None => true,
            Some(t) => t.elapsed() >= self.cooldown,
        }
    }

    fn mark_completed(&mut self) {
        self.last_run = Some(Instant::now());
    }
}

/// Unified grouping scheduler that coordinates all grouping strategies.
pub struct GroupingScheduler {
    dependency: StrategyState,
    workspace: StrategyState,
    git_org: StrategyState,
    affinity: StrategyState,
    tag_affinity: StrategyState,
}

impl GroupingScheduler {
    /// Create a new scheduler with default cooldowns (1 hour per strategy).
    pub fn new() -> Self {
        Self {
            dependency: StrategyState::new("dependency", DEFAULT_COOLDOWN_SECS),
            workspace: StrategyState::new("workspace", DEFAULT_COOLDOWN_SECS),
            git_org: StrategyState::new("git_org", DEFAULT_COOLDOWN_SECS),
            affinity: StrategyState::new("affinity", DEFAULT_COOLDOWN_SECS),
            tag_affinity: StrategyState::new("tag_affinity", DEFAULT_COOLDOWN_SECS),
        }
    }

    /// Returns true if any strategy needs to run.
    pub fn has_stale_strategies(&self) -> bool {
        self.dependency.is_stale()
            || self.workspace.is_stale()
            || self.git_org.is_stale()
            || self.affinity.is_stale()
            || self.tag_affinity.is_stale()
    }

    /// Snapshot of last-run timestamps for observability.
    ///
    /// Returns pairs of (strategy_name, seconds_since_last_run).
    /// The order matches execution order: phase 1 then phase 2.
    pub fn strategy_status(&self) -> Vec<(&str, Option<u64>)> {
        [
            &self.dependency,
            &self.workspace,
            &self.git_org,
            &self.affinity,
            &self.tag_affinity,
        ]
        .iter()
        .map(|s| (s.name, s.last_run.map(|t| t.elapsed().as_secs())))
        .collect()
    }

    /// Run all stale grouping strategies in the correct order.
    ///
    /// Phase 1 strategies (dependency, workspace, git_org) run first since
    /// they produce input data. Phase 1.5 (bootstrap) seeds aggregate
    /// embeddings for projects missing them. Phase 2 (affinity, tag_affinity)
    /// runs after because they derive from phase 1 outputs.
    ///
    /// Only strategies whose cooldown has expired are rerun. Failures in
    /// one strategy do not block others.
    pub async fn run_stale(&mut self, pool: &SqlitePool) -> GroupingResult {
        self.run_stale_with_storage(pool, None).await
    }

    /// Run all stale strategies with optional Qdrant access for bootstrap.
    pub async fn run_stale_with_storage(
        &mut self,
        pool: &SqlitePool,
        storage_client: Option<&Arc<crate::storage::StorageClient>>,
    ) -> GroupingResult {
        let mut result = GroupingResult {
            dependency_groups: None,
            workspace_groups: None,
            git_org_groups: None,
            affinity_groups: None,
            tag_affinity_groups: None,
            skipped_strategies: Vec::new(),
            failed_strategies: Vec::new(),
        };

        // -- Phase 1: input-data strategies (independent, can run in any order) --

        run_dependency(&mut self.dependency, pool, &mut result).await;
        run_workspace(&mut self.workspace, pool, &mut result).await;
        run_git_org(&mut self.git_org, pool, &mut result).await;

        // -- Phase 1.5: bootstrap aggregate embeddings for new projects --
        if let Some(sc) = storage_client {
            let boot = super::affinity::bootstrap_missing_embeddings(pool, sc).await;
            if boot.bootstrapped > 0 {
                info!(
                    bootstrapped = boot.bootstrapped,
                    empty = boot.empty,
                    "Bootstrap seeded aggregate embeddings"
                );
            }
        }

        // -- Phase 2: derived strategies (depend on phase 1 data) --

        run_affinity(&mut self.affinity, pool, &mut result).await;
        run_tag_affinity(&mut self.tag_affinity, pool, &mut result).await;

        info!(
            total = result.total_groups(),
            dependency = ?result.dependency_groups,
            workspace = ?result.workspace_groups,
            git_org = ?result.git_org_groups,
            affinity = ?result.affinity_groups,
            tag_affinity = ?result.tag_affinity_groups,
            skipped = result.skipped_strategies.len(),
            failed = result.failed_strategies.len(),
            "Grouping scheduler cycle complete"
        );

        result
    }
}

// ---- Strategy runners (extracted to keep run_stale concise) ----------------

async fn run_dependency(state: &mut StrategyState, pool: &SqlitePool, result: &mut GroupingResult) {
    if !state.is_stale() {
        result.skipped_strategies.push("dependency".into());
        return;
    }
    match dependency::compute_dependency_groups(pool, None).await {
        Ok(groups) => {
            result.dependency_groups = Some(groups);
            state.mark_completed();
            debug!(groups, "Dependency grouping complete");
        }
        Err(e) => {
            let msg = e.to_string();
            warn!(error = msg.as_str(), "Dependency grouping failed");
            result.failed_strategies.push(("dependency".into(), msg));
        }
    }
}

async fn run_workspace(state: &mut StrategyState, pool: &SqlitePool, result: &mut GroupingResult) {
    if !state.is_stale() {
        result.skipped_strategies.push("workspace".into());
        return;
    }
    match workspace::compute_workspace_groups(pool).await {
        Ok(groups) => {
            result.workspace_groups = Some(groups);
            state.mark_completed();
            debug!(groups, "Workspace grouping complete");
        }
        Err(e) => {
            let msg = e.to_string();
            warn!(error = msg.as_str(), "Workspace grouping failed");
            result.failed_strategies.push(("workspace".into(), msg));
        }
    }
}

async fn run_git_org(state: &mut StrategyState, pool: &SqlitePool, result: &mut GroupingResult) {
    if !state.is_stale() {
        result.skipped_strategies.push("git_org".into());
        return;
    }
    match git_org::compute_git_org_groups(pool).await {
        Ok(groups) => {
            result.git_org_groups = Some(groups);
            state.mark_completed();
            debug!(groups, "Git org grouping complete");
        }
        Err(e) => {
            let msg = e.to_string();
            warn!(error = msg.as_str(), "Git org grouping failed");
            result.failed_strategies.push(("git_org".into(), msg));
        }
    }
}

async fn run_affinity(state: &mut StrategyState, pool: &SqlitePool, result: &mut GroupingResult) {
    if !state.is_stale() {
        result.skipped_strategies.push("affinity".into());
        return;
    }
    let grouper = AffinityGrouper::new(pool.clone(), AffinityConfig::default());
    match grouper.compute_affinity_groups().await {
        Ok(groups) => {
            result.affinity_groups = Some(groups);
            state.mark_completed();
            debug!(groups, "Affinity grouping complete");
        }
        Err(e) => {
            let msg = e.to_string();
            warn!(error = msg.as_str(), "Affinity grouping failed");
            result.failed_strategies.push(("affinity".into(), msg));
        }
    }
}

async fn run_tag_affinity(
    state: &mut StrategyState,
    pool: &SqlitePool,
    result: &mut GroupingResult,
) {
    if !state.is_stale() {
        result.skipped_strategies.push("tag_affinity".into());
        return;
    }
    let config = TagAffinityConfig::default();
    match compute_tag_affinity_groups(pool, &config).await {
        Ok(groups) => {
            result.tag_affinity_groups = Some(groups);
            state.mark_completed();
            debug!(groups, "Tag affinity grouping complete");
        }
        Err(e) => {
            let msg = e.to_string();
            warn!(error = msg.as_str(), "Tag affinity grouping failed");
            result.failed_strategies.push(("tag_affinity".into(), msg));
        }
    }
}

/// Compute all groups unconditionally (ignoring staleness).
///
/// Intended for startup reconciliation and manual rebuild.
/// Runs all five strategies in the correct order.
pub async fn compute_all_groups(pool: &SqlitePool) -> GroupingResult {
    let mut result = GroupingResult {
        dependency_groups: None,
        workspace_groups: None,
        git_org_groups: None,
        affinity_groups: None,
        tag_affinity_groups: None,
        skipped_strategies: Vec::new(),
        failed_strategies: Vec::new(),
    };

    // Phase 1: input data strategies
    match dependency::compute_dependency_groups(pool, None).await {
        Ok(groups) => {
            result.dependency_groups = Some(groups);
            debug!(groups, "Startup: dependency grouping complete");
        }
        Err(e) => {
            let msg = e.to_string();
            warn!(error = msg.as_str(), "Startup: dependency grouping failed");
            result.failed_strategies.push(("dependency".into(), msg));
        }
    }

    match workspace::compute_workspace_groups(pool).await {
        Ok(groups) => {
            result.workspace_groups = Some(groups);
            debug!(groups, "Startup: workspace grouping complete");
        }
        Err(e) => {
            let msg = e.to_string();
            warn!(error = msg.as_str(), "Startup: workspace grouping failed");
            result.failed_strategies.push(("workspace".into(), msg));
        }
    }

    match git_org::compute_git_org_groups(pool).await {
        Ok(groups) => {
            result.git_org_groups = Some(groups);
            debug!(groups, "Startup: git org grouping complete");
        }
        Err(e) => {
            let msg = e.to_string();
            warn!(error = msg.as_str(), "Startup: git org grouping failed");
            result.failed_strategies.push(("git_org".into(), msg));
        }
    }

    // Phase 2: derived strategies
    let grouper = AffinityGrouper::new(pool.clone(), AffinityConfig::default());
    match grouper.compute_affinity_groups().await {
        Ok(groups) => {
            result.affinity_groups = Some(groups);
            debug!(groups, "Startup: affinity grouping complete");
        }
        Err(e) => {
            let msg = e.to_string();
            warn!(error = msg.as_str(), "Startup: affinity grouping failed");
            result.failed_strategies.push(("affinity".into(), msg));
        }
    }

    let tag_config = TagAffinityConfig::default();
    match compute_tag_affinity_groups(pool, &tag_config).await {
        Ok(groups) => {
            result.tag_affinity_groups = Some(groups);
            debug!(groups, "Startup: tag affinity grouping complete");
        }
        Err(e) => {
            let msg = e.to_string();
            warn!(
                error = msg.as_str(),
                "Startup: tag affinity grouping failed"
            );
            result.failed_strategies.push(("tag_affinity".into(), msg));
        }
    }

    info!(
        total = result.total_groups(),
        failed = result.failed_strategies.len(),
        "Startup group computation complete"
    );

    result
}

/// List all group memberships, optionally filtered by tenant.
///
/// Returns tuples of (group_id, tenant_id, group_type, confidence).
pub async fn list_all_groups(
    pool: &SqlitePool,
    tenant_filter: Option<&str>,
) -> Result<Vec<(String, String, String, f64)>, sqlx::Error> {
    let rows = if let Some(tenant) = tenant_filter {
        sqlx::query_as::<_, (String, String, String, f64)>(
            r#"
            SELECT group_id, tenant_id, group_type, confidence
            FROM project_groups
            WHERE tenant_id = ?
            ORDER BY group_type, group_id, tenant_id
            "#,
        )
        .bind(tenant)
        .fetch_all(pool)
        .await?
    } else {
        sqlx::query_as::<_, (String, String, String, f64)>(
            r#"
            SELECT group_id, tenant_id, group_type, confidence
            FROM project_groups
            ORDER BY group_type, group_id, tenant_id
            "#,
        )
        .fetch_all(pool)
        .await?
    };

    Ok(rows)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grouping::schema::{
        add_to_group, CREATE_PROJECT_GROUPS_INDEXES_SQL, CREATE_PROJECT_GROUPS_SQL,
    };
    use sqlx::sqlite::SqlitePoolOptions;

    /// Number of grouping strategies tracked by the scheduler.
    const STRATEGY_COUNT: usize = 5;

    async fn setup_pool() -> SqlitePool {
        let pool = SqlitePoolOptions::new()
            .max_connections(1)
            .connect("sqlite::memory:")
            .await
            .unwrap();

        sqlx::query(CREATE_PROJECT_GROUPS_SQL)
            .execute(&pool)
            .await
            .unwrap();

        for idx_sql in CREATE_PROJECT_GROUPS_INDEXES_SQL {
            sqlx::query(idx_sql).execute(&pool).await.unwrap();
        }

        pool
    }

    async fn create_all_strategy_tables(pool: &SqlitePool) {
        sqlx::query(
            "CREATE TABLE IF NOT EXISTS project_dependencies \
             (tenant_id TEXT, dependency_name TEXT, dep_type TEXT, updated_at TEXT, \
              PRIMARY KEY (tenant_id, dependency_name))",
        )
        .execute(pool)
        .await
        .unwrap();

        // Real watch_folders schema — fixtures must not drift from production
        // (a hand-rolled folder_path/watch_type table masked a column-name bug).
        for stmt in include_str!("../schema/watch_folders_schema.sql").split(';') {
            let stmt = stmt.trim();
            if !stmt.is_empty() {
                sqlx::query(stmt).execute(pool).await.unwrap();
            }
        }

        sqlx::query(
            "CREATE TABLE IF NOT EXISTS project_embeddings \
             (tenant_id TEXT PRIMARY KEY, embedding BLOB NOT NULL, \
              dim INTEGER NOT NULL, chunk_count INTEGER NOT NULL DEFAULT 0, \
              label TEXT, updated_at TEXT NOT NULL)",
        )
        .execute(pool)
        .await
        .unwrap();

        sqlx::query(
            "CREATE TABLE IF NOT EXISTS affinity_labels \
             (group_id TEXT PRIMARY KEY, label TEXT NOT NULL, \
              category TEXT NOT NULL, score REAL NOT NULL, updated_at TEXT NOT NULL)",
        )
        .execute(pool)
        .await
        .unwrap();

        sqlx::query(
            "CREATE TABLE IF NOT EXISTS tags \
             (tag_id INTEGER PRIMARY KEY AUTOINCREMENT, \
              doc_id TEXT NOT NULL, tag TEXT NOT NULL, \
              tag_type TEXT NOT NULL DEFAULT 'concept', \
              score REAL NOT NULL DEFAULT 0.0, \
              diversity_score REAL NOT NULL DEFAULT 0.0, \
              basket_id INTEGER, collection TEXT NOT NULL, \
              tenant_id TEXT NOT NULL, created_at TEXT NOT NULL DEFAULT '')",
        )
        .execute(pool)
        .await
        .unwrap();

        sqlx::query(
            "CREATE TABLE IF NOT EXISTS tracked_files \
             (file_id TEXT PRIMARY KEY, tenant_id TEXT, file_path TEXT, status TEXT)",
        )
        .execute(pool)
        .await
        .unwrap();
    }

    #[test]
    fn test_strategy_state_initially_stale() {
        let state = StrategyState::new("test", 3600);
        assert!(state.is_stale());
    }

    #[test]
    fn test_strategy_state_not_stale_after_mark() {
        let mut state = StrategyState::new("test", 3600);
        state.mark_completed();
        assert!(!state.is_stale());
    }

    #[test]
    fn test_strategy_state_stale_after_cooldown() {
        let mut state = StrategyState::new("test", 0);
        state.mark_completed();
        // With 0-second cooldown, should be immediately stale
        assert!(state.is_stale());
    }

    #[test]
    fn test_scheduler_new_all_stale() {
        let scheduler = GroupingScheduler::new();
        assert!(scheduler.has_stale_strategies());
    }

    #[test]
    fn test_scheduler_status_initially_none() {
        let scheduler = GroupingScheduler::new();
        let status = scheduler.strategy_status();
        assert_eq!(status.len(), STRATEGY_COUNT);
        for (_, last_run) in &status {
            assert!(last_run.is_none());
        }
    }

    #[test]
    fn test_grouping_result_total() {
        let result = GroupingResult {
            dependency_groups: Some(2),
            workspace_groups: Some(1),
            git_org_groups: Some(3),
            affinity_groups: None,
            tag_affinity_groups: Some(1),
            skipped_strategies: vec![],
            failed_strategies: vec![],
        };
        assert_eq!(result.total_groups(), 7);
    }

    #[test]
    fn test_grouping_result_total_all_none() {
        let result = GroupingResult {
            dependency_groups: None,
            workspace_groups: None,
            git_org_groups: None,
            affinity_groups: None,
            tag_affinity_groups: None,
            skipped_strategies: vec![],
            failed_strategies: vec![],
        };
        assert_eq!(result.total_groups(), 0);
    }

    #[tokio::test]
    async fn test_run_stale_on_empty_db() {
        let pool = setup_pool().await;
        create_all_strategy_tables(&pool).await;

        let mut scheduler = GroupingScheduler::new();
        let result = scheduler.run_stale(&pool).await;

        // All strategies should have run (no skips), producing 0 groups on empty DB
        assert!(result.skipped_strategies.is_empty());
        assert!(
            result.failed_strategies.is_empty(),
            "Failed strategies: {:?}",
            result.failed_strategies
        );
        assert_eq!(result.total_groups(), 0);

        // After running, no strategies should be stale
        assert!(!scheduler.has_stale_strategies());
    }

    #[tokio::test]
    async fn test_run_stale_skips_non_stale() {
        let pool = setup_pool().await;
        create_all_strategy_tables(&pool).await;

        let mut scheduler = GroupingScheduler::new();

        // First run: all strategies execute
        let result1 = scheduler.run_stale(&pool).await;
        assert!(result1.skipped_strategies.is_empty());

        // Second run immediately: all strategies skipped (cooldown not expired)
        let result2 = scheduler.run_stale(&pool).await;
        assert_eq!(result2.skipped_strategies.len(), STRATEGY_COUNT);
        assert_eq!(result2.total_groups(), 0);
    }

    #[tokio::test]
    async fn test_list_all_groups_empty() {
        let pool = setup_pool().await;
        let groups = list_all_groups(&pool, None).await.unwrap();
        assert!(groups.is_empty());
    }

    #[tokio::test]
    async fn test_list_all_groups_with_data() {
        let pool = setup_pool().await;

        add_to_group(&pool, "grp-dep-1", "proj-a", "dependency", 0.8)
            .await
            .unwrap();
        add_to_group(&pool, "grp-dep-1", "proj-b", "dependency", 0.8)
            .await
            .unwrap();
        add_to_group(&pool, "grp-ws-1", "proj-a", "workspace", 1.0)
            .await
            .unwrap();

        let all = list_all_groups(&pool, None).await.unwrap();
        assert_eq!(all.len(), 3);

        let filtered = list_all_groups(&pool, Some("proj-a")).await.unwrap();
        assert_eq!(filtered.len(), 2);

        let filtered_b = list_all_groups(&pool, Some("proj-b")).await.unwrap();
        assert_eq!(filtered_b.len(), 1);
    }

    #[tokio::test]
    async fn test_list_all_groups_tenant_filter_no_match() {
        let pool = setup_pool().await;

        add_to_group(&pool, "grp-1", "proj-a", "dependency", 0.8)
            .await
            .unwrap();

        let filtered = list_all_groups(&pool, Some("proj-nonexistent"))
            .await
            .unwrap();
        assert!(filtered.is_empty());
    }

    #[tokio::test]
    async fn test_compute_all_groups_on_empty_db() {
        let pool = setup_pool().await;
        create_all_strategy_tables(&pool).await;

        let result = compute_all_groups(&pool).await;
        assert!(result.failed_strategies.is_empty());
        assert_eq!(result.total_groups(), 0);
    }

    #[test]
    fn test_scheduler_ordering_phase1_before_phase2() {
        // Verify the strategy names match the expected ordering
        let scheduler = GroupingScheduler::new();
        let status = scheduler.strategy_status();

        let names: Vec<&str> = status.iter().map(|(n, _)| *n).collect();
        assert_eq!(
            names,
            vec![
                "dependency",
                "workspace",
                "git_org",
                "affinity",
                "tag_affinity"
            ]
        );
    }
}
