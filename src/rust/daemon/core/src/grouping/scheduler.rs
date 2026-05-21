//! Grouping scheduler -- unified coordinator for all project grouping strategies.
//!
//! Orchestrates the four grouping strategies in the correct order:
//!
//! 1. **Phase 1** (input data, independent):
//!    - Dependency-based (Jaccard on dep sets)
//!    - Workspace-based (Cargo/npm/Go workspaces)
//!    - Git-org-based (shared remote URL org)
//!
//! 2. **Phase 2** (derived from phase 1 data):
//!    - Tag-affinity-based (Jaccard on tag profiles / embedding similarity)
//!
//! Each strategy is tracked independently for staleness. A strategy only
//! reruns if its cooldown has expired since its last successful run.

use std::time::{Duration, Instant};

use sqlx::SqlitePool;
use tracing::{debug, info, warn};

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
}

impl GroupingScheduler {
    /// Create a new scheduler with default cooldowns (1 hour per strategy).
    pub fn new() -> Self {
        Self {
            dependency: StrategyState::new("dependency", DEFAULT_COOLDOWN_SECS),
            workspace: StrategyState::new("workspace", DEFAULT_COOLDOWN_SECS),
            git_org: StrategyState::new("git_org", DEFAULT_COOLDOWN_SECS),
            affinity: StrategyState::new("affinity", DEFAULT_COOLDOWN_SECS),
        }
    }

    /// Create a scheduler with custom cooldowns (in seconds).
    pub fn with_cooldowns(
        dependency_secs: u64,
        workspace_secs: u64,
        git_org_secs: u64,
        affinity_secs: u64,
    ) -> Self {
        Self {
            dependency: StrategyState::new("dependency", dependency_secs),
            workspace: StrategyState::new("workspace", workspace_secs),
            git_org: StrategyState::new("git_org", git_org_secs),
            affinity: StrategyState::new("affinity", affinity_secs),
        }
    }

    /// Returns true if any strategy needs to run.
    pub fn has_stale_strategies(&self) -> bool {
        self.dependency.is_stale()
            || self.workspace.is_stale()
            || self.git_org.is_stale()
            || self.affinity.is_stale()
    }

    /// Snapshot of last-run timestamps for observability.
    pub fn strategy_status(&self) -> Vec<(&str, Option<u64>)> {
        [
            &self.dependency,
            &self.workspace,
            &self.git_org,
            &self.affinity,
        ]
        .iter()
        .map(|s| (s.name, s.last_run.map(|t| t.elapsed().as_secs())))
        .collect()
    }

    /// Run all stale grouping strategies in the correct order.
    ///
    /// Phase 1 strategies (dependency, workspace, git_org) run first since
    /// they produce input data. Phase 2 (affinity) runs after because it
    /// derives from phase 1 outputs.
    ///
    /// Only strategies whose cooldown has expired are rerun. Failures in
    /// one strategy do not block others.
    pub async fn run_stale(&mut self, pool: &SqlitePool) -> GroupingResult {
        let mut result = GroupingResult {
            dependency_groups: None,
            workspace_groups: None,
            git_org_groups: None,
            affinity_groups: None,
            skipped_strategies: Vec::new(),
            failed_strategies: Vec::new(),
        };

        // -- Phase 1: input-data strategies (independent, can run in any order) --

        if self.dependency.is_stale() {
            match dependency::compute_dependency_groups(pool, None).await {
                Ok(groups) => {
                    result.dependency_groups = Some(groups);
                    self.dependency.mark_completed();
                    debug!(groups, "Dependency grouping complete");
                }
                Err(e) => {
                    let msg = e.to_string();
                    warn!(error = msg.as_str(), "Dependency grouping failed");
                    result.failed_strategies.push(("dependency".into(), msg));
                }
            }
        } else {
            result.skipped_strategies.push("dependency".into());
        }

        if self.workspace.is_stale() {
            match workspace::compute_workspace_groups(pool).await {
                Ok(groups) => {
                    result.workspace_groups = Some(groups);
                    self.workspace.mark_completed();
                    debug!(groups, "Workspace grouping complete");
                }
                Err(e) => {
                    let msg = e.to_string();
                    warn!(error = msg.as_str(), "Workspace grouping failed");
                    result.failed_strategies.push(("workspace".into(), msg));
                }
            }
        } else {
            result.skipped_strategies.push("workspace".into());
        }

        if self.git_org.is_stale() {
            match git_org::compute_git_org_groups(pool).await {
                Ok(groups) => {
                    result.git_org_groups = Some(groups);
                    self.git_org.mark_completed();
                    debug!(groups, "Git org grouping complete");
                }
                Err(e) => {
                    let msg = e.to_string();
                    warn!(error = msg.as_str(), "Git org grouping failed");
                    result.failed_strategies.push(("git_org".into(), msg));
                }
            }
        } else {
            result.skipped_strategies.push("git_org".into());
        }

        // -- Phase 2: derived strategies (depend on phase 1 data) --

        if self.affinity.is_stale() {
            let grouper = AffinityGrouper::new(pool.clone(), AffinityConfig::default());
            match grouper.compute_affinity_groups().await {
                Ok(groups) => {
                    result.affinity_groups = Some(groups);
                    self.affinity.mark_completed();
                    debug!(groups, "Affinity grouping complete");
                }
                Err(e) => {
                    let msg = e.to_string();
                    warn!(error = msg.as_str(), "Affinity grouping failed");
                    result.failed_strategies.push(("affinity".into(), msg));
                }
            }
        } else {
            result.skipped_strategies.push("affinity".into());
        }

        info!(
            total = result.total_groups(),
            dependency = ?result.dependency_groups,
            workspace = ?result.workspace_groups,
            git_org = ?result.git_org_groups,
            affinity = ?result.affinity_groups,
            skipped = result.skipped_strategies.len(),
            failed = result.failed_strategies.len(),
            "Grouping scheduler cycle complete"
        );

        result
    }
}

/// Compute all groups unconditionally (ignoring staleness).
///
/// Intended for startup reconciliation and manual rebuild.
/// Runs all four strategies in the correct order.
pub async fn compute_all_groups(pool: &SqlitePool) -> GroupingResult {
    let mut result = GroupingResult {
        dependency_groups: None,
        workspace_groups: None,
        git_org_groups: None,
        affinity_groups: None,
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
        assert_eq!(status.len(), 4);
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
            skipped_strategies: vec![],
            failed_strategies: vec![],
        };
        assert_eq!(result.total_groups(), 6);
    }

    #[test]
    fn test_grouping_result_total_all_none() {
        let result = GroupingResult {
            dependency_groups: None,
            workspace_groups: None,
            git_org_groups: None,
            affinity_groups: None,
            skipped_strategies: vec![],
            failed_strategies: vec![],
        };
        assert_eq!(result.total_groups(), 0);
    }

    #[test]
    fn test_scheduler_with_custom_cooldowns() {
        let scheduler = GroupingScheduler::with_cooldowns(60, 120, 180, 240);
        assert_eq!(scheduler.dependency.cooldown, Duration::from_secs(60));
        assert_eq!(scheduler.workspace.cooldown, Duration::from_secs(120));
        assert_eq!(scheduler.git_org.cooldown, Duration::from_secs(180));
        assert_eq!(scheduler.affinity.cooldown, Duration::from_secs(240));
    }

    #[tokio::test]
    async fn test_run_stale_on_empty_db() {
        let pool = setup_pool().await;

        // Create minimal required tables for dependency/workspace/git_org/affinity
        sqlx::query(
            "CREATE TABLE IF NOT EXISTS project_dependencies \
             (tenant_id TEXT, dep_name TEXT, dep_type TEXT, updated_at TEXT, \
              PRIMARY KEY (tenant_id, dep_name))",
        )
        .execute(&pool)
        .await
        .unwrap();

        sqlx::query(
            "CREATE TABLE IF NOT EXISTS watch_folders \
             (tenant_id TEXT PRIMARY KEY, folder_path TEXT, watch_type TEXT, \
              is_active INTEGER DEFAULT 1, last_activity_at TEXT, updated_at TEXT, \
              git_remote_url TEXT, name TEXT)",
        )
        .execute(&pool)
        .await
        .unwrap();

        sqlx::query(
            "CREATE TABLE IF NOT EXISTS project_embeddings \
             (tenant_id TEXT PRIMARY KEY, embedding BLOB, updated_at TEXT)",
        )
        .execute(&pool)
        .await
        .unwrap();

        sqlx::query(
            "CREATE TABLE IF NOT EXISTS affinity_labels \
             (group_id TEXT PRIMARY KEY, label TEXT, category TEXT, score REAL)",
        )
        .execute(&pool)
        .await
        .unwrap();

        let mut scheduler = GroupingScheduler::new();
        let result = scheduler.run_stale(&pool).await;

        // All strategies should have run (no skips), producing 0 groups on empty DB
        assert!(result.skipped_strategies.is_empty());
        assert!(result.failed_strategies.is_empty());
        assert_eq!(result.total_groups(), 0);

        // After running, no strategies should be stale
        assert!(!scheduler.has_stale_strategies());
    }

    #[tokio::test]
    async fn test_run_stale_skips_non_stale() {
        let pool = setup_pool().await;

        // Create required tables
        sqlx::query(
            "CREATE TABLE IF NOT EXISTS project_dependencies \
             (tenant_id TEXT, dep_name TEXT, dep_type TEXT, updated_at TEXT, \
              PRIMARY KEY (tenant_id, dep_name))",
        )
        .execute(&pool)
        .await
        .unwrap();
        sqlx::query(
            "CREATE TABLE IF NOT EXISTS watch_folders \
             (tenant_id TEXT PRIMARY KEY, folder_path TEXT, watch_type TEXT, \
              is_active INTEGER DEFAULT 1, last_activity_at TEXT, updated_at TEXT, \
              git_remote_url TEXT, name TEXT)",
        )
        .execute(&pool)
        .await
        .unwrap();
        sqlx::query(
            "CREATE TABLE IF NOT EXISTS project_embeddings \
             (tenant_id TEXT PRIMARY KEY, embedding BLOB, updated_at TEXT)",
        )
        .execute(&pool)
        .await
        .unwrap();
        sqlx::query(
            "CREATE TABLE IF NOT EXISTS affinity_labels \
             (group_id TEXT PRIMARY KEY, label TEXT, category TEXT, score REAL)",
        )
        .execute(&pool)
        .await
        .unwrap();

        let mut scheduler = GroupingScheduler::new();

        // First run: all strategies execute
        let result1 = scheduler.run_stale(&pool).await;
        assert!(result1.skipped_strategies.is_empty());

        // Second run immediately: all strategies skipped (cooldown not expired)
        let result2 = scheduler.run_stale(&pool).await;
        assert_eq!(result2.skipped_strategies.len(), 4);
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

        // Create required tables
        sqlx::query(
            "CREATE TABLE IF NOT EXISTS project_dependencies \
             (tenant_id TEXT, dep_name TEXT, dep_type TEXT, updated_at TEXT, \
              PRIMARY KEY (tenant_id, dep_name))",
        )
        .execute(&pool)
        .await
        .unwrap();
        sqlx::query(
            "CREATE TABLE IF NOT EXISTS watch_folders \
             (tenant_id TEXT PRIMARY KEY, folder_path TEXT, watch_type TEXT, \
              is_active INTEGER DEFAULT 1, last_activity_at TEXT, updated_at TEXT, \
              git_remote_url TEXT, name TEXT)",
        )
        .execute(&pool)
        .await
        .unwrap();
        sqlx::query(
            "CREATE TABLE IF NOT EXISTS project_embeddings \
             (tenant_id TEXT PRIMARY KEY, embedding BLOB, updated_at TEXT)",
        )
        .execute(&pool)
        .await
        .unwrap();
        sqlx::query(
            "CREATE TABLE IF NOT EXISTS affinity_labels \
             (group_id TEXT PRIMARY KEY, label TEXT, category TEXT, score REAL)",
        )
        .execute(&pool)
        .await
        .unwrap();

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
            vec!["dependency", "workspace", "git_org", "affinity"]
        );
    }
}
