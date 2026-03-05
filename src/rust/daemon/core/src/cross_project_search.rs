/// Cross-project search with scope-based tenant filtering.
///
/// Supports three search scopes:
/// - `Project`: Filter by the caller's tenant_id (default, existing behavior).
/// - `Group`: Search across all projects sharing a group with the caller.
/// - `All`: No tenant filter — search the entire collection.
///
/// Results from non-current projects are subject to relevance decay
/// to prioritize local matches while still surfacing cross-project context.
use serde::{Deserialize, Serialize};
use sqlx::SqlitePool;
use tracing::debug;

use crate::project_groups_schema;
use crate::storage::SearchResult;

// ─── Scope enum ─────────────────────────────────────────────────────────

/// Search scope controlling tenant filtering.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum SearchScope {
    /// Search only the current project (default).
    Project,
    /// Search the current project + grouped projects.
    Group,
    /// Search all projects, no tenant filter.
    All,
}

impl Default for SearchScope {
    fn default() -> Self {
        Self::Project
    }
}

impl SearchScope {
    /// Parse from string (case-insensitive).
    pub fn from_str_loose(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "group" => Self::Group,
            "all" => Self::All,
            _ => Self::Project,
        }
    }
}

// ─── Relevance decay ────────────────────────────────────────────────────

/// Relevance decay multipliers for cross-project results.
#[derive(Debug, Clone)]
pub struct RelevanceDecay {
    /// Multiplier for results from the current project (default: 1.0).
    pub current_project: f32,
    /// Multiplier for results from grouped projects (default: 0.7).
    pub group_project: f32,
    /// Multiplier for results from other projects (default: 0.4).
    pub other_project: f32,
}

impl Default for RelevanceDecay {
    fn default() -> Self {
        Self {
            current_project: 1.0,
            group_project: 0.7,
            other_project: 0.4,
        }
    }
}

// ─── Scope resolution ───────────────────────────────────────────────────

/// Resolved tenant filter based on scope.
#[derive(Debug, Clone)]
pub enum TenantFilter {
    /// Filter to a single tenant_id.
    Single(String),
    /// Filter to a set of tenant_ids (IN clause).
    Multiple(Vec<String>),
    /// No filter — search all tenants.
    None,
}

/// Resolve the search scope into a concrete tenant filter.
///
/// For `Group` scope, queries the `project_groups` table to find all
/// tenant_ids that share a group with the current tenant. Falls back
/// to `Single` if the tenant has no group memberships.
pub async fn resolve_scope(
    scope: SearchScope,
    current_tenant_id: &str,
    pool: &SqlitePool,
) -> TenantFilter {
    match scope {
        SearchScope::Project => TenantFilter::Single(current_tenant_id.to_string()),
        SearchScope::All => TenantFilter::None,
        SearchScope::Group => {
            match project_groups_schema::get_group_members(pool, current_tenant_id).await {
                Ok(members) if members.len() > 1 => {
                    debug!(
                        tenant = current_tenant_id,
                        group_members = members.len(),
                        "Resolved group scope"
                    );
                    TenantFilter::Multiple(members)
                }
                Ok(_) => {
                    // No groups or only self — fall back to single
                    debug!(
                        tenant = current_tenant_id,
                        "No group members found, falling back to project scope"
                    );
                    TenantFilter::Single(current_tenant_id.to_string())
                }
                Err(e) => {
                    debug!(
                        tenant = current_tenant_id,
                        error = %e,
                        "Failed to resolve group members, falling back to project scope"
                    );
                    TenantFilter::Single(current_tenant_id.to_string())
                }
            }
        }
    }
}

/// Apply relevance decay to search results based on tenant ownership.
///
/// Results from the current project keep their score unchanged.
/// Grouped project results are multiplied by `decay.group_project`.
/// All other results are multiplied by `decay.other_project`.
/// Results are re-sorted by adjusted score.
pub fn apply_relevance_decay(
    results: &mut Vec<SearchResult>,
    current_tenant_id: &str,
    group_tenant_ids: &[String],
    decay: &RelevanceDecay,
) {
    for result in results.iter_mut() {
        let tenant = result
            .payload
            .get("tenant_id")
            .and_then(|v| v.as_str())
            .unwrap_or("");

        let multiplier = if tenant == current_tenant_id {
            decay.current_project
        } else if group_tenant_ids.iter().any(|t| t == tenant) {
            decay.group_project
        } else {
            decay.other_project
        };

        result.score *= multiplier;
    }

    // Re-sort by decayed score (descending)
    results.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn make_result(id: &str, score: f32, tenant: &str) -> SearchResult {
        let mut payload = HashMap::new();
        payload.insert("tenant_id".to_string(), serde_json::json!(tenant));

        SearchResult {
            id: id.to_string(),
            score,
            payload,
            dense_vector: None,
            sparse_vector: None,
        }
    }

    #[test]
    fn test_search_scope_default() {
        assert_eq!(SearchScope::default(), SearchScope::Project);
    }

    #[test]
    fn test_search_scope_from_str() {
        assert_eq!(SearchScope::from_str_loose("project"), SearchScope::Project);
        assert_eq!(SearchScope::from_str_loose("group"), SearchScope::Group);
        assert_eq!(SearchScope::from_str_loose("all"), SearchScope::All);
        assert_eq!(SearchScope::from_str_loose("GROUP"), SearchScope::Group);
        assert_eq!(SearchScope::from_str_loose("unknown"), SearchScope::Project);
    }

    #[test]
    fn test_relevance_decay_defaults() {
        let decay = RelevanceDecay::default();
        assert_eq!(decay.current_project, 1.0);
        assert_eq!(decay.group_project, 0.7);
        assert_eq!(decay.other_project, 0.4);
    }

    #[test]
    fn test_apply_decay_current_project() {
        let decay = RelevanceDecay::default();
        let mut results = vec![make_result("r1", 0.9, "proj-a")];

        apply_relevance_decay(&mut results, "proj-a", &[], &decay);

        // Current project keeps full score
        assert!((results[0].score - 0.9).abs() < 1e-6);
    }

    #[test]
    fn test_apply_decay_group_project() {
        let decay = RelevanceDecay::default();
        let mut results = vec![make_result("r1", 1.0, "proj-b")];

        let group = vec!["proj-a".to_string(), "proj-b".to_string()];
        apply_relevance_decay(&mut results, "proj-a", &group, &decay);

        // Group project: 1.0 * 0.7 = 0.7
        assert!((results[0].score - 0.7).abs() < 1e-6);
    }

    #[test]
    fn test_apply_decay_other_project() {
        let decay = RelevanceDecay::default();
        let mut results = vec![make_result("r1", 1.0, "proj-x")];

        apply_relevance_decay(&mut results, "proj-a", &[], &decay);

        // Other project: 1.0 * 0.4 = 0.4
        assert!((results[0].score - 0.4).abs() < 1e-6);
    }

    #[test]
    fn test_apply_decay_reorders() {
        let decay = RelevanceDecay::default();
        let mut results = vec![
            make_result("other", 0.95, "proj-x"),   // 0.95 * 0.4 = 0.38
            make_result("group", 0.80, "proj-b"),   // 0.80 * 0.7 = 0.56
            make_result("current", 0.60, "proj-a"), // 0.60 * 1.0 = 0.60
        ];

        let group = vec!["proj-a".to_string(), "proj-b".to_string()];
        apply_relevance_decay(&mut results, "proj-a", &group, &decay);

        // Should be reordered: current(0.60) > group(0.56) > other(0.38)
        assert_eq!(results[0].id, "current");
        assert_eq!(results[1].id, "group");
        assert_eq!(results[2].id, "other");
    }

    #[test]
    fn test_apply_decay_custom_multipliers() {
        let decay = RelevanceDecay {
            current_project: 1.0,
            group_project: 0.9,
            other_project: 0.1,
        };
        let mut results = vec![
            make_result("r1", 0.8, "proj-a"),
            make_result("r2", 0.9, "proj-x"),
        ];

        apply_relevance_decay(&mut results, "proj-a", &[], &decay);

        // Current: 0.8 * 1.0 = 0.8
        // Other: 0.9 * 0.1 = 0.09
        assert!((results[0].score - 0.8).abs() < 1e-6);
        assert!((results[1].score - 0.09).abs() < 1e-6);
    }

    #[tokio::test]
    async fn test_resolve_scope_project() {
        // Project scope doesn't need a pool
        use sqlx::sqlite::SqlitePoolOptions;
        let pool = SqlitePoolOptions::new()
            .max_connections(1)
            .connect("sqlite::memory:")
            .await
            .unwrap();

        let filter = resolve_scope(SearchScope::Project, "proj-a", &pool).await;
        match filter {
            TenantFilter::Single(t) => assert_eq!(t, "proj-a"),
            _ => panic!("Expected Single filter"),
        }
    }

    #[tokio::test]
    async fn test_resolve_scope_all() {
        use sqlx::sqlite::SqlitePoolOptions;
        let pool = SqlitePoolOptions::new()
            .max_connections(1)
            .connect("sqlite::memory:")
            .await
            .unwrap();

        let filter = resolve_scope(SearchScope::All, "proj-a", &pool).await;
        assert!(matches!(filter, TenantFilter::None));
    }

    #[tokio::test]
    async fn test_resolve_scope_group_no_table() {
        // Group scope without the table falls back to Single
        use sqlx::sqlite::SqlitePoolOptions;
        let pool = SqlitePoolOptions::new()
            .max_connections(1)
            .connect("sqlite::memory:")
            .await
            .unwrap();

        let filter = resolve_scope(SearchScope::Group, "proj-a", &pool).await;
        match filter {
            TenantFilter::Single(t) => assert_eq!(t, "proj-a"),
            _ => panic!("Expected fallback to Single"),
        }
    }

    #[tokio::test]
    async fn test_resolve_scope_group_with_members() {
        use sqlx::sqlite::SqlitePoolOptions;
        let pool = SqlitePoolOptions::new()
            .max_connections(1)
            .connect("sqlite::memory:")
            .await
            .unwrap();

        // Create the table
        sqlx::query(crate::project_groups_schema::CREATE_PROJECT_GROUPS_SQL)
            .execute(&pool)
            .await
            .unwrap();

        // Add group members
        project_groups_schema::add_to_group(&pool, "grp-1", "proj-a", "workspace", 1.0)
            .await
            .unwrap();
        project_groups_schema::add_to_group(&pool, "grp-1", "proj-b", "workspace", 1.0)
            .await
            .unwrap();

        let filter = resolve_scope(SearchScope::Group, "proj-a", &pool).await;
        match filter {
            TenantFilter::Multiple(tenants) => {
                assert_eq!(tenants.len(), 2);
                assert!(tenants.contains(&"proj-a".to_string()));
                assert!(tenants.contains(&"proj-b".to_string()));
            }
            _ => panic!("Expected Multiple filter"),
        }
    }
}
