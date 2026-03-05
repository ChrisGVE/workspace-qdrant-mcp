//! Project groups schema and operations.
//!
//! Groups related projects for cross-project search scoping.
//! A group can represent dependency relationships, git organization
//! membership, or workspace co-location.

use sqlx::{Row, SqlitePool};
use tracing::debug;
use wqm_common::timestamps::now_utc;

/// SQL to create the project_groups table (schema v24).
pub const CREATE_PROJECT_GROUPS_SQL: &str = r#"
CREATE TABLE IF NOT EXISTS project_groups (
    group_id TEXT NOT NULL,
    tenant_id TEXT NOT NULL,
    group_type TEXT NOT NULL,
    confidence REAL NOT NULL DEFAULT 1.0,
    created_at TEXT NOT NULL,
    PRIMARY KEY (group_id, tenant_id)
)
"#;

/// SQL to create indexes on project_groups.
pub const CREATE_PROJECT_GROUPS_INDEXES_SQL: &[&str] = &[
    "CREATE INDEX IF NOT EXISTS idx_project_groups_tenant ON project_groups(tenant_id)",
    "CREATE INDEX IF NOT EXISTS idx_project_groups_type ON project_groups(group_type)",
];

/// Get all tenant IDs in the same group(s) as the given tenant.
///
/// Returns the set of tenant_ids that share at least one group with the
/// provided tenant_id, including the tenant itself.
pub async fn get_group_members(
    pool: &SqlitePool,
    tenant_id: &str,
) -> Result<Vec<String>, sqlx::Error> {
    let rows = sqlx::query(
        r#"
        SELECT DISTINCT pg2.tenant_id
        FROM project_groups pg1
        JOIN project_groups pg2 ON pg1.group_id = pg2.group_id
        WHERE pg1.tenant_id = ?
        ORDER BY pg2.tenant_id
        "#,
    )
    .bind(tenant_id)
    .fetch_all(pool)
    .await?;

    let members: Vec<String> = rows
        .iter()
        .map(|r| r.get::<String, _>("tenant_id"))
        .collect();

    debug!(tenant_id, members = members.len(), "Fetched group members");

    Ok(members)
}

/// Add a tenant to a group.
pub async fn add_to_group(
    pool: &SqlitePool,
    group_id: &str,
    tenant_id: &str,
    group_type: &str,
    confidence: f64,
) -> Result<(), sqlx::Error> {
    let now = now_utc();

    sqlx::query(
        r#"
        INSERT OR REPLACE INTO project_groups
            (group_id, tenant_id, group_type, confidence, created_at)
        VALUES (?, ?, ?, ?, ?)
        "#,
    )
    .bind(group_id)
    .bind(tenant_id)
    .bind(group_type)
    .bind(confidence)
    .bind(&now)
    .execute(pool)
    .await?;

    debug!(group_id, tenant_id, group_type, "Added tenant to group");
    Ok(())
}

/// Remove a tenant from a group.
pub async fn remove_from_group(
    pool: &SqlitePool,
    group_id: &str,
    tenant_id: &str,
) -> Result<bool, sqlx::Error> {
    let result = sqlx::query("DELETE FROM project_groups WHERE group_id = ? AND tenant_id = ?")
        .bind(group_id)
        .bind(tenant_id)
        .execute(pool)
        .await?;

    Ok(result.rows_affected() > 0)
}

/// List all groups a tenant belongs to.
pub async fn list_tenant_groups(
    pool: &SqlitePool,
    tenant_id: &str,
) -> Result<Vec<(String, String, f64)>, sqlx::Error> {
    let rows = sqlx::query(
        r#"
        SELECT group_id, group_type, confidence
        FROM project_groups
        WHERE tenant_id = ?
        ORDER BY group_id
        "#,
    )
    .bind(tenant_id)
    .fetch_all(pool)
    .await?;

    let groups: Vec<(String, String, f64)> = rows
        .iter()
        .map(|r| {
            (
                r.get::<String, _>("group_id"),
                r.get::<String, _>("group_type"),
                r.get::<f64, _>("confidence"),
            )
        })
        .collect();

    Ok(groups)
}

#[cfg(test)]
mod tests {
    use super::*;
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

    #[tokio::test]
    async fn test_add_and_get_group_members() {
        let pool = setup_pool().await;

        // Add three projects to the same group
        add_to_group(&pool, "grp-1", "proj-a", "workspace", 1.0)
            .await
            .unwrap();
        add_to_group(&pool, "grp-1", "proj-b", "workspace", 1.0)
            .await
            .unwrap();
        add_to_group(&pool, "grp-1", "proj-c", "workspace", 0.8)
            .await
            .unwrap();

        let members = get_group_members(&pool, "proj-a").await.unwrap();
        assert_eq!(members.len(), 3);
        assert!(members.contains(&"proj-a".to_string()));
        assert!(members.contains(&"proj-b".to_string()));
        assert!(members.contains(&"proj-c".to_string()));
    }

    #[tokio::test]
    async fn test_empty_group_members() {
        let pool = setup_pool().await;

        let members = get_group_members(&pool, "nonexistent").await.unwrap();
        assert!(members.is_empty());
    }

    #[tokio::test]
    async fn test_remove_from_group() {
        let pool = setup_pool().await;

        add_to_group(&pool, "grp-1", "proj-a", "workspace", 1.0)
            .await
            .unwrap();
        add_to_group(&pool, "grp-1", "proj-b", "workspace", 1.0)
            .await
            .unwrap();

        let removed = remove_from_group(&pool, "grp-1", "proj-b").await.unwrap();
        assert!(removed);

        let members = get_group_members(&pool, "proj-a").await.unwrap();
        assert_eq!(members.len(), 1);
        assert_eq!(members[0], "proj-a");
    }

    #[tokio::test]
    async fn test_remove_nonexistent() {
        let pool = setup_pool().await;

        let removed = remove_from_group(&pool, "grp-1", "none").await.unwrap();
        assert!(!removed);
    }

    #[tokio::test]
    async fn test_list_tenant_groups() {
        let pool = setup_pool().await;

        add_to_group(&pool, "grp-1", "proj-a", "workspace", 1.0)
            .await
            .unwrap();
        add_to_group(&pool, "grp-2", "proj-a", "git_org", 0.9)
            .await
            .unwrap();

        let groups = list_tenant_groups(&pool, "proj-a").await.unwrap();
        assert_eq!(groups.len(), 2);
        assert_eq!(groups[0].0, "grp-1");
        assert_eq!(groups[0].1, "workspace");
        assert_eq!(groups[1].0, "grp-2");
        assert_eq!(groups[1].1, "git_org");
    }

    #[tokio::test]
    async fn test_multi_group_membership() {
        let pool = setup_pool().await;

        // proj-a and proj-b share grp-1
        add_to_group(&pool, "grp-1", "proj-a", "workspace", 1.0)
            .await
            .unwrap();
        add_to_group(&pool, "grp-1", "proj-b", "workspace", 1.0)
            .await
            .unwrap();

        // proj-a and proj-c share grp-2
        add_to_group(&pool, "grp-2", "proj-a", "git_org", 0.9)
            .await
            .unwrap();
        add_to_group(&pool, "grp-2", "proj-c", "git_org", 0.9)
            .await
            .unwrap();

        // proj-a should see all three (itself via both groups)
        let members = get_group_members(&pool, "proj-a").await.unwrap();
        assert_eq!(members.len(), 3);
        assert!(members.contains(&"proj-a".to_string()));
        assert!(members.contains(&"proj-b".to_string()));
        assert!(members.contains(&"proj-c".to_string()));

        // proj-b only sees grp-1 members
        let members = get_group_members(&pool, "proj-b").await.unwrap();
        assert_eq!(members.len(), 2);
        assert!(members.contains(&"proj-a".to_string()));
        assert!(members.contains(&"proj-b".to_string()));
    }
}
