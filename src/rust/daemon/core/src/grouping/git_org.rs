/// Automated project grouping by Git organization/user.
///
/// Parses normalized git remote URLs to extract the organization or user
/// component, then groups all projects sharing the same org into
/// `project_groups` entries with `group_type = "git_org"`.

use sqlx::{Row, SqlitePool};
use tracing::{debug, info};

use super::schema;
use wqm_common::project_id::ProjectIdCalculator;

// ---- Org extraction --------------------------------------------------------

/// Extract the organization/user component from a git remote URL.
///
/// Uses `ProjectIdCalculator::normalize_git_url` to canonicalize, then
/// splits on `/` to extract `host/org`. Handles GitHub, GitLab, Bitbucket,
/// and generic git hosts.
///
/// # Examples
/// ```text
/// "https://github.com/ChrisGVE/my-repo.git" -> Some("github.com/chrisgve")
/// "git@gitlab.com:my-org/my-repo.git"        -> Some("gitlab.com/my-org")
/// "https://bitbucket.org/team/repo"           -> Some("bitbucket.org/team")
/// "https://github.com/solo-repo"              -> None (no org component)
/// ```
pub fn extract_git_org(remote_url: &str) -> Option<String> {
    if remote_url.is_empty() {
        return None;
    }

    // Normalize: "github.com/org/repo"
    let normalized = ProjectIdCalculator::normalize_git_url(remote_url);

    // Split: ["github.com", "org", "repo", ...]
    let parts: Vec<&str> = normalized.split('/').collect();

    // Need at least host/org/repo (3 parts)
    if parts.len() < 3 {
        return None;
    }

    let host = parts[0];
    let org = parts[1];

    if host.is_empty() || org.is_empty() {
        return None;
    }

    // Return "host/org" as the group key (lowercase from normalize)
    Some(format!("{}/{}", host, org))
}

/// Generate a deterministic group_id from a git org key.
///
/// Format: `git_org:<host>/<org>` -- human-readable and deterministic.
pub fn org_to_group_id(org_key: &str) -> String {
    format!("git_org:{}", org_key)
}

// ---- Grouping logic --------------------------------------------------------

/// Recompute git-org-based groups for all registered projects.
///
/// Reads `watch_folders` to get all tenant_ids with git remote URLs,
/// extracts orgs, and creates groups in `project_groups` with
/// `group_type = "git_org"` and `confidence = 1.0`.
///
/// Returns the number of groups created (orgs with 2+ projects).
pub async fn compute_git_org_groups(pool: &SqlitePool) -> Result<usize, sqlx::Error> {
    // Clear existing git_org groups
    sqlx::query("DELETE FROM project_groups WHERE group_type = 'git_org'")
        .execute(pool)
        .await?;

    // Load all projects with remote URLs
    let rows = sqlx::query(
        r#"
        SELECT tenant_id, git_remote_url
        FROM watch_folders
        WHERE git_remote_url IS NOT NULL AND git_remote_url != ''
        "#,
    )
    .fetch_all(pool)
    .await?;

    // Group tenants by org
    let mut org_tenants: std::collections::HashMap<String, Vec<String>> =
        std::collections::HashMap::new();

    for row in &rows {
        let tenant_id: String = row.get("tenant_id");
        let remote_url: String = row.get("git_remote_url");

        if let Some(org_key) = extract_git_org(&remote_url) {
            org_tenants
                .entry(org_key)
                .or_default()
                .push(tenant_id);
        }
    }

    let mut groups_created = 0;

    // Create groups for orgs with 2+ projects
    for (org_key, tenants) in &org_tenants {
        if tenants.len() < 2 {
            debug!(
                org = org_key.as_str(),
                "Skipping single-project org"
            );
            continue;
        }

        let group_id = org_to_group_id(org_key);

        for tenant_id in tenants {
            schema::add_to_group(pool, &group_id, tenant_id, "git_org", 1.0)
                .await?;
        }

        debug!(
            org = org_key.as_str(),
            members = tenants.len(),
            group_id = group_id.as_str(),
            "Created git org group"
        );
        groups_created += 1;
    }

    info!(
        projects = rows.len(),
        orgs = org_tenants.len(),
        groups = groups_created,
        "Git org group computation complete"
    );

    Ok(groups_created)
}

/// When we are the first project in an org group, scan all other projects
/// and add any that share the same org key.
async fn backfill_org_peers(
    pool: &SqlitePool,
    tenant_id: &str,
    org_key: &str,
    group_id: &str,
) -> Result<(), sqlx::Error> {
    let peers = sqlx::query(
        r#"
        SELECT tenant_id, git_remote_url
        FROM watch_folders
        WHERE git_remote_url IS NOT NULL
          AND git_remote_url != ''
          AND tenant_id != ?
        "#,
    )
    .bind(tenant_id)
    .fetch_all(pool)
    .await?;

    for row in &peers {
        let peer_tenant: String = row.get("tenant_id");
        let peer_url: String = row.get("git_remote_url");

        if let Some(peer_org) = extract_git_org(&peer_url) {
            if peer_org == org_key {
                schema::add_to_group(pool, group_id, &peer_tenant, "git_org", 1.0).await?;
                debug!(
                    peer = peer_tenant.as_str(),
                    org = org_key,
                    "Added existing peer to git org group"
                );
            }
        }
    }
    Ok(())
}

/// Update git org groups for a single project.
///
/// Call this when a new project is registered or its remote URL changes.
/// More efficient than recomputing all groups.
pub async fn update_project_org_group(
    pool: &SqlitePool,
    tenant_id: &str,
    remote_url: &str,
) -> Result<bool, sqlx::Error> {
    // Remove any existing git_org membership for this tenant
    sqlx::query(
        "DELETE FROM project_groups WHERE tenant_id = ? AND group_type = 'git_org'",
    )
    .bind(tenant_id)
    .execute(pool)
    .await?;

    let org_key = match extract_git_org(remote_url) {
        Some(key) => key,
        None => {
            debug!(
                tenant_id,
                remote_url,
                "No org extractable from remote URL"
            );
            return Ok(false);
        }
    };

    let group_id = org_to_group_id(&org_key);

    // Check if other projects share this org
    let existing = sqlx::query(
        "SELECT COUNT(*) as cnt FROM project_groups WHERE group_id = ?",
    )
    .bind(&group_id)
    .fetch_one(pool)
    .await?;

    let count: i64 = existing.get("cnt");

    // Always add to the group (even if we're the first -- others may join later)
    schema::add_to_group(pool, &group_id, tenant_id, "git_org", 1.0).await?;

    // If there are already members, also ensure any single-member that was
    // previously skipped during compute_git_org_groups is now included
    if count == 0 {
        backfill_org_peers(pool, tenant_id, &org_key, &group_id).await?;
    }

    debug!(
        tenant_id,
        org = org_key.as_str(),
        group_id = group_id.as_str(),
        "Updated project git org group"
    );

    Ok(true)
}

#[cfg(test)]
mod tests {
    use super::*;
    use sqlx::sqlite::SqlitePoolOptions;

    // ---- URL parsing tests -------------------------------------------------

    #[test]
    fn test_extract_github_https() {
        let org = extract_git_org("https://github.com/ChrisGVE/my-repo.git");
        assert_eq!(org, Some("github.com/chrisgve".to_string()));
    }

    #[test]
    fn test_extract_github_ssh() {
        let org = extract_git_org("git@github.com:ChrisGVE/workspace-qdrant-mcp.git");
        assert_eq!(org, Some("github.com/chrisgve".to_string()));
    }

    #[test]
    fn test_extract_gitlab_https() {
        let org = extract_git_org("https://gitlab.com/my-org/my-project");
        assert_eq!(org, Some("gitlab.com/my-org".to_string()));
    }

    #[test]
    fn test_extract_gitlab_ssh() {
        let org = extract_git_org("git@gitlab.com:my-org/sub-project.git");
        assert_eq!(org, Some("gitlab.com/my-org".to_string()));
    }

    #[test]
    fn test_extract_bitbucket_https() {
        let org = extract_git_org("https://bitbucket.org/team-name/repo.git");
        assert_eq!(org, Some("bitbucket.org/team-name".to_string()));
    }

    #[test]
    fn test_extract_self_hosted() {
        let org = extract_git_org("https://git.internal.corp/engineering/service.git");
        assert_eq!(org, Some("git.internal.corp/engineering".to_string()));
    }

    #[test]
    fn test_extract_nested_github() {
        // GitHub doesn't support nested groups, but GitLab does: org/subgroup/repo
        let org = extract_git_org("https://gitlab.com/org/subgroup/deep-repo.git");
        // We extract only host/first-level: the top-level org
        assert_eq!(org, Some("gitlab.com/org".to_string()));
    }

    #[test]
    fn test_extract_empty_url() {
        assert_eq!(extract_git_org(""), None);
    }

    #[test]
    fn test_extract_no_org() {
        // URL with only host/repo (no org level)
        assert_eq!(extract_git_org("https://example.com/repo"), None);
    }

    #[test]
    fn test_extract_case_insensitive() {
        let org1 = extract_git_org("https://github.com/MyOrg/Repo1.git");
        let org2 = extract_git_org("git@github.com:myorg/Repo2.git");
        assert_eq!(org1, org2);
    }

    #[test]
    fn test_org_to_group_id() {
        assert_eq!(
            org_to_group_id("github.com/chrisgve"),
            "git_org:github.com/chrisgve"
        );
    }

    // ---- SQLite integration tests ------------------------------------------

    async fn setup_pool() -> SqlitePool {
        let pool = SqlitePoolOptions::new()
            .max_connections(1)
            .connect("sqlite::memory:")
            .await
            .unwrap();

        // Create watch_folders table (minimal for our needs)
        sqlx::query(
            r#"
            CREATE TABLE IF NOT EXISTS watch_folders (
                watch_id TEXT PRIMARY KEY,
                folder_path TEXT NOT NULL,
                watch_type TEXT NOT NULL DEFAULT 'project',
                tenant_id TEXT NOT NULL,
                git_remote_url TEXT,
                is_active INTEGER DEFAULT 1,
                created_at TEXT NOT NULL DEFAULT (datetime('now'))
            )
            "#,
        )
        .execute(&pool)
        .await
        .unwrap();

        // Create project_groups tables
        sqlx::query(schema::CREATE_PROJECT_GROUPS_SQL)
            .execute(&pool)
            .await
            .unwrap();
        for idx in schema::CREATE_PROJECT_GROUPS_INDEXES_SQL {
            sqlx::query(idx).execute(&pool).await.unwrap();
        }

        pool
    }

    async fn insert_watch_folder(pool: &SqlitePool, tenant_id: &str, remote_url: &str) {
        sqlx::query(
            r#"
            INSERT INTO watch_folders (watch_id, folder_path, tenant_id, git_remote_url)
            VALUES (?, '/tmp/' || ?, ?, ?)
            "#,
        )
        .bind(tenant_id) // use tenant_id as watch_id for simplicity
        .bind(tenant_id)
        .bind(tenant_id)
        .bind(remote_url)
        .execute(pool)
        .await
        .unwrap();
    }

    #[tokio::test]
    async fn test_compute_groups_same_org() {
        let pool = setup_pool().await;

        insert_watch_folder(&pool, "proj-a", "https://github.com/MyOrg/repo-a.git").await;
        insert_watch_folder(&pool, "proj-b", "https://github.com/MyOrg/repo-b.git").await;
        insert_watch_folder(&pool, "proj-c", "git@github.com:MyOrg/repo-c.git").await;

        let groups = compute_git_org_groups(&pool).await.unwrap();
        assert_eq!(groups, 1, "All three share the same org");

        let members = schema::get_group_members(&pool, "proj-a")
            .await
            .unwrap();
        assert_eq!(members.len(), 3);
        assert!(members.contains(&"proj-a".to_string()));
        assert!(members.contains(&"proj-b".to_string()));
        assert!(members.contains(&"proj-c".to_string()));
    }

    #[tokio::test]
    async fn test_compute_groups_different_orgs() {
        let pool = setup_pool().await;

        insert_watch_folder(&pool, "proj-a", "https://github.com/org-one/repo.git").await;
        insert_watch_folder(&pool, "proj-b", "https://github.com/org-two/repo.git").await;

        let groups = compute_git_org_groups(&pool).await.unwrap();
        assert_eq!(groups, 0, "Different orgs, no groups");
    }

    #[tokio::test]
    async fn test_compute_groups_mixed_hosts() {
        let pool = setup_pool().await;

        insert_watch_folder(&pool, "proj-a", "https://github.com/myorg/repo-a.git").await;
        insert_watch_folder(&pool, "proj-b", "https://gitlab.com/myorg/repo-b.git").await;

        let groups = compute_git_org_groups(&pool).await.unwrap();
        assert_eq!(groups, 0, "Same org name but different hosts -> different groups");
    }

    #[tokio::test]
    async fn test_compute_groups_multiple_orgs() {
        let pool = setup_pool().await;

        // Org A: 2 projects
        insert_watch_folder(&pool, "proj-1", "https://github.com/orgA/repo1.git").await;
        insert_watch_folder(&pool, "proj-2", "https://github.com/orgA/repo2.git").await;
        // Org B: 3 projects
        insert_watch_folder(&pool, "proj-3", "https://gitlab.com/orgB/app1.git").await;
        insert_watch_folder(&pool, "proj-4", "https://gitlab.com/orgB/app2.git").await;
        insert_watch_folder(&pool, "proj-5", "https://gitlab.com/orgB/app3.git").await;
        // Solo project
        insert_watch_folder(&pool, "proj-6", "https://github.com/solo/lonely.git").await;

        let groups = compute_git_org_groups(&pool).await.unwrap();
        assert_eq!(groups, 2, "Two orgs with 2+ projects");
    }

    #[tokio::test]
    async fn test_update_project_new_org() {
        let pool = setup_pool().await;

        // proj-a already in org
        insert_watch_folder(&pool, "proj-a", "https://github.com/MyOrg/repo-a.git").await;

        // Register proj-b in same org
        insert_watch_folder(&pool, "proj-b", "https://github.com/MyOrg/repo-b.git").await;
        let added = update_project_org_group(
            &pool,
            "proj-b",
            "https://github.com/MyOrg/repo-b.git",
        )
        .await
        .unwrap();
        assert!(added);

        // Both should be in the same group
        let members = schema::get_group_members(&pool, "proj-a")
            .await
            .unwrap();
        assert_eq!(members.len(), 2);
    }

    #[tokio::test]
    async fn test_update_project_no_org() {
        let pool = setup_pool().await;

        // URL with no org
        let added = update_project_org_group(&pool, "proj-x", "https://example.com/repo")
            .await
            .unwrap();
        assert!(!added);
    }

    #[tokio::test]
    async fn test_recompute_clears_stale_groups() {
        let pool = setup_pool().await;

        // First compute: 2 in same org
        insert_watch_folder(&pool, "proj-a", "https://github.com/OrgX/repo1.git").await;
        insert_watch_folder(&pool, "proj-b", "https://github.com/OrgX/repo2.git").await;
        compute_git_org_groups(&pool).await.unwrap();

        let members = schema::get_group_members(&pool, "proj-a")
            .await
            .unwrap();
        assert_eq!(members.len(), 2);

        // Change proj-b's remote URL to different org
        sqlx::query("UPDATE watch_folders SET git_remote_url = ? WHERE tenant_id = ?")
            .bind("https://github.com/OtherOrg/repo2.git")
            .bind("proj-b")
            .execute(&pool)
            .await
            .unwrap();

        // Recompute -- should clear old group
        let groups = compute_git_org_groups(&pool).await.unwrap();
        assert_eq!(groups, 0, "No org has 2+ projects now");

        let members = schema::get_group_members(&pool, "proj-a")
            .await
            .unwrap();
        assert_eq!(members.len(), 0, "proj-a has no group members after recompute");
    }
}
