//! Tests for git worktree registration and data model correctness.

use super::*;

// ── worktree registration ───────────────────────────────────────────

#[tokio::test]
async fn test_worktree_registration_fields() {
    let temp_dir = tempdir().unwrap();
    let db_path = temp_dir.path().join("test_worktree_reg.db");
    let manager = DaemonStateManager::new(&db_path).await.unwrap();
    manager.initialize().await.unwrap();

    // Register the main repo
    let main_repo = WatchFolderRecord {
        is_git_tracked: true,
        ..make_test_watch_folder("main-001", "/repos/myproject", "shared-tenant")
    };
    manager.store_watch_folder(&main_repo).await.unwrap();

    // Register a worktree pointing back to the main repo
    let worktree = WatchFolderRecord {
        is_git_tracked: true,
        is_worktree: true,
        main_worktree_watch_id: Some("main-001".to_string()),
        ..make_test_watch_folder("wt-001", "/repos/myproject-feature", "shared-tenant")
    };
    manager.store_watch_folder(&worktree).await.unwrap();

    // Verify main repo fields
    let main_record = manager.get_watch_folder("main-001").await.unwrap().unwrap();
    assert!(
        !main_record.is_worktree,
        "main repo should not be a worktree"
    );
    assert!(
        main_record.main_worktree_watch_id.is_none(),
        "main repo should have no main_worktree_watch_id"
    );

    // Verify worktree fields
    let wt_record = manager.get_watch_folder("wt-001").await.unwrap().unwrap();
    assert!(
        wt_record.is_worktree,
        "worktree entry must have is_worktree = true"
    );
    assert_eq!(
        wt_record.main_worktree_watch_id.as_deref(),
        Some("main-001"),
        "worktree must reference the main repo's watch_id"
    );
    assert_eq!(
        wt_record.tenant_id, main_record.tenant_id,
        "worktree must share the same tenant_id as the main repo"
    );
}

#[tokio::test]
async fn test_multiple_worktrees_share_tenant_id() {
    let temp_dir = tempdir().unwrap();
    let db_path = temp_dir.path().join("test_multi_worktree.db");
    let manager = DaemonStateManager::new(&db_path).await.unwrap();
    manager.initialize().await.unwrap();

    let tenant = "multi-wt-tenant";

    // Main repo
    let main_repo = WatchFolderRecord {
        is_git_tracked: true,
        ..make_test_watch_folder("main-001", "/repos/project", tenant)
    };
    manager.store_watch_folder(&main_repo).await.unwrap();

    // Worktree A
    let wt_a = WatchFolderRecord {
        is_git_tracked: true,
        is_worktree: true,
        main_worktree_watch_id: Some("main-001".to_string()),
        ..make_test_watch_folder("wt-a", "/repos/project-wt-a", tenant)
    };
    manager.store_watch_folder(&wt_a).await.unwrap();

    // Worktree B
    let wt_b = WatchFolderRecord {
        is_git_tracked: true,
        is_worktree: true,
        main_worktree_watch_id: Some("main-001".to_string()),
        ..make_test_watch_folder("wt-b", "/repos/project-wt-b", tenant)
    };
    manager.store_watch_folder(&wt_b).await.unwrap();

    // Retrieve and verify
    let records = manager
        .list_watch_folders(Some("projects"), false)
        .await
        .unwrap();
    assert_eq!(records.len(), 3);

    let worktrees: Vec<_> = records.iter().filter(|r| r.is_worktree).collect();
    assert_eq!(
        worktrees.len(),
        2,
        "exactly two entries should be worktrees"
    );

    for wt in &worktrees {
        assert_eq!(
            wt.tenant_id, tenant,
            "worktree must share the main tenant_id"
        );
        assert_eq!(
            wt.main_worktree_watch_id.as_deref(),
            Some("main-001"),
            "worktree must reference the main repo"
        );
    }

    let main = records.iter().find(|r| !r.is_worktree).unwrap();
    assert_eq!(main.watch_id, "main-001");
    assert!(main.main_worktree_watch_id.is_none());
}

// ── concurrent activation ───────────────────────────────────────────

#[tokio::test]
async fn test_worktree_concurrent_activation_deactivate_one() {
    let temp_dir = tempdir().unwrap();
    let db_path = temp_dir.path().join("test_wt_activation.db");
    let manager = DaemonStateManager::new(&db_path).await.unwrap();
    manager.initialize().await.unwrap();

    let tenant = "act-tenant";

    // Main repo
    let main_repo = WatchFolderRecord {
        is_active: true,
        is_git_tracked: true,
        last_activity_at: Some(Utc::now()),
        ..make_test_watch_folder("main-001", "/repos/proj", tenant)
    };
    manager.store_watch_folder(&main_repo).await.unwrap();

    // Worktree A (active)
    let wt_a = WatchFolderRecord {
        is_active: true,
        is_git_tracked: true,
        is_worktree: true,
        main_worktree_watch_id: Some("main-001".to_string()),
        last_activity_at: Some(Utc::now()),
        ..make_test_watch_folder("wt-a", "/repos/proj-wt-a", tenant)
    };
    manager.store_watch_folder(&wt_a).await.unwrap();

    // Worktree B (active)
    let wt_b = WatchFolderRecord {
        is_active: true,
        is_git_tracked: true,
        is_worktree: true,
        main_worktree_watch_id: Some("main-001".to_string()),
        last_activity_at: Some(Utc::now()),
        ..make_test_watch_folder("wt-b", "/repos/proj-wt-b", tenant)
    };
    manager.store_watch_folder(&wt_b).await.unwrap();

    // Verify both worktrees are active
    let wt_a_rec = manager.get_watch_folder("wt-a").await.unwrap().unwrap();
    let wt_b_rec = manager.get_watch_folder("wt-b").await.unwrap().unwrap();
    assert!(wt_a_rec.is_active);
    assert!(wt_b_rec.is_active);

    // Deactivate only worktree A by updating its record directly
    let deactivated = WatchFolderRecord {
        is_active: false,
        ..wt_a_rec
    };
    manager.store_watch_folder(&deactivated).await.unwrap();

    // Worktree A should be inactive
    let wt_a_after = manager.get_watch_folder("wt-a").await.unwrap().unwrap();
    assert!(!wt_a_after.is_active, "worktree A should be deactivated");

    // Worktree B should remain active
    let wt_b_after = manager.get_watch_folder("wt-b").await.unwrap().unwrap();
    assert!(wt_b_after.is_active, "worktree B should remain active");

    // Main repo should remain active
    let main_after = manager.get_watch_folder("main-001").await.unwrap().unwrap();
    assert!(main_after.is_active, "main repo should remain active");
}

// ── lifecycle path-scoped deactivation with worktree metadata ───────

#[tokio::test]
async fn test_worktree_path_scoped_deactivation_preserves_sibling() {
    let temp_dir = tempdir().unwrap();
    let db_path = temp_dir.path().join("test_wt_path_deact.db");
    let manager = DaemonStateManager::new(&db_path).await.unwrap();
    manager.initialize().await.unwrap();

    let tenant = "path-deact-tenant";

    // Main repo with is_worktree metadata
    let main_repo = WatchFolderRecord {
        is_git_tracked: true,
        ..make_test_watch_folder("main-001", "/repos/proj", tenant)
    };
    manager.store_watch_folder(&main_repo).await.unwrap();

    // Two worktrees with proper worktree metadata
    let wt_a = WatchFolderRecord {
        is_git_tracked: true,
        is_worktree: true,
        main_worktree_watch_id: Some("main-001".to_string()),
        ..make_test_watch_folder("wt-a", "/repos/proj-wt-a", tenant)
    };
    manager.store_watch_folder(&wt_a).await.unwrap();

    let wt_b = WatchFolderRecord {
        is_git_tracked: true,
        is_worktree: true,
        main_worktree_watch_id: Some("main-001".to_string()),
        ..make_test_watch_folder("wt-b", "/repos/proj-wt-b", tenant)
    };
    manager.store_watch_folder(&wt_b).await.unwrap();

    // Use WatchFolderLifecycle for path-scoped activation/deactivation
    use crate::lifecycle::WatchFolderLifecycle;
    let lc = WatchFolderLifecycle::new(manager.pool().clone());

    // Activate all via tenant
    lc.activate_by_tenant(tenant, "projects").await.unwrap();

    // All three should be active (is_active = 1)
    for id in &["main-001", "wt-a", "wt-b"] {
        let rec = manager.get_watch_folder(id).await.unwrap().unwrap();
        assert!(
            rec.is_active,
            "{} should be active after tenant activation",
            id
        );
    }

    // Deactivate only worktree A by path
    lc.deactivate_by_tenant_and_path(tenant, "/repos/proj-wt-a")
        .await
        .unwrap();

    // Worktree A should now be inactive
    let wt_a_after = manager.get_watch_folder("wt-a").await.unwrap().unwrap();
    assert!(
        !wt_a_after.is_active,
        "worktree A should be deactivated by path"
    );
    // Worktree A should still have worktree metadata intact
    assert!(wt_a_after.is_worktree);
    assert_eq!(
        wt_a_after.main_worktree_watch_id.as_deref(),
        Some("main-001")
    );

    // Worktree B and main should remain active
    let wt_b_after = manager.get_watch_folder("wt-b").await.unwrap().unwrap();
    assert!(wt_b_after.is_active, "worktree B should remain active");

    let main_after = manager.get_watch_folder("main-001").await.unwrap().unwrap();
    assert!(main_after.is_active, "main repo should remain active");
}

// ── worktree appears in active project listing ──────────────────────

#[tokio::test]
async fn test_active_worktrees_listed_as_active_projects() {
    let temp_dir = tempdir().unwrap();
    let db_path = temp_dir.path().join("test_wt_active_list.db");
    let manager = DaemonStateManager::new(&db_path).await.unwrap();
    manager.initialize().await.unwrap();

    let tenant = "list-tenant";

    // Main repo (active)
    let main_repo = WatchFolderRecord {
        is_active: true,
        is_git_tracked: true,
        last_activity_at: Some(Utc::now()),
        ..make_test_watch_folder("main-001", "/repos/proj", tenant)
    };
    manager.store_watch_folder(&main_repo).await.unwrap();

    // Active worktree
    let wt = WatchFolderRecord {
        is_active: true,
        is_git_tracked: true,
        is_worktree: true,
        main_worktree_watch_id: Some("main-001".to_string()),
        last_activity_at: Some(Utc::now()),
        ..make_test_watch_folder("wt-001", "/repos/proj-wt", tenant)
    };
    manager.store_watch_folder(&wt).await.unwrap();

    let active = manager.list_active_projects().await.unwrap();
    assert_eq!(
        active.len(),
        2,
        "both main and worktree should be listed as active"
    );

    let worktree_entry = active.iter().find(|r| r.is_worktree);
    assert!(
        worktree_entry.is_some(),
        "the worktree should appear in active list"
    );
    assert_eq!(worktree_entry.unwrap().watch_id, "wt-001");
}

// ── worktree update preserves metadata ──────────────────────────────

#[tokio::test]
async fn test_worktree_update_preserves_metadata() {
    let temp_dir = tempdir().unwrap();
    let db_path = temp_dir.path().join("test_wt_update.db");
    let manager = DaemonStateManager::new(&db_path).await.unwrap();
    manager.initialize().await.unwrap();

    let tenant = "update-tenant";

    // Register main + worktree
    let main_repo = WatchFolderRecord {
        is_git_tracked: true,
        ..make_test_watch_folder("main-001", "/repos/proj", tenant)
    };
    manager.store_watch_folder(&main_repo).await.unwrap();

    let wt = WatchFolderRecord {
        is_git_tracked: true,
        is_worktree: true,
        main_worktree_watch_id: Some("main-001".to_string()),
        ..make_test_watch_folder("wt-001", "/repos/proj-wt", tenant)
    };
    manager.store_watch_folder(&wt).await.unwrap();

    // Update worktree (e.g., scan timestamp) via store_watch_folder (INSERT OR REPLACE)
    let mut updated_wt = manager.get_watch_folder("wt-001").await.unwrap().unwrap();
    updated_wt.last_scan = Some(Utc::now());
    manager.store_watch_folder(&updated_wt).await.unwrap();

    // Verify worktree metadata survived the update
    let reloaded = manager.get_watch_folder("wt-001").await.unwrap().unwrap();
    assert!(reloaded.is_worktree, "is_worktree must survive update");
    assert_eq!(
        reloaded.main_worktree_watch_id.as_deref(),
        Some("main-001"),
        "main_worktree_watch_id must survive update"
    );
    assert_eq!(reloaded.tenant_id, tenant);
    assert!(reloaded.last_scan.is_some(), "last_scan should be set");
}
